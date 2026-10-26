import importlib
import math

import pytest
import torch

from frac_ssm.ops.refs.chunk_cumsum import chunk_cumsum_ref
from frac_ssm.ops.refs.chunk_state import chunk_state_ref
from frac_ssm.ops.registry import available
from tests.utils import LOG_RHO_GENS, is_cuda_oom


_IMPL_MODULES = {
    "triton": "frac_ssm.ops.triton.chunk_state",
    "cute_sm90": "frac_ssm.ops.cute.chunk_state",
    # "cute_sm90_tma_k_lcr": "frac_ssm.ops.cute.variants.chunk_state_tma_k_lcr",
}


@pytest.fixture(params=tuple(_IMPL_MODULES), ids=tuple(_IMPL_MODULES))
def state_impl(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    backend = request.param
    if backend != "triton":
        ok, reason = available("cute_sm90")
        if not ok:
            pytest.skip(reason or f"Backend {backend!r} is unavailable")

    return importlib.import_module(_IMPL_MODULES[backend])


def _make_inputs(B, L, R, M, chunk_size, dtype, *, seed=0, gen="model_like"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    u     = torch.randn(B, L, R, device="cuda", dtype=dtype, generator=g)
    kappa = torch.randn(B, L, M, device="cuda", dtype=dtype, generator=g)
    log_rho = LOG_RHO_GENS[gen](B, L, M, seed=seed + 1)
    log_cum_rho, _ = chunk_cumsum_ref(log_rho, chunk_size)
    return u, kappa, log_cum_rho


def _assert_ref_error(out, out_pt, out_ref):
    assert out.shape == out_ref.shape
    assert out.dtype == out_ref.dtype
    error = (out.float() - out_ref.float()).abs().max().item()
    error_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert error <= 2 * error_pt + 1e-4


def _check_fwd_bwd(impl, B, L, R, M, chunk_size, dtype, gen, *, seed=0, use_compile=False):
    u, kappa, lcr = _make_inputs(B, L, R, M, chunk_size, dtype, seed=seed, gen=gen)

    u = u.detach().requires_grad_()
    kappa = kappa.detach().requires_grad_()
    lcr = lcr.detach().requires_grad_()

    u_r   = u.detach().clone().requires_grad_()
    k_r   = kappa.detach().clone().requires_grad_()
    lcr_r = lcr.detach().clone().requires_grad_()

    # ---- Forward ----
    fn = impl.chunk_state
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)

    states     = fn(u, kappa, lcr, chunk_size)
    states_ref = chunk_state_ref(u_r, k_r, lcr_r, chunk_size, upcast=True)
    states_pt  = chunk_state_ref(u_r, k_r, lcr_r, chunk_size, upcast=False)
    _assert_ref_error(states, states_pt, states_ref)

    # ---- Backward ----
    g = torch.Generator(device="cuda").manual_seed(seed + 17)
    d_states = torch.randn(states.shape, device="cuda", dtype=states.dtype, generator=g) * 0.1
    torch.cuda.synchronize()

    d_u, d_kappa, d_lcr = torch.autograd.grad(
        states, (u, kappa, lcr), grad_outputs=d_states,
    )
    d_u_ref, d_kappa_ref, d_lcr_ref = torch.autograd.grad(
        states_ref, (u_r, k_r, lcr_r), grad_outputs=d_states,
    )
    d_u_pt, d_kappa_pt, d_lcr_pt = torch.autograd.grad(
        states_pt, (u_r, k_r, lcr_r), grad_outputs=d_states,
    )

    assert d_u.dtype == u.dtype
    assert d_kappa.dtype == kappa.dtype
    assert d_lcr.dtype == torch.float32

    _assert_ref_error(d_u, d_u_pt, d_u_ref)
    _assert_ref_error(d_kappa, d_kappa_pt, d_kappa_ref)
    _assert_ref_error(d_lcr, d_lcr_pt, d_lcr_ref)


# ============================================================================
# Main correctness sweep
# ============================================================================

@pytest.mark.parametrize("input_gen",  list(LOG_RHO_GENS), ids=list(LOG_RHO_GENS))
@pytest.mark.parametrize("dtype",      [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("chunk_size", [32, 128, 1024])
@pytest.mark.parametrize("L",          [32, 4001, 32768])
@pytest.mark.parametrize("M",          [8, 16])
@pytest.mark.parametrize("R",          [64, 128, 192])
@pytest.mark.parametrize("B",          [1, 17])
def test_chunk_state(state_impl, B, R, M, L, chunk_size, dtype, input_gen):
    try:
        _check_fwd_bwd(state_impl, B, L, R, M, chunk_size, dtype, input_gen)
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        pytest.skip("Skipping chunk_state case after CUDA OOM")


def test_chunk_state_compile(state_impl):
    _check_fwd_bwd(
        state_impl, B=2, L=4001, R=64, M=16, chunk_size=128,
        dtype=torch.bfloat16, gen="model_like", use_compile=True,
    )


# ============================================================================
# Edge cases
# ============================================================================

def test_L_one(state_impl):
    _check_fwd_bwd(
        state_impl, B=2, L=1, R=64, M=16, chunk_size=32,
        dtype=torch.bfloat16, gen="model_like",
    )


def test_zero_batch(state_impl):
    L, R, M, cs = 128, 64, 16, 32
    nchunks = math.ceil(L / cs)
    u     = torch.empty(0, L, R, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kappa = torch.empty(0, L, M, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    lcr   = torch.empty(0, L, M, device="cuda", dtype=torch.float32, requires_grad=True)
    states = state_impl.chunk_state(u, kappa, lcr, cs)
    assert states.shape == (0, nchunks, R, M)

    torch.autograd.backward(states, torch.empty_like(states))
    assert u.grad.shape == (0, L, R)
    assert kappa.grad.shape == (0, L, M)
    assert lcr.grad.shape == (0, L, M)

def test_input_validation(state_impl):
    u = torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16)
    kappa = torch.randn(2, 32, 16, device="cuda", dtype=torch.bfloat16)
    log_cum_rho = torch.randn(2, 32, 16, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError, match="3D"):
        state_impl.chunk_state(u[0], kappa, log_cum_rho, 32)
    with pytest.raises(AssertionError, match="CUDA"):
        state_impl.chunk_state(u.cpu(), kappa.cpu(), log_cum_rho.cpu(), 32)
    with pytest.raises(AssertionError, match="dtype"):
        state_impl.chunk_state(u, kappa, log_cum_rho.to(torch.bfloat16), 32)
    with pytest.raises(AssertionError, match="chunk_size"):
        state_impl.chunk_state(u, kappa, log_cum_rho, 0)
