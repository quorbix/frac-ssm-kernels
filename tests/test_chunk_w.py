import importlib
import math

import pytest
import torch

from frac_ssm.ops.refs.chunk_cumsum import chunk_cumsum_ref
from frac_ssm.ops.refs.chunk_w import chunk_w_ref
from tests.utils import LOG_RHO_GENS, is_cuda_oom


_IMPL_MODULES = {
    "triton": "frac_ssm.ops.triton.chunk_w",
}


@pytest.fixture(params=tuple(_IMPL_MODULES), ids=tuple(_IMPL_MODULES))
def chunk_w_impl(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    return importlib.import_module(_IMPL_MODULES[request.param])


def _assert_ref_error(out, out_pt, out_ref):
    assert out.shape == out_ref.shape
    assert out.dtype == out_ref.dtype
    error = (out.float() - out_ref.float()).abs().max().item()
    error_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert error <= 2 * error_pt + 1e-5


def _make_inputs(B, L, M, chunk_size, dtype, input_gen, *, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    c = 0.25 * torch.randn(B, L, M, device=device, dtype=dtype, generator=g)
    kappa = 0.25 * torch.randn(B, L, M, device=device, dtype=dtype, generator=g)
    log_rho = LOG_RHO_GENS[input_gen](
        B, L, M, seed=seed + 1, device=device
    )
    log_cum_rho, _ = chunk_cumsum_ref(log_rho, chunk_size)
    return c, kappa, log_cum_rho


def _check_fwd_bwd(
    impl, c, kappa, log_cum_rho, chunk_size, *, seed=0, use_compile=False,
):
    assert c.dtype == kappa.dtype
    assert log_cum_rho.dtype == torch.float32

    c = c.detach().requires_grad_()
    kappa = kappa.detach().requires_grad_()
    log_cum_rho = log_cum_rho.detach().requires_grad_()

    c_r = c.detach().double().requires_grad_()
    kappa_r = kappa.detach().double().requires_grad_()
    lcr_r = log_cum_rho.detach().double().requires_grad_()

    c_pt = c.detach().clone().requires_grad_()
    kappa_pt = kappa.detach().clone().requires_grad_()
    lcr_pt = log_cum_rho.detach().clone().requires_grad_()

    fn = impl.chunk_w
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)

    W = fn(c, kappa, log_cum_rho, chunk_size)
    W_ref = chunk_w_ref(c_r, kappa_r, lcr_r, chunk_size, upcast=True)
    W_pt = chunk_w_ref(c_pt, kappa_pt, lcr_pt, chunk_size, upcast=False)

    assert W.dtype == torch.float32
    assert W.shape == (
        c.shape[0],
        math.ceil(c.shape[1] / chunk_size),
        chunk_size,
        chunk_size,
    )
    _assert_ref_error(W, W_pt, W_ref)

    g = torch.Generator(device=c.device).manual_seed(seed + 17)
    d_W = torch.randn(W.shape, device=c.device, dtype=torch.float32, generator=g)

    grads = torch.autograd.grad(
        W, (c, kappa, log_cum_rho), grad_outputs=d_W,
    )
    grads_ref = torch.autograd.grad(
        W_ref, (c_r, kappa_r, lcr_r), grad_outputs=d_W,
    )
    grads_pt = torch.autograd.grad(
        W_pt, (c_pt, kappa_pt, lcr_pt), grad_outputs=d_W,
    )

    for grad, grad_pt, grad_ref, x in zip(
        grads, grads_pt, grads_ref, (c, kappa, log_cum_rho),
    ):
        assert grad.dtype == x.dtype
        _assert_ref_error(grad.float(), grad_pt.float(), grad_ref.float())


# ============================================================================
# Main correctness sweep
# ============================================================================

@pytest.mark.parametrize("input_gen", list(LOG_RHO_GENS), ids=list(LOG_RHO_GENS))
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("chunk_size", [32, 128, 1024])
@pytest.mark.parametrize("L", [32, 4001, 4096, 32768])
@pytest.mark.parametrize("M", [16])
@pytest.mark.parametrize("B", [1, 17, 256])
def test_chunk_w(chunk_w_impl, B, M, L, chunk_size, dtype, input_gen):
    try:
        inputs = _make_inputs(B, L, M, chunk_size, dtype, input_gen)
        _check_fwd_bwd(chunk_w_impl, *inputs, chunk_size)
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        pytest.skip("Skipping chunk_w case after CUDA OOM")


def test_chunk_w_compile(chunk_w_impl):
    inputs = _make_inputs(
        2, 4001, 16, 128, torch.bfloat16, "model_like",
    )
    _check_fwd_bwd(chunk_w_impl, *inputs, 128, use_compile=True)


@pytest.mark.parametrize("chunk_size", [32, 128])
def test_M_eight(chunk_w_impl, chunk_size):
    inputs = _make_inputs(
        2, 257, 8, chunk_size, torch.bfloat16, "model_like",
    )
    _check_fwd_bwd(chunk_w_impl, *inputs, chunk_size)


# ============================================================================
# Edge cases
# ============================================================================

def test_L_one(chunk_w_impl):
    B, L, M, chunk_size = 2, 1, 16, 32
    inputs = _make_inputs(B, L, M, chunk_size, torch.bfloat16, "model_like", seed=7)
    _check_fwd_bwd(chunk_w_impl, *inputs, chunk_size, seed=7)


def test_zero_batch(chunk_w_impl):
    L, M, chunk_size = 128, 16, 32
    nchunks = math.ceil(L / chunk_size)
    c = torch.empty(
        0, L, M, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    kappa = torch.empty(
        0, L, M, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    log_cum_rho = torch.empty(
        0, L, M, device="cuda", dtype=torch.float32, requires_grad=True,
    )

    W = chunk_w_impl.chunk_w(c, kappa, log_cum_rho, chunk_size)
    assert W.shape == (0, nchunks, chunk_size, chunk_size)
    assert W.dtype == torch.float32

    torch.autograd.backward(W, torch.empty_like(W))
    assert c.grad.shape == (0, L, M)
    assert kappa.grad.shape == (0, L, M)
    assert log_cum_rho.grad.shape == (0, L, M)


def test_strided_inputs(chunk_w_impl):
    B, L, M, chunk_size = 2, 4001, 16, 128
    dtype = torch.bfloat16
    c, kappa, log_cum_rho = _make_inputs(
        B, L, M, chunk_size, dtype, "model_like", seed=11
    )

    c_storage = torch.empty(B, L, M + 5, device="cuda", dtype=dtype)
    kappa_storage = torch.empty(B, L, M + 5, device="cuda", dtype=dtype)
    lcr_storage = torch.empty(B, L, M + 5, device="cuda", dtype=torch.float32)
    c_storage[..., :M].copy_(c)
    kappa_storage[..., :M].copy_(kappa)
    lcr_storage[..., :M].copy_(log_cum_rho)

    strided = (
        c_storage[..., :M],
        kappa_storage[..., :M],
        lcr_storage[..., :M],
    )
    assert not strided[0].is_contiguous()
    assert not strided[1].is_contiguous()
    assert not strided[2].is_contiguous()
    _check_fwd_bwd(chunk_w_impl, *strided, chunk_size, seed=11)


def test_input_validation(chunk_w_impl):
    c = torch.randn(2, 32, 16, device="cuda", dtype=torch.bfloat16)
    kappa = torch.randn(2, 32, 16, device="cuda", dtype=torch.bfloat16)
    log_cum_rho = torch.randn(2, 32, 16, device="cuda", dtype=torch.float32)

    with pytest.raises(AssertionError, match="3D"):
        chunk_w_impl.chunk_w(c[0], kappa, log_cum_rho, 32)
    with pytest.raises(AssertionError, match="CUDA"):
        chunk_w_impl.chunk_w(c.cpu(), kappa.cpu(), log_cum_rho.cpu(), 32)
    with pytest.raises(AssertionError, match="dtype"):
        chunk_w_impl.chunk_w(c, kappa, log_cum_rho.to(torch.bfloat16), 32)
    with pytest.raises(AssertionError, match="chunk_size"):
        chunk_w_impl.chunk_w(c, kappa, log_cum_rho, 0)
    with pytest.raises(AssertionError, match="multiple of 32"):
        chunk_w_impl.chunk_w(c, kappa, log_cum_rho, 48)
