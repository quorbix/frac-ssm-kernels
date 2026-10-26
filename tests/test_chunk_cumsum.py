import importlib
import math

import pytest
import torch

from frac_ssm.ops.refs.chunk_cumsum import chunk_cumsum_ref
from frac_ssm.ops.registry import available
from tests.utils import LOG_RHO_GENS

_IMPL_MODULES = {
    "triton": "frac_ssm.ops.triton.chunk_cumsum",
    "cute_sm90": "frac_ssm.ops.cute.chunk_cumsum",
}


@pytest.fixture(params=tuple(_IMPL_MODULES), ids=tuple(_IMPL_MODULES))
def cumsum_impl(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    backend = request.param
    if backend != "triton":
        ok, reason = available("cute_sm90")
        if not ok:
            pytest.skip(reason or f"Backend {backend!r} is unavailable")

    return importlib.import_module(_IMPL_MODULES[backend])


def _assert_ref_error(out, out_pt, out_ref):
    assert out.shape == out_ref.shape
    assert out.dtype == out_ref.dtype
    error = (out - out_ref).abs().max().item()
    error_pt = (out_pt - out_ref).abs().max().item()
    assert error <= 2 * error_pt


def _check_fwd_bwd(impl, log_rho, chunk_size, use_compile=False):
    log_rho = log_rho.detach().requires_grad_()
    log_rho_ref = log_rho.detach().clone().requires_grad_()

    fn = impl.chunk_cumsum
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)

    out = fn(log_rho, chunk_size)
    out_ref = chunk_cumsum_ref(log_rho_ref, chunk_size, upcast=True)
    out_pt = chunk_cumsum_ref(log_rho_ref, chunk_size, upcast=False)
    for o, o_pt, o_ref in zip(out, out_pt, out_ref):
        _assert_ref_error(o, o_pt, o_ref)

    grads = tuple(torch.randn_like(o) for o in out)
    torch.cuda.synchronize()
    (d,) = torch.autograd.grad(out, log_rho, grads)
    (d_ref,) = torch.autograd.grad(out_ref, log_rho_ref, grads)
    (d_pt,) = torch.autograd.grad(out_pt, log_rho_ref, grads)
    assert d.dtype == log_rho.dtype
    _assert_ref_error(d, d_pt, d_ref)


# ============================================================================
# Main correctness sweep
# ============================================================================

@pytest.mark.parametrize("input_gen",   list(LOG_RHO_GENS), ids=list(LOG_RHO_GENS))
@pytest.mark.parametrize("chunk_size",  [32, 128, 1024])
@pytest.mark.parametrize("L",           [32, 4001, 4096, 32768])
@pytest.mark.parametrize("M",           [8, 16])
@pytest.mark.parametrize("B",           [1, 17, 256])
def test_chunk_cumsum(cumsum_impl, B, M, L, chunk_size, input_gen):
    log_rho = LOG_RHO_GENS[input_gen](B, L, M)
    _check_fwd_bwd(cumsum_impl, log_rho, chunk_size)


def test_chunk_cumsum_compile(cumsum_impl):
    log_rho = LOG_RHO_GENS["model_like"](2, 4001, 16)
    _check_fwd_bwd(cumsum_impl, log_rho, 128, use_compile=True)


# ============================================================================
# Chunk decomposition invariant
# ============================================================================

# log_cum_rho resets at chunk boundaries. Adding the decay of all preceding
# chunks must reconstruct the full-sequence cumulative sum for every chunk_size.
@pytest.mark.parametrize("chunk_size", [32, 128, 1024])
def test_chunk_decomposition_reconstructs_full_cumsum(cumsum_impl, chunk_size):
    log_rho = LOG_RHO_GENS["model_like"](
        2, 4001, 16, seed=42
    )
    log_cum_rho, chunk_decay = cumsum_impl.chunk_cumsum(log_rho, chunk_size)

    preceding_decay = torch.cat(
        [torch.zeros_like(chunk_decay[:, :1]), chunk_decay.cumsum(1)[:, :-1]],
        dim=1,
    )
    reconstructed = log_cum_rho + preceding_decay.repeat_interleave(
        chunk_size, dim=1
    )[:, :log_rho.shape[1]]
    expected_ref = torch.cumsum(log_rho.double(), dim=1).float()
    expected_pt = torch.cumsum(log_rho, dim=1)

    _assert_ref_error(reconstructed, expected_pt, expected_ref)
    _assert_ref_error(chunk_decay.sum(dim=1), expected_pt[:, -1], expected_ref[:, -1])


# ============================================================================
# Edge cases
# ============================================================================

def test_L_one(cumsum_impl):
    log_rho = torch.randn(2, 1, 16, device="cuda", dtype=torch.float32)
    lcr, cd = cumsum_impl.chunk_cumsum(log_rho, 32)
    torch.testing.assert_close(lcr, log_rho)
    torch.testing.assert_close(cd, log_rho)


def test_zero_batch(cumsum_impl):
    L, M, cs = 128, 16, 32
    log_rho = torch.empty(0, L, M, device="cuda", requires_grad=True)
    lcr, cd = cumsum_impl.chunk_cumsum(log_rho, cs)
    assert lcr.shape == (0, L, M) and cd.shape == (0, math.ceil(L / cs), M)
    torch.autograd.backward((lcr, cd), (torch.empty_like(lcr), torch.empty_like(cd)))
    assert log_rho.grad.shape == (0, L, M)


def test_input_validation(cumsum_impl):
    x = torch.randn(2, 32, 16, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError, match="3D"):
        cumsum_impl.chunk_cumsum(x[0], 32)
    with pytest.raises(AssertionError, match="CUDA"):
        cumsum_impl.chunk_cumsum(x.cpu(), 32)
    with pytest.raises(AssertionError, match="dtype"):
        cumsum_impl.chunk_cumsum(x.double(), 32)
    with pytest.raises(AssertionError, match="chunk_size"):
        cumsum_impl.chunk_cumsum(x, 0)
