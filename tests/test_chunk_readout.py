import importlib
import math

import pytest
import torch

from frac_ssm.ops.refs.chunk_cumsum import chunk_cumsum_ref
from frac_ssm.ops.refs.chunk_readout import chunk_readout_ref
from frac_ssm.ops.refs.chunk_state import chunk_state_ref
from frac_ssm.ops.refs.chunk_w import chunk_w_ref
from frac_ssm.ops.refs.state_passing import state_passing_ref
from tests.utils import LOG_RHO_GENS, is_cuda_oom


_IMPL_MODULES = {
    "triton": "frac_ssm.ops.triton.chunk_readout",
}


@pytest.fixture(params=tuple(_IMPL_MODULES), ids=tuple(_IMPL_MODULES))
def readout_impl(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    return importlib.import_module(_IMPL_MODULES[request.param])


def _assert_ref_error(out, out_pt, out_ref):
    assert out.shape == out_ref.shape
    error = (out.float() - out_ref.float()).abs().max().item()
    error_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert error <= 2 * error_pt + 1e-5


def _make_inputs(
    B, L, R, M, chunk_size, dtype, input_gen, *, seed=0, device="cuda",
):
    g = torch.Generator(device=device).manual_seed(seed)
    u = 0.25 * torch.randn(B, L, R, device=device, dtype=dtype, generator=g)
    c = 0.25 * torch.randn(B, L, M, device=device, dtype=dtype, generator=g)
    log_rho = LOG_RHO_GENS[input_gen](
        B, L, M, seed=seed + 1, device=device,
    )
    log_cum_rho, chunk_decay = chunk_cumsum_ref(log_rho, chunk_size)
    nchunks = math.ceil(L / chunk_size)

    if input_gen == "random":
        states_acc = 0.25 * torch.randn(
            B, nchunks, R, M, device=device, dtype=torch.float32, generator=g,
        )
        W = torch.tril(0.25 * torch.randn(
            B, nchunks, chunk_size, chunk_size,
            device=device, dtype=torch.float32, generator=g,
        ))
    else:
        kappa = 0.25 * torch.randn(
            B, L, M, device=device, dtype=dtype, generator=g,
        )
        states = chunk_state_ref(
            u, kappa, log_cum_rho, chunk_size, upcast=True,
        )
        states_acc, _ = state_passing_ref(states, chunk_decay)
        W = chunk_w_ref(c, kappa, log_cum_rho, chunk_size)

    return u, c, states_acc, W, log_cum_rho


def _check_fwd_bwd(
    impl, inputs, chunk_size, *, seed=0, use_compile=False,
):
    inputs = tuple(x.detach().requires_grad_() for x in inputs)
    inputs_ref = tuple(x.detach().float().requires_grad_() for x in inputs)
    inputs_pt = tuple(x.detach().clone().requires_grad_() for x in inputs)

    fn = impl.chunk_readout
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)

    out = fn(*inputs, chunk_size)
    out_ref = chunk_readout_ref(*inputs_ref, chunk_size, upcast=True)
    out_pt = chunk_readout_ref(*inputs_pt, chunk_size, upcast=False)

    assert out.dtype == inputs[0].dtype
    _assert_ref_error(out, out_pt, out_ref)

    g = torch.Generator(device=out.device).manual_seed(seed + 17)
    d_h = torch.randn(out.shape, device=out.device, dtype=out.dtype, generator=g)

    grads = torch.autograd.grad(out, inputs, grad_outputs=d_h)
    grads_ref = torch.autograd.grad(
        out_ref, inputs_ref, grad_outputs=d_h.float(),
    )
    grads_pt = torch.autograd.grad(out_pt, inputs_pt, grad_outputs=d_h)

    for grad, grad_pt, grad_ref, x in zip(
        grads, grads_pt, grads_ref, inputs,
    ):
        assert grad.dtype == x.dtype
        _assert_ref_error(grad, grad_pt, grad_ref)


# ============================================================================
# Main correctness sweep
# ============================================================================

@pytest.mark.parametrize("input_gen", list(LOG_RHO_GENS), ids=list(LOG_RHO_GENS))
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("chunk_size", [32, 128, 1024])
@pytest.mark.parametrize("L", [32, 4001, 4096, 32768])
@pytest.mark.parametrize("M", [16])
@pytest.mark.parametrize("R", [64, 192, 512])
@pytest.mark.parametrize("B", [1, 17, 256])
def test_chunk_readout(readout_impl, B, R, M, L, chunk_size, dtype, input_gen):
    try:
        inputs = _make_inputs(B, L, R, M, chunk_size, dtype, input_gen)
        _check_fwd_bwd(readout_impl, inputs, chunk_size)
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        pytest.skip("Skipping chunk_readout case after CUDA OOM")


def test_chunk_readout_compile(readout_impl):
    inputs = _make_inputs(
        2, 4001, 64, 16, 128, torch.bfloat16, "model_like",
    )
    _check_fwd_bwd(readout_impl, inputs, 128, use_compile=True)


@pytest.mark.parametrize("chunk_size", [32, 128])
def test_M_eight(readout_impl, chunk_size):
    inputs = _make_inputs(
        2, 257, 64, 8, chunk_size, torch.bfloat16, "model_like",
    )
    _check_fwd_bwd(readout_impl, inputs, chunk_size)


# ============================================================================
# Supported intermediate dtypes
# ============================================================================

# The public API supports BF16 intermediates even though the standard pipeline produces FP32.
def test_bf16_intermediates(readout_impl):
    inputs = list(_make_inputs(
        2, 257, 64, 16, 64, torch.bfloat16, "model_like", seed=15,
    ))
    inputs[2] = inputs[2].to(torch.bfloat16)
    inputs[3] = inputs[3].to(torch.bfloat16)
    _check_fwd_bwd(readout_impl, tuple(inputs), 64, seed=15)


# ============================================================================
# Edge cases
# ============================================================================

def test_L_one(readout_impl):
    inputs = _make_inputs(
        2, 1, 64, 16, 32, torch.bfloat16, "model_like", seed=7,
    )
    _check_fwd_bwd(readout_impl, inputs, 32, seed=7)


def test_zero_batch(readout_impl):
    L, R, M, chunk_size = 128, 64, 16, 32
    nchunks = math.ceil(L / chunk_size)
    inputs = (
        torch.empty(
            0, L, R, device="cuda", dtype=torch.bfloat16, requires_grad=True,
        ),
        torch.empty(
            0, L, M, device="cuda", dtype=torch.bfloat16, requires_grad=True,
        ),
        torch.empty(
            0, nchunks, R, M,
            device="cuda", dtype=torch.float32, requires_grad=True,
        ),
        torch.empty(
            0, nchunks, chunk_size, chunk_size,
            device="cuda", dtype=torch.float32, requires_grad=True,
        ),
        torch.empty(
            0, L, M, device="cuda", dtype=torch.float32, requires_grad=True,
        ),
    )

    h = readout_impl.chunk_readout(*inputs, chunk_size)
    assert h.shape == (0, L, R)
    assert h.dtype == torch.bfloat16

    torch.autograd.backward(h, torch.empty_like(h))
    for x in inputs:
        assert x.grad.shape == x.shape


def test_strided_inputs(readout_impl):
    B, L, R, M, chunk_size = 2, 4001, 64, 16, 128
    inputs = _make_inputs(
        B, L, R, M, chunk_size, torch.bfloat16, "model_like", seed=13,
    )
    u, c, states_acc, W, log_cum_rho = inputs
    nchunks = math.ceil(L / chunk_size)

    u_storage = torch.empty(B, L, R + 4, device="cuda", dtype=u.dtype)
    c_storage = torch.empty(B, L, M + 5, device="cuda", dtype=c.dtype)
    states_storage = torch.empty(
        B, nchunks, R, M + 5, device="cuda", dtype=states_acc.dtype,
    )
    W_storage = torch.empty(
        B,
        nchunks,
        chunk_size,
        chunk_size + 5,
        device="cuda",
        dtype=W.dtype,
    )
    lcr_storage = torch.empty(
        B, L, M + 5, device="cuda", dtype=log_cum_rho.dtype,
    )
    u_storage[..., :R].copy_(u)
    c_storage[..., :M].copy_(c)
    states_storage[..., :M].copy_(states_acc)
    W_storage[..., :chunk_size].copy_(W)
    lcr_storage[..., :M].copy_(log_cum_rho)

    strided = (
        u_storage[..., :R],
        c_storage[..., :M],
        states_storage[..., :M],
        W_storage[..., :chunk_size],
        lcr_storage[..., :M],
    )
    assert not strided[0].is_contiguous()
    assert not strided[1].is_contiguous()
    assert not strided[2].is_contiguous()
    assert not strided[3].is_contiguous()
    assert not strided[4].is_contiguous()
    _check_fwd_bwd(readout_impl, strided, chunk_size, seed=13)


def test_input_validation(readout_impl):
    inputs = _make_inputs(
        2, 32, 64, 16, 32, torch.bfloat16, "random",
    )
    u, c, states_acc, W, log_cum_rho = inputs

    with pytest.raises(AssertionError, match="3D"):
        readout_impl.chunk_readout(u[0], c, states_acc, W, log_cum_rho, 32)
    with pytest.raises(AssertionError, match="CUDA"):
        readout_impl.chunk_readout(
            u.cpu(), c.cpu(), states_acc.cpu(), W.cpu(), log_cum_rho.cpu(), 32,
        )
    with pytest.raises(AssertionError, match="states_acc dtype"):
        readout_impl.chunk_readout(
            u, c, states_acc.to(torch.float16), W, log_cum_rho, 32,
        )
    with pytest.raises(AssertionError, match="W dtype"):
        readout_impl.chunk_readout(
            u, c, states_acc, W.to(torch.float16), log_cum_rho, 32,
        )
    with pytest.raises(AssertionError, match="log_cum_rho dtype"):
        readout_impl.chunk_readout(
            u, c, states_acc, W, log_cum_rho.to(torch.bfloat16), 32,
        )
    with pytest.raises(AssertionError, match="chunk_size"):
        readout_impl.chunk_readout(u, c, states_acc, W, log_cum_rho, 0)
    with pytest.raises(AssertionError, match="c shape"):
        readout_impl.chunk_readout(
            u, c[:, :-1], states_acc, W, log_cum_rho, 32,
        )
    with pytest.raises(AssertionError, match="log_cum_rho shape"):
        readout_impl.chunk_readout(
            u, c, states_acc, W, log_cum_rho[:, :-1], 32,
        )
    with pytest.raises(AssertionError, match="states_acc shape"):
        readout_impl.chunk_readout(
            u, c, states_acc[:, :, :-1], W, log_cum_rho, 32,
        )
    with pytest.raises(AssertionError, match="W shape"):
        readout_impl.chunk_readout(
            u, c, states_acc, W[..., :-1], log_cum_rho, 32,
        )
