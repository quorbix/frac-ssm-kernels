import importlib

import pytest
import torch

from frac_ssm.ops.refs.state_passing import state_passing_ref
from frac_ssm.ops.registry import available
from tests.utils import LOG_RHO_GENS, is_cuda_oom


_IMPL_MODULES = {
    "triton": "frac_ssm.ops.triton.state_passing",
    "cute_sm90": "frac_ssm.ops.cute.state_passing",
    # "cute_sm90_seq": "frac_ssm.ops.cute.variants.state_passing_seq",

}

_TOKENS_PER_CHUNK = 32


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


def _make_inputs(B, nchunks, R, M, *, seed=0, gen="model_like", dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    states = 0.25 * torch.randn(
        B, nchunks, R, M, device="cuda", dtype=dtype, generator=g
    )
    log_rho = LOG_RHO_GENS[gen](
        B, nchunks * _TOKENS_PER_CHUNK, M, seed=seed + 1,
    )
    chunk_decay = log_rho.view(B, nchunks, _TOKENS_PER_CHUNK, M).sum(dim=2)
    initial_state = 0.25 * torch.randn(
        B, R, M, device="cuda", dtype=dtype, generator=g
    )
    return states, chunk_decay, initial_state


def _assert_ref_error(out, out_pt, out_ref):
    assert out.shape == out_ref.shape
    assert out.dtype == out_ref.dtype
    error = (out.float() - out_ref.float()).abs().max().item()
    error_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert error <= 2 * error_pt + 2e-4


def _autograd_grad(outputs, inputs, grad_outputs):
    differentiable = [
        (out, grad) for out, grad in zip(outputs, grad_outputs)
        if out.requires_grad
    ]
    assert differentiable
    return torch.autograd.grad(
        tuple(out for out, _ in differentiable),
        inputs,
        tuple(grad for _, grad in differentiable),
        allow_unused=False,
    )


def _check_fwd_bwd_inputs(
    impl, states, chunk_decay, initial_state=None, *, seed=0, use_compile=False,
):
    states = states.detach().requires_grad_()
    chunk_decay = chunk_decay.detach().requires_grad_()
    if initial_state is not None:
        initial_state = initial_state.detach().requires_grad_()

    states_r = states.detach().clone().requires_grad_()
    chunk_decay_r = chunk_decay.detach().clone().requires_grad_()
    initial_state_r = (
        initial_state.detach().clone().requires_grad_()
        if initial_state is not None else None
    )

    fn = impl.state_passing
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)
    out = fn(states, chunk_decay, initial_state)

    out_ref = state_passing_ref(
        states_r, chunk_decay_r, initial_state_r, upcast=True,
    )
    out_pt = state_passing_ref(
        states_r, chunk_decay_r, initial_state_r, upcast=False,
    )

    states_acc, final_state = out
    assert states_acc.dtype == states.dtype
    assert final_state.dtype == states.dtype
    for o, o_pt, o_ref in zip(out, out_pt, out_ref):
        _assert_ref_error(o, o_pt, o_ref)

    g = torch.Generator(device="cuda").manual_seed(seed + 17)
    grads = (
        torch.randn(states_acc.shape, device="cuda", dtype=states_acc.dtype, generator=g),
        torch.randn(final_state.shape, device="cuda", dtype=final_state.dtype, generator=g),
    )
    inputs = (states, chunk_decay)
    inputs_r = (states_r, chunk_decay_r)
    if initial_state is not None:
        inputs += (initial_state,)
        inputs_r += (initial_state_r,)

    d_inputs = _autograd_grad(out, inputs, grads)
    d_inputs_ref = _autograd_grad(out_ref, inputs_r, grads)
    d_inputs_pt = _autograd_grad(out_pt, inputs_r, grads)

    for d, d_pt, d_ref, x in zip(d_inputs, d_inputs_pt, d_inputs_ref, inputs):
        assert d.dtype == x.dtype
        _assert_ref_error(d, d_pt, d_ref)


def _check_fwd_bwd(
    impl, B, nchunks, R, M, gen, has_initial_state,
    *, seed=0, use_compile=False,
):
    states, chunk_decay, initial_state = _make_inputs(
        B, nchunks, R, M, seed=seed, gen=gen,
    )
    _check_fwd_bwd_inputs(
        impl,
        states,
        chunk_decay,
        initial_state if has_initial_state else None,
        seed=seed,
        use_compile=use_compile,
    )


# ============================================================================
# Main correctness sweep
# ============================================================================

@pytest.mark.parametrize("input_gen", list(LOG_RHO_GENS), ids=list(LOG_RHO_GENS))
@pytest.mark.parametrize("has_initial_state", [False, True], ids=["no_init", "init"])
@pytest.mark.parametrize("nchunks", [1, 17, 512])
@pytest.mark.parametrize("M", [8, 16])
@pytest.mark.parametrize("R", [64, 192, 512])
@pytest.mark.parametrize("B", [1, 65])
def test_state_passing(
    state_impl, B, R, M, nchunks, has_initial_state, input_gen
):
    try:
        _check_fwd_bwd(
            state_impl, B, nchunks, R, M, input_gen, has_initial_state,
        )
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        pytest.skip("Skipping state_passing case after CUDA OOM")


@pytest.mark.parametrize("has_initial_state", [False, True], ids=["no_init", "init"])
def test_state_passing_compile(state_impl, has_initial_state):
    _check_fwd_bwd(
        state_impl,
        B=2,
        nchunks=32,
        R=64,
        M=16,
        gen="model_like",
        has_initial_state=has_initial_state,
        use_compile=True,
    )


@pytest.mark.parametrize("has_initial_state", [False, True], ids=["no_init", "init"])
@pytest.mark.parametrize("nchunks", [1, 17, 512])
def test_bf16_states(state_impl, nchunks, has_initial_state):
    states, chunk_decay, initial_state = _make_inputs(
        2, nchunks, 64, 16, seed=7, dtype=torch.bfloat16,
    )
    _check_fwd_bwd_inputs(
        state_impl, states, chunk_decay,
        initial_state if has_initial_state else None, seed=7,
    )


# ============================================================================
# Edge cases
# ============================================================================


def test_zero_batch(state_impl):
    nchunks, R, M = 4, 64, 16
    states = torch.empty(
        0, nchunks, R, M, device="cuda", dtype=torch.float32, requires_grad=True,
    )
    chunk_decay = torch.empty(
        0, nchunks, M, device="cuda", dtype=torch.float32, requires_grad=True,
    )

    states_acc, final_state = state_impl.state_passing(states, chunk_decay)
    assert states_acc.shape == (0, nchunks, R, M)
    assert final_state.shape == (0, R, M)

    torch.autograd.backward(
        (states_acc, final_state),
        (torch.empty_like(states_acc), torch.empty_like(final_state)),
    )
    assert states.grad.shape == (0, nchunks, R, M)
    assert chunk_decay.grad.shape == (0, nchunks, M)


def test_strided_inputs(state_impl):
    B, nchunks, R, M = 2, 32, 64, 16
    states, chunk_decay, initial_state = _make_inputs(B, nchunks, R, M, seed=13)

    states_storage = torch.empty(B, nchunks, R, M + 5, device="cuda", dtype=torch.float32)
    decay_storage = torch.empty(B, nchunks, M + 5, device="cuda", dtype=torch.float32)
    initial_storage = torch.empty(B, R, M + 5, device="cuda", dtype=torch.float32)
    states_storage[..., :M].copy_(states)
    decay_storage[..., :M].copy_(chunk_decay)
    initial_storage[..., :M].copy_(initial_state)

    states = states_storage[..., :M]
    chunk_decay = decay_storage[..., :M]
    initial_state = initial_storage[..., :M]
    assert not states.is_contiguous()
    assert not chunk_decay.is_contiguous()
    assert not initial_state.is_contiguous()

    _check_fwd_bwd_inputs(
        state_impl, states, chunk_decay, initial_state, seed=13,
    )


def test_input_validation(state_impl):
    states = torch.randn(2, 4, 64, 16, device="cuda", dtype=torch.float32)
    chunk_decay = torch.randn(2, 4, 16, device="cuda", dtype=torch.float32)
    initial_state = torch.randn(2, 64, 16, device="cuda", dtype=torch.float32)

    with pytest.raises(AssertionError, match="4D"):
        state_impl.state_passing(states[0], chunk_decay)
    with pytest.raises(AssertionError, match="CUDA"):
        state_impl.state_passing(states.cpu(), chunk_decay.cpu())
    with pytest.raises(AssertionError, match="states dtype"):
        state_impl.state_passing(states.to(torch.float64), chunk_decay)
    with pytest.raises(AssertionError, match="chunk_decay dtype"):
        state_impl.state_passing(states, chunk_decay.to(torch.bfloat16))
    with pytest.raises(AssertionError, match="initial_state dtype"):
        state_impl.state_passing(states, chunk_decay, initial_state.to(torch.float64))
