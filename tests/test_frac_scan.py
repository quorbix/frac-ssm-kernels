import importlib

import pytest
import torch

from frac_ssm.ops.refs.frac_scan import frac_scan_ref
from frac_ssm.ops.registry import available
from tests.utils import LOG_RHO_GENS, is_cuda_oom


_IMPL_MODULES = {
    "triton": "frac_ssm.ops.triton.frac_scan",
    "cute_sm90": "frac_ssm.ops.cute.frac_scan",
}


@pytest.fixture(params=tuple(_IMPL_MODULES), ids=tuple(_IMPL_MODULES))
def scan_impl(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    backend = request.param
    if backend != "triton":
        ok, reason = available("cute_sm90")
        if not ok:
            pytest.skip(reason or f"Backend {backend!r} is unavailable")

    return importlib.import_module(_IMPL_MODULES[backend])


def _state_dtype(impl, u):
    if u.dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float32


def _assert_ref_error(out, out_pt, out_ref):
    assert out.shape == out_ref.shape
    error = (out.float() - out_ref.float()).abs().max().item()
    error_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert error <= 2 * error_pt + 1e-4


def _make_inputs(B, L, R, M, dtype, input_gen, *, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    u = 0.25 * torch.randn(B, L, R, device=device, dtype=dtype, generator=g)
    c = 0.25 * torch.randn(B, L, M, device=device, dtype=dtype, generator=g)
    kappa = 0.25 * torch.randn(
        B, L, M, device=device, dtype=dtype, generator=g,
    )
    log_rho = LOG_RHO_GENS[input_gen](
        B, L, M, seed=seed + 1, device=device,
    )
    rho = log_rho.exp()
    initial_state = 0.25 * torch.randn(
        B, R, M, device=device, dtype=torch.float32, generator=g,
    )
    return u, c, kappa, rho, initial_state


def _check_fwd_bwd_inputs(
    impl,
    u,
    c,
    kappa,
    rho,
    chunk_size,
    initial_state=None,
    *,
    seed=0,
    return_final_state=True,
    use_compile=False,
):
    u = u.detach().requires_grad_()
    c = c.detach().requires_grad_()
    kappa = kappa.detach().requires_grad_()
    rho = rho.detach().requires_grad_()
    if initial_state is not None:
        initial_state = initial_state.detach().requires_grad_()

    u_r = u.detach().float().requires_grad_()
    c_r = c.detach().float().requires_grad_()
    kappa_r = kappa.detach().float().requires_grad_()
    rho_r = rho.detach().clone().requires_grad_()
    initial_state_r = (
        initial_state.detach().clone().requires_grad_()
        if initial_state is not None else None
    )

    u_pt = u.detach().clone().requires_grad_()
    c_pt = c.detach().clone().requires_grad_()
    kappa_pt = kappa.detach().clone().requires_grad_()
    rho_pt = rho.detach().clone().requires_grad_()
    initial_state_pt = (
        initial_state.detach().clone().requires_grad_()
        if initial_state is not None else None
    )

    fn = impl.frac_scan
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)

    if return_final_state:
        h, final_state = fn(
            u, c, kappa, rho, chunk_size, initial_state, True,
        )
        h_ref, final_state_ref = frac_scan_ref(
            u_r, c_r, kappa_r, rho_r, chunk_size, initial_state_r, True,
            upcast=True,
        )
        h_pt, final_state_pt = frac_scan_ref(
            u_pt, c_pt, kappa_pt, rho_pt, chunk_size, initial_state_pt, True,
            upcast=False,
        )
    else:
        h = fn(u, c, kappa, rho, chunk_size, initial_state, False)
        h_ref = frac_scan_ref(
            u_r, c_r, kappa_r, rho_r, chunk_size, initial_state_r, False,
            upcast=True,
        )
        h_pt = frac_scan_ref(
            u_pt, c_pt, kappa_pt, rho_pt, chunk_size, initial_state_pt, False,
            upcast=False,
        )

    assert h.dtype == u.dtype
    _assert_ref_error(h, h_pt, h_ref)
    if return_final_state:
        assert final_state.dtype == _state_dtype(impl, u)
        _assert_ref_error(final_state, final_state_pt, final_state_ref)

    g = torch.Generator(device=u.device).manual_seed(seed + 17)
    d_h = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=g)

    inputs = (u, c, kappa, rho)
    inputs_r = (u_r, c_r, kappa_r, rho_r)
    inputs_pt = (u_pt, c_pt, kappa_pt, rho_pt)
    if initial_state is not None:
        inputs += (initial_state,)
        inputs_r += (initial_state_r,)
        inputs_pt += (initial_state_pt,)

    if return_final_state:
        d_final_state = torch.randn(
            final_state.shape,
            device=final_state.device,
            dtype=final_state.dtype,
            generator=g,
        )
        outputs = (h, final_state)
        outputs_ref = (h_ref, final_state_ref)
        outputs_pt = (h_pt, final_state_pt)
        grad_outputs = (d_h, d_final_state)
        grad_outputs_ref = (d_h.float(), d_final_state)
    else:
        outputs = h
        outputs_ref = h_ref
        outputs_pt = h_pt
        grad_outputs = d_h
        grad_outputs_ref = d_h.float()

    grads = torch.autograd.grad(outputs, inputs, grad_outputs)
    grads_ref = torch.autograd.grad(
        outputs_ref, inputs_r, grad_outputs_ref,
    )
    grads_pt = torch.autograd.grad(
        outputs_pt, inputs_pt, grad_outputs,
    )

    for grad, grad_pt, grad_ref, x in zip(
        grads, grads_pt, grads_ref, inputs,
    ):
        assert grad.dtype == x.dtype
        # Without initial state, the L=1 decay-gradient cancellation is amplified by 1 / rho.
        if u.shape[1] == 1 and initial_state is None and x is rho:
            torch.testing.assert_close(
                grad.float(), grad_ref.float(), atol=1e-2, rtol=1e-3,
            )
        else:
            _assert_ref_error(grad, grad_pt, grad_ref)


def _check_fwd_bwd(
    impl,
    B,
    L,
    R,
    M,
    chunk_size,
    dtype,
    input_gen,
    has_initial_state,
    *,
    seed=0,
    return_final_state=True,
    use_compile=False,
):
    u, c, kappa, rho, initial_state = _make_inputs(
        B, L, R, M, dtype, input_gen, seed=seed,
    )
    _check_fwd_bwd_inputs(
        impl,
        u,
        c,
        kappa,
        rho,
        chunk_size,
        initial_state if has_initial_state else None,
        seed=seed,
        return_final_state=return_final_state,
        use_compile=use_compile,
    )


# ============================================================================
# Main correctness sweep
# ============================================================================

@pytest.mark.parametrize("input_gen", list(LOG_RHO_GENS), ids=list(LOG_RHO_GENS))
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("chunk_size", [64, 128])
@pytest.mark.parametrize("L", [1, 32, 4001, 4096, 32768])
@pytest.mark.parametrize("M", [16])
@pytest.mark.parametrize("R", [64, 128, 192])
@pytest.mark.parametrize("B", [1, 17, 256])
@pytest.mark.parametrize("has_initial_state", [False, True], ids=["no_init", "init"])
def test_frac_scan(
    scan_impl, B, R, M, L, chunk_size, dtype, input_gen, has_initial_state,
):
    try:
        _check_fwd_bwd(
            scan_impl,
            B,
            L,
            R,
            M,
            chunk_size,
            dtype,
            input_gen,
            has_initial_state,
        )
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        pytest.skip("Skipping frac_scan case after CUDA OOM")


@pytest.mark.parametrize("chunk_size", [32, 128])
def test_frac_scan_compile(scan_impl, chunk_size):
    _check_fwd_bwd(
        scan_impl,
        B=2,
        L=4001,
        R=64,
        M=16,
        chunk_size=chunk_size,
        dtype=torch.bfloat16,
        input_gen="model_like",
        has_initial_state=True,
        use_compile=True,
    )


# ============================================================================
# API paths and supported chunk sizes
# ============================================================================

def test_model_scan_fwd_bwd():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from frac_ssm.models import modeling_frac
    from frac_ssm.models.configuration_frac import FracConfig

    config = FracConfig(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        frac_num_modes=8,
        frac_chunk_size=32,
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        mixer = modeling_frac.FracMixer(config)
    mixer = mixer.to(device="cuda", dtype=torch.bfloat16).train()
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        1, 33, 64, device="cuda", dtype=torch.bfloat16, generator=g,
        requires_grad=True,
    )

    out, (state, conv_state) = mixer(x, use_cache=True)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert state.shape == (1, 1, 8, 64) and state.dtype == torch.bfloat16
    assert conv_state.dtype == x.dtype
    assert torch.isfinite(out).all() and torch.isfinite(state).all()

    (out.float().square().mean() + state.square().mean()).backward()
    for value in (x, *mixer.parameters()):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("chunk_size", [32, 128])
def test_M_eight(scan_impl, chunk_size):
    _check_fwd_bwd(
        scan_impl,
        B=2,
        L=257,
        R=64,
        M=8,
        chunk_size=chunk_size,
        dtype=torch.bfloat16,
        input_gen="model_like",
        has_initial_state=True,
    )


@pytest.mark.parametrize("L", [1, 257])
@pytest.mark.parametrize("has_initial_state", [False, True], ids=["no_init", "init"])
def test_chunk_size_32(scan_impl, L, has_initial_state):
    _check_fwd_bwd(
        scan_impl,
        B=2,
        L=L,
        R=64,
        M=16,
        chunk_size=32,
        dtype=torch.bfloat16,
        input_gen="model_like",
        has_initial_state=has_initial_state,
    )


@pytest.mark.parametrize("chunk_size", [32, 128])
def test_return_h_only(scan_impl, chunk_size):
    _check_fwd_bwd(
        scan_impl,
        B=2,
        L=4001,
        R=64,
        M=16,
        chunk_size=chunk_size,
        dtype=torch.bfloat16,
        input_gen="model_like",
        has_initial_state=False,
        seed=11,
        return_final_state=False,
    )
    _check_fwd_bwd(
        scan_impl,
        B=2,
        L=4001,
        R=64,
        M=16,
        chunk_size=chunk_size,
        dtype=torch.bfloat16,
        input_gen="model_like",
        has_initial_state=True,
        seed=11,
        return_final_state=False,
    )


def test_masked_span(scan_impl):
    u, c, kappa, rho, initial_state = _make_inputs(
        2, 257, 64, 16, torch.bfloat16, "model_like", seed=19,
    )
    # Cross both chunk boundaries and cover one complete chunk.
    kappa[:, 61:132] = 0
    rho[:, 61:132] = 1
    _check_fwd_bwd_inputs(
        scan_impl,
        u,
        c,
        kappa,
        rho,
        64,
        initial_state,
        seed=19,
    )


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("has_initial_state", [False, True], ids=["no_init", "init"])
def test_fp32_coefficients(scan_impl, chunk_size, has_initial_state):
    if scan_impl.__name__ != _IMPL_MODULES["triton"] and chunk_size == 16:
        pytest.skip("cute_sm90 supports chunk sizes 32, 64, and 128")
    u, c, kappa, rho, initial_state = _make_inputs(
        2, 4001, 64, 16, torch.bfloat16, "model_like", seed=29,
    )
    _check_fwd_bwd_inputs(
        scan_impl, u, c.float(), kappa.float(), rho, chunk_size,
        initial_state if has_initial_state else None, seed=29,
    )


# ============================================================================
# Edge cases
# ============================================================================

def test_zero_batch(scan_impl):
    L, R, M, chunk_size = 128, 64, 16, 64
    u = torch.empty(
        0, L, R, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    c = torch.empty(
        0, L, M, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    kappa = torch.empty(
        0, L, M, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    rho = torch.empty(
        0, L, M, device="cuda", dtype=torch.float32, requires_grad=True,
    )
    initial_state = torch.empty(
        0, R, M, device="cuda", dtype=torch.float32, requires_grad=True,
    )

    h, final_state = scan_impl.frac_scan(
        u, c, kappa, rho, chunk_size, initial_state, True,
    )
    assert h.shape == (0, L, R)
    assert final_state.shape == (0, R, M)
    assert h.dtype == torch.bfloat16
    assert final_state.dtype == _state_dtype(scan_impl, u)

    torch.autograd.backward(
        (h, final_state), (torch.empty_like(h), torch.empty_like(final_state)),
    )
    assert u.grad.shape == u.shape
    assert c.grad.shape == c.shape
    assert kappa.grad.shape == kappa.shape
    assert rho.grad.shape == rho.shape
    assert initial_state.grad.shape == initial_state.shape


def test_strided_inputs(scan_impl):
    B, L, R, M, chunk_size = 2, 4001, 64, 16, 128
    dtype = torch.bfloat16
    u, c, kappa, rho, initial_state = _make_inputs(
        B, L, R, M, dtype, "model_like", seed=13,
    )

    u_storage = torch.empty(B, L, R + 4, device="cuda", dtype=dtype)
    c_storage = torch.empty(B, L, M + 5, device="cuda", dtype=dtype)
    kappa_storage = torch.empty(B, L, M + 5, device="cuda", dtype=dtype)
    rho_storage = torch.empty(B, L, M + 5, device="cuda", dtype=torch.float32)
    initial_storage = torch.empty(B, R, M + 5, device="cuda", dtype=torch.float32)
    u_storage[..., :R].copy_(u)
    c_storage[..., :M].copy_(c)
    kappa_storage[..., :M].copy_(kappa)
    rho_storage[..., :M].copy_(rho)
    initial_storage[..., :M].copy_(initial_state)

    u = u_storage[..., :R]
    c = c_storage[..., :M]
    kappa = kappa_storage[..., :M]
    rho = rho_storage[..., :M]
    initial_state = initial_storage[..., :M]
    assert not u.is_contiguous()
    assert not c.is_contiguous()
    assert not kappa.is_contiguous()
    assert not rho.is_contiguous()
    assert not initial_state.is_contiguous()
    _check_fwd_bwd_inputs(
        scan_impl, u, c, kappa, rho, chunk_size, initial_state, seed=13,
    )


def test_input_validation(scan_impl):
    u, c, kappa, rho, initial_state = _make_inputs(
        2, 32, 64, 16, torch.bfloat16, "model_like",
    )

    with pytest.raises(AssertionError, match="rho"):
        scan_impl.frac_scan(
            u, c, kappa, rho.to(torch.bfloat16), 64,
        )
    with pytest.raises(AssertionError, match="initial_state"):
        scan_impl.frac_scan(
            u, c, kappa, rho, 64, initial_state.to(torch.bfloat16),
        )
