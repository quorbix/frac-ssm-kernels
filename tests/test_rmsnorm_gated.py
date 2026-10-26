import pytest
import torch

from frac_ssm.modules.rmsnorm_gated import (
    MambaRMSNormGated,
    rmsnorm_gated_ref,
)
from tests.utils import is_cuda_oom


def _assert_ref_error(out, out_pt, out_ref, *, atol=1e-3):
    assert out.shape == out_ref.shape
    error = (out.float() - out_ref.float()).abs().max().item()
    error_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert error <= 2 * error_pt + atol


def _make_input(shape, dtype, *, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = 0.25 * torch.randn(
        shape, device="cuda", dtype=dtype, generator=g,
    )
    gate = 0.5 * torch.randn(
        shape, device="cuda", dtype=dtype, generator=g,
    )
    return x, gate


def _check_fwd_bwd(
    x,
    gate,
    *,
    eps=1e-6,
    seed=0,
    use_compile=False,
    strided_grad=False,
):
    hidden_size = x.shape[-1]
    g = torch.Generator(device=x.device).manual_seed(seed + 1)
    weight = 1 + 0.1 * torch.randn(
        hidden_size, device=x.device, dtype=x.dtype, generator=g,
    )

    x = x.detach().requires_grad_()
    if gate is not None:
        gate = gate.detach().requires_grad_()
    module = MambaRMSNormGated(hidden_size, eps=eps).to(
        device=x.device, dtype=x.dtype,
    )
    with torch.no_grad():
        module.weight.copy_(weight)

    fn = module
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)

    out = fn(x, gate)

    x_pt = x.detach().clone().requires_grad_()
    weight_pt = weight.detach().clone().requires_grad_()
    gate_pt = (
        gate.detach().clone().requires_grad_() if gate is not None else None
    )
    x_ref = x.detach().double().requires_grad_()
    weight_ref = weight.detach().double().requires_grad_()
    gate_ref = (
        gate.detach().double().requires_grad_() if gate is not None else None
    )

    out_pt = rmsnorm_gated_ref(
        x_pt, weight_pt, gate_pt, eps, upcast=False,
    )
    out_ref = rmsnorm_gated_ref(
        x_ref, weight_ref, gate_ref, eps, upcast=True,
    )

    assert out.dtype == x.dtype
    _assert_ref_error(out, out_pt, out_ref)

    if strided_grad:
        d_out_storage = torch.randn(
            *out.shape[:-1], out.shape[-1] + 5,
            device=out.device,
            dtype=out.dtype,
            generator=g,
        )
        d_out = d_out_storage[..., :out.shape[-1]]
        assert not d_out.is_contiguous()
    else:
        d_out = torch.randn(
            out.shape, device=out.device, dtype=out.dtype, generator=g,
        )
    inputs = (x, module.weight)
    inputs_pt = (x_pt, weight_pt)
    inputs_ref = (x_ref, weight_ref)
    if gate is not None:
        inputs += (gate,)
        inputs_pt += (gate_pt,)
        inputs_ref += (gate_ref,)

    grads = torch.autograd.grad(out, inputs, d_out)
    grads_pt = torch.autograd.grad(out_pt, inputs_pt, d_out)
    grads_ref = torch.autograd.grad(
        out_ref, inputs_ref, d_out.double(),
    )

    for grad, grad_pt, grad_ref, value in zip(
        grads, grads_pt, grads_ref, inputs,
    ):
        assert grad.dtype == value.dtype
        _assert_ref_error(grad, grad_pt, grad_ref)


# ============================================================================
# Main correctness sweep
# ============================================================================

@pytest.mark.parametrize("has_gate", [False, True], ids=["no_gate", "gate"])
@pytest.mark.parametrize("dtype",    [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("D",        [128, 768])
@pytest.mark.parametrize("L",        [32, 4001, 32768])
@pytest.mark.parametrize("B",        [1, 17])
def test_rmsnorm_gated(B, L, D, dtype, has_gate):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    try:
        x, gate = _make_input((B, L, D), dtype)
        _check_fwd_bwd(x, gate if has_gate else None)
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        pytest.skip("Skipping rmsnorm_gated case after CUDA OOM")


def test_rmsnorm_gated_leading_dimensions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    x, gate = _make_input((2, 3, 17, 64), torch.bfloat16, seed=3)
    _check_fwd_bwd(x, gate, seed=3)


def test_rmsnorm_gated_compile():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    x, gate = _make_input((2, 127, 768), torch.bfloat16, seed=5)
    _check_fwd_bwd(x, gate, seed=5, use_compile=True)


def test_rmsnorm_gated_strided_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    shape = (2, 127, 768)
    x, gate = _make_input(shape, torch.bfloat16, seed=7)
    x_storage = torch.empty(
        *shape[:-1], shape[-1] + 5,
        device="cuda",
        dtype=x.dtype,
    )
    gate_storage = torch.empty_like(x_storage)
    x_storage[..., :shape[-1]].copy_(x)
    gate_storage[..., :shape[-1]].copy_(gate)
    x = x_storage[..., :shape[-1]]
    gate = gate_storage[..., :shape[-1]]
    assert not x.is_contiguous()
    assert not gate.is_contiguous()
    _check_fwd_bwd(x, gate, seed=7, strided_grad=True)


@pytest.mark.parametrize("has_gate", [False, True], ids=["ungated", "gated"])
def test_rmsnorm_gated_zero_batch(has_gate):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    x = torch.empty(
        0, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    gate = torch.empty_like(x, requires_grad=True) if has_gate else None
    module = MambaRMSNormGated(64).cuda().to(torch.bfloat16)
    out = module(x, gate)
    assert out.shape == x.shape
    torch.autograd.backward(out, torch.empty_like(out))
    assert x.grad.shape == x.shape
    if gate is not None:
        assert gate.grad.shape == gate.shape
    torch.testing.assert_close(module.weight.grad, torch.zeros_like(module.weight))
