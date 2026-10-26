import pytest
import torch

from frac_ssm.modules.rmsnorm import RMSNorm, rmsnorm_ref
from tests.utils import is_cuda_oom


def _assert_ref_error(out, out_pt, out_ref, *, atol=1e-3):
    assert out.shape == out_ref.shape
    error = (out.float() - out_ref.float()).abs().max().item()
    error_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert error <= 2 * error_pt + atol


def _make_input(shape, dtype, *, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return 0.25 * torch.randn(
        shape, device=device, dtype=dtype, generator=g,
    )


def _check_fwd_bwd(
    x,
    *,
    eps=1e-6,
    seed=0,
    row_mode=None,
    use_compile=False,
    weight_dtype=None,
    strided_grad=False,
):
    hidden_size = x.shape[-1]
    weight_dtype = x.dtype if weight_dtype is None else weight_dtype
    g = torch.Generator(device=x.device).manual_seed(seed + 1)
    weight = 1 + 0.1 * torch.randn(
        hidden_size, device=x.device, dtype=weight_dtype, generator=g,
    )

    x = x.detach().requires_grad_()
    module = RMSNorm(
        hidden_size,
        eps=eps,
        row_mode=row_mode,
    ).to(
        device=x.device, dtype=weight_dtype,
    )
    with torch.no_grad():
        module.weight.copy_(weight)

    fn = module
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)

    out = fn(x)

    x_pt = x.detach().clone().requires_grad_()
    weight_pt = weight.detach().clone().requires_grad_()
    x_ref = x.detach().double().requires_grad_()
    weight_ref = weight.detach().double().requires_grad_()

    out_pt = rmsnorm_ref(x_pt, weight_pt, eps, upcast=False)
    out_ref = rmsnorm_ref(x_ref, weight_ref, eps, upcast=True)

    assert out.dtype == out_pt.dtype
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
    grads = torch.autograd.grad(out, (x, module.weight), d_out)
    grads_pt = torch.autograd.grad(out_pt, (x_pt, weight_pt), d_out)
    grads_ref = torch.autograd.grad(
        out_ref, (x_ref, weight_ref), d_out.double(),
    )

    for grad, grad_pt, grad_ref, value in zip(
        grads, grads_pt, grads_ref, (x, module.weight),
    ):
        assert grad.dtype == value.dtype
        _assert_ref_error(grad, grad_pt, grad_ref)


# ============================================================================
# Main correctness sweep
# ============================================================================

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("D",     [256, 768])
@pytest.mark.parametrize("L",     [32, 4001, 32768])
@pytest.mark.parametrize("B",     [1, 17])
def test_rmsnorm(B, L, D, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    try:
        _check_fwd_bwd(_make_input((B, L, D), dtype))
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        pytest.skip("Skipping rmsnorm case after CUDA OOM")


@pytest.mark.parametrize("row_mode", [True, False], ids=["row", "block"])
def test_rmsnorm_kernel_modes(row_mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    x = _make_input((2, 97, 128), torch.bfloat16, seed=3)
    _check_fwd_bwd(x, row_mode=row_mode, seed=3)


def test_rmsnorm_compile():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    x = _make_input((2, 127, 768), torch.bfloat16, seed=5)
    _check_fwd_bwd(x, seed=5, use_compile=True)


def test_rmsnorm_strided_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    shape = (2, 127, 768)
    x = _make_input(shape, torch.bfloat16, seed=7)
    storage = torch.empty(
        *shape[:-1], shape[-1] + 5,
        device="cuda",
        dtype=x.dtype,
    )
    storage[..., :shape[-1]].copy_(x)
    x = storage[..., :shape[-1]]
    assert not x.is_contiguous()
    _check_fwd_bwd(x, seed=7, strided_grad=True)


def test_rmsnorm_mixed_weight_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    x = _make_input((2, 127, 768), torch.bfloat16, seed=9)
    _check_fwd_bwd(x, seed=9, weight_dtype=torch.float32)


def test_rmsnorm_zero_batch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    x = torch.empty(
        0, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    module = RMSNorm(64).cuda().to(torch.bfloat16)
    out = module(x)
    assert out.shape == x.shape
    torch.autograd.backward(out, torch.empty_like(out))
    assert x.grad.shape == x.shape
    torch.testing.assert_close(module.weight.grad, torch.zeros_like(module.weight))
