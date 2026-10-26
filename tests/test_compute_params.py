import copy
import importlib

import pytest
import torch

from frac_ssm.layers.refs.compute_params import frac_compute_scan_params_ref
from frac_ssm.models.configuration_frac import FracConfig
from frac_ssm.models.modeling_frac import FracMixer
from tests.utils import is_cuda_oom


_IMPL_MODULES = {
    "triton": "frac_ssm.layers.compute_params",
}


@pytest.fixture(params=tuple(_IMPL_MODULES), ids=tuple(_IMPL_MODULES))
def compute_params_impl(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    return importlib.import_module(_IMPL_MODULES[request.param])


def _make_model(H, R, M, dtype, *, seed=0):
    config = FracConfig(
        hidden_size=H * R,
        intermediate_size=H * R,
        num_hidden_layers=1,
        num_attention_heads=H,
        frac_num_modes=M,
        frac_head_dim=R,
        frac_write_scale_init=0.5,
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = FracMixer(config)
    return model.to(device="cuda", dtype=dtype)


def _assert_ref_error(out, out_pt, out_ref, *, atol=1e-4):
    assert out.shape == out_ref.shape
    if out.numel() == 0:
        return
    error = (out.float() - out_ref.float()).abs().max().item()
    error_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert error <= 2 * error_pt + atol


def _make_inputs(B, L, H, R, dtype, has_mask, *, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x_conv = 0.25 * torch.randn(
        B, L, H * R, device="cuda", dtype=dtype, generator=g,
    )
    x = 0.25 * torch.randn(
        B, L, H * R, device="cuda", dtype=dtype, generator=g,
    )
    attention_mask = None
    if has_mask:
        attention_mask = torch.ones(B, L, device="cuda", dtype=torch.long)
        attention_mask[:, L // 3:L // 2] = 0
        attention_mask[:, -1] = 0
    return x_conv, x, attention_mask


def _model_inputs(model, x_conv, x):
    return (
        x_conv,
        x,
        model.dyn_proj.weight,
        model.dyn_proj.bias,
        model.rw_proj.weight,
        model.dt_bias,
        model.read_residual_logit,
        model.write_residual_scale,
    )


def _make_grad(output, generator, *, strided):
    if not strided:
        return 0.1 * torch.randn(
            output.shape,
            device=output.device,
            dtype=output.dtype,
            generator=generator,
        )

    storage = torch.randn(
        *output.shape[:-1],
        output.shape[-1] + 3,
        device=output.device,
        dtype=output.dtype,
        generator=generator,
    )
    storage.mul_(0.1)
    grad = storage[..., :output.shape[-1]]
    assert not grad.is_contiguous()
    return grad


def _check_fwd_bwd(
    impl,
    model,
    x_conv,
    x,
    attention_mask,
    *,
    seed=0,
    use_compile=False,
    strided_grad=False,
):
    x_conv = x_conv.detach().requires_grad_()
    x = x.detach().requires_grad_()

    fn = lambda xc, xi, mask: impl.frac_compute_scan_params(
        model, xc, xi, mask,
    )
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)

    out = fn(x_conv, x, attention_mask)

    model_pt = copy.deepcopy(model)
    x_conv_pt = x_conv.detach().clone().requires_grad_()
    x_pt = x.detach().clone().requires_grad_()

    model_ref = copy.deepcopy(model).double()
    x_conv_ref = x_conv.detach().double().requires_grad_()
    x_ref = x.detach().double().requires_grad_()

    out_pt = frac_compute_scan_params_ref(
        model_pt, x_conv_pt, x_pt, attention_mask, upcast=False,
    )
    out_ref = frac_compute_scan_params_ref(
        model_ref, x_conv_ref, x_ref, attention_mask, upcast=True,
    )

    assert out[0].dtype == x_conv.dtype
    for value in out[1:]:
        assert value.dtype == torch.float32
    for value, value_pt, value_ref in zip(out, out_pt, out_ref):
        _assert_ref_error(value, value_pt, value_ref)

    g = torch.Generator(device=x.device).manual_seed(seed + 17)
    grad_outputs = tuple(
        _make_grad(value, g, strided=strided_grad) for value in out
    )
    inputs = _model_inputs(model, x_conv, x)
    inputs_pt = _model_inputs(model_pt, x_conv_pt, x_pt)
    inputs_ref = _model_inputs(model_ref, x_conv_ref, x_ref)

    grads = torch.autograd.grad(out, inputs, grad_outputs)
    grads_pt = torch.autograd.grad(out_pt, inputs_pt, grad_outputs)
    grads_ref = torch.autograd.grad(
        out_ref,
        inputs_ref,
        tuple(grad.double() for grad in grad_outputs),
    )

    for grad, grad_pt, grad_ref, value in zip(
        grads, grads_pt, grads_ref, inputs,
    ):
        assert grad.dtype == value.dtype
        _assert_ref_error(grad, grad_pt, grad_ref)


# ============================================================================
# Main correctness sweep
# ============================================================================

@pytest.mark.parametrize("has_mask", [False, True], ids=["no_mask", "mask"])
@pytest.mark.parametrize("dtype",    [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("M",        [8, 16])
@pytest.mark.parametrize("R",        [64, 128, 512])
@pytest.mark.parametrize("H",        [1, 4, 12])
@pytest.mark.parametrize("L",        [32, 4001, 32768])
@pytest.mark.parametrize("B",        [1, 17])
def test_compute_params(
    compute_params_impl, B, L, H, R, M, dtype, has_mask,
):
    try:
        model = _make_model(H, R, M, dtype)
        inputs = _make_inputs(B, L, H, R, dtype, has_mask)
        _check_fwd_bwd(compute_params_impl, model, *inputs)
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        pytest.skip("Skipping compute_params case after CUDA OOM")


def test_compute_params_compile(compute_params_impl):
    H, R, M = 2, 64, 8
    model = _make_model(H, R, M, torch.bfloat16, seed=3)
    inputs = _make_inputs(
        2, 257, H, R, torch.bfloat16, True, seed=3,
    )
    _check_fwd_bwd(
        compute_params_impl, model, *inputs, seed=3, use_compile=True,
    )


# ============================================================================
# Mask semantics
# ============================================================================

def test_masked_positions_are_recurrence_identities(compute_params_impl):
    B, L, H, R, M = 2, 127, 2, 64, 8
    model = _make_model(H, R, M, torch.bfloat16, seed=5)
    x_conv, x, attention_mask = _make_inputs(
        B, L, H, R, torch.bfloat16, True, seed=5,
    )
    _, _, kappa, rho = compute_params_impl.frac_compute_scan_params(
        model, x_conv, x, attention_mask,
    )
    mask = attention_mask[:, None, :, None].expand(B, H, L, 1)
    mask = mask.reshape(B * H, L, 1).expand_as(kappa)

    torch.testing.assert_close(kappa[~mask.bool()], torch.zeros_like(kappa[~mask.bool()]))
    torch.testing.assert_close(rho[~mask.bool()], torch.ones_like(rho[~mask.bool()]))


# ============================================================================
# Edge cases
# ============================================================================

def test_L_one_and_clamped_parameters(compute_params_impl):
    H, R, M = 2, 64, 8
    model = _make_model(H, R, M, torch.bfloat16, seed=7)
    with torch.no_grad():
        model.dt_bias.copy_(
            torch.tensor([-20.0, 20.0], device="cuda", dtype=torch.bfloat16)
        )
        model.read_residual_logit.fill_(12.0)
        model.write_residual_scale.fill_(-12.0)
    inputs = _make_inputs(2, 1, H, R, torch.bfloat16, True, seed=7)
    _check_fwd_bwd(compute_params_impl, model, *inputs, seed=7)


def test_strided_inputs(compute_params_impl):
    B, L, H, R, M = 2, 257, 2, 64, 8
    D = H * R
    model = _make_model(H, R, M, torch.bfloat16, seed=11)
    x_conv, x, attention_mask = _make_inputs(
        B, L, H, R, torch.bfloat16, True, seed=11,
    )

    x_conv_storage = torch.empty(
        B, L, D + 5, device="cuda", dtype=x_conv.dtype,
    )
    x_storage = torch.empty_like(x_conv_storage)
    mask_storage = torch.empty(
        B, 2 * L, device="cuda", dtype=attention_mask.dtype,
    )
    x_conv_storage[..., :D].copy_(x_conv)
    x_storage[..., :D].copy_(x)
    mask_storage[:, ::2].copy_(attention_mask)
    x_conv = x_conv_storage[..., :D]
    x = x_storage[..., :D]
    attention_mask = mask_storage[:, ::2]

    assert not x_conv.is_contiguous()
    assert not x.is_contiguous()
    assert not attention_mask.is_contiguous()
    _check_fwd_bwd(
        compute_params_impl,
        model,
        x_conv,
        x,
        attention_mask,
        seed=11,
        strided_grad=True,
    )


def test_zero_batch(compute_params_impl):
    L, H, R, M = 127, 2, 64, 8
    model = _make_model(H, R, M, torch.bfloat16, seed=13)
    x_conv = torch.empty(
        0, L, H * R, device="cuda", dtype=torch.bfloat16,
    )
    x = torch.empty_like(x_conv)
    attention_mask = torch.empty(0, L, device="cuda", dtype=torch.long)
    _check_fwd_bwd(
        compute_params_impl,
        model,
        x_conv,
        x,
        attention_mask,
        seed=13,
    )
