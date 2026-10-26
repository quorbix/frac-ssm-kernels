import copy
import importlib

import pytest
import torch

from frac_ssm.layers.refs.output_mix import frac_skip_mix_ref
from frac_ssm.models.configuration_frac import FracConfig
from frac_ssm.models.modeling_frac import FracMixer
from tests.utils import is_cuda_oom


_IMPL_MODULES = {
    "triton": "frac_ssm.layers.output_mix",
}


@pytest.fixture(params=tuple(_IMPL_MODULES), ids=tuple(_IMPL_MODULES))
def output_mix_impl(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    return importlib.import_module(_IMPL_MODULES[request.param])


def _make_model(H, R, dtype, *, seed=0):
    config = FracConfig(
        hidden_size=H * R,
        intermediate_size=H * R,
        num_hidden_layers=1,
        num_attention_heads=H,
        frac_num_modes=8,
        frac_head_dim=R,
        frac_write_scale_init=0.5,
    )
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = FracMixer(config)
    model = model.to(device="cuda", dtype=dtype)

    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        model.D.copy_(0.1 * torch.randn_like(model.D, generator=g))
    return model


def _assert_ref_error(out, out_pt, out_ref, *, atol=1e-4):
    assert out.shape == out_ref.shape
    if out.numel() == 0:
        return
    error = (out.float() - out_ref.float()).abs().max().item()
    error_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert error <= 2 * error_pt + atol


def _make_inputs(B, L, H, R, dtype, *, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mixed_flat = 0.25 * torch.randn(
        B * H, L, R, device="cuda", dtype=dtype, generator=g,
    )
    x_conv = 0.25 * torch.randn(
        B, L, H * R, device="cuda", dtype=dtype, generator=g,
    )
    return mixed_flat, x_conv


def _check_fwd_bwd(
    impl,
    model,
    mixed_flat,
    x_conv,
    *,
    out_dtype=None,
    seed=0,
    use_compile=False,
    strided_grad=False,
):
    mixed_flat = mixed_flat.detach().requires_grad_()
    x_conv = x_conv.detach().requires_grad_()

    fn = lambda mf, xc: impl.frac_skip_mix(
        model, mf, xc, out_dtype=out_dtype,
    )
    if use_compile:
        fn = torch.compile(fn, fullgraph=True)

    out = fn(mixed_flat, x_conv)

    model_pt = copy.deepcopy(model)
    mixed_flat_pt = mixed_flat.detach().clone().requires_grad_()
    x_conv_pt = x_conv.detach().clone().requires_grad_()

    model_ref = copy.deepcopy(model).double()
    mixed_flat_ref = mixed_flat.detach().double().requires_grad_()
    x_conv_ref = x_conv.detach().double().requires_grad_()

    out_pt = frac_skip_mix_ref(
        model_pt,
        mixed_flat_pt,
        x_conv_pt,
        out_dtype,
        upcast=False,
    )
    out_ref = frac_skip_mix_ref(
        model_ref,
        mixed_flat_ref,
        x_conv_ref,
        out_dtype,
        upcast=True,
    )

    assert out.dtype == (out_dtype or x_conv.dtype)
    _assert_ref_error(out, out_pt, out_ref)

    g = torch.Generator(device=out.device).manual_seed(seed + 17)
    if strided_grad:
        d_out_storage = torch.randn(
            *out.shape[:-1],
            out.shape[-1] + 5,
            device=out.device,
            dtype=out.dtype,
            generator=g,
        )
        d_out_storage.mul_(0.1)
        d_out = d_out_storage[..., :out.shape[-1]]
        assert not d_out.is_contiguous()
    else:
        d_out = 0.1 * torch.randn(
            out.shape,
            device=out.device,
            dtype=out.dtype,
            generator=g,
        )

    inputs = (mixed_flat, x_conv, model.D)
    inputs_pt = (mixed_flat_pt, x_conv_pt, model_pt.D)
    inputs_ref = (mixed_flat_ref, x_conv_ref, model_ref.D)
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

@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("R",     [64, 128])
@pytest.mark.parametrize("H",     [2, 12])
@pytest.mark.parametrize("L",     [32, 4001, 32768])
@pytest.mark.parametrize("B",     [1, 17])
def test_output_mix(output_mix_impl, B, L, H, R, dtype):
    try:
        model = _make_model(H, R, dtype)
        inputs = _make_inputs(B, L, H, R, dtype)
        _check_fwd_bwd(output_mix_impl, model, *inputs)
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        pytest.skip("Skipping output_mix case after CUDA OOM")


def test_output_mix_compile(output_mix_impl):
    H, R = 2, 64
    model = _make_model(H, R, torch.bfloat16, seed=3)
    inputs = _make_inputs(2, 257, H, R, torch.bfloat16, seed=3)
    _check_fwd_bwd(
        output_mix_impl, model, *inputs, seed=3, use_compile=True,
    )


# ============================================================================
# Output dtype
# ============================================================================

def test_float32_output(output_mix_impl):
    H, R = 2, 64
    model = _make_model(H, R, torch.bfloat16, seed=5)
    inputs = _make_inputs(2, 257, H, R, torch.bfloat16, seed=5)
    _check_fwd_bwd(
        output_mix_impl,
        model,
        *inputs,
        out_dtype=torch.float32,
        seed=5,
    )


# ============================================================================
# Edge cases
# ============================================================================

def test_L_one(output_mix_impl):
    H, R = 2, 64
    model = _make_model(H, R, torch.bfloat16, seed=7)
    inputs = _make_inputs(2, 1, H, R, torch.bfloat16, seed=7)
    _check_fwd_bwd(output_mix_impl, model, *inputs, seed=7)


def test_strided_inputs(output_mix_impl):
    B, L, H, R = 2, 257, 2, 64
    model = _make_model(H, R, torch.bfloat16, seed=11)
    mixed_flat, x_conv = _make_inputs(
        B, L, H, R, torch.bfloat16, seed=11,
    )

    mixed_storage = torch.empty(
        B * H, L, R + 5, device="cuda", dtype=mixed_flat.dtype,
    )
    x_conv_storage = torch.empty(
        B, L, H * R + 5, device="cuda", dtype=x_conv.dtype,
    )
    mixed_storage[..., :R].copy_(mixed_flat)
    x_conv_storage[..., :H * R].copy_(x_conv)
    mixed_flat = mixed_storage[..., :R]
    x_conv = x_conv_storage[..., :H * R]

    assert not mixed_flat.is_contiguous()
    assert not x_conv.is_contiguous()
    _check_fwd_bwd(
        output_mix_impl,
        model,
        mixed_flat,
        x_conv,
        seed=11,
        strided_grad=True,
    )


def test_zero_batch(output_mix_impl):
    L, H, R = 127, 2, 64
    model = _make_model(H, R, torch.bfloat16, seed=13)
    mixed_flat = torch.empty(
        0, L, R, device="cuda", dtype=torch.bfloat16,
    )
    x_conv = torch.empty(
        0, L, H * R, device="cuda", dtype=torch.bfloat16,
    )
    _check_fwd_bwd(
        output_mix_impl, model, mixed_flat, x_conv, seed=13,
    )
