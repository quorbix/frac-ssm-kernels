import copy

import torch

from common import (
    BenchmarkCall,
    BenchmarkResult,
    BenchmarkSpec,
    benchmark_main,
    DTYPE_MAP,
    KernelBenchmark,
    bandwidth_gbps,
    benchmark_calls,
    ensure_cuda,
    load_backend_modules,
    print_result,
)

from frac_ssm.layers.refs.compute_params import frac_compute_scan_params_ref
from frac_ssm.models.configuration_frac import FracConfig
from frac_ssm.models.modeling_frac import FracMixer


_BACKEND_MODULES = {
    "triton": "frac_ssm.layers.compute_params",
}


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


def _mem_bytes_fwd(B, L, H, R, M, dtype, has_mask):
    element_size = torch.empty((), dtype=dtype).element_size()
    D = H * R
    parameter_elements = 3 * H * D + 3 * H + 2 * H * M * D + H + 2 + M
    return (
        2 * B * L * D * element_size  # read x and x_conv
        + parameter_elements * element_size  # read projections and parameters
        + has_mask * B * L * 8  # read attention_mask (int64)
        + B * L * D * element_size  # write u
        + 3 * B * H * L * M * 4  # write c, kappa, and rho (fp32)
    )


def _mem_bytes_bwd(B, L, H, R, M, dtype, has_mask):
    element_size = torch.empty((), dtype=dtype).element_size()
    D = H * R
    parameter_elements = 3 * H * D + 3 * H + 2 * H * M * D + H + 2 + M
    return (
        B * L * D * element_size  # read d_u
        + 3 * B * H * L * M * 4  # read d_c, d_kappa, and d_rho
        + 2 * B * L * D * element_size  # read x and x_conv
        + parameter_elements * element_size  # read projections and parameters
        + has_mask * B * L * 8  # read attention_mask (int64)
        + 2 * B * L * D * element_size  # write d_x and d_x_conv
        + parameter_elements * element_size  # write parameter gradients
    )


def _make_inputs(B, L, H, R, dtype, has_mask, *, seed=0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    x_conv = 0.25 * torch.randn(
        B, L, H * R, device=device, dtype=dtype, generator=g,
    )
    x = 0.25 * torch.randn(
        B, L, H * R, device=device, dtype=dtype, generator=g,
    )
    attention_mask = None
    if has_mask:
        attention_mask = torch.ones(B, L, device=device, dtype=torch.long)
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


def run_compute_params(
    B, L, H, R, M, dtype_name,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    has_mask=True,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, H={H}, R={R}, M={M}], dtype={dtype_name}")
        print(f"Attention mask {has_mask}")

    torch_dtype = DTYPE_MAP[dtype_name]
    x_conv, x, attention_mask = _make_inputs(
        B, L, H, R, torch_dtype, has_mask,
    )
    model = _make_model(H, R, M, torch_dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if verbose:
        print("Input tensor shapes")
        print(f"x_conv {x_conv.shape}, dtype {x_conv.dtype}")
        print(f"x {x.shape}, dtype {x.dtype}")
        if attention_mask is not None:
            print(f"attention_mask {attention_mask.shape}, dtype {attention_mask.dtype}")

    mem_bytes = _mem_bytes_fwd(B, L, H, R, M, torch_dtype, has_mask)

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")
    kernel_calls = {
        name: BenchmarkCall(
            impl.frac_compute_scan_params,
            args=(model, x_conv, x, attention_mask),
        )
        for name, impl in impls.items()
    }

    compiled_ref = torch.compile(frac_compute_scan_params_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(model, x_conv, x, attention_mask),
        kwargs={"upcast": False},
    )
    kernel_times, ref_time = benchmark_calls(
        kernel_calls,
        reference,
        warmup_ms,
        rep_ms,
        preheat_ms=preheat_ms,
        warmup_launches=warmup_launches,
    )
    ref_ms = ref_time.ms
    ref_bw = bandwidth_gbps(mem_bytes, ref_ms)
    kernels = tuple(
        KernelBenchmark(name, timing.ms, bandwidth_gbps(mem_bytes, timing.ms), timing.metadata())
        for name, timing in kernel_times.items()
    )

    result = BenchmarkResult(
        ref_ms=ref_ms, ref_bw=ref_bw, kernels=kernels, ref_timing=ref_time.metadata(),
    )
    if verbose:
        print_result(result)
    return result


def run_compute_params_backward(
    B, L, H, R, M, dtype_name,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    has_mask=True,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, H={H}, R={R}, M={M}], dtype={dtype_name}")
        print(f"Attention mask {has_mask}")

    torch_dtype = DTYPE_MAP[dtype_name]
    x_conv, x, attention_mask = _make_inputs(
        B, L, H, R, torch_dtype, has_mask,
    )
    x_conv.requires_grad_()
    x.requires_grad_()
    x_conv_ref = x_conv.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    model = _make_model(H, R, M, torch_dtype)
    model_ref = copy.deepcopy(model)

    if verbose:
        print("Input tensor shapes")
        print(f"x_conv {x_conv.shape}, dtype {x_conv.dtype}")
        print(f"x {x.shape}, dtype {x.dtype}")
        if attention_mask is not None:
            print(f"attention_mask {attention_mask.shape}, dtype {attention_mask.dtype}")

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")

    forward_calls = {
        name: BenchmarkCall(
            impl.frac_compute_scan_params,
            args=(model, x_conv, x, attention_mask),
        )
        for name, impl in impls.items()
    }
    sample_call = next(iter(forward_calls.values()))
    output = sample_call.kernel(*sample_call.args, **sample_call.kwargs)
    g = torch.Generator(device="cuda").manual_seed(17)
    grad_outputs = tuple(
        0.1 * torch.randn(
            value.shape,
            device="cuda",
            dtype=value.dtype,
            generator=g,
        )
        for value in output
    )

    inputs = _model_inputs(model, x_conv, x)
    mem_bytes = _mem_bytes_bwd(B, L, H, R, M, torch_dtype, has_mask)
    del output
    kernel_calls = {
        name: call.backward(inputs, grad_outputs)
        for name, call in forward_calls.items()
    }

    compiled_ref = torch.compile(frac_compute_scan_params_ref)

    inputs_ref = _model_inputs(model_ref, x_conv_ref, x_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(model_ref, x_conv_ref, x_ref, attention_mask),
        kwargs={'upcast': False},
    ).backward(
        inputs_ref, grad_outputs,
    )
    kernel_times, ref_time = benchmark_calls(
        kernel_calls,
        reference,
        warmup_ms,
        rep_ms,
        preheat_ms=preheat_ms,
        warmup_launches=warmup_launches,
    )
    ref_ms = ref_time.ms
    ref_bw = bandwidth_gbps(mem_bytes, ref_ms)
    kernels = tuple(
        KernelBenchmark(name, timing.ms, bandwidth_gbps(mem_bytes, timing.ms), timing.metadata())
        for name, timing in kernel_times.items()
    )

    result = BenchmarkResult(
        ref_ms=ref_ms, ref_bw=ref_bw, kernels=kernels, ref_timing=ref_time.metadata(),
    )
    if verbose:
        print_result(result)
    return result


SPEC = BenchmarkSpec(
    name="compute_params",
    forward=run_compute_params,
    backward=run_compute_params_backward,
    backends=_BACKEND_MODULES,
    dimensions={"H": 12, "R": 128, "M": 16},
    dtypes=tuple(DTYPE_MAP),
    takes_dtype=True,
    options={"has_mask": True},
)


if __name__ == "__main__":
    raise SystemExit(benchmark_main(SPEC))
