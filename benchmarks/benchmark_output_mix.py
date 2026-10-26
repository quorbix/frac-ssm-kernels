
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

from frac_ssm.layers.refs.output_mix import frac_skip_mix_ref
from frac_ssm.models.configuration_frac import FracConfig
from frac_ssm.models.modeling_frac import FracMixer


_BACKEND_MODULES = {
    "triton": "frac_ssm.layers.output_mix",
}


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


def _mem_bytes_fwd(B, L, H, R, dtype):
    element_size = torch.empty((), dtype=dtype).element_size()
    return (
        B * H * L * R * element_size  # read mixed_flat
        + B * L * H * R * element_size  # read x_conv
        + H * R * element_size  # read D
        + B * L * H * R * element_size  # write mixed
    )


def _mem_bytes_bwd(B, L, H, R, dtype):
    element_size = torch.empty((), dtype=dtype).element_size()
    return (
        B * L * H * R * element_size  # read d_mixed
        + B * L * H * R * element_size  # read x_conv
        + H * R * element_size  # read D
        + B * H * L * R * element_size  # write d_mixed_flat
        + B * L * H * R * element_size  # write d_x_conv
        + H * R * element_size  # write d_D
    )


def _make_inputs(B, L, H, R, dtype, *, seed=0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    mixed_flat = 0.25 * torch.randn(
        B * H, L, R, device=device, dtype=dtype, generator=g,
    )
    x_conv = 0.25 * torch.randn(
        B, L, H * R, device=device, dtype=dtype, generator=g,
    )
    return mixed_flat, x_conv


def run_output_mix(
    B, L, H, R, dtype_name,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, H={H}, R={R}], dtype={dtype_name}")

    torch_dtype = DTYPE_MAP[dtype_name]
    mixed_flat, x_conv = _make_inputs(B, L, H, R, torch_dtype)
    model = _make_model(H, R, torch_dtype)
    model.D.requires_grad_(False)

    if verbose:
        print("Input tensor shapes")
        print(f"mixed_flat {mixed_flat.shape}, dtype {mixed_flat.dtype}")
        print(f"x_conv {x_conv.shape}, dtype {x_conv.dtype}")
        print(f"D {model.D.shape}, dtype {model.D.dtype}")

    mem_bytes = _mem_bytes_fwd(B, L, H, R, torch_dtype)

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")
    kernel_calls = {
        name: BenchmarkCall(
            impl.frac_skip_mix,
            args=(model, mixed_flat, x_conv),
        )
        for name, impl in impls.items()
    }

    compiled_ref = torch.compile(frac_skip_mix_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(model, mixed_flat, x_conv),
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


def run_output_mix_backward(
    B, L, H, R, dtype_name,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, H={H}, R={R}], dtype={dtype_name}")

    torch_dtype = DTYPE_MAP[dtype_name]
    mixed_flat, x_conv = _make_inputs(B, L, H, R, torch_dtype)
    mixed_flat.requires_grad_()
    x_conv.requires_grad_()
    mixed_flat_ref = mixed_flat.detach().clone().requires_grad_()
    x_conv_ref = x_conv.detach().clone().requires_grad_()
    model = _make_model(H, R, torch_dtype)
    model_ref = _make_model(H, R, torch_dtype)
    with torch.no_grad():
        model_ref.D.copy_(model.D)

    if verbose:
        print("Input tensor shapes")
        print(f"mixed_flat {mixed_flat.shape}, dtype {mixed_flat.dtype}")
        print(f"x_conv {x_conv.shape}, dtype {x_conv.dtype}")
        print(f"D {model.D.shape}, dtype {model.D.dtype}")

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")

    forward_calls = {
        name: BenchmarkCall(
            impl.frac_skip_mix,
            args=(model, mixed_flat, x_conv),
        )
        for name, impl in impls.items()
    }
    sample_call = next(iter(forward_calls.values()))
    output = sample_call.kernel(*sample_call.args, **sample_call.kwargs)
    g = torch.Generator(device="cuda").manual_seed(17)
    d_mixed = 0.1 * torch.randn(
        output.shape, device="cuda", dtype=output.dtype, generator=g,
    )

    inputs = (mixed_flat, x_conv, model.D)
    mem_bytes = _mem_bytes_bwd(B, L, H, R, torch_dtype)
    del output
    kernel_calls = {
        name: call.backward(inputs, d_mixed)
        for name, call in forward_calls.items()
    }

    compiled_ref = torch.compile(frac_skip_mix_ref)

    inputs_ref = (mixed_flat_ref, x_conv_ref, model_ref.D)
    reference = BenchmarkCall(
        compiled_ref,
        args=(model_ref, mixed_flat_ref, x_conv_ref),
        kwargs={'upcast': False},
    ).backward(
        inputs_ref, d_mixed,
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
    name="output_mix",
    forward=run_output_mix,
    backward=run_output_mix_backward,
    backends=_BACKEND_MODULES,
    dimensions={"H": 12, "R": 128},
    dtypes=tuple(DTYPE_MAP),
    takes_dtype=True,
)


if __name__ == "__main__":
    raise SystemExit(benchmark_main(SPEC))
