
import torch

from frac_ssm.modules.rmsnorm import rmsnorm_ref

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


_BACKEND_MODULES = {
    "triton": "frac_ssm.modules.rmsnorm",
}

def _mem_bytes_fwd(B, L, D, dtype):
    element_size = torch.empty((), dtype=dtype).element_size()
    return (
        B * L * D * element_size  # read x
        + D * element_size  # read weight
        + B * L * D * element_size  # write output
    )


def _mem_bytes_bwd(B, L, D, dtype):
    element_size = torch.empty((), dtype=dtype).element_size()
    return (
        B * L * D * element_size  # read d_output
        + B * L * D * element_size  # read x
        + D * element_size  # read weight
        + B * L * D * element_size  # write d_x
        + D * element_size  # write d_weight
    )


def _make_inputs(B, L, D, dtype, seed=0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    x = 0.25 * torch.randn(
        B, L, D, device=device, dtype=dtype, generator=g
    )
    weight = 1 + 0.1 * torch.randn(
        D, device=device, dtype=dtype, generator=g
    )
    return x, weight


def run_rmsnorm(
    B, L, D, dtype_name,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    eps=1e-6,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, D={D}], dtype={dtype_name}")

    torch_dtype = DTYPE_MAP[dtype_name]
    x, weight = _make_inputs(B, L, D, torch_dtype)

    if verbose:
        print("Input tensor shapes")
        print(f"x {x.shape}, dtype {x.dtype}")
        print(f"weight {weight.shape}, dtype {weight.dtype}")

    mem_bytes = _mem_bytes_fwd(B, L, D, torch_dtype)

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")
    modules = {
        name: impl.RMSNorm(D, eps=eps).cuda().to(torch_dtype)
        for name, impl in impls.items()
    }
    for module in modules.values():
        with torch.no_grad():
            module.weight.copy_(weight)
        module.weight.requires_grad_(False)
    kernel_calls = {
        name: BenchmarkCall(
            module,
            args=(x,),
        )
        for name, module in modules.items()
    }

    compiled_ref = torch.compile(rmsnorm_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(x, weight, eps),
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


def run_rmsnorm_backward(
    B, L, D, dtype_name,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    eps=1e-6,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, D={D}], dtype={dtype_name}")

    torch_dtype = DTYPE_MAP[dtype_name]
    x, weight = _make_inputs(B, L, D, torch_dtype)
    x.requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()

    if verbose:
        print("Input tensor shapes")
        print(f"x {x.shape}, dtype {x.dtype}")
        print(f"weight {weight.shape}, dtype {weight.dtype}")

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")
    modules = {
        name: impl.RMSNorm(D, eps=eps).cuda().to(torch_dtype)
        for name, impl in impls.items()
    }
    for module in modules.values():
        with torch.no_grad():
            module.weight.copy_(weight)

    forward_calls = {
        name: BenchmarkCall(
            module,
            args=(x,),
        )
        for name, module in modules.items()
    }
    sample_call = next(iter(forward_calls.values()))
    output = sample_call.kernel(*sample_call.args, **sample_call.kwargs)
    g = torch.Generator(device="cuda").manual_seed(17)
    d_output = torch.randn(
        output.shape, device="cuda", dtype=output.dtype, generator=g
    )

    mem_bytes = _mem_bytes_bwd(B, L, D, torch_dtype)
    del output
    kernel_calls = {
        name: call.backward((x, modules[name].weight), d_output)
        for name, call in forward_calls.items()
    }

    compiled_ref = torch.compile(rmsnorm_ref)

    reference = BenchmarkCall(
        compiled_ref,
        args=(x_ref, weight_ref, eps),
        kwargs={'upcast': False},
    ).backward(
        (x_ref, weight_ref), d_output,
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
    name="rmsnorm",
    forward=run_rmsnorm,
    backward=run_rmsnorm_backward,
    backends=_BACKEND_MODULES,
    dimensions={"D": 768},
    dtypes=tuple(DTYPE_MAP),
    takes_dtype=True,
    sweep_dtypes=("bfloat16", "float32"),
    options={"eps": 1e-6},
)


if __name__ == "__main__":
    raise SystemExit(benchmark_main(SPEC))
