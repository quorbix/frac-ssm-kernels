import math

import torch

from common import (
    BenchmarkCall,
    BenchmarkResult,
    BenchmarkSpec,
    KernelBenchmark,
    bandwidth_gbps,
    benchmark_calls,
    benchmark_main,
    ensure_cuda,
    load_backend_modules,
    print_result,
)

from frac_ssm.ops.refs.chunk_cumsum import chunk_cumsum_ref


_BACKEND_MODULES = {
    "triton": "frac_ssm.ops.triton.chunk_cumsum",
    "cute_sm90": "frac_ssm.ops.cute.chunk_cumsum",
}


def _mem_bytes_fwd(B, L, M, chunk_size):
    nchunks = math.ceil(L / chunk_size)
    return (
        B * L * M * 4  # read log_rho (fp32)
        + B * L * M * 4  # write log_cum_rho (fp32)
        + B * nchunks * M * 4  # write chunk_decay (fp32)
    )


def _mem_bytes_bwd(B, L, M, chunk_size):
    nchunks = math.ceil(L / chunk_size)
    return (
        B * L * M * 4  # read d_log_cum_rho (fp32)
        + B * nchunks * M * 4  # read d_chunk_decay (fp32)
        + B * L * M * 4  # write d_log_rho (fp32)
    )


def run_chunk_cumsum(
    B, L, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, M={M}], chunk_size={chunk_size}")

    device = "cuda"
    log_rho = torch.randn(B, L, M, device=device, dtype=torch.float32)

    if verbose:
        print("Input tensor shapes")
        print(f"log_rho {log_rho.shape}, dtype {log_rho.dtype}")

    mem_bytes = _mem_bytes_fwd(B, L, M, chunk_size)

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")
    kernel_calls = {
        name: BenchmarkCall(
            impl.chunk_cumsum,
            args=(log_rho, chunk_size),
        )
        for name, impl in impls.items()
    }

    compiled_ref = torch.compile(chunk_cumsum_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(log_rho, chunk_size),
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


def run_chunk_cumsum_backward(
    B, L, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, M={M}], chunk_size={chunk_size}")

    device = "cuda"
    log_rho = torch.randn(B, L, M, device=device, dtype=torch.float32, requires_grad=True)
    log_rho_ref = log_rho.detach().clone().requires_grad_()

    if verbose:
        print("Input tensor shapes")
        print(f"log_rho {log_rho.shape}, dtype {log_rho.dtype}")

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")

    forward_calls = {
        name: BenchmarkCall(
            impl.chunk_cumsum,
            args=(log_rho, chunk_size),
        )
        for name, impl in impls.items()
    }
    sample_call = next(iter(forward_calls.values()))
    (log_cum_rho, chunk_decay) = sample_call.kernel(*sample_call.args, **sample_call.kwargs)
    d_lcr = torch.randn_like(log_cum_rho)
    d_cd = torch.randn_like(chunk_decay)

    mem_bytes = _mem_bytes_bwd(B, L, M, chunk_size)
    del log_cum_rho, chunk_decay
    kernel_calls = {
        name: call.backward(log_rho, (d_lcr, d_cd))
        for name, call in forward_calls.items()
    }

    compiled_ref = torch.compile(chunk_cumsum_ref)

    reference = BenchmarkCall(
        compiled_ref,
        args=(log_rho_ref, chunk_size),
        kwargs={'upcast': False},
    ).backward(
        log_rho_ref, (d_lcr, d_cd),
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
    name="chunk_cumsum",
    forward=run_chunk_cumsum,
    backward=run_chunk_cumsum_backward,
    backends=_BACKEND_MODULES,
    dimensions={"M": 16, "chunk_size": 64},
    dtypes=("float32",),
    sweep_dtypes=("float32",),
)


if __name__ == "__main__":
    raise SystemExit(benchmark_main(SPEC))
