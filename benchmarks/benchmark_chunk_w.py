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
from frac_ssm.ops.refs.chunk_w import chunk_w_ref


_BACKEND_MODULES = {
    "triton": "frac_ssm.ops.triton.chunk_w",
}


def _mem_bytes_fwd(B, L, M, chunk_size):
    nchunks = math.ceil(L / chunk_size)
    return (
        B * L * M * 2  # read c (bf16)
        + B * L * M * 2  # read kappa (bf16)
        + B * L * M * 4  # read log_cum_rho (fp32)
        + B * nchunks * chunk_size * chunk_size * 4  # write W (fp32)
    )


def _mem_bytes_bwd(B, L, M, chunk_size):
    nchunks = math.ceil(L / chunk_size)
    return (
        B * nchunks * chunk_size * chunk_size * 4  # read d_W (fp32)
        + B * L * M * 2  # read c (bf16)
        + B * L * M * 2  # read kappa (bf16)
        + B * L * M * 4  # read log_cum_rho (fp32)
        + B * L * M * 2  # write d_c (bf16)
        + B * L * M * 2  # write d_kappa (bf16)
        + B * L * M * 4  # write d_log_cum_rho (fp32)
    )


def _make_inputs(B, L, M, chunk_size, seed=0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    c = 0.25 * torch.randn(
        B, L, M, device=device, dtype=torch.bfloat16, generator=g
    )
    kappa = 0.25 * torch.randn(
        B, L, M, device=device, dtype=torch.bfloat16, generator=g
    )

    rho_g = torch.Generator(device=device).manual_seed(seed + 1)
    log_rho = torch.rand(
        B, L, M, device=device, dtype=torch.float32, generator=rho_g
    ).clamp(0.01, 0.99).log()
    log_cum_rho, _ = chunk_cumsum_ref(
        log_rho, chunk_size, upcast=False
    )
    return c, kappa, log_cum_rho


def run_chunk_w(
    B, L, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, M={M}], chunk_size={chunk_size}")

    c, kappa, log_cum_rho = _make_inputs(B, L, M, chunk_size)

    if verbose:
        print("Input tensor shapes")
        print(f"c {c.shape}, dtype {c.dtype}")
        print(f"kappa {kappa.shape}, dtype {kappa.dtype}")
        print(f"log_cum_rho {log_cum_rho.shape}, dtype {log_cum_rho.dtype}")

    mem_bytes = _mem_bytes_fwd(B, L, M, chunk_size)

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")
    kernel_calls = {
        name: BenchmarkCall(
            impl.chunk_w,
            args=(c, kappa, log_cum_rho, chunk_size),
        )
        for name, impl in impls.items()
    }

    compiled_ref = torch.compile(chunk_w_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(c, kappa, log_cum_rho, chunk_size),
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


def run_chunk_w_backward(
    B, L, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, M={M}], chunk_size={chunk_size}")

    c, kappa, log_cum_rho = _make_inputs(B, L, M, chunk_size)
    c.requires_grad_()
    kappa.requires_grad_()
    log_cum_rho.requires_grad_()
    c_ref = c.detach().clone().requires_grad_()
    kappa_ref = kappa.detach().clone().requires_grad_()
    log_cum_rho_ref = log_cum_rho.detach().clone().requires_grad_()

    if verbose:
        print("Input tensor shapes")
        print(f"c {c.shape}, dtype {c.dtype}")
        print(f"kappa {kappa.shape}, dtype {kappa.dtype}")
        print(f"log_cum_rho {log_cum_rho.shape}, dtype {log_cum_rho.dtype}")

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")

    forward_calls = {
        name: BenchmarkCall(
            impl.chunk_w,
            args=(c, kappa, log_cum_rho, chunk_size),
        )
        for name, impl in impls.items()
    }
    sample_call = next(iter(forward_calls.values()))
    W = sample_call.kernel(*sample_call.args, **sample_call.kwargs)
    g = torch.Generator(device="cuda").manual_seed(17)
    d_W = torch.randn(W.shape, device="cuda", dtype=torch.float32, generator=g)

    inputs = (c, kappa, log_cum_rho)
    mem_bytes = _mem_bytes_bwd(B, L, M, chunk_size)
    del W
    kernel_calls = {
        name: call.backward(inputs, d_W)
        for name, call in forward_calls.items()
    }

    compiled_ref = torch.compile(chunk_w_ref)

    ref_inputs = (c_ref, kappa_ref, log_cum_rho_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(c_ref, kappa_ref, log_cum_rho_ref, chunk_size),
        kwargs={'upcast': False},
    ).backward(
        ref_inputs, d_W,
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
    name="chunk_w",
    forward=run_chunk_w,
    backward=run_chunk_w_backward,
    backends=_BACKEND_MODULES,
    dimensions={"M": 16, "chunk_size": 64},
)


if __name__ == "__main__":
    raise SystemExit(benchmark_main(SPEC))
