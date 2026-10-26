import math

import torch
import torch.nn.functional as F

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
from frac_ssm.ops.refs.chunk_state import chunk_state_ref


_BACKEND_MODULES = {
    "triton": "frac_ssm.ops.triton.chunk_state",
    "cute_sm90": "frac_ssm.ops.cute.chunk_state",
    "cute_sm90_tma_k_lcr": "frac_ssm.ops.cute.variants.chunk_state_tma_k_lcr",
}


def _model_like_log_rho(B, L, M, seed=0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    tau = torch.linspace(
        math.log(1.0), math.log(65536.0), M,
        device=device, dtype=torch.float32,
    ).exp()

    raw = torch.randn(B, L, device=device, dtype=torch.float32, generator=g)
    raw = F.avg_pool1d(raw[:, None, :], 4, stride=1, padding=2)[..., :L].squeeze(1)
    dt = F.softplus(raw).clamp(1e-4, 1.0)

    def uniform(lo, hi):
        return lo + (hi - lo) * torch.rand(
            B, L, device=device, dtype=torch.float32, generator=g
        )

    alpha = uniform(0.20, 0.95)
    lam = uniform(0.25, 4.0)
    lambda_factor = lam ** (1.0 / alpha)
    rho = torch.exp(-dt[..., None] * lambda_factor[..., None] / tau)
    return rho.clamp(1e-6, 1.0 - 1e-6).log()


def _mem_bytes_fwd(B, L, R, M, chunk_size):
    nchunks = math.ceil(L / chunk_size)
    return (
        B * L * R * 2  # read u (bf16)
        + B * L * M * 2  # read kappa (bf16)
        + B * L * M * 4  # read log_cum_rho (fp32)
        + B * nchunks * R * M * 4  # write states (fp32)
    )


def _mem_bytes_bwd(B, L, R, M, chunk_size):
    nchunks = math.ceil(L / chunk_size)
    return (
        B * nchunks * R * M * 4  # read d_states (fp32)
        + B * L * R * 2  # read u (bf16)
        + B * L * M * 2  # read kappa (bf16)
        + B * L * M * 4  # read log_cum_rho (fp32)
        + B * L * R * 2  # write d_u (bf16)
        + B * L * M * 2  # write d_kappa (bf16)
        + B * L * M * 4  # write d_log_cum_rho (fp32)
    )


def _make_inputs(B, L, R, M, chunk_size, seed=0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    u = torch.randn(B, L, R, device=device, dtype=torch.bfloat16, generator=g)
    kappa = torch.randn(B, L, M, device=device, dtype=torch.bfloat16, generator=g)
    log_rho = _model_like_log_rho(B, L, M, seed=seed + 1)
    log_cum_rho, _ = chunk_cumsum_ref(log_rho, chunk_size, upcast=False)
    return u, kappa, log_cum_rho


def run_chunk_state(
    B, L, R, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, R={R}, M={M}], chunk_size={chunk_size}")

    u, kappa, log_cum_rho = _make_inputs(B, L, R, M, chunk_size)

    if verbose:
        print("Input tensor shapes")
        print(f"u {u.shape}, dtype {u.dtype}")
        print(f"kappa {kappa.shape}, dtype {kappa.dtype}")
        print(f"log_cum_rho {log_cum_rho.shape}, dtype {log_cum_rho.dtype}")

    mem_bytes = _mem_bytes_fwd(B, L, R, M, chunk_size)

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")
    kernel_calls = {
        name: BenchmarkCall(
            impl.chunk_state,
            args=(u, kappa, log_cum_rho, chunk_size),
        )
        for name, impl in impls.items()
    }

    compiled_ref = torch.compile(chunk_state_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(u, kappa, log_cum_rho, chunk_size),
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


def run_chunk_state_backward(
    B, L, R, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, R={R}, M={M}], chunk_size={chunk_size}")

    u, kappa, log_cum_rho = _make_inputs(B, L, R, M, chunk_size)
    u.requires_grad_()
    kappa.requires_grad_()
    log_cum_rho.requires_grad_()
    u_ref = u.detach().clone().requires_grad_()
    kappa_ref = kappa.detach().clone().requires_grad_()
    log_cum_rho_ref = log_cum_rho.detach().clone().requires_grad_()

    if verbose:
        print("Input tensor shapes")
        print(f"u {u.shape}, dtype {u.dtype}")
        print(f"kappa {kappa.shape}, dtype {kappa.dtype}")
        print(f"log_cum_rho {log_cum_rho.shape}, dtype {log_cum_rho.dtype}")

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")

    forward_calls = {
        name: BenchmarkCall(
            impl.chunk_state,
            args=(u, kappa, log_cum_rho, chunk_size),
        )
        for name, impl in impls.items()
    }
    sample_call = next(iter(forward_calls.values()))
    states = sample_call.kernel(*sample_call.args, **sample_call.kwargs)
    g = torch.Generator(device="cuda").manual_seed(17)
    d_states = torch.randn(
        states.shape, device="cuda", dtype=states.dtype, generator=g
    ) * 0.1

    mem_bytes = _mem_bytes_bwd(B, L, R, M, chunk_size)
    del states
    kernel_calls = {
        name: call.backward((u, kappa, log_cum_rho), d_states)
        for name, call in forward_calls.items()
    }

    compiled_ref = torch.compile(chunk_state_ref)

    reference = BenchmarkCall(
        compiled_ref,
        args=(u_ref, kappa_ref, log_cum_rho_ref, chunk_size),
        kwargs={'upcast': False},
    ).backward(
        (u_ref, kappa_ref, log_cum_rho_ref), d_states,
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
    name="chunk_state",
    forward=run_chunk_state,
    backward=run_chunk_state_backward,
    backends=_BACKEND_MODULES,
    dimensions={"R": 128, "M": 16, "chunk_size": 64},
)


if __name__ == "__main__":
    raise SystemExit(benchmark_main(SPEC))
