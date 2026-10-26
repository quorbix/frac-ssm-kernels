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
from frac_ssm.ops.refs.chunk_readout import chunk_readout_ref
from frac_ssm.ops.refs.chunk_w import chunk_w_ref
from frac_ssm.ops.triton.chunk_w import chunk_w


_BACKEND_MODULES = {
    "triton": "frac_ssm.ops.triton.chunk_readout",
    "cute_sm90": "frac_ssm.ops.cute.chunk_fused_readout",
    "cute_sm90_warp_specialized": (
        "frac_ssm.ops.cute.variants.chunk_fused_readout_warp_specialized"
    ),
    "cute_sm90_cooperative": (
        "frac_ssm.ops.cute.variants.chunk_fused_readout_cooperative"
    ),
}


def _mem_bytes_fwd(B, L, R, M, chunk_size):
    nchunks = math.ceil(L / chunk_size)
    return (
        B * L * R * 2  # read u (bf16)
        + B * L * M * 2  # read c (bf16)
        + B * L * M * 2  # read kappa (bf16)
        + B * nchunks * R * M * 4  # read states_acc (fp32)
        + B * L * M * 4  # read log_cum_rho (fp32)
        + B * L * R * 2  # write h (bf16)
    )


def _mem_bytes_bwd(B, L, R, M, chunk_size):
    nchunks = math.ceil(L / chunk_size)
    return (
        B * L * R * 2  # read d_h (bf16)
        + B * L * R * 2  # read u (bf16)
        + B * L * M * 2  # read c (bf16)
        + B * L * M * 2  # read kappa (bf16)
        + B * nchunks * R * M * 4  # read states_acc (fp32)
        + B * L * M * 4  # read log_cum_rho (fp32)
        + B * L * R * 2  # write d_u (bf16)
        + B * L * M * 2  # write d_c (bf16)
        + B * L * M * 2  # write d_kappa (bf16)
        + B * nchunks * R * M * 4  # write d_states_acc (fp32)
        + B * L * M * 4  # write d_log_cum_rho (fp32)
    )


def _make_inputs(B, L, R, M, chunk_size, seed=0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    u = 0.25 * torch.randn(
        B, L, R, device=device, dtype=torch.bfloat16, generator=g
    )
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

    nchunks = math.ceil(L / chunk_size)
    states_acc = 0.25 * torch.randn(
        B, nchunks, R, M, device=device, dtype=torch.float32, generator=g
    )
    return u, c, kappa, states_acc, log_cum_rho


def _chunk_fused_readout_ref(
    u, c, kappa, states_acc, log_cum_rho, chunk_size,
):
    W = chunk_w_ref(c, kappa, log_cum_rho, chunk_size, upcast=False)
    return chunk_readout_ref(
        u, c, states_acc, W, log_cum_rho, chunk_size, upcast=False
    )


def _run_backend(name, impl, u, c, kappa, states_acc, log_cum_rho, chunk_size):
    if name == "triton":
        W = chunk_w(c, kappa, log_cum_rho, chunk_size)
        return impl.chunk_readout(
            u, c, states_acc, W, log_cum_rho, chunk_size
        )
    return impl.chunk_fused_readout(
        u, c, kappa, states_acc, log_cum_rho, chunk_size
    )


def run_chunk_fused_readout(
    B, L, R, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, R={R}, M={M}], chunk_size={chunk_size}")

    u, c, kappa, states_acc, log_cum_rho = _make_inputs(
        B, L, R, M, chunk_size
    )

    if verbose:
        print("Input tensor shapes")
        print(f"u {u.shape}, dtype {u.dtype}")
        print(f"c {c.shape}, dtype {c.dtype}")
        print(f"kappa {kappa.shape}, dtype {kappa.dtype}")
        print(f"states_acc {states_acc.shape}, dtype {states_acc.dtype}")
        print(f"log_cum_rho {log_cum_rho.shape}, dtype {log_cum_rho.dtype}")

    mem_bytes = _mem_bytes_fwd(B, L, R, M, chunk_size)

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")
    kernel_calls = {
        name: BenchmarkCall(
            _run_backend,
            args=(name, impl, u, c, kappa, states_acc, log_cum_rho, chunk_size),
        )
        for name, impl in impls.items()
    }

    compiled_ref = torch.compile(_chunk_fused_readout_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(u, c, kappa, states_acc, log_cum_rho, chunk_size),
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


def run_chunk_fused_readout_backward(
    B, L, R, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, R={R}, M={M}], chunk_size={chunk_size}")

    u, c, kappa, states_acc, log_cum_rho = _make_inputs(
        B, L, R, M, chunk_size
    )
    u.requires_grad_()
    c.requires_grad_()
    kappa.requires_grad_()
    states_acc.requires_grad_()
    log_cum_rho.requires_grad_()
    u_ref = u.detach().clone().requires_grad_()
    c_ref = c.detach().clone().requires_grad_()
    kappa_ref = kappa.detach().clone().requires_grad_()
    states_acc_ref = states_acc.detach().clone().requires_grad_()
    log_cum_rho_ref = log_cum_rho.detach().clone().requires_grad_()

    if verbose:
        print("Input tensor shapes")
        print(f"u {u.shape}, dtype {u.dtype}")
        print(f"c {c.shape}, dtype {c.dtype}")
        print(f"kappa {kappa.shape}, dtype {kappa.dtype}")
        print(f"states_acc {states_acc.shape}, dtype {states_acc.dtype}")
        print(f"log_cum_rho {log_cum_rho.shape}, dtype {log_cum_rho.dtype}")

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")

    forward_calls = {
        name: BenchmarkCall(
            _run_backend,
            args=(name, impl, u, c, kappa, states_acc, log_cum_rho, chunk_size),
        )
        for name, impl in impls.items()
    }
    sample_call = next(iter(forward_calls.values()))
    h = sample_call.kernel(*sample_call.args, **sample_call.kwargs)
    g = torch.Generator(device="cuda").manual_seed(17)
    d_h = torch.randn(h.shape, device="cuda", dtype=h.dtype, generator=g)

    inputs = (u, c, kappa, states_acc, log_cum_rho)
    mem_bytes = _mem_bytes_bwd(B, L, R, M, chunk_size)
    del h
    kernel_calls = {
        name: call.backward(inputs, d_h)
        for name, call in forward_calls.items()
    }

    compiled_ref = torch.compile(_chunk_fused_readout_ref)

    ref_inputs = (u_ref, c_ref, kappa_ref, states_acc_ref, log_cum_rho_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(u_ref, c_ref, kappa_ref, states_acc_ref, log_cum_rho_ref, chunk_size),
    ).backward(
        ref_inputs, d_h,
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
    name="chunk_fused_readout",
    forward=run_chunk_fused_readout,
    backward=run_chunk_fused_readout_backward,
    backends=_BACKEND_MODULES,
    dimensions={"R": 64, "M": 16, "chunk_size": 64},
)


if __name__ == "__main__":
    raise SystemExit(benchmark_main(SPEC))
