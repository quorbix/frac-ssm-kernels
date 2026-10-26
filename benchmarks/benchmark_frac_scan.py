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

from frac_ssm.ops.refs.frac_scan import frac_scan_ref


_BACKEND_MODULES = {
    "triton": "frac_ssm.ops.triton.frac_scan",
    "cute_sm90": "frac_ssm.ops.cute.frac_scan",
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


def _mem_bytes_fwd(B, L, R, M, has_initial_state):
    return (
        B * L * R * 2  # read u (bf16)
        + B * L * M * 2  # read c (bf16)
        + B * L * M * 2  # read kappa (bf16)
        + B * L * M * 4  # read rho (fp32)
        + (B * R * M * 4 if has_initial_state else 0)  # read initial_state (fp32)
        + B * L * R * 2  # write h (bf16)
        + B * R * M * 4  # write final_state (fp32)
    )


def _mem_bytes_bwd(B, L, R, M, has_initial_state):
    return (
        B * L * R * 2  # read d_h (bf16)
        + B * R * M * 4  # read d_final_state (fp32)
        + B * L * R * 2  # read u (bf16)
        + B * L * M * 2  # read c (bf16)
        + B * L * M * 2  # read kappa (bf16)
        + B * L * M * 4  # read rho (fp32)
        + B * L * R * 2  # write d_u (bf16)
        + B * L * M * 2  # write d_c (bf16)
        + B * L * M * 2  # write d_kappa (bf16)
        + B * L * M * 4  # write d_rho (fp32)
        + (B * R * M * 4 if has_initial_state else 0)  # write d_initial_state (fp32)
    )


def _make_inputs(B, L, R, M, seed=0):
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
    rho = _model_like_log_rho(B, L, M, seed=seed + 1).exp()
    initial_state = 0.25 * torch.randn(
        B, R, M, device=device, dtype=torch.float32, generator=g
    )
    return u, c, kappa, rho, initial_state


def run_frac_scan(
    B, L, R, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    has_initial_state=False,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, R={R}, M={M}], chunk_size={chunk_size}")
        print(f"Initial state {has_initial_state}")

    u, c, kappa, rho, initial_state = _make_inputs(B, L, R, M)
    initial_state_arg = initial_state if has_initial_state else None

    if verbose:
        print("Input tensor shapes")
        print(f"u {u.shape}, dtype {u.dtype}")
        print(f"c {c.shape}, dtype {c.dtype}")
        print(f"kappa {kappa.shape}, dtype {kappa.dtype}")
        print(f"rho {rho.shape}, dtype {rho.dtype}")
        if has_initial_state:
            print(f"initial_state {initial_state.shape}, dtype {initial_state.dtype}")

    mem_bytes = _mem_bytes_fwd(B, L, R, M, has_initial_state)

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")
    kernel_calls = {
        name: BenchmarkCall(
            impl.frac_scan,
            args=(u, c, kappa, rho, chunk_size, initial_state_arg, True),
        )
        for name, impl in impls.items()
    }

    compiled_ref = torch.compile(frac_scan_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(u, c, kappa, rho, chunk_size, initial_state_arg, True),
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


def run_frac_scan_backward(
    B, L, R, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    has_initial_state=False,
    verbose=True,
    backends=None,
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, R={R}, M={M}], chunk_size={chunk_size}")
        print(f"Initial state {has_initial_state}")

    u, c, kappa, rho, initial_state = _make_inputs(B, L, R, M)
    u.requires_grad_()
    c.requires_grad_()
    kappa.requires_grad_()
    rho.requires_grad_()
    initial_state_arg = (
        initial_state.requires_grad_() if has_initial_state else None
    )
    u_ref = u.detach().clone().requires_grad_()
    c_ref = c.detach().clone().requires_grad_()
    kappa_ref = kappa.detach().clone().requires_grad_()
    rho_ref = rho.detach().clone().requires_grad_()
    initial_state_ref = (
        initial_state.detach().clone().requires_grad_()
        if has_initial_state else None
    )

    if verbose:
        print("Input tensor shapes")
        print(f"u {u.shape}, dtype {u.dtype}")
        print(f"c {c.shape}, dtype {c.dtype}")
        print(f"kappa {kappa.shape}, dtype {kappa.dtype}")
        print(f"rho {rho.shape}, dtype {rho.dtype}")
        if has_initial_state:
            print(f"initial_state {initial_state_arg.shape}, dtype {initial_state_arg.dtype}")

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")

    forward_calls = {
        name: BenchmarkCall(
            impl.frac_scan,
            args=(u, c, kappa, rho, chunk_size, initial_state_arg, True),
        )
        for name, impl in impls.items()
    }
    sample_call = next(iter(forward_calls.values()))
    (h, final_state) = sample_call.kernel(*sample_call.args, **sample_call.kwargs)
    g = torch.Generator(device="cuda").manual_seed(17)
    d_h = torch.randn(h.shape, device="cuda", dtype=h.dtype, generator=g)
    d_final_state = torch.randn(
        final_state.shape, device="cuda", dtype=torch.float32, generator=g
    )
    inputs = (
        (u, c, kappa, rho, initial_state_arg)
        if has_initial_state else (u, c, kappa, rho)
    )

    mem_bytes = _mem_bytes_bwd(B, L, R, M, has_initial_state)
    del h, final_state
    kernel_calls = {
        name: call.backward(inputs, (d_h, d_final_state))
        for name, call in forward_calls.items()
    }

    ref_inputs = (
        (u_ref, c_ref, kappa_ref, rho_ref, initial_state_ref)
        if has_initial_state
        else (u_ref, c_ref, kappa_ref, rho_ref)
    )
    compiled_ref = torch.compile(frac_scan_ref)

    reference = BenchmarkCall(
        compiled_ref,
        args=(u_ref, c_ref, kappa_ref, rho_ref, chunk_size, initial_state_ref, True),
        kwargs={'upcast': False},
    ).backward(
        ref_inputs, (d_h, d_final_state),
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


def _unsupported(case, backend, direction):
    if backend == "cute_sm90" and case["chunk_size"] not in (32, 64, 128):
        return "CuTe frac_scan supports chunk_size 32, 64, or 128"
    return None


SPEC = BenchmarkSpec(
    name="frac_scan",
    forward=run_frac_scan,
    backward=run_frac_scan_backward,
    backends=_BACKEND_MODULES,
    dimensions={"R": 128, "M": 16, "chunk_size": 64},
    options={"has_initial_state": False},
    unsupported=_unsupported,
)


if __name__ == "__main__":
    raise SystemExit(benchmark_main(SPEC))
