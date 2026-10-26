import math

import torch
import torch.nn.functional as F

from common import (
    DTYPE_MAP,
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

from frac_ssm.ops.refs.state_passing import state_passing_ref


_BACKEND_MODULES = {
    "triton": "frac_ssm.ops.triton.state_passing",
    "cute_sm90": "frac_ssm.ops.cute.state_passing",
    "cute_sm90_seq": "frac_ssm.ops.cute.variants.state_passing_seq",
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


def _chunk_decay(log_rho, chunk_size):
    B, L, M = log_rho.shape
    nchunks = math.ceil(L / chunk_size)
    padded_len = nchunks * chunk_size
    if padded_len != L:
        log_rho = F.pad(log_rho, (0, 0, 0, padded_len - L), value=0.0)
    return log_rho.view(B, nchunks, chunk_size, M).sum(dim=2)


def _mem_bytes_fwd(B, L, R, M, chunk_size, has_initial_state):
    nchunks = math.ceil(L / chunk_size)
    return (
        B * nchunks * R * M * 4  # read states (fp32)
        + B * nchunks * M * 4  # read chunk_decay (fp32)
        + (B * R * M * 4 if has_initial_state else 0)  # read initial_state (fp32)
        + B * nchunks * R * M * 4  # write states_acc (fp32)
        + B * R * M * 4  # write final_state (fp32)
    )


def _mem_bytes_bwd(B, L, R, M, chunk_size, has_initial_state):
    nchunks = math.ceil(L / chunk_size)
    return (
        B * nchunks * R * M * 4  # read d_states_acc (fp32)
        + B * R * M * 4  # read d_final_state (fp32)
        + B * nchunks * R * M * 4  # read states_acc (fp32)
        + B * nchunks * M * 4  # read chunk_decay (fp32)
        + B * nchunks * R * M * 4  # write d_states (fp32)
        + B * nchunks * M * 4  # write d_chunk_decay (fp32)
        + (B * R * M * 4 if has_initial_state else 0)  # write d_initial_state (fp32)
    )


def _make_inputs(B, L, R, M, chunk_size, seed=0, dtype=torch.float32):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    nchunks = math.ceil(L / chunk_size)
    states = 0.25 * torch.randn(
        B, nchunks, R, M, device=device, dtype=torch.float32, generator=g
    ).to(dtype)
    log_rho = _model_like_log_rho(B, L, M, seed=seed + 1)
    chunk_decay = _chunk_decay(log_rho, chunk_size)
    initial_state = 0.25 * torch.randn(
        B, R, M, device=device, dtype=torch.float32, generator=g
    ).to(dtype)
    return states, chunk_decay, initial_state


def run_state_passing(
    B, L, R, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    has_initial_state=False,
    verbose=True,
    backends=None,
    dtype_name="float32",
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, R={R}, M={M}], chunk_size={chunk_size}")
        print(f"Initial state {has_initial_state}")

    states, chunk_decay, initial_state = _make_inputs(B, L, R, M, chunk_size, dtype=DTYPE_MAP[dtype_name])
    initial_state_arg = initial_state if has_initial_state else None

    if verbose:
        print("Input tensor shapes")
        print(f"states {states.shape}, dtype {states.dtype}")
        print(f"chunk_decay {chunk_decay.shape}, dtype {chunk_decay.dtype}")
        if has_initial_state:
            print(f"initial_state {initial_state.shape}, dtype {initial_state.dtype}")

    mem_bytes = _mem_bytes_fwd(B, L, R, M, chunk_size, has_initial_state)

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")
    kernel_calls = {
        name: BenchmarkCall(
            impl.state_passing,
            args=(states, chunk_decay, initial_state_arg),
        )
        for name, impl in impls.items()
    }

    compiled_ref = torch.compile(state_passing_ref)
    reference = BenchmarkCall(
        compiled_ref,
        args=(states, chunk_decay, initial_state_arg),
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


def run_state_passing_backward(
    B, L, R, M, chunk_size,
    warmup_ms=10, rep_ms=100, preheat_ms=200, warmup_launches=5,
    has_initial_state=False,
    verbose=True,
    backends=None,
    dtype_name="float32",
):
    ensure_cuda()

    if verbose:
        print(f"Tensor dimensions [B={B}, L={L}, R={R}, M={M}], chunk_size={chunk_size}")
        print(f"Initial state {has_initial_state}")

    states, chunk_decay, initial_state = _make_inputs(B, L, R, M, chunk_size, dtype=DTYPE_MAP[dtype_name])
    states.requires_grad_()
    chunk_decay.requires_grad_()
    initial_state_arg = (
        initial_state.requires_grad_() if has_initial_state else None
    )
    states_ref = states.detach().clone().requires_grad_()
    chunk_decay_ref = chunk_decay.detach().clone().requires_grad_()
    initial_state_ref = (
        initial_state.detach().clone().requires_grad_()
        if has_initial_state else None
    )

    if verbose:
        print("Input tensor shapes")
        print(f"states {states.shape}, dtype {states.dtype}")
        print(f"chunk_decay {chunk_decay.shape}, dtype {chunk_decay.dtype}")
        if has_initial_state:
            print(f"initial_state {initial_state_arg.shape}, dtype {initial_state_arg.dtype}")

    impls = load_backend_modules(backends, _BACKEND_MODULES)
    if verbose:
        print(f"Executing kernels {', '.join(impls)}")

    forward_calls = {
        name: BenchmarkCall(
            impl.state_passing,
            args=(states, chunk_decay, initial_state_arg),
        )
        for name, impl in impls.items()
    }
    sample_call = next(iter(forward_calls.values()))
    (states_acc, final_state) = sample_call.kernel(*sample_call.args, **sample_call.kwargs)
    g = torch.Generator(device="cuda").manual_seed(17)
    d_states_acc = torch.randn(
        states_acc.shape, device="cuda", dtype=torch.float32, generator=g
    ).to(states_acc.dtype)
    d_final_state = torch.randn(
        final_state.shape, device="cuda", dtype=torch.float32, generator=g
    ).to(final_state.dtype)
    inputs = (
        (states, chunk_decay, initial_state_arg)
        if has_initial_state else (states, chunk_decay)
    )

    mem_bytes = _mem_bytes_bwd(B, L, R, M, chunk_size, has_initial_state)
    del states_acc, final_state
    kernel_calls = {
        name: call.backward(inputs, (d_states_acc, d_final_state))
        for name, call in forward_calls.items()
    }

    compiled_ref = torch.compile(state_passing_ref)

    ref_inputs = (
        (states_ref, chunk_decay_ref, initial_state_ref)
        if has_initial_state else (states_ref, chunk_decay_ref)
    )
    reference = BenchmarkCall(
        compiled_ref,
        args=(states_ref, chunk_decay_ref, initial_state_ref),
        kwargs={'upcast': False},
    ).backward(
        ref_inputs, (d_states_acc, d_final_state),
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
    name="state_passing",
    forward=run_state_passing,
    backward=run_state_passing_backward,
    backends=_BACKEND_MODULES,
    dimensions={"R": 64, "M": 16, "chunk_size": 64},
    dtypes=("float32", "bfloat16"),
    sweep_dtypes=("float32",),
    options={"has_initial_state": False},
    takes_dtype=True,
    batches=(2, 16, 32, 64, 256),
    lengths=(512, 4096, 8192, 16384, 32768),
)


if __name__ == "__main__":
    raise SystemExit(benchmark_main(SPEC))
