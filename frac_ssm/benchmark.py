# Adapted from Dao-AILab/quack at c8ec3170057987da0ec99883736f381ea1937cf3.
# Licensed under Apache-2.0 as in the repository LICENSE.
# Local changes share graph timing, backward preparation and buffer restoration.
"""GPU measurements shared by standalone benchmarks and CuTe autotuning."""

from __future__ import annotations

import copy
import inspect
import math
import os
import statistics
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Callable, Mapping

import torch
import torch._functorch.config as _functorch_config
from torch import Tensor
from torch.utils._pytree import tree_leaves, tree_map


PROTOCOL = "cuda_graph_rotation_v1"
DEFAULT_WARMUP_MS = 10
DEFAULT_REP_MS = 100
DEFAULT_PREHEAT_MS = 200
DEFAULT_WARMUP_LAUNCHES = 5


def _bench_cuda_graph_l2_rotate(
    fn,
    arg_sets,
    kwarg_sets,
    extra_kwargs,
    warmup_target_ms: float = 200.0,
    n_timed_calls: int = 200,
    quantiles=None,
    *,
    rep_ms: float = DEFAULT_REP_MS,
    preheat_ms: float = DEFAULT_PREHEAT_MS,
    warmup_launches: int = DEFAULT_WARMUP_LAUNCHES,
    restore=None,
    stream=None,
    return_measurement=False,
):
    """CUDA-graph benchmark over pre-cloned rotating input sets.

    Calls fn(*arg_sets[i], **kwarg_sets[i], **extra_kwargs) round-robin.
    Compilation and autotuning are discovered in eager priming launches.

    Local extensions to Quack use graph warmup and repeated timed replays.
    Each replay reports a batch-average latency, and the result is their
    median. Requested quantiles use these samples instead of repeating one
    value. Rotation is bounded and does not guarantee a cold L2 cache.

    An optional restore callback runs before each rotation round, outside
    the timed event interval. Backward workloads pass the stream used to
    prepare their saved autograd state. return_measurement includes samples
    and rotation metadata for the shared benchmark runner.
    """
    n_sets = len(arg_sets)
    if n_sets == 0 or len(kwarg_sets) != n_sets:
        raise ValueError("Matching nonempty argument sets are required")
    if warmup_target_ms < 0 or rep_ms <= 0 or warmup_launches < 1 or n_timed_calls < 1:
        raise ValueError("Timing requires nonnegative warmup and positive repeats")
    if any(kwargs.keys() & extra_kwargs.keys() for kwargs in kwarg_sets):
        raise ValueError("Benchmark arguments conflict with candidate arguments")

    if stream is None:
        device = _device(BenchmarkCall(fn, (arg_sets, kwarg_sets)))
        stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(stream.device))

    def capture(rounds_timed):
        events = [
            (torch.cuda.Event(enable_timing=True, external=True),
             torch.cuda.Event(enable_timing=True, external=True))
            for _ in range(rounds_timed)
        ]
        timed_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(timed_graph, stream=stream):
            for start_evt, end_evt in events:
                if restore is not None:
                    restore()
                start_evt.record()
                for i in range(n_sets):
                    fn(*arg_sets[i], **kwarg_sets[i], **extra_kwargs)
                end_evt.record()
        return timed_graph, events

    try:
        with (
            torch.cuda.device(stream.device), torch.cuda.stream(stream),
            _functorch_config.patch(donated_buffer=False),
        ):
            # Prime outside graph capture so deferred compilation can escape.
            for _ in range(warmup_launches):
                if restore is not None:
                    restore()
                for i in range(n_sets):
                    fn(*arg_sets[i], **kwarg_sets[i], **extra_kwargs)
            torch.cuda.synchronize()
            preheat_gpu(preheat_ms, device=stream.device)

            # Probe one rotation round to size the warmup and timed graph.
            probe, probe_events = capture(1)
            probe.replay()
            torch.cuda.synchronize()
            est_ms = max(probe_events[0][0].elapsed_time(probe_events[0][1]), 1e-6)
            del probe, probe_events

            # Round the call count to whole rotations and bound graph size.
            target_ms = min(1.0, rep_ms / 5)
            rounds_timed = max(
                1, min(n_timed_calls // n_sets, math.ceil(target_ms / est_ms)),
            )
            total_timed_calls = rounds_timed * n_sets
            timed_graph, events = capture(rounds_timed)
            for _ in range(math.ceil(warmup_target_ms / (rounds_timed * est_ms))):
                timed_graph.replay()
            torch.cuda.synchronize()

            samples = []
            elapsed = 0.0
            while len(samples) < 5 or elapsed < rep_ms:
                timed_graph.replay()
                torch.cuda.synchronize()
                batch_ms = sum(start.elapsed_time(end) for start, end in events)
                if not math.isfinite(batch_ms) or batch_ms <= 0:
                    raise RuntimeError("CUDA graph produced an invalid timing")
                samples.append(batch_ms / total_timed_calls)
                elapsed += batch_ms
    finally:
        torch.cuda.current_stream(stream.device).wait_stream(stream)

    measurement = Measurement(tuple(samples), n_sets, total_timed_calls)
    if return_measurement:
        return measurement
    if quantiles is not None:
        return measurement.quantiles(quantiles)
    return measurement.ms


def _clone_l2_rotate_inputs(args, kwargs, n_buffers: int):
    """Clone tensor args AND tensor kwargs n_buffers times.

    Returns (arg_sets, kwarg_sets) with the same positional and keyword
    structure as the originals. Non-tensor values are shared across sets.

    Local extensions clone nested tensors and modules, preserve tensor
    strides and pointer alignment, and retain repeated tensor identity.
    Distinct aliased views are rejected rather than losing their aliases.
    """
    if n_buffers < 1:
        raise ValueError("At least one rotation buffer is required")
    storages = {}
    for tensor in _tensors((args, kwargs)):
        if tensor.numel():
            storage = (tensor.device, tensor.untyped_storage().data_ptr())
            if storage in storages and storages[storage] is not tensor:
                raise ValueError("Aliased tensor views need explicit independent benchmark inputs")
            storages[storage] = tensor

    arg_sets = []
    kwarg_sets = []
    for _ in range(n_buffers):
        cloned_args, cloned_kwargs = _clone_tree((args, dict(kwargs)), {})
        arg_sets.append(cloned_args)
        kwarg_sets.append(cloned_kwargs)
    return arg_sets, kwarg_sets


def _pick_l2_rotate_count(
    args, kwargs, target_ratio: int = 3, min_buffers: int = 1, max_buffers: int = 16,
    *, tensor_bytes=None, free_bytes=None,
):
    """Pick a rotation count targeting target_ratio * L2_size.

    Local extensions accept the measured preparation footprint, including
    backward saved activations, and the free memory before preparation.
    Memory headroom takes precedence over min_buffers. Small workloads
    capped at max_buffers can still fit in L2.
    """
    if min_buffers < 1 or max_buffers < min_buffers:
        raise ValueError("Rotation buffer bounds must be positive and ordered")
    if not torch.cuda.is_available():
        return min_buffers
    if tensor_bytes is None:
        tensors = {id(tensor): tensor for tensor in _tensors((args, kwargs))}
        tensor_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    if tensor_bytes == 0:
        return min_buffers
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_size = props.L2_cache_size
    n_by_l2 = (target_ratio * l2_size + tensor_bytes - 1) // tensor_bytes
    if free_bytes is None:
        free_bytes, _ = torch.cuda.mem_get_info()
    # Leave half of free memory headroom for scratch and other allocations.
    n_by_mem = max(1, int(free_bytes * 0.5) // tensor_bytes)
    return max(1, min(max_buffers, max(min_buffers, n_by_l2), n_by_mem))


@dataclass(frozen=True)
class BenchmarkCall:
    """Explicit forward call, optionally used to prepare backward-only work."""

    kernel: Callable
    args: tuple = ()
    kwargs: Mapping = field(default_factory=dict)
    grad_inputs: tuple[Tensor, ...] | None = None
    grad_outputs: object = None

    def backward(self, inputs, grad_outputs):
        inputs = (inputs,) if isinstance(inputs, Tensor) else tuple(inputs)
        return replace(self, grad_inputs=inputs, grad_outputs=grad_outputs)


@dataclass(frozen=True)
class Measurement:
    samples_ms: tuple[float, ...]
    buffers: int
    calls_per_batch: int

    @property
    def ms(self):
        return statistics.median(self.samples_ms)

    def quantiles(self, values):
        samples = sorted(self.samples_ms)
        result = []
        for q in values:
            if not 0 <= q <= 1:
                raise ValueError("Quantiles must be between zero and one")
            position = q * (len(samples) - 1)
            lo, hi = math.floor(position), math.ceil(position)
            result.append(samples[lo] + (samples[hi] - samples[lo]) * (position - lo))
        return result

    def metadata(self):
        return {
            "protocol": PROTOCOL,
            "statistic": "median_of_batch_means",
            "buffers": self.buffers,
            "calls_per_batch": self.calls_per_batch,
            "samples_ms": list(self.samples_ms),
        }


def _tensors(value):
    for leaf in tree_leaves(value):
        if isinstance(leaf, Tensor):
            yield leaf
        elif isinstance(leaf, torch.nn.Module):
            yield from leaf.parameters()
            yield from leaf.buffers()


def _device(call):
    devices = {x.device for x in _tensors((call.kernel, call.args, call.kwargs, call.grad_outputs))}
    cuda_devices = {device for device in devices if device.type == "cuda"}
    if len(cuda_devices) != 1:
        raise ValueError("A benchmark call must use exactly one CUDA device")
    return next(iter(cuda_devices))


@lru_cache(maxsize=8)
def _preheat_tensors(device, size):
    generator = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(size, size, device=device, dtype=torch.bfloat16, generator=generator)
    b = torch.randn(size, size, device=device, dtype=torch.bfloat16, generator=generator)
    return a, b, torch.empty_like(a)


def preheat_gpu(duration_ms=DEFAULT_PREHEAT_MS, *, device="cuda"):
    """Preheat with fixed finite operands without advancing the caller's RNG."""
    if duration_ms < 0:
        raise ValueError("Preheat duration must be nonnegative")
    with torch.cuda.device(device):
        if duration_ms == 0:
            return
        device = torch.device("cuda", torch.cuda.current_device())
        size = int(os.environ.get("FRAC_SSM_BENCH_PREHEAT_SIZE", "4096"))
        a, b, out = _preheat_tensors(device, size)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        elapsed = 0.0
        while elapsed < duration_ms:
            for _ in range(32):
                torch.mm(a, b, out=out)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)


def _clone_tree(value, memo):
    def clone(leaf):
        if id(leaf) in memo:
            return memo[id(leaf)]
        if isinstance(leaf, Tensor):
            # Copy the storage span so padded and broadcast strides survive.
            # Preserve pointer alignment as well as the logical tensor layout.
            span = (
                1 + sum((size - 1) * stride for size, stride in zip(leaf.shape, leaf.stride()))
                if leaf.numel() else 0
            )
            alignment = 256
            storage = torch.empty(
                span + alignment // leaf.element_size(), dtype=leaf.dtype, device=leaf.device,
            )
            offset = ((leaf.data_ptr() - storage.data_ptr()) % alignment) // leaf.element_size()
            storage[offset:offset + span].copy_(leaf.detach().as_strided((span,), (1,)))
            cloned = storage.as_strided(leaf.shape, leaf.stride(), offset)
            if isinstance(leaf, torch.nn.Parameter):
                cloned = torch.nn.Parameter(cloned, requires_grad=leaf.requires_grad)
            else:
                cloned.requires_grad_(leaf.requires_grad)
            memo[id(leaf)] = cloned
            return cloned
        if isinstance(leaf, torch.nn.Module):
            for tensor in (*leaf.parameters(), *leaf.buffers()):
                clone(tensor)
            return copy.deepcopy(leaf, memo)
        return leaf
    return tree_map(clone, value)


def _prepare_call(call):
    if call.grad_inputs is not None:
        input_ids = {id(x) for x in _tensors((call.kernel, call.args, call.kwargs))}
        if any(id(x) not in input_ids for x in call.grad_inputs):
            raise ValueError("Backward inputs must belong to the call or its modules")
    arg_sets, kwarg_sets = _clone_l2_rotate_inputs(
        (call.kernel, call.args, call.grad_outputs, call.grad_inputs), call.kwargs, 1,
    )
    kernel, args, grad_outputs, inputs = arg_sets[0]
    kwargs = kwarg_sets[0]
    if inputs is None:
        return BenchmarkCall(kernel, args, kwargs)
    for x in inputs:
        x.requires_grad_(True)
    # Retained backward workloads cannot donate their saved buffers.
    with torch.enable_grad(), _functorch_config.patch(donated_buffer=False):
        outputs = kernel(*args, **kwargs)
    outputs = tuple(tree_leaves(outputs))
    gradients = tuple(tree_leaves(grad_outputs))
    if len(outputs) != len(gradients):
        raise ValueError("Backward gradients must match the forward outputs")
    return BenchmarkCall(
        torch.autograd.grad,
        args=(outputs, inputs),
        kwargs={"grad_outputs": gradients, "retain_graph": True},
    )


def _invoke_call(call, **extra_kwargs):
    return call.kernel(*call.args, **call.kwargs, **extra_kwargs)


class PreparedBenchmark:
    """Own prepared calls and delegate timing to Quack's rotating graph helper."""

    def __init__(self, call, *, restore_value=(), max_buffers=16):
        if max_buffers < 1:
            raise ValueError("At least one rotation buffer is required")
        self.device = _device(call)
        # Saved autograd nodes must see the same stream during capture.
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.calls = []
        self.restores = []
        self.restore_value = tuple(restore_value)

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            free_bytes, _ = torch.cuda.mem_get_info()
            before = torch.cuda.memory_allocated()
            self._append(call)
            footprint = max(1, torch.cuda.memory_allocated() - before)
            count = _pick_l2_rotate_count(
                call.args, call.kwargs, max_buffers=max_buffers,
                tensor_bytes=footprint, free_bytes=free_bytes,
            )
            for _ in range(1, count):
                free_now, _ = torch.cuda.mem_get_info()
                if free_now < 2 * footprint:
                    break
                oom = False
                try:
                    self._append(call)
                except torch.OutOfMemoryError:
                    oom = True
                if oom:
                    # Keep the already prepared slots under the same protocol.
                    torch.cuda.empty_cache()
                    break

    def _append(self, call):
        prepared = _prepare_call(call)
        restores = []
        if self.restore_value:
            named = inspect.signature(prepared.kernel).bind_partial(
                *prepared.args, **prepared.kwargs,
            ).arguments
            for name in self.restore_value:
                target = named[name]
                if not isinstance(target, Tensor):
                    raise ValueError("Restored arguments must be tensors")
                restores.append((target, target.detach().clone()))
        self.calls.append(prepared)
        self.restores.append(restores)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        self.calls.clear()
        self.restores.clear()

    def _reset(self):
        with torch.no_grad():
            for restores in self.restores:
                for target, snapshot in restores:
                    target.copy_(snapshot)

    def warmup(self, launches=DEFAULT_WARMUP_LAUNCHES, *, extra_kwargs=None):
        """Discover compilation and tuning outside any CUDA graph capture."""
        extra_kwargs = {} if extra_kwargs is None else extra_kwargs
        with (
            torch.cuda.device(self.device), torch.cuda.stream(self.stream),
            _functorch_config.patch(donated_buffer=False),
        ):
            for _ in range(launches):
                self._reset()
                for call in self.calls:
                    _invoke_call(call, **extra_kwargs)
            torch.cuda.synchronize()

    def measure(self, *, extra_kwargs=None, warmup_ms=DEFAULT_WARMUP_MS,
                rep_ms=DEFAULT_REP_MS, preheat_ms=DEFAULT_PREHEAT_MS,
                warmup_launches=DEFAULT_WARMUP_LAUNCHES):
        extra_kwargs = {} if extra_kwargs is None else extra_kwargs
        if any(call.kwargs.keys() & extra_kwargs.keys() for call in self.calls):
            raise ValueError("Benchmark arguments conflict with candidate arguments")
        # Each slot can own a different module or retained autograd graph.
        return _bench_cuda_graph_l2_rotate(
            _invoke_call,
            arg_sets=[(call,) for call in self.calls],
            kwarg_sets=[{} for _ in self.calls],
            extra_kwargs=extra_kwargs,
            warmup_target_ms=warmup_ms, rep_ms=rep_ms,
            preheat_ms=preheat_ms, warmup_launches=warmup_launches,
            restore=self._reset, stream=self.stream, return_measurement=True,
        )


def benchmark(kernel, *, args=(), kwargs=None, warmup_ms=DEFAULT_WARMUP_MS,
              rep_ms=DEFAULT_REP_MS, preheat_ms=DEFAULT_PREHEAT_MS):
    """Return median batch-average GPU latency in milliseconds."""
    call = BenchmarkCall(kernel, args, {} if kwargs is None else kwargs)
    with PreparedBenchmark(call) as prepared:
        return prepared.measure(
            warmup_ms=warmup_ms, rep_ms=rep_ms, preheat_ms=preheat_ms,
        ).ms
