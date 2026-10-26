# Adapted from Dao-AILab/quack at 14b0b7122e252a4df40a0b762c7ffd0f8b67a015.
# Licensed under Apache-2.0. See LICENSE and AUTOTUNER_LICENSE.
# Local differences are explicit keys, device/thread guards and shared graph measurements.
# Adapted from https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py
# Copyright (C) 2025, Tri Dao.
from __future__ import annotations

import builtins
import os
import sys
import time
import inspect
import threading
from contextlib import nullcontext
import base64
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from frac_ssm.benchmark import (
    PROTOCOL, BenchmarkCall, PreparedBenchmark, preheat_gpu as _gpu_warmup,
)

import torch
from torch import Tensor

import triton

from frac_ssm import __version__


PACKAGE_NAME = "frac_ssm"
VERSION = __version__

# Quack's active compile pool and candidate loop are process-global.
_tuning_lock = threading.RLock()


def get_home_dir():
    return os.getenv(f"{PACKAGE_NAME.upper()}_HOME", Path.home())


def default_cache_dir():
    return os.path.join(get_home_dir(), f".{PACKAGE_NAME}", "cache")


class FileCacheManager(triton.runtime.cache.FileCacheManager):
    def __init__(self, key):
        super().__init__(key)
        self.cache_dir = (
            os.getenv(f"{PACKAGE_NAME.upper()}_CACHE_DIR", "").strip() or default_cache_dir()
        )
        if self.cache_dir:
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            raise RuntimeError("Could not create or locate cache dir")


def _base32(key):
    # Assume key is a hex string.
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")


#: How long a deferred config may wait on its pool compile before the bench
#: loop stops trusting the pool and benches it with the pool suppressed
#: (in-process compile). Guards against a wedged worker / a foreign flock
#: holder that never produces the .o; without it a permanently-"pending"
#: sha would rotate forever. Tests override this.
_POOL_WEDGE_TIMEOUT_S = 300.0


# ---------------------------------------------------------------------------
# Candidate-config compilation
#
# There is no separate precompile phase: the bench loop in ``benchmark()``
# (inside ``Autotuner.__call__``) runs under ``pool_scope()`` from
# frac_ssm.ops.cute.cache_utils.async_compile. A config whose kernel misses the .o cache
# raises ``CompilePending`` from jit_cache after shipping the pickled
# ``_compile_*`` key to a CPU worker; the loop rotates that config to the
# back and benches whichever config is ready. Total wall stays
# max(parallel_compile, serial_bench), key discovery uses the real tensors
# in-process, and workers never launch kernels (they call the tensor-free
# ``_compile_*`` functions directly).
# ---------------------------------------------------------------------------


class Autotuner:
    def __init__(
        self,
        fn,
        key,
        configs,
        restore_value=None,
        prune_configs_by: Optional[Dict] = None,
        cache_results=False,
    ):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It takes configs:List[Config] as its input, and returns pruned configs.
        """
        if not configs:
            self.configs = [AutotuneConfig()]
        else:
            self.configs = configs
        signature = inspect.signature(fn)
        self.signature = signature
        self.keys = key
        self.cache: Dict[Tuple, AutotuneConfig] = {}
        self.arg_names = list(signature.parameters.keys())
        self.cache_results = (
            cache_results or os.getenv(f"{PACKAGE_NAME.upper()}_CACHE_AUTOTUNING", None) == "1"
        )

        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get(
                "early_config_prune", self.early_config_prune
            )

        self.fn = fn

    def _bench(self, *args, config, **meta):
        if meta.keys() & config.kwargs.keys():
            raise ValueError("Conflicting candidate and call arguments")
        try:
            measurement = self._prepared_benchmark.measure(
                extra_kwargs=config.all_kwargs(),
                warmup_ms=200, rep_ms=5, preheat_ms=0, warmup_launches=3,
            )
            return measurement.quantiles((0.5, 0.2, 0.8))
        except (RuntimeError, MemoryError) as exc:
            if os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING") == "1":
                print(f"Autotuning failed with {type(exc).__name__} {exc}")
            return [float("inf")] * 3

    @torch.compiler.disable
    def check_disk_cache(self, tuning_key, configs, bench_fn):
        if not tuning_key:
            bench_fn()
            return

        fn = self.fn
        config_str_list = [str(c) for c in configs]
        assert len(config_str_list) == len(set(config_str_list)), "Config strings must be unique"
        cache_key = [VERSION, PROTOCOL, str(tuning_key)] + config_str_list
        cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = FileCacheManager(_base32(cache_key))
        file_name = f"{fn.__name__[:150]}.autotune.json"
        path = cache.get_file(file_name)
        # There's an environment variable to force cache update
        if path and not os.environ.get(f"{PACKAGE_NAME.upper()}_FORCE_CACHE_UPDATE", False):
            str2config = {s: c for s, c in zip(config_str_list, configs)}
            with open(path, "r") as cached_configs:
                timings = json.load(cached_configs)["configs_timings"]
                timings = {str2config[config]: timing for config, timing in timings}
                self.cache[tuning_key] = builtins.min(timings, key=timings.get)
                self.configs_timings = timings
                self.bench_time = 0
            return

        bench_fn()
        cache.put(
            json.dumps(
                {
                    # str(): the key tuple holds torch dtypes/shape tuples, and
                    # this field is informational (only configs_timings is read back)
                    "key": str(tuning_key),
                    "configs_timings": [
                        (str(config), timings) for config, timings in self.configs_timings.items()
                    ],
                }
            ),
            file_name,
            binary=False,
        )

    def _make_key(self, args, kwargs):
        """Only declared scalar values or tensor dimensions trigger retuning.

        For example, key=["chunk_size", ("u", 2), ("c", 2)] selects R and M
        without including the batch or sequence dimensions. Tensor dtypes and
        devices remain safety keys. Shapes and strides are never implicit.
        """
        named = self.signature.bind_partial(*args, **kwargs)
        named.apply_defaults()
        named = named.arguments
        key = []
        for entry in self.keys:
            if isinstance(entry, str):
                key.append((entry, named[entry]))
            else:
                name, axis = entry
                key.append((name, axis, named[name].shape[axis]))
        key.extend(
            (name, arg.dtype, arg.device)
            for name, arg in named.items() if isinstance(arg, Tensor)
        )
        return tuple(key)

    def __call__(self, *args, **kwargs):
        tensors = [arg for arg in (*args, *kwargs.values()) if isinstance(arg, Tensor)]
        device = next((arg.device for arg in tensors if arg.is_cuda), None)
        context = torch.cuda.device(device) if device is not None else nullcontext()
        with context, _tuning_lock:
            try:
                return self._run(*args, **kwargs)
            finally:
                self.nargs = None

    def _run(self, *args, **kwargs):
        used_cached_result = True
        if len(self.configs) > 1:
            key = self._make_key(args, kwargs)
            if key not in self.cache:
                if any(
                    isinstance(arg, Tensor) and arg.is_cuda
                    for arg in (*args, *kwargs.values())
                ) and torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm up autotuning before CUDA graph capture")
                self.nargs = dict(self.signature.bind_partial(*args, **kwargs).arguments)
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                if not pruned_configs:
                    raise ValueError("Autotuning has no configurations after pruning")

                @torch.compiler.disable  # Don't want any tracing here
                def benchmark():
                    # Compile/bench overlap via the async compile pool
                    # (frac_ssm.ops.cute.cache_utils.async_compile): the bench loop runs inside
                    # pool_scope(). A config whose kernel isn't compiled yet
                    # raises CompilePending from jit_cache (after shipping the
                    # key to a CPU worker); the loop rotates it to the back
                    # and benches whichever config is ready, retrying once its
                    # .o lands. Discovery happens in-process with the real
                    # tensors (no fake-tensor reconstruction), and the pool
                    # workers replay the pickled _compile_* key directly --
                    # which never launches, by construction.
                    #
                    # CompilePending can only fire OUTSIDE CUDA graph capture:
                    # the shared timer does priming launches before capture,
                    # so a cold key raises at the first plain launch.
                    from collections import deque

                    from frac_ssm.ops.cute.cache_utils.async_compile import (
                        CompilePending,
                        pool_scope,
                        suppress_pool,
                    )

                    bench_start = time.time()
                    verbose = os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
                    timings = {}
                    _MAX_ATTEMPTS = 20
                    try:
                        _gpu_warmup()
                        self._prepared_benchmark = PreparedBenchmark(
                            BenchmarkCall(self.fn, args, kwargs),
                            restore_value=self.restore_value,
                        )

                        with pool_scope() as pool:
                            queue = deque(pruned_configs)
                            awaiting = {}  # id(config) -> sha
                            attempts = {}  # id(config) -> int
                            deadline = {}  # id(config) -> wedge deadline
                            spins = 0
                            while queue:
                                config = queue.popleft()
                                sha = awaiting.get(id(config))
                                wedged = sha is not None and time.monotonic() > deadline[id(config)]
                                if sha is not None and not wedged:
                                    state, _ = pool.poll(sha)
                                    if state == "pending":
                                        queue.append(config)
                                        spins += 1
                                        if spins >= len(queue):
                                            time.sleep(0.05)
                                            spins = 0
                                        continue
                                spins = 0
                                n = attempts.get(id(config), 0) + 1
                                attempts[id(config)] = n
                                try:
                                    if wedged or n > _MAX_ATTEMPTS:
                                        # Wedged pool: compile in-process so
                                        # the sweep always terminates.
                                        with suppress_pool():
                                            timings[config] = self._bench(
                                                *args, config=config, **kwargs
                                            )
                                    else:
                                        timings[config] = self._bench(
                                            *args, config=config, **kwargs
                                        )
                                except CompilePending as e:
                                    awaiting[id(config)] = e.sha
                                    deadline.setdefault(
                                        id(config),
                                        time.monotonic() + _POOL_WEDGE_TIMEOUT_S,
                                    )
                                    queue.append(config)
                    finally:
                        prepared = getattr(self, "_prepared_benchmark", None)
                        if prepared is not None:
                            prepared.close()
                        self._prepared_benchmark = None
                    bench_end = time.time()
                    if verbose:
                        for config, time_ in timings.items():
                            print(f"[{config}] -> {time_[0]:.3f}ms")
                    # Surface bench failures (configs returning inf timings)
                    # so smem-overflow / launch errors aren't silently masked.
                    n_failed = sum(1 for t in timings.values() if t[0] == float("inf"))
                    if n_failed:
                        print(
                            f"{PACKAGE_NAME} autotune: {n_failed}/{len(timings)} configs "
                            f"failed for {self.fn.__name__}{key}; "
                            f"set {PACKAGE_NAME.upper()}_PRINT_AUTOTUNING=1 for details",
                            file=sys.stderr,
                        )
                    if n_failed == len(timings):
                        raise RuntimeError(f"All autotuning configurations failed for {self.fn.__name__}")
                    self.bench_time = bench_end - bench_start
                    self.cache[key] = builtins.min(timings, key=timings.get)
                    self.configs_timings = timings

                if self.cache_results:
                    self.check_disk_cache(key, pruned_configs, benchmark)
                else:
                    benchmark()

            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        # Cheap flag first: the getenv (plus its f-string) is measurable on the
        # per-call cache-hit path.
        if not used_cached_result and (
            os.getenv(f"{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING", None) == "1"
        ):
            print(
                f"{PACKAGE_NAME} autotuning for function {self.fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        ret = self.fn(*args, **kwargs, **config.all_kwargs())
        self.nargs = None
        return ret

    def prune_configs(self, kwargs: Dict) -> List[Any]:
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            elif not isinstance(top_k, int):
                # Slice index must be an integer
                raise TypeError(
                    "Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int"
                )

            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs


class AutotuneConfig:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.kwargs = state.get("kwargs", {})

    def all_kwargs(self):
        return self.kwargs

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        return ", ".join(res)

    def __hash__(self):
        return hash(tuple(self.all_kwargs().items()))

    def __eq__(self, other):
        self_tuple = tuple(self.all_kwargs().items())
        other_tuple = tuple(other.all_kwargs().items())
        return self_tuple == other_tuple


def autotune(
    configs, key=None, prune_configs_by=None, restore_value=None, cache_results=True
):
    f"""
    Decorator for auto-tuning a function function.

    .. highlight:: python

    If the environment variable :code:`{PACKAGE_NAME.upper()}_PRINT_AUTOTUNING` is set to
    :code:`"1"`, we will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`AutotuneConfig` objects
    :type configs: list[AutotuneConfig]
    :param key: explicit scalar argument names or (tensor_name, axis) pairs. Tensor dtypes and devices are also keyed.
    :type key: list[str | tuple[str, int]]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param cache_results: whether to cache autotune timings to disk.  Defaults to False.
    "type cache_results: bool
    """

    if key is None:
        key = []

    def decorator(fn):
        return Autotuner(
            fn,
            key,
            configs,
            restore_value=restore_value,
            prune_configs_by=prune_configs_by,
            cache_results=cache_results,
        )

    return decorator
