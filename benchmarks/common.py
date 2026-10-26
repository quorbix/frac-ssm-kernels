from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Sequence

os.environ.setdefault("TORCH_COMPILE_DYNAMIC", "0")

import torch
from frac_ssm.benchmark import (
    PROTOCOL, BenchmarkCall, Measurement, PreparedBenchmark,
    DEFAULT_WARMUP_MS, DEFAULT_REP_MS, DEFAULT_PREHEAT_MS, DEFAULT_WARMUP_LAUNCHES,
)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
SWEEP_LENGTHS = (1024, 2048, 4096, 8192, 32768, 131072)
SWEEP_BATCHES = (2, 16, 64)
DIRECTIONS = ("fwd", "bwd")


@dataclass(frozen=True)
class KernelBenchmark:
    impl: str
    ms: float
    bw: float
    timing: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    kernels: tuple[KernelBenchmark, ...]
    ref_ms: float
    ref_bw: float
    ref_timing: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.kernels:
            raise ValueError("at least one kernel result is required")

    def as_rows(self) -> list[dict[str, object]]:
        return [
            {
                "impl": kernel.impl,
                "kern_ms": kernel.ms,
                "ref_ms": self.ref_ms,
                "kern_bw": kernel.bw,
                "ref_bw": self.ref_bw,
                "timing": {"kernel": kernel.timing, "reference": self.ref_timing},
            }
            for kernel in self.kernels
        ]


def ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this benchmark!")


def dtype_short(dtype_name: str) -> str:
    return {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
    }[dtype_name]


def _validate_backends(
    backends: Sequence[str] | None,
    choices: Sequence[str],
) -> tuple[str, ...]:
    if backends is None:
        backends = choices
    backends = tuple(name.strip() for name in backends)
    if not backends or any(not name for name in backends):
        raise ValueError(
            "Backends must be a comma-separated list without empty entries"
        )

    unknown = tuple(dict.fromkeys(name for name in backends if name not in choices))
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"Unknown backend(s) {names}. Choose from {', '.join(choices)}")

    duplicates = tuple(
        dict.fromkeys(name for name in backends if backends.count(name) > 1)
    )
    if duplicates:
        names = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"Duplicate backend(s) {names}")

    return backends


def _parse_backends(
    value: str,
    choices: Sequence[str],
) -> tuple[str, ...]:
    try:
        return _validate_backends(value.split(","), choices)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_backend_arg(
    parser,
    choices: Sequence[str],
) -> None:
    choices = tuple(choices)
    parser.add_argument(
        "--backends",
        type=lambda value: _parse_backends(value, choices),
        metavar="NAME[,NAME...]",
        help=(
            f"Comma-separated kernel backends to benchmark "
            f"{','.join(choices)} (default all listed backends). "
            "The compiled torch reference is always included."
        ),
    )


def load_backend_modules(
    backends: Sequence[str] | None,
    module_paths: Mapping[str, str],
) -> dict[str, ModuleType]:
    impls = {}
    for backend in _validate_backends(backends, tuple(module_paths)):
        impls[backend] = importlib.import_module(module_paths[backend])
    return impls


def add_benchmark_args(parser) -> None:
    parser.add_argument(
        "--warmup_ms",
        default=DEFAULT_WARMUP_MS,
        type=int,
        help="CUDA graph warmup target in milliseconds.",
    )
    parser.add_argument(
        "--rep_ms",
        default=DEFAULT_REP_MS,
        type=int,
        help="CUDA graph measurement target in milliseconds.",
    )
    parser.add_argument(
        "--preheat_ms",
        default=DEFAULT_PREHEAT_MS,
        type=int,
        help="Run a fixed GEMM preheat before each timed provider. Set 0 to disable.",
    )
    parser.add_argument(
        "--warmup_launches",
        default=DEFAULT_WARMUP_LAUNCHES,
        type=int,
        help="Extra eager launches per provider before timing, used for compile/autotune.",
    )


def cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def bandwidth_gbps(num_bytes: int, ms: float) -> float:
    return round(num_bytes / (ms / 1000) / 1e9, 2)


def benchmark_calls(
    calls: Mapping[str, BenchmarkCall],
    reference: BenchmarkCall,
    warmup_ms: int,
    rep_ms: int,
    *,
    preheat_ms: int = DEFAULT_PREHEAT_MS,
    warmup_launches: int = DEFAULT_WARMUP_LAUNCHES,
) -> tuple[dict[str, Measurement], Measurement]:
    # Compile and tune before timing, retaining only one provider at a time.
    for call in (*calls.values(), reference):
        with PreparedBenchmark(call, max_buffers=1) as prepared:
            prepared.warmup(warmup_launches)
        gc.collect()

    measurements = []
    for call in (*calls.values(), reference):
        with PreparedBenchmark(call) as prepared:
            measurements.append(prepared.measure(
                warmup_ms=warmup_ms, rep_ms=rep_ms, preheat_ms=preheat_ms,
                warmup_launches=warmup_launches,
            ))
        gc.collect()
    return dict(zip(calls, measurements[:-1])), measurements[-1]


def print_result(result: BenchmarkResult) -> None:
    print(f"Ref kernel execution time {result.ref_ms:.4f} ms")
    print(f"Ref mem throughput {result.ref_bw:.2f} GB/s")
    for kernel in result.kernels:
        print(f"{kernel.impl} execution time {kernel.ms:.4f} ms")
        print(f"{kernel.impl} mem throughput {kernel.bw:.2f} GB/s")
        print(f"{kernel.impl} speedup {result.ref_ms / kernel.ms:.2f}x")


@dataclass(frozen=True)
class BenchmarkSpec:
    """Kernel-specific setup. Sweeps, CLI, reporting and storage live here."""

    name: str
    forward: Callable[..., BenchmarkResult]
    backward: Callable[..., BenchmarkResult]
    backends: Mapping[str, str]
    dimensions: Mapping[str, int]
    dtypes: Sequence[str] = ("bfloat16",)
    sweep_dtypes: Sequence[str] = ("bfloat16",)
    takes_dtype: bool = False
    options: Mapping[str, object] = field(default_factory=dict)
    batches: Sequence[int] = SWEEP_BATCHES
    lengths: Sequence[int] = SWEEP_LENGTHS
    # Return a skip reason, or None. Unexpected exceptions abort the run.
    unsupported: Callable[[Mapping[str, object], str, str], str | None] | None = None

    @property
    def axes(self) -> dict[str, int]:
        return {"B": 2, "L": 32768, **self.dimensions}


FULL_SWEEP_AXES = {
    "R": (64, 128, 192),
    "chunk_size": (32, 64, 128),
    "D": (256, 768, 2048),
}
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _values(value) -> tuple:
    values = (value,) if isinstance(value, (int, str)) else tuple(dict.fromkeys(value))
    if not values:
        raise ValueError("sweep axes cannot be empty")
    return values


def resolve_axes(spec: BenchmarkSpec, sweep=None, overrides=None) -> dict[str, tuple[int, ...]]:
    """Explicit lists override presets, including B/L in the small sweep."""
    if sweep not in (None, "small", "full"):
        raise ValueError(f"unknown sweep preset {sweep}")
    axes = {name: (value,) for name, value in spec.axes.items()}
    if sweep:
        axes.update(B=tuple(spec.batches), L=tuple(spec.lengths))
    if sweep == "full":
        axes.update({name: values for name, values in FULL_SWEEP_AXES.items() if name in axes})
    for name, values in (overrides or {}).items():
        if name not in axes:
            raise ValueError(f"unknown sweep axis {name}")
        if values is not None:
            axes[name] = _values(values)
    if any(not isinstance(v, int) or v <= 0 for values in axes.values() for v in values):
        raise ValueError("all dimensions must be positive integers")
    return axes


def _build_parser(spec: BenchmarkSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Benchmark {spec.name} forward and backward")
    for name, default in spec.axes.items():
        parser.add_argument(
            f"--{name}", nargs="+", type=_positive_int,
            help=f"One or more values. Overrides the sweep preset (single-case default {default}).",
        )
    parser.add_argument("--dtype", nargs="+", choices=tuple(spec.dtypes))
    parser.add_argument(
        "--sweep", nargs="?", const="small", choices=("small", "full"),
        help="small existing B/L grid. Full also vary R=64/128/192, CS=32/64/128 "
             "and D=256/768/2048 where applicable. Explicit axes override either.",
    )
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--backward", action="store_true", help="Backward only.")
    direction.add_argument(
        "--directions", nargs="+", choices=DIRECTIONS,
        help="Default fwd for a single/custom grid. Fwd bwd for --sweep.",
    )
    for name, default in spec.options.items():
        if isinstance(default, bool):
            stem = name.removeprefix("has_")
            flag = f"--no_{stem}" if default else f"--{stem}"
            parser.add_argument(flag, dest=name, default=default,
                                action="store_false" if default else "store_true")
        else:
            parser.add_argument(f"--{name}", default=default, type=type(default))
    add_backend_arg(parser, tuple(spec.backends))
    add_benchmark_args(parser)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT,
                        help="Results root. Each new run creates a kernel/timestamp directory.")
    parser.add_argument("--summary", action="store_true", help="Print the full results table.")
    parser.add_argument("--verbose", action="store_true", help="Also print tensor shapes and setup details for each case.")
    saved = parser.add_mutually_exclusive_group()
    saved.add_argument("--report", type=Path,
                       help="Print a saved run without benchmarking. Axis/backend/direction flags filter rows.")
    saved.add_argument("--resume", type=Path,
                       help="Resume a run directory using its saved settings. Retry unfinished and OOM cases.")
    parser.add_argument("--allow-changes", action="store_true",
                        help="Allow source, GPU model or software changes when resuming. Recorded per session.")
    return parser


def _source_metadata() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(
                ("git", *args), cwd=_REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    digest = hashlib.sha256()
    paths = sorted((_REPO_ROOT / "frac_ssm").rglob("*.py"))
    paths += sorted(Path(__file__).resolve().parent.glob("*.py"))
    for path in paths:
        digest.update(path.relative_to(_REPO_ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python_source_sha256": digest.hexdigest(),
    }


def _environment_metadata() -> dict:
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu": {
            "name": props.name,
            "uuid": str(getattr(props, "uuid", "")),
            "device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "compute_capability": [props.major, props.minor],
            "multiprocessor_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": importlib.metadata.version("triton"),
        "cuda": torch.version.cuda,
        **_source_metadata(),
    }


def _new_run_directory(root: Path, metadata: dict) -> Path:
    parent = root / metadata["benchmark"]
    parent.mkdir(parents=True, exist_ok=True)
    name = datetime.fromisoformat(metadata["started_at"]).strftime("%Y%m%dT%H%M%SZ")
    target = parent / name
    try:
        target.mkdir()
    except FileExistsError:
        target = parent / f"{name}-{uuid.uuid4().hex[:8]}"
        target.mkdir()
    return target


@contextmanager
def _run_lock(directory: Path):
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Benchmark directory is already in use {directory}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_json(path: Path, value) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}-",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _save_run(directory: Path, metadata: dict, rows: list[dict]) -> None:
    # One atomic snapshot is authoritative if the process stops between exports.
    _write_json(directory / "checkpoint.json", {"metadata": metadata, "rows": rows})
    _write_json(directory / "metadata.json", metadata)
    _write_json(directory / "results.json", rows)


def _load_run(directory: Path, *, checkpoint_required=False) -> tuple[dict, list[dict]]:
    checkpoint = directory / "checkpoint.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        return saved["metadata"], saved["rows"]
    if checkpoint_required:
        raise ValueError(f"No checkpoint in {directory}. Older runs cannot be resumed.")
    return (json.loads((directory / "metadata.json").read_text()),
            json.loads((directory / "results.json").read_text()))


def _resume_changes(previous: dict, current: dict) -> list[str]:
    changes = [key for key in ("python_source_sha256", "python", "torch", "triton", "cuda")
               if previous.get(key) != current.get(key)]
    changes.extend(f"gpu.{key}" for key in
                   ("name", "compute_capability", "multiprocessor_count", "total_memory_bytes")
                   if previous["gpu"].get(key) != current["gpu"].get(key))
    return changes


def print_sweep_summary(rows: Sequence[Mapping[str, object]], context: str = "") -> None:
    """One dimension-aware table, also usable from a saved results directory."""
    print(f"\n{context} speedup vs same-shape compiled Torch reference")
    if not rows:
        print("No matching results.")
        return
    axes = [name for name in ("B", "L", "H", "D", "R", "M", "chunk_size") if name in rows[0]]
    columns = ["dtype", *axes, "dir", "impl", "status", "kern_ms", "ref_ms", "kern_bw", "ref_bw", "speedup"]
    if len({row.get("session_id") for row in rows}) > 1:
        columns.insert(0, "session_id")
    labels = {"chunk_size": "CS", "kern_ms": "Kern ms", "ref_ms": "Ref ms",
              "kern_bw": "Kern GB/s", "ref_bw": "Ref GB/s", "session_id": "Session"}
    # Put CS comparisons for a given backend/shape next to each other.
    group_axes = [name for name in axes if name != "chunk_size"]
    ordered = sorted(rows, key=lambda row: (
        str(row["dtype"]), *(row[name] for name in group_axes),
        row["dir"] != "fwd", str(row["impl"]), row.get("chunk_size", 0),
    ))
    table = []
    for row in ordered:
        cells = {name: str(row.get(name, "-")) for name in columns}
        if row["status"] == "ok":
            for name in ("kern_ms", "ref_ms"):
                cells[name] = f"{row[name]:.5f}"
            for name in ("kern_bw", "ref_bw"):
                cells[name] = f"{row[name]:.2f}"
            cells["speedup"] = f"{row['ref_ms'] / row['kern_ms']:.2f}x"
        table.append(cells)
    widths = {name: max(len(labels.get(name, name)), *(len(row[name]) for row in table))
              for name in columns}
    print("  ".join(labels.get(name, name).rjust(widths[name]) for name in columns))
    for cells, row in zip(table, ordered):
        line = "  ".join(cells[name].rjust(widths[name]) for name in columns)
        print(line + (f"  ({row['reason']})" if row.get("reason") else ""))


def report_run(spec: BenchmarkSpec, args) -> None:
    metadata, rows = _load_run(args.report)
    if metadata["benchmark"] != spec.name:
        raise ValueError(f"saved run is for {metadata['benchmark']}, not {spec.name}")
    filters = {name: getattr(args, name) for name in spec.axes if getattr(args, name) is not None}
    if args.dtype:
        filters["dtype"] = [dtype_short(name) for name in args.dtype]
    if args.backends:
        filters["impl"] = args.backends
    if args.directions or args.backward:
        filters["dir"] = ("bwd",) if args.backward else args.directions
    rows = [row for row in rows if all(row[name] in values for name, values in filters.items())]
    print(f"Started {metadata['started_at']}. Options {metadata['options']}")
    if "sessions" in metadata:
        print(f"Status {metadata['status']}. Sessions {len(metadata['sessions'])}")
        if len(metadata["sessions"]) > 1:
            print("Measurements span separate sessions. See metadata for each environment.")
    print_sweep_summary(rows, f"{spec.name} {args.report}")


def run_benchmark(
    spec: BenchmarkSpec,
    *,
    axes: Mapping[str, object] | None = None,
    sweep: str | None = "small",
    dtypes: Sequence[str] | None = None,
    directions: Sequence[str] = DIRECTIONS,
    backends: Sequence[str] | None = None,
    options: Mapping[str, object] | None = None,
    warmup_ms: int = DEFAULT_WARMUP_MS,
    rep_ms: int = DEFAULT_REP_MS,
    preheat_ms: int = DEFAULT_PREHEAT_MS,
    warmup_launches: int = DEFAULT_WARMUP_LAUNCHES,
    output_dir: Path = RESULTS_ROOT,
    summary: bool = False,
    verbose: bool = False,
    resume: Path | None = None,
    allow_changes: bool = False,
) -> list[dict[str, object]]:
    """Checkpoint whole comparisons and continue after case-level CUDA OOMs.

    Resume restores the saved settings and retries unfinished or OOM cases.
    Unexpected errors stop the session without losing completed comparisons.
    """
    with ExitStack() as stack:
        metadata, rows = None, []
        if resume is not None:
            directory = Path(resume)
            stack.enter_context(_run_lock(directory))
            metadata, rows = _load_run(directory, checkpoint_required=True)
            if metadata["benchmark"] != spec.name:
                raise ValueError(f"Checkpoint does not match benchmark {spec.name}")
            if metadata["timing"].get("protocol") != PROTOCOL:
                raise ValueError("Checkpoint uses a different timing protocol. Start a new run.")
            if set(metadata["axes"]) != set(spec.axes) or set(metadata["options"]) != set(spec.options):
                raise ValueError("Checkpoint dimensions or kernel options no longer match this benchmark")
            if any(spec.backends.get(name) != module for name, module in metadata["backends"].items()):
                raise ValueError("Checkpoint backend definitions no longer match this benchmark")
            axes, sweep = metadata["axes"], metadata["sweep"]
            dtypes, directions = metadata["dtypes"], metadata["directions"]
            backends, options = tuple(metadata["backends"]), metadata["options"]
            timing = {name: metadata["timing"][name] for name in
                      ("warmup_ms", "rep_ms", "preheat_ms", "warmup_launches")}
        else:
            timing = dict(warmup_ms=warmup_ms, rep_ms=rep_ms,
                          preheat_ms=preheat_ms, warmup_launches=warmup_launches)

        resolved = resolve_axes(spec, sweep, axes)
        if metadata is not None:
            # Saved axis order defines the stable case indices across sessions.
            resolved = {name: resolved[name] for name in metadata["axes"]}
        backends = _validate_backends(backends, tuple(spec.backends))
        dtypes = _values(dtypes if dtypes is not None else spec.sweep_dtypes if sweep else spec.dtypes[:1])
        directions = _values(directions)
        if any(dtype not in spec.dtypes for dtype in dtypes):
            raise ValueError(f"supported dtypes {', '.join(spec.dtypes)}")
        if any(direction not in DIRECTIONS for direction in directions):
            raise ValueError(f"directions must be chosen from {DIRECTIONS}")
        if timing["warmup_ms"] < 0 or timing["rep_ms"] <= 0 or timing["preheat_ms"] < 0 or timing["warmup_launches"] < 1:
            raise ValueError("timing windows must be nonnegative, rep_ms > 0 and warmup_launches >= 1")
        if set(options or {}) - set(spec.options):
            raise ValueError("unknown benchmark options")
        options = {**spec.options, **(options or {})}

        count = math.prod(map(len, resolved.values())) * len(dtypes) * len(directions)
        completed = set()
        previous_cases = {}
        for row in rows:
            previous_cases.setdefault(row["case_index"], []).append(row)
        for index, case_rows in previous_cases.items():
            if not 1 <= index <= count or len(case_rows) != len(backends) or {r["impl"] for r in case_rows} != set(backends):
                raise ValueError("Checkpoint contains an incomplete or invalid comparison")
            if any(r["status"] not in ("ok", "skipped", "oom") for r in case_rows):
                raise ValueError("Checkpoint contains an unknown measurement status")
            if all(r["status"] != "oom" for r in case_rows):
                completed.add(index)

        mode = "resume" if resume is not None else f"{sweep} sweep" if sweep else "custom grid" if count > 1 else "single case"
        print(f"\n=== {spec.name} {mode} ===")
        for name, values in resolved.items():
            print(f"  {name} {list(values)}")
        print(f"  Dtypes {', '.join(dtype_short(dtype) for dtype in dtypes)}")
        print(f"  Directions {', '.join(directions)}")
        print("  Implementations")
        for backend in backends:
            print(f"    {backend}")
        print("  Reference compiled Torch")
        if options:
            print("  Options " + ", ".join(f"{name}={value}" for name, value in options.items()))
        print("  Timing " + ", ".join(f"{name}={value}" for name, value in timing.items()))
        print(f"  Protocol {PROTOCOL}")
        print(f"  Cases {count}, restored {len(completed)}, remaining {count - len(completed)}")
        if len(completed) == count:
            print(f"No pending cases. Saved {directory}", flush=True)
            if summary:
                print_sweep_summary(rows, spec.name)
            return rows

        ensure_cuda()
        environment = _environment_metadata()
        started = datetime.now(timezone.utc).isoformat()
        changes = []
        if metadata is None:
            metadata = {
                "schema_version": 3, "benchmark": spec.name, "started_at": started,
                "sweep": sweep, "axes": resolved, "dtypes": dtypes, "directions": directions,
                "backends": {name: spec.backends[name] for name in backends},
                "options": options, "seed": 0, "seed_scope": "case",
                "timing": {**timing, "protocol": PROTOCOL, "timer": "frac_ssm.benchmark",
                           "statistic": "median_of_batch_means",
                           "cache": "bounded rotating buffers", "preheat": "fixed bf16 GEMM",
                           "preheat_matrix_size": int(os.environ.get("FRAC_SSM_BENCH_PREHEAT_SIZE", "4096")),
                           "provider_order": [*backends, "torch_reference"]},
                "environment": environment, "sessions": [],
            }
            directory = _new_run_directory(Path(output_dir), metadata)
            stack.enter_context(_run_lock(directory))
        else:
            changes = _resume_changes(metadata["environment"], environment)
            if changes and not allow_changes:
                raise ValueError("Resume environment changed " + ", ".join(changes)
                                 + ". Start a new run or explicitly use --allow-changes.")
            if changes:
                print("Warning resuming with changes " + ", ".join(changes), flush=True)
            print("Resuming in a separate measurement session. Providers warm up before timing.", flush=True)
            for previous in metadata["sessions"]:
                if previous["status"] == "running":
                    previous["status"] = "interrupted"
            os.environ["FRAC_SSM_BENCH_PREHEAT_SIZE"] = str(metadata["timing"]["preheat_matrix_size"])

        session = {"id": len(metadata["sessions"]) + 1, "started_at": started,
                   "environment": environment, "changes_from_original": changes,
                   "status": "running", "oom_cases": []}
        metadata["sessions"].append(session)
        elapsed_start = time.perf_counter()
        backend_order = {name: i for i, name in enumerate(backends)}

        def checkpoint(status="running"):
            now = datetime.now(timezone.utc).isoformat()
            session.update(status=status, elapsed_seconds=time.perf_counter() - elapsed_start)
            if status != "running":
                session["finished_at"] = now
            metadata.update(status=status, updated_at=now,
                            finished_at=now if status.startswith("completed") else None,
                            elapsed_seconds=sum(s.get("elapsed_seconds", 0) for s in metadata["sessions"]),
                            counts={s: sum(r["status"] == s for r in rows) for s in ("ok", "skipped", "oom")})
            metadata["counts"]["oom_cases"] = len({r["case_index"] for r in rows if r["status"] == "oom"})
            _save_run(directory, metadata, rows)

        checkpoint()
        print(f"Checkpoint {directory}", flush=True)
        active_case = None
        try:
            grid = itertools.product(*(resolved.values()), dtypes, directions)
            for index, values in enumerate(grid, 1):
                case = dict(zip(resolved, values[:-2]))
                dtype, direction = values[-2:]
                shape = " ".join(f"{name}={value}" for name, value in case.items())
                print(f"[{index}/{count}] {shape} {dtype_short(dtype)} {direction} ...",
                      end="\n" if verbose else " ", flush=True)
                if index in completed:
                    print("RESTORED", flush=True)
                    continue
                active_case = {**case, "dtype": dtype_short(dtype), "dir": direction, "case_index": index}
                row_base = {**active_case, **options, "session_id": session["id"]}
                case_rows, selected = [], []
                for backend in backends:
                    reason = spec.unsupported(case, backend, direction) if spec.unsupported else None
                    if reason:
                        case_rows.append({**row_base, "impl": backend, "status": "skipped", "reason": reason})
                        print(f"{backend}=SKIPPED ({reason})", end="\n" if verbose else "  ", flush=True)
                    else:
                        selected.append(backend)
                if selected:
                    cleanup_cuda_memory()
                    # The case seed also reproduces inputs after skipping restored cases.
                    torch.manual_seed(metadata["seed"])
                    # Dynamo caches per code object, so a second shape would switch
                    # the compiled reference to dynamic shapes (much slower for the
                    # chunk loops) and later shapes would exhaust the recompile limit.
                    torch._dynamo.reset()
                    fn = spec.backward if direction == "bwd" else spec.forward
                    call_args = {**case, **options, **timing, "backends": tuple(selected), "verbose": verbose}
                    if spec.takes_dtype:
                        call_args["dtype_name"] = dtype
                    try:
                        result = fn(**call_args)
                    except torch.OutOfMemoryError as exc:
                        reason = str(exc)
                        session["oom_cases"].append({**active_case, "reason": reason})
                        case_rows.extend({**row_base, "impl": name, "status": "oom", "scope": "case",
                                          "reason": reason} for name in selected)
                        print(f"CASE OOM {reason.splitlines()[0] if reason else 'CUDA out of memory'}. Continuing.", flush=True)
                    else:
                        measured = result.as_rows()
                        if len(measured) != len(selected) or {r["impl"] for r in measured} != set(selected):
                            raise ValueError("Benchmark did not return exactly the selected implementations")
                        case_rows.extend({**row_base, **row, "status": "ok"} for row in measured)
                        if not verbose:
                            parts = "  ".join(f"{k.impl}={k.ms:.4f}ms/{result.ref_ms / k.ms:.2f}x" for k in result.kernels)
                            print(f"ref={result.ref_ms:.4f}ms  {parts}", flush=True)
                    # Leave the exception handler before releasing failed autograd graphs.
                    cleanup_cuda_memory()
                elif not verbose:
                    print(flush=True)
                rows = [r for r in rows if r["case_index"] != index] + case_rows
                rows.sort(key=lambda r: (r["case_index"], backend_order[r["impl"]]))
                checkpoint()
            checkpoint("completed_with_oom" if metadata["counts"]["oom_cases"] else "completed")
        except BaseException as exc:
            session["error"] = {"type": type(exc).__name__, "message": str(exc), "case": active_case}
            checkpoint("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            print(f"\nProgress saved {directory}", file=sys.stderr, flush=True)
            raise

        counts = metadata["counts"]
        print(f"Completed {counts['ok']} measurements, {counts['skipped']} skipped, "
              f"{counts['oom_cases']} OOM cases.\nSaved {directory}", flush=True)
        if summary:
            context = f"{spec.name}. Options {options}" if options else spec.name
            print_sweep_summary(rows, context)
        return rows


def benchmark_main(spec: BenchmarkSpec, argv=None) -> int:
    parser = _build_parser(spec)
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    if args.allow_changes and not args.resume:
        parser.error("--allow-changes requires --resume")
    if args.resume:
        allowed = {"--resume", "--allow-changes", "--summary", "--verbose"}
        if any(arg.startswith("--") and arg.split("=", 1)[0] not in allowed for arg in argv):
            parser.error("--resume restores saved settings. Do not pass sweep, dimension, backend or timing flags.")
    if args.report:
        report_run(spec, args)
        return 0
    directions = ("bwd",) if args.backward else args.directions or (DIRECTIONS if args.sweep else ("fwd",))
    try:
        run_benchmark(
            spec, axes={name: getattr(args, name) for name in spec.axes},
            sweep=args.sweep, dtypes=args.dtype, directions=directions,
            backends=args.backends,
            options={name: getattr(args, name) for name in spec.options},
            warmup_ms=args.warmup_ms, rep_ms=args.rep_ms, preheat_ms=args.preheat_ms,
            warmup_launches=args.warmup_launches,
            output_dir=args.output_dir, summary=args.summary, verbose=args.verbose,
            resume=args.resume, allow_changes=args.allow_changes,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Use --resume with the checkpoint directory to continue.", file=sys.stderr)
        return 130
    except Exception:
        print("\nBenchmark aborted. Any completed cases remain in the checkpoint directory.", file=sys.stderr)
        raise
    return 0
