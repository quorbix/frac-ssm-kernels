# Benchmarks

Run from the repository root after installing the package in your CUDA-enabled
environment. Choose an idle GPU; GPU 0 in the examples is a placeholder. Do not
run benchmarks alongside another workload on the same GPU.

All scripts share the same CLI. Each script's `--help` lists its supported
dimensions, backends, dtypes, and options.

**Omitting `--backends` runs every registered implementation for that benchmark.**
Use `--backends triton` to run only Triton, or a comma-separated list such as
`--backends triton,cute_sm90` to select a subset. The compiled Torch reference
is always included. Implementation choices are listed under each kernel below;
their hardware and shape requirements still apply.

## Single runs and sweep modes

Without `--sweep`, a command measures one shape and forward only by default:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_chunk_fused_readout.py \
  --B 2 --L 8192 --R 128 --M 16 --chunk_size 64
```

- `--sweep` or `--sweep small`: the benchmark's standard B/L grid and sweep
  dtypes, with both forward and backward. Other dimensions stay fixed.
- `--sweep full`: also varies R over 64/128/192, chunk size over 32/64/128,
  and D over 256/768/2048, where those dimensions apply.
- Explicit dimension lists override a preset. All combinations of the selected
  values are measured.
- `--directions fwd`, `--directions bwd`, or `--backward` restricts the direction.
  A custom grid without `--sweep` defaults to forward; use
  `--directions fwd bwd` for both.

**Supplying `--B` or `--L` limits the sweep.** For example, adding `--sweep`
to the single-shape command above keeps B=2 and L=8192; it does not expand them
to the standard grid. Omit these flags to sweep the preset batches and lengths.

## Per-kernel sweeps

Each command below uses `--sweep`: standard batches and lengths, both
directions, the kernel-specific dimensions shown, and all listed implementations.
Add `--backends` only when you want to restrict the implementations.

### Chunk cumsum

Implementations: `triton`, `cute_sm90`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_chunk_cumsum.py \
  --sweep --M 16 --chunk_size 64
```

### Chunk W

Implementations: `triton`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_chunk_w.py \
  --sweep --M 16 --chunk_size 64
```

### Chunk state

Implementations: `triton`, `cute_sm90`, `cute_sm90_tma_k_lcr`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_chunk_state.py \
  --sweep --R 128 --M 16 --chunk_size 64
```

### Chunk readout

Implementations: `triton`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_chunk_readout.py \
  --sweep --R 128 --M 16 --chunk_size 64
```

### Fused chunk readout

Implementations:

- `triton`: chunk W followed by chunk readout.
- `cute_sm90`: autotuned persistent forward and backward.
- `cute_sm90_warp_specialized`: autotuned warp-specialized forward with unrolled consumer loops.
- `cute_sm90_cooperative`: cooperative forward and backward.

All entries support backward through their wrappers.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_chunk_fused_readout.py \
  --sweep --R 128 --M 16 --chunk_size 64
```

### State passing

Implementations: `triton`, `cute_sm90`, `cute_sm90_seq`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_state_passing.py \
  --sweep --R 128 --M 16 --chunk_size 64
```

Add `--initial_state` to include an initial state.

### Frac scan

Implementations: `triton`, `cute_sm90`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_frac_scan.py \
  --sweep --R 128 --M 16 --chunk_size 64
```

Add `--initial_state` to include an initial state.

### Compute parameters

Implementations: `triton`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_compute_params.py \
  --sweep --H 12 --R 128 --M 16 --dtype bfloat16
```

Add `--no_mask` to disable the attention mask.

### Output mix

Implementations: `triton`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_output_mix.py \
  --sweep --H 12 --R 128 --dtype bfloat16
```

### RMSNorm

Implementations: `triton`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_rmsnorm.py \
  --sweep --D 768 --dtype bfloat16 float32
```

### Gated RMSNorm

Implementations: `triton`.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_rmsnorm_gated.py \
  --sweep --D 768 --dtype bfloat16 float32
```

Add `--no_gate` to disable the gate. Both RMSNorm benchmarks also accept `--eps`.

## Custom grids and backend comparisons

Compare all fused-readout implementations over selected shapes:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_chunk_fused_readout.py \
  --B 2 --L 1024 8192 --R 64 128 --M 16 --chunk_size 32 64 128 \
  --directions fwd bwd
```

For only cooperative and persistent, add
`--backends cute_sm90_cooperative,cute_sm90` to that command.

To use the full R/chunk-size preset but limit batches, lengths, and direction:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_chunk_fused_readout.py \
  --sweep full --B 2 --L 1024 8192 --directions fwd
```

## Progress and saved results

Before running, the benchmark prints the resolved grid: every dimension's
values, dtypes, directions, selected implementations, kernel options, timing
settings, and the total case count.

Every case prints its dimensions, dtype, and direction before running, followed
by the reference time and each backend's time and speedup when it completes.
Skipped backends are printed with their reasons. Progress is printed for every
case, without throttling.

The end of the run prints only the measurement, skip, and OOM counts and saved directory
unless you request the full summary table.

- `--summary`: also print the full results table.
- `--verbose`: also print tensor shapes and setup details for each case.
- `--output-dir PATH`: change the results root.

New runs create `benchmarks/results/<benchmark>/<timestamp>/` immediately and
print its path. After every complete shape/dtype/direction comparison, the runner
atomically replaces `checkpoint.json` with the settings, session history, and
results so far. It also refreshes `metadata.json` and `results.json` for analysis.
The checkpoint is authoritative if a process stops between those exports.
Writes happen between cases, outside the timing windows.

CUDA out-of-memory errors are recorded as `oom` and the sweep continues after
memory cleanup. This is a **case-level** result, not a claim that each selected
kernel ran out of memory. For example, the Torch reference can fail before the
comparison is measured. Such a case has no valid timings or speedups. Known
unsupported combinations remain `skipped`. Other errors still abort, and Ctrl-C
still stops the run, but completed comparisons are preserved.

## Resume

Use the same benchmark script and the directory printed after `Checkpoint` or
`Saved`. No need to repeat the original sweep arguments.

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_chunk_fused_readout.py \
  --resume PATH_TO_RUN
```

- Saved dimensions, dtypes, directions, implementations, kernel options, and
  timing settings are restored. Do not combine `--resume` with overrides.
- Completed comparisons are printed as `RESTORED` and are not remeasured.
  Unfinished comparisons and recorded OOM cases are retried. Another OOM is
  recorded and the run continues again. Old OOM diagnostics remain in the
  session history even if a retry succeeds.
- Checkpoints cover whole comparisons, including all selected implementations
  and the reference. An interrupted comparison starts over. Input seeds are
  reset per case so skipping restored cases does not change later random inputs.
- Every invocation records its own environment and timestamps. Rows identify
  their measurement session. Resuming is not treated as an uninterrupted run.
  Use an idle GPU and allow normal warmup. A different GPU of the same model is
  allowed and its identity is recorded.
- Changed source code, GPU model, or software versions are rejected by default.
  Prefer a fresh run for performance comparisons. If mixing sessions under
  changed conditions is intentional, add `--allow-changes`. Differences are
  printed and recorded. Two processes cannot write the same run concurrently.
- A different measurement protocol cannot be resumed, even with `--allow-changes`.
  Old event-timed runs remain readable with `--report`. Start a new directory
  for graph-timed measurements.
- `--summary` and `--verbose` are also accepted with `--resume`.

An OOM can still recur if the reference and retained backward graphs do not fit
in memory. Resume preserves work but does not reduce the memory requirement.
Runs made before checkpoint support have no checkpoint and cannot be resumed.

Read or filter either a completed or partial run without a GPU or rerunning
measurements. The report reads the latest atomic checkpoint when present.

```bash
python benchmarks/benchmark_chunk_fused_readout.py \
  --report PATH_TO_RUN --R 128 --directions fwd
```

Times are in ms, estimated bandwidth in GB/s, and speedup is relative to the
same-shape compiled Torch reference, which is always included.

## Timing options

The shared timer in `frac_ssm/benchmark.py` is also used by CuTe autotuning.
It reports median batch-average GPU latency from CUDA graphs, with input buffers
rotated between calls. Python launch overhead, compilation, input preparation
and buffer restoration are outside the measured intervals.

Backward benchmarks run forward once per rotation slot to prepare its saved
autograd state, then measure backward only. Providers are prepared and released
one at a time.
Modules, parameters and saved activations are included in this preparation.
Copies preserve tensor strides and pointer alignment. Distinct aliased input
views are rejected rather than silently losing their alias relationship.

Rotation is bounded by memory headroom and may use only one slot for a large
workload. Small workloads can still fit in L2, so rotation does not guarantee
a cold cache. Each result stores the actual slot count, calls per batch and
individual batch measurements. Capture failures are reported without silently
switching to another timer.

- `--warmup_ms`: GPU graph warmup target, default 10 ms.
- `--rep_ms`: GPU measurement target, default 100 ms, with at least five batches.
- `--preheat_ms`: fixed GEMM preheat before each timed provider, default 200 ms.
- `--warmup_launches`: eager launches for compilation/autotuning before timing,
  default 5.

CuTe candidate timing uses the same engine with a 200 ms warmup target and a
5 ms measurement target. Its compilation pool and tuning keys are unchanged.
Triton's own internal autotuner still uses its native measurement policy.
