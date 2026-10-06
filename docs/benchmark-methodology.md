# Benchmark methodology (scaffold)

## Current measurement contract

The initial workload is square FP32 matmul, `C = A @ B`, on one explicitly
selected device. NumPy generates inputs using seed 0; JAX uses `HIGHEST` matmul
precision. Inputs are placed on-device and synchronized before timing.

1. Disable JAX's persistent compilation cache for this smoke run.
2. Measure JIT lowering plus compilation separately with a monotonic clock.
3. Execute once, synchronize, and compare with NumPy (`rtol=1e-4`, `atol=1e-4`).
   Abort on incorrect or non-finite output.
4. Run the requested untimed warmups with synchronization.
5. Time each execution through `block_until_ready()`. Retain all samples and
   report median, minimum, and maximum in milliseconds.

Execution timings include host dispatch and synchronization. Input generation,
host/device transfer, compilation, correctness checks, and JSON writing are
excluded. These are latency samples, not pure device kernel time. Repeated calls
reuse resident inputs; the smoke harness does not sweep sizes or control caches,
CPU threads, clocks, thermals, or competing load.

Each JSON report (schema version 2) also has a `runtime` section: the
`JAX_PLATFORMS` selection as JAX read it (`null` when unset), the platform
version string reported by the XLA client of the measured device, and the
versions of any installed JAX CUDA or ROCm plugin wheels. Because `--backend gpu`
covers both CUDA and ROCm, these fields are meant to show which accelerator
runtime produced a GPU report. On the CPU baseline they read `cpu`, `cpu`, and
an empty mapping. They have not yet been exercised on accelerator hardware.

The approach follows [JAX's benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html)
on compilation, synchronization, dtype, and data placement.

## Before publishing a comparative result

- State the question, baseline, workload, shapes, dtype, precision, seed,
  tolerances, warmups, iterations, and exact commands before running.
- Record repository commit and working-tree state; archive dependency versions
  or lockfile, container digest (if used), and reviewed configuration.
- Record CPU/GPU model and counts, memory, OS, driver, CUDA/ROCm, plugin/XLA,
  RCCL, power/clock settings, CPU thread controls, and concurrent load.
- Keep raw JSON and correctness evidence. Repeat in independent fresh processes
  under comparable conditions; show variation across runs, not just a best run.
- Compare identical math, dtype, device counts, precision, and timing boundaries.
  Record failures, out-of-memory cases, and exclusions.
- Define throughput or bandwidth explicitly. Separate compilation, transfers,
  steady-state latency, and end-to-end time. Use a profiler for kernel-only claims.
- Sanitize artifacts before upload; exclude hostnames, user paths, credentials,
  and arbitrary environment dumps.

## Result record template

```text
Question / baseline:
Commit / working-tree state:
Hardware / topology / device counts:
OS / driver / accelerator runtime / library versions:
Container or dependency manifest:
Command / explicit configuration:
Workload / shape / dtype / precision / seed:
Correctness oracle / tolerances / result:
Warmups / iterations / independent runs:
Timing boundaries / clock / synchronization:
Raw artifacts / summary / between-run variation:
Failures / exclusions / limitations:
Conclusion supported by these measurements:
```

RCCL records require message sizes, collective type, rank count, topology, and a
documented bandwidth definition; the first one is
[rccl-mi210-baseline.md](rccl-mi210-baseline.md). FFI will require stream,
buffer, and ABI checks. MaxText will require model/config revision, batch and
sequence lengths, sharding, loss checks, and a clear step-time boundary. These
two extensions remain future work.
