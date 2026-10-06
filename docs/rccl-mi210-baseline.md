# RCCL AllReduce baseline on 2x MI210

Roadmap stage 1 ([issue #1](https://github.com/egesamuray/jax-rocm-systems-lab/issues/1)).
Measured 2026-10-06. Raw evidence:
[`results/rccl/mi210-2gpu-allreduce-20261006/`](../results/rccl/mi210-2gpu-allreduce-20261006/).

## Question

Can the available MI210 configuration execute a correct, reproducible RCCL
AllReduce, and what are the baseline latency and bandwidth samples?

On the configuration below: yes. Two ranks, one MPI process per GPU, completed
every AllReduce with zero wrong elements in all checks, across three independent
runs. The numbers are a baseline for this setup only, not an optimization result
or a comparison.

## Environment

| Item | Recorded value |
| --- | --- |
| GPUs | 2x AMD Instinct MI210, `gfx90a:sramecc+:xnack-`, 104 CUs, VBIOS 113-D67301-059 |
| Placement | Both GPUs in one node (same-node run, 2 of 2 GPUs on the node) |
| GPU-to-GPU path | PCIe only (`rocm-smi`: link type PCIE, 2 hops, weight 40; no xGMI). Each GPU is PCIe Gen4 x16 behind its own Broadcom PEX880xx switch, on two different root complexes of CPU socket 0 |
| CPU / NUMA | 2x AMD EPYC 7713; both GPUs on NUMA node 0; the job held 16 cores of NUMA node 0 and both ranks were bound to them |
| OS / driver | RHEL 9.8, kernel 5.14.0-687.38.1.el9_8.x86_64, amdgpu 6.16.13 |
| ROCm / HIP | ROCm 7.2.0 (7.2.0-43), HIP 7.2.26015-fc0010cf6a, HSA runtime 1.18 |
| RCCL | 2.27.7, site package `rccl-2.27.7.70200-43.el8` (runtime banner `2.27.7-HEAD:0d2c4fd`) |
| MPI | Open MPI 5.0.11, built for this run from the checksum-verified release tarball; used only to start the ranks, share the RCCL unique ID, and combine timings and error counts |
| rccl-tests | [ROCm/rccl-tests](https://github.com/ROCm/rccl-tests) `40b1b17901370a7880d4a56854b5361c89f8d324`, built with `MPI=1 GPU_TARGETS=gfx90a` (gfx90a code objects only) |
| Scheduler context | One Slurm allocation; no other Slurm jobs on the node and both GPUs at 0% use before each run |
| Environment variables | `NCCL_DEBUG=VERSION` only (plus Slurm's `ROCR_VISIBLE_DEVICES=0,1`); no RCCL tuning variables |

The standalone `ROCm/rccl-tests` repository is now marked retired, and
development continues in [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems).
The pinned commit is the final state of the standalone repository. Exact tool
output is in [`versions.txt`](../results/rccl/mi210-2gpu-allreduce-20261006/versions.txt)
and [`topology.txt`](../results/rccl/mi210-2gpu-allreduce-20261006/topology.txt).
JAX is not part of this measurement.

## Command

Build Open MPI 5.0.11 and rccl-tests as listed in
[`commands.txt`](../results/rccl/mi210-2gpu-allreduce-20261006/commands.txt),
inside an allocation with both GPUs of one node. Then run each repetition as a
fresh process:

```bash
export LD_LIBRARY_PATH=<openmpi-prefix>/lib:$LD_LIBRARY_PATH
mpirun -np 2 --bind-to numa -x NCCL_DEBUG=VERSION -x LD_LIBRARY_PATH \
  <rccl-tests>/build/all_reduce_perf -g 1 -d float -o sum \
  -b 8 -e 128M -f 2 -w 5 -n 20 -c 1 -Z json -x raw-01.json < /dev/null > raw-01.log
python -m benchmarks.rccl_report raw-01.json raw-02.json raw-03.json \
  --ranks 2 --output summary.json
```

`benchmarks/rccl_report.py` needs only the Python standard library. It reads
each JSON file together with the stdout log of the same name.

## Correctness

- **Built-in validator (required).** For each size and placement, rccl-tests
  first runs the timed iterations. It then refills the buffers with seeded
  per-rank inputs, runs one more AllReduce (`-c 1`), and compares every output
  element on every rank with an expected result generated on the GPU. For FP32
  sum the inputs are constructed so the exact sum does not depend on reduction
  order, and the comparison tolerance is 0 ULP. `#wrong` is the number of
  mismatching elements summed over ranks. The check runs on separate iterations
  from the timed ones.
- **Result.** `#wrong` is 0 in all 150 baseline records (25 sizes x 2 placements
  x 3 runs) and all 22 smoke records (1 KiB to 1 MiB). Every log ends with
  `Out of bounds values : 0 OK`. `rccl_report.py` rejects any record whose
  `#wrong` is non-zero or `N/A`, so the summary could not have been produced
  otherwise.
- **Known-value check (secondary).** [`benchmarks/rccl_known_value.cpp`](../benchmarks/rccl_known_value.cpp)
  has rank r contribute the constant r + 1. Every output element must equal 3.0
  exactly, and the receive buffer is pre-filled with NaN. It covered 1, 1000,
  2^20 and 2^24 + 3 elements, out-of-place and in-place: 71,307,184 elements in
  total over both ranks, with 0 wrong
  ([`known-value.log`](../results/rccl/mi210-2gpu-allreduce-20261006/known-value.log)).
- **Participation.** Every log lists rank 0 on device `0000:08:00` and rank 1 on
  device `0000:48:00`.

## Measurement definition

- AllReduce, FP32 (`-d float`), sum, 2 ranks, 1 GPU per rank (`-g 1`, one MPI
  process per GPU), out-of-place and in-place. RCCL picks the algorithm itself
  (default selection).
- Message size S is the bytes per rank (element count x 4). The sweep runs from
  8 B to 128 MiB by factors of 2: 25 sizes.
- Outside the timed region: process launch, RCCL communicator setup and buffer
  allocation. Before the sweep there are 5 untimed AllReduce calls at 128 MiB and
  5 at 8 B (`-w 5`). Each size and placement also gets 1 untimed call before
  timing.
- Timed region: after an MPI barrier, 20 back-to-back AllReduce calls are
  enqueued on one stream (`-n 20`). A host `steady_clock` timer runs from before
  the first enqueue until polling shows the stream has completed. `time` is
  elapsed / 20, averaged over the two ranks (rccl-tests default `-a 1`). This is
  host-observed time per collective, launch overhead included, not kernel-only
  profiler time. rccl-tests reports one mean per size and placement per run, not
  per-iteration samples.
- `algbw = S / t`, in GB/s with GB = 1e9 bytes.
- `busbw = algbw * 2 * (n - 1) / n` for AllReduce with n ranks. With n = 2,
  `busbw = algbw`.
- `rccl_report.py` recomputes `algbw` from `size` and `time` for every record and
  checks `busbw` against the formula. The largest difference was 5.0e-7 GB/s,
  which is rounding in the 6-decimal JSON fields.
- Repetitions: 3 independent `mpirun` invocations with identical arguments,
  run back to back in the same allocation.

## Results

Out-of-place records: median of the 3 runs, with the minimum and maximum
([`summary.json`](../results/rccl/mi210-2gpu-allreduce-20261006/summary.json)
has every size, both placements, and all raw samples).

| Size | Time median [min, max] (us) | algbw = busbw, median (GB/s) | Time spread across runs |
| --- | --- | --- | --- |
| 8 B | 17.84 [17.64, 18.16] | 0.000448 | 2.9% |
| 1 KiB | 16.58 [16.56, 16.69] | 0.0618 | 0.8% |
| 8 KiB | 17.16 [17.15, 17.53] | 0.477 | 2.2% |
| 16 KiB | 27.73 [27.50, 28.05] | 0.591 | 2.0% |
| 64 KiB | 27.12 [27.11, 27.25] | 2.42 | 0.5% |
| 1 MiB | 68.27 [67.41, 68.58] | 15.4 | 1.7% |
| 4 MiB | 199.5 [199.4, 199.7] | 21.0 | 0.2% |
| 16 MiB | 730.9 [730.9, 730.9] | 23.0 | 0.0% |
| 64 MiB | 2867.9 [2866.8, 2868.2] | 23.4 | 0.0% |
| 128 MiB | 5714.3 [5706.7, 5716.3] | 23.5 | 0.2% |

Spread is (max - min) / median of the per-run times.

- Up to 8 KiB, the median time per AllReduce stays between 16.2 and 19.6 us.
  From 16 KiB to 128 KiB it is between 26.2 and 29.8 us. Beyond that it grows
  with message size.
- That step lines up with RCCL's default selection. A separate diagnostic run
  with `-M 1` reported Ring/LL up to 8 KiB and Ring/Simple from 16 KiB
  ([`algo-info.log`](../results/rccl/mi210-2gpu-allreduce-20261006/algo-info.log)).
  That run is not part of the baseline.
- From 16 MiB to 128 MiB, busbw medians are between 22.95 and 23.51 GB/s.
- 46 of the 50 size/placement points have a spread of 3.5% or less (median
  1.1%). The largest spreads are 33% at 32 KiB in-place, where one run took
  35.16 us against 26.39 and 26.47 us, and 8.8% at 2 MiB out-of-place. Those
  outliers are left in, not removed.
- From 1 MiB up, in-place and out-of-place medians agree within 0.9%. Below
  1 MiB they differ by up to 9.7%, in both directions.

## Limitations

- One node and one hardware configuration: two MI210s connected only through
  PCIe. Other MI210 nodes in the same cluster may attach their GPUs to different
  CPU sockets.
- Two ranks only. There is no scaling claim, and with two ranks busbw equals
  algbw.
- One collective, one data type, one reduction, and default RCCL settings. No
  tuning variables were set, and GPU clocks and power were not pinned.
- Three runs within one allocation, all within about 13 seconds. That says
  nothing about variation across days, nodes, or driver versions.
- Shared cluster: no other Slurm jobs were on the node, but the partition allows
  CPU oversubscription, and cluster-wide activity was not controlled.
- Times are host-observed means over 20 enqueued iterations, not kernel-only
  profiler measurements. The per-iteration distribution is not available from
  rccl-tests.
- Correctness is checked on a separate validation iteration for each size and
  placement, not on the timed iterations themselves.
- MPI was built for this record (Open MPI 5.0.11, no UCX or GPU-aware
  transport). It does not carry the AllReduce data.
- No comparison with another accelerator, another RCCL version, or a
  theoretical peak; no MI300X claim.
