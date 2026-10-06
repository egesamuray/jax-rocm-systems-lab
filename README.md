# jax-rocm-systems-lab

A small systems lab for reproducible JAX experiments, with a future focus on
AMD ROCm. Start with a trustworthy measurement harness, then add one experiment
at a time.

**Status: early.** The repository contains a CPU matmul smoke baseline, a
benchmark methodology scaffold, CPU-only CI, and a first accelerator record: a
correctness-checked, reproducible RCCL AllReduce baseline on 2x AMD Instinct
MI210 ([docs/rccl-mi210-baseline.md](docs/rccl-mi210-baseline.md)). HIP/XLA FFI
is the next roadmap stage. FFI, MaxText, and GPU runs of the JAX harness are
pending. There are no speedup, scaling, or MI300X optimization claims.

## Goals

- Establish correctness checks and reproducible, synchronized timings.
- Explore collectives through RCCL, then one HIP kernel through XLA FFI.
- Evaluate a small MaxText workload only after the lower-level checks pass.

This setup is intentionally small. The roadmap is an ordered backlog, with no
six-week completion commitment, hardware provisioning, or training campaign.

## Quick start (CPU)

Use Python 3.14 in a fresh virtual environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python -m benchmarks.matmul --backend cpu \
  --size 256 --warmup 3 --iterations 10 --output results/cpu-smoke.json
```

The harness checks FP32 results against NumPy, separates lowering/compilation
from execution, and saves every synchronized timing sample with stack and device
metadata. This tiny single-process run validates the harness; it is insufficient
for hardware comparisons or optimization claims. CI tests correctness and report
generation, with no timing thresholds.

For a future GPU baseline, prepare a separate environment using the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
and run the same command with `--backend gpu` and `JAX_PLATFORMS=cuda` or
`JAX_PLATFORMS=rocm` as appropriate. Record the accelerator stack separately;
the CPU requirements file is not an accelerator installation recipe. An
unavailable requested backend fails instead of falling back to CPU.

## Layout

- `benchmarks/matmul.py`: one baseline and a JSON report.
- `benchmarks/rccl_report.py`: validates and summarizes rccl-tests AllReduce
  output; `benchmarks/rccl_known_value.cpp`: known-value RCCL check.
- `docs/rccl-mi210-baseline.md` and `results/rccl/`: the first RCCL record and
  its sanitized raw evidence.
- `docs/benchmark-methodology.md`: measurement contract and reporting checklist.
- `docs/roadmap.md`: RCCL → HIP/XLA FFI → MaxText gates.
- `tests/` and `.github/workflows/ci.yml`: CPU correctness and smoke checks.

See the [issue roadmap](https://github.com/egesamuray/jax-rocm-systems-lab/issues)
for future work. Publish results only with commands, raw samples, correctness
evidence, and sufficient hardware/software context.
