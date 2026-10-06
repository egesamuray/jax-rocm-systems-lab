"""Single-device FP32 matmul smoke baseline; not a hardware comparison."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np


ACCELERATOR_PLUGIN = re.compile(r"jax-(cuda|rocm)[0-9]*-(plugin|pjrt)")


def accelerator_plugin_versions():
    """Versions of installed JAX CUDA/ROCm plugin wheels; empty on a CPU-only stack."""
    versions = {}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if ACCELERATOR_PLUGIN.fullmatch(name):
            versions[name] = dist.version
    return dict(sorted(versions.items()))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def run(*, backend="cpu", size=256, warmup=3, iterations=10):
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be cpu or gpu")
    if min(size, warmup, iterations) < 1:
        raise ValueError("size, warmup, and iterations must be positive")
    jax.config.update("jax_enable_compilation_cache", False)
    device = jax.devices(backend)[0]  # Explicit request: unavailable backend raises.
    rng = np.random.default_rng(0)
    a_host = rng.standard_normal((size, size)).astype(np.float32)
    b_host = rng.standard_normal((size, size)).astype(np.float32)
    a, b = (jax.device_put(x, device) for x in (a_host, b_host))
    a.block_until_ready()
    b.block_until_ready()
    operation = jax.jit(lambda x, y: jnp.matmul(x, y, precision="highest"))
    start = time.perf_counter_ns()
    compiled = operation.lower(a, b).compile()
    compile_ms = (time.perf_counter_ns() - start) / 1e6
    actual = np.asarray(compiled(a, b).block_until_ready())
    reference = a_host @ b_host
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
    if not np.isfinite(actual).all():
        raise ValueError("non-finite output")
    for _ in range(warmup):
        compiled(a, b).block_until_ready()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        compiled(a, b).block_until_ready()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "schema_version": 2,
        "purpose": "harness_smoke_only",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision,
        "working_tree_dirty": dirty,
        "stack": {name: importlib.metadata.version(name) for name in
                  ("jax", "jaxlib", "numpy", "scipy", "ml_dtypes", "opt_einsum")},
        "runtime": {"jax_platforms": jax.config.jax_platforms,
                    "platform_version": getattr(device.client, "platform_version", None),
                    "accelerator_plugins": accelerator_plugin_versions()},
        "host": {"os": platform.system(), "os_release": platform.release(),
                 "architecture": platform.machine(), "python": platform.python_version(),
                 "logical_cpu_count": os.cpu_count()},
        "device": {"requested_backend": backend, "platform": device.platform,
                   "kind": device.device_kind, "id": device.id,
                   "visible_device_count": len(jax.devices(backend))},
        "workload": {"operation": "matmul", "shape": [size, size],
                     "dtype": "float32", "precision": "highest", "seed": 0,
                     "warmup": warmup, "iterations": iterations},
        "correctness": {"passed": True, "oracle": "numpy.matmul",
                        "rtol": 1e-4, "atol": 1e-4,
                        "max_abs_error": float(np.max(np.abs(actual - reference)))},
        "timing": {"clock": "perf_counter_ns", "synchronization": "block_until_ready",
                   "persistent_compilation_cache": False,
                   "lower_and_compile_ms": compile_ms, "samples_ms": samples,
                   "median_ms": statistics.median(samples),
                   "min_ms": min(samples), "max_ms": max(samples)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--size", type=positive_int, default=256)
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--iterations", type=positive_int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(backend=args.backend, size=args.size,
                 warmup=args.warmup, iterations=args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Correctness passed; {args.iterations} samples saved to {args.output}")


if __name__ == "__main__":
    main()
