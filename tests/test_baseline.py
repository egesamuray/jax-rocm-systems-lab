import argparse
import json
import math

import pytest

from benchmarks.matmul import positive_int, run


@pytest.mark.parametrize("size", [7, 32])
def test_cpu_correctness_and_report(size):
    report = run(size=size, warmup=1, iterations=2)
    assert report["correctness"]["passed"]
    assert report["device"]["requested_backend"] == "cpu"
    assert report["device"]["platform"] == "cpu"
    assert report["workload"]["shape"] == [size, size]
    samples = report["timing"]["samples_ms"]
    assert len(samples) == 2
    assert all(math.isfinite(sample) and sample > 0 for sample in samples)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_cli_rejects_nonpositive_counts(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


def test_invalid_backend_fails():
    with pytest.raises(ValueError, match="backend"):
        run(backend="automatic")
