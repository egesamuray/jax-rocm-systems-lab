import argparse
import json
import math
from types import SimpleNamespace
import warnings

import jax
import pytest

from benchmarks import matmul
from benchmarks.matmul import accelerator_plugin_versions, positive_int, run


@pytest.mark.parametrize("size", [7, 32])
def test_cpu_correctness_and_report(size):
    report = run(size=size, warmup=1, iterations=2)
    assert report["correctness"]["passed"]
    assert report["device"]["requested_backend"] == "cpu"
    assert report["device"]["platform"] == "cpu"
    assert report["workload"]["shape"] == [size, size]
    assert report["schema_version"] == 2
    assert report["runtime"]["jax_platforms"] == jax.config.jax_platforms
    assert report["runtime"]["platform_version"] == "cpu"
    assert isinstance(report["runtime"]["accelerator_plugins"], dict)
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


def test_accelerator_plugin_detection(monkeypatch):
    installed = {"jax": "0.11.2", "jaxlib": "0.11.2", "jax_rocm7_plugin": "0.11.2",
                 "jax-rocm7-pjrt": "0.11.2", "jax-cuda12-plugin": "0.11.2",
                 "jax-rocm-tools": "1.0", "numpy": "2.5.3"}
    fakes = [SimpleNamespace(metadata={"Name": name, "Version": version})
             for name, version in installed.items()]
    monkeypatch.setattr(matmul.importlib.metadata, "distributions", lambda: fakes)
    assert accelerator_plugin_versions() == {
        "jax-cuda12-plugin": "0.11.2", "jax-rocm7-pjrt": "0.11.2", "jax-rocm7-plugin": "0.11.2"}


def test_plugin_detection_tolerates_incomplete_metadata(monkeypatch, tmp_path):
    dist_infos = {"broken-1.0": None,  # no METADATA, as left by an interrupted install
                  "jax_rocm7_plugin-0.11.2": "Name: jax_rocm7_plugin\nVersion: 0.11.2",
                  "jax_cuda12_pjrt-0.11.2": "Name: jax_cuda12_pjrt"}
    for stem, fields in dist_infos.items():
        path = tmp_path / f"{stem}.dist-info"
        path.mkdir()
        if fields:
            (path / "METADATA").write_text(f"Metadata-Version: 2.1\n{fields}\n")
    real = matmul.importlib.metadata.distributions
    monkeypatch.setattr(matmul.importlib.metadata, "distributions",
                        lambda: real(path=[str(tmp_path)]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert accelerator_plugin_versions() == {"jax-cuda12-pjrt": None,
                                                 "jax-rocm7-plugin": "0.11.2"}
