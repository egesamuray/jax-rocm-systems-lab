import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import rccl_report
from benchmarks.rccl_report import (ReportError, allreduce_busbw, load_run,
                                    parse_log, summarize, validate_record)


SIZES = [8, 16, 32, 64]
HOST = "private-host-name"


def log_text(*, oob="0 OK", validation=1, ranks=2, max_bytes=64,
             tests_version="HEAD:40b1b17", rccl_version="2.27.7-HEAD:0d2c4fd"):
    devices = "".join(
        f"#  Rank {r:2d} Group  0 Pid {40000 + r:6d} on {HOST:>10s} device {r:2d} "
        f"[0000:{8 + 64 * r:02x}:00] AMD Instinct MI210\n" for r in range(ranks))
    return ("# Collective test starting: all_reduce_perf\n"
            f"# nThread 1 nGpus 1 minBytes 8 maxBytes {max_bytes} step: 2(factor) "
            f"warmup iters: 5 iters: 20 agg iters: 1 validation: {validation} graph: 0\n"
            f"#\nrccl-tests: Version {tests_version}\n# Using devices\n" + devices +
            (f"RCCL version : {rccl_version}\n" if rccl_version else "") +
            "HIP version  : 7.2.26015-fc0010cf6a\nROCm version : 7.2.0.0-43-fc0010cf6a\n"
            f"Hostname     : {HOST}.example\n"
            "#\n# Out of bounds values : " + oob + "\n# Avg bus bandwidth    : 0.001 \n#\n"
            "# Collective test concluded: all_reduce_perf\n")


def record_text(size, in_place, time_us, *, ranks=2, wrong='"0"', busbw=None, time_text=None):
    """One record exactly as rccl-tests writes it (std::to_string: 6 decimals)."""
    algbw = size / 1.0e9 / (time_us * 1e-6)
    busbw = algbw * 2 * (ranks - 1) / ranks if busbw is None else busbw
    time_text = f"{time_us:f}" if time_text is None else time_text
    return (f'{{"numCycle":0, "name":"AllReduce", "nodes":1, "ranks":{ranks}, '
            f'"ranksPerNode":{ranks}, "gpusPerRank":1, "size":{size}, "type":"float", '
            f'"redop":"sum", "inPlace":{in_place}, "time":{time_text}, '
            f'"algBw":{algbw:f}, "busBw":{busbw:f}, "wrong":{wrong}}}')


def write_run(directory, name, *, scale=1.0, records=None, log=None):
    if records is None:
        records = [record_text(size, in_place, scale * (10.0 + size / 8))
                   for size in SIZES for in_place in (0, 1)]
    path = Path(directory) / f"{name}.json"
    path.write_text("[\n" + ",\n".join(records) + "\n]\n")
    path.with_suffix(".log").write_text(log_text() if log is None else log)
    return path


def as_record(text):
    return json.loads(text)


def test_valid_runs_are_summarized_with_raw_samples(tmp_path):
    runs = [load_run(write_run(tmp_path, f"raw-0{i}", scale=s), ranks=2)
            for i, s in enumerate((1.0, 1.2, 0.9), start=1)]
    report = summarize(runs, ranks=2)
    assert report["correctness"] == {
        "validator": "rccl-tests built-in check (#wrong); FP32 sum compared exactly",
        "records_checked": 24, "wrong_total": 0, "passed": True}
    assert report["settings"]["warmup_iters"] == 5
    assert report["settings"]["iters"] == 20
    assert report["settings"]["check_iters"] == 1
    assert report["same_node"] is True
    assert report["versions"] == {"rccl_tests": "HEAD:40b1b17", "rccl": "2.27.7-HEAD:0d2c4fd",
                                  "hip": "7.2.26015-fc0010cf6a", "rocm": "7.2.0.0-43-fc0010cf6a"}
    assert [d["pci_bus"] for d in report["runs"][0]["devices"]] == ["0000:08:00", "0000:48:00"]
    row = report["results"][0]
    assert (row["size_bytes"], row["in_place"]) == (8, False)
    assert row["time_us"] == [11.0, pytest.approx(13.2), pytest.approx(9.9)]
    assert row["time_us_median"] == pytest.approx(11.0)
    assert (row["time_us_min"], row["time_us_max"]) == (pytest.approx(9.9), pytest.approx(13.2))
    assert row["time_spread_pct"] == pytest.approx(100 * 3.3 / 11.0)
    assert row["busbw_gbps"] == row["algbw_gbps"]  # two ranks: factor 1
    assert row["wrong"] == [0, 0, 0]
    assert len(report["results"]) == 8


@pytest.mark.parametrize(("ranks", "expected"), [(1, 0.0), (2, 10.0), (4, 15.0), (8, 17.5)])
def test_allreduce_busbw_formula(ranks, expected):
    assert allreduce_busbw(10.0, ranks) == pytest.approx(expected)


def test_record_cross_checks_both_bandwidths():
    row = validate_record(as_record(record_text(1 << 20, 0, 60.5)), ranks=2)
    assert row["algbw_recomputed_gbps"] == pytest.approx(row["algbw_gbps"], abs=2e-6)
    with pytest.raises(ReportError, match="busBw"):
        validate_record(as_record(record_text(1 << 20, 0, 60.5, busbw=1.0)), ranks=2)
    bad_time = as_record(record_text(1 << 20, 0, 60.5))
    bad_time["time"] = 70.0
    with pytest.raises(ReportError, match="algBw"):
        validate_record(bad_time, ranks=2)


@pytest.mark.parametrize(("wrong", "message"), [('"3"', "correctness failure: 3"),
                                                 ('"N/A"', "not checked")])
def test_failed_or_unchecked_correctness_is_rejected(tmp_path, wrong, message):
    records = [record_text(size, p, 10.0, wrong=wrong if size == 32 else '"0"')
               for size in SIZES for p in (0, 1)]
    with pytest.raises(ReportError, match=message):
        load_run(write_run(tmp_path, "raw", records=records), ranks=2)


@pytest.mark.parametrize("token", ["inf", "-nan", "NaN", "Infinity"])
def test_non_finite_values_are_rejected(tmp_path, token):
    records = [record_text(size, p, 10.0, time_text=token if size == 16 else None)
               for size in SIZES for p in (0, 1)]
    with pytest.raises(ReportError, match="malformed JSON|non-finite"):
        load_run(write_run(tmp_path, "raw", records=records), ranks=2)


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("busBw"),
    lambda r: r.update(extra=1),
    lambda r: r.update(size="8"),
    lambda r: r.update(inPlace=True),
    lambda r: r.update(ranks=4, ranksPerNode=4),
    lambda r: r.update(gpusPerRank=2),
    lambda r: r.update(name="AllGather"),
    lambda r: r.update(type="half"),
    lambda r: r.update(redop="max"),
    lambda r: r.update(size=6),
    lambda r: r.update(time=0.0),
])
def test_malformed_records_are_rejected(mutate):
    record = as_record(record_text(8, 0, 10.0))
    mutate(record)
    with pytest.raises(ReportError):
        validate_record(record, ranks=2)


@pytest.mark.parametrize("body", ["{}", "[]", "[1, 2]", "not json"])
def test_malformed_files_are_rejected(tmp_path, body):
    path = write_run(tmp_path, "raw")
    path.write_text(body)
    with pytest.raises(ReportError):
        load_run(path, ranks=2)


def test_missing_duplicate_or_extra_sweep_records_are_rejected(tmp_path):
    full = [record_text(size, p, 10.0) for size in SIZES for p in (0, 1)]
    for records in (full[:-1], full + full[-1:], full + [record_text(128, 0, 10.0)]):
        with pytest.raises(ReportError):
            load_run(write_run(tmp_path, "raw", records=records), ranks=2)


@pytest.mark.parametrize("log", [log_text(oob="2 FAILED"), log_text(validation=0),
                                 log_text(ranks=1), log_text().replace("device  1", "device  0"),
                                 log_text() + log_text(), "no header\n",
                                 log_text(tests_version="HEAD:40b1b17+"),
                                 log_text(rccl_version=None)])
def test_log_must_show_two_distinct_gpus_and_a_passing_validator(log):
    with pytest.raises(ReportError):
        parse_log(log, ranks=2)


def test_runs_must_share_settings_and_sizes(tmp_path):
    first = load_run(write_run(tmp_path, "a"), ranks=2)
    other = write_run(tmp_path, "b", records=[record_text(size, p, 10.0)
                                              for size in SIZES[:-1] for p in (0, 1)],
                      log=log_text(max_bytes=32))
    with pytest.raises(ReportError, match="different"):
        summarize([first, load_run(other, ranks=2)], ranks=2)
    newer = write_run(tmp_path, "c", log=log_text(rccl_version="2.28.3-HEAD:abc1234"))
    with pytest.raises(ReportError, match="versions"):
        summarize([first, load_run(newer, ranks=2)], ranks=2)


def test_sanitized_report_schema_via_cli(tmp_path):
    paths = [write_run(tmp_path, f"raw-0{i}", scale=1 + i / 10) for i in (1, 2, 3)]
    output = tmp_path / "out" / "summary.json"
    subprocess.run([sys.executable, "-m", "benchmarks.rccl_report", *map(str, paths),
                    "--ranks", "2", "--output", str(output)],
                   check=True, cwd=Path(rccl_report.__file__).resolve().parents[1])
    text = output.read_text()
    report = json.loads(text)
    assert set(report) == {"schema_version", "purpose", "collective", "dtype", "redop",
                           "ranks", "gpus_per_rank", "same_node", "settings", "versions",
                           "definitions",
                           "correctness", "formula_check", "runs", "results"}
    assert report["schema_version"] == 1
    assert [report[k] for k in ("collective", "dtype", "redop")] == ["AllReduce", "float32", "sum"]
    assert [run["file"] for run in report["runs"]] == ["raw-01.json", "raw-02.json", "raw-03.json"]
    assert all(set(run) == {"file", "sha256", "records", "devices"} for run in report["runs"])
    assert all(set(d) == {"rank", "device", "pci_bus", "name"}
               for run in report["runs"] for d in run["devices"])
    assert HOST not in text and str(tmp_path) not in text and "Pid" not in text
    numbers = [v for row in report["results"] for k, v in row.items()
               if isinstance(v, float)]
    assert numbers and all(math.isfinite(v) for v in numbers)
    rerun = tmp_path / "again.json"
    report_again = summarize([load_run(p, ranks=2) for p in paths], ranks=2)
    rerun.write_text(json.dumps(report_again, indent=2, allow_nan=False) + "\n")
    assert rerun.read_text() == text  # stable output for the same inputs
