"""Validate and summarize rccl-tests AllReduce output; standard library only.

Input is one or more independent runs of an MPI build of `all_reduce_perf`, each
saved as the JSON file written by `-Z json -x <file>` plus the matching stdout
log (same path, `.log` suffix). The report rejects correctness failures and
malformed or non-finite values, recomputes both bandwidths, keeps every raw
sample, and writes only the fields below (no hostnames, paths, or PIDs).
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


SCHEMA_VERSION = 1
FIELDS = {"numCycle": int, "name": str, "nodes": int, "ranks": int,
          "ranksPerNode": int, "gpusPerRank": int, "size": int, "type": str,
          "redop": str, "inPlace": int, "time": float, "algBw": float,
          "busBw": float, "wrong": str}
# JSON numbers are printed with 6 decimals: allow rounding in both operands.
BW_ABS_TOL = 2e-6
BW_REL_TOL = 1e-6
SETTINGS = re.compile(
    r"^# nThread (?P<threads>\d+) nGpus (?P<gpus>\d+) minBytes (?P<min_bytes>\d+) "
    r"maxBytes (?P<max_bytes>\d+) step: (?P<step>\d+)\((?P<step_kind>factor|bytes)\) "
    r"warmup iters: (?P<warmup_iters>\d+) iters: (?P<iters>\d+) "
    r"agg iters: (?P<agg_iters>\d+) validation: (?P<check_iters>\d+) "
    r"graph: (?P<graph_launches>\d+)\s*$", re.M)
DEVICE = re.compile(
    r"^#\s+Rank\s+(?P<rank>\d+) Group\s+\d+ Pid\s+\d+ on\s+(?P<host>\S+) "
    r"device\s+(?P<device>\d+) \[(?P<pci>[0-9a-fA-F:.]+)\] (?P<name>.+?)\s*$", re.M)
OUT_OF_BOUNDS = re.compile(r"^# Out of bounds values : (\d+) (OK|FAILED)\s*$", re.M)
# Banner lines: rccl-tests always prints its git revision; RCCL prints the rest
# when NCCL_DEBUG=VERSION is set (HIP/ROCm lines are optional).
VERSIONS = {"rccl_tests": re.compile(r"^rccl-tests: Version (\S+)\s*$", re.M),
            "rccl": re.compile(r"^RCCL version\s*: (\S+)\s*$", re.M),
            "hip": re.compile(r"^HIP version\s*: (\S+)\s*$", re.M),
            "rocm": re.compile(r"^ROCm version\s*: (\S+)\s*$", re.M)}


class ReportError(ValueError):
    """Raised when rccl-tests output cannot support a published result."""


def _reject_constant(token):
    raise ReportError(f"non-finite JSON constant {token}")


def _number(record, key, kind):
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{key} must be numeric, got {value!r}")
    if kind is int and not isinstance(value, int):
        raise ReportError(f"{key} must be an integer, got {value!r}")
    if not math.isfinite(value):
        raise ReportError(f"{key} is not finite")
    return kind(value)


def allreduce_busbw(algbw, ranks):
    """rccl-tests AllReduce bus bandwidth: algbw * 2 * (n - 1) / n."""
    if ranks < 1:
        raise ReportError("ranks must be positive")
    return algbw * 2 * (ranks - 1) / ranks


def validate_record(record, *, ranks, gpus_per_rank=1):
    """Check one rccl-tests JSON record and return its normalized form."""
    if not isinstance(record, dict) or set(record) != set(FIELDS):
        raise ReportError(f"malformed record; expected keys {sorted(FIELDS)}")
    for key, kind in FIELDS.items():
        if kind is str and not isinstance(record[key], str):
            raise ReportError(f"{key} must be a string, got {record[key]!r}")
    if (record["name"], record["type"], record["redop"]) != ("AllReduce", "float", "sum"):
        raise ReportError("expected an AllReduce float sum record")
    values = {key: _number(record, key, kind) for key, kind in FIELDS.items()
              if kind is not str}
    if values["ranks"] != ranks or values["ranksPerNode"] != ranks:
        raise ReportError(f"expected {ranks} ranks in one communicator")
    if values["gpusPerRank"] != gpus_per_rank:
        raise ReportError(f"expected {gpus_per_rank} GPU per rank")
    if values["inPlace"] not in (0, 1):
        raise ReportError("inPlace must be 0 or 1")
    size, time_us = values["size"], values["time"]
    if size <= 0 or size % 4 or time_us <= 0:
        raise ReportError("size must be a positive multiple of 4 and time positive")
    if values["algBw"] < 0 or values["busBw"] < 0:
        raise ReportError("bandwidth must be non-negative")
    if not record["wrong"].isdigit():
        raise ReportError(f"correctness was not checked (#wrong={record['wrong']!r})")
    wrong = int(record["wrong"])
    if wrong:
        raise ReportError(f"correctness failure: {wrong} wrong elements at {size} B")
    n = ranks * gpus_per_rank
    algbw = size / (time_us * 1e3)  # bytes / microseconds / 1e3 = 1e9 bytes/s
    for name, expected, reported in (("algBw", algbw, values["algBw"]),
                                     ("busBw", allreduce_busbw(values["algBw"], n),
                                      values["busBw"])):
        if abs(expected - reported) > BW_ABS_TOL + BW_REL_TOL * abs(expected):
            raise ReportError(f"{name} {reported} disagrees with formula ({expected})")
    return {"size_bytes": size, "in_place": bool(values["inPlace"]),
            "time_us": time_us, "algbw_gbps": values["algBw"],
            "busbw_gbps": values["busBw"], "wrong": wrong,
            "algbw_recomputed_gbps": algbw}


def parse_log(text, *, ranks):
    """Extract settings, per-rank devices, and the final validator line."""
    matches = list(SETTINGS.finditer(text))
    if len(matches) != 1:
        raise ReportError("log must contain exactly one settings line")
    parsed = {key: (value if key == "step_kind" else int(value))
              for key, value in matches[0].groupdict().items()}
    if parsed["check_iters"] < 1:
        raise ReportError("rccl-tests validation was disabled")
    devices = [m.groupdict() for m in DEVICE.finditer(text)]
    if sorted(int(d["rank"]) for d in devices) != list(range(ranks)):
        raise ReportError(f"log must list devices for ranks 0..{ranks - 1}")
    if len({(d["host"], d["device"]) for d in devices}) != ranks:
        raise ReportError("ranks must use distinct GPUs")
    oob = OUT_OF_BOUNDS.findall(text)
    if len(oob) != 1 or oob[0] != ("0", "OK"):
        raise ReportError("log must end with 'Out of bounds values : 0 OK'")
    versions = {}
    for key, pattern in VERSIONS.items():
        found = set(pattern.findall(text))
        if len(found) > 1 or (not found and key in ("rccl_tests", "rccl")):
            raise ReportError(f"log must report exactly one {key} version "
                              "(run with NCCL_DEBUG=VERSION)")
        versions[key] = found.pop() if found else None
    if versions["rccl_tests"].endswith("+"):
        raise ReportError("rccl-tests was built from a modified source tree")
    hosts = sorted({d["host"] for d in devices})
    return {"settings": parsed, "versions": versions,
            "same_node": len(hosts) == 1,
            "devices": [{"rank": int(d["rank"]), "device": int(d["device"]),
                         "pci_bus": d["pci"], "name": d["name"]}
                        for d in sorted(devices, key=lambda d: int(d["rank"]))]}


def sweep_sizes(settings):
    """Message sizes visited by rccl-tests for the logged min/max/step."""
    factor = settings["step_kind"] == "factor"
    step, size, sizes = settings["step"], settings["min_bytes"], []
    if step < (2 if factor else 1) or size < 1:
        raise ReportError("invalid sweep settings")
    while size <= settings["max_bytes"]:
        sizes.append(size)
        size = size * step if factor else size + step
    return sizes


def load_run(json_path, *, ranks, gpus_per_rank=1):
    """Load one run (JSON plus sibling .log) and validate every record."""
    json_path = Path(json_path)
    raw = json_path.read_bytes()
    try:
        records = json.loads(raw, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ReportError(f"malformed JSON in {json_path.name}: {error}") from None
    if not isinstance(records, list) or not records:
        raise ReportError(f"{json_path.name} must be a non-empty JSON list")
    rows = {}
    for record in records:
        row = validate_record(record, ranks=ranks, gpus_per_rank=gpus_per_rank)
        key = (row["size_bytes"], row["in_place"])
        if key in rows:
            raise ReportError(f"duplicate record for {key}")
        rows[key] = row
    log = parse_log(json_path.with_suffix(".log").read_text(), ranks=ranks)
    if set(rows) != {(size, in_place) for size in sweep_sizes(log["settings"])
                     for in_place in (False, True)}:
        raise ReportError("JSON records do not match the logged sweep "
                          "(every size, out-of-place and in-place)")
    return {"file": json_path.name, "sha256": hashlib.sha256(raw).hexdigest(),
            "records": len(rows), **log, "rows": rows}


def summarize(runs, *, ranks, gpus_per_rank=1):
    """Combine independent runs; every run must cover the same records."""
    if not runs:
        raise ReportError("at least one run is required")
    keys = sorted(runs[0]["rows"])
    for run in runs[1:]:
        if sorted(run["rows"]) != keys:
            raise ReportError(f"{run['file']} covers different sizes than {runs[0]['file']}")
        if (run["settings"], run["versions"]) != (runs[0]["settings"], runs[0]["versions"]):
            raise ReportError(f"{run['file']} used different rccl-tests settings or versions")
    results = []
    worst = 0.0
    for key in keys:
        rows = [run["rows"][key] for run in runs]
        times = [row["time_us"] for row in rows]
        median_time = statistics.median(times)
        worst = max(worst, max(abs(row["algbw_recomputed_gbps"] - row["algbw_gbps"])
                               for row in rows))
        results.append({
            "size_bytes": key[0], "in_place": key[1],
            "time_us": times, "time_us_median": median_time,
            "time_us_min": min(times), "time_us_max": max(times),
            "time_spread_pct": 100 * (max(times) - min(times)) / median_time,
            "algbw_gbps": [row["algbw_gbps"] for row in rows],
            "algbw_gbps_median": statistics.median(row["algbw_gbps"] for row in rows),
            "busbw_gbps": [row["busbw_gbps"] for row in rows],
            "busbw_gbps_median": statistics.median(row["busbw_gbps"] for row in rows),
            "wrong": [row["wrong"] for row in rows]})
    n = ranks * gpus_per_rank
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "rccl_allreduce_baseline",
        "collective": "AllReduce", "dtype": "float32", "redop": "sum",
        "ranks": n, "gpus_per_rank": gpus_per_rank,
        "same_node": all(run["same_node"] for run in runs),
        "settings": runs[0]["settings"],
        "versions": runs[0]["versions"],
        "definitions": {
            "time_us": "mean microseconds per collective over the timed iterations, "
                       "averaged across ranks (rccl-tests default -a 1)",
            "algbw_gbps": "size_bytes / time, in 1e9 bytes per second",
            "busbw_gbps": f"algbw_gbps * 2 * (ranks - 1) / ranks = algbw_gbps * "
                          f"{allreduce_busbw(1.0, n):g}"},
        "correctness": {
            "validator": "rccl-tests built-in check (#wrong); FP32 sum compared exactly",
            "records_checked": sum(run["records"] for run in runs),
            "wrong_total": 0, "passed": True},
        "formula_check": {"max_abs_algbw_diff_gbps": worst,
                          "tolerance_gbps": f"{BW_ABS_TOL:g} + {BW_REL_TOL:g} * value"},
        "runs": [{key: run[key] for key in ("file", "sha256", "records", "devices")}
                 for run in runs],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path,
                        help="rccl-tests JSON files; each needs a sibling .log")
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [load_run(path, ranks=args.ranks) for path in args.runs]
    report = summarize(runs, ranks=args.ranks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['correctness']['records_checked']} records from {len(runs)} runs "
          f"passed validation; summary saved to {args.output}")


if __name__ == "__main__":
    main()
