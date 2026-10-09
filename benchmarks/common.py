"""Shared benchmark plumbing: environment capture, percentiles, result files."""

from __future__ import annotations

import csv
import json
import os
import platform
import sqlite3
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psutil

RESULTS = Path(__file__).parent / "results"


def environment() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "src")),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "disk_note": os.environ.get("BENCH_DISK_NOTE", ""),
    }


def pct(values: Sequence[float], q: float) -> float:
    """Percentile (q in 0..100) with linear interpolation between closest ranks."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(values: Sequence[float], scale: float = 1.0) -> dict[str, float]:
    v = [x * scale for x in values]
    return {
        "n": len(v),
        "mean": round(statistics.fmean(v), 3) if v else float("nan"),
        "p50": round(pct(v, 50), 3),
        "p95": round(pct(v, 95), 3),
        "p99": round(pct(v, 99), 3),
        "max": round(max(v), 3) if v else float("nan"),
    }


def write_results(name: str, payload: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    """``results/<name>/summary.json`` (environment + config + summary) and ``rows.csv``."""
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps({"environment": environment(), **payload}, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        with (out / "rows.csv").open("w", newline="", encoding="utf-8") as f:
            fields = list(dict.fromkeys(k for row in rows for k in row))  # union, in order
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return out
