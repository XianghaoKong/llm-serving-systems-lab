#!/usr/bin/env python3
"""Select the smallest near-saturated, queue-safe S8-D concurrency."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

try:
    from src.s8_analyze_interference import summarize_trial
except ModuleNotFoundError:
    from s8_analyze_interference import summarize_trial


def read_gpu_utilization(path: Path) -> List[float]:
    values: List[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                try:
                    values.append(float(row[1].strip()))
                except ValueError:
                    continue
    return values


def choose(rows: List[Dict[str, Any]], utilization_target: float) -> Dict[str, Any]:
    queue_safe = [row for row in rows if float(row["impact_max_waiting"] or 0) <= 1]
    saturated = [
        row for row in queue_safe
        if float(row["gpu_active_utilization_median_pct"]) >= utilization_target
    ]
    if saturated:
        return min(saturated, key=lambda row: int(row["background_concurrency"]))
    if queue_safe:
        return max(queue_safe, key=lambda row: int(row["background_concurrency"]))
    return min(
        rows,
        key=lambda row: (
            float(row["impact_max_waiting"] or 0),
            -float(row["gpu_active_utilization_median_pct"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--utilization-target", type=float, default=90.0)
    args = parser.parse_args()
    rows: List[Dict[str, Any]] = []
    for trial_path in sorted(args.input_root.glob("c*/trial/trial.json")):
        summary = summarize_trial(trial_path, baseline_seconds=4.0)
        samples = read_gpu_utilization(trial_path.parent.with_suffix(".gpu.csv"))
        if not samples:
            raise RuntimeError(f"no GPU telemetry for {trial_path}")
        active_samples = [value for value in samples if value > 0]
        if not active_samples:
            raise RuntimeError(f"no active GPU telemetry for {trial_path}")
        rows.append(
            {
                "background_concurrency": summary["background_concurrency"],
                "gpu_samples": len(samples),
                "gpu_active_samples": len(active_samples),
                "gpu_busy_sample_fraction": len(active_samples) / len(samples),
                "gpu_active_utilization_median_pct": statistics.median(active_samples),
                "gpu_utilization_min_pct": min(samples),
                "impact_max_waiting": summary["impact_max_waiting"],
                "baseline_gap_p99_ms": summary["baseline_gap_p99_ms"],
                "impact_gap_p99_ms": summary["impact_gap_p99_ms"],
                "preemptions_delta": summary["preemptions_delta"],
            }
        )
    if not rows:
        raise RuntimeError("no S8-D calibration trials found")
    selected = choose(rows, args.utilization_target)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "calibration_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_version": 1,
        "utilization_target_pct": args.utilization_target,
        "selection_rule": (
            "smallest concurrency with median utilization across nonzero GPU "
            "samples at or above the target and scheduler waiting <= 1; "
            "otherwise highest queue-safe candidate"
        ),
        "selected_background_concurrency": selected["background_concurrency"],
        "candidates": rows,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
