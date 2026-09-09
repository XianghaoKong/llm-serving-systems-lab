#!/usr/bin/env python3
"""Combine S8 client measurements with extracted Nsight GPU intervals."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from src.s8_analyze_interference import summarize_trial
except ModuleNotFoundError:
    from s8_analyze_interference import summarize_trial


def finite(values: Iterable[Optional[float]]) -> List[float]:
    return [float(value) for value in values if value is not None]


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    rows = finite(values)
    return statistics.median(rows) if rows else None


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    rows = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]
    position = (len(rows) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return rows[lower]
    return rows[lower] * (upper - position) + rows[upper] * (position - lower)


def bootstrap_median_ci(
    values: Sequence[float],
    seed: int,
    samples: int = 5000,
) -> tuple[Optional[float], Optional[float]]:
    rows = finite(values)
    if not rows:
        return None, None
    if len(rows) == 1:
        return rows[0], rows[0]
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(rows, k=len(rows)))
        for _ in range(samples)
    ]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def read_busy_intervals(path: Path) -> List[Dict[str, float]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [
            {
                "relative_start_ms": float(row["relative_start_ms"]),
                "relative_end_ms": float(row["relative_end_ms"]),
                "duration_ms": float(row["duration_ms"]),
            }
            for row in csv.DictReader(handle)
        ]


def maximum_overlapping_duration(
    intervals: Sequence[Mapping[str, float]],
    start_ms: float,
    end_ms: float,
) -> Optional[float]:
    overlapping = [
        float(row["duration_ms"])
        for row in intervals
        if float(row["relative_end_ms"]) >= start_ms
        and float(row["relative_start_ms"]) <= end_ms
    ]
    return max(overlapping, default=None)


def summarize_profile_trial(trial_json: Path) -> Dict[str, Any]:
    trial_dir = trial_json.parent
    case_dir = trial_dir.parent
    metadata = json.loads(trial_json.read_text(encoding="utf-8"))
    if not metadata.get("profile_enabled"):
        raise RuntimeError(f"profile was not enabled for {trial_json}")
    profile_start = metadata.get("profile_start") or {}
    reference_t_s = profile_start.get("return_t_s")
    if reference_t_s is None:
        raise RuntimeError(f"missing profile start alignment in {trial_json}")
    nsys_dir = case_dir / "nsys"
    nsys_summary = json.loads(
        (nsys_dir / "nsys_summary.json").read_text(encoding="utf-8")
    )
    intervals = read_busy_intervals(nsys_dir / "busy_intervals.csv.gz")
    injection_offset_ms = (
        float(metadata["injection_t_s"]) - float(reference_t_s)
    ) * 1000.0
    impact_end_offset_ms = (
        float(metadata["impact_end_t_s"]) - float(reference_t_s)
    ) * 1000.0
    causal = summarize_trial(trial_json, baseline_seconds=4.0)
    return {
        "case": case_dir.name,
        "replicate": int(metadata["replicate"]),
        "trial_kind": metadata["trial_kind"],
        "config_label": metadata["config_label"],
        "interferer_input_tokens": int(metadata["interferer_input_tokens"]),
        "profile_start_latency_ms": profile_start.get("latency_ms"),
        "profile_stop_latency_ms": (metadata.get("profile_stop") or {}).get(
            "latency_ms"
        ),
        "injection_offset_ms": injection_offset_ms,
        "impact_end_offset_ms": impact_end_offset_ms,
        "kernel_count": nsys_summary["kernel_count"],
        "gpu_span_ms": nsys_summary["gpu_span_ms"],
        "gpu_active_fraction": nsys_summary["gpu_active_fraction"],
        "max_kernel_duration_ms": nsys_summary["max_kernel_duration_ms"],
        "max_busy_interval_ms": nsys_summary["max_busy_interval_ms"],
        "impact_max_busy_interval_ms": maximum_overlapping_duration(
            intervals,
            max(0.0, injection_offset_ms),
            max(0.0, impact_end_offset_ms),
        ),
        "background_p99_stall_ratio": causal["p99_stall_ratio"],
        "background_impact_max_gap_ms": causal["impact_gap_max_ms"],
        "long_request_ttft_p50_ms": causal["long_ttft_p50_ms"],
        "impact_max_waiting": causal["impact_max_waiting"],
        "impact_max_kv_usage": causal["impact_max_kv_usage"],
        "preemptions_delta": causal["preemptions_delta"],
    }


def aggregate(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case"])].append(row)
    metrics = [
        "impact_max_busy_interval_ms",
        "max_kernel_duration_ms",
        "gpu_active_fraction",
        "background_p99_stall_ratio",
        "background_impact_max_gap_ms",
        "long_request_ttft_p50_ms",
        "impact_max_waiting",
        "impact_max_kv_usage",
        "preemptions_delta",
    ]
    output: List[Dict[str, Any]] = []
    for case_index, (case, case_rows) in enumerate(sorted(grouped.items())):
        row: Dict[str, Any] = {
            "case": case,
            "n_runs": len(case_rows),
            "trial_kind": case_rows[0]["trial_kind"],
            "config_label": case_rows[0]["config_label"],
        }
        for metric_index, metric in enumerate(metrics):
            values = finite(item.get(metric) for item in case_rows)
            row[f"{metric}_median"] = median(values)
            low, high = bootstrap_median_ci(
                values,
                seed=seed + case_index * 100 + metric_index,
            )
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        output.append(row)
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    order = [
        "control_off_32768",
        "inject_off_32768",
        "inject_on_4096",
        "inject_on_1024",
    ]
    by_case = {str(row["case"]): row for row in rows}
    available = [case for case in order if case in by_case]
    if not available:
        return
    labels = [case.replace("control_", "control\n").replace("inject_", "inject\n")
              for case in available]
    bursts = [
        by_case[case]["impact_max_busy_interval_ms_median"] or 0.0
        for case in available
    ]
    gaps = [
        by_case[case]["background_impact_max_gap_ms_median"] or 0.0
        for case in available
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, bursts, color="#4C78A8")
    axes[0].set_ylabel("Median maximum GPU busy interval (ms)")
    axes[0].set_title("GPU execution interval")
    axes[1].bar(labels, gaps, color="#E45756")
    axes[1].set_ylabel("Median maximum background SSE gap (ms)")
    axes[1].set_title("Decode stream stall")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("S8-B: 24K prefill execution interval and decode stall")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    trial_files = sorted(args.input_root.glob("block_*/*/trial/trial.json"))
    if not trial_files:
        raise RuntimeError(f"no profile trials found below {args.input_root}")
    rows = [summarize_profile_trial(path) for path in trial_files]
    aggregates = aggregate(rows, seed=args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "profile_trial_summary.csv", rows)
    write_csv(args.output_dir / "profile_aggregate_summary.csv", aggregates)
    make_figure(aggregates, args.output_dir / "gpu_interval_vs_sse_gap.png")
    audit = {
        "schema_version": 1,
        "trial_count": len(rows),
        "expected_trial_count": 20,
        "complete": len(rows) == 20 and all(row["n_runs"] == 5 for row in aggregates),
        "run_is_statistical_unit": True,
        "bootstrap_samples": 5000,
        "cases": [row["case"] for row in aggregates],
        "alignment_note": (
            "GPU intervals use the first captured GPU event as zero; injection "
            "offset uses the client-clock return from /start_profile and is "
            "therefore approximate at sub-request latency scale."
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
