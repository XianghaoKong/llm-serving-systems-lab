#!/usr/bin/env python3
"""Build the S8-C TTFT/decode-gap Pareto and dual-SLO analysis."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    from src.s8_analyze_interference import bootstrap_median_ci, finite
except ModuleNotFoundError:
    from s8_analyze_interference import bootstrap_median_ci, finite


METRICS = (
    "long_ttft_p50_ms",
    "impact_gap_p99_ms",
    "p99_stall_ratio",
)


def pareto_flags(rows: Sequence[Mapping[str, Any]]) -> List[bool]:
    """Mark points that are not dominated while minimizing TTFT and gap."""
    output: List[bool] = []
    for row in rows:
        x = float(row["long_ttft_p50_ms_median"])
        y = float(row["impact_gap_p99_ms_median"])
        dominated = any(
            float(other["long_ttft_p50_ms_median"]) <= x
            and float(other["impact_gap_p99_ms_median"]) <= y
            and (
                float(other["long_ttft_p50_ms_median"]) < x
                or float(other["impact_gap_p99_ms_median"]) < y
            )
            for other in rows
            if other is not row
        )
        output.append(not dominated)
    return output


def aggregate(
    trials: Iterable[Mapping[str, str]],
    seed: int,
    ttft_slo_ms: float,
    gap_slo_ms: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[int, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in trials:
        if row["trial_kind"] == "inject":
            grouped[(int(row["interferer_input_tokens"]), row["config_label"])].append(row)
    output: List[Dict[str, Any]] = []
    for group_index, ((input_tokens, label), rows) in enumerate(sorted(grouped.items())):
        item: Dict[str, Any] = {
            "interferer_input_tokens": input_tokens,
            "config_label": label,
            "max_num_batched_tokens": int(rows[0]["max_num_batched_tokens"]),
            "n_runs": len(rows),
        }
        for metric_index, metric in enumerate(METRICS):
            values = finite(row.get(metric) for row in rows)
            item[f"{metric}_median"] = statistics.median(values)
            low, high = bootstrap_median_ci(
                values,
                seed=seed + group_index * 100 + metric_index,
            )
            item[f"{metric}_ci_low"] = low
            item[f"{metric}_ci_high"] = high
        item["meets_ttft_slo"] = item["long_ttft_p50_ms_median"] <= ttft_slo_ms
        item["meets_gap_slo"] = item["impact_gap_p99_ms_median"] <= gap_slo_ms
        item["meets_both_slos"] = item["meets_ttft_slo"] and item["meets_gap_slo"]
        output.append(item)

    by_length: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_length[int(row["interferer_input_tokens"])].append(row)
    for rows in by_length.values():
        for row, is_pareto in zip(rows, pareto_flags(rows), strict=True):
            row["pareto_optimal"] = is_pareto
    return output


def policy_map(
    rows: Sequence[Mapping[str, Any]],
    ttft_slo_ms: float,
    gap_slo_ms: float,
) -> List[Dict[str, Any]]:
    by_length: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_length[int(row["interferer_input_tokens"])].append(row)
    output: List[Dict[str, Any]] = []
    for input_tokens, candidates in sorted(by_length.items()):
        for step in range(21):
            alpha = step / 20.0
            scored = []
            for row in candidates:
                objective = (
                    alpha * float(row["long_ttft_p50_ms_median"]) / ttft_slo_ms
                    + (1.0 - alpha)
                    * float(row["impact_gap_p99_ms_median"])
                    / gap_slo_ms
                )
                scored.append((objective, int(row["max_num_batched_tokens"]), row))
            objective, _, winner = min(scored, key=lambda item: (item[0], -item[1]))
            output.append(
                {
                    "interferer_input_tokens": input_tokens,
                    "alpha_ttft": alpha,
                    "selected_config": winner["config_label"],
                    "normalized_objective": objective,
                }
            )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    ttft_slo_ms: float,
    gap_slo_ms: float,
) -> None:
    import matplotlib.pyplot as plt

    lengths = sorted({int(row["interferer_input_tokens"]) for row in rows})
    figure, axes = plt.subplots(1, len(lengths), figsize=(14, 4.7), sharey=True)
    if len(lengths) == 1:
        axes = [axes]
    for axis, input_tokens in zip(axes, lengths, strict=True):
        selected = [row for row in rows if int(row["interferer_input_tokens"]) == input_tokens]
        frontier = sorted(
            (row for row in selected if row["pareto_optimal"]),
            key=lambda row: float(row["long_ttft_p50_ms_median"]),
        )
        axis.plot(
            [row["long_ttft_p50_ms_median"] for row in frontier],
            [row["impact_gap_p99_ms_median"] for row in frontier],
            color="#222222",
            linewidth=1.4,
            zorder=1,
        )
        for row in selected:
            color = "#2CA02C" if row["meets_both_slos"] else "#4C78A8"
            marker = "D" if row["pareto_optimal"] else "o"
            axis.scatter(
                row["long_ttft_p50_ms_median"],
                row["impact_gap_p99_ms_median"],
                color=color,
                marker=marker,
                s=62,
                zorder=2,
            )
            label = str(row["config_label"])
            display = label.replace("off_32768", "off").replace("on_", "")
            offsets = {
                "off_32768": (4, 18),
                "on_32768": (4, 8),
                "on_16384": (4, -2),
                "on_8192": (4, -13),
            }
            axis.annotate(
                display,
                (row["long_ttft_p50_ms_median"], row["impact_gap_p99_ms_median"]),
                xytext=offsets.get(label, (4, 4)),
                textcoords="offset points",
                fontsize=8,
            )
        axis.axvline(ttft_slo_ms, color="#E45756", linestyle="--", linewidth=1)
        axis.axhline(gap_slo_ms, color="#E45756", linestyle="--", linewidth=1)
        axis.set_yscale("log")
        axis.set_title(f"{input_tokens // 1024}K-token prefill")
        axis.set_xlabel("Long-request median TTFT (ms)")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("Background impact-window P99 SSE gap (ms)")
    figure.suptitle("S8-C: Chunk Budget Pareto Frontiers and Dual-SLO Region")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, default=1500.0)
    parser.add_argument("--gap-slo-ms", type=float, default=100.0)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    with args.trial_summary.open(encoding="utf-8", newline="") as handle:
        trials = list(csv.DictReader(handle))
    rows = aggregate(
        trials,
        seed=args.bootstrap_seed,
        ttft_slo_ms=args.ttft_slo_ms,
        gap_slo_ms=args.gap_slo_ms,
    )
    if not rows:
        raise RuntimeError("no injection trials found")
    policies = policy_map(rows, args.ttft_slo_ms, args.gap_slo_ms)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pareto_summary.csv", rows)
    write_csv(args.output_dir / "weighted_policy_map.csv", policies)
    make_figure(
        rows,
        args.output_dir / "ttft_decode_gap_pareto.png",
        args.ttft_slo_ms,
        args.gap_slo_ms,
    )
    audit = {
        "schema_version": 1,
        "input": str(args.trial_summary),
        "injection_trial_count": sum(int(row["n_runs"]) for row in rows),
        "aggregate_count": len(rows),
        "input_lengths": sorted({int(row["interferer_input_tokens"]) for row in rows}),
        "ttft_slo_ms": args.ttft_slo_ms,
        "background_p99_gap_slo_ms": args.gap_slo_ms,
        "run_is_statistical_unit": True,
        "bootstrap_samples": 5000,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
