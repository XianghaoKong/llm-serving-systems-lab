#!/usr/bin/env python3
"""Audit and plot the S8-D five-block Qwen2.5-7B extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED = ("off_32768", "on_4096", "on_1024")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    trial_paths = sorted(args.input_root.glob("block_*/*/trial/trial.json"))
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in trial_paths]
    with (args.analysis_dir / "aggregate_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        aggregates: List[Dict[str, Any]] = list(csv.DictReader(handle))
    by_label = {row["config_label"]: row for row in aggregates}
    counts = {
        label: sum(row.get("config_label") == label for row in metadata)
        for label in EXPECTED
    }
    concurrencies = sorted({int(row["background_concurrency"]) for row in metadata})
    complete = (
        len(metadata) == 15
        and set(by_label) == set(EXPECTED)
        and all(count == 5 for count in counts.values())
        and all(row.get("status") == "complete" and row.get("error") is None for row in metadata)
        and all(int(row["interferer_input_tokens"]) == 24576 for row in metadata)
        and len(concurrencies) == 1
    )
    summary = {
        label: {
            "p99_stall_ratio_median": float(by_label[label]["p99_stall_ratio_median"]),
            "p99_stall_ratio_ci_low": float(by_label[label]["p99_stall_ratio_ci_low"]),
            "p99_stall_ratio_ci_high": float(by_label[label]["p99_stall_ratio_ci_high"]),
            "long_ttft_p50_ms_median": float(by_label[label]["long_ttft_p50_ms_median"]),
            "impact_gap_max_ms_median": float(by_label[label]["impact_gap_max_ms_median"]),
        }
        for label in EXPECTED
        if label in by_label
    }
    if set(summary) == set(EXPECTED):
        baseline = summary["off_32768"]["p99_stall_ratio_median"]
        for label in ("on_4096", "on_1024"):
            summary[label]["stall_reduction_vs_off_pct"] = (
                (baseline - summary[label]["p99_stall_ratio_median"]) / baseline * 100.0
            )
    audit = {
        "schema_version": 1,
        "complete": complete,
        "trial_count": len(metadata),
        "expected_trial_count": 15,
        "runs_per_configuration": counts,
        "background_concurrencies": concurrencies,
        "request_errors": sum(row.get("error") is not None for row in metadata),
        "preemptions_total": sum(float(row.get("preemptions_delta") or 0) for row in metadata),
        "summary": summary,
    }
    (args.analysis_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    if set(summary) == set(EXPECTED):
        import matplotlib.pyplot as plt

        labels = ["off / 32K", "on / 4K", "on / 1K"]
        figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
        stall = [summary[label]["p99_stall_ratio_median"] for label in EXPECTED]
        ttft = [summary[label]["long_ttft_p50_ms_median"] for label in EXPECTED]
        axes[0].bar(labels, stall, color=["#E45756", "#4C78A8", "#54A24B"])
        axes[0].set_ylabel("Median background P99 stall ratio")
        axes[0].set_title("Decode protection")
        axes[1].bar(labels, ttft, color=["#E45756", "#4C78A8", "#54A24B"])
        axes[1].set_ylabel("Median long-request TTFT (ms)")
        axes[1].set_title("Long-request cost")
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle("S8-D: Qwen2.5-7B Directional Validation")
        figure.tight_layout()
        figure.savefig(args.analysis_dir / "model_extension_7b.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    print(json.dumps(audit, indent=2))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
