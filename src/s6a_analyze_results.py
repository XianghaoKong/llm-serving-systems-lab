#!/usr/bin/env python3

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


CONCURRENCIES = [1, 8, 32, 64, 128]
ENGINES = ["vllm", "sglang"]
REPEATS = 5

PATTERNS = {
    "successful_requests": r"Successful requests:\s+([0-9]+)",
    "benchmark_duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "request_throughput_rps": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "input_throughput_tps": r"Input token throughput \(tok/s\):\s+([0-9.]+)",
    "output_throughput_tps": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "peak_output_throughput_tps": r"Peak output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_throughput_tps": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
    "concurrency_observed": r"^Concurrency:\s+([0-9.]+)",

    "mean_e2e_ms": r"Mean E2E Latency \(ms\):\s+([0-9.]+)",
    "median_e2e_ms": r"Median E2E Latency \(ms\):\s+([0-9.]+)",
    "p90_e2e_ms": r"P90 E2E Latency \(ms\):\s+([0-9.]+)",
    "p95_e2e_ms": r"P95 E2E Latency \(ms\):\s+([0-9.]+)",
    "p99_e2e_ms": r"P99 E2E Latency \(ms\):\s+([0-9.]+)",

    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "p90_ttft_ms": r"P90 TTFT \(ms\):\s+([0-9.]+)",
    "p95_ttft_ms": r"P95 TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",

    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p90_tpot_ms": r"P90 TPOT \(ms\):\s+([0-9.]+)",
    "p95_tpot_ms": r"P95 TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",

    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    "median_itl_ms": r"Median ITL \(ms\):\s+([0-9.]+)",
    "p90_itl_ms": r"P90 ITL \(ms\):\s+([0-9.]+)",
    "p95_itl_ms": r"P95 ITL \(ms\):\s+([0-9.]+)",
    "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
    "max_itl_ms": r"Max ITL \(ms\):\s+([0-9.]+)",
}


def parse_stdout(path: Path):
    text = path.read_text(errors="replace")
    row = {}

    for key, pattern in PATTERNS.items():
        m = re.search(pattern, text, flags=re.MULTILINE)
        if not m:
            raise RuntimeError(f"Missing metric {key} in {path}")

        value = m.group(1)
        if key == "successful_requests":
            row[key] = int(value)
        else:
            row[key] = float(value)

    return row


def median(xs):
    return statistics.median(xs)


def mean(xs):
    return statistics.mean(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs)


def cv_pct(xs):
    m = mean(xs)
    if m == 0:
        return math.nan
    return 100.0 * std(xs) / m


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary_lookup, out_dir, metric, ylabel, filename):
    plt.figure(figsize=(7.2, 4.8))

    for engine in ENGINES:
        ys = [
            summary_lookup[(engine, c)][metric]
            for c in CONCURRENCIES
        ]
        plt.plot(CONCURRENCIES, ys, marker="o", label=engine)

    plt.xlabel("Max request concurrency")
    plt.ylabel(ylabel)
    plt.xticks(CONCURRENCIES)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("formal_dir", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    formal_dir = args.formal_dir.resolve()

    if not formal_dir.exists():
        raise SystemExit(f"Formal directory not found: {formal_dir}")

    formal_id = formal_dir.name

    if args.out_dir is None:
        out_dir = (
            Path("results/s6/analysis/s6a")
            / formal_id
        )
    else:
        out_dir = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print(f"Formal directory: {formal_dir}")
    print()

    for engine in ENGINES:
        for c in CONCURRENCIES:
            found = 0

            for rep in range(1, REPEATS + 1):
                path = (
                    formal_dir
                    / engine
                    / f"c{c}"
                    / f"repeat_{rep}"
                    / "stdout.txt"
                )

                if not path.exists():
                    raise RuntimeError(
                        f"Missing formal repeat: {path}"
                    )

                metrics = parse_stdout(path)

                row = {
                    "formal_id": formal_id,
                    "engine": engine,
                    "concurrency": c,
                    "repeat": rep,
                    **metrics,
                }

                rows.append(row)
                found += 1

            print(f"{engine:7s} C={c:<3d}: {found}/{REPEATS}")

    print()

    expected = len(ENGINES) * len(CONCURRENCIES) * REPEATS
    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} measured runs, got {len(rows)}"
        )

    print(f"Measured runs: {len(rows)}/{expected} PASS")

    repeat_fields = list(rows[0].keys())
    write_csv(
        out_dir / "repeat_metrics.csv",
        rows,
        repeat_fields,
    )

    metric_names = [
        k for k in rows[0].keys()
        if k not in {
            "formal_id",
            "engine",
            "concurrency",
            "repeat",
            "successful_requests",
        }
    ]

    summary_rows = []
    summary_lookup = {}

    for engine in ENGINES:
        for c in CONCURRENCIES:
            subset = [
                r for r in rows
                if r["engine"] == engine
                and r["concurrency"] == c
            ]

            summary = {
                "formal_id": formal_id,
                "engine": engine,
                "concurrency": c,
                "repeat_count": len(subset),
                "successful_requests_min": min(
                    r["successful_requests"] for r in subset
                ),
            }

            for metric in metric_names:
                xs = [r[metric] for r in subset]

                summary[f"{metric}_median"] = median(xs)
                summary[f"{metric}_mean"] = mean(xs)
                summary[f"{metric}_std"] = std(xs)
                summary[f"{metric}_cv_pct"] = cv_pct(xs)

            summary_rows.append(summary)

            summary_lookup[(engine, c)] = {
                metric: summary[f"{metric}_median"]
                for metric in metric_names
            }

    write_csv(
        out_dir / "summary_by_engine_concurrency.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )

    comparison_metrics = [
        "output_throughput_tps",
        "request_throughput_rps",
        "median_e2e_ms",
        "p95_e2e_ms",
        "p99_e2e_ms",
        "median_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "max_itl_ms",
    ]

    comparison_rows = []

    for c in CONCURRENCIES:
        v = summary_lookup[("vllm", c)]
        s = summary_lookup[("sglang", c)]

        for metric in comparison_metrics:
            vv = v[metric]
            ss = s[metric]

            pct = 100.0 * (ss - vv) / vv

            higher_is_better = metric in {
                "output_throughput_tps",
                "request_throughput_rps",
            }

            if higher_is_better:
                winner = (
                    "sglang" if ss > vv
                    else "vllm" if vv > ss
                    else "tie"
                )
            else:
                winner = (
                    "sglang" if ss < vv
                    else "vllm" if vv < ss
                    else "tie"
                )

            comparison_rows.append({
                "concurrency": c,
                "metric": metric,
                "vllm_median": vv,
                "sglang_median": ss,
                "sglang_vs_vllm_pct": pct,
                "winner": winner,
            })

    write_csv(
        out_dir / "engine_comparison.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )

    plot_metric(
        summary_lookup,
        out_dir,
        "output_throughput_tps",
        "Output throughput (tokens/s)",
        "throughput_vs_concurrency.png",
    )

    plot_metric(
        summary_lookup,
        out_dir,
        "p95_ttft_ms",
        "P95 TTFT (ms)",
        "p95_ttft_vs_concurrency.png",
    )

    plot_metric(
        summary_lookup,
        out_dir,
        "p99_ttft_ms",
        "P99 TTFT (ms)",
        "p99_ttft_vs_concurrency.png",
    )

    plot_metric(
        summary_lookup,
        out_dir,
        "p95_tpot_ms",
        "P95 TPOT (ms/token)",
        "p95_tpot_vs_concurrency.png",
    )

    plot_metric(
        summary_lookup,
        out_dir,
        "p95_itl_ms",
        "P95 ITL (ms)",
        "p95_itl_vs_concurrency.png",
    )

    print()
    print("Median-of-repeat headline metrics")
    print()

    header = (
        f"{'Engine':<8}"
        f"{'C':>6}"
        f"{'Out tok/s':>14}"
        f"{'P95 TTFT':>14}"
        f"{'P99 TTFT':>14}"
        f"{'P95 TPOT':>14}"
        f"{'P95 ITL':>14}"
    )
    print(header)
    print("-" * len(header))

    for c in CONCURRENCIES:
        for engine in ENGINES:
            x = summary_lookup[(engine, c)]

            print(
                f"{engine:<8}"
                f"{c:>6}"
                f"{x['output_throughput_tps']:>14.2f}"
                f"{x['p95_ttft_ms']:>14.2f}"
                f"{x['p99_ttft_ms']:>14.2f}"
                f"{x['p95_tpot_ms']:>14.2f}"
                f"{x['p95_itl_ms']:>14.2f}"
            )

    print()
    print("Repeat stability warnings (CV > 5%)")
    print()

    warning_count = 0

    stability_metrics = [
        "output_throughput_tps",
        "p95_ttft_ms",
        "p95_tpot_ms",
        "p95_itl_ms",
    ]

    for row in summary_rows:
        for metric in stability_metrics:
            cv = row[f"{metric}_cv_pct"]

            if cv > 5.0:
                warning_count += 1
                print(
                    f"WARNING "
                    f"{row['engine']} "
                    f"C={row['concurrency']} "
                    f"{metric}: CV={cv:.2f}%"
                )

    if warning_count == 0:
        print("None.")

    print()
    print("Analysis complete.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
