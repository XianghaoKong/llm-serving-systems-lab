#!/usr/bin/env python3
"""
R1-A — Reproducible analysis for the 1,000-request sequential serving baseline.

Expected project inputs
-----------------------
results/r1/raw/full/
├── eager_requests.csv
├── flash_requests.csv
├── eager_token_latency.csv.gz
├── flash_token_latency.csv.gz
├── eager_run_metadata.json
└── flash_run_metadata.json

Outputs
-------
results/r1/
├── summary/
│   ├── backend_summary.csv
│   ├── category_summary.csv
│   ├── paired_request_summary.csv
│   ├── request_pairs.csv
│   ├── input_band_summary.csv
│   ├── token_tail_summary.csv
│   ├── token_sequence_band_summary.csv
│   ├── token_position_band_summary.csv
│   ├── boundary_512_summary.csv
│   ├── same_output_paired_token_latency.csv
│   ├── same_output_token_pair_summary.csv
│   ├── r1_results_summary.json
│   └── r1_key_findings.md
└── figures/
    ├── ttft_by_category.png
    ├── ttft_speedup_by_input_band.png
    ├── decode_tpot_ecdf.png
    ├── decode_tpot_by_sequence_band.png
    └── peak_allocated_memory_by_category.png

Design principles
-----------------
- Request-level Eager/Flash data are merged one-to-one by request_id.
- TTFT/TPOT/ITL/memory are the primary backend-oriented metrics.
- E2E is reported as an observed production metric, but NOT interpreted as
  a pure kernel speedup because stochastic generation can diverge.
- Same-output-hash requests are used for the cleanest token-by-token paired
  decode comparison.
- "Effective sequence length" for decode token k is:
      runtime_input_tokens + token_index - 1
- No scipy dependency is required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_RAW_DIR = PROJECT_ROOT / "results" / "r1" / "raw" / "full"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "r1"

EXPECTED_REQUESTS = 1000

CATEGORY_ORDER = [
    "short_interactive",
    "knowledge_qa",
    "coding_request",
    "document_qa",
    "long_context_qa",
    "long_output",
]

CATEGORY_LABELS = {
    "short_interactive": "Short Interactive",
    "knowledge_qa": "Knowledge QA",
    "coding_request": "Coding",
    "document_qa": "Document QA",
    "long_context_qa": "Long-context QA",
    "long_output": "Long Output",
}

REQUEST_REQUIRED_COLUMNS = [
    "request_id",
    "workload_category",
    "backend",
    "status",
    "finish_reason",
    "prompt_hash",
    "frozen_input_tokens",
    "runtime_input_tokens",
    "max_new_tokens",
    "output_tokens",
    "output_hash",
    "sampling_seed",
    "model_ttft_ms",
    "request_ttft_ms",
    "mean_tpot_ms",
    "mean_stream_itl_ms",
    "e2e_request_ms",
    "decode_tokens_per_s",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "start_allocated_gib",
    "start_reserved_gib",
]

TOKEN_REQUIRED_COLUMNS = [
    "request_id",
    "workload_category",
    "backend",
    "token_index",
    "token_phase",
    "gpu_step_ms",
    "stream_cpu_ms",
    "stream_itl_ms",
    "is_eos",
]

REQUEST_METRICS = [
    "model_ttft_ms",
    "request_ttft_ms",
    "mean_tpot_ms",
    "mean_stream_itl_ms",
    "decode_tokens_per_s",
    "e2e_request_ms",
    "peak_allocated_gib",
    "peak_reserved_gib",
]

INPUT_BANDS = [
    (0, 128, "<=128"),
    (129, 512, "129-512"),
    (513, 1024, "513-1024"),
    (1025, 2048, "1025-2048"),
    (2049, 3072, "2049-3072"),
    (3073, 4096, "3073-4096"),
    (4097, math.inf, ">4096"),
]

SEQUENCE_BANDS = [
    (0, 128, "<=128"),
    (129, 256, "129-256"),
    (257, 512, "257-512"),
    (513, 1024, "513-1024"),
    (1025, 2048, "1025-2048"),
    (2049, 3072, "2049-3072"),
    (3073, 4096, "3073-4096"),
    (4097, math.inf, ">4096"),
]

TOKEN_POSITION_BANDS = [
    (2, 32, "2-32"),
    (33, 64, "33-64"),
    (65, 128, "65-128"),
    (129, 256, "129-256"),
    (257, 512, "257-512"),
    (513, 1024, "513-1024"),
    (1025, math.inf, ">1024"),
]


# =============================================================================
# Helpers
# =============================================================================

def q(series: pd.Series, value: float) -> float:
    return float(series.quantile(value))


def mean(series: pd.Series) -> float:
    return float(series.mean())


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float) / denominator.astype(float)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def ratio_percent_reduction(speedup: float) -> float:
    if speedup <= 0 or math.isnan(speedup):
        return float("nan")
    return 100.0 * (1.0 - 1.0 / speedup)


def percent_change(new: float, old: float) -> float:
    if old == 0:
        return float("nan")
    return 100.0 * (new - old) / old


def assign_band(value: float, bands: Sequence[Tuple[float, float, str]]) -> str:
    for lower, upper, label in bands:
        if lower <= value <= upper:
            return label
    return "unassigned"


def validate_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"{label}: missing required columns: {missing}")


def assert_unique_pairs(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    duplicate_count = int(df.duplicated(list(columns)).sum())
    if duplicate_count:
        raise RuntimeError(
            f"{label}: found {duplicate_count} duplicate keys for {list(columns)}"
        )


def metric_stats(series: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "p50": float(clean.quantile(0.50)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def metadata_key(meta: Dict[str, Any], *keys: str) -> Any:
    current: Any = meta
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


# =============================================================================
# Input loading and validation
# =============================================================================

def load_inputs(raw_dir: Path):
    paths = {
        "eager_requests": raw_dir / "eager_requests.csv",
        "flash_requests": raw_dir / "flash_requests.csv",
        "eager_tokens": raw_dir / "eager_token_latency.csv.gz",
        "flash_tokens": raw_dir / "flash_token_latency.csv.gz",
        "eager_meta": raw_dir / "eager_run_metadata.json",
        "flash_meta": raw_dir / "flash_run_metadata.json",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing R1 Full input files:\n  " + "\n  ".join(missing)
        )

    eager = pd.read_csv(paths["eager_requests"])
    flash = pd.read_csv(paths["flash_requests"])
    eager_tokens = pd.read_csv(paths["eager_tokens"], compression="gzip")
    flash_tokens = pd.read_csv(paths["flash_tokens"], compression="gzip")

    eager_meta = json.loads(paths["eager_meta"].read_text(encoding="utf-8"))
    flash_meta = json.loads(paths["flash_meta"].read_text(encoding="utf-8"))

    validate_columns(eager, REQUEST_REQUIRED_COLUMNS, "eager_requests")
    validate_columns(flash, REQUEST_REQUIRED_COLUMNS, "flash_requests")
    validate_columns(eager_tokens, TOKEN_REQUIRED_COLUMNS, "eager_tokens")
    validate_columns(flash_tokens, TOKEN_REQUIRED_COLUMNS, "flash_tokens")

    return eager, flash, eager_tokens, flash_tokens, eager_meta, flash_meta, paths


def validate_request_runs(
    eager: pd.DataFrame,
    flash: pd.DataFrame,
    eager_meta: Dict[str, Any],
    flash_meta: Dict[str, Any],
) -> Dict[str, Any]:
    if len(eager) != EXPECTED_REQUESTS or len(flash) != EXPECTED_REQUESTS:
        raise RuntimeError(
            f"Expected {EXPECTED_REQUESTS} request rows/backend, "
            f"found eager={len(eager)}, flash={len(flash)}"
        )

    if not (eager["status"] == "ok").all():
        raise RuntimeError("Eager Full contains non-ok requests.")

    if not (flash["status"] == "ok").all():
        raise RuntimeError("Flash Full contains non-ok requests.")

    assert_unique_pairs(eager, ["request_id"], "eager_requests")
    assert_unique_pairs(flash, ["request_id"], "flash_requests")

    eager_ids = eager["request_id"].tolist()
    flash_ids = flash["request_id"].tolist()

    if eager_ids != flash_ids:
        raise RuntimeError(
            "Eager and Flash request execution order is not identical."
        )

    for backend_name, frame in [("eager", eager), ("flash", flash)]:
        mismatch = int(
            (
                frame["frozen_input_tokens"].astype(int)
                != frame["runtime_input_tokens"].astype(int)
            ).sum()
        )
        if mismatch:
            raise RuntimeError(
                f"{backend_name}: {mismatch} frozen/runtime input-token mismatches."
            )

    metadata_checks = {
        "eager_status_complete": eager_meta.get("status") == "complete",
        "flash_status_complete": flash_meta.get("status") == "complete",
        "same_model": eager_meta.get("model_id") == flash_meta.get("model_id"),
        "same_dtype": eager_meta.get("dtype") == flash_meta.get("dtype"),
        "same_protocol_version": (
            eager_meta.get("r1_protocol_version")
            == flash_meta.get("r1_protocol_version")
        ),
        "same_corpus_sha256": (
            eager_meta.get("corpus_sha256")
            == flash_meta.get("corpus_sha256")
        ),
        "same_execution_seed": (
            eager_meta.get("execution_seed")
            == flash_meta.get("execution_seed")
        ),
        "same_sampling_policy": (
            eager_meta.get("sampling") == flash_meta.get("sampling")
        ),
        "eager_completed_1000": (
            eager_meta.get("completed_request_count") == EXPECTED_REQUESTS
        ),
        "flash_completed_1000": (
            flash_meta.get("completed_request_count") == EXPECTED_REQUESTS
        ),
        "eager_backend_verification_pass": (
            metadata_key(eager_meta, "backend_verification", "status") == "pass"
        ),
        "flash_backend_verification_pass": (
            metadata_key(flash_meta, "backend_verification", "status") == "pass"
        ),
        "flash_forced_backend": bool(
            metadata_key(flash_meta, "backend_verification", "forced_flash_context")
        ),
    }

    failed = [key for key, value in metadata_checks.items() if not value]
    if failed:
        raise RuntimeError(f"Metadata validation failed: {failed}")

    return {
        "request_rows_per_backend": EXPECTED_REQUESTS,
        "execution_order_identical": True,
        "frozen_runtime_input_token_mismatches": 0,
        "metadata_checks": metadata_checks,
        "flash_already_completed_on_resume": int(
            flash_meta.get("already_completed_on_resume", 0)
        ),
    }


def validate_token_trace(
    tokens: pd.DataFrame,
    requests: pd.DataFrame,
    backend: str,
) -> Dict[str, Any]:
    assert_unique_pairs(
        tokens,
        ["request_id", "token_index"],
        f"{backend}_token_trace",
    )

    request_ids = set(requests["request_id"].astype(str))
    trace_ids = set(tokens["request_id"].astype(str))

    if request_ids != trace_ids:
        raise RuntimeError(
            f"{backend}: token trace request IDs do not exactly match request CSV."
        )

    trace_counts = (
        tokens.groupby("request_id", observed=True)
        .size()
        .rename("trace_tokens")
    )

    expected = requests.set_index("request_id")["output_tokens"].astype(int)
    joined = pd.concat([expected, trace_counts], axis=1)

    mismatch_count = int(
        (joined["output_tokens"] != joined["trace_tokens"]).sum()
    )

    if mismatch_count:
        raise RuntimeError(
            f"{backend}: {mismatch_count} request/token-trace length mismatches."
        )

    first_token_count = int((tokens["token_phase"] == "prefill_first_token").sum())
    decode_count = int((tokens["token_phase"] == "decode").sum())

    if first_token_count != EXPECTED_REQUESTS:
        raise RuntimeError(
            f"{backend}: expected {EXPECTED_REQUESTS} first-token rows, "
            f"found {first_token_count}"
        )

    return {
        "rows": int(len(tokens)),
        "unique_requests": int(tokens["request_id"].nunique()),
        "first_token_rows": first_token_count,
        "decode_token_rows": decode_count,
        "duplicate_request_token_keys": 0,
        "trace_length_mismatches": 0,
    }


# =============================================================================
# Request-level analysis
# =============================================================================

def build_request_pairs(eager: pd.DataFrame, flash: pd.DataFrame) -> pd.DataFrame:
    paired = eager.merge(
        flash,
        on="request_id",
        how="inner",
        suffixes=("_eager", "_flash"),
        validate="one_to_one",
        sort=False,
    )

    identity_columns = [
        "workload_category",
        "prompt_hash",
        "frozen_input_tokens",
        "runtime_input_tokens",
        "max_new_tokens",
        "sampling_seed",
    ]

    for column in identity_columns:
        left = paired[f"{column}_eager"]
        right = paired[f"{column}_flash"]

        if not left.equals(right):
            raise RuntimeError(
                f"Cross-backend identity mismatch in column: {column}"
            )

        paired[column] = left

    paired["same_output_hash"] = (
        paired["output_hash_eager"] == paired["output_hash_flash"]
    )
    paired["same_output_tokens"] = (
        paired["output_tokens_eager"] == paired["output_tokens_flash"]
    )
    paired["same_finish_reason"] = (
        paired["finish_reason_eager"] == paired["finish_reason_flash"]
    )

    paired["output_token_delta_flash_minus_eager"] = (
        paired["output_tokens_flash"] - paired["output_tokens_eager"]
    )

    # Latency speedup: >1 means Flash is faster.
    for metric in [
        "model_ttft_ms",
        "request_ttft_ms",
        "mean_tpot_ms",
        "mean_stream_itl_ms",
        "e2e_request_ms",
    ]:
        paired[f"{metric}_speedup_eager_over_flash"] = safe_ratio(
            paired[f"{metric}_eager"],
            paired[f"{metric}_flash"],
        )

    # Throughput speedup: >1 means Flash is faster.
    paired["decode_throughput_speedup_flash_over_eager"] = safe_ratio(
        paired["decode_tokens_per_s_flash"],
        paired["decode_tokens_per_s_eager"],
    )

    paired["peak_allocated_reduction_pct"] = (
        100.0
        * (
            1.0
            - safe_ratio(
                paired["peak_allocated_gib_flash"],
                paired["peak_allocated_gib_eager"],
            )
        )
    )

    paired["peak_reserved_reduction_pct"] = (
        100.0
        * (
            1.0
            - safe_ratio(
                paired["peak_reserved_gib_flash"],
                paired["peak_reserved_gib_eager"],
            )
        )
    )

    paired["input_band"] = paired["runtime_input_tokens"].apply(
        lambda x: assign_band(float(x), INPUT_BANDS)
    )

    return paired


def backend_summary(eager: pd.DataFrame, flash: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for backend, frame in [("eager", eager), ("flash", flash)]:
        row: Dict[str, Any] = {
            "backend": backend,
            "requests": int(len(frame)),
            "total_output_tokens": int(frame["output_tokens"].sum()),
            "eos_count": int((frame["finish_reason"] == "eos").sum()),
            "length_count": int((frame["finish_reason"] == "length").sum()),
            "eos_rate": float((frame["finish_reason"] == "eos").mean()),
            "length_rate": float((frame["finish_reason"] == "length").mean()),
            "service_time_sum_minutes": float(
                frame["e2e_request_ms"].sum() / 60000.0
            ),
            "output_tokens_per_service_second": float(
                frame["output_tokens"].sum()
                / (frame["e2e_request_ms"].sum() / 1000.0)
            ),
            "max_peak_allocated_gib": float(frame["peak_allocated_gib"].max()),
            "max_peak_reserved_gib": float(frame["peak_reserved_gib"].max()),
            "start_allocated_min_gib": float(frame["start_allocated_gib"].min()),
            "start_allocated_max_gib": float(frame["start_allocated_gib"].max()),
        }

        for metric in REQUEST_METRICS:
            stats = metric_stats(frame[metric])
            for stat_name, value in stats.items():
                if stat_name == "count":
                    continue
                row[f"{metric}_{stat_name}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def category_summary(eager: pd.DataFrame, flash: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for category in CATEGORY_ORDER:
        for backend, frame in [("eager", eager), ("flash", flash)]:
            subset = frame[frame["workload_category"] == category]

            row: Dict[str, Any] = {
                "workload_category": category,
                "backend": backend,
                "requests": int(len(subset)),
                "input_tokens_p50": q(subset["runtime_input_tokens"], 0.50),
                "input_tokens_p95": q(subset["runtime_input_tokens"], 0.95),
                "output_tokens_p50": q(subset["output_tokens"], 0.50),
                "output_tokens_p95": q(subset["output_tokens"], 0.95),
                "eos_rate": float((subset["finish_reason"] == "eos").mean()),
                "length_rate": float((subset["finish_reason"] == "length").mean()),
                "request_ttft_ms_p50": q(subset["request_ttft_ms"], 0.50),
                "request_ttft_ms_p95": q(subset["request_ttft_ms"], 0.95),
                "mean_tpot_ms_p50": q(subset["mean_tpot_ms"], 0.50),
                "mean_stream_itl_ms_p50": q(
                    subset["mean_stream_itl_ms"], 0.50
                ),
                "decode_tokens_per_s_p50": q(
                    subset["decode_tokens_per_s"], 0.50
                ),
                "e2e_request_ms_p50": q(subset["e2e_request_ms"], 0.50),
                "peak_allocated_gib_p50": q(
                    subset["peak_allocated_gib"], 0.50
                ),
                "peak_reserved_gib_p50": q(
                    subset["peak_reserved_gib"], 0.50
                ),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def paired_request_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    groups = [("ALL", paired)] + [
        (category, paired[paired["workload_category"] == category])
        for category in CATEGORY_ORDER
    ]

    for label, subset in groups:
        row = {
            "workload_category": label,
            "requests": int(len(subset)),
            "same_output_hash_count": int(subset["same_output_hash"].sum()),
            "same_output_hash_rate": float(subset["same_output_hash"].mean()),
            "same_output_tokens_count": int(subset["same_output_tokens"].sum()),
            "same_output_tokens_rate": float(subset["same_output_tokens"].mean()),
            "same_finish_reason_count": int(subset["same_finish_reason"].sum()),
            "same_finish_reason_rate": float(
                subset["same_finish_reason"].mean()
            ),
            "median_model_ttft_speedup": q(
                subset["model_ttft_ms_speedup_eager_over_flash"], 0.50
            ),
            "median_request_ttft_speedup": q(
                subset["request_ttft_ms_speedup_eager_over_flash"], 0.50
            ),
            "median_tpot_speedup": q(
                subset["mean_tpot_ms_speedup_eager_over_flash"], 0.50
            ),
            "median_stream_itl_speedup": q(
                subset["mean_stream_itl_ms_speedup_eager_over_flash"], 0.50
            ),
            "median_decode_throughput_speedup": q(
                subset["decode_throughput_speedup_flash_over_eager"], 0.50
            ),
            "median_observed_e2e_speedup": q(
                subset["e2e_request_ms_speedup_eager_over_flash"], 0.50
            ),
            "median_peak_allocated_reduction_pct": q(
                subset["peak_allocated_reduction_pct"], 0.50
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def input_band_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, _, label in INPUT_BANDS:
        subset = paired[paired["input_band"] == label]

        if subset.empty:
            continue

        rows.append({
            "input_band": label,
            "requests": int(len(subset)),
            "input_tokens_min": int(subset["runtime_input_tokens"].min()),
            "input_tokens_p50": q(subset["runtime_input_tokens"], 0.50),
            "input_tokens_max": int(subset["runtime_input_tokens"].max()),
            "eager_request_ttft_ms_p50": q(
                subset["request_ttft_ms_eager"], 0.50
            ),
            "flash_request_ttft_ms_p50": q(
                subset["request_ttft_ms_flash"], 0.50
            ),
            "paired_request_ttft_speedup_p50": q(
                subset["request_ttft_ms_speedup_eager_over_flash"], 0.50
            ),
            "paired_model_ttft_speedup_p50": q(
                subset["model_ttft_ms_speedup_eager_over_flash"], 0.50
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Token-level analysis
# =============================================================================

def prepare_decode_tokens(
    tokens: pd.DataFrame,
    requests: pd.DataFrame,
) -> pd.DataFrame:
    decode = tokens[tokens["token_phase"] == "decode"].copy()

    input_lookup = requests[
        ["request_id", "runtime_input_tokens"]
    ].drop_duplicates("request_id")

    decode = decode.merge(
        input_lookup,
        on="request_id",
        how="left",
        validate="many_to_one",
    )

    if decode["runtime_input_tokens"].isna().any():
        raise RuntimeError("Token trace contains request IDs missing from request CSV.")

    decode["effective_sequence_length"] = (
        decode["runtime_input_tokens"].astype(int)
        + decode["token_index"].astype(int)
        - 1
    )

    decode["sequence_band"] = decode["effective_sequence_length"].apply(
        lambda x: assign_band(float(x), SEQUENCE_BANDS)
    )

    decode["token_position_band"] = decode["token_index"].apply(
        lambda x: assign_band(float(x), TOKEN_POSITION_BANDS)
    )

    return decode


def token_tail_summary(
    eager_decode: pd.DataFrame,
    flash_decode: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for backend, frame in [
        ("eager", eager_decode),
        ("flash", flash_decode),
    ]:
        gpu = frame["gpu_step_ms"].dropna()
        stream = frame["stream_itl_ms"].dropna()
        cpu = frame["stream_cpu_ms"].dropna()

        rows.append({
            "backend": backend,
            "decode_token_events": int(len(frame)),
            "gpu_tpot_mean_ms": mean(gpu),
            "gpu_tpot_p50_ms": q(gpu, 0.50),
            "gpu_tpot_p95_ms": q(gpu, 0.95),
            "gpu_tpot_p99_ms": q(gpu, 0.99),
            "gpu_tpot_p999_ms": q(gpu, 0.999),
            "gpu_tpot_max_ms": float(gpu.max()),
            "stream_itl_p50_ms": q(stream, 0.50),
            "stream_itl_p95_ms": q(stream, 0.95),
            "stream_itl_p99_ms": q(stream, 0.99),
            "stream_itl_p999_ms": q(stream, 0.999),
            "stream_itl_max_ms": float(stream.max()),
            "stream_cpu_p50_ms": q(cpu, 0.50),
            "stream_cpu_p99_ms": q(cpu, 0.99),
            "gpu_over_25ms_count": int((gpu > 25.0).sum()),
            "gpu_over_25ms_rate": float((gpu > 25.0).mean()),
            "gpu_over_30ms_count": int((gpu > 30.0).sum()),
            "gpu_over_30ms_rate": float((gpu > 30.0).mean()),
        })

    return pd.DataFrame(rows)


def token_band_summary(
    eager_decode: pd.DataFrame,
    flash_decode: pd.DataFrame,
    band_column: str,
    band_definition: Sequence[Tuple[float, float, str]],
    label_column: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, _, label in band_definition:
        eager_subset = eager_decode[eager_decode[band_column] == label]
        flash_subset = flash_decode[flash_decode[band_column] == label]

        if eager_subset.empty and flash_subset.empty:
            continue

        rows.append({
            label_column: label,
            "eager_tokens": int(len(eager_subset)),
            "flash_tokens": int(len(flash_subset)),
            "eager_gpu_tpot_p50_ms": (
                q(eager_subset["gpu_step_ms"], 0.50)
                if not eager_subset.empty
                else float("nan")
            ),
            "eager_gpu_tpot_p95_ms": (
                q(eager_subset["gpu_step_ms"], 0.95)
                if not eager_subset.empty
                else float("nan")
            ),
            "flash_gpu_tpot_p50_ms": (
                q(flash_subset["gpu_step_ms"], 0.50)
                if not flash_subset.empty
                else float("nan")
            ),
            "flash_gpu_tpot_p95_ms": (
                q(flash_subset["gpu_step_ms"], 0.95)
                if not flash_subset.empty
                else float("nan")
            ),
            "p50_speedup_ratio": (
                q(eager_subset["gpu_step_ms"], 0.50)
                / q(flash_subset["gpu_step_ms"], 0.50)
                if not eager_subset.empty and not flash_subset.empty
                else float("nan")
            ),
        })

    return pd.DataFrame(rows)


def boundary_512_summary(
    eager_decode: pd.DataFrame,
    flash_decode: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for sequence_length in range(508, 517):
        for backend, frame in [
            ("eager", eager_decode),
            ("flash", flash_decode),
        ]:
            subset = frame[
                frame["effective_sequence_length"] == sequence_length
            ]

            rows.append({
                "effective_sequence_length": sequence_length,
                "backend": backend,
                "token_events": int(len(subset)),
                "gpu_tpot_p50_ms": (
                    q(subset["gpu_step_ms"], 0.50)
                    if not subset.empty
                    else float("nan")
                ),
                "gpu_tpot_p95_ms": (
                    q(subset["gpu_step_ms"], 0.95)
                    if not subset.empty
                    else float("nan")
                ),
                "gpu_tpot_max_ms": (
                    float(subset["gpu_step_ms"].max())
                    if not subset.empty
                    else float("nan")
                ),
            })

    return pd.DataFrame(rows)


def same_output_token_pair_analysis(
    paired_requests: pd.DataFrame,
    eager_decode: pd.DataFrame,
    flash_decode: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    same_ids = set(
        paired_requests.loc[
            paired_requests["same_output_hash"],
            "request_id",
        ].astype(str)
    )

    eager_same = eager_decode[
        eager_decode["request_id"].astype(str).isin(same_ids)
    ][
        [
            "request_id",
            "token_index",
            "gpu_step_ms",
            "stream_itl_ms",
            "effective_sequence_length",
        ]
    ].copy()

    flash_same = flash_decode[
        flash_decode["request_id"].astype(str).isin(same_ids)
    ][
        [
            "request_id",
            "token_index",
            "gpu_step_ms",
            "stream_itl_ms",
            "effective_sequence_length",
        ]
    ].copy()

    token_pairs = eager_same.merge(
        flash_same,
        on=["request_id", "token_index"],
        how="inner",
        suffixes=("_eager", "_flash"),
        validate="one_to_one",
    )

    if not (
        token_pairs["effective_sequence_length_eager"]
        == token_pairs["effective_sequence_length_flash"]
    ).all():
        raise RuntimeError(
            "Same-output token pairs disagree on effective sequence length."
        )

    token_pairs["gpu_tpot_speedup_eager_over_flash"] = safe_ratio(
        token_pairs["gpu_step_ms_eager"],
        token_pairs["gpu_step_ms_flash"],
    )

    token_pairs["stream_itl_speedup_eager_over_flash"] = safe_ratio(
        token_pairs["stream_itl_ms_eager"],
        token_pairs["stream_itl_ms_flash"],
    )

    summary = pd.DataFrame([{
        "same_output_hash_requests": int(len(same_ids)),
        "paired_decode_token_events": int(len(token_pairs)),
        "median_gpu_tpot_speedup": q(
            token_pairs["gpu_tpot_speedup_eager_over_flash"], 0.50
        ),
        "p05_gpu_tpot_speedup": q(
            token_pairs["gpu_tpot_speedup_eager_over_flash"], 0.05
        ),
        "p95_gpu_tpot_speedup": q(
            token_pairs["gpu_tpot_speedup_eager_over_flash"], 0.95
        ),
        "median_stream_itl_speedup": q(
            token_pairs["stream_itl_speedup_eager_over_flash"], 0.50
        ),
    }])

    return token_pairs, summary


# =============================================================================
# Figures
# =============================================================================

def save_ttft_by_category(
    category_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    pivot = category_df.pivot(
        index="workload_category",
        columns="backend",
        values="request_ttft_ms_p50",
    ).reindex(CATEGORY_ORDER)

    x = np.arange(len(pivot))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(x - width / 2, pivot["eager"], width, label="Eager")
    ax.bar(x + width / 2, pivot["flash"], width, label="Flash")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CATEGORY_LABELS[value] for value in pivot.index],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Median request TTFT (ms)")
    ax.set_title("R1 Full: TTFT by workload category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "ttft_by_category.png", dpi=180)
    plt.close(fig)


def save_ttft_speedup_by_input_band(
    band_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(
        band_df["input_band"],
        band_df["paired_request_ttft_speedup_p50"],
    )
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("Median paired speedup (Eager / Flash)")
    ax.set_xlabel("Runtime input-token band")
    ax.set_title("R1 Full: Flash TTFT speedup grows with context length")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "ttft_speedup_by_input_band.png", dpi=180)
    plt.close(fig)


def ecdf_xy(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy())
    y = np.arange(1, len(arr) + 1) / len(arr)
    return arr, y


def save_decode_tpot_ecdf(
    eager_decode: pd.DataFrame,
    flash_decode: pd.DataFrame,
    figures_dir: Path,
) -> None:
    eager_x, eager_y = ecdf_xy(eager_decode["gpu_step_ms"])
    flash_x, flash_y = ecdf_xy(flash_decode["gpu_step_ms"])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(eager_x, eager_y, label="Eager")
    ax.plot(flash_x, flash_y, label="Flash")
    ax.set_xlim(
        0,
        max(
            q(eager_decode["gpu_step_ms"], 0.9995),
            q(flash_decode["gpu_step_ms"], 0.9995),
        ),
    )
    ax.set_xlabel("Decode GPU token latency (ms)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("R1 Full: Decode token-latency distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "decode_tpot_ecdf.png", dpi=180)
    plt.close(fig)


def save_decode_tpot_by_sequence_band(
    sequence_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    x = np.arange(len(sequence_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar(
        x - width / 2,
        sequence_df["eager_gpu_tpot_p50_ms"],
        width,
        label="Eager",
    )
    ax.bar(
        x + width / 2,
        sequence_df["flash_gpu_tpot_p50_ms"],
        width,
        label="Flash",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sequence_df["sequence_band"], rotation=20, ha="right")
    ax.set_ylabel("Median decode GPU latency (ms/token)")
    ax.set_xlabel("Effective sequence-length band")
    ax.set_title("R1 Full: Decode latency remains stable as sequence grows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "decode_tpot_by_sequence_band.png", dpi=180)
    plt.close(fig)


def save_peak_allocated_memory_by_category(
    category_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    pivot = category_df.pivot(
        index="workload_category",
        columns="backend",
        values="peak_allocated_gib_p50",
    ).reindex(CATEGORY_ORDER)

    x = np.arange(len(pivot))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(x - width / 2, pivot["eager"], width, label="Eager")
    ax.bar(x + width / 2, pivot["flash"], width, label="Flash")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CATEGORY_LABELS[value] for value in pivot.index],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Median peak allocated GPU memory (GiB)")
    ax.set_title("R1 Full: Peak allocated memory by workload category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "peak_allocated_memory_by_category.png",
        dpi=180,
    )
    plt.close(fig)


# =============================================================================
# Findings / JSON summary
# =============================================================================

def build_results_json(
    eager: pd.DataFrame,
    flash: pd.DataFrame,
    paired: pd.DataFrame,
    backend_df: pd.DataFrame,
    category_df: pd.DataFrame,
    input_band_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    same_output_summary: pd.DataFrame,
    integrity: Dict[str, Any],
    eager_meta: Dict[str, Any],
    flash_meta: Dict[str, Any],
) -> Dict[str, Any]:
    eager_backend = backend_df.set_index("backend").loc["eager"]
    flash_backend = backend_df.set_index("backend").loc["flash"]

    long_ctx_pairs = paired[
        paired["workload_category"] == "long_context_qa"
    ]
    long_ctx_eager = eager[
        eager["workload_category"] == "long_context_qa"
    ]
    long_ctx_flash = flash[
        flash["workload_category"] == "long_context_qa"
    ]

    all_pair = paired_request_summary(paired).iloc[0]

    result = {
        "stage": "R1-A",
        "status": "PASS",
        "integrity": integrity,
        "protocol": {
            "model_id": eager_meta.get("model_id"),
            "dtype": eager_meta.get("dtype"),
            "protocol_version": eager_meta.get("r1_protocol_version"),
            "concurrency": eager_meta.get("concurrency"),
            "batch_size": eager_meta.get("batch_size"),
            "sampling": eager_meta.get("sampling"),
            "corpus_sha256": eager_meta.get("corpus_sha256"),
            "execution_seed": eager_meta.get("execution_seed"),
            "eager_backend_verification": eager_meta.get(
                "backend_verification"
            ),
            "flash_backend_verification": flash_meta.get(
                "backend_verification"
            ),
        },
        "request_level": {
            "same_output_hash_requests": int(paired["same_output_hash"].sum()),
            "same_output_hash_rate": float(paired["same_output_hash"].mean()),
            "same_output_length_requests": int(
                paired["same_output_tokens"].sum()
            ),
            "same_output_length_rate": float(
                paired["same_output_tokens"].mean()
            ),
            "same_finish_reason_requests": int(
                paired["same_finish_reason"].sum()
            ),
            "eager_total_output_tokens": int(eager["output_tokens"].sum()),
            "flash_total_output_tokens": int(flash["output_tokens"].sum()),
            "flash_output_token_change_pct": percent_change(
                float(flash["output_tokens"].sum()),
                float(eager["output_tokens"].sum()),
            ),
            "eager_service_time_minutes": float(
                eager["e2e_request_ms"].sum() / 60000.0
            ),
            "flash_service_time_minutes": float(
                flash["e2e_request_ms"].sum() / 60000.0
            ),
            "service_time_reduction_pct": (
                100.0
                * (
                    1.0
                    - flash["e2e_request_ms"].sum()
                    / eager["e2e_request_ms"].sum()
                )
            ),
            "median_request_ttft_speedup": float(
                all_pair["median_request_ttft_speedup"]
            ),
            "median_tpot_speedup": float(
                all_pair["median_tpot_speedup"]
            ),
            "median_decode_throughput_speedup": float(
                all_pair["median_decode_throughput_speedup"]
            ),
            "eager_decode_throughput_p50": float(
                eager_backend["decode_tokens_per_s_p50"]
            ),
            "flash_decode_throughput_p50": float(
                flash_backend["decode_tokens_per_s_p50"]
            ),
        },
        "long_context": {
            "requests": int(len(long_ctx_pairs)),
            "eager_request_ttft_p50_ms": q(
                long_ctx_eager["request_ttft_ms"], 0.50
            ),
            "flash_request_ttft_p50_ms": q(
                long_ctx_flash["request_ttft_ms"], 0.50
            ),
            "paired_request_ttft_speedup_p50": q(
                long_ctx_pairs[
                    "request_ttft_ms_speedup_eager_over_flash"
                ],
                0.50,
            ),
            "eager_peak_allocated_p50_gib": q(
                long_ctx_eager["peak_allocated_gib"], 0.50
            ),
            "flash_peak_allocated_p50_gib": q(
                long_ctx_flash["peak_allocated_gib"], 0.50
            ),
            "median_peak_allocated_reduction_pct": (
                100.0
                * (
                    1.0
                    - q(long_ctx_flash["peak_allocated_gib"], 0.50)
                    / q(long_ctx_eager["peak_allocated_gib"], 0.50)
                )
            ),
        },
        "token_level": {
            "eager_decode_events": int(
                tail_df.set_index("backend").loc["eager", "decode_token_events"]
            ),
            "flash_decode_events": int(
                tail_df.set_index("backend").loc["flash", "decode_token_events"]
            ),
            "decode_events_total": int(
                tail_df["decode_token_events"].sum()
            ),
            "eager_gpu_tpot_p50_ms": float(
                tail_df.set_index("backend").loc["eager", "gpu_tpot_p50_ms"]
            ),
            "flash_gpu_tpot_p50_ms": float(
                tail_df.set_index("backend").loc["flash", "gpu_tpot_p50_ms"]
            ),
            "eager_gpu_tpot_p99_ms": float(
                tail_df.set_index("backend").loc["eager", "gpu_tpot_p99_ms"]
            ),
            "flash_gpu_tpot_p99_ms": float(
                tail_df.set_index("backend").loc["flash", "gpu_tpot_p99_ms"]
            ),
            "same_output_hash_token_pairs": int(
                same_output_summary.iloc[0]["paired_decode_token_events"]
            ),
            "same_output_hash_median_gpu_tpot_speedup": float(
                same_output_summary.iloc[0]["median_gpu_tpot_speedup"]
            ),
            "same_output_hash_median_stream_itl_speedup": float(
                same_output_summary.iloc[0]["median_stream_itl_speedup"]
            ),
        },
        "interpretation_notes": {
            "e2e": (
                "Observed E2E latency includes stochastic generation-trajectory "
                "differences and must not be interpreted as pure kernel speedup."
            ),
            "reserved_memory": (
                "CUDA reserved memory is caching-allocator reservation, not "
                "active tensor allocation."
            ),
            "boundary_512": (
                "The earlier controlled 512-token boundary effect is not "
                "treated as reproduced unless the R1 508-516 summary shows a "
                "consistent 512->513 jump across the production-like workload."
            ),
        },
    }

    return result


def write_findings_markdown(
    result: Dict[str, Any],
    input_band_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    output_path: Path,
) -> None:
    req = result["request_level"]
    long_ctx = result["long_context"]
    tok = result["token_level"]

    tail_index = tail_df.set_index("backend")

    long_band = input_band_df[
        input_band_df["input_band"] == "3073-4096"
    ]

    long_band_speedup = (
        float(long_band.iloc[0]["paired_request_ttft_speedup_p50"])
        if not long_band.empty
        else float("nan")
    )

    text = f"""# R1 Sequential Serving Baseline — Key Findings

## Protocol

- Model: `{result["protocol"]["model_id"]}`
- Dtype: `{result["protocol"]["dtype"]}`
- Concurrency: `{result["protocol"]["concurrency"]}`
- Batch size: `{result["protocol"]["batch_size"]}`
- Frozen corpus SHA-256: `{result["protocol"]["corpus_sha256"]}`
- Requests per backend: 1,000
- Eager and Flash used the same frozen corpus and deterministic execution order.
- Flash was verified through the forced Flash SDPA backend context.

## Main findings

1. **Long-context prefill:** For 3,073–4,096-token requests, median paired
   request-TTFT speedup was **{long_band_speedup:.2f}x**. Across the dedicated
   Long-context QA category, Eager median request TTFT was
   **{long_ctx["eager_request_ttft_p50_ms"]:.1f} ms** versus
   **{long_ctx["flash_request_ttft_p50_ms"]:.1f} ms** for Flash.

2. **Decode:** Across **{tok["decode_events_total"]:,}** measured decode-token
   events, median GPU token latency fell from
   **{tok["eager_gpu_tpot_p50_ms"]:.2f} ms** to
   **{tok["flash_gpu_tpot_p50_ms"]:.2f} ms**. Request-level median decode
   throughput increased from **{req["eager_decode_throughput_p50"]:.2f}**
   to **{req["flash_decode_throughput_p50"]:.2f} tok/s**.

3. **Tail token latency:** P99 GPU decode latency was
   **{tok["eager_gpu_tpot_p99_ms"]:.2f} ms** for Eager and
   **{tok["flash_gpu_tpot_p99_ms"]:.2f} ms** for Flash.

4. **Memory:** Long-context median peak allocated GPU memory fell from
   **{long_ctx["eager_peak_allocated_p50_gib"]:.3f} GiB** to
   **{long_ctx["flash_peak_allocated_p50_gib"]:.3f} GiB**
   (**{long_ctx["median_peak_allocated_reduction_pct"]:.1f}% reduction**).

5. **Same-output sanity check:** Only
   **{req["same_output_hash_requests"]}/1000** requests produced exactly the
   same output token sequence across stochastic Eager and Flash runs. On those
   exactly matched generations, the median paired token-level GPU speedup was
   still **{tok["same_output_hash_median_gpu_tpot_speedup"]:.3f}x**.

6. **Observed service time:** The sum of request service times fell from
   **{req["eager_service_time_minutes"]:.2f} min** to
   **{req["flash_service_time_minutes"]:.2f} min**
   (**{req["service_time_reduction_pct"]:.1f}% lower**) even though Flash
   generated **{req["flash_output_token_change_pct"]:+.1f}%** more output
   tokens in this stochastic run.

## Interpretation boundary

E2E latency is a production-observed metric, not a pure kernel-causal metric,
because stochastic generation trajectories diverge across backends. TTFT,
TPOT, streaming ITL, decode throughput, and memory are the primary
backend-performance metrics.

CUDA `reserved` memory is allocator-reserved capacity and must not be described
as active/resident tensor memory.

The R1 production-like BF16 workload does not by itself establish the earlier
controlled 512-token boundary effect. See `boundary_512_summary.csv` for the
508–516-token measurements.
"""

    output_path.write_text(text, encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze R1 Eager vs Flash Full results."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing the six R1 Full raw files.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="R1 output root containing summary/ and figures/.",
    )

    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    output_root = args.output_root.resolve()
    summary_dir = output_root / "summary"
    figures_dir = output_root / "figures"

    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("R1-A — FULL REQUEST + TOKEN-LEVEL ANALYSIS")
    print("=" * 88)
    print(f"Raw dir:      {raw_dir}")
    print(f"Summary dir:  {summary_dir}")
    print(f"Figures dir:  {figures_dir}")

    (
        eager,
        flash,
        eager_tokens,
        flash_tokens,
        eager_meta,
        flash_meta,
        paths,
    ) = load_inputs(raw_dir)

    print("\n[1/7] Validating request-level integrity...")
    request_integrity = validate_request_runs(
        eager,
        flash,
        eager_meta,
        flash_meta,
    )

    print("[2/7] Validating token traces...")
    eager_trace_integrity = validate_token_trace(
        eager_tokens,
        eager,
        "eager",
    )
    flash_trace_integrity = validate_token_trace(
        flash_tokens,
        flash,
        "flash",
    )

    integrity = {
        "request_level": request_integrity,
        "eager_token_trace": eager_trace_integrity,
        "flash_token_trace": flash_trace_integrity,
    }

    print("[3/7] Building request-level summaries...")
    paired = build_request_pairs(eager, flash)
    backend_df = backend_summary(eager, flash)
    category_df = category_summary(eager, flash)
    paired_df = paired_request_summary(paired)
    input_band_df = input_band_summary(paired)

    print("[4/7] Building token-level summaries...")
    eager_decode = prepare_decode_tokens(eager_tokens, eager)
    flash_decode = prepare_decode_tokens(flash_tokens, flash)

    tail_df = token_tail_summary(eager_decode, flash_decode)

    sequence_df = token_band_summary(
        eager_decode,
        flash_decode,
        band_column="sequence_band",
        band_definition=SEQUENCE_BANDS,
        label_column="sequence_band",
    )

    position_df = token_band_summary(
        eager_decode,
        flash_decode,
        band_column="token_position_band",
        band_definition=TOKEN_POSITION_BANDS,
        label_column="token_position_band",
    )

    boundary_df = boundary_512_summary(
        eager_decode,
        flash_decode,
    )

    same_output_token_pairs, same_output_summary = (
        same_output_token_pair_analysis(
            paired,
            eager_decode,
            flash_decode,
        )
    )

    print("[5/7] Writing summary tables...")
    backend_df.to_csv(summary_dir / "backend_summary.csv", index=False)
    category_df.to_csv(summary_dir / "category_summary.csv", index=False)
    paired_df.to_csv(
        summary_dir / "paired_request_summary.csv",
        index=False,
    )
    paired.to_csv(summary_dir / "request_pairs.csv", index=False)
    input_band_df.to_csv(
        summary_dir / "input_band_summary.csv",
        index=False,
    )
    tail_df.to_csv(summary_dir / "token_tail_summary.csv", index=False)
    sequence_df.to_csv(
        summary_dir / "token_sequence_band_summary.csv",
        index=False,
    )
    position_df.to_csv(
        summary_dir / "token_position_band_summary.csv",
        index=False,
    )
    boundary_df.to_csv(
        summary_dir / "boundary_512_summary.csv",
        index=False,
    )
    same_output_token_pairs.to_csv(
        summary_dir / "same_output_paired_token_latency.csv",
        index=False,
    )
    same_output_summary.to_csv(
        summary_dir / "same_output_token_pair_summary.csv",
        index=False,
    )

    print("[6/7] Generating figures...")
    save_ttft_by_category(category_df, figures_dir)
    save_ttft_speedup_by_input_band(input_band_df, figures_dir)
    save_decode_tpot_ecdf(eager_decode, flash_decode, figures_dir)
    save_decode_tpot_by_sequence_band(sequence_df, figures_dir)
    save_peak_allocated_memory_by_category(category_df, figures_dir)

    print("[7/7] Writing machine-readable + Markdown findings...")
    result = build_results_json(
        eager,
        flash,
        paired,
        backend_df,
        category_df,
        input_band_df,
        tail_df,
        same_output_summary,
        integrity,
        eager_meta,
        flash_meta,
    )

    (summary_dir / "r1_results_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    write_findings_markdown(
        result,
        input_band_df,
        tail_df,
        summary_dir / "r1_key_findings.md",
    )

    print("\n" + "=" * 88)
    print("R1-A COMPLETE")
    print("=" * 88)

    all_pairs = paired_df[paired_df["workload_category"] == "ALL"].iloc[0]
    long_band = input_band_df[
        input_band_df["input_band"] == "3073-4096"
    ]

    eager_tail = tail_df.set_index("backend").loc["eager"]
    flash_tail = tail_df.set_index("backend").loc["flash"]

    print(f"Requests/backend:               {len(eager)}")
    print(
        f"Exact output-hash matches:      "
        f"{int(paired['same_output_hash'].sum())}/{len(paired)}"
    )
    print(
        f"Median paired request TTFT:     "
        f"{all_pairs['median_request_ttft_speedup']:.3f}x"
    )
    print(
        f"Median paired TPOT speedup:     "
        f"{all_pairs['median_tpot_speedup']:.3f}x"
    )

    if not long_band.empty:
        print(
            f"3K-4K paired TTFT speedup:      "
            f"{long_band.iloc[0]['paired_request_ttft_speedup_p50']:.3f}x"
        )

    print(
        f"Decode events:                  "
        f"{int(eager_tail['decode_token_events'] + flash_tail['decode_token_events']):,}"
    )
    print(
        f"Token P50 GPU latency:          "
        f"{eager_tail['gpu_tpot_p50_ms']:.2f} -> "
        f"{flash_tail['gpu_tpot_p50_ms']:.2f} ms"
    )
    print(
        f"Token P99 GPU latency:          "
        f"{eager_tail['gpu_tpot_p99_ms']:.2f} -> "
        f"{flash_tail['gpu_tpot_p99_ms']:.2f} ms"
    )
    print(
        f"Same-output paired token gain:  "
        f"{same_output_summary.iloc[0]['median_gpu_tpot_speedup']:.3f}x"
    )

    print("\nSaved summary tables to:")
    print(f"  {summary_dir}")
    print("Saved figures to:")
    print(f"  {figures_dir}")
    print()
    print(
        "R1 is now reproducibly summarized. Keep the raw Full files unchanged "
        "and move on to S1 only after committing the analysis script/results."
    )


if __name__ == "__main__":
    main()
