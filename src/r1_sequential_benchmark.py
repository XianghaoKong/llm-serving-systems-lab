#!/usr/bin/env python3
"""
R1 — Sequential Real-World LLM Serving Baseline (Protocol v2)

Purpose
-------
Run the frozen R0 1,000-request corpus through one long-lived, concurrency-1
inference worker and instrument the full in-process request lifecycle.

This is intentionally NOT a fixed-shape microbenchmark. Each request keeps its
real prompt length and category-specific max_new_tokens cap, and generation
stops at EOS or the cap.

Backends
--------
eager:
    Hugging Face Qwen attention implementation = "eager"

flash:
    Hugging Face Qwen attention implementation = "sdpa"
    AND every model forward is forced through PyTorch
    SDPBackend.FLASH_ATTENTION using sdpa_kernel().

Execution model
---------------
- One model process per backend.
- Model remains resident on GPU for the whole run.
- No torch.cuda.empty_cache() between requests.
- No artificial sleep between requests.
- Deterministically shuffled mixed workload.
- Same execution order for Eager and Flash.
- BF16 model execution.
- Locked Qwen-native sampling: temperature=0.7, top_p=0.8, top_k=20,\n  repetition_penalty=1.1, with deterministic per-request CUDA seeds.\n- Six representative warm-up requests (one/category).
- Pilot = first 10 requests/category in the fixed mixed order (60 total).
- Full = all 1,000 requests.
- --resume skips successfully completed request IDs.

Measured request lifecycle
--------------------------
request_start
  -> apply Qwen chat template
  -> CPU tokenization
  -> H2D
  -> prefill + first-token selection
  -> first token transferred to CPU and incrementally detokenized
  -> autoregressive decode with KV cache
  -> incremental streaming-style detokenization per token
  -> EOS or max_new_tokens

Outputs
-------
results/r1/raw/execution_order.json
results/r1/raw/<mode>/<backend>_requests.csv
results/r1/raw/<mode>/<backend>_token_latency.csv.gz
results/r1/raw/<mode>/<backend>_run_metadata.json

Examples
--------
Pilot:
    python src/r1_sequential_benchmark.py --backend eager --mode pilot
    python src/r1_sequential_benchmark.py --backend flash --mode pilot

Resume:
    python src/r1_sequential_benchmark.py --backend flash --mode pilot --resume

Full:
    python src/r1_sequential_benchmark.py --backend eager --mode full
    python src/r1_sequential_benchmark.py --backend flash --mode full

Important
---------
The script requires:
    workloads/final/SHA256SUMS.txt

Create or refresh it AFTER the final R0 patch:
    sha256sum workloads/final/realistic_requests.json \
      > workloads/final/SHA256SUMS.txt

Then verify:
    sha256sum -c workloads/final/SHA256SUMS.txt
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TextStreamer,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Protocol v2: use the model's native BF16 numerical regime.
DTYPE = torch.bfloat16

# Locked Qwen-native generation policy for reproducibility and realism.
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.1

# Each request gets its own deterministic CUDA RNG seed so results are
# independent of execution order, warmup, resume position, and backend process.
SAMPLING_SEED_BASE = 42026000

EXECUTION_SEED = 2026
PILOT_PER_CATEGORY = 10
WARMUP_MAX_NEW_TOKENS = 32

CATEGORY_ORDER = [
    "short_interactive",
    "knowledge_qa",
    "coding_request",
    "document_qa",
    "long_context_qa",
    "long_output",
]

EXPECTED_COUNTS = {
    "short_interactive": 300,
    "knowledge_qa": 200,
    "coding_request": 150,
    "document_qa": 150,
    "long_context_qa": 100,
    "long_output": 100,
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

WORKLOAD_FILE = PROJECT_ROOT / "workloads" / "final" / "realistic_requests.json"
CHECKSUM_FILE = PROJECT_ROOT / "workloads" / "final" / "SHA256SUMS.txt"

RESULT_ROOT = PROJECT_ROOT / "results" / "r1"
RAW_ROOT = RESULT_ROOT / "raw"
EXECUTION_ORDER_FILE = RAW_ROOT / "execution_order.json"


REQUEST_FIELDS = [
    "request_id",
    "workload_category",
    "backend",
    "status",
    "finish_reason",
    "source",
    "source_id",
    "prompt_hash",
    "frozen_input_tokens",
    "runtime_input_tokens",
    "max_new_tokens",
    "output_tokens",
    "output_hash",
    "ended_with_eos",
    "hit_max_new_tokens",
    "generation_ratio",
    "sampling_seed",
    "template_ms",
    "tokenization_ms",
    "preprocess_ms",
    "h2d_ms",
    "model_ttft_ms",
    "stream_cpu_first_token_ms",
    "request_ttft_ms",
    "decode_total_gpu_ms",
    "decode_wall_ms",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_stream_itl_ms",
    "p50_stream_itl_ms",
    "p95_stream_itl_ms",
    "p99_stream_itl_ms",
    "e2e_model_ms",
    "e2e_request_ms",
    "decode_tokens_per_s",
    "stream_decode_tokens_per_s",
    "stream_total_chars",
    "start_allocated_gib",
    "start_reserved_gib",
    "post_prefill_allocated_gib",
    "post_prefill_reserved_gib",
    "final_allocated_gib",
    "final_reserved_gib",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "error_type",
    "error_message",
]

TOKEN_FIELDS = [
    "request_id",
    "workload_category",
    "backend",
    "token_index",
    "token_phase",
    "gpu_step_ms",
    "stream_cpu_ms",
    "stream_itl_ms",
    "piece_chars",
    "is_eos",
]


# =============================================================================
# Small helpers
# =============================================================================

def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def output_token_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(x) for x in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def gib(n_bytes: int) -> float:
    return n_bytes / (1024.0 ** 3)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None

    vals = sorted(float(v) for v in values)

    if len(vals) == 1:
        return vals[0]

    pos = (len(vals) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return vals[lower]

    weight = pos - lower
    return vals[lower] * (1.0 - weight) + vals[upper] * weight


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(float(v) for v in values) / len(values)


def request_sampling_seed(request_id: str) -> int:
    match = re.search(r"(\d+)$", str(request_id))

    if not match:
        raise RuntimeError(
            f"Could not derive sampling seed from request_id={request_id!r}"
        )

    return SAMPLING_SEED_BASE + int(match.group(1))


SAMPLING_PIPELINE = (
    RepetitionPenaltyLogitsProcessor(REPETITION_PENALTY),
    TemperatureLogitsWarper(TEMPERATURE),
    TopKLogitsWarper(TOP_K),
    TopPLogitsWarper(TOP_P),
)


def sample_next_token(
    *,
    logits: torch.Tensor,
    sequence_ids: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Locked Qwen-native next-token policy:
    repetition penalty -> temperature -> top-k -> top-p -> sample.
    """
    scores = logits

    for processor in SAMPLING_PIPELINE:
        scores = processor(sequence_ids, scores)

    probs = torch.softmax(scores, dim=-1)

    return torch.multinomial(
        probs,
        num_samples=1,
        generator=generator,
    ).squeeze(1)


def safe_rate(numerator: float, milliseconds: Optional[float]) -> Optional[float]:
    if milliseconds is None or milliseconds <= 0:
        return None
    return numerator / (milliseconds / 1000.0)


def as_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def run_command_optional(command: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_commit_optional() -> Optional[str]:
    return run_command_optional(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]
    )


# =============================================================================
# Corpus / checksum / execution order
# =============================================================================

def load_workload() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not WORKLOAD_FILE.exists():
        raise FileNotFoundError(
            f"Frozen workload not found: {WORKLOAD_FILE}"
        )

    with WORKLOAD_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        requests = payload
        metadata: Dict[str, Any] = {}
    elif isinstance(payload, dict):
        requests = payload.get("requests")
        metadata = payload.get("metadata", {})
    else:
        raise RuntimeError("Unsupported workload JSON structure.")

    if not isinstance(requests, list):
        raise RuntimeError("Workload JSON does not contain a requests list.")

    if len(requests) != 1000:
        raise RuntimeError(
            f"Frozen corpus must contain exactly 1000 requests; "
            f"found {len(requests)}."
        )

    ids = [str(row.get("request_id")) for row in requests]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate request_id detected in frozen corpus.")

    counts = {category: 0 for category in CATEGORY_ORDER}
    for row in requests:
        category = str(row.get("workload_category"))
        if category not in counts:
            raise RuntimeError(f"Unknown workload category: {category}")
        counts[category] += 1

    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            "Frozen category counts do not match R0 definition.\n"
            f"Expected: {EXPECTED_COUNTS}\n"
            f"Actual:   {counts}"
        )

    return metadata, requests


def parse_checksum_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            "\nFrozen-corpus checksum file is missing:\n"
            f"  {path}\n\n"
            "Create it AFTER the final R0 patch:\n"
            "  sha256sum workloads/final/realistic_requests.json "
            "> workloads/final/SHA256SUMS.txt\n"
        )

    target_name = WORKLOAD_FILE.name

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        digest = parts[0]
        filename = parts[-1].lstrip("*")

        if Path(filename).name == target_name:
            if len(digest) != 64:
                raise RuntimeError(
                    f"Invalid SHA-256 digest in {path}: {digest}"
                )
            return digest.lower()

    raise RuntimeError(
        f"Could not find {target_name} in {path}."
    )


def verify_frozen_checksum() -> str:
    expected = parse_checksum_file(CHECKSUM_FILE)
    actual = sha256_file(WORKLOAD_FILE)

    if expected != actual:
        raise RuntimeError(
            "\nFROZEN CORPUS CHECKSUM MISMATCH.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}\n\n"
            "If the final R0 replacement patch changed the corpus, refresh "
            "SHA256SUMS.txt only after confirming that patched corpus is the "
            "version you intend to freeze."
        )

    return actual


def create_or_validate_execution_order(
    requests: Sequence[Dict[str, Any]],
    corpus_sha256: str,
) -> Dict[str, Any]:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    request_by_id = {
        str(row["request_id"]): row
        for row in requests
    }

    all_ids = list(request_by_id)
    rng = random.Random(EXECUTION_SEED)
    rng.shuffle(all_ids)

    pilot_counts = {category: 0 for category in CATEGORY_ORDER}
    pilot_ids: List[str] = []

    for request_id in all_ids:
        category = str(request_by_id[request_id]["workload_category"])

        if pilot_counts[category] < PILOT_PER_CATEGORY:
            pilot_ids.append(request_id)
            pilot_counts[category] += 1

        if all(
            pilot_counts[category] == PILOT_PER_CATEGORY
            for category in CATEGORY_ORDER
        ):
            break

    if len(pilot_ids) != PILOT_PER_CATEGORY * len(CATEGORY_ORDER):
        raise RuntimeError(
            f"Pilot selection produced {len(pilot_ids)} requests; expected 60."
        )

    desired = {
        "version": 1,
        "execution_seed": EXECUTION_SEED,
        "corpus_sha256": corpus_sha256,
        "full_request_ids": all_ids,
        "pilot_request_ids": pilot_ids,
        "pilot_counts": pilot_counts,
    }

    if EXECUTION_ORDER_FILE.exists():
        existing = json.loads(
            EXECUTION_ORDER_FILE.read_text(encoding="utf-8")
        )

        checks = [
            existing.get("execution_seed") == EXECUTION_SEED,
            existing.get("corpus_sha256") == corpus_sha256,
            existing.get("full_request_ids") == all_ids,
            existing.get("pilot_request_ids") == pilot_ids,
        ]

        if not all(checks):
            raise RuntimeError(
                "\nExisting R1 execution_order.json does not match the "
                "current frozen corpus/seed. Do not silently regenerate it.\n"
                f"Inspect or remove: {EXECUTION_ORDER_FILE}"
            )

        return existing

    EXECUTION_ORDER_FILE.write_text(
        json.dumps(desired, indent=2),
        encoding="utf-8",
    )
    return desired


# =============================================================================
# Tokenization
# =============================================================================

def render_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_rendered(tokenizer, rendered: str):
    return tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
    )


def preflight_validate_runtime_token_counts(
    tokenizer,
    selected_requests: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    mismatches: List[Dict[str, Any]] = []
    max_input = 0

    print(
        f"\nPreflight: validating Qwen chat-template token counts "
        f"for {len(selected_requests)} requests..."
    )

    start = time.perf_counter()

    for i, row in enumerate(selected_requests, start=1):
        prompt = str(row["prompt"])
        rendered = render_chat(tokenizer, prompt)
        encoded = tokenize_rendered(tokenizer, rendered)
        runtime_count = int(encoded["input_ids"].shape[-1])
        frozen_count = int(row["input_tokens"])
        max_input = max(max_input, runtime_count)

        if runtime_count != frozen_count:
            mismatches.append({
                "request_id": row["request_id"],
                "category": row["workload_category"],
                "frozen": frozen_count,
                "runtime": runtime_count,
            })

        if i % 100 == 0 or i == len(selected_requests):
            print(
                f"  validated {i}/{len(selected_requests)}",
                end="\r",
                flush=True,
            )

    print()

    elapsed = time.perf_counter() - start

    if mismatches:
        preview = "\n".join(
            f"  {m['request_id']} {m['category']}: "
            f"frozen={m['frozen']} runtime={m['runtime']}"
            for m in mismatches[:10]
        )

        raise RuntimeError(
            "\nRuntime tokenizer no longer reproduces frozen R0 token counts.\n"
            f"Mismatches: {len(mismatches)}\n"
            f"{preview}\n\n"
            "Do not benchmark until tokenizer/model-version drift is resolved."
        )

    print(
        f"Preflight PASS: exact token-count match for all "
        f"{len(selected_requests)} requests "
        f"({elapsed:.2f}s, max input={max_input})."
    )

    return {
        "validated_requests": len(selected_requests),
        "mismatch_count": 0,
        "max_runtime_input_tokens": max_input,
        "elapsed_seconds": elapsed,
    }


# =============================================================================
# Attention backend
# =============================================================================

def attention_context(backend: str):
    if backend == "eager":
        return contextlib.nullcontext()

    if backend != "flash":
        raise ValueError(f"Unsupported backend: {backend}")

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:
        raise RuntimeError(
            "This PyTorch build does not expose torch.nn.attention.sdpa_kernel."
        ) from exc

    return sdpa_kernel(SDPBackend.FLASH_ATTENTION)


def model_attn_implementation(model) -> Optional[str]:
    value = getattr(model.config, "_attn_implementation", None)
    if value is None:
        value = getattr(model.config, "attn_implementation", None)
    return None if value is None else str(value)


def load_model_and_tokenizer(backend: str):
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    attn_implementation = "eager" if backend == "eager" else "sdpa"

    print(
        f"Loading model {MODEL_ID} in BF16 "
        f"(attn_implementation={attn_implementation})..."
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        attn_implementation=attn_implementation,
    )

    model.to("cuda")
    model.eval()

    return tokenizer, model


# =============================================================================
# Streaming-style detokenization
# =============================================================================

class SilentCountingStreamer(TextStreamer):
    """
    Reuse Hugging Face TextStreamer's incremental decoding behavior, but do not
    print model text or save it to disk.

    This measures the CPU-side work that a streaming path has to do after each
    GPU token is ready, while preserving privacy/repository cleanliness.
    """

    def __init__(self, tokenizer):
        super().__init__(
            tokenizer,
            skip_prompt=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.last_piece_chars = 0
        self.total_chars = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.last_piece_chars = len(text)
        self.total_chars += len(text)

    def put_token_id(self, token_id: int) -> int:
        self.last_piece_chars = 0
        token_tensor = torch.tensor([token_id], dtype=torch.long)
        super().put(token_tensor)
        return self.last_piece_chars

    def finish(self) -> int:
        before = self.total_chars
        super().end()
        return self.total_chars - before


# =============================================================================
# Request execution
# =============================================================================

def eos_token_ids(tokenizer, model) -> set:
    values = set()

    candidates = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    ]

    for value in candidates:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.update(int(x) for x in value)
        else:
            values.add(int(value))

    if not values:
        raise RuntimeError("Could not determine EOS token ID(s).")

    return values


def request_template(
    row: Dict[str, Any],
    backend: str,
) -> Dict[str, Any]:
    result = {field: "" for field in REQUEST_FIELDS}

    result.update({
        "request_id": row["request_id"],
        "workload_category": row["workload_category"],
        "backend": backend,
        "status": "error",
        "finish_reason": "error",
        "source": row.get("source", ""),
        "source_id": row.get("source_id", ""),
        "prompt_hash": row.get("prompt_hash", ""),
        "frozen_input_tokens": row.get("input_tokens", ""),
        "max_new_tokens": row.get("max_new_tokens", ""),
        "sampling_seed": request_sampling_seed(str(row["request_id"])),
        "ended_with_eos": 0,
        "hit_max_new_tokens": 0,
    })

    return result


@torch.inference_mode()
def run_one_request(
    *,
    model,
    tokenizer,
    row: Dict[str, Any],
    backend: str,
    eos_ids: set,
    max_new_tokens_override: Optional[int] = None,
    capture_token_trace: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    request_result = request_template(row, backend)
    token_rows: List[Dict[str, Any]] = []

    request_start = time.perf_counter()

    try:
        # ------------------------------------------------------------------
        # Request preprocessing
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        rendered = render_chat(tokenizer, str(row["prompt"]))
        t1 = time.perf_counter()

        encoded_cpu = tokenize_rendered(tokenizer, rendered)
        t2 = time.perf_counter()

        template_ms = (t1 - t0) * 1000.0
        tokenization_ms = (t2 - t1) * 1000.0
        preprocess_ms = (t2 - t0) * 1000.0

        runtime_input_tokens = int(
            encoded_cpu["input_ids"].shape[-1]
        )

        frozen_input_tokens = int(row["input_tokens"])

        if runtime_input_tokens != frozen_input_tokens:
            raise RuntimeError(
                f"Runtime input token mismatch: frozen={frozen_input_tokens}, "
                f"runtime={runtime_input_tokens}"
            )

        max_new_tokens = int(
            max_new_tokens_override
            if max_new_tokens_override is not None
            else row["max_new_tokens"]
        )

        if max_new_tokens <= 0:
            raise RuntimeError("max_new_tokens must be positive.")

        sampling_seed = request_sampling_seed(str(row["request_id"]))
        sampling_generator = torch.Generator(device="cuda")
        sampling_generator.manual_seed(sampling_seed)

        # ------------------------------------------------------------------
        # Start-of-request GPU state
        # ------------------------------------------------------------------
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start_allocated = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()

        # H2D timing using CUDA events.
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)

        h2d_start.record()
        input_ids = encoded_cpu["input_ids"].to("cuda")
        attention_mask = encoded_cpu.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones_like(encoded_cpu["input_ids"])

        attention_mask = attention_mask.to("cuda")
        h2d_end.record()

        # ------------------------------------------------------------------
        # Prefill + first token
        # ------------------------------------------------------------------
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)

        prefill_start.record()

        with attention_context(backend):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

            next_token = sample_next_token(
                logits=outputs.logits[:, -1, :],
                sequence_ids=input_ids,
                generator=sampling_generator,
            )

        prefill_end.record()

        # .item() synchronizes the token to CPU; after this all preceding
        # CUDA events are complete and can be read accurately.
        transfer_first_start = time.perf_counter()
        first_token_id = int(next_token.item())

        h2d_ms = float(h2d_start.elapsed_time(h2d_end))
        model_ttft_ms = float(
            prefill_start.elapsed_time(prefill_end)
        )

        post_prefill_allocated = torch.cuda.memory_allocated()
        post_prefill_reserved = torch.cuda.memory_reserved()

        # Streaming-style CPU detokenization for the first generated token.
        streamer = SilentCountingStreamer(tokenizer)
        stream_first_start = time.perf_counter()
        first_piece_chars = streamer.put_token_id(first_token_id)
        first_piece_ready = time.perf_counter()

        stream_cpu_first_token_ms = (
            first_piece_ready - stream_first_start
        ) * 1000.0

        # transfer_first_start is intentionally not a separate metric: the
        # GPU->CPU scalar synchronization is part of request-visible TTFT.
        _ = transfer_first_start

        request_ttft_ms = (
            first_piece_ready - request_start
        ) * 1000.0

        output_ids: List[int] = [first_token_id]

        if capture_token_trace:
            token_rows.append({
                "request_id": row["request_id"],
                "workload_category": row["workload_category"],
                "backend": backend,
                "token_index": 1,
                "token_phase": "prefill_first_token",
                "gpu_step_ms": model_ttft_ms,
                "stream_cpu_ms": stream_cpu_first_token_ms,
                "stream_itl_ms": "",
                "piece_chars": first_piece_chars,
                "is_eos": int(first_token_id in eos_ids),
            })

        past_key_values = outputs.past_key_values
        current_token = next_token.view(1, 1)

        # Transformers repetition penalty is conditioned on prompt +
        # generated tokens, so maintain the full sequence explicitly.
        sequence_ids = torch.cat(
            [input_ids, current_token],
            dim=-1,
        )

        ended_with_eos = first_token_id in eos_ids

        decode_gpu_steps: List[float] = []
        stream_itls: List[float] = []

        previous_piece_ready = first_piece_ready

        # ------------------------------------------------------------------
        # Decode
        # ------------------------------------------------------------------
        while (
            not ended_with_eos
            and len(output_ids) < max_new_tokens
        ):
            # Growing attention_mask is application/runtime work rather than
            # part of the model-forward CUDA event. It is still naturally
            # included in stream ITL.
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=-1,
            )

            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)

            decode_start.record()

            with attention_context(backend):
                outputs = model(
                    input_ids=current_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                next_token = sample_next_token(
                    logits=outputs.logits[:, -1, :],
                    sequence_ids=sequence_ids,
                    generator=sampling_generator,
                )

            decode_end.record()

            token_id = int(next_token.item())
            gpu_step_ms = float(
                decode_start.elapsed_time(decode_end)
            )

            stream_cpu_start = time.perf_counter()
            piece_chars = streamer.put_token_id(token_id)
            piece_ready = time.perf_counter()

            stream_cpu_ms = (
                piece_ready - stream_cpu_start
            ) * 1000.0

            stream_itl_ms = (
                piece_ready - previous_piece_ready
            ) * 1000.0

            previous_piece_ready = piece_ready

            output_ids.append(token_id)
            decode_gpu_steps.append(gpu_step_ms)
            stream_itls.append(stream_itl_ms)

            ended_with_eos = token_id in eos_ids

            if capture_token_trace:
                token_rows.append({
                    "request_id": row["request_id"],
                    "workload_category": row["workload_category"],
                    "backend": backend,
                    "token_index": len(output_ids),
                    "token_phase": "decode",
                    "gpu_step_ms": gpu_step_ms,
                    "stream_cpu_ms": stream_cpu_ms,
                    "stream_itl_ms": stream_itl_ms,
                    "piece_chars": piece_chars,
                    "is_eos": int(ended_with_eos),
                })

            past_key_values = outputs.past_key_values
            current_token = next_token.view(1, 1)
            sequence_ids = torch.cat(
                [sequence_ids, current_token],
                dim=-1,
            )

        # Flush any held tokenizer text fragment.
        final_flush_start = time.perf_counter()
        streamer.finish()
        final_piece_ready = time.perf_counter()
        final_flush_ms = (
            final_piece_ready - final_flush_start
        ) * 1000.0

        # The final stream flush belongs to request completion, but it does
        # not represent an additional generated token.
        _ = final_flush_ms

        e2e_request_ms = (
            final_piece_ready - request_start
        ) * 1000.0

        decode_wall_ms = (
            final_piece_ready - first_piece_ready
        ) * 1000.0

        output_tokens = len(output_ids)
        hit_max = (
            not ended_with_eos
            and output_tokens >= max_new_tokens
        )

        finish_reason = "eos" if ended_with_eos else "length"

        decode_total_gpu_ms = sum(decode_gpu_steps)
        e2e_model_ms = model_ttft_ms + decode_total_gpu_ms

        final_allocated = torch.cuda.memory_allocated()
        final_reserved = torch.cuda.memory_reserved()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        decode_token_count = max(0, output_tokens - 1)

        request_result.update({
            "status": "ok",
            "finish_reason": finish_reason,
            "runtime_input_tokens": runtime_input_tokens,
            "max_new_tokens": max_new_tokens,
            "output_tokens": output_tokens,
            "output_hash": output_token_hash(output_ids),
            "ended_with_eos": int(ended_with_eos),
            "hit_max_new_tokens": int(hit_max),
            "generation_ratio": output_tokens / max_new_tokens,
            "sampling_seed": sampling_seed,
            "template_ms": template_ms,
            "tokenization_ms": tokenization_ms,
            "preprocess_ms": preprocess_ms,
            "h2d_ms": h2d_ms,
            "model_ttft_ms": model_ttft_ms,
            "stream_cpu_first_token_ms": stream_cpu_first_token_ms,
            "request_ttft_ms": request_ttft_ms,
            "decode_total_gpu_ms": decode_total_gpu_ms,
            "decode_wall_ms": decode_wall_ms,
            "mean_tpot_ms": mean_or_none(decode_gpu_steps),
            "p50_tpot_ms": percentile(decode_gpu_steps, 0.50),
            "p95_tpot_ms": percentile(decode_gpu_steps, 0.95),
            "p99_tpot_ms": percentile(decode_gpu_steps, 0.99),
            "mean_stream_itl_ms": mean_or_none(stream_itls),
            "p50_stream_itl_ms": percentile(stream_itls, 0.50),
            "p95_stream_itl_ms": percentile(stream_itls, 0.95),
            "p99_stream_itl_ms": percentile(stream_itls, 0.99),
            "e2e_model_ms": e2e_model_ms,
            "e2e_request_ms": e2e_request_ms,
            "decode_tokens_per_s": safe_rate(
                decode_token_count,
                decode_total_gpu_ms,
            ),
            "stream_decode_tokens_per_s": safe_rate(
                decode_token_count,
                decode_wall_ms,
            ),
            "stream_total_chars": streamer.total_chars,
            "start_allocated_gib": gib(start_allocated),
            "start_reserved_gib": gib(start_reserved),
            "post_prefill_allocated_gib": gib(
                post_prefill_allocated
            ),
            "post_prefill_reserved_gib": gib(
                post_prefill_reserved
            ),
            "final_allocated_gib": gib(final_allocated),
            "final_reserved_gib": gib(final_reserved),
            "peak_allocated_gib": gib(peak_allocated),
            "peak_reserved_gib": gib(peak_reserved),
            "error_type": "",
            "error_message": "",
        })

        # Drop request-specific references. We intentionally do NOT call
        # torch.cuda.empty_cache(); the CUDA caching allocator stays warm.
        del (
            encoded_cpu,
            input_ids,
            attention_mask,
            outputs,
            past_key_values,
            current_token,
            next_token,
            sequence_ids,
            sampling_generator,
        )

        return request_result, token_rows

    except Exception as exc:
        request_result.update({
            "status": "error",
            "finish_reason": (
                "oom"
                if isinstance(exc, torch.cuda.OutOfMemoryError)
                else "error"
            ),
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "e2e_request_ms": (
                time.perf_counter() - request_start
            ) * 1000.0,
        })

        return request_result, []


# =============================================================================
# Warmup / backend verification
# =============================================================================

def representative_warmup_requests(
    requests: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []

    for category in CATEGORY_ORDER:
        group = [
            row
            for row in requests
            if row["workload_category"] == category
        ]

        ordered = sorted(
            group,
            key=lambda r: int(r["input_tokens"]),
        )

        chosen.append(ordered[len(ordered) // 2])

    return chosen


def verify_backend_with_real_request(
    *,
    model,
    tokenizer,
    backend: str,
    row: Dict[str, Any],
    eos_ids: set,
) -> Dict[str, Any]:
    impl = model_attn_implementation(model)

    expected_impl = "eager" if backend == "eager" else "sdpa"

    if impl is not None and expected_impl not in impl.lower():
        raise RuntimeError(
            f"Model attention implementation mismatch: "
            f"expected {expected_impl}, got {impl}"
        )

    if backend == "flash":
        available_fn = getattr(
            torch.backends.cuda,
            "is_flash_attention_available",
            None,
        )

        if callable(available_fn):
            if not bool(available_fn()):
                raise RuntimeError(
                    "PyTorch reports Flash Attention is unavailable."
                )

    result, _ = run_one_request(
        model=model,
        tokenizer=tokenizer,
        row=row,
        backend=backend,
        eos_ids=eos_ids,
        max_new_tokens_override=2,
        capture_token_trace=False,
    )

    if result["status"] != "ok":
        raise RuntimeError(
            f"Backend verification request failed: "
            f"{result['error_type']}: {result['error_message']}"
        )

    # For Flash this is a real Qwen forward under a context where ONLY the
    # FLASH_ATTENTION SDPA backend is enabled. If the shape/kernel were not
    # supported, PyTorch would raise instead of silently falling back to math.
    return {
        "backend": backend,
        "model_attn_implementation": impl,
        "verification_request_id": row["request_id"],
        "verification_input_tokens": row["input_tokens"],
        "forced_flash_context": backend == "flash",
        "status": "pass",
    }


def warmup_worker(
    *,
    model,
    tokenizer,
    backend: str,
    warmups: Sequence[Dict[str, Any]],
    eos_ids: set,
) -> None:
    print("\nWarm-up: one representative request per workload category...")

    for i, row in enumerate(warmups, start=1):
        result, _ = run_one_request(
            model=model,
            tokenizer=tokenizer,
            row=row,
            backend=backend,
            eos_ids=eos_ids,
            max_new_tokens_override=min(
                int(row["max_new_tokens"]),
                WARMUP_MAX_NEW_TOKENS,
            ),
            capture_token_trace=False,
        )

        if result["status"] != "ok":
            raise RuntimeError(
                f"Warm-up failed for {row['workload_category']}: "
                f"{result['error_type']}: {result['error_message']}"
            )

        print(
            f"  [{i}/6] {row['workload_category']:20s} "
            f"input={row['input_tokens']:4d} "
            f"generated={result['output_tokens']:3d}"
        )

    torch.cuda.synchronize()
    print("Warm-up PASS.")


# =============================================================================
# Results I/O / resume
# =============================================================================

def mode_paths(mode: str, backend: str) -> Dict[str, Path]:
    mode_dir = RAW_ROOT / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    return {
        "request_csv": mode_dir / f"{backend}_requests.csv",
        "token_csv_gz": mode_dir / f"{backend}_token_latency.csv.gz",
        "metadata_json": mode_dir / f"{backend}_run_metadata.json",
    }


def completed_request_ids(path: Path) -> set:
    if not path.exists():
        return set()

    completed = set()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("status") == "ok":
                completed.add(str(row["request_id"]))

    return completed


def append_request_row(path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=REQUEST_FIELDS,
            extrasaction="ignore",
        )

        if not exists:
            writer.writeheader()

        writer.writerow({
            key: as_csv_value(row.get(key))
            for key in REQUEST_FIELDS
        })

        f.flush()
        os.fsync(f.fileno())


def append_token_rows_gz(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    if not rows:
        return

    exists = path.exists() and path.stat().st_size > 0

    # Appending gzip members is valid; standard gzip readers transparently
    # concatenate them. One completed request is written atomically-ish as a
    # small member, so resume never needs partial token traces.
    with gzip.open(
        path,
        "at",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=TOKEN_FIELDS,
            extrasaction="ignore",
        )

        if not exists:
            writer.writeheader()

        for row in rows:
            writer.writerow({
                key: as_csv_value(row.get(key))
                for key in TOKEN_FIELDS
            })


# =============================================================================
# Run metadata
# =============================================================================

def gpu_metadata() -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(0)

    nvidia_smi = run_command_optional([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ])

    return {
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": (
            f"{props.major}.{props.minor}"
        ),
        "total_memory_gib": gib(props.total_memory),
        "torch_cuda_runtime": torch.version.cuda,
        "nvidia_smi": nvidia_smi,
    }


def environment_metadata() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit_optional(),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="R1 sequential real-world LLM serving baseline"
    )

    parser.add_argument(
        "--backend",
        choices=["eager", "flash"],
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip request IDs already completed successfully.",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R1.")

    if DTYPE is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "R1 protocol v2 requires BF16, but this GPU/PyTorch build "
            "does not report BF16 support."
        )

    print("=" * 88)
    print("R1 — SEQUENTIAL REAL-WORLD LLM SERVING BASELINE")
    print("=" * 88)
    print(f"Project:      {PROJECT_ROOT}")
    print(f"Model:        {MODEL_ID}")
    print(f"Backend:      {args.backend}")
    print(f"Mode:         {args.mode}")
    print(f"Resume:       {args.resume}")
    print(f"Execution seed: {EXECUTION_SEED}")
    print(f"Dtype:        {DTYPE}")
    print(
        "Generation:   Qwen-native locked sampling "
        f"(T={TEMPERATURE}, top_p={TOP_P}, top_k={TOP_K}, "
        f"rep_penalty={REPETITION_PENALTY})"
    )

    # ----------------------------------------------------------------------
    # Frozen corpus verification
    # ----------------------------------------------------------------------
    print("\n[1/7] Verifying frozen R0 corpus checksum...")
    corpus_sha256 = verify_frozen_checksum()
    print(f"  SHA-256 PASS: {corpus_sha256}")

    workload_metadata, all_requests = load_workload()
    request_by_id = {
        str(row["request_id"]): row
        for row in all_requests
    }

    order = create_or_validate_execution_order(
        all_requests,
        corpus_sha256,
    )

    selected_ids = (
        order["pilot_request_ids"]
        if args.mode == "pilot"
        else order["full_request_ids"]
    )

    selected_requests = [
        request_by_id[request_id]
        for request_id in selected_ids
    ]

    print(
        f"  Selected requests: {len(selected_requests)} "
        f"({'10/category' if args.mode == 'pilot' else 'full corpus'})"
    )

    # ----------------------------------------------------------------------
    # Tokenizer and token-count preflight
    # ----------------------------------------------------------------------
    print("\n[2/7] Loading tokenizer and validating runtime token counts...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    preflight = preflight_validate_runtime_token_counts(
        tokenizer,
        selected_requests,
    )

    # ----------------------------------------------------------------------
    # Model load
    # ----------------------------------------------------------------------
    print("\n[3/7] Loading persistent GPU worker...")
    attn_implementation = (
        "eager"
        if args.backend == "eager"
        else "sdpa"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        attn_implementation=attn_implementation,
    )
    model.to("cuda")
    model.eval()

    eos_ids = eos_token_ids(tokenizer, model)

    print(
        f"  Loaded on: {torch.cuda.get_device_name(0)}\n"
        f"  dtype:     {DTYPE}\n"
        f"  model attn implementation: "
        f"{model_attn_implementation(model)}\n"
        f"  EOS IDs:   {sorted(eos_ids)}"
    )

    # ----------------------------------------------------------------------
    # Backend verification
    # ----------------------------------------------------------------------
    print("\n[4/7] Verifying requested attention backend with a real Qwen request...")

    warmups = representative_warmup_requests(all_requests)

    backend_verification = verify_backend_with_real_request(
        model=model,
        tokenizer=tokenizer,
        backend=args.backend,
        row=warmups[0],
        eos_ids=eos_ids,
    )

    print(
        f"  Backend verification PASS "
        f"(model impl={backend_verification['model_attn_implementation']}, "
        f"forced_flash={backend_verification['forced_flash_context']})"
    )

    # No measured request has run yet. Clean up verification temporaries once,
    # before warmup. There is deliberately no empty_cache() after this point.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # Warmup
    # ----------------------------------------------------------------------
    print("\n[5/7] Warming persistent worker...")
    warmup_worker(
        model=model,
        tokenizer=tokenizer,
        backend=args.backend,
        warmups=warmups,
        eos_ids=eos_ids,
    )

    # ----------------------------------------------------------------------
    # Resume / metadata
    # ----------------------------------------------------------------------
    print("\n[6/7] Preparing run outputs...")
    paths = mode_paths(args.mode, args.backend)

    if paths["request_csv"].exists() and not args.resume:
        raise RuntimeError(
            f"\nResult file already exists:\n"
            f"  {paths['request_csv']}\n\n"
            "Use --resume to continue it, or move/delete the existing "
            "pilot/full output intentionally before starting a new run."
        )

    completed = (
        completed_request_ids(paths["request_csv"])
        if args.resume
        else set()
    )

    remaining = [
        row
        for row in selected_requests
        if str(row["request_id"]) not in completed
    ]

    run_metadata: Dict[str, Any] = {
        "r1_protocol_version": 2,
        "started_at_utc": utc_now_iso(),
        "finished_at_utc": None,
        "status": "running",
        "model_id": MODEL_ID,
        "dtype": str(DTYPE),
        "backend": args.backend,
        "mode": args.mode,
        "concurrency": 1,
        "batch_size": 1,
        "sampling": {
            "policy": "qwen_native_locked",
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "per_request_seed_base": SAMPLING_SEED_BASE,
            "seed_is_request_id_derived": True,
        },
        "torch_compile": False,
        "quantization": False,
        "artificial_inter_request_sleep": False,
        "empty_cache_between_requests": False,
        "corpus_file": str(WORKLOAD_FILE),
        "corpus_sha256": corpus_sha256,
        "checksum_file": str(CHECKSUM_FILE),
        "execution_order_file": str(EXECUTION_ORDER_FILE),
        "execution_seed": EXECUTION_SEED,
        "selected_request_count": len(selected_requests),
        "already_completed_on_resume": len(completed),
        "remaining_at_start": len(remaining),
        "preflight": preflight,
        "backend_verification": backend_verification,
        "workload_metadata": workload_metadata,
        "environment": environment_metadata(),
        "gpu": gpu_metadata(),
    }

    paths["metadata_json"].write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    print(f"  Request CSV: {paths['request_csv']}")
    print(f"  Token trace: {paths['token_csv_gz']}")
    print(f"  Remaining:   {len(remaining)}")

    # ----------------------------------------------------------------------
    # Measured run
    # ----------------------------------------------------------------------
    print("\n[7/7] Starting measured requests...")
    run_start = time.perf_counter()

    successful_this_run = 0

    for index, row in enumerate(remaining, start=1):
        absolute_position = selected_ids.index(str(row["request_id"])) + 1

        print(
            f"  [{absolute_position:4d}/{len(selected_requests):4d}] "
            f"{row['request_id']} "
            f"{row['workload_category']:20s} "
            f"in={int(row['input_tokens']):4d} "
            f"cap={int(row['max_new_tokens']):4d}",
            end="",
            flush=True,
        )

        request_result, token_rows = run_one_request(
            model=model,
            tokenizer=tokenizer,
            row=row,
            backend=args.backend,
            eos_ids=eos_ids,
            capture_token_trace=True,
        )

        append_request_row(
            paths["request_csv"],
            request_result,
        )

        if request_result["status"] == "ok":
            append_token_rows_gz(
                paths["token_csv_gz"],
                token_rows,
            )

            successful_this_run += 1

            print(
                f" -> out={int(request_result['output_tokens']):4d} "
                f"TTFT={float(request_result['request_ttft_ms']):8.2f}ms "
                f"E2E={float(request_result['e2e_request_ms'])/1000.0:7.2f}s "
                f"[{request_result['finish_reason']}]"
            )
        else:
            print(
                f" -> ERROR {request_result['error_type']}: "
                f"{request_result['error_message']}"
            )

            run_metadata.update({
                "status": "failed",
                "finished_at_utc": utc_now_iso(),
                "failure_request_id": row["request_id"],
                "failure_error_type": request_result["error_type"],
                "failure_error_message": request_result["error_message"],
                "elapsed_seconds": time.perf_counter() - run_start,
                "successful_this_run": successful_this_run,
            })

            paths["metadata_json"].write_text(
                json.dumps(run_metadata, indent=2),
                encoding="utf-8",
            )

            raise RuntimeError(
                f"R1 stopped after request failure. "
                f"Fix/investigate, then rerun with --resume."
            )

    elapsed = time.perf_counter() - run_start

    final_completed = completed_request_ids(
        paths["request_csv"]
    )

    run_metadata.update({
        "status": "complete",
        "finished_at_utc": utc_now_iso(),
        "elapsed_seconds_this_invocation": elapsed,
        "successful_this_run": successful_this_run,
        "completed_request_count": len(final_completed),
        "expected_request_count": len(selected_requests),
    })

    paths["metadata_json"].write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("R1 RUN COMPLETE")
    print("=" * 88)
    print(f"Backend:            {args.backend}")
    print(f"Mode:               {args.mode}")
    print(f"Completed requests: {len(final_completed)}/{len(selected_requests)}")
    print(f"This invocation:    {elapsed/60.0:.2f} min")
    print(f"Request results:    {paths['request_csv']}")
    print(f"Token trace:        {paths['token_csv_gz']}")
    print(f"Run metadata:       {paths['metadata_json']}")
    print()
    print(
        "For the pilot, run BOTH eager and flash before analyzing or "
        "starting the 1,000-request full runs."
    )


if __name__ == "__main__":
    main()
