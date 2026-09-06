import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_WORKLOAD = Path("workloads/final/realistic_requests.json")
DEFAULT_OUTPUT_DIR = Path("results/s3/raw/sequential/smoke6")

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
URL = "http://127.0.0.1:8001/v1/chat/completions"

TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.1
DEFAULT_MAX_TOKENS = 512


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def get_first(d: Dict[str, Any], names: List[str], default=None):
    for name in names:
        if name in d and d[name] is not None:
            return d[name]
    return default


def get_request_id(req: Dict[str, Any], idx: int) -> str:
    return str(get_first(req, ["request_id", "id", "sample_id", "uid"], f"request_{idx:04d}"))


def get_category(req: Dict[str, Any]) -> str:
    return str(get_first(req, ["category", "workload_category", "type", "task_type"], "unknown"))


def get_prompt(req: Dict[str, Any]) -> str:
    prompt = get_first(req, ["prompt", "text", "input", "query", "question", "content"])
    if isinstance(prompt, str):
        return prompt

    messages = req.get("messages")
    if isinstance(messages, list):
        parts = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user"))
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
        if parts:
            return "\n".join(parts)

    raise KeyError(f"Could not find prompt text. Available fields: {sorted(req.keys())}")


def get_max_tokens(req: Dict[str, Any]) -> int:
    value = get_first(
        req,
        ["max_output_tokens", "max_tokens", "output_cap", "generation_cap", "target_output_tokens"],
        DEFAULT_MAX_TOKENS,
    )
    try:
        return int(value)
    except Exception:
        return DEFAULT_MAX_TOKENS


def get_seed(req: Dict[str, Any], idx: int) -> int:
    existing = get_first(req, ["seed", "sampling_seed", "rng_seed"])
    if existing is not None:
        return int(existing)
    return 43000000 + idx


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_representative_requests(requests: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    chosen = []
    used_indices = set()
    seen_categories = set()

    for i, req in enumerate(requests):
        category = get_category(req)
        if category not in seen_categories:
            chosen.append(req)
            used_indices.add(i)
            seen_categories.add(category)
        if len(chosen) >= count:
            return chosen

    for i, req in enumerate(requests):
        if i in used_indices:
            continue
        chosen.append(req)
        if len(chosen) >= count:
            break
    return chosen


def run_one(client: httpx.Client, req: Dict[str, Any], index: int) -> Dict[str, Any]:
    request_id = get_request_id(req, index)
    category = get_category(req)
    prompt = get_prompt(req)
    max_tokens = get_max_tokens(req)
    seed = get_seed(req, index)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start_ns = time.perf_counter_ns()
    first_event_ns = None
    first_visible_ns = None
    event_times_ns: List[int] = []
    visible_event_times_ns: List[int] = []
    output_parts: List[str] = []
    usage = {}
    finish_reason = None
    response_status = None
    error = None

    try:
        with client.stream("POST", URL, json=payload, timeout=None) as response:
            response_status = response.status_code
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                now_ns = time.perf_counter_ns()
                choices = obj.get("choices") or []

                if choices:
                    choice = choices[0] or {}
                    delta = choice.get("delta") or {}
                    content = delta.get("content")

                    if first_event_ns is None:
                        first_event_ns = now_ns
                    event_times_ns.append(now_ns)

                    if isinstance(content, str):
                        output_parts.append(content)
                        if content:
                            if first_visible_ns is None:
                                first_visible_ns = now_ns
                            visible_event_times_ns.append(now_ns)

                    if choice.get("finish_reason") is not None:
                        finish_reason = choice.get("finish_reason")

                if obj.get("usage"):
                    usage = obj["usage"]

        end_ns = time.perf_counter_ns()

    except Exception as exc:
        end_ns = time.perf_counter_ns()
        error = f"{type(exc).__name__}: {exc}"

    output_text = "".join(output_parts)
    token_event_itls_ms = [(b - a) / 1e6 for a, b in zip(event_times_ns, event_times_ns[1:])]
    visible_itls_ms = [(b - a) / 1e6 for a, b in zip(visible_event_times_ns, visible_event_times_ns[1:])]

    ttft_event_ms = (first_event_ns - start_ns) / 1e6 if first_event_ns is not None else None
    ttft_visible_ms = (first_visible_ns - start_ns) / 1e6 if first_visible_ns is not None else None
    e2e_ms = (end_ns - start_ns) / 1e6

    return {
        "request_id": request_id,
        "category": category,
        "status": "ok" if error is None else "error",
        "http_status": response_status,
        "error": error,
        "seed": seed,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": max_tokens,
        "prompt_chars": len(prompt),
        "prompt_sha256": sha256_text(prompt),
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": finish_reason,
        "client_first_token_event_ttft_ms": ttft_event_ms,
        "client_first_visible_text_ttft_ms": ttft_visible_ms,
        "client_e2e_ms": e2e_ms,
        "token_event_count": len(event_times_ns),
        "visible_event_count": len(visible_event_times_ns),
        "mean_token_event_itl_ms": statistics.mean(token_event_itls_ms) if token_event_itls_ms else None,
        "p95_token_event_itl_ms": percentile(token_event_itls_ms, 0.95),
        "mean_visible_event_itl_ms": statistics.mean(visible_itls_ms) if visible_itls_ms else None,
        "output_chars": len(output_text),
        "output_sha256": sha256_text(output_text),
        "output_text": output_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    with args.workload.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        requests = obj
    elif isinstance(obj, dict):
        requests = get_first(obj, ["requests", "data", "items", "samples"])
        if not isinstance(requests, list):
            raise ValueError("Workload JSON is a dict but no request list was found.")
    else:
        raise ValueError("Unsupported workload JSON format.")

    selected = choose_representative_requests(requests, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("S3-A vLLM Sequential Validation")
    print("=" * 72)
    print(f"workload: {args.workload}")
    print(f"total workload requests: {len(requests)}")
    print(f"selected: {len(selected)}")
    print(f"server: {URL}")
    print(f"model: {MODEL}")
    print()

    rows = []
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)

    with httpx.Client(http2=False, limits=limits, headers={"Connection": "keep-alive"}) as client:
        for i, req in enumerate(selected, start=1):
            rid = get_request_id(req, i)
            category = get_category(req)
            print(f"[{i}/{len(selected)}] {rid} | {category}")

            row = run_one(client, req, i)
            rows.append(row)

            if row["client_first_token_event_ttft_ms"] is not None:
                print(
                    f"  status={row['status']} finish={row['finish_reason']} "
                    f"tokens={row['output_tokens']} "
                    f"TTFT_event={row['client_first_token_event_ttft_ms']:.2f}ms"
                )
            else:
                print(f"  status={row['status']} no TTFT")

            if row["client_first_visible_text_ttft_ms"] is not None:
                print(
                    f"  TTFT_visible={row['client_first_visible_text_ttft_ms']:.2f}ms "
                    f"E2E={row['client_e2e_ms']:.2f}ms "
                    f"output_chars={row['output_chars']}"
                )

            if row["error"]:
                print(f"  ERROR: {row['error']}")

    csv_path = args.output_dir / "requests.csv"
    json_path = args.output_dir / "requests.json"
    metadata_path = args.output_dir / "metadata.json"

    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    successful = [r for r in rows if r["status"] == "ok"]

    metadata = {
        "phase": "S3-A",
        "mode": "natural_generation_sequential_smoke",
        "count_requested": args.count,
        "count_completed": len(rows),
        "count_successful": len(successful),
        "model": MODEL,
        "endpoint": URL,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "http2": False,
        "client_max_connections": 1,
        "client_keepalive_connections": 1,
        "server_expected_config": {
            "vllm_version": "0.28.0",
            "dtype": "bfloat16",
            "attention_backend": "FlashAttention 2",
            "flashinfer_sampler": False,
            "torch_compile": True,
            "cuda_graphs": True,
            "prefix_caching": True,
            "chunked_prefill": True,
            "gpu_memory_utilization": 0.85,
            "wsl2_pin_memory": True,
        },
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"success: {len(successful)}/{len(rows)}")

    if successful:
        event_ttfts = [r["client_first_token_event_ttft_ms"] for r in successful if r["client_first_token_event_ttft_ms"] is not None]
        visible_ttfts = [r["client_first_visible_text_ttft_ms"] for r in successful if r["client_first_visible_text_ttft_ms"] is not None]
        e2es = [r["client_e2e_ms"] for r in successful]

        print(f"P50 first-token-event TTFT: {percentile(event_ttfts, 0.5):.2f} ms" if event_ttfts else "P50 first-token-event TTFT: N/A")
        print(f"P50 first-visible-text TTFT: {percentile(visible_ttfts, 0.5):.2f} ms" if visible_ttfts else "P50 first-visible-text TTFT: N/A")
        print(f"P50 E2E: {percentile(e2es, 0.5):.2f} ms")

    print()
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"saved: {metadata_path}")


if __name__ == "__main__":
    main()
