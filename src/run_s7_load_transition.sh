#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/workspace/llm-serving-systems-lab"
BENCH="/root/.venv-s6-sglang/bin/python"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

cd "$PROJECT"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/s7/raw/load_transition/${RUN_ID}"
mkdir -p "$OUT"

TIMELINE="${OUT}/timeline.csv"
echo "stage,event,utc_iso,unix_ts,concurrency,num_prompts" > "$TIMELINE"

check_url() {
  local name="$1"
  local url="$2"

  if ! curl -fsS "$url" >/dev/null; then
    echo "ERROR: $name unavailable: $url"
    exit 1
  fi
}

mark() {
  local stage="$1"
  local event="$2"
  local c="$3"
  local n="$4"

  local iso ts
  iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ts="$(date -u +%s)"

  echo "${stage},${event},${iso},${ts},${c},${n}" \
    | tee -a "$TIMELINE"
}

idle_stage() {
  local name="$1"
  local seconds="$2"

  mark "$name" start 0 0
  sleep "$seconds"
  mark "$name" end 0 0
}

run_stage() {
  local stage="$1"
  local c="$2"
  local n="$3"

  echo
  echo "========================================"
  echo "$stage: C=$c N=$n"
  echo "========================================"

  mark "$stage" start "$c" "$n"

  "$BENCH" \
    -m sglang.benchmark.serving \
    --backend vllm \
    --base-url http://127.0.0.1:8002 \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --dataset-name random-ids \
    --random-input-len 512 \
    --random-output-len 128 \
    --random-range-ratio 1.0 \
    --num-prompts "$n" \
    --request-rate inf \
    --max-concurrency "$c" \
    --seed 2026 \
    --warmup-requests 0 \
    --output-file "${OUT}/${stage}.jsonl" \
    --output-details \
    2>&1 | tee "${OUT}/${stage}.stdout.txt"

  mark "$stage" end "$c" "$n"
}

echo "Checking telemetry stack..."

check_url "vLLM" \
  "http://127.0.0.1:8002/v1/models"

check_url "Prometheus" \
  "http://127.0.0.1:9090/-/ready"

check_url "DCGM" \
  "http://127.0.0.1:9400/metrics"

echo "All services ready."

idle_stage idle_initial 15

# ~30 s each based on S6-A measured throughput
run_stage c8    8    400
idle_stage idle_after_c8 10

run_stage c32   32   1150
idle_stage idle_after_c32 10

run_stage c64_a 64   1600
idle_stage idle_after_c64 10

run_stage c128  128  1850
idle_stage idle_after_c128 10

run_stage c64_b 64   1600

idle_stage idle_final 15

echo
echo "========================================"
echo "S7 LOAD TRANSITION COMPLETE"
echo "Run ID: $RUN_ID"
echo "Output: $OUT"
echo "========================================"
