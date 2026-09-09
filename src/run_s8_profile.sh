#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${S8_PROJECT_ROOT:-/workspace/llm-serving-systems-lab}"
MODEL="${S8_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${S8_PORT:-8002}"
VLLM_BIN="${S8_VLLM_BIN:-/root/.venv-s6-vllm/bin/vllm}"
CLIENT_PY="${S8_CLIENT_PY:-/root/.venv-s6-sglang/bin/python}"
REPEATS="${S8_PROFILE_REPEATS:-5}"
BACKGROUND_CONCURRENCY="${S8_BACKGROUND_CONCURRENCY:-64}"
WARMUP_SECONDS="${S8_PROFILE_WARMUP_SECONDS:-15}"
RECOVERY_SECONDS="${S8_PROFILE_RECOVERY_SECONDS:-2}"
INPUT_TOKENS="${S8_PROFILE_INPUT_TOKENS:-24576}"
CASES=(control_off_32768 inject_off_32768 inject_on_4096 inject_on_1024)

if [[ -n "${S8_PROFILE_CASES:-}" ]]; then
  read -r -a CASES <<< "$S8_PROFILE_CASES"
fi

cd "$PROJECT"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

for command in nsys curl ss setsid; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "ERROR: required command is missing: $command"
    exit 2
  fi
done

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="results/s8/raw/profile/${RUN_ID}"
mkdir -p "$ROOT"

SERVER_PID=""
SERVER_LOG=""

group_alive() {
  [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null
}

cleanup_server() {
  if [[ -z "$SERVER_PID" ]]; then
    return
  fi
  if group_alive; then
    kill -INT -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 120); do
      group_alive || break
      sleep 1
    done
  fi
  if group_alive; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      group_alive || break
      sleep 1
    done
  fi
  if group_alive; then
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  SERVER_LOG=""
  sleep 3
  if ss -ltnp | grep -q ":${PORT}"; then
    echo "ERROR: port ${PORT} remains occupied after server shutdown."
    exit 1
  fi
}

trap cleanup_server EXIT INT TERM

wait_ready() {
  for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      return
    fi
    if ! group_alive; then
      echo "ERROR: profiled vLLM process exited during startup."
      tail -120 "$SERVER_LOG" || true
      exit 1
    fi
    sleep 1
  done
  echo "ERROR: profiled vLLM startup timed out."
  tail -120 "$SERVER_LOG" || true
  exit 1
}

save_command() {
  local path="$1"
  shift
  printf '%q ' "$@" > "$path"
  printf '\n' >> "$path"
}

start_server() {
  local label="$1"
  local output_dir="$2"
  local chunk_mode="${label%%_*}"
  local token_budget="${label##*_}"
  local chunk_flag="--enable-chunked-prefill"
  if [[ "$chunk_mode" == "off" ]]; then
    chunk_flag="--no-enable-chunked-prefill"
  fi
  mkdir -p "$output_dir/nsys"
  SERVER_LOG="$output_dir/server.log"
  CMD=(
    nsys profile
    --trace=cuda,nvtx,osrt
    --sample=none
    --cpuctxsw=none
    --trace-fork-before-exec=true
    --cuda-graph-trace=node
    --capture-range=cudaProfilerApi
    --capture-range-end=repeat
    --force-overwrite=true
    --output "$output_dir/nsys/profile"
    "$VLLM_BIN" serve "$MODEL"
    --dtype bfloat16
    --host 0.0.0.0
    --port "$PORT"
    --max-model-len 32768
    --gpu-memory-utilization 0.90
    --max-num-seqs 128
    --max-num-batched-tokens "$token_budget"
    --long-prefill-token-threshold 0
    --scheduling-policy fcfs
    --scheduler-reserve-full-isl
    --no-enable-prefix-caching
    "$chunk_flag"
    --profiler-config.profiler cuda
  )
  save_command "$output_dir/server_command.txt" "${CMD[@]}"
  setsid "${CMD[@]}" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_ready
  curl -fsS "http://127.0.0.1:${PORT}/metrics" \
    > "$output_dir/metrics_after_startup.txt"
}

run_client() {
  local kind="$1"
  local label="$2"
  local replicate="$3"
  local output_dir="$4"
  local seed=$((208000 + replicate))
  CMD=(
    "$CLIENT_PY" src/s8_interference_client.py
    --base-url "http://127.0.0.1:${PORT}"
    --model "$MODEL"
    --trial-kind "$kind"
    --config-label "$label"
    --replicate "$replicate"
    --seed "$seed"
    --background-concurrency "$BACKGROUND_CONCURRENCY"
    --background-input-tokens 256
    --background-output-tokens 8192
    --interferer-input-tokens "$INPUT_TOKENS"
    --interferer-output-tokens 16
    --interferer-count 1
    --warmup-seconds "$WARMUP_SECONDS"
    --control-impact-seconds 2
    --recovery-seconds "$RECOVERY_SECONDS"
    --metrics-interval 0.05
    --profile
    --output-dir "$output_dir"
  )
  save_command "${output_dir}.command.txt" "${CMD[@]}"
  "${CMD[@]}"
}

export_nsys() {
  local output_dir="$1"
  local report
  report="$(find "$output_dir/nsys" -maxdepth 1 -type f -name '*.nsys-rep' | sort | head -1)"
  if [[ -z "$report" ]]; then
    echo "ERROR: Nsight report was not produced below $output_dir/nsys"
    exit 1
  fi
  nsys stats --report cuda_gpu_kern_sum --format csv "$report" \
    > "$output_dir/nsys/cuda_gpu_kern_sum.csv"
  nsys stats --report cuda_api_sum --format csv "$report" \
    > "$output_dir/nsys/cuda_api_sum.csv"
  nsys export --type sqlite --force-overwrite=true \
    --output "$output_dir/nsys/profile.sqlite" "$report"
  "$CLIENT_PY" src/s8_nsys_extract.py \
    --sqlite "$output_dir/nsys/profile.sqlite" \
    --output-dir "$output_dir/nsys"
}

if ss -ltnp | grep -q ":${PORT}"; then
  echo "ERROR: port ${PORT} is already occupied."
  exit 1
fi

{
  echo "S8-B Nsight Systems mechanism validation"
  echo "run_id=${RUN_ID}"
  echo "model=${MODEL}"
  echo "background_concurrency=${BACKGROUND_CONCURRENCY}"
  echo "background_shape=256/8192"
  echo "interferer_shape=${INPUT_TOKENS}/16"
  echo "repeats=${REPEATS}"
  echo "cases=${CASES[*]}"
  echo "warmup_seconds=${WARMUP_SECONDS}"
  echo "profile_recovery_seconds=${RECOVERY_SECONDS}"
  echo "busy_interval_gap_us=50"
  echo "prefix_caching=false"
  echo "scheduling_policy=fcfs"
} > "$ROOT/protocol.txt"

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  git rev-parse HEAD
  git status --short
  nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader
  nsys --version
  "$VLLM_BIN" --version
  "$CLIENT_PY" -c \
    'import httpx, torch, transformers; print("httpx", httpx.__version__); print("torch", torch.__version__); print("transformers", transformers.__version__)'
} > "$ROOT/environment.txt" 2>&1

for replicate in $(seq 1 "$REPEATS"); do
  case_count="${#CASES[@]}"
  case_offset=$(((replicate - 1) % case_count))
  for case_step in $(seq 0 $((case_count - 1))); do
    case_index=$(((case_step + case_offset) % case_count))
    case_name="${CASES[$case_index]}"
    kind="${case_name%%_*}"
    label="${case_name#*_}"
    if [[ "$kind" != "control" && "$kind" != "inject" ]]; then
      echo "ERROR: invalid profile case: $case_name"
      exit 2
    fi
    case_dir="$ROOT/block_${replicate}/${case_name}"
    echo "Block ${replicate}/${REPEATS}: ${case_name}"
    start_server "$label" "$case_dir"
    run_client "$kind" "$label" "$replicate" "$case_dir/trial"
    cleanup_server
    export_nsys "$case_dir"
  done
done

"$CLIENT_PY" src/s8_analyze_profile.py \
  --input-root "$ROOT" \
  --output-dir "$ROOT/analysis"

echo "S8-B profiling complete: $ROOT"
