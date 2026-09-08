#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${S8_PROJECT_ROOT:-/workspace/llm-serving-systems-lab}"
MODEL="${S8_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${S8_PORT:-8002}"
VLLM_BIN="${S8_VLLM_BIN:-/root/.venv-s6-vllm/bin/vllm}"
CLIENT_PY="${S8_CLIENT_PY:-/root/.venv-s6-sglang/bin/python}"
PHASE="${S8_PHASE:-pilot}"
BACKGROUND_CONCURRENCY="${S8_BACKGROUND_CONCURRENCY:-32}"
INTERFERER_COUNT="${S8_INTERFERER_COUNT:-1}"

case "$PHASE" in
  pilot)
    REPEATS=1
    WARMUP_SECONDS=10
    RECOVERY_SECONDS=5
    CONFIGS=(off_32768 on_32768 on_8192 on_2048 on_512)
    INPUT_LENGTHS=(16384)
    ;;
  formal)
    if [[ "${S8_PROTOCOL_FROZEN:-0}" != "1" ]]; then
      echo "ERROR: formal runs require S8_PROTOCOL_FROZEN=1 after pilot review."
      exit 2
    fi
    REPEATS=5
    WARMUP_SECONDS=15
    RECOVERY_SECONDS=15
    CONFIGS=(off_32768 on_32768 on_16384 on_8192 on_4096 on_2048 on_1024 on_512)
    INPUT_LENGTHS=(8192 16384 24576)
    ;;
  *)
    echo "ERROR: S8_PHASE must be pilot or formal."
    exit 2
    ;;
esac

if [[ -n "${S8_CONFIGS:-}" ]]; then
  read -r -a CONFIGS <<< "$S8_CONFIGS"
fi
if [[ -n "${S8_INPUT_LENGTHS:-}" ]]; then
  read -r -a INPUT_LENGTHS <<< "$S8_INPUT_LENGTHS"
fi
if [[ -n "${S8_REPEATS:-}" ]]; then
  REPEATS="$S8_REPEATS"
fi

cd "$PROJECT"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="results/s8/raw/interference_${PHASE}/${RUN_ID}"
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
    for _ in $(seq 1 20); do
      group_alive || break
      sleep 1
    done
  fi
  if group_alive; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
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
      echo "ERROR: vLLM exited during startup."
      tail -120 "$SERVER_LOG" || true
      exit 1
    fi
    sleep 1
  done
  echo "ERROR: vLLM startup timed out."
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

  mkdir -p "$output_dir"
  SERVER_LOG="${output_dir}/server.log"
  CMD=(
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
  )
  save_command "${output_dir}/server_command.txt" "${CMD[@]}"
  setsid "${CMD[@]}" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_ready
  curl -fsS "http://127.0.0.1:${PORT}/metrics" \
    > "${output_dir}/metrics_after_startup.txt"
}

run_client() {
  local kind="$1"
  local label="$2"
  local replicate="$3"
  local input_tokens="$4"
  local output_dir="$5"
  local seed=$((202600 + replicate))
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
    --interferer-input-tokens "$input_tokens"
    --interferer-output-tokens 16
    --interferer-count "$INTERFERER_COUNT"
    --warmup-seconds "$WARMUP_SECONDS"
    --control-impact-seconds 2
    --recovery-seconds "$RECOVERY_SECONDS"
    --metrics-interval 0.05
    --output-dir "$output_dir"
  )
  save_command "${output_dir}.command.txt" "${CMD[@]}"
  "${CMD[@]}"
}

if ss -ltnp | grep -q ":${PORT}"; then
  echo "ERROR: port ${PORT} is already occupied."
  exit 1
fi

{
  echo "S8 prefill/decode interference ${PHASE}"
  echo "run_id=${RUN_ID}"
  echo "model=${MODEL}"
  echo "background_concurrency=${BACKGROUND_CONCURRENCY}"
  echo "background_shape=256/8192"
  echo "interferer_output_tokens=16"
  echo "interferer_count=${INTERFERER_COUNT}"
  echo "repeats=${REPEATS}"
  echo "warmup_seconds=${WARMUP_SECONDS}"
  echo "recovery_seconds=${RECOVERY_SECONDS}"
  echo "max_model_len=32768"
  echo "max_num_seqs=128"
  echo "long_prefill_token_threshold=0"
  echo "prefix_caching=false"
  echo "scheduling_policy=fcfs"
} > "${ROOT}/protocol.txt"

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  git rev-parse HEAD
  git status --short
  nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader
  "$VLLM_BIN" --version
  "$CLIENT_PY" -c \
    'import httpx, torch, transformers; print("httpx", httpx.__version__); print("torch", torch.__version__); print("transformers", transformers.__version__)'
} > "${ROOT}/environment.txt" 2>&1

for replicate in $(seq 1 "$REPEATS"); do
  config_count="${#CONFIGS[@]}"
  config_offset=$(((replicate - 1) % config_count))
  for config_step in $(seq 0 $((config_count - 1))); do
    config_index=$(((config_step + config_offset) % config_count))
    label="${CONFIGS[$config_index]}"
    config_dir="${ROOT}/block_${replicate}/${label}"
    echo "Block ${replicate}/${REPEATS}: ${label}"
    start_server "$label" "$config_dir"

    run_client \
      control "$label" "$replicate" 16384 \
      "${config_dir}/excluded_stabilization"

    run_client \
      control "$label" "$replicate" 16384 \
      "${config_dir}/control"

    length_count="${#INPUT_LENGTHS[@]}"
    length_offset=$(((replicate - 1) % length_count))
    for length_step in $(seq 0 $((length_count - 1))); do
      length_index=$(((length_step + length_offset) % length_count))
      input_tokens="${INPUT_LENGTHS[$length_index]}"
      run_client \
        inject "$label" "$replicate" "$input_tokens" \
        "${config_dir}/inject_${input_tokens}"
    done
    cleanup_server
  done
done

"$CLIENT_PY" src/s8_analyze_interference.py \
  --input-root "$ROOT" \
  --output-dir "${ROOT}/analysis"

echo "S8 ${PHASE} complete: ${ROOT}"
