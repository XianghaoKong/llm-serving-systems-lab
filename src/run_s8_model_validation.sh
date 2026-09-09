#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${S8_PROJECT_ROOT:-/workspace/llm-serving-systems-lab}"
MODEL="${S8D_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${S8_PORT:-8002}"
VLLM_BIN="${S8_VLLM_BIN:-/root/.venv-s6-vllm/bin/vllm}"
CLIENT_PY="${S8_CLIENT_PY:-/root/.venv-s6-sglang/bin/python}"
PHASE="${S8D_PHASE:-calibration}"

case "$PHASE" in
  calibration)
    read -r -a CONCURRENCIES <<< "${S8D_CALIBRATION_CONCURRENCIES:-16 32}"
    REPEATS=1
    WARMUP_SECONDS=15
    RECOVERY_SECONDS=2
    ;;
  formal)
    CONCURRENCY="${S8D_BACKGROUND_CONCURRENCY:-32}"
    REPEATS=5
    WARMUP_SECONDS=15
    RECOVERY_SECONDS=15
    CASES=(off_32768 on_4096 on_1024)
    ;;
  *)
    echo "ERROR: S8D_PHASE must be calibration or formal."
    exit 2
    ;;
esac

cd "$PROJECT"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/.cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

for command in curl nvidia-smi setsid ss; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "ERROR: required command is missing: $command"
    exit 2
  }
done

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="results/s8/raw/model_validation_7b_${PHASE}/${RUN_ID}"
mkdir -p "$ROOT"
SERVER_PID=""
SERVER_LOG=""
TELEMETRY_PID=""

group_alive() {
  [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null
}

stop_telemetry() {
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill -INT "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=""
  fi
}

cleanup_server() {
  stop_telemetry
  if [[ -z "$SERVER_PID" ]]; then
    return
  fi
  if group_alive; then
    kill -INT -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
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
  for _ in $(seq 1 600); do
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
  SERVER_LOG="$output_dir/server.log"
  CMD=(
    "$VLLM_BIN" serve "$MODEL"
    --dtype bfloat16
    --host 0.0.0.0
    --port "$PORT"
    --max-model-len 32768
    --gpu-memory-utilization 0.90
    --max-num-seqs 64
    --max-num-batched-tokens "$token_budget"
    --long-prefill-token-threshold 0
    --scheduling-policy fcfs
    --scheduler-reserve-full-isl
    --no-enable-prefix-caching
    "$chunk_flag"
  )
  save_command "$output_dir/server_command.txt" "${CMD[@]}"
  setsid "${CMD[@]}" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_ready
  curl -fsS "http://127.0.0.1:${PORT}/metrics" > "$output_dir/metrics_after_startup.txt"
}

start_telemetry() {
  local output_path="$1"
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used \
    --format=csv,noheader,nounits \
    -l 1 > "$output_path" 2>&1 &
  TELEMETRY_PID=$!
}

run_client() {
  local kind="$1"
  local label="$2"
  local replicate="$3"
  local concurrency="$4"
  local output_dir="$5"
  local control_seconds=2
  if [[ "$PHASE" == "calibration" ]]; then
    control_seconds=4
  fi
  CMD=(
    "$CLIENT_PY" src/s8_interference_client.py
    --base-url "http://127.0.0.1:${PORT}"
    --model "$MODEL"
    --trial-kind "$kind"
    --config-label "$label"
    --replicate "$replicate"
    --seed "$((209000 + replicate))"
    --background-concurrency "$concurrency"
    --background-input-tokens 256
    --background-output-tokens 8192
    --interferer-input-tokens 24576
    --interferer-output-tokens 16
    --interferer-count 1
    --warmup-seconds "$WARMUP_SECONDS"
    --control-impact-seconds "$control_seconds"
    --recovery-seconds "$RECOVERY_SECONDS"
    --metrics-interval 0.05
    --idle-timeout 300
    --interferer-timeout 300
    --output-dir "$output_dir"
  )
  save_command "${output_dir}.command.txt" "${CMD[@]}"
  start_telemetry "${output_dir}.gpu.csv"
  "${CMD[@]}"
  stop_telemetry
}

if ss -ltnp | grep -q ":${PORT}"; then
  echo "ERROR: port ${PORT} is already occupied."
  exit 1
fi

{
  echo "S8-D 7B model-extension ${PHASE}"
  echo "run_id=${RUN_ID}"
  echo "model=${MODEL}"
  echo "background_shape=256/8192"
  echo "injection_shape=24576/16"
  echo "warmup_seconds=${WARMUP_SECONDS}"
  echo "recovery_seconds=${RECOVERY_SECONDS}"
  echo "prefix_caching=false"
  echo "scheduling_policy=fcfs"
} > "$ROOT/protocol.txt"

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  git rev-parse HEAD
  git status --short
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  "$VLLM_BIN" --version
  "$CLIENT_PY" -c 'import httpx, torch, transformers; print("httpx", httpx.__version__); print("torch", torch.__version__); print("transformers", transformers.__version__)'
} > "$ROOT/environment.txt" 2>&1

if [[ "$PHASE" == "calibration" ]]; then
  for concurrency in "${CONCURRENCIES[@]}"; do
    case_dir="$ROOT/c${concurrency}"
    echo "Calibration: C${concurrency}"
    start_server on_4096 "$case_dir"
    run_client control on_4096 1 "$concurrency" "$case_dir/trial"
    cleanup_server
  done
  "$CLIENT_PY" src/s8_select_7b_concurrency.py \
    --input-root "$ROOT" \
    --output-dir "$ROOT/analysis"
else
  for replicate in $(seq 1 "$REPEATS"); do
    case_count="${#CASES[@]}"
    case_offset=$(((replicate - 1) % case_count))
    for case_step in $(seq 0 $((case_count - 1))); do
      case_index=$(((case_step + case_offset) % case_count))
      label="${CASES[$case_index]}"
      case_dir="$ROOT/block_${replicate}/inject_${label}"
      echo "Block ${replicate}/${REPEATS}: inject_${label} at C${CONCURRENCY}"
      start_server "$label" "$case_dir"
      run_client inject "$label" "$replicate" "$CONCURRENCY" "$case_dir/trial"
      cleanup_server
    done
  done
  "$CLIENT_PY" src/s8_analyze_interference.py \
    --input-root "$ROOT" \
    --output-dir "$ROOT/analysis" \
    --baseline-seconds 4
  "$CLIENT_PY" src/s8_validate_model_extension.py \
    --input-root "$ROOT" \
    --analysis-dir "$ROOT/analysis"
fi

echo "S8-D ${PHASE} complete: $ROOT"
