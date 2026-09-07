#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8002
CONCURRENCY=8

cd /workspace/llm-serving-systems-lab
source /root/.venv-s5/bin/activate
export HF_HOME=/workspace/hf-cache

FORMAL_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="results/s5/raw/kv_pressure_formal/${FORMAL_ID}"

mkdir -p "$ROOT"

cat > "$ROOT/protocol.txt" <<EOF
S5-B KV Pressure Formal
model=${MODEL}
GPU=A100 80GB PCIe
input_tokens=32640
output_tokens=128
concurrency=8
max_model_len=32768
max_num_batched_tokens=2048
max_num_seqs=256
scheduler_reserve_full_isl=true
prefix_caching=false
chunked_prefill=true
warmup_batches=1
measured_batches=5
EOF

declare -a LABELS=(
  "baseline_66p54GiB"
  "kv_14GiB"
  "kv_7p78GiB"
  "kv_7GiB"
  "kv_6p75GiB"
)

declare -a BYTES=(
  "71442297140"
  "15032385536"
  "8351580160"
  "7516192768"
  "7247757312"
)

SERVER_PID=""

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping vLLM server PID ${SERVER_PID}..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    sleep 5
  fi
  SERVER_PID=""
}

trap cleanup_server EXIT INT TERM

if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "ERROR: port ${PORT} already has a running server."
  echo "Stop the existing vLLM server first."
  exit 1
fi

for i in "${!LABELS[@]}"; do
  LABEL="${LABELS[$i]}"
  KV_BYTES="${BYTES[$i]}"
  CFG_DIR="${ROOT}/${LABEL}"

  mkdir -p "$CFG_DIR"

  echo
  echo "================================================"
  echo "Formal config: ${LABEL}"
  echo "KV bytes: ${KV_BYTES}"
  echo "================================================"

  cat > "${CFG_DIR}/server_config.txt" <<EOF
model=${MODEL}
kv_cache_memory_bytes=${KV_BYTES}
max_model_len=32768
max_num_batched_tokens=2048
max_num_seqs=256
scheduler_reserve_full_isl=true
prefix_caching=false
chunked_prefill=true
EOF

  vllm serve "$MODEL" \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 32768 \
    --kv-cache-memory "$KV_BYTES" \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 256 \
    --scheduler-reserve-full-isl \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    > "${CFG_DIR}/server.log" 2>&1 &

  SERVER_PID=$!

  echo "Waiting for server..."

  READY=0

  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      READY=1
      break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server exited unexpectedly."
      tail -100 "${CFG_DIR}/server.log"
      exit 1
    fi

    sleep 1
  done

  if [[ "$READY" != "1" ]]; then
    echo "Server startup timed out."
    tail -100 "${CFG_DIR}/server.log"
    exit 1
  fi

  echo "Server ready."

  curl -s "http://127.0.0.1:${PORT}/metrics" \
    | grep 'vllm:cache_config_info' \
    > "${CFG_DIR}/cache_config_info.txt"

  cat "${CFG_DIR}/cache_config_info.txt"

  echo
  echo "--- Warmup batch (excluded) ---"

  python src/s5_kv_pressure_client.py \
    --concurrency "$CONCURRENCY" \
    --output-root "${CFG_DIR}/warmup" \
    --run-label "${LABEL}_warmup" \
    --purpose "S5-B formal warmup; excluded from analysis"

  for REP in 1 2 3 4 5; do
    echo
    echo "--- Measured repeat ${REP}/5 ---"

    python src/s5_kv_pressure_client.py \
      --concurrency "$CONCURRENCY" \
      --output-root "${CFG_DIR}/measured_repeat_${REP}" \
      --run-label "${LABEL}_repeat_${REP}" \
      --purpose "S5-B formal measured repeat"
  done

  cleanup_server

  echo "Completed ${LABEL}"
done

echo
echo "================================================"
echo "S5-B FORMAL COMPLETE"
echo "Results:"
echo "${ROOT}"
echo "================================================"
