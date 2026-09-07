#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/workspace/llm-serving-systems-lab"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8002

VLLM_BIN="/root/.venv-s6-vllm/bin/vllm"
SGLANG_PY="/root/.venv-s6-sglang/bin/python"
BENCH_PY="/root/.venv-s6-sglang/bin/python"

INPUT_LEN=512
OUTPUT_LEN=128
SEED=2026
STABILIZE_SEED=2025
REPEATS=5

CONCURRENCIES=(1 8 32 64 128)
NUM_PROMPTS=(10 40 160 320 640)

cd "$PROJECT"

export HF_HOME=/workspace/hf-cache
export UV_CACHE_DIR=/root/.cache/uv
export XDG_CACHE_HOME=/root/.cache

FORMAL_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="results/s6/raw/s6a_formal/${FORMAL_ID}"

mkdir -p "$ROOT"

SERVER_PID=""
SERVER_LOG=""

group_alive() {
  if [[ -z "${SERVER_PID}" ]]; then
    return 1
  fi
  kill -0 -- "-${SERVER_PID}" 2>/dev/null
}

cleanup_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi

  echo
  echo "Stopping server process group ${SERVER_PID}..."

  if group_alive; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true

    for _ in $(seq 1 20); do
      if ! group_alive; then
        break
      fi
      sleep 1
    done
  fi

  if group_alive; then
    echo "Server did not exit after SIGINT; sending SIGTERM..."
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true

    for _ in $(seq 1 10); do
      if ! group_alive; then
        break
      fi
      sleep 1
    done
  fi

  if group_alive; then
    echo "Server still alive; sending SIGKILL..."
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    sleep 2
  fi

  wait "${SERVER_PID}" 2>/dev/null || true

  SERVER_PID=""
  SERVER_LOG=""

  sleep 3

  if ss -ltnp | grep -q ":${PORT}"; then
    echo "ERROR: port ${PORT} is still occupied after shutdown."
    ss -ltnp | grep ":${PORT}" || true
    exit 1
  fi
}

trap cleanup_server EXIT INT TERM


wait_ready() {
  echo "Waiting for server on port ${PORT}..."

  for _ in $(seq 1 300); do
    if curl -fsS \
      "http://127.0.0.1:${PORT}/v1/models" \
      >/dev/null 2>&1; then

      echo "Server ready."
      return 0
    fi

    if ! group_alive; then
      echo "ERROR: server exited during startup."

      if [[ -n "${SERVER_LOG}" && -f "${SERVER_LOG}" ]]; then
        tail -120 "${SERVER_LOG}"
      fi

      exit 1
    fi

    sleep 1
  done

  echo "ERROR: server startup timed out."

  if [[ -n "${SERVER_LOG}" && -f "${SERVER_LOG}" ]]; then
    tail -120 "${SERVER_LOG}"
  fi

  exit 1
}


save_command() {
  local file="$1"
  shift

  printf '%q ' "$@" > "$file"
  printf '\n' >> "$file"
}


start_vllm() {
  local engine_dir="$1"

  SERVER_LOG="${engine_dir}/server.log"

  CMD=(
    "$VLLM_BIN"
    serve
    "$MODEL"
    --dtype bfloat16
    --host 0.0.0.0
    --port "$PORT"
    --max-model-len 32768
    --gpu-memory-utilization 0.90
    --no-enable-prefix-caching
    --enable-chunked-prefill
  )

  save_command "${engine_dir}/server_command.txt" "${CMD[@]}"

  echo
  echo "========================================"
  echo "Starting vLLM..."
  echo "========================================"

  setsid "${CMD[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  wait_ready
}


start_sglang() {
  local engine_dir="$1"

  SERVER_LOG="${engine_dir}/server.log"

  CMD=(
    "$SGLANG_PY"
    -m
    sglang.launch_server
    --model-path "$MODEL"
    --dtype bfloat16
    --host 0.0.0.0
    --port "$PORT"
    --context-length 32768
    --disable-radix-cache
    --mem-fraction-static 0.90
  )

  save_command "${engine_dir}/server_command.txt" "${CMD[@]}"

  echo
  echo "========================================"
  echo "Starting SGLang..."
  echo "========================================"

  setsid "${CMD[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  wait_ready
}


run_benchmark() {
  local backend="$1"
  local concurrency="$2"
  local num_prompts="$3"
  local seed="$4"
  local out_dir="$5"

  mkdir -p "$out_dir"

  local expected_input=$((num_prompts * INPUT_LEN))
  local expected_output=$((num_prompts * OUTPUT_LEN))

  CMD=(
    "$BENCH_PY"
    -m
    sglang.benchmark.serving
    --backend "$backend"
    --base-url "http://127.0.0.1:${PORT}"
    --model "$MODEL"
    --tokenizer "$MODEL"
    --dataset-name random-ids
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --random-range-ratio 1.0
    --num-prompts "$num_prompts"
    --request-rate inf
    --max-concurrency "$concurrency"
    --seed "$seed"
    --warmup-requests 1
    --output-file "${out_dir}/result.jsonl"
    --output-details
  )

  save_command "${out_dir}/command.txt" "${CMD[@]}"

  echo
  echo "----------------------------------------"
  echo "backend=${backend}"
  echo "concurrency=${concurrency}"
  echo "num_prompts=${num_prompts}"
  echo "seed=${seed}"
  echo "----------------------------------------"

  set +e
  "${CMD[@]}" 2>&1 | tee "${out_dir}/stdout.txt"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "ERROR: benchmark failed with status ${status}"
    exit "$status"
  fi

  if ! grep -Eq \
    "Successful requests:[[:space:]]+${num_prompts}[[:space:]]*$" \
    "${out_dir}/stdout.txt"; then

    echo "ERROR: successful request count mismatch."
    exit 1
  fi

  if ! grep -Eq \
    "Total input tokens:[[:space:]]+${expected_input}[[:space:]]*$" \
    "${out_dir}/stdout.txt"; then

    echo "ERROR: input token count mismatch."
    exit 1
  fi

  if ! grep -Eq \
    "Total generated tokens:[[:space:]]+${expected_output}[[:space:]]*$" \
    "${out_dir}/stdout.txt"; then

    echo "ERROR: output token count mismatch."
    exit 1
  fi

  echo "Validation PASS."
}


cat > "${ROOT}/protocol.txt" <<EOF
S6-A Formal Serving Engine Comparison

GPU:
  NVIDIA A100 80GB PCIe

Model:
  Qwen/Qwen2.5-1.5B-Instruct

Serving engines:
  vLLM 0.28.0
  SGLang 0.5.19

Benchmark client:
  SGLang 0.5.19 serving benchmark

dtype:
  BF16

max context:
  32768

workload:
  random-ids
  input_tokens_per_request=512
  output_tokens_per_request=128
  random_range_ratio=1.0
  temperature=0.0
  top_p=1.0
  ignore_eos=true
  request_rate=inf

prefix caching:
  disabled on both engines

concurrency:
  1, 8, 32, 64, 128

num_prompts:
  C1=10
  C8=40
  C32=160
  C64=320
  C128=640

formal repeats:
  5 per concurrency per engine

formal seed:
  2026

excluded stabilization:
  C32, 160 requests, seed=2025

memory policy:
  vLLM gpu-memory-utilization=0.90
  SGLang mem-fraction-static=0.90

Important:
  These are engine-native memory-budget controls and are
  not assumed to imply identical KV-cache byte capacity.
EOF


echo "=== GPU ===" > "${ROOT}/environment.txt"

nvidia-smi \
  --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader \
  >> "${ROOT}/environment.txt"

echo >> "${ROOT}/environment.txt"
echo "=== Git ===" >> "${ROOT}/environment.txt"

git rev-parse HEAD >> "${ROOT}/environment.txt"
git status --short >> "${ROOT}/environment.txt"

echo >> "${ROOT}/environment.txt"
echo "=== vLLM ===" >> "${ROOT}/environment.txt"

"$VLLM_BIN" --version \
  >> "${ROOT}/environment.txt" 2>&1 || true

echo >> "${ROOT}/environment.txt"
echo "=== SGLang / Torch ===" >> "${ROOT}/environment.txt"

"$SGLANG_PY" - <<'PY' \
  >> "${ROOT}/environment.txt"
import torch
import sglang

print("SGLang:", sglang.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY


if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is not installed."
  exit 1
fi

if ss -ltnp | grep -q ":${PORT}"; then
  echo "ERROR: port ${PORT} is already occupied."
  ss -ltnp | grep ":${PORT}" || true
  exit 1
fi


for engine in vllm sglang; do

  ENGINE_DIR="${ROOT}/${engine}"
  mkdir -p "$ENGINE_DIR"

  if [[ "$engine" == "vllm" ]]; then
    BACKEND="vllm"
    start_vllm "$ENGINE_DIR"
  else
    BACKEND="sglang-oai"
    start_sglang "$ENGINE_DIR"
  fi

  echo
  echo "========================================"
  echo "${engine}: excluded stabilization run"
  echo "========================================"

  run_benchmark \
    "$BACKEND" \
    32 \
    160 \
    "$STABILIZE_SEED" \
    "${ENGINE_DIR}/stabilization_c32"

  for idx in "${!CONCURRENCIES[@]}"; do

    C="${CONCURRENCIES[$idx]}"
    N="${NUM_PROMPTS[$idx]}"

    for REP in $(seq 1 "$REPEATS"); do

      echo
      echo "========================================"
      echo "${engine} C=${C} repeat ${REP}/${REPEATS}"
      echo "========================================"

      OUT_DIR="${ENGINE_DIR}/c${C}/repeat_${REP}"

      run_benchmark \
        "$BACKEND" \
        "$C" \
        "$N" \
        "$SEED" \
        "$OUT_DIR"

    done
  done

  cleanup_server

  echo
  echo "Completed engine: ${engine}"

done


echo
echo "========================================"
echo "S6-A FORMAL COMPLETE"
echo "Results:"
echo "${ROOT}"
echo "========================================"
