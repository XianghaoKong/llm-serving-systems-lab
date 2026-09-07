#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/workspace/llm-serving-systems-lab"

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
VLLM="/root/.venv-s6-vllm/bin/vllm"
BENCH="/root/.venv-s6-sglang/bin/python"

PORT=8002
C=64
N=20000
SEGMENTS=5

cd "$PROJECT"

export HF_HOME=/workspace/hf-cache
export UV_CACHE_DIR=/root/.cache/uv
export XDG_CACHE_HOME=/root/.cache

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/s7/raw/soak/${RUN_ID}"

mkdir -p "$OUT"

TIMELINE="${OUT}/timeline.csv"
echo "stage,event,utc_iso,unix_ts" > "$TIMELINE"

SERVER_PID=""
SMI_PID=""

mark() {
    local stage="$1"
    local event="$2"

    local iso
    local ts

    iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ts="$(date -u +%s)"

    echo "${stage},${event},${iso},${ts}" \
        | tee -a "$TIMELINE"
}

cleanup() {
    set +e

    if [[ -n "${SMI_PID}" ]]; then
        kill -INT "${SMI_PID}" 2>/dev/null || true
        wait "${SMI_PID}" 2>/dev/null || true
    fi

    if [[ -n "${SERVER_PID}" ]]; then
        kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
        sleep 5

        if kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
            kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
        fi

        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

check_url() {
    local name="$1"
    local url="$2"

    if ! curl -fsS "$url" >/dev/null; then
        echo "ERROR: ${name} unavailable: ${url}"
        exit 1
    fi
}

wait_vllm() {
    echo "Waiting for vLLM..."

    for _ in $(seq 1 300); do
        if curl -fsS \
            "http://127.0.0.1:${PORT}/v1/models" \
            >/dev/null 2>&1; then

            echo "vLLM ready."
            return
        fi

        if ! kill -0 -- "-${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: vLLM exited during startup."
            tail -100 "${OUT}/server.log"
            exit 1
        fi

        sleep 1
    done

    echo "ERROR: vLLM startup timeout."
    exit 1
}

run_bench() {
    local stage="$1"
    local n="$2"
    local seed="$3"

    mark "$stage" start

    "$BENCH" \
        -m sglang.benchmark.serving \
        --backend vllm \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "$MODEL" \
        --tokenizer "$MODEL" \
        --dataset-name random-ids \
        --random-input-len 512 \
        --random-output-len 128 \
        --random-range-ratio 1.0 \
        --num-prompts "$n" \
        --request-rate inf \
        --max-concurrency "$C" \
        --seed "$seed" \
        --warmup-requests 0 \
        --disable-tqdm \
        --output-file "${OUT}/${stage}.jsonl" \
        2>&1 | tee "${OUT}/${stage}.stdout.txt"

    mark "$stage" end

    local expected_input=$((n * 512))
    local expected_output=$((n * 128))

    grep -Eq \
        "Successful requests:[[:space:]]+${n}[[:space:]]*$" \
        "${OUT}/${stage}.stdout.txt" \
        || {
            echo "ERROR: successful request mismatch"
            exit 1
        }

    grep -Eq \
        "Total input tokens:[[:space:]]+${expected_input}[[:space:]]*$" \
        "${OUT}/${stage}.stdout.txt" \
        || {
            echo "ERROR: input token mismatch"
            exit 1
        }

    grep -Eq \
        "Total generated tokens:[[:space:]]+${expected_output}[[:space:]]*$" \
        "${OUT}/${stage}.stdout.txt" \
        || {
            echo "ERROR: output token mismatch"
            exit 1
        }

    echo "${stage}: VALIDATION PASS"
}

echo "Checking monitoring stack..."

check_url \
    "Prometheus" \
    "http://127.0.0.1:9090/-/ready"

check_url \
    "DCGM exporter" \
    "http://127.0.0.1:9400/metrics"

if ss -ltnp | grep -q ":${PORT}"; then
    echo "ERROR: port ${PORT} already occupied."
    exit 1
fi

{
    echo "Run ID: ${RUN_ID}"
    echo "Model: ${MODEL}"
    echo "Concurrency: ${C}"
    echo "Input tokens/request: 512"
    echo "Output tokens/request: 128"
    echo "Segments: ${SEGMENTS}"
    echo "Requests/segment: ${N}"
    echo
    nvidia-smi \
        --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader
    echo
    "$VLLM" --version
    "$BENCH" - <<'PY'
import sglang, torch
print("SGLang benchmark:", sglang.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
PY
} > "${OUT}/environment.txt" 2>&1

echo
echo "Starting fresh vLLM..."

CMD=(
    "$VLLM"
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

printf '%q ' "${CMD[@]}" \
    > "${OUT}/server_command.txt"
printf '\n' \
    >> "${OUT}/server_command.txt"

setsid "${CMD[@]}" \
    > "${OUT}/server.log" 2>&1 &

SERVER_PID=$!

wait_vllm

echo
echo "Excluded stabilization..."

run_bench stabilization 640 2025

echo
echo "Post-warmup idle baseline..."

mark idle_before_soak start
sleep 20
mark idle_before_soak end

echo
echo "Starting 1 Hz nvidia-smi telemetry..."

nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv,noheader,nounits \
    -lms 1000 \
    > "${OUT}/nvidia_smi_1hz.csv" &

SMI_PID=$!

mark soak start

for SEG in $(seq 1 "$SEGMENTS"); do
    echo
    echo "========================================"
    echo "SOAK SEGMENT ${SEG}/${SEGMENTS}"
    echo "========================================"

    run_bench \
        "segment_${SEG}" \
        "$N" \
        "$((2026 + SEG))"
done

mark soak end

echo
echo "Post-soak idle observation..."

mark idle_after_soak start
sleep 30
mark idle_after_soak end

kill -INT "${SMI_PID}" 2>/dev/null || true
wait "${SMI_PID}" 2>/dev/null || true
SMI_PID=""

echo
echo "========================================"
echo "S7 FORMAL SOAK COMPLETE"
echo "Run ID: ${RUN_ID}"
echo "Output: ${OUT}"
echo "========================================"
