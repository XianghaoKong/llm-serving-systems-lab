#!/usr/bin/env bash
# Run from repository root in the pinned CUDA environment. GPU stages are serial.
set -euo pipefail
PYTHON="${PYTHON:-python}"
OUTPUT="${OUTPUT:-results/s9/raw}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
mkdir -p "$OUTPUT"
"$PYTHON" src/s9_validate.py --backend triton --output "$OUTPUT/correctness-triton.json"
"$PYTHON" src/s9_validate.py --backend tilelang --output "$OUTPUT/correctness-tilelang.json"
"$PYTHON" src/s9_benchmark.py --output "$OUTPUT/formal-norm-swiglu"
"$PYTHON" src/s9_benchmark.py --output "$OUTPUT/formal-swiglu-wide" --ops swiglu --widths 8960,18944
"$PYTHON" src/s9_benchmark.py --output "$OUTPUT/formal-w4" --ops w4a16 --backends unfused,cublas_dense,triton,tilelang --rows 1,32,128,512 --widths 1536,3584
"$PYTHON" src/s9_profile.py --output "$OUTPUT/profile"
"$PYTHON" src/s9_qwen_replay.py --model "$MODEL" --manifest workloads/final/public_workload_manifest.csv --output "$OUTPUT/qwen-formal"
"$PYTHON" src/s9_analyze.py "$OUTPUT/formal-norm-swiglu/measurements.jsonl" "$OUTPUT/formal-swiglu-wide/measurements.jsonl" "$OUTPUT/formal-w4/measurements.jsonl" --output results/s9/summary
