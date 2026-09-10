"""Run an explicit S10 case list sequentially, retaining failures and commands."""
import argparse
import datetime
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def command(case):
    world = case["world"]
    prefix = [sys.executable, "-m", "torch.distributed.run", "--standalone",
              f"--nproc_per_node={world}"]
    output = str(Path(case["output"]).resolve())
    warmup = case.get("warmup", 20)
    steps = case.get("steps", 100)
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    environment.update(OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                       CUDA_DEVICE_MAX_CONNECTIONS="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
                       TORCH_EXTENSIONS_DIR="/workspace/torch-extensions")
    if case["backend"] == "zero":
        cmd = prefix + [str(Path(__file__).with_name("s10_zero_train.py")),
            "--output", output, "--stage", str(case["stage"]),
            "--model", case.get("model", "qwen7b"), "--sequence", str(case.get("sequence", 512)),
            "--global-tokens", str(case.get("global_tokens", 8192)),
            "--warmup", str(warmup), "--steps", str(steps)]
        if case.get("profile"):
            cmd.append("--profile")
    else:
        environment.update(S10_OUTPUT=output, S10_WARMUP=str(warmup), S10_STEPS=str(steps),
                           S10_PROFILE=str(int(case.get("profile", False))),
                           S10_MEGATRON_ROOT="/workspace/Megatron-LM")
        smoke = case.get("model") == "smoke"
        sequence = case.get("sequence", 512)
        cmd = prefix + [str(Path(__file__).with_name("s10_megatron_measure.py")),
            "--tensor-model-parallel-size", str(case.get("tp", 1)),
            "--pipeline-model-parallel-size", str(case.get("pp", 1)),
            "--num-layers", str(4 if smoke else 28),
            "--hidden-size", str(512 if smoke else 3584),
            "--ffn-hidden-size", str(1536 if smoke else 18944),
            "--num-attention-heads", str(8 if smoke else 28),
            "--group-query-attention", "--num-query-groups", str(4),
            "--seq-length", str(sequence), "--max-position-embeddings", "4096",
            "--micro-batch-size", "1", "--global-batch-size", str(case.get("global_tokens",8192)//sequence),
            "--train-iters", str(warmup+steps), "--lr", "0.0001", "--min-lr", "0.0001",
            "--lr-decay-style", "constant", "--weight-decay", "0.01",
            "--adam-beta1", "0.9", "--adam-beta2", "0.95", "--adam-eps", "1e-8",
            "--clip-grad", "1.0", "--bf16", "--seed", "2026",
            "--mock-data", "--tokenizer-type", "NullTokenizer",
            "--vocab-size", str(4096 if smoke else 152064),
            "--transformer-impl", "local", "--normalization", "RMSNorm",
            "--norm-epsilon", "1e-6", "--swiglu", "--position-embedding-type", "rope",
            "--untie-embeddings-and-output-weights", "--disable-bias-linear",
            "--attention-dropout", "0", "--hidden-dropout", "0",
            "--no-masked-softmax-fusion", "--no-bias-swiglu-fusion",
            "--no-bias-dropout-fusion", "--no-rope-fusion", "--no-persist-layer-norm",
            "--no-gradient-accumulation-fusion", "--num-workers", "0",
            "--eval-iters", "0", "--eval-interval", "100000", "--log-interval", "10",
            "--no-save-optim", "--no-save-rng", "--distributed-timeout-minutes", "5"]
    return cmd, environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.matrix.read_text())
    for case in cases:
        output = Path(case["output"])
        output.mkdir(parents=True, exist_ok=False)
        cmd, env = command(case)
        info = {"case": case, "command": cmd, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (output/"launch.json").write_text(json.dumps(info, indent=2))
        print("START", str(output), flush=True)
        start = time.monotonic()
        with (output/"process.log").open("w") as log:
            try:
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                info["returncode"] = process.wait(timeout=case.get("timeout", 1800))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                info["returncode"] = "timeout"
        info["wall_seconds"] = time.monotonic()-start
        (output/"launch.json").write_text(json.dumps(info, indent=2))
        print("END", str(output), info["returncode"], info["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
