"""Instrument the pinned, unmodified Megatron-LM pretrain_gpt entrypoint.

S10_OUTPUT and S10_MEGATRON_ROOT select artifact and upstream directories.
All remaining CLI flags are passed unchanged to upstream. Timing wraps an entire
optimizer update, synchronizes CUDA and reports the slowest rank. Profiling is a
separate invocation. JSON is flushed every update to preserve interrupted runs.
"""
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

root = Path(os.environ["S10_MEGATRON_ROOT"]).resolve()
sys.path.insert(0, str(root))
import torch
import torch.distributed as dist
import megatron.training.training as training

original_step = training.train_step
out = Path(os.environ["S10_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)
warmup = int(os.environ.get("S10_WARMUP", "20"))
expected = int(os.environ.get("S10_STEPS", "100"))
profile = os.environ.get("S10_PROFILE", "0") == "1"
records = []
profiler = None
metadata = {"complete": False, "argv": sys.argv[1:], "warmup": warmup,
            "expected_steps": expected, "profile": profile,
            "upstream_commit": "23e00ed0963c35382dfe8a5a94fb3cda4d21e133"}


def write_result():
    if dist.is_initialized():
        metadata.update(rank=dist.get_rank(), world_size=dist.get_world_size())
    metadata["records"] = records
    (out / f"rank-{metadata.get('rank', os.environ.get('RANK', 'unknown'))}.json").write_text(
        json.dumps(metadata, indent=2, default=str))


def measured_step(*args, **kwargs):
    global profiler
    index = len(records)
    if index == warmup:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if profile:
            profiler = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            profiler.start()
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = original_step(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = torch.tensor(time.perf_counter() - start, device="cuda", dtype=torch.float64)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    losses = {key: float(value) for key, value in result[0].items()}
    norm = None if result[5] is None else float(result[5])
    valid = all(math.isfinite(v) for v in losses.values()) and (norm is None or math.isfinite(norm))
    valid = torch.tensor(int(valid and not result[1]), device="cuda")
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise RuntimeError("nonfinite loss/gradient or skipped optimizer update")
    records.append({"step": index, "measured": index >= warmup,
                    "seconds": float(elapsed), "loss": losses, "gradient_norm": norm})
    metadata.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    if profiler is not None and index == warmup + 1:
        profiler.stop()
        profiler.export_chrome_trace(str(out / f"trace-rank{dist.get_rank()}.json"))
        profiler = None
    write_result()
    return result


training.train_step = measured_step
try:
    runpy.run_path(str(root / "pretrain_gpt.py"), run_name="__main__")
    metadata["complete"] = len(records) == warmup + expected
    metadata["formal"] = metadata["complete"] and warmup >= 20 and expected >= 100 and not profile
except BaseException as error:
    metadata.update(error_type=type(error).__name__, error=str(error))
    raise
finally:
    write_result()
