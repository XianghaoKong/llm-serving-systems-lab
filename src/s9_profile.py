"""Separate Kineto traces: launch structure and compiler resource metadata.

Profile timings are diagnostic and must not be merged into formal latency data.
Occupancy fields, when present, are estimates, not hardware counter readings.
"""
import argparse
import gzip
import json
from pathlib import Path
import torch
from s9_benchmark import get_impl
import s9_kernels as custom
import s9_tilelang


def capture(fn, path):
    for _ in range(4): fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function("s9_diagnostic_call"):
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    trace=json.loads(path.read_text())
    kernels=[e for e in trace["traceEvents"] if e.get("cat")=="kernel"]
    # Retain the original trace as well; compressed copy is convenient for backup.
    with gzip.open(str(path)+".gz","wt") as f: json.dump(trace,f)
    return dict(kernel_launches=len(kernels),summed_kernel_us=sum(e.get("dur",0) for e in kernels),
                kernels=[dict(name=e["name"],duration_us=e.get("dur"),args=e.get("args",{})) for e in kernels])


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);args=ap.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(2026);torch._dynamo.config.cache_size_limit=256
    result=[]
    for op in ("rms","swiglu"):
        for m in (1,512):
            x=torch.randn(m,3584,device="cuda",dtype=torch.bfloat16,requires_grad=True)
            w=torch.randn((3584,) if op=="rms" else x.shape,device="cuda",dtype=x.dtype,requires_grad=True)
            dy=torch.randn_like(x)
            for backend in ("eager","compile","liger","triton","tilelang"):
                fn=get_impl(op,backend)
                for phase in ("forward","forward_backward"):
                    def call():
                        y=fn(x,w)
                        return torch.autograd.grad(y,(x,w),dy) if phase=="forward_backward" else y
                    tag=f"{op}-{m}-{backend}-{phase}"
                    row=dict(op=op,rows=m,width=3584,backend=backend,phase=phase,
                             **capture(call,output/(tag+".json")))
                    result.append(row);print(tag,row["kernel_launches"],flush=True)
    m,n,k=32,3584,3584
    a=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)
    q=torch.randint(256,(n,k//2),device="cuda",dtype=torch.uint8)
    s=torch.rand(n,k//128,device="cuda",dtype=a.dtype)*.05
    z=torch.full(s.shape,8,device="cuda",dtype=torch.uint8)
    dense=custom.dequantize(q,s,z).to(a.dtype)
    for backend,fn in {"unfused":lambda:a@custom.dequantize(q,s,z).to(a.dtype).T,
                       "cublas_dense":lambda:a@dense.T,
                       "triton":lambda:custom.w4a16(a,q,s,z),
                       "tilelang":lambda:s9_tilelang.w4a16(a,q,s,z)}.items():
        row=dict(op="w4a16",rows=m,width=n,backend=backend,phase="forward",
                 **capture(fn,output/f"w4a16-{backend}.json"))
        result.append(row);print(backend,row["kernel_launches"],flush=True)
    (output/"summary.json").write_text(json.dumps({"formal_timing":False,
        "occupancy_note":"Kineto estimates when available; no Nsight Compute hardware counters collected",
        "cases":result},indent=2))
    (output/"complete.json").write_text(json.dumps({"complete":True,"cases":len(result)}))


if __name__=="__main__": main()
