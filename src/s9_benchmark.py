"""S9: rotated CUDA-graph microbenchmarks; compile and correctness outside timing."""
import argparse
import gc
import importlib.metadata
import json
import statistics
import subprocess
import time
from pathlib import Path
import torch
import s9_kernels as custom


def percentile(xs, q):
    a = sorted(xs)
    p = (len(a)-1)*q
    lo = int(p)
    return a[lo] + (a[min(lo+1,len(a)-1)]-a[lo])*(p-lo)


def measure(fn, samples=30, inner=10):
    # Amortize launch overhead; same graph policy for every backend.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(inner):
            out = fn()
    values = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end)*1000/inner)
    del out, graph
    return dict(p50_us=statistics.median(values),p95_us=percentile(values,0.95),samples_us=values)


def get_impl(op, backend):
    ref = custom.rms_reference if op=="rms" else custom.swiglu_reference
    if backend=="eager":
        return ref
    if backend=="compile":
        return torch.compile(ref, fullgraph=True, dynamic=True)
    if backend=="triton":
        return custom.rms_norm if op=="rms" else custom.swiglu
    if backend=="tilelang":
        import s9_tilelang
        return s9_tilelang.rms_norm if op=="rms" else s9_tilelang.swiglu
    if backend=="liger":
        if op=="rms":
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction
            return lambda x,w:LigerRMSNormFunction.apply(x,w,1e-6,0.0,"gemma",False)
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        # Liger backward overwrites both saved inputs. Include preservation
        # copies in this non-mutating operator contract (and disclose the cost).
        return lambda x,w:LigerSiLUMulFunction.apply(x.clone(),w.clone())
    raise ValueError(backend)


def record_case(args, stream, op, m, h, dtype, backends):
    x = torch.randn(m,h,device="cuda",dtype=dtype,requires_grad=True)
    w = torch.randn((h,) if op=="rms" else (m,h),device="cuda",dtype=dtype,requires_grad=True)
    dy = torch.randn_like(x)
    ref = custom.rms_reference if op=="rms" else custom.swiglu_reference
    yr = ref(x,w)
    gr = torch.autograd.grad(yr,(x,w),dy)
    impls = {}
    for backend in backends:
        fn = get_impl(op,backend)
        t0=time.perf_counter()
        y=fn(x,w)
        gs=torch.autograd.grad(y,(x,w),dy)
        torch.cuda.synchronize()
        first=(time.perf_counter()-t0)*1000
        for a,b in zip((y,*gs),(yr,*gr)):
            torch.testing.assert_close(a,b,atol=0.03 if dtype==torch.bfloat16 else 0.004,
                                       rtol=0.02 if dtype==torch.bfloat16 else 0.003)
        impls[backend]=(fn,first,(y.float()-yr.float()).abs().max().item())
    for block in range(args.repeats):
        order=backends[block%len(backends):]+backends[:block%len(backends)]
        for backend in order:
            fn,first,error=impls[backend]
            for phase in ("forward","forward_backward"):
                def call():
                    y=fn(x,w)
                    if phase=="forward_backward":
                        return torch.autograd.grad(y,(x,w),dy)
                    return y
                gc.collect()
                torch.cuda.synchronize()
                base=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                temporary=call()
                torch.cuda.synchronize()
                peak=torch.cuda.max_memory_allocated()-base
                del temporary
                timing=measure(call,args.samples)
                row=dict(op=op,rows=m,width=h,dtype=str(dtype),backend=backend,
                         phase=phase,block=block+1,first_forward_backward_ms=first,
                         incremental_peak_allocated_bytes=peak,output_max_abs_error=error,
                         correctness=True,**timing)
                # Logical tensor bytes, excluding cache effects and recomputation.
                if phase=="forward":
                    row["modeled_min_bytes"]=(2*m*h+h if op=="rms" else 3*m*h)*x.element_size()
                    row["effective_GBps"]=row["modeled_min_bytes"]/timing["p50_us"]/1000
                stream.write(json.dumps(row)+"\n");stream.flush()
                print(op,m,h,str(dtype),backend,phase,round(timing["p50_us"],3),flush=True)


def w4_case(args, stream, m,n,k,dtype,backends):
    a=torch.randn(m,k,device="cuda",dtype=dtype)
    q=torch.randint(0,256,(n,k//2),device="cuda",dtype=torch.uint8)
    s=torch.rand(n,k//128,device="cuda",dtype=dtype)*0.05
    z=torch.randint(0,16,s.shape,device="cuda",dtype=torch.uint8)
    dense=custom.dequantize(q,s,z).to(dtype)
    ref=a.float()@dense.float().T
    import s9_tilelang
    methods={"unfused":lambda:a@custom.dequantize(q,s,z).to(dtype).T,
             "cublas_dense":lambda:a@dense.T,
             "triton":lambda:custom.w4a16(a,q,s,z),
             "tilelang":lambda:s9_tilelang.w4a16(a,q,s,z)}
    first={}
    for name in backends:
        t0=time.perf_counter(); got=methods[name]();torch.cuda.synchronize()
        first[name]=(time.perf_counter()-t0)*1000
        torch.testing.assert_close(got.float(),ref,atol=0.06 if dtype==torch.bfloat16 else 0.01,rtol=0.025)
    for block in range(args.repeats):
        for name in backends[block%len(backends):]+backends[:block%len(backends)]:
            fn=methods[name]
            torch.cuda.synchronize(); base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            got=fn();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-base
            error=(got.float()-ref).abs().max().item();del got
            timing=measure(fn,args.samples)
            row=dict(op="w4a16",rows=m,width=n,k=k,dtype=str(dtype),backend=name,phase="forward",
                block=block+1,first_forward_ms=first[name],incremental_peak_allocated_bytes=peak,
                output_max_abs_error=error,correctness=True,tflops=2*m*n*k/timing["p50_us"]/1e6,**timing)
            stream.write(json.dumps(row)+"\n");stream.flush()
            print("w4a16",m,n,k,str(dtype),name,round(timing["p50_us"],3),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--ops",default="rms,swiglu")
    ap.add_argument("--backends",default="eager,compile,liger,triton,tilelang")
    ap.add_argument("--rows",default="1,8,32,128,512,2048")
    ap.add_argument("--widths",default="1536,3584,5120")
    ap.add_argument("--dtypes",default="float16,bfloat16")
    ap.add_argument("--repeats",type=int,default=5)
    ap.add_argument("--samples",type=int,default=30)
    args=ap.parse_args()
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.cache_size_limit=256
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    env={"args":vars(args),"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
         "cuda":torch.version.cuda,"method":"CUDA graph, 10 calls/replay; event samples; compile excluded",
         "liger_swiglu_note":"Includes two input clones because upstream backward is destructive; non-mutating API comparison."}
    for name in ("triton","tilelang","liger-kernel"):
        try: env[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: env[name]=None
    (output/"environment.json").write_text(json.dumps(env,indent=2))
    path=output/"measurements.jsonl"
    # Construct leaves, gradient seeds and captures on the same non-default
    # stream so autograd cannot introduce a legacy-stream dependency in capture.
    bench_stream=torch.cuda.Stream()
    with torch.cuda.stream(bench_stream), path.open("x") as stream:
        for op in args.ops.split(','):
            for d in args.dtypes.split(','):
                for m in map(int,args.rows.split(',')):
                    for h in map(int,args.widths.split(',')):
                        if op=="w4a16":
                            w4_case(args,stream,m,h,h,getattr(torch,d),args.backends.split(','))
                        else:
                            record_case(args,stream,op,m,h,getattr(torch,d),args.backends.split(','))
    (output/"complete.json").write_text(json.dumps({"complete":True}))


if __name__=="__main__":
    main()
