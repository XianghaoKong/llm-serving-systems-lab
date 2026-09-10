"""One warmed NVTX range for external Nsight Compute; never a timing result."""
import argparse
import torch
from s9_benchmark import get_impl
import s9_kernels as custom


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--op",choices=("rms","swiglu","w4a16"),required=True)
    ap.add_argument("--backend",default="triton")
    ap.add_argument("--phase",choices=("forward","backward"),default="forward")
    ap.add_argument("--rows",type=int,required=True);ap.add_argument("--width",type=int,required=True)
    args=ap.parse_args();torch.manual_seed(2026)
    x=torch.randn(args.rows,args.width,device="cuda",dtype=torch.bfloat16,requires_grad=args.op!="w4a16")
    if args.op=="w4a16":
        q=torch.randint(256,(args.width,args.width//2),device="cuda",dtype=torch.uint8)
        s=torch.rand(args.width,args.width//128,device="cuda",dtype=x.dtype)*.05
        z=torch.full(s.shape,8,device="cuda",dtype=torch.uint8)
        if args.backend=="cublas_dense":
            dense=custom.dequantize(q,s,z).to(x.dtype)
            fn=lambda:x@dense.T
        else:
            fn=lambda:custom.w4a16(x,q,s,z)
    else:
        w=torch.randn((args.width,) if args.op=="rms" else x.shape,device="cuda",dtype=x.dtype,requires_grad=True)
        dy=torch.randn_like(x);op=get_impl(args.op,args.backend)
        fn=lambda:op(x,w)
    for _ in range(8):
        y=fn()
        if args.phase=="backward": torch.autograd.grad(y,(x,w),dy)
    if args.phase=="backward": y=fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("s9_target")
    if args.phase=="backward": torch.autograd.grad(y,(x,w),dy)
    else: fn()
    torch.cuda.nvtx.range_pop();torch.cuda.synchronize()


if __name__=="__main__": main()
