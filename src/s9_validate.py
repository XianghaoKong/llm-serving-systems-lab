"""GPU correctness gate; failures exit nonzero and never become speedups."""
import argparse
import importlib
import json
from pathlib import Path
import torch
from s9_kernels import rms_reference, swiglu_reference, dequantize


def check(module, dtype, m, h, strided):
    torch.manual_seed(2026)
    x = torch.randn((m, h * (2 if strided else 1)), device="cuda", dtype=dtype)
    x = x[:, ::2] if strided else x
    w = torch.randn(h, device="cuda", dtype=dtype)
    u = torch.randn_like(x)
    dy = torch.randn_like(x)
    tol = dict(atol=0.03 if dtype == torch.bfloat16 else 0.004,
               rtol=0.02 if dtype == torch.bfloat16 else 0.003)
    errors = {}
    for name, fn, ref, second in (("rms", module.rms_norm, rms_reference, w),
                                   ("swiglu", module.swiglu, swiglu_reference, u)):
        a, b = x.detach().requires_grad_(), second.detach().requires_grad_()
        aa, bb = x.detach().requires_grad_(), second.detach().requires_grad_()
        y, yr = fn(a, b), ref(aa, bb)
        grads = torch.autograd.grad(y, (a, b), dy)
        refs = torch.autograd.grad(yr, (aa, bb), dy)
        for label, got, expected in zip(("output", "dx", "dsecond"), (y, *grads), (yr, *refs)):
            torch.testing.assert_close(got, expected, **tol)
            errors[name + "_" + label] = (got.float()-expected.float()).abs().max().item() if got.numel() else 0
        torch.testing.assert_close(fn(a, b), y, atol=0, rtol=0)
    return dict(rows=m, width=h, dtype=str(dtype), strided=strided, errors=errors, passed=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("triton", "tilelang"), default="triton")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    module = importlib.import_module("s9_kernels" if args.backend == "triton" else "s9_tilelang")
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for m, h, strided in ((0,1536,False), (1,1536,False), (7,1537,True),
                              (32,3584,False), (128,5120,False), (17,8960,True)):
            rows.append(check(module, dtype, m, h, strided))
            print(args.backend, rows[-1], flush=True)
        for m, n, k in ((1,128,128), (17,193,256), (32,256,512)):
            a = torch.randn(m,k,device="cuda",dtype=dtype)
            q = torch.randint(0,256,(n,k//2),device="cuda",dtype=torch.uint8)
            s = torch.rand(n,k//128,device="cuda",dtype=dtype)*0.05
            z = torch.randint(0,16,(n,k//128),device="cuda",dtype=torch.uint8)
            ref = a.float() @ dequantize(q,s,z).to(dtype).float().T
            got = module.w4a16(a,q,s,z)
            torch.testing.assert_close(got.float(),ref,atol=0.04 if dtype==torch.bfloat16 else 0.006,rtol=0.02)
            rows.append(dict(op="w4a16",m=m,n=n,k=k,dtype=str(dtype),passed=True,
                             max_error=(got.float()-ref).abs().max().item()))
        for invalid in (torch.empty(1,0,device="cuda",dtype=dtype), torch.empty(3,device="cuda",dtype=dtype)):
            try:
                module.rms_norm(invalid,torch.ones(1,device="cuda",dtype=dtype))
            except ValueError:
                pass
            else:
                raise AssertionError("invalid shape accepted")
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(backend=args.backend,complete=True,cases=rows),indent=2))
    print("PASS",args.backend,flush=True)


if __name__ == "__main__":
    main()
