"""Experimental bridge from MLIR-produced LLVM bitcode to S9 Triton.

Supports both the scalar oracle prototype and the ordered warp32 reduction.
Unsupported shapes fall back to the existing ATen compatibility path.
"""
import json
import hashlib
import math
from pathlib import Path

import torch
import triton
import triton.language as tl
from s9_qwen_adapter import qwen_rms_compatible


@tl.core.extern
def compiled_mean(pointer, library, _semantic=None):
    library = tl.core._unwrap_if_constexpr(library)
    return tl.core.extern_elementwise('s9_mean', library, [pointer],
        {(tl.uint64,): ('s9_aten_mean', tl.float32)},
        is_pure=True, _semantic=_semantic)


@triton.jit
def _qwen_mlir(X,W,Y,V,H:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr,LIB:tl.constexpr,
               PARALLEL:tl.constexpr=False):
    row = tl.program_id(0)
    c = tl.arange(0,B)
    if PARALLEL:
        variance = compiled_warp_mean((X + row*H).to(tl.uint64),LIB)
    else:
        variance = compiled_mean((X + row*H).to(tl.uint64),LIB)
    x = tl.load(X+row*H+c,c<H,0).to(tl.float32)
    w = tl.load(W+c,c<H,0).to(tl.float32)
    r = tl.rsqrt(variance+EPS)
    normalized = (x*r).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+row*H+c,normalized*w,c<H)
    tl.store(V+row,variance)


@tl.core.extern
def compiled_warp_mean(pointer, library, _semantic=None):
    library = tl.core._unwrap_if_constexpr(library)
    # A full-warp collective, not a pure elementwise scalar function. Every
    # physical lane participates; the final broadcast makes the result uniform.
    return tl.core.extern_elementwise('s9_mean', library, [pointer],
        {(tl.uint64,): ('s9_aten_mean', tl.float32)},
        is_pure=False, _semantic=_semantic)


class MLIRRMSNorm:
    def __init__(self, directory):
        if torch.__version__.split('+')[0] != '2.8.0' or triton.__version__ != '3.4.0':
            raise RuntimeError('this experimental contract requires torch 2.8.0 / Triton 3.4.0')
        if torch.version.hip is not None:
            raise RuntimeError('this experiment uses the CUDA reduction contract')
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory/'libraries.json').read_text())
        for entry in self.manifest.values():
            kind=entry.get('implementation','scalar')
            if kind not in ('scalar','warp32') or entry.get('num_warps',4)!=(1 if kind=='warp32' else 4):
                raise ValueError('invalid compiled reduction launch contract')
            library=self.directory/entry['bitcode']
            if hashlib.sha256(library.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError(f'bitcode checksum mismatch: {library.name}')
        self.hits = 0
        self.fallbacks = 0

    def __call__(self,x,w,eps):
        if torch.is_grad_enabled():
            raise RuntimeError('MLIR adapter is inference-only')
        if x.ndim < 1 or x.shape[-1] == 0 or w.shape != (x.shape[-1],):
            raise ValueError('expected activations with positive width and matching weight')
        if w.device != x.device or not math.isfinite(eps) or eps <= 0:
            raise ValueError('weight device and positive epsilon required')
        if not x.is_cuda or x.dtype not in (torch.bfloat16,torch.float16) or w.dtype != x.dtype:
            raise ValueError('CUDA BF16/FP16 activations and matching weight dtype required')
        if not x.numel():
            return torch.empty_like(x)
        w=w.contiguous()
        shape = (x.numel()//x.shape[-1],x.shape[-1])
        key = f'{shape[0]}x{shape[1]}'
        if key not in self.manifest or x.dtype != torch.bfloat16:
            self.fallbacks += 1
            return qwen_rms_compatible(x,w,eps)
        y,v,kernel = self.run(x,w,eps)
        self.hits += 1
        return y

    def run(self,x,w,eps):
        if torch.is_grad_enabled():
            raise RuntimeError('MLIR adapter is inference-only')
        if (x.ndim<1 or x.shape[-1]!=1536 or not x.numel() or
            not x.is_cuda or w.device!=x.device or w.shape!=(1536,) or
            x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16 or
            not math.isfinite(eps) or eps<=0):
            raise ValueError('compiled path requires nonempty CUDA BF16 width 1536 and positive epsilon')
        x=x.contiguous()
        w=w.contiguous()
        m,h=x.numel()//x.shape[-1],x.shape[-1]
        entry=self.manifest[f'{m}x{h}']
        parallel=entry.get('implementation','scalar')=='warp32'
        if parallel and x.data_ptr()%8:
            x=x.clone()  # Four-BF16 vector loads require an aligned row base.
        y=torch.empty_like(x)
        v=torch.empty((m,1),device=x.device,dtype=torch.float32)
        library=str(self.directory/entry['bitcode'])
        kernel=_qwen_mlir[(m,)](x,w,y,v,h,eps,triton.next_power_of_2(h),
                   library,parallel,num_warps=entry.get('num_warps',4),
                   enable_fp_fusion=False,extern_libs={'s9_mean':library})
        return y,v,kernel

    def install(self,model):
        import types
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        for module in model.modules():
            if isinstance(module,Qwen2RMSNorm):
                if not hasattr(module,'_s9_original_forward'):
                    module._s9_original_forward=module.forward
                def forward(mod,x):
                    return self(x,mod.weight,mod.variance_epsilon)
                module.forward=types.MethodType(forward,module)
