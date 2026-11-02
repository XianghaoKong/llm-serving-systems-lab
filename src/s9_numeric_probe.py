"""Reproduce Qwen RMSNorm divergence on identical captured activations.

Diagnostic stores can change optimization. We therefore compare the probe's
output bit-for-bit with the uninstrumented S9 kernel before attributing errors.
No timings from this script are performance measurements.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM
from s9_qwen_adapter import _qwen_rms, set_backend


@triton.jit
def probe_kernel(X, W, V, SQ, SUM, MEAN, EPSV, INV, NORM, ROUND, Y,
                 H: tl.constexpr, E: tl.constexpr, B: tl.constexpr,
                 OVERRIDE: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + row * H + c, c < H, 0).to(tl.float32)
    w = tl.load(W + c, c < H, 0).to(tl.float32)
    sq = x * x
    s = tl.sum(sq, 0)
    v = s / H
    if OVERRIDE:
        v = tl.load(V + row)
    z = v + E
    r = tl.rsqrt(z)
    n = x * r
    q = n.to(tl.bfloat16).to(tl.float32)
    tl.store(SQ + row * H + c, sq, c < H)
    tl.store(SUM + row, s)
    tl.store(MEAN + row, v)
    tl.store(EPSV + row, z)
    tl.store(INV + row, r)
    tl.store(NORM + row * H + c, n, c < H)
    tl.store(ROUND + row * H + c, q, c < H)
    tl.store(Y + row * H + c, q * w, c < H)


def stats(a, b):
    a, b = a.detach().contiguous().cpu(), b.detach().contiguous().cpu()
    bits = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    mismatch = a.view(bits) != b.view(bits)
    af, bf = a.double(), b.double()
    result = dict(elements=a.numel(), different=int(mismatch.sum()),
                  max_abs=float((af-bf).abs().max()),
                  nrmse=float((af-bf).square().mean().sqrt() /
                              bf.square().mean().sqrt().clamp_min(1e-30)),
                  finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
    # Monotonic IEEE ordering; zero signs remain distinguishable in bit counts.
    ai = a.view(bits).to(torch.int64)
    bi = b.view(bits).to(torch.int64)
    width = 16 if bits == torch.int16 else 32
    mask = (1 << width) - 1
    au, bu = ai & mask, bi & mask
    ao = torch.where(ai < 0, mask-au, au+(1 << (width-1)))
    bo = torch.where(bi < 0, mask-bu, bu+(1 << (width-1)))
    result['max_ulp'] = int((ao-bo).abs().max())
    return result


def reference_stages(x, w, eps):
    xf = x.float()
    sq = xf.square()
    v = sq.mean(-1, keepdim=True)
    z = v + eps
    r = torch.rsqrt(z)
    n = xf * r
    q = n.to(x.dtype).float()
    # sum is an additional diagnostic: the production ATen graph calls mean.
    return dict(square=sq, sum=sq.sum(-1,keepdim=True), mean=v, epsilon=z, rsqrt=r, normalized=n,
                rounded=q, output=(q*w.float()).to(x.dtype))


def probe(x, w, eps, variance, override=False):
    m, h = x.shape
    out = {k: torch.empty_like(x, dtype=torch.float32)
           for k in ('square', 'normalized', 'rounded')}
    out.update({k: torch.empty((m,1), device=x.device, dtype=torch.float32)
                for k in ('sum', 'mean', 'epsilon', 'rsqrt')})
    out['output'] = torch.empty_like(x)
    kernel = probe_kernel[(m,)](x,w,variance,out['square'],out['sum'],out['mean'],
        out['epsilon'],out['rsqrt'],out['normalized'],out['rounded'],out['output'],
        h,eps,triton.next_power_of_2(h),override,num_warps=4,enable_fp_fusion=False)
    return out, kernel


@torch.inference_mode()
def replay(model, prompt, steps, forced=None):
    tokens, cache, logits, predictions = prompt, None, [], []
    for i in range(steps):
        out = model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        logits.append(out.logits[:,-1,:].float().cpu())
        prediction = out.logits[:,-1,:].argmax(-1,keepdim=True)
        predictions.append(prediction)
        tokens = prediction if forced is None else forced[i]
    return logits, predictions


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--length',type=int,default=346)
    ap.add_argument('--steps',type=int,default=32)
    args = ap.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True,exist_ok=False)
    report = {'environment': {k: importlib.metadata.version(k)
              for k in ('torch','triton','transformers')}, 'args':vars(args),
              'gpu':torch.cuda.get_device_name(), 'model_revision':
              '989aa7980e4cf806f80c7fef2b1adb7bc71aa306', 'layers':[]}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                  attn_implementation='sdpa',local_files_only=True).cuda().eval()
    gen = torch.Generator(device='cuda').manual_seed(2026)
    prompt = torch.randint(100,10000,(1,args.length),device='cuda',generator=gen)
    torch.save(prompt.cpu(),outdir/'prompt.pt')
    captures, handles = [], []
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    for name, module in model.named_modules():
        if isinstance(module,Qwen2RMSNorm):
            def capture(mod, inputs, name=name):
                captures.append((name,inputs[0].detach().reshape(-1,inputs[0].shape[-1]).cpu(),
                                 mod.weight.detach().cpu(),mod.variance_epsilon))
            handles.append(module.register_forward_pre_hook(capture))
    model(input_ids=prompt,use_cache=False,logits_to_keep=1)
    for h in handles: h.remove()
    saved = False
    for name, xc, wc, eps in captures:
        x,w = xc.cuda().contiguous(),wc.cuda()
        ref = reference_stages(x,w,eps)
        got, kernel = probe(x,w,eps,ref['mean'])
        injected,_ = probe(x,w,eps,ref['mean'],True)
        native = torch.empty_like(x)
        original = _qwen_rms[(x.shape[0],)](x,w,native,x.shape[1],eps,
                    triton.next_power_of_2(x.shape[1]),num_warps=4,enable_fp_fusion=False)
        stage = {k:stats(got[k],ref[k]) for k in ref}
        fidelity = stats(got['output'],native)
        if fidelity['different']:
            raise RuntimeError('instrumented probe differs from original kernel')
        item = dict(name=name,shape=list(x.shape),eps=eps,stages=stage,
                    first_difference=next((k for k in ref if stage[k]['different']),None),
                    probe_vs_original=fidelity,
                    injected_variance={k:stats(injected[k],ref[k]) for k in ref})
        item['scaling_controls'] = dict(
            aten_mean_vs_aten_sum_scaled=stats(ref['mean'],ref['sum']*(1.0/x.shape[1])),
            triton_mean_vs_triton_sum_scaled=stats(got['mean'],got['sum']*(1.0/x.shape[1])))
        report['layers'].append(item)
        print(name,item['first_difference'],stage['output'],flush=True)
        if not saved and stage['output']['different']:
            torch.save(dict(x=xc,w=wc,eps=eps,name=name),outdir/'captured_input.pt')
            np.savez(outdir/'captured_stages.npz',**{
                f'{prefix}_{k}':v.float().cpu().numpy()
                for prefix,d in [('aten',ref),('triton',got)] for k,v in d.items()})
            rows = (got['output'] != ref['output']).any(-1).nonzero().flatten()
            report['fixture'] = dict(layer=name,first_output_mismatch_row=int(rows[0]))
            for key,value in original.asm.items():
                if isinstance(value,str): (outdir/f'original.{key}').write_text(value)
            for key,value in kernel.asm.items():
                if isinstance(value,str): (outdir/f'probe.{key}').write_text(value)
            saved = True
        (outdir/'diagnosis.json').write_text(json.dumps(report,indent=2))
    set_backend(model,'eager')
    baseline,forced = replay(model,prompt,args.steps)
    report['model'] = {}
    for backend in ('triton','triton_compatible'):
        set_backend(model,backend)
        actual,predictions = replay(model,prompt,args.steps,forced)
        checks = [stats(a,b) for a,b in zip(actual,baseline)]
        report['model'][backend] = dict(positions=checks,
            max_nrmse=max(c['nrmse'] for c in checks),
            checked_original_gate_positions=[0,args.steps-1],
            original_gate_nrmse=max(checks[i]['nrmse'] for i in (0,args.steps-1)),
            greedy_matches=sum(int(torch.equal(a,b)) for a,b in zip(predictions,forced)))
        print(backend,report['model'][backend]['max_nrmse'],flush=True)
    (outdir/'diagnosis.json').write_text(json.dumps(report,indent=2))
    torch.save(dict(logits=baseline,forced=[x.cpu() for x in forced]),outdir/'baseline.pt')
    (outdir/'complete.json').write_text(json.dumps({'complete':True,'fixture_saved':saved}))


if __name__ == '__main__':
    main()
