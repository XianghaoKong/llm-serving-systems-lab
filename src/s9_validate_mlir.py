"""Validate MLIR-generated CPU/GPU code against ATen and model logits."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from s9_numeric_probe import stats, reference_stages, replay
from s9_mlir_adapter import MLIRRMSNorm


def cpu_mean(x, adapter):
    x=x.contiguous().cpu()
    m,h=x.shape
    entry=adapter.manifest[f'{m}x{h}']
    library=ctypes.CDLL(str(adapter.directory/entry['cpu_library']))
    fn=library.s9_aten_mean
    fn.argtypes=[ctypes.c_uint64]
    fn.restype=ctypes.c_float
    array=x.view(torch.int16).numpy()
    result=np.array([fn(array.ctypes.data+i*h*2) for i in range(m)],dtype=np.float32)
    return torch.from_numpy(result).reshape(m,1)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--libraries',required=True)
    ap.add_argument('--fixture',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--model')
    ap.add_argument('--replay-dir')
    args=ap.parse_args()
    outdir=Path(args.output)
    outdir.mkdir(parents=True,exist_ok=False)
    adapter=MLIRRMSNorm(args.libraries)
    fixture=torch.load(args.fixture,weights_only=True)
    cases=[('captured',fixture['x'].cuda(),fixture['w'].cuda(),fixture['eps'])]
    gen=torch.Generator(device='cuda').manual_seed(7709)
    for key,entry in adapter.manifest.items():
        m,h=entry['rows'],entry['width']
        x=torch.randn((m,h),device='cuda',generator=gen).bfloat16()
        w=torch.randn((h,),device='cuda',generator=gen).bfloat16()
        cases.append((f'random_{key}',x,w,1e-6))
        # Positive/negative zeros, tiny values, wide dynamic range and outliers.
        z=x.clone()
        z[:,::11]=0
        z[:,1::11]=-0.0
        z[:,2::11]*=1e-6
        z[:,3::11]*=1e6
        cases.append((f'dynamic_{key}',z,w,1e-6))
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'cases':[],'performance_claim':False}
    passed=True
    for name,x,w,eps in cases:
        ref=reference_stages(x,w,eps)
        cpu=cpu_mean(x,adapter)
        actual,variance,kernel=adapter.run(x,w,eps)
        item=dict(name=name,shape=list(x.shape),
                  cpu_mean=stats(cpu,ref['mean']),gpu_mean=stats(variance,ref['mean']),
                  gpu_output=stats(actual,ref['output']))
        # Execute compiled code twice to check repeated-run stability too.
        again,_,_=adapter.run(x,w,eps)
        item['repeat']=stats(again,actual)
        passed &= all(item[k]['different']==0 and item[k]['finite']
                      for k in ('cpu_mean','gpu_mean','gpu_output','repeat'))
        report['cases'].append(item)
        print(json.dumps(item),flush=True)
        if name=='captured':
            for key,value in kernel.asm.items():
                if isinstance(value,str):(outdir/f'compiled.{key}').write_text(value)
        (outdir/'validation.json').write_text(json.dumps(report,indent=2))
    if args.model:
        if not args.replay_dir:raise ValueError('--model requires --replay-dir')
        replay_dir=Path(args.replay_dir)
        prompt=torch.load(replay_dir/'prompt.pt',weights_only=True).cuda()
        baseline=torch.load(replay_dir/'baseline.pt',weights_only=True)
        forced=[x.cuda() for x in baseline['forced']]
        model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
                   attn_implementation='sdpa',local_files_only=True).cuda().eval()
        adapter.install(model)
        logits,predictions=replay(model,prompt,len(forced),forced)
        checks=[stats(a,b) for a,b in zip(logits,baseline['logits'])]
        report['model']=dict(positions=checks,max_nrmse=max(x['nrmse'] for x in checks),
            greedy_matches=sum(int(torch.equal(a,b)) for a,b in zip(predictions,forced)),
            compiled_calls=adapter.hits,fallback_calls=adapter.fallbacks)
        passed &= all(x['different']==0 and x['finite'] for x in checks)
        passed &= adapter.hits==57*len(forced) and adapter.fallbacks==0
        print(json.dumps(report['model']),flush=True)
    report['passed']=bool(passed)
    (outdir/'validation.json').write_text(json.dumps(report,indent=2))
    if not passed:raise SystemExit('numeric validation failed; inspect validation.json')


if __name__=='__main__':main()
