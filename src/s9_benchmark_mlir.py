"""Paired performance study of eager, full Triton, ATen-compatible and MLIR.

Numerical acceptance is separate from timing. A fast incompatible kernel is
retained as a diagnostic comparison, never promoted by its timing result.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch
import triton
from transformers import AutoModelForCausalLM
from s9_mlir_adapter import MLIRRMSNorm
from s9_numeric_probe import stats, replay
from s9_qwen_adapter import qwen_rms, qwen_rms_compatible, set_backend, _qwen_rms
from s9_benchmark_order import balanced_order, order_cycle_length

BACKENDS=('eager','triton','triton_compatible','mlir')


def eager(x,w,eps):
    xf=x.float()
    variance=xf.square().mean(-1,keepdim=True)
    normalized=xf*torch.rsqrt(variance+eps)
    return w*normalized.to(x.dtype)


def quantile(values,q):
    return float(np.quantile(values,q))


def telemetry():
    result=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,'
        'power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'],capture_output=True,text=True)
    return dict(time_unix=time.time(),csv=result.stdout.strip(),error=result.stderr.strip())


def resources(kernel):
    return dict(registers_per_thread=kernel.n_regs,
                n_spills_reported_by_triton=kernel.n_spills,
                shared_bytes=kernel.metadata.shared,num_warps=kernel.metadata.num_warps,
                note='Compiler/runtime metadata, not hardware counters or measured spill traffic')


def capture_graph(fn,repeats):
    for _ in range(5):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):result=fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph,result


def graph_samples(graph,repeats,samples):
    starts=[torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends=[torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start,end in zip(starts,ends):
        start.record();graph.replay();end.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b)*1000/repeats for a,b in zip(starts,ends)]


def host_samples(fn,repeats=30,samples=5):
    # Synchronized batch wall time includes Python dispatch and allocations.
    values=[]
    for _ in range(samples):
        torch.cuda.synchronize()
        begin=time.perf_counter_ns()
        for _ in range(repeats):result=fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns()-begin)/1000/repeats)
    return values


def profile_call(fn,path):
    for _ in range(4):fn()
    torch.cuda.synchronize()
    sentinel=torch.ones(1,device='cuda')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        sentinel.sin_();torch.cuda.synchronize()
        with torch.profiler.record_function('s9_mlir_target'):
            result=fn();torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    trace=json.loads(path.read_text())['traceEvents']
    region=next(e for e in trace if e.get('name')=='s9_mlir_target' and e.get('ph')=='X')
    kernels=[e for e in trace if e.get('cat')=='kernel'
             and region['ts']<=e['ts']<region['ts']+region['dur']]
    if not kernels:raise RuntimeError('empty target profile')
    return dict(kernel_launches=len(kernels),
        kernels=[dict(name=e['name'],duration_us=e['dur'],args=e.get('args',{})) for e in kernels],
        diagnostic_only=True)


def bench_operators(args,out,adapter,scalar_adapter=None):
    fixture=torch.load(args.fixture,weights_only=True)
    impls={'eager':eager,'triton':qwen_rms,'triton_compatible':qwen_rms_compatible,'mlir':adapter}
    if scalar_adapter is not None:impls['mlir_scalar']=scalar_adapter
    report=[]
    with (out/'operator-records.jsonl').open('x') as stream:
        for m in args.rows:
            h=1536
            if f'{m}x{h}' not in adapter.manifest:raise ValueError('uncompiled shape')
            if m==fixture['x'].shape[0]:
                x,w=fixture['x'].cuda(),fixture['w'].cuda()
                input_kind='captured Qwen layer activation'
            else:
                gen=torch.Generator(device='cuda').manual_seed(20261102+m)
                x=torch.randn(m,h,device='cuda',generator=gen).bfloat16()
                w=torch.randn(h,device='cuda',generator=gen).bfloat16()
                input_kind='fixed-seed Gaussian BF16'
            eps=fixture['eps']
            fns={b:(lambda b=b:impls[b](x,w,eps)) for b in BACKENDS}
            reference=eager(x,w,eps)
            correctness={b:stats(fn(),reference) for b,fn in fns.items()}
            for b in ('triton_compatible','mlir')+ (('mlir_scalar',) if scalar_adapter else ()):
                if correctness[b]['different'] or not correctness[b]['finite']:
                    raise RuntimeError(f'numerical gate failed: {b}, rows={m}')
            _,_,kernel=adapter.run(x,w,eps)
            resource={'mlir':resources(kernel)}
            if scalar_adapter is not None:
                _,_,scalar_kernel=scalar_adapter.run(x,w,eps)
                resource['mlir_scalar']=resources(scalar_kernel)
            y=torch.empty_like(x)
            kernel=_qwen_rms[(m,)](x,w,y,h,eps,triton.next_power_of_2(h),
                                    num_warps=4,enable_fp_fusion=False)
            resource['triton']=resources(kernel)
            graphs={b:capture_graph(fn,args.graph_repeats) for b,fn in fns.items()}
            blocks=[]
            for block in range(args.blocks):
                order=balanced_order(block,args.rows.index(m),BACKENDS)
                telem=telemetry()
                for b in order:
                    values=graph_samples(graphs[b][0],args.graph_repeats,args.samples)
                    host=host_samples(fns[b])
                    row=dict(rows=m,width=h,backend=b,block=block+1,
                        order=list(order),gpu_graph_us=values,host_batch_us=host,
                        gpu_p50_us=statistics.median(values),gpu_p95_us=quantile(values,.95),
                        host_p50_us=statistics.median(host),telemetry=telem)
                    stream.write(json.dumps(row)+'\n');stream.flush();blocks.append(row)
            timing={}
            for b in BACKENDS:
                records=[r for r in blocks if r['backend']==b]
                timing[b]={k:statistics.median(r[k] for r in records)
                           for k in ('gpu_p50_us','gpu_p95_us','host_p50_us')}
            profiles={}
            if m in (1,346):
                for b in BACKENDS:
                    profiles[b]=profile_call(fns[b],out/f'profile-r{m}-{b}.json')
            row=dict(rows=m,input_kind=input_kind,correctness=correctness,
                     timing=timing,resources=resource,profiles=profiles)
            report.append(row)
            (out/'operators.json').write_text(json.dumps(report,indent=2))
            print('operator',m,json.dumps(timing),flush=True)
            del graphs,fns,x,w,y,reference,kernel
            gc.collect();torch.cuda.empty_cache()
    return report


def timed_request(model,prompt,forced):
    # CPU copies, argmax, correctness checks and event construction are outside
    # the timed forward loop. EOS is ignored; all paths consume the same tokens.
    # Forward events include host launch gaps. A faster prefill can move those
    # gaps into decode intervals as the GPU queue drains; these are not isolated
    # per-step kernel-active times. Use a fixed-position graph to separate them.
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            for _ in forced]
    for a,b in events:a.record();b.record()
    torch.cuda.synchronize()
    begin=time.perf_counter_ns()
    cache=None;tokens=prompt
    for i,(a,b) in enumerate(events):
        a.record()
        output=model(input_ids=tokens,past_key_values=cache,use_cache=True,logits_to_keep=1)
        cache=output.past_key_values
        b.record()
        tokens=forced[i]
    torch.cuda.synchronize()
    wall=(time.perf_counter_ns()-begin)/1e6
    values=[a.elapsed_time(b) for a,b in events]
    return dict(prefill_gpu_ms=values[0],decode_gpu_p50_ms=statistics.median(values[1:]),
                decode_gpu_p95_ms=quantile(values[1:],.95),model_loop_wall_ms=wall,
                forward_event_ms=values)


def install(model,b,adapter,scalar_adapter=None):
    if b=='mlir':adapter.install(model)
    elif b=='mlir_scalar':scalar_adapter.install(model)
    else:set_backend(model,b)


def bench_model(args,out,adapter,scalar_adapter=None):
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
                    attn_implementation='sdpa',local_files_only=True).cuda().eval()
    cases=[]
    correctness=[]
    for length in args.lengths:
        if f'{length}x1536' not in adapter.manifest:raise ValueError('uncompiled model prefill shape')
        seed=2026 if length==346 else 2026+length
        gen=torch.Generator(device='cuda').manual_seed(seed)
        prompt=torch.randint(100,10000,(1,length),device='cuda',generator=gen)
        install(model,'eager',adapter)
        reference,forced=replay(model,prompt,args.steps)
        case={'length':length,'seed':seed,'prompt':prompt,'forced':forced}
        cases.append(case)
        row=dict(length=length,seed=seed,backends={})
        for b in BACKENDS:
            install(model,b,adapter,scalar_adapter)
            active=scalar_adapter if b=='mlir_scalar' else adapter
            hit0,fall0=active.hits,active.fallbacks
            actual,predictions=replay(model,prompt,args.steps,forced)
            checks=[stats(a,r) for a,r in zip(actual,reference)]
            row['backends'][b]=dict(max_nrmse=max(c['nrmse'] for c in checks),
                first_last_nrmse=max(checks[i]['nrmse'] for i in (0,args.steps-1)),
                different_logits=sum(c['different'] for c in checks),positions=checks,
                greedy_matches=sum(int(torch.equal(a,f)) for a,f in zip(predictions,forced)),
                compiled_calls=active.hits-hit0,fallback_calls=active.fallbacks-fall0)
            if b in ('eager','triton_compatible','mlir','mlir_scalar'):
                if any(c['different'] or not c['finite'] for c in checks):
                    raise RuntimeError(f'model correctness failed: {b}, length={length}')
            if b.startswith('mlir') and (active.hits-hit0!=57*args.steps or active.fallbacks!=fall0):
                raise RuntimeError('model used unexpected MLIR fallback')
            # Warm full request at the exact shape, outside measured runs.
            timed_request(model,prompt,forced)
        correctness.append(row)
        (out/'model-correctness.json').write_text(json.dumps(correctness,indent=2))
        print('model correctness',length,{b:row['backends'][b]['max_nrmse'] for b in BACKENDS},flush=True)
    records=[]
    with (out/'model-records.jsonl').open('x') as stream:
        for block in range(args.blocks):
            for index,case in enumerate(cases):
                order=balanced_order(block,index,BACKENDS)
                telem=telemetry()
                for b in order:
                    install(model,b,adapter,scalar_adapter)
                    active=scalar_adapter if b=='mlir_scalar' else adapter
                    hit0,fall0=active.hits,active.fallbacks
                    row=dict(block=block+1,length=case['length'],seed=case['seed'],backend=b,
                        order=list(order),telemetry=telem,
                        **timed_request(model,case['prompt'],case['forced']))
                    row['compiled_calls']=active.hits-hit0
                    row['fallback_calls']=active.fallbacks-fall0
                    if b.startswith('mlir') and (row['compiled_calls']!=57*args.steps or row['fallback_calls']):
                        raise RuntimeError('timed MLIR path fell back')
                    records.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    print('model timing',block+1,case['length'],b,row['model_loop_wall_ms'],flush=True)
    summary=[]
    metrics=('prefill_gpu_ms','decode_gpu_p50_ms','decode_gpu_p95_ms','model_loop_wall_ms')
    for length in args.lengths:
        timing={b:{k:statistics.median(r[k] for r in records
                       if r['length']==length and r['backend']==b) for k in metrics} for b in BACKENDS}
        paired=[]
        for block in range(1,args.blocks+1):
            pair={r['backend']:r for r in records if r['length']==length and r['block']==block}
            paired.append({k:pair['triton_compatible'][k]/pair['mlir'][k] for k in metrics})
        summary.append(dict(length=length,timing=timing,
            compatible_over_mlir_paired_speedup={k:statistics.median(p[k] for p in paired) for k in metrics},
            paired_speedup_blocks=paired))
    (out/'models.json').write_text(json.dumps(summary,indent=2))
    return summary


@torch.inference_mode()
def main():
    global BACKENDS
    ap=argparse.ArgumentParser()
    ap.add_argument('--libraries',required=True)
    ap.add_argument('--scalar-libraries',help='also remeasure the previous scalar MLIR prototype')
    ap.add_argument('--fixture',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--rows',nargs='+',type=int,default=[1,2,4,8,16,33,346,4096])
    ap.add_argument('--lengths',nargs='+',type=int,default=[33,346,4096])
    ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--blocks',type=int,help='default: complete balanced order cycles (8 or 10 blocks)')
    ap.add_argument('--samples',type=int,default=30)
    ap.add_argument('--graph-repeats',type=int,default=10)
    args=ap.parse_args()
    if args.scalar_libraries:BACKENDS=BACKENDS+('mlir_scalar',)
    cycle=order_cycle_length(BACKENDS)
    if args.blocks is None:args.blocks=cycle if cycle>=8 else 2*cycle
    if args.steps<2 or min(args.blocks,args.samples,args.graph_repeats)<1:raise ValueError('invalid sampling parameters')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    adapter=MLIRRMSNorm(args.libraries)
    scalar_adapter=MLIRRMSNorm(args.scalar_libraries) if args.scalar_libraries else None
    environment=dict(args=vars(args),versions={k:importlib.metadata.version(k)
        for k in ('torch','triton','transformers')},start_telemetry=telemetry(),
        source_sha256={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
            for p in ('s9_benchmark_mlir.py','s9_benchmark_order.py','s9_mlir_adapter.py','s9_qwen_adapter.py','s9_numeric_probe.py')},
        bitcode_manifest=adapter.manifest,scalar_bitcode_manifest=scalar_adapter.manifest if scalar_adapter else None,
        model_revision_expected='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
        gpu_clocks_locked=False,shared_desktop_gpu=True,
        protocol='Serial jobs; warm exact shapes; Williams balanced backend order; no compilation/profiling inside timing',
        order_cycle_blocks=cycle,complete_order_cycles=args.blocks%cycle==0,
        model_scope=f'{len(args.lengths)} synthetic requests, common forced prefixes; no HTTP/tokenization/sampling; not 200-request S9 replay',
        graph_scope=f'{args.graph_repeats} invocations/replay, hot inputs; API allocations captured; not request latency',
        model_event_scope='Forward CUDA-event elapsed time can include host launch gaps; not pure kernel active time')
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        operators=bench_operators(args,out,adapter,scalar_adapter)
        models=bench_model(args,out,adapter,scalar_adapter)
    (out/'complete.json').write_text(json.dumps(dict(complete=True,operator_cases=len(operators)*len(BACKENDS),
        operator_blocks=len(operators)*len(BACKENDS)*args.blocks,
        model_calls=len(models)*len(BACKENDS)*args.blocks,end_telemetry=telemetry()),indent=2))


if __name__=='__main__':main()
