"""Numerically gated host/graph ablation with complete balanced order cycles."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from s9_mlir_adapter import MLIRRMSNorm
from s9_decode_graph import ExactDecodeGraph
from s9_numeric_probe import replay,stats
from s9_qwen_adapter import set_backend
from s9_benchmark_mlir import timed_request,telemetry,host_samples
from s9_benchmark_order import balanced_order,order_cycle_length

BACKENDS=('eager','compatible','before','fast','graph')


def load_before(path):
    spec=importlib.util.spec_from_file_location('s9_adapter_before_host',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.MLIRRMSNorm


def timed_graph(runner,prompt,forced):
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in forced]
    for a,b in events:a.record();b.record()
    torch.cuda.synchronize();begin=time.perf_counter_ns()
    events[0][0].record();output=runner.begin(prompt);events[0][1].record()
    for i,(a,b) in enumerate(events[1:],start=1):
        a.record();output=runner.step(forced[i-1]);b.record()
    torch.cuda.synchronize();wall=(time.perf_counter_ns()-begin)/1e6
    values=[a.elapsed_time(b) for a,b in events]
    return dict(model_loop_wall_ms=wall,prefill_gpu_ms=values[0],
        decode_gpu_p50_ms=st.median(values[1:]),decode_gpu_p95_ms=float(np.quantile(values[1:],.95)),
        forward_event_ms=values)


def install(model,b,before,fast):
    if b=='before':before.install(model)
    elif b in ('fast','graph'):fast.install(model)
    else:set_backend(model,'triton_compatible' if b=='compatible' else 'eager')


def measure(model,case,b,before,fast):
    install(model,b,before,fast)
    h0=(before.hits,fast.hits);f0=(before.fallbacks,fast.fallbacks)
    g0=case['runner'].replays
    if b=='graph':result=timed_graph(case['runner'],case['prompt'],case['forced'])
    else:result=timed_request(model,case['prompt'],case['forced'])
    calls=(before.hits-h0[0],fast.hits-h0[1]);graphs=case['runner'].replays-g0
    assert (before.fallbacks,fast.fallbacks)==f0
    expected={'eager':(0,0),'compatible':(0,0),'before':(1824,0),'fast':(0,1824),'graph':(0,57)}
    assert calls==expected[b] and graphs==(31 if b=='graph' else 0)
    return dict(**result,python_norm_calls=dict(before=calls[0],fast=calls[1]),
                graph_replays=graphs,fallback_calls=0)


def graph_check(runner,prompt,forced,reference,greedy=False):
    output=runner.begin(prompt);actual=[];tokens=[]
    for i in range(32):
        actual.append(output[:,-1,:].float().cpu())
        token=output[:,-1,:].argmax(-1,keepdim=True)
        tokens.append(token)
        if i<31:output=runner.step(token if greedy else forced[i])
    checks=[stats(a,r) for a,r in zip(actual,reference)]
    assert all(c['different']==0 and c['finite'] for c in checks)
    assert sum(torch.equal(a,b) for a,b in zip(tokens,forced))==32
    return dict(positions=checks,greedy_matches=32,mode='actual greedy' if greedy else 'forced common prefix')


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--libraries',required=True);ap.add_argument('--before-adapter',required=True)
    ap.add_argument('--model',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--lengths',nargs='+',type=int,default=[33,346,4096])
    ap.add_argument('--blocks',type=int,default=10)
    args=ap.parse_args()
    if args.blocks<10 or args.blocks%order_cycle_length(BACKENDS):
        raise ValueError('use complete balanced ten-block cycles')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    fast=MLIRRMSNorm(args.libraries);before=load_before(args.before_adapter)(args.libraries)
    environment=dict(args=vars(args),torch=torch.__version__,start=telemetry(),
        backends=BACKENDS,order_cycle_blocks=10,shared_desktop_gpu=True,clocks_locked=False,
        bitcode_manifest=fast.manifest,before_sha256=hashlib.sha256(Path(args.before_adapter).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in
                       ('s9_benchmark_host.py','s9_mlir_adapter.py','s9_decode_graph.py','s9_benchmark_mlir.py','s9_benchmark_order.py')},
        scope='32 forwards; common eager prefixes; no sampling/HTTP/tokenization; graph initial KV copy and per-step token copies included',
        graph_setup='Preparation/capture excluded from warm latency; reported separately; exact-position graphs reused across fresh requests')
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
               attn_implementation='sdpa',local_files_only=True).cuda().eval()
    cases=[];correctness=[];preparations=[]
    for length in args.lengths:
        gen=torch.Generator(device='cuda').manual_seed(2026 if length==346 else 2026+length)
        prompt=torch.randint(100,10000,(1,length),device='cuda',generator=gen)
        set_backend(model,'eager');reference,forced=replay(model,prompt,32)
        row=dict(length=length,paths={})
        for b in ('compatible','before','fast'):
            install(model,b,before,fast)
            actual,tokens=replay(model,prompt,32,forced)
            checks=[stats(a,r) for a,r in zip(actual,reference)]
            assert all(c['different']==0 and c['finite'] for c in checks)
            assert sum(torch.equal(a,f) for a,f in zip(tokens,forced))==32
            row['paths'][b]=dict(positions=checks,greedy_matches=32)
        fast.install(model);torch.cuda.synchronize()
        allocated_before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        runner=ExactDecodeGraph(model,prompt,31)
        preparations.append(dict(length=length,prepare_ms=runner.prepare_ms,
            kv_buffer_bytes=runner.kv_buffer_bytes,allocated_before=allocated_before,
            allocated_after=torch.cuda.memory_allocated(),peak_allocated=torch.cuda.max_memory_allocated(),
            reserved_after=torch.cuda.memory_reserved(),graphs=len(runner.graphs)))
        row['paths']['graph']=graph_check(runner,prompt,forced,reference)
        # Reuse the same graphs with different prompt data and genuine greedy
        # token feedback. This catches stale KV buffers and fixed-token replay.
        second=torch.randint(100,10000,(1,length),device='cuda',generator=gen)
        set_backend(model,'eager');second_ref,second_tokens=replay(model,second,32)
        fast.install(model)
        row['paths']['graph_second_request']=graph_check(runner,second,second_tokens,second_ref,True)
        cases.append(dict(length=length,prompt=prompt,forced=forced,runner=runner))
        correctness.append(row)
        (out/'correctness.json').write_text(json.dumps(correctness,indent=2))
        (out/'preparation.json').write_text(json.dumps(preparations,indent=2))
        print('correctness',length,'passed; graph prepare ms',runner.prepare_ms,flush=True)
    assert before.fallbacks==fast.fallbacks==0
    for case in cases:
        for b in BACKENDS:measure(model,case,b,before,fast)
    records=[]
    with (out/'model-records.jsonl').open('x') as stream:
        for block in range(args.blocks):
            for index,case in enumerate(cases):
                order=balanced_order(block,index,BACKENDS);telem=telemetry();batch=[]
                for b in order:
                    batch.append(dict(length=case['length'],block=block+1,backend=b,order=order,
                        telemetry=telem,**measure(model,case,b,before,fast)))
                for row in batch:stream.write(json.dumps(row)+'\n')
                stream.flush();records.extend(batch)
            print('timing block',block+1,'of',args.blocks,flush=True)
    metrics=('model_loop_wall_ms','prefill_gpu_ms','decode_gpu_p50_ms','decode_gpu_p95_ms')
    summary=[]
    for length in args.lengths:
        timing={b:{k:st.median(r[k] for r in records if r['length']==length and r['backend']==b)
                   for k in metrics} for b in BACKENDS}
        comparisons={}
        for candidate,baseline in (('fast','before'),('graph','fast'),('graph','before'),('graph','compatible'),('fast','compatible')):
            paired=[]
            for block in range(1,args.blocks+1):
                group={r['backend']:r for r in records if r['length']==length and r['block']==block}
                paired.append({k:100*(group[candidate][k]/group[baseline][k]-1) for k in metrics})
            comparisons[f'{candidate}_vs_{baseline}']=dict(
                median_percent={k:st.median(p[k] for p in paired) for k in metrics},blocks=paired,
                faster_wall_blocks=sum(p['model_loop_wall_ms']<0 for p in paired))
        summary.append(dict(length=length,timing=timing,comparisons=comparisons))
    (out/'models.json').write_text(json.dumps(summary,indent=2))
    # API allocation/dispatch comparison is separate from model timing.
    micro=[]
    for rows in (1,346,4096):
        x=torch.randn(rows,1536,device='cuda').bfloat16();w=torch.ones(1536,device='cuda',dtype=x.dtype)
        before(x,w,1e-6);fast(x,w,1e-6)
        values={'before':[],'fast':[]}
        for block in range(10):
            for b in (('before','fast') if block%2==0 else ('fast','before')):
                a=before if b=='before' else fast
                values[b].extend(host_samples(lambda:a(x,w,1e-6)))
        micro.append(dict(rows=rows,samples_us=values,median_us={b:st.median(v) for b,v in values.items()}))
    (out/'host-micro.json').write_text(json.dumps(micro,indent=2))
    # Observe one real graph decode after a fresh request, outside timing.
    case=next((c for c in cases if c['length']==346),cases[0])
    fast.install(model);case['runner'].begin(case['prompt']);torch.cuda.synchronize()
    sentinel=torch.ones(1,device='cuda')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        sentinel.sin_();torch.cuda.synchronize()
        with torch.profiler.record_function('graph_decode_target'):
            case['runner'].step(case['forced'][0]);torch.cuda.synchronize()
    prof.export_chrome_trace(str(out/'graph-decode-profile.json'))
    case['runner'].reset()
    (out/'complete.json').write_text(json.dumps(dict(passed=True,model_requests=len(records),
        correctness_positions=len(cases)*5*32,fallbacks=fast.fallbacks+before.fallbacks,end=telemetry()),indent=2))
    print('complete',flush=True)


if __name__=='__main__':
    with torch.cuda.stream(torch.cuda.Stream()):main()
