"""Pretrained Qwen integration: length-matched synthetic tokens, not source text.

The public R0 manifest provides lengths/categories; private frozen request text
is unavailable. This is an in-process model-path benchmark, not an HTTP/SSE
serving run or a semantic-quality evaluation. Fixed continuation length avoids
EOS and output-length confounding; both paths consume the same baseline tokens.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from s9_qwen_adapter import set_backend
from s9_analyze import quantile, median_ci


def select_requests(path, count):
    all_rows=list(csv.DictReader(Path(path).open()))
    groups=defaultdict(list)
    for row in all_rows: groups[row["workload_category"]].append(row)
    rng=random.Random(2026)
    selected=[]
    for category,rows in sorted(groups.items()):
        rng.shuffle(rows)
        selected.extend(rows[:round(count*len(rows)/len(all_rows))])
    rng.shuffle(selected)
    if len(selected)!=count: raise ValueError("choose a count with exact stratified allocation")
    return selected


@torch.inference_mode()
def request(model, prompt, steps, forced=None):
    tokens=prompt
    cache=None
    generated=[];events=[];checkpoints=[]
    torch.cuda.synchronize();start_wall=time.perf_counter()
    for step in range(steps):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        out=model(input_ids=tokens,past_key_values=cache,use_cache=True,logits_to_keep=1)
        cache=out.past_key_values
        predicted=out.logits[:,-1,:].argmax(-1,keepdim=True)
        end.record();events.append((start,end));generated.append(predicted)
        if step in (0,steps-1): checkpoints.append(out.logits[:,-1,:].detach())
        tokens=predicted if forced is None else forced[step]
    torch.cuda.synchronize();wall_ms=(time.perf_counter()-start_wall)*1000
    times=[a.elapsed_time(b) for a,b in events]
    return dict(ttft_gpu_ms=times[0],decode_gpu_p50_ms=statistics.median(times[1:]),
                decode_gpu_p95_ms=quantile(times[1:],.95),model_wall_ms=wall_ms,
                generated_ids=torch.cat(generated,1).cpu().tolist()[0]),generated,checkpoints


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True);ap.add_argument("--manifest",required=True)
    ap.add_argument("--output",required=True);ap.add_argument("--count",type=int,default=200)
    ap.add_argument("--steps",type=int,default=32);ap.add_argument("--blocks",type=int,default=3)
    ap.add_argument("--optimized-backend",choices=("triton","triton_compatible"),default="triton_compatible")
    args=ap.parse_args()
    if args.steps<2: raise ValueError("at least two steps required")
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    selected=select_requests(args.manifest,args.count)
    torch.manual_seed(2026)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda().eval()
    count=set_backend(model,"eager")
    metadata=dict(args=vars(args),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"),rms_modules=count,
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        config=model.config.to_dict(),workload="R0 manifest length-matched synthetic token IDs; not original prompts",
        model_path_only=True,tokenization_and_HTTP_excluded=True,
        precision_contract="Qwen intermediate cast; triton_compatible preserves native FP32 reduction",
        optimized_backend=args.optimized_backend,
        attention="Transformers SDPA",fixed_continuation_steps=args.steps,EOS_ignored=True,
        seed=2026,formal=args.count==200 and args.blocks>=3,
        order="block 1 eager then Triton per request; later blocks alternate paired order")
    (output/"environment.json").write_text(json.dumps(metadata,indent=2,default=str))
    (output/"request_manifest.json").write_text(json.dumps(selected,indent=2))
    # Warm model, attention and shape-specific RMS kernels before all measurements.
    for backend in ("eager",args.optimized_backend):
        set_backend(model,backend)
        for length in (33,128,512,1024,2048,4085):
            p=torch.randint(100,10000,(1,length),device="cuda")
            request(model,p,4)
    prompts=[]
    for index,row in enumerate(selected):
        g=torch.Generator(device="cuda").manual_seed(2026+index)
        prompts.append(torch.randint(100,10000,(1,int(row["input_tokens"])),device="cuda",generator=g))
    references={};results=[]
    with (output/"requests.jsonl").open("x") as stream:
        for block in range(args.blocks):
            for index,(row,prompt) in enumerate(zip(selected,prompts)):
                order=("eager",args.optimized_backend) if block==0 or (block+index)%2==0 else (args.optimized_backend,"eager")
                for backend in order:
                    set_backend(model,backend)
                    reference=references.get(index)
                    measured,generated,logits=request(model,prompt,args.steps,None if reference is None else reference[0])
                    if reference is None:
                        references[index]=(generated,[x.cpu() for x in logits]);reference=references[index]
                    max_error=0;normalized_error=0
                    for got,ref in zip(logits,reference[1]):
                        got=got.float().cpu();ref=ref.float()
                        if not torch.isfinite(got).all(): raise ValueError("nonfinite model logits")
                        max_error=max(max_error,(got-ref).abs().max().item())
                        normalized_error=max(normalized_error,((got-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-6)).item())
                    # Tolerance applies to model logits; report divergence rather than hide it.
                    if normalized_error>.02: raise ValueError(f"model logit NRMSE exceeds 2%: {normalized_error}")
                    baseline_ids=torch.cat(reference[0],1).cpu().tolist()[0]
                    measured.update(backend=backend,block=block+1,request_id=row["request_id"],
                        category=row["workload_category"],input_tokens=int(row["input_tokens"]),
                        output_steps=args.steps,logit_max_abs_error=max_error,logit_nrmse=normalized_error,
                        greedy_token_matches=sum(a==b for a,b in zip(measured["generated_ids"],baseline_ids)))
                    stream.write(json.dumps(measured)+"\n");stream.flush();results.append(measured)
                if (index+1)%10==0: print("block",block+1,"requests",index+1,flush=True)
    paired=defaultdict(dict)
    for r in results: paired[(r["block"],r["request_id"])][r["backend"]]=r
    summary={"pairs":len(paired),"model_path_only":True,"formal":metadata["formal"],"optimized_backend":args.optimized_backend}
    for metric in ("ttft_gpu_ms","decode_gpu_p50_ms","decode_gpu_p95_ms","model_wall_ms"):
        ratios=[v["eager"][metric]/v[args.optimized_backend][metric] for v in paired.values()]
        # Bootstrap requests (clusters), retaining all repeated blocks for each request.
        clusters=defaultdict(list)
        for (_,rid),v in paired.items(): clusters[rid].append(v["eager"][metric]/v[args.optimized_backend][metric])
        request_ratios=[statistics.median(v) for v in clusters.values()]
        summary[metric]=dict(eager_median=statistics.median(r[metric] for r in results if r["backend"]=="eager"),
            optimized_median=statistics.median(r[metric] for r in results if r["backend"]==args.optimized_backend),
            median_paired_speedup=statistics.median(ratios),
            median_request_speedup=statistics.median(request_ratios),request_cluster_ci95=median_ci(request_ratios))
    optimized=[r for r in results if r["backend"]==args.optimized_backend]
    summary.update(max_logit_nrmse=max(r["logit_nrmse"] for r in optimized),
        max_logit_abs_error=max(r["logit_max_abs_error"] for r in optimized),
        token_agreement=sum(r["greedy_token_matches"] for r in optimized)/(len(optimized)*args.steps))
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    if metadata["formal"]:
        from s9_profile import capture
        profile_input=torch.randint(100,10000,(1,512),device="cuda")
        profiles={}
        for backend in ("eager",args.optimized_backend):
            set_backend(model,backend)
            profiles[backend]=capture(lambda:request(model,profile_input,4),
                                      output/f"model-profile-{backend}.json")
        (output/"model-profile-summary.json").write_text(json.dumps(dict(
            formal_timing=False,input_tokens=512,continuation_steps=4,backends=profiles),indent=2))
    (output/"complete.json").write_text(json.dumps({"complete":True,"requests":len(results)}))
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()
