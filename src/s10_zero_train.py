"""DeepSpeed ZeRO study using random Qwen2 weights and fixed synthetic tokens."""
import argparse
import datetime
import importlib.metadata
import json
import os
import math
import time
from pathlib import Path
import torch
import torch.distributed as dist
from s10_config import accumulation, zero_state_bytes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--stage",type=int,choices=range(4),required=True)
    ap.add_argument("--model",choices=("smoke","qwen7b"),default="smoke")
    ap.add_argument("--sequence",type=int,default=512)
    ap.add_argument("--global-tokens",type=int,default=8192)
    ap.add_argument("--microbatch",type=int,default=1)
    ap.add_argument("--warmup",type=int,default=20)
    ap.add_argument("--steps",type=int,default=100)
    ap.add_argument("--local_rank",type=int,default=-1)
    ap.add_argument("--profile",action="store_true")
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args()
    rank=int(os.environ.get("RANK",0)); world=int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0));torch.cuda.set_device(local)
    dist.init_process_group("nccl",timeout=datetime.timedelta(minutes=10))
    import deepspeed
    deepspeed.init_distributed(dist_backend="nccl")
    from transformers import Qwen2Config,Qwen2ForCausalLM
    grad_acc=accumulation(args.global_tokens,args.sequence,args.microbatch,world)
    config=dict(train_micro_batch_size_per_gpu=args.microbatch,gradient_accumulation_steps=grad_acc,
        train_batch_size=args.global_tokens//args.sequence,bf16={"enabled":True},
        zero_optimization={"stage":args.stage,"overlap_comm":False,"contiguous_gradients":True,
            "reduce_bucket_size":50000000,"allgather_bucket_size":50000000},
        optimizer={"type":"AdamW","params":{"lr":1e-4,"betas":[0.9,0.95],"eps":1e-8,"weight_decay":0.01,
            "torch_adam":True,"foreach":False}},gradient_clipping=1.0,steps_per_print=100000,
        wall_clock_breakdown=False,zero_allow_untested_optimizer=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result={"rank":rank,"world_size":world,"args":vars(args),"grad_accumulation":grad_acc,
            "gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
            "deepspeed":importlib.metadata.version("deepspeed"),"complete":False}
    try:
        torch.manual_seed(args.seed)
        dims=dict(hidden_size=3584,intermediate_size=18944,num_hidden_layers=28,
                  num_attention_heads=28,num_key_value_heads=4,vocab_size=152064) if args.model=="qwen7b" else dict(
                  hidden_size=512,intermediate_size=1536,num_hidden_layers=4,
                  num_attention_heads=8,num_key_value_heads=2,vocab_size=4096)
        cfg=Qwen2Config(**dims,max_position_embeddings=max(4096,args.sequence),
            tie_word_embeddings=False,use_cache=False,attention_dropout=0.0,rms_norm_eps=1e-6)
        cfg._attn_implementation="sdpa"
        # ZeRO-3 must partition initialization too; construct directly in BF16.
        with deepspeed.zero.Init(config_dict_or_path=config,enabled=args.stage==3,dtype=torch.bfloat16):
            original=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
            try:
                with torch.device(f"cuda:{local}"):
                    model=Qwen2ForCausalLM(cfg)
            finally: torch.set_default_dtype(original)
        params=sum(getattr(p,"ds_numel",p.numel()) for p in model.parameters())
        result.update(parameters=params,model_config=cfg.to_dict(),state_estimate=zero_state_bytes(params,world,args.stage))
        engine,optimizer,_,_=deepspeed.initialize(model=model,model_parameters=model.parameters(),config=config)
        result["deepspeed_config"]=config
        result["optimizer_wrapper"]=type(optimizer).__name__
        result["model_and_accumulation_dtypes"]=[str(t) for t in engine.get_data_types()]
        result["allocated_after_initialize"]=torch.cuda.memory_allocated()
        # Same fixed global synthetic batch; rank partitions differ with world.
        generator=torch.Generator(device="cpu").manual_seed(91)
        global_ids=torch.randint(0,dims["vocab_size"],(args.global_tokens//args.sequence,args.sequence),generator=generator)
        local_ids=global_ids[rank::world].to(engine.device)
        steps=[];loss_values=[];gradient_norms=[]
        profiler=None
        if args.profile:
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
        engine.train()
        for step in range(args.warmup+args.steps):
            if step==args.warmup:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                if profiler:profiler.start()
            dist.barrier();torch.cuda.synchronize();start=time.perf_counter()
            losses=[]
            for micro in range(grad_acc):
                tokens=local_ids[micro*args.microbatch:(micro+1)*args.microbatch]
                loss=engine(input_ids=tokens,labels=tokens).loss
                engine.backward(loss);engine.step();losses.append(loss.detach().float())
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            worst=torch.tensor(elapsed,device=engine.device,dtype=torch.float64)
            dist.all_reduce(worst,op=dist.ReduceOp.MAX)
            finite=torch.stack(losses).isfinite().all().int()
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item():raise RuntimeError("nonfinite loss on one or more ranks")
            norm=engine.get_global_grad_norm()
            norm=float(norm) if norm is not None else None
            if norm is not None and not math.isfinite(norm):
                raise RuntimeError("nonfinite gradient norm")
            if step>=args.warmup:
                steps.append(worst.item());loss_values.append(torch.stack(losses).mean().item())
                gradient_norms.append(norm)
            if profiler and step==args.warmup+1:
                profiler.stop();profiler.export_chrome_trace(str(out/f"trace-rank{rank}.json"));profiler=None
            if rank==0 and step%10==0:print("step",step,"seconds",worst.item(),"loss",losses[-1].item(),flush=True)
            if step%10==0:
                (out/f"progress-rank{rank}.json").write_text(json.dumps({"step":step,"seconds":worst.item(),"loss":float(losses[-1]),"gradient_norm":norm}))
        result.update(complete=True,formal=args.warmup>=20 and args.steps>=100 and not args.profile,
            steps_seconds=steps,loss=loss_values,gradient_norm=gradient_norms,tokens_per_second=args.global_tokens*len(steps)/sum(steps),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        (out/f"rank-{rank}.json").write_text(json.dumps(result,indent=2,default=str))
        if dist.is_initialized():dist.destroy_process_group()


if __name__=="__main__":main()
