"""Compare one ZeRO AdamW update against an explicit BF16/FP32 DP reference.

Run under torchrun with two ranks. Common initial weights are loaded into every
stage, avoiding stage-dependent initialization. This is a tiny-model numerical
check, separate from all timed experiments.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",type=int,choices=(1,2,3),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    local=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank=dist.get_rank(); world=dist.get_world_size()
    import deepspeed
    from deepspeed.utils import safe_get_full_fp32_param
    from transformers import Qwen2Config,Qwen2ForCausalLM
    deepspeed.init_distributed(dist_backend="nccl")
    torch.manual_seed(177)
    cfg=Qwen2Config(hidden_size=128,intermediate_size=256,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=256,
        tie_word_embeddings=False,use_cache=False,attention_dropout=0.0)
    cfg._attn_implementation="sdpa"
    reference=Qwen2ForCausalLM(cfg).to(device="cuda",dtype=torch.bfloat16)
    initial={n:p.detach().float().clone() for n,p in reference.named_parameters()}
    master=[torch.nn.Parameter(p.clone()) for p in initial.values()]
    optimizer=torch.optim.AdamW(master,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
    generator=torch.Generator().manual_seed(42)
    global_tokens=torch.randint(0,256,(8,32),generator=generator).cuda()
    accum=8//world
    grads=[torch.zeros_like(p) for p in master]
    reference_losses=[]
    for index in range(rank,8,world):
        tokens=global_tokens[index:index+1]
        loss=reference(input_ids=tokens,labels=tokens).loss
        reference_losses.append(float(loss.detach()))
        (loss/accum).backward()
        for p,g in zip(reference.parameters(),grads):
            if p.grad is not None: g.add_(p.grad.float())
        reference.zero_grad(set_to_none=True)
    for p,g in zip(master,grads):
        dist.all_reduce(g)
        p.grad=g/world
    ref_norm=float(torch.nn.utils.clip_grad_norm_(master,1.0))
    optimizer.step()
    config={"train_micro_batch_size_per_gpu":1,"gradient_accumulation_steps":accum,
        "train_batch_size":8,"bf16":{"enabled":True},
        "zero_optimization":{"stage":args.stage,"overlap_comm":False,"contiguous_gradients":True,
                             "reduce_bucket_size":50000000,"allgather_bucket_size":50000000},
        "optimizer":{"type":"AdamW","params":{"lr":1e-4,"betas":[.9,.95],"eps":1e-8,
                    "weight_decay":.01,"torch_adam":True,"foreach":False}},
        "gradient_clipping":1.0,"zero_allow_untested_optimizer":True,"steps_per_print":100000}
    with deepspeed.zero.Init(config_dict_or_path=config,enabled=args.stage==3,dtype=torch.bfloat16):
        model=Qwen2ForCausalLM(cfg).to(dtype=torch.bfloat16) if args.stage!=3 else Qwen2ForCausalLM(cfg)
    if args.stage==3:
        with deepspeed.zero.GatheredParameters(list(model.parameters()),modifier_rank=0):
            if rank==0:
                for n,p in model.named_parameters():p.data.copy_(initial[n])
    else:
        for n,p in model.named_parameters():p.data.copy_(initial[n].cpu())
    engine,_,_,_=deepspeed.initialize(model=model,model_parameters=model.parameters(),config=config)
    ds_losses=[]
    for index in range(rank,8,world):
        tokens=global_tokens[index:index+1]
        loss=engine(input_ids=tokens,labels=tokens).loss
        ds_losses.append(float(loss.detach()))
        engine.backward(loss)
        engine.step()
    diff_sq=0.; update_sq=0.; max_error=0.
    for (name,p),ref in zip(model.named_parameters(),master):
        actual=safe_get_full_fp32_param(p)
        assert actual is not None, name
        diff=actual.detach().float()-ref.detach()
        diff_sq+=float(diff.square().sum())
        update_sq+=float((ref.detach()-initial[name]).square().sum())
        max_error=max(max_error,float(diff.abs().max()))
    relative=(diff_sq/max(update_sq,1e-30))**.5
    ds_norm=float(engine.get_global_grad_norm())
    norm_error=abs(ds_norm-ref_norm)/max(ref_norm,1e-30)
    result={"stage":args.stage,"rank":rank,"world":world,"deepspeed":deepspeed.__version__,
        "torch":torch.__version__,"relative_update_l2_error":relative,
        "max_parameter_abs_error":max_error,"reference_gradient_norm":ref_norm,
        "deepspeed_gradient_norm":ds_norm,"gradient_norm_relative_error":norm_error,
        "reference_losses":reference_losses,"deepspeed_losses":ds_losses,
        "criterion":"relative update L2 error <= 0.05; gradient norm relative error <= 0.01; finite losses",
        "passed":relative<=.05 and norm_error<=.01 and all(torch.isfinite(torch.tensor(ds_losses)))}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/f"stage-{args.stage}-rank-{rank}.json").write_text(json.dumps(result,indent=2))
    if rank==0:print(json.dumps(result),flush=True)
    dist.destroy_process_group()
    if not result["passed"]:raise RuntimeError("optimizer update differs from explicit reference")


if __name__=="__main__":main()
