"""CPU-only S10 matrix validation and memory planning."""
def accumulation(global_tokens, sequence, microbatch, data_parallel):
    divisor=sequence*microbatch*data_parallel
    if min(global_tokens,sequence,microbatch,data_parallel)<=0 or global_tokens%divisor:
        raise ValueError("global tokens must divide exactly across sequence, microbatch and DP")
    return global_tokens//divisor


def parallel_degrees(world,tp,pp,layers,heads):
    if min(world,tp,pp)<=0 or world%(tp*pp) or layers%pp or heads%tp:
        raise ValueError("invalid world/TP/PP/layers/heads factorization")
    return world//(tp*pp)


def zero_state_bytes(parameters,world,stage):
    if world<1 or stage not in range(4):
        raise ValueError("invalid ZeRO configuration")
    # BF16 model/grad + FP32 master/Adam moments; excludes activations, buckets,
    # allocator and framework-specific duplicate buffers.
    return {"parameters":2*parameters/(world if stage==3 else 1),
            "gradients":2*parameters/(world if stage>=2 else 1),
            "master_and_adam":12*parameters/(world if stage>=1 else 1)}
