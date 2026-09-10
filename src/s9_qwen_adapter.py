"""Inference-only RMSNorm preserving Transformers Qwen2's intermediate cast.

The microbenchmark uses a single final cast. Qwen2 first casts the normalized
activation to BF16/FP16, then multiplies by the weight. Keep this distinction
explicit rather than silently changing the pretrained model's arithmetic.
"""
import types
import torch
import triton
import triton.language as tl


@triton.jit
def _qwen_rms(X, W, Y, H: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + row * H + c, c < H, 0).to(tl.float32)
    w = tl.load(W + c, c < H, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    normalized = (x * r).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * H + c, normalized * w, c < H)


def qwen_rms(x, w, eps):
    if torch.is_grad_enabled():
        raise RuntimeError("Qwen adapter is inference-only; use inference_mode")
    x = x.contiguous()
    h = x.shape[-1]
    y = torch.empty_like(x)
    _qwen_rms[(x.numel() // h,)](x, w, y, h, eps,
                               triton.next_power_of_2(h), num_warps=4)
    return y


def set_backend(model, backend):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    count = 0
    for module in model.modules():
        if isinstance(module, Qwen2RMSNorm):
            if not hasattr(module, "_s9_original_forward"):
                module._s9_original_forward = module.forward
            if backend == "eager":
                module.forward = module._s9_original_forward
            elif backend == "triton":
                def forward(self, x):
                    return qwen_rms(x, self.weight, self.variance_epsilon)
                module.forward = types.MethodType(forward, module)
            else:
                raise ValueError(backend)
            count += 1
    if not count:
        raise ValueError("no Qwen2RMSNorm modules found")
    return count
