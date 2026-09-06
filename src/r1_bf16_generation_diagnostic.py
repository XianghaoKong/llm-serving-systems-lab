#!/usr/bin/env python3
"""
R1 BF16 generation diagnostic.

Why
---
The first R1 Eager pilot forced Qwen2.5-1.5B-Instruct to FP16 and every request
hit max_new_tokens. A follow-up FP16 diagnostic showed:
- greedy decoding degenerating into repeated "!" tokens
- native Qwen sampling failing with invalid probability values

Qwen2.5-1.5B-Instruct's model config declares bfloat16 as its native dtype.
This script retests the SAME six pilot requests in BF16 using the official
chat-template path.

It does NOT modify the frozen corpus or any R1 result file.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKLOAD_FILE = PROJECT_ROOT / "workloads" / "final" / "realistic_requests.json"

REQUEST_IDS = [
    "req_0102",  # short_interactive
    "req_0382",  # knowledge_qa
    "req_0794",  # document_qa
    "req_0891",  # long_context_qa
    "req_0553",  # coding_request
    "req_0994",  # long_output
]

DIAGNOSTIC_CAP = 256


def generation_config_summary(model):
    gc = model.generation_config
    keys = [
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "eos_token_id",
        "pad_token_id",
        "bos_token_id",
    ]
    return {key: getattr(gc, key, None) for key in keys}


def prepare_inputs(tokenizer, prompt: str):
    # Official Qwen-style path: tokenize directly through the chat template.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def eos_set_from_model(tokenizer, model):
    eos = set()

    for value in [
        tokenizer.eos_token_id,
        model.generation_config.eos_token_id,
    ]:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            eos.update(int(x) for x in value)
        else:
            eos.add(int(value))

    return eos


def finish_reason(generated: torch.Tensor, eos_ids: set, cap: int) -> str:
    ids = generated.tolist()

    if ids and ids[-1] in eos_ids:
        return "eos"

    if len(ids) >= cap:
        return "length"

    return "other"


def preview(tokenizer, generated: torch.Tensor) -> str:
    text = tokenizer.decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).replace("\n", "\\n")

    if len(text) <= 700:
        return text

    return text[:450] + " ... <SNIP> ... " + text[-200:]


def first_step_logits_check(model, inputs):
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )

    logits = outputs.logits[:, -1, :]

    result = {
        "dtype": str(logits.dtype),
        "all_finite": bool(torch.isfinite(logits).all().item()),
        "nan_count": int(torch.isnan(logits).sum().item()),
        "inf_count": int(torch.isinf(logits).sum().item()),
        "min": float(logits.float().min().item()),
        "max": float(logits.float().max().item()),
    }

    del outputs, logits
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    bf16_supported = bool(torch.cuda.is_bf16_supported())

    print("=" * 88)
    print("R1 BF16 GENERATION DIAGNOSTIC")
    print("=" * 88)
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"BF16 supported:  {bf16_supported}")

    if not bf16_supported:
        raise RuntimeError(
            "This GPU/PyTorch build does not report BF16 support."
        )

    payload = json.loads(
        WORKLOAD_FILE.read_text(encoding="utf-8")
    )
    by_id = {
        row["request_id"]: row
        for row in payload["requests"]
    }

    missing = [rid for rid in REQUEST_IDS if rid not in by_id]
    if missing:
        raise RuntimeError(f"Missing request IDs: {missing}")

    print("\nLoading tokenizer and BF16 model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")

    model.eval()

    print(f"Model parameter dtype: {next(model.parameters()).dtype}")
    print("\nNative generation config:")
    print(json.dumps(generation_config_summary(model), indent=2))

    eos_ids = eos_set_from_model(tokenizer, model)
    print(f"EOS IDs: {sorted(eos_ids)}")

    native_eos_count = 0
    native_length_count = 0
    greedy_length_count = 0

    for index, request_id in enumerate(REQUEST_IDS):
        row = by_id[request_id]
        cap = min(int(row["max_new_tokens"]), DIAGNOSTIC_CAP)

        inputs_cpu = prepare_inputs(
            tokenizer,
            str(row["prompt"]),
        )

        runtime_tokens = int(inputs_cpu["input_ids"].shape[-1])
        frozen_tokens = int(row["input_tokens"])

        if runtime_tokens != frozen_tokens:
            raise RuntimeError(
                f"{request_id}: token-count drift "
                f"(frozen={frozen_tokens}, runtime={runtime_tokens})"
            )

        inputs = {
            key: value.to("cuda")
            for key, value in inputs_cpu.items()
            if torch.is_tensor(value)
        }

        print("\n" + "-" * 88)
        print(
            f"{request_id} | {row['workload_category']} | "
            f"input={runtime_tokens} | diagnostic_cap={cap}"
        )

        logits_info = first_step_logits_check(model, inputs)
        print("[BF16 first-step logits]")
        print(json.dumps(logits_info, indent=2))

        if not logits_info["all_finite"]:
            raise RuntimeError(
                f"{request_id}: BF16 first-step logits are not finite."
            )

        # ------------------------------------------------------------------
        # A) Qwen native generation config FIRST.
        # ------------------------------------------------------------------
        torch.manual_seed(2026 + index)
        torch.cuda.manual_seed_all(2026 + index)

        with torch.inference_mode():
            native_full = model.generate(
                **inputs,
                max_new_tokens=cap,
                use_cache=True,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        native_generated = native_full[0, prompt_len:].detach().cpu()
        native_finish = finish_reason(native_generated, eos_ids, cap)

        if native_finish == "eos":
            native_eos_count += 1
        elif native_finish == "length":
            native_length_count += 1

        print(
            f"[BF16 HF native] "
            f"out={len(native_generated):4d} "
            f"finish={native_finish}"
        )
        print(
            "[BF16 HF native preview] "
            + preview(tokenizer, native_generated)
        )

        # ------------------------------------------------------------------
        # B) Greedy in BF16, to distinguish dtype from decoding-policy effects.
        # ------------------------------------------------------------------
        with torch.inference_mode():
            greedy_full = model.generate(
                **inputs,
                do_sample=False,
                repetition_penalty=1.0,
                max_new_tokens=cap,
                eos_token_id=sorted(eos_ids),
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        greedy_generated = greedy_full[0, prompt_len:].detach().cpu()
        greedy_finish = finish_reason(greedy_generated, eos_ids, cap)

        if greedy_finish == "length":
            greedy_length_count += 1

        print(
            f"[BF16 HF greedy] "
            f"out={len(greedy_generated):4d} "
            f"finish={greedy_finish}"
        )
        print(
            "[BF16 HF greedy preview] "
            + preview(tokenizer, greedy_generated)
        )

        del (
            inputs_cpu,
            inputs,
            native_full,
            native_generated,
            greedy_full,
            greedy_generated,
        )

    print("\n" + "=" * 88)
    print("BF16 DIAGNOSTIC COMPLETE")
    print("=" * 88)
    print(
        f"Native generation: EOS={native_eos_count}/6, "
        f"LENGTH={native_length_count}/6"
    )
    print(
        f"Greedy generation: LENGTH={greedy_length_count}/6"
    )
    print()
    print(
        "Do not rerun R1 yet. Send this output back so the final R1 "
        "dtype + decoding policy can be locked."
    )


if __name__ == "__main__":
    main()
