#!/usr/bin/env python3
"""
R1 generation-policy diagnostic.

Goal
----
The first Eager pilot showed 60/60 requests terminating by max_new_tokens.
That is not realistic enough to continue with R1.

This diagnostic compares, on a few already-measured pilot requests:

1) Hugging Face model.generate() with GREEDY decoding.
2) Hugging Face model.generate() with Qwen's native/recommended generation
   configuration loaded from the model repository.

Interpretation
--------------
- If greedy model.generate ALSO always hits the cap, but native Qwen generation
  stops at EOS, the issue is the greedy policy (likely repetitive degeneration).
- If greedy model.generate stops normally while the current manual R1 loop does
  not, the manual KV-cache decode loop is wrong.
- If both modes hit the cap, inspect the decoded text and generation config
  before changing the benchmark.

This script does not modify the frozen corpus or R1 result files.
"""

from __future__ import annotations

import json
import random
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

# Keep diagnosis reasonably fast. This is enough to see whether EOS appears.
DIAGNOSTIC_CAP = 256


def render_and_tokenize(tokenizer, prompt: str):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
    )


def eos_ids(tokenizer, model):
    values = set()

    for value in [
        tokenizer.eos_token_id,
        model.generation_config.eos_token_id,
    ]:
        if value is None:
            continue

        if isinstance(value, (list, tuple, set)):
            values.update(int(x) for x in value)
        else:
            values.add(int(value))

    return sorted(values)


def finish_reason(generated, eos_set, cap):
    ids = generated.tolist()

    if ids and ids[-1] in eos_set:
        return "eos"

    if len(ids) >= cap:
        return "length"

    return "other"


def preview(tokenizer, generated):
    text = tokenizer.decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    text = text.replace("\n", "\\n")

    if len(text) <= 700:
        return text

    return text[:450] + " ... <SNIP> ... " + text[-200:]


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

    return {
        key: getattr(gc, key, None)
        for key in keys
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    payload = json.loads(
        WORKLOAD_FILE.read_text(encoding="utf-8")
    )

    requests = payload["requests"]
    by_id = {
        row["request_id"]: row
        for row in requests
    }

    missing = [
        request_id
        for request_id in REQUEST_IDS
        if request_id not in by_id
    ]

    if missing:
        raise RuntimeError(
            f"Missing diagnostic request IDs: {missing}"
        )

    print("=" * 88)
    print("R1 GENERATION-POLICY DIAGNOSTIC")
    print("=" * 88)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda")

    model.eval()

    eos = eos_ids(tokenizer, model)
    eos_set = set(eos)

    print("\nLoaded generation config:")
    print(
        json.dumps(
            generation_config_summary(model),
            indent=2,
        )
    )
    print(f"EOS IDs used for diagnosis: {eos}")

    for index, request_id in enumerate(REQUEST_IDS):
        row = by_id[request_id]

        cap = min(
            int(row["max_new_tokens"]),
            DIAGNOSTIC_CAP,
        )

        encoded = render_and_tokenize(
            tokenizer,
            str(row["prompt"]),
        )

        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")

        print("\n" + "-" * 88)
        print(
            f"{request_id} | {row['workload_category']} | "
            f"input={row['input_tokens']} | diagnostic_cap={cap}"
        )

        # ------------------------------------------------------------------
        # A) HF generate, greedy
        # ------------------------------------------------------------------
        torch.manual_seed(2026 + index)
        torch.cuda.manual_seed_all(2026 + index)

        with torch.inference_mode():
            greedy_full = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=cap,
                eos_token_id=eos,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        greedy_generated = greedy_full[
            0,
            input_ids.shape[-1]:,
        ].detach().cpu()

        print(
            f"[HF greedy] "
            f"out={len(greedy_generated):4d} "
            f"finish={finish_reason(greedy_generated, eos_set, cap)}"
        )
        print(
            "[HF greedy preview] "
            + preview(tokenizer, greedy_generated)
        )

        # ------------------------------------------------------------------
        # B) HF generate, model-native generation config
        # ------------------------------------------------------------------
        torch.manual_seed(2026 + index)
        torch.cuda.manual_seed_all(2026 + index)

        with torch.inference_mode():
            native_full = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cap,
                use_cache=True,
            )

        native_generated = native_full[
            0,
            input_ids.shape[-1]:,
        ].detach().cpu()

        print(
            f"[HF native] "
            f"out={len(native_generated):4d} "
            f"finish={finish_reason(native_generated, eos_set, cap)}"
        )
        print(
            "[HF native preview] "
            + preview(tokenizer, native_generated)
        )

        del (
            input_ids,
            attention_mask,
            greedy_full,
            greedy_generated,
            native_full,
            native_generated,
        )

    print("\n" + "=" * 88)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 88)
    print(
        "Send the terminal output back before running the Flash pilot."
    )


if __name__ == "__main__":
    main()
