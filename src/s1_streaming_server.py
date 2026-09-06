#!/usr/bin/env python3
"""
S1 — Streaming inference API server.

Goal
----
Wrap the validated R1 Flash/BF16 single-worker inference path in a persistent
HTTP/SSE service so we can measure real client-observed TTFT, streaming ITL,
and API overhead without yet introducing intentional load/concurrency.

Protocol
--------
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Dtype: BF16
- Backend: PyTorch SDPA Flash (forced with SDPBackend.FLASH_ATTENTION)
- Sampling: temperature=0.7, top_p=0.8, top_k=20,
  repetition_penalty=1.1
- Deterministic per-request sampling seed, identical to R1
- Persistent model on GPU
- No torch.cuda.empty_cache() between requests
- One inference worker protected by an asyncio lock
- SSE streaming over HTTP/1.1

Endpoint
--------
POST /v1/chat/completions

Request shape (OpenAI-like):
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "messages": [{"role": "user", "content": "..."}],
  "stream": true,
  "max_tokens": 256,
  "request_id": "req_0001"
}

The final SSE event contains server-side metrics. Generated text is streamed
to the client but is never written by this server to disk.

Run
---
python src/s1_streaming_server.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Sequence

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TextStreamer,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.bfloat16

TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.1
SAMPLING_SEED_BASE = 42026000

SAMPLING_PIPELINE = (
    RepetitionPenaltyLogitsProcessor(REPETITION_PENALTY),
    TemperatureLogitsWarper(TEMPERATURE),
    TopKLogitsWarper(TOP_K),
    TopPLogitsWarper(TOP_P),
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[ChatMessage]
    stream: bool = True
    max_tokens: int = Field(default=256, ge=1, le=4096)
    request_id: str


def output_token_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(x) for x in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def request_sampling_seed(request_id: str) -> int:
    digits = ""
    for ch in reversed(str(request_id)):
        if ch.isdigit():
            digits = ch + digits
        else:
            break

    if not digits:
        raise ValueError(
            f"Benchmark request_id must end in digits: {request_id!r}"
        )

    return SAMPLING_SEED_BASE + int(digits)


def gib(n_bytes: int) -> float:
    return n_bytes / (1024.0 ** 3)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None

    vals = sorted(float(v) for v in values)

    if len(vals) == 1:
        return vals[0]

    pos = (len(vals) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return vals[lower]

    weight = pos - lower
    return vals[lower] * (1.0 - weight) + vals[upper] * weight


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(float(v) for v in values) / len(values)


def sse(data: Any) -> str:
    if data == "[DONE]":
        return "data: [DONE]\n\n"

    return (
        "data: "
        + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        + "\n\n"
    )


class PieceStreamer(TextStreamer):
    """
    Use the same incremental tokenizer logic as R1, but capture finalized text
    instead of printing it.
    """

    def __init__(self, tokenizer):
        super().__init__(
            tokenizer,
            skip_prompt=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.last_piece = ""
        self.total_chars = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.last_piece = text
        self.total_chars += len(text)

    def put_token_id(self, token_id: int) -> str:
        self.last_piece = ""
        token_tensor = torch.tensor([token_id], dtype=torch.long)
        super().put(token_tensor)
        return self.last_piece

    def finish(self) -> str:
        self.last_piece = ""
        super().end()
        return self.last_piece


class InferenceEngine:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.eos_ids: set[int] = set()
        self.lock = asyncio.Lock()
        self.loaded = False

    def attention_context(self):
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except Exception as exc:
            raise RuntimeError(
                "PyTorch does not expose torch.nn.attention.sdpa_kernel."
            ) from exc

        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)

    def sample_next_token(
        self,
        *,
        logits: torch.Tensor,
        sequence_ids: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        scores = logits

        for processor in SAMPLING_PIPELINE:
            scores = processor(sequence_ids, scores)

        probs = torch.softmax(scores, dim=-1)

        return torch.multinomial(
            probs,
            num_samples=1,
            generator=generator,
        ).squeeze(1)

    def render_chat(self, messages: List[ChatMessage]) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def tokenize_rendered(self, rendered: str):
        return self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def determine_eos_ids(self) -> set[int]:
        values: set[int] = set()

        candidates = [
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.model.generation_config, "eos_token_id", None),
        ]

        for value in candidates:
            if value is None:
                continue

            if isinstance(value, (list, tuple, set)):
                values.update(int(x) for x in value)
            else:
                values.add(int(value))

        if not values:
            raise RuntimeError("Could not determine EOS token IDs.")

        return values

    def load(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")

        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("S1 protocol requires BF16 support.")

        print("=" * 88)
        print("S1 STREAMING INFERENCE SERVER")
        print("=" * 88)
        print(f"Model:       {MODEL_ID}")
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        print(f"Dtype:       {DTYPE}")
        print("Backend:     forced PyTorch SDPA Flash")
        print(
            "Generation:  "
            f"T={TEMPERATURE}, top_p={TOP_P}, top_k={TOP_K}, "
            f"rep_penalty={REPETITION_PENALTY}"
        )

        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_fast=True,
        )

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=DTYPE,
            attn_implementation="sdpa",
        ).to("cuda")

        self.model.eval()
        self.eos_ids = self.determine_eos_ids()

        print(f"EOS IDs:     {sorted(self.eos_ids)}")

        # Real forward under forced Flash. If unsupported, PyTorch should raise.
        print("\nVerifying forced Flash backend...")
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")

        with torch.inference_mode(), self.attention_context():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            _ = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())

        del outputs, input_ids, attention_mask
        torch.cuda.synchronize()

        # One startup cleanup is allowed; never empty cache between requests.
        torch.cuda.empty_cache()

        self.loaded = True
        print("Forced Flash verification: PASS")
        print("Server engine ready.\n")

    async def stream_request(
        self,
        payload: ChatCompletionRequest,
        http_request: Request,
        endpoint_start: float,
    ):
        if not self.loaded:
            raise RuntimeError("Inference engine is not loaded.")

        request_id = payload.request_id

        async with self.lock:
            lock_acquired = time.perf_counter()
            queue_ms = (lock_acquired - endpoint_start) * 1000.0

            request_start = endpoint_start

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            start_allocated = torch.cuda.memory_allocated()
            start_reserved = torch.cuda.memory_reserved()

            # --------------------------------------------------------------
            # Template + tokenization
            # --------------------------------------------------------------
            t0 = time.perf_counter()
            rendered = self.render_chat(payload.messages)
            t1 = time.perf_counter()

            encoded_cpu = self.tokenize_rendered(rendered)
            t2 = time.perf_counter()

            template_ms = (t1 - t0) * 1000.0
            tokenization_ms = (t2 - t1) * 1000.0
            preprocess_ms = (t2 - t0) * 1000.0

            runtime_input_tokens = int(
                encoded_cpu["input_ids"].shape[-1]
            )

            # --------------------------------------------------------------
            # H2D
            # --------------------------------------------------------------
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)

            h2d_start.record()
            input_ids = encoded_cpu["input_ids"].to("cuda")
            attention_mask = encoded_cpu.get("attention_mask")

            if attention_mask is None:
                attention_mask = torch.ones_like(encoded_cpu["input_ids"])

            attention_mask = attention_mask.to("cuda")
            h2d_end.record()

            sampling_seed = request_sampling_seed(request_id)
            sampling_generator = torch.Generator(device="cuda")
            sampling_generator.manual_seed(sampling_seed)

            # --------------------------------------------------------------
            # Prefill + first token
            # --------------------------------------------------------------
            prefill_start = torch.cuda.Event(enable_timing=True)
            prefill_end = torch.cuda.Event(enable_timing=True)

            prefill_start.record()

            with torch.inference_mode(), self.attention_context():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )

                next_token = self.sample_next_token(
                    logits=outputs.logits[:, -1, :],
                    sequence_ids=input_ids,
                    generator=sampling_generator,
                )

            prefill_end.record()

            first_token_id = int(next_token.item())

            h2d_ms = float(h2d_start.elapsed_time(h2d_end))
            model_ttft_ms = float(
                prefill_start.elapsed_time(prefill_end)
            )

            post_prefill_allocated = torch.cuda.memory_allocated()
            post_prefill_reserved = torch.cuda.memory_reserved()

            streamer = PieceStreamer(self.tokenizer)

            cpu_stream_start = time.perf_counter()
            first_piece = streamer.put_token_id(first_token_id)
            first_piece_ready = time.perf_counter()
            first_stream_cpu_ms = (
                first_piece_ready - cpu_stream_start
            ) * 1000.0

            server_request_ttft_ms = (
                first_piece_ready - request_start
            ) * 1000.0

            output_ids = [first_token_id]
            decode_gpu_steps: List[float] = []
            stream_itls: List[float] = []
            previous_piece_ready = first_piece_ready

            past_key_values = outputs.past_key_values
            current_token = next_token.view(1, 1)
            sequence_ids = torch.cat(
                [input_ids, current_token],
                dim=-1,
            )

            ended_with_eos = first_token_id in self.eos_ids

            first_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": first_piece},
                        "finish_reason": None,
                    }
                ],
                "_metrics": {
                    "token_index": 1,
                    "server_request_ttft_ms": server_request_ttft_ms,
                    "model_ttft_ms": model_ttft_ms,
                },
            }

            yield sse(first_chunk)

            # --------------------------------------------------------------
            # Decode
            # --------------------------------------------------------------
            while (
                not ended_with_eos
                and len(output_ids) < payload.max_tokens
            ):
                if await http_request.is_disconnected():
                    break

                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )

                decode_start = torch.cuda.Event(enable_timing=True)
                decode_end = torch.cuda.Event(enable_timing=True)

                decode_start.record()

                with torch.inference_mode(), self.attention_context():
                    outputs = self.model(
                        input_ids=current_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

                    next_token = self.sample_next_token(
                        logits=outputs.logits[:, -1, :],
                        sequence_ids=sequence_ids,
                        generator=sampling_generator,
                    )

                decode_end.record()

                token_id = int(next_token.item())
                gpu_step_ms = float(
                    decode_start.elapsed_time(decode_end)
                )

                cpu_stream_start = time.perf_counter()
                piece = streamer.put_token_id(token_id)
                piece_ready = time.perf_counter()

                stream_cpu_ms = (
                    piece_ready - cpu_stream_start
                ) * 1000.0

                stream_itl_ms = (
                    piece_ready - previous_piece_ready
                ) * 1000.0

                previous_piece_ready = piece_ready

                output_ids.append(token_id)
                decode_gpu_steps.append(gpu_step_ms)
                stream_itls.append(stream_itl_ms)

                ended_with_eos = token_id in self.eos_ids

                token_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }
                    ],
                    "_metrics": {
                        "token_index": len(output_ids),
                        "gpu_step_ms": gpu_step_ms,
                        "server_stream_itl_ms": stream_itl_ms,
                        "stream_cpu_ms": stream_cpu_ms,
                    },
                }

                yield sse(token_chunk)

                past_key_values = outputs.past_key_values
                current_token = next_token.view(1, 1)
                sequence_ids = torch.cat(
                    [sequence_ids, current_token],
                    dim=-1,
                )

            flush_piece = streamer.finish()

            if flush_piece:
                yield sse({
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": flush_piece},
                            "finish_reason": None,
                        }
                    ],
                    "_metrics": {"flush": True},
                })

            response_ready = time.perf_counter()

            output_tokens = len(output_ids)
            hit_max = (
                not ended_with_eos
                and output_tokens >= payload.max_tokens
            )

            if ended_with_eos:
                finish_reason = "stop"
            elif hit_max:
                finish_reason = "length"
            else:
                finish_reason = "client_disconnect"

            final_allocated = torch.cuda.memory_allocated()
            final_reserved = torch.cuda.memory_reserved()
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()

            server_e2e_ms = (
                response_ready - request_start
            ) * 1000.0

            decode_total_gpu_ms = sum(decode_gpu_steps)

            done_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": runtime_input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": runtime_input_tokens + output_tokens,
                },
                "_server_metrics": {
                    "queue_ms": queue_ms,
                    "template_ms": template_ms,
                    "tokenization_ms": tokenization_ms,
                    "preprocess_ms": preprocess_ms,
                    "h2d_ms": h2d_ms,
                    "model_ttft_ms": model_ttft_ms,
                    "first_stream_cpu_ms": first_stream_cpu_ms,
                    "server_request_ttft_ms": server_request_ttft_ms,
                    "decode_total_gpu_ms": decode_total_gpu_ms,
                    "mean_tpot_ms": mean_or_none(decode_gpu_steps),
                    "p50_tpot_ms": percentile(decode_gpu_steps, 0.50),
                    "p95_tpot_ms": percentile(decode_gpu_steps, 0.95),
                    "mean_server_stream_itl_ms": mean_or_none(stream_itls),
                    "server_e2e_ms": server_e2e_ms,
                    "sampling_seed": sampling_seed,
                    "output_hash": output_token_hash(output_ids),
                    "start_allocated_gib": gib(start_allocated),
                    "start_reserved_gib": gib(start_reserved),
                    "post_prefill_allocated_gib": gib(
                        post_prefill_allocated
                    ),
                    "post_prefill_reserved_gib": gib(
                        post_prefill_reserved
                    ),
                    "final_allocated_gib": gib(final_allocated),
                    "final_reserved_gib": gib(final_reserved),
                    "peak_allocated_gib": gib(peak_allocated),
                    "peak_reserved_gib": gib(peak_reserved),
                },
            }

            yield sse(done_chunk)
            yield sse("[DONE]")

            del (
                encoded_cpu,
                input_ids,
                attention_mask,
                outputs,
                past_key_values,
                current_token,
                next_token,
                sequence_ids,
                sampling_generator,
            )


ENGINE = InferenceEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ENGINE.load()
    yield


app = FastAPI(
    title="S1 LLM Streaming Benchmark Server",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok" if ENGINE.loaded else "loading",
        "model": MODEL_ID,
        "dtype": str(DTYPE),
        "backend": "flash_sdpa_forced",
        "worker_concurrency": 1,
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
):
    endpoint_start = time.perf_counter()

    if payload.model != MODEL_ID:
        raise HTTPException(
            status_code=400,
            detail=f"Only model {MODEL_ID!r} is loaded.",
        )

    if not payload.stream:
        raise HTTPException(
            status_code=400,
            detail="S1 only benchmarks stream=true.",
        )

    if not payload.messages:
        raise HTTPException(
            status_code=400,
            detail="messages must not be empty.",
        )

    return StreamingResponse(
        ENGINE.stream_request(
            payload,
            request,
            endpoint_start,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
