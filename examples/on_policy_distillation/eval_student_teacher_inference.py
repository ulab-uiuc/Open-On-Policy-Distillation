#!/usr/bin/env python3
"""
Standalone inference-only evaluation for student and teacher models via an
OpenAI-compatible chat completions API.
"""


'''
python examples/on_policy_distillation/eval_student_teacher_inference.py \
  --model /root/checkpoints_siqi/Qwen3-1.7B \
  --teacher-model /root/checkpoints_siqi/Qwen3-1.7B \
  --student-api-base http://127.0.0.1:30000/v1 \
  --teacher-api-base http://127.0.0.1:30000/v1 \
  --student-api-key EMPTY \
  --teacher-api-key EMPTY \
  --dataset /root/math/data/train_dapo.jsonl \
  --teacher-info-mode answer_only \
  --max-new-tokens 32768 \
  --n-samples 32 \
  --concurrency 256 \
  --seed 42

  
  /root/math/data/train_dapo.jsonl
  /root/math/data/train_openthoughts_math.jsonl


python examples/on_policy_distillation/eval_student_teacher_inference.py \
  --model /root/checkpoints_siqi/Qwen3-1.7B \
  --teacher-model /root/checkpoints_siqi/Qwen3-1.7B \
  --student-api-base http://127.0.0.1:30000/v1 \
  --teacher-api-base http://127.0.0.1:30000/v1 \
  --student-api-key EMPTY \
  --teacher-api-key EMPTY \
  --dataset /root/math/data/train_dapo.jsonl \
  --teacher-info-mode answer_only \
  --max-new-tokens 32768 \
  --n-samples 512 \
  --concurrency 256 \
  --seed 42 \
  --score-concurrency 1 \
  --score-chunk-tokens 256 \
  --output ./eval_math500_student_teacher_inference_s1.7t1.7b_answeronly_b512.jsonl \
#   --score-context-window-tokens 2048 \
#   --max-score-response-tokens 2048


python examples/on_policy_distillation/eval_student_teacher_inference.py \
  --model /root/checkpoints_siqi/Qwen3-1.7B \
  --teacher-model /root/checkpoints_siqi/Qwen3-1.7B \
  --student-api-base http://127.0.0.1:30000/v1 \
  --teacher-api-base http://127.0.0.1:30000/v1 \
  --student-api-key EMPTY \
  --teacher-api-key EMPTY \
  --dataset /root/math/data/train_dapo.jsonl \
  --teacher-info-mode answer_only \
  --max-new-tokens 32768 \
  --n-samples 512 \
  --concurrency 256 \
  --seed 42 \
  --score-concurrency 1 \
  --score-chunk-tokens 256 \
  --output ./eval_math500_student_teacher_inference_s1.7t1.7b_answeronly_disablethinking_b512.jsonl \
  --student-enable-thinking false

python examples/on_policy_distillation/eval_student_teacher_inference.py \
  --model /root/checkpoints_siqi/Qwen3-1.7B \
  --teacher-model /root/checkpoints_siqi/Qwen3-8B \
  --student-api-base http://127.0.0.1:30000/v1 \
  --teacher-api-base http://127.0.0.1:30001/v1 \
  --student-api-key EMPTY \
  --teacher-api-key EMPTY \
  --dataset /root/math/data/train_dapo.jsonl \
  --teacher-info-mode same_as_student \
  --max-new-tokens 32768 \
  --n-samples 512 \
  --concurrency 256 \
  --seed 42 \
  --score-concurrency 1 \
  --score-chunk-tokens 256 \
  --output ./eval_math500_student_teacher_inference_s1.7t8b_noanswer_b512.jsonl

  


python examples/on_policy_distillation/eval_student_teacher_inference.py \
  --model /root/checkpoints_siqi/Qwen3-1.7B \
  --teacher-model /root/checkpoints_siqi/Qwen3-8B \
  --student-api-base http://127.0.0.1:30002/v1 \
  --teacher-api-base http://127.0.0.1:30001/v1 \
  --student-api-key EMPTY \
  --teacher-api-key EMPTY \
  --dataset /root/math/data/train_dapo.jsonl \
  --teacher-info-mode same_as_student \
  --max-new-tokens 32768 \
  --n-samples 512 \
  --concurrency 32 \
  --seed 42 \
  --score-concurrency 2 \
  --score-chunk-tokens 256 \
  --output ./eval_math500_student_teacher_inference_s1.7t8b_same_as_student_b512.jsonl

  

python examples/on_policy_distillation/eval_student_teacher_inference.py \
  --model /root/checkpoints_siqi/Qwen3-8B \
  --teacher-model /root/checkpoints_siqi/Qwen3-8B \
  --student-api-base http://127.0.0.1:30001/v1 \
  --teacher-api-base http://127.0.0.1:30001/v1 \
  --student-api-key EMPTY \
  --teacher-api-key EMPTY \
  --dataset /root/math/data/train_dapo.jsonl \
  --teacher-info-mode answer_only \
  --max-new-tokens 64 \
  --n-samples 32 \
  --concurrency 256 \
  --seed 42 \
  --score-concurrency 1 \
  --score-chunk-tokens 256 \
  --output ./eval_math500_student_teacher_inference_s8t8b_answeronly.jsonl


./eval_math500_student_teacher_inference_s8t8b_answeronly.jsonl


python examples/on_policy_distillation/eval_student_teacher_inference.py \
  --model /root/checkpoints_siqi/Qwen3-8B \
  --teacher-model /root/checkpoints_siqi/Qwen3-8B \
  --student-api-base http://127.0.0.1:30001/v1 \
  --teacher-api-base http://127.0.0.1:30001/v1 \
  --student-api-key EMPTY \
  --teacher-api-key EMPTY \
  --dataset /root/math/data/train_dapo.jsonl \
  --teacher-info-mode answer_only \
  --max-new-tokens 64 \
  --n-samples 32 \
  --concurrency 256 \
  --seed 42 \
  --score-concurrency 1 \
  --score-chunk-tokens 256 \
  --output ./eval_math500_student_teacher_inference_s8t8b_answeronly.jsonl


python examples/on_policy_distillation/eval_student_teacher_inference.py \
  --model Qwen/Qwen3-4B \
  --teacher-model Qwen/Qwen3-4B \
  --student-api-base http://127.0.0.1:30001/v1 \
  --teacher-api-base http://127.0.0.1:30001/v1 \
  --student-api-key EMPTY \
  --teacher-api-key EMPTY \
  --dataset /root/math/data/train_dapo.jsonl \
  --teacher-info-mode answer_only \
  --max-new-tokens 64 \
  --n-samples 32 \
  --concurrency 256 \
  --seed 42 \
  --score-concurrency 1 \
  --score-chunk-tokens 256 \
  --output ./eval_math500_student_teacher_inference_s4t4b_answeronly.jsonl

'''
import argparse
import asyncio
import copy
import json
import logging
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import aiohttp
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
_ENABLE_THINKING_FALLBACK_WARNED = False

_DEFAULT_FORMAT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
_ANSWER_ONLY_PROMPT = (
    "The correct final answer to this problem is: {answer}\n"
    "Now solve the problem yourself step by step and arrive at the same answer:"
)
_MASKED_TRANSITION_PROMPT = (
    "After understanding the reference solution and the rationale behind each step, "
    "now articulate your own step-by-step reasoning that derives the same final answer "
    "to the problem above:"
)
_FULL_TRANSITION_PROMPT = _MASKED_TRANSITION_PROMPT
_DEFAULT_CONCISENESS_INSTRUCTION = (
    "Solve the following math problem concisely and correctly. "
    "Be direct -- avoid unnecessary elaboration, redundant steps, or restating the problem. "
    "Focus only on the key reasoning steps needed to reach the answer."
)


def _first_nonempty(*values):
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            return s
    return ""


def _infer_answer_format(metadata: dict) -> str:
    fmt = metadata.get("format_instruction", "") or ""
    if "boxed" in fmt:
        return "boxed"
    if "Answer" in fmt:
        return "answer"
    return "auto"


def grade(response: str, label: str, answer_format: str) -> float:
    from slime.rollout.rm_hub import grade_answer_verl

    if not label:
        return 0.0
    try:
        return 1.0 if grade_answer_verl(response, label, mode=answer_format) else 0.0
    except TypeError as e:
        if "unexpected keyword argument 'mode'" not in str(e):
            raise
        logger.warning(
            "grade_answer_verl does not accept `mode`; falling back to the legacy 2-arg call."
        )
        return 1.0 if grade_answer_verl(response, label) else 0.0


def load_dataset(path: str, n_samples: int | None, seed: int) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    total = len(samples)
    if n_samples is not None and n_samples < total:
        rng = random.Random(seed)
        samples = rng.sample(samples, n_samples)
        logger.info(f"Sampled {n_samples} of {total} examples (seed={seed})")
    else:
        logger.info(f"Loaded {total} examples from {path}")
    return samples


def extract_user_content(row: dict) -> str:
    prompt_field = row.get("prompt", [])
    if isinstance(prompt_field, list):
        return next(
            (m["content"] for m in prompt_field if isinstance(m, dict) and m.get("role") == "user"),
            "",
        )
    return str(prompt_field)


def build_student_messages(row: dict) -> list[dict]:
    prompt_field = row.get("prompt", [])
    if isinstance(prompt_field, list):
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in prompt_field
            if isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"}
        ]
        if messages:
            return messages
    return [{"role": "user", "content": extract_user_content(row)}]


def maybe_load_tokenizer(args):
    if args.teacher_info_mode not in {"hidden_think", "hidden_think_full", "masked_reasoning"}:
        return None
    from transformers import AutoTokenizer

    tokenizer_source = args.teacher_model or args.model
    logger.info(f"Loading tokenizer from {tokenizer_source} for teacher prompt construction")
    return AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)


def build_teacher_user_content(metadata: dict, label: str, mode: str, args) -> str | None:
    if mode == "none":
        return None
    raw_content = metadata.get("raw_content", "") or ""
    student_user_content = metadata.get("student_user_content") or raw_content
    format_instruction = metadata.get("format_instruction") or _DEFAULT_FORMAT_SUFFIX
    reference_solution = _first_nonempty(
        metadata.get("reference_solution"),
        metadata.get("solution"),
        label,
    )

    if mode == "answer_only":
        answer_hint = label if label else reference_solution
        return f"{student_user_content}\n\n" + _ANSWER_ONLY_PROMPT.format(answer=answer_hint) if answer_hint else student_user_content
    if mode == "pi":
        pi_instruction = metadata.get("pi_instruction", "")
        return f"{pi_instruction}\n\n{student_user_content}" if pi_instruction else student_user_content
    if mode == "conciseness":
        return f"{args.conciseness_instruction}\n\n{student_user_content}"
    if mode == "full":
        if reference_solution:
            return (
                f"{raw_content}\n\n"
                f"Here is a reference solution to this problem:\n{reference_solution}\n"
                f"{_FULL_TRANSITION_PROMPT}\n"
                f"{format_instruction}"
            )
        return f"{raw_content}\n\n{format_instruction}"
    if mode == "masked_reasoning":
        if reference_solution and args.mask_ratio > 0.0:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            token_ids = tok.encode(reference_solution, add_special_tokens=False)
            mask_token = getattr(tok, "mask_token", None) or "[MASK]"
            masked = "".join(
                mask_token if random.random() < args.mask_ratio else tok.decode([t], skip_special_tokens=True)
                for t in token_ids
            )
        else:
            masked = reference_solution
        if masked:
            return (
                f"{raw_content}\n\n"
                f"Here is a reference solution to this problem:\n{masked}\n"
                f"{_MASKED_TRANSITION_PROMPT}\n"
                f"{format_instruction}"
            )
        return f"{raw_content}\n\n{format_instruction}"
    raise ValueError(f"Unsupported teacher-info-mode: {mode!r}")


def build_teacher_prefill_content(
    metadata: dict,
    label: str,
    mode: str,
    tokenizer,
    max_think_tokens: int,
) -> str | None:
    student_user_content = metadata.get("student_user_content") or metadata.get("raw_content", "")
    if not student_user_content:
        return None

    if mode == "hidden_think":
        answer_hint = _first_nonempty(
            label,
            metadata.get("reference_solution"),
            metadata.get("solution"),
        )
        if answer_hint:
            fmt = metadata.get("format_instruction", "") or ""
            formatted_answer = f"\\boxed{{{answer_hint}}}" if "boxed" in fmt else answer_hint
            think_content = f"The answer to this problem is {formatted_answer}."
        else:
            think_content = ""
    elif mode == "hidden_think_full":
        think_content = _first_nonempty(
            metadata.get("reference_solution"),
            metadata.get("solution"),
            label,
        )
    else:
        return None

    if max_think_tokens > 0 and think_content:
        think_ids = tokenizer.encode(think_content, add_special_tokens=False)
        if len(think_ids) > max_think_tokens:
            think_content = tokenizer.decode(think_ids[-max_think_tokens:], skip_special_tokens=True)
    return f"<think>{think_content}\n</think>\n"


def build_teacher_messages(row: dict, args, teacher_tokenizer) -> list[dict] | None:
    metadata = row.get("metadata", {}) or {}
    label = row.get("label", "") or ""
    if args.teacher_info_mode == "none":
        return None
    if args.teacher_info_mode == "same_as_student":
        return build_student_messages(row)
    if args.teacher_info_mode in {"hidden_think", "hidden_think_full"}:
        student_user_content = metadata.get("student_user_content") or metadata.get("raw_content", "")
        prefill = build_teacher_prefill_content(
            metadata,
            label,
            args.teacher_info_mode,
            teacher_tokenizer,
            args.teacher_think_max_tokens,
        )
        if prefill is None:
            return None
        return [
            {"role": "user", "content": student_user_content},
            {"role": "assistant", "content": prefill},
        ]

    teacher_user_content = build_teacher_user_content(metadata, label, args.teacher_info_mode, args)
    return [{"role": "user", "content": teacher_user_content}] if teacher_user_content is not None else None


def build_client(base_url: str, api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _normalize_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed[: -len("/v1")]
    return trimmed


def _to_generate_url(base_url: str) -> str:
    return f"{_normalize_base_url(base_url)}/generate"


def _is_probable_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if any(
        key in text
        for key in (
            "out of memory",
            "cuda out of memory",
            "cuda error: out of memory",
            "memory pool exhausted",
            "memory allocation failed",
        )
    ):
        return True
    return re.search(r"\boom\b", text) is not None


async def _post_json_checked(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    async with session.post(url, json=payload) as resp:
        body = await resp.text()
        status = resp.status
    if status >= 400:
        body_snippet = body[:800].strip()
        raise RuntimeError(f"HTTP {status} from {url}: {body_snippet}")
    try:
        return json.loads(body) if body else {}
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Non-JSON response from {url}: {body[:200]!r}") from e


def _ensure_list_token_ids(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return [int(x) for x in token_ids]


def _apply_chat_template_tokenize(
    tokenizer,
    messages: list[dict],
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is None:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
        return _ensure_list_token_ids(token_ids)
    try:
        token_ids = tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError as e:
        if "enable_thinking" not in str(e):
            raise
        global _ENABLE_THINKING_FALLBACK_WARNED
        if not _ENABLE_THINKING_FALLBACK_WARNED:
            logger.warning(
                "Tokenizer.apply_chat_template does not support `enable_thinking`; "
                "falling back to tokenizer default behavior."
            )
            _ENABLE_THINKING_FALLBACK_WARNED = True
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    return _ensure_list_token_ids(token_ids)


def build_generation_input_ids(
    tokenizer,
    prompt_messages: list[dict],
    continue_final_message: bool,
    enable_thinking: bool | None = None,
) -> list[int]:
    return _apply_chat_template_tokenize(
        tokenizer,
        prompt_messages,
        add_generation_prompt=not continue_final_message,
        enable_thinking=enable_thinking,
    )


def _shared_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def build_forced_score_input_ids(
    tokenizer,
    prompt_messages: list[dict],
    response_text: str,
    continue_final_message: bool,
    enable_thinking: bool | None = None,
) -> tuple[list[int], int, list[int]]:
    if continue_final_message:
        full_messages = copy.deepcopy(prompt_messages)
        if not full_messages or full_messages[-1]["role"] != "assistant":
            raise ValueError("continue_final_message=True requires the last message role to be assistant.")
        full_messages[-1]["content"] = f"{full_messages[-1].get('content', '')}{response_text}"
        prefix_ids = _apply_chat_template_tokenize(
            tokenizer,
            prompt_messages,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        full_ids = _apply_chat_template_tokenize(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
    else:
        full_messages = [*prompt_messages, {"role": "assistant", "content": response_text}]
        prefix_ids = _apply_chat_template_tokenize(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        full_ids = _apply_chat_template_tokenize(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )

    response_start = _shared_prefix_len(prefix_ids, full_ids)
    if response_start < len(prefix_ids):
        logger.warning(
            "Prompt token prefix mismatch (matched=%d, expected=%d). "
            "Falling back to longest shared prefix for response span.",
            response_start,
            len(prefix_ids),
        )
    return full_ids, response_start, full_ids[response_start:]


def _to_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _extract_scalar_logprob(item: Any) -> float:
    if isinstance(item, (list, tuple)):
        if not item:
            return float("nan")
        return _to_float_or_nan(item[0])
    if isinstance(item, dict):
        if "logprob" in item:
            return _to_float_or_nan(item.get("logprob"))
        if "value" in item:
            return _to_float_or_nan(item.get("value"))
    return _to_float_or_nan(item)


def _slice_response_logprobs(
    input_token_logprobs: list[Any],
    full_input_len: int,
    response_start: int,
    response_len: int,
) -> list[float]:
    if response_len <= 0:
        return []

    values = [_extract_scalar_logprob(x) for x in input_token_logprobs]
    if not values:
        return []

    if len(values) == response_len:
        return values

    if len(values) == full_input_len:
        begin = response_start
        end = response_start + response_len
        if end <= len(values):
            return values[begin:end]
    elif len(values) == full_input_len - 1:
        begin = max(response_start - 1, 0)
        end = begin + response_len
        if end <= len(values):
            return values[begin:end]
    elif len(values) == full_input_len + 1:
        begin = response_start + 1
        end = begin + response_len
        if end <= len(values):
            return values[begin:end]

    logger.warning(
        "Unexpected input_token_logprobs length=%d for input_len=%d. Falling back to tail slice len=%d.",
        len(values),
        full_input_len,
        response_len,
    )
    return values[-response_len:]


def _build_score_requests_for_response(
    sample_idx: int,
    full_input_ids: list[int],
    response_start: int,
    response_len: int,
    chunk_tokens: int | None,
    context_window_tokens: int | None,
) -> list[tuple[int, int, list[int], int, int]]:
    if response_len <= 0:
        return []

    if chunk_tokens is None or chunk_tokens <= 0:
        chunk_tokens = response_len

    requests: list[tuple[int, int, list[int], int, int]] = []
    for offset in range(0, response_len, chunk_tokens):
        chunk_len = min(chunk_tokens, response_len - offset)
        abs_chunk_start = response_start + offset
        abs_chunk_end = abs_chunk_start + chunk_len

        context_start = 0
        if context_window_tokens is not None and context_window_tokens > 0:
            context_start = max(0, abs_chunk_start - context_window_tokens)

        chunk_input_ids = full_input_ids[context_start:abs_chunk_end]
        local_response_start = abs_chunk_start - context_start
        requests.append((sample_idx, offset, chunk_input_ids, local_response_start, chunk_len))

    return requests


async def _score_one(
    session: aiohttp.ClientSession,
    generate_url: str,
    input_ids: list[int],
    response_start: int,
    response_len: int,
    retries: int,
    score_logprob_start: str,
    score_temperature: float,
    adaptive_oom_retry: bool,
    min_response_tokens_on_oom: int,
    oom_cooldown_seconds: float,
) -> list[float]:
    if score_logprob_start not in {"full", "response"}:
        raise ValueError(f"Unsupported score_logprob_start: {score_logprob_start!r}")
    active_response_len = max(0, int(response_len))
    active_input_ids = list(input_ids)

    last_error = None
    for attempt in range(retries + 1):
        try:
            logprob_start_len = 0 if score_logprob_start == "full" else max(response_start, 0)
            payload = {
                "input_ids": active_input_ids,
                "sampling_params": {
                    "temperature": float(score_temperature),
                    "max_new_tokens": 0,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": logprob_start_len,
            }
            output = await _post_json_checked(session, generate_url, payload)
            input_token_logprobs = output.get("meta_info", {}).get("input_token_logprobs")
            if input_token_logprobs is None:
                raise KeyError("`meta_info.input_token_logprobs` not found in scoring response.")
            return _slice_response_logprobs(
                input_token_logprobs=input_token_logprobs,
                full_input_len=len(active_input_ids),
                response_start=response_start,
                response_len=active_response_len,
            )
        except Exception as e:
            last_error = e
            if (
                adaptive_oom_retry
                and _is_probable_oom_error(e)
                and active_response_len > max(1, min_response_tokens_on_oom)
            ):
                new_response_len = max(max(1, min_response_tokens_on_oom), active_response_len // 2)
                if new_response_len < active_response_len:
                    active_response_len = new_response_len
                    active_input_ids = input_ids[: response_start + active_response_len]
                    logger.warning(
                        "Teacher scoring hit OOM; retrying with shorter response span: %d tokens.",
                        active_response_len,
                    )
                    if oom_cooldown_seconds > 0:
                        await asyncio.sleep(oom_cooldown_seconds)
                    continue
            if attempt == retries:
                break
            await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Scoring failed after {retries + 1} attempts: {last_error}") from last_error


async def _generate_one_with_logprobs(
    session: aiohttp.ClientSession,
    generate_url: str,
    input_ids: list[int],
    sampling_params: dict,
    seed: int,
    retries: int,
    adaptive_oom_retry: bool,
    oom_min_max_new_tokens: int,
    oom_cooldown_seconds: float,
) -> tuple[str, list[int], list[float]]:
    max_new_tokens = max(1, int(sampling_params["max_tokens"]))
    min_new_tokens = max(1, int(oom_min_max_new_tokens))

    last_error = None
    for attempt in range(retries + 1):
        try:
            payload = {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": sampling_params["temperature"],
                    "top_p": sampling_params["top_p"],
                    "max_new_tokens": max_new_tokens,
                    "skip_special_tokens": True,
                    "sampling_seed": seed,
                },
                "return_logprob": True,
            }
            output = await _post_json_checked(session, generate_url, payload)
            text = output.get("text", "") or ""
            output_token_logprobs = output.get("meta_info", {}).get("output_token_logprobs")
            if output_token_logprobs is None:
                raise KeyError("`meta_info.output_token_logprobs` not found in generation response.")
            token_ids = [int(item[1]) for item in output_token_logprobs]
            token_logprobs = [_extract_scalar_logprob(item) for item in output_token_logprobs]
            return text, token_ids, token_logprobs
        except Exception as e:
            last_error = e
            if adaptive_oom_retry and _is_probable_oom_error(e) and max_new_tokens > min_new_tokens:
                new_max_new_tokens = max(min_new_tokens, max_new_tokens // 2)
                if new_max_new_tokens < max_new_tokens:
                    max_new_tokens = new_max_new_tokens
                    logger.warning(
                        "Student generation hit OOM; retrying with max_new_tokens=%d.",
                        max_new_tokens,
                    )
                    if oom_cooldown_seconds > 0:
                        await asyncio.sleep(oom_cooldown_seconds)
                    continue
            if attempt == retries:
                break
            await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Generation with logprobs failed after {retries + 1} attempts: {last_error}") from last_error


async def _generate_one(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    sampling_params: dict,
    seed: int,
    retries: int,
    continue_final_message: bool,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
        "max_tokens": sampling_params["max_tokens"],
        "seed": seed,
    }
    if continue_final_message:
        payload["extra_body"] = {"continue_final_message": True}

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = await client.chat.completions.create(**payload)
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            if attempt == retries:
                break
            await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Generation failed after {retries + 1} attempts: {last_error}") from last_error


async def _async_batch_generate_with_logprobs(
    requests: list[tuple[int, list[int]]],
    sampling_params: dict,
    generate_url: str,
    concurrency: int,
    progress_desc: str,
    seed: int,
    retries: int,
    adaptive_oom_retry: bool,
    oom_min_max_new_tokens: int,
    oom_cooldown_seconds: float,
) -> tuple[list[str], list[list[int]], list[list[float]]]:
    from tqdm import tqdm

    texts: list[str | None] = [None] * len(requests)
    token_ids_list: list[list[int] | None] = [None] * len(requests)
    token_logprobs_list: list[list[float] | None] = [None] * len(requests)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    timeout = aiohttp.ClientTimeout(total=None, connect=120, sock_connect=120, sock_read=1800)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def _run(result_idx: int, input_ids: list[int]):
            async with semaphore:
                text, token_ids, token_logprobs = await _generate_one_with_logprobs(
                    session=session,
                    generate_url=generate_url,
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    seed=seed + result_idx,
                    retries=retries,
                    adaptive_oom_retry=adaptive_oom_retry,
                    oom_min_max_new_tokens=oom_min_max_new_tokens,
                    oom_cooldown_seconds=oom_cooldown_seconds,
                )
                return result_idx, text, token_ids, token_logprobs

        tasks = [
            asyncio.create_task(_run(result_idx, input_ids))
            for result_idx, (_, input_ids) in enumerate(requests)
        ]
        with tqdm(total=len(tasks), desc=progress_desc) as pbar:
            for task in asyncio.as_completed(tasks):
                result_idx, text, token_ids, token_logprobs = await task
                texts[result_idx] = text
                token_ids_list[result_idx] = token_ids
                token_logprobs_list[result_idx] = token_logprobs
                pbar.update(1)

    return (
        [x if x is not None else "" for x in texts],
        [x if x is not None else [] for x in token_ids_list],
        [x if x is not None else [] for x in token_logprobs_list],
    )


def batch_generate_with_logprobs(
    requests: list[tuple[int, list[int]]],
    sampling_params: dict,
    generate_url: str,
    concurrency: int,
    progress_desc: str,
    seed: int,
    retries: int,
    adaptive_oom_retry: bool,
    oom_min_max_new_tokens: int,
    oom_cooldown_seconds: float,
) -> tuple[list[str], list[list[int]], list[list[float]]]:
    return asyncio.run(
        _async_batch_generate_with_logprobs(
            requests=requests,
            sampling_params=sampling_params,
            generate_url=generate_url,
            concurrency=concurrency,
            progress_desc=progress_desc,
            seed=seed,
            retries=retries,
            adaptive_oom_retry=adaptive_oom_retry,
            oom_min_max_new_tokens=oom_min_max_new_tokens,
            oom_cooldown_seconds=oom_cooldown_seconds,
        )
    )


async def _async_batch_generate(
    client: AsyncOpenAI,
    model: str,
    requests: list[tuple[int, list[dict], bool]],
    sampling_params: dict,
    concurrency: int,
    progress_desc: str,
    seed: int,
    retries: int,
) -> list[str]:
    from tqdm import tqdm

    results = [None] * len(requests)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _run(result_idx: int, messages: list[dict], continue_final_message: bool):
        async with semaphore:
            text = await _generate_one(
                client=client,
                model=model,
                messages=messages,
                sampling_params=sampling_params,
                seed=seed + result_idx,
                retries=retries,
                continue_final_message=continue_final_message,
            )
            return result_idx, text

    tasks = [
        asyncio.create_task(_run(result_idx, messages, continue_final_message))
        for result_idx, (_, messages, continue_final_message) in enumerate(requests)
    ]

    with tqdm(total=len(tasks), desc=progress_desc) as pbar:
        for task in asyncio.as_completed(tasks):
            result_idx, text = await task
            results[result_idx] = text
            pbar.update(1)
    return results


def batch_generate(
    client: AsyncOpenAI,
    model: str,
    requests: list[tuple[int, list[dict], bool]],
    sampling_params: dict,
    concurrency: int,
    progress_desc: str,
    seed: int,
    retries: int,
) -> list[str]:
    return asyncio.run(
        _async_batch_generate(
            client=client,
            model=model,
            requests=requests,
            sampling_params=sampling_params,
            concurrency=concurrency,
            progress_desc=progress_desc,
            seed=seed,
            retries=retries,
        )
    )


async def _async_batch_score(
    requests: list[tuple[int, int, list[int], int, int]],
    generate_url: str,
    concurrency: int,
    progress_desc: str,
    retries: int,
    score_logprob_start: str,
    score_temperature: float,
    adaptive_oom_retry: bool,
    oom_min_score_response_tokens: int,
    oom_cooldown_seconds: float,
) -> list[list[float]]:
    from tqdm import tqdm

    results: list[list[float] | None] = [None] * len(requests)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    timeout = aiohttp.ClientTimeout(total=None, connect=120, sock_connect=120, sock_read=1800)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def _run(result_idx: int, input_ids: list[int], response_start: int, response_len: int):
            async with semaphore:
                logprobs = await _score_one(
                    session=session,
                    generate_url=generate_url,
                    input_ids=input_ids,
                    response_start=response_start,
                    response_len=response_len,
                    retries=retries,
                    score_logprob_start=score_logprob_start,
                    score_temperature=score_temperature,
                    adaptive_oom_retry=adaptive_oom_retry,
                    min_response_tokens_on_oom=oom_min_score_response_tokens,
                    oom_cooldown_seconds=oom_cooldown_seconds,
                )
                return result_idx, logprobs

        tasks = [
            asyncio.create_task(_run(result_idx, input_ids, response_start, response_len))
            for result_idx, (_, _, input_ids, response_start, response_len) in enumerate(requests)
        ]
        with tqdm(total=len(tasks), desc=progress_desc) as pbar:
            for task in asyncio.as_completed(tasks):
                result_idx, logprobs = await task
                results[result_idx] = logprobs
                pbar.update(1)

    return [r if r is not None else [] for r in results]


def batch_score(
    requests: list[tuple[int, int, list[int], int, int]],
    generate_url: str,
    concurrency: int,
    progress_desc: str,
    retries: int,
    score_logprob_start: str,
    score_temperature: float,
    adaptive_oom_retry: bool,
    oom_min_score_response_tokens: int,
    oom_cooldown_seconds: float,
) -> list[list[float]]:
    return asyncio.run(
        _async_batch_score(
            requests=requests,
            generate_url=generate_url,
            concurrency=concurrency,
            progress_desc=progress_desc,
            retries=retries,
            score_logprob_start=score_logprob_start,
            score_temperature=score_temperature,
            adaptive_oom_retry=adaptive_oom_retry,
            oom_min_score_response_tokens=oom_min_score_response_tokens,
            oom_cooldown_seconds=oom_cooldown_seconds,
        )
    )


def _load_tokenizer(source: str):
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer from %s", source)
    return AutoTokenizer.from_pretrained(source, trust_remote_code=True)


def _find_answer_char_start(text: str) -> int | None:
    boxed_pos = text.rfind("\\boxed{")
    if boxed_pos >= 0:
        return boxed_pos
    matches = list(re.finditer(r"(?im)^\s*answer\s*[:：]", text))
    if matches:
        return matches[-1].start()
    final_matches = list(re.finditer(r"(?im)final\s+answer\s*[:：]", text))
    if final_matches:
        return final_matches[-1].start()
    return None


def _find_thinking_char_start(text: str) -> int | None:
    pos = text.find("<think>")
    return pos if pos >= 0 else None


def _parse_enable_thinking_arg(value: str) -> bool | None:
    v = str(value).strip().lower()
    if v == "auto":
        return None
    if v == "true":
        return True
    if v == "false":
        return False
    raise ValueError(f"Unsupported enable_thinking value: {value!r}")


def _format_enable_thinking(value: bool | None) -> str:
    if value is None:
        return "auto(tokenizer default)"
    return "true" if value else "false"


def _char_to_token_position(
    tokenizer,
    text: str,
    char_pos: int | None,
    fallback_token_count: int,
) -> int | None:
    if char_pos is None or fallback_token_count <= 0:
        return None
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
        if not offsets:
            raise ValueError("offset_mapping not available")
        for idx, (start, end) in enumerate(offsets):
            if start <= char_pos < end:
                return idx + 1
            if char_pos < start:
                return idx + 1
        return len(offsets)
    except Exception:
        if not text:
            return None
        ratio = min(max(char_pos / max(len(text), 1), 0.0), 1.0)
        return min(max(int(round(ratio * fallback_token_count)), 1), fallback_token_count)


def _count_zero_logprobs(values: list[float], atol: float = 1e-12) -> tuple[int, int]:
    zero_count = 0
    finite_count = 0
    for v in values:
        if not math.isfinite(v):
            continue
        finite_count += 1
        if abs(v) <= atol:
            zero_count += 1
    return zero_count, finite_count


def run_eval(args):
    samples = load_dataset(args.dataset, args.n_samples, args.seed)
    teacher_prompt_tokenizer = maybe_load_tokenizer(args)
    student_scoring_tokenizer = _load_tokenizer(args.model)
    teacher_model = args.teacher_model or args.model
    teacher_scoring_tokenizer = _load_tokenizer(teacher_model) if args.teacher_info_mode != "none" else None

    student_requests = []
    teacher_prompt_messages = [None] * len(samples)
    labels = []
    answer_formats = []
    metadata_list = []

    for sample_idx, row in enumerate(samples):
        metadata = row.get("metadata", {}) or {}
        label = row.get("label", "") or ""

        student_messages = build_student_messages(row)
        student_continue_final = bool(student_messages) and student_messages[-1]["role"] == "assistant"
        student_requests.append((sample_idx, student_messages, student_continue_final))

        teacher_messages = build_teacher_messages(row, args, teacher_prompt_tokenizer)
        teacher_prompt_messages[sample_idx] = teacher_messages

        labels.append(label)
        answer_formats.append(_infer_answer_format(metadata))
        metadata_list.append(metadata)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
    }

    student_generation_requests = []
    for sample_idx, student_messages, student_continue_final in student_requests:
        student_input_ids = build_generation_input_ids(
            tokenizer=student_scoring_tokenizer,
            prompt_messages=student_messages,
            continue_final_message=student_continue_final,
            enable_thinking=args.student_enable_thinking,
        )
        student_generation_requests.append((sample_idx, student_input_ids))

    logger.info(
        "Running student generation via /generate (with output_token_logprobs): base=%s model=%s",
        args.student_api_base,
        args.model,
    )
    student_responses, student_response_token_ids_raw, student_response_logprobs_raw = batch_generate_with_logprobs(
        requests=student_generation_requests,
        sampling_params=sampling_params,
        generate_url=_to_generate_url(args.student_api_base),
        concurrency=args.generation_concurrency,
        progress_desc="Student generation",
        seed=args.seed,
        retries=args.retries,
        adaptive_oom_retry=not args.disable_adaptive_oom_retry,
        oom_min_max_new_tokens=args.oom_min_max_new_tokens,
        oom_cooldown_seconds=args.oom_cooldown_seconds,
    )
    student_response_token_count_by_idx = {
        sample_idx: len(token_ids)
        for (sample_idx, _), token_ids in zip(student_generation_requests, student_response_token_ids_raw)
    }
    # Token IDs are only needed for length alignment; release them early to reduce memory.
    del student_response_token_ids_raw
    student_logprob_by_idx = {
        sample_idx: list(logprobs)
        for (sample_idx, _), logprobs in zip(student_generation_requests, student_response_logprobs_raw)
    }

    teacher_score_requests = []
    teacher_score_expected_len: dict[int, int] = {}

    for sample_idx, s_resp in enumerate(student_responses):
        t_messages = teacher_prompt_messages[sample_idx]
        if t_messages is None or teacher_scoring_tokenizer is None:
            continue
        teacher_continue_final = t_messages[-1]["role"] == "assistant"
        t_full_ids, t_start, t_resp_ids = build_forced_score_input_ids(
            tokenizer=teacher_scoring_tokenizer,
            prompt_messages=t_messages,
            response_text=s_resp,
            continue_final_message=teacher_continue_final,
            enable_thinking=args.teacher_enable_thinking,
        )
        score_input_ids = t_full_ids
        score_response_len = len(t_resp_ids)
        if args.max_score_response_tokens is not None and score_response_len > args.max_score_response_tokens:
            score_response_len = max(1, int(args.max_score_response_tokens))
            score_input_ids = t_full_ids[: t_start + score_response_len]
            logger.warning(
                "Sample %d teacher scoring truncated to first %d response tokens (from %d) to reduce memory.",
                sample_idx,
                score_response_len,
                len(t_resp_ids),
            )
        teacher_score_expected_len[sample_idx] = score_response_len
        teacher_score_requests.extend(
            _build_score_requests_for_response(
                sample_idx=sample_idx,
                full_input_ids=score_input_ids,
                response_start=t_start,
                response_len=score_response_len,
                chunk_tokens=args.score_chunk_tokens,
                context_window_tokens=args.score_context_window_tokens,
            )
        )

    teacher_logprob_by_idx = {}
    if not teacher_score_requests:
        teacher_prompt_none_count = sum(1 for m in teacher_prompt_messages if m is None)
        if args.teacher_info_mode == "none":
            logger.info(
                "Teacher scoring disabled because teacher_info_mode='none'; teacher_logprobs will be null."
            )
        else:
            logger.warning(
                "No teacher score requests were built (teacher_info_mode=%s, teacher_prompt_messages_none=%d/%d). "
                "teacher_logprobs will be null.",
                args.teacher_info_mode,
                teacher_prompt_none_count,
                len(teacher_prompt_messages),
            )

    if teacher_score_requests:
        logger.info(
            "Scoring student sampled tokens with teacher model via /generate: base=%s model=%s score_temperature=%.4g",
            args.teacher_api_base,
            teacher_model,
            args.score_temperature,
        )
        teacher_scored = batch_score(
            requests=teacher_score_requests,
            generate_url=_to_generate_url(args.teacher_api_base),
            concurrency=args.score_concurrency,
            progress_desc="Teacher token scoring",
            retries=args.retries,
            score_logprob_start=args.score_logprob_start,
            score_temperature=args.score_temperature,
            adaptive_oom_retry=not args.disable_adaptive_oom_retry,
            oom_min_score_response_tokens=args.oom_min_score_response_tokens,
            oom_cooldown_seconds=args.oom_cooldown_seconds,
        )
        chunked_logprobs_by_idx: dict[int, list[tuple[int, list[float]]]] = {}
        for (sample_idx, response_offset, *_), scored in zip(teacher_score_requests, teacher_scored):
            chunked_logprobs_by_idx.setdefault(sample_idx, []).append((response_offset, scored))

        for sample_idx, pieces in chunked_logprobs_by_idx.items():
            pieces.sort(key=lambda x: x[0])
            merged: list[float] = []
            cursor = 0
            for offset, chunk_vals in pieces:
                values = list(chunk_vals)
                if offset > cursor:
                    gap = offset - cursor
                    logger.warning(
                        "Teacher score chunk gap for sample %d: offset=%d cursor=%d (filling %d NaNs).",
                        sample_idx,
                        offset,
                        cursor,
                        gap,
                    )
                    merged.extend([float("nan")] * gap)
                    cursor = offset
                elif offset < cursor:
                    overlap = cursor - offset
                    if overlap >= len(values):
                        logger.warning(
                            "Teacher score chunk overlap fully skipped for sample %d: offset=%d cursor=%d chunk_len=%d.",
                            sample_idx,
                            offset,
                            cursor,
                            len(values),
                        )
                        continue
                    logger.warning(
                        "Teacher score chunk overlap for sample %d: offset=%d cursor=%d (dropping %d tokens).",
                        sample_idx,
                        offset,
                        cursor,
                        overlap,
                    )
                    values = values[overlap:]

                merged.extend(values)
                cursor += len(values)

            expected_len = teacher_score_expected_len.get(sample_idx)
            if expected_len is not None and len(merged) != expected_len:
                if len(merged) > expected_len:
                    logger.warning(
                        "Teacher score length overflow for sample %d: expected=%d got=%d (truncating).",
                        sample_idx,
                        expected_len,
                        len(merged),
                    )
                    merged = merged[:expected_len]
                else:
                    logger.warning(
                        "Teacher score length underflow for sample %d: expected=%d got=%d (padding NaNs).",
                        sample_idx,
                        expected_len,
                        len(merged),
                    )
                    merged.extend([float("nan")] * (expected_len - len(merged)))

            teacher_logprob_by_idx[sample_idx] = merged

    student_rewards = []
    student_zero_count = 0
    student_finite_count = 0
    teacher_zero_count = 0
    teacher_finite_count = 0
    results = []

    for i, (s_resp, label, fmt) in enumerate(zip(student_responses, labels, answer_formats)):
        s_reward = grade(s_resp, label, fmt)
        student_rewards.append(s_reward)

        s_logprobs_raw = student_logprob_by_idx.get(i, [])
        t_logprobs_raw = teacher_logprob_by_idx.get(i)
        s_token_count = student_response_token_count_by_idx.get(i, 0)

        length_candidates = [s_token_count, len(s_logprobs_raw)]
        if t_logprobs_raw is not None:
            length_candidates.append(len(t_logprobs_raw))
        common_len = min(length_candidates) if length_candidates else 0

        if s_token_count != common_len or len(s_logprobs_raw) != common_len:
            logger.warning(
                "Sample %d student logprob length mismatch: tokens=%d, student_lp=%d, using %d",
                i,
                s_token_count,
                len(s_logprobs_raw),
                common_len,
            )
        if t_logprobs_raw is not None and len(t_logprobs_raw) != common_len:
            logger.warning(
                "Sample %d teacher logprob length mismatch: teacher_lp=%d, using %d",
                i,
                len(t_logprobs_raw),
                common_len,
            )

        s_logprobs = s_logprobs_raw[:common_len]
        t_logprobs = t_logprobs_raw[:common_len] if t_logprobs_raw is not None else None
        s_zero, s_finite = _count_zero_logprobs(s_logprobs)
        student_zero_count += s_zero
        student_finite_count += s_finite
        if t_logprobs is not None:
            t_zero, t_finite = _count_zero_logprobs(t_logprobs)
            teacher_zero_count += t_zero
            teacher_finite_count += t_finite

        answer_char_pos = _find_answer_char_start(s_resp)
        thinking_char_pos = _find_thinking_char_start(s_resp)
        answer_token_start = _char_to_token_position(student_scoring_tokenizer, s_resp, answer_char_pos, common_len)
        thinking_token_start = _char_to_token_position(
            student_scoring_tokenizer, s_resp, thinking_char_pos, common_len
        )

        results.append(
            {
                "index": i,
                "label": label,
                "student_response": s_resp,
                "student_reward": s_reward,
                "teacher_response": None,
                "teacher_reward": None,
                "metadata": metadata_list[i],
                "token_stats": {
                    "student_logprobs": s_logprobs,
                    "teacher_logprobs": t_logprobs,
                    "answer_token_start": answer_token_start,
                    "thinking_token_start": thinking_token_start,
                },
            }
        )

    n = len(student_rewards)
    student_acc = sum(student_rewards) / n if n else 0.0
    student_zero_ratio = (student_zero_count / student_finite_count) if student_finite_count > 0 else None
    teacher_zero_ratio = (teacher_zero_count / teacher_finite_count) if teacher_finite_count > 0 else None

    if student_zero_ratio is not None:
        logger.info(
            "Student logprob zero ratio: %.2f%% (%d/%d finite tokens, generation temperature=%.4g)",
            student_zero_ratio * 100.0,
            student_zero_count,
            student_finite_count,
            args.temperature,
        )
        if student_zero_ratio >= 0.20:
            logger.warning(
                "High student zero-logprob ratio detected (%.2f%%). "
                "This may indicate post-temperature/sampled logprobs rather than original model logprobs.",
                student_zero_ratio * 100.0,
            )
    if teacher_zero_ratio is not None:
        logger.info(
            "Teacher logprob zero ratio: %.2f%% (%d/%d finite tokens, score temperature=%.4g)",
            teacher_zero_ratio * 100.0,
            teacher_zero_count,
            teacher_finite_count,
            args.score_temperature,
        )
        if teacher_zero_ratio >= 0.20:
            logger.warning(
                "High teacher zero-logprob ratio detected (%.2f%%). "
                "If you need distribution-level logprobs, try score temperature=1.0 and/or "
                "enable original-logprob mode on the backend.",
                teacher_zero_ratio * 100.0,
            )

    print(f"\n{'=' * 60}")
    print(f"Dataset : {args.dataset}")
    print(f"Student : {args.model} @ {args.student_api_base}")
    print(f"Teacher : {teacher_model} @ {args.teacher_api_base}")
    print(f"Teacher info mode : {args.teacher_info_mode}")
    print(f"Samples : {n}")
    print(f"{'=' * 60}")
    print(f"Student accuracy : {student_acc:.4f}  ({sum(student_rewards):.0f}/{n})")
    if student_zero_ratio is not None:
        print(f"Student zero-logprob ratio              : {student_zero_ratio:.2%}")
    if teacher_zero_ratio is not None:
        print(f"Teacher zero-logprob ratio              : {teacher_zero_ratio:.2%}")
    print(f"{'=' * 60}\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Saved predictions to %s", args.output)

    return student_acc, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", required=True, help="Student API model name.")
    parser.add_argument("--teacher-model", default=None, help="Teacher API model name. Defaults to --model.")

    parser.add_argument("--dataset", required=True, help="Path to preprocessed JSONL dataset.")
    parser.add_argument("--n-samples", type=int, default=None, help="Randomly sample N rows.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Legacy global concurrency (used as default for generation/score concurrency).",
    )
    parser.add_argument(
        "--generation-concurrency",
        type=int,
        default=None,
        help="Max in-flight student generation requests. Defaults to --concurrency.",
    )
    parser.add_argument(
        "--score-concurrency",
        type=int,
        default=2,
        help="Max in-flight teacher scoring requests. Defaults to min(--concurrency, 8).",
    )
    parser.add_argument(
        "--score-logprob-start",
        default="response",
        choices=["full", "response"],
        help="Scoring logprob start span: full input or only response span. `response` greatly reduces memory.",
    )
    parser.add_argument(
        "--score-temperature",
        type=float,
        default=1.0,
        help="Teacher scoring temperature used in /generate with max_new_tokens=0.",
    )
    parser.add_argument("--retries", type=int, default=2, help="Retries per request on API failure.")
    parser.add_argument(
        "--max-score-response-tokens",
        type=int,
        default=None,
        help="Optional cap for teacher scoring span (first N response tokens). Useful for long-response OOM mitigation.",
    )
    parser.add_argument(
        "--score-chunk-tokens",
        type=int,
        default=None,
        help=(
            "Split teacher scoring into chunks of response tokens. "
            "Used with --score-context-window-tokens to reduce peak memory."
        ),
    )
    parser.add_argument(
        "--score-context-window-tokens",
        type=int,
        default=None,
        help=(
            "When set, teacher scoring only keeps the last N context tokens before each scored chunk "
            "(approximate logprobs, lower memory)."
        ),
    )
    parser.add_argument(
        "--disable-adaptive-oom-retry",
        action="store_true",
        help="Disable adaptive OOM fallback (automatic token-span / max_new_tokens reduction).",
    )
    parser.add_argument(
        "--oom-min-max-new-tokens",
        type=int,
        default=1024,
        help="Lower bound for adaptive max_new_tokens reduction after OOM.",
    )
    parser.add_argument(
        "--oom-min-score-response-tokens",
        type=int,
        default=512,
        help="Lower bound for adaptive teacher scoring span reduction after OOM.",
    )
    parser.add_argument(
        "--oom-cooldown-seconds",
        type=float,
        default=1.5,
        help="Sleep duration between adaptive OOM retries (seconds).",
    )

    parser.add_argument(
        "--student-api-base",
        default=os.environ.get("OPENAI_API_BASE") or os.environ.get("STUDENT_API_BASE") or "http://127.0.0.1:30000/v1",
        help="OpenAI-compatible API base for the student model.",
    )
    parser.add_argument(
        "--student-api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("STUDENT_API_KEY") or "EMPTY",
        help="API key for the student endpoint.",
    )
    parser.add_argument("--teacher-api-base", default=None, help="Teacher API base. Defaults to student API base.")
    parser.add_argument("--teacher-api-key", default=None, help="Teacher API key. Defaults to student API key.")

    parser.add_argument(
        "--teacher-info-mode",
        default="answer_only",
        choices=[
            "none",
            "same_as_student",
            "answer_only",
            "pi",
            "full",
            "masked_reasoning",
            "conciseness",
            "hidden_think",
            "hidden_think_full",
        ],
        help="Teacher privileged prompt mode.",
    )
    parser.add_argument("--teacher-think-max-tokens", type=int, default=-1)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--conciseness-instruction", default=_DEFAULT_CONCISENESS_INSTRUCTION)
    parser.add_argument(
        "--student-enable-thinking",
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "enable_thinking passed to student-side tokenizer.apply_chat_template. "
            "`auto` means tokenizer default behavior."
        ),
    )
    parser.add_argument(
        "--teacher-enable-thinking",
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "enable_thinking passed to teacher-side tokenizer.apply_chat_template for scoring. "
            "`auto` follows student setting; if student is also auto, it uses tokenizer default."
        ),
    )

    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")
    parser.add_argument(
        "--output",
        default="./eval_math500_student_teacher_inference_s1.7t8b_noanswer.jsonl",
        help="Path to save per-sample predictions as JSONL.",
    )

    args = parser.parse_args()
    if args.generation_concurrency is None:
        args.generation_concurrency = args.concurrency
    if args.score_concurrency is None:
        args.score_concurrency = max(1, min(args.concurrency, 8))
    if args.teacher_api_base is None:
        args.teacher_api_base = args.student_api_base
    if args.teacher_api_key is None:
        args.teacher_api_key = args.student_api_key
    args.student_enable_thinking = _parse_enable_thinking_arg(args.student_enable_thinking)
    if str(args.teacher_enable_thinking).strip().lower() == "auto":
        args.teacher_enable_thinking = args.student_enable_thinking
    else:
        args.teacher_enable_thinking = _parse_enable_thinking_arg(args.teacher_enable_thinking)
    if args.teacher_enable_thinking and args.teacher_info_mode in {"hidden_think", "hidden_think_full"}:
        logger.warning(
            "teacher_enable_thinking=true with %s may create an extra thinking prefix around the "
            "manually injected <think> prefill. For safety, forcing teacher_enable_thinking=false.",
            args.teacher_info_mode,
        )
        args.teacher_enable_thinking = False
    args.oom_min_max_new_tokens = max(1, int(args.oom_min_max_new_tokens))
    args.oom_min_score_response_tokens = max(1, int(args.oom_min_score_response_tokens))
    if args.max_score_response_tokens is not None:
        args.max_score_response_tokens = max(1, int(args.max_score_response_tokens))
    if args.score_chunk_tokens is not None:
        args.score_chunk_tokens = max(1, int(args.score_chunk_tokens))
    if args.score_context_window_tokens is not None:
        args.score_context_window_tokens = max(1, int(args.score_context_window_tokens))
    args.oom_cooldown_seconds = max(0.0, float(args.oom_cooldown_seconds))
    same_server = _normalize_base_url(args.student_api_base) == _normalize_base_url(args.teacher_api_base)
    if same_server and args.teacher_info_mode != "none":
        logger.warning(
            "Student and teacher endpoints are the same server (%s). Long generation + scoring on one server can trigger OOM.",
            _normalize_base_url(args.student_api_base),
        )
    if args.max_new_tokens >= 8192:
        logger.warning(
            "max_new_tokens=%d is high. If OOM persists, try reducing to 2048/4096 or set --max-score-response-tokens.",
            args.max_new_tokens,
        )
    if args.score_chunk_tokens is not None and args.score_context_window_tokens is None:
        logger.warning(
            "--score-chunk-tokens is set without --score-context-window-tokens. "
            "This may increase total compute time but not reduce memory peak much."
        )
    if args.score_context_window_tokens is not None:
        logger.warning(
            "Teacher scoring context window is enabled (%d tokens). "
            "This reduces memory but makes teacher logprobs approximate.",
            args.score_context_window_tokens,
        )
    logger.info(
        "Concurrency: generation=%d, score=%d, score_logprob_start=%s, score_temperature=%.4g, adaptive_oom_retry=%s",
        args.generation_concurrency,
        args.score_concurrency,
        args.score_logprob_start,
        args.score_temperature,
        "off" if args.disable_adaptive_oom_retry else "on",
    )
    logger.info(
        "Chat template enable_thinking: student=%s, teacher=%s",
        _format_enable_thinking(args.student_enable_thinking),
        _format_enable_thinking(args.teacher_enable_thinking),
    )
    run_eval(args)


if __name__ == "__main__":
    main()
