#!/usr/bin/env python3
"""
Standalone inference-only evaluation for student and teacher models via an
OpenAI-compatible chat completions API.
"""


'''
python examples/on_policy_distillation/eval_student_teacher_inference.py \
  --model Qwen3-1.7B \
  --teacher-model Qwen3-1.7B \
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
  --model Qwen3-1.7B \
  --teacher-model Qwen3-1.7B \
  --student-api-base http://127.0.0.1:30000/v1 \
  --teacher-api-base http://127.0.0.1:30001/v1 \
  --student-api-key EMPTY \
  --teacher-api-key EMPTY \
  --dataset /root/math/data/train_dapo.jsonl \
  --teacher-info-mode none \
  --max-new-tokens 32768 \
  --n-samples 32 \
  --concurrency 256 \
  --seed 42

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


def run_eval(args):
    samples = load_dataset(args.dataset, args.n_samples, args.seed)
    teacher_tokenizer = maybe_load_tokenizer(args)

    student_requests = []
    teacher_requests = []
    labels = []
    answer_formats = []
    metadata_list = []

    for sample_idx, row in enumerate(samples):
        metadata = row.get("metadata", {}) or {}
        label = row.get("label", "") or ""

        student_requests.append((sample_idx, build_student_messages(row), False))

        teacher_messages = build_teacher_messages(row, args, teacher_tokenizer)
        if teacher_messages is not None:
            teacher_requests.append((sample_idx, teacher_messages, teacher_messages[-1]["role"] == "assistant"))

        labels.append(label)
        answer_formats.append(_infer_answer_format(metadata))
        metadata_list.append(metadata)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
    }

    student_client = build_client(args.student_api_base, args.student_api_key)
    teacher_client = build_client(args.teacher_api_base, args.teacher_api_key)
    teacher_model = args.teacher_model or args.model

    logger.info(f"Running student inference via API: base={args.student_api_base} model={args.model}")
    student_responses = batch_generate(
        client=student_client,
        model=args.model,
        requests=student_requests,
        sampling_params=sampling_params,
        concurrency=args.concurrency,
        progress_desc="Student generation",
        seed=args.seed,
        retries=args.retries,
    )

    teacher_responses = [None] * len(samples)
    if args.teacher_info_mode != "none" and teacher_requests:
        logger.info(f"Running teacher inference via API: base={args.teacher_api_base} model={teacher_model}")
        valid_teacher_responses = batch_generate(
            client=teacher_client,
            model=teacher_model,
            requests=teacher_requests,
            sampling_params=sampling_params,
            concurrency=args.concurrency,
            progress_desc="Teacher generation",
            seed=args.seed + 1000000,
            retries=args.retries,
        )
        for request_offset, (sample_idx, _, _) in enumerate(teacher_requests):
            teacher_responses[sample_idx] = valid_teacher_responses[request_offset]

    student_rewards, teacher_rewards = [], []
    results = []
    for i, (row, s_resp, t_resp, label, fmt) in enumerate(
        zip(samples, student_responses, teacher_responses, labels, answer_formats)
    ):
        s_reward = grade(s_resp, label, fmt)
        student_rewards.append(s_reward)
        t_reward = None
        if t_resp is not None:
            t_reward = grade(t_resp, label, fmt)
            teacher_rewards.append(t_reward)
        results.append(
            {
                "index": i,
                "label": label,
                "student_response": s_resp,
                "student_reward": s_reward,
                "teacher_response": t_resp,
                "teacher_reward": t_reward,
                "metadata": metadata_list[i],
            }
        )

    n = len(student_rewards)
    student_acc = sum(student_rewards) / n if n else 0.0
    print(f"\n{'=' * 60}")
    print(f"Dataset : {args.dataset}")
    print(f"Student : {args.model} @ {args.student_api_base}")
    print(f"Samples : {n}")
    print(f"{'=' * 60}")
    print(f"Student accuracy : {student_acc:.4f}  ({sum(student_rewards):.0f}/{n})")
    if teacher_rewards:
        nt = len(teacher_rewards)
        teacher_acc = sum(teacher_rewards) / nt if nt else 0.0
        print(
            f"Teacher accuracy : {teacher_acc:.4f}  ({sum(teacher_rewards):.0f}/{nt})"
            f"  [mode={args.teacher_info_mode}, model={teacher_model}]"
        )
    print(f"{'=' * 60}\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved predictions to {args.output}")

    return student_acc, (sum(teacher_rewards) / len(teacher_rewards)) if teacher_rewards else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", required=True, help="Student API model name.")
    parser.add_argument("--teacher-model", default=None, help="Teacher API model name. Defaults to --model.")

    parser.add_argument("--dataset", required=True, help="Path to preprocessed JSONL dataset.")
    parser.add_argument("--n-samples", type=int, default=None, help="Randomly sample N rows.")
    parser.add_argument("--concurrency", type=int, default=16, help="Max in-flight API requests.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per request on API failure.")

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

    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and reproducibility.")
    parser.add_argument(
        "--output",
        default="./eval_math500_student_teacher_inference.jsonl",
        help="Path to save per-sample predictions as JSONL.",
    )

    args = parser.parse_args()
    if args.teacher_api_base is None:
        args.teacher_api_base = args.student_api_base
    if args.teacher_api_key is None:
        args.teacher_api_key = args.student_api_key
    run_eval(args)


if __name__ == "__main__":
    main()
