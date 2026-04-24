import copy
import json
import logging
import asyncio
from functools import lru_cache

import aiohttp
import torch

from slime.rollout.rm_hub import grade_answer_verl
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.opd_token_stats import compute_teacher_rank_at_k, compute_topk_overlap_ratio
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
_RM_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}
_OPTIONAL_LOGPROB_WARNING_COUNTS: dict[str, int] = {}
_MASK_GENERATORS: dict[tuple[str, str], MultiTurnLossMaskGenerator] = {}


def _get_rm_semaphore(args) -> asyncio.Semaphore:
    limit = int(getattr(args, "rm_max_concurrency", 8) or 8)
    if limit <= 0:
        limit = 1
    loop = asyncio.get_running_loop()
    key = (id(loop), limit)
    semaphore = _RM_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        _RM_SEMAPHORES[key] = semaphore
    return semaphore

_ANSWER_ONLY_PROMPT = (
    "The correct final answer to this problem is: {answer}\n"
    "Now solve the problem yourself step by step and arrive at the same answer:"
)

_FULL_TRANSITION_PROMPT = (
    "After understanding the reference solution and the rationale behind each step, "
    "now articulate your own step-by-step reasoning that derives the same final answer "
    "to the problem above:"
)


def _first_nonempty_text(*values) -> str:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            return s
    return ""


def _extract_user_content_from_prompt(prompt) -> str:
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    return str(prompt or "")


def _infer_answer_format(metadata: dict) -> str:
    fmt = str((metadata or {}).get("format_instruction", "") or "")
    if "boxed" in fmt:
        return "boxed"
    if "Answer" in fmt:
        return "answer"
    return "auto"


def _resolve_chat_template_kwargs(args) -> dict:
    raw_kwargs = getattr(args, "apply_chat_template_kwargs", None)
    if not raw_kwargs:
        return {}
    if isinstance(raw_kwargs, dict):
        return raw_kwargs
    try:
        parsed = json.loads(raw_kwargs)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


@lru_cache(maxsize=4)
def _load_cached_tokenizer(name_or_path: str):
    return load_tokenizer(name_or_path, trust_remote_code=True)


def _get_teacher_tokenizer(args):
    tokenizer_source = getattr(args, "opd_teacher_tokenizer", None) or getattr(args, "hf_checkpoint", None)
    if not tokenizer_source:
        raise ValueError(
            "Cannot build privileged teacher prompt for OPD-SGLang because tokenizer source is empty. "
            "Set --opd-teacher-tokenizer or --hf-checkpoint."
        )
    return _load_cached_tokenizer(tokenizer_source)


def _get_actor_tokenizer(args):
    tokenizer_source = getattr(args, "hf_checkpoint", None) or getattr(args, "opd_teacher_tokenizer", None)
    if not tokenizer_source:
        raise ValueError("Cannot build teacher-SFT loss mask because --hf-checkpoint is empty.")
    return _load_cached_tokenizer(tokenizer_source)


def _get_mask_generator(args):
    tokenizer_source = getattr(args, "hf_checkpoint", None) or getattr(args, "opd_teacher_tokenizer", None)
    tokenizer_type = getattr(args, "loss_mask_type", "qwen")
    key = (str(tokenizer_source), str(tokenizer_type))
    generator = _MASK_GENERATORS.get(key)
    if generator is None:
        generator = MultiTurnLossMaskGenerator(_get_actor_tokenizer(args), tokenizer_type=tokenizer_type)
        _MASK_GENERATORS[key] = generator
    return generator


def _build_teacher_user_content(args, sample: Sample) -> str:
    mode = getattr(args, "opd_teacher_info_mode", "same_as_student")
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    raw_content = _first_nonempty_text(metadata.get("raw_content"))
    raw_problem = _first_nonempty_text(metadata.get("raw_problem"), raw_content)
    student_user_content = _first_nonempty_text(
        metadata.get("student_user_content"),
        raw_content,
        _extract_user_content_from_prompt(sample.prompt),
    )

    if mode == "same_as_student":
        return student_user_content

    if mode == "full":
        reference_solution = _first_nonempty_text(
            metadata.get("reference_solution"),
            metadata.get("solution"),
        )
        format_instruction = _first_nonempty_text(metadata.get("format_instruction"))
        if reference_solution:
            if format_instruction:
                return (
                    f"{raw_problem}\n\n"
                    f"Here is a reference solution to this problem:\n{reference_solution}\n"
                    f"{_FULL_TRANSITION_PROMPT}\n"
                    f"{format_instruction}"
                )
            return (
                f"{raw_problem}\n\n"
                f"Here is a reference solution to this problem:\n{reference_solution}\n"
                f"{_FULL_TRANSITION_PROMPT}"
            )
        return student_user_content

    if mode != "answer_only":
        raise ValueError(f"Unsupported opd_teacher_info_mode: {mode!r}")

    answer_hint = _first_nonempty_text(
        sample.label,
        metadata.get("solution"),
        metadata.get("reference_solution"),
    )
    return (
        f"{student_user_content}\n\n" + _ANSWER_ONLY_PROMPT.format(answer=answer_hint)
        if answer_hint
        else student_user_content
    )


def _build_teacher_prompt_input_ids(args, sample: Sample) -> list[int]:
    mode = getattr(args, "opd_teacher_info_mode", "same_as_student")
    if mode == "same_as_student":
        response_len = int(sample.response_length or 0)
        if response_len <= 0:
            return list(sample.tokens)
        return list(sample.tokens[:-response_len])

    teacher_user_content = _build_teacher_user_content(args, sample)

    tokenizer = _get_teacher_tokenizer(args)
    template_kwargs = _resolve_chat_template_kwargs(args)
    teacher_enable_thinking = template_kwargs.get("enable_thinking", True)
    teacher_messages = [{"role": "user", "content": teacher_user_content}]
    try:
        teacher_prompt_text = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=teacher_enable_thinking,
        )
    except TypeError as e:
        if "enable_thinking" not in str(e):
            raise
        teacher_prompt_text = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    teacher_prompt_tokens = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)
    return teacher_prompt_tokens


def _build_teacher_input_ids(args, sample: Sample) -> list[int]:
    teacher_prompt_tokens = _build_teacher_prompt_input_ids(args, sample)
    response_tokens = list(sample.tokens[-sample.response_length:]) if sample.response_length > 0 else []
    return teacher_prompt_tokens + response_tokens


def _extract_generated_token_ids(output: dict) -> list[int]:
    meta_info = output.get("meta_info", {}) if isinstance(output, dict) else {}
    output_items = meta_info.get("output_token_logprobs")
    if isinstance(output_items, list):
        token_ids = []
        for item in output_items:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                token_ids.append(int(item[1]))
            elif isinstance(item, dict):
                token_id = item.get("token_id", item.get("id", item.get("token")))
                if token_id is not None:
                    token_ids.append(int(token_id))
        if token_ids:
            return token_ids

    output_ids = output.get("output_ids") if isinstance(output, dict) else None
    if isinstance(output_ids, list):
        return [int(token_id) for token_id in output_ids]
    return []


def _build_teacher_sft_response_loss_mask(
    args,
    sample: Sample,
    *,
    prompt_input_ids: list[int],
    response_token_ids: list[int],
    response_text: str,
) -> list[int]:
    # prompt_input_ids already includes the generation-prompt header tokens
    # (e.g. <|im_start|>assistant\n), so response_token_ids are purely the
    # content tokens generated by SGLang — all should be trained on.
    # Re-tokenizing the decoded response_text to compute a mask is unreliable
    # (decode→re-encode is not always a round-trip), and both the match and
    # the fallback branch produce the same [1]*len(response_token_ids) result.
    return [1] * len(response_token_ids)


def _extract_scalar_logprob(item) -> float:
    if item is None:
        raise ValueError("Logprob item is None.")
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, str):
        return float(item)
    if isinstance(item, (list, tuple)):
        if not item:
            raise ValueError("Empty logprob item.")
        return _extract_scalar_logprob(item[0])
    if isinstance(item, dict):
        if "logprob" in item:
            return _extract_scalar_logprob(item["logprob"])
        if "value" in item:
            return _extract_scalar_logprob(item["value"])
    raise ValueError(f"Unsupported logprob item format: {type(item)}")


def _is_missing_logprob_item(item) -> bool:
    if item is None:
        return True
    if isinstance(item, (list, tuple)):
        if not item:
            return False
        return _is_missing_logprob_item(item[0])
    if isinstance(item, dict):
        if "logprob" in item:
            return _is_missing_logprob_item(item.get("logprob"))
        if "value" in item:
            return _is_missing_logprob_item(item.get("value"))
    return False


def _slice_response_items(
    items: list,
    full_input_len: int,
    response_start: int,
    response_len: int,
) -> list:
    if response_len <= 0:
        return []
    if len(items) == response_len:
        return list(items)
    if len(items) == full_input_len:
        begin = response_start
        end = response_start + response_len
        if end <= len(items):
            return list(items[begin:end])
    elif len(items) == full_input_len - 1:
        begin = max(response_start - 1, 0)
        end = begin + response_len
        if end <= len(items):
            return list(items[begin:end])
    elif len(items) == full_input_len + 1:
        begin = response_start + 1
        end = begin + response_len
        if end <= len(items):
            return list(items[begin:end])
    # Conservative fallback: keep old behavior (tail slice) for robustness.
    return list(items[-response_len:])


def _candidate_response_slices(
    items: list,
    full_input_len: int,
    response_start: int,
    response_len: int,
) -> list[list]:
    if response_len <= 0:
        return [[]]

    candidates: list[list] = []

    def _add(candidate: list | None) -> None:
        if candidate is None:
            return
        candidate = list(candidate)
        if len(candidate) != response_len:
            return
        if candidate not in candidates:
            candidates.append(candidate)

    _add(_slice_response_items(items, full_input_len, response_start, response_len))

    if len(items) >= response_len:
        for shift in (-1, 1):
            begin = response_start + shift
            end = begin + response_len
            if begin >= 0 and end <= len(items):
                _add(items[begin:end])
        _add(items[-response_len:])

    return candidates


def _extract_response_log_probs_with_mask(
    teacher_output: dict,
    *,
    full_input_len: int,
    response_start: int,
    response_len: int,
) -> tuple[list[float], list[int]]:
    if response_len <= 0:
        return [], []

    meta_info = teacher_output.get("meta_info", {}) if isinstance(teacher_output, dict) else {}
    candidate_specs: list[tuple[str, list[list]]] = []

    input_items = meta_info.get("input_token_logprobs")
    if isinstance(input_items, list):
        candidate_specs.append(
            (
                "meta_info.input_token_logprobs",
                _candidate_response_slices(input_items, full_input_len, response_start, response_len),
            )
        )

    output_items = meta_info.get("output_token_logprobs")
    if isinstance(output_items, list):
        candidate_specs.append(
            (
                "meta_info.output_token_logprobs",
                _candidate_response_slices(
                    output_items,
                    full_input_len=len(output_items),
                    response_start=max(len(output_items) - response_len, 0),
                    response_len=response_len,
                ),
            )
        )

    errors: list[str] = []
    for source_name, candidates in candidate_specs:
        for idx, candidate in enumerate(candidates):
            values: list[float] = []
            mask: list[int] = []
            candidate_errors: list[str] = []
            missing_positions: list[int] = []
            for pos, item in enumerate(candidate):
                try:
                    values.append(_extract_scalar_logprob(item))
                    mask.append(1)
                except Exception as exc:
                    if _is_missing_logprob_item(item):
                        values.append(0.0)
                        mask.append(0)
                        missing_positions.append(pos)
                        candidate_errors.append(f"pos={pos}: {exc}")
                    else:
                        candidate_errors.append(f"pos={pos}: {exc}")
                        values = []
                        mask = []
                        break
            valid_positions = sum(mask)
            if values and valid_positions > 0:
                missing = len(missing_positions)
                if missing > 0:
                    preview = missing_positions[:8]
                    logger.warning(
                        "Teacher response logprobs contain %s missing positions; they will be masked out. "
                        "source=%s candidate=%s missing_positions=%s response_len=%s response_start=%s "
                        "full_input_len=%s",
                        missing,
                        source_name,
                        idx,
                        preview,
                        response_len,
                        response_start,
                        full_input_len,
                    )
                return values, mask
            if candidate_errors:
                if values and valid_positions == 0:
                    candidate_errors.append("all positions are missing")
                errors.append(f"{source_name}[candidate={idx}]: {'; '.join(candidate_errors[:4])}")

    available_keys = sorted(meta_info.keys()) if isinstance(meta_info, dict) else []
    raise ValueError(
        "Failed to extract teacher response logprobs. "
        f"response_len={response_len}, response_start={response_start}, "
        f"full_input_len={full_input_len}, available_meta_info_keys={available_keys}, "
        f"errors={errors[:4]}"
    )


def _extract_id_logprob_map(raw) -> dict[int, float]:
    mapping: dict[int, float] = {}
    if raw is None:
        return mapping
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                tid = int(k)
            except Exception:
                continue
            try:
                mapping[tid] = _extract_scalar_logprob(v)
            except Exception:
                continue
        return mapping
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                token_id = item.get("token_id", item.get("id", item.get("token")))
                if token_id is None:
                    continue
                try:
                    mapping[int(token_id)] = _extract_scalar_logprob(item)
                except Exception:
                    continue
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                a, b = item[0], item[1]
                try:
                    tid = int(b)
                    lp = _extract_scalar_logprob(a)
                except Exception:
                    try:
                        tid = int(a)
                        lp = _extract_scalar_logprob(b)
                    except Exception:
                        continue
                mapping[tid] = lp
    return mapping


def _extract_response_aux_id_logprob_maps(
    teacher_output: dict,
    response_items: list,
    *,
    full_input_len: int,
    response_start: int,
) -> list[dict[int, float]]:
    meta_info = teacher_output.get("meta_info", {}) if isinstance(teacher_output, dict) else {}
    sidecar = None
    for key in [
        "input_token_ids_logprobs",
        "token_ids_logprob",
        "input_top_logprobs",
        "input_token_top_logprobs",
    ]:
        if key in meta_info:
            sidecar = meta_info[key]
            break

    if sidecar is not None:
        # Reuse the same response alignment as input_token_logprobs rather than
        # inferring offsets from the sidecar length alone.
        response_sidecar = _slice_response_items(
            sidecar,
            full_input_len=full_input_len,
            response_start=response_start,
            response_len=len(response_items),
        )
    else:
        response_sidecar = [None] * len(response_items)

    aux_maps: list[dict[int, float]] = []
    for pos, item in enumerate(response_items):
        aux_map = {}
        if pos < len(response_sidecar):
            aux_map = _extract_id_logprob_map(response_sidecar[pos])
        if not aux_map and isinstance(item, dict):
            for key in ["token_ids_logprob", "top_logprobs", "top_logprob", "logprobs"]:
                if key in item:
                    aux_map = _extract_id_logprob_map(item[key])
                    if aux_map:
                        break
        if not aux_map and isinstance(item, (list, tuple)) and len(item) >= 3:
            aux_map = _extract_id_logprob_map(item[2])
        aux_maps.append(aux_map)
    return aux_maps


def _extract_response_top_logprob_maps(
    teacher_output: dict,
    response_items: list,
    *,
    full_input_len: int,
    response_start: int,
) -> list[dict[int, float]]:
    meta_info = teacher_output.get("meta_info", {}) if isinstance(teacher_output, dict) else {}
    sidecar = None
    for key in ["input_top_logprobs", "input_token_top_logprobs"]:
        if key in meta_info:
            sidecar = meta_info[key]
            break

    if sidecar is not None:
        response_sidecar = _slice_response_items(
            sidecar,
            full_input_len=full_input_len,
            response_start=response_start,
            response_len=len(response_items),
        )
    else:
        response_sidecar = [None] * len(response_items)

    aux_maps: list[dict[int, float]] = []
    for pos, item in enumerate(response_items):
        aux_map = {}
        if pos < len(response_sidecar):
            aux_map = _extract_id_logprob_map(response_sidecar[pos])
        if not aux_map and isinstance(item, dict):
            for key in ["top_logprobs", "top_logprob", "logprobs"]:
                if key in item:
                    aux_map = _extract_id_logprob_map(item[key])
                    if aux_map:
                        break
        if not aux_map and isinstance(item, (list, tuple)) and len(item) >= 3:
            aux_map = _extract_id_logprob_map(item[2])
        aux_maps.append(aux_map)
    return aux_maps


def _extract_requested_token_log_probs(
    aux_maps: list[dict[int, float]],
    requested_token_ids: list[list[int]],
    *,
    missing_error_prefix: str,
) -> list[list[float]]:
    rows: list[list[float]] = []
    for pos, tok_ids in enumerate(requested_token_ids):
        aux_map = aux_maps[pos] if pos < len(aux_maps) else {}
        if not aux_map:
            raise ValueError(f"{missing_error_prefix} at position {pos}: token-level auxiliary logprobs are missing.")
        row = []
        for tid in tok_ids:
            if int(tid) not in aux_map:
                raise ValueError(f"{missing_error_prefix} at position {pos}: token id {tid} is missing.")
            row.append(float(aux_map[int(tid)]))
        rows.append(row)
    return rows


def _extract_optional_requested_token_log_probs(
    aux_maps: list[dict[int, float]],
    requested_token_ids: list[list[int]],
    *,
    fallback_aux_maps: list[dict[int, float]] | None = None,
    missing_value: float = float("nan"),
    warning_prefix: str | None = None,
) -> list[list[float]]:
    rows: list[list[float]] = []
    fallback_positions: list[int] = []
    missing_positions: list[int] = []
    for pos, tok_ids in enumerate(requested_token_ids):
        aux_map = aux_maps[pos] if pos < len(aux_maps) else {}
        fallback_map = fallback_aux_maps[pos] if fallback_aux_maps is not None and pos < len(fallback_aux_maps) else {}
        row = []
        used_fallback = False
        has_missing = False
        for tid in tok_ids:
            tid = int(tid)
            if tid in aux_map:
                row.append(float(aux_map[tid]))
            elif tid in fallback_map:
                row.append(float(fallback_map[tid]))
                used_fallback = True
            else:
                row.append(float(missing_value))
                has_missing = True
        if used_fallback:
            fallback_positions.append(pos)
        if has_missing:
            missing_positions.append(pos)
        rows.append(row)

    if warning_prefix is not None and (fallback_positions or missing_positions):
        count = _OPTIONAL_LOGPROB_WARNING_COUNTS.get(warning_prefix, 0) + 1
        _OPTIONAL_LOGPROB_WARNING_COUNTS[warning_prefix] = count
        if count <= 3 or count % 50 == 0:
            suffix = ""
            if count > 3:
                suffix = f" (occurrence={count}; intermediate warnings suppressed)"
            logger.warning(
                "%s: exact_positions=%d/%d, fallback_positions=%s, missing_positions=%s%s",
                warning_prefix,
                len(requested_token_ids) - len(fallback_positions) - len(missing_positions),
                len(requested_token_ids),
                fallback_positions[:20],
                missing_positions[:20],
                suffix,
            )
    return rows


def _extract_teacher_topk_token_ids(
    aux_maps: list[dict[int, float]],
    topk: int,
) -> list[list[int]]:
    teacher_topk_ids: list[list[int]] = []
    for pos, aux_map in enumerate(aux_maps):
        if not aux_map:
            raise ValueError(f"Missing teacher top-k logprobs at position {pos}.")
        sorted_pairs = sorted(aux_map.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_pairs) < topk:
            raise ValueError(
                f"Teacher top-k at position {pos} has only {len(sorted_pairs)} entries, expected at least {topk}."
            )
        teacher_topk_ids.append([int(token_id) for token_id, _ in sorted_pairs[:topk]])
    return teacher_topk_ids


def _extract_teacher_topk_log_probs(
    aux_maps: list[dict[int, float]],
    requested_token_ids: list[list[int]],
) -> list[list[float]]:
    return _extract_requested_token_log_probs(
        aux_maps,
        requested_token_ids,
        missing_error_prefix="Missing teacher token-level top-k logprobs",
    )


def _combine_requested_token_ids(
    primary: list[list[int]] | None,
    extra: list[list[int]] | None,
) -> list[list[int]] | None:
    if primary is None and extra is None:
        return None
    if primary is None:
        return [list(row) for row in extra]
    if extra is None:
        return [list(row) for row in primary]
    if len(primary) != len(extra):
        raise ValueError(f"Requested token-id length mismatch: {len(primary)} vs {len(extra)}.")
    combined: list[list[int]] = []
    for row_primary, row_extra in zip(primary, extra, strict=False):
        merged = list(dict.fromkeys([int(x) for x in row_primary] + [int(x) for x in row_extra]))
        combined.append(merged)
    return combined


def _get_sample_student_topk_ids(sample: Sample) -> list[list[int]] | None:
    if getattr(sample, "opd_diag_student_topk_token_ids", None) is not None:
        return getattr(sample, "opd_diag_student_topk_token_ids")
    if getattr(sample, "opd_topk_token_ids", None) is not None:
        return getattr(sample, "opd_topk_token_ids")
    return None


async def reward_func(args, sample, **kwargs):
    evaluation = bool(kwargs.get("evaluation", False))
    teacher_input_ids = _build_teacher_input_ids(args, sample)
    teacher_prompt_input_ids = _build_teacher_prompt_input_ids(args, sample)
    response_len = int(sample.response_length or 0)
    response_start = max(len(teacher_input_ids) - response_len, 0)
    use_opd_sglang = getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang"
    use_topk_kl = use_opd_sglang and getattr(args, "opd_kl_mode", "token_reverse_kl") in (
        "full_vocab_topk_reverse_kl",
        "topk_reverse_kl_notail",
        "topk_reverse_kl_notail_sg",
        "topk_reverse_kl_intersect_sg_norm",
    )
    diag_enabled = use_opd_sglang and bool(getattr(args, "opd_token_stats", False))
    diag_topk = int(getattr(args, "opd_token_stats_topk", 50))
    teacher_sft_enabled = use_opd_sglang and bool(getattr(args, "opd_teacher_sft", False)) and not evaluation
    payload = {
        "input_ids": teacher_input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # For OPD we only need teacher logprobs on the generated response tokens.
        # We request from (response_start - 1) rather than response_start because SGLang
        # always prepends a None to input_token_logprobs_val for the boundary token at
        # logprob_start_len (it needs the hidden state one position *before* to compute
        # that token's logprob). By including one extra prompt token, the first response
        # token's logprob is available and the leading None lands on the last prompt token
        # instead, which we discard via the tail-slice in _slice_response_items.
        "logprob_start_len": max(response_start - 1, 0),
    }
    topk_request_size = 0
    requested_topk_token_ids = None
    if use_topk_kl:
        topk_token_ids = getattr(sample, "opd_topk_token_ids", None)
        topk_student_lp = getattr(sample, "opd_topk_student_log_probs", None)
        if topk_token_ids is None or topk_student_lp is None:
            raise ValueError(
                "full_vocab_topk_reverse_kl requires student top-k logprobs from rollout, but they are missing."
            )
        if len(topk_token_ids) != response_len or len(topk_student_lp) != response_len:
            raise ValueError(
                f"Student top-k data length mismatch: token_ids={len(topk_token_ids)}, "
                f"student_logprobs={len(topk_student_lp)}, response_len={response_len}."
            )
        requested_topk_token_ids = topk_token_ids
        topk_request_size = max(topk_request_size, int(getattr(args, "opd_topk", 50)))
    if diag_enabled:
        topk_request_size = max(topk_request_size, diag_topk)


    if topk_request_size > 0:
        payload["top_logprobs_num"] = topk_request_size

    teacher_sft_payload = None
    if teacher_sft_enabled:
        sft_max_response_len = getattr(args, "opd_teacher_sft_max_response_len", None)
        if sft_max_response_len is None:
            sft_max_response_len = getattr(args, "rollout_max_response_len", None)
        if sft_max_response_len is None:
            raise ValueError("--opd-teacher-sft requires --opd-teacher-sft-max-response-len or --rollout-max-response-len.")
        teacher_sft_payload = {
            "input_ids": teacher_prompt_input_ids,
            "sampling_params": {
                "temperature": float(getattr(args, "opd_teacher_sft_temperature", 0.0)),
                "top_p": float(getattr(args, "opd_teacher_sft_top_p", 1.0)),
                "max_new_tokens": int(sft_max_response_len),
                "skip_special_tokens": False,
                "no_stop_trim": True,
            },
            "return_logprob": True,
        }

    session_kwargs = {}
    rm_semaphore = _get_rm_semaphore(args)
    teacher_sft_output = None

    async def _post_with_sem(p):
        # Each request acquires its own slot so rm_max_concurrency limits total
        # outstanding teacher server requests across all callers.
        async with rm_semaphore:
            async with aiohttp.ClientSession(**session_kwargs) as session:
                async with session.post(args.rm_url, json=p) as resp:
                    resp.raise_for_status()
                    return await resp.json()

    if teacher_sft_payload is not None:
        # Fire the cheap logprob pass and the expensive SFT generation in
        # parallel — they are independent and the server can execute both
        # concurrently. Each acquires its own semaphore slot.
        teacher_output, teacher_sft_output = await asyncio.gather(
            _post_with_sem(payload),
            _post_with_sem(teacher_sft_payload),
        )
    else:
        teacher_output = await _post_with_sem(payload)

    teacher_topk_log_probs = None
    teacher_topk_intersect_mask = None
    teacher_topk_token_ids = None
    if use_topk_kl or diag_enabled:
        input_items = teacher_output.get("meta_info", {}).get("input_token_logprobs")
        if input_items is None:
            raise ValueError(
                "OPD token diagnostics require `meta_info.input_token_logprobs` from teacher response."
            )
        response_items = _slice_response_items(
            input_items,
            full_input_len=len(teacher_input_ids),
            response_start=response_start,
            response_len=response_len,
        )
        # Build per-position {token_id: logprob} maps from teacher's top-k output.
        # This is the single source of truth for both topk_kl and diag paths.
        teacher_top_logprob_maps = _extract_response_top_logprob_maps(
            teacher_output=teacher_output,
            response_items=response_items,
            full_input_len=len(teacher_input_ids),
            response_start=response_start,
        )
        if use_topk_kl:
            # For each position look up the teacher logprob of the student's top-k
            # token IDs. Tokens absent from teacher's top-k are filled with the
            # student's own rollout logprob, making that term zero in the KL loss
            # (effectively masking the token rather than penalising it).
            # Also emit a 0/1 intersect mask (student-topk ∩ teacher-topk) for
            # the topk_reverse_kl_intersect_sg_norm mode.
            teacher_topk_log_probs = []
            teacher_topk_intersect_mask = []
            for pos, tok_ids in enumerate(requested_topk_token_ids):
                m = teacher_top_logprob_maps[pos] if pos < len(teacher_top_logprob_maps) else {}
                row_lp = []
                row_mask = []
                for j, t in enumerate(tok_ids):
                    ti = int(t)
                    if ti in m:
                        row_lp.append(float(m[ti]))
                        row_mask.append(1)
                    else:
                        row_lp.append(float(topk_student_lp[pos][j]))
                        row_mask.append(0)
                teacher_topk_log_probs.append(row_lp)
                teacher_topk_intersect_mask.append(row_mask)
        if diag_enabled:
            teacher_topk_token_ids = _extract_teacher_topk_token_ids(
                teacher_top_logprob_maps,
                diag_topk,
            )


    # Dual-metric grading for diagnosis:
    # - strict follows dataset format (boxed/answer) inferred from metadata
    # - relaxed uses auto extraction to reduce format-mismatch false negatives
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    strict_mode = _infer_answer_format(metadata)
    response = sample.response or ""
    label = sample.label or ""
    accuracy_strict = 1.0 if grade_answer_verl(response, label, mode=strict_mode) else 0.0
    # True relaxed metric: accept if any extraction mode gets the correct final answer.
    # This avoids `auto` under-counting when an incorrect "Answer:" line appears before
    # a correct final boxed answer.
    relaxed_hit = (
        grade_answer_verl(response, label, mode="boxed")
        or grade_answer_verl(response, label, mode="answer")
        or grade_answer_verl(response, label, mode="auto")
    )
    accuracy_relaxed = 1.0 if relaxed_hit else 0.0
    return {
        "teacher_output": teacher_output,
        "teacher_input_len": len(teacher_input_ids),
        "teacher_response_start": response_start,
        "teacher_topk_log_probs": teacher_topk_log_probs,
        "teacher_topk_intersect_mask": teacher_topk_intersect_mask,
        "teacher_topk_token_ids": teacher_topk_token_ids,
        "teacher_sft_output": teacher_sft_output,
        "teacher_sft_prompt_input_ids": teacher_prompt_input_ids if teacher_sft_output is not None else None,
        "accuracy_strict": accuracy_strict,
        "accuracy_relaxed": accuracy_relaxed,
        # Backward-compatible alias for existing scripts/checkpoints.
        "accuracy": accuracy_strict,
    }


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    reward_key = getattr(args, "reward_key", None) or "accuracy"
    original_samples = list(samples)
    raw_rewards = []
    response_lengths = [sample.response_length for sample in original_samples]
    use_opd_sglang = getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang"
    opd_distill_max_response_len = int(getattr(args, "opd_distill_max_response_len", 2048))
    diag_enabled = use_opd_sglang and bool(getattr(args, "opd_token_stats", False))
    diag_topk = int(getattr(args, "opd_token_stats_topk", 50))
    teacher_sft_enabled = use_opd_sglang and bool(getattr(args, "opd_teacher_sft", False))

    if use_opd_sglang:
        for sample, response_length in zip(original_samples, response_lengths, strict=False):
            if opd_distill_max_response_len == -1:
                sample.opd_distill_sample_mask = 1
            else:
                sample.opd_distill_sample_mask = 1 if response_length <= opd_distill_max_response_len else 0
            sample.opd_sft_sample_mask = 0

    teacher_outputs = []
    teacher_input_lens = []
    teacher_response_starts = []
    teacher_topk_logprobs_list = []
    teacher_topk_intersect_list = []
    teacher_topk_token_ids_list = []
    teacher_sft_outputs = []
    teacher_sft_prompt_input_ids = []
    for sample in original_samples:
        reward = sample.reward
        if isinstance(reward, dict) and "teacher_output" in reward:
            teacher_output = reward["teacher_output"]
            raw_rewards.append(float(reward.get(reward_key, reward.get("accuracy", 0.0))))
            teacher_input_lens.append(int(reward.get("teacher_input_len", len(sample.tokens))))
            teacher_response_starts.append(
                int(reward.get("teacher_response_start", max(len(sample.tokens) - sample.response_length, 0)))
            )
            teacher_topk_logprobs_list.append(reward.get("teacher_topk_log_probs"))
            teacher_topk_intersect_list.append(reward.get("teacher_topk_intersect_mask"))
            teacher_topk_token_ids_list.append(reward.get("teacher_topk_token_ids"))
            teacher_sft_outputs.append(reward.get("teacher_sft_output"))
            teacher_sft_prompt_input_ids.append(reward.get("teacher_sft_prompt_input_ids"))
        else:
            # Backward-compatible path for historical checkpoints/scripts.
            teacher_output = reward
            raw_rewards.append(0.0)
            teacher_input_lens.append(len(sample.tokens))
            teacher_response_starts.append(max(len(sample.tokens) - sample.response_length, 0))
            teacher_topk_logprobs_list.append(None)
            teacher_topk_intersect_list.append(None)
            teacher_topk_token_ids_list.append(None)
            teacher_sft_outputs.append(None)
            teacher_sft_prompt_input_ids.append(None)
        teacher_outputs.append(teacher_output)

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = []
    for sample, reward, response_length, input_len, response_start in zip(
        original_samples, teacher_outputs, response_lengths, teacher_input_lens, teacher_response_starts, strict=False
    ):
        extracted, extracted_mask = _extract_response_log_probs_with_mask(
            reward,
            full_input_len=input_len,
            response_start=response_start,
            response_len=response_length,
        )
        teacher_log_probs.append(torch.tensor(extracted, dtype=torch.float32))
        if any(x == 0 for x in extracted_mask):
            sample.teacher_logprob_mask = torch.tensor(extracted_mask, dtype=torch.int)
        else:
            sample.teacher_logprob_mask = torch.ones(response_length, dtype=torch.int)

    for sample, t_log_probs in zip(original_samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    use_topk_kl = (
        getattr(args, "use_opd", False)
        and getattr(args, "opd_type", None) == "sglang"
        and getattr(args, "opd_kl_mode", "token_reverse_kl") in (
            "full_vocab_topk_reverse_kl",
            "topk_reverse_kl_notail",
            "topk_reverse_kl_notail_sg",
            "topk_reverse_kl_intersect_sg_norm",
        )
    )
    need_intersect_mask = use_topk_kl and getattr(args, "opd_kl_mode", "token_reverse_kl") == (
        "topk_reverse_kl_intersect_sg_norm"
    )
    if use_topk_kl:
        for idx, (sample, teacher_topk, intersect_mask) in enumerate(
            zip(original_samples, teacher_topk_logprobs_list, teacher_topk_intersect_list, strict=False)
        ):
            student_topk = getattr(sample, "opd_topk_student_log_probs", None)
            if student_topk is None or teacher_topk is None:
                raise ValueError(
                    f"Sample {idx}: missing top-k logprob data for full_vocab_topk_reverse_kl."
                )
            if len(student_topk) != sample.response_length or len(teacher_topk) != sample.response_length:
                raise ValueError(
                    f"Sample {idx}: top-k length mismatch. "
                    f"student={len(student_topk)}, teacher={len(teacher_topk)}, response={sample.response_length}."
                )
            sample.opd_topk_student_log_probs = torch.tensor(student_topk, dtype=torch.float32)
            sample.opd_topk_teacher_log_probs = torch.tensor(teacher_topk, dtype=torch.float32)
            if need_intersect_mask:
                if intersect_mask is None:
                    raise ValueError(
                        f"Sample {idx}: missing teacher_topk_intersect_mask for topk_reverse_kl_intersect_sg_norm."
                    )
                if len(intersect_mask) != sample.response_length:
                    raise ValueError(
                        f"Sample {idx}: intersect_mask length mismatch: "
                        f"{len(intersect_mask)} vs response={sample.response_length}."
                    )
                sample.opd_topk_intersect_mask = torch.tensor(intersect_mask, dtype=torch.int)

    if diag_enabled:
        for idx, (sample, teacher_topk_ids) in enumerate(
            zip(original_samples, teacher_topk_token_ids_list, strict=False)
        ):
            student_topk_ids = _get_sample_student_topk_ids(sample)
            if student_topk_ids is None or teacher_topk_ids is None:
                raise ValueError(f"Sample {idx}: missing student/teacher top-k ids for OPD token diagnostics.")
            if len(student_topk_ids) != sample.response_length or len(teacher_topk_ids) != sample.response_length:
                raise ValueError(
                    f"Sample {idx}: top-k length mismatch for token diagnostics. "
                    f"student={len(student_topk_ids)}, teacher={len(teacher_topk_ids)}, response={sample.response_length}."
                )
            response_tokens = list(sample.tokens[-sample.response_length:]) if sample.response_length > 0 else []
            sample.opd_diag_topk_overlap = torch.tensor(
                [
                    compute_topk_overlap_ratio(student_topk_ids[pos], teacher_topk_ids[pos], diag_topk)
                    for pos in range(sample.response_length)
                ],
                dtype=torch.float32,
            )
            sample.opd_diag_teacher_rank_at_k = torch.tensor(
                [
                    compute_teacher_rank_at_k(response_tokens[pos], teacher_topk_ids[pos], diag_topk)
                    for pos in range(sample.response_length)
                ],
                dtype=torch.float32,
            )

    sft_raw_rewards = []
    sft_scalar_rewards = []
    if teacher_sft_enabled:
        for sample, teacher_sft_output, prompt_input_ids in zip(
            original_samples,
            teacher_sft_outputs,
            teacher_sft_prompt_input_ids,
            strict=False,
        ):
            if not isinstance(teacher_sft_output, dict) or not prompt_input_ids:
                continue
            teacher_response_tokens = _extract_generated_token_ids(teacher_sft_output)
            if not teacher_response_tokens:
                continue

            sft_sample = copy.deepcopy(sample)
            sft_sample.tokens = list(prompt_input_ids) + teacher_response_tokens
            sft_sample.response = str(teacher_sft_output.get("text", "") or "")
            sft_sample.response_length = len(teacher_response_tokens)
            sft_sample.loss_mask = _build_teacher_sft_response_loss_mask(
                args,
                sample,
                prompt_input_ids=list(prompt_input_ids),
                response_token_ids=teacher_response_tokens,
                response_text=sft_sample.response,
            )
            if len(sft_sample.loss_mask) != sft_sample.response_length:
                raise ValueError(
                    "Teacher-SFT loss mask length mismatch: "
                    f"{len(sft_sample.loss_mask)} vs response_length={sft_sample.response_length}."
                )
            sft_sample.reward = 0.0
            sft_sample.rollout_log_probs = [0.0] * sft_sample.response_length
            sft_sample.teacher_log_probs = torch.zeros(sft_sample.response_length, dtype=torch.float32)
            sft_sample.teacher_logprob_mask = torch.zeros(sft_sample.response_length, dtype=torch.int)
            sft_sample.opd_distill_sample_mask = 0
            sft_sample.opd_sft_sample_mask = 1
            sft_sample.opd_topk_token_ids = (
                [[] for _ in range(sft_sample.response_length)]
                if getattr(sample, "opd_topk_token_ids", None) is not None
                else None
            )
            sft_sample.opd_topk_student_log_probs = (
                [[] for _ in range(sft_sample.response_length)]
                if getattr(sample, "opd_topk_student_log_probs", None) is not None
                else None
            )
            sft_sample.opd_topk_teacher_log_probs = (
                [[] for _ in range(sft_sample.response_length)]
                if getattr(sample, "opd_topk_teacher_log_probs", None) is not None
                else None
            )
            sft_sample.opd_topk_intersect_mask = (
                [[] for _ in range(sft_sample.response_length)]
                if getattr(sample, "opd_topk_intersect_mask", None) is not None
                else None
            )
            sft_sample.opd_diag_student_topk_token_ids = None
            sft_sample.opd_diag_topk_overlap = (
                [0.0] * sft_sample.response_length
                if getattr(sample, "opd_diag_topk_overlap", None) is not None
                else None
            )
            sft_sample.opd_diag_teacher_rank_at_k = (
                [0.0] * sft_sample.response_length
                if getattr(sample, "opd_diag_teacher_rank_at_k", None) is not None
                else None
            )
            sft_sample.teacher_tokens = None
            sft_sample.teacher_prompt_length = None
            sft_sample.metadata = dict(sft_sample.metadata or {})
            sft_sample.metadata["opd_teacher_sft"] = True
            sft_sample.metadata["opd_teacher_sft_source_index"] = sample.index
            samples.append(sft_sample)
            sft_raw_rewards.append(0.0)
            sft_scalar_rewards.append(0.0)

    # Interleave SFT samples with their source originals so that every
    # training mini-batch contains a mix of student and SFT samples.
    # Without this, all SFT samples land at the end of the list and entire
    # training steps see only student samples (opd_teacher_sft_loss = 0).
    if sft_raw_rewards:
        n_orig = len(original_samples)
        orig_samples = samples[:n_orig]
        sft_samples = samples[n_orig:]
        orig_raw = raw_rewards[:]  # n_orig entries, not yet extended
        # Interleave: orig_0, sft_0, orig_1, sft_1, ...
        # zip stops at the shorter list; trailing originals are appended after.
        interleaved_samples = [s for pair in zip(orig_samples, sft_samples) for s in pair]
        interleaved_samples.extend(orig_samples[len(sft_samples):])
        interleaved_raw = [r for pair in zip(orig_raw, sft_raw_rewards) for r in pair]
        interleaved_raw.extend(orig_raw[len(sft_raw_rewards):])
        samples[:] = interleaved_samples
        raw_rewards[:] = interleaved_raw
        sft_raw_rewards = []
        sft_scalar_rewards = []

    # Return scalar rewards for GRPO/PPO advantage estimator.
    # When --opd-zero-task-reward is enabled, task rewards are zeroed so training
    # is driven by OPD KL only. Otherwise use raw task rewards.
    if getattr(args, "opd_zero_task_reward", False):
        scalar_rewards = [0.0] * len(samples)
    else:
        scalar_rewards = list(raw_rewards)
        scalar_rewards.extend(sft_scalar_rewards)
    raw_rewards.extend(sft_raw_rewards)

    return raw_rewards, scalar_rewards
