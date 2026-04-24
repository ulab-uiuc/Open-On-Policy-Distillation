"""On-Policy Self-Distillation (OPSD) reward function and post-processing.

In OPSD, the same model serves as both student and teacher. The teacher receives
a *privileged prompt* (original prompt + ground-truth hint) while the student
receives the normal prompt. The learning signal is a full-vocabulary JSD between
their distributions, combined with a standard RL math reward.

IMPORTANT: Because ``--apply-chat-template`` converts ``sample.prompt`` from a
list of message dicts to a rendered string, the data preprocessing step in
``run-qwen3-8b-opsd.sh`` stores the original user question as
``metadata['raw_content']``. This function uses that field to build the
privileged teacher prompt.

Teacher information density modes (``--opsd-teacher-info-mode``):
  - ``full`` (default): Teacher receives the complete reference solution and is
    asked to re-derive the answer via step-by-step reasoning.  This is the
    original OPSD behaviour from arXiv 2601.18734.
  - ``answer_only``: Teacher only receives the ground-truth final answer
    (no reference reasoning steps).  The teacher prompt stays in the same
    "response" regime but the privileged information is reduced to a bare
    answer, lowering information density.
  - ``masked_reasoning``: Teacher still receives the reference solution in its
    privileged prompt, but the reference solution *text* itself is randomly
    masked at the token level before being inserted.  At rate
    ``--opsd-reasoning-mask-ratio`` every token of the reference solution is
    replaced with a mask placeholder (``"___"``), so the teacher sees only a
    partial view of the reasoning chain.  This ablates information density
    inside the teacher prompt without changing the JSD computation.
  - ``conciseness``: OPSDC mode (arXiv 2603.05433).  Teacher receives a
    *conciseness instruction* prepended to the original problem; the student
    receives only the original problem.  No ground-truth solution is required.
    Use with ``--opsd-loss-type reverse_kl`` and ``--opsd-pure-mode``.
    The conciseness instruction text is controlled by
    ``--opsd-conciseness-instruction``.

Exported symbols used via CLI args:
    --custom-rm-path examples.on_policy_distillation.on_policy_self_distillation.reward_func
    --custom-reward-post-process-path examples.on_policy_distillation.on_policy_self_distillation.post_process_rewards
    --custom-reward-post-process-path examples.on_policy_distillation.on_policy_self_distillation.post_process_rewards_grpo_only
    --reward-key math_reward
"""

import json
import logging
import random
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)

# Default format suffix used when metadata['format_instruction'] is absent
# (e.g. data preprocessed before this feature was added).
_DEFAULT_FORMAT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def _first_nonempty_text(*values: str | None) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return ""


@lru_cache(maxsize=1)
def _get_tokenizer(model_path: str):
    """Lazy-load and cache the HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"OPSD: loaded tokenizer from {model_path}")
    return tokenizer


def _grade_math(response: str, label: str, answer_format: str = "auto") -> float:
    """Return 1.0 if the response correctly answers the math problem, else 0.0."""
    from slime.rollout.rm_hub import grade_answer_verl

    if not label:
        return 0.0
    return 1.0 if grade_answer_verl(response, label, mode=answer_format) else 0.0


def _extract_mode_from_metadata(metadata: dict) -> str:
    """Infer answer extraction mode from the format_instruction stored in metadata.

    Returns ``"boxed"``, ``"answer"``, or ``"auto"`` (fallback).
    """
    fmt = metadata.get("format_instruction", "") or ""
    if "boxed" in fmt:
        return "boxed"
    if "Answer" in fmt:
        return "answer"
    return "auto"


# ---------------------------------------------------------------------------
# _mask_reference_solution
# ---------------------------------------------------------------------------

_FALLBACK_MASK_TOKEN = "[MASK]"  # Standard BERT-style mask token


def _get_mask_token(tokenizer) -> str:
    """Return the tokenizer's own mask token, or ``[MASK]`` as fallback.

    Causal LMs (Qwen, LLaMA, …) are trained without a dedicated mask token,
    but ``[MASK]`` appears abundantly in their pretraining corpora (BERT papers,
    code, documentation), so the model recognises its "hidden / redacted" semantics.
    """
    mask_token = getattr(tokenizer, "mask_token", None)
    return mask_token if mask_token else _FALLBACK_MASK_TOKEN


def _build_hidden_think_prefix_tokens(
    tokenizer, user_content: str, think_content: str, max_think_tokens: int = -1
) -> list[int]:
    """Build teacher prefix tokens with privileged info hidden inside a <think> block.

    The resulting token sequence is:
        [chat_template ... <think>\\n] + think_content + "\\n</think>\\n"

    The caller appends student response tokens to form sample.teacher_tokens and
    sets sample.teacher_prompt_length = len(returned list).

    This always uses enable_thinking=True for the teacher so the generation prompt
    ends with "<think>\\n" regardless of what the student uses.  The student is
    expected to run with enable_thinking=False (no <think> tokens in its response).

    Args:
        max_think_tokens: If > 0, truncate think_content at this many tokens before
            embedding it in the think block.  Use this to prevent OOM when
            think_content is a long reference reasoning chain (hidden_think_full).
            -1 (default) means no limit.
    """
    messages = [{"role": "user", "content": user_content}]
    try:
        think_open_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        # Tokenizer does not support enable_thinking; fall back without it.
        think_open_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Optionally truncate think_content at the token level to avoid OOM from
    # very long reference solutions (e.g. OpenThoughts-114k DeepSeek chains).
    if max_think_tokens > 0 and think_content:
        think_ids = tokenizer.encode(think_content, add_special_tokens=False)
        if len(think_ids) > max_think_tokens:
            think_content = tokenizer.decode(think_ids[-max_think_tokens:], skip_special_tokens=True)

    prefix_text = think_open_text + think_content + "\n</think>\n"
    return tokenizer.encode(prefix_text, add_special_tokens=False)


def _mask_reference_solution(
    text: str,
    tokenizer,
    mask_ratio: float,
) -> str:
    """Randomly mask ``mask_ratio`` fraction of tokens in ``text``.

    Each token is independently replaced by the tokenizer's mask token
    (``tokenizer.mask_token`` if available, otherwise ``[MASK]``) with
    probability ``mask_ratio``.  The masked token sequence is then decoded back
    to a string and returned.

    This is applied to the **reference solution** inside the teacher's
    privileged prompt, reducing the amount of reasoning information available
    to the teacher while preserving the overall prompt structure.

    Args:
        text:        The reference solution text to partially mask.
        tokenizer:   HuggingFace tokenizer (used for token-level granularity).
        mask_ratio:  Fraction of tokens to replace (0.0 = no masking, 1.0 = all masked).

    Returns:
        The (partially) masked text string, ready to insert into the prompt.
    """
    if not text or mask_ratio <= 0.0:
        return text

    mask_placeholder = _get_mask_token(tokenizer)

    # Encode to token IDs (no special tokens)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return text

    # Decode each individual token so we can replace them selectively.
    token_strings = [tokenizer.decode([tid], skip_special_tokens=True) for tid in token_ids]

    masked_parts = []
    for tok_str in token_strings:
        if random.random() < mask_ratio:
            masked_parts.append(mask_placeholder)
        else:
            masked_parts.append(tok_str)

    return "".join(masked_parts)


# ---------------------------------------------------------------------------
# reward_func  (async, called per sample during rollout)
# ---------------------------------------------------------------------------

async def reward_func(args, sample, **kwargs):
    """Compute math reward for a single sample.

    Returns a dict so that ``--reward-key math_reward`` can extract the scalar.
    """
    response = sample.response
    label = sample.label or ""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    answer_format = _extract_mode_from_metadata(metadata)
    math_reward = _grade_math(response, label, answer_format)
    return {"math_reward": math_reward}


# ---------------------------------------------------------------------------
# post_process_rewards  (called on the full batch after rollout)
# ---------------------------------------------------------------------------

def post_process_rewards(args, samples, **kwargs):
    """Process rewards and build OPSD teacher tokens.

    For every sample:
    1. Extract the scalar math reward.
    2. Build a *privileged* teacher prompt according to ``--opsd-teacher-info-mode``:

       - ``full``            : Teacher receives the complete reference solution
                               plus a step-by-step re-derivation instruction.
                               This is the original OPSD behaviour.
       - ``answer_only``     : Teacher only receives the ground-truth final answer
                               (no reasoning steps), lowering information density.
       - ``masked_reasoning``: Teacher receives the reference solution, but the
                               reference solution *text* is first randomly masked at
                               token level (rate ``--opsd-reasoning-mask-ratio``),
                               reducing the reasoning information in the prompt.
       - ``conciseness``     : OPSDC mode (arXiv 2603.05433). Teacher receives a
                               conciseness instruction prepended to the original
                               problem. No ground-truth solution is needed.

    3. Tokenize the teacher prompt and concatenate with student response tokens.
    4. Store ``sample.teacher_tokens`` and ``sample.teacher_prompt_length``.
    5. Perform GRPO group normalisation on the rewards.

    Returns
    -------
    raw_rewards : list[float]
        Un-normalised scalar rewards (for logging).
    normalised_rewards : list[float]
        Rewards after GRPO mean/std normalisation (for advantage computation).
    """
    tokenizer = _get_tokenizer(args.hf_checkpoint)

    # Read teacher info mode from args (default to 'full' for backward compat)
    teacher_info_mode = getattr(args, "opsd_teacher_info_mode", "full")
    reasoning_mask_ratio = float(getattr(args, "opsd_reasoning_mask_ratio", 0.5))

    # Derive enable_thinking from --apply-chat-template-kwargs so the teacher
    # prompt template exactly matches the student's generation mode.
    # If the student runs with enable_thinking=false the teacher must also use
    # enable_thinking=False, otherwise the teacher prompt ends with <think>\n
    # but the student response tokens contain no <think>, causing a systematic
    # KL mismatch at the very first token.
    _chat_template_kwargs: dict = {}
    _raw_kwargs = getattr(args, "apply_chat_template_kwargs", None)
    if _raw_kwargs:
        if isinstance(_raw_kwargs, dict):
            _chat_template_kwargs = _raw_kwargs
        else:
            try:
                _chat_template_kwargs = json.loads(_raw_kwargs)
            except Exception:
                pass
    _teacher_enable_thinking: bool = _chat_template_kwargs.get("enable_thinking", True)
    _teacher_think_max_tokens: int = int(getattr(args, "opsd_teacher_think_max_tokens", -1))

    # # Debug: log _teacher_enable_thinking to verify it is correctly derived.
    # _debug_msg = (
    #     f"[OPSD DEBUG] apply_chat_template_kwargs raw type={type(_raw_kwargs).__name__}, "
    #     f"value={_raw_kwargs!r}\n"
    #     f"[OPSD DEBUG] _chat_template_kwargs={_chat_template_kwargs!r}\n"
    #     f"[OPSD DEBUG] _teacher_enable_thinking={_teacher_enable_thinking}\n"
    # )
    # logger.info(_debug_msg)
    # import os as _os
    # _debug_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "debug_teacher_enable_thinking.txt")
    # with open(_debug_path, "a") as _f:
    #     _f.write(_debug_msg)

    # Per-sample format_instruction is read from metadata below; this is only
    # used as a last-resort fallback when the field is missing from metadata.
    _global_format_suffix = _DEFAULT_FORMAT_SUFFIX

    # ---- 1. Extract raw math rewards ----
    raw_rewards = []
    for sample in samples:
        r = sample.get_reward_value(args)
        if isinstance(r, dict):
            r = r.get("math_reward", 0.0)
        raw_rewards.append(float(r))

    # ---- 2-4. Construct teacher tokens for each sample ----
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        raw_content = metadata.get("raw_content", "")
        label = sample.label or ""
        reference_solution = _first_nonempty_text(
            metadata.get("reference_solution"),
            metadata.get("solution"),
            label,
        )

        # Format instruction used by both student and teacher.
        # Stored in metadata by preprocess_dataset.py (--answer-format flag).
        # Falls back to the \boxed{} suffix for data pre-dating this feature.
        format_instruction = metadata.get("format_instruction") or _global_format_suffix

        if not raw_content:
            # Fallback: try to extract from sample.prompt if it's still a list of dicts
            if isinstance(sample.prompt, list):
                for msg in sample.prompt:
                    if msg.get("role") == "user":
                        raw_content = msg.get("content", "")
                        break
            if not raw_content:
                logger.warning(
                    "OPSD: raw_content not found in metadata and prompt is not a message list. "
                    "Teacher tokens will be empty for this sample."
                )

        # ------------------------------------------------------------------
        # Build privileged teacher prompt based on teacher_info_mode.
        #
        # Standard modes (answer_only / masked_reasoning / pi / conciseness / full):
        #   Set teacher_user_content; the bottom of the loop applies the chat
        #   template uniformly with enable_thinking=_teacher_enable_thinking.
        #
        # hidden_think modes (hidden_think / hidden_think_full):
        #   Set teacher_prompt_tokens directly (bypassing apply_chat_template).
        #   The teacher always uses enable_thinking=True so its generation prefix
        #   ends with "<think>\n"; privileged info is then prepended inside the
        #   think block, followed by "</think>\n".  The student is expected to run
        #   with enable_thinking=False (no <think> tokens in its response).
        # ------------------------------------------------------------------
        teacher_user_content = None     # used by standard modes
        teacher_prompt_tokens = None    # used by hidden_think modes (skips template step)

        if _teacher_enable_thinking and teacher_info_mode in ("hidden_think", "hidden_think_full"):
            logger.warning(
                "OPSD: hidden_think mode is designed for student enable_thinking=False "
                "(--apply-chat-template-kwargs '{\"enable_thinking\":false}'). "
                "Detected enable_thinking=True for the student; the student response "
                "will contain <think> tokens which may cause a token-distribution mismatch."
            )

        if teacher_info_mode == "hidden_think":
            # Privileged answer hint hidden inside the teacher's <think> block.
            # Both teacher and student receive the identical user message; the only
            # difference is that the teacher's generation context is prefilled with
            # the answer inside <think>…</think> before the student response tokens.
            # This tests whether "silent" knowledge of the answer can be distilled
            # into a student that never uses the thinking mode.
            student_user_content = metadata.get("student_user_content") or raw_content
            answer_hint = label if label else reference_solution
            if answer_hint:
                # Wrap in \boxed{} when the dataset uses boxed answer format so the
                # hint matches the output format the model is trained to produce.
                if "boxed" in format_instruction:
                    formatted_answer = f"\\boxed{{{answer_hint}}}"
                else:
                    formatted_answer = answer_hint
                think_content = f"The answer to this problem is {formatted_answer}."
            else:
                think_content = ""
            _debug_msg = (
                f"[OPSD DEBUG] teacher_user_content={student_user_content}\n"
                f"[OPSD DEBUG] think_content={think_content}\n"
            )
            logger.info(_debug_msg)
            teacher_prompt_tokens = _build_hidden_think_prefix_tokens(
                tokenizer, student_user_content, think_content,
                max_think_tokens=_teacher_think_max_tokens,
            )

        elif teacher_info_mode == "hidden_think_full":
            # Full reference reasoning chain hidden inside the teacher's <think> block.
            # Requires a dataset that provides reference_solution (e.g. OpenThoughts-114k).
            # When reference_solution is empty, the think block is left empty and the
            # teacher degrades to the same context as the student.
            # Use --opsd-teacher-think-max-tokens to cap very long reference solutions
            # and prevent OOM (recommended: 4096 for OpenThoughts-114k).
            student_user_content = metadata.get("student_user_content") or raw_content
            think_content = reference_solution  # may be empty for eval-only datasets
            _debug_msg = (
                f"[OPSD DEBUG] teacher_user_content={student_user_content}\n"
                f"[OPSD DEBUG] think_content={think_content}\n"
            )
            logger.info(_debug_msg)
            teacher_prompt_tokens = _build_hidden_think_prefix_tokens(
                tokenizer, student_user_content, think_content,
                max_think_tokens=_teacher_think_max_tokens,
            )

        elif teacher_info_mode == "answer_only":
            # Teacher only gets the final answer, not the full reasoning chain.
            # We still frame it as "here is the answer, now solve it yourself"
            # so the teacher stays in the response distribution.
            # Use label (extracted final answer) preferentially over reference_solution
            # (which may contain full reasoning steps for datasets like OpenThoughts-114k).
            #
            # Use student_user_content (which already has the correct format instruction
            # embedded for all dataset types) as the base.  This avoids duplicating the
            # format instruction for DAPO datasets where raw_content == student_user_content
            # (i.e., the format instruction is already baked into the prompt text).
            # The pattern mirrors the 'conciseness' and 'pi' modes.
            _ANSWER_ONLY_PROMPT = (
                "The correct final answer to this problem is: {answer}\n"
                "Now solve the problem yourself step by step and arrive at the same answer:"
            )
            answer_hint = label if label else reference_solution
            student_user_content = metadata.get("student_user_content") or raw_content
            if answer_hint:
                teacher_user_content = (
                    f"{student_user_content}\n\n"
                    + _ANSWER_ONLY_PROMPT.format(answer=answer_hint)
                )
            else:
                # No reference answer available; fall back to standard student prompt.
                teacher_user_content = student_user_content

        elif teacher_info_mode == "masked_reasoning":
            # Teacher receives the full reference solution structure, but the
            # reference solution *text* is randomly token-masked at mask_ratio.
            # This degrades the information density inside the privileged prompt
            # while keeping the prompt template unchanged.
            _TRANSITION_PROMPT = (
                "After understanding the reference solution and the rationale behind each step, "
                "now articulate your own step-by-step reasoning that derives the same final answer "
                "to the problem above:"
            )
            if reference_solution and reasoning_mask_ratio > 0.0:
                masked_solution = _mask_reference_solution(
                    reference_solution, tokenizer, mask_ratio=reasoning_mask_ratio
                )
            else:
                masked_solution = reference_solution

            if masked_solution:
                teacher_user_content = (
                    f"{raw_content}\n\n"
                    f"Here is a reference solution to this problem:\n{masked_solution}\n"
                    f"{_TRANSITION_PROMPT}\n"
                    f"{format_instruction}"
                )
            else:
                teacher_user_content = f"{raw_content}\n\n{format_instruction}"

        elif teacher_info_mode == "pi":
            # Generic Prompt Internalization mode.
            # Teacher receives the alignment instruction stored per-sample in
            # metadata['pi_instruction'] prepended to the bare student prompt.
            # Student always sees only the bare prompt (no alignment instruction).
            pi_instruction = metadata.get("pi_instruction", "")
            student_user_content = metadata.get("student_user_content") or raw_content
            teacher_user_content = (
                f"{pi_instruction}\n\n{student_user_content}"
                if pi_instruction
                else student_user_content
            )

        elif teacher_info_mode == "conciseness":
            # OPSDC mode (arXiv 2603.05433): teacher gets a conciseness instruction
            # prepended to the original problem; NO ground-truth solution is needed.
            # Student prompt: original problem only (unchanged).
            # Teacher prompt: conciseness_instruction + student_user_content.
            # Per Figure 2 of the paper: teacher = [c] + [student prompt content],
            # so both share the same answer-format instructions and problem text.
            #
            # student_user_content already contains the format instruction (appended
            # by preprocess_dataset.py), so we do NOT append it again here.
            conciseness_instruction = getattr(
                args,
                "opsd_conciseness_instruction",
                (
                    "Solve the following math problem concisely and correctly. "
                    "Be direct -- avoid unnecessary elaboration, redundant steps, or restating the problem. "
                    "Focus only on the key reasoning steps needed to reach the answer."
                ),
            )
            # Use the full student user content (problem + format instruction) if stored
            # in metadata['student_user_content'] by the data preprocessing script.
            # Fall back to raw_content for backward compatibility.
            student_user_content = metadata.get("student_user_content") or raw_content
            teacher_user_content = f"{conciseness_instruction}\n\n{student_user_content}"

        else:
            # ``full`` mode (default): teacher gets the complete reference solution.
            # Original opsd_ucla transition_prompt (data_collator.py, line ~90):
            #   "After understanding the reference solution and the rationale behind each
            #    step, now articulate your own step-by-step reasoning that derives the
            #    same final answer to the problem below:"
            _TRANSITION_PROMPT = (
                "After understanding the reference solution and the rationale behind each step, "
                "now articulate your own step-by-step reasoning that derives the same final answer "
                "to the problem above:"
            )
            if reference_solution:
                teacher_user_content = (
                    f"{raw_content}\n\n"
                    f"Here is a reference solution to this problem:\n{reference_solution}\n"
                    f"{_TRANSITION_PROMPT}\n"
                    f"{format_instruction}"
                )
            else:
                teacher_user_content = f"{raw_content}\n\n{format_instruction}"

        # Student response tokens (last response_length tokens of the full sequence)
        response_tokens = sample.tokens[-sample.response_length:]

        if teacher_prompt_tokens is None:
            # Standard path: apply chat template to the user-content string set above.
            # enable_thinking must match --apply-chat-template-kwargs used for student
            # rollout to avoid a systematic KL mismatch at the very first token
            # (teacher prompt ending with <think>\n while student response has no <think>).
            teacher_messages = [{"role": "user", "content": teacher_user_content}]
            teacher_prompt_text = tokenizer.apply_chat_template(
                teacher_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=_teacher_enable_thinking
            )
            teacher_prompt_tokens = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)

        # Teacher tokens = privileged prefix + student response
        sample.teacher_tokens = teacher_prompt_tokens + list(response_tokens)
        sample.teacher_prompt_length = len(teacher_prompt_tokens)

    # ---- 5. GRPO group normalisation ----
    n = getattr(args, "n_samples_per_prompt", 1)
    rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float)

    if n > 1 and len(raw_rewards) >= n:
        rewards_tensor = rewards_tensor.view(-1, n)
        mean = rewards_tensor.mean(dim=-1, keepdim=True)
        std = rewards_tensor.std(dim=-1, keepdim=True)
        normalised = (rewards_tensor - mean) / (std + 1e-6)
        # Zero out groups where all rewards are the same (zero std → no signal)
        zero_std_mask = std.squeeze(-1) < 1e-8
        normalised[zero_std_mask] = 0.0
        normalised_rewards = normalised.flatten().tolist()
    else:
        normalised_rewards = list(raw_rewards)

    return raw_rewards, normalised_rewards


# ---------------------------------------------------------------------------
# post_process_rewards_grpo_only
# ---------------------------------------------------------------------------

def post_process_rewards_grpo_only(args, samples, **kwargs):
    """Pure GRPO reward post-processing: math reward + group normalisation, no teacher tokens.

    Use this when privileged information is already baked into the rollout prompt
    (e.g. via ``filter_openthoughts_math.py --privileged-mode``).  The rollout is
    fully on-policy under the privileged prompt, so no token replacement or teacher
    forward pass is needed here.

    CLI wiring::

        --custom-rm-path      examples.on_policy_distillation.on_policy_self_distillation.reward_func
        --custom-reward-post-process-path \\
            examples.on_policy_distillation.on_policy_self_distillation.post_process_rewards_grpo_only
        --reward-key math_reward
    """
    raw_rewards = []
    for sample in samples:
        r = sample.get_reward_value(args)
        if isinstance(r, dict):
            r = r.get("math_reward", 0.0)
        raw_rewards.append(float(r))

    n = getattr(args, "n_samples_per_prompt", 1)
    rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float)

    if n > 1 and len(raw_rewards) >= n:
        rewards_tensor = rewards_tensor.view(-1, n)
        mean = rewards_tensor.mean(dim=-1, keepdim=True)
        std = rewards_tensor.std(dim=-1, keepdim=True)
        normalised = (rewards_tensor - mean) / (std + 1e-6)
        zero_std_mask = std.squeeze(-1) < 1e-8
        normalised[zero_std_mask] = 0.0
        normalised_rewards = normalised.flatten().tolist()
    else:
        normalised_rewards = list(raw_rewards)

    return raw_rewards, normalised_rewards


