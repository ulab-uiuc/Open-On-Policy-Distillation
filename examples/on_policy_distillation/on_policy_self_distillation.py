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

Exported symbols used via CLI args:
    --custom-rm-path examples.on_policy_distillation.on_policy_self_distillation.reward_func
    --custom-reward-post-process-path examples.on_policy_distillation.on_policy_self_distillation.post_process_rewards
    --reward-key math_reward
"""

import logging
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_tokenizer(model_path: str):
    """Lazy-load and cache the HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"OPSD: loaded tokenizer from {model_path}")
    return tokenizer


def _grade_math(response: str, label: str) -> float:
    """Return 1.0 if the response correctly answers the math problem, else 0.0."""
    from slime.rollout.rm_hub import grade_answer_verl

    if not label:
        return 0.0
    return 1.0 if grade_answer_verl(response, label) else 0.0


# ---------------------------------------------------------------------------
# reward_func  (async, called per sample during rollout)
# ---------------------------------------------------------------------------

async def reward_func(args, sample, **kwargs):
    """Compute math reward for a single sample.

    Returns a dict so that ``--reward-key math_reward`` can extract the scalar.
    """
    response = sample.response
    label = sample.label or ""
    math_reward = _grade_math(response, label)
    return {"math_reward": math_reward}


# ---------------------------------------------------------------------------
# post_process_rewards  (called on the full batch after rollout)
# ---------------------------------------------------------------------------

def post_process_rewards(args, samples, **kwargs):
    """Process rewards and build OPSD teacher tokens.

    For every sample:
    1. Extract the scalar math reward.
    2. Build a *privileged* teacher prompt that includes the ground-truth answer.
       Uses ``metadata['raw_content']`` (the original question text) because
       ``sample.prompt`` is already a rendered chat-template string.
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

        # Build privileged teacher prompt: user question + generation prompt + hint.
        # We use add_generation_prompt=True so the assistant turn stays OPEN
        # (no <|im_end|> inserted), then append the hint text.  This way the
        # student response tokens that follow sit inside a valid assistant turn,
        # matching the context the student model sees during its own forward pass.
        teacher_messages = [
            {"role": "user", "content": raw_content},
        ]

        # Tokenize teacher prompt (privileged prompt including answer hint)
        teacher_prompt_text = tokenizer.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=True
        )
        teacher_prompt_text += f"The answer is {label}. Let me verify this step by step.\n"
        teacher_prompt_tokens = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)

        # Student response tokens (last response_length tokens of the full sequence)
        response_tokens = sample.tokens[-sample.response_length:]

        # Teacher tokens = privileged prompt + student response
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
