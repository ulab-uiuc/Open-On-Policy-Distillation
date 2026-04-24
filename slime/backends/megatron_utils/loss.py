from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.opd_token_stats import build_token_repetition_mask, compute_token_stats_metrics
from slime.utils.opsd_grpo_reward import recompute_local_opsd_grpo_r2_rewards
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_log_probs,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    if args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = mpu.get_context_parallel_rank()
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits: torch.Tensor,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> None:
    """Redistribute response tensors from allgather-CP layout to zigzag ring-attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits: Model output used only to determine the local sequence length
            (``logits.size(1)``).
        args: Configuration (needs ``qkv_format``).
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()

    logits_local_len = logits.size(1)  # logits shape: [1, T_local, ...]
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        # Reconstruct full response tensors with each rank's contiguous contribution
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            if e <= s:
                # This rank has no response logprobs for this sample
                full_resp = torch.zeros(
                    response_length,
                    dtype=value.dtype,
                    device=value.device,
                    requires_grad=True,
                )
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                full_resp = F.pad(value, (resp_start, response_length - resp_end))

            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        # Single differentiable all-reduce to gather full response from all CP ranks
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(
            zip(all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            new_values.append(
                slice_log_prob_with_cp(full_resp, total_length, response_length, args.qkv_format, max_seq_len)
            )

        res[key] = new_values


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    log_probs_list = []
    entropy_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            mpu.get_tensor_model_parallel_group(),
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=logits.device), res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=logits.device), res


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `use_opd` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of student log-probability tensors.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None:
        return

    raw_distill_sample_mask = rollout_data.get("opd_distill_sample_mask")
    if raw_distill_sample_mask is None:
        opd_distill_sample_mask = [1.0] * len(advantages)
    else:
        if len(raw_distill_sample_mask) != len(advantages):
            raise ValueError(
                "opd_distill_sample_mask length mismatch: "
                f"{len(raw_distill_sample_mask)} vs {len(advantages)}."
            )
        opd_distill_sample_mask = []
        for value in raw_distill_sample_mask:
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        "opd_distill_sample_mask tensor entries must be scalar. "
                        f"Got shape={tuple(value.shape)}."
                    )
                value = float(value.item())
            else:
                value = float(value)
            opd_distill_sample_mask.append(1.0 if value > 0 else 0.0)

    kl_mode = getattr(args, "opd_kl_mode", "token_reverse_kl")
    device = student_log_probs[0].device

    def _compute_topk_tail_reverse_kl(
        student_topk_lp: torch.Tensor,
        teacher_topk_lp: torch.Tensor,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Inputs are log probabilities over the full vocabulary, restricted to
        # student top-k token positions: [response_len, k].
        p_s = student_topk_lp.exp()
        p_t = teacher_topk_lp.exp()

        p_s_mass = p_s.sum(dim=-1).clamp(min=0.0, max=1.0)
        p_t_mass = p_t.sum(dim=-1).clamp(min=0.0, max=1.0)

        kl_topk = (p_s * (student_topk_lp - teacher_topk_lp)).sum(dim=-1)
        p_s_tail = (1.0 - p_s_mass).clamp(min=eps)
        p_t_tail = (1.0 - p_t_mass).clamp(min=eps)
        kl_tail = p_s_tail * (p_s_tail.log() - p_t_tail.log())
        return kl_topk + kl_tail, p_s_mass, p_t_mass

    def _compute_topk_notail_reverse_kl(
        student_topk_lp: torch.Tensor,
        teacher_topk_lp: torch.Tensor,
        stop_grad_weight: bool = False,
    ) -> torch.Tensor:
        # top-k reverse KL without tail-mass correction: KL = Σ P_s * (log P_s - log P_t)
        # When stop_grad_weight=True, log P_s is detached so gradients flow only through P_s weight.
        p_s = student_topk_lp.exp()
        lp_s = student_topk_lp.detach() if stop_grad_weight else student_topk_lp
        return (p_s * (lp_s - teacher_topk_lp)).sum(dim=-1)

    def _compute_topk_intersect_sg_norm_reverse_kl(
        student_topk_lp: torch.Tensor,
        teacher_topk_lp: torch.Tensor,
        intersect_mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        # Reverse KL restricted to the intersection I of student/teacher top-k sets,
        # with P_s and P_t renormalised on I to be valid distributions, then computed
        # in the _sg style (stop-grad on log P̃_s so gradients flow only through the
        # P̃_s weight).  Positions with empty intersection contribute zero.
        mask = intersect_mask.to(dtype=student_topk_lp.dtype)
        p_s = student_topk_lp.exp() * mask
        p_t = teacher_topk_lp.exp() * mask
        Z_s = p_s.sum(dim=-1, keepdim=True)
        Z_t = p_t.sum(dim=-1, keepdim=True)
        log_Z_s = Z_s.clamp(min=eps).log()
        log_Z_t = Z_t.clamp(min=eps).log()
        p_s_norm = p_s / Z_s.clamp(min=eps)
        lp_s_norm = (student_topk_lp - log_Z_s).detach()
        lp_t_norm = teacher_topk_lp - log_Z_t
        kl = (p_s_norm * (lp_s_norm - lp_t_norm) * mask).sum(dim=-1)
        valid = (mask.sum(dim=-1) > 0).to(dtype=kl.dtype)
        return kl * valid

    _TOPK_KL_MODES = frozenset(
        [
            "full_vocab_topk_reverse_kl",
            "topk_reverse_kl_notail",
            "topk_reverse_kl_notail_sg",
            "topk_reverse_kl_intersect_sg_norm",
        ]
    )

    reverse_kls = []
    if kl_mode in _TOPK_KL_MODES:
        student_topk = rollout_data.get("opd_topk_student_log_probs")
        teacher_topk = rollout_data.get("opd_topk_teacher_log_probs")
        if student_topk is None or teacher_topk is None:
            raise ValueError(
                f"OPD {kl_mode} requires opd_topk_student_log_probs and "
                "opd_topk_teacher_log_probs in rollout_data."
            )
        student_topk = [t.to(device=device) for t in student_topk]
        teacher_topk = [t.to(device=device) for t in teacher_topk]
        intersect_masks = None
        if kl_mode == "topk_reverse_kl_intersect_sg_norm":
            intersect_masks = rollout_data.get("opd_topk_intersect_mask")
            if intersect_masks is None:
                raise ValueError(
                    f"OPD {kl_mode} requires opd_topk_intersect_mask in rollout_data."
                )
            intersect_masks = [m.to(device=device) for m in intersect_masks]

        topk_student_mass = []
        topk_teacher_mass = []
        topk_tail_kl = []
        for i, adv in enumerate(advantages):
            if opd_distill_sample_mask[i] <= 0:
                reverse_kl = torch.zeros_like(adv)
                advantages[i] = adv - args.opd_kl_coef * reverse_kl
                reverse_kls.append(reverse_kl)
                topk_student_mass.append(torch.zeros_like(reverse_kl))
                topk_teacher_mass.append(torch.zeros_like(reverse_kl))
                topk_tail_kl.append(torch.zeros_like(reverse_kl))
                continue
            if student_topk[i].dim() != 2 or teacher_topk[i].dim() != 2:
                raise ValueError(
                    "OPD top-k tensors must be rank-2 [response_len, k]. "
                    f"Got {student_topk[i].shape} and {teacher_topk[i].shape}."
                )
            if student_topk[i].shape != teacher_topk[i].shape:
                raise ValueError(
                    "Student/teacher top-k tensor shape mismatch: "
                    f"{student_topk[i].shape} vs {teacher_topk[i].shape}."
                )
            if kl_mode == "full_vocab_topk_reverse_kl":
                reverse_kl, ps_mass, pt_mass = _compute_topk_tail_reverse_kl(student_topk[i], teacher_topk[i])
                topk_student_mass.append(ps_mass)
                topk_teacher_mass.append(pt_mass)
                topk_tail_kl.append(
                    reverse_kl - (student_topk[i].exp() * (student_topk[i] - teacher_topk[i])).sum(dim=-1)
                )
            elif kl_mode == "topk_reverse_kl_intersect_sg_norm":
                reverse_kl = _compute_topk_intersect_sg_norm_reverse_kl(
                    student_topk[i], teacher_topk[i], intersect_masks[i]
                )
                p_s = student_topk[i].exp()
                p_s_mass = p_s.sum(dim=-1).clamp(min=0.0, max=1.0)
                p_t_mass = teacher_topk[i].exp().sum(dim=-1).clamp(min=0.0, max=1.0)
                topk_student_mass.append(p_s_mass)
                topk_teacher_mass.append(p_t_mass)
                topk_tail_kl.append(torch.zeros_like(reverse_kl))
            else:
                stop_grad = kl_mode == "topk_reverse_kl_notail_sg"
                reverse_kl = _compute_topk_notail_reverse_kl(student_topk[i], teacher_topk[i], stop_grad)
                p_s = student_topk[i].exp()
                p_s_mass = p_s.sum(dim=-1).clamp(min=0.0, max=1.0)
                p_t_mass = teacher_topk[i].exp().sum(dim=-1).clamp(min=0.0, max=1.0)
                topk_student_mass.append(p_s_mass)
                topk_teacher_mass.append(p_t_mass)
                topk_tail_kl.append(torch.zeros_like(reverse_kl))
            advantages[i] = adv - args.opd_kl_coef * reverse_kl
            reverse_kls.append(reverse_kl)

        rollout_data["opd_topk_student_mass"] = topk_student_mass
        rollout_data["opd_topk_teacher_mass"] = topk_teacher_mass
        rollout_data["opd_topk_tail_kl"] = topk_tail_kl
    else:
        teacher_log_probs = rollout_data.get("teacher_log_probs")
        if teacher_log_probs is None:
            raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")
        teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]
        teacher_logprob_mask = rollout_data.get("teacher_logprob_mask")
        if teacher_logprob_mask is not None:
            teacher_logprob_mask = [m.to(device=device, dtype=torch.float32) for m in teacher_logprob_mask]
        for i, adv in enumerate(advantages):
            reverse_kl = student_log_probs[i] - teacher_log_probs[i]
            if teacher_logprob_mask is not None:
                reverse_kl = reverse_kl * teacher_logprob_mask[i]
            if opd_distill_sample_mask[i] <= 0:
                reverse_kl = torch.zeros_like(reverse_kl)
            advantages[i] = adv - args.opd_kl_coef * reverse_kl
            reverse_kls.append(reverse_kl)

    # Store reverse KL for logging
    rollout_data["opd_reverse_kl"] = reverse_kls


def _maybe_apply_opsd_grpo_r2_reward_bonus(args: Namespace, rollout_data: RolloutBatch) -> None:
    alpha = float(getattr(args, "opsd_grpo_r2_alpha", 0.0))
    sign_mode = str(getattr(args, "opsd_grpo_r2_sign_mode", "flip_on_reward0"))
    if alpha <= 0.0:
        return

    raw_rewards_full = rollout_data.get("raw_reward")
    local_batch_indices = rollout_data.get("batch_sample_indices")
    local_gap_means = rollout_data.get("opsd_reward_gap_mean")
    if raw_rewards_full is None:
        raise ValueError("--opsd-grpo-r2-alpha requires raw_reward in rollout_data, but it is missing.")
    if local_batch_indices is None:
        raise ValueError("--opsd-grpo-r2-alpha requires batch_sample_indices in rollout_data, but they are missing.")
    if local_gap_means is None:
        raise ValueError(
            "--opsd-grpo-r2-alpha requires opsd_reward_gap_mean in rollout_data, but it is missing. "
            "Make sure the Megatron actor pre-computed teacher-vs-rollout gap means."
        )

    local_batch_indices = [int(idx) for idx in local_batch_indices]
    local_gap_means = [float(value) for value in local_gap_means]
    if len(local_batch_indices) != len(local_gap_means):
        raise ValueError(
            "--opsd-grpo-r2-alpha local batch-index/gap count mismatch: "
            f"{len(local_batch_indices)} vs {len(local_gap_means)}."
        )

    dp_world_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
    if dp_world_size > 1:
        dp_group = mpu.get_data_parallel_group_gloo(with_context_parallel=True)
        gathered_batch_indices: list[list[int] | None] = [None] * dp_world_size
        gathered_gap_means: list[list[float] | None] = [None] * dp_world_size
        dist.all_gather_object(gathered_batch_indices, local_batch_indices, group=dp_group)
        dist.all_gather_object(gathered_gap_means, local_gap_means, group=dp_group)
    else:
        gathered_batch_indices = [local_batch_indices]
        gathered_gap_means = [local_gap_means]

    local_reward_payload = recompute_local_opsd_grpo_r2_rewards(
        raw_rewards_full=[float(value) for value in raw_rewards_full],
        local_batch_indices=local_batch_indices,
        gathered_batch_indices=[[int(idx) for idx in shard] for shard in gathered_batch_indices],
        gathered_gap_means=[[float(value) for value in shard] for shard in gathered_gap_means],
        alpha=alpha,
        sign_mode=sign_mode,
        n_samples_per_prompt=int(getattr(args, "n_samples_per_prompt", 1)),
    )
    rollout_data.update(local_reward_payload)


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    if mpu.is_pipeline_last_stage():
        _maybe_apply_opsd_grpo_r2_reward_bonus(args, rollout_data)

    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    # Skip for opsd: OPSD computes full-vocabulary JSD in the forward pass, not via log-prob KL on advantages.
    if args.use_opd and getattr(args, "opd_type", None) != "opsd":
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def _compute_opsd_center_and_scale(
    smoothed: torch.Tensor,
    normalization_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(normalization_mode).strip().lower()

    if mode == "standard":
        center = smoothed.mean()
        scale = smoothed.std()
        if not torch.isfinite(scale):
            scale = smoothed.std(unbiased=False)
    elif mode == "robust":
        center = smoothed.median()
        mad = (smoothed - center).abs().median()
        scale = 1.4826 * mad
        if not torch.isfinite(scale) or scale < 1e-8:
            scale = smoothed.std()
            if not torch.isfinite(scale):
                scale = smoothed.std(unbiased=False)
    else:
        raise ValueError(
            "Unsupported --opsd-advantage-normalization: "
            f"{normalization_mode!r}. Expected one of ['standard', 'robust']."
        )

    if not torch.isfinite(scale):
        scale = torch.zeros_like(center)
    return center, scale.clamp(min=1e-8)


def _build_repetition_mask_list(
    args: Namespace,
    batch: RolloutBatch,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None,
) -> list[torch.Tensor]:
    ngram = int(getattr(args, "opd_token_stats_repeat_ngram", 3))
    repetition_masks: list[torch.Tensor] = []
    for i, (tokens_i, total_length, response_length) in enumerate(
        zip(batch["unconcat_tokens"], total_lengths, response_lengths, strict=False)
    ):
        response_tokens_i = tokens_i[-response_length:] if response_length > 0 else tokens_i[0:0]
        repeat_mask_full = build_token_repetition_mask(response_tokens_i, ngram=ngram)
        repeat_mask_local = slice_log_prob_with_cp(
            repeat_mask_full,
            total_length,
            response_length,
            args.qkv_format,
            max_seq_lens[i] if args.qkv_format == "bshd" else None,
        )
        repetition_masks.append(
            torch.tensor(repeat_mask_local, dtype=torch.float32, device=torch.cuda.current_device())
        )
    return repetition_masks




def _maybe_collect_opd_token_stats_metrics(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    *,
    log_probs_list: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None,
    opd_distill_sample_mask: list[float] | None,
) -> dict[str, torch.Tensor]:
    if not (args.use_opd and getattr(args, "opd_token_stats", False)):
        return {}
    if getattr(args, "opd_type", None) not in {"sglang", "opsd"}:
        return {}

    if getattr(args, "opd_type", None) == "opsd":
        teacher_log_probs_list = batch.get("opsd_teacher_sampled_log_probs")
    else:
        teacher_log_probs_list = batch.get("teacher_log_probs")

    topk_overlap_list = batch.get("opd_diag_topk_overlap")
    teacher_rank_list = batch.get("opd_diag_teacher_rank_at_k")
    if teacher_log_probs_list is None:
        raise ValueError("OPD token diagnostics require teacher sampled-token logprobs in batch.")
    if topk_overlap_list is None or teacher_rank_list is None:
        raise ValueError(
            "OPD token diagnostics require opd_diag_topk_overlap and opd_diag_teacher_rank_at_k in batch."
        )

    repetition_mask_list = _build_repetition_mask_list(
        args,
        batch,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )

    teacher_logprob_mask = batch.get("teacher_logprob_mask")
    effective_masks: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    for i, loss_mask_i in enumerate(batch["loss_masks"]):
        effective_mask_i = loss_mask_i.to(device=log_probs_list[i].device, dtype=torch.float32)
        valid_mask_i = effective_mask_i.clone()
        if teacher_logprob_mask is not None:
            valid_mask_i = valid_mask_i * teacher_logprob_mask[i].to(device=valid_mask_i.device, dtype=torch.float32)
        if (
            getattr(args, "opd_type", None) == "sglang"
            and opd_distill_sample_mask is not None
            and opd_distill_sample_mask[i] <= 0
        ):
            valid_mask_i = torch.zeros_like(valid_mask_i)
        effective_masks.append(effective_mask_i)
        valid_masks.append(valid_mask_i)

    metrics = compute_token_stats_metrics(
        repeat_mask=torch.cat(repetition_mask_list, dim=0),
        effective_mask=torch.cat(effective_masks, dim=0),
        valid_mask=torch.cat(valid_masks, dim=0),
        student_log_probs=torch.cat(log_probs_list, dim=0).detach(),
        teacher_log_probs=torch.cat(
            [tensor.to(device=log_probs_list[i].device, dtype=torch.float32) for i, tensor in enumerate(teacher_log_probs_list)],
            dim=0,
        ).detach(),
        teacher_rank_at_k=torch.cat(
            [tensor.to(device=log_probs_list[i].device, dtype=torch.float32) for i, tensor in enumerate(teacher_rank_list)],
            dim=0,
        ).detach(),
        topk_overlap=torch.cat(
            [tensor.to(device=log_probs_list[i].device, dtype=torch.float32) for i, tensor in enumerate(topk_overlap_list)],
            dim=0,
        ).detach(),
    )
    return {f"opd_token_stats_{key}": value.detach() for key, value in metrics.items()}


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    opd_distill_sample_mask = None
    opd_distill_sample_mask_tensor = None
    if args.use_opd and getattr(args, "opd_type", None) == "sglang":
        raw_distill_sample_mask = batch.get("opd_distill_sample_mask")
        if raw_distill_sample_mask is None:
            opd_distill_sample_mask = [1.0] * len(response_lengths)
        else:
            if len(raw_distill_sample_mask) != len(response_lengths):
                raise ValueError(
                    "opd_distill_sample_mask length mismatch: "
                    f"{len(raw_distill_sample_mask)} vs {len(response_lengths)}."
                )
            opd_distill_sample_mask = []
            for value in raw_distill_sample_mask:
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        raise ValueError(
                            "opd_distill_sample_mask tensor entries must be scalar. "
                            f"Got shape={tuple(value.shape)}."
                        )
                    value = float(value.item())
                else:
                    value = float(value)
                opd_distill_sample_mask.append(1.0 if value > 0 else 0.0)
        opd_distill_sample_mask_tensor = torch.tensor(
            opd_distill_sample_mask,
            dtype=torch.float32,
            device=logits.device,
        )

    opd_sft_sample_mask = None
    opd_sft_sample_mask_tensor = None
    if (
        args.use_opd
        and getattr(args, "opd_type", None) == "sglang"
        and getattr(args, "opd_teacher_sft", False)
    ):
        raw_sft_sample_mask = batch.get("opd_sft_sample_mask")
        if raw_sft_sample_mask is None:
            opd_sft_sample_mask = [0.0] * len(response_lengths)
        else:
            if len(raw_sft_sample_mask) != len(response_lengths):
                raise ValueError(
                    "opd_sft_sample_mask length mismatch: "
                    f"{len(raw_sft_sample_mask)} vs {len(response_lengths)}."
                )
            opd_sft_sample_mask = []
            for value in raw_sft_sample_mask:
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        raise ValueError(
                            "opd_sft_sample_mask tensor entries must be scalar. "
                            f"Got shape={tuple(value.shape)}."
                        )
                    value = float(value.item())
                else:
                    value = float(value)
                opd_sft_sample_mask.append(1.0 if value > 0 else 0.0)
        opd_sft_sample_mask_tensor = torch.tensor(
            opd_sft_sample_mask,
            dtype=torch.float32,
            device=logits.device,
        )

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs_list = log_probs

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # ---- OPSD advantage masking & weighting (optional) ----
    # Both features share the same smoothed & normalized signal pipeline.
    # Masking: zero out pg_loss for tokens inconsistent with advantage direction.
    # Weighting: multiply pg_loss by a configurable function of normalized_signal
    # (exp or 2*sigmoid), then clamp to [1-eps, 1+eps].
    opsd_adv_mask_ratio = None
    opsd_adv_weight_mean = None
    _do_masking = getattr(args, "opsd_advantage_masking", False) and "opsd_token_signal" in batch
    _do_weighting = getattr(args, "opsd_advantage_weighting", False) and "opsd_token_signal" in batch
    opsd_normalized_signal_list = []  # collected for wandb logging (pos/neg fraction)
    if _do_masking or _do_weighting:
        opsd_token_signal_list = batch["opsd_token_signal"]
        advantages_list = batch["advantages"]
        window_size = getattr(args, "opsd_signal_window_size", 32)
        normalization_mode = str(getattr(args, "opsd_advantage_normalization", "standard")).lower()
        weighting_eps = getattr(args, "opsd_advantage_weighting_epsilon", 0.2)
        weighting_fn = str(getattr(args, "opsd_advantage_weighting_fn", "sigmoid")).lower()
        weighting_sign_mode = str(
            getattr(args, "opsd_advantage_weighting_sign_mode", "flip_on_negative_advantage")
        ).lower()

        mask_list = []
        weight_list = []
        for signal_i, advantage_i in zip(opsd_token_signal_list, advantages_list):
            signal_i = signal_i.to(device=advantage_i.device, dtype=torch.float32)
            resp_len = signal_i.numel()

            # 1. Centered moving average smoothing (truncated boundary).
            # Even windows are right-biased:
            # left_span=(w-1)//2, right_span=w//2.
            w = max(1, int(window_size))
            if w > 1 and resp_len > 1:
                left_span = (w - 1) // 2
                right_span = w // 2
                idx = torch.arange(resp_len, device=signal_i.device)
                starts = torch.clamp(idx - left_span, min=0)
                ends = torch.clamp(idx + right_span + 1, max=resp_len)  # exclusive
                prefix = torch.cat(
                    [
                        torch.zeros(1, device=signal_i.device, dtype=signal_i.dtype),
                        torch.cumsum(signal_i, dim=0),
                    ],
                    dim=0,
                )
                sums = prefix[ends] - prefix[starts]
                counts = (ends - starts).to(dtype=signal_i.dtype).clamp(min=1.0)
                smoothed = sums / counts
            else:
                smoothed = signal_i

            # 2. Per-response normalization of the smoothed signal.
            center, scale = _compute_opsd_center_and_scale(smoothed, normalization_mode=normalization_mode)
            normalized = (smoothed - center) / scale

            # Collect normalized signal for logging
            opsd_normalized_signal_list.append(normalized.detach())

            adv_mean = advantage_i.mean()

            # 3. Masking: keep tokens consistent with advantage direction
            if _do_masking:
                if adv_mean > 0:
                    token_mask = (normalized >= 0).float()
                elif adv_mean < 0:
                    token_mask = (normalized <= 0).float()
                else:
                    token_mask = torch.ones_like(normalized)
                # Edge case: if all tokens masked, keep all
                if token_mask.sum() == 0:
                    token_mask = torch.ones_like(token_mask)
                mask_list.append(token_mask)

            # 4. Weighting: configurable function over normalized signal, then clamp
            if _do_weighting:
                if weighting_sign_mode == "none":
                    normalized_for_weight = normalized
                elif weighting_sign_mode == "flip_on_negative_advantage":
                    normalized_for_weight = -normalized if adv_mean < 0 else normalized
                else:
                    raise ValueError(
                        "Unsupported --opsd-advantage-weighting-sign-mode: "
                        f"{weighting_sign_mode!r}. Expected one of "
                        "['none', 'flip_on_negative_advantage']."
                    )

                normalized_for_weight = normalized_for_weight.clamp(min=-60.0, max=60.0)
                if weighting_fn == "exp":
                    token_weight_raw = torch.exp(normalized_for_weight)
                elif weighting_fn == "sigmoid":
                    token_weight_raw = 2.0 * torch.sigmoid(normalized_for_weight)
                else:
                    raise ValueError(
                        "Unsupported --opsd-advantage-weighting-fn: "
                        f"{weighting_fn!r}. Expected one of ['sigmoid', 'exp']."
                    )
                token_weight = token_weight_raw.clamp(min=1.0 - weighting_eps, max=1.0 + weighting_eps)
                weight_list.append(token_weight)

        # Apply masking
        if _do_masking:
            opsd_adv_mask = torch.cat(mask_list, dim=0).detach()
            opsd_adv_mask_ratio = 1.0 - opsd_adv_mask.mean()
            pg_loss = pg_loss * opsd_adv_mask

        # Apply weighting (stop gradient: weights only re-scale, no grad through weighting function)
        if _do_weighting:
            opsd_adv_weight = torch.cat(weight_list, dim=0).detach()
            opsd_adv_weight_mean = opsd_adv_weight.mean().detach()
            pg_loss = pg_loss * opsd_adv_weight

    # Apply off-policy correction using importance sampling if enabled
    # opsd_is_weights_flat: IS weights flattened to [total_tokens], used to re-weight OPSD JSD loss below.
    opsd_is_weights_flat = None
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function

        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # Capture IS weights for OPSD JSD re-weighting.
        # When use_tis is False (only get_mismatch_metrics), opsd_is_weights_flat stays None.
        if args.use_tis and args.use_opd and getattr(args, "opd_type", None) == "opsd":
            # Compute IS weights directly from train vs rollout log-probs.
            # This is safe regardless of pg_loss sign (avoids inferring weights from pg_loss ratios).
            # Boundaries: prefer MIS params (tis_lower_bound / tis_upper_bound) when set by custom
            # config, fall back to vanilla TIS params (tis_clip_low / tis_clip).
            # RS veto effect is separately captured via modified_response_masks → sum_of_sample_mean.
            _w_lo = getattr(args, "tis_lower_bound", None) or getattr(args, "tis_clip_low", 0.0)
            _w_hi = getattr(args, "tis_upper_bound", None) or getattr(args, "tis_clip", 2.0)
            _train_lp = torch.cat(batch["log_probs"], dim=0)
            _rollout_lp = torch.cat(batch["rollout_log_probs"], dim=0)
            _raw_is = torch.exp(
                torch.clamp(_train_lp - _rollout_lp, min=-20.0, max=20.0)
            ).detach()
            opsd_is_weights_flat = torch.clamp(_raw_is, min=_w_lo, max=_w_hi)
            del _train_lp, _rollout_lp, _raw_is

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    # OPSD: optionally zero out pg_loss in pure mode (JSD-only training)
    if getattr(args, "opsd_pure_mode", False) and args.use_opd and args.opd_type == "opsd":
        loss = torch.zeros_like(pg_loss) - args.entropy_coef * entropy_loss
    else:
        loss = pg_loss - args.entropy_coef * entropy_loss

    # OPSD: add JSD loss
    # When TIS is enabled, we also re-weight the per-token JSD contributions by the IS weights
    # so that tokens with large training-inference mismatch contribute less to the JSD gradient.
    # RS rejection (modified_response_masks) is already baked into sum_of_sample_mean above.
    if args.use_opd and args.opd_type == "opsd" and args.opsd_jsd_coef > 0 and "opsd_jsd_values" in batch:
        opsd_jsd_list = batch["opsd_jsd_values"]

        # -- Optional: length-normalized reduction --
        # Weight each sample by (L_i / L_mean) so longer responses contribute
        # proportionally more KL loss, creating an implicit length penalty.
        if getattr(args, "opsd_kl_length_normalize", False):
            loss_masks_list = batch["loss_masks"]
            token_counts = [mask.sum().item() for mask in loss_masks_list]
            total_tokens = sum(token_counts)
            n_samples = len(opsd_jsd_list)
            if total_tokens > 0:
                length_weights = [tc * n_samples / total_tokens for tc in token_counts]
            else:
                length_weights = [1.0] * n_samples
            opsd_jsd_list = [jsd_i * w for jsd_i, w in zip(opsd_jsd_list, length_weights)]

        # -- Optional: position decay --
        # Suppress KL gradient at late token positions to break the length-explosion
        # feedback loop in forward KL training.  Token at absolute position t receives
        # weight exp(-alpha * t / max_len).  This prevents the teacher's low-EOS
        # probability at late positions from being propagated to the student, allowing
        # the student to assign higher EOS probability there and produce shorter rollouts.
        _position_decay = getattr(args, "opsd_kl_position_decay", 0.0)
        if _position_decay > 0.0:
            _max_len = float(getattr(args, "rollout_max_response_len", 8192) or 8192)
            decayed = []
            for jsd_i in opsd_jsd_list:
                L = jsd_i.size(0)
                if L > 0:
                    t = torch.arange(L, dtype=torch.float32, device=jsd_i.device)
                    decay_weights = torch.exp(-_position_decay * t / _max_len)
                    decayed.append(jsd_i * decay_weights)
                else:
                    decayed.append(jsd_i)
            opsd_jsd_list = decayed

        opsd_jsd = torch.cat(opsd_jsd_list, dim=0)
        if opsd_is_weights_flat is not None:
            # IS-weight the per-token JSD contributions (same weights used for pg_loss).
            opsd_jsd = opsd_jsd * opsd_is_weights_flat
        opsd_jsd_loss = sum_of_sample_mean(opsd_jsd)
        loss = loss + args.opsd_jsd_coef * opsd_jsd_loss

    # OPD-SGLang: optional explicit distillation loss (coef * loss), in addition to
    # the advantage-side OPD term. This makes OPD support an explicit additive form
    # similar to OPSD's coef * distillation_loss.
    opd_explicit_loss = None
    if (
        args.use_opd
        and getattr(args, "opd_type", None) == "sglang"
        and getattr(args, "opd_explicit_loss_coef", 0.0) > 0
    ):
        kl_mode = getattr(args, "opd_kl_mode", "token_reverse_kl")
        if kl_mode == "token_reverse_kl":
            if "teacher_log_probs" not in batch:
                raise ValueError(
                    "OPD explicit token_reverse_kl requires teacher_log_probs in batch, but it is missing."
                )
            teacher_log_probs = batch["teacher_log_probs"]
            if len(teacher_log_probs) != len(log_probs_list):
                raise ValueError(
                    "OPD explicit token_reverse_kl teacher/student sample count mismatch: "
                    f"{len(teacher_log_probs)} vs {len(log_probs_list)}."
                )

            teacher_logprob_mask = batch.get("teacher_logprob_mask")
            if teacher_logprob_mask is not None and len(teacher_logprob_mask) != len(log_probs_list):
                raise ValueError(
                    "OPD explicit token_reverse_kl teacher mask sample count mismatch: "
                    f"{len(teacher_logprob_mask)} vs {len(log_probs_list)}."
                )

            opd_explicit_values_list = []
            combined_masks = []
            for i, (student_lp_i, teacher_lp_i, loss_mask_i) in enumerate(
                zip(log_probs_list, teacher_log_probs, batch["loss_masks"], strict=False)
            ):
                teacher_lp_i = teacher_lp_i.to(device=student_lp_i.device, dtype=student_lp_i.dtype)
                explicit_i = student_lp_i - teacher_lp_i

                if teacher_logprob_mask is not None:
                    teacher_mask_i_float = teacher_logprob_mask[i].to(device=student_lp_i.device, dtype=student_lp_i.dtype)
                    teacher_mask_i_int = teacher_logprob_mask[i].to(device=student_lp_i.device)
                    explicit_i = explicit_i * teacher_mask_i_float
                    combined_mask_i = (loss_mask_i.to(device=student_lp_i.device) * teacher_mask_i_int).int()
                else:
                    combined_mask_i = loss_mask_i.to(device=student_lp_i.device).int()

                if opd_distill_sample_mask is not None and opd_distill_sample_mask[i] <= 0:
                    explicit_i = torch.zeros_like(explicit_i)
                    combined_mask_i = torch.zeros_like(combined_mask_i)

                opd_explicit_values_list.append(explicit_i)
                combined_masks.append(combined_mask_i)

            opd_explicit_values = torch.cat(opd_explicit_values_list, dim=0)
            opd_sum_of_sample_mean = get_sum_of_sample_mean(
                total_lengths,
                response_lengths,
                combined_masks,
                args.calculate_per_token_loss,
                args.qkv_format,
                max_seq_lens,
            )
            opd_explicit_loss = opd_sum_of_sample_mean(opd_explicit_values)
        elif kl_mode in (
            "full_vocab_topk_reverse_kl",
            "topk_reverse_kl_notail",
            "topk_reverse_kl_notail_sg",
            "topk_reverse_kl_intersect_sg_norm",
        ):
            if "opd_topk_token_ids" not in batch or "opd_topk_teacher_log_probs" not in batch:
                raise ValueError(
                    f"OPD explicit {kl_mode} requires opd_topk_token_ids and "
                    "opd_topk_teacher_log_probs in batch."
                )
            if kl_mode == "topk_reverse_kl_intersect_sg_norm" and "opd_topk_intersect_mask" not in batch:
                raise ValueError(
                    f"OPD explicit {kl_mode} requires opd_topk_intersect_mask in batch."
                )

            tp_group = mpu.get_tensor_model_parallel_group()
            eps = 1e-8
            per_sample_kl = []
            response_iter = get_responses(
                logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                max_seq_lens=max_seq_lens,
            )
            for i, (student_logits_i, _) in enumerate(response_iter):
                if opd_distill_sample_mask is not None and opd_distill_sample_mask[i] <= 0:
                    per_sample_kl.append(torch.zeros((student_logits_i.size(0),), dtype=torch.float32, device=student_logits_i.device))
                    continue

                topk_ids_i = batch["opd_topk_token_ids"][i]
                teacher_topk_lp_i = batch["opd_topk_teacher_log_probs"][i]

                if topk_ids_i.numel() == 0:
                    per_sample_kl.append(torch.zeros((0,), dtype=torch.float32, device=student_logits_i.device))
                    continue

                if topk_ids_i.dim() != 2 or teacher_topk_lp_i.dim() != 2:
                    raise ValueError(
                        "OPD top-k explicit loss expects rank-2 tensors [response_len, k]. "
                        f"Got ids={topk_ids_i.shape}, teacher_lp={teacher_topk_lp_i.shape}."
                    )
                if topk_ids_i.shape != teacher_topk_lp_i.shape:
                    raise ValueError(
                        "OPD top-k explicit loss shape mismatch between ids and teacher log-probs: "
                        f"{topk_ids_i.shape} vs {teacher_topk_lp_i.shape}."
                    )
                if student_logits_i.size(0) != topk_ids_i.size(0):
                    raise ValueError(
                        "OPD top-k explicit loss response length mismatch: "
                        f"student_logits={student_logits_i.size(0)}, topk={topk_ids_i.size(0)}."
                    )

                k = topk_ids_i.size(1)
                current_student_lp_cols = []
                for j in range(k):
                    lp_col = compute_log_probs(
                        student_logits_i.float().clone(),
                        topk_ids_i[:, j].long(),
                        tp_group,
                    ).squeeze(-1)
                    current_student_lp_cols.append(lp_col)
                current_student_lp = torch.stack(current_student_lp_cols, dim=-1)

                teacher_topk_lp_i = teacher_topk_lp_i.float().to(device=current_student_lp.device)
                p_s = current_student_lp.exp()

                if kl_mode == "full_vocab_topk_reverse_kl":
                    p_t = teacher_topk_lp_i.exp()
                    kl_topk = (p_s * (current_student_lp - teacher_topk_lp_i)).sum(dim=-1)
                    p_s_tail = (1.0 - p_s.sum(dim=-1)).clamp(min=eps)
                    p_t_tail = (1.0 - p_t.sum(dim=-1)).clamp(min=eps)
                    kl_tail = p_s_tail * (p_s_tail.log() - p_t_tail.log())
                    per_sample_kl.append(kl_topk + kl_tail)
                elif kl_mode == "topk_reverse_kl_intersect_sg_norm":
                    intersect_mask_i = batch["opd_topk_intersect_mask"][i]
                    if intersect_mask_i.shape != teacher_topk_lp_i.shape:
                        raise ValueError(
                            "OPD intersect mask shape mismatch vs teacher log-probs: "
                            f"{intersect_mask_i.shape} vs {teacher_topk_lp_i.shape}."
                        )
                    mask = intersect_mask_i.to(dtype=current_student_lp.dtype, device=current_student_lp.device)
                    p_s_masked = p_s * mask
                    p_t_masked = teacher_topk_lp_i.exp() * mask
                    Z_s = p_s_masked.sum(dim=-1, keepdim=True)
                    Z_t = p_t_masked.sum(dim=-1, keepdim=True)
                    log_Z_s = Z_s.clamp(min=eps).log()
                    log_Z_t = Z_t.clamp(min=eps).log()
                    p_s_norm = p_s_masked / Z_s.clamp(min=eps)
                    lp_s_norm = (current_student_lp - log_Z_s).detach()
                    lp_t_norm = teacher_topk_lp_i - log_Z_t
                    kl_i = (p_s_norm * (lp_s_norm - lp_t_norm) * mask).sum(dim=-1)
                    valid = (mask.sum(dim=-1) > 0).to(dtype=kl_i.dtype)
                    per_sample_kl.append(kl_i * valid)
                else:
                    # topk_reverse_kl_notail and topk_reverse_kl_notail_sg: no tail correction
                    # For _sg: stopgrad on log P_s so gradients flow only through P_s weight.
                    lp_s = current_student_lp.detach() if kl_mode == "topk_reverse_kl_notail_sg" else current_student_lp
                    per_sample_kl.append((p_s * (lp_s - teacher_topk_lp_i)).sum(dim=-1))

            opd_explicit_values = torch.cat(per_sample_kl, dim=0)
            opd_explicit_loss = sum_of_sample_mean(opd_explicit_values)
        else:
            raise ValueError(f"Unsupported --opd-kl-mode for explicit OPD loss: {kl_mode}")

        loss = loss + args.opd_explicit_loss_coef * opd_explicit_loss

    # OPSD: position-weighted EOS encouragement loss
    # loss_t = -(t / max_len) * log p_S(EOS | context_t), pre-computed in the forward pass.
    # Encourages the student to assign higher EOS probability at later positions,
    # directly counteracting length explosion without modifying the JSD objective.
    if args.use_opd and args.opd_type == "opsd" and getattr(args, "opsd_eos_loss_coef", 0.0) > 0 and "opsd_eos_values" in batch:
        opsd_eos = torch.cat(batch["opsd_eos_values"], dim=0)
        opsd_eos_loss = sum_of_sample_mean(opsd_eos)
        loss = loss + args.opsd_eos_loss_coef * opsd_eos_loss

    opd_teacher_sft_loss = None
    if (
        args.use_opd
        and getattr(args, "opd_type", None) == "sglang"
        and getattr(args, "opd_teacher_sft", False)
    ):
        sft_values_list = []
        sft_masks = []
        for i, (student_lp_i, loss_mask_i) in enumerate(zip(log_probs_list, batch["loss_masks"], strict=False)):
            sample_is_sft = opd_sft_sample_mask is not None and opd_sft_sample_mask[i] > 0
            if sample_is_sft:
                sft_values_list.append(-student_lp_i)
                sft_masks.append(loss_mask_i.to(device=student_lp_i.device).int())
            else:
                sft_values_list.append(torch.zeros_like(student_lp_i))
                sft_masks.append(torch.zeros_like(loss_mask_i, device=student_lp_i.device).int())

        sft_values = torch.cat(sft_values_list, dim=0)
        sft_sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            sft_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )
        opd_teacher_sft_loss = sft_sum_of_sample_mean(sft_values)
        loss = loss + args.opd_teacher_sft_loss_coef * opd_teacher_sft_loss

    # kl_biased_ppo: per-token loss values are stored in opsd_jsd_values and consumed
    # by the regular OPSD JSD path above (lines ~805-847).  No special treatment needed.

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }
    reported_loss.update(
        _maybe_collect_opd_token_stats_metrics(
            args,
            batch,
            logits,
            log_probs_list=log_probs_list,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            opd_distill_sample_mask=opd_distill_sample_mask,
        )
    )

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # OPSD advantage masking metrics
    if opsd_adv_mask_ratio is not None:
        reported_loss["opsd_adv_mask_ratio"] = (
            opsd_adv_mask_ratio.clone().detach()
            if isinstance(opsd_adv_mask_ratio, torch.Tensor)
            else torch.tensor(opsd_adv_mask_ratio, device=logits.device)
        )
    # OPSD advantage weighting metrics
    if opsd_adv_weight_mean is not None:
        reported_loss["opsd_adv_weight_mean"] = (
            opsd_adv_weight_mean.clone().detach()
            if isinstance(opsd_adv_weight_mean, torch.Tensor)
            else torch.tensor(opsd_adv_weight_mean, device=logits.device)
        )
    # OPSD normalized signal positive/negative fraction (from smoothed & z-scored signal)
    if opsd_normalized_signal_list:
        _norm_sig_cat = torch.cat(opsd_normalized_signal_list, dim=0)
        _total = max(_norm_sig_cat.numel(), 1)
        reported_loss["opsd_norm_signal_pos_frac"] = torch.tensor(
            (_norm_sig_cat > 0).float().sum().item() / _total, device=logits.device
        )
        reported_loss["opsd_norm_signal_neg_frac"] = torch.tensor(
            (_norm_sig_cat < 0).float().sum().item() / _total, device=logits.device
        )
    if "opsd_token_signal" in batch:
        _signal_cat = torch.cat(batch["opsd_token_signal"], dim=0)
        reported_loss["opsd_signal_mean"] = sum_of_sample_mean(_signal_cat).clone().detach()
        reported_loss["opsd_signal_pos_frac"] = sum_of_sample_mean(
            (_signal_cat > 0).float()
        ).clone().detach()

    if opd_distill_sample_mask_tensor is not None:
        if opd_distill_sample_mask_tensor.numel() > 0:
            active_ratio = opd_distill_sample_mask_tensor.mean()
        else:
            active_ratio = torch.tensor(1.0, dtype=torch.float32, device=logits.device)
        reported_loss["opd_distill_active_ratio"] = active_ratio.clone().detach()
        reported_loss["opd_distill_skip_ratio"] = (1.0 - active_ratio).clone().detach()
    if opd_sft_sample_mask_tensor is not None:
        if opd_sft_sample_mask_tensor.numel() > 0:
            sft_active_ratio = opd_sft_sample_mask_tensor.mean()
        else:
            sft_active_ratio = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
        reported_loss["opd_teacher_sft_active_ratio"] = sft_active_ratio.clone().detach()

    # Add OPD metrics if available
    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        if "teacher_logprob_mask" in batch:
            combined_masks = [
                (loss_mask.to(device=opd_reverse_kl.device) * teacher_mask.to(device=opd_reverse_kl.device)).int()
                for loss_mask, teacher_mask in zip(batch["loss_masks"], batch["teacher_logprob_mask"], strict=False)
            ]
            opd_sum_of_sample_mean = get_sum_of_sample_mean(
                batch["total_lengths"],
                batch["response_lengths"],
                combined_masks,
                args.calculate_per_token_loss,
                args.qkv_format,
                batch.get("max_seq_lens", None),
            )
            reported_loss["opd_reverse_kl"] = opd_sum_of_sample_mean(opd_reverse_kl).clone().detach()
        else:
            reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()
    if opd_explicit_loss is not None:
        reported_loss["opd_explicit_loss"] = opd_explicit_loss.clone().detach()
    if opd_teacher_sft_loss is not None:
        reported_loss["opd_teacher_sft_loss"] = opd_teacher_sft_loss.clone().detach()

    # Add OPSD JSD / KL-biased-PPO metrics if available
    if "opsd_jsd_values" in batch:
        opsd_jsd = torch.cat(batch["opsd_jsd_values"], dim=0)
        if getattr(args, "opsd_loss_type", "jsd") == "kl_biased_ppo":
            reported_loss["kl_biased_ppo_loss"] = sum_of_sample_mean(opsd_jsd).clone().detach()
        else:
            reported_loss["opsd_jsd"] = sum_of_sample_mean(opsd_jsd).clone().detach()

    # Add Wiener gate weight metrics if available (wiener_kl loss type only)
    if "opsd_wiener_weights" in batch:
        wiener_w = torch.cat(batch["opsd_wiener_weights"], dim=0)
        reported_loss["opsd_wiener_weight"] = sum_of_sample_mean(wiener_w).clone().detach()

    # Add EOS loss metric if available
    if "opsd_eos_values" in batch:
        opsd_eos = torch.cat(batch["opsd_eos_values"], dim=0)
        reported_loss["opsd_eos_loss"] = sum_of_sample_mean(opsd_eos).clone().detach()

    # Fraction of tokens where entropy gate fired (kl_biased_ppo only)
    if "opsd_gate_values" in batch:
        gate_vals = torch.cat(batch["opsd_gate_values"], dim=0)
        reported_loss["kl_biased_ppo_gate_frac"] = sum_of_sample_mean(gate_vals).clone().detach()

    # Fraction of tokens clipped by opsd_token_clip
    if "opsd_clip_fracs" in batch:
        clip_vals = torch.cat(batch["opsd_clip_fracs"], dim=0)
        reported_loss["opsd_token_clip_frac"] = sum_of_sample_mean(clip_vals).clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        loss = (
            loss * num_microbatches / global_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    # These metrics are already normalized scalars inside policy_loss_function.
    # Multiply them back by the step normalizer so the outer train-step reducer
    # does not divide them by sample count a second time.
    pre_averaged_metric_keys = {
        "opsd_adv_mask_ratio",
        "opsd_adv_weight_mean",
        "opsd_norm_signal_pos_frac",
        "opsd_norm_signal_neg_frac",
        "opd_distill_active_ratio",
        "opd_distill_skip_ratio",
        "opd_teacher_sft_active_ratio",
    }

    step_normalizer = (
        num_tokens.to(device=logits.device, dtype=torch.float32)
        if args.calculate_per_token_loss
        else torch.tensor(float(num_samples), device=logits.device)
    )

    packed_log_values = []
    for key, value in log.items():
        value_tensor = (
            value.to(device=logits.device, dtype=torch.float32)
            if isinstance(value, torch.Tensor)
            else torch.tensor(float(value), device=logits.device)
        )
        if key in pre_averaged_metric_keys:
            value_tensor = value_tensor * step_normalizer
        packed_log_values.append(value_tensor)

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            "values": torch.stack([step_normalizer] + packed_log_values),
        },
    )
