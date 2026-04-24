import asyncio
import copy
import inspect
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)


def _truncate_log_text(value: Any, limit: int = 600) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _summarize_reward_for_log(reward: Any) -> Any:
    if not isinstance(reward, dict):
        return reward

    summary: dict[str, Any] = {}
    for key in ("accuracy", "accuracy_strict", "accuracy_relaxed", "teacher_input_len", "teacher_response_start"):
        if key in reward:
            summary[key] = reward[key]

    teacher_topk = reward.get("teacher_topk_log_probs")
    if teacher_topk is not None:
        summary["teacher_topk_log_probs_len"] = len(teacher_topk)

    teacher_topk_ids = reward.get("teacher_topk_token_ids")
    if teacher_topk_ids is not None:
        summary["teacher_topk_token_ids_len"] = len(teacher_topk_ids)

    teacher_output = reward.get("teacher_output")
    if isinstance(teacher_output, dict):
        meta_info = teacher_output.get("meta_info", {}) if isinstance(teacher_output.get("meta_info"), dict) else {}
        teacher_output_summary: dict[str, Any] = {
            "text_len": len(teacher_output.get("text", "") or ""),
            "output_ids_len": len(teacher_output.get("output_ids", []) or []),
            "meta_info_keys": sorted(meta_info.keys())[:20],
        }
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "weight_version",
        ):
            if key in meta_info:
                teacher_output_summary[key] = meta_info[key]
        for key in (
            "input_token_logprobs",
            "output_token_logprobs",
            "token_ids_logprob",
            "input_top_logprobs",
            "input_token_top_logprobs",
        ):
            value = meta_info.get(key)
            if isinstance(value, list):
                teacher_output_summary[f"{key}_len"] = len(value)
        summary["teacher_output"] = teacher_output_summary

    return summary or sorted(reward.keys())


def _summarize_sample_for_log(sample: Sample) -> str:
    prompt_and_response = _truncate_log_text(str(sample.prompt) + sample.response, limit=800)
    label = _truncate_log_text(sample.label or "", limit=120)
    reward = _summarize_reward_for_log(sample.reward)
    return f"prompt+response={prompt_and_response}, label={label}, reward={reward}"


def _should_use_custom_rm(args: Namespace, sample: Sample, evaluation: bool) -> bool:
    # During eval, allow dataset rm_type to bypass custom RM (e.g., OPD teacher calls).
    if not evaluation or args.custom_rm_path is None:
        return True
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    return not bool(rm_type)


def _extract_scalar_logprob(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Empty logprob entry.")
        return _extract_scalar_logprob(value[0])
    if isinstance(value, dict):
        if "logprob" in value:
            return _extract_scalar_logprob(value["logprob"])
        if "value" in value:
            return _extract_scalar_logprob(value["value"])
    raise ValueError(f"Unsupported logprob value format: {type(value)}")


def _parse_top_logprobs_entry(raw_entry: Any, topk: int) -> tuple[list[int], list[float]]:
    if raw_entry is None:
        raise ValueError("top_logprobs entry is None.")

    pairs: list[tuple[int, float]] = []
    if isinstance(raw_entry, dict):
        for k, v in raw_entry.items():
            token_id = int(k)
            logp = _extract_scalar_logprob(v)
            pairs.append((token_id, logp))
    elif isinstance(raw_entry, (list, tuple)):
        for item in raw_entry:
            if isinstance(item, dict):
                token_id = item.get("token_id", item.get("id", item.get("token")))
                if token_id is None:
                    continue
                pairs.append((int(token_id), _extract_scalar_logprob(item)))
            elif isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    # Common forms: [logprob, token_id, ...] or [token_id, logprob]
                    a, b = item[0], item[1]
                    try:
                        # Prefer [logprob, token_id] format used by sglang entries.
                        token_id = int(b)
                        logp = _extract_scalar_logprob(a)
                    except Exception:
                        token_id = int(a)
                        logp = _extract_scalar_logprob(b)
                    pairs.append((token_id, logp))
    else:
        raise ValueError(f"Unsupported top_logprobs format: {type(raw_entry)}")

    if not pairs:
        raise ValueError("No token-logprob pairs found in top_logprobs entry.")

    # Deduplicate by keeping the highest logprob per token id.
    dedup: dict[int, float] = {}
    for tid, lp in pairs:
        if tid not in dedup or lp > dedup[tid]:
            dedup[tid] = lp
    sorted_pairs = sorted(dedup.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_pairs[:topk]
    if len(selected) < topk:
        raise ValueError(f"top_logprobs has only {len(selected)} entries, expected at least {topk}.")
    token_ids = [p[0] for p in selected]
    logps = [p[1] for p in selected]
    return token_ids, logps


def _get_output_top_logprobs_entry(meta_info: dict[str, Any], output_item: Any, position: int) -> Any:
    if isinstance(output_item, dict):
        for key in ("top_logprobs", "top_logprob", "logprobs"):
            if output_item.get(key) is not None:
                return output_item[key]
    elif isinstance(output_item, (list, tuple)) and len(output_item) >= 3 and output_item[2] is not None:
        return output_item[2]

    if not isinstance(meta_info, dict):
        return None
    for key in ("output_top_logprobs", "output_token_top_logprobs", "top_logprobs"):
        sidecar = meta_info.get(key)
        if isinstance(sidecar, (list, tuple)) and position < len(sidecar) and sidecar[position] is not None:
            return sidecar[position]
    return None


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if state.processor:
        processor_kwargs = build_processor_kwargs(sample.multimodal_inputs)
        processor_output = state.processor(text=sample.prompt, **processor_kwargs)
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    use_opd_sglang = args.use_opd and getattr(args, "opd_type", None) == "sglang"
    use_topk_kl = use_opd_sglang and getattr(args, "opd_kl_mode", "token_reverse_kl") in (
        "full_vocab_topk_reverse_kl",
        "topk_reverse_kl_notail",
        "topk_reverse_kl_notail_sg",
        "topk_reverse_kl_intersect_sg_norm",
    )
    diag_enabled = use_opd_sglang and getattr(args, "opd_token_stats", False)
    requested_topk = []
    if use_topk_kl:
        requested_topk.append(int(getattr(args, "opd_topk", 50)))
    if diag_enabled:
        requested_topk.append(int(getattr(args, "opd_token_stats_topk", 50)))
    if requested_topk:
        payload["top_logprobs_num"] = max(requested_topk)

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    if sample.multimodal_inputs and sample.multimodal_inputs["images"]:
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing if router uses consistent_hashing policy
    headers = None
    if args.sglang_router_policy == "consistent_hashing" and sample.session_id:
        headers = {"X-SMG-Routing-Key": sample.session_id}

    output = await post(url, payload, headers=headers)

    if args.use_slime_router and "RadixTreeMiddleware" in args.slime_router_middleware_paths:
        from slime.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        sample = await postprocess_sample_with_radix_tree(args, sample, output)
    else:
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens, new_response_log_probs = [], []

        if use_topk_kl or diag_enabled:
            if "output_token_logprobs" not in output["meta_info"]:
                raise ValueError(
                    "OPD token diagnostics require `meta_info.output_token_logprobs`, but it is missing."
                )
            topk_for_kl = int(getattr(args, "opd_topk", 50))
            topk_for_diag = int(getattr(args, "opd_token_stats_topk", 50))
            if use_topk_kl and sample.opd_topk_token_ids is None:
                sample.opd_topk_token_ids = []
            if use_topk_kl and sample.opd_topk_student_log_probs is None:
                sample.opd_topk_student_log_probs = []
            if diag_enabled and sample.opd_diag_student_topk_token_ids is None:
                sample.opd_diag_student_topk_token_ids = []
            for pos, item in enumerate(output["meta_info"]["output_token_logprobs"]):
                max_topk = max(payload["top_logprobs_num"], 1)
                top_logprobs_entry = _get_output_top_logprobs_entry(output["meta_info"], item, pos)
                if top_logprobs_entry is None:
                    available_sidecars = [
                        key
                        for key in ("output_top_logprobs", "output_token_top_logprobs", "top_logprobs")
                        if output["meta_info"].get(key) is not None
                    ]
                    raise ValueError(
                        "OPD token diagnostics require top_logprobs for each generated token, "
                        f"but none were found at position {pos}. available_sidecars={available_sidecars}"
                    )
                top_token_ids, top_logps = _parse_top_logprobs_entry(top_logprobs_entry, max_topk)
                if use_topk_kl:
                    sample.opd_topk_token_ids.append(top_token_ids[:topk_for_kl])
                    sample.opd_topk_student_log_probs.append(top_logps[:topk_for_kl])
                if diag_enabled:
                    sample.opd_diag_student_topk_token_ids.append(top_token_ids[:topk_for_diag])

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        # When partial rollout and masking off policy is enabled, update the loss mask
        if sample.loss_mask is not None:
            assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
            sample.loss_mask += [1] * len(new_response_tokens)

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.update_from_meta_info(args, output["meta_info"])

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        if samples_need_reward:
            if evaluation and args.custom_rm_path is not None:
                # Eval can contain mixed rm_type settings; compute per-sample to choose RM path precisely.
                tasks = [
                    async_rm(
                        args,
                        sample,
                        use_custom_rm=_should_use_custom_rm(args, sample, evaluation=True),
                        evaluation=True,
                    )
                    for sample in samples_need_reward
                ]
                rewards = await asyncio.gather(*tasks)
            else:
                rewards = await batched_async_rm(args, samples_need_reward)
            for sample, reward in zip(samples_need_reward, rewards, strict=False):
                sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(
                args,
                sample,
                use_custom_rm=_should_use_custom_rm(args, sample, evaluation=evaluation),
                evaluation=evaluation,
            )

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    # await asyncio.gather(*[post(f"{url}/abort_request", {"abort_all": True}) for url in urls])
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for url, result in zip(urls, abort_results, strict=False):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info("First rollout sample: %s", _summarize_sample_for_log(sample))
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info("Finish rollout: %s", _summarize_sample_for_log(sample))

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            # Per-dataset kwargs override the global --apply-chat-template-kwargs.
            # This lets eval use a different template (e.g. enable_thinking=true) than rollout.
            apply_chat_template_kwargs=(
                dataset_cfg.apply_chat_template_kwargs
                if dataset_cfg.apply_chat_template_kwargs is not None
                else args.apply_chat_template_kwargs
            ),
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(sample.prompt) + sample.response]} "
                f"reward={sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key

    def get_eval_reward(sample: Sample):
        reward = sample.reward
        if reward_key and isinstance(reward, dict):
            return reward[reward_key]
        return reward

    return {
        dataset_cfg.name: {
            "rewards": [get_eval_reward(sample) for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
            "n_samples_per_eval_prompt": dataset_cfg.n_samples_per_eval_prompt,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to get and store samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
