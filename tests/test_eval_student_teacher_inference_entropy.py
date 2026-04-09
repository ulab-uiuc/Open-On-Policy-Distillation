import argparse
import asyncio
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from examples.on_policy_distillation import eval_student_teacher_inference as mod


def _make_args(tmp_path: pathlib.Path, record_entropy: bool) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=str(tmp_path / "dummy.jsonl"),
        n_samples=None,
        seed=42,
        model="student-model",
        student_api_base="http://127.0.0.1:30000/v1",
        student_api_key="EMPTY",
        teacher_configs=[
            {
                "index": 0,
                "mode": "answer_only",
                "model": "teacher-model",
                "api_base": "http://127.0.0.1:30001/v1",
                "api_key": "EMPTY",
            }
        ],
        teacher_enable_thinking=None,
        teacher_think_max_tokens=-1,
        mask_ratio=0.5,
        conciseness_instruction="concise",
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=32,
        generation_concurrency=1,
        score_concurrency=1,
        retries=0,
        disable_adaptive_oom_retry=True,
        oom_min_max_new_tokens=8,
        oom_cooldown_seconds=0.0,
        score_logprob_start="response",
        score_temperature=1.0,
        max_score_response_tokens=None,
        score_chunk_tokens=None,
        score_context_window_tokens=None,
        oom_min_score_response_tokens=4,
        output=str(tmp_path / "out.jsonl"),
        student_enable_thinking=None,
        record_student_token_entropy=record_entropy,
        student_token_entropy_mode="strict_exact",
        student_token_entropy_topk=50,
    )


def _patch_minimal_run_eval(monkeypatch):
    sample = {
        "prompt": [{"role": "user", "content": "Q?"}],
        "label": "A",
        "metadata": {"raw_content": "Q?"},
    }
    monkeypatch.setattr(mod, "load_dataset", lambda path, n_samples, seed: [sample])
    monkeypatch.setattr(mod, "_load_tokenizer", lambda source: object())
    monkeypatch.setattr(mod, "build_generation_input_ids", lambda **kwargs: [1, 2, 3])
    monkeypatch.setattr(mod, "grade", lambda response, label, fmt: 1.0)
    monkeypatch.setattr(mod, "_char_to_token_position", lambda *args, **kwargs: None)


def test_run_eval_record_false_outputs_null_entropy(tmp_path, monkeypatch):
    _patch_minimal_run_eval(monkeypatch)
    args = _make_args(tmp_path, record_entropy=False)

    monkeypatch.setattr(
        mod,
        "batch_generate_with_logprobs",
        lambda **kwargs: (["student"], [[11, 12, 13]], [[-0.1, -0.2, -0.3]], [None]),
    )
    monkeypatch.setattr(mod, "_score_with_teacher_config", lambda **kwargs: {0: [-0.4, -0.5, -0.6]})

    mod.run_eval(args)

    line = (tmp_path / "out.jsonl").read_text(encoding="utf-8").strip()
    row = json.loads(line)
    assert row["token_stats"]["student_entropies"] is None


def test_extract_student_entropy_strict_exact_success():
    ent = mod.extract_student_output_token_entropies(
        meta_info={"output_token_entropy": [0.2, 0.3]},
        output_token_logprobs=[[-0.1, 1], [-0.2, 2]],
        mode="strict_exact",
    )
    assert ent == [0.2, 0.3]


def test_extract_student_entropy_strict_exact_from_output_items():
    ent = mod.extract_student_output_token_entropies(
        meta_info={},
        output_token_logprobs=[[-0.1, 1, {"1": -0.1}, 0.21], [-0.2, 2, {"2": -0.2}, 0.31]],
        mode="strict_exact",
    )
    assert ent == [0.21, 0.31]


def test_extract_student_entropy_strict_exact_from_output_items_entropy_val_alias():
    ent = mod.extract_student_output_token_entropies(
        meta_info={},
        output_token_logprobs=[
            {"logprob": -0.1, "token_id": 1, "output_token_entropy_val": 0.21},
            {"logprob": -0.2, "token_id": 2, "entropy_val": 0.31},
        ],
        mode="strict_exact",
    )
    assert ent == [0.21, 0.31]


def test_extract_student_entropy_topk_approx_accepts_string_token_keys():
    ent = mod.extract_student_output_token_entropies(
        meta_info={},
        output_token_logprobs=[
            [-0.1, 1, {"hello": -0.2, "world": -1.5}],
            [-0.3, 2, {"A": -0.7, "B": -1.2}],
        ],
        mode="topk_approx",
    )
    assert len(ent) == 2
    assert all(isinstance(x, float) for x in ent)


def test_extract_student_entropy_topk_approx_accepts_dict_list_mix():
    ent = mod.extract_student_output_token_entropies(
        meta_info={},
        output_token_logprobs=[
            [-0.1, 1, [{"token": "x", "logprob": -0.4}, {"token_id": 12, "logprob": -1.1}]],
        ],
        mode="topk_approx",
    )
    assert len(ent) == 1
    assert isinstance(ent[0], float)


def test_extract_student_entropy_topk_approx_from_sidecar_output_top_logprobs():
    ent = mod.extract_student_output_token_entropies(
        meta_info={
            "output_top_logprobs": [
                {"hello": -0.2, "world": -1.3},
                {"foo": -0.4, "bar": -1.1},
            ]
        },
        output_token_logprobs=[[-0.1, 1, None], [-0.2, 2, None]],
        mode="topk_approx",
    )
    assert len(ent) == 2
    assert all(isinstance(x, float) for x in ent)


def test_extract_student_entropy_strict_exact_accepts_fullmass_sidecar():
    # exp(log(0.7)) + exp(log(0.3)) == 1.0
    ent = mod.extract_student_output_token_entropies(
        meta_info={"output_top_logprobs": [{"a": -0.3566749439, "b": -1.2039728043}]},
        output_token_logprobs=[[-0.1, 1, None]],
        mode="strict_exact",
    )
    assert len(ent) == 1
    assert isinstance(ent[0], float)


def test_extract_student_entropy_strict_exact_accepts_fullmass_embedded_toplogprobs():
    # exp(log(0.7)) + exp(log(0.3)) == 1.0
    ent = mod.extract_student_output_token_entropies(
        meta_info={},
        output_token_logprobs=[[-0.1, 1, {"1": -0.3566749439, "2": -1.2039728043}]],
        mode="strict_exact",
    )
    assert len(ent) == 1
    assert isinstance(ent[0], float)


def test_extract_student_entropy_strict_exact_rejects_truncated_sidecar():
    with pytest.raises(ValueError, match="strict_exact"):
        mod.extract_student_output_token_entropies(
            meta_info={"output_top_logprobs": [{"a": -2.0, "b": -2.0}]},
            output_token_logprobs=[[-0.1, 1, None]],
            mode="strict_exact",
        )


def test_generate_strict_exact_raises_when_exact_entropy_missing(monkeypatch):
    async def _fake_post_json_checked(session, url, payload):
        return {
            "text": "x",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 10, [[-0.1, 10]]]],
            },
        }

    monkeypatch.setattr(mod, "_post_json_checked", _fake_post_json_checked)

    async def _run():
        await mod._generate_one_with_logprobs(
            session=None,
            generate_url="http://x/generate",
            input_ids=[1],
            sampling_params={"temperature": 1.0, "top_p": 1.0, "max_tokens": 8},
            seed=0,
            retries=0,
            adaptive_oom_retry=False,
            oom_min_max_new_tokens=1,
            oom_cooldown_seconds=0.0,
            record_student_token_entropy=True,
            student_token_entropy_mode="strict_exact",
            student_token_entropy_topk=50,
        )

    with pytest.raises(RuntimeError, match="strict_exact"):
        asyncio.run(_run())


def test_generate_strict_exact_raises_on_entropy_length_mismatch(monkeypatch):
    async def _fake_post_json_checked(session, url, payload):
        return {
            "text": "x",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 10], [-0.2, 11]],
                "output_token_entropy": [0.5],
            },
        }

    monkeypatch.setattr(mod, "_post_json_checked", _fake_post_json_checked)

    async def _run():
        await mod._generate_one_with_logprobs(
            session=None,
            generate_url="http://x/generate",
            input_ids=[1],
            sampling_params={"temperature": 1.0, "top_p": 1.0, "max_tokens": 8},
            seed=0,
            retries=0,
            adaptive_oom_retry=False,
            oom_min_max_new_tokens=1,
            oom_cooldown_seconds=0.0,
            record_student_token_entropy=True,
            student_token_entropy_mode="strict_exact",
            student_token_entropy_topk=50,
        )

    with pytest.raises(RuntimeError, match="length mismatch"):
        asyncio.run(_run())


def test_generate_strict_exact_sends_top_logprobs_num(monkeypatch):
    captured = {}

    async def _fake_post_json_checked(session, url, payload):
        captured.update(payload)
        return {
            "text": "x",
            "meta_info": {
                "output_token_logprobs": [
                    [-0.1, 10, {"1": -0.3566749439, "2": -1.2039728043}],
                ],
            },
        }

    monkeypatch.setattr(mod, "_post_json_checked", _fake_post_json_checked)

    async def _run():
        await mod._generate_one_with_logprobs(
            session=None,
            generate_url="http://x/generate",
            input_ids=[1],
            sampling_params={"temperature": 1.0, "top_p": 1.0, "max_tokens": 8},
            seed=0,
            retries=0,
            adaptive_oom_retry=False,
            oom_min_max_new_tokens=1,
            oom_cooldown_seconds=0.0,
            record_student_token_entropy=True,
            student_token_entropy_mode="strict_exact",
            student_token_entropy_topk=77,
        )

    asyncio.run(_run())
    assert captured.get("top_logprobs_num") == 77


def test_run_eval_entropy_aligned_to_common_len(tmp_path, monkeypatch):
    _patch_minimal_run_eval(monkeypatch)
    args = _make_args(tmp_path, record_entropy=True)

    monkeypatch.setattr(
        mod,
        "batch_generate_with_logprobs",
        lambda **kwargs: (["student"], [[11, 12, 13]], [[-0.1, -0.2, -0.3]], [[0.9, 0.8, 0.7]]),
    )
    monkeypatch.setattr(mod, "_score_with_teacher_config", lambda **kwargs: {0: [-0.4, -0.5]})

    mod.run_eval(args)

    line = (tmp_path / "out.jsonl").read_text(encoding="utf-8").strip()
    row = json.loads(line)
    assert row["token_stats"]["student_logprobs"] == [-0.1, -0.2]
    assert row["token_stats"]["student_entropies"] == [0.9, 0.8]
