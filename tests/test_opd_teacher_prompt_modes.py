from argparse import Namespace
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from slime.rollout.on_policy_distillation import _build_teacher_user_content  # noqa: E402
from slime.utils.types import Sample  # noqa: E402
from examples.on_policy_distillation.filter_openthoughts_math import _build_privileged_user_content  # noqa: E402


def test_opd_teacher_full_mode_matches_privileged_grpo_prompt_template():
    args = Namespace(opd_teacher_info_mode="full")
    sample = Sample(
        prompt=[{"role": "user", "content": "student prompt"}],
        label=r"\boxed{2}",
        metadata={
            "raw_problem": "Compute 1+1.",
            "raw_content": "Compute 1+1.",
            "student_user_content": "Compute 1+1.\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
            "reference_solution": "1+1=2, so the answer is \\boxed{2}.",
            "format_instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
    )

    expected = (
        "Compute 1+1.\n\n"
        "Here is a reference solution to this problem:\n"
        "1+1=2, so the answer is \\boxed{2}.\n"
        "After understanding the reference solution and the rationale behind each step, "
        "now articulate your own step-by-step reasoning that derives the same final answer "
        "to the problem above:\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

    assert _build_teacher_user_content(args, sample) == expected


def test_opd_teacher_full_mode_matches_filter_privileged_full_builder():
    args = Namespace(opd_teacher_info_mode="full")
    raw_problem = "Find x if x^2=9."
    student_user_content = (
        "Find x if x^2=9.\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    reference_solution = "Since x^2=9, x=\\pm 3."
    format_instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    sample = Sample(
        prompt=[{"role": "user", "content": "unused"}],
        label=r"\boxed{\pm 3}",
        metadata={
            "raw_problem": raw_problem,
            "student_user_content": student_user_content,
            "reference_solution": reference_solution,
            "format_instruction": format_instruction,
        },
    )

    expected_from_filter = _build_privileged_user_content(
        mode="full",
        student_user_content=student_user_content,
        raw_problem=raw_problem,
        label=r"\boxed{\pm 3}",
        reference_solution=reference_solution,
        format_suffix=format_instruction,
    )

    assert _build_teacher_user_content(args, sample) == expected_from_filter


def test_opd_teacher_full_mode_falls_back_to_student_prompt_when_reference_missing():
    args = Namespace(opd_teacher_info_mode="full")
    student_user_content = "Solve x+1=2.\n\nPlease reason step by step."
    sample = Sample(
        prompt=[{"role": "user", "content": "fallback prompt"}],
        metadata={
            "raw_problem": "Solve x+1=2.",
            "student_user_content": student_user_content,
            "reference_solution": "",
            "format_instruction": "Please reason step by step.",
        },
    )

    assert _build_teacher_user_content(args, sample) == student_user_content


def test_opd_teacher_full_mode_uses_raw_content_when_raw_problem_missing():
    args = Namespace(opd_teacher_info_mode="full")
    sample = Sample(
        prompt=[{"role": "user", "content": "student prompt"}],
        metadata={
            "raw_content": "Compute 2+2.",
            "student_user_content": "Compute 2+2.\n\nPlease reason step by step.",
            "reference_solution": "2+2=4.",
            "format_instruction": "Please reason step by step.",
        },
    )

    result = _build_teacher_user_content(args, sample)
    assert result.startswith("Compute 2+2.\n\nHere is a reference solution to this problem:\n2+2=4.")


def test_opd_teacher_mode_rejects_unknown_value():
    args = Namespace(opd_teacher_info_mode="not_a_mode")
    sample = Sample(prompt=[{"role": "user", "content": "hi"}], metadata={})
    with pytest.raises(ValueError, match="Unsupported opd_teacher_info_mode"):
        _build_teacher_user_content(args, sample)
