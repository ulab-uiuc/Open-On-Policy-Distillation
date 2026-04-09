from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_verl


def test_auto_falls_back_to_boxed_for_final_answer_heading():
    passage = (
        "We simplify the expression.\n\n"
        "### Final Answer:\n\n"
        "$$\n"
        "\\boxed{15x - 80}\n"
        "$$"
    )
    assert extract_answer(passage, mode="auto") == "15x - 80"
    assert extract_answer(passage, mode="answer") is None


def test_single_line_answer_keeps_existing_behavior():
    passage = "Reasoning...\nAnswer: 42\n"
    assert extract_answer(passage, mode="answer") == "42"
    assert extract_answer(passage, mode="auto") == "42"


def test_answer_mode_uses_last_valid_inline_answer():
    passage = (
        "Answer: $$\n"
        "Answer:\n"
        "Answer: 17\n"
        "Answer: 99<|im_end|>\n"
    )
    assert extract_answer(passage, mode="answer") == "99"
    assert extract_answer(passage, mode="auto") == "99"


def test_grade_answer_verl_auto_accepts_boxed_fallback():
    passage = (
        "<think>\n\n</think>\n\n"
        "### Final Answer:\n\n"
        "$$\n"
        "\\boxed{15x - 80}\n"
        "$$<|im_end|>"
    )
    assert grade_answer_verl(passage, "15x - 80", mode="auto")
