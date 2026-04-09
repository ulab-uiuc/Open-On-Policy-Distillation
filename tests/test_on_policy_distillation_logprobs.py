import logging
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from slime.rollout.on_policy_distillation import _extract_response_log_probs_with_mask


def _teacher_output_from_input(items):
    return {"meta_info": {"input_token_logprobs": items}}

def test_extract_response_log_probs_masks_leading_none():
    values, mask = _extract_response_log_probs_with_mask(
        _teacher_output_from_input([None, [-1.25, 11], [-0.5, 12]]),
        full_input_len=3,
        response_start=0,
        response_len=3,
    )

    assert values == [0.0, -1.25, -0.5]
    assert mask == [0, 1, 1]


def test_extract_response_log_probs_masks_none_after_precise_slice():
    values, mask = _extract_response_log_probs_with_mask(
        _teacher_output_from_input(
            [
                [-9.0, 0],
                [-8.0, 1],
                [-7.0, 2],
                [-0.3, 3],
                None,
                [-0.7, 5],
            ]
        ),
        full_input_len=6,
        response_start=3,
        response_len=3,
    )

    assert values == [-0.3, 0.0, -0.7]
    assert mask == [1, 0, 1]


def test_extract_response_log_probs_masks_structured_list_missing():
    values, mask = _extract_response_log_probs_with_mask(
        _teacher_output_from_input([[None, 11], [-1.25, 12], [-0.5, 13]]),
        full_input_len=3,
        response_start=0,
        response_len=3,
    )

    assert values == [0.0, -1.25, -0.5]
    assert mask == [0, 1, 1]


def test_extract_response_log_probs_masks_structured_dict_missing():
    values, mask = _extract_response_log_probs_with_mask(
        _teacher_output_from_input(
            [
                {"token_id": 11, "logprob": None},
                {"token_id": 12, "logprob": -1.25},
                {"token_id": 13, "value": -0.5},
            ]
        ),
        full_input_len=3,
        response_start=0,
        response_len=3,
    )

    assert values == [0.0, -1.25, -0.5]
    assert mask == [0, 1, 1]


@pytest.mark.parametrize(
    ("items", "full_input_len", "response_start", "response_len", "expected_values"),
    [
        ([[-9.0, 0], [-8.0, 1], [-0.4, 2], [-0.5, 3]], 5, 3, 2, [-0.4, -0.5]),
        ([[-9.0, 0], [-8.0, 1], [-7.0, 2], [-0.4, 3], [-0.5, 4], [-0.6, 5]], 5, 2, 2, [-0.4, -0.5]),
    ],
)
def test_extract_response_log_probs_handles_shifted_input_lengths(
    items, full_input_len, response_start, response_len, expected_values
):
    values, mask = _extract_response_log_probs_with_mask(
        _teacher_output_from_input(items),
        full_input_len=full_input_len,
        response_start=response_start,
        response_len=response_len,
    )

    assert values == expected_values
    assert mask == [1] * response_len


def test_extract_response_log_probs_falls_back_to_output_tail_slice(caplog):
    teacher_output = {
        "meta_info": {
            "input_token_logprobs": ["bad", "still bad", "not parseable"],
            "output_token_logprobs": [[-0.1, 101], [-0.2, 102], [-0.3, 103]],
        }
    }

    with caplog.at_level(logging.WARNING):
        values, mask = _extract_response_log_probs_with_mask(
            teacher_output,
            full_input_len=10,
            response_start=7,
            response_len=3,
        )

    assert values == [-0.1, -0.2, -0.3]
    assert mask == [1, 1, 1]
    assert not any("missing positions" in rec.message for rec in caplog.records)


def test_extract_response_log_probs_raises_on_non_none_invalid_item():
    with pytest.raises(ValueError, match="Failed to extract teacher response logprobs"):
        _extract_response_log_probs_with_mask(
            _teacher_output_from_input([[-0.1, 1], object(), [-0.2, 2]]),
            full_input_len=3,
            response_start=0,
            response_len=3,
        )


def test_extract_response_log_probs_raises_when_all_positions_missing():
    with pytest.raises(ValueError, match="all positions are missing"):
        _extract_response_log_probs_with_mask(
            _teacher_output_from_input([None, None, None]),
            full_input_len=3,
            response_start=0,
            response_len=3,
        )


def test_extract_response_log_probs_raises_when_all_structured_positions_missing():
    with pytest.raises(ValueError, match="all positions are missing"):
        _extract_response_log_probs_with_mask(
            _teacher_output_from_input([[None, 11], {"logprob": None}, {"value": None}]),
            full_input_len=3,
            response_start=0,
            response_len=3,
        )


def test_extract_response_log_probs_logs_missing_metadata(caplog):
    with caplog.at_level(logging.WARNING):
        values, mask = _extract_response_log_probs_with_mask(
            _teacher_output_from_input([[-1.0, 1], None, [-0.5, 2]]),
            full_input_len=3,
            response_start=0,
            response_len=3,
        )

    assert values == [-1.0, 0.0, -0.5]
    assert mask == [1, 0, 1]
    assert any("missing_positions=[1]" in rec.message for rec in caplog.records)
