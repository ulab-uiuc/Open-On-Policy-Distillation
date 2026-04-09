#!/usr/bin/env python3
"""
Analyze teacher-student logprob deltas from eval JSONL.

Metrics:
1) Mean (teacher - student) logprob for reward=1 and reward=0.
2) Mean (teacher - student) logprob for \\boxed{} spans in:
   - content inside <think>...</think>
   - final output (outside <think>...</think>)
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _as_float_array(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    return np.asarray([float(v) for v in values], dtype=np.float64)


def _teacher_minus_student(record: dict) -> np.ndarray:
    token_stats = record.get("token_stats") or {}
    student_lp = _as_float_array(token_stats.get("student_logprobs"))
    teacher_lp = _as_float_array(token_stats.get("teacher_logprobs"))
    if student_lp.size == 0 or teacher_lp.size == 0:
        return np.array([], dtype=np.float64)
    n = min(student_lp.size, teacher_lp.size)
    return teacher_lp[:n] - student_lp[:n]


def _short_model_name(model: str | None) -> str:
    if not model:
        return "unknown-model"
    s = str(model).rstrip("/")
    return s.split("/")[-1] or s


def _iter_group_deltas(record: dict):
    token_stats = record.get("token_stats") or {}
    student_lp = _as_float_array(token_stats.get("student_logprobs"))
    if student_lp.size == 0:
        return

    # Keep the original top-level teacher channel as a separate baseline group.
    top_teacher = _as_float_array(token_stats.get("teacher_logprobs"))
    if top_teacher.size > 0:
        n = min(student_lp.size, top_teacher.size)
        if n > 0:
            yield "default_teacher_logprobs", top_teacher[:n] - student_lp[:n]

    teachers = token_stats.get("teachers") or []
    if not isinstance(teachers, list):
        return

    for i, teacher in enumerate(teachers):
        if not isinstance(teacher, dict):
            continue
        teacher_lp = _as_float_array(teacher.get("logprobs"))
        if teacher_lp.size == 0:
            continue
        n = min(student_lp.size, teacher_lp.size)
        if n <= 0:
            continue
        mode = teacher.get("mode") or "unknown-mode"
        model = _short_model_name(teacher.get("model"))
        api = teacher.get("api_base") or "unknown-api"
        label = f"teacher[{i}] mode={mode} model={model} api={api}"
        yield label, teacher_lp[:n] - student_lp[:n]


def _reward_bucket(v) -> int | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isclose(fv, 1.0, rel_tol=0.0, abs_tol=1e-8):
        return 1
    if math.isclose(fv, 0.0, rel_tol=0.0, abs_tol=1e-8):
        return 0
    return None


def _find_think_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    open_tag = "<think>"
    close_tag = "</think>"
    while True:
        s = text.find(open_tag, pos)
        if s < 0:
            break
        content_start = s + len(open_tag)
        e = text.find(close_tag, content_start)
        if e < 0:
            spans.append((content_start, len(text)))
            break
        spans.append((content_start, e))
        pos = e + len(close_tag)
    return spans


def _find_boxed_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    needle = r"\boxed{"
    pos = 0
    n = len(text)
    while True:
        s = text.find(needle, pos)
        if s < 0:
            break
        i = s + len(needle)
        depth = 1
        while i < n and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth == 0:
            spans.append((s, i))
            pos = i
        else:
            # Unclosed \boxed{...}; skip this occurrence safely.
            pos = s + len(needle)
    return spans


def _span_midpoint_in_any(span: tuple[int, int], regions: list[tuple[int, int]]) -> bool:
    s, e = span
    mid = (s + e) // 2
    for rs, re in regions:
        if rs <= mid < re:
            return True
    return False


def _char_span_to_token_span(span: tuple[int, int], text_len: int, token_count: int) -> tuple[int, int]:
    if token_count <= 0:
        return (0, 0)
    if text_len <= 0:
        return (0, token_count)
    s, e = span
    s = max(0, min(s, text_len))
    e = max(s, min(e, text_len))
    start = int(math.floor((s / text_len) * token_count))
    end = int(math.ceil((e / text_len) * token_count))
    start = max(0, min(start, token_count))
    end = max(start, min(end, token_count))
    if end == start and start < token_count:
        end = start + 1
    return (start, end)


@dataclass
class Stats:
    token_sum: float = 0.0
    token_count: int = 0
    span_sum: float = 0.0
    span_count: int = 0
    record_sum: float = 0.0
    record_count: int = 0

    def add_token_values(self, values: np.ndarray):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        self.token_sum += float(np.sum(finite))
        self.token_count += int(finite.size)

    def add_span(self, values: np.ndarray):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        self.token_sum += float(np.sum(finite))
        self.token_count += int(finite.size)
        self.span_sum += float(np.mean(finite))
        self.span_count += 1

    def add_record(self, values: np.ndarray):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        self.record_sum += float(np.mean(finite))
        self.record_count += 1

    def token_mean(self) -> float | None:
        if self.token_count == 0:
            return None
        return self.token_sum / self.token_count

    def span_mean(self) -> float | None:
        if self.span_count == 0:
            return None
        return self.span_sum / self.span_count

    def record_mean(self) -> float | None:
        if self.record_count == 0:
            return None
        return self.record_sum / self.record_count


def _fmt(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.6f}"


@dataclass
class GroupMetrics:
    reward_stats: dict[int, Stats]
    boxed_think_stats: Stats
    boxed_final_stats: Stats
    boxed_think_by_reward: dict[int, Stats]
    boxed_final_by_reward: dict[int, Stats]
    valid_delta_records: int = 0

    @classmethod
    def create(cls):
        return cls(
            reward_stats={0: Stats(), 1: Stats()},
            boxed_think_stats=Stats(),
            boxed_final_stats=Stats(),
            boxed_think_by_reward={0: Stats(), 1: Stats()},
            boxed_final_by_reward={0: Stats(), 1: Stats()},
            valid_delta_records=0,
        )


def analyze(path: Path):
    reward_stats = {0: Stats(), 1: Stats()}
    boxed_think_stats = Stats()
    boxed_final_stats = Stats()
    groups: dict[str, GroupMetrics] = defaultdict(GroupMetrics.create)

    total_records = 0
    valid_delta_records = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            record = json.loads(line)
            delta = _teacher_minus_student(record)
            if delta.size == 0:
                continue
            valid_delta_records += 1

            reward = _reward_bucket(record.get("student_reward"))
            if reward in reward_stats:
                reward_stats[reward].add_token_values(delta)
                reward_stats[reward].add_record(delta)

            text = record.get("student_response") or ""
            boxed_spans = _find_boxed_spans(text)
            if not boxed_spans:
                continue

            think_spans = _find_think_spans(text)
            for cspan in boxed_spans:
                ts, te = _char_span_to_token_span(cspan, len(text), delta.size)
                if te <= ts:
                    continue
                seg = delta[ts:te]
                if _span_midpoint_in_any(cspan, think_spans):
                    boxed_think_stats.add_span(seg)
                    if reward in (0, 1):
                        groups["__overall__"].boxed_think_by_reward[reward].add_span(seg)
                else:
                    boxed_final_stats.add_span(seg)
                    if reward in (0, 1):
                        groups["__overall__"].boxed_final_by_reward[reward].add_span(seg)

            for group_name, g_delta in _iter_group_deltas(record):
                g = groups[group_name]
                g.valid_delta_records += 1

                reward = _reward_bucket(record.get("student_reward"))
                if reward in g.reward_stats:
                    g.reward_stats[reward].add_token_values(g_delta)
                    g.reward_stats[reward].add_record(g_delta)

                for cspan in boxed_spans:
                    ts, te = _char_span_to_token_span(cspan, len(text), g_delta.size)
                    if te <= ts:
                        continue
                    seg = g_delta[ts:te]
                    if _span_midpoint_in_any(cspan, think_spans):
                        g.boxed_think_stats.add_span(seg)
                        if reward in (0, 1):
                            g.boxed_think_by_reward[reward].add_span(seg)
                    else:
                        g.boxed_final_stats.add_span(seg)
                        if reward in (0, 1):
                            g.boxed_final_by_reward[reward].add_span(seg)

    print(f"Input: {path}")
    print(f"Total records: {total_records}")
    print(f"Records with valid teacher/student logprobs: {valid_delta_records}")
    print()
    print("[Reward buckets] teacher - student logprob")
    for r in (1, 0):
        s = reward_stats[r]
        print(
            f"reward={r}: token_mean={_fmt(s.token_mean())} (tokens={s.token_count}), "
            f"record_mean={_fmt(s.record_mean())} (records={s.record_count})"
        )
    print()
    print("[\\\\boxed{} spans] teacher - student logprob")
    print(
        "boxed_in_think: "
        f"token_mean={_fmt(boxed_think_stats.token_mean())} (tokens={boxed_think_stats.token_count}), "
        f"span_mean={_fmt(boxed_think_stats.span_mean())} (spans={boxed_think_stats.span_count})"
    )
    print(
        "boxed_in_final: "
        f"token_mean={_fmt(boxed_final_stats.token_mean())} (tokens={boxed_final_stats.token_count}), "
        f"span_mean={_fmt(boxed_final_stats.span_mean())} (spans={boxed_final_stats.span_count})"
    )
    for r in (1, 0):
        s_think = groups["__overall__"].boxed_think_by_reward[r]
        s_final = groups["__overall__"].boxed_final_by_reward[r]
        print(
            f"boxed_in_think reward={r}: "
            f"token_mean={_fmt(s_think.token_mean())} (tokens={s_think.token_count}), "
            f"span_mean={_fmt(s_think.span_mean())} (spans={s_think.span_count})"
        )
        print(
            f"boxed_in_final reward={r}: "
            f"token_mean={_fmt(s_final.token_mean())} (tokens={s_final.token_count}), "
            f"span_mean={_fmt(s_final.span_mean())} (spans={s_final.span_count})"
        )
    print()
    print("[By experiment group] teacher - student logprob")
    for group_name in sorted(k for k in groups.keys() if k != "__overall__"):
        g = groups[group_name]
        print()
        print(f"- {group_name} | valid_records={g.valid_delta_records}")
        for r in (1, 0):
            s = g.reward_stats[r]
            print(
                f"  reward={r}: token_mean={_fmt(s.token_mean())} (tokens={s.token_count}), "
                f"record_mean={_fmt(s.record_mean())} (records={s.record_count})"
            )
        print(
            "  boxed_in_think: "
            f"token_mean={_fmt(g.boxed_think_stats.token_mean())} (tokens={g.boxed_think_stats.token_count}), "
            f"span_mean={_fmt(g.boxed_think_stats.span_mean())} (spans={g.boxed_think_stats.span_count})"
        )
        print(
            "  boxed_in_final: "
            f"token_mean={_fmt(g.boxed_final_stats.token_mean())} (tokens={g.boxed_final_stats.token_count}), "
            f"span_mean={_fmt(g.boxed_final_stats.span_mean())} (spans={g.boxed_final_stats.span_count})"
        )
        for r in (1, 0):
            s_think = g.boxed_think_by_reward[r]
            s_final = g.boxed_final_by_reward[r]
            print(
                f"  boxed_in_think reward={r}: "
                f"token_mean={_fmt(s_think.token_mean())} (tokens={s_think.token_count}), "
                f"span_mean={_fmt(s_think.span_mean())} (spans={s_think.span_count})"
            )
            print(
                f"  boxed_in_final reward={r}: "
                f"token_mean={_fmt(s_final.token_mean())} (tokens={s_final.token_count}), "
                f"span_mean={_fmt(s_final.span_mean())} (spans={s_final.span_count})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze reward-conditioned and boxed-span teacher-student logprob deltas."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval_openthoughts_student_teacher_inference_all_nothinking_b256.jsonl"),
        help="Path to eval JSONL file.",
    )
    args = parser.parse_args()
    analyze(args.input)


if __name__ == "__main__":
    main()
