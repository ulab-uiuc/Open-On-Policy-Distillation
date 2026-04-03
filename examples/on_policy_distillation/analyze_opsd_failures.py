#!/usr/bin/env python3
"""
Analyze OPSD failure patterns from student/teacher token-level logprobs.

This script focuses on trajectory-level diagnostics available from eval JSONL files:
- Reverse-KL estimator on student rollouts: E[log p_student - log p_teacher]
- Token-level dominance (teacher win rate)
- Segment-level stats (pre-think / think / answer)
- Reward-conditioned failure analysis (reward=0 vs reward=1)
- Optional paired comparison between answeronly and noanswer settings

Notes:
- We only have logprobs on sampled tokens, so we estimate reverse KL on trajectories,
  not full-distribution forward KL.
- All metrics are descriptive diagnostics; they are not training-time dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FILE_RE = re.compile(
    r"_s(?P<student>[^t_]+)t(?P<teacher>[^_]+)_(?P<mode>answeronly|noanswer)(?:_(?P<variant>[^.]+))?\.jsonl$"
)


@dataclass
class FileTag:
    student: str | None
    teacher: str | None
    mode: str | None
    variant: str | None


@dataclass
class SegmentStats:
    mean_delta: float
    teacher_win_rate: float
    p10_delta: float
    p50_delta: float
    p90_delta: float
    frac_delta_lt_minus1: float
    frac_delta_lt_minus2: float
    count: int


@dataclass
class SampleStats:
    index: int
    reward: float | None
    seq_len: int
    thinking_start: int | None
    answer_start: int | None
    overall: SegmentStats
    prethink: SegmentStats
    think: SegmentStats
    answer: SegmentStats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze OPSD failure diagnostics from eval JSONL files")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files. You can pass explicit files and/or glob patterns.",
    )
    p.add_argument(
        "--outdir",
        default="./opsd_failure_analysis",
        help="Output directory for CSV/Markdown reports.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally cap samples per file for quick debugging.",
    )
    return p.parse_args()


def expand_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if any(ch in item for ch in "*?[]"):
            files.extend(sorted(Path().glob(item)))
        elif p.exists():
            files.append(p)
    uniq: list[Path] = []
    seen = set()
    for f in files:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            uniq.append(f)
    return uniq


def safe_float(x) -> float:
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y


def parse_tag(path: Path) -> FileTag:
    m = FILE_RE.search(path.name)
    if not m:
        return FileTag(student=None, teacher=None, mode=None, variant=None)
    gd = m.groupdict()
    return FileTag(
        student=gd.get("student"),
        teacher=gd.get("teacher"),
        mode=gd.get("mode"),
        variant=gd.get("variant"),
    )


def segment_metrics(delta: np.ndarray) -> SegmentStats:
    if delta.size == 0:
        nan = float("nan")
        return SegmentStats(
            mean_delta=nan,
            teacher_win_rate=nan,
            p10_delta=nan,
            p50_delta=nan,
            p90_delta=nan,
            frac_delta_lt_minus1=nan,
            frac_delta_lt_minus2=nan,
            count=0,
        )

    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        nan = float("nan")
        return SegmentStats(
            mean_delta=nan,
            teacher_win_rate=nan,
            p10_delta=nan,
            p50_delta=nan,
            p90_delta=nan,
            frac_delta_lt_minus1=nan,
            frac_delta_lt_minus2=nan,
            count=0,
        )

    return SegmentStats(
        mean_delta=float(np.mean(finite)),
        teacher_win_rate=float(np.mean(finite < 0.0)),
        p10_delta=float(np.quantile(finite, 0.10)),
        p50_delta=float(np.quantile(finite, 0.50)),
        p90_delta=float(np.quantile(finite, 0.90)),
        frac_delta_lt_minus1=float(np.mean(finite < -1.0)),
        frac_delta_lt_minus2=float(np.mean(finite < -2.0)),
        count=int(finite.size),
    )


def split_segments(delta: np.ndarray, thinking_start: int | None, answer_start: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = delta.size
    ts = 0 if thinking_start is None else int(np.clip(thinking_start, 0, n))
    if answer_start is None:
        ans = n
    else:
        ans = int(np.clip(answer_start, 0, n))

    pre = delta[:ts]
    think = delta[ts:ans]
    answer = delta[ans:]
    return pre, think, answer


def compute_sample_stats(obj: dict) -> SampleStats | None:
    ts = obj.get("token_stats") or {}
    s_raw = ts.get("student_logprobs")
    t_raw = ts.get("teacher_logprobs")
    if s_raw is None or t_raw is None:
        return None

    s = np.asarray([safe_float(x) for x in s_raw], dtype=np.float64)
    t = np.asarray([safe_float(x) for x in t_raw], dtype=np.float64)
    n = min(s.size, t.size)
    if n == 0:
        return None

    s = s[:n]
    t = t[:n]
    delta = s - t
    if not np.any(np.isfinite(delta)):
        return None

    thinking_start = ts.get("thinking_token_start")
    answer_start = ts.get("answer_token_start")
    pre, think, answer = split_segments(delta, thinking_start, answer_start)

    reward = obj.get("student_reward")
    reward = None if reward is None else float(reward)

    return SampleStats(
        index=int(obj.get("index", -1)),
        reward=reward,
        seq_len=int(delta.size),
        thinking_start=int(thinking_start) if isinstance(thinking_start, int) else None,
        answer_start=int(answer_start) if isinstance(answer_start, int) else None,
        overall=segment_metrics(delta),
        prethink=segment_metrics(pre),
        think=segment_metrics(think),
        answer=segment_metrics(answer),
    )


def flatten_sample_row(file_name: str, tag: FileTag, s: SampleStats) -> dict:
    row = {
        "file": file_name,
        "student": tag.student,
        "teacher": tag.teacher,
        "mode": tag.mode,
        "variant": tag.variant,
        "index": s.index,
        "reward": s.reward,
        "seq_len": s.seq_len,
        "thinking_start": s.thinking_start,
        "answer_start": s.answer_start,
    }

    for seg_name, seg in [("overall", s.overall), ("prethink", s.prethink), ("think", s.think), ("answer", s.answer)]:
        row[f"{seg_name}_mean_delta"] = seg.mean_delta
        row[f"{seg_name}_teacher_win_rate"] = seg.teacher_win_rate
        row[f"{seg_name}_p10_delta"] = seg.p10_delta
        row[f"{seg_name}_p50_delta"] = seg.p50_delta
        row[f"{seg_name}_p90_delta"] = seg.p90_delta
        row[f"{seg_name}_frac_delta_lt_minus1"] = seg.frac_delta_lt_minus1
        row[f"{seg_name}_frac_delta_lt_minus2"] = seg.frac_delta_lt_minus2
        row[f"{seg_name}_count"] = seg.count

    return row


def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def aggregate_rows(rows: list[dict], key_filter=None) -> dict:
    if key_filter is not None:
        rows = [r for r in rows if key_filter(r)]
    n = len(rows)
    out = {"n_samples": n}
    if n == 0:
        return out

    mean_fields = [
        "seq_len",
        "overall_mean_delta",
        "overall_teacher_win_rate",
        "overall_frac_delta_lt_minus1",
        "overall_frac_delta_lt_minus2",
        "think_mean_delta",
        "think_teacher_win_rate",
        "answer_mean_delta",
        "answer_teacher_win_rate",
    ]
    for f in mean_fields:
        out[f] = nanmean(r.get(f, float("nan")) for r in rows)

    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paired_mode_comparison(all_rows: list[dict]) -> list[dict]:
    # group by (file stem tag sans mode, index), compare answeronly - noanswer
    grouped = {}
    for r in all_rows:
        mode = r.get("mode")
        if mode not in {"answeronly", "noanswer"}:
            continue
        key = (r.get("student"), r.get("teacher"), r.get("index"))
        grouped.setdefault(key, {})[mode] = r

    out = []
    for (student, teacher, idx), d in grouped.items():
        if "answeronly" not in d or "noanswer" not in d:
            continue
        a = d["answeronly"]
        n = d["noanswer"]
        out.append(
            {
                "student": student,
                "teacher": teacher,
                "index": idx,
                "reward_answeronly": a.get("reward"),
                "reward_noanswer": n.get("reward"),
                "reward_diff_answer_minus_noanswer": safe_float(a.get("reward")) - safe_float(n.get("reward")),
                "overall_mean_delta_diff": safe_float(a.get("overall_mean_delta")) - safe_float(n.get("overall_mean_delta")),
                "overall_teacher_win_rate_diff": safe_float(a.get("overall_teacher_win_rate")) - safe_float(n.get("overall_teacher_win_rate")),
                "think_mean_delta_diff": safe_float(a.get("think_mean_delta")) - safe_float(n.get("think_mean_delta")),
                "answer_mean_delta_diff": safe_float(a.get("answer_mean_delta")) - safe_float(n.get("answer_mean_delta")),
                "seq_len_diff": safe_float(a.get("seq_len")) - safe_float(n.get("seq_len")),
            }
        )
    return out


def fmt(x) -> str:
    if x is None:
        return "NA"
    try:
        if math.isnan(float(x)):
            return "NA"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def write_report_md(path: Path, per_file_summary: list[dict], pair_rows: list[dict]):
    lines = []
    lines.append("# OPSD Failure Analysis Report")
    lines.append("")
    lines.append("Interpretation anchor:")
    lines.append("- `overall_mean_delta = E[log p_student - log p_teacher]` on sampled tokens; this is a reverse-KL trajectory estimator.")
    lines.append("- `teacher_win_rate` is token fraction where teacher logprob > student logprob.")
    lines.append("- Negative delta means teacher is more confident on sampled tokens.")
    lines.append("")
    lines.append("## Per-file Summary")
    lines.append("")
    lines.append("| file | mode | student | teacher | n | acc(mean reward) | revKL_est(mean delta) | teacher_win | think_delta | answer_delta | seq_len |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in per_file_summary:
        lines.append(
            "| {file} | {mode} | {student} | {teacher} | {n} | {acc} | {d} | {tw} | {td} | {ad} | {sl} |".format(
                file=r.get("file"),
                mode=r.get("mode"),
                student=r.get("student"),
                teacher=r.get("teacher"),
                n=r.get("n_samples", 0),
                acc=fmt(r.get("acc")),
                d=fmt(r.get("overall_mean_delta")),
                tw=fmt(r.get("overall_teacher_win_rate")),
                td=fmt(r.get("think_mean_delta")),
                ad=fmt(r.get("answer_mean_delta")),
                sl=fmt(r.get("seq_len")),
            )
        )

    lines.append("")
    lines.append("## Reward-conditioned Diagnostics")
    lines.append("")
    lines.append("For each file, compare reward=0 vs reward=1 groups in `summary_by_file.csv`:")
    lines.append("- If reward=0 has lower `overall_mean_delta` and higher `teacher_win_rate`, failures are linked to student under-confidence vs teacher.")
    lines.append("- If gap is mostly in `think_mean_delta`, failure originates in reasoning trajectory.")
    lines.append("- If gap is mostly in `answer_mean_delta`, failure is concentrated near final answer emission.")

    if pair_rows:
        lines.append("")
        lines.append("## Paired answeronly vs noanswer (same index)")
        lines.append("")
        lines.append("Aggregated over available paired samples:")
        lines.append("- `reward_diff_answer_minus_noanswer` > 0 means answeronly improves reward.")
        lines.append("- `overall_mean_delta_diff` > 0 means answeronly shifts student closer to/above teacher on sampled tokens.")

        arr_reward = np.asarray([safe_float(r.get("reward_diff_answer_minus_noanswer")) for r in pair_rows], dtype=np.float64)
        arr_delta = np.asarray([safe_float(r.get("overall_mean_delta_diff")) for r in pair_rows], dtype=np.float64)
        arr_twr = np.asarray([safe_float(r.get("overall_teacher_win_rate_diff")) for r in pair_rows], dtype=np.float64)
        lines.append("")
        lines.append(f"- paired_n: {len(pair_rows)}")
        lines.append(f"- mean reward diff (answer - noanswer): {fmt(np.nanmean(arr_reward))}")
        lines.append(f"- mean overall delta diff: {fmt(np.nanmean(arr_delta))}")
        lines.append(f"- mean teacher-win diff: {fmt(np.nanmean(arr_twr))}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    files = expand_inputs(args.inputs)
    if not files:
        raise FileNotFoundError("No input files found.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    per_file_summary: list[dict] = []
    summary_by_file_reward: list[dict] = []

    for file_path in files:
        tag = parse_tag(file_path)
        sample_rows: list[dict] = []

        with file_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.max_samples is not None and i >= args.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                s = compute_sample_stats(obj)
                if s is None:
                    continue
                sample_rows.append(flatten_sample_row(file_path.name, tag, s))

        all_rows.extend(sample_rows)

        # save per-file sample rows
        write_csv(outdir / "per_file_samples" / f"{file_path.stem}.csv", sample_rows)

        # aggregate per file
        agg_all = aggregate_rows(sample_rows)
        acc = nanmean(r.get("reward", float("nan")) for r in sample_rows)
        per_file_summary.append(
            {
                "file": file_path.name,
                "student": tag.student,
                "teacher": tag.teacher,
                "mode": tag.mode,
                "variant": tag.variant,
                "acc": acc,
                **agg_all,
            }
        )

        for reward_value in (0.0, 1.0):
            agg = aggregate_rows(sample_rows, key_filter=lambda r, rv=reward_value: safe_float(r.get("reward")) == rv)
            summary_by_file_reward.append(
                {
                    "file": file_path.name,
                    "student": tag.student,
                    "teacher": tag.teacher,
                    "mode": tag.mode,
                    "variant": tag.variant,
                    "reward": reward_value,
                    **agg,
                }
            )

    # global outputs
    write_csv(outdir / "all_samples.csv", all_rows)
    write_csv(outdir / "summary_by_file.csv", per_file_summary)
    write_csv(outdir / "summary_by_file_reward.csv", summary_by_file_reward)

    pair_rows = paired_mode_comparison(all_rows)
    write_csv(outdir / "paired_answeronly_vs_noanswer.csv", pair_rows)

    write_report_md(outdir / "report.md", per_file_summary, pair_rows)

    print(f"[done] analyzed {len(files)} files")
    print(f"[done] outputs under: {outdir.resolve()}")


if __name__ == "__main__":
    main()
