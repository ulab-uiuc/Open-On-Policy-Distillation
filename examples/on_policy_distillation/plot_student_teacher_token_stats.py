#!/usr/bin/env python3
"""
Plot token-level teacher-student log-prob deltas from
eval_student_teacher_inference.py output.
"""


'''
pip install matplotlib
python examples/on_policy_distillation/plot_student_teacher_token_stats.py \
  --input ./eval_math500_student_teacher_inference.jsonl \
  --output-dir ./student_teacher_plots

'''
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _as_float_array(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    return np.asarray([float(v) for v in values], dtype=np.float64)


def _extract_teacher_minus_student(token_stats: dict) -> np.ndarray:
    student_lp = _as_float_array(token_stats.get("student_logprobs"))
    teacher_lp = _as_float_array(token_stats.get("teacher_logprobs"))
    if student_lp.size == 0 or teacher_lp.size == 0:
        return np.array([], dtype=np.float64)
    n = min(student_lp.size, teacher_lp.size)
    return teacher_lp[:n] - student_lp[:n]


def _select_plot_series(token_stats: dict) -> tuple[np.ndarray, str, str, str, str]:
    student_lp = _as_float_array(token_stats.get("student_logprobs"))
    teacher_lp = _as_float_array(token_stats.get("teacher_logprobs"))
    if student_lp.size > 0 and teacher_lp.size > 0:
        n = min(student_lp.size, teacher_lp.size)
        return (
            teacher_lp[:n] - student_lp[:n],
            "teacher - student",
            "teacher logprob - student logprob",
            "teacher_minus_student",
            "#1f77b4",
        )
    if student_lp.size > 0:
        return (
            student_lp,
            "student logprob",
            "student logprob",
            "student_only",
            "#2ca02c",
        )
    if teacher_lp.size > 0:
        return (
            teacher_lp,
            "teacher logprob",
            "teacher logprob",
            "teacher_only",
            "#ff7f0e",
        )
    return np.array([], dtype=np.float64), "", "", "", "#1f77b4"


def _char_to_token_position_from_ratio(text: str, char_pos: int | None, fallback_token_count: int) -> int | None:
    if char_pos is None or fallback_token_count <= 0:
        return None
    if not text:
        return fallback_token_count
    ratio = min(max(char_pos / max(len(text), 1), 0.0), 1.0)
    return min(max(int(round(ratio * fallback_token_count)), 1), fallback_token_count)


def _find_thinking_end_token_position(record: dict, token_stats: dict, fallback_token_count: int) -> tuple[int | None, bool]:
    # Preferred: consume explicit field if upstream writer already provides it.
    explicit_end = token_stats.get("thinking_token_end")
    if isinstance(explicit_end, int) and explicit_end > 0:
        return min(explicit_end, fallback_token_count), False

    response_text = record.get("student_response") or ""
    think_start = token_stats.get("thinking_token_start")
    if not (isinstance(think_start, int) and think_start > 0):
        if "<think>" not in response_text:
            return None, False

    think_open_idx = response_text.find("<think>")
    close_tag = "</think>"
    close_idx = response_text.find(close_tag, think_open_idx + len("<think>")) if think_open_idx >= 0 else -1
    if close_idx >= 0:
        # Mark thinking end right after the closing tag.
        char_end = close_idx + len(close_tag)
        return _char_to_token_position_from_ratio(response_text, char_end, fallback_token_count), False

    # If think was started but not closed, treat response end as thinking end.
    return fallback_token_count if fallback_token_count > 0 else None, True


def _get_y_anchor(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite))


def plot_per_request(records: list[dict], output_dir: Path, max_requests: int | None, dpi: int):
    per_req_dir = output_dir / "per_request_teacher_minus_student"
    per_req_dir.mkdir(parents=True, exist_ok=True)
    if not per_req_dir.is_dir() or not per_req_dir.exists():
        raise RuntimeError(f"Failed to create output directory: {per_req_dir}")

    count = 0
    series_kind_counts: dict[str, int] = {}
    for record in records:
        if max_requests is not None and count >= max_requests:
            break

        idx = int(record.get("index", count))
        token_stats = record.get("token_stats") or {}
        plot_values, line_label, y_label, series_kind, line_color = _select_plot_series(token_stats)
        if plot_values.size == 0:
            continue

        series_kind_counts[series_kind] = series_kind_counts.get(series_kind, 0) + 1
        n = plot_values.size
        x = np.arange(1, n + 1)

        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(x, plot_values, lw=1.2, label=line_label, color=line_color)
        ax.axhline(0.0, color="#444444", ls="--", lw=1.0, alpha=0.7)

        answer_pos = token_stats.get("answer_token_start")
        think_pos = token_stats.get("thinking_token_start")
        think_end_pos, think_end_is_truncated = _find_thinking_end_token_position(record, token_stats, n)
        ymax = _get_y_anchor(plot_values)

        if isinstance(answer_pos, int) and answer_pos > 0:
            ax.axvline(answer_pos, color="#d62728", ls="--", lw=1.0, alpha=0.85, label="answer start")
            ax.text(answer_pos, ymax, " answer", color="#d62728", va="bottom", ha="left", fontsize=9)
        if isinstance(think_pos, int) and think_pos > 0:
            ax.axvline(think_pos, color="#2ca02c", ls="--", lw=1.0, alpha=0.85, label="thinking start")
            ax.text(think_pos, ymax, " think", color="#2ca02c", va="bottom", ha="left", fontsize=9)
        if isinstance(think_end_pos, int) and think_end_pos > 0:
            end_label = "thinking end (response end)" if think_end_is_truncated else "thinking end"
            ax.axvline(think_end_pos, color="#9467bd", ls="-.", lw=1.0, alpha=0.9, label=end_label)
            end_text = " think_end*" if think_end_is_truncated else " think_end"
            ax.text(think_end_pos, ymax, end_text, color="#9467bd", va="bottom", ha="left", fontsize=9)

        ax.set_xlabel("token position")
        ax.set_ylabel(y_label)
        ax.set_title(
            f"Request {idx} | tokens={n} | reward={record.get('student_reward')} "
            f"| series={series_kind} | mean={np.nanmean(plot_values):.4f} p50={np.nanpercentile(plot_values, 50):.4f}"
        )
        ax.grid(alpha=0.22)
        ax.legend(loc="best")
        fig.tight_layout()
        out_file = per_req_dir / f"request_{idx:05d}_{series_kind}.png"
        try:
            fig.savefig(out_file, dpi=dpi)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot write to {out_file}. Check --output-dir permissions or use a new directory."
            ) from e
        plt.close(fig)
        count += 1

    if count == 0:
        print(
            "No per-request figures generated. "
            "Need at least one of token_stats.student_logprobs or token_stats.teacher_logprobs."
        )
    else:
        print(f"Generated {count} per-request figures in {per_req_dir}")
        print(f"Per-request series breakdown: {series_kind_counts}")


def plot_overall_quantiles(records: list[dict], output_dir: Path, dpi: int):
    all_values_by_kind: dict[str, list[np.ndarray]] = {
        "student_only": [],
        "teacher_only": [],
    }
    ylabels_by_kind = {
        "student_only": "student logprob",
        "teacher_only": "teacher logprob",
    }
    colors_by_kind = {
        "student_only": "#2ca02c",
        "teacher_only": "#1f77b4",
    }

    # Collect per-position values for teacher_minus_student
    pos_values: dict[int, list[float]] = {}
    n_tms_records = 0

    for record in records:
        token_stats = record.get("token_stats") or {}
        values, _, _, series_kind, _ = _select_plot_series(token_stats)
        if values.size == 0:
            continue
        if series_kind == "teacher_minus_student":
            n_tms_records += 1
            for i, v in enumerate(values):
                if np.isfinite(v):
                    pos_values.setdefault(i, []).append(float(v))
        else:
            valid = values[np.isfinite(values)]
            if valid.size > 0:
                all_values_by_kind.setdefault(series_kind, []).append(valid)

    generated = 0

    # --- teacher_minus_student: x=token position, y=mean + quantile bands ---
    if pos_values:
        positions = sorted(pos_values.keys())
        x = [p + 1 for p in positions]  # 1-indexed token positions
        means = [float(np.mean(pos_values[p])) for p in positions]
        q10 = [float(np.percentile(pos_values[p], 10)) for p in positions]
        q25 = [float(np.percentile(pos_values[p], 25)) for p in positions]
        q50 = [float(np.percentile(pos_values[p], 50)) for p in positions]
        q75 = [float(np.percentile(pos_values[p], 75)) for p in positions]
        q90 = [float(np.percentile(pos_values[p], 90)) for p in positions]

        color = "#1f77b4"
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.fill_between(x, q10, q90, alpha=0.15, color=color, label="p10–p90")
        ax.fill_between(x, q25, q75, alpha=0.30, color=color, label="p25–p75")
        ax.plot(x, q50, lw=1.2, color=color, ls="--", label="median (p50)")
        ax.plot(x, means, lw=1.5, color="#d62728", label="mean")
        ax.axhline(0.0, color="#444444", ls="--", lw=1.0, alpha=0.7)
        ax.set_xlabel("token position")
        ax.set_ylabel("teacher logprob - student logprob")
        ax.set_title(
            f"Teacher − Student log-prob by token position "
            f"(n_records={n_tms_records}, max_pos={max(x)})"
        )
        ax.grid(alpha=0.22)
        ax.legend(loc="best")
        fig.tight_layout()
        out_file = output_dir / "summary_teacher_minus_student_quantiles.png"
        try:
            fig.savefig(out_file, dpi=dpi)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot write to {out_file}. Check --output-dir permissions or use a new directory."
            ) from e
        plt.close(fig)
        generated += 1

    # --- other series: original percentile-curve plots ---
    for series_kind, chunks in all_values_by_kind.items():
        if not chunks:
            continue
        merged = np.concatenate(chunks)
        if merged.size == 0:
            continue
        quantiles = np.linspace(0.0, 100.0, 201)
        quantile_values = np.percentile(merged, quantiles)
        fig, ax = plt.subplots(figsize=(11, 5.2))
        ax.plot(
            quantiles,
            quantile_values,
            color=colors_by_kind.get(series_kind, "#ff7f0e"),
            lw=1.5,
            label=f"{series_kind} quantile curve",
        )
        ax.axhline(0.0, color="#444444", ls="--", lw=1.0, alpha=0.7)
        ax.set_xlabel("percentile")
        ax.set_ylabel(ylabels_by_kind.get(series_kind, "logprob"))
        ax.set_title(f"Overall Quantiles ({series_kind}, n_tokens={merged.size})")
        ax.grid(alpha=0.22)
        ax.legend(loc="best")
        fig.tight_layout()
        out_file = output_dir / f"summary_{series_kind}_quantiles.png"
        try:
            fig.savefig(out_file, dpi=dpi)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot write to {out_file}. Check --output-dir permissions or use a new directory."
            ) from e
        plt.close(fig)
        generated += 1

    if generated == 0:
        print("No summary quantile figure generated (no finite token-level logprob values).")
    else:
        print(f"Generated {generated} summary quantile figure(s) in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to JSONL output from eval_student_teacher_inference.py.")
    parser.add_argument(
        "--output-dir",
        default="./eval_student_teacher_plots",
        help="Directory to save per-request and summary plots.",
    )
    parser.add_argument("--max-requests", type=int, default=None, help="Optional cap on number of per-request figures.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    args = parser.parse_args()

    records = load_records(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_per_request(records, output_dir=output_dir, max_requests=args.max_requests, dpi=args.dpi)
    plot_overall_quantiles(records, output_dir=output_dir, dpi=args.dpi)

    print(f"Loaded {len(records)} records from {args.input}")
    print(f"Saved figures under {output_dir}")


if __name__ == "__main__":
    main()
