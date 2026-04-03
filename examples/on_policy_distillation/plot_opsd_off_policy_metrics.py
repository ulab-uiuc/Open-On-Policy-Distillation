#!/usr/bin/env python3
"""
Plot token-position OPSD off-policy metrics from
`eval_student_teacher_inference.py` JSONL output.

Example:
  python examples/on_policy_distillation/plot_opsd_off_policy_metrics.py \
    --input ./eval_math500_student_teacher_inference_s1.7t1.7b_answeronly_b512.jsonl \
    --segment all \
    --format pdf \
    --max-show-tokens 25000 \
    --y-min -2 \
    --y-max 2 \
    --output-dir ./opsd_offpolicy_plots_s1.7t1.7b_answeronly_b512
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with: pip install matplotlib"
    ) from e


ALLOWED_SEGMENTS = {"all", "thinking", "answer"}


def load_records(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        x = int(v)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def _as_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    out = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            out.append(x)
    return np.asarray(out, dtype=np.float64)


def _aligned_logprobs(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ts = record.get("token_stats") or {}
    student = _as_float_array(ts.get("student_logprobs"))
    teacher = _as_float_array(ts.get("teacher_logprobs"))
    if student.size == 0 or teacher.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    n = min(student.size, teacher.size)
    return student[:n], teacher[:n]


def _segment_bounds(
    token_count: int,
    thinking_start_1idx: int | None,
    answer_start_1idx: int | None,
    segment: str,
) -> tuple[bool, int | None, int | None]:
    if segment == "all":
        return True, 0, token_count

    if segment == "thinking":
        if thinking_start_1idx is None:
            return False, None, None
        start = max(0, thinking_start_1idx - 1)
        end = token_count
        if answer_start_1idx is not None:
            end = min(token_count, max(start, answer_start_1idx - 1))
        return True, start, end

    if segment == "answer":
        if answer_start_1idx is None:
            return False, None, None
        start = max(0, answer_start_1idx - 1)
        return True, start, token_count

    return False, None, None


def _append_position_values(dest: dict[int, list[float]], values: np.ndarray) -> None:
    for i, v in enumerate(values):
        if np.isfinite(v):
            dest.setdefault(i, []).append(float(v))


def _ess_from_log_weights(log_w: list[float]) -> float | None:
    vals = [v for v in log_w if math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return None
    m = max(vals)
    ws = [math.exp(v - m) for v in vals]
    sum_w = sum(ws)
    sum_w2 = sum(w * w for w in ws)
    if sum_w2 <= 0.0:
        return None
    ess = (sum_w * sum_w) / sum_w2
    ess = min(max(ess, 1.0), float(n))
    return ess / float(n)


def _stats_from_pos_values(values_by_pos: dict[int, list[float]]) -> dict[str, np.ndarray]:
    if not values_by_pos:
        return {
            "x": np.array([], dtype=np.int32),
            "mean": np.array([], dtype=np.float64),
            "q10": np.array([], dtype=np.float64),
            "q25": np.array([], dtype=np.float64),
            "q50": np.array([], dtype=np.float64),
            "q75": np.array([], dtype=np.float64),
            "q90": np.array([], dtype=np.float64),
            "count": np.array([], dtype=np.int32),
        }

    positions = sorted(values_by_pos.keys())
    x = np.asarray([p + 1 for p in positions], dtype=np.int32)
    means = []
    q10 = []
    q25 = []
    q50 = []
    q75 = []
    q90 = []
    counts = []
    for p in positions:
        arr = np.asarray(values_by_pos[p], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        counts.append(int(arr.size))
        if arr.size == 0:
            means.append(np.nan)
            q10.append(np.nan)
            q25.append(np.nan)
            q50.append(np.nan)
            q75.append(np.nan)
            q90.append(np.nan)
            continue
        means.append(float(np.mean(arr)))
        q10.append(float(np.percentile(arr, 10)))
        q25.append(float(np.percentile(arr, 25)))
        q50.append(float(np.percentile(arr, 50)))
        q75.append(float(np.percentile(arr, 75)))
        q90.append(float(np.percentile(arr, 90)))

    return {
        "x": x,
        "mean": np.asarray(means, dtype=np.float64),
        "q10": np.asarray(q10, dtype=np.float64),
        "q25": np.asarray(q25, dtype=np.float64),
        "q50": np.asarray(q50, dtype=np.float64),
        "q75": np.asarray(q75, dtype=np.float64),
        "q90": np.asarray(q90, dtype=np.float64),
        "count": np.asarray(counts, dtype=np.int32),
    }


def _plot_quantile_band(
    stats: dict[str, np.ndarray],
    title: str,
    y_label: str,
    out_path: Path,
    dpi: int,
    color: str,
    add_zero_line: bool,
    y_min: float | None = None,
    y_max: float | None = None,
) -> bool:
    x = stats["x"]
    if x.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(
        x,
        stats["q10"],
        stats["q90"],
        alpha=0.28,
        color=color,
        edgecolor="none",
        linewidth=0.0,
        label="p10-p90 range",
    )
    # Keep only one center line to reduce visual clutter.
    ax.plot(x, stats["mean"], lw=2.1, color="#6a6a6a", label="mean")
    if add_zero_line:
        ax.axhline(0.0, color="#444444", ls="--", lw=1.0, alpha=0.7)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_xlabel("token position (within segment)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.text(
        0.012,
        0.985,
        "Band: p10-p90, line: mean.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f8f8f8", "edgecolor": "#dddddd", "alpha": 0.92},
    )
    ax.grid(alpha=0.12)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_count_curve(x: np.ndarray, count: np.ndarray, out_path: Path, title: str, dpi: int) -> bool:
    if x.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(x, count, lw=1.6, color="#8c564b")
    ax.set_xlabel("token position (within segment)")
    ax.set_ylabel("sample count")
    ax.set_title(title)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_prefix_ness(prefix_logw_by_pos: dict[int, list[float]], out_path: Path, dpi: int, segment: str) -> bool:
    if not prefix_logw_by_pos:
        return False
    positions = sorted(prefix_logw_by_pos.keys())
    x = np.asarray([p + 1 for p in positions], dtype=np.int32)
    ness = []
    for p in positions:
        v = _ess_from_log_weights(prefix_logw_by_pos[p])
        ness.append(np.nan if v is None else float(v))
    y = np.asarray(ness, dtype=np.float64)
    if not np.isfinite(y).any():
        return False

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(x, y, lw=1.7, color="#9467bd")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("token position (within segment)")
    ax.set_ylabel("normalized ESS")
    ax.set_title(f"Prefix Normalized ESS by Token Position [{segment}]")
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_token_position_offpolicy(
    records: list[dict[str, Any]],
    segment: str,
    output_dir: Path,
    dpi: int,
    output_format: str,
    kept_only: bool,
    min_valid_tokens: int,
    max_show_tokens: int | None,
    y_min: float | None,
    y_max: float | None,
) -> list[Path]:
    delta_by_pos: dict[int, list[float]] = {}
    abs_delta_by_pos: dict[int, list[float]] = {}
    nll_teacher_by_pos: dict[int, list[float]] = {}
    rev_kl_proxy_by_pos: dict[int, list[float]] = {}
    teacher_win_by_pos: dict[int, list[float]] = {}
    prefix_logw_by_pos: dict[int, list[float]] = {}

    used_records = 0
    skipped_no_segment = 0
    skipped_too_short = 0
    for row in records:
        student, teacher = _aligned_logprobs(row)
        if student.size == 0 or teacher.size == 0:
            continue

        ts = row.get("token_stats") or {}
        thinking_start = _safe_int(ts.get("thinking_token_start"))
        answer_start = _safe_int(ts.get("answer_token_start"))
        ok, start, end = _segment_bounds(student.size, thinking_start, answer_start, segment)
        if not ok or start is None or end is None or end <= start:
            skipped_no_segment += 1
            continue

        s = student[start:end]
        t = teacher[start:end]
        valid_n = int(min(s.size, t.size))
        if kept_only and valid_n < min_valid_tokens:
            skipped_too_short += 1
            continue
        if valid_n <= 0:
            continue

        if max_show_tokens is not None and max_show_tokens > 0:
            valid_n = min(valid_n, max_show_tokens)
            s = s[:valid_n]
            t = t[:valid_n]

        delta = t - s
        _append_position_values(delta_by_pos, delta)
        _append_position_values(abs_delta_by_pos, np.abs(delta))
        _append_position_values(nll_teacher_by_pos, -t)
        _append_position_values(rev_kl_proxy_by_pos, s - t)
        _append_position_values(teacher_win_by_pos, (delta > 0.0).astype(np.float64))

        prefix = np.cumsum(delta)
        _append_position_values(prefix_logw_by_pos, prefix)

        used_records += 1

    stats_delta = _stats_from_pos_values(delta_by_pos)
    stats_abs_delta = _stats_from_pos_values(abs_delta_by_pos)
    stats_nll = _stats_from_pos_values(nll_teacher_by_pos)
    stats_rev = _stats_from_pos_values(rev_kl_proxy_by_pos)
    stats_win = _stats_from_pos_values(teacher_win_by_pos)

    generated: list[Path] = []
    suffix = "kept_only" if kept_only else "all_records"
    ext = output_format.lower().lstrip(".")

    out = output_dir / f"tokenpos_{segment}_{suffix}_delta_quantiles.{ext}"
    if _plot_quantile_band(
        stats_delta,
        title=f"Teacher - Student by Token Position [{segment}] (n_records={used_records})",
        y_label="teacher logprob - student logprob",
        out_path=out,
        dpi=dpi,
        color="#1f77b4",
        add_zero_line=True,
        y_min=y_min,
        y_max=y_max,
    ):
        generated.append(out)

    out = output_dir / f"tokenpos_{segment}_{suffix}_abs_delta_quantiles.{ext}"
    if _plot_quantile_band(
        stats_abs_delta,
        title=f"|Teacher - Student| by Token Position [{segment}] (n_records={used_records})",
        y_label="abs(teacher - student)",
        out_path=out,
        dpi=dpi,
        color="#ff7f0e",
        add_zero_line=False,
        y_min=y_min,
        y_max=y_max,
    ):
        generated.append(out)

    out = output_dir / f"tokenpos_{segment}_{suffix}_teacher_nll_quantiles.{ext}"
    if _plot_quantile_band(
        stats_nll,
        title=f"Teacher NLL on Student Tokens by Position [{segment}] (n_records={used_records})",
        y_label="- teacher logprob",
        out_path=out,
        dpi=dpi,
        color="#2ca02c",
        add_zero_line=False,
        y_min=y_min,
        y_max=y_max,
    ):
        generated.append(out)

    out = output_dir / f"tokenpos_{segment}_{suffix}_reverse_kl_proxy_quantiles.{ext}"
    if _plot_quantile_band(
        stats_rev,
        title=f"Sampled Reverse-KL Proxy (Student-Teacher) by Position [{segment}] (n_records={used_records})",
        y_label="student logprob - teacher logprob",
        out_path=out,
        dpi=dpi,
        color="#17becf",
        add_zero_line=True,
        y_min=y_min,
        y_max=y_max,
    ):
        generated.append(out)

    out = output_dir / f"tokenpos_{segment}_{suffix}_teacher_win_rate_quantiles.{ext}"
    if _plot_quantile_band(
        stats_win,
        title=f"Teacher Win Rate by Token Position [{segment}] (n_records={used_records})",
        y_label="P(teacher > student)",
        out_path=out,
        dpi=dpi,
        color="#d62728",
        add_zero_line=False,
        y_min=0.0 if y_min is None else y_min,
        y_max=1.02 if y_max is None else y_max,
    ):
        generated.append(out)

    out = output_dir / f"tokenpos_{segment}_{suffix}_prefix_normalized_ess.{ext}"
    if _plot_prefix_ness(prefix_logw_by_pos, out_path=out, dpi=dpi, segment=segment):
        generated.append(out)

    out = output_dir / f"tokenpos_{segment}_{suffix}_sample_count_by_position.{ext}"
    if _plot_count_curve(
        stats_delta["x"],
        stats_delta["count"],
        out_path=out,
        dpi=dpi,
        title=f"Coverage by Token Position [{segment}] (n_records={used_records})",
    ):
        generated.append(out)

    print(f"records_total={len(records)} records_used={used_records}")
    print(f"skipped_no_segment={skipped_no_segment} skipped_too_short={skipped_too_short}")
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL output path from eval_student_teacher_inference.py",
    )
    parser.add_argument("--output-dir", default="./opsd_offpolicy_plots", help="Output plot directory")
    parser.add_argument(
        "--segment",
        default="all",
        choices=sorted(ALLOWED_SEGMENTS),
        help="Token segment to analyze",
    )
    parser.add_argument(
        "--kept-only",
        action="store_true",
        help="If set, only keep records with valid segment token count >= --min-valid-tokens",
    )
    parser.add_argument(
        "--min-valid-tokens",
        type=int,
        default=32,
        help="Minimum valid tokens per selected segment when --kept-only is set",
    )
    parser.add_argument(
        "--max-show-tokens",
        type=int,
        default=None,
        help="Optional max number of token positions to show (within segment)",
    )
    parser.add_argument(
        "--max-position",
        dest="max_show_tokens",
        type=int,
        default=None,
        help="Backward-compatible alias of --max-show-tokens",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png"],
        help="Output figure format",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional y-axis lower bound for quantile plots",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional y-axis upper bound for quantile plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_records(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = plot_token_position_offpolicy(
        records=rows,
        segment=args.segment,
        output_dir=out_dir,
        dpi=args.dpi,
        output_format=args.format,
        kept_only=args.kept_only,
        min_valid_tokens=max(1, int(args.min_valid_tokens)),
        max_show_tokens=args.max_show_tokens,
        y_min=args.y_min,
        y_max=args.y_max,
    )

    print(f"Loaded records: {len(rows)} from {args.input}")
    print(f"Segment: {args.segment}")
    print(f"Generated {len(generated)} figures under: {out_dir}")
    for p in generated:
        print(f" - {p}")


if __name__ == "__main__":
    main()
