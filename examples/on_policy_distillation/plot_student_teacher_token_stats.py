#!/usr/bin/env python3
"""
Plot token-level teacher-student log-prob deltas from
eval_student_teacher_inference.py output.
"""


'''

python examples/on_policy_distillation/plot_student_teacher_token_stats.py \
  --input /data/siqizhu4/opsd_slime/eval_openthoughts_student_teacher_inference_all_nothinking_b256.jsonl \
  --output-dir ./student_teacher_plots_new \
  --y-min -0.3 \
  --y-max 0.1 \
  --smooth-window 31


  

python examples/on_policy_distillation/plot_student_teacher_token_stats.py \
  --input ./eval_openthoughts_student8b_teacher8b_inference_all_nothinking_b256.jsonl \
  --output-dir ./student8b_teacher_plots \
  --y-min -0.2 \
  --y-max 0.1 \
  --smooth-window 51 \
  --raw-alpha 0.05 \
  --min-count-per-pos 16 \
  --weighted-smooth

python examples/on_policy_distillation/plot_student_teacher_token_stats.py \
  --input /data/siqizhu4/opsd_slime/eval_openthoughts_student_teacher_inference_all_nothinking_b256.jsonl \
  --output-dir ./student_teacher_plots \
  --y-min -0.3 \
  --y-max 0.2 \
  --smooth-window 51 \
  --raw-alpha 0.05 \
  --min-count-per-pos 16 \
  --weighted-smooth



python examples/on_policy_distillation/plot_student_teacher_token_stats.py \
  --input eval_openthoughts_student1.7b_teacher8b_inference_all_nothinking_entropy_b128.jsonl \
  --output-dir ./student1.7b_teacher_plots_nothinking_entropy \
  --y-min -0.8 \
  --y-max 0.4 \
  --smooth-window 51 \
  --raw-alpha 0.05 \
  --min-count-per-pos 16 \
  --weighted-smooth \
  --entropy-bin-count 58 \
  --entropy-bin-strategy equal_width

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


def _apply_y_limits(ax, y_min: float | None, y_max: float | None):
    if y_min is None and y_max is None:
        return
    ax.set_ylim(bottom=y_min, top=y_max)


def _smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    if y.size == 0:
        return y
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w <= 1 or y.size < 3:
        return y.copy()
    w = min(w, y.size if y.size % 2 == 1 else max(1, y.size - 1))
    if w <= 1:
        return y.copy()
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(y, kernel, mode="same")


def _smooth_series_weighted(y: np.ndarray, weights: np.ndarray, window: int) -> np.ndarray:
    if y.size == 0:
        return y
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w <= 1 or y.size < 3:
        return y.copy()
    w = min(w, y.size if y.size % 2 == 1 else max(1, y.size - 1))
    if w <= 1:
        return y.copy()
    kernel = np.ones(w, dtype=np.float64)
    # Weighted moving average: sum(y*w) / sum(w)
    num = np.convolve(y * weights, kernel, mode="same")
    den = np.convolve(weights, kernel, mode="same")
    den = np.maximum(den, 1e-12)
    return num / den


def _short_model_name(model: str | None) -> str:
    if not model:
        return "unknown-model"
    m = str(model).rstrip("/")
    return m.split("/")[-1] or m


def _teacher_setting_label(mode: str | None, model: str | None, api_base: str | None) -> str:
    # Keep legend concise and tied to real JSON settings.
    mode_text_raw = (mode or "unknown-mode").strip()
    model_text = _short_model_name(model)
    return f"PI: {mode_text_raw} | teacher: {model_text}"


def _as_float_array_allow_nan(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float64)


def _sanitize_mode_for_filename(mode: str | None) -> str:
    text = str(mode or "unknown").strip().lower()
    chars = [(c if c.isalnum() else "_") for c in text]
    slug = "".join(chars)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "unknown"


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x0 * y0) / denom)


def _append_entropy_delta_chunk(
    bucket: dict,
    student_entropies: np.ndarray,
    student_logprobs: np.ndarray,
    teacher_logprobs: np.ndarray,
):
    n = min(student_entropies.size, student_logprobs.size, teacher_logprobs.size)
    if n <= 0:
        return
    ent = student_entropies[:n]
    delta = teacher_logprobs[:n] - student_logprobs[:n]
    mask = np.isfinite(ent) & np.isfinite(delta)
    if not np.any(mask):
        return
    bucket["entropy_chunks"].append(ent[mask])
    bucket["delta_chunks"].append(delta[mask])


def _collect_entropy_delta_sources(records: list[dict]) -> list[dict]:
    buckets: dict[str, dict] = {}
    order: list[str] = []

    def _ensure_bucket(key: str, file_suffix: str, title: str) -> dict:
        if key not in buckets:
            buckets[key] = {
                "file_suffix": file_suffix,
                "title": title,
                "entropy_chunks": [],
                "delta_chunks": [],
            }
            order.append(key)
        return buckets[key]

    for record in records:
        token_stats = record.get("token_stats") or {}
        student_logprobs = _as_float_array_allow_nan(token_stats.get("student_logprobs"))
        student_entropies = _as_float_array_allow_nan(token_stats.get("student_entropies"))
        if student_logprobs.size == 0 or student_entropies.size == 0:
            continue

        primary_teacher_logprobs = _as_float_array_allow_nan(token_stats.get("teacher_logprobs"))
        primary_bucket = _ensure_bucket(
            key="primary",
            file_suffix="primary",
            title="primary teacher (token_stats.teacher_logprobs)",
        )
        _append_entropy_delta_chunk(primary_bucket, student_entropies, student_logprobs, primary_teacher_logprobs)

        teachers = token_stats.get("teachers") or []
        if not isinstance(teachers, list):
            continue
        for idx, tinfo in enumerate(teachers):
            if not isinstance(tinfo, dict):
                continue
            mode = str(tinfo.get("mode") or "unknown")
            mode_slug = _sanitize_mode_for_filename(mode)
            key = f"teacher_{idx}_{mode_slug}"
            bucket = _ensure_bucket(
                key=key,
                file_suffix=key,
                title=f"teacher[{idx}] mode={mode}",
            )
            teacher_logprobs = _as_float_array_allow_nan(tinfo.get("logprobs"))
            _append_entropy_delta_chunk(bucket, student_entropies, student_logprobs, teacher_logprobs)

    sources: list[dict] = []
    for key in order:
        bucket = buckets[key]
        if not bucket["entropy_chunks"]:
            continue
        entropy = np.concatenate(bucket["entropy_chunks"])
        delta = np.concatenate(bucket["delta_chunks"])
        if entropy.size == 0 or delta.size == 0:
            continue
        sources.append(
            {
                "file_suffix": bucket["file_suffix"],
                "title": bucket["title"],
                "entropy": entropy,
                "delta": delta,
            }
        )
    return sources


def _choose_entropy_bin_count(n_points: int) -> int:
    if n_points <= 0:
        return 10
    # Keep bins dense enough for trend smoothness while preventing sparse/noisy bins.
    return int(min(60, max(10, round(np.sqrt(float(n_points)) / 8.0))))


def _choose_entropy_mean_smooth_window(n_bins: int) -> int:
    if n_bins <= 4:
        return 1
    # Auto smooth with ~12% of bins so high-bin setups are less jittery.
    w = int(round(0.12 * float(n_bins)))
    w = max(5, min(41, w))
    if w % 2 == 0:
        w += 1
    return w


def _legend_handles_labels_dedup(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        text = str(label).strip()
        if not text or text in unique:
            continue
        unique[text] = handle
    return list(unique.values()), list(unique.keys())


def _build_equal_frequency_bin_stats(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    edges = np.percentile(x, np.linspace(0.0, 100.0, n_bins + 1))
    centers: list[float] = []
    means: list[float] = []
    medians: list[float] = []
    q25s: list[float] = []
    q75s: list[float] = []
    counts: list[float] = []

    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < n_bins - 1:
            mask = (x >= lo) & (x < hi)
        else:
            mask = (x >= lo) & (x <= hi)
        if not np.any(mask):
            continue
        y_bin = y[mask]
        x_bin = x[mask]
        centers.append(float(np.median(x_bin)))
        means.append(float(np.mean(y_bin)))
        medians.append(float(np.median(y_bin)))
        q25s.append(float(np.percentile(y_bin, 25)))
        q75s.append(float(np.percentile(y_bin, 75)))
        counts.append(float(y_bin.size))

    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(medians, dtype=np.float64),
        np.asarray(q25s, dtype=np.float64),
        np.asarray(q75s, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def _build_equal_width_bin_stats(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    # Degenerate case: all entropy values are identical.
    if x_max <= x_min + 1e-12:
        y_bin = y[np.isfinite(y)]
        if y_bin.size == 0:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
        center = np.asarray([x_min], dtype=np.float64)
        mean = np.asarray([float(np.mean(y_bin))], dtype=np.float64)
        median = np.asarray([float(np.median(y_bin))], dtype=np.float64)
        q25 = np.asarray([float(np.percentile(y_bin, 25))], dtype=np.float64)
        q75 = np.asarray([float(np.percentile(y_bin, 75))], dtype=np.float64)
        count = np.asarray([float(y_bin.size)], dtype=np.float64)
        return center, mean, median, q25, q75, count

    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers: list[float] = []
    means: list[float] = []
    medians: list[float] = []
    q25s: list[float] = []
    q75s: list[float] = []
    counts: list[float] = []

    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < n_bins - 1:
            mask = (x >= lo) & (x < hi)
        else:
            mask = (x >= lo) & (x <= hi)
        if not np.any(mask):
            continue
        y_bin = y[mask]
        x_bin = x[mask]
        centers.append(float(np.median(x_bin)))
        means.append(float(np.mean(y_bin)))
        medians.append(float(np.median(y_bin)))
        q25s.append(float(np.percentile(y_bin, 25)))
        q75s.append(float(np.percentile(y_bin, 75)))
        counts.append(float(y_bin.size))

    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(medians, dtype=np.float64),
        np.asarray(q25s, dtype=np.float64),
        np.asarray(q75s, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def _plot_entropy_delta_scatter_with_mean_line(
    entropy: np.ndarray,
    delta: np.ndarray,
    title: str,
    out_file: Path,
    dpi: int,
    y_min: float | None,
    y_max: float | None,
    entropy_bin_count: int | None,
    entropy_bin_strategy: str,
):
    corr = _pearson_corr(entropy, delta)
    n_bins = int(entropy_bin_count) if entropy_bin_count is not None else _choose_entropy_bin_count(int(entropy.size))
    if entropy_bin_strategy == "equal_width":
        centers, means, _medians, _q25s, _q75s, counts = _build_equal_width_bin_stats(entropy, delta, n_bins=n_bins)
    else:
        centers, means, _medians, _q25s, _q75s, counts = _build_equal_frequency_bin_stats(entropy, delta, n_bins=n_bins)
    if centers.size == 0:
        return
    smooth_window = _choose_entropy_mean_smooth_window(int(centers.size))
    means_smoothed = _smooth_series_weighted(means, counts, window=smooth_window)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.scatter(
        entropy,
        delta,
        s=4,
        alpha=0.08,
        color="#1f77b4",
        edgecolors="none",
        rasterized=True,
        # label=f"all points (n={entropy.size:,})",
    )
    ax.plot(
        centers,
        means,
        lw=1.0,
        color="#d62728",
        alpha=0.30,
        label=None,
        zorder=2,
    )
    ax.plot(
        centers,
        means_smoothed,
        lw=2.0,
        color="#d62728",
        # label=f"binned mean (smoothed, bins={centers.size}, {entropy_bin_strategy})",
        zorder=3,
    )

    ax.axhline(0.0, color="#444444", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("student token entropy")
    ax.set_ylabel("teacher logprob - student logprob")
    ax.set_title(f"Entropy vs teacher-student delta ({title})")
    _apply_y_limits(ax, y_min=y_min, y_max=y_max)
    ax.grid(alpha=0.20)
    handles, labels = _legend_handles_labels_dedup(ax)
    ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        framealpha=0.90,
        title=f"n_tokens={entropy.size:,}, r={corr:.4f}",
    )
    fig.tight_layout()
    try:
        fig.savefig(out_file, dpi=dpi)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot write to {out_file}. Check --output-dir permissions or use a new directory."
        ) from e
    plt.close(fig)


def plot_entropy_vs_teacher_minus_student(
    records: list[dict],
    output_dir: Path,
    dpi: int,
    y_min: float | None,
    y_max: float | None,
    entropy_bin_count: int | None,
    entropy_bin_strategy: str,
):
    sources = _collect_entropy_delta_sources(records)
    if not sources:
        print(
            "No entropy-vs-delta figure generated "
            "(need token_stats.student_entropies + aligned student/teacher logprobs)."
        )
        return

    generated = 0
    for src in sources:
        suffix = src["file_suffix"]
        title = src["title"]
        entropy = src["entropy"]
        delta = src["delta"]

        out_file = output_dir / f"summary_entropy_vs_teacher_minus_student_{suffix}_scatter.png"
        _plot_entropy_delta_scatter_with_mean_line(
            entropy=entropy,
            delta=delta,
            title=title,
            out_file=out_file,
            dpi=dpi,
            y_min=y_min,
            y_max=y_max,
            entropy_bin_count=entropy_bin_count,
            entropy_bin_strategy=entropy_bin_strategy,
        )
        generated += 1

    print(f"Generated {generated} entropy-vs-delta figure(s) in {output_dir}")


def plot_teacher_group_smoothed_lines(
    records: list[dict],
    output_dir: Path,
    dpi: int,
    y_min: float | None,
    y_max: float | None,
    smooth_window: int,
    raw_alpha: float,
    min_count_per_pos: int,
    weighted_smooth: bool,
):
    group_pos_values: dict[int, dict[int, list[float]]] = {}
    group_labels: dict[int, str] = {}

    for record in records:
        token_stats = record.get("token_stats") or {}
        student_lp = _as_float_array(token_stats.get("student_logprobs"))
        teachers = token_stats.get("teachers") or []
        if student_lp.size == 0 or not isinstance(teachers, list):
            continue

        for g_idx, tinfo in enumerate(teachers):
            if not isinstance(tinfo, dict):
                continue
            if g_idx not in group_labels:
                group_labels[g_idx] = _teacher_setting_label(
                    tinfo.get("mode"),
                    tinfo.get("model"),
                    tinfo.get("api_base"),
                )
            teacher_lp = _as_float_array(tinfo.get("logprobs"))
            if teacher_lp.size == 0:
                continue
            n = min(student_lp.size, teacher_lp.size)
            if n <= 0:
                continue
            delta = teacher_lp[:n] - student_lp[:n]
            for pos, v in enumerate(delta):
                if np.isfinite(v):
                    group_pos_values.setdefault(g_idx, {}).setdefault(pos, []).append(float(v))

    if not group_pos_values:
        print("No teacher-group smoothed line figure generated (missing token_stats.teachers[*].logprobs).")
        return

    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(14, 6))

    # Draw groups sorted by global mean so similar curves are less likely to fully cover each other.
    sortable = []
    for g_idx, pos_map in group_pos_values.items():
        vals = [v for pos in pos_map.values() for v in pos]
        gm = float(np.mean(vals)) if vals else 0.0
        sortable.append((gm, g_idx))
    sortable.sort(key=lambda x: x[0])

    for _, g_idx in sortable:
        pos_map = group_pos_values[g_idx]
        positions = sorted(pos_map.keys())
        x_all = np.asarray([p + 1 for p in positions], dtype=np.int64)
        raw_mean_all = np.asarray([float(np.mean(pos_map[p])) for p in positions], dtype=np.float64)
        counts_all = np.asarray([len(pos_map[p]) for p in positions], dtype=np.float64)

        # Tail positions usually have very low sample count; drop them to avoid misleading spikes.
        if min_count_per_pos > 1:
            keep = counts_all >= float(min_count_per_pos)
        else:
            keep = np.ones_like(counts_all, dtype=bool)
        x = x_all[keep]
        raw_mean = raw_mean_all[keep]
        counts = counts_all[keep]
        if x.size == 0:
            continue

        if weighted_smooth:
            smoothed = _smooth_series_weighted(raw_mean, counts, window=smooth_window)
        else:
            smoothed = _smooth_series(raw_mean, window=smooth_window)

        color = colors[g_idx % len(colors)]
        ax.plot(
            x,
            raw_mean,
            color=color,
            lw=1.0,
            alpha=max(0.0, min(1.0, raw_alpha)),
            solid_capstyle="round",
            label=None,
            zorder=1,
        )
        ax.plot(
            x,
            smoothed,
            color=color,
            lw=2.0,
            alpha=0.95,
            solid_capstyle="round",
            label=group_labels.get(g_idx, f"group {g_idx + 1}"),
            zorder=3,
        )

    ax.axhline(0.0, color="#444444", ls="--", lw=1.0, alpha=0.7, zorder=2)
    ax.set_xlabel("token position")
    ax.set_ylabel("teacher logprob - student logprob")
    ax.set_title(
        "teacher logprob - student logprob "
        f"(openthoughts, student: qwen3-8b enable_thinking=false)"
    )
    _apply_y_limits(ax, y_min=y_min, y_max=y_max)
    ax.grid(alpha=0.22)

    # Compact and clear legend: one entry per group.
    ax.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        framealpha=0.90,
        title="Teacher Setting",
    )
    fig.tight_layout()
    out_file = output_dir / "summary_teacher_groups_smoothed_line.png"
    try:
        fig.savefig(out_file, dpi=dpi)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot write to {out_file}. Check --output-dir permissions or use a new directory."
        ) from e
    plt.close(fig)
    print(f"Generated teacher-group smoothed line figure in {out_file}")


def plot_per_request(
    records: list[dict],
    output_dir: Path,
    max_requests: int | None,
    dpi: int,
    y_min: float | None,
    y_max: float | None,
):
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
        _apply_y_limits(ax, y_min=y_min, y_max=y_max)
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


def plot_overall_quantiles(
    records: list[dict],
    output_dir: Path,
    dpi: int,
    y_min: float | None,
    y_max: float | None,
):
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
        _apply_y_limits(ax, y_min=y_min, y_max=y_max)
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

        # Dedicated line plot (mean by token position).
        fig, ax = plt.subplots(figsize=(14, 5.0))
        ax.plot(x, means, lw=1.5, color="#1f77b4", label="mean(teacher - student)")
        ax.axhline(0.0, color="#444444", ls="--", lw=1.0, alpha=0.7)
        ax.set_xlabel("token position")
        ax.set_ylabel("teacher logprob - student logprob")
        ax.set_title(
            f"Teacher - Student mean delta by token position "
            f"(n_records={n_tms_records}, max_pos={max(x)})"
        )
        _apply_y_limits(ax, y_min=y_min, y_max=y_max)
        ax.grid(alpha=0.22)
        ax.legend(loc="best")
        fig.tight_layout()
        out_file = output_dir / "summary_teacher_minus_student_line_mean.png"
        try:
            fig.savefig(out_file, dpi=dpi)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot write to {out_file}. Check --output-dir permissions or use a new directory."
            ) from e
        plt.close(fig)
        generated += 1

        # Dedicated scatter plot (all finite deltas at each token position).
        scatter_x: list[int] = []
        scatter_y: list[float] = []
        for p in positions:
            vals = pos_values[p]
            scatter_x.extend([p + 1] * len(vals))
            scatter_y.extend(vals)

        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.scatter(
            scatter_x,
            scatter_y,
            s=4,
            alpha=0.08,
            color="#1f77b4",
            edgecolors="none",
            # label=f"all points (n={len(scatter_y)})",
        )
        ax.axhline(0.0, color="#444444", ls="--", lw=1.0, alpha=0.7)
        ax.set_xlabel("token position")
        ax.set_ylabel("teacher logprob - student logprob")
        ax.set_title(
            f"Teacher − Student scatter by token position "
            f"(n_records={n_tms_records}, max_pos={max(x)})"
        )
        _apply_y_limits(ax, y_min=y_min, y_max=y_max)
        ax.grid(alpha=0.22)
        ax.legend(loc="best")
        fig.tight_layout()
        out_file = output_dir / "summary_teacher_minus_student_scatter.png"
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
        _apply_y_limits(ax, y_min=y_min, y_max=y_max)
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
    parser.add_argument("--y-min", type=float, default=None, help="Optional fixed y-axis minimum for all plots.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional fixed y-axis maximum for all plots.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=31,
        help="Moving-average window size for per-teacher-group smoothed line plot.",
    )
    parser.add_argument(
        "--raw-alpha",
        type=float,
        default=0.08,
        help="Alpha of background raw lines in per-teacher-group plot (0~1).",
    )
    parser.add_argument(
        "--min-count-per-pos",
        type=int,
        default=8,
        help="Only draw token positions with at least this many samples in per-teacher-group plot.",
    )
    parser.add_argument(
        "--weighted-smooth",
        action="store_true",
        default=True,
        help="Use sample-count weighted smoothing in per-teacher-group plot.",
    )
    parser.add_argument(
        "--no-weighted-smooth",
        dest="weighted_smooth",
        action="store_false",
        help="Disable sample-count weighted smoothing.",
    )
    parser.add_argument(
        "--entropy-bin-count",
        type=int,
        default=None,
        help="Optional fixed bin count for entropy mean line (default: auto).",
    )
    parser.add_argument(
        "--entropy-bin-strategy",
        choices=["equal_freq", "equal_width"],
        default="equal_freq",
        help="Binning strategy for entropy mean line.",
    )
    args = parser.parse_args()

    if args.y_min is not None and args.y_max is not None and args.y_min >= args.y_max:
        raise ValueError(f"--y-min must be smaller than --y-max, got y_min={args.y_min}, y_max={args.y_max}")
    if args.entropy_bin_count is not None and args.entropy_bin_count < 1:
        raise ValueError(f"--entropy-bin-count must be >= 1, got {args.entropy_bin_count}")

    records = load_records(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_per_request(
        records,
        output_dir=output_dir,
        max_requests=args.max_requests,
        dpi=args.dpi,
        y_min=args.y_min,
        y_max=args.y_max,
    )
    plot_overall_quantiles(
        records,
        output_dir=output_dir,
        dpi=args.dpi,
        y_min=args.y_min,
        y_max=args.y_max,
    )
    plot_teacher_group_smoothed_lines(
        records,
        output_dir=output_dir,
        dpi=args.dpi,
        y_min=args.y_min,
        y_max=args.y_max,
        smooth_window=args.smooth_window,
        raw_alpha=args.raw_alpha,
        min_count_per_pos=args.min_count_per_pos,
        weighted_smooth=args.weighted_smooth,
    )
    plot_entropy_vs_teacher_minus_student(
        records,
        output_dir=output_dir,
        dpi=args.dpi,
        y_min=args.y_min,
        y_max=args.y_max,
        entropy_bin_count=args.entropy_bin_count,
        entropy_bin_strategy=args.entropy_bin_strategy,
    )

    print(f"Loaded {len(records)} records from {args.input}")
    print(f"Saved figures under {output_dir}")


if __name__ == "__main__":
    main()
