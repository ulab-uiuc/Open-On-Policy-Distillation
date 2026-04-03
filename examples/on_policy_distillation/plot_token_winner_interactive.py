#!/usr/bin/env python3
"""
Generate an interactive HTML heatmap showing per-token teacher vs student logprob dominance.

Red   = teacher logprob > student logprob
Green = student logprob > teacher logprob
Color intensity = magnitude of the difference

Each row = one request. Reward (0/1) is shown in the row label.
Click any row in the heatmap to open a detailed per-token view below.

Usage:
  pip install numpy
  python examples/on_policy_distillation/plot_token_winner_interactive.py \
    --input ./eval_math500_student_teacher_inference.jsonl \
    --output ./token_winner_interactive.html \
    --tokenizer Qwen/Qwen3-1.7B


python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input ./eval_math500_student_teacher_inference.jsonl \
  --output ./token_winner_interactive.html \
  --tokenizer Qwen/Qwen3-1.7B \
  --show-last-k-tokens 120


python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input ./eval_math500_student_teacher_inference_s1.7t1.7b_answeronly_disablethinking_b512.jsonl \
  --output ./token_winner_interactive_s1.7t1.7b_answeronly_disablethinking_b512.html \
  --tokenizer Qwen/Qwen3-1.7B \
  --show-last-k-tokens 32768 \
  --max-requests 8

python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input ./eval_math500_student_teacher_inference_s1.7t8b_same_as_student_b512.jsonl \
  --output ./token_winner_interactive_s1.7t8b_same_as_student_b512.html \
  --tokenizer Qwen/Qwen3-1.7B \
  --show-last-k-tokens 32768


python examples/on_policy_distillation/eval_student_teacher_inference.py \
  --model /root/checkpoints_siqi/Qwen3-1.7B \
  --teacher-model /root/checkpoints_siqi/Qwen3-8B \
  --student-api-base http://127.0.0.1:30000/v1 \
  --teacher-api-base http://127.0.0.1:30001/v1 \
  --student-api-key EMPTY \
  --teacher-api-key EMPTY \
  --dataset /root/math/data/train_dapo.jsonl \
  --teacher-info-mode same_as_student \
  --max-new-tokens 32768 \
  --n-samples 512 \
  --concurrency 256 \
  --seed 42 \
  --score-concurrency 1 \
  --score-chunk-tokens 256 \
  --output ./eval_math500_student_teacher_inference_s1.7t8b_same_as_student_b512.jsonl \
  --student-enable-thinking false


python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input ./eval_math500_student_teacher_inference_s1.7t1.7b_answeronly.jsonl \
  --output ./token_winner_interactive.html \
  --tokenizer Qwen/Qwen3-1.7B \
  --show-last-k-tokens 120


  ./eval_math500_student_teacher_inference_s1.7t8b_answeronly_disablethinking_b512.jsonl

python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input ./eval_math500_student_teacher_inference_s1.7t8b_same_as_student_b512.jsonl \
  --output ./token_winner_interactive_s1.7t8b_same_as_student_b512.html \
  --tokenizer Qwen/Qwen3-1.7B \
  --show-last-k-tokens 64

python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input ./eval_math500_student_teacher_inference_s1.7t1.7b_answeronly.jsonl \
  --output ./token_winner_interactive_s1.7t1.7b_answeronly.html \
  --tokenizer Qwen/Qwen3-1.7B \
  --show-first-k-tokens 64


python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input ./eval_math500_student_teacher_inference_s1.7t8b_answeronly.jsonl \
  --output ./token_winner_interactive_s1.7t8b_answeronly.html \
  --tokenizer Qwen/Qwen3-1.7B \
  --show-first-k-tokens 64

  
python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input eval_math500_student_teacher_inference_s1.7t1.7b_answeronly_b512.jsonl \
  --output ./token_winner_interactive_s1.7t1.7b_answeronly_b512.html \
  --tokenizer Qwen/Qwen3-1.7B \
  --show-first-k-tokens 64


python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input ./eval_math500_student_teacher_inference_s1.7t1.7b_answeronly_v2.jsonl \
  --output ./token_winner_interactive_s1.7t1.7b_answeronly_v2.html \
  --tokenizer Qwen/Qwen3-1.7B \
  --show-first-k-tokens 64

python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input ./eval_math500_student_teacher_inference_s1.7t1.7b_answeronly.jsonl \
  --output ./token_winner_interactive_s1.7t1.7b_answeronly.html \
  --tokenizer Qwen/Qwen3-1.7B \
  --show-first-k-tokens 64

  
python examples/on_policy_distillation/plot_token_winner_interactive.py \
  --input ./eval_math500_student_teacher_inference_s8t8b_answeronly.jsonl \
  --output ./token_winner_interactive_s8t8b_answeronly.html \
  --tokenizer Qwen/Qwen3-8B \
  --show-first-k-tokens 64

"""

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"
_PLOTLY_CACHE = Path.home() / ".cache" / "plotly-2.27.0.min.js"


def _get_plotly_js() -> str | None:
    """
    Return Plotly.js source string (for inline embedding).
    Priority:
      1. plotly Python package (no network needed)
      2. local cache at ~/.cache/plotly-2.27.0.min.js
      3. download from CDN
    Returns None on failure (caller will use <script src=CDN> as last resort).
    """
    # 1. Try plotly Python package (bundled JS, no network required)
    try:
        import plotly.offline as _plo
        src = _plo.get_plotlyjs()
        if src:
            print("Using Plotly.js from installed plotly package.")
            return src
    except Exception:
        pass

    # 2. Local cache
    if _PLOTLY_CACHE.exists():
        print(f"Using cached Plotly.js from {_PLOTLY_CACHE}.")
        return _PLOTLY_CACHE.read_text(encoding="utf-8")

    # 3. Download from CDN
    print(f"Downloading Plotly.js from {_PLOTLY_CDN} (will cache to {_PLOTLY_CACHE}) ...")
    try:
        with urllib.request.urlopen(_PLOTLY_CDN, timeout=30) as resp:
            src = resp.read().decode("utf-8")
        _PLOTLY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _PLOTLY_CACHE.write_text(src, encoding="utf-8")
        print("Download complete.")
        return src
    except Exception as e:
        print(
            f"\nWARNING: Could not obtain Plotly.js ({e}).\n"
            "The HTML will try to load it from CDN — this requires internet access in the browser.\n"
            "To fix offline: run  pip install plotly  and re-run this script.\n"
        )
        return None


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def as_float_array(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float64)
    return np.asarray([float(v) for v in values], dtype=np.float64)


def _normalize_token_list(values, n: int) -> list[str]:
    if not isinstance(values, list):
        return []
    out = [str(v) for v in values[:n]]
    if len(out) < n:
        out.extend([""] * (n - len(out)))
    return out


def _decode_ids_to_text_tokens(tokenizer, token_ids: list[int], n: int) -> list[str]:
    """Decode token ids into human-readable token text pieces."""
    out = []
    for tid in token_ids[:n]:
        try:
            piece = tokenizer.decode(
                [int(tid)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            piece = str(tid)
        out.append(piece)
    if len(out) < n:
        out.extend([""] * (n - len(out)))
    return out


def _is_empty_token_text(token) -> bool:
    if token is None:
        return True
    return str(token) == ""


def _effective_token_count_without_trailing_empty(rec: dict, n_full: int) -> int:
    """
    If trailing response tokens are explicitly recorded as empty text,
    drop all of them.
    """
    if n_full <= 0:
        return 0

    ts = rec.get("token_stats") or {}
    text_keys = [
        "student_tokens",
        "student_token_texts",
        "token_texts",
        "tokens",
    ]
    for source in (ts, rec):
        for k in text_keys:
            values = source.get(k)
            if isinstance(values, list) and len(values) >= n_full:
                effective = n_full
                while effective > 0 and _is_empty_token_text(values[effective - 1]):
                    effective -= 1
                return effective

    return n_full


def _find_boxed_spans(text: str) -> list[tuple[int, int]]:
    """Return content spans of \\boxed{...} as [start, end) char offsets."""
    spans = []
    i = 0
    key = "\\boxed{"
    n = len(text)
    while i < n:
        j = text.find(key, i)
        if j < 0:
            break
        start = j + len(key)
        depth = 1
        k = start
        while k < n and depth > 0:
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        if depth == 0:
            # content is [start, k-1)
            spans.append((start, k - 1))
            i = k
        else:
            break
    return spans


def _token_indices_overlapping(offsets, spans: list[tuple[int, int]], max_len: int) -> list[int]:
    out = []
    for ti, (a, b) in enumerate(offsets[:max_len]):
        if b <= a:
            continue
        for s, e in spans:
            if a < e and b > s:
                out.append(ti)
                break
    return out


def _mean_or_none(values: list[float]):
    return float(np.mean(values)) if values else None


def compute_boxed_stats(records: list[dict], tokenizer=None) -> dict:
    """
    Compute reward-grouped boxed-region stats for teacher-student logprob delta.
    Returns JSON-serializable dict consumed by HTML.
    """
    if tokenizer is None:
        return {
            "available": False,
            "reason": "Tokenizer not provided. Pass --tokenizer to compute boxed token stats.",
        }

    grouped_all = {0: [], 1: []}
    grouped_last = {0: [], 1: []}
    sample_means_all = {0: [], 1: []}
    sample_means_last = {0: [], 1: []}
    sample_count_all = {0: 0, 1: 0}
    sample_count_last = {0: 0, 1: 0}

    missing_boxed = 0
    nan_tokens_all = 0
    nan_tokens_last = 0
    eligible_records = 0

    for rec in records:
        reward = rec.get("student_reward", rec.get("reward"))
        if reward is None:
            continue
        try:
            reward = int(round(float(reward)))
        except Exception:
            continue
        if reward not in (0, 1):
            continue

        ts = rec.get("token_stats") or {}
        s = as_float_array(ts.get("student_logprobs"))
        t = as_float_array(ts.get("teacher_logprobs"))
        n = _effective_token_count_without_trailing_empty(rec, min(s.size, t.size))
        if n <= 0:
            continue
        eligible_records += 1

        text = rec.get("student_response") or ""
        spans = _find_boxed_spans(text)
        if not spans:
            missing_boxed += 1
            continue

        try:
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
        except Exception:
            continue

        deltas = t[:n] - s[:n]

        idx_all = _token_indices_overlapping(offsets, spans, n)
        if idx_all:
            vals = [float(deltas[i]) for i in idx_all if np.isfinite(deltas[i])]
            nan_tokens_all += len(idx_all) - len(vals)
            if vals:
                grouped_all[reward].extend(vals)
                sample_means_all[reward].append(float(np.mean(vals)))
                sample_count_all[reward] += 1

        idx_last = _token_indices_overlapping(offsets, [spans[-1]], n)
        if idx_last:
            vals_last = [float(deltas[i]) for i in idx_last if np.isfinite(deltas[i])]
            nan_tokens_last += len(idx_last) - len(vals_last)
            if vals_last:
                grouped_last[reward].extend(vals_last)
                sample_means_last[reward].append(float(np.mean(vals_last)))
                sample_count_last[reward] += 1

    return {
        "available": True,
        "eligible_records": eligible_records,
        "missing_boxed_records": missing_boxed,
        "nan_tokens_all_boxed": nan_tokens_all,
        "nan_tokens_last_boxed": nan_tokens_last,
        "all_boxed": {
            "reward_1": {
                "token_count": len(grouped_all[1]),
                "sample_count": sample_count_all[1],
                "pooled_mean_delta": _mean_or_none(grouped_all[1]),
                "sample_mean_delta": _mean_or_none(sample_means_all[1]),
            },
            "reward_0": {
                "token_count": len(grouped_all[0]),
                "sample_count": sample_count_all[0],
                "pooled_mean_delta": _mean_or_none(grouped_all[0]),
                "sample_mean_delta": _mean_or_none(sample_means_all[0]),
            },
        },
        "last_boxed_only": {
            "reward_1": {
                "token_count": len(grouped_last[1]),
                "sample_count": sample_count_last[1],
                "pooled_mean_delta": _mean_or_none(grouped_last[1]),
                "sample_mean_delta": _mean_or_none(sample_means_last[1]),
            },
            "reward_0": {
                "token_count": len(grouped_last[0]),
                "sample_count": sample_count_last[0],
                "pooled_mean_delta": _mean_or_none(grouped_last[0]),
                "sample_mean_delta": _mean_or_none(sample_means_last[0]),
            },
        },
    }


def _extract_token_texts_from_record(rec: dict, start: int, end: int, tokenizer=None) -> list[str]:
    """
    Best-effort token text extraction for one request, aligned to n tokens.
    Priority:
      1) explicit token text arrays in record/token_stats
      2) token-id arrays decoded by tokenizer
      3) re-tokenize student_response by tokenizer
      4) whitespace-based fallback split of student_response
    """
    n = max(0, end - start)
    if n <= 0:
        return []
    ts = rec.get("token_stats") or {}

    # 1) Explicit token text arrays
    text_keys = [
        "student_tokens",
        "student_token_texts",
        "token_texts",
        "tokens",
    ]
    for source in (ts, rec):
        for k in text_keys:
            values = source.get(k)
            if isinstance(values, list):
                normalized = _normalize_token_list(values[start:end], n)
            else:
                normalized = []
            if normalized:
                return normalized

    # 2) Decode token IDs (if present and tokenizer available)
    if tokenizer is not None:
        id_keys = [
            "student_token_ids",
            "student_response_token_ids",
            "token_ids",
            "response_token_ids",
        ]
        for source in (ts, rec):
            for k in id_keys:
                values = source.get(k)
                if isinstance(values, list) and values:
                    try:
                        ids = [int(x) for x in values[start:end]]
                        normalized = _decode_ids_to_text_tokens(tokenizer, ids, n)
                        if normalized:
                            return normalized
                    except Exception:
                        pass

    # 3) Re-tokenize response text
    response_text = rec.get("student_response")
    if tokenizer is not None and isinstance(response_text, str) and response_text:
        try:
            ids = tokenizer.encode(response_text, add_special_tokens=False)
            normalized = _decode_ids_to_text_tokens(tokenizer, ids[start:end], n)
            if normalized:
                return normalized
        except Exception:
            pass

    # 4) Fallback: simple text chunks (not model tokenizer-accurate)
    if isinstance(response_text, str) and response_text:
        chunks = re.findall(r"\S+\s*|\s+", response_text)
        normalized = _normalize_token_list(chunks[start:end], n)
        if normalized:
            return normalized

    return [""] * n


def _remap_marker_to_window(pos_1idx, start_0idx: int, end_0idx: int):
    """Map original 1-indexed token position into window-relative 1-indexed position."""
    if pos_1idx is None:
        return None
    try:
        p = int(pos_1idx)
    except Exception:
        return None
    if p <= start_0idx or p > end_0idx:
        return None
    return p - start_0idx


def _extract_question_text(rec: dict) -> str:
    """Best-effort question text extraction for per-request detail header."""
    metadata = rec.get("metadata")
    if isinstance(metadata, dict):
        for key in ("question", "problem", "query", "instruction", "prompt"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                pieces = [str(v).strip() for v in value if str(v).strip()]
                if pieces:
                    return "\n".join(pieces)

    prompt_field = rec.get("prompt")
    if isinstance(prompt_field, list):
        for item in prompt_field:
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    elif isinstance(prompt_field, str) and prompt_field.strip():
        return prompt_field.strip()

    label = rec.get("label")
    if isinstance(label, str) and label.strip():
        return f"(No explicit question field) Label: {label.strip()}"

    return "(Question text not available in record)"


def build_heatmap_data(
    records: list[dict],
    tokenizer=None,
    n_bins: int = 0,
    first_k_tokens: int | None = None,
    last_k_tokens: int | None = None,
) -> tuple:  # n_bins unused, kept for CLI compat
    """
    Returns:
      heatmap_rows  : list of lists (n_records x max_tokens), exact per-token delta
      row_labels    : list of str
      rewards       : list of float/int/None
      x_positions   : list of int (1-indexed token positions)
      detail_data   : list of dicts for per-request detail view
      max_tokens    : int
      token_rows    : list of lists (n_records x max_tokens), per-position token text
      window_desc   : str, display description for selected token window
    """
    # First pass: find max tokens
    max_tokens = 0
    for rec in records:
        ts = rec.get("token_stats") or {}
        s = as_float_array(ts.get("student_logprobs"))
        t = as_float_array(ts.get("teacher_logprobs"))
        n_full = _effective_token_count_without_trailing_empty(rec, min(s.size, t.size))
        if first_k_tokens is not None:
            n = min(n_full, first_k_tokens)
        elif last_k_tokens is not None:
            n = min(n_full, last_k_tokens)
        else:
            n = n_full
        max_tokens = max(max_tokens, n)

    if max_tokens == 0:
        return [], [], [], [], [], 0, [], "all tokens"

    if first_k_tokens is not None:
        window_desc = f"first {first_k_tokens} tokens per request"
    elif last_k_tokens is not None:
        window_desc = f"last {last_k_tokens} tokens per request"
    else:
        window_desc = "all tokens"

    x_positions = list(range(1, max_tokens + 1))

    heatmap_rows = []
    row_labels = []
    rewards = []
    detail_data = []
    token_rows = []

    for rec in records:
        ts = rec.get("token_stats") or {}
        s = as_float_array(ts.get("student_logprobs"))
        t = as_float_array(ts.get("teacher_logprobs"))
        n_full = _effective_token_count_without_trailing_empty(rec, min(s.size, t.size))
        if n_full == 0:
            continue

        if first_k_tokens is not None:
            start = 0
            end = min(n_full, first_k_tokens)
        elif last_k_tokens is not None:
            end = n_full
            start = max(0, n_full - last_k_tokens)
        else:
            start = 0
            end = n_full
        n = end - start
        if n <= 0:
            continue

        delta = t[start:end] - s[start:end]
        reward = rec.get("student_reward", rec.get("reward"))
        idx = rec.get("index", len(heatmap_rows))
        question_text = _extract_question_text(rec)
        response_text = str(rec.get("student_response", "") or "")

        # Exact per-token delta; pad shorter responses with None
        row = []
        for i in range(max_tokens):
            if i < n and np.isfinite(delta[i]):
                row.append(float(delta[i]))
            else:
                row.append(None)

        heatmap_rows.append(row)
        tokens = _extract_token_texts_from_record(rec, start=start, end=end, tokenizer=tokenizer)
        token_row = []
        for i in range(max_tokens):
            token_row.append(tokens[i] if i < n else None)
        token_rows.append(token_row)

        # Reward display
        if reward is None:
            r_str = "?"
        else:
            r_str = str(int(reward)) if float(reward) in (0.0, 1.0) else f"{reward:.2f}"
        row_labels.append(f"[{r_str}] req {idx}")
        rewards.append(reward)

        finite_delta = delta[np.isfinite(delta)]
        if finite_delta.size > 0:
            teacher_win_rate = float(np.mean(finite_delta > 0) * 100.0)
            mean_delta = float(np.mean(finite_delta))
        else:
            teacher_win_rate = None
            mean_delta = None

        # Detail data — trim to avoid massive HTML; keep full arrays up to 4096 tokens
        cap = 99999
        detail_data.append({
            "idx": idx,
            "reward": reward,
            "r_str": r_str,
            "question_text": question_text,
            "student_response_text": response_text,
            "student_logprobs": s[start:min(end, start + cap)].tolist(),
            "teacher_logprobs": t[start:min(end, start + cap)].tolist(),
            "delta": delta[:min(n, cap)].tolist(),
            "tokens": tokens[:min(n, cap)],
            "n": min(n, cap),
            "teacher_win_rate_pct": teacher_win_rate,
            "mean_delta": mean_delta,
            "window_start_1idx": start + 1,
            "window_end_1idx": min(end, start + cap),
            "n_full": n_full,
            "answer_token_start": _remap_marker_to_window(ts.get("answer_token_start"), start, end),
            "thinking_token_start": _remap_marker_to_window(ts.get("thinking_token_start"), start, end),
            "thinking_token_end": _remap_marker_to_window(ts.get("thinking_token_end"), start, end),
        })

    return heatmap_rows, row_labels, rewards, x_positions, detail_data, max_tokens, token_rows, window_desc


def generate_html(
    heatmap_rows: list,
    row_labels: list,
    rewards: list,
    x_positions: list,
    detail_data: list,
    max_tokens: int,
    token_rows: list,
    window_desc: str,
    boxed_stats: dict,
    plotly_js_src: str | None = None,
) -> str:
    # Compute symmetric color range
    abs_max = 1.0  # color range fixed to [-1, 1]; values outside saturate to extreme colors

    n_records = len(heatmap_rows)
    heatmap_height = max(320, min(12000, n_records * 18 + 140))

    heatmap_json = json.dumps(heatmap_rows)
    labels_json = json.dumps(row_labels)
    rewards_json = json.dumps(rewards)
    x_positions_json = json.dumps(x_positions)
    detail_json = json.dumps(detail_data)
    token_rows_json = json.dumps(token_rows)
    window_desc_json = json.dumps(window_desc)
    boxed_stats_json = json.dumps(boxed_stats)

    if plotly_js_src:
        plotly_script_tag = f"<script>{plotly_js_src}</script>"
    else:
        plotly_script_tag = f'<script src="{_PLOTLY_CDN}" charset="utf-8"></script>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Teacher vs Student Token Logprob Explorer</title>
  {plotly_script_tag}
  <style>
    :root {{
      --bg: #edf0f4;
      --card: #ffffff;
      --border: #d9e0e9;
      --text: #1d2430;
      --muted: #586376;
      --accent: #3d5a80;
      --shadow: 0 8px 24px rgba(26, 32, 44, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 16px;
      background:
        radial-gradient(circle at 90% -10%, rgba(176, 196, 222, 0.40), rgba(176, 196, 222, 0) 30%),
        linear-gradient(180deg, #f4f6fa 0%, var(--bg) 100%);
      color: var(--text);
      font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      line-height: 1.45;
    }}
    .page {{ max-width: 1380px; margin: 0 auto; }}
    h1 {{ margin: 0; font-size: 1.35rem; font-weight: 700; }}
    .subtitle {{ margin-top: 6px; color: var(--muted); font-size: 0.92rem; }}
    .legend {{ display: flex; gap: 16px; align-items: center; margin-top: 12px; flex-wrap: wrap; }}
    .legend-item {{ display: flex; align-items: center; gap: 7px; font-size: 0.86rem; color: #293344; }}
    .swatch {{ width: 22px; height: 14px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.08); flex-shrink: 0; }}
    .colorbar-card {{
      margin-top: 10px;
      padding: 10px 12px;
      border: 1px solid #dbe3ee;
      border-radius: 10px;
      background: #f8fbfd;
    }}
    .colorbar-title {{ font-size: 0.82rem; color: #455269; margin-bottom: 6px; font-weight: 600; }}
    .colorbar-gradient {{
      width: 100%;
      height: 14px;
      border-radius: 8px;
      border: 1px solid rgba(0,0,0,0.12);
      background: linear-gradient(to right, #1a7a1a 0%, #ebf6eb 48%, #ffffff 50%, #fdeeee 52%, #8b0000 100%);
    }}
    .colorbar-ticks {{
      display: flex;
      justify-content: space-between;
      margin-top: 5px;
      font-size: 0.78rem;
      color: #4d5a70;
      font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}
    .colorbar-desc {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-top: 6px;
      font-size: 0.8rem;
      color: #3f4c62;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: var(--shadow);
      padding: 14px;
      margin-top: 14px;
    }}
    .section-title {{ margin: 0 0 8px; font-size: 1rem; font-weight: 650; }}
    .hint {{ color: #5d697d; font-size: 0.83rem; }}
    #stats {{ font-size: 0.87rem; color: #39465a; margin-bottom: 10px; line-height: 1.5; }}
    #boxed-summary {{ font-size: 0.86rem; color: #2f3948; line-height: 1.5; }}
    .boxed-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.83rem; }}
    .boxed-table th, .boxed-table td {{ border: 1px solid #e3e8ef; padding: 6px 8px; text-align: left; }}
    .boxed-table th {{ background: #f6f8fb; }}
    #overview-section {{ display: block; }}
    #detail-page {{ display: none; }}
    .nav-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .nav-controls {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    button, select {{
      font: inherit;
      border: 1px solid #c6d0df;
      background: #ffffff;
      color: #1f2937;
      border-radius: 8px;
      padding: 6px 10px;
    }}
    button {{
      cursor: pointer;
      transition: background 0.16s ease, border-color 0.16s ease;
    }}
    button:hover {{ background: #f3f6fa; border-color: #aebbd0; }}
    button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
    #detail-title {{ font-size: 1rem; font-weight: 650; color: #1f2a3b; }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      margin: 10px 0 12px;
    }}
    .metric {{
      border: 1px solid #dde4ed;
      border-radius: 10px;
      background: #f9fbfd;
      padding: 8px 9px;
    }}
    .metric .label {{ font-size: 0.76rem; color: #5f6c80; margin-bottom: 2px; }}
    .metric .value {{ font-size: 0.89rem; font-weight: 600; color: #202b3c; }}
    #question-block {{
      border: 1px solid #dde4ed;
      border-radius: 10px;
      background: #f9fbfd;
      padding: 10px;
      margin-bottom: 12px;
    }}
    #question-title {{ font-size: 0.83rem; color: #4f5d73; margin-bottom: 4px; font-weight: 600; }}
    #question-text {{
      margin: 0;
      font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.42;
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    #token-wall {{
      width: 100%;
      min-height: 78vh;
      border: 1px solid #dde4ed;
      border-radius: 12px;
      background: #f8fafd;
      padding: 10px;
      overflow: auto;
      display: flex;
      flex-wrap: wrap;
      align-content: flex-start;
      gap: 6px;
    }}
    .token-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 8px;
      border-radius: 8px;
      border: 1px solid rgba(0, 0, 0, 0.08);
      font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.25;
      white-space: pre-wrap;
      word-break: break-word;
      max-width: 100%;
    }}
    .token-pos {{
      font-size: 10px;
      opacity: 0.75;
      flex-shrink: 0;
    }}
    .token-text {{
      overflow-wrap: anywhere;
    }}
    #detail-chart {{ width: 100%; height: 500px; margin-top: 10px; }}
    #token-preview {{
      margin-top: 10px;
      border: 1px solid #dde4ed;
      border-radius: 10px;
      background: #f9fbfd;
      padding: 9px;
      max-height: 240px;
      overflow: auto;
      font-size: 12px;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}
    @media (max-width: 1100px) {{
      .detail-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      #detail-chart {{ height: 440px; }}
      #token-wall {{ min-height: 72vh; }}
    }}
    @media (max-width: 768px) {{
      body {{ padding: 12px; }}
      .detail-grid {{ grid-template-columns: 1fr; }}
      .nav-row {{ align-items: flex-start; }}
      .nav-controls {{ width: 100%; }}
      #detail-chart {{ height: 400px; }}
      #token-wall {{ min-height: 66vh; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Teacher vs Student Token Logprob Explorer</h1>
    <div class="subtitle">
      Color encodes <strong>teacher logprob − student logprob</strong> at each token position.
      Keep the overview heatmap for global comparison, then open one request in an independent detail view.
    </div>
    <div class="legend">
      <div class="legend-item">
        <div class="swatch" style="background: linear-gradient(to right, #8b0000, #f0b1b1);"></div>
        <span><strong>Red</strong>: teacher stronger (Δ &gt; 0)</span>
      </div>
      <div class="legend-item">
        <div class="swatch" style="background: linear-gradient(to right, #e8f5e8, #1a7a1a);"></div>
        <span><strong>Green</strong>: student stronger (Δ &lt; 0)</span>
      </div>
      <div class="legend-item">
        <div class="swatch" style="background: #ffffff;"></div>
        <span>White: near tie (Δ ≈ 0)</span>
      </div>
    </div>
    <div class="colorbar-card">
      <div class="colorbar-title">Color Direction & Magnitude for Δ = teacher logprob − student logprob</div>
      <div class="colorbar-gradient"></div>
      <div class="colorbar-ticks">
        <span>-{abs_max:.1f}</span>
        <span>0</span>
        <span>+{abs_max:.1f}</span>
      </div>
      <div class="colorbar-desc">
        <span>Student stronger (green)</span>
        <span>Teacher stronger (red)</span>
      </div>
    </div>

    <div class="card">
      <div id="boxed-summary"></div>
    </div>

    <div class="card" id="overview-section">
      <h2 class="section-title">Overview Heatmap</h2>
      <div id="stats"></div>
      <div class="hint">Click any heatmap row to open an independent per-question detail page below.</div>
      <div id="heatmap" style="width:100%; height:{heatmap_height}px;"></div>
    </div>

    <div class="card" id="detail-page">
      <div class="nav-row">
        <div id="detail-title">Question detail</div>
        <div class="nav-controls">
          <button id="back-overview" type="button">Back to Overview</button>
          <button id="nav-prev" type="button">Previous</button>
          <button id="nav-next" type="button">Next</button>
          <label for="req-select">Request:</label>
          <select id="req-select"></select>
        </div>
      </div>
      <div class="detail-grid" id="detail-summary"></div>
      <div id="question-block">
        <div id="question-title">Question</div>
        <pre id="question-text"></pre>
      </div>
      <div id="token-wall"></div>
      <div id="detail-chart"></div>
      <div id="token-preview"></div>
    </div>
  </div>

  <script>
  (function() {{
    var Z = {heatmap_json};
    var yLabels = {labels_json};
    var rewards = {rewards_json};
    var xCenters = {x_positions_json};
    var details = {detail_json};
    var tokenRows = {token_rows_json};
    var windowDesc = {window_desc_json};
    var boxedStats = {boxed_stats_json};
    var absMax = {abs_max:.6f};
    var currentRowIdx = null;

    var colorscale = [
      [0.00, '#1a7a1a'],
      [0.35, '#8ac48a'],
      [0.49, '#ebf6eb'],
      [0.50, '#ffffff'],
      [0.51, '#fdeeee'],
      [0.65, '#de8f8f'],
      [1.00, '#8b0000'],
    ];

    function fmtNum(v, digits) {{
      if (v === null || v === undefined || !isFinite(v)) return 'N/A';
      var d = (digits === undefined) ? 4 : digits;
      return Number(v).toFixed(d);
    }}

    function visibleToken(tok) {{
      if (tok === null || tok === undefined || tok === '') return '(empty)';
      return String(tok)
        .replace(/\\n/g, '\\\\n')
        .replace(/\\r/g, '\\\\r')
        .replace(/\\t/g, '\\\\t');
    }}

    function escHtml(s) {{
      return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function windowAbsoluteX(d) {{
      var x = [];
      for (var i = 0; i < d.n; i++) x.push(d.window_start_1idx + i);
      return x;
    }}

    function toAbsPosition(relativePos, windowStart) {{
      if (!relativePos || !isFinite(relativePos)) return null;
      return Number(relativePos) + Number(windowStart) - 1;
    }}

    function clamp(v, lo, hi) {{
      return Math.max(lo, Math.min(hi, v));
    }}

    function deltaToChipStyle(delta) {{
      if (!isFinite(delta)) {{
        return {{ bg: 'rgba(140,140,140,0.18)', fg: '#111111' }};
      }}
      var x = clamp(delta / absMax, -1, 1);
      var strength = Math.abs(x);
      // x > 0 => teacher win (red), x < 0 => student win (green)
      if (x >= 0) {{
        var r = 255;
        var g = Math.round(255 - 120 * strength);
        var b = Math.round(255 - 120 * strength);
        return {{
          bg: 'rgb(' + r + ',' + g + ',' + b + ')',
          fg: '#111111'
        }};
      }}
      var r2 = Math.round(255 - 120 * strength);
      var g2 = 255;
      var b2 = Math.round(255 - 120 * strength);
      return {{
        bg: 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')',
        fg: '#111111'
      }};
    }}

    function summarizeOverviewStats() {{
      var nRecords = Z.length;
      var rewardCounts = {{}};
      rewards.forEach(function(r) {{
        var k = (r === null || r === undefined) ? '?' : String(r);
        rewardCounts[k] = (rewardCounts[k] || 0) + 1;
      }});
      var rewardStr = Object.entries(rewardCounts)
        .map(function(e) {{ return 'reward=' + e[0] + ': ' + e[1]; }})
        .join(' | ');

      var totalTokens = 0, teacherWins = 0, deltaSum = 0;
      Z.forEach(function(row) {{
        row.forEach(function(v) {{
          if (v !== null && v !== undefined && isFinite(v)) {{
            totalTokens++;
            deltaSum += v;
            if (v > 0) teacherWins++;
          }}
        }});
      }});
      var teacherWinRate = totalTokens > 0 ? (teacherWins / totalTokens * 100).toFixed(1) : 'N/A';
      var meanDelta = totalTokens > 0 ? (deltaSum / totalTokens).toFixed(4) : 'N/A';

      document.getElementById('stats').innerHTML =
        nRecords + ' requests  |  ' + rewardStr +
        '  |  max token position shown: ' + {max_tokens} +
        '  |  window: ' + escHtml(windowDesc) +
        '<br><strong>Teacher win rate:</strong> ' + teacherWinRate + '%  |  ' +
        '<strong>Mean Δ (teacher − student):</strong> ' + meanDelta;
    }}

    function renderBoxedSummary() {{
      var boxedEl = document.getElementById('boxed-summary');
      if (!boxedStats || !boxedStats.available) {{
        boxedEl.innerHTML =
          '<strong>Boxed Token Stats</strong><br>' +
          escHtml((boxedStats && boxedStats.reason) ? boxedStats.reason : 'Not available.');
        return;
      }}
      var a1 = boxedStats.all_boxed.reward_1;
      var a0 = boxedStats.all_boxed.reward_0;
      var l1 = boxedStats.last_boxed_only.reward_1;
      var l0 = boxedStats.last_boxed_only.reward_0;
      boxedEl.innerHTML =
        '<strong>Boxed Token Stats (teacher − student)</strong><br>' +
        'Eligible records: ' + boxedStats.eligible_records +
        ' | Missing \\\\boxed records: ' + boxedStats.missing_boxed_records +
        ' | NaN filtered (all/last): ' + boxedStats.nan_tokens_all_boxed + '/' + boxedStats.nan_tokens_last_boxed +
        '<table class="boxed-table">' +
          '<thead><tr><th>Region</th><th>Reward</th><th>Samples</th><th>Tokens</th><th>Pooled Mean Δ</th><th>Sample Mean Δ</th></tr></thead>' +
          '<tbody>' +
            '<tr><td>All boxed spans</td><td>1</td><td>' + a1.sample_count + '</td><td>' + a1.token_count + '</td><td>' + fmtNum(a1.pooled_mean_delta) + '</td><td>' + fmtNum(a1.sample_mean_delta) + '</td></tr>' +
            '<tr><td>All boxed spans</td><td>0</td><td>' + a0.sample_count + '</td><td>' + a0.token_count + '</td><td>' + fmtNum(a0.pooled_mean_delta) + '</td><td>' + fmtNum(a0.sample_mean_delta) + '</td></tr>' +
            '<tr><td>Last boxed only</td><td>1</td><td>' + l1.sample_count + '</td><td>' + l1.token_count + '</td><td>' + fmtNum(l1.pooled_mean_delta) + '</td><td>' + fmtNum(l1.sample_mean_delta) + '</td></tr>' +
            '<tr><td>Last boxed only</td><td>0</td><td>' + l0.sample_count + '</td><td>' + l0.token_count + '</td><td>' + fmtNum(l0.pooled_mean_delta) + '</td><td>' + fmtNum(l0.sample_mean_delta) + '</td></tr>' +
          '</tbody>' +
        '</table>';
    }}

    function renderOverviewHeatmap() {{
      var trace = {{
        type: 'heatmap',
        z: Z,
        x: xCenters,
        y: yLabels,
        customdata: tokenRows,
        colorscale: colorscale,
        zmin: -absMax,
        zmax: absMax,
        zmid: 0,
        colorbar: {{
          title: {{ text: 'teacher − student<br>logprob', side: 'right' }},
          thickness: 14,
          len: 0.55,
          tickfont: {{ size: 10 }},
        }},
        hoverongaps: false,
        hovertemplate:
          '<b>%{{y}}</b><br>' +
          'token pos ≈ %{{x}}<br>' +
          'token: %{{customdata}}<br>' +
          'Δ logprob: %{{z:.4f}}<extra></extra>',
      }};
      var layout = {{
        margin: {{ l: 150, r: 85, t: 14, b: 56 }},
        xaxis: {{ title: 'token position', tickfont: {{ size: 10 }} }},
        yaxis: {{ autorange: 'reversed', tickfont: {{ size: 10 }}, automargin: true }},
        plot_bgcolor: '#fafbfc',
      }};
      Plotly.newPlot('heatmap', [trace], layout, {{ responsive: true, displayModeBar: true }});

      document.getElementById('heatmap').on('plotly_click', function(evt) {{
        var pt = evt.points[0];
        var rowIdx = Array.isArray(pt.pointIndex) ? pt.pointIndex[0] : pt.pointIndex;
        if (rowIdx >= 0 && rowIdx < details.length) {{
          location.hash = '#req-' + encodeURIComponent(String(details[rowIdx].idx));
        }}
      }});
    }}

    function renderQuestionDetail(rowIdx) {{
      if (rowIdx < 0 || rowIdx >= details.length) return;
      currentRowIdx = rowIdx;
      var d = details[rowIdx];
      var tokens = (d.tokens || []).slice(0, d.n);
      while (tokens.length < d.n) tokens.push('');
      var shownTokens = tokens.map(visibleToken);
      var xAbs = windowAbsoluteX(d);
      var selectEl = document.getElementById('req-select');
      if (selectEl.selectedIndex !== rowIdx) {{
        selectEl.selectedIndex = rowIdx;
      }}

      document.getElementById('overview-section').style.display = 'none';
      document.getElementById('detail-page').style.display = 'block';
      document.getElementById('detail-title').textContent =
        'Request ' + d.idx + ' | reward=' + d.r_str +
        ' | token window ' + d.window_start_1idx + '...' + d.window_end_1idx + ' / ' + d.n_full;

      var summaryHtml = '';
      var cards = [
        ['Request', String(d.idx)],
        ['Reward', String(d.r_str)],
        ['Token Window', d.window_start_1idx + '...' + d.window_end_1idx],
        ['Teacher Win Rate', fmtNum(d.teacher_win_rate_pct, 1) + (isFinite(d.teacher_win_rate_pct) ? '%' : '')],
        ['Mean Δ', fmtNum(d.mean_delta, 5)]
      ];
      cards.forEach(function(item) {{
        summaryHtml +=
          '<div class="metric"><div class="label">' + escHtml(item[0]) +
          '</div><div class="value">' + escHtml(item[1]) + '</div></div>';
      }});
      document.getElementById('detail-summary').innerHTML = summaryHtml;

      document.getElementById('question-text').textContent = String(d.question_text || '(Question text unavailable)');

      var wall = document.getElementById('token-wall');
      var chips = [];
      for (var ti = 0; ti < shownTokens.length; ti++) {{
        var style = deltaToChipStyle(d.delta[ti]);
        var absPos = xAbs[ti];
        var tip =
          'pos=' + absPos +
          '\\ntoken=' + shownTokens[ti] +
          '\\nΔ=' + fmtNum(d.delta[ti], 5) +
          '\\nteacher=' + fmtNum(d.teacher_logprobs[ti], 5) +
          '\\nstudent=' + fmtNum(d.student_logprobs[ti], 5);
        chips.push(
          '<span class="token-chip" title="' + escHtml(tip) + '" style="background:' + style.bg + ';color:' + style.fg + ';">' +
            '<span class="token-pos">[' + absPos + ']</span>' +
            '<span class="token-text">' + escHtml(shownTokens[ti]) + '</span>' +
          '</span>'
        );
      }}
      wall.innerHTML = chips.join('');

      var lineCustom = shownTokens;
      var teacherTrace = {{
        type: 'scatter', mode: 'lines+markers',
        x: xAbs, y: d.teacher_logprobs, customdata: lineCustom,
        name: 'teacher logprob',
        line: {{ color: '#d17a20', width: 1.5 }},
        marker: {{ size: 3, color: '#d17a20' }},
        hovertemplate: 'pos=%{{x}}<br>token=%{{customdata}}<br>teacher=%{{y:.5f}}<extra>teacher</extra>',
      }};
      var studentTrace = {{
        type: 'scatter', mode: 'lines+markers',
        x: xAbs, y: d.student_logprobs, customdata: lineCustom,
        name: 'student logprob',
        line: {{ color: '#2f5a9b', width: 1.5 }},
        marker: {{ size: 3, color: '#2f5a9b' }},
        hovertemplate: 'pos=%{{x}}<br>token=%{{customdata}}<br>student=%{{y:.5f}}<extra>student</extra>',
      }};
      var deltaBar = {{
        type: 'bar', yaxis: 'y2', x: xAbs,
        y: d.delta.map(function(v) {{ return isFinite(v) ? v : 0; }}),
        customdata: lineCustom,
        name: 'Δ (teacher−student)',
        marker: {{
          color: d.delta.map(function(v) {{
            if (!isFinite(v)) return 'rgba(140,140,140,0.30)';
            return v > 0 ? 'rgba(139,0,0,0.35)' : 'rgba(26,122,26,0.35)';
          }}),
        }},
        hovertemplate: 'pos=%{{x}}<br>token=%{{customdata}}<br>Δ=%{{y:.5f}}<extra>Δ</extra>',
      }};

      var shapes = [], annotations = [];
      var thinkStartAbs = toAbsPosition(d.thinking_token_start, d.window_start_1idx);
      var thinkEndAbs = toAbsPosition(d.thinking_token_end, d.window_start_1idx);
      var ansAbs = toAbsPosition(d.answer_token_start, d.window_start_1idx);
      if (thinkStartAbs) {{
        shapes.push({{ type:'line', x0:thinkStartAbs, x1:thinkStartAbs, y0:0, y1:1, yref:'paper',
          line:{{ color:'#2ca02c', dash:'dash', width:1.5 }} }});
        annotations.push({{ x:thinkStartAbs, y:1, yref:'paper', text:'think', showarrow:false,
          font:{{ color:'#2ca02c', size:10 }}, xanchor:'left' }});
      }}
      if (thinkEndAbs) {{
        shapes.push({{ type:'line', x0:thinkEndAbs, x1:thinkEndAbs, y0:0, y1:1, yref:'paper',
          line:{{ color:'#9467bd', dash:'dashdot', width:1.5 }} }});
        annotations.push({{ x:thinkEndAbs, y:0.93, yref:'paper', text:'think_end', showarrow:false,
          font:{{ color:'#9467bd', size:10 }}, xanchor:'left' }});
      }}
      if (ansAbs) {{
        shapes.push({{ type:'line', x0:ansAbs, x1:ansAbs, y0:0, y1:1, yref:'paper',
          line:{{ color:'#d62728', dash:'dash', width:1.5 }} }});
        annotations.push({{ x:ansAbs, y:0.86, yref:'paper', text:'answer', showarrow:false,
          font:{{ color:'#d62728', size:10 }}, xanchor:'left' }});
      }}
      var layout2 = {{
        title: {{ text: 'Per-token logprob curves for request ' + d.idx, font: {{ size: 13 }} }},
        xaxis: {{ title: 'token position (absolute)', tickfont: {{ size: 10 }} }},
        yaxis: {{ title: 'logprob', tickfont: {{ size: 10 }} }},
        yaxis2: {{
          title: 'Δ logprob', overlaying: 'y', side: 'right',
          zeroline: true, zerolinecolor: '#a3acba', tickfont: {{ size: 10 }},
        }},
        hovermode: 'x unified',
        legend: {{ orientation: 'h', y: -0.18, font: {{ size: 10 }} }},
        margin: {{ l: 58, r: 66, t: 52, b: 78 }},
        shapes: shapes,
        annotations: annotations,
        plot_bgcolor: '#fafbfc',
        bargap: 0,
      }};
      Plotly.react('detail-chart', [teacherTrace, studentTrace, deltaBar], layout2, {{ responsive: true }});

      var previewLimit = 450;
      var preview = shownTokens.slice(0, previewLimit).map(function(tok, i) {{
        return '[' + xAbs[i] + '] ' + tok;
      }}).join(' ');
      if (shownTokens.length > previewLimit) {{
        preview += '\\n... (' + (shownTokens.length - previewLimit) + ' more tokens hidden in preview)';
      }}
      document.getElementById('token-preview').textContent = preview;

      document.getElementById('nav-prev').disabled = rowIdx <= 0;
      document.getElementById('nav-next').disabled = rowIdx >= details.length - 1;
      document.getElementById('detail-page').scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}

    function bindNavigator() {{
      var selectEl = document.getElementById('req-select');
      details.forEach(function(d, i) {{
        var option = document.createElement('option');
        option.value = String(i);
        option.textContent = 'req ' + d.idx + ' | reward=' + d.r_str;
        selectEl.appendChild(option);
      }});

      document.getElementById('nav-prev').addEventListener('click', function() {{
        if (currentRowIdx === null) return;
        var next = Math.max(0, currentRowIdx - 1);
        location.hash = '#req-' + encodeURIComponent(String(details[next].idx));
      }});

      document.getElementById('nav-next').addEventListener('click', function() {{
        if (currentRowIdx === null) return;
        var next = Math.min(details.length - 1, currentRowIdx + 1);
        location.hash = '#req-' + encodeURIComponent(String(details[next].idx));
      }});

      selectEl.addEventListener('change', function(evt) {{
        var targetRow = Number(evt.target.value);
        if (isFinite(targetRow) && targetRow >= 0 && targetRow < details.length) {{
          location.hash = '#req-' + encodeURIComponent(String(details[targetRow].idx));
        }}
      }});

      document.getElementById('back-overview').addEventListener('click', function() {{
        history.replaceState(null, '', location.pathname + location.search);
        currentRowIdx = null;
        document.getElementById('detail-page').style.display = 'none';
        document.getElementById('overview-section').style.display = 'block';
        document.getElementById('overview-section').scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      }});
    }}

    function findRowIndexByReqId(reqId) {{
      if (reqId === null || reqId === undefined) return -1;
      var target = String(reqId);
      for (var i = 0; i < details.length; i++) {{
        if (String(details[i].idx) === target) return i;
      }}
      var numeric = Number(target);
      if (isFinite(numeric) && numeric >= 0 && numeric < details.length) return numeric;
      return -1;
    }}

    function syncHashRoute() {{
      var h = location.hash || '';
      if (!h || h === '#overview') {{
        currentRowIdx = null;
        document.getElementById('detail-page').style.display = 'none';
        document.getElementById('overview-section').style.display = 'block';
        return;
      }}
      var m = h.match(/^#req-(.+)$/);
      if (!m) return;
      var reqId = decodeURIComponent(m[1] || '');
      var rowIdx = findRowIndexByReqId(reqId);
      if (rowIdx >= 0) {{
        renderQuestionDetail(rowIdx);
      }}
    }}

    summarizeOverviewStats();
    renderBoxedSummary();
    renderOverviewHeatmap();
    bindNavigator();
    syncHashRoute();
    window.addEventListener('hashchange', syncHashRoute);
  }})();
  </script>
</body>
</html>"""


def serve(out_path: Path, port: int):
    """Start a simple HTTP server and print a clickable URL."""
    import http.server
    import socket
    import threading

    serve_dir = out_path.parent.resolve()
    filename = out_path.name

    # Find a free port if port=0
    if port == 0:
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, fmt, *args):  # silence request logs
            pass

    server = http.server.HTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}/{filename}"
    print()
    print("=" * 60)
    print(f"  Serving at:  {url}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)
    print()

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL file from eval_student_teacher_inference.py")
    parser.add_argument("--output", default="./token_winner_interactive.html", help="Output HTML path")
    parser.add_argument("--max-requests", type=int, default=None, help="Cap number of requests (default: all)")
    parser.add_argument("--n-bins", type=int, default=200, help="Number of token-position bins for heatmap columns (default: 200)")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path/name (Hugging Face) for reconstructing per-token text when not present in JSONL.",
    )
    parser.add_argument(
        "--show-first-k-tokens",
        type=int,
        default=None,
        help="Only show the first K tokens of each request in plots.",
    )
    parser.add_argument(
        "--show-last-k-tokens",
        type=int,
        default=None,
        help="Only show the last K tokens of each request in plots.",
    )
    parser.add_argument("--port", type=int, default=0, help="HTTP server port (default: auto)")
    parser.add_argument("--no-serve", action="store_true", help="Just save the HTML file, don't start a server")
    args = parser.parse_args()

    records = load_records(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    if args.max_requests:
        records = records[: args.max_requests]
        print(f"Using first {len(records)} records")

    if args.show_first_k_tokens is not None and args.show_first_k_tokens <= 0:
        raise ValueError("--show-first-k-tokens must be > 0")
    if args.show_last_k_tokens is not None and args.show_last_k_tokens <= 0:
        raise ValueError("--show-last-k-tokens must be > 0")
    if args.show_first_k_tokens is not None and args.show_last_k_tokens is not None:
        raise ValueError("Please set only one of --show-first-k-tokens / --show-last-k-tokens")

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            print(f"Loading tokenizer from {args.tokenizer} ...")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        except Exception as e:
            print(f"WARNING: failed to load tokenizer from {args.tokenizer}: {e}")
            print("Will fall back to best-effort whitespace token chunks.")

    heatmap_rows, row_labels, rewards, x_positions, detail_data, max_tokens, token_rows, window_desc = build_heatmap_data(
        records,
        tokenizer=tokenizer,
        n_bins=args.n_bins,
        first_k_tokens=args.show_first_k_tokens,
        last_k_tokens=args.show_last_k_tokens,
    )

    if not heatmap_rows:
        print("No records with both student_logprobs and teacher_logprobs found. Nothing to plot.")
        return

    boxed_stats = compute_boxed_stats(records, tokenizer=tokenizer)

    print(f"Building heatmap: {len(heatmap_rows)} rows x {max_tokens} tokens | max_tokens={max_tokens}")

    plotly_js_src = _get_plotly_js()
    html = generate_html(
        heatmap_rows, row_labels, rewards, x_positions, detail_data, max_tokens, token_rows, window_desc, boxed_stats,
        plotly_js_src=plotly_js_src,
    )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved HTML to {out_path}")

    if args.no_serve:
        print("Open the HTML file in a browser to explore.")
    else:
        serve(out_path, port=args.port)


if __name__ == "__main__":
    main()
