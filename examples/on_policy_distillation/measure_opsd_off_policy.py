#!/usr/bin/env python3
"""
Measure off-policy degree between teacher policy and student rollouts from
`eval_student_teacher_inference.py` JSONL outputs.

Default outputs:
- summary JSON: ./opsd_offpolicy_summary.json
- per-sample JSONL: ./opsd_offpolicy_per_sample.jsonl

Example:
  python examples/on_policy_distillation/measure_opsd_off_policy.py \
    --input ./eval_math500_student_teacher_inference_s1.7t1.7b_answeronly_b64_v1.jsonl

    python examples/on_policy_distillation/measure_opsd_off_policy.py \
    --input ./eval_math500_student_teacher_inference_s1.7t8b_answeronly.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_SEGMENTS = {"all", "thinking", "answer"}


@dataclass
class TokenPair:
    idx0: int
    student_lp: float
    teacher_lp: float


@dataclass
class SegmentData:
    available: bool
    start0: int | None
    end0: int | None
    tokens: list[TokenPair]


@dataclass
class SampleParsed:
    index: int
    raw_token_count: int
    valid_token_count: int
    dropped_token_count: int
    thinking_token_start: int | None
    answer_token_start: int | None
    segments: dict[str, SegmentData]


@dataclass
class SampleMetrics:
    index: int
    kept_for_global: bool
    raw_token_count: int
    valid_token_count_all: int
    dropped_token_count: int
    per_segment: dict[str, dict[str, Any]]


def _parse_comma_floats(text: str) -> list[float]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def _parse_segments(text: str) -> list[str]:
    segs = [x.strip() for x in text.split(",") if x.strip()]
    if not segs:
        raise ValueError("--segments is empty")
    unknown = [s for s in segs if s not in ALLOWED_SEGMENTS]
    if unknown:
        raise ValueError(f"Unsupported segments: {unknown}. Allowed: {sorted(ALLOWED_SEGMENTS)}")
    # keep order, dedup
    seen = set()
    ordered = []
    for s in segs:
        if s not in seen:
            ordered.append(s)
            seen.add(s)
    return ordered


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return None
    return iv if iv > 0 else None


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _percentile(values: list[float], q: float) -> float:
    """q in [0, 100], linear interpolation."""
    if not values:
        return float("nan")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _summary_stats(values: list[float]) -> dict[str, Any]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "median": _percentile(finite, 50.0),
        "p10": _percentile(finite, 10.0),
        "p90": _percentile(finite, 90.0),
        "min": min(finite),
        "max": max(finite),
    }


def _logsumexp(log_values: list[float]) -> float:
    if not log_values:
        return float("nan")
    m = max(log_values)
    if not math.isfinite(m):
        return float("nan")
    return m + math.log(sum(math.exp(v - m) for v in log_values))


def _ess_from_log_weights(log_w: list[float]) -> tuple[float | None, float | None]:
    """
    Returns (ess, normalized_ess).
    normalized_ess = ess / n.
    """
    vals = [v for v in log_w if math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return None, None
    lse_w = _logsumexp(vals)
    lse_w2 = _logsumexp([2.0 * v for v in vals])
    if not (math.isfinite(lse_w) and math.isfinite(lse_w2)):
        return None, None
    log_ess = 2.0 * lse_w - lse_w2
    ess = math.exp(log_ess)
    ess = min(max(ess, 1.0), float(n))
    return ess, ess / float(n)


def _segment_bounds(
    n_raw: int,
    thinking_start_1idx: int | None,
    answer_start_1idx: int | None,
) -> dict[str, tuple[bool, int | None, int | None]]:
    out: dict[str, tuple[bool, int | None, int | None]] = {
        "all": (True, 0, n_raw),
        "thinking": (False, None, None),
        "answer": (False, None, None),
    }

    if thinking_start_1idx is not None:
        start = max(0, thinking_start_1idx - 1)
        end = n_raw
        if answer_start_1idx is not None:
            end = min(n_raw, max(start, answer_start_1idx - 1))
        out["thinking"] = (True, start, end)

    if answer_start_1idx is not None:
        start = max(0, answer_start_1idx - 1)
        end = n_raw
        out["answer"] = (True, start, end)

    return out


def _extract_token_pairs(row: dict) -> tuple[list[TokenPair], int]:
    ts = row.get("token_stats") or {}
    s = ts.get("student_logprobs") or []
    t = ts.get("teacher_logprobs") or []

    n = min(len(s), len(t))
    out: list[TokenPair] = []
    dropped = 0

    for i in range(n):
        sv = _safe_float(s[i])
        tv = _safe_float(t[i])
        if sv is None or tv is None or (not math.isfinite(sv)) or (not math.isfinite(tv)):
            dropped += 1
            continue
        out.append(TokenPair(idx0=i, student_lp=sv, teacher_lp=tv))

    return out, n


def _parse_row(row: dict, segments: list[str]) -> SampleParsed:
    idx = int(row.get("index", 0))
    ts = row.get("token_stats") or {}
    thinking_start = _safe_int(ts.get("thinking_token_start"))
    answer_start = _safe_int(ts.get("answer_token_start"))

    valid_pairs, raw_n = _extract_token_pairs(row)
    dropped = raw_n - len(valid_pairs)

    bounds = _segment_bounds(raw_n, thinking_start, answer_start)
    seg_data: dict[str, SegmentData] = {}
    for seg in segments:
        available, start0, end0 = bounds[seg]
        if not available or start0 is None or end0 is None:
            seg_data[seg] = SegmentData(available=False, start0=None, end0=None, tokens=[])
            continue
        seg_tokens = [tp for tp in valid_pairs if start0 <= tp.idx0 < end0]
        seg_data[seg] = SegmentData(available=True, start0=start0, end0=end0, tokens=seg_tokens)

    return SampleParsed(
        index=idx,
        raw_token_count=raw_n,
        valid_token_count=len(valid_pairs),
        dropped_token_count=dropped,
        thinking_token_start=thinking_start,
        answer_token_start=answer_start,
        segments=seg_data,
    )


def _compute_segment_metrics(seg: SegmentData) -> dict[str, Any]:
    if not seg.available:
        return {
            "status": "not_available",
            "valid_token_count": 0,
            "nll_teacher_on_student": None,
            "delta_mean_teacher_minus_student": None,
            "teacher_win_rate": None,
            "sampled_reverse_kl_proxy_student_minus_teacher": None,
            "log_weight_sum": None,
            "prefix_scores": [],
        }

    tokens = seg.tokens
    if not tokens:
        return {
            "status": "available_but_empty",
            "valid_token_count": 0,
            "nll_teacher_on_student": None,
            "delta_mean_teacher_minus_student": None,
            "teacher_win_rate": None,
            "sampled_reverse_kl_proxy_student_minus_teacher": None,
            "log_weight_sum": 0.0,
            "prefix_scores": [],
        }

    teacher_vals = [t.teacher_lp for t in tokens]
    student_vals = [t.student_lp for t in tokens]
    deltas = [tv - sv for tv, sv in zip(teacher_vals, student_vals)]

    cum = 0.0
    prefix_scores = []
    for i, tv in enumerate(teacher_vals, start=1):
        cum += tv
        prefix_scores.append(cum / float(i))

    delta_mean = sum(deltas) / len(deltas)
    return {
        "status": "ok",
        "valid_token_count": len(tokens),
        "nll_teacher_on_student": -sum(teacher_vals) / len(teacher_vals),
        "delta_mean_teacher_minus_student": delta_mean,
        "teacher_win_rate": sum(1 for d in deltas if d > 0.0) / len(deltas),
        "sampled_reverse_kl_proxy_student_minus_teacher": -delta_mean,
        "log_weight_sum": sum(deltas),
        "prefix_scores": prefix_scores,
    }


def _low_density_rates(
    prefix_scores: list[float],
    quantile_thresholds: list[float],
    absolute_thresholds: list[float],
) -> dict[str, Any]:
    finite = [v for v in prefix_scores if math.isfinite(v)]
    if not finite:
        return {
            "prefix_score_count": 0,
            "quantile_thresholds": {},
            "absolute_thresholds": {},
        }

    q_obj = {}
    for q in quantile_thresholds:
        q_label = f"p{q * 100:g}"
        thr = _percentile(finite, q * 100.0)
        rate = sum(1 for v in finite if v < thr) / len(finite)
        q_obj[q_label] = {
            "quantile": q,
            "threshold": thr,
            "rate_below_threshold": rate,
        }

    a_obj = {}
    for thr in absolute_thresholds:
        key = f"{thr:g}"
        rate = sum(1 for v in finite if v < thr) / len(finite)
        a_obj[key] = {
            "threshold": thr,
            "rate_below_threshold": rate,
        }

    return {
        "prefix_score_count": len(finite),
        "quantile_thresholds": q_obj,
        "absolute_thresholds": a_obj,
    }


def _compute_global_segment_metrics(
    sample_metrics: list[SampleMetrics],
    seg: str,
    quantile_thresholds: list[float],
    absolute_thresholds: list[float],
) -> dict[str, Any]:
    seg_rows = [sm.per_segment.get(seg, {}) for sm in sample_metrics if sm.kept_for_global]

    token_count = sum(int(r.get("valid_token_count") or 0) for r in seg_rows)
    nll_vals = [r["nll_teacher_on_student"] for r in seg_rows if r.get("nll_teacher_on_student") is not None]
    delta_vals = [
        r["delta_mean_teacher_minus_student"]
        for r in seg_rows
        if r.get("delta_mean_teacher_minus_student") is not None
    ]
    win_vals = [r["teacher_win_rate"] for r in seg_rows if r.get("teacher_win_rate") is not None]
    proxy_vals = [
        r["sampled_reverse_kl_proxy_student_minus_teacher"]
        for r in seg_rows
        if r.get("sampled_reverse_kl_proxy_student_minus_teacher") is not None
    ]
    log_w_vals = [r["log_weight_sum"] for r in seg_rows if r.get("log_weight_sum") is not None]
    prefix_vals = []
    for r in seg_rows:
        prefix_vals.extend(r.get("prefix_scores") or [])

    ess, n_ess = _ess_from_log_weights(log_w_vals)

    return {
        "kept_sample_count": len(seg_rows),
        "token_count": token_count,
        "nll_teacher_on_student": _summary_stats(nll_vals),
        "delta_mean_teacher_minus_student": _summary_stats(delta_vals),
        "teacher_win_rate": _summary_stats(win_vals),
        "sampled_reverse_kl_proxy_student_minus_teacher": _summary_stats(proxy_vals),
        "log_weight_sum": _summary_stats(log_w_vals),
        "ess": {
            "ess": ess,
            "normalized_ess": n_ess,
            "sample_count_for_ess": len(log_w_vals),
        },
        "low_density": _low_density_rates(
            prefix_scores=prefix_vals,
            quantile_thresholds=quantile_thresholds,
            absolute_thresholds=absolute_thresholds,
        ),
    }


def _build_distribution_stats(sample_metrics: list[SampleMetrics], segments: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    kept = [sm for sm in sample_metrics if sm.kept_for_global]

    out["sample_level"] = {
        "valid_token_count_all": _summary_stats([float(sm.valid_token_count_all) for sm in kept]),
        "dropped_token_count": _summary_stats([float(sm.dropped_token_count) for sm in kept]),
    }

    per_seg: dict[str, Any] = {}
    for seg in segments:
        rows = [sm.per_segment.get(seg, {}) for sm in kept]
        per_seg[seg] = {
            "valid_token_count": _summary_stats([float(r.get("valid_token_count") or 0) for r in rows]),
            "nll_teacher_on_student": _summary_stats(
                [r["nll_teacher_on_student"] for r in rows if r.get("nll_teacher_on_student") is not None]
            ),
            "delta_mean_teacher_minus_student": _summary_stats(
                [
                    r["delta_mean_teacher_minus_student"]
                    for r in rows
                    if r.get("delta_mean_teacher_minus_student") is not None
                ]
            ),
            "teacher_win_rate": _summary_stats(
                [r["teacher_win_rate"] for r in rows if r.get("teacher_win_rate") is not None]
            ),
            "sampled_reverse_kl_proxy_student_minus_teacher": _summary_stats(
                [
                    r["sampled_reverse_kl_proxy_student_minus_teacher"]
                    for r in rows
                    if r.get("sampled_reverse_kl_proxy_student_minus_teacher") is not None
                ]
            ),
            "log_weight_sum": _summary_stats(
                [r["log_weight_sum"] for r in rows if r.get("log_weight_sum") is not None]
            ),
        }
    out["per_segment"] = per_seg
    return out


def _print_terminal_summary(summary: dict[str, Any], segments: list[str]) -> None:
    meta = summary["meta"]
    print("=== OPSD Off-Policy Summary ===")
    print(f"input: {meta['input']}")
    print(
        "samples: total={total} parsed={parsed} kept={kept} dropped_by_min_valid={drop}".format(
            total=meta["total_lines"],
            parsed=meta["parsed_samples"],
            kept=meta["kept_samples"],
            drop=meta["dropped_samples_by_min_valid_tokens"],
        )
    )

    gm = summary["global_metrics"]
    for seg in segments:
        segm = gm.get(seg, {})
        print(f"\n[{seg}] kept_samples={segm.get('kept_sample_count')} token_count={segm.get('token_count')}")
        nll_mean = (segm.get("nll_teacher_on_student") or {}).get("mean")
        delta_mean = (segm.get("delta_mean_teacher_minus_student") or {}).get("mean")
        win_mean = (segm.get("teacher_win_rate") or {}).get("mean")
        ess = ((segm.get("ess") or {}).get("ess"))
        ness = ((segm.get("ess") or {}).get("normalized_ess"))
        print(
            "  nll_mean={nll}  delta_mean={delta}  win_rate_mean={win}  ess={ess}  n_ess={ness}".format(
                nll="None" if nll_mean is None else f"{nll_mean:.6f}",
                delta="None" if delta_mean is None else f"{delta_mean:.6f}",
                win="None" if win_mean is None else f"{win_mean:.4f}",
                ess="None" if ess is None else f"{ess:.4f}",
                ness="None" if ness is None else f"{ness:.6f}",
            )
        )
        low = segm.get("low_density") or {}
        q = low.get("quantile_thresholds") or {}
        a = low.get("absolute_thresholds") or {}
        if q:
            q_txt = ", ".join(
                f"{k}:thr={v['threshold']:.6f},rate={v['rate_below_threshold']:.4f}" for k, v in q.items()
            )
            print(f"  low-density quantiles: {q_txt}")
        if a:
            a_txt = ", ".join(
                f"{k}:rate={v['rate_below_threshold']:.4f}" for k, v in a.items()
            )
            print(f"  low-density absolute: {a_txt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL from eval_student_teacher_inference.py")
    parser.add_argument("--output-json", default="./opsd_offpolicy_summary.json", help="Summary JSON output path")
    parser.add_argument(
        "--output-per-sample-jsonl",
        default="./opsd_offpolicy_per_sample.jsonl",
        help="Per-sample metrics JSONL output path",
    )
    parser.add_argument(
        "--quantile-thresholds",
        default="0.1,0.05",
        help="Comma-separated quantile thresholds in [0,1], e.g. '0.1,0.05'",
    )
    parser.add_argument(
        "--absolute-thresholds",
        default="-2.0,-3.0",
        help="Comma-separated absolute prefix-score thresholds, e.g. '-2.0,-3.0'",
    )
    parser.add_argument(
        "--segments",
        default="all,thinking,answer",
        help="Comma-separated segment names from: all,thinking,answer",
    )
    parser.add_argument(
        "--min-valid-tokens",
        type=int,
        default=32,
        help="Minimum valid tokens in 'all' segment for including a sample in global metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segments = _parse_segments(args.segments)
    quantile_thresholds = _parse_comma_floats(args.quantile_thresholds)
    absolute_thresholds = _parse_comma_floats(args.absolute_thresholds)

    for q in quantile_thresholds:
        if q < 0.0 or q > 1.0:
            raise ValueError(f"Quantile threshold out of range [0,1]: {q}")

    in_path = Path(args.input)
    out_json = Path(args.output_json)
    out_per = Path(args.output_per_sample_jsonl)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_per.parent.mkdir(parents=True, exist_ok=True)

    sample_metrics: list[SampleMetrics] = []
    total_lines = 0
    parsed_samples = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            parsed = _parse_row(row, segments)
            parsed_samples += 1

            per_seg = {}
            for seg in segments:
                per_seg[seg] = _compute_segment_metrics(parsed.segments[seg])

            valid_all = int(per_seg.get("all", {}).get("valid_token_count") or 0)
            kept = valid_all >= args.min_valid_tokens

            sample_metrics.append(
                SampleMetrics(
                    index=parsed.index,
                    kept_for_global=kept,
                    raw_token_count=parsed.raw_token_count,
                    valid_token_count_all=valid_all,
                    dropped_token_count=parsed.dropped_token_count,
                    per_segment=per_seg,
                )
            )

    kept_samples = sum(1 for s in sample_metrics if s.kept_for_global)
    dropped_by_min = len(sample_metrics) - kept_samples

    global_metrics = {
        seg: _compute_global_segment_metrics(
            sample_metrics=sample_metrics,
            seg=seg,
            quantile_thresholds=quantile_thresholds,
            absolute_thresholds=absolute_thresholds,
        )
        for seg in segments
    }

    summary = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(in_path),
            "total_lines": total_lines,
            "parsed_samples": parsed_samples,
            "kept_samples": kept_samples,
            "dropped_samples_by_min_valid_tokens": dropped_by_min,
            "config": {
                "segments": segments,
                "quantile_thresholds": quantile_thresholds,
                "absolute_thresholds": absolute_thresholds,
                "min_valid_tokens": args.min_valid_tokens,
            },
            "notes": {
                "teacher_kl_proxy": (
                    "sampled_reverse_kl_proxy_student_minus_teacher is a proxy on sampled tokens, "
                    "not exact full-distribution token KL"
                )
            },
        },
        "global_metrics": global_metrics,
        "distribution_stats": _build_distribution_stats(sample_metrics, segments),
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with out_per.open("w", encoding="utf-8") as f:
        for sm in sample_metrics:
            row = {
                "index": sm.index,
                "kept_for_global": sm.kept_for_global,
                "raw_token_count": sm.raw_token_count,
                "valid_token_count_all": sm.valid_token_count_all,
                "dropped_token_count": sm.dropped_token_count,
                "per_segment": sm.per_segment,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _print_terminal_summary(summary, segments)
    print(f"\nWrote summary JSON: {out_json}")
    print(f"Wrote per-sample JSONL: {out_per}")


if __name__ == "__main__":
    main()
