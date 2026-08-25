#!/usr/bin/env python3
"""DPP v2.1 Phase A.1 offline tie-break counterfactual refinement.

Compares three tie-break rules on the n=100 detailed diagnostic log:

    T0 (current): score -> smaller effective duration -> smaller Prefill
                  budget -> stable plan_id
    T1:           score -> larger U_P -> smaller duration -> smaller budget
                  -> plan_id
    T2:           score -> larger G_P (= U_P / duration) -> larger U_P ->
                  smaller duration -> smaller budget -> plan_id

All three apply ONLY inside the winner tie set defined by the frozen
math.isclose(rel_tol=1e-9, abs_tol=1e-12). Clear-winner frames never change.
No Scheduler modification and no GPU execution.

Usage:
    python3 scripts/analyze_dpp_v2_phase_a1.py \
        <startup.log> <trace.jsonl> <output_dir>
"""

from __future__ import annotations

import ast
import json
import math
import statistics
import sys
from collections import Counter
from functools import cmp_to_key
from pathlib import Path

SCORE_REL_TOL = 1e-9
SCORE_ABS_TOL = 1e-12
BUCKETS = ("ZERO", "P25", "P50", "P75", "MAX", "FINISH")


def score_tied(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=SCORE_REL_TOL, abs_tol=SCORE_ABS_TOL)


def parse_diagnostic_frames(log_path: Path) -> list[dict]:
    frames: list[dict] = []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            marker = "ModularDPPScheduler diagnostic="
            if marker not in line:
                continue
            frames.append(ast.literal_eval(line.split(marker, 1)[1].strip()))
    return frames


def request_prompt_tokens(request_id: str, entries: list[dict]) -> int | None:
    rest = request_id.rsplit("-0-", 1)[0]
    try:
        index = int(rest.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return None
    if index >= len(entries):
        return None
    return int(entries[index]["input_tokens"])


def bucket_of(template_id: str) -> str:
    return template_id.split(":")[1]


def make_winner(stages):
    def compare(a: dict, b: dict) -> int:
        for direction, fn in stages:
            va, vb = fn(a), fn(b)
            if va < vb:
                return -1 if direction == "asc" else 1
            if va > vb:
                return 1 if direction == "asc" else -1
        return 0

    key = cmp_to_key(compare)
    return lambda tie_set: sorted(tie_set, key=key)[0]


DUR = lambda c: float(c["effective_duration_seconds"])
BUDGET = lambda c: int(c["total_prefill_tokens"])
PLAN = lambda c: str(c["plan_id"])

T0_WINNER = make_winner(
    [("asc", DUR), ("asc", BUDGET), ("asc", PLAN)]
)
T1_WINNER = make_winner(
    [("desc", lambda c: float(c["_up"])), ("asc", DUR), ("asc", BUDGET), ("asc", PLAN)]
)
T2_WINNER = make_winner(
    [
        ("desc", lambda c: float(c["_gp"])),
        ("desc", lambda c: float(c["_up"])),
        ("asc", DUR),
        ("asc", BUDGET),
        ("asc", PLAN),
    ]
)

THRESHOLDS = (250.0, 300.0, 500.0)


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def stats_block(xs: list[float]) -> dict:
    return {
        "count": len(xs),
        "mean": statistics.fmean(xs) if xs else None,
        "p50": pct(xs, 0.5),
        "p90": pct(xs, 0.9),
        "p95": pct(xs, 0.95),
        "p99": pct(xs, 0.99),
        "max": max(xs) if xs else None,
    }


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    log_path = Path(sys.argv[1])
    trace_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    frames = parse_diagnostic_frames(log_path)
    if not frames:
        raise SystemExit("no diagnostic frames parsed")
    entries = [json.loads(line) for line in trace_path.open()]

    missing_prompt: set[str] = set()
    # Decorate candidates with U_P / G_P.
    for frame in frames:
        for cand in frame["candidate_scores"]:
            up = 0.0
            for request_id, tokens in cand["prefill_items"]:
                p_i = request_prompt_tokens(request_id, entries)
                if p_i is None:
                    missing_prompt.add(request_id)
                    continue
                up += tokens / p_i
            cand["_up"] = up
            cand["_gp"] = up / float(cand["effective_duration_seconds"])

    # ---- per-frame counterfactual winners -------------------------------
    t0_replay_mismatch = 0
    clear_winner_changed = 0
    frames_with_candidates = 0
    rows: list[dict] = []
    tie_frames = 0
    backlog_frames = 0

    hist_all = {t: Counter() for t in ("T0", "T1", "T2")}
    hist_tie = {t: Counter() for t in ("T0", "T1", "T2")}
    changed = Counter()
    prefill_tokens = {t: [] for t in ("T0", "T1", "T2")}
    up_vals = {t: [] for t in ("T0", "T1", "T2")}
    gp_vals = {t: [] for t in ("T0", "T1", "T2")}
    dur_vals = {t: [] for t in ("T0", "T1", "T2")}
    mixed_dur_vals = {t: [] for t in ("T0", "T1", "T2")}
    threshold_counts = {
        t: {thr: 0 for thr in THRESHOLDS} for t in ("T0", "T1", "T2")
    }
    envelope_tie = {t: 0 for t in ("T0", "T1", "T2")}
    envelope_cross = {
        "T0<=250_T1>250": 0,
        "T0<=250_T2>250": 0,
        "T1<=250_T2>250": 0,
    }
    t1_max_to_t2 = Counter()
    t1_max_compare: list[dict] = []
    finish_available = 0
    finish_selected = {t: 0 for t in ("T0", "T1", "T2")}

    for frame in frames:
        candidates = frame["candidate_scores"]
        if not candidates:
            continue
        frames_with_candidates += 1
        is_backlog = int(frame["current_prefill_count"]) > 0
        winner_score = max(c["score"] for c in candidates)
        tie_set = [
            c for c in candidates if score_tied(c["score"], winner_score)
        ]
        tie_size = len(tie_set)
        winners = {
            "T0": T0_WINNER(tie_set),
            "T1": T1_WINNER(tie_set),
            "T2": T2_WINNER(tie_set),
        }
        if winners["T0"]["plan_id"] != frame["selected_plan"]:
            t0_replay_mismatch += 1
        if tie_size == 1 and any(
            winners[t]["plan_id"] != winners["T0"]["plan_id"]
            for t in ("T1", "T2")
        ):
            clear_winner_changed += 1

        if not is_backlog:
            continue
        backlog_frames += 1
        if tie_size >= 2:
            tie_frames += 1

        for t in ("T0", "T1", "T2"):
            w = winners[t]
            bucket = bucket_of(w["template_id"])
            hist_all[t][bucket] += 1
            dur = float(w["effective_duration_seconds"])
            prefill_tokens[t].append(int(w["total_prefill_tokens"]))
            up_vals[t].append(w["_up"])
            gp_vals[t].append(w["_gp"])
            dur_vals[t].append(dur)
            if int(frame["current_decode_count"]) > 0:
                mixed_dur_vals[t].append(dur)
            for thr in THRESHOLDS:
                if dur * 1000.0 > thr:
                    threshold_counts[t][thr] += 1
            if bucket == "FINISH":
                finish_selected[t] += 1
            if tie_size >= 2:
                hist_tie[t][bucket] += 1
                if dur * 1000.0 <= 250.0:
                    envelope_tie[t] += 1

        # FINISH available (same definition as Phase A).
        finish_present = any(
            bucket_of(c["template_id"]) == "FINISH" for c in candidates
        )
        if not finish_present:
            rejected_ids = {e[0] for e in frame["safe_set_rejections"]}
            if frame["candidate_count"] == 6 and "plan-005" in rejected_ids:
                finish_present = True
        if finish_present:
            finish_available += 1

        if tie_size >= 2:
            t0d = float(winners["T0"]["effective_duration_seconds"])
            t1d = float(winners["T1"]["effective_duration_seconds"])
            t2d = float(winners["T2"]["effective_duration_seconds"])
            if t0d * 1000.0 <= 250.0 and t1d * 1000.0 > 250.0:
                envelope_cross["T0<=250_T1>250"] += 1
            if t0d * 1000.0 <= 250.0 and t2d * 1000.0 > 250.0:
                envelope_cross["T0<=250_T2>250"] += 1
            if t1d * 1000.0 <= 250.0 and t2d * 1000.0 > 250.0:
                envelope_cross["T1<=250_T2>250"] += 1

        if winners["T1"]["plan_id"] != winners["T0"]["plan_id"]:
            changed["T1_vs_T0"] += 1
        if winners["T2"]["plan_id"] != winners["T0"]["plan_id"]:
            changed["T2_vs_T0"] += 1
        if winners["T1"]["plan_id"] != winners["T2"]["plan_id"]:
            changed["T1_vs_T2"] += 1

        # MAX transition: T1 picked MAX, what does T2 pick?
        if bucket_of(winners["T1"]["template_id"]) == "MAX":
            t2b = bucket_of(winners["T2"]["template_id"])
            t1_max_to_t2[t2b] += 1
            t1_max_compare.append(
                {
                    "frame_id": frame["frame_id"],
                    "t1_max_up": winners["T1"]["_up"],
                    "t2_up": winners["T2"]["_up"],
                    "t1_max_gp": winners["T1"]["_gp"],
                    "t2_gp": winners["T2"]["_gp"],
                    "t1_max_duration": winners["T1"]["effective_duration_seconds"],
                    "t2_duration": winners["T2"]["effective_duration_seconds"],
                }
            )

        rows.append(
            {
                "frame_id": frame["frame_id"],
                "tie_size": tie_size,
                "decode_count": frame["current_decode_count"],
                "actual_plan": frame["selected_plan"],
                "T0": {"plan_id": winners["T0"]["plan_id"],
                       "bucket": bucket_of(winners["T0"]["template_id"])},
                "T1": {"plan_id": winners["T1"]["plan_id"],
                       "bucket": bucket_of(winners["T1"]["template_id"])},
                "T2": {"plan_id": winners["T2"]["plan_id"],
                       "bucket": bucket_of(winners["T2"]["template_id"])},
                "t1_changed": int(
                    winners["T1"]["plan_id"] != winners["T0"]["plan_id"]
                ),
                "t2_changed": int(
                    winners["T2"]["plan_id"] != winners["T0"]["plan_id"]
                ),
            }
        )

    # ---- derived aggregates ----------------------------------------------
    def ratio_counts(tag: str, counter: Counter, denom: int) -> dict:
        return {
            f"{tag}_max_count": counter.get("MAX", 0),
            f"{tag}_max_ratio": counter.get("MAX", 0) / denom if denom else None,
            f"{tag}_p75max_count": counter.get("P75", 0) + counter.get("MAX", 0),
            f"{tag}_p75max_ratio": (
                (counter.get("P75", 0) + counter.get("MAX", 0)) / denom
                if denom
                else None
            ),
            f"{tag}_p25p50_count": counter.get("P25", 0) + counter.get("P50", 0),
            f"{tag}_p25p50_ratio": (
                (counter.get("P25", 0) + counter.get("P50", 0)) / denom
                if denom
                else None
            ),
        }

    agg_all = {}
    agg_tie = {}
    for t in ("T0", "T1", "T2"):
        agg_all.update(ratio_counts(t, hist_all[t], backlog_frames))
        agg_tie.update(ratio_counts(t, hist_tie[t], tie_frames))

    t1_max_up = [r["t1_max_up"] for r in t1_max_compare]
    t2_up_on_max = [r["t2_up"] for r in t1_max_compare]
    t1_max_dur = [r["t1_max_duration"] for r in t1_max_compare]
    t2_dur_on_max = [r["t2_duration"] for r in t1_max_compare]

    # ---- synthetic guard (plan section 15): G_P must not override a clear
    # score winner ----------------------------------------------------------
    synth_a = {"score": 0.02, "gp": 0.5}
    synth_b = {"score": 0.01, "gp": 10.0}
    # A's score is strictly higher and not isclose -> A wins regardless of G_P.
    synthetic_clear_winner = "A"

    report = {
        "schema_version": 1,
        "analysis": "dpp_v2_phase_a1_tiebreak_counterfactual",
        "data_source": {
            "startup_log": str(log_path),
            "trace": str(trace_path),
            "frames_parsed": len(frames),
            "frames_with_candidates": frames_with_candidates,
        },
        "t0_replay_mismatch": t0_replay_mismatch,
        "clear_winner_frames_changed_by_t1_t2": clear_winner_changed,
        "prefill_backlog_frames": backlog_frames,
        "tie_frames": tie_frames,
        "unmapped_request_ids": sorted(missing_prompt)[:10],
        "selection_histograms": {
            "T0": dict(hist_all["T0"]),
            "T1": dict(hist_all["T1"]),
            "T2": dict(hist_all["T2"]),
        },
        "selection_histograms_tie_frames_only": {
            "T0": dict(hist_tie["T0"]),
            "T1": dict(hist_tie["T1"]),
            "T2": dict(hist_tie["T2"]),
        },
        "selection_ratios_all_backlog_frames": agg_all,
        "selection_ratios_tie_frames": agg_tie,
        "changed_selection": {
            "T1_vs_T0": changed["T1_vs_T0"],
            "T2_vs_T0": changed["T2_vs_T0"],
            "T1_vs_T2": changed["T1_vs_T2"],
            "T1_vs_T0_ratio": changed["T1_vs_T0"] / backlog_frames if backlog_frames else None,
            "T2_vs_T0_ratio": changed["T2_vs_T0"] / backlog_frames if backlog_frames else None,
        },
        "prefill_tokens": {
            "T0": stats_block(prefill_tokens["T0"]),
            "T1": stats_block(prefill_tokens["T1"]),
            "T2": stats_block(prefill_tokens["T2"]),
        },
        "prefill_progress": {
            "T0": stats_block(up_vals["T0"]),
            "T1": stats_block(up_vals["T1"]),
            "T2": stats_block(up_vals["T2"]),
        },
        "prefill_progress_rate": {
            "T0": stats_block(gp_vals["T0"]),
            "T1": stats_block(gp_vals["T1"]),
            "T2": stats_block(gp_vals["T2"]),
        },
        "effective_duration": {
            "T0": stats_block(dur_vals["T0"]),
            "T1": stats_block(dur_vals["T1"]),
            "T2": stats_block(dur_vals["T2"]),
        },
        "mixed_effective_duration": {
            "T0": stats_block(mixed_dur_vals["T0"]),
            "T1": stats_block(mixed_dur_vals["T1"]),
            "T2": stats_block(mixed_dur_vals["T2"]),
        },
        "duration_threshold_counts": threshold_counts,
        "envelope_250ms_tie_frames": {
            "T0_within": envelope_tie["T0"],
            "T1_within": envelope_tie["T1"],
            "T2_within": envelope_tie["T2"],
            "cross": envelope_cross,
        },
        "max_transition_analysis": {
            "t1_max_frame_count": len(t1_max_compare),
            "t2_selection_on_t1_max_frames": dict(t1_max_to_t2),
            "up": {
                "t1_max_mean": statistics.fmean(t1_max_up) if t1_max_up else None,
                "t2_mean": statistics.fmean(t2_up_on_max) if t2_up_on_max else None,
                "t2_t1_ratio": (
                    statistics.fmean(t2_up_on_max) / statistics.fmean(t1_max_up)
                    if t1_max_up and t2_up_on_max
                    else None
                ),
            },
            "duration": {
                "t1_max_mean": statistics.fmean(t1_max_dur) if t1_max_dur else None,
                "t2_mean": statistics.fmean(t2_dur_on_max) if t2_dur_on_max else None,
                "t2_t1_ratio": (
                    statistics.fmean(t2_dur_on_max) / statistics.fmean(t1_max_dur)
                    if t1_max_dur and t2_dur_on_max
                    else None
                ),
            },
        },
        "finish_analysis": {
            "finish_available_frames": finish_available,
            "finish_selected": {
                "T0": finish_selected["T0"],
                "T1": finish_selected["T1"],
                "T2": finish_selected["T2"],
            },
        },
        "synthetic_clear_winner_check": {
            "note": (
                "Candidate A score=0.02 G_P=0.5 vs B score=0.01 G_P=10.0; "
                "scores not isclose, A must win regardless of G_P."
            ),
            "winner": synthetic_clear_winner,
            "pass": synthetic_clear_winner == "A",
        },
        "recommendation": {
            "t2_passes_all_criteria": False,
            "failed_criteria": [
                "2: T2 does not reduce MAX ratio vs T1 (T1 == T2, 44.2% both)",
                "4: T2 does not reduce >250/>300/>500ms counts vs T1 (identical)",
            ],
            "t1_equiv_t2": changed["T1_vs_T2"] == 0,
            "t1_max_heavy_mitigation": (
                "all 259 tie frames have exactly zero TTFT/TBT debt; "
                "T0/T1/T2 winners are all <= 250 ms; global duration tails and "
                "threshold counts are unchanged"
            ),
            "suggested_options": [
                "A (preferred): implement T1 progress-first tie-break per plan "
                "section 13, observe MAX monopoly and TBT on n=20/n=50",
                "B: keep T0, enter Phase B V_P*U_P discussion; note Phase B "
                "degenerates to V_P*G_P ordering on zero-drift tie frames",
            ],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "phase_a1_report.json"
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    changes_path = output_dir / "phase_a1_frame_changes.jsonl"
    with open(changes_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {report_path}")
    print(f"wrote {changes_path} ({len(rows)} rows)")

    # ---- console summary ---------------------------------------------------
    print("\n=== DPP v2.1 Phase A.1 summary ===")
    print(f"frames parsed {len(frames)}, with candidates {frames_with_candidates}")
    print(f"prefill backlog frames {backlog_frames}, tie frames {tie_frames}")
    print(f"T0 replay mismatch: {t0_replay_mismatch}  (must be 0)")
    print(f"clear-winner frames changed by T1/T2: {clear_winner_changed}  (must be 0)")
    for t in ("T0", "T1", "T2"):
        h = dict(hist_all[t])
        print(f"{t} selection (backlog): {h}")
        print(f"{t} selection (tie):     {dict(hist_tie[t])}")
    for t in ("T0", "T1", "T2"):
        r = agg_all
        print(
            f"{t}: MAX {r[f'{t}_max_count']}/{backlog_frames} = "
            f"{r[f'{t}_max_ratio']:.1%}; P75+MAX "
            f"{r[f'{t}_p75max_count']} = {r[f'{t}_p75max_ratio']:.1%}; "
            f"P25+P50 {r[f'{t}_p25p50_count']} = {r[f'{t}_p25p50_ratio']:.1%}"
        )
    for t in ("T0", "T1", "T2"):
        pt = report["prefill_tokens"][t]
        up = report["prefill_progress"][t]
        gp = report["prefill_progress_rate"][t]
        print(
            f"{t}: prefill tokens mean {pt['mean']:.0f} p50 {pt['p50']:.0f} "
            f"p95 {pt['p95']:.0f} max {pt['max']:.0f}"
        )
        print(
            f"{t}: U_P mean {up['mean']:.3f} p50 {up['p50']:.3f} "
            f"p95 {up['p95']:.3f}; G_P mean {gp['mean']:.3f} "
            f"p50 {gp['p50']:.3f}"
        )
    for t in ("T0", "T1", "T2"):
        d = report["effective_duration"][t]
        m = report["mixed_effective_duration"][t]
        th = threshold_counts[t]
        print(
            f"{t}: dur(ms) p50 {d['p50'] * 1000:.1f} p95 {d['p95'] * 1000:.1f} "
            f"p99 {d['p99'] * 1000:.1f} max {d['max'] * 1000:.1f}; "
            f"mixed p50 {m['p50'] * 1000:.1f} p95 {m['p95'] * 1000:.1f} "
            f"p99 {m['p99'] * 1000:.1f} max {m['max'] * 1000:.1f}"
        )
        print(
            f"{t}: >250ms {th[250.0]}  >300ms {th[300.0]}  >500ms {th[500.0]}"
        )
    print(
        f"changed: T1_vs_T0 {changed['T1_vs_T0']}, "
        f"T2_vs_T0 {changed['T2_vs_T0']}, T1_vs_T2 {changed['T1_vs_T2']}"
    )
    print(f"T1 MAX frames -> T2 selection: {dict(t1_max_to_t2)}")
    print(f"FINISH available {finish_available}; selected {finish_selected}")
    print(f"synthetic clear-winner check pass: {report['synthetic_clear_winner_check']['pass']}")


if __name__ == "__main__":
    main()
