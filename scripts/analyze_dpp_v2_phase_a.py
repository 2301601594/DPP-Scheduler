#!/usr/bin/env python3
"""DPP v2.1 Phase A offline near-tie diagnostic analysis.

Parses the detailed DPP iteration log (DPP_DIAGNOSTIC_ITERATION_LOG=1) from
the n=100 diagnostic run and answers the Phase A questions in
docs/DPP-v2.1-Scheduler-Agent-Plan.md:

  1. Prefill backlog frame statistics.
  2. Winner tie-set analysis (math.isclose score ties).
  3. Zero / near-zero total-drift analysis.
  4. FINISH candidate audit (generated / selected / tied / lost / clear-loser).
  5. Prefill progress utility U_P and rate G_P (diagnostic only).
  6. Progress-first tie-break counterfactual selection distribution.

Local-only, dependency-free analysis of already-collected raw results. It
does not modify Scheduler behavior and does not import project modules.

Usage:
    python3 scripts/analyze_dpp_v2_phase_a.py \
        <startup.log> <trace.jsonl> <output_dir>
"""

from __future__ import annotations

import ast
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

# Frozen Selector isclose tolerances (dpp_scheduler.settings defaults).
SCORE_REL_TOL = 1e-9
SCORE_ABS_TOL = 1e-12
# Diagnostic-only near-zero drift threshold (plan section 5; never a Scheduler
# parameter).
ZERO_DRIFT_ABS = 1e-9

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
            payload = line.split(marker, 1)[1].strip()
            frames.append(ast.literal_eval(payload))
    return frames


def request_prompt_tokens(request_id: str, entries: list[dict]) -> int | None:
    # request id: cmpl-poisson_q0.2_s1001_0000-0-bc042d62
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


def isclose_pair_tie(score: float, winner: float) -> bool:
    return score_tied(score, winner)


def actual_tie_break_key(candidate: dict) -> tuple:
    return (
        -float(candidate["effective_duration_seconds"]),
        -int(candidate["total_prefill_tokens"]),
        str(candidate["plan_id"]),
    )


def counterfactual_tie_break_key(candidate: dict, up: float) -> tuple:
    return (
        up,
        -float(candidate["effective_duration_seconds"]),
        -int(candidate["total_prefill_tokens"]),
        str(candidate["plan_id"]),
    )


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

    # ---- per-frame / per-candidate pass ---------------------------------
    frames_out: list[dict] = []
    missing_prompt: set[str] = set()
    actual_repro_fail = 0
    report_mismatch: list[tuple] = []

    total = Counter()
    backlog = Counter()
    tie_size2 = tie_size3 = 0
    all_near_zero = 0
    near_zero_candidates = 0
    scored_candidates = 0
    finish_generated = 0
    finish_selected = 0
    finish_in_tie = 0
    finish_lost_tiebreak = 0
    finish_clear_loser = 0
    finish_audit: list[dict] = []
    counterfactual = Counter()
    changed = 0
    backlog_frames: list[dict] = []
    selected_up: list[float] = []
    selected_gp: list[float] = []
    counterfactual_up: list[float] = []
    winner_clear_buckets = Counter()
    winner_tied_buckets = Counter()
    tie_common_drift_zero = 0
    clear_margins_rel: list[float] = []
    tie_set_compositions: Counter = Counter()
    finish_loss_drift = 0
    finish_loss_duration = 0

    for frame in frames:
        is_backlog = int(frame["current_prefill_count"]) > 0
        is_mixed = is_backlog and int(frame["current_decode_count"]) > 0
        candidates = frame["candidate_scores"]
        selected_plan = frame["selected_plan"]
        frame_entry = {
            "frame_id": frame["frame_id"],
            "current_prefill_count": frame["current_prefill_count"],
            "current_decode_count": frame["current_decode_count"],
            "sum_ttft_debt": frame["sum_ttft_debt"],
            "max_ttft_debt": frame["max_ttft_debt"],
            "sum_tbt_debt": frame["sum_tbt_debt"],
            "max_tbt_debt": frame["max_tbt_debt"],
            "candidate_count": frame["candidate_count"],
            "safe_candidate_count": frame["safe_candidate_count"],
            "selected_plan": selected_plan,
            "decision_reason": frame["decision_reason"],
            "fallback_reason": frame["fallback_reason"],
            "candidates": [],
        }
        total["all"] += 1
        total[("backlog" if is_backlog else "decode_only")] += 1
        if is_mixed:
            total["mixed"] += 1

        # FINISH generated: safe candidate, or the distinct 6th candidate
        # (plan-005) rejected by Safe-Set.
        finish_present = any(
            bucket_of(c["template_id"]) == "FINISH" for c in candidates
        )
        if not finish_present:
            rejected_ids = {entry[0] for entry in frame["safe_set_rejections"]}
            if frame["candidate_count"] == 6 and "plan-005" in rejected_ids:
                finish_present = True

        if is_backlog:
            backlog["all"] += 1
            if is_mixed:
                backlog["mixed"] += 1
            else:
                backlog["prefill_only"] += 1

        per_candidate: list[dict] = []
        up_by_plan: dict[str, float] = {}
        for cand in candidates:
            up = 0.0
            for request_id, tokens in cand["prefill_items"]:
                p_i = request_prompt_tokens(request_id, entries)
                if p_i is None:
                    missing_prompt.add(request_id)
                    continue
                up += tokens / p_i
            up_by_plan[cand["plan_id"]] = up
            gp = up / cand["effective_duration_seconds"] if up else 0.0
            per_candidate.append(
                {
                    "plan_id": cand["plan_id"],
                    "template_id": cand["template_id"],
                    "bucket": bucket_of(cand["template_id"]),
                    "prefill_tokens": cand["total_prefill_tokens"],
                    "expected_duration": cand["expected_duration_seconds"],
                    "conservative_duration": cand["conservative_duration_seconds"],
                    "effective_duration": cand["effective_duration_seconds"],
                    "prefill_drift": cand["prefill_normalized_drift"],
                    "decode_drift": cand["decode_normalized_drift"],
                    "total_drift": cand["total_normalized_drift"],
                    "score": cand["score"],
                    "selected": cand["selected"],
                    "selection_rank": cand["selection_rank"],
                    "prefill_progress_utility": up,
                    "prefill_progress_rate": gp,
                }
            )
        frame_entry["candidates"] = per_candidate

        if candidates:
            scored_candidates += len(candidates)
            winner_score = max(c["score"] for c in candidates)
            tie_set = [
                c for c in candidates if isclose_pair_tie(c["score"], winner_score)
            ]
            tie_size = len(tie_set)
            up_winner = max(
                tie_set,
                key=lambda c: counterfactual_tie_break_key(
                    c, up_by_plan[c["plan_id"]]
                ),
            )
            actual_repro = max(tie_set, key=actual_tie_break_key)
            if actual_repro["plan_id"] != selected_plan:
                actual_repro_fail += 1
                report_mismatch.append(
                    (frame["frame_id"], selected_plan, actual_repro["plan_id"])
                )
            cf_selected = up_winner["plan_id"]

            if is_backlog:
                sel_bucket = bucket_of(
                    next(c["template_id"] for c in candidates if c["selected"])
                )
                if tie_size >= 2:
                    tie_size2 += 1
                    winner_tied_buckets[sel_bucket] += 1
                    common = {
                        repr(c["total_normalized_drift"]) for c in tie_set
                    }
                    if len(common) == 1 and abs(float(next(iter(common)))) <= 1e-12:
                        tie_common_drift_zero += 1
                    tie_set_compositions[
                        tuple(
                            sorted(bucket_of(c["template_id"]) for c in tie_set)
                        )
                    ] += 1
                if tie_size >= 3:
                    tie_size3 += 1
                if tie_size == 1:
                    winner_clear_buckets[sel_bucket] += 1
                    runner_up = sorted(
                        (c["score"] for c in candidates), reverse=True
                    )[1]
                    if winner_score != 0.0:
                        clear_margins_rel.append(
                            (winner_score - runner_up) / abs(winner_score)
                        )
                if all(
                    abs(c["total_normalized_drift"]) <= ZERO_DRIFT_ABS
                    for c in candidates
                ):
                    all_near_zero += 1
                near_zero_candidates += sum(
                    1
                    for c in candidates
                    if abs(c["total_normalized_drift"]) <= ZERO_DRIFT_ABS
                )
                backlog[sel_bucket] += 1
                sel = next(c for c in candidates if c["selected"])
                selected_up.append(up_by_plan[sel["plan_id"]])
                if up_by_plan[sel["plan_id"]] > 0:
                    selected_gp.append(
                        up_by_plan[sel["plan_id"]]
                        / sel["effective_duration_seconds"]
                    )

            # ---- FINISH audit (backlog frames only) ----------------------
            finish_cand = next(
                (c for c in candidates if bucket_of(c["template_id"]) == "FINISH"),
                None,
            )
            if is_backlog:
                if finish_present:
                    finish_generated += 1
                    if finish_cand is None:
                        frame_entry["finish_status"] = "rejected_by_safe_set"
                    else:
                        is_selected = bool(finish_cand["selected"])
                        in_tie = isclose_pair_tie(finish_cand["score"], winner_score)
                        finish_entry = {
                            "frame_id": frame["frame_id"],
                            "finish_rank": finish_cand["selection_rank"],
                            "finish_selected": is_selected,
                            "finish_in_winner_tie": in_tie,
                            "finish_score": finish_cand["score"],
                            "winner_score": winner_score,
                            "selected_plan": selected_plan,
                            "selected_score": next(
                                c["score"] for c in candidates if c["selected"]
                            ),
                            "score_gap": winner_score - finish_cand["score"],
                            "finish_effective_duration": finish_cand[
                                "effective_duration_seconds"
                            ],
                            "selected_effective_duration": next(
                                c["effective_duration_seconds"]
                                for c in candidates
                                if c["selected"]
                            ),
                            "finish_prefill_drift": finish_cand[
                                "prefill_normalized_drift"
                            ],
                            "selected_prefill_drift": next(
                                c["prefill_normalized_drift"]
                                for c in candidates
                                if c["selected"]
                            ),
                            "finish_decode_drift": finish_cand[
                                "decode_normalized_drift"
                            ],
                            "selected_decode_drift": next(
                                c["decode_normalized_drift"]
                                for c in candidates
                                if c["selected"]
                            ),
                            "finish_up": up_by_plan[finish_cand["plan_id"]],
                            "finish_gp": (
                                up_by_plan[finish_cand["plan_id"]]
                                / finish_cand["effective_duration_seconds"]
                            ),
                        }
                        finish_audit.append(finish_entry)
                        if is_selected:
                            finish_selected += 1
                        elif in_tie:
                            finish_in_tie += 1
                            finish_lost_tiebreak += 1
                            frame_entry["finish_status"] = "tied_lost"
                        else:
                            finish_clear_loser += 1
                            frame_entry["finish_status"] = "clear_loser"
                            sel_cand = next(c for c in candidates if c["selected"])
                            if finish_cand["total_normalized_drift"] > sel_cand[
                                "total_normalized_drift"
                            ]:
                                finish_loss_drift += 1
                            else:
                                finish_loss_duration += 1

            # ---- counterfactual ------------------------------------------
            cf_cand = next(
                c for c in candidates if c["plan_id"] == up_winner["plan_id"]
            )
            cf_bucket = bucket_of(cf_cand["template_id"])
            if is_backlog:
                counterfactual[cf_bucket] += 1
                if cf_selected != selected_plan:
                    changed += 1
                counterfactual_up.append(up_by_plan[cf_cand["plan_id"]])
            if is_backlog:
                frame_entry["counterfactual_plan"] = cf_selected
                frame_entry["counterfactual_bucket"] = cf_bucket
                frame_entry["winner_tie_size"] = tie_size

        frames_out.append(frame_entry)
        if is_backlog:
            backlog_frames.append(frame_entry)

    # ---- report -----------------------------------------------------------
    report = {
        "schema_version": 1,
        "analysis": "dpp_v2_phase_a_near_tie",
        "source": {
            "startup_log": str(log_path),
            "trace": str(trace_path),
            "frames_parsed": len(frames),
        },
        "validation": {
            "actual_selection_reproduced": actual_repro_fail == 0,
            "actual_selection_mismatches": report_mismatch[:10],
            "unmapped_request_ids": sorted(missing_prompt)[:10],
            "unmapped_request_count": len(missing_prompt),
        },
        "1_prefill_backlog_frames": {
            "total_frames": total["all"],
            "prefill_backlog_frames": total["backlog"],
            "prefill_only_frames": total["prefill_only"],
            "mixed_frames": total["mixed"],
            "decode_only_frames": total["decode_only"],
        },
        "2_winner_ties": {
            "backlog_frames_with_tie_size_ge_2": tie_size2,
            "backlog_frames_with_tie_size_ge_3": tie_size3,
            "tie_frame_ratio": (
                tie_size2 / total["backlog"] if total["backlog"] else None
            ),
            "tie_set_common_drift_exactly_zero": tie_common_drift_zero,
            "tie_set_compositions": {str(k): v for k, v in tie_set_compositions.items()},
            "tie_frame_actual_selection_histogram": dict(winner_tied_buckets),
            "clear_winner_frame_selection_histogram": dict(winner_clear_buckets),
            "clear_winner_rel_margin_p50": (
                statistics.median(clear_margins_rel) if clear_margins_rel else None
            ),
        },
        "3_near_zero_drift": {
            "all_candidates_near_zero_frames": all_near_zero,
            "all_candidates_near_zero_frame_ratio": (
                all_near_zero / total["backlog"] if total["backlog"] else None
            ),
            "near_zero_candidate_count": near_zero_candidates,
            "near_zero_candidate_ratio": (
                near_zero_candidates / scored_candidates
                if scored_candidates
                else None
            ),
        },
        "4_finish_audit": {
            "finish_generated_frames": finish_generated,
            "finish_selected_frames": finish_selected,
            "finish_available_but_rejected": finish_generated - len(finish_audit),
            "finish_tied_with_winner": finish_in_tie,
            "finish_lost_on_tiebreak": finish_lost_tiebreak,
            "finish_clear_loser": finish_clear_loser,
            "finish_clear_loser_loss_driven_by_worse_drift": finish_loss_drift,
            "finish_clear_loser_loss_driven_by_duration": finish_loss_duration,
            "finish_audit_rows": finish_audit,
        },
        "5_prefill_progress": {
            "selected_up_mean": statistics.fmean(selected_up) if selected_up else None,
            "selected_up_p50": statistics.median(selected_up) if selected_up else None,
            "selected_gp_mean": statistics.fmean(selected_gp) if selected_gp else None,
            "selected_gp_p50": statistics.median(selected_gp) if selected_gp else None,
            "selected_gp_p90": (
                sorted(selected_gp)[int(0.9 * len(selected_gp))] if selected_gp else None
            ),
            "counterfactual_selected_up_mean": (
                statistics.fmean(counterfactual_up) if counterfactual_up else None
            ),
        },
        "6_actual_selection": {
            "all_frames_histogram": dict(total),
            "backlog_frames_selection_histogram": dict(backlog),
        },
        "7_counterfactual_progress_first": {
            "backlog_frames_selection_histogram": dict(counterfactual),
            "changed_frames": changed,
            "changed_frame_ratio": (
                changed / total["backlog"] if total["backlog"] else None
            ),
        },
        "8_backlog_frame_detail": backlog_frames,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "phase_a_report.json"
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {report_path}")

    # ---- console summary ---------------------------------------------------
    b = report["1_prefill_backlog_frames"]
    t = report["2_winner_ties"]
    n = report["3_near_zero_drift"]
    f = report["4_finish_audit"]
    p = report["5_prefill_progress"]
    a = report["6_actual_selection"]
    c = report["7_counterfactual_progress_first"]
    print("\n=== DPP v2.1 Phase A summary ===")
    print(f"frames parsed: {b['total_frames']}")
    print(f"prefill backlog frames: {b['prefill_backlog_frames']}"
          f" (prefill-only {b['prefill_only_frames']}, mixed {b['mixed_frames']},"
          f" decode-only {b['decode_only_frames']})")
    print(f"winner tie size>=2: {t['backlog_frames_with_tie_size_ge_2']}"
          f" ({t['tie_frame_ratio']:.1%}); size>=3: {t['backlog_frames_with_tie_size_ge_3']}")
    print(f"tie sets with common drift exactly zero: {t['tie_set_common_drift_exactly_zero']}")
    print(f"tie-set compositions: {t['tie_set_compositions']}")
    print(f"tie-frame actual selection: {t['tie_frame_actual_selection_histogram']}")
    print(f"clear-winner frame selection: {t['clear_winner_frame_selection_histogram']}"
          f" (rel margin p50 {t['clear_winner_rel_margin_p50']:.3f})")
    print(f"all-candidates near-zero drift frames: {n['all_candidates_near_zero_frames']}"
          f" ({n['all_candidates_near_zero_frame_ratio']:.1%});"
          f" near-zero candidate ratio {n['near_zero_candidate_ratio']:.1%}")
    print(f"FINISH generated {f['finish_generated_frames']}, selected {f['finish_selected_frames']},"
          f" tied {f['finish_tied_with_winner']}, lost-on-tiebreak {f['finish_lost_on_tiebreak']},"
          f" clear-loser {f['finish_clear_loser']}"
          f" (loss: drift-driven {f['finish_clear_loser_loss_driven_by_worse_drift']},"
          f" duration-driven {f['finish_clear_loser_loss_driven_by_duration']})")
    print(f"selected U_P mean {p['selected_up_mean']:.3f} p50 {p['selected_up_p50']:.3f};"
          f" G_P mean {p['selected_gp_mean']:.4f} p50 {p['selected_gp_p50']:.4f};"
          f" counterfactual selected U_P mean {p['counterfactual_selected_up_mean']:.3f}")
    print(f"actual selection (backlog frames): {a['backlog_frames_selection_histogram']}")
    print(f"counterfactual selection (backlog frames): {c['backlog_frames_selection_histogram']}")
    print(f"changed frames: {c['changed_frames']} ({c['changed_frame_ratio']:.1%})")
    print(f"actual-selection reproduction mismatches: {report['validation']['actual_selection_mismatches']}")
    print(f"unmapped request ids: {report['validation']['unmapped_request_count']}")


if __name__ == "__main__":
    main()
