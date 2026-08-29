#!/usr/bin/env python3
"""Offline counterfactual replay for Stage-1 V2-A: ZERO-relative ΔN_TBT = 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def higher_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = math.ceil(q * (len(ordered) - 1))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "p50": higher_quantile(values, 0.50),
        "p90": higher_quantile(values, 0.90),
        "p95": higher_quantile(values, 0.95),
        "p99": higher_quantile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def is_zero_template(candidate: dict[str, Any]) -> bool:
    value = str(candidate.get("template_id", ""))
    return value == "ZERO" or value.startswith("ZERO:")


def is_stock_template(candidate: dict[str, Any]) -> bool:
    value = str(candidate.get("template_id", ""))
    return value == "STOCK" or value.startswith("STOCK:")


def plan_service_tokens(candidate: dict[str, Any]) -> int:
    return int(candidate["plan"]["total_prefill_tokens"])


def plan_decode_ids(candidate: dict[str, Any]) -> set[str]:
    return {str(x) for x in candidate["plan"]["decode_items"]}


def stage2_effective_duration(candidate: dict[str, Any]) -> float:
    """Preserve current V1 Stage-2 denominator semantics."""
    duration = candidate["duration"]
    if bool(duration["in_support"]) and duration["prediction_mode"] == "INTERPOLATION":
        return finite_positive(duration["expected"], "expected duration")
    if (
        not bool(duration["in_support"])
        and duration["prediction_mode"] == "CONSTRAINED_EXTRAPOLATION"
    ):
        return finite_positive(duration["conservative"], "conservative duration")
    raise ValueError("prediction support flag/mode mismatch")


def stage1_risk_duration(candidate: dict[str, Any]) -> float:
    """V2-A target semantics: Stage 1 always uses conservative duration."""
    return finite_positive(candidate["duration"]["conservative"], "risk duration")


def resolve_zero_reference(
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    candidates = list(record["candidates"])
    active_decode_count = int(record["state"]["active_decode_count"])

    exact_zero = [
        c
        for c in candidates
        if is_zero_template(c)
        and plan_service_tokens(c) == 0
        and len(plan_decode_ids(c)) == active_decode_count
    ]
    if exact_zero:
        return sorted(exact_zero, key=lambda c: str(c["plan_id"]))[0], "ZERO_TEMPLATE"

    # Schema v3 does not store the full active-decode ID set in state.
    # This fallback is only for canonical-dedup cases; report its use explicitly.
    zero_service_full_count = [
        c
        for c in candidates
        if plan_service_tokens(c) == 0
        and len(plan_decode_ids(c)) == active_decode_count
    ]
    if zero_service_full_count:
        return (
            sorted(zero_service_full_count, key=lambda c: str(c["plan_id"]))[0],
            "ZERO_SERVICE_COUNT_MATCH_FALLBACK",
        )
    return None, "MISSING"


def obligation_miss(
    *,
    timestamp: float,
    deadline: float,
    request_id: str,
    decode_ids: set[str],
    duration: float,
) -> bool:
    """Match consequence_estimator._misses TBT semantics."""
    end = timestamp + duration
    if request_id in decode_ids:
        return end > deadline
    return end >= deadline


def candidate_risk(
    record: dict[str, Any],
    candidate: dict[str, Any],
    duration: float,
) -> tuple[dict[str, bool], dict[str, float]]:
    timestamp = float(record["state"]["timestamp"])
    decode_ids = plan_decode_ids(candidate)
    misses: dict[str, bool] = {}
    lateness: dict[str, float] = {}
    for item in record["state"]["tbt_request_slacks"]:
        request_id = str(item["request_id"])
        deadline = float(item["deadline"])
        misses[request_id] = obligation_miss(
            timestamp=timestamp,
            deadline=deadline,
            request_id=request_id,
            decode_ids=decode_ids,
            duration=duration,
        )
        lateness[request_id] = max(0.0, timestamp + duration - deadline)
    return misses, lateness


def delta_risk(
    zero_miss: dict[str, bool],
    zero_lateness: dict[str, float],
    cand_miss: dict[str, bool],
    cand_lateness: dict[str, float],
) -> tuple[int, float]:
    request_ids = set(zero_miss) | set(cand_miss)
    delta_n = sum(
        1
        for request_id in request_ids
        if cand_miss.get(request_id, False)
        and not zero_miss.get(request_id, False)
    )
    delta_l = sum(
        max(
            0.0,
            cand_lateness.get(request_id, 0.0)
            - zero_lateness.get(request_id, 0.0),
        )
        for request_id in request_ids
    )
    return delta_n, delta_l


def rank_service_rate(
    candidates: list[dict[str, Any]],
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    scores: list[dict[str, Any]] = []
    for candidate in candidates:
        tau = stage2_effective_duration(candidate)
        service_tokens = plan_service_tokens(candidate)
        score = service_tokens / tau
        scores.append(
            {
                "candidate": candidate,
                "plan_id": str(candidate["plan_id"]),
                "score": score,
                "duration": tau,
                "service_tokens": service_tokens,
                "budget": service_tokens,
            }
        )

    remaining = sorted(scores, key=lambda x: (-x["score"], x["plan_id"]))
    ranked: list[dict[str, Any]] = []
    winner_tie: list[str] = []
    while remaining:
        leader = remaining[0]
        group = [
            x
            for x in remaining
            if (x["score"] == 0.0) == (leader["score"] == 0.0)
            and math.isclose(
                x["score"],
                leader["score"],
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        ]
        group_ids = {x["plan_id"] for x in group}
        remaining = [x for x in remaining if x["plan_id"] not in group_ids]
        group.sort(
            key=lambda x: (
                x["duration"],
                -x["service_tokens"],
                x["budget"],
                x["plan_id"],
            )
        )
        if not ranked:
            winner_tie = [x["plan_id"] for x in group]
        ranked.extend(group)
    return ranked, winner_tie


def replay_record(record: dict[str, Any]) -> dict[str, Any]:
    if int(record.get("schema_version", -1)) != 3:
        raise ValueError("Stage1 V2-A replay requires Selector Diagnosis schema v3")

    candidates = list(record["candidates"])
    state = record["state"]
    selector = record["selector"]
    tbt_items = list(state["tbt_request_slacks"])
    backlog_tokens = int(state["waiting_prefill_tokens"])
    rel_tol = float(selector.get("score_rel_tol", 1e-9))
    abs_tol = float(selector.get("score_abs_tol", 1e-12))

    old_selected_id = record["decision"]["selector_selected_plan_id"]
    candidate_by_id = {str(c["plan_id"]): c for c in candidates}
    old_selected = (
        candidate_by_id.get(str(old_selected_id))
        if old_selected_id is not None
        else None
    )
    old_selected_service = (
        plan_service_tokens(old_selected) if old_selected is not None else None
    )

    if not candidates:
        return {
            "frame_id": int(record["frame_id"]),
            "status": "NO_SAFE_CANDIDATES",
            "backlog_tokens": backlog_tokens,
            "active_tbt": bool(tbt_items),
            "old_selected_service": old_selected_service,
        }

    risk_by_id: dict[str, dict[str, Any]] = {}
    baseline_resolution = "NOT_NEEDED"
    zero_ref = None

    if not tbt_items:
        eligible = candidates
        for candidate in candidates:
            risk_by_id[str(candidate["plan_id"])] = {
                "delta_n": 0,
                "delta_lateness_seconds": 0.0,
                "risk_duration_seconds": stage1_risk_duration(candidate),
            }
    else:
        zero_ref, baseline_resolution = resolve_zero_reference(record)
        if zero_ref is None:
            return {
                "frame_id": int(record["frame_id"]),
                "status": "ZERO_REFERENCE_MISSING",
                "backlog_tokens": backlog_tokens,
                "active_tbt": True,
                "old_selected_service": old_selected_service,
                "baseline_resolution": baseline_resolution,
            }

        zero_duration = stage1_risk_duration(zero_ref)
        zero_miss, zero_lateness = candidate_risk(record, zero_ref, zero_duration)
        eligible = []
        for candidate in candidates:
            risk_duration = stage1_risk_duration(candidate)
            cand_miss, cand_lateness = candidate_risk(
                record, candidate, risk_duration
            )
            delta_n, delta_l = delta_risk(
                zero_miss,
                zero_lateness,
                cand_miss,
                cand_lateness,
            )
            risk_by_id[str(candidate["plan_id"])] = {
                "delta_n": delta_n,
                "delta_lateness_seconds": delta_l,
                "risk_duration_seconds": risk_duration,
                "violation_count": sum(cand_miss.values()),
                "zero_violation_count": sum(zero_miss.values()),
            }
            if delta_n == 0:
                eligible.append(candidate)

    if not eligible:
        raise RuntimeError("ΔN=0 eligible set unexpectedly empty")

    ranked, winner_tie = rank_service_rate(
        eligible, rel_tol=rel_tol, abs_tol=abs_tol
    )
    winner_entry = ranked[0]
    winner = winner_entry["candidate"]
    winner_id = winner_entry["plan_id"]
    winner_service = winner_entry["service_tokens"]
    winner_risk = risk_by_id[winner_id]

    stock = next((c for c in candidates if is_stock_template(c)), None)
    stock_id = str(stock["plan_id"]) if stock is not None else None
    stock_risk = risk_by_id.get(stock_id) if stock_id is not None else None
    stock_old_passed = (
        bool(stock["stage1_tbt"]["passed"]) if stock is not None else None
    )

    old_zero_only = bool(
        tbt_items
        and backlog_tokens > 0
        and old_selected_service == 0
        and int(
            record.get("service_rate_diagnosis", {}).get(
                "stage1_eligible_nonzero_candidate_count", 0
            )
        )
        == 0
    )
    released = old_zero_only and winner_service > 0

    template = str(winner.get("template_id", ""))
    return {
        "frame_id": int(record["frame_id"]),
        "status": "OK",
        "backlog_tokens": backlog_tokens,
        "active_tbt": bool(tbt_items),
        "baseline_resolution": baseline_resolution,
        "zero_reference_plan_id": (
            str(zero_ref["plan_id"]) if zero_ref is not None else None
        ),
        "old_selected_plan_id": old_selected_id,
        "old_selected_service": old_selected_service,
        "new_selected_plan_id": winner_id,
        "new_selected_template_id": template,
        "new_selected_service": winner_service,
        "new_selected_score": winner_entry["score"],
        "new_selected_delta_n": int(winner_risk["delta_n"]),
        "new_selected_delta_lateness_seconds": float(
            winner_risk["delta_lateness_seconds"]
        ),
        "new_selected_risk_duration_seconds": float(
            winner_risk["risk_duration_seconds"]
        ),
        "winner_tie_plan_ids": winner_tie,
        "eligible_count": len(eligible),
        "eligible_nonzero_count": sum(
            1 for c in eligible if plan_service_tokens(c) > 0
        ),
        "old_zero_only": old_zero_only,
        "released": released,
        "winner_changed": str(old_selected_id) != winner_id,
        "stock_present": stock is not None,
        "stock_old_passed": stock_old_passed,
        "stock_delta_n": (
            int(stock_risk["delta_n"]) if stock_risk is not None else None
        ),
        "stock_delta_lateness_seconds": (
            float(stock_risk["delta_lateness_seconds"])
            if stock_risk is not None
            else None
        ),
        "stock_new_eligible": (
            bool(stock_risk is not None and stock_risk["delta_n"] == 0)
            if stock is not None
            else None
        ),
        "stock_newly_released": bool(
            stock is not None
            and stock_old_passed is False
            and stock_risk is not None
            and stock_risk["delta_n"] == 0
        ),
        "stock_new_winner": bool(stock_id is not None and winner_id == stock_id),
        "candidate_risk": risk_by_id,
    }


def replay_file(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha.lower() != expected_sha256.lower():
        raise ValueError(
            f"diagnosis SHA256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )

    counts = Counter()
    winner_templates = Counter()
    baseline_resolution = Counter()
    stock_delta_n_hist = Counter()
    released_delta_l: list[float] = []
    winner_delta_l: list[float] = []
    winner_risk_duration: list[float] = []
    winner_service_tokens: list[float] = []
    released_examples: list[dict[str, Any]] = []
    frame_results = 0

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                result = replay_record(record)
            except Exception as error:
                raise RuntimeError(f"replay failed at line {line_number}: {error}") from error

            frame_results += 1
            counts["frames"] += 1
            counts[f"status:{result['status']}"] += 1
            if result.get("active_tbt"):
                counts["active_tbt_frames"] += 1
            if result.get("backlog_tokens", 0) > 0:
                counts["prefill_backlog_frames"] += 1
            if result.get("active_tbt") and result.get("backlog_tokens", 0) > 0:
                counts["active_tbt_and_backlog_frames"] += 1

            if result["status"] != "OK":
                continue

            baseline_resolution[result["baseline_resolution"]] += 1
            if result["winner_changed"]:
                counts["winner_changed_frames"] += 1
            if result["new_selected_service"] == 0:
                counts["new_zero_winner_frames"] += 1
                if result["active_tbt"] and result["backlog_tokens"] > 0:
                    counts["new_zero_winner_active_tbt_backlog_frames"] += 1
            else:
                counts["new_nonzero_winner_frames"] += 1

            if result["old_zero_only"]:
                counts["old_zero_only_active_tbt_backlog_frames"] += 1
                if result["eligible_nonzero_count"] > 0:
                    counts["old_zero_only_with_new_nonzero_eligible"] += 1
                if result["released"]:
                    counts["released_old_zero_only_frames"] += 1
                    released_delta_l.append(
                        result["new_selected_delta_lateness_seconds"]
                    )
                    if len(released_examples) < 50:
                        released_examples.append(
                            {
                                "frame_id": result["frame_id"],
                                "old_selected_plan_id": result["old_selected_plan_id"],
                                "new_selected_plan_id": result["new_selected_plan_id"],
                                "new_selected_template_id": result[
                                    "new_selected_template_id"
                                ],
                                "new_selected_service": result["new_selected_service"],
                                "new_selected_delta_lateness_seconds": result[
                                    "new_selected_delta_lateness_seconds"
                                ],
                                "eligible_nonzero_count": result[
                                    "eligible_nonzero_count"
                                ],
                            }
                        )

            if result["stock_present"]:
                counts["stock_present_frames"] += 1
                if result["stock_new_eligible"]:
                    counts["stock_new_eligible_frames"] += 1
                if result["stock_newly_released"]:
                    counts["stock_newly_released_frames"] += 1
                if result["stock_new_winner"]:
                    counts["stock_new_winner_frames"] += 1
                if result["stock_delta_n"] is not None:
                    stock_delta_n_hist[str(result["stock_delta_n"])] += 1

            winner_templates[result["new_selected_template_id"]] += 1
            winner_delta_l.append(result["new_selected_delta_lateness_seconds"])
            winner_risk_duration.append(result["new_selected_risk_duration_seconds"])
            winner_service_tokens.append(float(result["new_selected_service"]))

    old_zero_only = counts["old_zero_only_active_tbt_backlog_frames"]
    released = counts["released_old_zero_only_frames"]
    active_backlog = counts["active_tbt_and_backlog_frames"]
    summary = {
        "schema_version": 1,
        "replay_algorithm": "stage1_zero_relative_delta_n_zero_v2a",
        "source": {
            "path": str(path),
            "sha256": actual_sha,
            "bytes": path.stat().st_size,
        },
        "counts": dict(counts),
        "rates": {
            "old_zero_only_release_rate": (
                released / old_zero_only if old_zero_only else None
            ),
            "new_zero_rate_with_active_tbt_and_backlog": (
                counts["new_zero_winner_active_tbt_backlog_frames"] / active_backlog
                if active_backlog
                else None
            ),
            "winner_change_rate": (
                counts["winner_changed_frames"] / frame_results
                if frame_results
                else None
            ),
        },
        "baseline_resolution_histogram": dict(baseline_resolution),
        "new_winner_template_histogram": dict(winner_templates),
        "stock_delta_n_histogram": dict(stock_delta_n_hist),
        "winner_delta_lateness_seconds": distribution(winner_delta_l),
        "released_frame_winner_delta_lateness_seconds": distribution(released_delta_l),
        "winner_risk_duration_seconds": distribution(winner_risk_duration),
        "winner_prefill_service_tokens": distribution(winner_service_tokens),
        "released_examples_first_50": released_examples,
        "interpretation_guardrails": [
            "This is a counterfactual replay over recorded candidate predictions; it is not a performance benchmark.",
            "Stage 1 risk uses recorded conservative duration.",
            "Stage 2 ranking preserves the current V1 effective-duration service-rate semantics.",
            "Alternative selections were not actually executed, so replay cannot infer actual TTFT/TBT/E2E.",
        ],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnosis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    summary = replay_file(args.diagnosis, args.expected_sha256)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
