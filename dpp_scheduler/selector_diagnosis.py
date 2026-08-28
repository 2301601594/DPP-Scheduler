"""Replayable JSONL diagnostics for the two-stage DPP Selector."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, TextIO

from dpp_scheduler.contracts import ControlState, Decision, SafeCandidate, StateSnapshot
from dpp_scheduler.dpp_selector import SelectorAudit
from dpp_scheduler.settings import DPPSettings


DPP_SELECTOR_DIAGNOSIS_ENV = "DPP_SELECTOR_DIAGNOSIS"
DPP_SELECTOR_DIAGNOSIS_PATH_ENV = "DPP_SELECTOR_DIAGNOSIS_PATH"
SELECTOR_DIAGNOSIS_SCHEMA_VERSION = 3
SUPPORTED_SELECTOR_DIAGNOSIS_SCHEMA_VERSIONS = frozenset({1, 2, 3})


def _is_zero_template(template_id: object) -> bool:
    value = str(template_id)
    return value == "ZERO" or value.startswith("ZERO:")


def _is_stock_template(template_id: object) -> bool:
    value = str(template_id)
    return value == "STOCK" or value.startswith("STOCK:")


def resolve_selector_diagnosis(
    settings: DPPSettings,
    *,
    selection_mode: str,
    environment: Mapping[str, str],
) -> tuple[bool, Path | None]:
    raw_enabled = environment.get(
        DPP_SELECTOR_DIAGNOSIS_ENV,
        "1" if settings.diagnosis_enabled_default else "0",
    )
    if raw_enabled not in {"0", "1"}:
        raise ValueError(f"{DPP_SELECTOR_DIAGNOSIS_ENV} must be 0 or 1")
    enabled = raw_enabled == "1"
    raw_path = environment.get(DPP_SELECTOR_DIAGNOSIS_PATH_ENV)
    if enabled and not raw_path:
        raise ValueError(
            f"{DPP_SELECTOR_DIAGNOSIS_PATH_ENV} is required when Selector "
            "diagnosis is enabled"
        )
    if not enabled and raw_path:
        raise ValueError(
            f"{DPP_SELECTOR_DIAGNOSIS_PATH_ENV} requires "
            f"{DPP_SELECTOR_DIAGNOSIS_ENV}=1"
        )
    if enabled and selection_mode != "normal":
        raise ValueError("Selector diagnosis requires normal DPP selection mode")
    return enabled, Path(raw_path) if raw_path else None


class SelectorDiagnosisWriter:
    """Exclusively create and flush one replayable record per Selector frame."""

    def __init__(
        self,
        path: Path,
        *,
        config_sha256: str,
        predictor_version: str,
        schema_version: int,
    ) -> None:
        if schema_version != SELECTOR_DIAGNOSIS_SCHEMA_VERSION:
            raise ValueError("Selector diagnosis schema version mismatch")
        self.path = path
        self.config_sha256 = config_sha256
        self.predictor_version = predictor_version
        self.schema_version = schema_version
        self._stream: TextIO = path.open("x", encoding="utf-8")

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def write(
        self,
        *,
        snapshot: StateSnapshot,
        control: ControlState,
        safe_candidates: tuple[SafeCandidate, ...],
        audit: SelectorAudit,
        selector_decision: Decision,
        controller_decision: Decision,
        executed_plan_id: str | None,
    ) -> dict[str, Any]:
        selector_plan_id = (
            selector_decision.selected_plan.plan_id
            if selector_decision.selected_plan is not None
            else None
        )
        if (
            selector_decision.snapshot_hash != snapshot.snapshot_hash
            or controller_decision.snapshot_hash != snapshot.snapshot_hash
        ):
            raise ValueError("diagnosis Decision snapshot_hash mismatch")
        if (
            selector_plan_id != audit.selected_plan_id
            or selector_decision.reason != audit.decision_reason
        ):
            raise ValueError("diagnosis Selector audit/Decision mismatch")
        stage1_by_id = {item.plan_id: item for item in audit.stage1.candidates}
        stage2_by_id = {item.plan_id: item for item in audit.stage2_scores}
        candidates: list[dict[str, Any]] = []
        for candidate in safe_candidates:
            plan = candidate.plan
            prediction = candidate.prediction
            stage1 = stage1_by_id[plan.plan_id]
            stage2 = stage2_by_id.get(plan.plan_id)
            stage2_payload = asdict(stage2) if stage2 is not None else None
            if stage2_payload is not None:
                stage2_payload["tie_break_key"] = {
                    "prefill_service_rate_desc": stage2.prefill_service_rate,
                    "effective_duration_asc": stage2.effective_duration,
                    "prefill_service_tokens_desc": stage2.prefill_service_tokens,
                    "prefill_budget_asc": stage2.prefill_budget,
                    "plan_id_asc": stage2.plan_id,
                }
            candidates.append(
                {
                    "plan_id": plan.plan_id,
                    "template_id": plan.template_id,
                    "plan": {
                        "prefill_items": list(plan.prefill_items),
                        "decode_items": list(plan.decode_items),
                        "total_prefill_tokens": plan.total_prefill_tokens,
                        "total_decode_tokens": plan.total_decode_tokens,
                    },
                    "prediction": asdict(prediction),
                    "duration": {
                        "expected": prediction.expected_duration,
                        "conservative": prediction.conservative_duration,
                        "effective": stage1.effective_duration,
                        "in_support": prediction.in_support,
                        "prediction_mode": prediction.prediction_mode,
                    },
                    "stage1_tbt": asdict(stage1),
                    "stage2_prefill_service_rate": stage2_payload,
                    "selected": audit.selected_plan_id == plan.plan_id,
                }
            )
        backlog_tokens = sum(
            request.remaining_tokens
            for request in snapshot.waiting_prefill_requests
        )
        eligible_scores = list(audit.stage2_scores)
        eligible_nonzero = [
            score for score in eligible_scores if score.prefill_service_tokens > 0
        ]
        selected_score = stage2_by_id.get(audit.selected_plan_id)
        selected_is_zero = bool(
            selected_score is not None
            and selected_score.prefill_service_tokens == 0
        )
        zero_with_eligible_nonzero = bool(selected_is_zero and eligible_nonzero)
        if zero_with_eligible_nonzero:
            raise RuntimeError(
                "Selector diagnosis detected ZERO with eligible Prefill service"
            )
        service_rate_diagnosis = {
            "prefill_backlog_count": len(snapshot.waiting_prefill_requests),
            "prefill_backlog_tokens": backlog_tokens,
            "stage1_eligible_candidate_count": len(eligible_scores),
            "stage1_eligible_nonzero_candidate_count": len(eligible_nonzero),
            "selected_plan_id": audit.selected_plan_id,
            "selected_prefill_service_tokens": (
                selected_score.prefill_service_tokens
                if selected_score is not None
                else None
            ),
            "selected_prefill_service_rate": (
                selected_score.prefill_service_rate
                if selected_score is not None
                else None
            ),
            "selected_is_zero": selected_is_zero,
            "zero_with_eligible_nonzero": zero_with_eligible_nonzero,
        }
        record = {
            "schema_version": self.schema_version,
            "frame_id": snapshot.frame_id,
            "snapshot_hash": snapshot.snapshot_hash,
            "config_sha256": self.config_sha256,
            "predictor_version": self.predictor_version,
            "selector_version": audit.algorithm,
            "selector": {
                "algorithm": audit.algorithm,
                "score_rel_tol": 1e-9,
                "score_abs_tol": 1e-12,
                "tbt_delta_seconds": audit.stage1.delta_seconds,
                "tie_break_order": list(audit.tie_break_order),
            },
            "state": {
                "timestamp": snapshot.timestamp,
                "active_decode_count": len(snapshot.active_decode_requests),
                "waiting_prefill_count": len(snapshot.waiting_prefill_requests),
                "waiting_prefill_tokens": backlog_tokens,
                "tbt_request_slacks": [
                    asdict(item) for item in audit.stage1.request_slacks
                ],
                "min_tbt_slack_seconds": audit.stage1.min_slack_seconds,
                "current_ttft_debts": list(control.ttft_service_debts),
                "safe_candidate_count": len(safe_candidates),
            },
            "stage1": asdict(audit.stage1),
            "candidates": candidates,
            "service_rate_diagnosis": service_rate_diagnosis,
            "decision": {
                "selector_selected_plan_id": audit.selected_plan_id,
                "selector_reason": selector_decision.reason,
                "controller_selected_plan_id": (
                    controller_decision.selected_plan.plan_id
                    if controller_decision.selected_plan is not None
                    else None
                ),
                "controller_reason": controller_decision.reason,
                "executed_plan_id": executed_plan_id,
                "winner_tie_plan_ids": list(audit.winner_tie_plan_ids),
                "tie_detected": len(audit.winner_tie_plan_ids) > 1,
                "tie_size": len(audit.winner_tie_plan_ids),
            },
        }
        self._stream.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        self._stream.flush()
        return record


def _close(first: float, second: float, *, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(first, second, rel_tol=rel_tol, abs_tol=abs_tol)


def _rank_candidates(
    scores: list[dict[str, Any]],
    *,
    score_key: str,
    rel_tol: float,
    abs_tol: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    ranked: list[dict[str, Any]] = []
    remaining = sorted(
        scores, key=lambda item: (-float(item[score_key]), item["plan_id"])
    )
    winner_tie: list[str] = []
    while remaining:
        leader = remaining[0]
        group = [
            item
            for item in remaining
            if _close(
                float(item[score_key]),
                float(leader[score_key]),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        ]
        group_ids = {item["plan_id"] for item in group}
        remaining = [
            item for item in remaining if item["plan_id"] not in group_ids
        ]
        group.sort(
            key=lambda item: (
                -item["completed"],
                -item["progress"],
                item["duration"],
                item["budget"],
                item["plan_id"],
            )
        )
        if not ranked:
            winner_tie = [item["plan_id"] for item in group]
        ranked.extend(group)
    return ranked, winner_tie


def _rank_service_candidates(
    scores: list[dict[str, Any]],
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    ranked: list[dict[str, Any]] = []
    remaining = sorted(
        scores, key=lambda item: (-float(item["score"]), item["plan_id"])
    )
    winner_tie: list[str] = []
    while remaining:
        leader = remaining[0]
        group = [
            item
            for item in remaining
            if (
                (float(item["score"]) == 0.0)
                == (float(leader["score"]) == 0.0)
                and _close(
                    float(item["score"]),
                    float(leader["score"]),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
            )
        ]
        group_ids = {item["plan_id"] for item in group}
        remaining = [
            item for item in remaining if item["plan_id"] not in group_ids
        ]
        group.sort(
            key=lambda item: (
                item["duration"],
                -item["service_tokens"],
                item["budget"],
                item["plan_id"],
            )
        )
        if not ranked:
            winner_tie = [item["plan_id"] for item in group]
        ranked.extend(group)
    return ranked, winner_tie


def replay_record(record: Mapping[str, Any]) -> dict[str, int]:
    counters = {
        "stage1_mismatch": 0,
        "ttft_debt_mismatch": 0,
        "stage2_score_mismatch": 0,
        "winner_mismatch": 0,
        "tie_break_mismatch": 0,
        "service_rate_invariant_mismatch": 0,
    }
    schema_version = record.get("schema_version")
    if schema_version not in SUPPORTED_SELECTOR_DIAGNOSIS_SCHEMA_VERSIONS:
        raise ValueError("Selector diagnosis schema version mismatch")
    selector = record["selector"]
    state = record["state"]
    stage1 = record["stage1"]
    candidates = list(record["candidates"])
    decision = record["decision"]
    rel_tol = float(selector["score_rel_tol"])
    abs_tol = float(selector["score_abs_tol"])
    timestamp = float(state["timestamp"])
    delta = float(selector["tbt_delta_seconds"])

    slacks = []
    for item in state["tbt_request_slacks"]:
        replayed = float(item["deadline"]) - timestamp
        slacks.append(replayed)
        if not _close(
            replayed,
            float(item["slack_seconds"]),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            counters["stage1_mismatch"] += 1
    min_slack = min(slacks) if slacks else None
    recorded_min = state["min_tbt_slack_seconds"]
    if (min_slack is None) != (recorded_min is None) or (
        min_slack is not None
        and not _close(
            min_slack, float(recorded_min), rel_tol=rel_tol, abs_tol=abs_tol
        )
    ):
        counters["stage1_mismatch"] += 1

    duration_limit = min_slack + delta if min_slack is not None else None
    recorded_limit = stage1["duration_limit_seconds"]
    if (duration_limit is None) != (recorded_limit is None) or (
        duration_limit is not None
        and not _close(
            duration_limit,
            float(recorded_limit),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
    ):
        counters["stage1_mismatch"] += 1
    replayed_pass: list[str] = []
    effective_by_id: dict[str, float] = {}
    for candidate in candidates:
        duration = candidate["duration"]
        if duration["in_support"] and duration["prediction_mode"] == "INTERPOLATION":
            effective = float(duration["expected"])
        elif (
            not duration["in_support"]
            and duration["prediction_mode"] == "CONSTRAINED_EXTRAPOLATION"
        ):
            effective = float(duration["conservative"])
        else:
            counters["stage1_mismatch"] += 1
            continue
        effective_by_id[str(candidate["plan_id"])] = effective
        stage1_candidate = candidate["stage1_tbt"]
        if not _close(
            effective,
            float(duration["effective"]),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            counters["stage1_mismatch"] += 1
        passed = duration_limit is None or effective <= duration_limit
        if passed:
            replayed_pass.append(str(candidate["plan_id"]))
        if bool(stage1_candidate["passed"]) != passed:
            counters["stage1_mismatch"] += 1

    replayed_status: str
    replayed_eligible: list[str]
    if not candidates:
        replayed_status = "NO_SAFE_CANDIDATES"
        replayed_eligible = []
    elif duration_limit is None:
        replayed_status = "NO_ACTIVE_TBT_OBLIGATION"
        replayed_eligible = replayed_pass
    elif replayed_pass:
        replayed_status = "WITHIN_SLACK"
        replayed_eligible = replayed_pass
    else:
        replayed_status = "NO_CANDIDATE_WITHIN_SLACK"
        fallback = min(
            candidates,
            key=lambda item: (
                effective_by_id[str(item["plan_id"])],
                int(item["plan"]["total_prefill_tokens"]),
                str(item["plan_id"]),
            ),
        )
        replayed_eligible = [str(fallback["plan_id"])]
    if replayed_status != stage1["status"] or replayed_eligible != list(
        stage1["eligible_plan_ids"]
    ):
        counters["stage1_mismatch"] += 1
    replayed_fallback = (
        replayed_eligible[0]
        if replayed_status == "NO_CANDIDATE_WITHIN_SLACK"
        else None
    )
    if replayed_fallback != stage1["fallback_plan_id"]:
        counters["stage1_mismatch"] += 1
    for candidate in candidates:
        stage1_candidate = candidate["stage1_tbt"]
        plan_id = str(candidate["plan_id"])
        selected_by_fallback = plan_id == replayed_fallback
        if bool(stage1_candidate["selected_by_fallback"]) != selected_by_fallback:
            counters["stage1_mismatch"] += 1

    if schema_version == 3:
        replayed_scores: list[dict[str, Any]] = []
        backlog_tokens = int(state["waiting_prefill_tokens"])
        for candidate in candidates:
            plan_id = str(candidate["plan_id"])
            details = candidate["stage2_prefill_service_rate"]
            if plan_id not in replayed_eligible:
                if details is not None:
                    counters["stage2_score_mismatch"] += 1
                continue
            if details is None:
                counters["stage2_score_mismatch"] += 1
                continue
            tau = effective_by_id[plan_id]
            service_tokens = int(candidate["plan"]["total_prefill_tokens"])
            service_rate = service_tokens / tau
            decode_items = list(candidate["plan"]["decode_items"])
            decode_coverage = (
                len(decode_items) == len(set(decode_items))
                and len(decode_items) == int(state["active_decode_count"])
            )
            if (
                int(details["prefill_service_tokens"]) != service_tokens
                or not _close(
                    service_rate,
                    float(details["prefill_service_rate"]),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
                or not _close(
                    service_rate,
                    float(details["score"]),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
                or not _close(
                    tau,
                    float(details["effective_duration"]),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
                or int(details["prefill_budget"]) != service_tokens
                or int(details["current_prefill_count"])
                != int(state["waiting_prefill_count"])
                or int(details["current_prefill_backlog_tokens"])
                != backlog_tokens
                or int(details["current_decode_count"])
                != int(state["active_decode_count"])
                or bool(details["decode_coverage_complete"]) != decode_coverage
            ):
                counters["stage2_score_mismatch"] += 1
            tie_key = details["tie_break_key"]
            if (
                not _close(
                    service_rate,
                    float(tie_key["prefill_service_rate_desc"]),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
                or not _close(
                    tau,
                    float(tie_key["effective_duration_asc"]),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
                or int(tie_key["prefill_service_tokens_desc"])
                != service_tokens
                or int(tie_key["prefill_budget_asc"]) != service_tokens
                or str(tie_key["plan_id_asc"]) != plan_id
            ):
                counters["tie_break_mismatch"] += 1
            replayed_scores.append(
                {
                    "plan_id": plan_id,
                    "score": service_rate,
                    "duration": tau,
                    "service_tokens": service_tokens,
                    "budget": service_tokens,
                    "recorded_rank": int(details["rank"]),
                }
            )

        ranked, winner_tie = _rank_service_candidates(
            replayed_scores, rel_tol=rel_tol, abs_tol=abs_tol
        )
        if any(
            item["recorded_rank"] != rank
            for rank, item in enumerate(ranked, start=1)
        ):
            counters["tie_break_mismatch"] += 1
        if winner_tie != list(decision["winner_tie_plan_ids"]):
            counters["tie_break_mismatch"] += 1
        replayed_winner = ranked[0]["plan_id"] if ranked else None
        if replayed_winner != decision["selector_selected_plan_id"]:
            counters["winner_mismatch"] += 1

        diagnosis = record["service_rate_diagnosis"]
        eligible_nonzero = sum(
            int(item["service_tokens"] > 0) for item in replayed_scores
        )
        selected = ranked[0] if ranked else None
        selected_is_zero = bool(
            selected is not None and selected["service_tokens"] == 0
        )
        invariant = bool(selected_is_zero and eligible_nonzero)
        selected_tokens = selected["service_tokens"] if selected is not None else None
        selected_rate = selected["score"] if selected is not None else None
        recorded_selected_tokens = diagnosis["selected_prefill_service_tokens"]
        recorded_selected_rate = diagnosis["selected_prefill_service_rate"]
        selected_value_mismatch = (
            (selected_tokens is None) != (recorded_selected_tokens is None)
            or (
                selected_tokens is not None
                and int(recorded_selected_tokens) != selected_tokens
            )
            or (selected_rate is None) != (recorded_selected_rate is None)
            or (
                selected_rate is not None
                and not _close(
                    selected_rate,
                    float(recorded_selected_rate),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
            )
        )
        if (
            int(diagnosis["prefill_backlog_count"])
            != int(state["waiting_prefill_count"])
            or int(diagnosis["prefill_backlog_tokens"]) != backlog_tokens
            or int(diagnosis["stage1_eligible_candidate_count"])
            != len(replayed_scores)
            or int(diagnosis["stage1_eligible_nonzero_candidate_count"])
            != eligible_nonzero
            or diagnosis["selected_plan_id"] != replayed_winner
            or bool(diagnosis["selected_is_zero"]) != selected_is_zero
            or bool(diagnosis["zero_with_eligible_nonzero"]) != invariant
            or selected_value_mismatch
        ):
            counters["stage2_score_mismatch"] += 1
        if invariant:
            counters["service_rate_invariant_mismatch"] += 1
        return counters

    prefill_ref = selector["prefill_reference_concurrency"]
    replayed_scores: list[dict[str, Any]] = []
    for candidate in candidates:
        details = candidate["stage2_ttft"]
        if str(candidate["plan_id"]) not in replayed_eligible:
            if details is not None:
                counters["stage2_score_mismatch"] += 1
            continue
        if details is None or prefill_ref is None:
            counters["stage2_score_mismatch"] += 1
            continue
        contributions: list[float] = []
        completed = 0
        progress: list[float] = []
        tau = float(candidate["duration"]["effective"])
        for request in details["request_results"]:
            current = float(request["current_debt"])
            prompt = int(request["prompt_tokens"])
            remaining = int(request["remaining_tokens"])
            scheduled = int(request["scheduled_tokens"])
            slo = float(request["ttft_slo_seconds"])
            duration_increment = tau / slo
            normalized = scheduled / prompt
            completion = scheduled >= remaining
            predicted = (
                0.0
                if completion
                else max(0.0, current + tau / slo - normalized)
            )
            contribution = predicted * predicted - current * current
            contributions.append(contribution)
            progress.append(normalized)
            completed += int(completion)
            if not _close(
                duration_increment,
                float(request["duration_increment"]),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            ):
                counters["ttft_debt_mismatch"] += 1
            if not _close(
                predicted,
                float(request["predicted_next_debt"]),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            ):
                counters["ttft_debt_mismatch"] += 1
            for replayed, field in (
                (normalized, "normalized_service"),
                (current * current, "current_debt_squared"),
                (predicted * predicted, "next_debt_squared"),
                (contribution, "drift_contribution"),
            ):
                if not _close(
                    replayed,
                    float(request[field]),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                ):
                    counters["ttft_debt_mismatch"] += 1
            if bool(request["completion_this_frame"]) != completion:
                counters["ttft_debt_mismatch"] += 1
        drift = math.fsum(contributions) / (2.0 * int(prefill_ref))
        score_rate_old = -drift / tau
        score_absolute_new = -drift
        online_score = (
            score_rate_old if schema_version == 1 else score_absolute_new
        )
        replayed_progress = math.fsum(progress)
        if not _close(
            drift,
            float(details["prefill_drift"]),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ) or not _close(
            online_score,
            float(details["score"]),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            counters["stage2_score_mismatch"] += 1
        if schema_version == 2:
            for replayed, field in (
                (score_rate_old, "ttft_score_rate_old"),
                (score_absolute_new, "ttft_score_absolute_new"),
            ):
                if field not in details or not _close(
                    replayed,
                    float(details[field]),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                ):
                    counters["stage2_score_mismatch"] += 1
        if int(details["completed_prefill_count"]) != completed or not _close(
            replayed_progress,
            float(details["prefill_progress"]),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            counters["tie_break_mismatch"] += 1
        tie_key = details["tie_break_key"]
        if (
            int(tie_key["completed_prefill_count_desc"]) != completed
            or not _close(
                replayed_progress,
                float(tie_key["prefill_progress_desc"]),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            or not _close(
                tau,
                float(tie_key["effective_duration_asc"]),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            or int(tie_key["prefill_budget_asc"])
            != int(candidate["plan"]["total_prefill_tokens"])
            or str(tie_key["plan_id_asc"]) != str(candidate["plan_id"])
        ):
            counters["tie_break_mismatch"] += 1
        replayed_scores.append(
            {
                "plan_id": str(candidate["plan_id"]),
                "score": online_score,
                "score_rate_old": score_rate_old,
                "score_absolute_new": score_absolute_new,
                "completed": completed,
                "progress": replayed_progress,
                "duration": tau,
                "budget": int(candidate["plan"]["total_prefill_tokens"]),
                "recorded_rank": int(details["rank"]),
                "recorded_rank_rate_old": int(
                    details.get("rank_rate_old", details["rank"])
                ),
                "recorded_rank_absolute_new": int(
                    details.get("rank_absolute_new", details["rank"])
                ),
            }
        )

    ranked, winner_tie = _rank_candidates(
        replayed_scores,
        score_key="score",
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    if any(item["recorded_rank"] != rank for rank, item in enumerate(ranked, 1)):
        counters["tie_break_mismatch"] += 1
    if schema_version == 2:
        rate_ranked, _ = _rank_candidates(
            replayed_scores,
            score_key="score_rate_old",
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        absolute_ranked, _ = _rank_candidates(
            replayed_scores,
            score_key="score_absolute_new",
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        if any(
            item["recorded_rank_rate_old"] != rank
            for rank, item in enumerate(rate_ranked, 1)
        ) or any(
            item["recorded_rank_absolute_new"] != rank
            for rank, item in enumerate(absolute_ranked, 1)
        ):
            counters["tie_break_mismatch"] += 1
    if winner_tie != list(decision["winner_tie_plan_ids"]):
        counters["tie_break_mismatch"] += 1
    replayed_winner = ranked[0]["plan_id"] if ranked else None
    if replayed_winner != decision["selector_selected_plan_id"]:
        counters["winner_mismatch"] += 1
    return counters


def replay_file(path: Path) -> dict[str, int]:
    summary = {
        "frames_replayed": 0,
        "stage1_mismatch": 0,
        "ttft_debt_mismatch": 0,
        "stage2_score_mismatch": 0,
        "winner_mismatch": 0,
        "tie_break_mismatch": 0,
        "service_rate_invariant_mismatch": 0,
    }
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"diagnosis row {line_number} is not an object")
            result = replay_record(record)
            summary["frames_replayed"] += 1
            for key, value in result.items():
                summary[key] += value
    return summary


def _backlog_stratum(depth: int) -> str:
    if depth <= 0:
        return "none"
    if depth == 1:
        return "1"
    if depth <= 4:
        return "2-4"
    if depth <= 8:
        return "5-8"
    return ">8"


def _slack_stratum(value: object) -> str:
    if value is None:
        return "no_active_tbt_obligation"
    slack_ms = float(value) * 1000.0
    if slack_ms <= 0:
        return "<=0ms"
    if slack_ms <= 50:
        return "0-50ms"
    if slack_ms <= 100:
        return "50-100ms"
    if slack_ms <= 200:
        return "100-200ms"
    return ">200ms"


def counterfactual_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Rank one recorded Stage-2 set under rate and absolute TTFT scores."""

    schema_version = record.get("schema_version")
    if schema_version not in SUPPORTED_SELECTOR_DIAGNOSIS_SCHEMA_VERSIONS:
        raise ValueError("Selector diagnosis schema version mismatch")
    if schema_version == 3:
        raise ValueError(
            "TTFT Rate/Absolute counterfactual supports diagnosis schema 1/2 only"
        )
    selector = record["selector"]
    rel_tol = float(selector["score_rel_tol"])
    abs_tol = float(selector["score_abs_tol"])
    candidates: list[dict[str, Any]] = []
    nonzero_passed = False
    nonzero_present = False
    for candidate in record["candidates"]:
        template_id = str(candidate["template_id"])
        is_zero = _is_zero_template(template_id)
        nonzero_present = nonzero_present or not is_zero
        nonzero_passed = nonzero_passed or (
            not is_zero and bool(candidate["stage1_tbt"]["passed"])
        )
        details = candidate["stage2_ttft"]
        if details is None:
            continue
        drift = float(details["prefill_drift"])
        duration = float(candidate["duration"]["effective"])
        candidates.append(
            {
                "plan_id": str(candidate["plan_id"]),
                "template_id": template_id,
                "is_zero": is_zero,
                "is_stock": _is_stock_template(template_id),
                "score_rate_old": -drift / duration,
                "score_absolute_new": -drift,
                "completed": int(details["completed_prefill_count"]),
                "progress": float(details["prefill_progress"]),
                "duration": duration,
                "budget": int(candidate["plan"]["total_prefill_tokens"]),
            }
        )
    rate_ranked, _ = _rank_candidates(
        candidates,
        score_key="score_rate_old",
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    absolute_ranked, _ = _rank_candidates(
        candidates,
        score_key="score_absolute_new",
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    rate_rank = {
        item["plan_id"]: rank for rank, item in enumerate(rate_ranked, start=1)
    }
    absolute_rank = {
        item["plan_id"]: rank
        for rank, item in enumerate(absolute_ranked, start=1)
    }
    old_winner = rate_ranked[0] if rate_ranked else None
    new_winner = absolute_ranked[0] if absolute_ranked else None
    old_zero = bool(old_winner and old_winner["is_zero"])
    new_zero = bool(new_winner and new_winner["is_zero"])
    stock = next((item for item in candidates if item["is_stock"]), None)
    depth = int(record["state"]["waiting_prefill_count"])
    return {
        "frame_id": int(record["frame_id"]),
        "has_prefill_backlog": depth > 0,
        "backlog_depth": depth,
        "backlog_stratum": _backlog_stratum(depth),
        "slack_stratum": _slack_stratum(
            record["state"]["min_tbt_slack_seconds"]
        ),
        "nonzero_candidate_present": nonzero_present,
        "nonzero_passed_stage1": nonzero_passed,
        "all_nonzero_filtered_by_stage1": (
            nonzero_present and not nonzero_passed
        ),
        "old_winner_plan_id": old_winner["plan_id"] if old_winner else None,
        "new_winner_plan_id": new_winner["plan_id"] if new_winner else None,
        "old_zero_selected": old_zero,
        "new_zero_selected": new_zero,
        "winner_changed": bool(
            old_winner
            and new_winner
            and old_winner["plan_id"] != new_winner["plan_id"]
        ),
        "zero_to_nonzero": old_zero and not new_zero,
        "nonzero_to_zero": not old_zero and new_zero,
        "old_zero_with_nonzero_passed_stage1": old_zero and nonzero_passed,
        "new_zero_with_nonzero_passed_stage1": new_zero and nonzero_passed,
        "stock_rank_old": rate_rank.get(stock["plan_id"]) if stock else None,
        "stock_rank_new": (
            absolute_rank.get(stock["plan_id"]) if stock else None
        ),
    }


def _new_counterfactual_bucket() -> dict[str, Any]:
    return {
        "frames": 0,
        "old_zero_selected": 0,
        "new_zero_selected": 0,
        "winner_changed": 0,
        "zero_to_nonzero": 0,
        "nonzero_to_zero": 0,
        "all_nonzero_filtered_by_stage1": 0,
        "old_zero_with_nonzero_passed_stage1": 0,
        "new_zero_with_nonzero_passed_stage1": 0,
        "stock_rank_old_sum": 0,
        "stock_rank_new_sum": 0,
        "stock_rank_count": 0,
    }


def _add_counterfactual_frame(bucket: dict[str, Any], frame: Mapping[str, Any]) -> None:
    bucket["frames"] += 1
    for field in (
        "old_zero_selected",
        "new_zero_selected",
        "winner_changed",
        "zero_to_nonzero",
        "nonzero_to_zero",
        "all_nonzero_filtered_by_stage1",
        "old_zero_with_nonzero_passed_stage1",
        "new_zero_with_nonzero_passed_stage1",
    ):
        bucket[field] += int(bool(frame[field]))
    if frame["stock_rank_old"] is not None and frame["stock_rank_new"] is not None:
        bucket["stock_rank_old_sum"] += int(frame["stock_rank_old"])
        bucket["stock_rank_new_sum"] += int(frame["stock_rank_new"])
        bucket["stock_rank_count"] += 1


def _finalize_counterfactual_bucket(bucket: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(bucket)
    frames = int(result["frames"])
    stock_count = int(result.pop("stock_rank_count"))
    stock_old_sum = int(result.pop("stock_rank_old_sum"))
    stock_new_sum = int(result.pop("stock_rank_new_sum"))
    result["old_zero_ratio"] = (
        int(result["old_zero_selected"]) / frames if frames else None
    )
    result["new_zero_ratio"] = (
        int(result["new_zero_selected"]) / frames if frames else None
    )
    result["stock_rank_old_mean"] = (
        stock_old_sum / stock_count if stock_count else None
    )
    result["stock_rank_new_mean"] = (
        stock_new_sum / stock_count if stock_count else None
    )
    result["stock_rank_count"] = stock_count
    return result


def counterfactual_replay_file(path: Path) -> dict[str, Any]:
    overall = _new_counterfactual_bucket()
    backlog = _new_counterfactual_bucket()
    by_backlog: dict[str, dict[str, Any]] = {}
    by_slack: dict[str, dict[str, Any]] = {}
    total_frames = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"diagnosis row {line_number} is not an object")
            frame = counterfactual_record(record)
            total_frames += 1
            _add_counterfactual_frame(overall, frame)
            if not frame["has_prefill_backlog"]:
                continue
            _add_counterfactual_frame(backlog, frame)
            backlog_bucket = by_backlog.setdefault(
                str(frame["backlog_stratum"]), _new_counterfactual_bucket()
            )
            slack_bucket = by_slack.setdefault(
                str(frame["slack_stratum"]), _new_counterfactual_bucket()
            )
            _add_counterfactual_frame(backlog_bucket, frame)
            _add_counterfactual_frame(slack_bucket, frame)
    return {
        "total_frames": total_frames,
        "prefill_backlog_frames": int(backlog["frames"]),
        "overall": _finalize_counterfactual_bucket(overall),
        "prefill_backlog": _finalize_counterfactual_bucket(backlog),
        "by_backlog_depth": {
            key: _finalize_counterfactual_bucket(value)
            for key, value in sorted(by_backlog.items())
        },
        "by_min_tbt_slack": {
            key: _finalize_counterfactual_bucket(value)
            for key, value in sorted(by_slack.items())
        },
    }
