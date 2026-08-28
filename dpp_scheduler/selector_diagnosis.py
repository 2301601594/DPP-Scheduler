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
SELECTOR_DIAGNOSIS_SCHEMA_VERSION = 1


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
                    "completed_prefill_count_desc": stage2.completed_prefill_count,
                    "prefill_progress_desc": stage2.prefill_progress,
                    "effective_duration_asc": stage2.effective_duration,
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
                    "stage2_ttft": stage2_payload,
                    "selected": audit.selected_plan_id == plan.plan_id,
                }
            )
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
                "prefill_reference_concurrency": (
                    audit.stage2_scores[0].prefill_reference_concurrency
                    if audit.stage2_scores
                    else None
                ),
                "tbt_delta_seconds": audit.stage1.delta_seconds,
                "tie_break_order": list(audit.tie_break_order),
            },
            "state": {
                "timestamp": snapshot.timestamp,
                "active_decode_count": len(snapshot.active_decode_requests),
                "waiting_prefill_count": len(snapshot.waiting_prefill_requests),
                "tbt_request_slacks": [
                    asdict(item) for item in audit.stage1.request_slacks
                ],
                "min_tbt_slack_seconds": audit.stage1.min_slack_seconds,
                "current_ttft_debts": list(control.ttft_service_debts),
                "safe_candidate_count": len(safe_candidates),
            },
            "stage1": asdict(audit.stage1),
            "candidates": candidates,
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


def replay_record(record: Mapping[str, Any]) -> dict[str, int]:
    counters = {
        "stage1_mismatch": 0,
        "ttft_debt_mismatch": 0,
        "stage2_score_mismatch": 0,
        "winner_mismatch": 0,
        "tie_break_mismatch": 0,
    }
    if record.get("schema_version") != SELECTOR_DIAGNOSIS_SCHEMA_VERSION:
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
        score = -drift / tau
        replayed_progress = math.fsum(progress)
        if not _close(
            drift,
            float(details["prefill_drift"]),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ) or not _close(
            score,
            float(details["score"]),
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
                "score": score,
                "completed": completed,
                "progress": replayed_progress,
                "duration": tau,
                "budget": int(candidate["plan"]["total_prefill_tokens"]),
                "recorded_rank": int(details["rank"]),
            }
        )

    ranked: list[dict[str, Any]] = []
    remaining_scores = sorted(
        replayed_scores, key=lambda item: (-item["score"], item["plan_id"])
    )
    winner_tie: list[str] = []
    while remaining_scores:
        leader = remaining_scores[0]
        group = [
            item
            for item in remaining_scores
            if _close(
                item["score"], leader["score"], rel_tol=rel_tol, abs_tol=abs_tol
            )
        ]
        group_ids = {item["plan_id"] for item in group}
        remaining_scores = [
            item for item in remaining_scores if item["plan_id"] not in group_ids
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
    if any(item["recorded_rank"] != rank for rank, item in enumerate(ranked, 1)):
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
