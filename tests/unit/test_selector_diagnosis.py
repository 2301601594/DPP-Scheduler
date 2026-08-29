from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dpp_scheduler.contracts import (
    ControlState,
    DecodeRequest,
    Decision,
    Obligation,
    PrefillRequest,
)
from dpp_scheduler.dpp_selector import DPPSelector
from dpp_scheduler.selector_diagnosis import (
    DPP_SELECTOR_DIAGNOSIS_ENV,
    DPP_SELECTOR_DIAGNOSIS_PATH_ENV,
    SELECTOR_DIAGNOSIS_SCHEMA_VERSION,
    SelectorDiagnosisWriter,
    counterfactual_record,
    replay_file,
    replay_record,
    resolve_selector_diagnosis,
)
from tests.unit.test_dpp_selector import candidate, deadline_snapshot, settings, snapshot


class SelectorDiagnosisTests(unittest.TestCase):
    def _record(self, root: Path, *, slack: float = 0.15) -> tuple[Path, dict]:
        state = deadline_snapshot(slack=slack)
        control = ControlState(state.snapshot_hash)
        candidates = (
            candidate(state, "short", duration=0.1),
            candidate(state, "long", duration=0.2),
        )
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, control, candidates, capture_request_details=True
        )
        path = root / "selector.jsonl"
        writer = SelectorDiagnosisWriter(
            path,
            config_sha256="c" * 64,
            predictor_version="predictor-test",
            schema_version=SELECTOR_DIAGNOSIS_SCHEMA_VERSION,
        )
        record = writer.write(
            snapshot=state,
            control=control,
            safe_candidates=candidates,
            audit=audit,
            selector_decision=decision,
            controller_decision=decision,
            executed_plan_id=decision.selected_plan.plan_id,
        )
        writer.close()
        return path, record

    def _prefill_record(self, root: Path) -> dict:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0, ttft_slo_seconds=2.0),)
        )
        control = ControlState(state.snapshot_hash, (("p", 0.5),))
        candidates = (
            candidate(state, "zero", duration=0.1, template_id="ZERO"),
            candidate(
                state,
                "partial",
                prefill_items=(("p", 20),),
                duration=0.1,
                template_id="P20",
            ),
        )
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, control, candidates, capture_request_details=True
        )
        writer = SelectorDiagnosisWriter(
            root / "prefill.jsonl",
            config_sha256="e" * 64,
            predictor_version="predictor-test",
            schema_version=SELECTOR_DIAGNOSIS_SCHEMA_VERSION,
        )
        record = writer.write(
            snapshot=state,
            control=control,
            safe_candidates=candidates,
            audit=audit,
            selector_decision=decision,
            controller_decision=decision,
            executed_plan_id=decision.selected_plan.plan_id,
        )
        writer.close()
        return record

    def test_round_trip_normal_and_tbt_fallback(self) -> None:
        for slack in (0.15, -0.05):
            with self.subTest(slack=slack), tempfile.TemporaryDirectory() as root:
                path, _ = self._record(Path(root), slack=slack)
                summary = replay_file(path)
                self.assertEqual(summary["frames_replayed"], 1)
                self.assertEqual(
                    sum(
                        value
                        for key, value in summary.items()
                        if key.endswith("_mismatch")
                    ),
                    0,
                )

    def test_no_safe_decision_round_trips(self) -> None:
        state = snapshot()
        control = ControlState(state.snapshot_hash)
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, control, (), capture_request_details=True
        )
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "empty.jsonl"
            writer = SelectorDiagnosisWriter(
                path,
                config_sha256="d" * 64,
                predictor_version="predictor-test",
                schema_version=SELECTOR_DIAGNOSIS_SCHEMA_VERSION,
            )
            writer.write(
                snapshot=state,
                control=control,
                safe_candidates=(),
                audit=audit,
                selector_decision=decision,
                controller_decision=Decision(
                    state.frame_id,
                    state.snapshot_hash,
                    selected_plan=None,
                    reason="EMPTY_WORKLOAD_IDLE",
                ),
                executed_plan_id=None,
            )
            writer.close()
            self.assertEqual(replay_file(path)["winner_mismatch"], 0)

    def test_tampered_slack_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            _, record = self._record(Path(root))
            record["state"]["tbt_request_slacks"][0]["slack_seconds"] += 1.0
            self.assertGreater(replay_record(record)["stage1_mismatch"], 0)

    def test_tampered_service_tokens_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
            stage2 = next(
                item["stage2_prefill_service_rate"]
                for item in record["candidates"]
                if item["plan_id"] == "partial"
            )
            stage2["prefill_service_tokens"] += 1
            self.assertGreater(replay_record(record)["stage2_score_mismatch"], 0)

    def test_tampered_score_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
            stage2 = next(
                item["stage2_prefill_service_rate"]
                for item in record["candidates"]
                if item["plan_id"] == "partial"
            )
            stage2["score"] = float(stage2["score"]) + 1.0
            self.assertGreater(replay_record(record)["stage2_score_mismatch"], 0)

    def test_records_service_rate_and_zero_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
            partial = next(
                item["stage2_prefill_service_rate"]
                for item in record["candidates"]
                if item["template_id"] == "P20"
            )
            self.assertEqual(partial["prefill_service_tokens"], 20)
            self.assertEqual(partial["score"], partial["prefill_service_rate"])
            diagnosis = record["service_rate_diagnosis"]
            self.assertEqual(diagnosis["prefill_backlog_count"], 1)
            self.assertEqual(diagnosis["prefill_backlog_tokens"], 100)
            self.assertEqual(
                diagnosis["stage1_eligible_nonzero_candidate_count"], 1
            )
            self.assertFalse(diagnosis["selected_is_zero"])
            self.assertFalse(diagnosis["zero_with_eligible_nonzero"])

    def test_ttft_counterfactual_rejects_service_rate_schema(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
        with self.assertRaisesRegex(ValueError, "schema 1/2"):
            counterfactual_record(record)

    def _delta_n_record(self, root: Path, *, limit: int = 0) -> dict:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0),),
            decode=(DecodeRequest("d", 0.0, 100, tbt_deadline=10.4),),
            obligations=(Obligation("tbt:d", "d", "TBT", 10.4, 9.0),),
        )
        control = ControlState(state.snapshot_hash, (("p", 0.0),))
        candidates = (
            candidate(state, "zero", duration=0.1, template_id="ZERO"),
            candidate(
                state,
                "mixed",
                prefill_items=(("p", 4),),
                duration=0.3,
                template_id="P10",
            ),
        )
        decision, audit = DPPSelector(
            settings(stage1_max_delta_n=limit)
        ).select_with_audit(state, control, candidates, capture_request_details=True)
        writer = SelectorDiagnosisWriter(
            root / "delta_n.jsonl",
            config_sha256="f" * 64,
            predictor_version="predictor-test",
            schema_version=SELECTOR_DIAGNOSIS_SCHEMA_VERSION,
        )
        record = writer.write(
            snapshot=state,
            control=control,
            safe_candidates=candidates,
            audit=audit,
            selector_decision=decision,
            controller_decision=decision,
            executed_plan_id=decision.selected_plan.plan_id,
        )
        writer.close()
        return record

    def test_delta_n_admission_round_trips_at_limit_one(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._delta_n_record(Path(root), limit=1)
            stage1 = record["stage1"]
            self.assertEqual(stage1["status"], "DELTA_N_ADMITTED")
            self.assertEqual(stage1["zero_reference_resolution"], "ZERO_TEMPLATE")
            self.assertEqual(stage1["reference_plan_id"], "zero")
            self.assertEqual(
                stage1["maximum_incremental_tbt_violations"], 1
            )
            mixed = next(
                item["stage1_tbt"]
                for item in record["candidates"]
                if item["plan_id"] == "mixed"
            )
            self.assertEqual(mixed["delta_violation_count"], 1)
            self.assertTrue(mixed["passed"])
            self.assertGreater(mixed["delta_lateness_seconds"], 0.0)
            self.assertEqual(
                replay_file(Path(root) / "delta_n.jsonl")["stage1_mismatch"], 0
            )

    def test_tampered_delta_n_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._delta_n_record(Path(root), limit=1)
            mixed = next(
                item["stage1_tbt"]
                for item in record["candidates"]
                if item["plan_id"] == "mixed"
            )
            mixed["delta_violation_count"] -= 1
            self.assertGreater(replay_record(record)["stage1_mismatch"], 0)

    def test_tampered_admission_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._delta_n_record(Path(root), limit=0)
            mixed = next(
                item["stage1_tbt"]
                for item in record["candidates"]
                if item["plan_id"] == "mixed"
            )
            mixed["passed"] = True
            record["stage1"]["eligible_plan_ids"] = list(
                record["stage1"]["eligible_plan_ids"]
            ) + ["mixed"]
            self.assertGreater(replay_record(record)["stage1_mismatch"], 0)

    def test_tampered_zero_reference_resolution_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._delta_n_record(Path(root), limit=1)
            record["stage1"]["zero_reference_resolution"] = "ZERO_SERVICE_MATCH"
            self.assertGreater(replay_record(record)["stage1_mismatch"], 0)

    @staticmethod
    def _to_v3(record: dict) -> dict:
        converted = json.loads(json.dumps(record))
        converted["schema_version"] = 3
        state = converted["state"]
        selector = converted["selector"]
        timestamp = float(state["timestamp"])
        delta = float(selector["tbt_delta_seconds"])
        slacks = [
            float(item["deadline"]) - timestamp
            for item in state["tbt_request_slacks"]
        ]
        min_slack = min(slacks) if slacks else None
        duration_limit = min_slack + delta if min_slack is not None else None
        eligible: list[str] = []
        fallback = None
        for item in converted["candidates"]:
            effective = float(item["duration"]["effective"])
            passed = duration_limit is None or effective <= duration_limit
            item["stage1_tbt"] = {
                "plan_id": item["plan_id"],
                "effective_duration": effective,
                "duration_limit": duration_limit,
                "passed": passed,
                "selected_by_fallback": False,
                "rejection_reason": None,
            }
            if passed:
                eligible.append(item["plan_id"])
        stage1 = converted["stage1"]
        if not converted["candidates"]:
            status = "NO_SAFE_CANDIDATES"
        elif duration_limit is None:
            status = "NO_ACTIVE_TBT_OBLIGATION"
        elif eligible:
            status = "WITHIN_SLACK"
        else:
            status = "NO_CANDIDATE_WITHIN_SLACK"
            fallback = min(
                converted["candidates"],
                key=lambda item: (
                    float(item["duration"]["effective"]),
                    int(item["plan"]["total_prefill_tokens"]),
                    str(item["plan_id"]),
                ),
            )["plan_id"]
            eligible = [fallback]
            for item in converted["candidates"]:
                item["stage1_tbt"]["selected_by_fallback"] = (
                    item["plan_id"] == fallback
                )
                item["stage1_tbt"]["rejection_reason"] = (
                    "EFFECTIVE_DURATION_EXCEEDS_TBT_LIMIT"
                )
        stage1["status"] = status
        stage1["duration_limit_seconds"] = duration_limit
        stage1["eligible_plan_ids"] = eligible
        stage1["fallback_plan_id"] = fallback
        for key in (
            "maximum_incremental_tbt_violations",
            "reference_plan_id",
            "reference_template_id",
            "reference_risk_duration_seconds",
            "reference_violation_count",
            "zero_reference_resolution",
        ):
            stage1.pop(key, None)
        selector.pop("maximum_incremental_tbt_violations", None)
        state.pop("active_decode_request_ids", None)
        return converted

    def test_legacy_v3_replay_keeps_working(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path, _ = self._record(Path(root), slack=0.25)
            with path.open("r", encoding="utf-8") as stream:
                record = json.loads(stream.readline())
            converted = self._to_v3(record)
            result = replay_record(converted)
            self.assertEqual(
                sum(
                    value
                    for key, value in result.items()
                    if key.endswith("_mismatch")
                ),
                0,
            )

    def test_tampered_winner_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
            record["decision"]["selector_selected_plan_id"] = "tampered"
            self.assertGreater(replay_record(record)["winner_mismatch"], 0)

    def test_tampered_tie_key_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
            stage2 = next(
                item["stage2_prefill_service_rate"]
                for item in record["candidates"]
                if item["plan_id"] == "partial"
            )
            stage2["tie_break_key"]["prefill_service_tokens_desc"] += 1
            result = replay_record(record)
            self.assertGreater(result["tie_break_mismatch"], 0)

    def test_runtime_switch_and_path_must_be_paired(self) -> None:
        active, path = resolve_selector_diagnosis(
            settings(),
            selection_mode="normal",
            environment={
                DPP_SELECTOR_DIAGNOSIS_ENV: "1",
                DPP_SELECTOR_DIAGNOSIS_PATH_ENV: "/tmp/selector.jsonl",
            },
        )
        self.assertTrue(active)
        self.assertEqual(path, Path("/tmp/selector.jsonl"))
        with self.assertRaises(ValueError):
            resolve_selector_diagnosis(
                settings(),
                selection_mode="normal",
                environment={DPP_SELECTOR_DIAGNOSIS_ENV: "1"},
            )
        with self.assertRaises(ValueError):
            resolve_selector_diagnosis(
                settings(),
                selection_mode="normal",
                environment={DPP_SELECTOR_DIAGNOSIS_PATH_ENV: "/tmp/x"},
            )
        with self.assertRaises(ValueError):
            resolve_selector_diagnosis(
                settings(),
                selection_mode="forced_stock_plan",
                environment={
                    DPP_SELECTOR_DIAGNOSIS_ENV: "1",
                    DPP_SELECTOR_DIAGNOSIS_PATH_ENV: "/tmp/x",
                },
            )

    def test_writer_refuses_to_overwrite_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "selector.jsonl"
            path.write_text(json.dumps({"existing": True}), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                SelectorDiagnosisWriter(
                    path,
                    config_sha256="c" * 64,
                    predictor_version="p",
                    schema_version=SELECTOR_DIAGNOSIS_SCHEMA_VERSION,
                )


if __name__ == "__main__":
    unittest.main()
