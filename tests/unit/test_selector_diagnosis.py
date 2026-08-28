from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dpp_scheduler.contracts import ControlState, Decision, PrefillRequest
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
    def _record(self, root: Path, *, fallback: bool = False) -> tuple[Path, dict]:
        state = deadline_snapshot(slack=-0.05 if fallback else 0.15)
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
        for fallback in (False, True):
            with self.subTest(fallback=fallback), tempfile.TemporaryDirectory() as root:
                path, _ = self._record(Path(root), fallback=fallback)
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

    def test_tampered_next_debt_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
            stage2 = next(
                item["stage2_ttft"]
                for item in record["candidates"]
                if item["plan_id"] == "partial"
            )
            stage2["request_results"][0]["predicted_next_debt"] += 1.0
            self.assertGreater(replay_record(record)["ttft_debt_mismatch"], 0)

    def test_tampered_score_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
            stage2 = next(
                item["stage2_ttft"]
                for item in record["candidates"]
                if item["plan_id"] == "partial"
            )
            stage2["score"] = float(stage2["score"]) + 1.0
            self.assertGreater(replay_record(record)["stage2_score_mismatch"], 0)

    def test_records_absolute_and_legacy_rate_scores_and_zero_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
            zero = next(
                item["stage2_ttft"]
                for item in record["candidates"]
                if item["template_id"] == "ZERO"
            )
            self.assertEqual(zero["score"], zero["ttft_score_absolute_new"])
            self.assertEqual(zero["rank"], zero["rank_absolute_new"])
            self.assertIn("ttft_score_rate_old", zero)
            self.assertIn("rank_rate_old", zero)
            diagnosis = record["zero_diagnosis"]
            self.assertTrue(diagnosis["has_prefill_backlog"])
            self.assertTrue(diagnosis["zero_candidate_present"])
            self.assertTrue(diagnosis["nonzero_passed_stage1"])

    def test_counterfactual_reports_rate_and_absolute_winners(self) -> None:
        state = snapshot(
            prefill=(PrefillRequest("p", 0.0, 100, 0, ttft_slo_seconds=2.0),)
        )
        control = ControlState(state.snapshot_hash, (("p", 0.0),))
        candidates = (
            candidate(state, "zero", duration=0.1, template_id="ZERO"),
            candidate(
                state,
                "partial",
                prefill_items=(("p", 4),),
                duration=0.2,
                template_id="P10",
            ),
        )
        decision, audit = DPPSelector(settings()).select_with_audit(
            state, control, candidates, capture_request_details=True
        )
        with tempfile.TemporaryDirectory() as root:
            writer = SelectorDiagnosisWriter(
                Path(root) / "counterfactual.jsonl",
                config_sha256="a" * 64,
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
        counterfactual = counterfactual_record(record)
        self.assertEqual(counterfactual["old_winner_plan_id"], "partial")
        self.assertEqual(counterfactual["new_winner_plan_id"], "zero")
        self.assertTrue(counterfactual["nonzero_to_zero"])

    def test_tampered_winner_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
            record["decision"]["selector_selected_plan_id"] = "tampered"
            self.assertGreater(replay_record(record)["winner_mismatch"], 0)

    def test_tampered_tie_key_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            record = self._prefill_record(Path(root))
            stage2 = next(
                item["stage2_ttft"]
                for item in record["candidates"]
                if item["plan_id"] == "partial"
            )
            stage2["tie_break_key"]["completed_prefill_count_desc"] += 1
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
