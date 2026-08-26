from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dpp_scheduler.contracts import BatchPlan, DecodeRequest, PrefillRequest, StateSnapshot
from dpp_scheduler.predictor import (
    ACTIVE_FEATURES,
    BATCH_KINDS,
    CROSS_FEATURE_SEGMENTED_ONLINE_PREDICTOR_VERSION,
    MIXED_CROSS_FEATURE_NAMES,
    MIXED_DECODE_SEGMENTS,
    ONLINE_PREDICTOR_VERSION,
    SEGMENTED_ONLINE_PREDICTOR_VERSION,
    RidgeDurationPredictor,
    build_plan_features,
)


def _snapshot(
    frame: int = 1, *, decode_context: int = 20, decode_count: int = 1
) -> StateSnapshot:
    return StateSnapshot.create(
        frame_id=frame,
        timestamp=float(frame),
        waiting_prefill_requests=(
            PrefillRequest("p1", 0.0, 100, 10, is_running=True),
            PrefillRequest("p2", 0.1, 100, 0),
        ),
        active_decode_requests=tuple(
            DecodeRequest(f"d{index}", 0.0, decode_context)
            for index in range(1, decode_count + 1)
        ),
        active_ttft_obligations=(),
        active_tbt_obligations=(),
        recovery_requests=(),
        free_kv_blocks=100,
        kv_block_size=16,
        token_budget=2048,
        sequence_budget=64,
        total_kv_blocks=200,
    )


def _plan(snapshot: StateSnapshot, kind: str) -> BatchPlan:
    prefills = () if kind == "decode_only" else (("p1", 4), ("p2", 6))
    decodes = (
        ()
        if kind == "prefill_only"
        else tuple(request.request_id for request in snapshot.active_decode_requests)
    )
    return BatchPlan(
        plan_id=f"plan-{snapshot.frame_id}-{kind}",
        snapshot_hash=snapshot.snapshot_hash,
        template_id="test",
        prefill_items=prefills,
        decode_items=decodes,
        total_prefill_tokens=sum(tokens for _, tokens in prefills),
        total_decode_tokens=len(decodes),
        total_sequences=len(prefills) + len(decodes),
        projected_kv_blocks=2,
        mandatory_request_ids=(),
    )


def _artifact(
    root: Path,
    *,
    support_max: float = 1e9,
    segmented_mixed: bool = False,
    cross_feature_mixed: bool = False,
) -> Path:
    models = {}
    calibration = {}
    for kind in BATCH_KINDS:
        names = (
            MIXED_CROSS_FEATURE_NAMES
            if kind == "mixed" and cross_feature_mixed
            else ACTIVE_FEATURES[kind]
        )
        models[kind] = {
            "active_features": list(names),
            "alpha": 1.0,
            "intercept_seconds": 0.1,
            "coefficients_for_standardized_features": {
                name: (
                    0.001
                    if name == "x_9"
                    else 0.0001
                    if name == "x_10"
                    else 0.0
                )
                for name in names
            },
            "standardization": {
                name: {"mean": 0.0, "scale": 1.0} for name in names
            },
            "support_domain_train_marginal_box": {
                name: {"min": 0.0, "max": support_max} for name in names
            },
        }
        calibration[kind] = {
            "window_size": 32,
            "minimum_samples": 32,
            "cold_start_mean_seconds": 0.0,
            "cold_start_centered_p95_seconds": 0.02,
        }
    version = (
        CROSS_FEATURE_SEGMENTED_ONLINE_PREDICTOR_VERSION
        if cross_feature_mixed
        else SEGMENTED_ONLINE_PREDICTOR_VERSION
        if segmented_mixed
        else ONLINE_PREDICTOR_VERSION
    )
    if segmented_mixed:
        base_model = models["mixed"]
        models["mixed"] = {
            "dispatch": "decode_count_segments",
            "dispatch_feature": "x_4",
            "segments": [
                {
                    "segment_id": segment_id,
                    "minimum_decode_count_inclusive": minimum,
                    "maximum_decode_count_inclusive": maximum,
                    "model": {
                        **base_model,
                        "intercept_seconds": 0.1 * (index + 1),
                    },
                }
                for index, (segment_id, minimum, maximum) in enumerate(
                    MIXED_DECODE_SEGMENTS
                )
            ],
        }
    payload = {
        "schema_version": 1,
        "predictor_version": version,
        "model_family": "ridge_regression",
        "models": models,
        "residual_calibration": {
            "strategy": "online_window_per_batch_kind",
            "residual_definition": "actual_duration_seconds-base_duration_seconds",
            "quantile_method": "higher",
            "by_batch_kind": calibration,
        },
    }
    root.mkdir()
    predictor = root / "predictor.json"
    predictor.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(predictor.read_bytes()).hexdigest()
    (root / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictor_version": version,
                "status": "complete",
                "files": {
                    "predictor": {"file": "predictor.json", "sha256": digest}
                },
            }
        ),
        encoding="utf-8",
    )
    return root


class OnlinePredictorTests(unittest.TestCase):
    def test_feature_definitions_match_all_three_scenarios(self) -> None:
        snapshot = _snapshot()
        kind, features = build_plan_features(snapshot, _plan(snapshot, "mixed"))
        self.assertEqual(kind, "mixed")
        self.assertEqual(
            features,
            {
                "x_1": 4 * 14 + 6 * 6,
                "x_2": 4 * 4 + 6 * 6,
                "x_3": 10 + 0 + 20,
                "x_4": 1.0,
                "x_5": 20.0,
                "x_6": 10.0,
                "x_7": 6.0,
                "x_8": 2.0,
                "x_9": 10.0,
                "x_10": 200.0,
            },
        )
        self.assertEqual(
            build_plan_features(snapshot, _plan(snapshot, "decode_only"))[1]["x_5"],
            20.0,
        )
        self.assertEqual(
            build_plan_features(snapshot, _plan(snapshot, "prefill_only"))[1]["x_4"],
            0.0,
        )

    def test_cold_start_then_online_window_uses_only_prior_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictor = RidgeDurationPredictor.from_artifact(
                _artifact(Path(directory) / "artifact")
            )
            first = _snapshot(1)
            first_audit = predictor.predict_with_audit(first, _plan(first, "decode_only"))
            self.assertEqual(first_audit.calibration_source, "offline_oof_cold_start")
            self.assertAlmostEqual(first_audit.prediction.expected_duration, 0.1)
            for frame in range(1, 33):
                snapshot = _snapshot(frame)
                plan = _plan(snapshot, "decode_only")
                before = predictor.predict_with_audit(snapshot, plan)
                predictor.observe_actual(
                    snapshot,
                    plan,
                    0.11,
                    base_duration_seconds=before.base_duration_seconds,
                )
            next_snapshot = _snapshot(33)
            next_audit = predictor.predict_with_audit(
                next_snapshot, _plan(next_snapshot, "decode_only")
            )
            self.assertEqual(next_audit.calibration_source, "online_window")
            self.assertAlmostEqual(next_audit.prediction.expected_duration, 0.11)
            self.assertEqual(predictor.calibration_sample_count("mixed"), 0)

    def test_duplicate_feedback_and_ood_is_constrained_but_not_calibrated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictor = RidgeDurationPredictor.from_artifact(
                _artifact(Path(directory) / "artifact", support_max=100.0)
            )
            snapshot = _snapshot(1)
            plan = _plan(snapshot, "decode_only")
            audit = predictor.predict_with_audit(snapshot, plan)
            predictor.observe_actual(snapshot, plan, 0.11)
            with self.assertRaisesRegex(ValueError, "duplicate or out of order"):
                predictor.observe_actual(snapshot, plan, 0.11)
            self.assertTrue(audit.prediction.in_support)

            ood_snapshot = _snapshot(2, decode_context=101)
            ood = predictor.predict_with_audit(
                ood_snapshot, _plan(ood_snapshot, "decode_only")
            )
            self.assertFalse(ood.prediction.in_support)
            self.assertEqual(
                ood.prediction.prediction_mode, "CONSTRAINED_EXTRAPOLATION"
            )
            self.assertIsNotNone(ood.prediction.expected_duration)
            self.assertGreater(
                ood.prediction.conservative_duration,
                ood.prediction.expected_duration,
            )
            far_snapshot = _snapshot(3, decode_context=500)
            far = predictor.predict_with_audit(
                far_snapshot, _plan(far_snapshot, "decode_only")
            )
            self.assertGreaterEqual(
                far.prediction.expected_duration,
                ood.prediction.expected_duration,
            )
            self.assertGreater(far.prediction.ood_distance, ood.prediction.ood_distance)
            with self.assertRaisesRegex(ValueError, "out-of-support"):
                predictor.observe_actual(
                    ood_snapshot, _plan(ood_snapshot, "decode_only"), 0.2
                )

    def test_artifact_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _artifact(Path(directory) / "artifact")
            (root / "predictor.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                RidgeDurationPredictor.from_artifact(root)

    def test_segmented_mixed_model_dispatches_on_decode_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictor = RidgeDurationPredictor.from_artifact(
                _artifact(Path(directory) / "artifact", segmented_mixed=True)
            )
            for frame, decode_count, expected in (
                (1, 4, 0.1),
                (2, 5, 0.2),
                (3, 16, 0.2),
                (4, 17, 0.3),
                (5, 64, 0.3),
            ):
                snapshot = _snapshot(frame, decode_count=decode_count)
                audit = predictor.predict_with_audit(
                    snapshot, _plan(snapshot, "mixed")
                )
                self.assertTrue(audit.prediction.in_support)
                self.assertAlmostEqual(audit.base_duration_seconds, expected)

    def test_cross_feature_segmented_model_uses_both_interactions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictor = RidgeDurationPredictor.from_artifact(
                _artifact(
                    Path(directory) / "artifact",
                    segmented_mixed=True,
                    cross_feature_mixed=True,
                )
            )
            snapshot = _snapshot(1, decode_context=20, decode_count=4)
            audit = predictor.predict_with_audit(snapshot, _plan(snapshot, "mixed"))
            # x9=10*4=40 and x10=10*(20*4)=800.
            self.assertAlmostEqual(audit.base_duration_seconds, 0.22)
            self.assertEqual(
                predictor.predictor_version,
                CROSS_FEATURE_SEGMENTED_ONLINE_PREDICTOR_VERSION,
            )

    def test_segmented_mixed_model_rejects_boundary_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _artifact(
                Path(directory) / "artifact", segmented_mixed=True
            )
            payload = json.loads((root / "predictor.json").read_text(encoding="utf-8"))
            payload["models"]["mixed"]["segments"][1][
                "minimum_decode_count_inclusive"
            ] = 6
            encoded = json.dumps(payload)
            (root / "predictor.json").write_text(encoded, encoding="utf-8")
            manifest = json.loads(
                (root / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            manifest["files"]["predictor"]["sha256"] = hashlib.sha256(
                encoded.encode("utf-8")
            ).hexdigest()
            (root / "artifact_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "boundaries or order"):
                RidgeDurationPredictor.from_artifact(root)


if __name__ == "__main__":
    unittest.main()
