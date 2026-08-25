from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.freeze_ood_uncertainty import build_artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class OODUncertaintyFreezeTests(unittest.TestCase):
    def _source(
        self,
        root: Path,
        *,
        role: str,
        run_id: str,
        recipe_seed: int,
        validation_fail: bool = False,
    ) -> Path:
        run = root / role / "runs" / run_id
        run.mkdir(parents=True)
        profile = run / "predictor_evaluation.jsonl"
        rows = []
        kind_required = {
            "prefill_only": 0.10,
            "decode_only": 0.20,
            "mixed": 0.30,
        }
        index = 0
        for kind, maximum_required in kind_required.items():
            for sample in range(20):
                required = maximum_required * sample / 19
                if validation_fail and kind == "decode_only" and sample < 2:
                    required = 0.5
                rows.append(
                    {
                        "schema_version": 2,
                        "run_id": run_id,
                        "iteration_index": index,
                        "sample_role": "target",
                        "batch_kind": kind,
                        "in_support": False,
                        "prediction_mode": "CONSTRAINED_EXTRAPOLATION",
                        "actual_duration_seconds": 1.1 + required,
                        "expected_duration_seconds": 1.0,
                        "centered_residual_p95_seconds": 0.1,
                        "ood_distance": 1.0,
                        "timing_source": "vllm_official_iteration_details",
                    }
                )
                index += 1
        profile.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        runtime = {"model": "Qwen3-14B", "runtime": "test"}
        predictor_hash = "b" * 64
        manifest = {
            "kind": "qwen3_14b_predictor_online_shadow_evaluation",
            "status": "complete",
            "run_id": run_id,
            "campaign_id": (
                "predictor_ood_calibration_v2"
                if role == "calibration"
                else "predictor_ood_validation_v2"
            ),
            "source_seed": 1001 if role == "calibration" else 1002,
            "recipe_seed": recipe_seed,
            "recipe_mode": "ood",
            "file_sha256": {profile.name: _sha256(profile)},
            "resolved": {
                "request_count": 150,
                "diagnostic_prefix": True,
                "runtime_consistency": runtime,
                "runtime_consistency_sha256": _canonical_sha256(runtime),
                "predictor_artifact_manifest_sha256": predictor_hash,
                "predictor_version": "qwen3-14b-ridge-three-scenario-online-v1",
            },
        }
        (run / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return profile

    def test_higher_q95_by_kind_and_independent_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = self._source(
                root, role="calibration", run_id="cal-run", recipe_seed=7001
            )
            validation = self._source(
                root, role="validation", run_id="val-run", recipe_seed=8001
            )
            artifact = build_artifact(
                (calibration,), (validation,), repository=root
            )
            self.assertEqual(artifact["status"], "frozen")
            self.assertAlmostEqual(artifact["kappa_by_batch_kind"]["mixed"], 0.3)
            self.assertAlmostEqual(artifact["kappa_ood"], 0.3)
            self.assertGreaterEqual(artifact["validation_coverage"]["overall"], 0.95)
            self.assertTrue(artifact["independence"]["recipe_seeds_disjoint"])

    def test_validation_below_target_is_retained_but_not_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = self._source(
                root, role="calibration", run_id="cal-run", recipe_seed=7001
            )
            validation = self._source(
                root,
                role="validation",
                run_id="val-run",
                recipe_seed=8001,
                validation_fail=True,
            )
            artifact = build_artifact(
                (calibration,), (validation,), repository=root
            )
            self.assertEqual(artifact["status"], "validation_failed")
            self.assertLess(
                artifact["validation_coverage"]["by_batch_kind"]["decode_only"],
                0.95,
            )

    def test_development_n150_per_role_is_nonformal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = self._source(
                root, role="calibration", run_id="cal-run", recipe_seed=7001
            )
            validation = self._source(
                root, role="validation", run_id="val-run", recipe_seed=8001
            )
            artifact = build_artifact(
                (calibration,),
                (validation,),
                repository=root,
                scope="development_nonformal",
            )
            self.assertEqual(artifact["status"], "frozen_development")
            self.assertFalse(artifact["formal_benchmark_eligible"])

    def test_overlapping_seed_and_tampered_profile_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = self._source(
                root, role="calibration", run_id="cal-run", recipe_seed=7001
            )
            validation = self._source(
                root, role="validation", run_id="val-run", recipe_seed=7001
            )
            with self.assertRaisesRegex(ValueError, "recipe seeds overlap"):
                build_artifact((calibration,), (validation,), repository=root)
            validation.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                build_artifact((calibration,), (validation,), repository=root)


if __name__ == "__main__":
    unittest.main()
