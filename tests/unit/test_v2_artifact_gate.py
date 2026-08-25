from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    REPOSITORY_ROOT,
    ActiveConfigError,
    FrozenPredictor,
    candidate_runtime_signature,
    load_active_runtime,
    validate_frozen_v2_artifacts,
)
from dpp_scheduler.settings import DPPSettings, PredictorSettings


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V2ArtifactGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_runtime = load_active_runtime(
            REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE
        )

    def test_owned_hashed_runtime_consistent_artifacts_enable_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            runtime = replace(self.base_runtime, workspace=workspace)
            runtime_payload, runtime_sha = candidate_runtime_signature(runtime)
            predictor = FrozenPredictor(
                artifact_root=workspace / "predictor",
                artifact_manifest_sha256="b" * 64,
                predictor_version="qwen3-14b-ridge-three-scenario-online-v1",
            )
            reference_path = workspace / "artifacts" / "reference.json"
            reference_sha = _write_json(
                reference_path,
                {
                    "schema_version": 1,
                    "artifact_id": "qwen3_14b_dgx_spark_reference_concurrency_v2",
                    "status": "frozen",
                    "scope": "formal",
                    "formal_benchmark_eligible": True,
                    "runtime_consistency": runtime_payload,
                    "runtime_consistency_sha256": runtime_sha,
                    "prefill_reference_concurrency": 4,
                    "decode_reference_concurrency": 16,
                    "sources": [
                        {
                            "run_id": "reference-run",
                            "profile_path": "profile.jsonl",
                            "profile_sha256": "1" * 64,
                            "run_manifest_path": "run_manifest.json",
                            "run_manifest_sha256": "2" * 64,
                        }
                    ],
                },
            )
            ood_path = workspace / "artifacts" / "ood.json"
            ood_sha = _write_json(
                ood_path,
                {
                    "schema_version": 1,
                    "artifact_id": "qwen3_14b_dgx_spark_ood_uncertainty_v1",
                    "status": "frozen",
                    "scope": "formal",
                    "formal_benchmark_eligible": True,
                    "coverage_target": 0.95,
                    "runtime_consistency": runtime_payload,
                    "runtime_consistency_sha256": runtime_sha,
                    "predictor_artifact_manifest_sha256": "b" * 64,
                    "predictor_version": predictor.predictor_version,
                    "kappa_ood": 0.25,
                    "validation_coverage": {
                        "overall": 0.96,
                        "by_batch_kind": {
                            "prefill_only": 0.95,
                            "decode_only": 0.97,
                            "mixed": 0.96,
                        },
                    },
                    "sources": {
                        "calibration": [
                            {
                                "run_id": "cal",
                                "source_seed": 1001,
                                "recipe_seed": 7001,
                                "profile_path": "calibration.jsonl",
                                "profile_sha256": "3" * 64,
                                "run_manifest_path": "calibration_manifest.json",
                                "run_manifest_sha256": "4" * 64,
                            }
                        ],
                        "validation": [
                            {
                                "run_id": "val",
                                "source_seed": 1002,
                                "recipe_seed": 8001,
                                "profile_path": "validation.jsonl",
                                "profile_sha256": "5" * 64,
                                "run_manifest_path": "validation_manifest.json",
                                "run_manifest_sha256": "6" * 64,
                            }
                        ],
                    },
                },
            )
            dpp = DPPSettings(
                4,
                16,
                float.fromhex("0x1.fffffffffffffp+1023"),
                reference_parameter_status="frozen_from_stock_positive_frame_p50",
                reference_artifact_path="artifacts/reference.json",
                reference_artifact_sha256=reference_sha,
            )
            predictor_settings = PredictorSettings(
                0.25,
                parameter_status="frozen_from_held_out_ood_calibration",
                calibration_artifact_path="artifacts/ood.json",
                calibration_artifact_sha256=ood_sha,
            )
            result = validate_frozen_v2_artifacts(
                runtime,
                dpp_settings=dpp,
                predictor_settings=predictor_settings,
                predictor=predictor,
                execution_scope="formal",
            )
            self.assertEqual(result.reference_sha256, reference_sha)

            reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
            reference_payload.update(
                {
                    "status": "frozen_development",
                    "scope": "development_nonformal",
                    "formal_benchmark_eligible": False,
                }
            )
            development_reference_sha = _write_json(
                reference_path, reference_payload
            )
            ood_payload = json.loads(ood_path.read_text(encoding="utf-8"))
            ood_payload.update(
                {
                    "status": "frozen_development",
                    "scope": "development_nonformal",
                    "formal_benchmark_eligible": False,
                }
            )
            development_ood_sha = _write_json(ood_path, ood_payload)
            development_dpp = replace(
                dpp,
                reference_parameter_status=(
                    "frozen_from_development_stock_n300_positive_frame_p50"
                ),
                reference_artifact_sha256=development_reference_sha,
            )
            development_predictor = replace(
                predictor_settings,
                parameter_status=(
                    "frozen_from_development_held_out_ood_calibration"
                ),
                calibration_artifact_sha256=development_ood_sha,
            )
            with self.assertRaisesRegex(
                ActiveConfigError, "cannot enable a formal benchmark"
            ):
                validate_frozen_v2_artifacts(
                    runtime,
                    dpp_settings=development_dpp,
                    predictor_settings=development_predictor,
                    predictor=predictor,
                    execution_scope="formal",
                )
            validate_frozen_v2_artifacts(
                runtime,
                dpp_settings=development_dpp,
                predictor_settings=development_predictor,
                predictor=predictor,
                execution_scope="development_nonformal",
            )

            reference_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ActiveConfigError, "SHA256 mismatch"):
                validate_frozen_v2_artifacts(
                    runtime,
                    dpp_settings=development_dpp,
                    predictor_settings=development_predictor,
                    predictor=predictor,
                    execution_scope="development_nonformal",
                )


if __name__ == "__main__":
    unittest.main()
