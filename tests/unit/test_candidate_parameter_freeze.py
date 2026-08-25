from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.freeze_reference_concurrency import build_artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class ReferenceConcurrencyFreezeTests(unittest.TestCase):
    def _source(
        self,
        root: Path,
        *,
        qps: float,
        seed: int,
        name: str = "normal_stock",
        development: bool = False,
    ) -> Path:
        qps_tag = str(qps).replace(".", "p")
        run_id = f"qps_{qps_tag}_seed_{seed}"
        run = root / name / "runs" / run_id
        run.mkdir(parents=True)
        profile = run / "iteration_profile.jsonl"
        prefill = (0, 2, 4, 8, 8, 12)
        decode = (0, 0, 16, 32, 32, 48)
        rows = []
        for index, (nf, nd) in enumerate(zip(prefill, decode)):
            rows.append(
                {
                    "schema_version": 2,
                    "profile_kind": "stock_natural_workload",
                    "snapshot_concurrency_semantics": "dpp_stage_queues_v2",
                    "run_id": run_id,
                    "iteration_index": index,
                    "snapshot_prefill_count": nf,
                    "snapshot_decode_count": nd,
                    "snapshot_running_count": nf + nd,
                    "snapshot_waiting_count": 0,
                    "snapshot_running_prefill_count": nf,
                    "snapshot_running_decode_count": nd,
                    "snapshot_waiting_prefill_count": 0,
                    "snapshot_waiting_decode_count": 0,
                    "snapshot_preempted_count": 0,
                    "snapshot_other_waiting_count": 0,
                    "snapshot_requests_with_preemptions_count": 0,
                    "snapshot_total_preemptions": 0,
                }
            )
        profile.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        runtime = {"model": "Qwen3-14B", "runtime": "test"}
        manifest = {
            "status": "complete",
            "kind": "qwen3_14b_stock_predictor_profile",
            "run_id": run_id,
            "qps": qps,
            "seed": seed,
            "file_sha256": {"iteration_profile.jsonl": _sha256(profile)},
            "resolved": {
                "server_command": ["vllm", "Qwen3-14B"],
                "diagnostic_prefix": development,
                "request_count": 300 if development else 500,
                "comparison_scope": (
                    "stock_profile_development_reference_n300"
                    if development
                    else "stock_profile_formal_reference"
                ),
                "qps": qps,
                "seed": seed,
                "config": "configs/dgx_spark_experiment.yaml",
                "config_sha256": "a" * 64,
                "runtime_consistency": runtime,
                "runtime_consistency_sha256": _canonical_sha256(runtime),
            },
        }
        (run / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return profile

    def _matrix(self, root: Path, *, name: str = "normal_stock") -> tuple[Path, ...]:
        return tuple(
            self._source(root, qps=qps, seed=seed, name=name)
            for qps, seed in ((0.2, 1001), (0.25, 1003), (0.3, 1005))
        )

    def _rebind_profile_hash(self, profile: Path) -> None:
        manifest_path = profile.parent / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["file_sha256"][profile.name] = _sha256(profile)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_positive_frame_p50_is_deterministic_and_provenanced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self._matrix(root)
            first = build_artifact(sources, repository=root)
            second = build_artifact(sources, repository=root)
            self.assertEqual(first["prefill_reference_concurrency"], 8)
            self.assertEqual(first["decode_reference_concurrency"], 32)
            self.assertEqual(first["prefill"]["positive_sample_count"], 15)
            self.assertEqual(first["decode"]["positive_sample_count"], 12)
            self.assertEqual(first["sources"], second["sources"])
            self.assertEqual(len(first["sources"][0]["profile_sha256"]), 64)
            self.assertEqual(first["development_operating_region_qps"], [0.2, 0.25, 0.3])

    def test_targeted_source_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = self._matrix(root, name="targeted_profile")
            with self.assertRaisesRegex(ValueError, "forbidden profiling source"):
                build_artifact(sources, repository=root)

    def test_single_n300_development_reference_is_explicitly_nonformal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(
                root,
                qps=0.25,
                seed=1003,
                name="normal_stock_development",
                development=True,
            )
            artifact = build_artifact(
                (source,), repository=root, scope="development_nonformal"
            )
            self.assertEqual(artifact["status"], "frozen_development")
            self.assertEqual(artifact["scope"], "development_nonformal")
            self.assertFalse(artifact["formal_benchmark_eligible"])
            self.assertEqual(artifact["development_operating_region_qps"], [0.25])

    def test_missing_snapshot_counts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = list(self._matrix(root))
            sources[0].write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "qps_0p2_seed_1001",
                        "iteration_index": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self._rebind_profile_hash(sources[0])
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                build_artifact(sources, repository=root)

    def test_profile_hash_and_runtime_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = list(self._matrix(root))
            sources[0].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                build_artifact(sources, repository=root)
            self._rebind_profile_hash(sources[0])
            manifest_path = sources[0].parent / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["resolved"]["runtime_consistency_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime identity"):
                build_artifact(sources, repository=root)


if __name__ == "__main__":
    unittest.main()
