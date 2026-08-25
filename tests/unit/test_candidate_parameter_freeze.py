from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.freeze_reference_concurrency import build_artifact


class ReferenceConcurrencyFreezeTests(unittest.TestCase):
    def _source(self, root: Path, *, name: str = "normal_stock") -> Path:
        run = root / name / "runs" / "qps_0p2_seed_1001"
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
                    "run_id": "qps_0p2_seed_1001",
                    "iteration_index": index,
                    "snapshot_prefill_count": nf,
                    "snapshot_decode_count": nd,
                }
            )
        profile.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (run / "run_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "kind": "qwen3_14b_stock_predictor_profile",
                    "run_id": "qps_0p2_seed_1001",
                    "qps": 0.2,
                    "seed": 1001,
                    "resolved": {"server_command": ["vllm", "Qwen3-14B"]},
                }
            ),
            encoding="utf-8",
        )
        return profile

    def test_positive_frame_p50_is_deterministic_and_provenanced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            first = build_artifact((source,), repository=root)
            second = build_artifact((source,), repository=root)
            self.assertEqual(first["prefill_reference_concurrency"], 8)
            self.assertEqual(first["decode_reference_concurrency"], 32)
            self.assertEqual(first["prefill"]["positive_sample_count"], 5)
            self.assertEqual(first["decode"]["positive_sample_count"], 4)
            self.assertEqual(first["sources"], second["sources"])
            self.assertEqual(len(first["sources"][0]["profile_sha256"]), 64)

    def test_targeted_source_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root, name="targeted_profile")
            with self.assertRaisesRegex(ValueError, "forbidden profiling source"):
                build_artifact((source,), repository=root)

    def test_missing_snapshot_counts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._source(root)
            source.write_text(
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
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                build_artifact((source,), repository=root)


if __name__ == "__main__":
    unittest.main()
