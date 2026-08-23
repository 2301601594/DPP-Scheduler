#!/usr/bin/env python3
"""Select residual windows from training OOF predictions and build runtime artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.build_predictor_dataset import BATCH_KINDS, _json, sha256_file
from benchmarks.split_predictor_ridge_data import (
    ACTIVE_FEATURES,
    RIDGE_DATASET_ID,
    validate_ridge_splits,
)
from benchmarks.train_ridge_predictor import (
    PREDICTOR_VERSION as BASE_PREDICTOR_VERSION,
    _arrays,
    _feature_group_folds,
    _oof_predictions,
    _rows,
    _run_folds,
)
from dpp_scheduler.predictor import ONLINE_PREDICTOR_VERSION


ONLINE_ARTIFACT_SCHEMA_VERSION = 1
WINDOW_CANDIDATES = (32, 64, 128)
MINIMUM_SAMPLES = 32
SPLIT_CONTENT_IDENTITY = "configs/predictor_ridge_split_content_v1.json"


def _gzip_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_rebuilt_splits_and_base(
    *, split_root: Path, base_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_ridge_splits(split_root)
    repository = Path(__file__).resolve().parents[1]
    identity_path = repository / SPLIT_CONTENT_IDENTITY
    identity = _json(identity_path)
    split_manifest = _json(split_root / "dataset_manifest.json")
    if identity.get("dataset_id") != RIDGE_DATASET_ID:
        raise ValueError("Ridge split content identity mismatch")
    expected_content = identity.get("decompressed_jsonl_sha256", {})
    if set(expected_content) != set(split_manifest.get("files", {})):
        raise ValueError("Ridge split content identity file coverage mismatch")
    for key, expected_hash in expected_content.items():
        path = split_root / split_manifest["files"][key]["file"]
        if _gzip_content_sha256(path) != expected_hash:
            raise ValueError(f"rebuilt Ridge split content mismatch: {key}")

    artifact_manifest = _json(base_root / "artifact_manifest.json")
    if (
        artifact_manifest.get("status") != "complete"
        or artifact_manifest.get("predictor_version") != BASE_PREDICTOR_VERSION
    ):
        raise ValueError("base Predictor artifact identity/status mismatch")
    for record in artifact_manifest.get("files", {}).values():
        path = (base_root / record["file"]).resolve()
        if path.parent != base_root.resolve() or sha256_file(path) != record["sha256"]:
            raise ValueError("base Predictor artifact path/hash mismatch")
    report = _json(base_root / "training_report.json")
    if report.get("dataset_manifest_sha256") != identity.get(
        "original_dataset_manifest_sha256"
    ):
        raise ValueError("base Predictor does not reference the reviewed split identity")
    return split_manifest, artifact_manifest, identity


def _higher_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * (len(ordered) - 1))]


def _cold_start(residuals: np.ndarray) -> dict[str, float]:
    mean = float(residuals.mean())
    centered = [float(value - mean) for value in residuals]
    margin = _higher_quantile(centered, 0.95)
    if margin < 0:
        raise ValueError("OOF centered P95 cannot be a conservative margin")
    return {
        "cold_start_mean_seconds": mean,
        "cold_start_centered_p95_seconds": margin,
    }


def evaluate_window(
    *,
    rows: list[dict[str, Any]],
    actual: np.ndarray,
    base_prediction: np.ndarray,
    residuals: np.ndarray,
    window_size: int,
) -> dict[str, float | int]:
    by_run: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_run[str(row["run_id"])].append(index)
    absolute_errors: list[float] = []
    signed_errors: list[float] = []
    covered: list[bool] = []
    for run_id in sorted(by_run):
        ordered = sorted(
            by_run[run_id], key=lambda index: int(rows[index]["iteration_index"])
        )
        history: list[float] = []
        for index in ordered:
            if len(history) >= MINIMUM_SAMPLES:
                window = history[-window_size:]
                mean = sum(window) / len(window)
                centered_p95 = _higher_quantile(
                    [value - mean for value in window], 0.95
                )
                expected = float(base_prediction[index]) + mean
                conservative = expected + centered_p95
                error = expected - float(actual[index])
                absolute_errors.append(abs(error))
                signed_errors.append(error)
                covered.append(float(actual[index]) <= conservative)
            history.append(float(residuals[index]))
    if not absolute_errors:
        raise ValueError("window selection has no post-warmup rows")
    return {
        "window_size": window_size,
        "evaluated_rows": len(absolute_errors),
        "expected_mae_seconds": sum(absolute_errors) / len(absolute_errors),
        "expected_mean_error_seconds": sum(signed_errors) / len(signed_errors),
        "conservative_coverage": sum(covered) / len(covered),
    }


def select_window(candidates: list[dict[str, float | int]]) -> dict[str, Any]:
    qualifying = [
        item for item in candidates if float(item["conservative_coverage"]) >= 0.95
    ]
    if qualifying:
        selected = min(
            qualifying,
            key=lambda item: (
                float(item["expected_mae_seconds"]), int(item["window_size"])
            ),
        )
        reason = "coverage_at_least_0.95_then_lowest_expected_mae"
    else:
        selected = min(
            candidates,
            key=lambda item: (
                -float(item["conservative_coverage"]),
                float(item["expected_mae_seconds"]),
                int(item["window_size"]),
            ),
        )
        reason = "no_candidate_met_coverage_choose_highest_coverage_then_mae"
    return {"selected": selected, "selection_reason": reason}


def build_online_artifact(
    *, split_root: Path, base_root: Path, output_root: Path
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"append-only online artifact exists: {output_root}")
    split_manifest, base_manifest, split_identity = _validate_rebuilt_splits_and_base(
        split_root=split_root, base_root=base_root
    )
    base_predictor = _json(base_root / "predictor.json")
    if base_predictor.get("predictor_version") != BASE_PREDICTOR_VERSION:
        raise ValueError("base Predictor version mismatch")

    selection: dict[str, Any] = {}
    online_models: dict[str, Any] = {}
    by_kind: dict[str, Any] = {}
    for kind in BATCH_KINDS:
        names = ACTIVE_FEATURES[kind]
        train_rows = _rows(
            split_root / split_manifest["files"][f"{kind}_train"]["file"]
        )
        x, actual = _arrays(train_rows, names)
        folds = _feature_group_folds(x) if kind == "prefill_only" else _run_folds(train_rows)
        alpha = float(base_predictor["models"][kind]["alpha"])
        oof = _oof_predictions(x, actual, folds, alpha)
        residuals = actual - oof
        candidates = [
            evaluate_window(
                rows=train_rows,
                actual=actual,
                base_prediction=oof,
                residuals=residuals,
                window_size=size,
            )
            for size in WINDOW_CANDIDATES
        ]
        selected = select_window(candidates)
        cold = _cold_start(residuals)
        selection[kind] = {
            "cv_scheme": (
                "five_fold_identical_feature_grouped"
                if kind == "prefill_only"
                else "leave_one_training_run_out"
            ),
            "oof_rows": len(train_rows),
            "minimum_samples": MINIMUM_SAMPLES,
            "candidates": candidates,
            **selected,
            **cold,
        }
        online_models[kind] = base_predictor["models"][kind]
        by_kind[kind] = {
            "window_size": int(selected["selected"]["window_size"]),
            "minimum_samples": MINIMUM_SAMPLES,
            **cold,
        }

    payload = {
        "schema_version": ONLINE_ARTIFACT_SCHEMA_VERSION,
        "predictor_version": ONLINE_PREDICTOR_VERSION,
        "base_predictor_version": BASE_PREDICTOR_VERSION,
        "base_predictor_sha256": sha256_file(base_root / "predictor.json"),
        "model_family": "ridge_regression",
        "prediction_target": "current_plan_iteration_duration_seconds",
        "models": online_models,
        "residual_calibration": {
            "strategy": "online_window_per_batch_kind",
            "residual_definition": "actual_duration_seconds-base_duration_seconds",
            "expected_statistic": "mean",
            "conservative_statistic": "centered_p95",
            "quantile_method": "higher",
            "cold_start": "offline_oof_by_batch_kind",
            "reset_scope": "server_process",
            "by_batch_kind": by_kind,
        },
    }
    report = {
        "schema_version": ONLINE_ARTIFACT_SCHEMA_VERSION,
        "predictor_version": ONLINE_PREDICTOR_VERSION,
        "source_split_dataset_id": RIDGE_DATASET_ID,
        "source_split_manifest_sha256": sha256_file(
            split_root / "dataset_manifest.json"
        ),
        "source_split_content_identity_file": SPLIT_CONTENT_IDENTITY,
        "source_split_content_identity_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / SPLIT_CONTENT_IDENTITY
        ),
        "source_split_original_manifest_sha256": split_identity[
            "original_dataset_manifest_sha256"
        ],
        "base_artifact_manifest_sha256": sha256_file(
            base_root / "artifact_manifest.json"
        ),
        "base_artifact_files": base_manifest["files"],
        "test_split_used_for_window_selection": False,
        "selection_by_batch_kind": selection,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    try:
        predictor_path = temporary / "predictor.json"
        report_path = temporary / "window_selection_report.json"
        predictor_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": ONLINE_ARTIFACT_SCHEMA_VERSION,
            "predictor_version": ONLINE_PREDICTOR_VERSION,
            "status": "complete",
            "files": {
                "predictor": {
                    "file": predictor_path.name,
                    "sha256": sha256_file(predictor_path),
                },
                "window_selection_report": {
                    "file": report_path.name,
                    "sha256": sha256_file(report_path),
                },
            },
        }
        (temporary / "artifact_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_root)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_online_artifact(path: Path) -> dict[str, Any]:
    manifest = _json(path / "artifact_manifest.json")
    if (
        manifest.get("schema_version") != ONLINE_ARTIFACT_SCHEMA_VERSION
        or manifest.get("predictor_version") != ONLINE_PREDICTOR_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("online artifact identity/status mismatch")
    for record in manifest.get("files", {}).values():
        file_path = (path / str(record["file"])).resolve()
        if file_path.parent != path.resolve() or sha256_file(file_path) != record["sha256"]:
            raise ValueError("online artifact path/hash mismatch")
    from dpp_scheduler.predictor import RidgeDurationPredictor

    RidgeDurationPredictor.from_artifact(path)
    return {"valid": True, "predictor_version": ONLINE_PREDICTOR_VERSION}


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "status"))
    parser.add_argument(
        "--splits",
        default=str(repository / "results" / "dataset" / RIDGE_DATASET_ID),
    )
    parser.add_argument(
        "--base",
        default=str(repository / "predictors" / "qwen3_14b" / "ridge_three_scenario_v1"),
    )
    parser.add_argument(
        "--output",
        default=str(repository / "predictors" / "qwen3_14b" / "ridge_three_scenario_online_v1"),
    )
    args = parser.parse_args()
    output = Path(args.output)
    if args.command == "status":
        print(
            json.dumps(
                _json(output / "artifact_manifest.json")
                if output.exists()
                else {"predictor_version": ONLINE_PREDICTOR_VERSION, "status": "absent"},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build":
        report = build_online_artifact(
            split_root=Path(args.splits), base_root=Path(args.base), output_root=output
        )
        print(json.dumps(report["selection_by_batch_kind"], indent=2, sort_keys=True))
    print(json.dumps(validate_online_artifact(output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
