#!/usr/bin/env python3
"""Build the three-segment Mixed Ridge Predictor from frozen training splits."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.build_online_ridge_predictor import (
    MINIMUM_SAMPLES,
    WINDOW_CANDIDATES,
    _cold_start,
    evaluate_window,
    select_window,
)
from benchmarks.build_predictor_dataset import BATCH_KINDS, _json, sha256_file
from benchmarks.split_predictor_ridge_data import (
    ACTIVE_FEATURES,
    RIDGE_DATASET_ID,
    validate_ridge_splits,
)
from benchmarks.train_ridge_predictor import (
    _arrays,
    _metrics,
    _predict,
    _residual_summary,
    _rows,
    _run_folds,
    _subset_metrics,
    _support,
    _support_mask,
    fit_ridge,
)
from dpp_scheduler.predictor import (
    MIXED_DECODE_SEGMENTS,
    SEGMENTED_ONLINE_PREDICTOR_VERSION,
)


ARTIFACT_SCHEMA_VERSION = 1
MIXED_ALPHA = 10.0
LEGACY_ONLINE_ARTIFACT = "ridge_three_scenario_online_v1"
DEFAULT_OUTPUT = "ridge_mixed_decode_three_segment_online_v2"


def _model_payload(
    names: tuple[str, ...], x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[dict[str, Any], tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    intercept, coefficients, mean, scale = fit_ridge(x, y, alpha)
    return (
        {
            "active_features": list(names),
            "alpha": alpha,
            "intercept_seconds": intercept,
            "coefficients_for_standardized_features": {
                name: float(coefficients[index]) for index, name in enumerate(names)
            },
            "standardization": {
                name: {"mean": float(mean[index]), "scale": float(scale[index])}
                for index, name in enumerate(names)
            },
            "support_domain_train_marginal_box": _support(names, x),
        },
        (intercept, coefficients, mean, scale),
    )


def _decode_counts(x: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    values = x[:, names.index("x_4")]
    if not np.all(np.equal(values, np.floor(values))):
        raise ValueError("Mixed x_4 Decode counts must be integers")
    return values.astype(np.int64)


def _segment_mask(counts: np.ndarray, minimum: int, maximum: int) -> np.ndarray:
    return (counts >= minimum) & (counts <= maximum)


def _segmented_oof(
    x: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, Any]],
    names: tuple[str, ...],
) -> tuple[np.ndarray, list[np.ndarray]]:
    folds = _run_folds(rows)
    counts = _decode_counts(x, names)
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    all_indices = np.arange(len(y))
    for validation in folds:
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        for _, minimum, maximum in MIXED_DECODE_SEGMENTS:
            train_indices = training[
                _segment_mask(counts[training], minimum, maximum)
            ]
            validation_indices = validation[
                _segment_mask(counts[validation], minimum, maximum)
            ]
            if not len(validation_indices):
                continue
            if len(train_indices) < 2:
                raise ValueError("segmented OOF training fold is undersized")
            fitted = fit_ridge(x[train_indices], y[train_indices], MIXED_ALPHA)
            prediction[validation_indices] = _predict(x[validation_indices], *fitted)
    if not np.isfinite(prediction).all():
        raise ValueError("segmented OOF prediction coverage is incomplete")
    return prediction, folds


def _segment_evaluation(
    *,
    rows: list[dict[str, Any]],
    x: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, Any]:
    del rows
    counts = _decode_counts(x, names)
    return {
        segment_id: {
            "rows": int(mask.sum()),
            "metrics": _metrics(actual[mask], prediction[mask]),
        }
        for segment_id, minimum, maximum in MIXED_DECODE_SEGMENTS
        for mask in (_segment_mask(counts, minimum, maximum),)
    }


def _legacy_prediction(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    names = tuple(model["active_features"])
    coefficients = np.asarray(
        [model["coefficients_for_standardized_features"][name] for name in names],
        dtype=np.float64,
    )
    mean = np.asarray(
        [model["standardization"][name]["mean"] for name in names],
        dtype=np.float64,
    )
    scale = np.asarray(
        [model["standardization"][name]["scale"] for name in names],
        dtype=np.float64,
    )
    return _predict(x, float(model["intercept_seconds"]), coefficients, mean, scale)


def build_artifact(
    *, split_root: Path, legacy_root: Path, output_root: Path
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"append-only Predictor artifact exists: {output_root}")
    validate_ridge_splits(split_root)
    split_manifest = _json(split_root / "dataset_manifest.json")
    legacy_manifest = _json(legacy_root / "artifact_manifest.json")
    legacy_predictor = _json(legacy_root / "predictor.json")
    if legacy_manifest.get("status") != "complete":
        raise ValueError("legacy online Predictor artifact is incomplete")
    predictor_record = legacy_manifest.get("files", {}).get("predictor", {})
    if sha256_file(legacy_root / "predictor.json") != predictor_record.get("sha256"):
        raise ValueError("legacy online Predictor hash mismatch")

    names = ACTIVE_FEATURES["mixed"]
    train_rows = _rows(split_root / split_manifest["files"]["mixed_train"]["file"])
    test_rows = _rows(split_root / split_manifest["files"]["mixed_test"]["file"])
    x_train, y_train = _arrays(train_rows, names)
    x_test, y_test = _arrays(test_rows, names)
    oof, folds = _segmented_oof(x_train, y_train, train_rows, names)

    # Fit each segment on every eligible training row, then evaluate the untouched
    # historical test split. A second, newly generated run is required separately.
    train_counts = _decode_counts(x_train, names)
    test_counts = _decode_counts(x_test, names)
    test_prediction = np.full(len(y_test), np.nan, dtype=np.float64)
    in_support = np.zeros(len(y_test), dtype=bool)
    segment_models: list[dict[str, Any]] = []
    for segment_id, minimum, maximum in MIXED_DECODE_SEGMENTS:
        train_mask = _segment_mask(train_counts, minimum, maximum)
        test_mask = _segment_mask(test_counts, minimum, maximum)
        if int(train_mask.sum()) < 2 or not test_mask.any():
            raise ValueError(f"segment {segment_id} lacks train or held-out rows")
        model, fitted = _model_payload(
            names, x_train[train_mask], y_train[train_mask], MIXED_ALPHA
        )
        test_prediction[test_mask] = _predict(x_test[test_mask], *fitted)
        in_support[test_mask] = _support_mask(
            names, x_test[test_mask], model["support_domain_train_marginal_box"]
        )
        segment_models.append(
            {
                "segment_id": segment_id,
                "minimum_decode_count_inclusive": minimum,
                "maximum_decode_count_inclusive": maximum,
                "training_rows": int(train_mask.sum()),
                "model": model,
            }
        )
    if not np.isfinite(test_prediction).all():
        raise ValueError("held-out Mixed rows fall outside fixed segment coverage")

    residuals = y_train - oof
    window_candidates = [
        evaluate_window(
            rows=train_rows,
            actual=y_train,
            base_prediction=oof,
            residuals=residuals,
            window_size=size,
        )
        for size in WINDOW_CANDIDATES
    ]
    selected_window = select_window(window_candidates)
    cold = _cold_start(residuals)
    calibration = dict(legacy_predictor["residual_calibration"]["by_batch_kind"])
    calibration["mixed"] = {
        "window_size": int(selected_window["selected"]["window_size"]),
        "minimum_samples": MINIMUM_SAMPLES,
        **cold,
    }

    models = dict(legacy_predictor["models"])
    models["mixed"] = {
        "dispatch": "decode_count_segments",
        "dispatch_feature": "x_4",
        "segments": segment_models,
    }
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "predictor_version": SEGMENTED_ONLINE_PREDICTOR_VERSION,
        "model_family": "ridge_regression",
        "prediction_target": "current_plan_iteration_duration_seconds",
        "scenario_dispatch": list(BATCH_KINDS),
        "models": models,
        "residual_calibration": {
            **legacy_predictor["residual_calibration"],
            "by_batch_kind": calibration,
        },
    }
    legacy_mixed_prediction = _legacy_prediction(
        legacy_predictor["models"]["mixed"], x_test
    )
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "predictor_version": SEGMENTED_ONLINE_PREDICTOR_VERSION,
        "design_status": "candidate_pending_fresh_independent_validation",
        "training_data": {
            "dataset_id": RIDGE_DATASET_ID,
            "dataset_manifest_sha256": sha256_file(
                split_root / "dataset_manifest.json"
            ),
            "mixed_train_rows": len(train_rows),
            "mixed_held_out_rows": len(test_rows),
        },
        "design": {
            "scope": "Mixed_only",
            "dispatch_feature": "x_4_decode_count",
            "segments": [
                {
                    "segment_id": identity,
                    "minimum_decode_count_inclusive": minimum,
                    "maximum_decode_count_inclusive": maximum,
                }
                for identity, minimum, maximum in MIXED_DECODE_SEGMENTS
            ],
            "fixed_alpha": MIXED_ALPHA,
            "features": list(names),
            "residual_window_scope": "single_Mixed_window_not_per_segment",
        },
        "training_oof": {
            "scheme": "leave_one_training_run_out_with_fit_per_decode_segment",
            "fold_rows": [int(len(fold)) for fold in folds],
            "metrics": _metrics(y_train, oof),
            "residual_actual_minus_prediction": _residual_summary(y_train, oof),
            "window_candidates": window_candidates,
            **selected_window,
            **cold,
        },
        "historical_held_out_check_not_independent_of_design": {
            "segmented_overall": _metrics(y_test, test_prediction),
            "legacy_global_overall": _metrics(y_test, legacy_mixed_prediction),
            "segmented_by_decode_segment": _segment_evaluation(
                rows=test_rows,
                x=x_test,
                actual=y_test,
                prediction=test_prediction,
                names=names,
            ),
            "segmented_by_source_kind": _subset_metrics(
                test_rows, y_test, test_prediction
            ),
            "support": {
                "in_support_rows": int(in_support.sum()),
                "out_of_support_rows": int((~in_support).sum()),
                "out_of_support_rate": float(np.mean(~in_support)),
            },
        },
        "fresh_independent_validation": {
            "status": "pending",
            "data_must_not_participate_in_segment_or_model_design": True,
        },
        "legacy_online_artifact_manifest_sha256": sha256_file(
            legacy_root / "artifact_manifest.json"
        ),
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    try:
        predictor_path = temporary / "predictor.json"
        report_path = temporary / "training_report.json"
        predictor_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "predictor_version": SEGMENTED_ONLINE_PREDICTOR_VERSION,
            "status": "complete",
            "files": {
                "predictor": {
                    "file": predictor_path.name,
                    "sha256": sha256_file(predictor_path),
                },
                "training_report": {
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


def validate_artifact(path: Path) -> dict[str, Any]:
    from dpp_scheduler.predictor import RidgeDurationPredictor

    RidgeDurationPredictor.from_artifact(path)
    manifest = _json(path / "artifact_manifest.json")
    for record in manifest.get("files", {}).values():
        file_path = (path / str(record["file"])).resolve()
        if file_path.parent != path.resolve() or sha256_file(file_path) != record["sha256"]:
            raise ValueError("segmented Predictor artifact path/hash mismatch")
    return {"valid": True, "predictor_version": SEGMENTED_ONLINE_PREDICTOR_VERSION}


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "status"))
    parser.add_argument(
        "--splits",
        default=str(repository / "results" / "dataset" / RIDGE_DATASET_ID),
    )
    parser.add_argument(
        "--legacy-online",
        default=str(repository / "predictors" / "qwen3_14b" / LEGACY_ONLINE_ARTIFACT),
    )
    parser.add_argument(
        "--output",
        default=str(repository / "predictors" / "qwen3_14b" / DEFAULT_OUTPUT),
    )
    args = parser.parse_args()
    output = Path(args.output)
    if args.command == "status":
        print(
            json.dumps(
                _json(output / "artifact_manifest.json")
                if output.exists()
                else {
                    "predictor_version": SEGMENTED_ONLINE_PREDICTOR_VERSION,
                    "status": "absent",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build":
        report = build_artifact(
            split_root=Path(args.splits),
            legacy_root=Path(args.legacy_online),
            output_root=output,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(validate_artifact(output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
