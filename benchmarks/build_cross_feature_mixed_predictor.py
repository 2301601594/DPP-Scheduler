#!/usr/bin/env python3
"""Build three segmented Mixed Ridge models with two interaction features."""

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
from benchmarks.build_segmented_mixed_predictor import (
    MIXED_ALPHA,
    _decode_counts,
    _legacy_prediction,
    _model_payload,
    _segment_evaluation,
    _segment_mask,
    _segmented_oof,
)
from benchmarks.split_predictor_ridge_data import RIDGE_DATASET_ID, validate_ridge_splits
from benchmarks.train_ridge_predictor import (
    _arrays,
    _metrics,
    _predict,
    _residual_summary,
    _rows,
    _subset_metrics,
    _support_mask,
)
from dpp_scheduler.predictor import (
    CROSS_FEATURE_SEGMENTED_ONLINE_PREDICTOR_VERSION,
    FEATURE_NAMES,
    MIXED_CROSS_FEATURE_NAMES,
    MIXED_DECODE_SEGMENTS,
)


ARTIFACT_SCHEMA_VERSION = 1
BASELINE_ARTIFACT = "ridge_mixed_decode_three_segment_online_v2"
DEFAULT_OUTPUT = "ridge_mixed_decode_three_segment_cross_online_v3"


def add_mixed_cross_features(row: dict[str, Any]) -> dict[str, Any]:
    """Add only current-plan interactions; no future output information is used."""
    result = dict(row)
    features = dict(row["features"])
    features["x_9"] = float(features["x_6"]) * float(features["x_4"])
    features["x_10"] = float(features["x_6"]) * float(features["x_5"])
    result["features"] = features
    return result


def _segmented_prediction(
    model: dict[str, Any], rows: list[dict[str, Any]]
) -> np.ndarray:
    x, _ = _arrays(rows, FEATURE_NAMES)
    counts = _decode_counts(x, FEATURE_NAMES)
    prediction = np.full(len(rows), np.nan, dtype=np.float64)
    for item, expected in zip(model["segments"], MIXED_DECODE_SEGMENTS):
        segment_id, minimum, maximum = expected
        if (
            item["segment_id"],
            item["minimum_decode_count_inclusive"],
            item["maximum_decode_count_inclusive"],
        ) != expected:
            raise ValueError("baseline segment identity mismatch")
        mask = _segment_mask(counts, minimum, maximum)
        if mask.any():
            prediction[mask] = _legacy_prediction(item["model"], x[mask])
    if not np.isfinite(prediction).all():
        raise ValueError("baseline segmented prediction coverage is incomplete")
    return prediction


def build_artifact(
    *, split_root: Path, baseline_root: Path, output_root: Path
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"append-only Predictor artifact exists: {output_root}")
    validate_ridge_splits(split_root)
    split_manifest = _json(split_root / "dataset_manifest.json")
    baseline_manifest = _json(baseline_root / "artifact_manifest.json")
    baseline_predictor = _json(baseline_root / "predictor.json")
    if baseline_manifest.get("status") != "complete":
        raise ValueError("baseline segmented Predictor artifact is incomplete")
    predictor_record = baseline_manifest.get("files", {}).get("predictor", {})
    if sha256_file(baseline_root / "predictor.json") != predictor_record.get("sha256"):
        raise ValueError("baseline segmented Predictor hash mismatch")

    raw_train_rows = _rows(
        split_root / split_manifest["files"]["mixed_train"]["file"]
    )
    raw_test_rows = _rows(
        split_root / split_manifest["files"]["mixed_test"]["file"]
    )
    train_rows = [add_mixed_cross_features(row) for row in raw_train_rows]
    test_rows = [add_mixed_cross_features(row) for row in raw_test_rows]
    names = MIXED_CROSS_FEATURE_NAMES
    x_train, y_train = _arrays(train_rows, names)
    x_test, y_test = _arrays(test_rows, names)
    oof, folds = _segmented_oof(x_train, y_train, train_rows, names)

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
    calibration = dict(
        baseline_predictor["residual_calibration"]["by_batch_kind"]
    )
    calibration["mixed"] = {
        "window_size": int(selected_window["selected"]["window_size"]),
        "minimum_samples": MINIMUM_SAMPLES,
        **cold,
    }

    models = dict(baseline_predictor["models"])
    models["mixed"] = {
        "dispatch": "decode_count_segments",
        "dispatch_feature": "x_4",
        "segments": segment_models,
    }
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "predictor_version": CROSS_FEATURE_SEGMENTED_ONLINE_PREDICTOR_VERSION,
        "model_family": "ridge_regression",
        "prediction_target": "current_plan_iteration_duration_seconds",
        "scenario_dispatch": list(BATCH_KINDS),
        "models": models,
        "residual_calibration": {
            **baseline_predictor["residual_calibration"],
            "by_batch_kind": calibration,
        },
    }
    baseline_prediction = _segmented_prediction(
        baseline_predictor["models"]["mixed"], raw_test_rows
    )
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "predictor_version": CROSS_FEATURE_SEGMENTED_ONLINE_PREDICTOR_VERSION,
        "design_status": "candidate_pending_fresh_independent_validation",
        "training_data": {
            "dataset_id": RIDGE_DATASET_ID,
            "dataset_manifest_sha256": sha256_file(
                split_root / "dataset_manifest.json"
            ),
            "mixed_train_rows": len(train_rows),
            "mixed_held_out_rows": len(test_rows),
        },
        "controlled_change": {
            "unchanged": [
                "three_decode_segment_boundaries",
                "ridge_model_family",
                "alpha_10",
                "training_and_held_out_split",
                "base_features_x1_through_x8",
            ],
            "added_to_all_three_mixed_models": {
                "x_9": "x_6_total_prefill_tokens*x_4_decode_token_count",
                "x_10": "x_6_total_prefill_tokens*x_5_decode_context_total",
            },
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
        "historical_held_out_controlled_comparison": {
            "validation_role": "development_evidence_not_fresh_independent_data",
            "cross_feature_overall": _metrics(y_test, test_prediction),
            "baseline_segmented_overall": _metrics(y_test, baseline_prediction),
            "cross_feature_by_decode_segment": _segment_evaluation(
                rows=test_rows,
                x=x_test,
                actual=y_test,
                prediction=test_prediction,
                names=names,
            ),
            "cross_feature_by_source_kind": _subset_metrics(
                test_rows, y_test, test_prediction
            ),
            "support": {
                "in_support_rows": int(in_support.sum()),
                "out_of_support_rows": int((~in_support).sum()),
                "out_of_support_rate": float(np.mean(~in_support)),
            },
        },
        "fresh_independent_validation": {
            "status": "required_before_adoption",
            "existing_n200_run_is_now_post_hoc_development_data": True,
        },
        "baseline_segmented_artifact_manifest_sha256": sha256_file(
            baseline_root / "artifact_manifest.json"
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
            "predictor_version": CROSS_FEATURE_SEGMENTED_ONLINE_PREDICTOR_VERSION,
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

    predictor = RidgeDurationPredictor.from_artifact(path)
    if predictor.predictor_version != CROSS_FEATURE_SEGMENTED_ONLINE_PREDICTOR_VERSION:
        raise ValueError("cross-feature Predictor version mismatch")
    manifest = _json(path / "artifact_manifest.json")
    for record in manifest.get("files", {}).values():
        file_path = (path / str(record["file"])).resolve()
        if file_path.parent != path.resolve() or sha256_file(file_path) != record["sha256"]:
            raise ValueError("cross-feature Predictor artifact path/hash mismatch")
    return {"valid": True, "predictor_version": predictor.predictor_version}


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "status"))
    parser.add_argument(
        "--splits",
        default=str(repository / "results" / "dataset" / RIDGE_DATASET_ID),
    )
    parser.add_argument(
        "--baseline-segmented",
        default=str(repository / "predictors" / "qwen3_14b" / BASELINE_ARTIFACT),
    )
    parser.add_argument(
        "--output",
        default=str(repository / "predictors" / "qwen3_14b" / DEFAULT_OUTPUT),
    )
    args = parser.parse_args()
    output = Path(args.output)
    if args.command == "status":
        value = (
            _json(output / "artifact_manifest.json")
            if output.exists()
            else {
                "predictor_version": CROSS_FEATURE_SEGMENTED_ONLINE_PREDICTOR_VERSION,
                "status": "absent",
            }
        )
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "build":
        report = build_artifact(
            split_root=Path(args.splits),
            baseline_root=Path(args.baseline_segmented),
            output_root=output,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(validate_artifact(output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
