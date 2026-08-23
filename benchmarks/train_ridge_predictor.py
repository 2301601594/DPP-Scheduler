#!/usr/bin/env python3
"""Train and validate three scenario-specific Ridge duration models locally."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.build_predictor_dataset import BATCH_KINDS, _json, sha256_file
from benchmarks.split_predictor_ridge_data import (
    ACTIVE_FEATURES,
    RIDGE_DATASET_ID,
    validate_ridge_splits,
)


PREDICTOR_VERSION = "qwen3-14b-ridge-three-scenario-v1"
ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_ALPHAS = tuple(float(value) for value in np.logspace(-6, 6, 25))


def _rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _arrays(
    rows: list[dict[str, Any]], names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(row["features"][name]) for name in names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray(
        [float(row["actual_duration_seconds"]) for row in rows], dtype=np.float64
    )
    if x.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("invalid or undersized Ridge training matrix")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("Ridge data contains non-finite features or invalid durations")
    return x, y


def _scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    # A feature can be constant in an internal CV fold even though it is active
    # in the full training set. Leaving its standardized value at zero is safe.
    scale = np.where(scale > 0, scale, 1.0)
    return mean, scale


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if mean is None or scale is None:
        mean, scale = _scaler(x)
    z = (x - mean) / scale
    intercept = float(y.mean())
    gram = z.T @ z + alpha * np.eye(z.shape[1], dtype=np.float64)
    rhs = z.T @ (y - intercept)
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    if not np.isfinite(coefficients).all() or not math.isfinite(intercept):
        raise ValueError("Ridge fit produced non-finite parameters")
    return intercept, coefficients, mean, scale


def _predict(
    x: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return intercept + ((x - mean) / scale) @ coefficients


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - y
    absolute = np.abs(error)
    under = np.maximum(y - prediction, 0.0)
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "rows": int(len(y)),
        "mae_seconds": float(absolute.mean()),
        "rmse_seconds": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error_seconds": float(np.median(absolute)),
        "p95_absolute_error_seconds": float(np.quantile(absolute, 0.95)),
        "mean_error_seconds": float(error.mean()),
        "r2": float(1.0 - np.sum(error**2) / denominator)
        if denominator > 0
        else None,
        "underprediction_rate": float(np.mean(prediction < y)),
        "underprediction_p95_seconds": float(np.quantile(under, 0.95)),
        "nonpositive_predictions": int(np.sum(prediction <= 0)),
    }


def _run_folds(rows: list[dict[str, Any]]) -> list[np.ndarray]:
    runs = sorted({str(row["run_id"]) for row in rows})
    if len(runs) < 2:
        raise ValueError("run-grouped CV requires at least two runs")
    run_array = np.asarray([str(row["run_id"]) for row in rows])
    return [np.flatnonzero(run_array == run_id) for run_id in runs]


def _feature_group_folds(x: np.ndarray, fold_count: int = 5) -> list[np.ndarray]:
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, vector in enumerate(x):
        groups.setdefault(tuple(float(value) for value in vector), []).append(index)
    if len(groups) < fold_count:
        raise ValueError("not enough distinct feature vectors for grouped CV")
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    # Largest groups are distributed first. The digest makes equal-size ordering
    # deterministic without depending on input order.
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(repr(item[0]).encode("utf-8")).hexdigest(),
        ),
    )
    for _, indices in ordered:
        destination = min(range(fold_count), key=lambda i: (len(folds[i]), i))
        folds[destination].extend(indices)
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[np.ndarray],
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> tuple[float, list[dict[str, float]]]:
    if not folds or any(len(fold) == 0 for fold in folds):
        raise ValueError("CV folds must be non-empty")
    all_indices = np.arange(len(y))
    results: list[dict[str, float]] = []
    for alpha in alphas:
        fold_mae: list[float] = []
        fold_rmse: list[float] = []
        for validation in folds:
            training = np.setdiff1d(all_indices, validation, assume_unique=True)
            if len(training) < 2:
                raise ValueError("CV training fold is undersized")
            intercept, coefficients, mean, scale = fit_ridge(
                x[training], y[training], alpha
            )
            prediction = _predict(
                x[validation], intercept, coefficients, mean, scale
            )
            error = prediction - y[validation]
            fold_mae.append(float(np.mean(np.abs(error))))
            fold_rmse.append(float(np.sqrt(np.mean(error**2))))
        results.append(
            {
                "alpha": float(alpha),
                "macro_fold_mae_seconds": float(np.mean(fold_mae)),
                "macro_fold_rmse_seconds": float(np.mean(fold_rmse)),
            }
        )
    selected = min(
        results,
        key=lambda result: (
            result["macro_fold_mae_seconds"],
            result["macro_fold_rmse_seconds"],
            result["alpha"],
        ),
    )
    return selected["alpha"], results


def _oof_predictions(
    x: np.ndarray, y: np.ndarray, folds: list[np.ndarray], alpha: float
) -> np.ndarray:
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    all_indices = np.arange(len(y))
    for validation in folds:
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        intercept, coefficients, mean, scale = fit_ridge(
            x[training], y[training], alpha
        )
        prediction[validation] = _predict(
            x[validation], intercept, coefficients, mean, scale
        )
    if not np.isfinite(prediction).all():
        raise ValueError("OOF prediction coverage is incomplete")
    return prediction


def _residual_summary(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = y - prediction
    centered = residual - residual.mean()
    return {
        "rows": int(len(residual)),
        "mean_seconds": float(residual.mean()),
        "median_seconds": float(np.median(residual)),
        "p95_seconds": float(np.quantile(residual, 0.95)),
        "centered_p95_seconds": float(np.quantile(centered, 0.95)),
    }


def _support(names: tuple[str, ...], x: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {"min": float(x[:, index].min()), "max": float(x[:, index].max())}
        for index, name in enumerate(names)
    }


def _support_mask(
    names: tuple[str, ...], x: np.ndarray, support: dict[str, dict[str, float]]
) -> np.ndarray:
    mask = np.ones(len(x), dtype=bool)
    for index, name in enumerate(names):
        mask &= x[:, index] >= support[name]["min"]
        mask &= x[:, index] <= support[name]["max"]
    return mask


def _subset_metrics(
    rows: list[dict[str, Any]], y: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    source = np.asarray([str(row["source_kind"]) for row in rows])
    result = {}
    for value in sorted(set(source)):
        mask = source == value
        result[value] = _metrics(y[mask], prediction[mask])
    return result


def train_predictors(
    *,
    input_root: Path,
    output_root: Path,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"append-only Predictor artifact exists: {output_root}")
    validate_ridge_splits(input_root)
    dataset_manifest_path = input_root / "dataset_manifest.json"
    dataset_manifest = _json(dataset_manifest_path)
    models: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}

    for kind in BATCH_KINDS:
        names = ACTIVE_FEATURES[kind]
        train_rows = _rows(input_root / dataset_manifest["files"][f"{kind}_train"]["file"])
        test_rows = _rows(input_root / dataset_manifest["files"][f"{kind}_test"]["file"])
        x_train, y_train = _arrays(train_rows, names)
        x_test, y_test = _arrays(test_rows, names)
        if kind == "prefill_only":
            folds = _feature_group_folds(x_train)
            cv_scheme = "five_fold_identical_feature_grouped"
        else:
            folds = _run_folds(train_rows)
            cv_scheme = "leave_one_training_run_out"
        alpha, cv_results = select_alpha(x_train, y_train, folds, alphas)
        oof_prediction = _oof_predictions(x_train, y_train, folds, alpha)

        recorded_scaler = dataset_manifest["standardization"]["by_batch_kind"][kind]
        mean = np.asarray([recorded_scaler[name]["mean"] for name in names])
        scale = np.asarray([recorded_scaler[name]["scale"] for name in names])
        computed_mean, computed_scale = _scaler(x_train)
        if not np.allclose(mean, computed_mean, rtol=1e-12, atol=1e-12) or not np.allclose(
            scale, computed_scale, rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"train-only scaler mismatch for {kind}")
        intercept, coefficients, _, _ = fit_ridge(
            x_train, y_train, alpha, mean=mean, scale=scale
        )
        test_prediction = _predict(x_test, intercept, coefficients, mean, scale)
        support = _support(names, x_train)
        in_support = _support_mask(names, x_test, support)

        models[kind] = {
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
            "support_domain_train_marginal_box": support,
        }
        evaluation[kind] = {
            "cv": {
                "scheme": cv_scheme,
                "fold_rows": [int(len(fold)) for fold in folds],
                "selection_metric": "lowest macro fold MAE; RMSE then alpha tie-break",
                "selected_alpha": alpha,
                "candidates": cv_results,
            },
            "train_oof": {
                "metrics": _metrics(y_train, oof_prediction),
                "residual_actual_minus_prediction": _residual_summary(
                    y_train, oof_prediction
                ),
            },
            "held_out_test": {
                "overall": _metrics(y_test, test_prediction),
                "by_source_kind": _subset_metrics(test_rows, y_test, test_prediction),
                "support": {
                    "in_support_rows": int(in_support.sum()),
                    "out_of_support_rows": int((~in_support).sum()),
                    "out_of_support_rate": float(np.mean(~in_support)),
                    "in_support_metrics": _metrics(
                        y_test[in_support], test_prediction[in_support]
                    )
                    if in_support.any()
                    else None,
                    "out_of_support_metrics": _metrics(
                        y_test[~in_support], test_prediction[~in_support]
                    )
                    if (~in_support).any()
                    else None,
                },
            },
        }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    try:
        predictor = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "predictor_version": PREDICTOR_VERSION,
            "model_family": "ridge_regression",
            "prediction_target": "current_plan_iteration_duration_seconds",
            "scenario_dispatch": list(BATCH_KINDS),
            "models": models,
            "residual_calibration": {
                "applied_to_base_prediction": False,
                "future_strategy": "online_window_per_batch_kind",
                "window_size": None,
                "minimum_samples": None,
                "expected_residual_statistic": None,
                "conservative_residual_quantile": None,
            },
        }
        report = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "predictor_version": PREDICTOR_VERSION,
            "dataset_id": RIDGE_DATASET_ID,
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "test_policy": "held-out test used once after per-scenario alpha selection",
            "alpha_grid": list(alphas),
            "evaluation": evaluation,
        }
        predictor_path = temporary / "predictor.json"
        report_path = temporary / "training_report.json"
        predictor_path.write_text(
            json.dumps(predictor, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "predictor_version": PREDICTOR_VERSION,
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
            json.dumps(artifact_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_root)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_artifact(*, input_root: Path, artifact_root: Path) -> dict[str, Any]:
    validate_ridge_splits(input_root)
    artifact = _json(artifact_root / "artifact_manifest.json")
    if (
        artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or artifact.get("predictor_version") != PREDICTOR_VERSION
        or artifact.get("status") != "complete"
    ):
        raise ValueError("Predictor artifact identity/status mismatch")
    loaded: dict[str, dict[str, Any]] = {}
    for key in ("predictor", "training_report"):
        record = artifact["files"][key]
        path = artifact_root / str(record["file"])
        if path.parent != artifact_root or sha256_file(path) != record["sha256"]:
            raise ValueError("Predictor artifact path/hash mismatch")
        loaded[key] = _json(path)
    predictor = loaded["predictor"]
    report = loaded["training_report"]
    dataset_manifest = _json(input_root / "dataset_manifest.json")
    if report.get("dataset_manifest_sha256") != sha256_file(
        input_root / "dataset_manifest.json"
    ):
        raise ValueError("Predictor source dataset hash mismatch")
    if predictor.get("residual_calibration", {}).get("applied_to_base_prediction"):
        raise ValueError("base Ridge artifact must not contain residual calibration")

    replay: dict[str, Any] = {}
    for kind in BATCH_KINDS:
        model = predictor["models"][kind]
        names = tuple(model["active_features"])
        if names != ACTIVE_FEATURES[kind]:
            raise ValueError("Predictor feature schema mismatch")
        rows = _rows(input_root / dataset_manifest["files"][f"{kind}_test"]["file"])
        x, y = _arrays(rows, names)
        mean = np.asarray([model["standardization"][name]["mean"] for name in names])
        scale = np.asarray([model["standardization"][name]["scale"] for name in names])
        coefficients = np.asarray(
            [model["coefficients_for_standardized_features"][name] for name in names]
        )
        prediction = _predict(
            x, float(model["intercept_seconds"]), coefficients, mean, scale
        )
        observed = _metrics(y, prediction)
        recorded = report["evaluation"][kind]["held_out_test"]["overall"]
        for key, value in observed.items():
            recorded_value = recorded[key]
            if value is None and recorded_value is None:
                continue
            if isinstance(value, int):
                if value != recorded_value:
                    raise ValueError("Predictor evaluation replay mismatch")
            elif not math.isclose(value, float(recorded_value), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Predictor evaluation replay mismatch")
        replay[kind] = observed
    return {
        "valid": True,
        "predictor_version": PREDICTOR_VERSION,
        "held_out_test": replay,
    }


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "validate", "status"))
    parser.add_argument(
        "--input",
        default=str(repository / "results" / "dataset" / RIDGE_DATASET_ID),
    )
    parser.add_argument(
        "--output",
        default=str(repository / "predictors" / "qwen3_14b" / "ridge_three_scenario_v1"),
    )
    args = parser.parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    if args.command == "status":
        if not output_root.exists():
            print(json.dumps({"predictor_version": PREDICTOR_VERSION, "status": "absent"}))
            return 0
        print(json.dumps(_json(output_root / "artifact_manifest.json"), indent=2))
        return 0
    if args.command == "train":
        report = train_predictors(input_root=input_root, output_root=output_root)
        summary = {
            kind: {
                "alpha": report["evaluation"][kind]["cv"]["selected_alpha"],
                **report["evaluation"][kind]["held_out_test"]["overall"],
            }
            for kind in BATCH_KINDS
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        json.dumps(
            validate_artifact(input_root=input_root, artifact_root=output_root),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
