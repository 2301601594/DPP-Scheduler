#!/usr/bin/env python3
"""Compare segmented and legacy Mixed predictions on one untouched validation run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Any

from benchmarks.build_predictor_dataset import extract_features
from benchmarks.predictor_online_evaluation import load_evaluation_rows
from benchmarks.run_segmented_predictor_validation_campaign import (
    RECIPE_SEED,
    RUN_ID,
    SOURCE_SEED,
)
from benchmarks.run_stock_natural_eos import _atomic_json
from dpp_scheduler.predictor import MIXED_DECODE_SEGMENTS


def _higher(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(quantile * (len(ordered) - 1))]


def _legacy_base(
    model: dict[str, Any], features: dict[str, Any]
) -> tuple[float, bool, float]:
    names = tuple(model["active_features"])
    values = [float(features[name]) for name in names]
    coefficients = model["coefficients_for_standardized_features"]
    standardization = model["standardization"]
    support = model["support_domain_train_marginal_box"]
    clipped = [
        min(float(support[name]["max"]), max(float(support[name]["min"]), value))
        for name, value in zip(names, values)
    ]
    in_support = all(value == bounded for value, bounded in zip(values, clipped))
    boundary = float(model["intercept_seconds"]) + sum(
        float(coefficients[name])
        * (bounded - float(standardization[name]["mean"]))
        / float(standardization[name]["scale"])
        for name, bounded in zip(names, clipped)
    )
    high = sum(
        max(
            0.0,
            float(coefficients[name]) / float(standardization[name]["scale"]),
        )
        * max(0.0, value - float(support[name]["max"]))
        for name, value in zip(names, values)
    )
    distance = max(
        abs(value - bounded) / float(standardization[name]["scale"])
        for name, value, bounded in zip(names, values, clipped)
    )
    return boundary + high, in_support, distance


def _metrics(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("metrics require rows")
    errors = [row["prediction"] - row["actual"] for row in rows]
    absolute = [abs(value) for value in errors]
    relative = [absolute[index] / row["actual"] for index, row in enumerate(rows)]
    return {
        "rows": len(rows),
        "mae_seconds": sum(absolute) / len(rows),
        "rmse_seconds": math.sqrt(sum(value * value for value in errors) / len(rows)),
        "mean_error_seconds": sum(errors) / len(rows),
        "mape": sum(relative) / len(rows),
        "median_absolute_percentage_error": statistics.median(relative),
        "p95_absolute_percentage_error": _higher(relative, 0.95),
        "underprediction_rate": sum(value < 0 for value in errors) / len(rows),
        "conservative_coverage": sum(
            row["actual"] <= row["conservative"] for row in rows
        )
        / len(rows),
    }


def _segment_id(decode_count: int) -> str:
    for segment_id, minimum, maximum in MIXED_DECODE_SEGMENTS:
        if minimum <= decode_count <= maximum:
            return segment_id
    raise ValueError(f"Mixed Decode count is outside fixed segments: {decode_count}")


def analyze(*, run_root: Path, legacy_artifact: Path, output: Path) -> dict[str, Any]:
    rows = load_evaluation_rows(
        run_root / "predictor_evaluation.jsonl",
        expected_run_id=RUN_ID,
        recipe_seed=RECIPE_SEED,
        recipe_mode="formal",
    )
    legacy = json.loads((legacy_artifact / "predictor.json").read_text(encoding="utf-8"))
    legacy_model = legacy["models"]["mixed"]
    setting = legacy["residual_calibration"]["by_batch_kind"]["mixed"]
    history: deque[float] = deque(maxlen=int(setting["window_size"]))
    comparisons: list[dict[str, Any]] = []
    candidate_mixed_rows = 0
    legacy_mixed_in_support = 0
    for row in rows:
        if row["batch_kind"] != "mixed":
            continue
        candidate_mixed_rows += 1
        features = extract_features(row["selected_requests"])
        base, legacy_in_support, distance = _legacy_base(
            legacy_model, features
        )
        if len(history) >= int(setting["minimum_samples"]):
            residual_mean = sum(history) / len(history)
            centered_p95 = _higher(
                [value - residual_mean for value in history], 0.95
            )
        else:
            residual_mean = float(setting["cold_start_mean_seconds"])
            centered_p95 = float(setting["cold_start_centered_p95_seconds"])
        actual = float(row["actual_duration_seconds"])
        legacy_expected = base + residual_mean
        legacy_conservative = legacy_expected + centered_p95 + 0.0 * distance
        if legacy_in_support:
            legacy_mixed_in_support += 1
            history.append(actual - base)
        if bool(row["in_support"]) and legacy_in_support:
            comparisons.append(
                {
                    "iteration_index": int(row["iteration_index"]),
                    "segment_id": _segment_id(int(features["x_4"])),
                    "actual": actual,
                    "candidate": {
                        "prediction": float(row["expected_duration_seconds"]),
                        "conservative": float(row["conservative_duration_seconds"]),
                    },
                    "legacy": {
                        "prediction": legacy_expected,
                        "conservative": legacy_conservative,
                    },
                }
            )
    if not comparisons:
        raise ValueError("independent validation has no common-support Mixed rows")

    def metrics_for(label: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        return _metrics(
            [
                {
                    "actual": row["actual"],
                    "prediction": row[label]["prediction"],
                    "conservative": row[label]["conservative"],
                }
                for row in subset
            ]
        )

    candidate = metrics_for("candidate", comparisons)
    legacy_metrics = metrics_for("legacy", comparisons)
    by_segment: dict[str, Any] = {}
    for segment_id, _, _ in MIXED_DECODE_SEGMENTS:
        subset = [row for row in comparisons if row["segment_id"] == segment_id]
        by_segment[segment_id] = (
            {
                "candidate": metrics_for("candidate", subset),
                "legacy": metrics_for("legacy", subset),
            }
            if subset
            else {"candidate": None, "legacy": None, "rows": 0}
        )
    criteria = {
        "all_three_segments_observed": all(
            any(row["segment_id"] == segment_id for row in comparisons)
            for segment_id, _, _ in MIXED_DECODE_SEGMENTS
        ),
        "candidate_mae_lower_than_legacy": (
            candidate["mae_seconds"] < legacy_metrics["mae_seconds"]
        ),
        "candidate_mape_lower_than_legacy": candidate["mape"] < legacy_metrics["mape"],
        "candidate_p95_ape_lower_than_legacy": (
            candidate["p95_absolute_percentage_error"]
            < legacy_metrics["p95_absolute_percentage_error"]
        ),
        "candidate_conservative_coverage_at_least_0p95": (
            candidate["conservative_coverage"] >= 0.95
        ),
    }
    result = {
        "schema_version": 1,
        "validation_role": "fresh_independent_data_not_used_for_design_or_training",
        "run_id": RUN_ID,
        "source_seed": SOURCE_SEED,
        "recipe_seed": RECIPE_SEED,
        "mixed_rows": candidate_mixed_rows,
        "common_support_rows": len(comparisons),
        "candidate_out_of_support_rows": candidate_mixed_rows
        - sum(bool(row["in_support"]) for row in rows if row["batch_kind"] == "mixed"),
        "legacy_out_of_support_rows": candidate_mixed_rows - legacy_mixed_in_support,
        "candidate": candidate,
        "legacy_global_mixed": legacy_metrics,
        "relative_change_candidate_vs_legacy": {
            "mae": candidate["mae_seconds"] / legacy_metrics["mae_seconds"] - 1.0,
            "rmse": candidate["rmse_seconds"] / legacy_metrics["rmse_seconds"] - 1.0,
            "mape": candidate["mape"] / legacy_metrics["mape"] - 1.0,
            "p95_absolute_percentage_error": (
                candidate["p95_absolute_percentage_error"]
                / legacy_metrics["p95_absolute_percentage_error"]
                - 1.0
            ),
        },
        "by_decode_segment": by_segment,
        "predeclared_acceptance_criteria": criteria,
        "accepted": all(criteria.values()),
        "single_seed_diagnostic": True,
    }
    if output.exists():
        raise FileExistsError(f"append-only validation analysis exists: {output}")
    _atomic_json(output, result)
    return result


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--legacy-artifact",
        default=str(
            repository / "predictors" / "qwen3_14b" / "ridge_three_scenario_online_v1"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(
        run_root=Path(args.run_root),
        legacy_artifact=Path(args.legacy_artifact),
        output=Path(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
