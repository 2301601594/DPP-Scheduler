#!/usr/bin/env python3
"""Replay existing Mixed telemetry through segmented v2 and cross-feature v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import deque
from pathlib import Path
from typing import Any

from benchmarks.analyze_segmented_predictor_validation import _legacy_base
from benchmarks.build_cross_feature_mixed_predictor import add_mixed_cross_features
from benchmarks.build_predictor_dataset import extract_features
from benchmarks.predictor_online_evaluation import load_evaluation_rows
from benchmarks.run_segmented_predictor_validation_campaign import (
    RECIPE_SEED,
    RUN_ID,
)
from benchmarks.run_stock_natural_eos import _atomic_json
from dpp_scheduler.predictor import MIXED_DECODE_SEGMENTS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_artifact(root: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    record = manifest["files"]["predictor"]
    predictor_path = (root / record["file"]).resolve()
    if predictor_path.parent != root.resolve() or _sha256(predictor_path) != record["sha256"]:
        raise ValueError(f"Predictor artifact path/hash mismatch: {root}")
    payload = json.loads(predictor_path.read_text(encoding="utf-8"))
    if payload["predictor_version"] != manifest["predictor_version"]:
        raise ValueError(f"Predictor artifact version mismatch: {root}")
    return payload, _sha256(root / "artifact_manifest.json")


def _higher(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(quantile * (len(ordered) - 1))]


def _segment_id(decode_count: int) -> str:
    for segment_id, minimum, maximum in MIXED_DECODE_SEGMENTS:
        if minimum <= decode_count <= maximum:
            return segment_id
    raise ValueError(f"Mixed Decode count is outside fixed segments: {decode_count}")


def _segment_model(payload: dict[str, Any], decode_count: int) -> dict[str, Any]:
    mixed = payload["models"]["mixed"]
    if mixed.get("dispatch") != "decode_count_segments":
        raise ValueError("comparison requires segmented Mixed artifacts")
    for item in mixed["segments"]:
        if (
            int(item["minimum_decode_count_inclusive"])
            <= decode_count
            <= int(item["maximum_decode_count_inclusive"])
        ):
            return item["model"]
    raise ValueError(f"no Mixed segment covers Decode count {decode_count}")


def _metrics(rows: list[dict[str, float]], prefix: str) -> dict[str, float | int]:
    if not rows:
        raise ValueError("metrics require at least one row")
    actual = [row["actual"] for row in rows]
    prediction = [row[f"{prefix}_expected"] for row in rows]
    conservative = [row[f"{prefix}_conservative"] for row in rows]
    base = [row[f"{prefix}_base"] for row in rows]
    errors = [prediction[index] - value for index, value in enumerate(actual)]
    base_errors = [base[index] - value for index, value in enumerate(actual)]
    absolute = [abs(value) for value in errors]
    base_absolute = [abs(value) for value in base_errors]
    relative = [absolute[index] / value for index, value in enumerate(actual)]
    return {
        "rows": len(rows),
        "base_mae_seconds": sum(base_absolute) / len(rows),
        "mae_seconds": sum(absolute) / len(rows),
        "rmse_seconds": math.sqrt(sum(value * value for value in errors) / len(rows)),
        "mean_error_seconds": sum(errors) / len(rows),
        "mape": sum(relative) / len(rows),
        "median_absolute_percentage_error": statistics.median(relative),
        "p95_absolute_percentage_error": _higher(relative, 0.95),
        "underprediction_rate": sum(value < 0 for value in errors) / len(rows),
        "conservative_coverage": sum(
            actual[index] <= conservative[index] for index in range(len(rows))
        )
        / len(rows),
    }


def _replay(
    *, rows: list[dict[str, Any]], payload: dict[str, Any], prefix: str
) -> tuple[list[dict[str, Any]], int]:
    setting = payload["residual_calibration"]["by_batch_kind"]["mixed"]
    history: deque[float] = deque(maxlen=int(setting["window_size"]))
    output: list[dict[str, Any]] = []
    out_of_support = 0
    for row in rows:
        if row["batch_kind"] != "mixed":
            continue
        features = extract_features(row["selected_requests"])
        enriched = add_mixed_cross_features({"features": features})["features"]
        decode_count = int(features["x_4"])
        model = _segment_model(payload, decode_count)
        base, in_support, distance = _legacy_base(model, enriched)
        if len(history) >= int(setting["minimum_samples"]):
            residual_mean = sum(history) / len(history)
            centered_p95 = _higher(
                [value - residual_mean for value in history], 0.95
            )
        else:
            residual_mean = float(setting["cold_start_mean_seconds"])
            centered_p95 = float(setting["cold_start_centered_p95_seconds"])
        actual = float(row["actual_duration_seconds"])
        expected = base + residual_mean
        conservative = expected + centered_p95
        if in_support:
            history.append(actual - base)
        else:
            out_of_support += 1
        output.append(
            {
                "iteration_index": int(row["iteration_index"]),
                "segment_id": _segment_id(decode_count),
                "decode_count": decode_count,
                "prefill_tokens": int(features["x_6"]),
                "actual": actual,
                f"{prefix}_base": base,
                f"{prefix}_expected": expected,
                f"{prefix}_conservative": conservative,
                f"{prefix}_in_support": in_support,
                f"{prefix}_ood_distance": distance,
            }
        )
    return output, out_of_support


def compare(
    *, run_root: Path, baseline_root: Path, candidate_root: Path, output: Path
) -> dict[str, Any]:
    rows = load_evaluation_rows(
        run_root / "predictor_evaluation.jsonl",
        expected_run_id=RUN_ID,
        recipe_seed=RECIPE_SEED,
        recipe_mode="formal",
    )
    baseline, baseline_manifest_hash = _load_artifact(baseline_root)
    candidate, candidate_manifest_hash = _load_artifact(candidate_root)
    baseline_rows, baseline_ood = _replay(
        rows=rows, payload=baseline, prefix="baseline"
    )
    candidate_rows, candidate_ood = _replay(
        rows=rows, payload=candidate, prefix="candidate"
    )
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("replay row counts differ")
    merged: list[dict[str, Any]] = []
    for baseline_row, candidate_row in zip(baseline_rows, candidate_rows):
        if baseline_row["iteration_index"] != candidate_row["iteration_index"]:
            raise ValueError("replay iteration alignment mismatch")
        merged.append({**baseline_row, **candidate_row})
    common = [
        row
        for row in merged
        if row["baseline_in_support"] and row["candidate_in_support"]
    ]
    if not common:
        raise ValueError("replay has no common-support Mixed rows")
    baseline_metrics = _metrics(common, "baseline")
    candidate_metrics = _metrics(common, "candidate")
    by_segment: dict[str, Any] = {}
    for segment_id, _, _ in MIXED_DECODE_SEGMENTS:
        subset = [row for row in common if row["segment_id"] == segment_id]
        by_segment[segment_id] = {
            "baseline": _metrics(subset, "baseline") if subset else None,
            "candidate": _metrics(subset, "candidate") if subset else None,
        }
    high_decode_48_low_prefill = [
        row
        for row in common
        if row["segment_id"] == "decode_17_64"
        and row["decode_count"] == 48
        and row["prefill_tokens"] <= 64
    ]
    result = {
        "schema_version": 1,
        "validation_role": (
            "post_hoc_development_replay_existing_n200_data_used_to_motivate_features"
        ),
        "no_new_remote_benchmark": True,
        "run_id": RUN_ID,
        "mixed_rows": len(merged),
        "common_support_rows": len(common),
        "support": {
            "baseline_out_of_support_rows": baseline_ood,
            "candidate_out_of_support_rows": candidate_ood,
        },
        "baseline": {
            "predictor_version": baseline["predictor_version"],
            "artifact_manifest_sha256": baseline_manifest_hash,
            "metrics_on_common_support": baseline_metrics,
        },
        "candidate": {
            "predictor_version": candidate["predictor_version"],
            "artifact_manifest_sha256": candidate_manifest_hash,
            "metrics_on_common_support": candidate_metrics,
        },
        "relative_change_candidate_vs_baseline": {
            name: candidate_metrics[name] / baseline_metrics[name] - 1.0
            for name in (
                "base_mae_seconds",
                "mae_seconds",
                "rmse_seconds",
                "mape",
                "p95_absolute_percentage_error",
            )
        },
        "by_decode_segment": by_segment,
        "diagnostic_decode48_prefill_at_most64": {
            "rows": len(high_decode_48_low_prefill),
            "baseline": (
                _metrics(high_decode_48_low_prefill, "baseline")
                if high_decode_48_low_prefill
                else None
            ),
            "candidate": (
                _metrics(high_decode_48_low_prefill, "candidate")
                if high_decode_48_low_prefill
                else None
            ),
        },
        "interpretation_constraint": (
            "A fresh unused seed is required before adoption or an independent claim."
        ),
    }
    if output.exists():
        raise FileExistsError(f"append-only comparison exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    return result


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--baseline",
        default=str(
            repository
            / "predictors"
            / "qwen3_14b"
            / "ridge_mixed_decode_three_segment_online_v2"
        ),
    )
    parser.add_argument(
        "--candidate",
        default=str(
            repository
            / "predictors"
            / "qwen3_14b"
            / "ridge_mixed_decode_three_segment_cross_online_v3"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(
        run_root=Path(args.run_root),
        baseline_root=Path(args.baseline),
        candidate_root=Path(args.candidate),
        output=Path(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
