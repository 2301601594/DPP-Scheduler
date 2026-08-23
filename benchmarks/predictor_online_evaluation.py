"""Validate and summarize real-vLLM online Predictor shadow telemetry."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.predictor_profile import parse_iteration_durations
from dpp_scheduler.predictor import BATCH_KINDS, ONLINE_PREDICTOR_VERSION
from dpp_scheduler.targeted_profile import build_target_recipes
from dpp_scheduler.vllm_adapter import VLLM_OFFICIAL_ITERATION_TIMING


FORBIDDEN_KEYS = {
    "remaining_output_tokens",
    "expected_output_tokens",
    "output_length",
    "future_eos",
    "max_tokens",
}


def _p95(values: list[float]) -> float:
    if not values:
        raise ValueError("P95 requires values")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * (len(ordered) - 1))]


def _assert_no_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        overlap = FORBIDDEN_KEYS.intersection(value)
        if overlap:
            raise ValueError(f"forbidden Predictor telemetry fields: {sorted(overlap)}")
        for child in value.values():
            _assert_no_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden(child)


def load_evaluation_rows(
    path: Path, *, expected_run_id: str, recipe_seed: int, recipe_mode: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[tuple[int, str, str]] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"evaluation row is not an object: {line_number}")
            _assert_no_forbidden(row)
            if int(row.get("schema_version", 0)) != 2:
                raise ValueError("Predictor evaluation schema version mismatch")
            if row.get("run_id") != expected_run_id:
                raise ValueError("Predictor evaluation run_id mismatch")
            if row.get("predictor_version") != ONLINE_PREDICTOR_VERSION:
                raise ValueError("Predictor evaluation version mismatch")
            if int(row.get("recipe_seed", -1)) != recipe_seed:
                raise ValueError("Predictor evaluation recipe seed mismatch")
            if row.get("recipe_mode") != recipe_mode:
                raise ValueError("Predictor evaluation recipe mode mismatch")
            index = int(row.get("iteration_index", -1))
            identity = (
                int(row.get("frame_id", -1)),
                str(row.get("snapshot_hash", "")),
                str(row.get("plan_id", "")),
            )
            if index < 0 or identity in identities:
                raise ValueError("Predictor evaluation identity is invalid or duplicate")
            identities.add(identity)
            kind = row.get("batch_kind")
            if kind not in BATCH_KINDS:
                raise ValueError("Predictor evaluation batch kind is invalid")
            actual = float(row.get("actual_duration_seconds", 0.0))
            cpu = float(row.get("predictor_cpu_seconds", -1.0))
            if not math.isfinite(actual) or actual <= 0 or not math.isfinite(cpu) or cpu < 0:
                raise ValueError("Predictor evaluation duration is invalid")
            if row.get("timing_source") != VLLM_OFFICIAL_ITERATION_TIMING:
                raise ValueError("Predictor evaluation timing is not vLLM official")
            if row.get("timing_boundary") != (
                "after_execute_model_submission_through_model_result_and_sampling"
            ):
                raise ValueError("Predictor evaluation timing boundary mismatch")
            selected = row.get("selected_requests")
            if not isinstance(selected, list) or not selected:
                raise ValueError("Predictor evaluation selected requests are empty")
            request_ids: set[str] = set()
            phases: set[str] = set()
            for item in selected:
                request_id = str(item.get("request_id", ""))
                if not request_id or request_id in request_ids:
                    raise ValueError("Predictor evaluation request identity is invalid")
                request_ids.add(request_id)
                if item.get("phase") not in {"prefill", "decode"}:
                    raise ValueError("Predictor evaluation request phase is invalid")
                phases.add(str(item["phase"]))
                if int(item.get("current_context_tokens", -1)) < 0:
                    raise ValueError("Predictor evaluation context is invalid")
                scheduled = int(item.get("scheduled_tokens", 0))
                if scheduled <= 0 or (item["phase"] == "decode" and scheduled != 1):
                    raise ValueError("Predictor evaluation token count is invalid")
            derived_kind = (
                "mixed"
                if phases == {"prefill", "decode"}
                else "prefill_only"
                if phases == {"prefill"}
                else "decode_only"
                if phases == {"decode"}
                else None
            )
            if derived_kind != kind:
                raise ValueError("Predictor evaluation batch kind/phase mismatch")
            if bool(row.get("in_support")):
                for field in (
                    "base_duration_seconds",
                    "expected_duration_seconds",
                    "conservative_duration_seconds",
                    "residual_seconds",
                ):
                    value = float(row[field])
                    if not math.isfinite(value):
                        raise ValueError(f"supported Predictor field is invalid: {field}")
                if float(row["base_duration_seconds"]) <= 0:
                    raise ValueError("base Predictor duration is non-positive")
                if float(row["expected_duration_seconds"]) <= 0:
                    raise ValueError("expected Predictor duration is non-positive")
                if float(row["conservative_duration_seconds"]) < float(
                    row["expected_duration_seconds"]
                ):
                    raise ValueError("conservative duration is below expected duration")
                expected_residual = actual - float(row["base_duration_seconds"])
                if not math.isclose(
                    float(row["residual_seconds"]),
                    expected_residual,
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):
                    raise ValueError("Predictor residual definition mismatch")
                if not row.get("calibration_updated"):
                    raise ValueError("supported successful iteration was not calibrated")
            elif row.get("calibration_updated"):
                raise ValueError("out-of-support iteration updated calibration")
            rows.append(row)
    rows.sort(key=lambda row: int(row["iteration_index"]))
    if [int(row["iteration_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Predictor evaluation indices are not contiguous")

    recipes = build_target_recipes(recipe_seed, mode=recipe_mode)
    observed = [str(row.get("recipe_id")) for row in rows if row["sample_role"] == "target"]
    expected = [recipe.recipe_id for recipe in recipes]
    if observed != expected:
        raise ValueError("Predictor evaluation target recipe order is incomplete")
    return rows


def _prediction_metrics(
    rows: list[dict[str, Any]], prediction_field: str
) -> dict[str, float | int]:
    errors = [
        float(row[prediction_field]) - float(row["actual_duration_seconds"])
        for row in rows
    ]
    absolute = [abs(value) for value in errors]
    return {
        "rows": len(rows),
        "mae_seconds": sum(absolute) / len(absolute),
        "rmse_seconds": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "mean_error_seconds": sum(errors) / len(errors),
        "p95_absolute_error_seconds": _p95(absolute),
        "underprediction_rate": sum(value < 0 for value in errors) / len(errors),
    }


def _effectiveness_result(
    criteria: dict[str, bool], *, timing_compatible: bool
) -> dict[str, Any]:
    candidate = all(criteria.values())
    return {
        "criteria": criteria,
        "candidate_effective_without_timing_guard": candidate,
        "conclusion_available": timing_compatible,
        "effective": candidate if timing_compatible else None,
        "unavailable_reason": None if timing_compatible else "timing_incompatible",
    }


def analyze_evaluation(
    *,
    telemetry_path: Path,
    startup_log_path: Path,
    output_path: Path,
    expected_run_id: str,
    recipe_seed: int,
    recipe_mode: str,
) -> dict[str, Any]:
    rows = load_evaluation_rows(
        telemetry_path,
        expected_run_id=expected_run_id,
        recipe_seed=recipe_seed,
        recipe_mode=recipe_mode,
    )
    official = parse_iteration_durations(startup_log_path)
    if set(official) != set(range(len(rows))):
        raise ValueError("Predictor telemetry and official iteration timing mismatch")
    timing_differences = [
        abs(float(row["actual_duration_seconds"]) - official[index])
        for index, row in enumerate(rows)
    ]
    official_values = list(official.values())
    median_difference = statistics.median(timing_differences)
    p95_difference = _p95(timing_differences)
    median_limit = max(0.002, 0.02 * statistics.median(official_values))
    p95_limit = max(0.005, 0.05 * _p95(official_values))
    timing_compatible = median_difference <= median_limit and p95_difference <= p95_limit

    with output_path.open("x", encoding="utf-8") as stream:
        for index, source in enumerate(rows):
            row = dict(source)
            row["official_duration_seconds"] = official[index]
            row["timing_absolute_difference_seconds"] = timing_differences[index]
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    by_kind: dict[str, Any] = {}
    for kind in BATCH_KINDS:
        kind_rows = [row for row in rows if row["batch_kind"] == kind]
        supported = [row for row in kind_rows if row["in_support"]]
        active = [
            row for row in supported if row["calibration_source"] == "online_window"
        ]
        item: dict[str, Any] = {
            "rows": len(kind_rows),
            "in_support_rows": len(supported),
            "out_of_support_rows": len(kind_rows) - len(supported),
            "out_of_support_rate": (
                (len(kind_rows) - len(supported)) / len(kind_rows) if kind_rows else None
            ),
            "online_window_rows": len(active),
            "effectiveness": None,
        }
        if active:
            base = _prediction_metrics(active, "base_duration_seconds")
            expected = _prediction_metrics(active, "expected_duration_seconds")
            coverage = sum(
                float(row["actual_duration_seconds"])
                <= float(row["conservative_duration_seconds"])
                for row in active
            ) / len(active)
            criteria = {
                "mae_improved": expected["mae_seconds"] < base["mae_seconds"],
                "absolute_bias_not_worse": abs(expected["mean_error_seconds"])
                <= abs(base["mean_error_seconds"]),
                "conservative_coverage_at_least_0p95": coverage >= 0.95,
            }
            item.update(
                {
                    "base": base,
                    "calibrated_expected": expected,
                    "conservative_coverage": coverage,
                    "effectiveness": _effectiveness_result(
                        criteria, timing_compatible=timing_compatible
                    ),
                }
            )
        by_kind[kind] = item
    cpu_values = [float(row["predictor_cpu_seconds"]) for row in rows]
    return {
        "schema_version": 2,
        "valid": True,
        "run_id": expected_run_id,
        "predictor_version": ONLINE_PREDICTOR_VERSION,
        "iteration_count": len(rows),
        "sample_roles": dict(sorted(Counter(row["sample_role"] for row in rows).items())),
        "timing": {
            "compatible": timing_compatible,
            "median_absolute_difference_seconds": median_difference,
            "median_limit_seconds": median_limit,
            "p95_absolute_difference_seconds": p95_difference,
            "p95_limit_seconds": p95_limit,
        },
        "predictor_cpu_seconds": {
            "p50": statistics.median(cpu_values),
            "p95": _p95(cpu_values),
            "max": max(cpu_values),
        },
        "by_batch_kind": by_kind,
        "effectiveness_conclusion_available": timing_compatible,
        "overall_effective": (
            all(
                by_kind[kind]["effectiveness"] is not None
                and by_kind[kind]["effectiveness"]["effective"]
                for kind in BATCH_KINDS
            )
            if timing_compatible
            else None
        ),
        "effectiveness_is_single_seed_diagnostic": True,
    }
