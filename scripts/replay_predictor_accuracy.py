#!/usr/bin/env python3
"""Replay executed Predictor estimates against actual vLLM batch durations.

The replay joins an immutable Selector Diagnosis JSONL record to the Scheduler
feedback log by frame ID.  It never recomputes a prediction: expected,
conservative, and effective durations are the exact values used by the live
decision.  Actual durations come only from the aligned vLLM execution timing
recorded after the selected batch executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


BATCH_KINDS = ("prefill_only", "decode_only", "mixed")
DEFAULT_TIMING_SOURCE = "vllm_aligned_monotonic"
SELECTOR_DIAGNOSIS_SCHEMA_VERSION = 3
FEEDBACK_RE = re.compile(
    r"ModularDPPScheduler feedback frame=(?P<frame>\d+) "
    r"scheduled_tokens=(?P<scheduled>\d+) "
    r"actual_duration_seconds=(?P<duration>[0-9.eE+-]+) "
    r"timing_source=(?P<timing_source>\S+) "
    r"iteration_index=(?P<iteration_index>\S+) "
    r"actual_prefill=(?P<prefill>\d+) "
    r"actual_decode=(?P<decode>\d+)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[max(0, index)]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def prediction_metrics(
    rows: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    predicted = [float(row[field]) for row in rows]
    actual = [float(row["actual_duration_seconds"]) for row in rows]
    errors = [estimate - observed for estimate, observed in zip(predicted, actual)]
    absolute = [abs(value) for value in errors]
    relative = [value / observed for value, observed in zip(absolute, actual)]
    return {
        "rows": len(rows),
        "predicted_seconds": distribution(predicted),
        "actual_seconds": distribution(actual),
        "error_seconds_prediction_minus_actual": distribution(errors),
        "absolute_error_seconds": distribution(absolute),
        "relative_absolute_error": distribution(relative),
        "mae_seconds": statistics.fmean(absolute),
        "rmse_seconds": math.sqrt(statistics.fmean([value * value for value in errors])),
        "mean_error_seconds": statistics.fmean(errors),
        "underprediction_rate": sum(value < 0 for value in errors) / len(rows),
        "overprediction_rate": sum(value > 0 for value in errors) / len(rows),
        "within_5ms_rate": sum(value <= 0.005 for value in absolute) / len(rows),
        "within_10ms_rate": sum(value <= 0.010 for value in absolute) / len(rows),
        "within_10pct_rate": sum(value <= 0.10 for value in relative) / len(rows),
    }


def metric_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = prediction_metrics(rows, "expected_duration_seconds")
    conservative = prediction_metrics(rows, "conservative_duration_seconds")
    effective = prediction_metrics(rows, "effective_duration_seconds")
    misses = [
        float(row["actual_duration_seconds"])
        - float(row["conservative_duration_seconds"])
        for row in rows
        if float(row["actual_duration_seconds"])
        > float(row["conservative_duration_seconds"])
    ]
    conservative.update(
        {
            "coverage_rate": 1.0 - len(misses) / len(rows),
            "miss_count": len(misses),
            "miss_shortfall_seconds": distribution(misses),
        }
    )
    return {
        "rows": len(rows),
        "expected": expected,
        "conservative": conservative,
        "effective": effective,
    }


def load_diagnosis(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frames: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or int(row.get("schema_version", 0))
                != SELECTOR_DIAGNOSIS_SCHEMA_VERSION
            ):
                raise ValueError(f"invalid diagnosis schema at line {line_number}")
            frame = int(row.get("frame_id", -1))
            if frame < 0 or frame in frames:
                raise ValueError(f"invalid or duplicate diagnosis frame: {frame}")
            frames.add(frame)
            rows.append(row)
    if not rows:
        raise ValueError("Selector Diagnosis is empty")
    if [int(row["frame_id"]) for row in rows] != sorted(frames):
        raise ValueError("Selector Diagnosis frames are not monotonically ordered")
    return rows


def load_feedback(path: Path, expected_timing_source: str) -> dict[int, dict[str, Any]]:
    feedback: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = FEEDBACK_RE.search(line)
            if match is None:
                continue
            frame = int(match.group("frame"))
            if frame in feedback:
                raise ValueError(f"duplicate actual feedback frame: {frame}")
            duration = float(match.group("duration"))
            timing_source = match.group("timing_source")
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"invalid actual duration at frame {frame}")
            if timing_source != expected_timing_source:
                raise ValueError(
                    f"unexpected timing source at frame {frame}: {timing_source}"
                )
            feedback[frame] = {
                "frame_id": frame,
                "scheduled_tokens": int(match.group("scheduled")),
                "actual_duration_seconds": duration,
                "timing_source": timing_source,
                "iteration_index": match.group("iteration_index"),
                "actual_prefill_tokens": int(match.group("prefill")),
                "actual_decode_tokens": int(match.group("decode")),
            }
    if not feedback:
        raise ValueError("Scheduler feedback log is empty")
    return feedback


def classify_batch(prefill_tokens: int, decode_tokens: int) -> str:
    if prefill_tokens > 0 and decode_tokens > 0:
        return "mixed"
    if prefill_tokens > 0:
        return "prefill_only"
    if decode_tokens > 0:
        return "decode_only"
    raise ValueError("executed batch has no scheduled tokens")


def validate_sources(
    manifest_path: Path, diagnosis_path: Path, startup_log_path: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("source run manifest is not complete")
    if manifest.get("selector_diagnosis_valid") is not True:
        raise ValueError("source Selector Diagnosis is not valid")
    diagnosis_sha = sha256_file(diagnosis_path)
    startup_sha = sha256_file(startup_log_path)
    if manifest.get("selector_diagnosis_sha256") != diagnosis_sha:
        raise ValueError("Selector Diagnosis hash does not match run manifest")
    if manifest.get("startup_log_sha256") != startup_sha:
        raise ValueError("startup log hash does not match run manifest")
    replay = manifest.get("selector_diagnosis_replay")
    if not isinstance(replay, dict) or any(
        int(value) != 0
        for key, value in replay.items()
        if key.endswith("_mismatch")
    ):
        raise ValueError("source Selector Diagnosis replay has mismatches")
    return {
        "run_id": str(manifest.get("run_id")),
        "run_manifest_sha256": sha256_file(manifest_path),
        "selector_diagnosis_sha256": diagnosis_sha,
        "startup_log_sha256": startup_sha,
        "selector_diagnosis_replay": replay,
    }


def build_replay_rows(
    diagnosis: list[dict[str, Any]],
    feedback: dict[int, dict[str, Any]],
    *,
    minimum_samples: int,
    window_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if minimum_samples <= 0 or window_size < minimum_samples:
        raise ValueError("invalid calibration replay window settings")
    sample_counts = {kind: 0 for kind in BATCH_KINDS}
    joined: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    accounted_feedback_frames: set[int] = set()
    for record in diagnosis:
        frame = int(record["frame_id"])
        decision = record["decision"]
        executed_plan_id = str(decision["executed_plan_id"])
        candidates = {
            str(candidate["plan_id"]): candidate
            for candidate in record.get("candidates", [])
        }
        candidate = candidates.get(executed_plan_id)
        actual = feedback.get(frame)
        if candidate is None:
            if actual is not None and (
                int(actual["scheduled_tokens"]) != 0
                or int(actual["actual_prefill_tokens"]) != 0
                or int(actual["actual_decode_tokens"]) != 0
            ):
                raise ValueError(
                    "non-empty feedback exists without an executed diagnosis "
                    f"candidate: {frame}"
                )
            if actual is not None:
                accounted_feedback_frames.add(frame)
            omitted.append(
                {
                    "frame_id": frame,
                    "executed_plan_id": executed_plan_id,
                    "controller_reason": decision.get("controller_reason"),
                    "actual_feedback_present": actual is not None,
                    "actual_scheduled_tokens": (
                        int(actual["scheduled_tokens"]) if actual is not None else None
                    ),
                    "actual_duration_seconds": (
                        float(actual["actual_duration_seconds"])
                        if actual is not None
                        else None
                    ),
                }
            )
            continue
        if actual is None:
            raise ValueError(f"missing actual feedback for executed frame: {frame}")

        plan = candidate["plan"]
        prefill_tokens = int(plan["total_prefill_tokens"])
        decode_tokens = int(plan["total_decode_tokens"])
        total_tokens = prefill_tokens + decode_tokens
        if total_tokens != int(actual["scheduled_tokens"]):
            raise ValueError(f"scheduled-token mismatch at frame {frame}")
        if prefill_tokens != int(actual["actual_prefill_tokens"]):
            raise ValueError(f"actual Prefill-token mismatch at frame {frame}")
        if decode_tokens != int(actual["actual_decode_tokens"]):
            raise ValueError(f"actual Decode-token mismatch at frame {frame}")

        duration = candidate["duration"]
        expected = float(duration["expected"])
        conservative = float(duration["conservative"])
        effective = float(duration["effective"])
        actual_duration = float(actual["actual_duration_seconds"])
        if not all(
            math.isfinite(value) and value > 0
            for value in (expected, conservative, effective, actual_duration)
        ):
            raise ValueError(f"non-positive or non-finite duration at frame {frame}")
        if conservative < expected:
            raise ValueError(f"conservative duration below expected at frame {frame}")

        kind = classify_batch(prefill_tokens, decode_tokens)
        sample_count_before = sample_counts[kind]
        calibration_source = (
            "online_window"
            if sample_count_before >= minimum_samples
            else "offline_oof_cold_start"
        )
        row = {
            "frame_id": frame,
            "plan_id": executed_plan_id,
            "template_id": str(candidate["template_id"]),
            "batch_kind": kind,
            "prediction_mode": str(duration["prediction_mode"]),
            "in_support": bool(duration["in_support"]),
            "calibration_source_replayed": calibration_source,
            "calibration_sample_count_before": sample_count_before,
            "total_prefill_tokens": prefill_tokens,
            "total_decode_tokens": decode_tokens,
            "total_scheduled_tokens": total_tokens,
            "expected_duration_seconds": expected,
            "conservative_duration_seconds": conservative,
            "effective_duration_seconds": effective,
            "actual_duration_seconds": actual_duration,
            "expected_error_seconds_prediction_minus_actual": expected
            - actual_duration,
            "conservative_error_seconds_prediction_minus_actual": conservative
            - actual_duration,
            "effective_error_seconds_prediction_minus_actual": effective
            - actual_duration,
            "expected_absolute_error_seconds": abs(expected - actual_duration),
            "conservative_absolute_error_seconds": abs(
                conservative - actual_duration
            ),
            "effective_absolute_error_seconds": abs(effective - actual_duration),
            "conservative_covers_actual": conservative >= actual_duration,
            "timing_source": str(actual["timing_source"]),
            "selector_reason": str(decision["selector_reason"]),
            "controller_reason": str(decision["controller_reason"]),
        }
        joined.append(row)
        accounted_feedback_frames.add(frame)
        if row["in_support"]:
            sample_counts[kind] = min(window_size, sample_count_before + 1)

    extra_feedback = sorted(set(feedback) - accounted_feedback_frames)
    if extra_feedback:
        raise ValueError(f"actual feedback has no joined diagnosis: {extra_feedback[:10]}")
    return joined, omitted, sample_counts


def grouped_metrics(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    values = sorted({str(row[key]) for row in rows})
    return {
        value: metric_bundle([row for row in rows if str(row[key]) == value])
        for value in values
    }


def analyze_replay(
    *,
    diagnosis_path: Path,
    startup_log_path: Path,
    run_manifest_path: Path,
    minimum_samples: int,
    window_size: int,
    expected_timing_source: str = DEFAULT_TIMING_SOURCE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = validate_sources(run_manifest_path, diagnosis_path, startup_log_path)
    diagnosis = load_diagnosis(diagnosis_path)
    feedback = load_feedback(startup_log_path, expected_timing_source)
    rows, omitted, final_sample_counts = build_replay_rows(
        diagnosis,
        feedback,
        minimum_samples=minimum_samples,
        window_size=window_size,
    )
    if not rows:
        raise ValueError("Predictor accuracy replay has no joined frames")
    by_kind_and_source = {
        kind: {
            source_name: metric_bundle(
                [
                    row
                    for row in rows
                    if row["batch_kind"] == kind
                    and row["calibration_source_replayed"] == source_name
                ]
            )
            for source_name in sorted(
                {
                    str(row["calibration_source_replayed"])
                    for row in rows
                    if row["batch_kind"] == kind
                }
            )
        }
        for kind in BATCH_KINDS
    }
    summary = {
        "schema_version": 1,
        "analysis": "executed_predictor_accuracy_replay",
        "valid": True,
        "source": source,
        "alignment": {
            "diagnosis_frames": len(diagnosis),
            "actual_feedback_frames": len(feedback),
            "joined_executed_frames": len(rows),
            "omitted_nonexecuting_frames": omitted,
            "frame_id_min": min(int(row["frame_id"]) for row in rows),
            "frame_id_max": max(int(row["frame_id"]) for row in rows),
            "timing_source": expected_timing_source,
            "plan_and_token_alignment_mismatches": 0,
        },
        "calibration_replay": {
            "minimum_samples": minimum_samples,
            "window_size": window_size,
            "source_rule": "prior actual-only in-support executions per batch kind",
            "final_sample_counts": final_sample_counts,
            "rows_by_source": dict(
                sorted(Counter(row["calibration_source_replayed"] for row in rows).items())
            ),
        },
        "row_counts": {
            "by_batch_kind": dict(sorted(Counter(row["batch_kind"] for row in rows).items())),
            "by_template": dict(sorted(Counter(row["template_id"] for row in rows).items())),
            "by_prediction_mode": dict(
                sorted(Counter(row["prediction_mode"] for row in rows).items())
            ),
            "in_support": sum(bool(row["in_support"]) for row in rows),
            "out_of_support": sum(not bool(row["in_support"]) for row in rows),
        },
        "overall": metric_bundle(rows),
        "by_batch_kind": grouped_metrics(rows, "batch_kind"),
        "by_calibration_source": grouped_metrics(
            rows, "calibration_source_replayed"
        ),
        "by_batch_kind_and_calibration_source": by_kind_and_source,
        "by_template": grouped_metrics(rows, "template_id"),
        "limitations": [
            "Diagnosis stores live expected/conservative/effective durations but not the Ridge base duration, so this replay cannot compare calibrated expected against base.",
            "Calibration source is deterministically replayed from prior actual-only in-support executions and the frozen minimum/window sizes.",
            "This development run is single-trace diagnostic evidence, not a formal Predictor validation campaign.",
        ],
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selector_diagnosis", type=Path)
    parser.add_argument("startup_log", type=Path)
    parser.add_argument("run_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-samples", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument(
        "--expected-timing-source", default=DEFAULT_TIMING_SOURCE
    )
    args = parser.parse_args()

    rows, summary = analyze_replay(
        diagnosis_path=args.selector_diagnosis,
        startup_log_path=args.startup_log,
        run_manifest_path=args.run_manifest,
        minimum_samples=args.minimum_samples,
        window_size=args.window_size,
        expected_timing_source=args.expected_timing_source,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = args.output_dir / "per_frame_accuracy.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "summary.json"
    summary["artifacts"] = {
        "per_frame_accuracy_csv": str(csv_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_manifest = {
        "schema_version": 1,
        "analysis": summary["analysis"],
        "source": summary["source"],
        "files": {
            "per_frame_accuracy": {
                "file": csv_path.name,
                "sha256": sha256_file(csv_path),
            },
            "summary": {
                "file": summary_path.name,
                "sha256": sha256_file(summary_path),
            },
        },
    }
    manifest_path = args.output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "valid": summary["valid"],
                "run_id": summary["source"]["run_id"],
                "alignment": summary["alignment"],
                "row_counts": summary["row_counts"],
                "overall": summary["overall"],
                "output_dir": str(args.output_dir),
                "artifact_manifest_sha256": sha256_file(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
