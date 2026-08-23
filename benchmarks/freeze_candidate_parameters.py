#!/usr/bin/env python3
"""Analyze and freeze Candidate Generator horizon and Prefill knee parameters."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from benchmarks.qwen3_runtime import (
    ActiveRuntime,
    candidate_runtime_signature,
    load_active_runtime,
    resolve_under,
    sha256_file,
)
from benchmarks.run_candidate_knee_profile_campaign import (
    validate_campaign as validate_knee_campaign,
)
from benchmarks.run_stock_natural_eos import _git_state
from dpp_scheduler.targeted_profile import (
    ISOLATED_KNEE_CAMPAIGN_ID,
    KNEE_CAMPAIGN_MATRIX,
)


ARTIFACT_ID = "candidate_parameter_freeze_v2"
DATASET_ID = "predictor_iteration_dataset_v1"
SCHEMA_VERSION = 2
BOOTSTRAP_SEED = 20260823
BOOTSTRAP_REPLICATES = 2000
MIN_HORIZON_BUCKET_ROWS = 200
MIN_HORIZON_BUCKET_RUNS = 2
MIN_EXACT_KNEE_REPEATS = 4
KNEE_REPEATS = 5
KNEE_EFFICIENCY_THRESHOLD = 0.90
KNEE_LATER_GAIN_LIMIT = 0.05
KNEE_CAPS = (256, 384, 512, 768, 1024, 1280, 1536, 2048)
COUNT_BINS = (
    ("1", 1, 1),
    ("2-8", 2, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
)
ODD_STOCK_SEEDS = frozenset({1001, 1003, 1005, 1007, 1009, 1011})


def _quantile(values: np.ndarray, q: float, *, method: str) -> float:
    return float(np.quantile(values, q, method=method))


def _count_bin(count: int) -> str | None:
    for label, lower, upper in COUNT_BINS:
        if lower <= count <= upper:
            return label
    return None


def _iter_gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            yield row


def is_horizon_development_source(row: dict[str, Any], source: dict[str, Any] | None) -> bool:
    """Return true only for odd-seed Stock Decode-only development rows."""
    return bool(
        row.get("source_kind") == "stock"
        and row.get("batch_kind") == "decode_only"
        and source is not None
        and int(source.get("seed", -1)) in ODD_STOCK_SEEDS
    )


def _load_horizon_source(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = dataset_root / "dataset_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected Predictor dataset identity")
    decode_file = manifest.get("files", {}).get("decode_only", {})
    path = dataset_root / str(decode_file.get("file", ""))
    if not path.is_file() or sha256_file(path) != decode_file.get("sha256"):
        raise ValueError("Decode-only dataset file/hash mismatch")
    source_runs = {
        str(item["run_id"]): item
        for item in manifest.get("sources", {}).get("runs", [])
        if item.get("source_kind") == "stock"
    }
    rows: list[dict[str, Any]] = []
    for row in _iter_gzip_jsonl(path):
        run = source_runs.get(str(row.get("run_id")))
        if not is_horizon_development_source(row, run):
            continue
        selected = row.get("selected_requests")
        if not isinstance(selected, list) or not selected:
            raise ValueError("Decode-only source row has no selected requests")
        if any(item.get("phase") != "decode" for item in selected):
            raise ValueError("Decode-only source row contains non-Decode request")
        duration = float(row.get("actual_duration_seconds", math.nan))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Horizon source duration must be finite and positive")
        rows.append(
            {
                "run_id": str(row["run_id"]),
                "seed": int(run["seed"]),
                "decode_count": len(selected),
                "cumulative_context_tokens": sum(
                    int(item["current_context_tokens"]) for item in selected
                ),
                "duration_seconds": duration,
            }
        )
    if not rows:
        raise ValueError("no odd-seed Stock Decode-only development rows")
    return rows, manifest


def analyze_horizon(
    rows: list[dict[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    minimum_rows: int = MIN_HORIZON_BUCKET_ROWS,
    minimum_runs: int = MIN_HORIZON_BUCKET_RUNS,
) -> dict[str, Any]:
    """Compute stratified Decode P95 UCBs without Predictor outputs."""
    rng = np.random.default_rng(bootstrap_seed)
    bucket_reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for count_label, _, _ in COUNT_BINS:
        count_rows = [row for row in rows if _count_bin(int(row["decode_count"])) == count_label]
        contexts = np.asarray(
            [int(row["cumulative_context_tokens"]) for row in count_rows],
            dtype=np.int64,
        )
        if contexts.size == 0:
            failures.append(f"decode_count={count_label}: no rows")
            continue
        edges = [
            int(_quantile(contexts, q, method="higher"))
            for q in (0.25, 0.50, 0.75)
        ]
        for quartile_index in range(4):
            bucket = [
                row
                for row in count_rows
                if int(np.searchsorted(edges, row["cumulative_context_tokens"], side="right"))
                == quartile_index
            ]
            by_run: dict[str, list[float]] = defaultdict(list)
            for row in bucket:
                by_run[str(row["run_id"])].append(float(row["duration_seconds"]))
            eligible = len(bucket) >= minimum_rows and len(by_run) >= minimum_runs
            item: dict[str, Any] = {
                "bucket_id": f"decode_{count_label}_context_q{quartile_index + 1}",
                "decode_count_bin": count_label,
                "context_quartile": quartile_index + 1,
                "context_quantile_edges_higher": edges,
                "row_count": len(bucket),
                "run_count": len(by_run),
                "run_ids": sorted(by_run),
                "eligible": eligible,
                "p95_seconds": None,
                "p95_ucb95_seconds": None,
            }
            if not eligible:
                failures.append(
                    f"{item['bucket_id']}: rows={len(bucket)}, runs={len(by_run)}"
                )
            else:
                durations = np.asarray(
                    [float(row["duration_seconds"]) for row in bucket], dtype=float
                )
                bootstrap = np.empty(bootstrap_replicates, dtype=float)
                run_arrays = [np.asarray(by_run[run_id], dtype=float) for run_id in sorted(by_run)]
                for index in range(bootstrap_replicates):
                    sampled = np.concatenate(
                        [values[rng.integers(0, len(values), len(values))] for values in run_arrays]
                    )
                    bootstrap[index] = _quantile(sampled, 0.95, method="higher")
                item["p95_seconds"] = _quantile(durations, 0.95, method="higher")
                item["p95_ucb95_seconds"] = _quantile(
                    bootstrap, 0.95, method="higher"
                )
            bucket_reports.append(item)

    eligible = not failures and len(bucket_reports) == len(COUNT_BINS) * 4
    max_ucb = (
        max(float(item["p95_ucb95_seconds"]) for item in bucket_reports)
        if eligible
        else None
    )
    horizon = math.ceil(max_ucb * 1000.0) / 1000.0 if max_ucb is not None else None
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "critical_horizon",
        "eligible": eligible,
        "critical_horizon_seconds": horizon,
        "source_policy": "Stock Decode-only rows with odd seeds; Predictor held-out rows excluded",
        "allowed_seeds": sorted(ODD_STOCK_SEEDS),
        "row_count": len(rows),
        "run_ids": sorted({str(row["run_id"]) for row in rows}),
        "count_bins": [label for label, _, _ in COUNT_BINS],
        "context_partition": (
            "within-count cumulative-context quartiles using higher quantile edges"
        ),
        "duration_field": "actual_duration_seconds",
        "duration_timing_source": "vLLM official iteration timing",
        "bootstrap": {
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "method": "fixed-run-strata row resampling",
            "p95_method": "higher",
            "one_sided_ucb_quantile": 0.95,
        },
        "minimum_bucket_rows": minimum_rows,
        "minimum_bucket_runs": minimum_runs,
        "rounding": "ceil_to_1ms",
        "buckets": bucket_reports,
        "failures": failures,
    }


def _load_knee_rows(
    runtime: ActiveRuntime, campaign_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation = validate_knee_campaign(runtime, campaign_root)
    with (campaign_root / "campaign_checkpoint.json").open("r", encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    run_id = checkpoint["runs"][0]["valid_attempt"]
    run_root = campaign_root / "runs" / str(run_id)
    with (run_root / "run_manifest.json").open("r", encoding="utf-8") as stream:
        run_manifest = json.load(stream)
    rows: list[dict[str, Any]] = []
    with (run_root / "iteration_profile.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("sample_role") == "target":
                rows.append(row)
    return rows, {
        "validation": validation,
        "run_id": run_id,
        "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
        "iteration_profile_sha256": sha256_file(run_root / "iteration_profile.jsonl"),
        "campaign_manifest_sha256": sha256_file(campaign_root / "campaign_manifest.json"),
        "checkpoint_sha256": sha256_file(campaign_root / "campaign_checkpoint.json"),
        "config_sha256": run_manifest.get("resolved", {}).get("config_sha256"),
        "git": run_manifest.get("git"),
    }


def analyze_knee(
    rows: list[dict[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    minimum_exact_repeats: int = MIN_EXACT_KNEE_REPEATS,
) -> dict[str, Any]:
    """Select a measured knee from exact Prefill-only target iterations."""
    expected_shapes = [
        (request_cap, state, distribution)
        for request_cap in (1, 4, 8)
        for state in ("fresh", "partial")
        for distribution in ("balanced", "skewed")
    ]
    cells: dict[tuple[tuple[int, str, str], int], list[dict[str, Any]]] = defaultdict(list)
    mixed_validation_rows = 0
    for row in rows:
        requested = row.get("requested_shape")
        realized = row.get("realized_shape")
        duration = float(row.get("actual_duration_seconds", math.nan))
        if not isinstance(requested, dict) or not isinstance(realized, dict):
            raise ValueError("knee row is missing shape metadata")
        if requested.get("batch_kind") != "prefill_only":
            mixed_validation_rows += 1
            continue
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("knee duration must be finite and positive")
        shape = (
            int(requested["prefill_request_cap"]),
            str(requested["prefill_state"]),
            str(requested["prefill_distribution"]),
        )
        cap = int(requested["prefill_token_cap"])
        cells[(shape, cap)].append(
            {
                "exact": int(realized["prefill_tokens"]) == cap,
                "realized_tokens": int(realized["prefill_tokens"]),
                "realized_requests": int(realized["prefill_requests"]),
                "requested_state_match": (
                    int(realized[f"{requested['prefill_state']}_prefill_requests"])
                    == int(realized["prefill_requests"])
                ),
                "efficiency_tokens_per_second": int(realized["prefill_tokens"]) / duration,
            }
        )

    failures: list[str] = []
    cell_reports: list[dict[str, Any]] = []
    cell_efficiencies: dict[tuple[tuple[int, str, str], int], np.ndarray] = {}
    for shape in expected_shapes:
        for cap in KNEE_CAPS:
            values = cells.get((shape, cap), [])
            efficiencies = [
                float(item["efficiency_tokens_per_second"]) for item in values
            ]
            exact = [
                float(item["efficiency_tokens_per_second"])
                for item in values
                if item["exact"]
            ]
            eligible = len(values) == KNEE_REPEATS and len(exact) >= minimum_exact_repeats
            if not eligible:
                failures.append(
                    f"shape={shape}, cap={cap}: rows={len(values)}, exact={len(exact)}"
                )
            else:
                cell_efficiencies[(shape, cap)] = np.asarray(
                    efficiencies, dtype=float
                )
            cell_reports.append(
                {
                    "shape": {
                        "prefill_request_cap": shape[0],
                        "prefill_state": shape[1],
                        "prefill_distribution": shape[2],
                    },
                    "prefill_token_cap": cap,
                    "row_count": len(values),
                    "exact_count": len(exact),
                    "requested_state_match_count": sum(
                        bool(item["requested_state_match"]) for item in values
                    ),
                    "realized_request_count_median": (
                        float(np.median([item["realized_requests"] for item in values]))
                        if values
                        else None
                    ),
                    "eligible": eligible,
                    "median_efficiency_tokens_per_second": (
                        float(np.median(efficiencies)) if efficiencies else None
                    ),
                }
            )

    cap_reports: list[dict[str, Any]] = []
    knee: int | None = None
    if not failures:
        point_by_shape = {
            shape: {
                cap: float(np.median(cell_efficiencies[(shape, cap)]))
                for cap in KNEE_CAPS
            }
            for shape in expected_shapes
        }
        normalized = {
            shape: {
                cap: point_by_shape[shape][cap] / max(point_by_shape[shape].values())
                for cap in KNEE_CAPS
            }
            for shape in expected_shapes
        }
        point_by_cap = {
            cap: float(np.median([normalized[shape][cap] for shape in expected_shapes]))
            for cap in KNEE_CAPS
        }
        rng = np.random.default_rng(bootstrap_seed)
        boot_by_cap = {cap: np.empty(bootstrap_replicates, dtype=float) for cap in KNEE_CAPS}
        for replicate in range(bootstrap_replicates):
            sampled_by_shape: dict[tuple[int, str, str], dict[int, float]] = {}
            for shape in expected_shapes:
                sampled_by_shape[shape] = {}
                for cap in KNEE_CAPS:
                    values = cell_efficiencies[(shape, cap)]
                    sample = values[rng.integers(0, len(values), len(values))]
                    sampled_by_shape[shape][cap] = float(np.median(sample))
                maximum = max(sampled_by_shape[shape].values())
                sampled_by_shape[shape] = {
                    cap: value / maximum
                    for cap, value in sampled_by_shape[shape].items()
                }
            for cap in KNEE_CAPS:
                boot_by_cap[cap][replicate] = float(
                    np.median([sampled_by_shape[shape][cap] for shape in expected_shapes])
                )
        for cap_index, cap in enumerate(KNEE_CAPS):
            lcb = _quantile(boot_by_cap[cap], 0.05, method="lower")
            later_gain = max(
                [point_by_cap[later] - point_by_cap[cap] for later in KNEE_CAPS[cap_index + 1 :]]
                or [0.0]
            )
            qualifies = lcb >= KNEE_EFFICIENCY_THRESHOLD and later_gain <= KNEE_LATER_GAIN_LIMIT
            cap_reports.append(
                {
                    "prefill_token_cap": cap,
                    "normalized_efficiency_median": point_by_cap[cap],
                    "normalized_efficiency_lcb95": lcb,
                    "maximum_later_efficiency_gain": later_gain,
                    "qualifies": qualifies,
                }
            )
            if knee is None and qualifies:
                knee = cap
        if knee is None:
            failures.append("no measured cap satisfies the knee rule")

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "prefill_knee",
        "eligible": not failures and knee is not None,
        "knee_tokens": knee if not failures else None,
        "source_campaign_id": ISOLATED_KNEE_CAMPAIGN_ID,
        "seed_count": 1,
        "source_seed": KNEE_CAMPAIGN_MATRIX[0].source_seed,
        "recipe_seed": KNEE_CAMPAIGN_MATRIX[0].recipe_seed,
        "single_seed_limitation": "no cross-seed stability claim",
        "qps_role": "source trace reuse and audit only; not an analysis input",
        "target_row_count": sum(len(values) for values in cells.values()),
        "mixed_validation_row_count": mixed_validation_rows,
        "shape_count": len(expected_shapes),
        "caps": list(KNEE_CAPS),
        "repeats_per_cell": KNEE_REPEATS,
        "minimum_exact_repeats": minimum_exact_repeats,
        "efficiency": "realized_prefill_tokens / actual_duration_seconds",
        "bootstrap": {
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "method": "within-cell row resampling",
            "one_sided_lcb_quantile": 0.05,
            "quantile_method": "lower",
        },
        "selection": {
            "minimum_normalized_efficiency_lcb": KNEE_EFFICIENCY_THRESHOLD,
            "maximum_later_gain": KNEE_LATER_GAIN_LIMIT,
            "interpolation": False,
        },
        "cells": cell_reports,
        "caps_summary": cap_reports,
        "failures": failures,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_artifact(runtime: ActiveRuntime, output_root: Path) -> tuple[dict[str, Any], Path]:
    if output_root.exists():
        raise FileExistsError(f"append-only freeze artifact exists: {output_root}")
    dataset_root = resolve_under(runtime.processed_results, DATASET_ID, label="Predictor dataset")
    knee_root = resolve_under(
        runtime.raw_results, ISOLATED_KNEE_CAMPAIGN_ID, label="knee campaign"
    )
    horizon_rows, dataset_manifest = _load_horizon_source(dataset_root)
    knee_rows, knee_source = _load_knee_rows(runtime, knee_root)
    horizon_report = analyze_horizon(horizon_rows)
    knee_report = analyze_knee(knee_rows)
    eligible = bool(horizon_report["eligible"] and knee_report["eligible"])
    signature_payload, signature_sha = candidate_runtime_signature(runtime)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{ARTIFACT_ID}.tmp-", dir=output_root.parent))
    try:
        _atomic_json(temporary / "horizon_report.json", horizon_report)
        _atomic_json(temporary / "knee_report.json", knee_report)
        implementation_paths = (
            "benchmarks/freeze_candidate_parameters.py",
            "benchmarks/run_candidate_knee_profile_campaign.py",
            "benchmarks/run_isolated_candidate_profile.py",
            "benchmarks/isolated_candidate_profile.py",
            "dpp_scheduler/targeted_profile.py",
            "dpp_scheduler/isolated_profile_scheduler.py",
            "dpp_scheduler/vllm_adapter.py",
            "scripts/candidate_knee_profile_campaign.sh",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_id": ARTIFACT_ID,
            "status": "frozen" if eligible else "ineligible",
            "parameters_frozen": eligible,
            "critical_horizon_seconds": (
                horizon_report["critical_horizon_seconds"] if eligible else None
            ),
            "prefill_knee_tokens": knee_report["knee_tokens"] if eligible else None,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "official_timing_boundary": signature_payload["duration_timing_boundary"],
            "statistical_rules": {
                "horizon": {
                    "decode_count_bins": [label for label, _, _ in COUNT_BINS],
                    "context_buckets": "within-count quartiles",
                    "minimum_rows_per_bucket": MIN_HORIZON_BUCKET_ROWS,
                    "minimum_runs_per_bucket": MIN_HORIZON_BUCKET_RUNS,
                    "p95_quantile_method": "higher",
                    "ucb": "one-sided 95% stratified bootstrap",
                    "rounding": "ceil_to_1ms",
                },
                "knee": {
                    "minimum_exact_repeats": MIN_EXACT_KNEE_REPEATS,
                    "repeats_per_cell": KNEE_REPEATS,
                    "normalized_efficiency_lcb_minimum": KNEE_EFFICIENCY_THRESHOLD,
                    "maximum_later_gain": KNEE_LATER_GAIN_LIMIT,
                    "interpolation": False,
                },
            },
            "runtime_signature": signature_payload,
            "runtime_signature_sha256": signature_sha,
            "seed_count": 1,
            "single_seed_limitation": "Knee uses one seed; no cross-seed stability claim",
            "predictor_training_use": "knee campaign is excluded from Predictor training",
            "source": {
                "horizon_dataset_id": DATASET_ID,
                "horizon_dataset_manifest_sha256": sha256_file(
                    dataset_root / "dataset_manifest.json"
                ),
                "horizon_decode_file_sha256": dataset_manifest["files"]["decode_only"]["sha256"],
                "horizon_run_ids": horizon_report["run_ids"],
                "horizon_allowed_seeds": horizon_report["allowed_seeds"],
                "knee_campaign_id": ISOLATED_KNEE_CAMPAIGN_ID,
                **knee_source,
            },
            "reports": {
                "horizon_report.json": sha256_file(temporary / "horizon_report.json"),
                "knee_report.json": sha256_file(temporary / "knee_report.json"),
            },
            "implementation_sha256": {
                relative: sha256_file(runtime.workspace / relative)
                for relative in implementation_paths
            },
            "git": {
                "root": _git_state(runtime.workspace),
                "vllm": _git_state(runtime.workspace, "vllm"),
            },
        }
        _atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest, output_root


def validate_artifact(runtime: ActiveRuntime, output_root: Path) -> dict[str, Any]:
    with (output_root / "manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("artifact_id") != ARTIFACT_ID:
        raise ValueError("candidate freeze artifact identity mismatch")
    expected_report_names = {"horizon_report.json", "knee_report.json"}
    report_hashes = manifest.get("reports")
    if not isinstance(report_hashes, dict) or set(report_hashes) != expected_report_names:
        raise ValueError("candidate freeze report set mismatch")
    reports: dict[str, dict[str, Any]] = {}
    for name, expected in report_hashes.items():
        path = output_root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"candidate freeze report hash mismatch: {name}")
        with path.open("r", encoding="utf-8") as stream:
            reports[name] = json.load(stream)
    _, observed_signature = candidate_runtime_signature(runtime)
    if manifest.get("runtime_signature_sha256") != observed_signature:
        raise ValueError("candidate freeze runtime signature mismatch")
    frozen = bool(manifest.get("parameters_frozen"))
    reports_eligible = bool(
        reports["horizon_report.json"].get("eligible")
        and reports["knee_report.json"].get("eligible")
    )
    if frozen != reports_eligible:
        raise ValueError("candidate freeze report eligibility mismatch")
    if frozen != (manifest.get("status") == "frozen"):
        raise ValueError("candidate freeze status mismatch")
    if frozen and (
        manifest.get("critical_horizon_seconds") is None
        or manifest.get("prefill_knee_tokens") is None
    ):
        raise ValueError("frozen candidate parameters are incomplete")
    if frozen and (
        manifest.get("critical_horizon_seconds")
        != reports["horizon_report.json"].get("critical_horizon_seconds")
        or manifest.get("prefill_knee_tokens")
        != reports["knee_report.json"].get("knee_tokens")
    ):
        raise ValueError("candidate freeze values differ from reports")
    return {
        "valid": True,
        "artifact_id": ARTIFACT_ID,
        "parameters_frozen": frozen,
        "manifest_sha256": sha256_file(output_root / "manifest.json"),
        "runtime_signature_sha256": observed_signature,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    args = parser.parse_args()
    runtime = load_active_runtime(args.config)
    output_root = resolve_under(runtime.processed_results, ARTIFACT_ID, label="freeze artifact")
    if args.command == "build":
        manifest, _ = build_artifact(runtime, output_root)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if manifest["parameters_frozen"] else 1
    result = validate_artifact(runtime, output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
