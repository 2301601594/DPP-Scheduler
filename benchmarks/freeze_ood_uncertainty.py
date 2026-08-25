"""Freeze the v2 OOD uncertainty coefficient from independent held-out runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from benchmarks.qwen3_runtime import (
    ACTIVE_CONFIG_RELATIVE,
    REPOSITORY_ROOT,
    candidate_runtime_signature,
    load_active_runtime,
    load_frozen_predictor,
)
from dpp_scheduler.predictor import BATCH_KINDS, ONLINE_PREDICTOR_VERSION
from dpp_scheduler.vllm_adapter import VLLM_OFFICIAL_ITERATION_TIMING


SCHEMA_VERSION = 1
COVERAGE_TARGET = 0.95
ARTIFACT_ID = "qwen3_14b_dgx_spark_ood_uncertainty_v1"
ARTIFACT_SCOPES = ("formal", "development_nonformal")
DEVELOPMENT_ROLE_REQUEST_COUNT = 150
DEVELOPMENT_CALIBRATION_CAMPAIGN = "predictor_ood_calibration_v2"
DEVELOPMENT_VALIDATION_CAMPAIGN = "predictor_ood_validation_v2"
DEVELOPMENT_CALIBRATION_SOURCE_SEED = 1001
DEVELOPMENT_VALIDATION_SOURCE_SEED = 1002
DEVELOPMENT_CALIBRATION_RECIPE_SEED = 7001
DEVELOPMENT_VALIDATION_RECIPE_SEED = 8001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _higher_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("higher quantile requires at least one sample")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * (len(ordered) - 1))]


def _display_path(path: Path, repository: Path) -> str:
    try:
        return str(path.relative_to(repository))
    except ValueError:
        return str(path)


def _load_source(
    path: Path,
    *,
    repository: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file() or path.name != "predictor_evaluation.jsonl":
        raise ValueError(f"OOD source must be predictor_evaluation.jsonl: {path}")
    manifest_path = path.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"OOD source manifest is absent: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind")
        != "qwen3_14b_predictor_online_shadow_evaluation"
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"OOD source is not a complete shadow run: {manifest_path}")
    file_hashes = manifest.get("file_sha256")
    observed_profile_sha = _sha256(path)
    if not isinstance(file_hashes, dict) or file_hashes.get(path.name) != (
        observed_profile_sha
    ):
        raise ValueError(f"OOD profile hash is not bound by manifest: {path}")
    resolved = manifest.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError(f"OOD source has no resolved runtime: {manifest_path}")
    runtime_payload = resolved.get("runtime_consistency")
    runtime_sha = resolved.get("runtime_consistency_sha256")
    if not isinstance(runtime_payload, dict) or _canonical_sha256(
        runtime_payload
    ) != runtime_sha:
        raise ValueError(f"OOD source runtime identity is invalid: {manifest_path}")
    run_id = str(manifest.get("run_id", ""))
    recipe_seed = manifest.get("recipe_seed")
    if not run_id or isinstance(recipe_seed, bool) or not isinstance(recipe_seed, int):
        raise ValueError(f"OOD source run/seed identity is invalid: {manifest_path}")
    predictor_hash = resolved.get("predictor_artifact_manifest_sha256")
    predictor_version = resolved.get("predictor_version")
    if (
        not isinstance(predictor_hash, str)
        or len(predictor_hash) != 64
        or predictor_version != ONLINE_PREDICTOR_VERSION
    ):
        raise ValueError(f"OOD source Predictor identity is invalid: {manifest_path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") != run_id:
                raise ValueError(f"{path}:{line_number} run_id mismatch")
            if row.get("sample_role") != "target":
                continue
            if row.get("batch_kind") not in BATCH_KINDS:
                raise ValueError(f"{path}:{line_number} batch kind mismatch")
            if row.get("timing_source") != VLLM_OFFICIAL_ITERATION_TIMING:
                raise ValueError(f"{path}:{line_number} timing source mismatch")
            if (
                row.get("in_support") is not False
                or row.get("prediction_mode") != "CONSTRAINED_EXTRAPOLATION"
            ):
                continue
            numeric: dict[str, float] = {}
            for field in (
                "actual_duration_seconds",
                "expected_duration_seconds",
                "centered_residual_p95_seconds",
                "ood_distance",
            ):
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{path}:{line_number} has invalid {field}")
                numeric[field] = float(value)
            if (
                not all(math.isfinite(value) for value in numeric.values())
                or numeric["actual_duration_seconds"] <= 0
                or numeric["expected_duration_seconds"] <= 0
                or numeric["centered_residual_p95_seconds"] < 0
                or numeric["ood_distance"] <= 0
            ):
                raise ValueError(f"{path}:{line_number} has invalid OOD values")
            rows.append({**row, **numeric})
    if not rows:
        raise ValueError(f"OOD source has no held-out target OOD samples: {path}")
    source = {
        "run_id": run_id,
        "campaign_id": manifest.get("campaign_id"),
        "source_seed": manifest.get("source_seed"),
        "recipe_seed": recipe_seed,
        "recipe_mode": manifest.get("recipe_mode"),
        "profile_path": _display_path(path, repository),
        "profile_sha256": observed_profile_sha,
        "run_manifest_path": _display_path(manifest_path, repository),
        "run_manifest_sha256": _sha256(manifest_path),
        "runtime_consistency_sha256": runtime_sha,
        "predictor_artifact_manifest_sha256": predictor_hash,
        "sample_count": len(rows),
    }
    identity = {
        "runtime_consistency": runtime_payload,
        "runtime_consistency_sha256": runtime_sha,
        "predictor_artifact_manifest_sha256": predictor_hash,
        "predictor_version": predictor_version,
    }
    return rows, source, identity


def _load_role(
    paths: Iterable[Path], *, repository: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    resolved_paths = tuple(sorted({path.resolve() for path in paths}))
    if not resolved_paths:
        raise ValueError("each OOD role requires at least one source")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    identity: dict[str, Any] | None = None
    for path in resolved_paths:
        source_rows, source, observed_identity = _load_source(
            path, repository=repository
        )
        if identity is None:
            identity = observed_identity
        elif identity != observed_identity:
            raise ValueError("OOD sources use inconsistent runtime/Predictor identities")
        rows.extend(source_rows)
        sources.append(source)
    assert identity is not None
    for kind in BATCH_KINDS:
        if not any(row["batch_kind"] == kind for row in rows):
            raise ValueError(f"OOD role has no {kind} samples")
    return rows, sources, identity


def _coverage(rows: list[dict[str, Any]], kappa: float) -> dict[str, Any]:
    def covered(row: dict[str, Any]) -> bool:
        bound = (
            row["expected_duration_seconds"]
            + row["centered_residual_p95_seconds"]
            + kappa * row["ood_distance"]
        )
        return row["actual_duration_seconds"] <= bound

    by_kind: dict[str, float] = {}
    counts: dict[str, int] = {}
    for kind in BATCH_KINDS:
        selected = [row for row in rows if row["batch_kind"] == kind]
        counts[kind] = len(selected)
        by_kind[kind] = sum(covered(row) for row in selected) / len(selected)
    return {
        "overall": sum(covered(row) for row in rows) / len(rows),
        "by_batch_kind": by_kind,
        "sample_count": len(rows),
        "sample_count_by_batch_kind": counts,
    }


def build_artifact(
    calibration_paths: Iterable[Path],
    validation_paths: Iterable[Path],
    *,
    repository: Path,
    scope: str = "formal",
    expected_runtime_signature_sha256: str | None = None,
    expected_predictor_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    if scope not in ARTIFACT_SCOPES:
        raise ValueError(f"unknown OOD artifact scope: {scope}")
    calibration, calibration_sources, calibration_identity = _load_role(
        calibration_paths, repository=repository
    )
    validation, validation_sources, validation_identity = _load_role(
        validation_paths, repository=repository
    )
    if calibration_identity != validation_identity:
        raise ValueError("calibration and validation runtime/Predictor differ")
    if (
        expected_runtime_signature_sha256 is not None
        and calibration_identity["runtime_consistency_sha256"]
        != expected_runtime_signature_sha256
    ):
        raise ValueError("OOD sources differ from the active runtime")
    if (
        expected_predictor_artifact_sha256 is not None
        and calibration_identity["predictor_artifact_manifest_sha256"]
        != expected_predictor_artifact_sha256
    ):
        raise ValueError("OOD sources differ from the active Predictor artifact")

    calibration_run_ids = {source["run_id"] for source in calibration_sources}
    validation_run_ids = {source["run_id"] for source in validation_sources}
    calibration_seeds = {source["recipe_seed"] for source in calibration_sources}
    validation_seeds = {source["recipe_seed"] for source in validation_sources}
    calibration_source_seeds = {
        source["source_seed"] for source in calibration_sources
    }
    validation_source_seeds = {
        source["source_seed"] for source in validation_sources
    }
    if calibration_run_ids.intersection(validation_run_ids):
        raise ValueError("calibration and validation run IDs overlap")
    if calibration_seeds.intersection(validation_seeds):
        raise ValueError("calibration and validation recipe seeds overlap")
    if calibration_source_seeds.intersection(validation_source_seeds):
        raise ValueError("calibration and validation source seeds overlap")
    if scope == "development_nonformal":
        if len(calibration_sources) != 1 or len(validation_sources) != 1:
            raise ValueError(
                "development OOD calibration requires exactly one source per role"
            )
        for role, sources, expected_campaign, expected_source_seed, expected_recipe_seed in (
            (
                "calibration",
                calibration_sources,
                DEVELOPMENT_CALIBRATION_CAMPAIGN,
                DEVELOPMENT_CALIBRATION_SOURCE_SEED,
                DEVELOPMENT_CALIBRATION_RECIPE_SEED,
            ),
            (
                "validation",
                validation_sources,
                DEVELOPMENT_VALIDATION_CAMPAIGN,
                DEVELOPMENT_VALIDATION_SOURCE_SEED,
                DEVELOPMENT_VALIDATION_RECIPE_SEED,
            ),
        ):
            source = sources[0]
            manifest_path = Path(source["run_manifest_path"])
            if not manifest_path.is_absolute():
                manifest_path = repository / manifest_path
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            resolved = manifest.get("resolved", {})
            if (
                manifest.get("recipe_mode") != "ood"
                or manifest.get("campaign_id") != expected_campaign
                or manifest.get("source_seed") != expected_source_seed
                or manifest.get("recipe_seed") != expected_recipe_seed
                or int(resolved.get("request_count", -1))
                != DEVELOPMENT_ROLE_REQUEST_COUNT
                or resolved.get("diagnostic_prefix") is not True
            ):
                raise ValueError(
                    f"development OOD {role} source must be a targeted n=150 run"
                )

    kappa_by_kind: dict[str, float] = {}
    required_by_kind: dict[str, dict[str, float | int]] = {}
    for kind in BATCH_KINDS:
        kind_rows = [row for row in calibration if row["batch_kind"] == kind]
        required = [
            max(
                0.0,
                (
                    row["actual_duration_seconds"]
                    - row["expected_duration_seconds"]
                    - row["centered_residual_p95_seconds"]
                )
                / row["ood_distance"],
            )
            for row in kind_rows
        ]
        kappa = _higher_quantile(required, COVERAGE_TARGET)
        kappa_by_kind[kind] = kappa
        required_by_kind[kind] = {
            "higher_q95": kappa,
            "sample_count": len(required),
            "max": max(required),
        }
    kappa_ood = max(kappa_by_kind.values())
    calibration_coverage = _coverage(calibration, kappa_ood)
    validation_coverage = _coverage(validation, kappa_ood)
    validation_passed = validation_coverage["overall"] >= COVERAGE_TARGET and all(
        value >= COVERAGE_TARGET
        for value in validation_coverage["by_batch_kind"].values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "status": (
            "frozen" if scope == "formal" else "frozen_development"
        ) if validation_passed else "validation_failed",
        "scope": scope,
        "formal_benchmark_eligible": scope == "formal",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_target": COVERAGE_TARGET,
        "quantile_method": "higher",
        "kappa_required_formula": (
            "max(0,(actual_duration-expected_duration-centered_p95)/d_ood)"
        ),
        "kappa_by_batch_kind": kappa_by_kind,
        "kappa_required_distribution": required_by_kind,
        "kappa_ood": kappa_ood,
        "calibration_coverage": calibration_coverage,
        "validation_coverage": validation_coverage,
        "runtime_consistency": calibration_identity["runtime_consistency"],
        "runtime_consistency_sha256": calibration_identity[
            "runtime_consistency_sha256"
        ],
        "predictor_artifact_manifest_sha256": calibration_identity[
            "predictor_artifact_manifest_sha256"
        ],
        "predictor_version": calibration_identity["predictor_version"],
        "independence": {
            "run_ids_disjoint": True,
            "source_seeds_disjoint": True,
            "recipe_seeds_disjoint": True,
        },
        "sources": {
            "calibration": calibration_sources,
            "validation": validation_sources,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-source", action="append", required=True, type=Path)
    parser.add_argument("--validation-source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--scope", choices=ARTIFACT_SCOPES, default="formal")
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    runtime = load_active_runtime(args.config)
    _, runtime_signature = candidate_runtime_signature(runtime)
    predictor = load_frozen_predictor(runtime)
    artifact = build_artifact(
        args.calibration_source,
        args.validation_source,
        repository=args.repository.resolve(),
        scope=args.scope,
        expected_runtime_signature_sha256=runtime_signature,
        expected_predictor_artifact_sha256=predictor.artifact_manifest_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if artifact["status"] in {"frozen", "frozen_development"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
