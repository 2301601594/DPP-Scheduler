"""Validation helpers for exact targeted Predictor profiling."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.predictor_profile import load_scheduled_batches
from benchmarks.qwen3_runtime import ActiveRuntime, sha256_file
from dpp_scheduler.targeted_profile import (
    TARGET_PROFILE_SCHEMA_VERSION,
    TargetCampaignRun,
    build_target_recipes,
)


def validate_reused_stock_trace(
    trace_path: Path,
    manifest_path: Path,
    runtime: ActiveRuntime,
    *,
    source_qps: float,
    source_seed: int,
    expected_request_count: int,
) -> dict[str, Any]:
    """Validate a Stock trace while permitting only its stale whole-config hash."""
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    required = {
        "kind": "qwen3_14b_poisson_length_blind_traces",
        "model_revision": runtime.model_revision,
        "tokenizer_revision": runtime.tokenizer_revision,
        "client_safety_ceiling_tokens": runtime.client_safety_ceiling_tokens,
        "client_safety_ceiling_role": (
            "termination_guard_only_never_scheduler_input"
        ),
        "predetermined_output_length": False,
        "ignore_eos": runtime.ignore_eos,
        "temperature": runtime.temperature,
        "top_p": runtime.top_p,
        "seed_source": runtime.seed_source,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"reused Stock trace {key} mismatch: expected {expected!r}, "
                f"got {manifest.get(key)!r}"
            )
    if int(manifest.get("num_requests_per_trace", 0)) != expected_request_count:
        raise ValueError("reused Stock trace request count mismatch")
    matches = [
        entry
        for entry in manifest.get("files", [])
        if entry.get("file") == trace_path.name
    ]
    if len(matches) != 1:
        raise ValueError("reused Stock trace identity is ambiguous")
    entry = matches[0]
    if (
        float(entry.get("requested_qps", -1.0)) != source_qps
        or int(entry.get("seed", -1)) != source_seed
        or int(entry.get("num_requests", 0)) != expected_request_count
    ):
        raise ValueError("reused Stock trace QPS/seed/count mismatch")
    observed_hash = sha256_file(trace_path)
    if entry.get("sha256") != observed_hash:
        raise ValueError("reused Stock trace SHA256 mismatch")
    return {
        "valid": True,
        "trace_sha256": observed_hash,
        "source_config_sha256": manifest.get("config_sha256"),
        "current_config_sha256": runtime.config_sha256,
        "config_hash_mismatch_allowed": (
            manifest.get("config_sha256") != runtime.config_sha256
        ),
        "compatibility_scope": "model_tokenizer_generation_and_trace_content",
    }


def validate_target_rows(
    path: Path,
    *,
    expected_run_id: str,
    recipe_seed: int,
    recipe_mode: str,
) -> dict[str, Any]:
    rows = load_scheduled_batches(
        path,
        expected_run_id=expected_run_id,
        allowed_plan_prefixes=("stock-", "target-"),
    )
    recipes = build_target_recipes(recipe_seed, mode=recipe_mode)
    expected_recipe_ids = [recipe.recipe_id for recipe in recipes]
    target_recipe_ids: list[str] = []
    role_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    for row in rows:
        duration = float(row.get("actual_duration_seconds", 0.0))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("targeted profile duration must be positive")
        if int(row.get("recipe_seed", -1)) != recipe_seed:
            raise ValueError("targeted profile recipe seed mismatch")
        if row.get("recipe_mode") != recipe_mode:
            raise ValueError("targeted profile recipe mode mismatch")
        role = str(row.get("sample_role", ""))
        if role not in {"setup", "target", "drain"}:
            raise ValueError(f"invalid targeted sample role: {role!r}")
        role_counts[role] += 1

        selected = row["selected_requests"]
        prefills = [item for item in selected if item["phase"] == "prefill"]
        decodes = [item for item in selected if item["phase"] == "decode"]
        if any(int(item["scheduled_tokens"]) != 1 for item in decodes):
            raise ValueError("targeted Decode request did not schedule exactly one token")
        if prefills and decodes:
            kind = "mixed"
        elif prefills:
            kind = "prefill_only"
        else:
            kind = "decode_only"
        kind_counts[kind] += 1

        if role == "target":
            if not str(row["plan_id"]).startswith("target-"):
                raise ValueError("target row does not carry an exact target plan_id")
            recipe_id = str(row.get("recipe_id", ""))
            target_recipe_ids.append(recipe_id)
            requested = row.get("requested_shape")
            if not isinstance(requested, dict) or requested.get("recipe_id") != recipe_id:
                raise ValueError("target row requested shape is missing or mismatched")
            if requested.get("batch_kind") != kind:
                raise ValueError("target row realized the wrong batch kind")
            realized = row.get("realized_shape")
            if not isinstance(realized, dict) or realized.get("batch_kind") != kind:
                raise ValueError("target row realized-shape metadata is invalid")
            if int(realized.get("prefill_requests", -1)) != len(prefills):
                raise ValueError("target row Prefill request count mismatch")
            if int(realized.get("prefill_tokens", -1)) != sum(
                int(item["scheduled_tokens"]) for item in prefills
            ):
                raise ValueError("target row Prefill token count mismatch")
            if int(realized.get("decode_requests", -1)) != len(decodes):
                raise ValueError("target row Decode request count mismatch")
        elif str(row["plan_id"]).startswith("target-"):
            raise ValueError("non-target row carries a target plan_id")

    if target_recipe_ids != expected_recipe_ids:
        raise ValueError(
            "target recipe execution mismatch: "
            f"expected={len(expected_recipe_ids)}, observed={len(target_recipe_ids)}"
        )
    expected_kind_counts = Counter(recipe.batch_kind for recipe in recipes)
    observed_target_kinds = Counter(
        row["requested_shape"]["batch_kind"]
        for row in rows
        if row["sample_role"] == "target"
    )
    if observed_target_kinds != expected_kind_counts:
        raise ValueError("target batch-kind counts do not match the recipe matrix")
    return {
        "schema_version": TARGET_PROFILE_SCHEMA_VERSION,
        "valid": True,
        "run_id": expected_run_id,
        "iteration_count": len(rows),
        "target_count": role_counts["target"],
        "setup_count": role_counts["setup"],
        "drain_count": role_counts["drain"],
        "target_batch_kind_counts": dict(sorted(expected_kind_counts.items())),
        "all_batch_kind_counts": dict(sorted(kind_counts.items())),
    }


def validate_target_run_directory(
    path: Path,
    *,
    expected_run: TargetCampaignRun,
    expected_request_count: int,
    recipe_mode: str = "formal",
) -> dict[str, Any]:
    with (path / "run_manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with (path / "summary.json").open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    with (path / "target_validation.json").open("r", encoding="utf-8") as stream:
        recorded_validation = json.load(stream)

    if manifest.get("status") != "complete":
        raise ValueError(f"target run manifest is not complete: {path}")
    if int(manifest.get("source_seed", -1)) != expected_run.source_seed:
        raise ValueError(f"target run source seed mismatch: {path}")
    if int(manifest.get("recipe_seed", -1)) != expected_run.recipe_seed:
        raise ValueError(f"target run recipe seed mismatch: {path}")
    if int(summary.get("num_requests", 0)) != expected_request_count:
        raise ValueError(f"target run request count mismatch: {path}")
    if int(summary.get("completed", 0)) != expected_request_count:
        raise ValueError(f"target run has incomplete requests: {path}")
    if int(summary.get("failed", -1)) != 0:
        raise ValueError(f"target run has failed requests: {path}")
    validation = validate_target_rows(
        path / "iteration_profile.jsonl",
        expected_run_id=str(manifest["run_id"]),
        recipe_seed=expected_run.recipe_seed,
        recipe_mode=recipe_mode,
    )
    if recorded_validation != validation:
        raise ValueError(f"target validation artifact mismatch: {path}")
    return {
        "valid": True,
        "run_id": manifest["run_id"],
        "request_count": expected_request_count,
        "iteration_count": validation["iteration_count"],
        "target_count": validation["target_count"],
        "target_batch_kind_counts": validation["target_batch_kind_counts"],
    }
