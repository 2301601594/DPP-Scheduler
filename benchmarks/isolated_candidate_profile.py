"""Construction and validation helpers for isolated Candidate profiling."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from dpp_scheduler.targeted_profile import (
    ISOLATED_PARTIAL_SETUP_TOKENS,
    TargetRecipe,
    build_target_recipes,
    isolated_client_request_ids,
    isolated_prefill_allocations,
)
from dpp_scheduler.vllm_adapter import VLLM_OFFICIAL_ITERATION_TIMING


DECODE_CONTEXT_ANCHORS = (128, 192, 256, 384, 512)


def _pick_source(
    source_rows: list[dict[str, Any]],
    *,
    minimum_tokens: int,
    anchor_tokens: int,
    offset: int,
) -> dict[str, Any]:
    eligible = [
        row for row in source_rows if int(row["input_tokens"]) >= minimum_tokens
    ]
    if not eligible:
        raise ValueError(
            f"source trace has no prompt with at least {minimum_tokens} tokens"
        )
    ordered = sorted(
        eligible,
        key=lambda row: (
            abs(int(row["input_tokens"]) - anchor_tokens),
            int(row["input_tokens"]),
            str(row["request_id"]),
        ),
    )
    return ordered[offset % len(ordered)]


def build_isolated_request_rows(
    source_rows: list[dict[str, Any]],
    recipe: TargetRecipe,
    *,
    recipe_ordinal: int,
) -> list[dict[str, Any]]:
    """Construct only the requests needed by one target batch."""
    prefill_ids, decode_ids = isolated_client_request_ids(recipe)
    allocations = isolated_prefill_allocations(recipe)
    rows: list[dict[str, Any]] = []
    for index, (request_id, allocation) in enumerate(zip(prefill_ids, allocations)):
        setup = (
            ISOLATED_PARTIAL_SETUP_TOKENS
            if recipe.prefill_state == "partial"
            else 0
        )
        minimum = setup + allocation
        source = _pick_source(
            source_rows,
            minimum_tokens=minimum,
            anchor_tokens=minimum,
            offset=recipe_ordinal + index,
        )
        row = dict(source)
        row.update(
            {
                "request_id": request_id,
                "arrival_time_s": 0.0,
                "generation_seed": (
                    int(source["generation_seed"]) + recipe_ordinal * 97 + index
                ),
                "profile_recipe_id": recipe.recipe_id,
                "profile_request_role": "prefill",
            }
        )
        rows.append(row)

    anchor = DECODE_CONTEXT_ANCHORS[
        recipe.repeat_index % len(DECODE_CONTEXT_ANCHORS)
    ]
    for index, request_id in enumerate(decode_ids):
        source = _pick_source(
            source_rows,
            minimum_tokens=1,
            anchor_tokens=anchor,
            offset=recipe_ordinal * 53 + index,
        )
        row = dict(source)
        row.update(
            {
                "request_id": request_id,
                "arrival_time_s": 0.0,
                "generation_seed": (
                    int(source["generation_seed"]) + recipe_ordinal * 97
                    + len(prefill_ids) + index
                ),
                "profile_recipe_id": recipe.recipe_id,
                "profile_request_role": "decode",
            }
        )
        rows.append(row)
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def validate_isolated_profile(
    profile_path: Path,
    event_path: Path,
    *,
    expected_run_id: str,
    recipe_seed: int,
    recipe_mode: str,
) -> dict[str, Any]:
    recipes = build_target_recipes(recipe_seed, mode=recipe_mode)
    expected = {recipe.recipe_id: recipe for recipe in recipes}
    rows = load_jsonl(profile_path)
    events = load_jsonl(event_path)
    failures = [event for event in events if event.get("event") == "batch_failed"]
    completions = [event for event in events if event.get("event") == "batch_complete"]
    if failures:
        raise ValueError(f"isolated profile contains {len(failures)} failed batches")
    if len(rows) != len(recipes) or len(completions) != len(recipes):
        raise ValueError(
            "isolated profile target count mismatch: "
            f"recipes={len(recipes)}, rows={len(rows)}, completions={len(completions)}"
        )
    if [row.get("recipe_id") for row in rows] != [r.recipe_id for r in recipes]:
        raise ValueError("isolated target execution order differs from recipes")

    kinds: Counter[str] = Counter()
    for row in rows:
        recipe_id = str(row.get("recipe_id", ""))
        recipe = expected.get(recipe_id)
        if recipe is None:
            raise ValueError(f"unknown isolated recipe: {recipe_id}")
        if row.get("run_id") != expected_run_id:
            raise ValueError("isolated run_id mismatch")
        if row.get("timing_source") != VLLM_OFFICIAL_ITERATION_TIMING:
            raise ValueError("isolated target did not use official iteration timing")
        duration = float(row.get("actual_duration_seconds", 0.0))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("isolated target duration is invalid")
        if not row.get("execution_match") or not row.get("admission_closed"):
            raise ValueError("isolated target execution/admission proof failed")
        cleanup = row.get("cleanup")
        if not isinstance(cleanup, dict):
            raise ValueError("isolated target cleanup proof is missing")
        if (
            not cleanup.get("cleanup_started_after_timing")
            or not cleanup.get("resource_recovered")
            or int(cleanup.get("post_cleanup_request_count", -1)) != 0
            or int(cleanup.get("post_cleanup_running_count", -1)) != 0
            or int(cleanup.get("post_cleanup_waiting_count", -1)) != 0
            or int(cleanup.get("post_cleanup_free_kv_blocks", -1))
            != int(cleanup.get("baseline_free_kv_blocks", -2))
        ):
            raise ValueError("isolated target did not restore the clean baseline")
        selected = row.get("selected_requests")
        if not isinstance(selected, list):
            raise ValueError("isolated selected_requests is missing")
        prefills = [item for item in selected if item.get("phase") == "prefill"]
        decodes = [item for item in selected if item.get("phase") == "decode"]
        prefill_tokens = sum(int(item["scheduled_tokens"]) for item in prefills)
        if (
            len(prefills) != recipe.prefill_request_cap
            or prefill_tokens != recipe.prefill_token_cap
            or len(decodes) != recipe.decode_request_cap
            or any(int(item["scheduled_tokens"]) != 1 for item in decodes)
        ):
            raise ValueError("isolated target realized shape differs from recipe")
        requested = row.get("requested_shape")
        if requested != recipe.as_dict():
            raise ValueError("isolated requested_shape differs from recipe")
        expected_ratio = recipe.prefill_token_cap / (
            recipe.prefill_token_cap + recipe.decode_request_cap
        )
        row["prefill_ratio"] = expected_ratio
        kinds[recipe.batch_kind] += 1
    return {
        "schema_version": 2,
        "valid": True,
        "run_id": expected_run_id,
        "target_count": len(rows),
        "target_batch_kind_counts": dict(sorted(kinds.items())),
        "failed_batch_count": 0,
        "official_timing_count": len(rows),
        "clean_baseline_count": len(rows),
    }


def validate_isolated_run_directory(
    path: Path,
    *,
    expected_run_id: str,
    recipe_seed: int,
    recipe_mode: str,
) -> dict[str, Any]:
    with (path / "run_manifest.json").open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with (path / "summary.json").open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    with (path / "target_validation.json").open("r", encoding="utf-8") as stream:
        recorded = json.load(stream)
    if manifest.get("status") != "complete":
        raise ValueError("isolated run manifest is not complete")
    if int(summary.get("failed_batch_count", -1)) != 0:
        raise ValueError("isolated run summary contains failed batches")
    validation = validate_isolated_profile(
        path / "iteration_profile.jsonl",
        path / "batch_events.jsonl",
        expected_run_id=expected_run_id,
        recipe_seed=recipe_seed,
        recipe_mode=recipe_mode,
    )
    if validation != recorded or manifest.get("validation") != validation:
        raise ValueError("isolated validation artifacts disagree")
    return validation
