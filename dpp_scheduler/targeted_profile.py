"""Pure helpers for targeted Predictor profiling batches.

The recipes describe only work that can be selected from the current
``StateSnapshot``.  They never contain output-length or future-EOS state.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from dpp_scheduler.candidate_generator import (
    project_kv_blocks,
    project_sequence_count,
)
from dpp_scheduler.contracts import BatchPlan, PrefillRequest, StateSnapshot


TARGET_CAMPAIGN_ID = "predictor_profile_targeted_prefill_mixed_n500_v1"
TARGET_SMOKE_CAMPAIGN_ID = "predictor_profile_targeted_smoke_v1"
TARGET_PROFILE_SCHEMA_VERSION = 1
TARGET_REQUEST_COUNT = 500
TARGET_RECIPE_REPEATS = 3
TARGET_CAMPAIGN_TIMEOUT_SECONDS = 8 * 60 * 60
TARGET_RUN_TIMEOUT_SECONDS = 3 * 60 * 60
TARGET_REQUEST_TIMEOUT_SECONDS = 2 * 60 * 60
TARGET_MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class TargetCampaignRun:
    source_qps: float
    source_seed: int
    recipe_seed: int

    @property
    def key(self) -> str:
        qps = str(float(self.source_qps)).replace(".", "p")
        return f"source_qps_{qps}_seed_{self.source_seed}_recipe_{self.recipe_seed}"

    @property
    def source_trace_filename(self) -> str:
        return f"qps_{float(self.source_qps)}_seed_{self.source_seed}.jsonl"


TARGET_CAMPAIGN_MATRIX = (
    TargetCampaignRun(0.2, 1001, 2001),
    TargetCampaignRun(0.2, 1002, 2002),
)


@dataclass(frozen=True)
class TargetRecipe:
    recipe_id: str
    batch_kind: str
    prefill_token_cap: int
    prefill_request_cap: int
    decode_request_cap: int
    prefill_state: str
    prefill_distribution: str
    decode_context: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "recipe_id": self.recipe_id,
            "batch_kind": self.batch_kind,
            "prefill_token_cap": self.prefill_token_cap,
            "prefill_request_cap": self.prefill_request_cap,
            "decode_request_cap": self.decode_request_cap,
            "prefill_state": self.prefill_state,
            "prefill_distribution": self.prefill_distribution,
            "decode_context": self.decode_context,
        }


def build_target_recipes(seed: int, *, mode: str = "formal") -> tuple[TargetRecipe, ...]:
    """Build a deterministic, shuffled recipe list for one independent run."""
    if mode == "smoke":
        recipes = [
            TargetRecipe("smoke-p-000", "prefill_only", 16, 1, 0, "fresh", "balanced", "none"),
            TargetRecipe("smoke-m-000", "mixed", 16, 1, 1, "any", "balanced", "short"),
            TargetRecipe("smoke-p-001", "prefill_only", 64, 2, 0, "partial", "skewed", "none"),
            TargetRecipe("smoke-m-001", "mixed", 64, 2, 2, "any", "balanced", "mixed"),
        ]
        return tuple(recipes)
    if mode != "formal":
        raise ValueError(f"unknown targeted recipe mode: {mode}")

    recipes: list[TargetRecipe] = []
    ordinal = 0
    for repeat in range(TARGET_RECIPE_REPEATS):
        for token_cap in (16, 64, 256, 1024, 2048):
            for request_cap in (1, 2, 4, 8):
                for state in ("fresh", "partial"):
                    for distribution in ("balanced", "skewed"):
                        recipes.append(
                            TargetRecipe(
                                recipe_id=f"p-r{repeat}-{ordinal:03d}",
                                batch_kind="prefill_only",
                                prefill_token_cap=token_cap,
                                prefill_request_cap=request_cap,
                                decode_request_cap=0,
                                prefill_state=state,
                                prefill_distribution=distribution,
                                decode_context="none",
                            )
                        )
                        ordinal += 1

    ordinal = 0
    for repeat in range(TARGET_RECIPE_REPEATS):
        for token_cap in (16, 64, 256, 1024):
            for request_cap in (1, 4):
                for decode_cap in (1, 16, 48):
                    for decode_context in ("short", "long", "mixed"):
                        recipes.append(
                            TargetRecipe(
                                recipe_id=f"m-r{repeat}-{ordinal:03d}",
                                batch_kind="mixed",
                                prefill_token_cap=token_cap,
                                prefill_request_cap=request_cap,
                                decode_request_cap=decode_cap,
                                prefill_state="any",
                                prefill_distribution="balanced",
                                decode_context=decode_context,
                            )
                        )
                        ordinal += 1

    random.Random(seed).shuffle(recipes)
    return tuple(recipes)


def _prefill_order(
    snapshot: StateSnapshot, recipe: TargetRecipe
) -> tuple[PrefillRequest, ...]:
    def state_rank(request: PrefillRequest) -> int:
        if recipe.prefill_state == "fresh":
            return 0 if not request.is_partial else 1
        if recipe.prefill_state == "partial":
            return 0 if request.is_partial else 1
        return 0

    return tuple(
        sorted(
            snapshot.waiting_prefill_requests,
            key=lambda request: (
                state_rank(request),
                request.arrival_time,
                request.ordinal,
                request.request_id,
            ),
        )
    )


def _select_prefills(
    snapshot: StateSnapshot,
    recipe: TargetRecipe,
    *,
    token_cap: int,
) -> tuple[PrefillRequest, ...]:
    if token_cap <= 0:
        return ()
    active_sequences = len(snapshot.active_decode_requests) + sum(
        request.is_running for request in snapshot.waiting_prefill_requests
    )
    new_slots = max(0, snapshot.sequence_budget - active_sequences)
    selected: list[PrefillRequest] = []
    admitted = 0
    request_limit = min(recipe.prefill_request_cap, token_cap)
    for request in _prefill_order(snapshot, recipe):
        if request.remaining_tokens <= 0:
            continue
        if not request.is_running:
            if admitted >= new_slots:
                continue
            admitted += 1
        selected.append(request)
        if len(selected) >= request_limit:
            break
    return tuple(selected)


def _decode_order(snapshot: StateSnapshot, recipe: TargetRecipe) -> tuple[str, ...]:
    ordered = sorted(
        snapshot.active_decode_requests,
        key=lambda request: (
            request.kv_context_length,
            request.arrival_time,
            request.ordinal,
            request.request_id,
        ),
    )
    if recipe.decode_context == "long":
        ordered.reverse()
    elif recipe.decode_context == "mixed":
        mixed = []
        left = 0
        right = len(ordered) - 1
        while left <= right:
            mixed.append(ordered[left])
            left += 1
            if left <= right:
                mixed.append(ordered[right])
                right -= 1
        ordered = mixed
    return tuple(request.request_id for request in ordered)


def _balanced_capacities(capacities: tuple[int, ...], total: int) -> tuple[int, ...]:
    allocations = [0] * len(capacities)
    remaining = min(total, sum(capacities))
    active = list(range(len(capacities)))
    while remaining > 0 and active:
        share = max(1, (remaining + len(active) - 1) // len(active))
        next_active: list[int] = []
        for index in active:
            room = capacities[index] - allocations[index]
            if room <= 0:
                continue
            granted = min(room, share, remaining)
            allocations[index] += granted
            remaining -= granted
            if allocations[index] < capacities[index]:
                next_active.append(index)
            if remaining == 0:
                break
        active = next_active
    return tuple(allocations)


def _balanced_allocation(
    requests: tuple[PrefillRequest, ...], total: int
) -> tuple[int, ...]:
    return _balanced_capacities(
        tuple(request.remaining_tokens for request in requests), total
    )


def _allocate_prefill_tokens(
    requests: tuple[PrefillRequest, ...],
    total: int,
    distribution: str,
) -> tuple[tuple[str, int], ...]:
    if not requests or total <= 0:
        return ()
    total = min(total, sum(request.remaining_tokens for request in requests))
    if distribution == "balanced" or len(requests) == 1:
        allocations = _balanced_allocation(requests, total)
    elif distribution == "skewed":
        allocations_list = [1] * len(requests)
        remaining = total - len(requests)
        heavy = min(
            requests[0].remaining_tokens - 1,
            max(0, (remaining * 3 + 3) // 4),
        )
        allocations_list[0] += heavy
        remaining -= heavy
        capacities = tuple(
            request.remaining_tokens - allocated
            for request, allocated in zip(requests, allocations_list)
        )
        extra = _balanced_capacities(capacities, remaining)
        allocations_list = [
            allocated + increment
            for allocated, increment in zip(allocations_list, extra)
        ]
        allocations = tuple(allocations_list)
    else:
        raise ValueError(f"unknown prefill distribution: {distribution}")
    return tuple(
        (request.request_id, allocation)
        for request, allocation in zip(requests, allocations)
        if allocation > 0
    )


def _fit_kv_capacity(
    snapshot: StateSnapshot,
    prefill_items: tuple[tuple[str, int], ...],
    decode_items: tuple[str, ...],
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...], int] | None:
    items = list(prefill_items)
    decodes = list(decode_items)
    while items or decodes:
        projected = project_kv_blocks(snapshot, tuple(items), tuple(decodes))
        if projected <= snapshot.total_kv_blocks:
            return tuple(items), tuple(decodes), projected
        if items:
            index = max(range(len(items)), key=lambda item: items[item][1])
            request_id, tokens = items[index]
            reduced = max(0, tokens - snapshot.kv_block_size)
            if reduced == 0:
                items.pop(index)
            else:
                items[index] = (request_id, reduced)
        elif decodes:
            decodes.pop()
    return None


def build_target_plan(
    snapshot: StateSnapshot, recipe: TargetRecipe
) -> tuple[BatchPlan, dict[str, int | str]] | None:
    """Construct one feasible target plan, adapting only to current availability."""
    if recipe.batch_kind not in {"prefill_only", "mixed"}:
        raise ValueError(f"invalid target batch kind: {recipe.batch_kind}")

    decode_limit = 0
    decode_items: tuple[str, ...] = ()
    if recipe.batch_kind == "mixed":
        decode_limit = min(
            recipe.decode_request_cap,
            len(snapshot.active_decode_requests),
            max(0, snapshot.token_budget - 1),
        )
        if decode_limit <= 0:
            return None
        decode_items = _decode_order(snapshot, recipe)[:decode_limit]

    prefill_budget = min(
        recipe.prefill_token_cap,
        snapshot.token_budget - len(decode_items),
    )
    prefill_requests = _select_prefills(
        snapshot, recipe, token_cap=prefill_budget
    )
    if not prefill_requests:
        return None
    prefill_items = _allocate_prefill_tokens(
        prefill_requests,
        prefill_budget,
        recipe.prefill_distribution,
    )
    if not prefill_items:
        return None

    fitted = _fit_kv_capacity(snapshot, prefill_items, decode_items)
    if fitted is None:
        return None
    prefill_items, decode_items, projected_kv = fitted
    if not prefill_items or (recipe.batch_kind == "mixed" and not decode_items):
        return None

    total_prefill = sum(tokens for _, tokens in prefill_items)
    total_decode = len(decode_items)
    total_sequences = project_sequence_count(snapshot, prefill_items)
    if total_prefill + total_decode > snapshot.token_budget:
        raise RuntimeError("target planner exceeded token budget")
    if total_sequences > snapshot.sequence_budget:
        raise RuntimeError("target planner exceeded sequence budget")

    plan_payload = {
        "recipe_id": recipe.recipe_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "prefill_items": prefill_items,
        "decode_items": decode_items,
    }
    digest = hashlib.sha256(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan = BatchPlan(
        plan_id=f"target-{digest[:24]}",
        snapshot_hash=snapshot.snapshot_hash,
        template_id=f"targeted:{recipe.recipe_id}",
        prefill_items=prefill_items,
        decode_items=decode_items,
        total_prefill_tokens=total_prefill,
        total_decode_tokens=total_decode,
        total_sequences=total_sequences,
        projected_kv_blocks=projected_kv,
        mandatory_request_ids=(),
    )
    prefill_by_id = {
        request.request_id: request
        for request in snapshot.waiting_prefill_requests
    }
    realized = {
        "batch_kind": recipe.batch_kind,
        "prefill_requests": len(prefill_items),
        "prefill_tokens": total_prefill,
        "fresh_prefill_requests": sum(
            prefill_by_id[request_id].prefilled_tokens == 0
            for request_id, _ in prefill_items
        ),
        "partial_prefill_requests": sum(
            prefill_by_id[request_id].prefilled_tokens > 0
            for request_id, _ in prefill_items
        ),
        "decode_requests": total_decode,
    }
    return plan, realized


class TargetBatchPlanner:
    """Small mutable cursor over an immutable deterministic recipe list."""

    def __init__(self, recipes: tuple[TargetRecipe, ...]) -> None:
        if not recipes:
            raise ValueError("target recipe list cannot be empty")
        self.recipes = recipes
        self.index = 0

    @property
    def complete(self) -> bool:
        return self.index >= len(self.recipes)

    @property
    def current(self) -> TargetRecipe | None:
        return None if self.complete else self.recipes[self.index]

    def build(self, snapshot: StateSnapshot) -> tuple[BatchPlan, dict[str, int | str]] | None:
        recipe = self.current
        if recipe is None:
            return None
        return build_target_plan(snapshot, recipe)

    def advance(self) -> None:
        if self.complete:
            raise RuntimeError("cannot advance a completed target planner")
        self.index += 1
