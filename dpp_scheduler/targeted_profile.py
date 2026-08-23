"""Pure helpers for targeted Predictor profiling batches.

The recipes describe only work that can be selected from the current
``StateSnapshot``.  They never contain output-length or future-EOS state.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass

from dpp_scheduler.candidate_generator import (
    project_kv_blocks,
    project_sequence_count,
)
from dpp_scheduler.contracts import BatchPlan, PrefillRequest, StateSnapshot


TARGET_CAMPAIGN_ID = "predictor_profile_targeted_prefill_mixed_n500_v1"
TARGET_SMOKE_CAMPAIGN_ID = "predictor_profile_targeted_smoke_v1"
KNEE_CAMPAIGN_ID = "candidate_knee_profile_n500_v1"
KNEE_SMOKE_CAMPAIGN_ID = "candidate_knee_profile_smoke_n10_v1"
ISOLATED_KNEE_CAMPAIGN_ID = "candidate_knee_profile_isolated_v2"
ISOLATED_KNEE_SMOKE_CAMPAIGN_ID = "candidate_knee_profile_isolated_smoke_v4"
TARGET_PROFILE_SCHEMA_VERSION = 1
TARGET_REQUEST_COUNT = 500
TARGET_RECIPE_REPEATS = 3
TARGET_CAMPAIGN_TIMEOUT_SECONDS = 8 * 60 * 60
TARGET_RUN_TIMEOUT_SECONDS = 3 * 60 * 60
TARGET_REQUEST_TIMEOUT_SECONDS = 2 * 60 * 60
TARGET_MAX_ATTEMPTS = 2
KNEE_REQUEST_COUNT = 500
KNEE_RECIPE_REPEATS = 5
KNEE_CAMPAIGN_TIMEOUT_SECONDS = 6 * 60 * 60
KNEE_RUN_TIMEOUT_SECONDS = 4 * 60 * 60
KNEE_REQUEST_TIMEOUT_SECONDS = 2 * 60 * 60
KNEE_MAX_ATTEMPTS = 2
ISOLATED_KNEE_RECIPE_REPEATS = 5
ISOLATED_KNEE_PREFILL_CAPS = (256, 384, 512, 768, 1024, 1280, 1536, 2048)
ISOLATED_KNEE_DECODE_COUNTS = (0, 8, 16, 32, 48)
ISOLATED_KNEE_PREFILL_COUNTS = (1, 4, 8)
ISOLATED_PARTIAL_SETUP_TOKENS = 16
ISOLATED_PROFILE_TOKEN_BUDGET = 2048
ISOLATED_KNEE_TARGET_COUNT = (
    sum(
        prefill + decode <= ISOLATED_PROFILE_TOKEN_BUDGET
        for prefill in ISOLATED_KNEE_PREFILL_CAPS
        for decode in ISOLATED_KNEE_DECODE_COUNTS
    )
    * len(ISOLATED_KNEE_PREFILL_COUNTS)
    * 2
    * 2
    * ISOLATED_KNEE_RECIPE_REPEATS
)


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

KNEE_CAMPAIGN_MATRIX = (TargetCampaignRun(0.2, 1001, 3001),)


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
    repeat_index: int = 0

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
            "repeat_index": self.repeat_index,
        }


def build_target_recipes(seed: int, *, mode: str = "formal") -> tuple[TargetRecipe, ...]:
    """Build a deterministic, shuffled recipe list for one independent run."""
    if mode in {"smoke", "knee_smoke"}:
        recipes = [
            TargetRecipe("smoke-p-000", "prefill_only", 16, 1, 0, "fresh", "balanced", "none"),
            TargetRecipe("smoke-m-000", "mixed", 16, 1, 1, "any", "balanced", "short"),
            TargetRecipe("smoke-p-001", "prefill_only", 64, 2, 0, "partial", "skewed", "none"),
            TargetRecipe("smoke-m-001", "mixed", 64, 2, 2, "any", "balanced", "mixed"),
        ]
        return tuple(recipes)
    if mode == "isolated_knee_smoke":
        return (
            TargetRecipe(
                "iso-smoke-p", "prefill_only", 256, 4, 0,
                "partial", "balanced", "none", 0,
            ),
            TargetRecipe(
                "iso-smoke-m", "mixed", 384, 4, 8,
                "fresh", "skewed", "measured", 1,
            ),
        )
    if mode == "isolated_knee":
        recipes = []
        ordinal = 0
        for repeat in range(ISOLATED_KNEE_RECIPE_REPEATS):
            for token_cap in ISOLATED_KNEE_PREFILL_CAPS:
                for decode_cap in ISOLATED_KNEE_DECODE_COUNTS:
                    if token_cap + decode_cap > ISOLATED_PROFILE_TOKEN_BUDGET:
                        continue
                    for request_cap in ISOLATED_KNEE_PREFILL_COUNTS:
                        for state in ("fresh", "partial"):
                            for distribution in ("balanced", "skewed"):
                                recipes.append(
                                    TargetRecipe(
                                        recipe_id=f"iso-r{repeat}-{ordinal:04d}",
                                        batch_kind=(
                                            "prefill_only"
                                            if decode_cap == 0
                                            else "mixed"
                                        ),
                                        prefill_token_cap=token_cap,
                                        prefill_request_cap=request_cap,
                                        decode_request_cap=decode_cap,
                                        prefill_state=state,
                                        prefill_distribution=distribution,
                                        decode_context=(
                                            "none" if decode_cap == 0 else "measured"
                                        ),
                                        repeat_index=repeat,
                                    )
                                )
                                ordinal += 1
        random.Random(seed).shuffle(recipes)
        return tuple(recipes)
    if mode == "knee":
        recipes = []
        ordinal = 0
        for repeat in range(KNEE_RECIPE_REPEATS):
            for token_cap in (256, 384, 512, 768, 1024, 1280, 1536, 2048):
                for request_cap in (1, 4, 8):
                    for state in ("fresh", "partial"):
                        for distribution in ("balanced", "skewed"):
                            recipes.append(
                                TargetRecipe(
                                    recipe_id=f"k-r{repeat}-{ordinal:03d}",
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
        random.Random(seed).shuffle(recipes)
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


def isolated_client_request_ids(
    recipe: TargetRecipe,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the IDs sent through the OpenAI completion API."""
    prefills = tuple(
        f"{recipe.recipe_id}-p{index:02d}"
        for index in range(recipe.prefill_request_cap)
    )
    decodes = tuple(
        f"{recipe.recipe_id}-d{index:02d}"
        for index in range(recipe.decode_request_cap)
    )
    return prefills, decodes


def resolve_isolated_request_ids(
    observed_request_ids: set[str],
    recipe: TargetRecipe,
    *,
    require_complete: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Bind recipe roles to the locked vLLM engine's actual request IDs.

    The completion endpoint adds ``cmpl-`` and ``-0`` around the client ID,
    then V1 adds an optional eight-hex-character uniqueness suffix. Exact plans
    must therefore bind after admission instead of predicting internal IDs.
    """
    client_prefills, client_decodes = isolated_client_request_ids(recipe)
    expected = client_prefills + client_decodes
    matches: dict[str, str] = {}
    for observed in observed_request_ids:
        matched = [
            client_id
            for client_id in expected
            if re.fullmatch(
                rf"cmpl-{re.escape(client_id)}-0(?:-[0-9a-f]{{8}})?",
                observed,
            )
        ]
        if len(matched) != 1:
            raise RuntimeError(f"unexpected isolated request ID: {observed}")
        client_id = matched[0]
        if client_id in matches:
            raise RuntimeError(f"duplicate isolated request ID: {client_id}")
        matches[client_id] = observed
    if require_complete and set(matches) != set(expected):
        raise RuntimeError("isolated request admission is incomplete")
    return (
        tuple(matches[item] for item in client_prefills if item in matches),
        tuple(matches[item] for item in client_decodes if item in matches),
    )


def isolated_prefill_allocations(recipe: TargetRecipe) -> tuple[int, ...]:
    """Allocate the requested Prefill cap exactly, independent of live state."""
    count = recipe.prefill_request_cap
    total = recipe.prefill_token_cap
    if count <= 0 or total < count:
        raise ValueError("isolated Prefill shape cannot allocate every request")
    if recipe.prefill_distribution == "balanced" or count == 1:
        base, extra = divmod(total, count)
        return tuple(base + (index < extra) for index in range(count))
    if recipe.prefill_distribution != "skewed":
        raise ValueError(
            f"unknown prefill distribution: {recipe.prefill_distribution}"
        )
    allocations = [1] * count
    remaining = total - count
    heavy = (remaining * 3 + 3) // 4
    allocations[0] += heavy
    remaining -= heavy
    if count > 1:
        base, extra = divmod(remaining, count - 1)
        for index in range(1, count):
            allocations[index] += base + ((index - 1) < extra)
    return tuple(allocations)


def build_isolated_target_plan(
    snapshot: StateSnapshot, recipe: TargetRecipe
) -> tuple[BatchPlan, dict[str, int | str]]:
    """Build an exact isolated target plan; never adapt or silently truncate it."""
    prefills = {
        request.request_id: request
        for request in snapshot.waiting_prefill_requests
    }
    decodes = {
        request.request_id: request
        for request in snapshot.active_decode_requests
    }
    prefill_ids, decode_ids = resolve_isolated_request_ids(
        set(prefills) | set(decodes), recipe
    )
    expected_ids = set(prefill_ids) | set(decode_ids)
    observed_ids = set(prefills) | set(decodes)
    if observed_ids != expected_ids:
        raise RuntimeError("isolated target snapshot request set is not exact")
    if set(decode_ids) != set(decodes):
        raise RuntimeError("isolated target Decode requests are not all decode-ready")

    allocations = isolated_prefill_allocations(recipe)
    prefill_items: list[tuple[str, int]] = []
    for request_id, tokens in zip(prefill_ids, allocations):
        request = prefills.get(request_id)
        if request is None:
            raise RuntimeError("isolated target Prefill request is not in Prefill state")
        if recipe.prefill_state == "fresh":
            if request.prefilled_tokens != 0 or request.is_running:
                raise RuntimeError("fresh isolated Prefill request was prepared")
        elif recipe.prefill_state == "partial":
            if (
                request.prefilled_tokens != ISOLATED_PARTIAL_SETUP_TOKENS
                or not request.is_running
            ):
                raise RuntimeError("partial isolated Prefill state is not exact")
        else:
            raise ValueError(f"invalid isolated prefill state: {recipe.prefill_state}")
        if tokens > request.remaining_tokens:
            raise RuntimeError("isolated target Prefill allocation exceeds prompt")
        prefill_items.append((request_id, tokens))

    total_sequences = project_sequence_count(snapshot, tuple(prefill_items))
    total_tokens = recipe.prefill_token_cap + recipe.decode_request_cap
    if total_tokens > snapshot.token_budget:
        raise RuntimeError("isolated target exceeds token budget")
    if total_sequences > snapshot.sequence_budget:
        raise RuntimeError("isolated target exceeds sequence budget")
    projected_kv = project_kv_blocks(snapshot, tuple(prefill_items), decode_ids)
    if projected_kv > snapshot.total_kv_blocks:
        raise RuntimeError("isolated target exceeds KV capacity")

    payload = {
        "recipe_id": recipe.recipe_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "prefill_items": prefill_items,
        "decode_items": decode_ids,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan = BatchPlan(
        plan_id=f"target-{digest[:24]}",
        snapshot_hash=snapshot.snapshot_hash,
        template_id=f"isolated:{recipe.recipe_id}",
        prefill_items=tuple(prefill_items),
        decode_items=decode_ids,
        total_prefill_tokens=recipe.prefill_token_cap,
        total_decode_tokens=recipe.decode_request_cap,
        total_sequences=total_sequences,
        projected_kv_blocks=projected_kv,
        mandatory_request_ids=(),
    )
    realized = {
        "batch_kind": recipe.batch_kind,
        "prefill_requests": len(prefill_items),
        "prefill_tokens": recipe.prefill_token_cap,
        "fresh_prefill_requests": (
            len(prefill_items) if recipe.prefill_state == "fresh" else 0
        ),
        "partial_prefill_requests": (
            len(prefill_items) if recipe.prefill_state == "partial" else 0
        ),
        "decode_requests": len(decode_ids),
    }
    return plan, realized


def build_isolated_setup_plan(
    snapshot: StateSnapshot, recipe: TargetRecipe
) -> BatchPlan | None:
    """Build one exact, unmeasured preparation round for an isolated cell."""
    prefills = {
        request.request_id: request
        for request in snapshot.waiting_prefill_requests
    }
    decodes = {
        request.request_id: request
        for request in snapshot.active_decode_requests
    }
    prefill_ids, decode_ids = resolve_isolated_request_ids(
        set(prefills) | set(decodes), recipe
    )
    expected_ids = set(prefill_ids) | set(decode_ids)
    observed_ids = set(prefills) | set(decodes)
    if observed_ids != expected_ids:
        raise RuntimeError("isolated setup snapshot request set is not exact")

    items: list[tuple[str, int]] = []
    remaining_budget = snapshot.token_budget

    if recipe.prefill_state == "partial":
        for request_id in prefill_ids:
            request = prefills.get(request_id)
            if request is None:
                raise RuntimeError("isolated partial Prefill request left Prefill state")
            if request.prefilled_tokens == 0:
                if request.remaining_tokens <= ISOLATED_PARTIAL_SETUP_TOKENS:
                    raise RuntimeError("isolated partial Prefill prompt is too short")
                items.append((request_id, ISOLATED_PARTIAL_SETUP_TOKENS))
                remaining_budget -= ISOLATED_PARTIAL_SETUP_TOKENS
            elif request.prefilled_tokens != ISOLATED_PARTIAL_SETUP_TOKENS:
                raise RuntimeError("isolated partial Prefill was prepared more than once")
    elif recipe.prefill_state != "fresh":
        raise ValueError(f"invalid isolated prefill state: {recipe.prefill_state}")

    decode_prefills = [
        prefills[request_id]
        for request_id in decode_ids
        if request_id in prefills
    ]
    if remaining_budget < 0:
        raise RuntimeError("isolated partial setup exceeds token budget")
    if decode_prefills and remaining_budget > 0:
        remaining = remaining_budget
        active = list(decode_prefills)
        while remaining > 0 and active:
            share = max(1, (remaining + len(active) - 1) // len(active))
            next_active: list[PrefillRequest] = []
            for request in active:
                grant = min(request.remaining_tokens, share, remaining)
                if grant > 0:
                    items.append((request.request_id, grant))
                    remaining -= grant
                if grant < request.remaining_tokens:
                    next_active.append(request)
                if remaining == 0:
                    break
            active = next_active

    if not items:
        if set(decode_ids) != set(decodes):
            raise RuntimeError("isolated Decode preparation made no progress")
        return None

    prefill_items = tuple(items)
    total_sequences = project_sequence_count(snapshot, prefill_items)
    if total_sequences > snapshot.sequence_budget:
        raise RuntimeError("isolated setup exceeds sequence budget")
    projected_kv = project_kv_blocks(snapshot, prefill_items, ())
    if projected_kv > snapshot.total_kv_blocks:
        raise RuntimeError("isolated setup exceeds KV capacity")
    payload = {
        "recipe_id": recipe.recipe_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "prefill_items": prefill_items,
        "role": "setup",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return BatchPlan(
        plan_id=f"setup-{digest[:24]}",
        snapshot_hash=snapshot.snapshot_hash,
        template_id=f"isolated-setup:{recipe.recipe_id}",
        prefill_items=prefill_items,
        decode_items=(),
        total_prefill_tokens=sum(tokens for _, tokens in prefill_items),
        total_decode_tokens=0,
        total_sequences=total_sequences,
        projected_kv_blocks=projected_kv,
        mandatory_request_ids=(),
    )
