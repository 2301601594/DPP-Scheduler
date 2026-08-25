"""Pure deterministic v2 Candidate Generator.

Every normal candidate contains every active Decode request. The only search
dimension is the integer Prefill budget.
"""

from __future__ import annotations

import math

from dpp_scheduler.contracts import BatchPlan, PrefillRequest, StateSnapshot
from dpp_scheduler.settings import SchedulerSettings


def rank_decode_requests(snapshot: StateSnapshot) -> tuple:
    return tuple(
        sorted(
            snapshot.active_decode_requests,
            key=lambda item: (item.arrival_time, item.ordinal, item.request_id),
        )
    )


def rank_prefill_requests(snapshot: StateSnapshot) -> tuple[PrefillRequest, ...]:
    return tuple(
        sorted(
            snapshot.waiting_prefill_requests,
            key=lambda item: (
                0 if item.is_running else 1,
                item.arrival_time,
                item.ordinal,
                item.request_id,
            ),
        )
    )


def _highest_bindable_prefill(
    snapshot: StateSnapshot,
    prefill_order: tuple[PrefillRequest, ...],
) -> PrefillRequest | None:
    running = sum(item.is_running for item in snapshot.waiting_prefill_requests)
    new_slots = max(
        0,
        snapshot.sequence_budget - len(snapshot.active_decode_requests) - running,
    )
    for request in prefill_order:
        if request.is_running or new_slots > 0:
            return request
    return None


def highest_bindable_prefill(snapshot: StateSnapshot) -> PrefillRequest | None:
    return _highest_bindable_prefill(snapshot, rank_prefill_requests(snapshot))


def _prefill_budgets(
    snapshot: StateSnapshot,
    prefill_order: tuple[PrefillRequest, ...],
    settings: SchedulerSettings,
) -> tuple[tuple[str, int], ...]:
    decode_count = len(snapshot.active_decode_requests)
    remaining = sum(item.remaining_tokens for item in prefill_order)
    maximum = min(remaining, max(0, snapshot.token_budget - decode_count))
    raw: list[tuple[str, int]] = []
    for label, fraction in zip(
        ("ZERO", "P25", "P50", "P75", "MAX"),
        settings.prefill_budget_fractions,
    ):
        budget = maximum if fraction == 1.0 else math.floor(maximum * fraction)
        raw.append((label, max(0, min(maximum, budget))))
    highest = _highest_bindable_prefill(snapshot, prefill_order)
    if settings.include_finish_boundary and highest is not None:
        raw.append(("FINISH", min(maximum, highest.remaining_tokens)))
    deduped: dict[int, str] = {}
    for label, budget in raw:
        deduped.setdefault(budget, label)
    return tuple((label, budget) for budget, label in sorted(deduped.items()))


def _fill_prefill(
    snapshot: StateSnapshot,
    budget: int,
    order: tuple[PrefillRequest, ...],
    settings: SchedulerSettings,
) -> tuple[tuple[str, int], ...]:
    if budget <= 0:
        return ()
    running = sum(item.is_running for item in snapshot.waiting_prefill_requests)
    new_slots = max(
        0,
        snapshot.sequence_budget - len(snapshot.active_decode_requests) - running,
    )
    consumed = admitted = 0
    items: list[tuple[str, int]] = []
    for request in order:
        if not request.is_running and admitted >= new_slots:
            continue
        available = min(request.remaining_tokens, budget - consumed)
        if available <= 0:
            break
        if available < request.remaining_tokens:
            if not settings.allow_partial_prefill:
                continue
            if available < settings.minimum_prefill_chunk_tokens:
                continue
        items.append((request.request_id, available))
        consumed += available
        admitted += int(not request.is_running)
        if consumed >= budget:
            break
    return tuple(items)


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("kv_block_size must be positive")
    return (value + divisor - 1) // divisor


def project_kv_blocks(
    snapshot: StateSnapshot,
    prefill_items: tuple[tuple[str, int], ...],
    decode_items: tuple[str, ...],
) -> int:
    prefill = {item.request_id: item for item in snapshot.waiting_prefill_requests}
    decode = {item.request_id: item for item in snapshot.active_decode_requests}
    added = 0
    for request_id, tokens in prefill_items:
        request = prefill.get(request_id)
        if request is None:
            raise ValueError(f"unknown prefill request in plan: {request_id}")
        added += max(
            0,
            _ceil_div(request.prefilled_tokens + tokens, snapshot.kv_block_size)
            - _ceil_div(request.prefilled_tokens, snapshot.kv_block_size),
        )
    for request_id in decode_items:
        request = decode.get(request_id)
        if request is None:
            raise ValueError(f"unknown decode request in plan: {request_id}")
        added += max(
            0,
            _ceil_div(request.kv_context_length + 1, snapshot.kv_block_size)
            - _ceil_div(request.kv_context_length, snapshot.kv_block_size),
        )
    return max(0, snapshot.total_kv_blocks - snapshot.free_kv_blocks) + added


def project_sequence_count(
    snapshot: StateSnapshot,
    prefill_items: tuple[tuple[str, int], ...],
) -> int:
    prefill = {item.request_id: item for item in snapshot.waiting_prefill_requests}
    active = len(snapshot.active_decode_requests) + sum(
        item.is_running for item in snapshot.waiting_prefill_requests
    )
    for request_id, _ in prefill_items:
        request = prefill.get(request_id)
        if request is None:
            raise ValueError(f"unknown prefill request in plan: {request_id}")
        active += int(not request.is_running)
    return active


class CandidateGenerator:
    """Generate at most six all-Decode, Prefill-budget candidates."""

    def __init__(self, settings: SchedulerSettings | None = None) -> None:
        self.settings = settings or SchedulerSettings.provisional()

    def generate(self, snapshot: StateSnapshot) -> tuple[BatchPlan, ...]:
        if not snapshot.waiting_prefill_requests and not snapshot.active_decode_requests:
            return ()
        prefill_order = rank_prefill_requests(snapshot)
        decode_ids = tuple(item.request_id for item in rank_decode_requests(snapshot))
        plans: list[BatchPlan] = []
        seen: set[tuple[tuple[tuple[str, int], ...], tuple[str, ...]]] = set()
        for source, budget in _prefill_budgets(
            snapshot, prefill_order, self.settings
        ):
            prefill_items = _fill_prefill(
                snapshot, budget, prefill_order, self.settings
            )
            key = (prefill_items, decode_ids)
            if key in seen:
                continue
            seen.add(key)
            plans.append(
                BatchPlan(
                    plan_id=f"plan-{len(plans):03d}",
                    snapshot_hash=snapshot.snapshot_hash,
                    template_id=f"ALL_DECODE:{source}:requested_{budget}",
                    prefill_items=prefill_items,
                    decode_items=decode_ids,
                    total_prefill_tokens=sum(tokens for _, tokens in prefill_items),
                    total_decode_tokens=len(decode_ids),
                    total_sequences=project_sequence_count(snapshot, prefill_items),
                    projected_kv_blocks=project_kv_blocks(
                        snapshot, prefill_items, decode_ids
                    ),
                    mandatory_request_ids=(),
                )
            )
        if len(plans) > self.settings.maximum_seed_candidates:
            raise RuntimeError("v2 Candidate Generator exceeded six candidates")
        return tuple(plans)
