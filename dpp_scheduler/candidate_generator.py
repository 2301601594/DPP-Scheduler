"""Deterministic candidate generator for the G2 exact-BatchPlan path.

This module is pure Python.  It never touches vLLM internals or the real KV
block manager.  All resource calculations are side-effect-free projections.
"""

from __future__ import annotations

import math
from dataclasses import replace

from dpp_scheduler.contracts import (
    BatchPlan,
    DecodeRequest,
    PrefillRequest,
    StateSnapshot,
)
from dpp_scheduler.settings import SchedulerSettings


def _oldest_due_recovery(snapshot: StateSnapshot) -> str | None:
    recovery_set = set(snapshot.recovery_requests)
    due = [
        item
        for item in snapshot.active_decode_requests
        if item.recovery_due and item.request_id in recovery_set
    ]
    if not due:
        return None
    return min(
        due,
        key=lambda item: (
            item.recovery_first_miss_time
            if item.recovery_first_miss_time is not None
            else item.arrival_time,
            item.arrival_time,
            item.ordinal,
            item.request_id,
        ),
    ).request_id


def _rank_decode(snapshot: StateSnapshot) -> tuple[DecodeRequest, ...]:
    decode = list(snapshot.active_decode_requests)
    recovery_set = set(snapshot.recovery_requests)
    oldest_due = _oldest_due_recovery(snapshot)

    def category(item: DecodeRequest) -> int:
        if item.request_id == oldest_due:
            return 0
        if item.request_id not in recovery_set and item.tbt_deadline is not None:
            return 1
        if item.request_id not in recovery_set:
            return 2
        return 3

    def sort_key(item: DecodeRequest) -> tuple:
        if category(item) == 0:
            return (0, item.arrival_time, item.ordinal, item.request_id)
        if category(item) == 1:
            return (
                1,
                item.tbt_deadline if item.tbt_deadline is not None else math.inf,
                item.arrival_time,
                item.ordinal,
                item.request_id,
            )
        if category(item) == 2:
            return (2, item.arrival_time, item.ordinal, item.request_id)
        return (
            3,
            item.recovery_first_miss_time
            if item.recovery_first_miss_time is not None
            else item.arrival_time,
            item.arrival_time,
            item.ordinal,
            item.request_id,
        )

    return tuple(sorted(decode, key=sort_key))


def _rank_prefill(snapshot: StateSnapshot) -> tuple[PrefillRequest, ...]:
    prefill = list(snapshot.waiting_prefill_requests)

    def category(item: PrefillRequest) -> int:
        if item.hard_ttft_protected:
            return 0
        if item.is_partial:
            return 1
        return 2

    def sort_key(item: PrefillRequest) -> tuple:
        if category(item) == 0:
            return (
                0,
                item.ttft_deadline if item.ttft_deadline is not None else math.inf,
                item.arrival_time,
                item.ordinal,
                item.request_id,
            )
        if category(item) == 1:
            return (1, item.arrival_time, item.ordinal, item.request_id)
        return (2, item.arrival_time, item.ordinal, item.request_id)

    return tuple(sorted(prefill, key=sort_key))


def rank_decode_requests(snapshot: StateSnapshot) -> tuple[DecodeRequest, ...]:
    """Expose the frozen Decode order for Controller-owned Fallback reuse."""
    return _rank_decode(snapshot)


def rank_prefill_requests(snapshot: StateSnapshot) -> tuple[PrefillRequest, ...]:
    """Expose the frozen Prefill order for Controller-owned Fallback reuse."""
    return _rank_prefill(snapshot)


def _mandatory_decode(
    ordered: tuple[DecodeRequest, ...], oldest_due_recovery: str | None
) -> list[str]:
    return [
        item.request_id
        for item in ordered
        if item.mandatory or item.request_id == oldest_due_recovery
    ]


def _bind_decode(
    profile: str,
    ordered: tuple[DecodeRequest, ...],
    snapshot: StateSnapshot,
    settings: SchedulerSettings,
    oldest_due_recovery: str | None,
    recovery_set: set[str],
) -> tuple[str, ...]:
    mandatory = set(_mandatory_decode(ordered, oldest_due_recovery))
    if profile == "MANDATORY":
        return tuple(
            item.request_id for item in ordered if item.request_id in mandatory
        )
    if profile == "CRITICAL":
        selected = set(mandatory)
        horizon = settings.critical_horizon_seconds
        if horizon is not None:
            for item in ordered:
                if item.request_id in recovery_set or item.tbt_deadline is None:
                    continue
                if item.tbt_deadline - snapshot.timestamp <= horizon:
                    selected.add(item.request_id)
        return tuple(
            item.request_id for item in ordered if item.request_id in selected
        )
    if profile == "ALL":
        return tuple(item.request_id for item in ordered)
    raise ValueError(f"unknown decode profile: {profile}")


def _trim_decode_to_budget(
    decode_ids: tuple[str, ...],
    snapshot: StateSnapshot,
    ordered: tuple[DecodeRequest, ...],
) -> tuple[str, ...] | None:
    """Keep as many decode ids as the per-iteration token/sequence limits allow.

    Each decode item consumes one token in this design.  Mandatory items are
    kept as long as they individually fit; later items are trimmed from the end
    in ranked order.
    """
    if (
        len(decode_ids) <= snapshot.sequence_budget
        and len(decode_ids) <= snapshot.token_budget
    ):
        return decode_ids

    by_id = {item.request_id: item for item in ordered}
    oldest_due = _oldest_due_recovery(snapshot)
    mandatory_ids = [
        rid for rid in decode_ids
        if by_id[rid].mandatory or rid == oldest_due
    ]
    limit = min(snapshot.sequence_budget, snapshot.token_budget)
    if len(mandatory_ids) > limit:
        # A profile that silently drops protected Decode work is not the same
        # action. Let later fallback/preemption ownership handle this state.
        return None
    selected = set(mandatory_ids)
    for rid in decode_ids:
        if rid in selected:
            continue
        if len(selected) >= limit:
            break
        selected.add(rid)
    return tuple(rid for rid in decode_ids if rid in selected)


def _highest_bindable_prefill(
    snapshot: StateSnapshot,
    prefill_order: tuple[PrefillRequest, ...],
) -> PrefillRequest | None:
    running_prefill = sum(
        request.is_running for request in snapshot.waiting_prefill_requests
    )
    active_sequences = len(snapshot.active_decode_requests) + running_prefill
    new_sequence_slots = max(0, snapshot.sequence_budget - active_sequences)
    for request in prefill_order:
        if request.remaining_tokens <= 0:
            continue
        if request.is_running or new_sequence_slots > 0:
            return request
    return None


def highest_bindable_prefill(snapshot: StateSnapshot) -> PrefillRequest | None:
    """Return the first ranked Prefill request that can consume a sequence slot."""
    return _highest_bindable_prefill(snapshot, _rank_prefill(snapshot))


def _prefill_breakpoints(
    snapshot: StateSnapshot,
    decode_count: int,
    prefill_order: tuple[PrefillRequest, ...],
    settings: SchedulerSettings,
) -> tuple[tuple[str, int], ...]:
    """Return snapshot-only seed caps in stable semantic order."""
    breakpoints: list[tuple[str, int]] = [("ZERO", 0)]
    visible_backlog = sum(request.remaining_tokens for request in prefill_order)
    bindable_max = min(
        max(0, snapshot.token_budget - decode_count),
        visible_backlog,
    )
    if bindable_max <= 0:
        return tuple(breakpoints)

    highest = _highest_bindable_prefill(snapshot, prefill_order)
    if highest is not None and highest.remaining_tokens <= bindable_max:
        breakpoints.append(("FINISH", highest.remaining_tokens))

    if settings.prefill_knee_tokens is not None:
        breakpoints.append(
            ("KNEE", min(settings.prefill_knee_tokens, bindable_max))
        )
    breakpoints.append(("BINDABLE_MAX", bindable_max))
    return tuple(breakpoints)


def _fill_prefill(
    snapshot: StateSnapshot,
    decode_count: int,
    prefill_cap: int,
    prefill_order: tuple[PrefillRequest, ...],
    settings: SchedulerSettings,
) -> tuple[tuple[str, int], ...]:
    """Fill prefill items up to *prefill_cap* under token/sequence limits.

    Requests are scheduled as whole remaining prompts when they fit; otherwise
    they are partially scheduled if ``allow_partial_prefill`` is enabled.
    """
    if prefill_cap <= 0:
        return ()

    token_budget_for_prefill = max(
        0, snapshot.token_budget - decode_count
    )
    running_prefill = sum(
        request.is_running for request in snapshot.waiting_prefill_requests
    )
    active_sequences = len(snapshot.active_decode_requests) + running_prefill
    new_sequence_budget = max(0, snapshot.sequence_budget - active_sequences)
    remaining_cap = min(prefill_cap, token_budget_for_prefill)

    items: list[tuple[str, int]] = []
    scheduled_tokens = 0
    admitted_sequences = 0

    for request in prefill_order:
        if not request.is_running and admitted_sequences >= new_sequence_budget:
            # A new request cannot be admitted, but a later running partial
            # Prefill may still consume no additional sequence slot.
            continue
        if scheduled_tokens >= remaining_cap:
            break
        remaining = request.remaining_tokens
        if remaining <= 0:
            continue

        available = min(remaining_cap - scheduled_tokens, remaining)
        if available <= 0:
            break

        if available == remaining:
            scheduled = remaining
        else:
            if not settings.allow_partial_prefill:
                continue
            if available < settings.minimum_prefill_chunk_tokens:
                continue
            scheduled = available

        if scheduled <= 0:
            break

        items.append((request.request_id, scheduled))
        scheduled_tokens += scheduled
        if not request.is_running:
            admitted_sequences += 1

    return tuple(items)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("kv_block_size must be positive")
    return (numerator + denominator - 1) // denominator


def project_kv_blocks(
    snapshot: StateSnapshot,
    prefill_items: tuple[tuple[str, int], ...],
    decode_items: tuple[str, ...],
) -> int:
    prefill_by_id = {req.request_id: req for req in snapshot.waiting_prefill_requests}
    decode_by_id = {req.request_id: req for req in snapshot.active_decode_requests}

    current_allocated = max(0, snapshot.total_kv_blocks - snapshot.free_kv_blocks)
    added = 0
    block_size = snapshot.kv_block_size

    for request_id, scheduled_tokens in prefill_items:
        request = prefill_by_id.get(request_id)
        if request is None:
            raise ValueError(f"unknown prefill request in plan: {request_id}")
        old_blocks = _ceil_div(request.prefilled_tokens, block_size)
        new_blocks = _ceil_div(request.prefilled_tokens + scheduled_tokens, block_size)
        added += max(0, new_blocks - old_blocks)

    for request_id in decode_items:
        request = decode_by_id.get(request_id)
        if request is None:
            raise ValueError(f"unknown decode request in plan: {request_id}")
        old_blocks = _ceil_div(request.kv_context_length, block_size)
        new_blocks = _ceil_div(request.kv_context_length + 1, block_size)
        added += max(0, new_blocks - old_blocks)

    return current_allocated + added


def project_sequence_count(
    snapshot: StateSnapshot,
    prefill_items: tuple[tuple[str, int], ...],
) -> int:
    """Return projected active sequences, not scheduled items this round."""
    prefill_by_id = {
        request.request_id: request for request in snapshot.waiting_prefill_requests
    }
    active = len(snapshot.active_decode_requests) + sum(
        request.is_running for request in snapshot.waiting_prefill_requests
    )
    for request_id, _ in prefill_items:
        request = prefill_by_id.get(request_id)
        if request is None:
            raise ValueError(f"unknown prefill request in plan: {request_id}")
        if not request.is_running:
            active += 1
    return active


class CandidateGenerator:
    """Generate the fixed at-most-12 deterministic BatchPlan candidates."""

    def __init__(self, settings: SchedulerSettings | None = None) -> None:
        self.settings = settings or SchedulerSettings.provisional()

    def generate(self, snapshot: StateSnapshot) -> tuple[BatchPlan, ...]:
        if (
            not snapshot.waiting_prefill_requests
            and not snapshot.active_decode_requests
        ):
            return ()

        ordered_decode = _rank_decode(snapshot)
        ordered_prefill = _rank_prefill(snapshot)
        recovery_set = set(snapshot.recovery_requests)
        oldest_due = _oldest_due_recovery(snapshot)

        raw_plans: list[tuple[str, BatchPlan]] = []
        profile_order = self.settings.template_names

        for profile in profile_order:
            decode_ids = _bind_decode(
                profile,
                ordered_decode,
                snapshot,
                self.settings,
                oldest_due,
                recovery_set,
            )
            decode_ids = _trim_decode_to_budget(decode_ids, snapshot, ordered_decode)
            if decode_ids is None:
                continue
            cap_order = _prefill_breakpoints(
                snapshot, len(decode_ids), ordered_prefill, self.settings
            )
            for cap_source, cap in cap_order:
                prefill_items = _fill_prefill(
                    snapshot, len(decode_ids), cap, ordered_prefill, self.settings
                )
                total_prefill = sum(count for _, count in prefill_items)
                total_decode = len(decode_ids)
                total_sequences = project_sequence_count(snapshot, prefill_items)
                if total_prefill + total_decode > snapshot.token_budget:
                    continue
                if total_sequences > snapshot.sequence_budget:
                    continue
                projected_kv = project_kv_blocks(
                    snapshot, prefill_items, decode_ids
                )
                template_id = f"{profile}:{cap_source}:requested_{cap}"
                plan = BatchPlan(
                    plan_id="",  # assigned after canonical dedup
                    snapshot_hash=snapshot.snapshot_hash,
                    template_id=template_id,
                    prefill_items=prefill_items,
                    decode_items=decode_ids,
                    total_prefill_tokens=total_prefill,
                    total_decode_tokens=total_decode,
                    total_sequences=total_sequences,
                    projected_kv_blocks=projected_kv,
                    mandatory_request_ids=tuple(
                        rid for rid in decode_ids
                        if (
                            rid == oldest_due
                            or any(
                                r.request_id == rid and r.mandatory
                                for r in ordered_decode
                            )
                        )
                    ),
                )
                raw_plans.append((template_id, plan))

        if len(raw_plans) > self.settings.maximum_seed_candidates:
            raise RuntimeError("Candidate Generator exceeded the 12-seed bound")

        return self._canonical_deduplicate(raw_plans)

    def _canonical_deduplicate(
        self, raw_plans: list[tuple[str, BatchPlan]]
    ) -> tuple[BatchPlan, ...]:
        deduped: dict[tuple, BatchPlan] = {}
        # Deterministic first-seen order: profile order, then cap order.
        for _, plan in raw_plans:
            key = (plan.prefill_items, plan.decode_items)
            if key not in deduped:
                deduped[key] = plan

        ordered = sorted(
            deduped.values(),
            key=lambda p: (
                p.prefill_items,
                p.decode_items,
                p.template_id,
            ),
        )
        result: list[BatchPlan] = []
        for index, plan in enumerate(ordered):
            result.append(
                replace(plan, plan_id=f"plan-{index:03d}")
            )
        return tuple(result)
