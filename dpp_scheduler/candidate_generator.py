"""Deterministic candidate generator for the G2 exact-BatchPlan path.

This module is pure Python.  It never touches vLLM internals or the real KV
block manager.  All resource calculations are side-effect-free projections.
"""

from __future__ import annotations

import math
from dataclasses import replace
from dpp_scheduler.contracts import BatchPlan, DecodeRequest, PrefillRequest, StateSnapshot
from dpp_scheduler.settings import SchedulerSettings

def _rank_decode(snapshot: StateSnapshot) -> tuple[DecodeRequest, ...]:
    decode = list(snapshot.active_decode_requests)
    recovery_set = set(snapshot.recovery_requests)

    def category(item: DecodeRequest) -> int:
        if item.recovery_due or item.request_id in recovery_set:
            return 0
        if item.tbt_deadline is not None:
            return 1
        return 2

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
        return (2, item.arrival_time, item.ordinal, item.request_id)

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


def _mandatory_decode(
    ordered: tuple[DecodeRequest, ...], recovery_set: set[str]
) -> list[str]:
    return [
        item.request_id
        for item in ordered
        if item.mandatory or item.recovery_due or item.request_id in recovery_set
    ]


def _bind_decode(
    profile: str,
    ordered: tuple[DecodeRequest, ...],
    settings: SchedulerSettings,
    recovery_set: set[str],
) -> tuple[str, ...]:
    mandatory = _mandatory_decode(ordered, recovery_set)
    if profile == "MANDATORY":
        return tuple(mandatory)
    if profile == "URGENT":
        result = list(mandatory)
        for item in ordered:
            if item.request_id in result:
                continue
            if len(result) >= len(mandatory) + settings.urgent_limit_u:
                break
            result.append(item.request_id)
        return tuple(result)
    if profile == "ALL":
        return tuple(item.request_id for item in ordered)
    raise ValueError(f"unknown decode profile: {profile}")


def _trim_decode_to_budget(
    decode_ids: tuple[str, ...],
    snapshot: StateSnapshot,
    ordered: tuple[DecodeRequest, ...],
) -> tuple[str, ...]:
    """Keep as many decode ids as the per-iteration token/sequence limits allow.

    Each decode item consumes one token in this design.  Mandatory items are
    kept as long as they individually fit; later items are trimmed from the end
    in ranked order.
    """
    if len(decode_ids) <= snapshot.sequence_budget and len(decode_ids) <= snapshot.token_budget:
        return decode_ids

    by_id = {item.request_id: item for item in ordered}
    mandatory_ids = [rid for rid in decode_ids if by_id[rid].mandatory or by_id[rid].recovery_due or rid in set(snapshot.recovery_requests)]
    remaining = [rid for rid in decode_ids if rid not in mandatory_ids]
    selected = list(mandatory_ids[: min(snapshot.sequence_budget, snapshot.token_budget)])
    for rid in remaining:
        if len(selected) + 1 > snapshot.sequence_budget:
            break
        if len(selected) + 1 > snapshot.token_budget:
            break
        selected.append(rid)
    return tuple(selected)


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
    seed_budget_for_prefill = max(0, snapshot.sequence_budget - decode_count)
    remaining_cap = min(prefill_cap, token_budget_for_prefill)

    items: list[tuple[str, int]] = []
    scheduled_tokens = 0
    used_sequences = 0

    for request in prefill_order:
        if used_sequences >= seed_budget_for_prefill:
            break
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
            scheduled = max(
                settings.minimum_prefill_chunk_tokens,
                min(available, remaining),
            )
            # Do not exceed the remaining cap.
            scheduled = min(scheduled, remaining_cap - scheduled_tokens)

        if scheduled <= 0:
            break

        items.append((request.request_id, scheduled))
        scheduled_tokens += scheduled
        used_sequences += 1

    return tuple(items)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("kv_block_size must be positive")
    return (numerator + denominator - 1) // denominator


def _projected_kv_blocks(
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
            continue
        old_blocks = _ceil_div(request.prefilled_tokens, block_size)
        new_blocks = _ceil_div(request.prefilled_tokens + scheduled_tokens, block_size)
        added += max(0, new_blocks - old_blocks)

    for request_id in decode_items:
        request = decode_by_id.get(request_id)
        if request is None:
            continue
        old_blocks = _ceil_div(request.kv_context_length, block_size)
        new_blocks = _ceil_div(request.kv_context_length + 1, block_size)
        added += max(0, new_blocks - old_blocks)

    return current_allocated + added


class CandidateGenerator:
    """Generate the fixed at-most-12 deterministic BatchPlan candidates."""

    def __init__(self, settings: SchedulerSettings | None = None) -> None:
        self.settings = settings or SchedulerSettings.provisional()

    def generate(self, snapshot: StateSnapshot) -> tuple[BatchPlan, ...]:
        ordered_decode = _rank_decode(snapshot)
        ordered_prefill = _rank_prefill(snapshot)
        recovery_set = set(snapshot.recovery_requests)

        raw_plans: list[tuple[str, BatchPlan]] = []
        profile_order = ("MANDATORY", "URGENT", "ALL")
        cap_order = self.settings.prefill_caps

        for profile in profile_order:
            decode_ids = _bind_decode(profile, ordered_decode, self.settings, recovery_set)
            decode_ids = _trim_decode_to_budget(decode_ids, snapshot, ordered_decode)
            for cap in cap_order:
                prefill_items = _fill_prefill(
                    snapshot, len(decode_ids), cap, ordered_prefill, self.settings
                )
                total_prefill = sum(count for _, count in prefill_items)
                total_decode = len(decode_ids)
                total_sequences = len(prefill_items) + total_decode
                if total_prefill + total_decode > snapshot.token_budget:
                    continue
                if total_sequences > snapshot.sequence_budget:
                    continue
                projected_kv = _projected_kv_blocks(
                    snapshot, prefill_items, decode_ids
                )
                template_id = f"{profile}:cap_{cap}"
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
                            rid in recovery_set
                            or any(
                                r.request_id == rid and (r.mandatory or r.recovery_due)
                                for r in ordered_decode
                            )
                        )
                    ),
                )
                raw_plans.append((template_id, plan))

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
        return tuple(result[: self.settings.maximum_candidates])
