"""Deterministic fixed-fraction and Stock-like BatchPlan generation.

Normal candidates contain every active Decode request and vary only the
Prefill budget. A separate STOCK candidate mirrors the request-selection order
of the locked vLLM Scheduler for the supported text-only runtime path. No
Predictor or live allocator is called while candidates are constructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dpp_scheduler.contracts import (
    BatchPlan,
    DecodeRequest,
    PrefillRequest,
    StateSnapshot,
)
from dpp_scheduler.settings import SchedulerSettings


FRACTION_LABELS: tuple[str, ...] = tuple(
    f"P{index * 10}" for index in range(1, 11)
)
BUDGET_FRACTIONS: tuple[float, ...] = tuple(index / 10 for index in range(1, 11))


def rank_decode_requests(snapshot: StateSnapshot) -> tuple[DecodeRequest, ...]:
    """Return active Decode requests in stable FCFS order."""
    return tuple(
        sorted(
            snapshot.active_decode_requests,
            key=lambda item: (item.arrival_time, item.ordinal, item.request_id),
        )
    )


def rank_prefill_requests(snapshot: StateSnapshot) -> tuple[PrefillRequest, ...]:
    """Running Prefill first, then waiting Prefill in stable FCFS order."""
    return tuple(
        sorted(
            snapshot.waiting_prefill_requests,
            key=lambda item: (
                0 if item.is_running else 1,
                item.ordinal if item.is_running else item.arrival_time,
                item.ordinal,
                item.request_id,
            ),
        )
    )


# Backwards-compatible name for callers that used the former continuation policy.
rank_prefill_continuation = rank_prefill_requests


def highest_bindable_prefill(snapshot: StateSnapshot) -> PrefillRequest | None:
    """Return the first ordered Prefill that can occupy the current seq budget."""
    running_count = sum(
        request.is_running for request in snapshot.waiting_prefill_requests
    )
    waiting_slot_available = (
        len(snapshot.active_decode_requests) + running_count
        < snapshot.sequence_budget
    )
    for request in rank_prefill_requests(snapshot):
        if request.is_running or waiting_slot_available:
            return request
    return None


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
    decode_items = tuple(
        request.request_id for request in rank_decode_requests(snapshot)
    )

    def kv_clamp(request: PrefillRequest, requested: int) -> int:
        """Return the largest token count that fits the current KV projection."""
        low, high = 0, requested
        while low < high:
            midpoint = (low + high + 1) // 2
            tentative = (*items, (request.request_id, midpoint))
            if (
                project_kv_blocks(snapshot, tentative, decode_items)
                <= snapshot.total_kv_blocks
            ):
                low = midpoint
            else:
                high = midpoint - 1
        return low

    for request in order:
        if not request.is_running and admitted >= new_slots:
            continue
        available = min(request.remaining_tokens, budget - consumed)
        available = kv_clamp(request, available)
        if available <= 0:
            continue
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


def _build_batch_plan(
    snapshot: StateSnapshot,
    prefill_items: tuple[tuple[str, int], ...],
    decode_items: tuple[str, ...],
    template_id: str,
) -> BatchPlan:
    return BatchPlan(
        plan_id=f"plan-{template_id}",
        snapshot_hash=snapshot.snapshot_hash,
        template_id=template_id,
        prefill_items=prefill_items,
        decode_items=decode_items,
        total_prefill_tokens=sum(tokens for _, tokens in prefill_items),
        total_decode_tokens=len(decode_items),
        total_sequences=project_sequence_count(snapshot, prefill_items),
        projected_kv_blocks=project_kv_blocks(snapshot, prefill_items, decode_items),
        mandatory_request_ids=(),
    )


def derive_candidate_budgets(
    *, token_budget: int, decode_count: int, total_prefill_backlog: int
) -> tuple[tuple[str, int, float], ...]:
    """Return P10..P100 budgets after floor, clamp, and budget deduplication."""
    maximum = min(
        max(0, int(total_prefill_backlog)),
        max(0, int(token_budget) - int(decode_count)),
    )
    deduped: dict[int, tuple[str, float]] = {}
    for label, fraction in zip(FRACTION_LABELS, BUDGET_FRACTIONS):
        budget = max(0, min(maximum, math.floor(maximum * fraction)))
        deduped.setdefault(budget, (label, fraction))
    return tuple(
        (label, budget, fraction)
        for budget, (label, fraction) in sorted(deduped.items())
    )


def _stock_running_order(
    snapshot: StateSnapshot,
) -> tuple[PrefillRequest | DecodeRequest, ...]:
    running: list[PrefillRequest | DecodeRequest] = [
        item for item in snapshot.waiting_prefill_requests if item.is_running
    ]
    running.extend(snapshot.active_decode_requests)
    return tuple(sorted(running, key=lambda item: (item.ordinal, item.request_id)))


def build_stock_plan(snapshot: StateSnapshot) -> BatchPlan:
    """Build the supported Stock request selection without mutating vLLM state."""
    token_budget = max(0, int(snapshot.token_budget))
    prefill_items: list[tuple[str, int]] = []
    decode_items: list[str] = []

    def fits(
        candidate_prefill: list[tuple[str, int]], candidate_decode: list[str]
    ) -> bool:
        prefill_tuple = tuple(candidate_prefill)
        return (
            project_sequence_count(snapshot, prefill_tuple) <= snapshot.sequence_budget
            and project_kv_blocks(snapshot, prefill_tuple, tuple(candidate_decode))
            <= snapshot.total_kv_blocks
        )

    for request in _stock_running_order(snapshot):
        if token_budget <= 0:
            break
        if isinstance(request, PrefillRequest):
            scheduled = min(request.remaining_tokens, token_budget)
            if scheduled <= 0:
                continue
            tentative = prefill_items + [(request.request_id, scheduled)]
            if not fits(tentative, decode_items):
                continue
            prefill_items = tentative
            token_budget -= scheduled
        else:
            tentative_decode = decode_items + [request.request_id]
            if not fits(prefill_items, tentative_decode):
                continue
            decode_items = tentative_decode
            token_budget -= 1

    waiting = sorted(
        (item for item in snapshot.waiting_prefill_requests if not item.is_running),
        key=lambda item: (item.arrival_time, item.ordinal, item.request_id),
    )
    for request in waiting:
        if token_budget <= 0:
            break
        scheduled = min(request.remaining_tokens, token_budget)
        if scheduled <= 0:
            continue
        tentative = prefill_items + [(request.request_id, scheduled)]
        if not fits(tentative, decode_items):
            break
        prefill_items = tentative
        token_budget -= scheduled

    return _build_batch_plan(
        snapshot,
        tuple(prefill_items),
        tuple(decode_items),
        template_id="STOCK",
    )


@dataclass(frozen=True)
class GeneratorDiagnostics:
    maximum_prefill_budget: int
    raw_candidate_count: int
    deduplicated_candidate_count: int
    candidate_budget_values: tuple[int, ...]
    fraction_budgets: tuple[tuple[str, int], ...]
    stock_prefill_budget: int


class CandidateGenerator:
    """Generate ZERO, P10..P100, and one Stock-like candidate."""

    def __init__(self, settings: SchedulerSettings | None = None) -> None:
        self.settings = settings or SchedulerSettings.provisional()
        self._last_diagnostic: GeneratorDiagnostics | None = None

    @property
    def last_diagnostic(self) -> GeneratorDiagnostics | None:
        return self._last_diagnostic

    def build_stock_plan(self, snapshot: StateSnapshot) -> BatchPlan:
        return build_stock_plan(snapshot)

    def generate(self, snapshot: StateSnapshot) -> tuple[BatchPlan, ...]:
        if not snapshot.waiting_prefill_requests and not snapshot.active_decode_requests:
            self._last_diagnostic = None
            return ()

        decode_ids = tuple(item.request_id for item in rank_decode_requests(snapshot))
        prefill_order = rank_prefill_requests(snapshot)
        backlog = sum(item.remaining_tokens for item in prefill_order)
        maximum = min(backlog, max(0, snapshot.token_budget - len(decode_ids)))
        plans: list[BatchPlan] = [_build_batch_plan(snapshot, (), decode_ids, "ZERO")]
        fraction_budgets: list[tuple[str, int]] = []
        for label, budget, _fraction in derive_candidate_budgets(
            token_budget=snapshot.token_budget,
            decode_count=len(decode_ids),
            total_prefill_backlog=backlog,
        ):
            if budget <= 0:
                continue
            fraction_budgets.append((label, budget))
            prefill_items = _fill_prefill(snapshot, budget, prefill_order, self.settings)
            plans.append(_build_batch_plan(snapshot, prefill_items, decode_ids, label))

        stock_plan = self.build_stock_plan(snapshot)
        plans.append(stock_plan)
        raw_count = len(plans)

        seen: set[tuple[tuple[tuple[str, int], ...], tuple[str, ...]]] = set()
        deduplicated: list[BatchPlan] = []
        # Preserve STOCK identity if its material work duplicates a fraction plan.
        for plan in (stock_plan, *plans[:-1]):
            key = (plan.prefill_items, plan.decode_items)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(plan)
        if len(deduplicated) > self.settings.maximum_seed_candidates:
            raise RuntimeError(
                f"Candidate Generator produced {len(deduplicated)} candidates "
                f"exceeding maximum_seed_candidates={self.settings.maximum_seed_candidates}"
            )

        self._last_diagnostic = GeneratorDiagnostics(
            maximum_prefill_budget=maximum,
            raw_candidate_count=raw_count,
            deduplicated_candidate_count=len(deduplicated),
            candidate_budget_values=tuple(
                sorted({plan.total_prefill_tokens for plan in plans})
            ),
            fraction_budgets=tuple(fraction_budgets),
            stock_prefill_budget=stock_plan.total_prefill_tokens,
        )
        return tuple(deduplicated)
