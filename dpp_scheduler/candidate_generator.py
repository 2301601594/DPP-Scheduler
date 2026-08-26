"""Slack-centered v3 Candidate Generator with multiplier neighborhood.

The Generator consumes a :class:`BudgetResolution` from the injected
:class:`BudgetResolver` (Predictor-inversion-based) and emits up to 16 raw
BatchPlans: 1 ZERO plan + at most ``5 multipliers × 3 priority policies``
(M050/M075/M100/M125/M150 × URGENCY/COMPLETION_AWARE/CONTINUATION). Plans are
canonical-deduplicated by ``(prefill_items, decode_items)``.

This module never imports or calls Predictor, Safe-Set, or Selector directly.
The Predictor dependency is injected into the resolver at the Adapter boundary,
preserving the G2 contract test that introspects this module's source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dpp_scheduler.budget_resolver import (
    BudgetResolution,
    BudgetResolver,
    NullBudgetResolver,
    RESOLUTION_NO_DECODE_USE_MAX,
)
from dpp_scheduler.contracts import BatchPlan, PrefillRequest, StateSnapshot
from dpp_scheduler.settings import SchedulerSettings


# ---------------------------------------------------------------------------
# Decode ordering
# ---------------------------------------------------------------------------


def rank_decode_requests(snapshot: StateSnapshot) -> tuple:
    """Stable EDF-friendly arrival-time ordering for active Decode requests."""
    return tuple(
        sorted(
            snapshot.active_decode_requests,
            key=lambda item: (item.arrival_time, item.ordinal, item.request_id),
        )
    )


# ---------------------------------------------------------------------------
# Prefill ordering
# ---------------------------------------------------------------------------


def _urgency_score(request: PrefillRequest, t_now: float) -> float:
    remaining = max(0, int(request.remaining_tokens))
    deadline = request.ttft_deadline
    if deadline is None:
        return float(remaining)
    slack = deadline - t_now
    if slack <= 0:
        return float(remaining) * 1.0e9
    return float(remaining) / slack


def _tier_of(score: float) -> int:
    if score >= 1.0:
        return 0
    if score >= 0.5:
        return 1
    return 2


def rank_prefill_continuation(snapshot: StateSnapshot) -> tuple[PrefillRequest, ...]:
    """Continuation / running-first ordering; baseline priority policy."""
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


def rank_prefill_urgency(snapshot: StateSnapshot) -> tuple[PrefillRequest, ...]:
    """Urgency ordering: highest Prefill completion-rate first."""
    t_now = float(snapshot.timestamp)
    return tuple(
        sorted(
            snapshot.waiting_prefill_requests,
            key=lambda item: (
                -_urgency_score(item, t_now),
                item.arrival_time,
                item.ordinal,
                item.request_id,
            ),
        )
    )


def rank_prefill_completion_aware(
    snapshot: StateSnapshot,
) -> tuple[PrefillRequest, ...]:
    """Three-tier urgency, then smallest-remaining-first inside each tier."""
    t_now = float(snapshot.timestamp)
    scored = [
        (
            _tier_of(_urgency_score(item, t_now)),
            0 if item.is_running else 1,
            item.remaining_tokens,
            item.arrival_time,
            item.ordinal,
            item.request_id,
            item,
        )
        for item in snapshot.waiting_prefill_requests
    ]
    scored.sort(
        key=lambda entry: (
            entry[0],
            entry[1],
            entry[2],
            entry[3],
            entry[4],
            entry[5],
        )
    )
    return tuple(entry[6] for entry in scored)


def build_prefill_orders(
    snapshot: StateSnapshot,
) -> tuple[tuple[str, tuple[PrefillRequest, ...]], ...]:
    """Return the three frozen priority policies in deterministic order."""
    return (
        ("URGENCY", rank_prefill_urgency(snapshot)),
        ("COMPLETION_AWARE", rank_prefill_completion_aware(snapshot)),
        ("CONTINUATION", rank_prefill_continuation(snapshot)),
    )


# Backwards-compatible alias: the old name remains importable.
rank_prefill_requests = rank_prefill_continuation


# ---------------------------------------------------------------------------
# Resource clamps
# ---------------------------------------------------------------------------


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
    return _highest_bindable_prefill(snapshot, rank_prefill_continuation(snapshot))


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


# ---------------------------------------------------------------------------
# Budget neighborhood derivation
# ---------------------------------------------------------------------------


MULTIPLIER_LABELS: tuple[str, ...] = ("M050", "M075", "M100", "M125", "M150")
BUDGET_MULTIPLIERS: tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50)


def derive_candidate_budgets(
    base_budget: int,
    *,
    token_budget: int,
    decode_count: int,
    total_prefill_backlog: int,
) -> tuple[tuple[str, int, float], ...]:
    """Multiply ``base_budget`` by the frozen multipliers, floor, clamp, dedup.

    Returns a sorted tuple of ``(multiplier_label, clamped_budget, multiplier)``
    triples. The tuple length is at most ``len(BUDGET_MULTIPLIERS)`` and at
    least one (the entry for multiplier 0.5 floored to zero is preserved).
    """
    if base_budget <= 0:
        return ()
    max_prefill = min(
        max(0, total_prefill_backlog),
        max(0, int(token_budget) - int(decode_count)),
    )
    raw: list[tuple[str, int, float]] = []
    for multiplier, label in zip(BUDGET_MULTIPLIERS, MULTIPLIER_LABELS):
        raw_budget = math.floor(multiplier * float(base_budget))
        clamped = max(0, min(max_prefill, int(raw_budget)))
        raw.append((label, clamped, float(multiplier)))
    deduped: dict[int, tuple[str, float]] = {}
    for label, budget, multiplier in raw:
        deduped.setdefault(budget, (label, multiplier))
    return tuple(
        (label, budget, multiplier)
        for budget, (label, multiplier) in sorted(deduped.items())
    )


# ---------------------------------------------------------------------------
# Candidate Generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GeneratorDiagnostics:
    """Per-frame diagnostic summary exposed via ``CandidateGenerator.last_diagnostic``."""

    resolution: BudgetResolution
    raw_candidate_count: int
    deduplicated_candidate_count: int
    candidate_budget_values: tuple[int, ...]


class CandidateGenerator:
    """Generate up to 16 raw candidates: 1 ZERO + 5 multipliers × 3 policies.

    The Generator depends on a :class:`BudgetResolver` (Predictor inversion) for
    the slack-centered base budget ``P``. With the default ``NullBudgetResolver``
    only the ZERO candidate is emitted, preserving the G2 contract test that
    asserts no Predictor / Safe-Set coupling.
    """

    def __init__(
        self,
        settings: SchedulerSettings | None = None,
        *,
        budget_resolver: BudgetResolver | None = None,
    ) -> None:
        self.settings = settings or SchedulerSettings.provisional()
        self.budget_resolver: BudgetResolver = budget_resolver or NullBudgetResolver()
        self._last_diagnostic: _GeneratorDiagnostics | None = None

    @property
    def last_diagnostic(self) -> _GeneratorDiagnostics | None:
        """Diagnostic summary from the most recent :meth:`generate` call.

        Consumed by the Adapter's per-frame diagnostic logger. ``None`` until
        the first ``generate`` invocation.
        """
        return self._last_diagnostic

    @staticmethod
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
            projected_kv_blocks=project_kv_blocks(
                snapshot, prefill_items, decode_items
            ),
            mandatory_request_ids=(),
        )

    def generate(self, snapshot: StateSnapshot) -> tuple[BatchPlan, ...]:
        if not snapshot.waiting_prefill_requests and not snapshot.active_decode_requests:
            self._last_diagnostic = None
            return ()
        decode_ids = tuple(
            item.request_id for item in rank_decode_requests(snapshot)
        )
        backlog = sum(
            item.remaining_tokens for item in snapshot.waiting_prefill_requests
        )

        # Always-emitted ZERO candidate.
        zero_plan = self._build_batch_plan(
            snapshot,
            prefill_items=(),
            decode_items=decode_ids,
            template_id="ALL_DECODE:ZERO",
        )
        plans: list[BatchPlan] = [zero_plan]
        raw_count = 1

        resolution = self.budget_resolver.resolve(snapshot)
        if resolution.base_prefill_budget > 0:
            budgets = derive_candidate_budgets(
                resolution.base_prefill_budget,
                token_budget=snapshot.token_budget,
                decode_count=len(decode_ids),
                total_prefill_backlog=backlog,
            )
            orders = build_prefill_orders(snapshot)
            for multiplier_label, budget, _multiplier in budgets:
                if budget <= 0:
                    # Zero-budget Mixed candidate would degenerate to ZERO;
                    # skip and rely on the canonical ZERO plan above.
                    continue
                for policy_name, order in orders:
                    prefill_items = _fill_prefill(
                        snapshot, budget, order, self.settings
                    )
                    template_id = (
                        f"ALL_DECODE:SLACK_BUDGET:{multiplier_label}:{policy_name}"
                    )
                    plans.append(
                        self._build_batch_plan(
                            snapshot,
                            prefill_items=prefill_items,
                            decode_items=decode_ids,
                            template_id=template_id,
                        )
                    )
                    raw_count += 1

        # Canonical dedup on (prefill_items, decode_items).
        seen: dict[tuple[tuple[tuple[str, int], ...], tuple[str, ...]], BatchPlan] = {}
        deduplicated: list[BatchPlan] = []
        for plan in plans:
            key = (plan.prefill_items, plan.decode_items)
            if key in seen:
                continue
            seen[key] = plan
            deduplicated.append(plan)

        if len(deduplicated) > self.settings.maximum_seed_candidates:
            raise RuntimeError(
                f"Candidate Generator produced {len(deduplicated)} candidates "
                f"exceeding maximum_seed_candidates={self.settings.maximum_seed_candidates}"
            )

        candidate_budget_values = _budgets_from_plans(plans)
        self._last_diagnostic = _GeneratorDiagnostics(
            resolution=resolution,
            raw_candidate_count=raw_count,
            deduplicated_candidate_count=len(deduplicated),
            candidate_budget_values=candidate_budget_values,
        )
        return tuple(deduplicated)


def _budgets_from_plans(plans: list[BatchPlan]) -> tuple[int, ...]:
    """Return the sorted unique budget values actually emitted across plans."""
    seen: set[int] = set()
    for plan in plans:
        seen.add(int(plan.total_prefill_tokens))
    return tuple(sorted(seen))
