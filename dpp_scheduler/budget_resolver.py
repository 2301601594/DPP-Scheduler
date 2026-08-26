"""Slack-centered Prefill budget resolution via Predictor inversion.

The Candidate Generator calls :meth:`BudgetResolver.resolve` with the current
``StateSnapshot``; the resolver computes a base Prefill budget ``P`` by inverting
the validated duration Predictor under the current Decode TBT slack.

This module is pure and deterministic: it never mutates the snapshot, the
Predictor state, or any plan; it only reads ``predictor.predict`` outputs.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from dpp_scheduler.contracts import (
    BatchPlan,
    PrefillRequest,
    StateSnapshot,
)
from dpp_scheduler.predictor import DurationPredictor
from dpp_scheduler.settings import SchedulerSettings


RESOLUTION_INVERTED_OK = "INVERTED_OK"
RESOLUTION_INVERTED_OOD = "INVERTED_OOD"
RESOLUTION_NO_FEASIBLE_BUDGET = "NO_FEASIBLE_BUDGET"
RESOLUTION_NO_DECODE_USE_MAX = "NO_DECODE_USE_MAX"
RESOLUTION_NO_PREFILL_BACKLOG = "NO_PREFILL_BACKLOG"
RESOLUTION_PREDICTOR_INVALID = "PREDICTOR_INVALID"
RESOLUTION_NO_DECODE_NO_BACKLOG = "NO_DECODE_NO_BACKLOG"

DEFAULT_INVERSION_BUDGET_GRID: tuple[int, ...] = (
    0,
    64,
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
)
DEFAULT_SAFETY_MARGIN_SECONDS: float = 0.020
URGENCY_FLOOR_SECONDS: float = 1.0e-6


@dataclass(frozen=True)
class BudgetResolution:
    """Result of one Predictor-inversion budget resolution."""

    base_prefill_budget: int
    target_duration_seconds: float | None
    resolution_status: str
    requested_grid_budgets: tuple[int, ...] = ()
    feasible_grid_budgets: tuple[int, ...] = ()
    predictor_in_support_ratio: float | None = None


class BudgetResolver(ABC):
    """Abstract slack → Prefill budget resolver."""

    @abstractmethod
    def resolve(self, snapshot: StateSnapshot) -> BudgetResolution:
        raise NotImplementedError


class NullBudgetResolver(BudgetResolver):
    """G2 default: always returns ``base_prefill_budget = 0``.

    Used when no Predictor dependency is injected (Controller default tests and
    G2 fixtures). The Candidate Generator consumes ``base_prefill_budget = 0``
    as the signal to emit only the ZERO candidate.
    """

    def resolve(self, snapshot: StateSnapshot) -> BudgetResolution:
        del snapshot
        return BudgetResolution(
            base_prefill_budget=0,
            target_duration_seconds=None,
            resolution_status=RESOLUTION_NO_FEASIBLE_BUDGET,
            requested_grid_budgets=(),
            feasible_grid_budgets=(),
            predictor_in_support_ratio=None,
        )


def _continuation_order(
    snapshot: StateSnapshot,
) -> tuple[PrefillRequest, ...]:
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


def _shadow_fill(
    snapshot: StateSnapshot,
    budget: int,
    order: tuple[PrefillRequest, ...],
    settings: SchedulerSettings,
) -> tuple[tuple[str, int], ...]:
    """Minimal shadow Prefill filler used only for Predictor inversion sweeps.

    Mirrors ``candidate_generator._fill_prefill`` semantics but never touches
    the real block manager. Iteration constraints (running prefill precedence,
    minimum chunk, partial-prefill policy) are honored so that the inversion
    sweep produces realistic duration predictions.
    """
    if budget <= 0:
        return ()
    running = sum(item.is_running for item in snapshot.waiting_prefill_requests)
    new_slots = max(
        0,
        snapshot.sequence_budget
        - len(snapshot.active_decode_requests)
        - running,
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


def _build_shadow_plan(
    snapshot: StateSnapshot,
    prefill_items: tuple[tuple[str, int], ...],
    decode_items: tuple[str, ...],
    template_label: str,
) -> BatchPlan:
    return BatchPlan(
        plan_id=f"shadow-{template_label}",
        snapshot_hash=snapshot.snapshot_hash,
        template_id=f"SHADOW:{template_label}",
        prefill_items=prefill_items,
        decode_items=decode_items,
        total_prefill_tokens=sum(tokens for _, tokens in prefill_items),
        total_decode_tokens=len(decode_items),
        total_sequences=len(decode_items) + len(prefill_items),
        projected_kv_blocks=0,
        mandatory_request_ids=(),
    )


class RidgeBudgetResolver(BudgetResolver):
    """Slack-centered base budget via Predictor inversion.

    Algorithm (mirrors ``benchmarks/time_to_budget_validation.py:641-677``):

    1. Collect next-token TBT deadlines of active Decode requests that are
       still live (``d_j^{next} > t_k``). If none exist or no Decode requests
       are active, fall back to resource-cap ``P`` (§11 of the plan).
    2. Compute ``s_k^{min} = min_j(d_j^{next} - t_k)`` and
       ``T_k^{target} = max(0.0, s_k^{min} - safety_margin)``.
    3. Sweep ``budget_grid``, build a shadow BatchPlan per budget using the
       CONTINUATION order, and call ``predictor.predict(snapshot, plans)``.
    4. Pick ``P = max(b for b, p in zip(grid, predictions)
                  if p.expected_duration is finite and
                  p.expected_duration > 0 and
                  p.expected_duration <= T_k^{target})``.
    5. Report ``INVERTED_OK`` when all feasible predictions are in-support;
       otherwise ``INVERTED_OOD``. ``NO_FEASIBLE_BUDGET`` when the sweep
       yields no feasible budget.
    """

    def __init__(
        self,
        *,
        predictor: DurationPredictor,
        settings: SchedulerSettings,
        safety_margin_seconds: float = DEFAULT_SAFETY_MARGIN_SECONDS,
        budget_grid: tuple[int, ...] = DEFAULT_INVERSION_BUDGET_GRID,
    ) -> None:
        if predictor is None:
            raise ValueError("RidgeBudgetResolver requires a non-None Predictor")
        if (
            isinstance(safety_margin_seconds, bool)
            or not isinstance(safety_margin_seconds, (int, float))
            or not math.isfinite(safety_margin_seconds)
            or safety_margin_seconds <= 0
        ):
            raise ValueError("safety_margin_seconds must be finite and positive")
        if (
            isinstance(budget_grid, bool)
            or not isinstance(budget_grid, tuple)
            or not all(isinstance(item, int) and item >= 0 for item in budget_grid)
            or list(budget_grid) != sorted(budget_grid)
        ):
            raise ValueError(
                "budget_grid must be a non-decreasing tuple of non-negative ints"
            )
        self._predictor = predictor
        self._settings = settings
        self._safety_margin_seconds = float(safety_margin_seconds)
        self._budget_grid = tuple(int(item) for item in budget_grid)

    @property
    def predictor(self) -> DurationPredictor:
        return self._predictor

    @property
    def safety_margin_seconds(self) -> float:
        return self._safety_margin_seconds

    @property
    def budget_grid(self) -> tuple[int, ...]:
        return self._budget_grid

    def resolve(self, snapshot: StateSnapshot) -> BudgetResolution:
        backlog = sum(
            item.remaining_tokens for item in snapshot.waiting_prefill_requests
        )
        decode_count = len(snapshot.active_decode_requests)

        if decode_count == 0:
            if backlog <= 0:
                return BudgetResolution(
                    base_prefill_budget=0,
                    target_duration_seconds=None,
                    resolution_status=RESOLUTION_NO_DECODE_NO_BACKLOG,
                )
            return BudgetResolution(
                base_prefill_budget=min(
                    backlog, max(0, snapshot.token_budget - decode_count)
                ),
                target_duration_seconds=None,
                resolution_status=RESOLUTION_NO_DECODE_USE_MAX,
            )

        live_deadlines: list[float] = []
        for request in snapshot.active_decode_requests:
            deadline = request.tbt_deadline
            if deadline is None:
                continue
            slack = deadline - snapshot.timestamp
            if slack > 0:
                live_deadlines.append(slack)

        if not live_deadlines:
            return BudgetResolution(
                base_prefill_budget=min(
                    backlog, max(0, snapshot.token_budget - decode_count)
                ),
                target_duration_seconds=None,
                resolution_status=RESOLUTION_NO_DECODE_USE_MAX,
            )

        s_min = min(live_deadlines)
        target_duration = max(0.0, s_min - self._safety_margin_seconds)
        return self._invert(snapshot, target_duration, backlog)

    def _invert(
        self,
        snapshot: StateSnapshot,
        target_duration: float,
        backlog: int,
    ) -> BudgetResolution:
        decode_ids = tuple(item.request_id for item in snapshot.active_decode_requests)
        order = _continuation_order(snapshot)

        requested: list[int] = []
        feasible: list[tuple[int, float, bool]] = []
        plans: list[BatchPlan] = []
        plan_budgets: list[int] = []

        for budget in self._budget_grid:
            prefill_items = _shadow_fill(
                snapshot, budget, order, self._settings
            )
            plan = _build_shadow_plan(
                snapshot,
                prefill_items,
                decode_ids,
                f"INVERSION:requested_{budget}",
            )
            plans.append(plan)
            plan_budgets.append(budget)
            requested.append(budget)

        if not plans:
            return BudgetResolution(
                base_prefill_budget=0,
                target_duration_seconds=target_duration,
                resolution_status=RESOLUTION_NO_FEASIBLE_BUDGET,
                requested_grid_budgets=tuple(requested),
                feasible_grid_budgets=(),
                predictor_in_support_ratio=None,
            )

        predictions = self._predictor.predict(snapshot, plans)
        if len(predictions) != len(plans):
            return BudgetResolution(
                base_prefill_budget=0,
                target_duration_seconds=target_duration,
                resolution_status=RESOLUTION_PREDICTOR_INVALID,
                requested_grid_budgets=tuple(requested),
                feasible_grid_budgets=(),
                predictor_in_support_ratio=None,
            )

        chosen_budget = 0
        chosen_status = RESOLUTION_NO_FEASIBLE_BUDGET
        in_support_count = 0
        any_feasible = False

        for budget, prediction in zip(plan_budgets, predictions):
            if prediction.in_support:
                in_support_count += 1
            duration = prediction.expected_duration
            if (
                duration is None
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
                or duration > target_duration
            ):
                continue
            any_feasible = True
            if budget >= chosen_budget:
                chosen_budget = budget

        if not any_feasible:
            return BudgetResolution(
                base_prefill_budget=0,
                target_duration_seconds=target_duration,
                resolution_status=RESOLUTION_NO_FEASIBLE_BUDGET,
                requested_grid_budgets=tuple(requested),
                feasible_grid_budgets=(),
                predictor_in_support_ratio=_in_support_ratio(
                    in_support_count, len(predictions)
                ),
            )

        feasible_budgets = tuple(
            budget
            for budget, prediction in zip(plan_budgets, predictions)
            if _is_feasible(prediction.expected_duration, target_duration)
        )
        if in_support_count == len(predictions):
            chosen_status = RESOLUTION_INVERTED_OK
        else:
            chosen_status = RESOLUTION_INVERTED_OOD

        # The resolver returns the raw largest-feasible budget; resource
        # clamping against backlog / token-budget / sequence-budget is the
        # Candidate Generator's responsibility (via derive_candidate_budgets).

        return BudgetResolution(
            base_prefill_budget=int(chosen_budget),
            target_duration_seconds=target_duration,
            resolution_status=chosen_status,
            requested_grid_budgets=tuple(requested),
            feasible_grid_budgets=feasible_budgets,
            predictor_in_support_ratio=_in_support_ratio(
                in_support_count, len(predictions)
            ),
        )


def _is_feasible(duration: float | None, target_duration: float) -> bool:
    if duration is None:
        return False
    if not isinstance(duration, (int, float)):
        return False
    if not math.isfinite(duration) or duration <= 0:
        return False
    return duration <= target_duration


def _in_support_ratio(in_support: int, total: int) -> float | None:
    if total <= 0:
        return None
    return float(in_support) / float(total)


def collect_resolution_summary(
    resolutions: Iterable[BudgetResolution],
) -> dict[str, float | int | None]:
    """Aggregate diagnostic summary over a sequence of resolutions."""

    values = list(resolutions)
    if not values:
        return {
            "frames": 0,
            "inverted_ok": 0,
            "inverted_ood": 0,
            "no_feasible": 0,
            "no_decode_use_max": 0,
            "no_decode_no_backlog": 0,
            "predictor_invalid": 0,
            "p_min": None,
            "p_p50": None,
            "p_p95": None,
            "p_max": None,
            "support_ratio_mean": None,
        }

    counts: dict[str, int] = {
        RESOLUTION_INVERTED_OK: 0,
        RESOLUTION_INVERTED_OOD: 0,
        RESOLUTION_NO_FEASIBLE_BUDGET: 0,
        RESOLUTION_NO_DECODE_USE_MAX: 0,
        RESOLUTION_NO_DECODE_NO_BACKLOG: 0,
        RESOLUTION_PREDICTOR_INVALID: 0,
    }
    bases: list[int] = []
    supports: list[float] = []
    for value in values:
        counts[value.resolution_status] = counts.get(value.resolution_status, 0) + 1
        bases.append(int(value.base_prefill_budget))
        if value.predictor_in_support_ratio is not None:
            supports.append(value.predictor_in_support_ratio)

    bases_sorted = sorted(bases)
    p_min = bases_sorted[0]
    p_max = bases_sorted[-1]
    p_p50 = _percentile(bases_sorted, 0.50)
    p_p95 = _percentile(bases_sorted, 0.95)

    support_mean: float | None
    if supports:
        support_mean = sum(supports) / len(supports)
    else:
        support_mean = None

    return {
        "frames": len(values),
        "inverted_ok": counts[RESOLUTION_INVERTED_OK],
        "inverted_ood": counts[RESOLUTION_INVERTED_OOD],
        "no_feasible": counts[RESOLUTION_NO_FEASIBLE_BUDGET],
        "no_decode_use_max": counts[RESOLUTION_NO_DECODE_USE_MAX],
        "no_decode_no_backlog": counts[RESOLUTION_NO_DECODE_NO_BACKLOG],
        "predictor_invalid": counts[RESOLUTION_PREDICTOR_INVALID],
        "p_min": p_min,
        "p_p50": p_p50,
        "p_p95": p_p95,
        "p_max": p_max,
        "support_ratio_mean": support_mean,
    }


def _percentile(ordered: list[int], quantile: float) -> int | None:
    if not ordered:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(0, index)]
