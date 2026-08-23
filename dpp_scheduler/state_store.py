"""In-memory control state and actual returned-token obligation ledger."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dpp_scheduler.contracts import (
    ControlState,
    Obligation,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.settings import DPPSettings


@dataclass
class InMemoryStateStore:
    """Control state updated only from actual queue and ledger observations."""

    current: ControlState = ControlState(
        snapshot_hash="unbound",
        prefill_backlog=0,
        ttft_debt=0.0,
        tbt_debt=0.0,
    )
    settings: DPPSettings | None = None
    _applied_event_ids: set[str] = field(default_factory=set, init=False)
    _pending_prefill_arrivals: int = field(default=0, init=False)

    def get(self) -> ControlState:
        return self.current

    def set(self, state: ControlState) -> None:
        self.current = state

    def record_prefill_arrival(self, prompt_tokens: int) -> None:
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
        ):
            raise ValueError("prompt_tokens must be a positive integer")
        self._pending_prefill_arrivals += prompt_tokens

    def bind_snapshot(self, snapshot: StateSnapshot) -> ControlState:
        """Bind debts to a round and reconcile QP from actual visible state."""
        observed_backlog = sum(
            request.remaining_tokens
            for request in snapshot.waiting_prefill_requests
        )
        expected_backlog = (
            self.current.prefill_backlog + self._pending_prefill_arrivals
        )
        # Normal rounds follow Q-mu+A exactly. An actual Snapshot remains the
        # authority after cancellation/preemption changes visible Prefill work.
        bound_backlog = (
            expected_backlog
            if expected_backlog == observed_backlog
            else observed_backlog
        )
        self._pending_prefill_arrivals = 0
        self.current = ControlState(
            snapshot_hash=snapshot.snapshot_hash,
            prefill_backlog=bound_backlog,
            ttft_debt=self.current.ttft_debt,
            tbt_debt=self.current.tbt_debt,
        )
        return self.current

    def update_from_actual(
        self,
        *,
        snapshot_hash: str,
        actual_prefill_tokens: int,
        ledger_updates: tuple[LedgerUpdate, ...] = (),
    ) -> ControlState:
        if self.settings is None and ledger_updates:
            raise RuntimeError("DPP settings are required for debt updates")
        event_ids = [update.event_id for update in ledger_updates]
        if len(event_ids) != len(set(event_ids)):
            raise DuplicateLedgerEvent("duplicate event in control-state update")
        duplicates = self._applied_event_ids.intersection(event_ids)
        if duplicates:
            raise DuplicateLedgerEvent(
                f"control-state event already applied: {min(duplicates)}"
            )
        self.current = advance_control_state(
            self.current,
            snapshot_hash=snapshot_hash,
            actual_prefill_tokens=actual_prefill_tokens,
            ledger_updates=ledger_updates,
            settings=self.settings,
        )
        self._applied_event_ids.update(event_ids)
        return self.current


class DuplicateLedgerEvent(ValueError):
    """Raised when the same actual output callback is delivered twice."""


@dataclass(frozen=True)
class LedgerUpdate:
    """Actual one-event outcomes used by the later control-state update."""

    event_id: str
    request_id: str
    ttft_success: int = 0
    ttft_miss: int = 0
    tbt_success: int = 0
    tbt_miss: int = 0
    cancelled_obligations: int = 0
    terminal_reason: str | None = None


def advance_control_state(
    state: ControlState,
    *,
    snapshot_hash: str,
    actual_prefill_tokens: int,
    ledger_updates: tuple[LedgerUpdate, ...],
    settings: DPPSettings | None,
) -> ControlState:
    """Apply the frozen actual-feedback equations as a replayable pure step."""
    validate_snapshot_hash(state.snapshot_hash, snapshot_hash)
    if (
        isinstance(actual_prefill_tokens, bool)
        or not isinstance(actual_prefill_tokens, int)
        or actual_prefill_tokens < 0
    ):
        raise ValueError("actual_prefill_tokens must be a non-negative integer")

    totals = {
        "ttft_success": 0,
        "ttft_miss": 0,
        "tbt_success": 0,
        "tbt_miss": 0,
    }
    for update in ledger_updates:
        if not update.event_id:
            raise ValueError("LedgerUpdate event_id must be non-empty")
        for key in totals:
            value = getattr(update, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"LedgerUpdate {key} must be non-negative integer")
            totals[key] += value

    if ledger_updates and settings is None:
        raise RuntimeError("DPP settings are required for debt updates")
    epsilon_ttft = settings.epsilon_ttft if settings is not None else 0.0
    epsilon_tbt = settings.epsilon_tbt if settings is not None else 0.0
    next_ttft = max(
        0.0,
        state.ttft_debt
        + (1.0 - epsilon_ttft) * totals["ttft_miss"]
        - epsilon_ttft * totals["ttft_success"],
    )
    next_tbt = max(
        0.0,
        state.tbt_debt
        + (1.0 - epsilon_tbt) * totals["tbt_miss"]
        - epsilon_tbt * totals["tbt_success"],
    )
    next_prefill = max(0, state.prefill_backlog - actual_prefill_tokens)
    maximum = settings.maximum_numeric if settings is not None else float("inf")
    if (
        not math.isfinite(next_ttft)
        or not math.isfinite(next_tbt)
        or next_ttft > maximum
        or next_tbt > maximum
        or next_prefill > maximum
    ):
        raise ValueError("actual feedback produced out-of-range ControlState")
    return ControlState(
        snapshot_hash=snapshot_hash,
        prefill_backlog=next_prefill,
        ttft_debt=next_ttft,
        tbt_debt=next_tbt,
    )


@dataclass(frozen=True)
class LedgerRequestView:
    """Snapshot-facing deadline and Recovery state for one live request."""

    ttft_deadline: float | None
    tbt_deadline: float | None
    recovery: bool
    recovery_due: bool
    recovery_first_miss_time: float | None


@dataclass
class _RequestLedger:
    request_id: str
    arrival_time: float
    ttft: Obligation | None
    tbt: Obligation | None = None
    tbt_sequence: int = 0
    first_token_returned: bool = False
    recovery_first_miss_time: float | None = None


@dataclass
class ObligationLedger:
    """Create and settle TTFT/TBT obligations from actual output events.

    The ledger never reads an eventual or remaining output length. A token
    event settles the currently active obligation exactly once. A nonterminal
    token creates the next TBT obligation from its actual return timestamp;
    any terminal event creates no next obligation.
    """

    ttft_slo_seconds: float
    tbt_slo_seconds: float
    recovery_age_threshold_seconds: float | None = None
    _requests: dict[str, _RequestLedger] = field(default_factory=dict, init=False)
    _seen_event_ids: set[str] = field(default_factory=set, init=False)
    _settled_obligation_ids: set[str] = field(default_factory=set, init=False)
    _terminal_counts: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("ttft_slo_seconds", self.ttft_slo_seconds),
            ("tbt_slo_seconds", self.tbt_slo_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        threshold = self.recovery_age_threshold_seconds
        if threshold is not None and (
            not math.isfinite(threshold) or threshold < 0
        ):
            raise ValueError(
                "recovery_age_threshold_seconds must be finite and non-negative"
            )

    def register_request(self, request_id: str, arrival_time: float) -> None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if not math.isfinite(arrival_time):
            raise ValueError("arrival_time must be finite")
        if request_id in self._requests:
            raise ValueError(f"request is already registered: {request_id}")
        obligation = Obligation(
            obligation_id=f"{request_id}:TTFT:0",
            request_id=request_id,
            kind="TTFT",
            deadline=arrival_time + self.ttft_slo_seconds,
            created_at=arrival_time,
        )
        self._requests[request_id] = _RequestLedger(
            request_id=request_id,
            arrival_time=arrival_time,
            ttft=obligation,
        )

    def has_request(self, request_id: str) -> bool:
        return request_id in self._requests

    def _settle(
        self, obligation: Obligation, returned_at: float
    ) -> tuple[int, int]:
        if obligation.obligation_id in self._settled_obligation_ids:
            raise DuplicateLedgerEvent(
                f"obligation already settled: {obligation.obligation_id}"
            )
        self._settled_obligation_ids.add(obligation.obligation_id)
        return (1, 0) if returned_at <= obligation.deadline else (0, 1)

    def observe_output(
        self,
        *,
        event_id: str,
        request_id: str,
        returned_at: float,
        token_count: int,
        terminal_reason: str | None,
    ) -> LedgerUpdate:
        """Apply one Adapter-observed output event exactly once."""
        if event_id in self._seen_event_ids:
            raise DuplicateLedgerEvent(f"duplicate output event: {event_id}")
        if not event_id:
            raise ValueError("event_id must be non-empty")
        if not math.isfinite(returned_at):
            raise ValueError("returned_at must be finite")
        if isinstance(token_count, bool) or token_count not in (0, 1):
            raise ValueError("version 1 output events must contain zero or one token")
        if token_count == 0 and terminal_reason is None:
            raise ValueError("an output event must contain a token or terminal reason")
        request = self._requests.get(request_id)
        if request is None:
            raise ValueError(f"unknown ledger request: {request_id}")

        self._seen_event_ids.add(event_id)
        ttft_success = ttft_miss = tbt_success = tbt_miss = cancelled = 0

        if token_count == 1:
            if request.ttft is not None:
                ttft_success, ttft_miss = self._settle(request.ttft, returned_at)
                request.ttft = None
                request.first_token_returned = True
            elif request.tbt is not None:
                tbt_success, tbt_miss = self._settle(request.tbt, returned_at)
                request.tbt = None
            else:
                raise ValueError(
                    f"request has no active token obligation: {request_id}"
                )
            request.recovery_first_miss_time = None

        if terminal_reason is None:
            if token_count != 1:
                raise ValueError("a nonterminal output must return one token")
            request.tbt_sequence += 1
            request.tbt = Obligation(
                obligation_id=f"{request_id}:TBT:{request.tbt_sequence}",
                request_id=request_id,
                kind="TBT",
                deadline=returned_at + self.tbt_slo_seconds,
                created_at=returned_at,
            )
        else:
            # A terminal event without a token cancels, rather than fabricates,
            # a TTFT/TBT outcome. Normal EOS carries the returned token.
            for obligation in (request.ttft, request.tbt):
                if obligation is not None:
                    if obligation.obligation_id in self._settled_obligation_ids:
                        raise DuplicateLedgerEvent(
                            f"obligation already settled: {obligation.obligation_id}"
                        )
                    self._settled_obligation_ids.add(obligation.obligation_id)
                    cancelled += 1
            request.ttft = None
            request.tbt = None
            self._terminal_counts[terminal_reason] = (
                self._terminal_counts.get(terminal_reason, 0) + 1
            )
            del self._requests[request_id]

        return LedgerUpdate(
            event_id=event_id,
            request_id=request_id,
            ttft_success=ttft_success,
            ttft_miss=ttft_miss,
            tbt_success=tbt_success,
            tbt_miss=tbt_miss,
            cancelled_obligations=cancelled,
            terminal_reason=terminal_reason,
        )

    def request_view(self, request_id: str, now: float) -> LedgerRequestView:
        request = self._requests.get(request_id)
        if request is None:
            raise ValueError(f"unknown ledger request: {request_id}")
        if not math.isfinite(now):
            raise ValueError("snapshot timestamp must be finite")
        if request.tbt is not None and now >= request.tbt.deadline:
            if request.recovery_first_miss_time is None:
                request.recovery_first_miss_time = request.tbt.deadline
        first_miss = request.recovery_first_miss_time
        threshold = self.recovery_age_threshold_seconds
        recovery_due = bool(
            first_miss is not None
            and threshold is not None
            and now >= first_miss + threshold
        )
        return LedgerRequestView(
            ttft_deadline=(request.ttft.deadline if request.ttft else None),
            tbt_deadline=(request.tbt.deadline if request.tbt else None),
            recovery=first_miss is not None,
            recovery_due=recovery_due,
            recovery_first_miss_time=first_miss,
        )

    def active_obligations(
        self, request_ids: set[str]
    ) -> tuple[tuple[Obligation, ...], tuple[Obligation, ...]]:
        ttft: list[Obligation] = []
        tbt: list[Obligation] = []
        for request_id in sorted(request_ids):
            request = self._requests.get(request_id)
            if request is None:
                raise ValueError(f"live request is absent from ledger: {request_id}")
            if request.ttft is not None:
                ttft.append(request.ttft)
            if request.tbt is not None:
                tbt.append(request.tbt)
        return tuple(ttft), tuple(tbt)

    @property
    def terminal_counts(self) -> dict[str, int]:
        return dict(self._terminal_counts)
