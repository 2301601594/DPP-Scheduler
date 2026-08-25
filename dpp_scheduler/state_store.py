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
    """Request-level debts updated only from actual duration and service."""

    current: ControlState = ControlState(snapshot_hash="unbound")
    settings: DPPSettings | None = None
    _applied_snapshot_hashes: set[str] = field(default_factory=set, init=False)

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
        # Arrival debt is initialized from the authoritative Snapshot timestamp,
        # request arrival time, prompt progress, and request-level SLO.

    def bind_snapshot(self, snapshot: StateSnapshot) -> ControlState:
        """Initialize new Prefill debt and remove completed phase state."""
        prior_ttft = self.current.ttft_debt_map()
        prior_tbt = self.current.tbt_debt_map()
        ttft: dict[str, float] = {}
        for request in snapshot.waiting_prefill_requests:
            if request.ttft_slo_seconds <= 0 or request.token_count <= 0:
                raise ValueError("Prefill request SLO/prompt length must be positive")
            if request.request_id in prior_ttft:
                ttft[request.request_id] = prior_ttft[request.request_id]
            else:
                ttft[request.request_id] = max(
                    0.0,
                    (snapshot.timestamp - request.arrival_time)
                    / request.ttft_slo_seconds
                    - request.prefilled_tokens / request.token_count,
                )
        active_decode = {
            request.request_id for request in snapshot.active_decode_requests
        }
        tbt = {
            request_id: debt
            for request_id, debt in prior_tbt.items()
            if request_id in active_decode
        }
        self.current = ControlState(
            snapshot_hash=snapshot.snapshot_hash,
            ttft_service_debts=tuple(sorted(ttft.items())),
            tbt_service_debts=tuple(sorted(tbt.items())),
        )
        return self.current

    def update_from_actual(
        self,
        *,
        previous_snapshot: StateSnapshot,
        actual_duration_seconds: float,
        executed_prefill_items: tuple[tuple[str, int], ...],
        executed_decode_items: tuple[str, ...],
        initialized_tbt_request_ids: tuple[str, ...] = (),
        terminal_request_ids: tuple[str, ...] = (),
    ) -> ControlState:
        if previous_snapshot.snapshot_hash in self._applied_snapshot_hashes:
            raise DuplicateLedgerEvent("actual iteration feedback already applied")
        self.current = advance_service_debts(
            self.current,
            previous_snapshot=previous_snapshot,
            actual_duration_seconds=actual_duration_seconds,
            executed_prefill_items=executed_prefill_items,
            executed_decode_items=executed_decode_items,
            initialized_tbt_request_ids=initialized_tbt_request_ids,
            terminal_request_ids=terminal_request_ids,
            maximum_numeric=(
                self.settings.maximum_numeric if self.settings else float("inf")
            ),
        )
        self._applied_snapshot_hashes.add(previous_snapshot.snapshot_hash)
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
    initializes_tbt_service: bool = False
    tbt_service_tokens: int = 0


def advance_service_debts(
    state: ControlState,
    *,
    previous_snapshot: StateSnapshot,
    actual_duration_seconds: float,
    executed_prefill_items: tuple[tuple[str, int], ...],
    executed_decode_items: tuple[str, ...],
    initialized_tbt_request_ids: tuple[str, ...] = (),
    terminal_request_ids: tuple[str, ...] = (),
    maximum_numeric: float = float("inf"),
) -> ControlState:
    """Replay one actual iteration using per-request normalized service."""
    validate_snapshot_hash(state.snapshot_hash, previous_snapshot.snapshot_hash)
    if not math.isfinite(actual_duration_seconds) or actual_duration_seconds <= 0:
        raise ValueError("actual duration must be finite and positive")
    if len(dict(executed_prefill_items)) != len(executed_prefill_items):
        raise ValueError("actual Prefill feedback contains duplicate request IDs")
    if len(set(executed_decode_items)) != len(executed_decode_items):
        raise ValueError("actual Decode feedback contains duplicate request IDs")

    prefill = {
        request.request_id: request
        for request in previous_snapshot.waiting_prefill_requests
    }
    decode = {
        request.request_id: request
        for request in previous_snapshot.active_decode_requests
    }
    actual_prefill = dict(executed_prefill_items)
    if not set(actual_prefill).issubset(prefill):
        raise ValueError("actual Prefill feedback references an unknown request")
    if not set(executed_decode_items).issubset(decode):
        raise ValueError("actual Decode feedback references an unknown request")

    ttft: dict[str, float] = {}
    for request_id, current in state.ttft_service_debts:
        request = prefill[request_id]
        tokens = actual_prefill.get(request_id, 0)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("actual Prefill service must be non-negative integers")
        if tokens > request.remaining_tokens:
            raise ValueError("actual Prefill service exceeds remaining prompt")
        if request.prefilled_tokens + tokens >= request.token_count:
            continue
        ttft[request_id] = max(
            0.0,
            current
            + actual_duration_seconds / request.ttft_slo_seconds
            - tokens / request.token_count,
        )

    actual_decode = set(executed_decode_items)
    terminal = set(terminal_request_ids)
    tbt: dict[str, float] = {}
    for request_id, current in state.tbt_service_debts:
        if request_id in terminal:
            continue
        request = decode[request_id]
        tbt[request_id] = max(
            0.0,
            current
            + actual_duration_seconds / request.tbt_slo_seconds
            - (1.0 if request_id in actual_decode else 0.0),
        )
    for request_id in initialized_tbt_request_ids:
        if request_id not in terminal:
            tbt.setdefault(request_id, 0.0)

    if any(
        not math.isfinite(value) or value < 0 or value > maximum_numeric
        for value in tuple(ttft.values()) + tuple(tbt.values())
    ):
        raise ValueError("actual feedback produced out-of-range service debt")
    return ControlState(
        snapshot_hash=previous_snapshot.snapshot_hash,
        ttft_service_debts=tuple(sorted(ttft.items())),
        tbt_service_debts=tuple(sorted(tbt.items())),
    )


@dataclass(frozen=True)
class LedgerRequestView:
    """Snapshot-facing deadline and Recovery state for one live request."""

    ttft_deadline: float | None
    tbt_deadline: float | None
    recovery: bool
    recovery_due: bool
    recovery_first_miss_time: float | None
    goodput_eligible: bool


@dataclass
class _RequestLedger:
    request_id: str
    arrival_time: float
    ttft: Obligation | None
    tbt: Obligation | None = None
    tbt_sequence: int = 0
    first_token_returned: bool = False
    recovery_first_miss_time: float | None = None
    goodput_eligible: bool = True


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

    @staticmethod
    def _record_miss(request: _RequestLedger, obligation: Obligation) -> None:
        """Make request-level Goodput loss and Recovery state monotonic."""
        request.goodput_eligible = False
        if (
            obligation.kind == "TBT"
            and request.recovery_first_miss_time is None
        ):
            request.recovery_first_miss_time = obligation.deadline

    def _settle_request_obligation(
        self,
        request: _RequestLedger,
        obligation: Obligation,
        returned_at: float,
    ) -> tuple[int, int]:
        was_eligible = request.goodput_eligible
        success, miss = self._settle(obligation, returned_at)
        if not was_eligible:
            return (0, 0)
        if miss:
            self._record_miss(request, obligation)
        return (success, miss)

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
        was_first_token = not request.first_token_returned

        if token_count == 1:
            if request.ttft is not None:
                ttft_success, ttft_miss = self._settle_request_obligation(
                    request, request.ttft, returned_at
                )
                request.ttft = None
                request.first_token_returned = True
            elif request.tbt is not None:
                tbt_success, tbt_miss = self._settle_request_obligation(
                    request, request.tbt, returned_at
                )
                request.tbt = None
            elif request.first_token_returned or (
                f"{request_id}:TTFT:0" in self._settled_obligation_ids
            ):
                # The token is late for an already-expired obligation. It
                # advances execution but contributes no repeated outcome.
                request.first_token_returned = True
            else:
                raise ValueError(
                    f"request has no active token obligation: {request_id}"
                )

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
            initializes_tbt_service=bool(
                token_count == 1 and was_first_token and terminal_reason is None
            ),
            tbt_service_tokens=int(token_count == 1 and not was_first_token),
        )

    def expire_deadlines(self, now: float) -> tuple[LedgerUpdate, ...]:
        """Settle every active deadline at or before now exactly once.
        Expired obligations on an already-ineligible request are retired
        silently so they cannot keep adding debt or SLO risk.
        """
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
        ):
            raise ValueError("expiry timestamp must be finite")
        updates: list[LedgerUpdate] = []
        for request_id in sorted(self._requests):
            request = self._requests[request_id]
            for obligation in (request.ttft, request.tbt):
                if obligation is None or obligation.deadline > now:
                    continue
                if obligation.obligation_id in self._settled_obligation_ids:
                    raise DuplicateLedgerEvent(
                        f"obligation already settled: {obligation.obligation_id}"
                    )
                self._settled_obligation_ids.add(obligation.obligation_id)
                if obligation.kind == "TTFT":
                    request.ttft = None
                elif obligation.kind == "TBT":
                    request.tbt = None
                else:
                    raise ValueError(f"unknown obligation kind: {obligation.kind}")
                was_eligible = request.goodput_eligible
                self._record_miss(request, obligation)
                if was_eligible:
                    updates.append(
                        LedgerUpdate(
                            event_id=f"expiry:{obligation.obligation_id}",
                            request_id=request_id,
                            ttft_miss=int(obligation.kind == "TTFT"),
                            tbt_miss=int(obligation.kind == "TBT"),
                        )
                    )
        return tuple(updates)
    def request_view(self, request_id: str, now: float) -> LedgerRequestView:
        request = self._requests.get(request_id)
        if request is None:
            raise ValueError(f"unknown ledger request: {request_id}")
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
        ):
            raise ValueError("snapshot timestamp must be finite")
        first_miss = request.recovery_first_miss_time
        threshold = self.recovery_age_threshold_seconds
        recovery_due = bool(
            first_miss is not None
            and threshold is not None
            and now >= first_miss + threshold
        )
        eligible = request.goodput_eligible
        return LedgerRequestView(
            ttft_deadline=(
                request.ttft.deadline
                if eligible and request.ttft is not None
                else None
            ),
            tbt_deadline=(
                request.tbt.deadline
                if eligible and request.tbt is not None
                else None
            ),
            recovery=first_miss is not None,
            recovery_due=recovery_due,
            recovery_first_miss_time=first_miss,
            goodput_eligible=eligible,
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
            if not request.goodput_eligible:
                continue
            if request.ttft is not None:
                ttft.append(request.ttft)
            if request.tbt is not None:
                tbt.append(request.tbt)
        return tuple(ttft), tuple(tbt)

    @property
    def terminal_counts(self) -> dict[str, int]:
        return dict(self._terminal_counts)
