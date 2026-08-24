"""G2 minimal observer/decision logger.

The full observer with actual-feedback state updates is G6.  This version logs
decisions and observations in a bounded in-memory list for tests and audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dpp_scheduler.contracts import Decision, ExecutionObservation, StateSnapshot


@dataclass
class InMemoryObserver:
    max_records: int = 1024
    records: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_records <= 0:
            raise ValueError("max_records must be positive")

    def record(self, snapshot: StateSnapshot, decision: Decision, observation: ExecutionObservation | None) -> None:
        self.records.append(
            {
                "frame_id": snapshot.frame_id,
                "snapshot_hash": snapshot.snapshot_hash,
                "decision": decision,
                "observation": observation,
            }
        )
        del self.records[:-self.max_records]

    def clear(self) -> None:
        self.records.clear()


class ZeroProgressWatchdogError(RuntimeError):
    """Raised in development mode after a bounded zero-progress streak."""

    def __init__(self, diagnostic: dict[str, object]) -> None:
        super().__init__("non-empty workload made no scheduling progress")
        self.diagnostic = diagnostic


@dataclass
class ProgressWatchdog:
    """Keep bounded iteration diagnostics and detect zero-token stalls."""

    max_records: int
    zero_progress_limit: int
    fail_fast: bool = False
    records: list[dict[str, object]] = field(default_factory=list)
    consecutive_zero_progress: int = 0
    consecutive_decode_only: int = 0

    def __post_init__(self) -> None:
        if self.max_records <= 0 or self.zero_progress_limit <= 0:
            raise ValueError("watchdog limits must be positive")

    def record_iteration(
        self,
        *,
        workload_nonempty: bool,
        scheduled_tokens: int,
        prefill_tokens: int,
        decode_tokens: int,
        diagnostic: dict[str, object],
    ) -> dict[str, object]:
        for label, value in (
            ("scheduled_tokens", scheduled_tokens),
            ("prefill_tokens", prefill_tokens),
            ("decode_tokens", decode_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if scheduled_tokens != prefill_tokens + decode_tokens:
            raise ValueError("scheduled token diagnostic totals do not agree")

        if workload_nonempty and scheduled_tokens == 0:
            self.consecutive_zero_progress += 1
        else:
            self.consecutive_zero_progress = 0
        if scheduled_tokens > 0 and prefill_tokens == 0 and decode_tokens > 0:
            self.consecutive_decode_only += 1
        else:
            self.consecutive_decode_only = 0

        record = dict(diagnostic)
        record.update(
            {
                "workload_nonempty": workload_nonempty,
                "scheduled_tokens": scheduled_tokens,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "consecutive_zero_progress": self.consecutive_zero_progress,
                "consecutive_decode_only": self.consecutive_decode_only,
                "watchdog_triggered": bool(
                    workload_nonempty
                    and self.consecutive_zero_progress
                    >= self.zero_progress_limit
                ),
            }
        )
        self.records.append(record)
        del self.records[:-self.max_records]
        if record["watchdog_triggered"] and self.fail_fast:
            raise ZeroProgressWatchdogError(record)
        return record
