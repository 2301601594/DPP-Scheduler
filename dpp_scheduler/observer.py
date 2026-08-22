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
