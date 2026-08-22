"""Minimal in-memory control-state store.

The real DPP control state and obligation ledger are G5/G6.  This store exists
so G2 controller wiring has a deterministic place to keep ControlState without
yet implementing DPP updates.
"""

from __future__ import annotations

from dataclasses import dataclass

from dpp_scheduler.contracts import ControlState


@dataclass
class InMemoryStateStore:
    current: ControlState = ControlState(
        snapshot_hash="unbound",
        prefill_backlog=0,
        ttft_debt=0.0,
        tbt_debt=0.0,
    )

    def get(self) -> ControlState:
        return self.current

    def set(self, state: ControlState) -> None:
        self.current = state
