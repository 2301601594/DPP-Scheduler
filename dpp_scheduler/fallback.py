"""Fallback placeholder.

Fallback construction is G4.  G2 keeps a deterministic no-op interface so the
Controller/selector can explicitly report NO_SAFE_DECISION before fallback is
implemented.
"""

from __future__ import annotations

from dpp_scheduler.contracts import BatchPlan, StateSnapshot


class NullFallback:
    def build(self, snapshot: StateSnapshot) -> BatchPlan | None:
        return None
