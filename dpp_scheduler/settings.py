"""G2 settings container.

These values are deliberately provisional until G0/G1 profiling freezes them.
They are used by the pure Candidate Generator and temporary Selector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerSettings:
    """Deterministic scheduler settings for the G2 implementation."""

    critical_horizon_seconds: float | None = None
    prefill_knee_tokens: int | None = None
    recovery_age_threshold: float = 30.0
    maximum_seed_candidates: int = 12
    stable_tie_key: str = "request_id"
    allow_partial_prefill: bool = True
    minimum_prefill_chunk_tokens: int = 1
    frozen: bool = False

    def __post_init__(self) -> None:
        if (
            self.critical_horizon_seconds is not None
            and (
                not math.isfinite(self.critical_horizon_seconds)
                or self.critical_horizon_seconds < 0
            )
        ):
            raise ValueError(
                "critical_horizon_seconds must be finite and non-negative"
            )
        if self.prefill_knee_tokens is not None and self.prefill_knee_tokens <= 0:
            raise ValueError("prefill_knee_tokens must be positive")
        if self.maximum_seed_candidates != 12:
            raise ValueError("maximum_seed_candidates is fixed at 12")
        if self.minimum_prefill_chunk_tokens <= 0:
            raise ValueError("minimum_prefill_chunk_tokens must be positive")

    @property
    def template_names(self) -> tuple[str, ...]:
        return ("MANDATORY", "CRITICAL", "ALL")

    @classmethod
    def provisional(cls) -> SchedulerSettings:
        # Horizon and knee remain absent until same-configuration profiling
        # defines and freezes their selection rules. Candidate generation still
        # produces MANDATORY/ALL and ZERO/FINISH/BINDABLE_MAX actions.
        return cls(frozen=False)
