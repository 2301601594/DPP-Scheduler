"""G2 settings container.

These values are deliberately provisional until G0/G1 profiling freezes them.
They are used by the pure Candidate Generator and temporary Selector.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerSettings:
    """Deterministic scheduler settings for the G2 implementation."""

    prefill_caps_small: int
    prefill_caps_medium: int
    prefill_caps_large: int
    urgent_limit_u: int
    recovery_age_threshold: float
    maximum_candidates: int = 12
    stable_tie_key: str = "request_id"
    allow_partial_prefill: bool = True
    minimum_prefill_chunk_tokens: int = 1

    @property
    def prefill_caps(self) -> tuple[int, int, int, int]:
        return (0, self.prefill_caps_small, self.prefill_caps_medium, self.prefill_caps_large)

    @property
    def template_names(self) -> tuple[str, ...]:
        return ("MANDATORY", "URGENT", "ALL")

    @classmethod
    def provisional(cls) -> SchedulerSettings:
        # Placeholder values only; they must be replaced by frozen G0/G1 values
        # before any DPP comparison.
        return cls(
            prefill_caps_small=256,
            prefill_caps_medium=1024,
            prefill_caps_large=2048,
            urgent_limit_u=4,
            recovery_age_threshold=30.0,
        )
