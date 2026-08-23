"""G2 settings container.

These values are deliberately provisional until G0/G1 profiling freezes them.
They are used by the pure Candidate Generator and temporary Selector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SchedulerSettings:
        """Load the frozen decision values from the active config section."""
        if not isinstance(value, Mapping):
            raise ValueError("candidate_generator settings must be a mapping")
        breakpoints = value.get("prefill_breakpoints")
        if not isinstance(breakpoints, Mapping):
            raise ValueError("candidate_generator.prefill_breakpoints must be a mapping")
        frozen = value.get("parameters_frozen")
        if not isinstance(frozen, bool):
            raise ValueError("candidate_generator.parameters_frozen must be boolean")
        horizon = value.get("critical_horizon_seconds")
        knee = breakpoints.get("knee_tokens")
        knee_status = breakpoints.get(
            "knee_status", "frozen" if frozen else "pending"
        )
        if frozen and (horizon is None or knee is None):
            raise ValueError("frozen candidate parameters cannot be null")
        if frozen and knee_status != "frozen":
            raise ValueError("frozen Prefill knee must have knee_status=frozen")
        if not frozen and horizon is not None:
            raise ValueError("unfrozen critical horizon must remain null")
        if not frozen and knee is None and knee_status != "pending":
            raise ValueError("null Prefill knee must have knee_status=pending")
        if not frozen and knee is not None and knee_status != "provisional":
            raise ValueError(
                "unfrozen Prefill knee requires knee_status=provisional"
            )
        if horizon is not None and (
            isinstance(horizon, bool) or not isinstance(horizon, (int, float))
        ):
            raise ValueError("critical_horizon_seconds must be numeric")
        if knee is not None and (isinstance(knee, bool) or not isinstance(knee, int)):
            raise ValueError("prefill knee_tokens must be an integer")
        maximum = value.get("maximum_seed_candidates", 12)
        if isinstance(maximum, bool) or not isinstance(maximum, int):
            raise ValueError("maximum_seed_candidates must be an integer")
        minimum_prefill = value.get("minimum_prefill_chunk_tokens", 1)
        if isinstance(minimum_prefill, bool) or not isinstance(
            minimum_prefill, int
        ):
            raise ValueError("minimum_prefill_chunk_tokens must be an integer")
        return cls(
            critical_horizon_seconds=(float(horizon) if horizon is not None else None),
            prefill_knee_tokens=knee,
            maximum_seed_candidates=maximum,
            minimum_prefill_chunk_tokens=minimum_prefill,
            frozen=frozen,
        )


@dataclass(frozen=True)
class SafeSetSettings:
    """Frozen physical-guard and all-risk ranking parameters for G4."""

    rolling_kv_horizon_iterations: int
    reserve_blocks_r0: int
    top_k_when_all_risky: int

    def __post_init__(self) -> None:
        for label, value in (
            ("rolling_kv_horizon_iterations", self.rolling_kv_horizon_iterations),
            ("reserve_blocks_r0", self.reserve_blocks_r0),
            ("top_k_when_all_risky", self.top_k_when_all_risky),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an integer")
        if self.rolling_kv_horizon_iterations < 0:
            raise ValueError("rolling_kv_horizon_iterations must be non-negative")
        if self.reserve_blocks_r0 < 0:
            raise ValueError("reserve_blocks_r0 must be non-negative")
        if self.top_k_when_all_risky <= 0:
            raise ValueError("top_k_when_all_risky must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SafeSetSettings:
        if not isinstance(value, Mapping):
            raise ValueError("safe_set settings must be a mapping")
        required = (
            "rolling_kv_horizon_iterations",
            "reserve_blocks_r0",
            "top_k_when_all_risky",
        )
        missing = [key for key in required if value.get(key) is None]
        if missing:
            raise ValueError(
                "Safe-Set parameters are not frozen: " + ", ".join(missing)
            )
        if value.get("full_output_length_reservation") != "forbidden":
            raise ValueError("full output length reservation must be forbidden")
        if value.get("slo_risk_is_hard_filter") is not False:
            raise ValueError("SLO risk must not be configured as a hard filter")
        if tuple(value.get("all_risk_order", ())) != (
            "predicted_violation_count",
            "predicted_total_lateness",
            "stable_plan_key",
        ):
            raise ValueError("Safe-Set all-risk ordering contract mismatch")
        return cls(
            rolling_kv_horizon_iterations=value["rolling_kv_horizon_iterations"],
            reserve_blocks_r0=value["reserve_blocks_r0"],
            top_k_when_all_risky=value["top_k_when_all_risky"],
        )


@dataclass(frozen=True)
class FallbackSettings:
    """Deterministic Controller-owned Fallback construction settings."""

    minimum_prefill_chunk_tokens: int

    def __post_init__(self) -> None:
        value = self.minimum_prefill_chunk_tokens
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("minimum_prefill_chunk_tokens must be a positive integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FallbackSettings:
        if not isinstance(value, Mapping):
            raise ValueError("fallback settings must be a mapping")
        minimum = value.get("minimum_prefill_chunk_tokens")
        if minimum is None:
            raise ValueError("Fallback minimum_prefill_chunk_tokens is unresolved")
        if value.get("participates_in_dpp_scoring") is not False:
            raise ValueError("Fallback must remain independent of DPP scoring")
        if value.get("must_pass_hard_filters") is not True:
            raise ValueError("Fallback must pass hard filters")
        if value.get("with_decode") != "edf_decode_only_no_prefill":
            raise ValueError("Fallback Decode policy mismatch")
        if value.get("without_decode") != "minimum_physically_feasible_prefill":
            raise ValueError("Fallback Prefill policy mismatch")
        return cls(minimum_prefill_chunk_tokens=minimum)


@dataclass(frozen=True)
class ObligationSettings:
    """SLO deadlines and optional Recovery age used by the live ledger."""

    ttft_slo_seconds: float
    tbt_slo_seconds: float
    recovery_age_threshold_seconds: float | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("ttft_slo_seconds", self.ttft_slo_seconds),
            ("tbt_slo_seconds", self.tbt_slo_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{label} must be finite and positive")
        threshold = self.recovery_age_threshold_seconds
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError(
                "recovery_age_threshold_seconds must be finite and non-negative"
            )
