"""Validated settings for the modular Scheduler components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SchedulerSettings:
    """Deterministic slack-centered Prefill multiplier candidate settings."""

    prefill_budget_multipliers: tuple[float, ...] = (0.50, 0.75, 1.00, 1.25, 1.50)
    maximum_seed_candidates: int = 16
    predictor_inversion_safety_margin_seconds: float = 0.020
    predictor_inversion_budget_grid: tuple[int, ...] = (
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
    stable_tie_key: str = "request_id"
    completion_aware_tiering: str = "relative_urgency_tertiles"
    completion_aware_equal_score_policy: str = "same_tier_at_first_rank"
    completion_aware_order: tuple[str, ...] = (
        "relative_urgency_tier",
        "remaining_tokens",
        "running_before_waiting",
        "arrival_time",
        "ordinal",
        "request_id",
    )
    allow_partial_prefill: bool = True
    minimum_prefill_chunk_tokens: int = 1
    frozen: bool = True

    def __post_init__(self) -> None:
        if self.prefill_budget_multipliers != (0.50, 0.75, 1.00, 1.25, 1.50):
            raise ValueError(
                "Prefill budget multipliers are fixed at 0.50/0.75/1.00/1.25/1.50"
            )
        if self.maximum_seed_candidates != 16:
            raise ValueError("maximum_seed_candidates is fixed at 16")
        if self.completion_aware_tiering != "relative_urgency_tertiles":
            raise ValueError(
                "completion_aware_tiering must be relative_urgency_tertiles"
            )
        if self.completion_aware_equal_score_policy != "same_tier_at_first_rank":
            raise ValueError(
                "completion_aware_equal_score_policy must be "
                "same_tier_at_first_rank"
            )
        if self.completion_aware_order != (
            "relative_urgency_tier",
            "remaining_tokens",
            "running_before_waiting",
            "arrival_time",
            "ordinal",
            "request_id",
        ):
            raise ValueError("completion_aware_order does not match the v3 contract")
        margin = self.predictor_inversion_safety_margin_seconds
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(margin)
            or margin <= 0
        ):
            raise ValueError(
                "predictor_inversion_safety_margin_seconds must be finite and positive"
            )
        grid = self.predictor_inversion_budget_grid
        if (
            isinstance(grid, bool)
            or not isinstance(grid, tuple)
            or not all(isinstance(item, int) and item >= 0 for item in grid)
            or list(grid) != sorted(grid)
        ):
            raise ValueError(
                "predictor_inversion_budget_grid must be a non-decreasing tuple of non-negative ints"
            )
        if self.minimum_prefill_chunk_tokens <= 0:
            raise ValueError("minimum_prefill_chunk_tokens must be positive")

    @classmethod
    def provisional(cls) -> SchedulerSettings:
        return cls()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SchedulerSettings:
        """Load the frozen decision values from the active config section."""
        if not isinstance(value, Mapping):
            raise ValueError("candidate_generator settings must be a mapping")
        if "prefill_budget_fractions" in value or "include_finish_boundary" in value:
            raise ValueError(
                "candidate_generator v2 fields (prefill_budget_fractions, "
                "include_finish_boundary) are no longer accepted; use "
                "prefill_budget_multipliers and prefill_priority_policies"
            )
        multipliers = value.get("prefill_budget_multipliers")
        if not isinstance(multipliers, list):
            raise ValueError(
                "candidate_generator.prefill_budget_multipliers must be a list"
            )
        frozen = value.get("parameters_frozen")
        if not isinstance(frozen, bool):
            raise ValueError("candidate_generator.parameters_frozen must be boolean")
        maximum = value.get("maximum_seed_candidates", 16)
        if isinstance(maximum, bool) or not isinstance(maximum, int):
            raise ValueError("maximum_seed_candidates must be an integer")
        minimum_prefill = value.get("minimum_prefill_chunk_tokens", 1)
        if isinstance(minimum_prefill, bool) or not isinstance(
            minimum_prefill, int
        ):
            raise ValueError("minimum_prefill_chunk_tokens must be an integer")
        margin = value.get(
            "predictor_inversion_safety_margin_seconds", 0.020
        )
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(margin)
            or margin <= 0
        ):
            raise ValueError(
                "predictor_inversion_safety_margin_seconds must be finite and positive"
            )
        grid_raw = value.get(
            "predictor_inversion_budget_grid",
            list((0, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048)),
        )
        if not isinstance(grid_raw, list):
            raise ValueError(
                "candidate_generator.predictor_inversion_budget_grid must be a list"
            )
        grid = tuple(int(item) for item in grid_raw)
        tiering = value.get("completion_aware_tiering")
        if not isinstance(tiering, str):
            raise ValueError(
                "candidate_generator.completion_aware_tiering must be a string"
            )
        equal_score_policy = value.get("completion_aware_equal_score_policy")
        if not isinstance(equal_score_policy, str):
            raise ValueError(
                "candidate_generator.completion_aware_equal_score_policy "
                "must be a string"
            )
        order_raw = value.get("completion_aware_order")
        if not isinstance(order_raw, list) or not all(
            isinstance(item, str) for item in order_raw
        ):
            raise ValueError(
                "candidate_generator.completion_aware_order must be a list of strings"
            )
        return cls(
            prefill_budget_multipliers=tuple(float(item) for item in multipliers),
            maximum_seed_candidates=maximum,
            predictor_inversion_safety_margin_seconds=float(margin),
            predictor_inversion_budget_grid=grid,
            completion_aware_tiering=tiering,
            completion_aware_equal_score_policy=equal_score_policy,
            completion_aware_order=tuple(order_raw),
            minimum_prefill_chunk_tokens=minimum_prefill,
            frozen=frozen,
        )


@dataclass(frozen=True)
class SafeSetSettings:
    """Hard physical-feasibility settings; SLO risk is never an admission input."""

    rolling_kv_horizon_iterations: int
    reserve_blocks_r0: int
    top_k_when_all_risky: int = 0  # legacy compatibility; inactive in v2

    def __post_init__(self) -> None:
        for label, value in (
            ("rolling_kv_horizon_iterations", self.rolling_kv_horizon_iterations),
            ("reserve_blocks_r0", self.reserve_blocks_r0),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label} must be an integer")
        if self.rolling_kv_horizon_iterations < 0:
            raise ValueError("rolling_kv_horizon_iterations must be non-negative")
        if self.reserve_blocks_r0 < 0:
            raise ValueError("reserve_blocks_r0 must be non-negative")
        if isinstance(self.top_k_when_all_risky, bool) or not isinstance(
            self.top_k_when_all_risky, int
        ):
            raise ValueError("legacy top_k_when_all_risky must be an integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SafeSetSettings:
        if not isinstance(value, Mapping):
            raise ValueError("safe_set settings must be a mapping")
        required = (
            "rolling_kv_horizon_iterations",
            "reserve_blocks_r0",
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
        return cls(
            rolling_kv_horizon_iterations=value["rolling_kv_horizon_iterations"],
            reserve_blocks_r0=value["reserve_blocks_r0"],
            top_k_when_all_risky=int(value.get("top_k_when_all_risky", 0)),
        )


@dataclass(frozen=True)
class SchedulerDiagnosticsSettings:
    """Validated bounded diagnostic/watchdog integration settings."""

    parameter_status: str
    bounded_records: int
    zero_progress_watchdog_iterations: int
    fail_fast_development: bool
    performance_logging_default: bool
    performance_logging_enable_env: str

    def __post_init__(self) -> None:
        if self.parameter_status != "provisional_for_scheduler_integration":
            raise ValueError(
                "scheduler diagnostics must remain provisional for integration"
            )
        for label, value in (
            ("bounded_records", self.bounded_records),
            (
                "zero_progress_watchdog_iterations",
                self.zero_progress_watchdog_iterations,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        for label, value in (
            ("fail_fast_development", self.fail_fast_development),
            ("performance_logging_default", self.performance_logging_default),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be boolean")
        if (
            not isinstance(self.performance_logging_enable_env, str)
            or not self.performance_logging_enable_env
        ):
            raise ValueError("performance_logging_enable_env must be non-empty")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> SchedulerDiagnosticsSettings:
        if not isinstance(value, Mapping):
            raise ValueError("scheduler_diagnostics settings must be a mapping")
        required = (
            "parameter_status",
            "bounded_records",
            "zero_progress_watchdog_iterations",
            "fail_fast_development",
            "performance_logging_default",
            "performance_logging_enable_env",
        )
        missing = [key for key in required if value.get(key) is None]
        if missing:
            raise ValueError(
                "Scheduler diagnostics parameters are unresolved: "
                + ", ".join(missing)
            )
        return cls(
            parameter_status=value["parameter_status"],
            bounded_records=value["bounded_records"],
            zero_progress_watchdog_iterations=value[
                "zero_progress_watchdog_iterations"
            ],
            fail_fast_development=value["fail_fast_development"],
            performance_logging_default=value["performance_logging_default"],
            performance_logging_enable_env=value["performance_logging_enable_env"],
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
        if (
            value.get("final_path")
            != "native_preemption_or_empty_workload_idle"
        ):
            raise ValueError("Fallback final liveness path mismatch")
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


@dataclass(frozen=True)
class PredictorSettings:
    """Constrained Ridge extrapolation settings and live-readiness status."""

    ood_uncertainty_coefficient: float
    parameter_status: str = "provisional_pending_held_out_ood_calibration"
    calibration_artifact_path: str | None = None
    calibration_artifact_sha256: str | None = None
    development_default_acknowledged: bool = False

    def __post_init__(self) -> None:
        value = self.ood_uncertainty_coefficient
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("ood_uncertainty_coefficient must be finite and non-negative")
        if self.parameter_status not in {
            "provisional_pending_held_out_ood_calibration",
            "frozen_from_held_out_ood_calibration",
            "frozen_from_development_held_out_ood_calibration",
            "fixed_default_for_development_nonformal_comparison",
        }:
            raise ValueError("unknown OOD uncertainty parameter status")
        if self.parameter_status in {
            "frozen_from_held_out_ood_calibration",
            "frozen_from_development_held_out_ood_calibration",
        } and (
            not self.calibration_artifact_path
            or not self.calibration_artifact_sha256
            or len(self.calibration_artifact_sha256) != 64
        ):
            raise ValueError("frozen OOD uncertainty artifact path/hash is incomplete")
        if self.parameter_status == (
            "fixed_default_for_development_nonformal_comparison"
        ):
            if value != 0.0:
                raise ValueError("development default OOD coefficient must be zero")
            if not self.development_default_acknowledged:
                raise ValueError("development default OOD mode requires acknowledgement")
            if self.calibration_artifact_path or self.calibration_artifact_sha256:
                raise ValueError(
                    "development default OOD mode must not claim a calibration artifact"
                )
        elif self.development_default_acknowledged:
            raise ValueError(
                "development default acknowledgement requires development default status"
            )

    @property
    def uses_development_default(self) -> bool:
        return self.parameter_status == (
            "fixed_default_for_development_nonformal_comparison"
        )

    @property
    def live_v2_ready(self) -> bool:
        return self.parameter_status in {
            "frozen_from_held_out_ood_calibration",
            "frozen_from_development_held_out_ood_calibration",
            "fixed_default_for_development_nonformal_comparison",
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PredictorSettings:
        extrapolation = value.get("extrapolation")
        if not isinstance(extrapolation, Mapping):
            raise ValueError("predictor.extrapolation must be a mapping")
        if extrapolation.get("strategy") != "clipped_monotonic_ridge":
            raise ValueError("Predictor extrapolation strategy mismatch")
        if extrapolation.get("allow_extrapolation") is not True:
            raise ValueError("Predictor extrapolation must be enabled")
        return cls(
            ood_uncertainty_coefficient=extrapolation.get(
                "ood_uncertainty_coefficient"
            ),
            parameter_status=str(extrapolation.get("parameter_status", "")),
            calibration_artifact_path=extrapolation.get(
                "calibration_artifact_path"
            ),
            calibration_artifact_sha256=extrapolation.get(
                "calibration_artifact_sha256"
            ),
            development_default_acknowledged=(
                extrapolation.get("development_default_acknowledged") is True
            ),
        )


@dataclass(frozen=True)
class DPPSettings:
    """Request-level service-deficit v2 scoring settings."""

    prefill_reference_concurrency: int
    decode_reference_concurrency: int
    maximum_numeric: float
    reference_parameter_status: str = "provisional_pending_stock_profiling"
    score_rel_tol: float = 1e-9
    score_abs_tol: float = 1e-12
    algorithm: str = "request_service_deficit_v2"
    reference_artifact_path: str | None = None
    reference_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("prefill_reference_concurrency", self.prefill_reference_concurrency),
            ("decode_reference_concurrency", self.decode_reference_concurrency),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if not math.isfinite(self.maximum_numeric) or self.maximum_numeric <= 0:
            raise ValueError("maximum_numeric must be finite and positive")
        if self.algorithm != "request_service_deficit_v2":
            raise ValueError("DPP algorithm must be request_service_deficit_v2")
        if self.reference_parameter_status not in {
            "provisional_pending_stock_profiling",
            "frozen_from_stock_positive_frame_p50",
            "frozen_from_development_stock_n300_positive_frame_p50",
        }:
            raise ValueError("unknown reference concurrency parameter status")
        if self.reference_parameter_status in {
            "frozen_from_stock_positive_frame_p50",
            "frozen_from_development_stock_n300_positive_frame_p50",
        } and (
            not self.reference_artifact_path
            or not self.reference_artifact_sha256
            or len(self.reference_artifact_sha256) != 64
        ):
            raise ValueError("frozen reference artifact path/hash is incomplete")
        if self.score_rel_tol != 1e-9 or self.score_abs_tol != 1e-12:
            raise ValueError("DPP score tie tolerance is fixed")

    @property
    def live_v2_ready(self) -> bool:
        return self.reference_parameter_status in {
            "frozen_from_stock_positive_frame_p50",
            "frozen_from_development_stock_n300_positive_frame_p50",
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        token_budget: int,
        sequence_budget: int,
    ) -> DPPSettings:
        del token_budget, sequence_budget
        if not isinstance(value, Mapping):
            raise ValueError("dpp settings must be a mapping")
        reference = value.get("reference_concurrency")
        ranges = value.get("score_numeric_ranges")
        tolerance = value.get("score_tie_tolerance")
        if not isinstance(reference, Mapping):
            raise ValueError("dpp.reference_concurrency must be a mapping")
        if not isinstance(ranges, Mapping) or not isinstance(tolerance, Mapping):
            raise ValueError("DPP numeric range/tolerance must be mappings")
        if reference.get("statistic") != "p50_positive_frames":
            raise ValueError("reference concurrency statistic mismatch")
        if tuple(value.get("tie_break", ())) != (
            "smaller_effective_duration",
            "smaller_prefill_budget",
            "smaller_plan_id",
        ):
            raise ValueError("DPP v2 tie-break contract mismatch")
        return cls(
            prefill_reference_concurrency=reference.get("prefill"),
            decode_reference_concurrency=reference.get("decode"),
            maximum_numeric=float(ranges.get("finite_absolute_maximum")),
            reference_parameter_status=str(reference.get("parameter_status", "")),
            score_rel_tol=float(tolerance.get("relative")),
            score_abs_tol=float(tolerance.get("absolute")),
            algorithm=str(value.get("algorithm", "")),
            reference_artifact_path=reference.get("artifact_path"),
            reference_artifact_sha256=reference.get("artifact_sha256"),
        )
