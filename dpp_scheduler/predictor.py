"""Offline Ridge duration Predictor with bounded online residual calibration."""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dpp_scheduler.contracts import BatchPlan, Prediction, StateSnapshot


BATCH_KINDS = ("decode_only", "mixed", "prefill_only")
FEATURE_NAMES = tuple(f"x_{index}" for index in range(1, 9))
ACTIVE_FEATURES = {
    "decode_only": ("x_4", "x_5"),
    "mixed": FEATURE_NAMES,
    "prefill_only": ("x_1", "x_2", "x_3", "x_6", "x_7", "x_8"),
}
ONLINE_PREDICTOR_VERSION = "qwen3-14b-ridge-three-scenario-online-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _higher_quantile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    return ordered[math.ceil(quantile * (len(ordered) - 1))]


def classify_batch(plan: BatchPlan) -> str | None:
    if plan.prefill_items and plan.decode_items:
        return "mixed"
    if plan.prefill_items:
        return "prefill_only"
    if plan.decode_items:
        return "decode_only"
    return None


def build_plan_features(
    snapshot: StateSnapshot, plan: BatchPlan
) -> tuple[str, dict[str, float]]:
    """Build the frozen x1-x8 schema from current, length-blind state."""
    plan.validate_snapshot(snapshot)
    kind = classify_batch(plan)
    if kind is None:
        raise ValueError("empty BatchPlan has no Predictor model")
    prefill_ids = [request_id for request_id, _ in plan.prefill_items]
    decode_ids = list(plan.decode_items)
    if len(prefill_ids) != len(set(prefill_ids)):
        raise ValueError("duplicate Prefill request in BatchPlan")
    if len(decode_ids) != len(set(decode_ids)):
        raise ValueError("duplicate Decode request in BatchPlan")
    if set(prefill_ids).intersection(decode_ids):
        raise ValueError("request cannot be Prefill and Decode in one BatchPlan")
    if any(int(tokens) <= 0 for _, tokens in plan.prefill_items):
        raise ValueError("scheduled Prefill tokens must be positive")
    if sum(int(tokens) for _, tokens in plan.prefill_items) != int(
        plan.total_prefill_tokens
    ):
        raise ValueError("BatchPlan Prefill token total mismatch")
    if len(plan.decode_items) != int(plan.total_decode_tokens):
        raise ValueError("BatchPlan Decode token total mismatch")

    prefill = {
        request.request_id: request
        for request in snapshot.waiting_prefill_requests
    }
    decode = {
        request.request_id: request for request in snapshot.active_decode_requests
    }
    prefill_work: list[tuple[float, float]] = []
    for request_id, scheduled_tokens in plan.prefill_items:
        request = prefill.get(request_id)
        if request is None:
            raise ValueError(f"unknown Prefill request: {request_id}")
        prefill_work.append(
            (float(request.prefilled_tokens), float(scheduled_tokens))
        )
    decode_contexts: list[float] = []
    for request_id in plan.decode_items:
        request = decode.get(request_id)
        if request is None:
            raise ValueError(f"unknown Decode request: {request_id}")
        decode_contexts.append(float(request.kv_context_length))

    features = {
        "x_1": sum(tokens * (context + tokens) for context, tokens in prefill_work),
        "x_2": sum(tokens * tokens for _, tokens in prefill_work),
        "x_3": sum(context for context, _ in prefill_work) + sum(decode_contexts),
        "x_4": float(len(decode_contexts)),
        "x_5": sum(decode_contexts),
        "x_6": sum(tokens for _, tokens in prefill_work),
        "x_7": max((tokens for _, tokens in prefill_work), default=0.0),
        "x_8": float(len(prefill_work)),
    }
    if not all(math.isfinite(value) and value >= 0 for value in features.values()):
        raise ValueError("Predictor feature is negative or non-finite")
    return kind, features


@dataclass(frozen=True)
class PredictionAudit:
    prediction: Prediction
    batch_kind: str | None
    base_duration_seconds: float | None
    calibration_source: str | None
    calibration_sample_count: int
    centered_residual_p95_seconds: float | None = None
    predictor_cpu_seconds: float | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class CalibrationUpdate:
    batch_kind: str
    residual_seconds: float
    samples_before: int
    samples_after: int


class DurationPredictor(ABC):
    @abstractmethod
    def predict(
        self, snapshot: StateSnapshot, plans: Iterable[BatchPlan]
    ) -> tuple[Prediction, ...]:
        raise NotImplementedError


class NullDurationPredictor(DurationPredictor):
    """G2 placeholder: every prediction is explicitly out-of-support."""

    def predict(
        self, snapshot: StateSnapshot, plans: Iterable[BatchPlan]
    ) -> tuple[Prediction, ...]:
        return tuple(
            Prediction(
                plan_id=plan.plan_id,
                snapshot_hash=snapshot.snapshot_hash,
                expected_duration=None,
                conservative_duration=None,
                in_support=False,
                prediction_mode="INVALID",
                predictor_version="null-g2",
            )
            for plan in plans
        )


@dataclass(frozen=True)
class _RidgeModel:
    names: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    support: tuple[tuple[float, float], ...]

    def base_prediction(
        self, features: dict[str, float]
    ) -> tuple[float, bool, float]:
        values = tuple(_finite(features[name], label=name) for name in self.names)
        in_support = all(
            lower <= value <= upper
            for value, (lower, upper) in zip(values, self.support)
        )
        clipped = tuple(
            min(upper, max(lower, value))
            for value, (lower, upper) in zip(values, self.support)
        )
        boundary = self.intercept + sum(
            coefficient * ((value - mean) / scale)
            for value, coefficient, mean, scale in zip(
                clipped, self.coefficients, self.means, self.scales
            )
        )
        high_extrapolation = sum(
            max(0.0, coefficient / scale) * max(0.0, value - upper)
            for value, coefficient, scale, (_, upper) in zip(
                values, self.coefficients, self.scales, self.support
            )
        )
        ood_distance = max(
            (
                abs(value - clipped_value) / scale
                for value, clipped_value, scale in zip(
                    values, clipped, self.scales
                )
            ),
            default=0.0,
        )
        return boundary + high_extrapolation, in_support, ood_distance


@dataclass(frozen=True)
class _CalibrationSettings:
    window_size: int
    minimum_samples: int
    cold_mean: float
    cold_centered_p95: float


class OnlineResidualCalibrator:
    """Three independent, bounded residual windows; model weights stay frozen."""

    def __init__(self, settings: dict[str, _CalibrationSettings]) -> None:
        if set(settings) != set(BATCH_KINDS):
            raise ValueError("calibration settings must cover all batch kinds")
        self._settings = settings
        self._windows = {
            kind: deque(maxlen=value.window_size) for kind, value in settings.items()
        }
        self._last_observed_frame = -1

    def values(self, kind: str) -> tuple[float, float, str, int]:
        setting = self._settings[kind]
        window = self._windows[kind]
        if len(window) < setting.minimum_samples:
            return (
                setting.cold_mean,
                setting.cold_centered_p95,
                "offline_oof_cold_start",
                len(window),
            )
        mean = sum(window) / len(window)
        centered_p95 = _higher_quantile(
            (value - mean for value in window), 0.95
        )
        return mean, centered_p95, "online_window", len(window)

    def observe(self, *, frame_id: int, kind: str, residual: float) -> CalibrationUpdate:
        if frame_id <= self._last_observed_frame:
            raise ValueError("Predictor feedback frame is duplicate or out of order")
        residual = _finite(residual, label="residual")
        window = self._windows[kind]
        before = len(window)
        window.append(residual)
        self._last_observed_frame = frame_id
        return CalibrationUpdate(kind, residual, before, len(window))

    def sample_count(self, kind: str) -> int:
        return len(self._windows[kind])


class RidgeDurationPredictor(DurationPredictor):
    """Artifact-backed Ridge models composed with online residual windows."""

    def __init__(
        self,
        *,
        predictor_version: str,
        models: dict[str, _RidgeModel],
        calibrator: OnlineResidualCalibrator,
        ood_uncertainty_coefficient: float = 0.05,
    ) -> None:
        if (
            not math.isfinite(ood_uncertainty_coefficient)
            or ood_uncertainty_coefficient < 0
        ):
            raise ValueError("OOD uncertainty coefficient must be non-negative")
        self.predictor_version = predictor_version
        self._models = models
        self._calibrator = calibrator
        self.ood_uncertainty_coefficient = float(ood_uncertainty_coefficient)

    @classmethod
    def from_artifact(
        cls,
        artifact_root: str | Path,
        *,
        ood_uncertainty_coefficient: float = 0.05,
    ) -> RidgeDurationPredictor:
        root = Path(artifact_root).resolve()
        manifest = _load_json(root / "artifact_manifest.json")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("status") != "complete"
            or manifest.get("predictor_version") != ONLINE_PREDICTOR_VERSION
        ):
            raise ValueError("online Predictor artifact identity/status mismatch")
        record = manifest.get("files", {}).get("predictor")
        if not isinstance(record, dict):
            raise ValueError("online Predictor manifest is missing predictor file")
        predictor_path = (root / str(record.get("file", ""))).resolve()
        if predictor_path.parent != root:
            raise ValueError("online Predictor file escapes artifact directory")
        if _sha256_file(predictor_path) != record.get("sha256"):
            raise ValueError("online Predictor artifact hash mismatch")
        payload = _load_json(predictor_path)
        if (
            payload.get("schema_version") != 1
            or payload.get("predictor_version") != ONLINE_PREDICTOR_VERSION
            or payload.get("model_family") != "ridge_regression"
        ):
            raise ValueError("online Predictor payload identity mismatch")

        models: dict[str, _RidgeModel] = {}
        calibration: dict[str, _CalibrationSettings] = {}
        if set(payload.get("models", {})) != set(BATCH_KINDS):
            raise ValueError("online Predictor models do not cover all batch kinds")
        calibration_payload = payload.get("residual_calibration", {})
        if (
            calibration_payload.get("strategy") != "online_window_per_batch_kind"
            or calibration_payload.get("residual_definition")
            != "actual_duration_seconds-base_duration_seconds"
            or calibration_payload.get("quantile_method") != "higher"
        ):
            raise ValueError("online residual calibration contract mismatch")
        by_kind = calibration_payload.get("by_batch_kind", {})
        if set(by_kind) != set(BATCH_KINDS):
            raise ValueError("online calibration does not cover all batch kinds")

        for kind in BATCH_KINDS:
            model = payload["models"][kind]
            names = tuple(model.get("active_features", ()))
            if names != ACTIVE_FEATURES[kind]:
                raise ValueError(f"active feature schema mismatch for {kind}")
            coefficients = model.get("coefficients_for_standardized_features", {})
            standardization = model.get("standardization", {})
            support_payload = model.get("support_domain_train_marginal_box", {})
            if not all(
                set(mapping) == set(names)
                for mapping in (coefficients, standardization, support_payload)
            ):
                raise ValueError(f"model field coverage mismatch for {kind}")
            scales = tuple(
                _finite(standardization[name]["scale"], label=f"{kind}.{name}.scale")
                for name in names
            )
            if any(value <= 0 for value in scales):
                raise ValueError("Predictor feature scales must be positive")
            support = tuple(
                (
                    _finite(support_payload[name]["min"], label="support min"),
                    _finite(support_payload[name]["max"], label="support max"),
                )
                for name in names
            )
            if any(lower > upper for lower, upper in support):
                raise ValueError("Predictor support range is reversed")
            models[kind] = _RidgeModel(
                names=names,
                intercept=_finite(model["intercept_seconds"], label="intercept"),
                coefficients=tuple(
                    _finite(coefficients[name], label=f"{kind}.{name}.coefficient")
                    for name in names
                ),
                means=tuple(
                    _finite(standardization[name]["mean"], label=f"{kind}.{name}.mean")
                    for name in names
                ),
                scales=scales,
                support=support,
            )

            item = by_kind[kind]
            window_size = int(item["window_size"])
            minimum_samples = int(item["minimum_samples"])
            cold_mean = _finite(item["cold_start_mean_seconds"], label="cold mean")
            cold_p95 = _finite(
                item["cold_start_centered_p95_seconds"], label="cold p95"
            )
            if window_size not in {32, 64, 128}:
                raise ValueError("online window size is not a selected candidate")
            if minimum_samples != 32 or minimum_samples > window_size:
                raise ValueError("online calibration minimum sample contract mismatch")
            if cold_p95 < 0:
                raise ValueError("cold-start conservative margin must be non-negative")
            calibration[kind] = _CalibrationSettings(
                window_size, minimum_samples, cold_mean, cold_p95
            )

        return cls(
            predictor_version=ONLINE_PREDICTOR_VERSION,
            models=models,
            calibrator=OnlineResidualCalibrator(calibration),
            ood_uncertainty_coefficient=ood_uncertainty_coefficient,
        )

    def predict_with_audit(
        self, snapshot: StateSnapshot, plan: BatchPlan
    ) -> PredictionAudit:
        try:
            kind, features = build_plan_features(snapshot, plan)
            base, in_support, ood_distance = self._models[kind].base_prediction(
                features
            )
            if not math.isfinite(base) or base <= 0:
                raise ValueError("base duration prediction is non-positive or non-finite")
            residual_mean, centered_p95, source, sample_count = (
                self._calibrator.values(kind)
            )
            expected = base + residual_mean
            conservative = (
                expected
                + centered_p95
                + self.ood_uncertainty_coefficient * ood_distance
            )
            if (
                not math.isfinite(expected)
                or not math.isfinite(conservative)
                or expected <= 0
                or conservative <= 0
                or conservative < expected
            ):
                raise ValueError("calibrated duration is invalid")
            prediction = Prediction(
                plan_id=plan.plan_id,
                snapshot_hash=snapshot.snapshot_hash,
                expected_duration=expected,
                conservative_duration=conservative,
                in_support=in_support,
                ood_distance=ood_distance,
                prediction_mode=(
                    "INTERPOLATION" if in_support else "CONSTRAINED_EXTRAPOLATION"
                ),
                predictor_version=self.predictor_version,
            )
            return PredictionAudit(
                prediction=prediction,
                batch_kind=kind,
                base_duration_seconds=base,
                calibration_source=source,
                calibration_sample_count=sample_count,
                centered_residual_p95_seconds=centered_p95,
            )
        except (KeyError, TypeError, ValueError) as error:
            return PredictionAudit(
                prediction=Prediction(
                    plan_id=plan.plan_id,
                    snapshot_hash=snapshot.snapshot_hash,
                    expected_duration=None,
                    conservative_duration=None,
                    in_support=False,
                    prediction_mode="INVALID",
                    predictor_version=self.predictor_version,
                ),
                batch_kind=classify_batch(plan),
                base_duration_seconds=None,
                calibration_source=None,
                calibration_sample_count=0,
                centered_residual_p95_seconds=None,
                rejection_reason=f"{type(error).__name__}: {error}",
            )

    def predict(
        self, snapshot: StateSnapshot, plans: Iterable[BatchPlan]
    ) -> tuple[Prediction, ...]:
        return tuple(
            self.predict_with_audit(snapshot, plan).prediction for plan in plans
        )

    def observe_actual(
        self,
        snapshot: StateSnapshot,
        plan: BatchPlan,
        actual_duration_seconds: float,
        *,
        base_duration_seconds: float | None = None,
    ) -> CalibrationUpdate:
        actual = _finite(actual_duration_seconds, label="actual duration")
        if actual <= 0:
            raise ValueError("actual duration must be positive")
        audit = self.predict_with_audit(snapshot, plan)
        if not audit.prediction.in_support or audit.base_duration_seconds is None:
            raise ValueError("out-of-support prediction cannot update calibration")
        base = audit.base_duration_seconds
        if base_duration_seconds is not None and not math.isclose(
            base,
            _finite(base_duration_seconds, label="recorded base duration"),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("recorded base duration does not match current plan")
        assert audit.batch_kind is not None
        return self._calibrator.observe(
            frame_id=snapshot.frame_id,
            kind=audit.batch_kind,
            residual=actual - base,
        )

    def calibration_sample_count(self, kind: str) -> int:
        return self._calibrator.sample_count(kind)
