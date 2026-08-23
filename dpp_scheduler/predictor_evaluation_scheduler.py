"""Lazy entry point for the real-vLLM Predictor shadow scheduler."""

from __future__ import annotations

from dpp_scheduler.vllm_adapter import get_predictor_evaluation_scheduler_class

_scheduler_class: type | None = None


def __getattr__(name: str) -> type:
    global _scheduler_class
    if name != "PredictorEvaluationScheduler":
        raise AttributeError(name)
    if _scheduler_class is None:
        _scheduler_class = get_predictor_evaluation_scheduler_class()
    return _scheduler_class
