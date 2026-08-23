"""Lazy vLLM entry point for targeted Predictor profiling."""

from __future__ import annotations

from dpp_scheduler.vllm_adapter import get_targeted_profiling_scheduler_class

_scheduler_class: type | None = None


def __getattr__(name: str) -> type:
    global _scheduler_class
    if name != "TargetedProfilingScheduler":
        raise AttributeError(name)
    if _scheduler_class is None:
        _scheduler_class = get_targeted_profiling_scheduler_class()
    return _scheduler_class
