"""Lazy vLLM entry point for isolated exact-batch profiling."""

from __future__ import annotations

from typing import Any


_scheduler_class: type | None = None


def __getattr__(name: str) -> Any:
    global _scheduler_class
    if name != "IsolatedProfilingScheduler":
        raise AttributeError(name)
    if _scheduler_class is None:
        from dpp_scheduler.vllm_adapter import get_isolated_profiling_scheduler_class

        _scheduler_class = get_isolated_profiling_scheduler_class()
    return _scheduler_class

