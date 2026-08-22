"""Lazy vLLM Scheduler-class entry point.

All version-specific vLLM imports and behavior remain in ``vllm_adapter``.
"""

from __future__ import annotations

from dpp_scheduler.vllm_adapter import get_modular_scheduler_class

_scheduler_class: type | None = None


def __getattr__(name: str) -> type:
    global _scheduler_class
    if name != "ModularDPPScheduler":
        raise AttributeError(name)
    if _scheduler_class is None:
        _scheduler_class = get_modular_scheduler_class()
    return _scheduler_class
