"""Modular DPP scheduler: public contracts and G2 core.

The package intentionally contains no vLLM imports outside
``dpp_scheduler.vllm_adapter``.
"""

from dpp_scheduler.contracts import (
    BatchPlan,
    ControlState,
    Decision,
    DecodeRequest,
    ExecutionObservation,
    Obligation,
    Prediction,
    PrefillRequest,
    SafeSetResult,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.candidate_generator import CandidateGenerator

__all__ = [
    "BatchPlan",
    "CandidateGenerator",
    "ControlState",
    "Decision",
    "DecodeRequest",
    "ExecutionObservation",
    "Obligation",
    "Prediction",
    "PrefillRequest",
    "SafeSetResult",
    "StateSnapshot",
    "validate_snapshot_hash",
]
