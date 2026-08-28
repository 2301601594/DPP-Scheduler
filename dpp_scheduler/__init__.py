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
    FallbackResult,
    Obligation,
    Prediction,
    PrefillRequest,
    SafeCandidate,
    SafeSetResult,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.candidate_generator import CandidateGenerator
from dpp_scheduler.dpp_selector import DPPScore, DPPSelector, SelectorAudit

__all__ = [
    "BatchPlan",
    "CandidateGenerator",
    "ControlState",
    "Decision",
    "DecodeRequest",
    "DPPScore",
    "DPPSelector",
    "SelectorAudit",
    "ExecutionObservation",
    "FallbackResult",
    "Obligation",
    "Prediction",
    "PrefillRequest",
    "SafeCandidate",
    "SafeSetResult",
    "StateSnapshot",
    "validate_snapshot_hash",
]
