"""Scheduler-specific exceptions."""


class SchedulerError(Exception):
    """Base error for the modular scheduler."""


class SnapshotHashMismatch(SchedulerError):
    """Raised when a decision/plan is not bound to the active snapshot."""


class ExactExecutionMismatch(SchedulerError):
    """Raised when the executed plan differs from the selected BatchPlan."""


class NoSafeDecision(SchedulerError):
    """Raised when the selector has no safe candidate and no fallback applies."""
