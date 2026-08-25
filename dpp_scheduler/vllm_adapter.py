"""vLLM Adapter boundary for the modular DPP scheduler.

Only this module may import vLLM internal types.  The G2 implementation
provides the exact-plan adapter plus a callback-based adapter used by unit
tests.  The vLLM-specific methods are implemented lazily so that pure-Python
unit tests can import this module without a full vLLM installation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dpp_scheduler.contracts import (
    BatchPlan,
    DecodeRequest,
    Decision,
    ExecutionObservation,
    PrefillRequest,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.candidate_generator import (
    project_kv_blocks,
    project_sequence_count,
)
from dpp_scheduler.state_store import ObligationLedger


STOCK_PROFILE_PATH_ENV = "DPP_STOCK_PROFILE_PATH"
STOCK_PROFILE_RUN_ID_ENV = "DPP_STOCK_PROFILE_RUN_ID"
TARGET_PROFILE_PATH_ENV = "DPP_TARGET_PROFILE_PATH"
TARGET_PROFILE_RUN_ID_ENV = "DPP_TARGET_PROFILE_RUN_ID"
TARGET_PROFILE_RECIPE_SEED_ENV = "DPP_TARGET_PROFILE_RECIPE_SEED"
TARGET_PROFILE_RECIPE_MODE_ENV = "DPP_TARGET_PROFILE_RECIPE_MODE"
ISOLATED_PROFILE_PATH_ENV = "DPP_ISOLATED_PROFILE_PATH"
ISOLATED_PROFILE_EVENT_PATH_ENV = "DPP_ISOLATED_PROFILE_EVENT_PATH"
ISOLATED_PROFILE_RUN_ID_ENV = "DPP_ISOLATED_PROFILE_RUN_ID"
ISOLATED_PROFILE_RECIPE_SEED_ENV = "DPP_ISOLATED_PROFILE_RECIPE_SEED"
ISOLATED_PROFILE_RECIPE_MODE_ENV = "DPP_ISOLATED_PROFILE_RECIPE_MODE"
PREDICTOR_EVAL_PATH_ENV = "DPP_PREDICTOR_EVAL_PATH"
PREDICTOR_EVAL_RUN_ID_ENV = "DPP_PREDICTOR_EVAL_RUN_ID"
PREDICTOR_ARTIFACT_PATH_ENV = "DPP_PREDICTOR_ARTIFACT_PATH"
PREDICTOR_EVAL_RECIPE_SEED_ENV = "DPP_PREDICTOR_EVAL_RECIPE_SEED"
PREDICTOR_EVAL_RECIPE_MODE_ENV = "DPP_PREDICTOR_EVAL_RECIPE_MODE"
DPP_DIAGNOSTIC_ITERATION_LOG_ENV = "DPP_DIAGNOSTIC_ITERATION_LOG"
DPP_DIAGNOSTIC_AGGREGATE_PATH_ENV = "DPP_DIAGNOSTIC_AGGREGATE_PATH"
DPP_EXECUTION_SCOPE_ENV = "DPP_EXECUTION_SCOPE"

VLLM_OFFICIAL_ITERATION_TIMING = "vllm_official_iteration_details"
VLLM_ALIGNED_ITERATION_TIMING = "vllm_aligned_monotonic"
STOCK_PROFILE_SCHEMA_VERSION = 2
STOCK_CONCURRENCY_SEMANTICS = "dpp_stage_queues_v2"


def _build_iteration_timing_bridge(original: Callable[..., Any]) -> Callable[..., Any]:
    @contextmanager
    def capture_iteration_details(core: Any, scheduler_output: Any):
        details = None
        aligned_duration = 0.0
        with original(core, scheduler_output) as details:
            aligned_started = time.monotonic()
            try:
                yield details
            finally:
                aligned_duration = time.monotonic() - aligned_started

        if scheduler_output is None:
            return
        callback = getattr(core.scheduler, "_dpp_record_iteration_duration", None)
        if callback is None:
            return
        zero_token_iteration = int(scheduler_output.total_num_scheduled_tokens) == 0
        if zero_token_iteration and not bool(
            getattr(
                core.scheduler,
                "_dpp_capture_zero_token_iteration_timing",
                False,
            )
        ):
            return
        if details is not None:
            duration_seconds = float(details.elapsed_ms) / 1000.0
            iteration_index = int(details.iteration_index)
            timing_source = VLLM_OFFICIAL_ITERATION_TIMING
        else:
            duration_seconds = aligned_duration
            iteration_index = None
            timing_source = VLLM_ALIGNED_ITERATION_TIMING
        if not duration_seconds > 0:
            raise RuntimeError("vLLM iteration duration must be positive")
        callback(
            scheduler_output=scheduler_output,
            iteration_index=iteration_index,
            duration_seconds=duration_seconds,
            timing_source=timing_source,
        )

    return capture_iteration_details


def install_vllm_iteration_timing_bridge() -> None:
    """Forward vLLM's EngineCore iteration boundary to project schedulers.

    The locked vLLM timer starts after asynchronous model submission and ends
    after model-result collection/sampling, immediately before
    ``Scheduler.update_from_output``. When official iteration details are
    disabled, the bridge measures that same context-manager boundary locally.
    """
    from vllm.v1.engine.core import EngineCore

    marker = "_dpp_iteration_timing_bridge_installed"
    if getattr(EngineCore, marker, False):
        return

    EngineCore.capture_iteration_details = _build_iteration_timing_bridge(
        EngineCore.capture_iteration_details
    )
    setattr(EngineCore, marker, True)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_stock_profile_state(scheduler: Any) -> dict[str, Any]:
    """Capture only current, length-blind state before stock scheduling."""
    running = tuple(scheduler.running)
    waiting = tuple(scheduler.waiting)
    running_ids = {request.request_id for request in running}
    waiting_ids = {request.request_id for request in waiting}
    if running_ids.intersection(waiting_ids):
        raise RuntimeError("Stock request appears in both running and waiting queues")

    request_states = []
    current_context_tokens: dict[str, int] = {}
    running_prefill_ids: list[str] = []
    running_decode_ids: list[str] = []
    waiting_prefill_ids: list[str] = []
    waiting_decode_ids: list[str] = []
    preempted_ids: list[str] = []
    other_waiting_ids: list[str] = []
    requests_with_preemptions: list[str] = []
    total_preemptions = 0
    for request_id in sorted(scheduler.requests):
        request = scheduler.requests[request_id]
        context_tokens = int(request.num_computed_tokens)
        prompt_tokens = int(request.num_prompt_tokens)
        status = str(request.status)
        num_preemptions = int(getattr(request, "num_preemptions", 0))
        current_tokens = int(getattr(request, "num_tokens", prompt_tokens))
        current_context_tokens[request_id] = context_tokens
        if num_preemptions > 0:
            requests_with_preemptions.append(request_id)
            total_preemptions += num_preemptions
        if request_id in running_ids:
            if status != "RUNNING":
                raise RuntimeError("Stock running queue contains a non-RUNNING request")
            if context_tokens < prompt_tokens:
                running_prefill_ids.append(request_id)
            else:
                running_decode_ids.append(request_id)
        elif request_id in waiting_ids:
            decode_phase = current_tokens > prompt_tokens or context_tokens >= prompt_tokens
            if status == "PREEMPTED":
                preempted_ids.append(request_id)
            elif status == "WAITING" and not decode_phase and context_tokens == 0:
                waiting_prefill_ids.append(request_id)
            elif decode_phase:
                waiting_decode_ids.append(request_id)
            else:
                other_waiting_ids.append(request_id)
        else:
            other_waiting_ids.append(request_id)
        request_states.append(
            {
                "request_id": request_id,
                "status": status,
                "queue": (
                    "running"
                    if request_id in running_ids
                    else "waiting"
                    if request_id in waiting_ids
                    else "other"
                ),
                "current_context_tokens": context_tokens,
                "prompt_tokens": prompt_tokens,
                "current_tokens": current_tokens,
                "num_preemptions": num_preemptions,
            }
        )

    for values in (
        running_prefill_ids,
        running_decode_ids,
        waiting_prefill_ids,
        waiting_decode_ids,
        preempted_ids,
        other_waiting_ids,
        requests_with_preemptions,
    ):
        values.sort()

    snapshot_prefill_count = len(running_prefill_ids) + len(waiting_prefill_ids)
    snapshot_decode_count = len(running_decode_ids)

    snapshot_payload = {
        "requests": request_states,
        "running_request_ids": sorted(running_ids),
        "waiting_request_ids": sorted(waiting_ids),
        "free_kv_blocks": int(
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        ),
        "token_budget": int(scheduler.max_num_scheduled_tokens),
        "sequence_budget": int(scheduler.max_num_running_reqs),
    }
    return {
        "snapshot_hash": _canonical_sha256(snapshot_payload),
        "current_context_tokens": current_context_tokens,
        "snapshot_concurrency_semantics": STOCK_CONCURRENCY_SEMANTICS,
        "snapshot_prefill_count": snapshot_prefill_count,
        "snapshot_decode_count": snapshot_decode_count,
        "snapshot_running_count": len(running_ids),
        "snapshot_waiting_count": len(waiting_ids),
        "snapshot_running_prefill_count": len(running_prefill_ids),
        "snapshot_running_decode_count": len(running_decode_ids),
        "snapshot_waiting_prefill_count": len(waiting_prefill_ids),
        "snapshot_waiting_decode_count": len(waiting_decode_ids),
        "snapshot_preempted_count": len(preempted_ids),
        "snapshot_other_waiting_count": len(other_waiting_ids),
        "snapshot_requests_with_preemptions_count": len(requests_with_preemptions),
        "snapshot_total_preemptions": total_preemptions,
        "snapshot_running_request_ids": tuple(sorted(running_ids)),
        "snapshot_waiting_request_ids": tuple(sorted(waiting_ids)),
        "snapshot_running_prefill_request_ids": tuple(running_prefill_ids),
        "snapshot_running_decode_request_ids": tuple(running_decode_ids),
        "snapshot_waiting_prefill_request_ids": tuple(waiting_prefill_ids),
        "snapshot_waiting_decode_request_ids": tuple(waiting_decode_ids),
        "snapshot_preempted_request_ids": tuple(preempted_ids),
        "snapshot_other_waiting_request_ids": tuple(other_waiting_ids),
    }


def build_stock_profile_record(
    *,
    run_id: str,
    iteration_index: int,
    captured_state: dict[str, Any],
    scheduler_output: Any,
) -> dict[str, Any] | None:
    """Bind one unchanged stock decision to its pre-iteration request state."""
    if int(scheduler_output.total_num_scheduled_tokens) == 0:
        return None

    new_request_ids = {
        request.req_id for request in scheduler_output.scheduled_new_reqs
    }
    contexts = captured_state["current_context_tokens"]
    selected_requests: list[dict[str, Any]] = []
    for request_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
        if request_id not in contexts:
            raise RuntimeError(f"scheduled request missing from captured state: {request_id}")
        is_prefill = request_id in new_request_ids or (
            scheduler_output.scheduled_cached_reqs.is_context_phase(request_id)
        )
        selected_requests.append(
            {
                "request_id": request_id,
                "phase": "prefill" if is_prefill else "decode",
                "current_context_tokens": int(contexts[request_id]),
                "scheduled_tokens": int(scheduled_tokens),
            }
        )

    if not selected_requests:
        raise RuntimeError("non-empty stock iteration has no selected requests")
    if sum(item["scheduled_tokens"] for item in selected_requests) != int(
        scheduler_output.total_num_scheduled_tokens
    ):
        raise RuntimeError("stock profile token total does not match SchedulerOutput")

    plan_hash = _canonical_sha256(selected_requests)
    return {
        "schema_version": STOCK_PROFILE_SCHEMA_VERSION,
        "profile_kind": "stock_natural_workload",
        "run_id": run_id,
        "iteration_index": iteration_index,
        "plan_id": f"stock-{plan_hash[:24]}",
        "snapshot_hash": captured_state["snapshot_hash"],
        "snapshot_concurrency_semantics": captured_state[
            "snapshot_concurrency_semantics"
        ],
        "snapshot_prefill_count": int(
            captured_state["snapshot_prefill_count"]
        ),
        "snapshot_decode_count": int(captured_state["snapshot_decode_count"]),
        "snapshot_running_count": int(captured_state["snapshot_running_count"]),
        "snapshot_waiting_count": int(captured_state["snapshot_waiting_count"]),
        "snapshot_running_prefill_count": int(
            captured_state["snapshot_running_prefill_count"]
        ),
        "snapshot_running_decode_count": int(
            captured_state["snapshot_running_decode_count"]
        ),
        "snapshot_waiting_prefill_count": int(
            captured_state["snapshot_waiting_prefill_count"]
        ),
        "snapshot_waiting_decode_count": int(
            captured_state["snapshot_waiting_decode_count"]
        ),
        "snapshot_preempted_count": int(captured_state["snapshot_preempted_count"]),
        "snapshot_other_waiting_count": int(
            captured_state["snapshot_other_waiting_count"]
        ),
        "snapshot_requests_with_preemptions_count": int(
            captured_state["snapshot_requests_with_preemptions_count"]
        ),
        "snapshot_total_preemptions": int(
            captured_state["snapshot_total_preemptions"]
        ),
        "snapshot_running_request_ids": list(
            captured_state["snapshot_running_request_ids"]
        ),
        "snapshot_waiting_request_ids": list(
            captured_state["snapshot_waiting_request_ids"]
        ),
        "snapshot_running_prefill_request_ids": list(
            captured_state["snapshot_running_prefill_request_ids"]
        ),
        "snapshot_running_decode_request_ids": list(
            captured_state["snapshot_running_decode_request_ids"]
        ),
        "snapshot_waiting_prefill_request_ids": list(
            captured_state["snapshot_waiting_prefill_request_ids"]
        ),
        "snapshot_waiting_decode_request_ids": list(
            captured_state["snapshot_waiting_decode_request_ids"]
        ),
        "snapshot_preempted_request_ids": list(
            captured_state["snapshot_preempted_request_ids"]
        ),
        "snapshot_other_waiting_request_ids": list(
            captured_state["snapshot_other_waiting_request_ids"]
        ),
        "selected_requests": selected_requests,
    }


def build_target_profile_record(
    *,
    run_id: str,
    iteration_index: int,
    snapshot: StateSnapshot,
    plan: BatchPlan,
    recipe: Any,
    realized_shape: dict[str, Any],
) -> dict[str, Any]:
    """Bind a targeted exact plan to its pre-iteration observable state."""
    prefill_context = {
        request.request_id: request.prefilled_tokens
        for request in snapshot.waiting_prefill_requests
    }
    decode_context = {
        request.request_id: request.kv_context_length
        for request in snapshot.active_decode_requests
    }
    selected_requests = [
        {
            "request_id": request_id,
            "phase": "prefill",
            "current_context_tokens": int(prefill_context[request_id]),
            "scheduled_tokens": int(tokens),
        }
        for request_id, tokens in plan.prefill_items
    ]
    selected_requests.extend(
        {
            "request_id": request_id,
            "phase": "decode",
            "current_context_tokens": int(decode_context[request_id]),
            "scheduled_tokens": 1,
        }
        for request_id in plan.decode_items
    )
    if not selected_requests:
        raise RuntimeError("targeted exact plan cannot be empty")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "iteration_index": iteration_index,
        "plan_id": plan.plan_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "sample_role": "target",
        "recipe_id": recipe.recipe_id,
        "requested_shape": recipe.as_dict(),
        "realized_shape": realized_shape,
        "selected_requests": selected_requests,
    }


def _selected_shape(record: dict[str, Any]) -> dict[str, int | str]:
    selected = record["selected_requests"]
    prefills = [request for request in selected if request["phase"] == "prefill"]
    decodes = [request for request in selected if request["phase"] == "decode"]
    if prefills and decodes:
        kind = "mixed"
    elif prefills:
        kind = "prefill_only"
    else:
        kind = "decode_only"
    return {
        "batch_kind": kind,
        "prefill_requests": len(prefills),
        "prefill_tokens": sum(int(request["scheduled_tokens"]) for request in prefills),
        "decode_requests": len(decodes),
    }


def build_shadow_plan(
    *, snapshot: StateSnapshot, scheduler_output: Any
) -> BatchPlan | None:
    """Convert one unchanged Stock SchedulerOutput into an auditable BatchPlan."""
    if int(scheduler_output.total_num_scheduled_tokens) == 0:
        return None
    prefill_ids = {
        request.request_id for request in snapshot.waiting_prefill_requests
    }
    decode_ids = {
        request.request_id for request in snapshot.active_decode_requests
    }
    prefill_items: list[tuple[str, int]] = []
    decode_items: list[str] = []
    for request_id, token_count in scheduler_output.num_scheduled_tokens.items():
        if request_id in prefill_ids:
            prefill_items.append((request_id, int(token_count)))
        elif request_id in decode_ids:
            if int(token_count) != 1:
                raise RuntimeError("shadow Decode work must schedule exactly one token")
            decode_items.append(request_id)
        else:
            raise RuntimeError(f"shadow scheduled unknown request: {request_id}")
    payload = {
        "snapshot_hash": snapshot.snapshot_hash,
        "prefill_items": prefill_items,
        "decode_items": decode_items,
    }
    return BatchPlan(
        plan_id=f"shadow-{_canonical_sha256(payload)[:24]}",
        snapshot_hash=snapshot.snapshot_hash,
        template_id="SHADOW_STOCK_EXECUTED",
        prefill_items=tuple(prefill_items),
        decode_items=tuple(decode_items),
        total_prefill_tokens=sum(tokens for _, tokens in prefill_items),
        total_decode_tokens=len(decode_items),
        total_sequences=project_sequence_count(snapshot, tuple(prefill_items)),
        projected_kv_blocks=project_kv_blocks(
            snapshot, tuple(prefill_items), tuple(decode_items)
        ),
        mandatory_request_ids=(),
    )


class ExactPlanAdapter(ABC):
    """Adapter contract: make a snapshot and atomically execute a BatchPlan."""

    @abstractmethod
    def make_snapshot(self) -> StateSnapshot:
        raise NotImplementedError

    @abstractmethod
    def execute_exact_plan(self, plan: BatchPlan) -> ExecutionObservation:
        raise NotImplementedError


@dataclass(frozen=True)
class CallbackVllmAdapter(ExactPlanAdapter):
    """Adapter useful for tests and for wrapping a future vLLM binding."""

    snapshot_factory: Callable[[], StateSnapshot]
    executor: Callable[[BatchPlan], ExecutionObservation]

    def make_snapshot(self) -> StateSnapshot:
        return self.snapshot_factory()

    def execute_exact_plan(self, plan: BatchPlan) -> ExecutionObservation:
        observation = self.executor(plan)
        validate_snapshot_hash(plan.snapshot_hash, observation.snapshot_hash)
        if observation.frame_id < 0:
            raise RuntimeError("execution observation has invalid frame_id")
        if not observation.matches(plan):
            raise RuntimeError(
                f"selected plan {plan.plan_id} does not match executed plan "
                f"{observation.executed_plan_id}"
            )
        return observation


class VllmAdapter(ExactPlanAdapter):
    """Commit-specific vLLM adapter backed by a live ``Scheduler`` object.

    The adapter builds an immutable ``StateSnapshot`` from the scheduler's
    current request/KV state and submits the exact selected ``BatchPlan`` by
    materializing a vLLM ``SchedulerOutput`` directly.  Prefix caching and
    speculative decoding are disabled in this design, so the implementation
    only needs the simple prefill/decode path.
    """

    def __init__(
        self,
        scheduler: Any,
        *,
        frame_start: int = 1,
        obligation_ledger: ObligationLedger | None = None,
        critical_horizon_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._scheduler = scheduler
        self._frame = frame_start
        self._last_snapshot: StateSnapshot | None = None
        self._poisoned = False
        self._obligation_ledger = obligation_ledger
        self._critical_horizon_seconds = critical_horizon_seconds
        self._clock = clock
        self._prepared_snapshot_timestamp: float | None = None

    def expire_obligations_before_snapshot(self) -> tuple[Any, ...]:
        """Settle misses and pin the same timestamp for the next snapshot."""
        now = self._clock()
        self._prepared_snapshot_timestamp = now
        ledger = self._obligation_ledger
        if ledger is None:
            return ()
        return tuple(ledger.expire_deadlines(now))

    def _hard_ttft_protected(self, ledger_view: Any, now: float) -> bool:
        if ledger_view is None or not ledger_view.goodput_eligible:
            return False
        deadline = ledger_view.ttft_deadline
        horizon = self._critical_horizon_seconds
        return (
            deadline is not None
            and horizon is not None
            and deadline - now <= horizon
        )

    def make_snapshot(self) -> StateSnapshot:
        scheduler = self._require_scheduler()

        requests = scheduler.requests
        running = scheduler.running
        block_pool = scheduler.kv_cache_manager.block_pool
        block_size = scheduler.block_size
        token_budget = scheduler.max_num_scheduled_tokens
        sequence_budget = scheduler.max_num_running_reqs
        frame_id = self._frame
        self._frame += 1
        now = self._prepared_snapshot_timestamp
        if now is None:
            now = self._clock()
        self._prepared_snapshot_timestamp = None

        ledger_views = {}
        if self._obligation_ledger is not None:
            for request_id in requests:
                ledger_views[request_id] = self._obligation_ledger.request_view(
                    request_id, now
                )

        waiting: list[PrefillRequest] = []
        decode: list[DecodeRequest] = []
        ordinal = 0

        for request in running:
            if str(request.status) != "RUNNING":
                raise RuntimeError("running queue contains a non-RUNNING request")
            if request.num_computed_tokens < request.num_prompt_tokens:
                ledger_view = ledger_views.get(request.request_id)
                waiting.append(
                    PrefillRequest(
                        request_id=request.request_id,
                        arrival_time=request.arrival_time,
                        token_count=request.num_prompt_tokens,
                        prefilled_tokens=request.num_computed_tokens,
                        ttft_deadline=(
                            ledger_view.ttft_deadline if ledger_view else None
                        ),
                        hard_ttft_protected=self._hard_ttft_protected(
                            ledger_view, now
                        ),
                        is_running=True,
                        ordinal=ordinal,
                        goodput_eligible=(
                            ledger_view.goodput_eligible if ledger_view else True
                        ),
                        ttft_slo_seconds=(
                            self._obligation_ledger.ttft_slo_seconds
                            if self._obligation_ledger is not None
                            else 2.0
                        ),
                    )
                )
            else:
                ledger_view = ledger_views.get(request.request_id)
                decode.append(
                    DecodeRequest(
                        request_id=request.request_id,
                        arrival_time=request.arrival_time,
                        kv_context_length=request.num_computed_tokens,
                        tbt_deadline=(
                            ledger_view.tbt_deadline if ledger_view else None
                        ),
                        recovery_due=(
                            ledger_view.recovery_due if ledger_view else False
                        ),
                        recovery_first_miss_time=(
                            ledger_view.recovery_first_miss_time
                            if ledger_view
                            else None
                        ),
                        mandatory=(
                            ledger_view.recovery_due if ledger_view else False
                        ),
                        ordinal=ordinal,
                        goodput_eligible=(
                            ledger_view.goodput_eligible if ledger_view else True
                        ),
                        tbt_slo_seconds=(
                            self._obligation_ledger.tbt_slo_seconds
                            if self._obligation_ledger is not None
                            else 0.25
                        ),
                    )
                )
            ordinal += 1

        running_ids = {request.request_id for request in running}
        waiting_queue = tuple(scheduler.waiting)
        waiting_ids = {request.request_id for request in waiting_queue}
        if running_ids.intersection(waiting_ids):
            raise RuntimeError("vLLM request appears in both running and waiting")
        if set(requests) != running_ids | waiting_ids:
            raise RuntimeError(
                "unsupported vLLM request state outside running/waiting queues"
            )
        for request in waiting_queue:
            if str(request.status) != "WAITING" or request.num_computed_tokens != 0:
                raise RuntimeError(
                    "non-plain waiting/preempted requests require the G4 Recovery path"
                )
            ledger_view = ledger_views.get(request.request_id)
            waiting.append(
                PrefillRequest(
                    request_id=request.request_id,
                    arrival_time=request.arrival_time,
                    token_count=request.num_prompt_tokens,
                    prefilled_tokens=0,
                    ttft_deadline=(
                        ledger_view.ttft_deadline if ledger_view else None
                    ),
                    hard_ttft_protected=self._hard_ttft_protected(
                        ledger_view, now
                    ),
                    is_running=False,
                    ordinal=ordinal,
                    goodput_eligible=(
                        ledger_view.goodput_eligible if ledger_view else True
                    ),
                    ttft_slo_seconds=(
                        self._obligation_ledger.ttft_slo_seconds
                        if self._obligation_ledger is not None
                        else 2.0
                    ),
                )
            )
            ordinal += 1

        free_blocks = block_pool.get_num_free_blocks()
        total_blocks = block_pool.num_gpu_blocks
        active_ttft = ()
        active_tbt = ()
        recovery_requests = ()
        if self._obligation_ledger is not None:
            live_ids = set(requests)
            active_ttft, active_tbt = self._obligation_ledger.active_obligations(
                live_ids
            )
            recovery_requests = tuple(
                request_id
                for request_id in sorted(live_ids)
                if ledger_views[request_id].recovery
            )

        snapshot = StateSnapshot.create(
            frame_id=frame_id,
            timestamp=now,
            waiting_prefill_requests=tuple(waiting),
            active_decode_requests=tuple(decode),
            active_ttft_obligations=active_ttft,
            active_tbt_obligations=active_tbt,
            recovery_requests=recovery_requests,
            free_kv_blocks=free_blocks,
            kv_block_size=block_size,
            token_budget=token_budget,
            sequence_budget=sequence_budget,
            total_kv_blocks=total_blocks,
            provenance="vllm-live",
        )
        self._last_snapshot = snapshot
        return snapshot

    def _validate_plan_against_live_state(self, plan: BatchPlan) -> StateSnapshot:
        snapshot = self._last_snapshot
        if snapshot is None:
            raise RuntimeError("build_scheduler_output requires a fresh snapshot")
        plan.validate_snapshot(snapshot)
        if len({request_id for request_id, _ in plan.prefill_items}) != len(
            plan.prefill_items
        ):
            raise ValueError("duplicate prefill request in BatchPlan")
        if len(set(plan.decode_items)) != len(plan.decode_items):
            raise ValueError("duplicate decode request in BatchPlan")
        if {request_id for request_id, _ in plan.prefill_items}.intersection(
            plan.decode_items
        ):
            raise ValueError("request cannot be both Prefill and Decode")
        if any(token_count <= 0 for _, token_count in plan.prefill_items):
            raise ValueError("prefill token count must be positive")
        if sum(count for _, count in plan.prefill_items) != plan.total_prefill_tokens:
            raise ValueError("BatchPlan total_prefill_tokens mismatch")
        if len(plan.decode_items) != plan.total_decode_tokens:
            raise ValueError("BatchPlan total_decode_tokens mismatch")
        if plan.total_prefill_tokens + plan.total_decode_tokens > snapshot.token_budget:
            raise ValueError("BatchPlan exceeds token budget")
        projected_sequences = project_sequence_count(snapshot, plan.prefill_items)
        if projected_sequences != plan.total_sequences:
            raise ValueError("BatchPlan total_sequences mismatch")
        if projected_sequences > snapshot.sequence_budget:
            raise ValueError("BatchPlan exceeds sequence budget")
        projected_kv = project_kv_blocks(
            snapshot, plan.prefill_items, plan.decode_items
        )
        if projected_kv != plan.projected_kv_blocks:
            raise ValueError("BatchPlan projected_kv_blocks mismatch")
        if projected_kv > snapshot.total_kv_blocks:
            raise ValueError("BatchPlan exceeds current KV capacity")
        prefill_by_id = {
            request.request_id: request
            for request in snapshot.waiting_prefill_requests
        }
        decode_by_id = {
            request.request_id: request
            for request in snapshot.active_decode_requests
        }
        for request_id, token_count in plan.prefill_items:
            expected = prefill_by_id.get(request_id)
            if expected is None or token_count > expected.remaining_tokens:
                raise ValueError(f"invalid prefill work for {request_id}")
            live = self._scheduler.requests.get(request_id)
            if live is None or live.num_computed_tokens != expected.prefilled_tokens:
                raise RuntimeError(f"stale prefill state for {request_id}")
        for request_id in plan.decode_items:
            expected = decode_by_id.get(request_id)
            live = self._scheduler.requests.get(request_id)
            if expected is None or live is None:
                raise ValueError(f"invalid decode work for {request_id}")
            if live.num_computed_tokens != expected.kv_context_length:
                raise RuntimeError(f"stale decode state for {request_id}")
        return snapshot

    def build_scheduler_output(self, plan: BatchPlan) -> Any:
        """Build the exact vLLM SchedulerOutput for a BatchPlan.

        This is the operation the custom vLLM scheduler calls from its
        ``schedule()`` method.  It performs the same scheduler state updates as
        the stock path and returns the output that the engine feeds to the
        model runner.
        """
        scheduler = self._require_scheduler()
        snapshot = self._validate_plan_against_live_state(plan)
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            NewRequestData,
            SchedulerOutput,
        )
        from vllm.v1.request import RequestStatus

        scheduler.current_step += 1
        scheduler.kv_cache_manager.new_step_starts()

        scheduled_new_reqs: list[Any] = []
        scheduled_running_reqs: list[Any] = []
        req_to_new_blocks: dict[str, Any] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}

        snapshot_prefill = {
            request.request_id: request
            for request in snapshot.waiting_prefill_requests
        }

        # 1. Prefill items may be either first admission or another chunk of
        #    an already-running prompt. They map to different vLLM contracts.
        for request_id, token_count in plan.prefill_items:
            request = scheduler.requests.get(request_id)
            if request is None:
                raise RuntimeError(f"prefill request not found in scheduler: {request_id}")
            expected = snapshot_prefill[request_id]
            if request.num_computed_tokens != expected.prefilled_tokens:
                raise RuntimeError(f"stale prefill state for {request_id}")
            is_running = request in scheduler.running
            if is_running != expected.is_running:
                raise RuntimeError(f"prefill queue state changed for {request_id}")
            new_blocks = scheduler.kv_cache_manager.allocate_slots(
                request, token_count, has_scheduled_reqs=False
            )
            if new_blocks is None:
                self._poisoned = True
                raise RuntimeError(
                    f"KV allocation failed for prefill request {request_id}; "
                    "Adapter is poisoned because native allocation is not transactional"
                )
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = token_count

            if not is_running:
                try:
                    scheduler.waiting.remove_request(request)
                except ValueError:
                    raise RuntimeError(
                        f"new prefill request missing from waiting queue: {request_id}"
                    ) from None
                scheduler.running.append(request)
                scheduled_new_reqs.append(request)
            else:
                scheduled_running_reqs.append(request)
            request.status = RequestStatus.RUNNING

        # 2. Decode items: allocate one token slot for each selected running
        #    request.
        for request_id in plan.decode_items:
            request = scheduler.requests.get(request_id)
            if request is None:
                raise RuntimeError(f"decode request not found in scheduler: {request_id}")
            if request not in scheduler.running:
                raise RuntimeError(f"decode request is not running: {request_id}")
            new_blocks = scheduler.kv_cache_manager.allocate_slots(
                request, 1, has_scheduled_reqs=False
            )
            if new_blocks is None:
                self._poisoned = True
                raise RuntimeError(
                    f"KV allocation failed for decode request {request_id}; "
                    "Adapter is poisoned because native allocation is not transactional"
                )
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = 1
            scheduled_running_reqs.append(request)

        # 3. Build vLLM SchedulerOutput exactly from the plan.
        if scheduler.use_v2_model_runner:
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        cached_reqs_data: CachedRequestData = scheduler._make_cached_request_data(
            scheduled_running_reqs,
            [],
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        if not scheduler.use_v2_model_runner:
            scheduler.prev_step_scheduled_req_ids.clear()
            scheduler.prev_step_scheduled_req_ids.update(num_scheduled_tokens)

        total_scheduled = sum(num_scheduled_tokens.values())
        num_common_prefix_blocks = [0] * len(
            scheduler.kv_cache_config.kv_cache_groups
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_scheduled,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            scheduled_encoder_input_stats=None,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=scheduler.reset_preempted_req_ids,
            finished_req_ids=scheduler.finished_req_ids,
            free_encoder_mm_hashes=scheduler.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=scheduler._get_new_block_ids_to_zero(),
            ec_manager_metadata=scheduler.encoder_cache_manager.get_manager_metadata(),
        )

        scheduler._update_after_schedule(scheduler_output)
        self._last_snapshot = None
        return scheduler_output

    def execute_exact_plan(self, plan: BatchPlan) -> ExecutionObservation:
        if self._last_snapshot is None:
            raise RuntimeError("execute_exact_plan requires a fresh snapshot")
        frame_id = self._last_snapshot.frame_id
        start = time.time()
        self.build_scheduler_output(plan)
        # The plan is exact by construction: the SchedulerOutput was built from
        # exactly the prefill/decode items in this BatchPlan.
        return ExecutionObservation(
            frame_id=frame_id,
            snapshot_hash=plan.snapshot_hash,
            executed_plan_id=plan.plan_id,
            executed_prefill_items=plan.prefill_items,
            executed_decode_items=plan.decode_items,
            started_at=start,
            finished_at=time.time(),
        )

    def _require_scheduler(self) -> Any:
        if self._scheduler is None:
            raise RuntimeError("VllmAdapter requires a live vLLM scheduler")
        if self._poisoned:
            raise RuntimeError("VllmAdapter is poisoned after an allocation failure")
        scheduler = self._scheduler
        if getattr(scheduler.cache_config, "enable_prefix_caching", True):
            raise RuntimeError("Modular DPP requires prefix caching disabled")
        if getattr(scheduler, "num_spec_tokens", 1) != 0:
            raise RuntimeError("Modular DPP requires speculative decoding disabled")
        if getattr(scheduler.scheduler_config, "async_scheduling", True):
            raise RuntimeError("Modular DPP requires async scheduling disabled")
        if getattr(scheduler, "connector", None) is not None:
            raise RuntimeError("Modular DPP version 1 does not support KV connectors")
        return self._scheduler


def get_modular_scheduler_class() -> type:
    """Create the commit-specific vLLM subclass at the Adapter boundary."""
    from vllm.logger import init_logger
    from vllm.v1.core.sched.scheduler import Scheduler

    from dpp_scheduler.candidate_generator import CandidateGenerator
    from dpp_scheduler.consequence_estimator import ConsequenceEstimator
    from dpp_scheduler.dpp_selector import DPPSelector
    from dpp_scheduler.fallback import (
        DeterministicFallback,
        LIVENESS_ESCAPE_DECODE,
        LIVENESS_ESCAPE_PREFILL,
        PREEMPTION_REQUIRED_NATIVE_PROGRESS,
        build_liveness_escape,
        resolve_fallback,
    )
    from dpp_scheduler.observer import ProgressWatchdog
    from dpp_scheduler.predictor import RidgeDurationPredictor
    from dpp_scheduler.safe_set import ResourceAndRiskSafeSet
    from dpp_scheduler.state_store import (
        InMemoryStateStore,
        LedgerUpdate,
        ObligationLedger,
    )
    from benchmarks.qwen3_runtime import (
        ACTIVE_CONFIG_RELATIVE,
        REPOSITORY_ROOT,
        load_active_runtime,
        load_dpp_settings,
        load_predictor_settings,
        load_fallback_settings,
        load_frozen_candidate_settings,
        load_frozen_predictor,
        load_frozen_safe_set_settings,
        load_obligation_settings,
        load_scheduler_diagnostics_settings,
        validate_frozen_v2_artifacts,
    )

    # vLLM installs handlers on the ``vllm`` logger with propagation disabled.
    # Keep diagnostic messages below that namespace so INFO records reach the
    # configured EngineCore handler instead of disappearing at the root logger.
    logger = init_logger("vllm.dpp_scheduler")

    class ModularDPPScheduler(Scheduler):
        """Exact-plan development scheduler, disabled until inputs freeze."""

        # vLLM can emit one zero-token cleanup frame after a request reaches its
        # client length guard. Modular DPP binds feedback for that frame, so it
        # must receive the aligned context-manager duration even though vLLM's
        # official iteration-details logger intentionally omits zero-token work.
        _dpp_capture_zero_token_iteration_timing = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            install_vllm_iteration_timing_bridge()
            runtime = load_active_runtime(REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE)
            candidate = load_frozen_candidate_settings(runtime)
            safe_set_settings = load_frozen_safe_set_settings(runtime)
            fallback_settings = load_fallback_settings(runtime)
            obligation_settings = load_obligation_settings(runtime)
            dpp_settings = load_dpp_settings(runtime)
            predictor_settings = load_predictor_settings(runtime)
            diagnostics_settings = load_scheduler_diagnostics_settings(runtime)
            predictor = load_frozen_predictor(runtime)
            execution_scope = os.environ.get(DPP_EXECUTION_SCOPE_ENV, "")
            if not dpp_settings.live_v2_ready:
                raise RuntimeError(
                    "live v2 is disabled until Stock reference concurrency is frozen"
                )
            if not predictor_settings.live_v2_ready:
                raise RuntimeError(
                    "live v2 is disabled until held-out OOD calibration freezes kappa"
                )
            validate_frozen_v2_artifacts(
                runtime,
                dpp_settings=dpp_settings,
                predictor_settings=predictor_settings,
                predictor=predictor,
                execution_scope=execution_scope,
            )
            super().__init__(*args, **kwargs)
            self._dpp_obligation_ledger = ObligationLedger(
                ttft_slo_seconds=obligation_settings.ttft_slo_seconds,
                tbt_slo_seconds=obligation_settings.tbt_slo_seconds,
                recovery_age_threshold_seconds=(
                    obligation_settings.recovery_age_threshold_seconds
                ),
            )
            self._dpp_adapter = VllmAdapter(
                self,
                obligation_ledger=self._dpp_obligation_ledger,
                critical_horizon_seconds=None,
            )
            self._dpp_generator = CandidateGenerator(candidate.settings)
            self._dpp_consequence_estimator = ConsequenceEstimator()
            self._dpp_safe_set = ResourceAndRiskSafeSet(safe_set_settings)
            self._dpp_fallback = DeterministicFallback(fallback_settings)
            self._dpp_selector = DPPSelector(dpp_settings)
            self._dpp_state_store = InMemoryStateStore(settings=dpp_settings)
            self._dpp_predictor = RidgeDurationPredictor.from_artifact(
                predictor.artifact_root,
                ood_uncertainty_coefficient=(
                    predictor_settings.ood_uncertainty_coefficient
                ),
            )
            if (
                diagnostics_settings.performance_logging_enable_env
                != DPP_DIAGNOSTIC_ITERATION_LOG_ENV
            ):
                raise RuntimeError("diagnostic logging environment key mismatch")
            diagnostic_logging = os.environ.get(
                DPP_DIAGNOSTIC_ITERATION_LOG_ENV,
                "1" if diagnostics_settings.performance_logging_default else "0",
            )
            if diagnostic_logging not in {"0", "1"}:
                raise RuntimeError(
                    f"{DPP_DIAGNOSTIC_ITERATION_LOG_ENV} must be 0 or 1"
                )
            self._dpp_diagnostic_iteration_log = diagnostic_logging == "1"
            self._dpp_diagnostics_settings = diagnostics_settings
            self._dpp_progress_watchdog = ProgressWatchdog(
                max_records=diagnostics_settings.bounded_records,
                zero_progress_limit=(
                    diagnostics_settings.zero_progress_watchdog_iterations
                ),
                fail_fast=diagnostics_settings.fail_fast_development,
            )
            self._dpp_predictor_pending: dict[str, Any] | None = None
            self._dpp_obligation_updates: list[LedgerUpdate] = []
            self._dpp_control_pending_updates: list[LedgerUpdate] = []
            self._dpp_external_event_sequence = 0
            self._dpp_aggregate_path = os.environ.get(
                DPP_DIAGNOSTIC_AGGREGATE_PATH_ENV
            )
            self._dpp_aggregate = self._dpp_new_aggregate()

        def add_request(self, request: Any) -> None:
            is_new = request.request_id not in self.requests
            super().add_request(request)
            if is_new:
                self._dpp_obligation_ledger.register_request(
                    request.request_id, float(request.arrival_time)
                )
                self._dpp_state_store.record_prefill_arrival(
                    int(request.num_prompt_tokens)
                )

        @staticmethod
        def _dpp_terminal_reason(value: Any) -> str | None:
            if value is None:
                return None
            name = getattr(value, "name", None)
            return str(name if name is not None else value).lower()

        @staticmethod
        def _dpp_new_aggregate() -> dict[str, Any]:
            buckets = ("ZERO", "P25", "P50", "P75", "MAX", "FINISH", "OTHER")
            return {
                "selection_histogram": {b: 0 for b in buckets},
                "tie_selected_histogram": {b: 0 for b in buckets},
                "prefill_backlog_frame_count": 0,
                "tie_frame_count": 0,
                "mixed_iteration_count": 0,
                "decode_only_iteration_count": 0,
                "prefill_only_iteration_count": 0,
                "selected_prefill_progress": [],
                "selected_prefill_tokens": [],
                "actual_duration_seconds": {
                    "mixed": [],
                    "decode_only": [],
                    "prefill_only": [],
                },
                "max_selected_frames": {},
            }

        @staticmethod
        def _dpp_aggregate_bucket(template_id: str) -> str:
            parts = template_id.split(":")
            if len(parts) >= 2 and parts[1] in (
                "ZERO",
                "P25",
                "P50",
                "P75",
                "MAX",
                "FINISH",
            ):
                return parts[1]
            return "OTHER"

        @staticmethod
        def _dpp_percentile(values: list[float], p: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

        def _dpp_stats(self, values: list[float]) -> dict[str, Any]:
            return {
                "count": len(values),
                "mean": sum(values) / len(values) if values else None,
                "p50": self._dpp_percentile(values, 0.50),
                "p95": self._dpp_percentile(values, 0.95),
                "max": max(values) if values else None,
            }

        def _dpp_duration_stats(
            self,
            values: list[float],
            thresholds_ms: tuple[float, ...] = (250.0, 300.0, 500.0),
        ) -> dict[str, Any]:
            stats = self._dpp_stats(values)
            stats.update(
                {
                    "p90": self._dpp_percentile(values, 0.90),
                    "p99": self._dpp_percentile(values, 0.99),
                }
            )
            stats.update(
                {
                    f"gt{int(thr)}ms": sum(
                        1 for value in values if value * 1000.0 > thr
                    )
                    for thr in thresholds_ms
                }
            )
            return stats

        def _dpp_write_aggregate(self) -> None:
            path = self._dpp_aggregate_path
            if not path:
                return
            agg = self._dpp_aggregate
            max_frames = agg["max_selected_frames"]
            max_durations = [
                float(entry["actual_duration_seconds"])
                for entry in max_frames.values()
                if entry.get("actual_duration_seconds") is not None
            ]
            for field in (
                "sum_tbt_debt",
                "max_tbt_debt",
                "sum_ttft_debt",
                "max_ttft_debt",
            ):
                max_frames_by_id = max_frames
                values = [float(e[field]) for e in max_frames_by_id.values()]
                agg.setdefault("max_audit", {})[field + "_mean"] = (
                    sum(values) / len(values) if values else None
                )
                agg["max_audit"][field + "_max"] = max(values) if values else None
            payload = {
                "schema_version": 1,
                "kind": "dpp_diagnostic_aggregate",
                "selection_histogram": agg["selection_histogram"],
                "tie_selected_histogram": agg["tie_selected_histogram"],
                "prefill_backlog_frame_count": agg["prefill_backlog_frame_count"],
                "tie_frame_count": agg["tie_frame_count"],
                "mixed_iteration_count": agg["mixed_iteration_count"],
                "decode_only_iteration_count": agg["decode_only_iteration_count"],
                "prefill_only_iteration_count": agg["prefill_only_iteration_count"],
                "selected_prefill_progress": self._dpp_stats(
                    agg["selected_prefill_progress"]
                ),
                "selected_prefill_tokens": self._dpp_stats(
                    agg["selected_prefill_tokens"]
                ),
                "actual_duration_seconds": {
                    "mixed": self._dpp_duration_stats(
                        agg["actual_duration_seconds"]["mixed"]
                    ),
                    "decode_only": self._dpp_duration_stats(
                        agg["actual_duration_seconds"]["decode_only"]
                    ),
                    "prefill_only": self._dpp_duration_stats(
                        agg["actual_duration_seconds"]["prefill_only"]
                    ),
                },
                "max_audit": {
                    "count": len(max_frames),
                    "duration": self._dpp_duration_stats(max_durations),
                    **agg.get("max_audit", {}),
                },
            }
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=1)
            logger.info(
                "ModularDPPScheduler diagnostic aggregate=%s", path
            )

        def _dpp_accumulate_diagnostic_aggregate(
            self,
            *,
            snapshot: StateSnapshot,
            control: Any,
            plan: BatchPlan,
            selected_bucket: str,
            selected_prefill_progress: float | None,
            frame_tie: dict[str, Any],
        ) -> None:
            """Bounded in-memory aggregate for run-end diagnostic artifacts."""
            agg = self._dpp_aggregate
            agg["selection_histogram"][selected_bucket] += 1
            prefill_count = len(snapshot.waiting_prefill_requests)
            decode_count = len(snapshot.active_decode_requests)
            if prefill_count > 0:
                agg["prefill_backlog_frame_count"] += 1
            if prefill_count > 0 and decode_count > 0:
                agg["mixed_iteration_count"] += 1
            elif prefill_count > 0:
                agg["prefill_only_iteration_count"] += 1
            elif decode_count > 0:
                agg["decode_only_iteration_count"] += 1
            if frame_tie["winner_tie"]:
                agg["tie_frame_count"] += 1
                agg["tie_selected_histogram"][selected_bucket] += 1
            if selected_prefill_progress is not None:
                agg["selected_prefill_progress"].append(selected_prefill_progress)
                agg["selected_prefill_tokens"].append(
                    int(plan.total_prefill_tokens)
                )
            if selected_bucket == "MAX":
                agg["max_selected_frames"][snapshot.frame_id] = {
                    "frame_id": snapshot.frame_id,
                    "current_prefill_count": prefill_count,
                    "current_decode_count": decode_count,
                    "sum_ttft_debt": sum(
                        value for _, value in control.ttft_service_debts
                    ),
                    "max_ttft_debt": max(
                        (value for _, value in control.ttft_service_debts),
                        default=0.0,
                    ),
                    "sum_tbt_debt": sum(
                        value for _, value in control.tbt_service_debts
                    ),
                    "max_tbt_debt": max(
                        (value for _, value in control.tbt_service_debts),
                        default=0.0,
                    ),
                    "actual_duration_seconds": None,
                }

        def _dpp_record_obligation_update(self, update: LedgerUpdate) -> None:
            self._dpp_obligation_updates.append(update)
            del self._dpp_obligation_updates[
                : -self._dpp_diagnostics_settings.bounded_records
            ]
            self._dpp_control_pending_updates.append(update)

        def finish_requests(self, request_ids: Any, finished_status: Any) -> Any:
            finished = super().finish_requests(request_ids, finished_status)
            reason = self._dpp_terminal_reason(
                getattr(finished_status, "name", finished_status)
            )
            returned_at = time.time()
            for request in finished:
                if not self._dpp_obligation_ledger.has_request(request.request_id):
                    continue
                self._dpp_external_event_sequence += 1
                update = self._dpp_obligation_ledger.observe_output(
                    event_id=(
                        f"external:{self._dpp_external_event_sequence}:"
                        f"{request.request_id}"
                    ),
                    request_id=request.request_id,
                    returned_at=returned_at,
                    token_count=0,
                    terminal_reason=reason,
                )
                self._dpp_record_obligation_update(update)
            return finished

        def _dpp_record_iteration_duration(
            self,
            *,
            scheduler_output: Any,
            iteration_index: int | None,
            duration_seconds: float,
            timing_source: str,
        ) -> None:
            pending = self._dpp_predictor_pending
            if pending is None:
                raise RuntimeError("modular Predictor timing has no selected plan")
            if pending["scheduler_output"] is not scheduler_output:
                raise RuntimeError("modular Predictor timing output mismatch")
            if pending["actual_duration_seconds"] is not None:
                raise RuntimeError("duplicate modular Predictor iteration timing")
            pending["actual_duration_seconds"] = duration_seconds
            pending["actual_iteration_index"] = iteration_index
            pending["actual_timing_source"] = timing_source

        @staticmethod
        def _dpp_zero_plan(
            snapshot: StateSnapshot, reason: str
        ) -> BatchPlan:
            return BatchPlan(
                plan_id=f"zero-frame-{snapshot.frame_id}",
                snapshot_hash=snapshot.snapshot_hash,
                template_id=f"ZERO:{reason}",
                prefill_items=(),
                decode_items=(),
                total_prefill_tokens=0,
                total_decode_tokens=0,
                total_sequences=project_sequence_count(snapshot, ()),
                projected_kv_blocks=project_kv_blocks(snapshot, (), ()),
                mandatory_request_ids=(),
            )

        def _dpp_bind_pending(
            self,
            *,
            snapshot: StateSnapshot,
            plan: BatchPlan,
            scheduler_output: Any,
            base_duration_seconds: float | None,
            skip_predictor_update: bool,
        ) -> None:
            self._dpp_predictor_pending = {
                "snapshot": snapshot,
                "plan": plan,
                "base_duration_seconds": base_duration_seconds,
                "scheduler_output": scheduler_output,
                "actual_duration_seconds": None,
                "actual_iteration_index": None,
                "actual_timing_source": None,
                "skip_predictor_update": skip_predictor_update,
            }

        def _dpp_record_schedule_diagnostic(
            self,
            *,
            snapshot: StateSnapshot,
            control: Any,
            plans: tuple[BatchPlan, ...],
            safe_result: Any,
            decision: Decision,
            fallback_result: Any,
            plan: BatchPlan,
            scheduler_output: Any,
            predictor_in_support: bool,
            scheduler_cpu_seconds: float,
            candidate_scores: tuple[Any, ...] = (),
        ) -> None:
            candidate_score_records: list[dict[str, Any]] | None = None
            if self._dpp_diagnostic_iteration_log:
                candidates_by_plan_id = {
                    candidate.plan.plan_id: candidate
                    for candidate in safe_result.safe_candidates
                }
                ranked_scores = sorted(
                    candidate_scores,
                    key=lambda score: (
                        -score.score,
                        -score.prefill_progress,
                        score.effective_duration,
                        score.prefill_budget,
                        score.plan_id,
                    ),
                )
                rank_by_plan_id = {
                    score.plan_id: rank
                    for rank, score in enumerate(ranked_scores, start=1)
                }
                selected_scored_plan_id = (
                    decision.selected_plan.plan_id
                    if decision.reason == "DPP_V2_MAX_DRIFT_RATE"
                    and decision.selected_plan is not None
                    else None
                )
                winner_score = max(
                    (score.score for score in candidate_scores), default=None
                )
                candidate_score_records = []
                for score in candidate_scores:
                    candidate = candidates_by_plan_id[score.plan_id]
                    prediction = candidate.prediction
                    candidate_score_records.append(
                        {
                            "plan_id": score.plan_id,
                            "template_id": candidate.plan.template_id,
                            "prefill_items": list(candidate.plan.prefill_items),
                            "decode_request_ids": list(candidate.plan.decode_items),
                            "total_prefill_tokens": (
                                candidate.plan.total_prefill_tokens
                            ),
                            "total_decode_tokens": (
                                candidate.plan.total_decode_tokens
                            ),
                            "total_sequences": candidate.plan.total_sequences,
                            "projected_kv_blocks": (
                                candidate.plan.projected_kv_blocks
                            ),
                            "expected_duration_seconds": (
                                prediction.expected_duration
                            ),
                            "conservative_duration_seconds": (
                                prediction.conservative_duration
                            ),
                            "effective_duration_seconds": score.effective_duration,
                            "in_support": prediction.in_support,
                            "prediction_mode": prediction.prediction_mode,
                            "ood_distance": prediction.ood_distance,
                            "prefill_normalized_drift": score.prefill_drift,
                            "decode_normalized_drift": score.decode_drift,
                            "total_normalized_drift": score.total_drift,
                            "score": score.score,
                            "prefill_reference_concurrency": (
                                score.prefill_reference_concurrency
                            ),
                            "decode_reference_concurrency": (
                                score.decode_reference_concurrency
                            ),
                            "prefill_progress_utility": score.prefill_progress,
                            "score_tied_with_winner": (
                                winner_score is not None
                                and math.isclose(
                                    score.score,
                                    winner_score,
                                    rel_tol=self._dpp_selector.settings.score_rel_tol,
                                    abs_tol=self._dpp_selector.settings.score_abs_tol,
                                )
                            ),
                            "selection_rank": rank_by_plan_id[score.plan_id],
                            "selected": (
                                score.plan_id == selected_scored_plan_id
                            ),
                            "tie_break_key": {
                                "score_desc": score.score,
                                "prefill_progress_desc": score.prefill_progress,
                                "effective_duration_asc": score.effective_duration,
                                "prefill_budget_asc": score.prefill_budget,
                                "plan_id_asc": score.plan_id,
                            },
                        }
                    )

            selected_bucket = self._dpp_aggregate_bucket(plan.template_id)
            selected_prefill_progress: float | None = None
            frame_tie: dict[str, Any]
            if candidate_scores and decision.reason == "DPP_V2_MAX_DRIFT_RATE":
                winner_score = max(score.score for score in candidate_scores)
                tie_set = [
                    score
                    for score in candidate_scores
                    if math.isclose(
                        score.score,
                        winner_score,
                        rel_tol=self._dpp_selector.settings.score_rel_tol,
                        abs_tol=self._dpp_selector.settings.score_abs_tol,
                    )
                ]
                old_t0_winner = min(
                    tie_set,
                    key=lambda score: (
                        score.effective_duration,
                        score.prefill_budget,
                        score.plan_id,
                    ),
                )
                selected_score = next(
                    (
                        score
                        for score in candidate_scores
                        if score.plan_id == plan.plan_id
                    ),
                    None,
                )
                if selected_score is not None:
                    selected_prefill_progress = selected_score.prefill_progress
                frame_tie = {
                    "winner_tie_size": len(tie_set),
                    "winner_tie": len(tie_set) >= 2,
                    "tie_break_changed_winner": (
                        old_t0_winner.plan_id != plan.plan_id
                    ),
                }
            else:
                frame_tie = {
                    "winner_tie_size": None,
                    "winner_tie": False,
                    "tie_break_changed_winner": False,
                }
            record = self._dpp_progress_watchdog.record_iteration(
                workload_nonempty=bool(
                    snapshot.active_decode_requests
                    or snapshot.waiting_prefill_requests
                ),
                scheduled_tokens=int(scheduler_output.total_num_scheduled_tokens),
                prefill_tokens=plan.total_prefill_tokens,
                decode_tokens=plan.total_decode_tokens,
                diagnostic={
                    "frame_id": snapshot.frame_id,
                    "candidate_count": len(plans),
                    "safe_candidate_count": len(safe_result.safe_candidates),
                    "selected_plan": plan.plan_id,
                    "decision_reason": decision.reason,
                    "current_prefill_count": len(
                        snapshot.waiting_prefill_requests
                    ),
                    "current_decode_count": len(snapshot.active_decode_requests),
                    "prefill_reference_concurrency": (
                        self._dpp_selector.settings.prefill_reference_concurrency
                    ),
                    "decode_reference_concurrency": (
                        self._dpp_selector.settings.decode_reference_concurrency
                    ),
                    "sum_ttft_debt": sum(
                        value for _, value in control.ttft_service_debts
                    ),
                    "max_ttft_debt": max(
                        (value for _, value in control.ttft_service_debts),
                        default=0.0,
                    ),
                    "sum_tbt_debt": sum(
                        value for _, value in control.tbt_service_debts
                    ),
                    "max_tbt_debt": max(
                        (value for _, value in control.tbt_service_debts),
                        default=0.0,
                    ),
                    "number_of_extrapolated_candidates": sum(
                        candidate.prediction.prediction_mode
                        == "CONSTRAINED_EXTRAPOLATION"
                        for candidate in safe_result.safe_candidates
                    ),
                    "max_ood_distance": max(
                        (
                            candidate.prediction.ood_distance
                            for candidate in safe_result.safe_candidates
                        ),
                        default=0.0,
                    ),
                    "predictor_in_support": predictor_in_support,
                    "safe_set_rejections": safe_result.rejected,
                    "rejected_candidates_scoring_status": "not_scored",
                    "candidate_scores": candidate_score_records,
                    "selection_tie_break_order": (
                        "score_desc_with_isclose,prefill_progress_desc,"
                        "effective_duration_asc,prefill_budget_asc,plan_id_asc"
                    ),
                    "winner_tie_size": frame_tie["winner_tie_size"],
                    "winner_tie": frame_tie["winner_tie"],
                    "tie_break_changed_winner": frame_tie["tie_break_changed_winner"],
                    "selected_template": plan.template_id,
                    "selected_prefill_tokens": plan.total_prefill_tokens,
                    "selected_prefill_progress": selected_prefill_progress,
                    "fallback_reason": (
                        fallback_result.reason if fallback_result else None
                    ),
                    "scheduler_cpu_seconds": scheduler_cpu_seconds,
                },
            )
            if record["watchdog_triggered"]:
                logger.error("ModularDPPScheduler zero-progress dump=%s", record)
            elif self._dpp_diagnostic_iteration_log:
                logger.info("ModularDPPScheduler diagnostic=%s", record)
            self._dpp_accumulate_diagnostic_aggregate(
                snapshot=snapshot,
                control=control,
                plan=plan,
                selected_bucket=selected_bucket,
                selected_prefill_progress=selected_prefill_progress,
                frame_tie=frame_tie,
            )

        def schedule(self, throttle_prefills: bool = False) -> Any:
            del throttle_prefills
            scheduler_cpu_started = time.perf_counter()
            expiry_updates = (
                self._dpp_adapter.expire_obligations_before_snapshot()
            )
            for update in expiry_updates:
                self._dpp_record_obligation_update(update)
            snapshot = self._dpp_adapter.make_snapshot()
            control = self._dpp_state_store.bind_snapshot(snapshot)
            # Deadline/Goodput ledger events are diagnostic-only in v2.
            self._dpp_control_pending_updates = []
            plans = self._dpp_generator.generate(snapshot)
            predictions = self._dpp_predictor.predict(snapshot, plans)
            if len(predictions) != len(plans):
                raise RuntimeError("Predictor did not cover every BatchPlan")
            safe_result = self._dpp_safe_set.filter(snapshot, plans, predictions)
            if safe_result.snapshot_hash != snapshot.snapshot_hash:
                raise RuntimeError("Safe-Set snapshot_hash mismatch")
            # Candidate scores are always computed: the bounded diagnostic
            # aggregate needs them even when per-frame logging is disabled,
            # and scoring is pure arithmetic over at most six candidates.
            decision, candidate_scores = self._dpp_selector.select_with_audit(
                snapshot, control, safe_result.safe_candidates
            )
            fallback_result = None
            if decision.selected_plan is None:
                fallback_result = resolve_fallback(
                    snapshot,
                    self._dpp_fallback,
                    self._dpp_predictor,
                    self._dpp_safe_set,
                )
                if fallback_result.rejection_reasons:
                    fallback_result = build_liveness_escape(
                        snapshot, fallback_result
                    )
                decision = Decision(
                    frame_id=snapshot.frame_id,
                    snapshot_hash=snapshot.snapshot_hash,
                    selected_plan=fallback_result.plan,
                    reason=fallback_result.reason,
                )
            if decision.selected_plan is None:
                workload_nonempty = bool(
                    snapshot.active_decode_requests
                    or snapshot.waiting_prefill_requests
                )
                if workload_nonempty:
                    scheduler_output = super().schedule(False)
                    plan = build_shadow_plan(
                        snapshot=snapshot, scheduler_output=scheduler_output
                    ) or self._dpp_zero_plan(snapshot, decision.reason)
                    decision = Decision(
                        frame_id=snapshot.frame_id,
                        snapshot_hash=snapshot.snapshot_hash,
                        selected_plan=None,
                        reason=PREEMPTION_REQUIRED_NATIVE_PROGRESS,
                    )
                else:
                    plan = self._dpp_zero_plan(snapshot, decision.reason)
                    scheduler_output = self._dpp_adapter.build_scheduler_output(plan)
                self._dpp_bind_pending(
                    snapshot=snapshot,
                    plan=plan,
                    scheduler_output=scheduler_output,
                    base_duration_seconds=None,
                    skip_predictor_update=True,
                )
                self._dpp_record_schedule_diagnostic(
                    snapshot=snapshot,
                    control=control,
                    plans=plans,
                    safe_result=safe_result,
                    decision=decision,
                    fallback_result=fallback_result,
                    plan=plan,
                    scheduler_output=scheduler_output,
                    predictor_in_support=False,
                    scheduler_cpu_seconds=(
                        time.perf_counter() - scheduler_cpu_started
                    ),
                    candidate_scores=candidate_scores,
                )
                return scheduler_output

            liveness_escape = decision.reason in {
                LIVENESS_ESCAPE_DECODE,
                LIVENESS_ESCAPE_PREFILL,
            }
            if liveness_escape:
                scheduler_output = self._dpp_adapter.build_scheduler_output(
                    decision.selected_plan
                )
                base_duration = None
                predictor_in_support = False
            else:
                audit = self._dpp_predictor.predict_with_audit(
                    snapshot, decision.selected_plan
                )
                if audit.prediction.prediction_mode not in {
                    "INTERPOLATION",
                    "CONSTRAINED_EXTRAPOLATION",
                }:
                    raise RuntimeError("Safe-Set selected an invalid prediction")
                scheduler_output = self._dpp_adapter.build_scheduler_output(
                    decision.selected_plan
                )
                base_duration = audit.base_duration_seconds
                predictor_in_support = audit.prediction.in_support
            self._dpp_bind_pending(
                snapshot=snapshot,
                plan=decision.selected_plan,
                scheduler_output=scheduler_output,
                base_duration_seconds=base_duration,
                skip_predictor_update=(
                    liveness_escape or not predictor_in_support
                ),
            )
            self._dpp_record_schedule_diagnostic(
                snapshot=snapshot,
                control=control,
                plans=plans,
                safe_result=safe_result,
                decision=decision,
                fallback_result=fallback_result,
                plan=decision.selected_plan,
                scheduler_output=scheduler_output,
                predictor_in_support=predictor_in_support,
                scheduler_cpu_seconds=time.perf_counter() - scheduler_cpu_started,
                candidate_scores=candidate_scores,
            )
            return scheduler_output

        def update_from_output(
            self, scheduler_output: Any, model_runner_output: Any
        ) -> Any:
            pending = self._dpp_predictor_pending
            result = super().update_from_output(scheduler_output, model_runner_output)
            if pending is None:
                raise RuntimeError("modular Predictor feedback has no selected plan")
            returned_at = time.time()
            for client_index in sorted(result):
                outputs = result[client_index].outputs
                for output_index, output in enumerate(outputs):
                    token_count = len(output.new_token_ids)
                    terminal_reason = self._dpp_terminal_reason(output.finish_reason)
                    if token_count == 0 and terminal_reason is None:
                        continue
                    if not self._dpp_obligation_ledger.has_request(
                        output.request_id
                    ):
                        if token_count == 0 and terminal_reason is not None:
                            # finish_requests already recorded this terminal event.
                            continue
                        raise RuntimeError(
                            "actual token output has no active obligation ledger"
                        )
                    update = self._dpp_obligation_ledger.observe_output(
                        event_id=(
                            f"frame:{pending['snapshot'].frame_id}:"
                            f"client:{client_index}:output:{output_index}:"
                            f"{output.request_id}"
                        ),
                        request_id=output.request_id,
                        returned_at=returned_at,
                        token_count=token_count,
                        terminal_reason=terminal_reason,
                    )
                    self._dpp_record_obligation_update(update)
            prefill_ids = {
                request.request_id
                for request in pending["snapshot"].waiting_prefill_requests
            }
            actual_prefill_items = tuple(
                (request_id, int(tokens))
                for request_id, tokens in scheduler_output.num_scheduled_tokens.items()
                if request_id in prefill_ids
            )
            if actual_prefill_items != pending["plan"].prefill_items:
                raise RuntimeError(
                    "actual per-request Prefill service does not match BatchPlan"
                )
            feedback_updates = tuple(self._dpp_control_pending_updates)
            actual_decode_items = tuple(
                update.request_id
                for update in feedback_updates
                if update.tbt_service_tokens == 1
            )
            initialized_tbt = tuple(
                update.request_id
                for update in feedback_updates
                if update.initializes_tbt_service
            )
            terminal_request_ids = tuple(
                update.request_id
                for update in feedback_updates
                if update.terminal_reason is not None
            )
            actual_duration = pending["actual_duration_seconds"]
            if actual_duration is None:
                raise RuntimeError("modular Predictor iteration timing is missing")
            executed = pending["plan"]
            if executed.total_prefill_tokens > 0 and executed.total_decode_tokens > 0:
                self._dpp_aggregate["actual_duration_seconds"]["mixed"].append(
                    actual_duration
                )
            elif executed.total_prefill_tokens > 0:
                self._dpp_aggregate["actual_duration_seconds"]["prefill_only"].append(
                    actual_duration
                )
            elif executed.total_decode_tokens > 0:
                self._dpp_aggregate["actual_duration_seconds"]["decode_only"].append(
                    actual_duration
                )
            if self._dpp_aggregate_bucket(executed.template_id) == "MAX":
                max_entry = self._dpp_aggregate["max_selected_frames"].get(
                    pending["snapshot"].frame_id
                )
                if max_entry is not None:
                    max_entry["actual_duration_seconds"] = actual_duration
            next_control = self._dpp_state_store.update_from_actual(
                previous_snapshot=pending["snapshot"],
                actual_duration_seconds=actual_duration,
                executed_prefill_items=actual_prefill_items,
                executed_decode_items=actual_decode_items,
                initialized_tbt_request_ids=initialized_tbt,
                terminal_request_ids=terminal_request_ids,
            )
            if self._dpp_diagnostic_iteration_log:
                logger.info(
                    "ModularDPPScheduler feedback frame=%s scheduled_tokens=%s "
                    "actual_duration_seconds=%.9f timing_source=%s "
                    "iteration_index=%s actual_prefill=%s actual_decode=%s "
                    "initialized_tbt=%s terminal=%s ledger_events=%s "
                    "next_control=(%s,%.6f,%.6f)",
                    pending["snapshot"].frame_id,
                    int(scheduler_output.total_num_scheduled_tokens),
                    actual_duration,
                    pending["actual_timing_source"],
                    pending["actual_iteration_index"],
                    sum(tokens for _, tokens in actual_prefill_items),
                    len(actual_decode_items),
                    len(initialized_tbt),
                    len(terminal_request_ids),
                    len(self._dpp_control_pending_updates),
                    len(next_control.ttft_service_debts),
                    len(next_control.tbt_service_debts),
                    sum(value for _, value in next_control.tbt_service_debts),
                )
            self._dpp_control_pending_updates = []
            if pending["skip_predictor_update"]:
                self._dpp_predictor_pending = None
                return result
            self._dpp_predictor_pending = None
            self._dpp_predictor.observe_actual(
                pending["snapshot"],
                pending["plan"],
                actual_duration,
                base_duration_seconds=pending["base_duration_seconds"],
            )
            return result

        def shutdown(self) -> None:
            try:
                self._dpp_write_aggregate()
            except Exception:
                logger.exception(
                    "ModularDPPScheduler diagnostic aggregate write failed"
                )
            super().shutdown()

    ModularDPPScheduler.__module__ = "dpp_scheduler.vllm_scheduler"
    return ModularDPPScheduler


def get_predictor_evaluation_scheduler_class() -> type:
    """Create a real-vLLM shadow scheduler for online Predictor evaluation."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from dpp_scheduler.predictor import RidgeDurationPredictor
    from dpp_scheduler.targeted_profile import (
        TargetBatchPlanner,
        build_target_recipes,
    )

    class PredictorEvaluationScheduler(Scheduler):
        """Execute Stock/target batches unchanged while predicting them online."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            install_vllm_iteration_timing_bridge()
            super().__init__(*args, **kwargs)
            output_value = os.environ.get(PREDICTOR_EVAL_PATH_ENV)
            run_id = os.environ.get(PREDICTOR_EVAL_RUN_ID_ENV)
            artifact_value = os.environ.get(PREDICTOR_ARTIFACT_PATH_ENV)
            recipe_seed_value = os.environ.get(PREDICTOR_EVAL_RECIPE_SEED_ENV)
            recipe_mode = os.environ.get(PREDICTOR_EVAL_RECIPE_MODE_ENV, "formal")
            if not output_value or not run_id or not artifact_value:
                raise RuntimeError(
                    f"{PREDICTOR_EVAL_PATH_ENV}, {PREDICTOR_EVAL_RUN_ID_ENV}, "
                    f"and {PREDICTOR_ARTIFACT_PATH_ENV} are required"
                )
            if recipe_seed_value is None:
                raise RuntimeError(f"{PREDICTOR_EVAL_RECIPE_SEED_ENV} is required")
            output_path = Path(output_value)
            artifact_path = Path(artifact_value)
            if not output_path.is_absolute() or not artifact_path.is_absolute():
                raise RuntimeError("Predictor evaluation paths must be absolute")
            output_path = output_path.resolve()
            artifact_path = artifact_path.resolve()
            if not output_path.parent.is_dir():
                raise RuntimeError(
                    f"Predictor evaluation parent does not exist: {output_path.parent}"
                )
            self._predictor_eval_run_id = run_id
            self._predictor_eval_stream = output_path.open(
                "x", encoding="utf-8", buffering=1
            )
            self._predictor = RidgeDurationPredictor.from_artifact(artifact_path)
            self._predictor_eval_iteration = 0
            self._predictor_eval_pending: dict[str, Any] | None = None
            self._predictor_eval_adapter = VllmAdapter(self)
            self._predictor_eval_recipe_seed = int(recipe_seed_value)
            self._predictor_eval_recipe_mode = recipe_mode
            self._predictor_eval_planner = TargetBatchPlanner(
                build_target_recipes(
                    self._predictor_eval_recipe_seed, mode=recipe_mode
                )
            )

        def _selected_requests(
            self, snapshot: StateSnapshot, plan: BatchPlan
        ) -> list[dict[str, Any]]:
            prefill = {
                request.request_id: request.prefilled_tokens
                for request in snapshot.waiting_prefill_requests
            }
            decode = {
                request.request_id: request.kv_context_length
                for request in snapshot.active_decode_requests
            }
            selected = [
                {
                    "request_id": request_id,
                    "phase": "prefill",
                    "current_context_tokens": int(prefill[request_id]),
                    "scheduled_tokens": int(tokens),
                }
                for request_id, tokens in plan.prefill_items
            ]
            selected.extend(
                {
                    "request_id": request_id,
                    "phase": "decode",
                    "current_context_tokens": int(decode[request_id]),
                    "scheduled_tokens": 1,
                }
                for request_id in plan.decode_items
            )
            return selected

        def _prepare_pending(
            self,
            *,
            snapshot: StateSnapshot,
            plan: BatchPlan,
            sample_role: str,
            recipe: Any | None,
            realized_shape: dict[str, Any] | None,
        ) -> None:
            if self._predictor_eval_pending is not None:
                raise RuntimeError("prior Predictor evaluation iteration is unfinished")
            prediction_started = time.perf_counter()
            audit = self._predictor.predict_with_audit(snapshot, plan)
            predictor_cpu_seconds = time.perf_counter() - prediction_started
            self._predictor_eval_pending = {
                "snapshot": snapshot,
                "plan": plan,
                "audit": audit,
                "predictor_cpu_seconds": predictor_cpu_seconds,
                "sample_role": sample_role,
                "recipe": recipe,
                "realized_shape": realized_shape,
                "scheduler_output": None,
                "official_iteration_index": None,
                "actual_duration_seconds": None,
                "timing_source": None,
            }

        def _bind_scheduler_output(self, scheduler_output: Any) -> None:
            pending = self._predictor_eval_pending
            if pending is None:
                raise RuntimeError("Predictor evaluation output has no pending plan")
            if pending["scheduler_output"] is not None:
                raise RuntimeError("Predictor evaluation output is already bound")
            pending["scheduler_output"] = scheduler_output

        def _dpp_record_iteration_duration(
            self,
            *,
            scheduler_output: Any,
            iteration_index: int | None,
            duration_seconds: float,
            timing_source: str,
        ) -> None:
            pending = self._predictor_eval_pending
            if pending is None:
                raise RuntimeError("Predictor evaluation timing has no pending plan")
            if pending["scheduler_output"] is not scheduler_output:
                raise RuntimeError("Predictor evaluation timing output mismatch")
            if pending["actual_duration_seconds"] is not None:
                raise RuntimeError("duplicate Predictor evaluation iteration timing")
            pending["official_iteration_index"] = iteration_index
            pending["actual_duration_seconds"] = duration_seconds
            pending["timing_source"] = timing_source

        def schedule(self, throttle_prefills: bool = False) -> Any:
            snapshot = self._predictor_eval_adapter.make_snapshot()
            if not self._predictor_eval_planner.complete:
                built = self._predictor_eval_planner.build(snapshot)
                if built is not None:
                    plan, realized_shape = built
                    scheduler_output = self._predictor_eval_adapter.build_scheduler_output(
                        plan
                    )
                    recipe = self._predictor_eval_planner.current
                    assert recipe is not None
                    self._predictor_eval_planner.advance()
                    self._prepare_pending(
                        snapshot=snapshot,
                        plan=plan,
                        sample_role="target",
                        recipe=recipe,
                        realized_shape=realized_shape,
                    )
                    self._bind_scheduler_output(scheduler_output)
                    return scheduler_output

            scheduler_output = super().schedule(throttle_prefills)
            plan = build_shadow_plan(
                snapshot=snapshot, scheduler_output=scheduler_output
            )
            if plan is not None:
                self._prepare_pending(
                    snapshot=snapshot,
                    plan=plan,
                    sample_role=(
                        "drain" if self._predictor_eval_planner.complete else "setup"
                    ),
                    recipe=None,
                    realized_shape=None,
                )
                self._bind_scheduler_output(scheduler_output)
            return scheduler_output

        def update_from_output(
            self, scheduler_output: Any, model_runner_output: Any
        ) -> Any:
            pending = self._predictor_eval_pending
            result = super().update_from_output(scheduler_output, model_runner_output)
            if pending is None:
                if int(scheduler_output.total_num_scheduled_tokens) != 0:
                    raise RuntimeError("non-empty iteration has no Predictor evaluation")
                return result
            self._predictor_eval_pending = None
            actual_duration = pending["actual_duration_seconds"]
            if actual_duration is None:
                raise RuntimeError("Predictor evaluation iteration timing is missing")
            official_iteration_index = pending["official_iteration_index"]
            if official_iteration_index is None:
                raise RuntimeError("official vLLM iteration timing is required for evaluation")
            if official_iteration_index != self._predictor_eval_iteration:
                raise RuntimeError("Predictor evaluation/official iteration index mismatch")
            if pending["timing_source"] != VLLM_OFFICIAL_ITERATION_TIMING:
                raise RuntimeError("Predictor evaluation did not receive official timing")
            snapshot = pending["snapshot"]
            plan = pending["plan"]
            audit = pending["audit"]
            update = None
            update_error = None
            if audit.prediction.in_support:
                try:
                    update = self._predictor.observe_actual(
                        snapshot,
                        plan,
                        actual_duration,
                        base_duration_seconds=audit.base_duration_seconds,
                    )
                except Exception as error:
                    update_error = f"{type(error).__name__}: {error}"
            record = {
                "schema_version": 2,
                "run_id": self._predictor_eval_run_id,
                "iteration_index": self._predictor_eval_iteration,
                "frame_id": snapshot.frame_id,
                "snapshot_hash": snapshot.snapshot_hash,
                "plan_id": plan.plan_id,
                "sample_role": pending["sample_role"],
                "batch_kind": audit.batch_kind,
                "in_support": audit.prediction.in_support,
                "prediction_mode": audit.prediction.prediction_mode,
                "ood_distance": audit.prediction.ood_distance,
                "base_duration_seconds": audit.base_duration_seconds,
                "expected_duration_seconds": audit.prediction.expected_duration,
                "centered_residual_p95_seconds": (
                    audit.centered_residual_p95_seconds
                ),
                "conservative_duration_seconds": (
                    audit.prediction.conservative_duration
                ),
                "actual_duration_seconds": actual_duration,
                "timing_source": pending["timing_source"],
                "timing_boundary": (
                    "after_execute_model_submission_through_model_result_and_sampling"
                ),
                "residual_seconds": (
                    update.residual_seconds if update is not None else None
                ),
                "calibration_source": audit.calibration_source,
                "calibration_samples_before": audit.calibration_sample_count,
                "calibration_samples_after": (
                    update.samples_after
                    if update is not None
                    else audit.calibration_sample_count
                ),
                "calibration_updated": update is not None,
                "predictor_cpu_seconds": pending["predictor_cpu_seconds"],
                "rejection_reason": audit.rejection_reason or update_error,
                "predictor_version": audit.prediction.predictor_version,
                "selected_requests": self._selected_requests(snapshot, plan),
                "recipe_seed": self._predictor_eval_recipe_seed,
                "recipe_mode": self._predictor_eval_recipe_mode,
            }
            recipe = pending["recipe"]
            if recipe is not None:
                record["recipe_id"] = recipe.recipe_id
                record["requested_shape"] = recipe.as_dict()
                record["realized_shape"] = pending["realized_shape"]
            self._predictor_eval_stream.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            self._predictor_eval_stream.flush()
            self._predictor_eval_iteration += 1
            return result

    PredictorEvaluationScheduler.__module__ = (
        "dpp_scheduler.predictor_evaluation_scheduler"
    )
    return PredictorEvaluationScheduler


def get_stock_profiling_scheduler_class() -> type:
    """Create a stock Scheduler subclass that only records executed batches."""
    from vllm.v1.core.sched.scheduler import Scheduler

    class StockProfilingScheduler(Scheduler):
        """Pass-through stock Scheduler with append-only profiling telemetry."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            output_value = os.environ.get(STOCK_PROFILE_PATH_ENV)
            run_id = os.environ.get(STOCK_PROFILE_RUN_ID_ENV)
            if not output_value or not run_id:
                raise RuntimeError(
                    f"{STOCK_PROFILE_PATH_ENV} and {STOCK_PROFILE_RUN_ID_ENV} "
                    "are required for StockProfilingScheduler"
                )
            output_path = Path(output_value)
            if not output_path.is_absolute():
                raise RuntimeError(f"stock profile path must be absolute: {output_path}")
            output_path = output_path.resolve()
            if not output_path.parent.is_dir():
                raise RuntimeError(
                    f"stock profile parent does not exist: {output_path.parent}"
                )
            self._stock_profile_run_id = run_id
            self._stock_profile_iteration = 0
            self._stock_profile_stream = output_path.open(
                "x", encoding="utf-8", buffering=1
            )

        def schedule(self, throttle_prefills: bool = False) -> Any:
            captured_state = capture_stock_profile_state(self)
            scheduler_output = super().schedule(throttle_prefills)
            record = build_stock_profile_record(
                run_id=self._stock_profile_run_id,
                iteration_index=self._stock_profile_iteration,
                captured_state=captured_state,
                scheduler_output=scheduler_output,
            )
            if record is not None:
                self._stock_profile_stream.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                self._stock_profile_stream.flush()
                self._stock_profile_iteration += 1
            return scheduler_output

    StockProfilingScheduler.__module__ = "dpp_scheduler.stock_profile_scheduler"
    return StockProfilingScheduler


def get_targeted_profiling_scheduler_class() -> type:
    """Create an experiment-only scheduler for real targeted BatchPlans."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from dpp_scheduler.targeted_profile import (
        TargetBatchPlanner,
        build_target_recipes,
    )

    class TargetedProfilingScheduler(Scheduler):
        """Execute target recipes exactly and use Stock only for setup/drain."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            output_value = os.environ.get(TARGET_PROFILE_PATH_ENV)
            run_id = os.environ.get(TARGET_PROFILE_RUN_ID_ENV)
            seed_value = os.environ.get(TARGET_PROFILE_RECIPE_SEED_ENV)
            recipe_mode = os.environ.get(TARGET_PROFILE_RECIPE_MODE_ENV, "formal")
            if not output_value or not run_id or seed_value is None:
                raise RuntimeError(
                    f"{TARGET_PROFILE_PATH_ENV}, {TARGET_PROFILE_RUN_ID_ENV}, and "
                    f"{TARGET_PROFILE_RECIPE_SEED_ENV} are required"
                )
            output_path = Path(output_value)
            if not output_path.is_absolute():
                raise RuntimeError(f"target profile path must be absolute: {output_path}")
            output_path = output_path.resolve()
            if not output_path.parent.is_dir():
                raise RuntimeError(
                    f"target profile parent does not exist: {output_path.parent}"
                )
            recipe_seed = int(seed_value)
            self._target_profile_run_id = run_id
            self._target_profile_iteration = 0
            self._target_recipe_seed = recipe_seed
            self._target_recipe_mode = recipe_mode
            self._target_profile_stream = output_path.open(
                "x", encoding="utf-8", buffering=1
            )
            self._target_planner = TargetBatchPlanner(
                build_target_recipes(recipe_seed, mode=recipe_mode)
            )
            self._target_adapter = VllmAdapter(self)

        def _write_record(self, record: dict[str, Any] | None) -> None:
            if record is None:
                return
            record["recipe_seed"] = self._target_recipe_seed
            record["recipe_mode"] = self._target_recipe_mode
            self._target_profile_stream.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            self._target_profile_stream.flush()
            self._target_profile_iteration += 1

        def schedule(self, throttle_prefills: bool = False) -> Any:
            if not self._target_planner.complete:
                snapshot = self._target_adapter.make_snapshot()
                built = self._target_planner.build(snapshot)
                if built is not None:
                    plan, realized_shape = built
                    scheduler_output = self._target_adapter.build_scheduler_output(plan)
                    if tuple(scheduler_output.num_scheduled_tokens.items()) != (
                        plan.prefill_items
                        + tuple((request_id, 1) for request_id in plan.decode_items)
                    ):
                        raise RuntimeError(
                            "target BatchPlan does not match materialized SchedulerOutput"
                        )
                    recipe = self._target_planner.current
                    assert recipe is not None
                    record = build_target_profile_record(
                        run_id=self._target_profile_run_id,
                        iteration_index=self._target_profile_iteration,
                        snapshot=snapshot,
                        plan=plan,
                        recipe=recipe,
                        realized_shape=realized_shape,
                    )
                    self._target_planner.advance()
                    self._write_record(record)
                    return scheduler_output

            captured_state = capture_stock_profile_state(self)
            scheduler_output = super().schedule(throttle_prefills)
            record = build_stock_profile_record(
                run_id=self._target_profile_run_id,
                iteration_index=self._target_profile_iteration,
                captured_state=captured_state,
                scheduler_output=scheduler_output,
            )
            if record is not None:
                recipe = self._target_planner.current
                role = "drain" if recipe is None else "setup"
                record.update(
                    {
                        "sample_role": role,
                        "recipe_id": recipe.recipe_id if recipe is not None else None,
                        "requested_shape": (
                            recipe.as_dict() if recipe is not None else None
                        ),
                        "realized_shape": _selected_shape(record),
                    }
                )
            self._write_record(record)
            return scheduler_output

    TargetedProfilingScheduler.__module__ = (
        "dpp_scheduler.targeted_profile_scheduler"
    )
    return TargetedProfilingScheduler


def get_isolated_profiling_scheduler_class() -> type:
    """Create a profiling scheduler with one clean, exact target per batch."""
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    from dpp_scheduler.targeted_profile import (
        TargetBatchPlanner,
        build_isolated_setup_plan,
        build_isolated_target_plan,
        build_target_recipes,
        resolve_isolated_request_ids,
    )

    class IsolatedProfilingScheduler(Scheduler):
        """Prepare, execute, abort, and fully release one recipe at a time."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            install_vllm_iteration_timing_bridge()
            super().__init__(*args, **kwargs)
            output_value = os.environ.get(ISOLATED_PROFILE_PATH_ENV)
            event_value = os.environ.get(ISOLATED_PROFILE_EVENT_PATH_ENV)
            run_id = os.environ.get(ISOLATED_PROFILE_RUN_ID_ENV)
            seed_value = os.environ.get(ISOLATED_PROFILE_RECIPE_SEED_ENV)
            recipe_mode = os.environ.get(
                ISOLATED_PROFILE_RECIPE_MODE_ENV, "isolated_knee"
            )
            if not output_value or not event_value or not run_id or seed_value is None:
                raise RuntimeError("isolated profiling environment is incomplete")
            output_path = Path(output_value).resolve()
            event_path = Path(event_value).resolve()
            if not output_path.is_absolute() or not event_path.is_absolute():
                raise RuntimeError("isolated profiling paths must be absolute")
            if not output_path.parent.is_dir() or not event_path.parent.is_dir():
                raise RuntimeError("isolated profiling output parent does not exist")
            self._isolated_run_id = run_id
            self._isolated_recipe_seed = int(seed_value)
            self._isolated_recipe_mode = recipe_mode
            self._isolated_stream = output_path.open("x", encoding="utf-8", buffering=1)
            self._isolated_event_stream = event_path.open(
                "x", encoding="utf-8", buffering=1
            )
            self._isolated_planner = TargetBatchPlanner(
                build_target_recipes(self._isolated_recipe_seed, mode=recipe_mode)
            )
            self._isolated_adapter = VllmAdapter(self)
            self._isolated_pending: dict[str, Any] | None = None
            self._isolated_admission_closed = False
            self._isolated_baseline_free_blocks = int(
                self.kv_cache_manager.block_pool.get_num_free_blocks()
            )
            self._isolated_target_ordinal = 0

        def _event(self, event: str, **values: Any) -> None:
            recipe = self._isolated_planner.current
            payload = {
                "schema_version": 2,
                "event": event,
                "run_id": self._isolated_run_id,
                "recipe_id": recipe.recipe_id if recipe is not None else None,
                **values,
            }
            self._isolated_event_stream.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            )
            self._isolated_event_stream.flush()

        @staticmethod
        def _materialization_matches(plan: BatchPlan, output: Any) -> bool:
            return tuple(output.num_scheduled_tokens.items()) == (
                plan.prefill_items
                + tuple((request_id, 1) for request_id in plan.decode_items)
            )

        def _clean_current(self, *, status: str, reason: str | None = None) -> dict[str, Any]:
            recipe = self._isolated_planner.current
            if recipe is None:
                raise RuntimeError("isolated cleanup has no current recipe")
            live_before = sorted(self.requests)
            self.finish_requests(live_before, RequestStatus.FINISHED_ABORTED)
            free_blocks = int(self.kv_cache_manager.block_pool.get_num_free_blocks())
            queue_empty = not self.running and not self.waiting and not self.requests
            recovered = queue_empty and free_blocks == self._isolated_baseline_free_blocks
            cleanup = {
                "cleanup_started_after_timing": True,
                "aborted_request_ids": live_before,
                "post_cleanup_request_count": len(self.requests),
                "post_cleanup_running_count": len(self.running),
                "post_cleanup_waiting_count": len(self.waiting),
                "baseline_free_kv_blocks": self._isolated_baseline_free_blocks,
                "post_cleanup_free_kv_blocks": free_blocks,
                "resource_recovered": recovered,
            }
            self._event(status, reason=reason, cleanup=cleanup)
            if not recovered:
                raise RuntimeError(
                    "isolated profiling cleanup did not restore the KV/request baseline"
                )
            self._isolated_planner.advance()
            self._isolated_admission_closed = False
            return cleanup

        def _fail_current(self, reason: str) -> None:
            self._clean_current(status="batch_failed", reason=reason)

        def _dpp_record_iteration_duration(
            self,
            *,
            scheduler_output: Any,
            iteration_index: int | None,
            duration_seconds: float,
            timing_source: str,
        ) -> None:
            pending = self._isolated_pending
            if pending is None or pending["scheduler_output"] is not scheduler_output:
                raise RuntimeError("isolated timing callback does not match pending plan")
            if timing_source != VLLM_OFFICIAL_ITERATION_TIMING or iteration_index is None:
                raise RuntimeError("isolated profiling requires official vLLM timing")
            pending["actual_duration_seconds"] = duration_seconds
            pending["official_iteration_index"] = iteration_index
            pending["timing_source"] = timing_source

        def schedule(self, throttle_prefills: bool = False) -> Any:
            if self._isolated_pending is not None:
                raise RuntimeError("isolated scheduler has overlapping iterations")
            recipe = self._isolated_planner.current
            if recipe is None:
                return super().schedule(throttle_prefills)

            observed = set(self.requests)
            try:
                resolved_prefills, resolved_decodes = resolve_isolated_request_ids(
                    observed, recipe, require_complete=False
                )
            except RuntimeError as error:
                self.finish_requests(sorted(observed), RequestStatus.FINISHED_ABORTED)
                raise RuntimeError(f"isolated admission rejected: {error}") from error
            expected_count = recipe.prefill_request_cap + recipe.decode_request_cap
            admission_complete = (
                len(resolved_prefills) + len(resolved_decodes) == expected_count
            )
            if not self._isolated_admission_closed:
                if not admission_complete:
                    if not observed:
                        return super().schedule(throttle_prefills)
                    return SchedulerOutput.make_empty()
                self._isolated_admission_closed = True
                self._event(
                    "admission_closed",
                    request_count=len(observed),
                    requested_shape=recipe.as_dict(),
                )
            elif not admission_complete:
                self._fail_current("request_finished_during_setup")
                return super().schedule(throttle_prefills)

            snapshot = self._isolated_adapter.make_snapshot()
            try:
                setup_plan = build_isolated_setup_plan(snapshot, recipe)
                if setup_plan is None:
                    plan, realized = build_isolated_target_plan(snapshot, recipe)
                    role = "target"
                else:
                    plan = setup_plan
                    realized = None
                    role = "setup"
            except Exception as error:
                self._fail_current(f"{type(error).__name__}: {error}")
                return super().schedule(throttle_prefills)

            output = self._isolated_adapter.build_scheduler_output(plan)
            if not self._materialization_matches(plan, output):
                raise RuntimeError("isolated Exact BatchPlan materialization mismatch")
            self._isolated_pending = {
                "role": role,
                "snapshot": snapshot,
                "plan": plan,
                "realized": realized,
                "scheduler_output": output,
                "actual_duration_seconds": None,
                "official_iteration_index": None,
                "timing_source": None,
            }
            return output

        def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> Any:
            pending = self._isolated_pending
            result = super().update_from_output(scheduler_output, model_runner_output)
            if pending is None or pending["scheduler_output"] is not scheduler_output:
                raise RuntimeError("isolated update does not match pending iteration")
            if pending["actual_duration_seconds"] is None:
                raise RuntimeError("isolated iteration has no official duration")
            self._isolated_pending = None
            if pending["role"] == "setup":
                self._event(
                    "setup_complete",
                    plan_id=pending["plan"].plan_id,
                    official_iteration_index=pending["official_iteration_index"],
                )
                return result

            recipe = self._isolated_planner.current
            assert recipe is not None
            record = build_target_profile_record(
                run_id=self._isolated_run_id,
                iteration_index=int(pending["official_iteration_index"]),
                snapshot=pending["snapshot"],
                plan=pending["plan"],
                recipe=recipe,
                realized_shape=pending["realized"],
            )
            record.update(
                {
                    "schema_version": 2,
                    "recipe_seed": self._isolated_recipe_seed,
                    "recipe_mode": self._isolated_recipe_mode,
                    "target_ordinal": self._isolated_target_ordinal,
                    "actual_duration_seconds": pending["actual_duration_seconds"],
                    "timing_source": pending["timing_source"],
                    "execution_match": self._materialization_matches(
                        pending["plan"], scheduler_output
                    ),
                    "admission_closed": self._isolated_admission_closed,
                    "prefill_ratio": (
                        pending["plan"].total_prefill_tokens
                        / (
                            pending["plan"].total_prefill_tokens
                            + pending["plan"].total_decode_tokens
                        )
                    ),
                }
            )
            cleanup = self._clean_current(status="batch_complete")
            record["cleanup"] = cleanup
            record["valid"] = bool(record["execution_match"] and cleanup["resource_recovered"])
            self._isolated_stream.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            self._isolated_stream.flush()
            self._isolated_target_ordinal += 1
            return result

    IsolatedProfilingScheduler.__module__ = (
        "dpp_scheduler.isolated_profile_scheduler"
    )
    return IsolatedProfilingScheduler
