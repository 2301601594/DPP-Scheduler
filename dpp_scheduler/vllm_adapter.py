"""vLLM Adapter boundary for the modular DPP scheduler.

Only this module may import vLLM internal types.  The G2 implementation
provides the exact-plan adapter plus a callback-based adapter used by unit
tests.  The vLLM-specific methods are implemented lazily so that pure-Python
unit tests can import this module without a full vLLM installation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dpp_scheduler.contracts import (
    BatchPlan,
    DecodeRequest,
    ExecutionObservation,
    PrefillRequest,
    StateSnapshot,
    validate_snapshot_hash,
)
from dpp_scheduler.candidate_generator import (
    project_kv_blocks,
    project_sequence_count,
)


STOCK_PROFILE_PATH_ENV = "DPP_STOCK_PROFILE_PATH"
STOCK_PROFILE_RUN_ID_ENV = "DPP_STOCK_PROFILE_RUN_ID"
TARGET_PROFILE_PATH_ENV = "DPP_TARGET_PROFILE_PATH"
TARGET_PROFILE_RUN_ID_ENV = "DPP_TARGET_PROFILE_RUN_ID"
TARGET_PROFILE_RECIPE_SEED_ENV = "DPP_TARGET_PROFILE_RECIPE_SEED"
TARGET_PROFILE_RECIPE_MODE_ENV = "DPP_TARGET_PROFILE_RECIPE_MODE"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_stock_profile_state(scheduler: Any) -> dict[str, Any]:
    """Capture only current, length-blind state before stock scheduling."""
    request_states = []
    current_context_tokens: dict[str, int] = {}
    for request_id in sorted(scheduler.requests):
        request = scheduler.requests[request_id]
        context_tokens = int(request.num_computed_tokens)
        current_context_tokens[request_id] = context_tokens
        request_states.append(
            {
                "request_id": request_id,
                "status": str(request.status),
                "current_context_tokens": context_tokens,
                "prompt_tokens": int(request.num_prompt_tokens),
            }
        )

    snapshot_payload = {
        "requests": request_states,
        "running_request_ids": [request.request_id for request in scheduler.running],
        "waiting_request_ids": [request.request_id for request in scheduler.waiting],
        "free_kv_blocks": int(
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        ),
        "token_budget": int(scheduler.max_num_scheduled_tokens),
        "sequence_budget": int(scheduler.max_num_running_reqs),
    }
    return {
        "snapshot_hash": _canonical_sha256(snapshot_payload),
        "current_context_tokens": current_context_tokens,
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
        "schema_version": 1,
        "run_id": run_id,
        "iteration_index": iteration_index,
        "plan_id": f"stock-{plan_hash[:24]}",
        "snapshot_hash": captured_state["snapshot_hash"],
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

    def __init__(self, scheduler: Any, *, frame_start: int = 1) -> None:
        self._scheduler = scheduler
        self._frame = frame_start
        self._last_snapshot: StateSnapshot | None = None
        self._poisoned = False

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
        now = time.time()

        waiting: list[PrefillRequest] = []
        decode: list[DecodeRequest] = []
        ordinal = 0

        for request in running:
            if str(request.status) != "RUNNING":
                raise RuntimeError("running queue contains a non-RUNNING request")
            if request.num_computed_tokens < request.num_prompt_tokens:
                waiting.append(
                    PrefillRequest(
                        request_id=request.request_id,
                        arrival_time=request.arrival_time,
                        token_count=request.num_prompt_tokens,
                        prefilled_tokens=request.num_computed_tokens,
                        is_running=True,
                        ordinal=ordinal,
                    )
                )
            else:
                decode.append(
                    DecodeRequest(
                        request_id=request.request_id,
                        arrival_time=request.arrival_time,
                        kv_context_length=request.num_computed_tokens,
                        ordinal=ordinal,
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
            waiting.append(
                PrefillRequest(
                    request_id=request.request_id,
                    arrival_time=request.arrival_time,
                    token_count=request.num_prompt_tokens,
                    prefilled_tokens=0,
                    is_running=False,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

        free_blocks = block_pool.get_num_free_blocks()
        total_blocks = block_pool.num_gpu_blocks

        snapshot = StateSnapshot.create(
            frame_id=frame_id,
            timestamp=now,
            waiting_prefill_requests=tuple(waiting),
            active_decode_requests=tuple(decode),
            active_ttft_obligations=(),
            active_tbt_obligations=(),
            recovery_requests=(),
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
    from dpp_scheduler.dpp_selector import TemporarySelector
    from dpp_scheduler.settings import SchedulerSettings

    logger = init_logger("dpp_scheduler.vllm_scheduler")

    class ModularDPPScheduler(Scheduler):
        """Exact-plan development scheduler, disabled until inputs freeze."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            settings = SchedulerSettings.provisional()
            if not settings.frozen:
                raise RuntimeError(
                    "ModularDPPScheduler is not executable while its G0/G1 "
                    "candidate parameters are provisional"
                )
            super().__init__(*args, **kwargs)
            self._dpp_adapter = VllmAdapter(self)
            self._dpp_generator = CandidateGenerator(settings)
            self._dpp_selector = TemporarySelector()

        def schedule(self, throttle_prefills: bool = False) -> Any:
            del throttle_prefills
            snapshot = self._dpp_adapter.make_snapshot()
            plans = self._dpp_generator.generate(snapshot)
            decision = self._dpp_selector.select(snapshot, plans)
            logger.info(
                "ModularDPPScheduler frame=%s plans=%d selected=%s",
                snapshot.frame_id,
                len(plans),
                decision.selected_plan.plan_id
                if decision.selected_plan is not None
                else "NONE",
            )
            if decision.selected_plan is None:
                raise RuntimeError(
                    "no exact BatchPlan; G4 fallback/preemption path is not implemented"
                )
            return self._dpp_adapter.build_scheduler_output(decision.selected_plan)

    ModularDPPScheduler.__module__ = "dpp_scheduler.vllm_scheduler"
    return ModularDPPScheduler


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
