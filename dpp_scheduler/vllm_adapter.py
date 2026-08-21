"""vLLM Adapter boundary for the modular DPP scheduler.

Only this module may import vLLM internal types.  The G2 implementation
provides the exact-plan adapter plus a callback-based adapter used by unit
tests.  The vLLM-specific methods are implemented lazily so that pure-Python
unit tests can import this module without a full vLLM installation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dpp_scheduler.contracts import (
    BatchPlan,
    DecodeRequest,
    ExecutionObservation,
    PrefillRequest,
    StateSnapshot,
    validate_snapshot_hash,
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

    def make_snapshot(self) -> StateSnapshot:
        scheduler = self._require_scheduler()

        # Lazy vLLM imports keep this module importable without vLLM.
        from vllm.v1.request import RequestStatus  # noqa: F401

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

        running_ids = {req.request_id for req in running}

        for request in requests.values():
            if request.request_id in running_ids:
                continue
            # A request that has not finished its prompt is a waiting prefill.
            if request.num_computed_tokens < request.num_prompt_tokens:
                waiting.append(
                    PrefillRequest(
                        request_id=request.request_id,
                        arrival_time=request.arrival_time,
                        token_count=request.num_prompt_tokens,
                        prefilled_tokens=request.num_computed_tokens,
                        ordinal=ordinal,
                    )
                )
                ordinal += 1

        for request in running:
            # With one token per decode and no speculative decoding, any running
            # request that has reached its prompt length is a decode request.
            if request.num_computed_tokens >= request.num_prompt_tokens:
                decode.append(
                    DecodeRequest(
                        request_id=request.request_id,
                        arrival_time=request.arrival_time,
                        kv_context_length=request.num_computed_tokens,
                        ordinal=ordinal,
                    )
                )
                ordinal += 1

        free_blocks = block_pool.get_num_free_blocks()
        total_blocks = block_pool.num_gpu_blocks

        return StateSnapshot.create(
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

    def execute_exact_plan(self, plan: BatchPlan) -> ExecutionObservation:
        scheduler = self._require_scheduler()
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            NewRequestData,
            ScheduledEncoderInputStats,
            SchedulerOutput,
        )
        from vllm.v1.request import RequestStatus

        start = time.time()
        scheduler.current_step += 1
        scheduler.kv_cache_manager.new_step_starts()

        scheduled_new_reqs: list[Any] = []
        scheduled_running_reqs: list[Any] = []
        req_to_new_blocks: dict[str, Any] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}

        # 1. Prefill items: move selected waiting requests into running and
        #    allocate their KV slots.
        for request_id, token_count in plan.prefill_items:
            request = scheduler.requests.get(request_id)
            if request is None:
                raise RuntimeError(f"prefill request not found in scheduler: {request_id}")
            new_blocks = scheduler.kv_cache_manager.allocate_slots(
                request, token_count
            )
            if new_blocks is None:
                raise RuntimeError(
                    f"KV allocation failed for prefill request {request_id}"
                )
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = token_count

            if request not in scheduler.running:
                try:
                    scheduler.waiting.remove_request(request)
                except ValueError:
                    # The request may already have been moved between queue types.
                    pass
                scheduler.running.append(request)
            request.status = RequestStatus.RUNNING
            scheduled_new_reqs.append(request)

        # 2. Decode items: allocate one token slot for each selected running
        #    request.
        for request_id in plan.decode_items:
            request = scheduler.requests.get(request_id)
            if request is None:
                raise RuntimeError(f"decode request not found in scheduler: {request_id}")
            new_blocks = scheduler.kv_cache_manager.allocate_slots(request, 1)
            if new_blocks is None:
                raise RuntimeError(
                    f"KV allocation failed for decode request {request_id}"
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

        return ExecutionObservation(
            frame_id=0,  # filled by the caller/controller in later integration
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
        return self._scheduler
