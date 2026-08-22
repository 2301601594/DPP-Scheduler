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
from dpp_scheduler.candidate_generator import (
    project_kv_blocks,
    project_sequence_count,
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
