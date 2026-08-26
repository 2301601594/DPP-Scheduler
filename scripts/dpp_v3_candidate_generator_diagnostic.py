#!/usr/bin/env python3
"""v3 Candidate Generator diagnostic harness.

This is a Python-level harness that drives :class:`CandidateGenerator` and
:class:`RidgeBudgetResolver` against a synthesized per-iteration schedule
derived from a Qwen3-14B trace and an optional v2 per_request.jsonl. It does
**not** spin up vLLM; iteration durations come from the offline Ridge
Predictor and the per_request.jsonl's measured ``itls_ms`` (when provided).

Goals
-----
* Quantify base Prefill budget ``P`` distribution emitted by
  :class:`RidgeBudgetResolver` (status counts, P percentiles, support ratio).
* Quantify the multiplier × priority-policy candidate layout produced by
  :class:`CandidateGenerator` (raw vs deduplicated counts, per-frame
  candidate budgets).
* Render a DPP choice distribution by attaching a deterministic selector
  that picks the highest ``-predicted_duration`` plan among deduped
  candidates.
* Compare the simulated candidate set against the legacy P25/P50/P75/MAX
  layout for the same synthesized snapshots.

This harness is intentionally side-effect-free: it never modifies the
trace, the Predictor artifact, or any on-disk state.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dpp_scheduler.budget_resolver import (
    RESOLUTION_NO_DECODE_NO_BACKLOG,
    RESOLUTION_NO_DECODE_USE_MAX,
    BudgetResolution,
    RidgeBudgetResolver,
    collect_resolution_summary,
)
from dpp_scheduler.candidate_generator import CandidateGenerator
from dpp_scheduler.contracts import (
    DecodeRequest,
    PrefillRequest,
    StateSnapshot,
)
from dpp_scheduler.predictor import RidgeDurationPredictor
from dpp_scheduler.settings import SchedulerSettings


DEFAULT_TTFT_SLO_SECONDS = 2.0
DEFAULT_TBT_SLO_SECONDS = 0.25
ASSUMED_OUTPUT_TOKENS = 256
ASSUMED_DECODE_TOKENS_PER_ITERATION = 1
DEFAULT_TOKEN_BUDGET = 2048
DEFAULT_SEQUENCE_BUDGET = 64


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda row: float(row.get("arrival_time_s", 0.0)))
    return rows


# ---------------------------------------------------------------------------
# Snapshot synthesis
# ---------------------------------------------------------------------------


class _SynthRequest:
    __slots__ = (
        "request_id",
        "arrival_time",
        "input_tokens",
        "prefilled_tokens",
        "is_running",
        "ordinal",
        "ttft_deadline",
        "decode_started_at",
        "active",
        "tokens_decoded",
    )

    def __init__(
        self,
        *,
        request_id: str,
        arrival_time: float,
        input_tokens: int,
        ordinal: int,
        ttft_slo_seconds: float,
    ) -> None:
        self.request_id = request_id
        self.arrival_time = arrival_time
        self.input_tokens = int(input_tokens)
        self.prefilled_tokens = 0
        self.is_running = False
        self.ordinal = ordinal
        self.ttft_deadline = arrival_time + ttft_slo_seconds
        self.decode_started_at: float | None = None
        self.active = False
        self.tokens_decoded = 0


def build_schedule(
    trace: list[dict[str, Any]],
    *,
    assumed_output_tokens: int,
    ttft_slo_seconds: float,
    tbt_slo_seconds: float,
) -> list[tuple[float, list[_SynthRequest], list[_SynthRequest]]]:
    """Synthesize one snapshot per request arrival.

    Returns a list of ``(timestamp, waiting, decode)`` tuples representing
    the schedule at each arrival moment, with all requests in the system
    classified as either waiting for prefill or actively decoding.
    """
    requests: list[_SynthRequest] = []
    schedule: list[tuple[float, list[_SynthRequest], list[_SynthRequest]]] = []
    for index, row in enumerate(trace):
        req = _SynthRequest(
            request_id=f"r{index:04d}",
            arrival_time=float(row["arrival_time_s"]),
            input_tokens=int(row["input_tokens"]),
            ordinal=index,
            ttft_slo_seconds=ttft_slo_seconds,
        )
        requests.append(req)
        schedule.append((req.arrival_time, [], []))
    schedule.clear()
    last_event = 0.0
    for index, req in enumerate(requests):
        # At each request arrival, recompute who is still waiting vs. who is
        # actively decoding. The simplified model assumes each request
        # finishes prefill immediately upon arrival (so all old waiting
        # requests move to decode), and each decode request emits one token
        # per ``tbt_slo_seconds`` since its decode_started_at.
        waiting: list[_SynthRequest] = [req]
        decode: list[_SynthRequest] = []
        for other in requests[:index]:
            tokens_so_far = int(
                (req.arrival_time - other.arrival_time) / tbt_slo_seconds
            ) if req.arrival_time > other.arrival_time else 0
            if tokens_so_far >= assumed_output_tokens:
                continue  # already finished
            decode_started = other.arrival_time  # prefill done instantly
            tbt_deadline = decode_started + (
                assumed_output_tokens * tbt_slo_seconds
            )
            decode.append(
                _DecodeShim(other, tbt_deadline, decode_started, tokens_so_far)
            )
        schedule.append((req.arrival_time, waiting, decode))
        last_event = req.arrival_time
    return schedule


class _DecodeShim:
    """Adapter so :class:`_SynthRequest` slots look like :class:`DecodeRequest`."""

    __slots__ = (
        "request_id",
        "arrival_time",
        "kv_context_length",
        "tbt_deadline",
        "ordinal",
        "decode_started_at",
        "tokens_decoded",
    )

    def __init__(
        self,
        req: _SynthRequest,
        tbt_deadline: float,
        decode_started_at: float,
        tokens_decoded: int,
    ) -> None:
        self.request_id = req.request_id
        self.arrival_time = req.arrival_time
        self.kv_context_length = req.input_tokens + tokens_decoded
        self.tbt_deadline = tbt_deadline
        self.ordinal = req.ordinal
        self.decode_started_at = decode_started_at
        self.tokens_decoded = tokens_decoded


def to_state_snapshot(
    waiting: list[_SynthRequest],
    decode: list[_DecodeShim],
    *,
    timestamp: float,
    snapshot_hash_seed: str,
    token_budget: int,
    sequence_budget: int,
    kv_block_size: int,
    total_kv_blocks: int,
) -> StateSnapshot:
    prefill_requests = tuple(
        PrefillRequest(
            request_id=req.request_id,
            arrival_time=req.arrival_time,
            token_count=req.input_tokens,
            prefilled_tokens=req.prefilled_tokens,
            ttft_deadline=req.ttft_deadline,
            is_running=req.is_running,
            ordinal=req.ordinal,
        )
        for req in waiting
    )
    decode_requests = tuple(
        DecodeRequest(
            request_id=item.request_id,
            arrival_time=item.arrival_time,
            kv_context_length=item.kv_context_length,
            tbt_deadline=item.tbt_deadline,
            ordinal=item.ordinal,
        )
        for item in decode
    )
    free_kv_blocks = max(0, total_kv_blocks - 100 * len(decode_requests))
    return StateSnapshot.create(
        frame_id=abs(hash(snapshot_hash_seed)) % (2**31),
        timestamp=timestamp,
        waiting_prefill_requests=prefill_requests,
        active_decode_requests=decode_requests,
        active_ttft_obligations=(),
        active_tbt_obligations=(),
        recovery_requests=(),
        free_kv_blocks=free_kv_blocks,
        kv_block_size=kv_block_size,
        token_budget=token_budget,
        sequence_budget=sequence_budget,
        total_kv_blocks=total_kv_blocks,
        provenance="dpp_v3_candidate_generator_diagnostic",
    )


# ---------------------------------------------------------------------------
# DPP stand-in (deterministic, predictor-driven)
# ---------------------------------------------------------------------------


def pick_winner(
    plans: tuple,
    predictions: dict[str, Any],
) -> Any | None:
    """Pick the plan with the smallest expected duration among in-support predictions.

    A production selector uses the full DPP equation; this stand-in only
    needs to choose deterministically so the diagnostic can attribute the
    selection to a ``template_id``.
    """
    candidates: list[tuple[float, Any]] = []
    for plan in plans:
        prediction = predictions.get(plan.plan_id)
        if prediction is None:
            continue
        if not prediction.in_support:
            continue
        if prediction.expected_duration is None:
            continue
        candidates.append((float(prediction.expected_duration), plan))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Legacy v2 reference layout (for comparison)
# ---------------------------------------------------------------------------


def v2_budgets(
    snapshot: StateSnapshot,
    *,
    fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[int]:
    """Mimic the legacy fractions + MAX layout for the same snapshot."""
    decode_count = len(snapshot.active_decode_requests)
    backlog = sum(item.remaining_tokens for item in snapshot.waiting_prefill_requests)
    maximum = min(backlog, max(0, snapshot.token_budget - decode_count))
    raw: list[int] = []
    for fraction in fractions:
        raw.append(math.floor(maximum * fraction) if fraction != 1.0 else maximum)
    raw = [max(0, min(maximum, value)) for value in raw]
    return sorted(set(raw))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_harness(
    trace: list[dict[str, Any]],
    predictor: RidgeDurationPredictor,
    settings: SchedulerSettings,
    *,
    max_frames: int,
    token_budget: int,
    sequence_budget: int,
    kv_block_size: int,
    total_kv_blocks: int,
    ttft_slo_seconds: float,
    tbt_slo_seconds: float,
    assumed_output_tokens: int,
) -> dict[str, Any]:
    schedule = build_schedule(
        trace,
        assumed_output_tokens=assumed_output_tokens,
        ttft_slo_seconds=ttft_slo_seconds,
        tbt_slo_seconds=tbt_slo_seconds,
    )
    resolver = RidgeBudgetResolver(predictor=predictor, settings=settings)
    generator = CandidateGenerator(settings, budget_resolver=resolver)

    resolutions: list[BudgetResolution] = []
    raw_counts: list[int] = []
    dedup_counts: list[int] = []
    distinct_budget_counts: list[int] = []
    candidate_budget_samples: list[tuple[int, ...]] = []
    v2_budget_samples: list[list[int]] = []
    multiplier_selections: Counter[str] = Counter()
    policy_selections: Counter[str] = Counter()
    template_winners: Counter[str] = Counter()
    decode_count_distribution: list[int] = []
    backlog_distribution: list[int] = []
    distinct_budget_count_histogram: Counter[int] = Counter()
    skipped_empty = 0

    for index, (timestamp, waiting, decode) in enumerate(schedule):
        if index >= max_frames:
            break
        snapshot = to_state_snapshot(
            waiting,
            decode,
            timestamp=timestamp,
            snapshot_hash_seed=f"{index}-{timestamp}",
            token_budget=token_budget,
            sequence_budget=sequence_budget,
            kv_block_size=kv_block_size,
            total_kv_blocks=total_kv_blocks,
        )
        if (
            not snapshot.waiting_prefill_requests
            and not snapshot.active_decode_requests
        ):
            skipped_empty += 1
            continue
        decode_count_distribution.append(len(decode))
        backlog_distribution.append(
            sum(item.remaining_tokens for item in snapshot.waiting_prefill_requests)
        )
        plans = generator.generate(snapshot)
        diag = generator.last_diagnostic
        if diag is None:
            continue
        resolutions.append(diag.resolution)
        raw_counts.append(diag.raw_candidate_count)
        dedup_counts.append(diag.deduplicated_candidate_count)
        budget_values = diag.candidate_budget_values
        distinct_budget_counts.append(len(budget_values))
        distinct_budget_count_histogram[len(budget_values)] += 1
        candidate_budget_samples.append(budget_values)
        v2_budget_samples.append(v2_budgets(snapshot))

        predictions = predictor.predict(snapshot, plans)
        predictions_by_id = {pred.plan_id: pred for pred in predictions}
        winner = pick_winner(plans, predictions_by_id)
        if winner is not None:
            template_winners[winner.template_id] += 1
            parts = winner.template_id.split(":")
            if len(parts) >= 4 and parts[1] == "SLACK_BUDGET":
                multiplier_selections[parts[2]] += 1
                policy_selections[parts[3]] += 1
            elif len(parts) >= 2 and parts[1] == "ZERO":
                multiplier_selections["ZERO"] += 1
                policy_selections["ZERO"] += 1

    resolution_summary = collect_resolution_summary(resolutions)
    return {
        "frames_simulated": len(schedule),
        "frames_with_workload": len(resolutions),
        "frames_skipped_empty": skipped_empty,
        "resolution_summary": resolution_summary,
        "raw_candidate_count": summary_stats(raw_counts),
        "deduplicated_candidate_count": summary_stats(dedup_counts),
        "distinct_budget_count_per_frame": summary_stats(distinct_budget_counts),
        "distinct_budget_count_histogram": counter_to_dict(
            distinct_budget_count_histogram
        ),
        "candidate_budget_pool": counter_to_dict(
            Counter(_budget for sample in candidate_budget_samples for _budget in sample)
        ),
        "v2_legacy_budget_pool": counter_to_dict(
            Counter(_budget for sample in v2_budget_samples for _budget in sample)
        ),
        "multiplier_selections": counter_to_dict(multiplier_selections),
        "policy_selections": counter_to_dict(policy_selections),
        "template_winners_top": counter_to_dict(template_winners).get("items", []),
        "decode_count_distribution": summary_stats(decode_count_distribution),
        "backlog_distribution": summary_stats(backlog_distribution),
    }


def summary_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": ordered[max(0, math.ceil(0.50 * len(ordered)) - 1)],
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def counter_to_dict(counter: Counter) -> dict[str, Any]:
    return {
        "total": sum(counter.values()),
        "items": [
            {"key": key, "count": count}
            for key, count in sorted(
                counter.items(), key=lambda pair: pair[1], reverse=True
            )
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        required=True,
        type=Path,
        help="JSONL trace of request arrivals (arrival_time_s + input_tokens)",
    )
    parser.add_argument(
        "--predictor-artifact",
        type=Path,
        default=Path(
            "predictors/qwen3_14b/ridge_mixed_decode_three_segment_cross_online_v3"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Number of arrival moments to simulate",
    )
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--sequence-budget", type=int, default=DEFAULT_SEQUENCE_BUDGET)
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--total-kv-blocks", type=int, default=30149)
    parser.add_argument(
        "--ttft-slo-seconds",
        type=float,
        default=DEFAULT_TTFT_SLO_SECONDS,
    )
    parser.add_argument(
        "--tbt-slo-seconds",
        type=float,
        default=DEFAULT_TBT_SLO_SECONDS,
    )
    parser.add_argument(
        "--assumed-output-tokens",
        type=int,
        default=ASSUMED_OUTPUT_TOKENS,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.trace.exists():
        raise FileNotFoundError(f"trace not found: {args.trace}")
    if not args.predictor_artifact.exists():
        raise FileNotFoundError(
            f"predictor artifact not found: {args.predictor_artifact}"
        )

    trace = load_trace(args.trace)
    predictor = RidgeDurationPredictor.from_artifact(
        args.predictor_artifact, ood_uncertainty_coefficient=0.0
    )
    settings = SchedulerSettings.provisional()

    report = run_harness(
        trace,
        predictor,
        settings,
        max_frames=args.max_frames,
        token_budget=args.token_budget,
        sequence_budget=args.sequence_budget,
        kv_block_size=args.kv_block_size,
        total_kv_blocks=args.total_kv_blocks,
        ttft_slo_seconds=args.ttft_slo_seconds,
        tbt_slo_seconds=args.tbt_slo_seconds,
        assumed_output_tokens=args.assumed_output_tokens,
    )
    report["trace_path"] = str(args.trace)
    report["trace_request_count"] = len(trace)
    report["predictor_artifact"] = str(args.predictor_artifact)
    report["predictor_version"] = predictor.predictor_version
    report["settings"] = {
        "prefill_budget_multipliers": list(settings.prefill_budget_multipliers),
        "maximum_seed_candidates": settings.maximum_seed_candidates,
        "predictor_inversion_safety_margin_seconds": (
            settings.predictor_inversion_safety_margin_seconds
        ),
        "predictor_inversion_budget_grid": list(
            settings.predictor_inversion_budget_grid
        ),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
