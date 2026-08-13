from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    if not 0 <= probability <= 100:
        raise ValueError("percentile must be in [0, 100]")
    position = (len(ordered) - 1) * probability / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def request_derived_metrics(request: dict[str, Any]) -> dict[str, float]:
    itls = [float(value) for value in request.get("itls_ms", [])]
    output_tokens = int(request.get("actual_output_tokens") or 0)
    ttft = float(request.get("ttft_ms") or 0.0)
    # Use the frozen definition, excluding HTTP/SSE teardown after the last token.
    e2e = ttft + sum(itls)
    tpot = sum(itls) / (output_tokens - 1) if output_tokens > 1 else 0.0
    return {
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "itl_ms": statistics.fmean(itls) if itls else 0.0,
        "max_tbt_ms": max(itls, default=0.0),
        "e2e_ms": e2e,
    }


def summarize_requests(
    requests: list[dict[str, Any]],
    duration_s: float,
    percentiles: Iterable[float],
) -> dict[str, Any]:
    completed = [request for request in requests if request.get("success")]
    failed = len(requests) - len(completed)
    derived = [request_derived_metrics(request) for request in completed]
    output_tokens = sum(int(request.get("actual_output_tokens") or 0) for request in completed)
    input_tokens = sum(int(request.get("actual_input_tokens") or 0) for request in completed)
    summary: dict[str, Any] = {
        "requests": len(requests),
        "completed": len(completed),
        "failed": failed,
        "duration_s": duration_s,
        "request_throughput_rps": len(completed) / duration_s if duration_s else 0.0,
        "input_throughput_tps": input_tokens / duration_s if duration_s else 0.0,
        "output_throughput_tps": output_tokens / duration_s if duration_s else 0.0,
        "total_throughput_tps": (input_tokens + output_tokens) / duration_s
        if duration_s
        else 0.0,
        "output_length_mismatches": sum(
            int(request.get("actual_output_tokens") or -1)
            != int(request["expected_output_tokens"])
            for request in completed
        ),
        "input_length_mismatches": sum(
            int(request.get("actual_input_tokens") or -1)
            != int(request["expected_input_tokens"])
            for request in completed
        ),
        "multi_token_chunks": sum(int(request.get("multi_token_chunks", 0)) for request in completed),
    }
    for metric in ("ttft_ms", "tpot_ms", "itl_ms", "max_tbt_ms", "e2e_ms"):
        values = [row[metric] for row in derived]
        summary[f"mean_{metric}"] = statistics.fmean(values) if values else math.nan
        for value in percentiles:
            summary[f"p{value:g}_{metric}"] = percentile(values, float(value))
    return summary


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def confidence_interval_95(values: Iterable[float]) -> tuple[float, float, float]:
    data = [float(value) for value in values]
    if not data:
        return math.nan, math.nan, math.nan
    mean = statistics.fmean(data)
    if len(data) == 1:
        return mean, math.nan, math.nan
    degrees = len(data) - 1
    keys = sorted(_T_CRITICAL_95)
    key = min((value for value in keys if value >= degrees), default=30)
    critical = _T_CRITICAL_95[key] if degrees <= 30 else 1.96
    half_width = critical * statistics.stdev(data) / math.sqrt(len(data))
    return mean, mean - half_width, mean + half_width


def slo_attainment(
    requests: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]
) -> dict[str, float]:
    completed = 0
    ttft_pass = 0
    tpot_pass = 0
    joint_pass = 0
    for request in requests:
        if not request.get("success"):
            continue
        completed += 1
        workload = request["workload_class"]
        threshold = thresholds.get(workload) or thresholds.get("default")
        if threshold is None:
            raise KeyError(f"no SLO threshold for workload class {workload}")
        derived = request_derived_metrics(request)
        ttft_ok = derived["ttft_ms"] <= float(threshold["ttft_ms"])
        tpot_ok = derived["tpot_ms"] <= float(threshold["tpot_ms"])
        ttft_pass += ttft_ok
        tpot_pass += tpot_ok
        joint_pass += ttft_ok and tpot_ok
    denominator = len(requests) or 1
    return {
        "ttft_attainment": ttft_pass / denominator,
        "tpot_attainment": tpot_pass / denominator,
        "joint_attainment": joint_pass / denominator,
        "joint_passed": joint_pass,
        "completed": completed,
    }
