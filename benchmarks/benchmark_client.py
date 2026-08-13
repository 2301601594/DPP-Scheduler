#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.dppbench.io import atomic_write_json, read_jsonl, sha256_file
from benchmarks.dppbench.metrics import summarize_requests


def build_arrival_times(
    count: int,
    request_rate: float,
    burstiness: float,
    seed: int,
) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    if math.isinf(request_rate):
        return [0.0] * count
    if request_rate <= 0 or burstiness <= 0:
        raise ValueError("request_rate and burstiness must be positive")
    rng = random.Random(seed)
    if math.isinf(burstiness):
        delays = [1.0 / request_rate] * (count - 1)
    else:
        delays = [
            rng.gammavariate(burstiness, 1.0 / (request_rate * burstiness))
            for _ in range(count - 1)
        ]
    cumulative = [0.0]
    elapsed = 0.0
    for delay in delays:
        elapsed += delay
        cumulative.append(elapsed)
    target_duration = (count - 1) / request_rate
    scale = target_duration / cumulative[-1]
    return [value * scale for value in cumulative]


async def _request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    trace_row: dict[str, Any],
    temperature: float,
    ignore_eos: bool,
    seed: int,
) -> dict[str, Any]:
    prompt = trace_row.get("prompt_token_ids", trace_row.get("prompt"))
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(trace_row["output_tokens"]),
        "temperature": temperature,
        "ignore_eos": ignore_eos,
        "repetition_penalty": 1.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    headers = {
        "Content-Type": "application/json",
        "x-request-id": trace_row["request_id"],
    }
    result: dict[str, Any] = {
        "request_id": trace_row["request_id"],
        "workload_class": trace_row["workload_class"],
        "expected_input_tokens": int(trace_row["input_tokens"]),
        "expected_output_tokens": int(trace_row["output_tokens"]),
        "success": False,
        "error": None,
        "ttft_ms": None,
        "itls_ms": [],
        "e2e_ms": None,
        "actual_input_tokens": None,
        "actual_output_tokens": None,
        "multi_token_chunks": 0,
    }
    start = time.perf_counter()
    last_token = start
    first_token = False
    try:
        async with session.post(endpoint, json=payload, headers=headers) as response:
            result["http_status"] = response.status
            if response.status != 200:
                result["error"] = await response.text()
                result["e2e_ms"] = (time.perf_counter() - start) * 1000
                return result
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                message = line.removeprefix("data:").strip()
                if message == "[DONE]":
                    continue
                data = json.loads(message)
                if usage := data.get("usage"):
                    result["actual_input_tokens"] = usage.get("prompt_tokens")
                    result["actual_output_tokens"] = usage.get("completion_tokens")
                choices = data.get("choices") or []
                if not choices:
                    continue
                token_ids = choices[0].get("token_ids")
                token_count = len(token_ids) if isinstance(token_ids, list) else 1
                if token_count > 1:
                    result["multi_token_chunks"] += 1
                now = time.perf_counter()
                if not first_token:
                    result["ttft_ms"] = (now - start) * 1000
                    first_token = True
                else:
                    result["itls_ms"].append((now - last_token) * 1000)
                last_token = now
            result["e2e_ms"] = (time.perf_counter() - start) * 1000
            result["success"] = first_token and result["actual_output_tokens"] is not None
            if not result["success"]:
                result["error"] = "stream ended without token or final usage"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["e2e_ms"] = (time.perf_counter() - start) * 1000
    return result


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    trace_rows = list(read_jsonl(args.trace))
    if args.limit > 0:
        trace_rows = trace_rows[: args.limit]
    if not trace_rows:
        raise ValueError("trace contains no requests")
    if args.self_timed:
        origin = float(trace_rows[0]["arrival_time_s"] or 0.0)
        arrivals = [float(row["arrival_time_s"] or 0.0) - origin for row in trace_rows]
    else:
        arrivals = build_arrival_times(
            len(trace_rows), args.request_rate, args.burstiness, args.seed
        )

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, trust_env=False
    ) as session:
        for index in range(args.warmups):
            warmup_row = dict(trace_rows[index % len(trace_rows)])
            warmup_row["request_id"] = f"warmup-{index:04d}"
            warmup_result = await _request(
                session,
                args.endpoint,
                args.model,
                warmup_row,
                args.temperature,
                args.ignore_eos,
                args.server_seed,
            )
            if not warmup_result["success"]:
                raise RuntimeError(f"warmup failed: {warmup_result['error']}")

        semaphore = (
            asyncio.Semaphore(args.max_concurrency)
            if args.max_concurrency
            else None
        )
        benchmark_start = time.perf_counter()
        dispatch_offsets: list[float] = []

        async def dispatch(row: dict[str, Any], arrival: float) -> dict[str, Any]:
            remaining = benchmark_start + arrival - time.perf_counter()
            if remaining > 0:
                await asyncio.sleep(remaining)
            dispatch_offset = time.perf_counter() - benchmark_start
            dispatch_offsets.append(dispatch_offset)
            if semaphore is None:
                result = await _request(
                    session,
                    args.endpoint,
                    args.model,
                    row,
                    args.temperature,
                    args.ignore_eos,
                    args.server_seed,
                )
            else:
                async with semaphore:
                    result = await _request(
                        session,
                        args.endpoint,
                        args.model,
                        row,
                        args.temperature,
                        args.ignore_eos,
                        args.server_seed,
                    )
            result["scheduled_arrival_s"] = arrival
            result["dispatch_offset_s"] = dispatch_offset
            return result

        pending = asyncio.gather(
            *(dispatch(row, arrival) for row, arrival in zip(trace_rows, arrivals))
        )
        if args.drain_timeout > 0:
            requests = await asyncio.wait_for(
                pending, timeout=max(arrivals, default=0.0) + args.drain_timeout
            )
        else:
            requests = await pending
        duration = time.perf_counter() - benchmark_start

    offered_duration = max(dispatch_offsets) - min(dispatch_offsets) if len(dispatch_offsets) > 1 else 0
    actual_offered_rate = (
        (len(dispatch_offsets) - 1) / offered_duration if offered_duration > 0 else math.inf
    )
    expected_rate = (
        (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
        if len(arrivals) > 1 and arrivals[-1] > arrivals[0]
        else math.inf
    )
    result = {
        "schema_version": 1,
        "trace_path": str(Path(args.trace).resolve()),
        "trace_sha256": sha256_file(args.trace),
        "model": args.model,
        "endpoint": args.endpoint,
        "seed": args.seed,
        "server_seed": args.server_seed,
        "request_rate_rps": None if math.isinf(args.request_rate) else args.request_rate,
        "burstiness": None if math.isinf(args.burstiness) else args.burstiness,
        "self_timed": args.self_timed,
        "max_concurrency": args.max_concurrency,
        "warmup_requests": args.warmups,
        "scheduled_offered_rate_rps": None if math.isinf(expected_rate) else expected_rate,
        "actual_offered_rate_rps": None if math.isinf(actual_offered_rate) else actual_offered_rate,
        "summary": summarize_requests(requests, duration, args.percentiles),
        "requests": requests,
    }
    return result


def parse_rate(value: str) -> float:
    return math.inf if value.lower() == "inf" else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact-token vLLM benchmark client")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--request-rate", type=parse_rate, default=math.inf)
    parser.add_argument("--burstiness", type=parse_rate, default=1.0)
    parser.add_argument("--self-timed", action="store_true")
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--server-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=21600)
    parser.add_argument("--drain-timeout", type=float, default=600)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[50, 90, 95, 99])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = asyncio.run(run_benchmark(args))
    atomic_write_json(output, result)
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
