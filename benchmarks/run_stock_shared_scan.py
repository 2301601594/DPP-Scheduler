#!/usr/bin/env python3
"""Run one Stock vLLM shared-parameter scan configuration.

This script starts a stock vLLM server, replays a fixed request trace, records
per-request streaming metrics, then stops the server.  It is intentionally
small and not a final benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

import aiohttp


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def wait_for_health(port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5.0
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("vLLM health endpoint did not become ready")


async def send_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    row: dict,
) -> dict:
    payload = {
        "model": model,
        "prompt": row["prompt"],
        "max_tokens": int(row["max_tokens_safety"]),
        "temperature": float(row.get("temperature", 0.0)),
        "ignore_eos": bool(row.get("ignore_eos", False)),
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    headers = {"Content-Type": "application/json", "x-request-id": row["request_id"]}
    result = {
        "request_id": row["request_id"],
        "prompt_id": row["prompt_id"],
        "arrival_time_s": row["arrival_time_s"],
        "input_tokens": row["input_tokens"],
        "success": False,
        "error": None,
        "http_status": None,
        "ttft_ms": None,
        "itls_ms": [],
        "e2e_ms": None,
        "actual_input_tokens": None,
        "actual_output_tokens": None,
        "finish_reason": None,
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
                choice = choices[0]
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
                token_ids = choice.get("token_ids")
                token_count = len(token_ids) if isinstance(token_ids, list) else 1
                now = time.perf_counter()
                if not first_token:
                    result["ttft_ms"] = (now - start) * 1000
                    first_token = True
                else:
                    result["itls_ms"].append((now - last_token) * 1000)
                last_token = now
            result["e2e_ms"] = (time.perf_counter() - start) * 1000
            result["success"] = (
                first_token
                and result["actual_output_tokens"] is not None
                and result["error"] is None
            )
    except Exception as error:  # noqa: BLE001
        result["error"] = f"{type(error).__name__}: {error}"
        result["e2e_ms"] = (time.perf_counter() - start) * 1000
    return result


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * pct / 100)))
    return ordered[index]


def summarize(results: list[dict]) -> dict:
    ttft = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    itls = [x for r in results for x in r["itls_ms"]]
    e2e = [r["e2e_ms"] for r in results if r["e2e_ms"] is not None]
    outputs = [r["actual_output_tokens"] for r in results if r["actual_output_tokens"] is not None]
    failures = [r for r in results if not r["success"]]
    return {
        "num_requests": len(results),
        "success": sum(1 for r in results if r["success"]),
        "failed": len(failures),
        "ttft_ms": {
            "p50": percentile(ttft, 50),
            "p90": percentile(ttft, 90),
            "p95": percentile(ttft, 95),
            "p99": percentile(ttft, 99),
            "mean": round(statistics.mean(ttft), 2) if ttft else None,
        },
        "tbt_ms": {
            "p50": percentile(itls, 50),
            "p90": percentile(itls, 90),
            "p95": percentile(itls, 95),
            "p99": percentile(itls, 99),
            "mean": round(statistics.mean(itls), 2) if itls else None,
        },
        "e2e_ms": {
            "p50": percentile(e2e, 50),
            "p90": percentile(e2e, 90),
            "p95": percentile(e2e, 95),
            "p99": percentile(e2e, 99),
            "mean": round(statistics.mean(e2e), 2) if e2e else None,
        },
        "output_tokens": {
            "min": min(outputs) if outputs else None,
            "max": max(outputs) if outputs else None,
            "mean": round(statistics.mean(outputs), 2) if outputs else None,
        },
        "finish_reason_counts": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="/home/dongj/models/Qwen3-14B-BF16")
    parser.add_argument("--model-name", default="Qwen3-14B")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--workspace", default="/home/dongj/LLM")
    parser.add_argument("--venv-python", default="/home/dongj/LLM/.venv/bin/python")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(args.trace).resolve()
    trace_rows = read_jsonl(trace_path)

    env = os.environ.copy()
    venv_bin = Path(args.venv_python).parent
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    cmd = [
        str(venv_bin / "vllm"),
        "serve",
        args.model_path,
        "--served-model-name",
        args.model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--max-model-len",
        "40960",
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
    ]

    startup_log = output_dir / "startup.log"
    process = None
    try:
        with startup_log.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                cmd,
                cwd=args.workspace,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            wait_for_health(args.port, args.startup_timeout)

            async def run_client() -> list[dict]:
                timeout = aiohttp.ClientTimeout(total=args.request_timeout)
                connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
                async with aiohttp.ClientSession(
                    connector=connector, timeout=timeout
                ) as session:
                    benchmark_start = time.perf_counter()

                    async def dispatch(row: dict) -> dict:
                        delay = benchmark_start + row["arrival_time_s"] - time.perf_counter()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        result = await send_one(
                            session,
                            f"http://127.0.0.1:{args.port}/v1/completions",
                            args.model_name,
                            row,
                        )
                        result["dispatch_offset_s"] = round(
                            time.perf_counter() - benchmark_start, 6
                        )
                        return result

                    return await asyncio.gather(*(dispatch(row) for row in trace_rows))

            results = asyncio.run(run_client())
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    per_request_path = output_dir / "per_request.jsonl"
    with per_request_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    summary = summarize(results)
    finish_counts: dict[str, int] = {}
    for result in results:
        reason = result.get("finish_reason") or ("error" if result.get("error") else "unknown")
        finish_counts[reason] = finish_counts.get(reason, 0) + 1
    summary["finish_reason_counts"] = finish_counts
    summary["trace_sha256"] = __import__("hashlib").sha256(
        trace_path.read_bytes()
    ).hexdigest()
    summary["max_num_batched_tokens"] = args.max_num_batched_tokens
    summary["max_num_seqs"] = args.max_num_seqs

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
