#!/usr/bin/env python3
"""Run one persistent-server, clean-baseline Candidate profiling matrix."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.isolated_candidate_profile import (
    build_isolated_request_rows,
    validate_isolated_profile,
)
from benchmarks.qwen3_runtime import (
    ActiveConfigError,
    build_isolated_profile_server_command,
    load_active_runtime,
    require_frozen_for_execution,
    resolve_under,
    sha256_file,
)
from benchmarks.run_predictor_profile import _stop_server, _wait_for_server
from benchmarks.run_stock_natural_eos import (
    RUN_ID_PATTERN,
    _atomic_json,
    _git_state,
    _require_free_port,
    load_trace,
    send_one,
)
from benchmarks.targeted_predictor_profile import validate_reused_stock_trace
from dpp_scheduler.targeted_profile import (
    ISOLATED_KNEE_CAMPAIGN_ID,
    ISOLATED_KNEE_SMOKE_CAMPAIGN_ID,
    build_target_recipes,
)
from dpp_scheduler.vllm_adapter import (
    ISOLATED_PROFILE_EVENT_PATH_ENV,
    ISOLATED_PROFILE_PATH_ENV,
    ISOLATED_PROFILE_RECIPE_MODE_ENV,
    ISOLATED_PROFILE_RECIPE_SEED_ENV,
    ISOLATED_PROFILE_RUN_ID_ENV,
)


async def _wait_for_terminal_event(
    path: Path,
    *,
    offset: int,
    recipe_id: str,
    timeout: float,
) -> tuple[dict[str, Any], int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with path.open("r", encoding="utf-8") as stream:
            stream.seek(offset)
            while line := stream.readline():
                offset = stream.tell()
                event = json.loads(line)
                if (
                    event.get("recipe_id") == recipe_id
                    and event.get("event") in {"batch_complete", "batch_failed"}
                ):
                    return event, offset
        await asyncio.sleep(0.01)
    raise TimeoutError(f"isolated batch timed out: {recipe_id}")


async def _run_matrix(
    runtime: Any,
    source_rows: list[dict[str, Any]],
    recipes: tuple[Any, ...],
    *,
    port: int,
    event_path: Path,
    per_request_path: Path,
    batch_timeout: float,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=batch_timeout)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    started = time.perf_counter()
    event_offset = 0
    request_count = 0
    failed_batches: list[dict[str, Any]] = []
    context_tokens: list[int] = []
    with per_request_path.open("x", encoding="utf-8") as output_stream:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for ordinal, recipe in enumerate(recipes):
                rows = build_isolated_request_rows(
                    source_rows, recipe, recipe_ordinal=ordinal
                )
                request_count += len(rows)
                context_tokens.extend(
                    int(row["input_tokens"])
                    for row in rows
                    if row["profile_request_role"] == "decode"
                )
                dispatched = time.perf_counter()
                tasks = [
                    asyncio.create_task(
                        send_one(
                            session,
                            f"http://127.0.0.1:{port}/v1/completions",
                            runtime,
                            row,
                            benchmark_start=started,
                            dispatched_at=dispatched,
                        )
                    )
                    for row in rows
                ]
                event, event_offset = await _wait_for_terminal_event(
                    event_path,
                    offset=event_offset,
                    recipe_id=recipe.recipe_id,
                    timeout=batch_timeout,
                )
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks), timeout=min(60.0, batch_timeout)
                )
                for row, result in zip(rows, results):
                    result["profile_recipe_id"] = recipe.recipe_id
                    result["profile_request_role"] = row["profile_request_role"]
                    output_stream.write(
                        json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                output_stream.flush()
                if event["event"] == "batch_failed":
                    failed_batches.append(event)

    elapsed = time.perf_counter() - started
    return {
        "schema_version": 2,
        "recipe_count": len(recipes),
        "request_count": request_count,
        "failed_batch_count": len(failed_batches),
        "failed_batches": failed_batches,
        "elapsed_seconds": elapsed,
        "decode_context_tokens": {
            "count": len(context_tokens),
            "minimum": min(context_tokens) if context_tokens else None,
            "maximum": max(context_tokens) if context_tokens else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    parser.add_argument("--campaign-id", default=ISOLATED_KNEE_CAMPAIGN_ID)
    parser.add_argument("--source-campaign-id", required=True)
    parser.add_argument("--source-trace-dir", required=True)
    parser.add_argument("--source-trace", required=True)
    parser.add_argument("--source-qps", type=float, required=True)
    parser.add_argument("--source-seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recipe-seed", type=int, required=True)
    parser.add_argument(
        "--recipe-mode",
        choices=("isolated_knee", "isolated_knee_smoke"),
        default="isolated_knee",
    )
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--batch-timeout", type=float, default=600)
    parser.add_argument("--run-timeout", type=float, default=21600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runtime = load_active_runtime(args.config)
    if not all(
        RUN_ID_PATTERN.fullmatch(value)
        for value in (args.campaign_id, args.source_campaign_id, args.run_id)
    ):
        raise ActiveConfigError("invalid campaign/source/run identifier")
    expected_campaign = (
        ISOLATED_KNEE_SMOKE_CAMPAIGN_ID
        if args.recipe_mode == "isolated_knee_smoke"
        else ISOLATED_KNEE_CAMPAIGN_ID
    )
    if args.campaign_id != expected_campaign:
        raise ActiveConfigError("isolated recipe mode/campaign mismatch")
    if (
        not math.isfinite(args.source_qps)
        or args.source_qps <= 0
        or args.startup_timeout <= 0
        or args.batch_timeout <= 0
        or args.run_timeout <= 0
    ):
        raise ActiveConfigError("isolated profiling QPS/timeouts are invalid")

    source_root = resolve_under(
        runtime.raw_results, args.source_campaign_id, label="source campaign"
    )
    trace_root = resolve_under(
        source_root, args.source_trace_dir, label="source trace directory"
    )
    trace_path = resolve_under(trace_root, args.source_trace, label="source trace")
    trace_manifest_path = resolve_under(
        trace_root, "manifest.json", label="source trace manifest"
    )
    source_rows = load_trace(trace_path, runtime)
    trace_validation = validate_reused_stock_trace(
        trace_path,
        trace_manifest_path,
        runtime,
        source_qps=args.source_qps,
        source_seed=args.source_seed,
        expected_request_count=len(source_rows),
    )
    recipes = build_target_recipes(args.recipe_seed, mode=args.recipe_mode)
    campaign_root = resolve_under(
        runtime.raw_results, args.campaign_id, label="isolated campaign"
    )
    output_dir = resolve_under(
        campaign_root, Path("runs") / args.run_id, label="isolated run"
    )
    command = build_isolated_profile_server_command(runtime, port=args.port)
    preview = {
        "schema_version": 2,
        "campaign_id": args.campaign_id,
        "run_id": args.run_id,
        "config": str(runtime.config_path),
        "config_sha256": runtime.config_sha256,
        "recipe_mode": args.recipe_mode,
        "recipe_seed": args.recipe_seed,
        "target_batch_count": len(recipes),
        "prefill_only_target_count": sum(r.decode_request_cap == 0 for r in recipes),
        "mixed_target_count": sum(r.decode_request_cap > 0 for r in recipes),
        "decode_counts": sorted({r.decode_request_cap for r in recipes}),
        "maximum_sequences_per_target": max(
            r.decode_request_cap + r.prefill_request_cap for r in recipes
        ),
        "sequence_budget": runtime.max_num_seqs,
        "maximum_tokens_per_target": max(
            r.prefill_token_cap + r.decode_request_cap for r in recipes
        ),
        "token_budget": runtime.max_num_batched_tokens,
        "source_trace": str(trace_path),
        "source_trace_sha256": sha256_file(trace_path),
        "source_trace_validation": trace_validation,
        "output_dir": str(output_dir),
        "server_command": command,
        "batch_timeout_seconds": args.batch_timeout,
        "run_timeout_seconds": args.run_timeout,
    }
    if preview["maximum_sequences_per_target"] > runtime.max_num_seqs:
        raise ActiveConfigError("isolated matrix exceeds the frozen sequence budget")
    if preview["maximum_tokens_per_target"] > runtime.max_num_batched_tokens:
        raise ActiveConfigError("isolated matrix exceeds the frozen token budget")
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    require_frozen_for_execution(runtime)
    if output_dir.exists():
        raise FileExistsError(f"append-only isolated run exists: {output_dir}")
    _require_free_port(args.port)
    output_dir.mkdir(parents=True)
    _atomic_json(
        output_dir / "recipes.json",
        {
            "schema_version": 2,
            "mode": args.recipe_mode,
            "seed": args.recipe_seed,
            "recipes": [recipe.as_dict() for recipe in recipes],
        },
    )
    profile_path = output_dir / "iteration_profile.jsonl"
    event_path = output_dir / "batch_events.jsonl"
    startup_log = output_dir / "startup.log"
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "kind": "qwen3_14b_isolated_exact_batch_profile",
        "campaign_id": args.campaign_id,
        "run_id": args.run_id,
        "source_qps": args.source_qps,
        "source_seed": args.source_seed,
        "recipe_seed": args.recipe_seed,
        "recipe_mode": args.recipe_mode,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved": preview,
        "git": {
            "root": _git_state(runtime.workspace),
            "vllm": _git_state(runtime.workspace, "vllm"),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    _atomic_json(manifest_path, manifest)
    environment = os.environ.copy()
    environment.update(dict(runtime.required_env))
    environment["PATH"] = f"{runtime.python.parent}:{environment.get('PATH', '')}"
    environment.update(
        {
            ISOLATED_PROFILE_PATH_ENV: str(profile_path),
            ISOLATED_PROFILE_EVENT_PATH_ENV: str(event_path),
            ISOLATED_PROFILE_RUN_ID_ENV: args.run_id,
            ISOLATED_PROFILE_RECIPE_SEED_ENV: str(args.recipe_seed),
            ISOLATED_PROFILE_RECIPE_MODE_ENV: args.recipe_mode,
        }
    )
    process: subprocess.Popen[str] | None = None
    try:
        deadline = time.monotonic() + args.run_timeout
        with startup_log.open("x", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                command,
                cwd=runtime.workspace,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _wait_for_server(
                process,
                port=args.port,
                timeout=min(args.startup_timeout, args.run_timeout),
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("isolated run expired before matrix execution")
            summary = asyncio.run(
                asyncio.wait_for(
                    _run_matrix(
                        runtime,
                        source_rows,
                        recipes,
                        port=args.port,
                        event_path=event_path,
                        per_request_path=output_dir / "per_request.jsonl",
                        batch_timeout=args.batch_timeout,
                    ),
                    timeout=remaining,
                )
            )
        _atomic_json(output_dir / "summary.json", summary)
        validation = validate_isolated_profile(
            profile_path,
            event_path,
            expected_run_id=args.run_id,
            recipe_seed=args.recipe_seed,
            recipe_mode=args.recipe_mode,
        )
        _atomic_json(output_dir / "target_validation.json", validation)
        manifest.update(
            {"status": "complete", "summary": summary, "validation": validation}
        )
        return 0
    except Exception as error:
        manifest.update(
            {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        )
        raise
    finally:
        _stop_server(process)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if startup_log.exists():
            manifest["startup_log_sha256"] = sha256_file(startup_log)
        _atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
