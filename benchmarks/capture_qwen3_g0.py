#!/usr/bin/env python3
"""Capture G0 stock-Scheduler facts for Qwen3-14B on DGX Spark.

This script is a bounded, auditable G0 capture helper:

1. Resolves the exact vLLM EngineArgs/SchedulerConfig/CacheConfig.
2. Starts a stock vLLM server (default scheduler, no ModularDPPScheduler).
3. Waits for /health.
4. Sends one natural-EOS smoke completion.
5. Stops the server.
6. Writes resolved config, startup log, smoke response, and a G0 manifest.

It is intentionally not a DPP benchmark: no parameter search, no formal SLO
comparison, and no unapproved long-running matrix.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(workspace: Path, repo: str = ".") -> str:
    return subprocess.check_output(
        ["git", "-C", str(workspace / repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def git_dirty(workspace: Path, repo: str = ".") -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(workspace / repo), "status", "--porcelain"],
            text=True,
        ).strip()
    )


def resolve_engine_config(args: argparse.Namespace) -> tuple[Any, Any]:
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=args.model_path,
        served_model_name=["Qwen3-14B"],
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        scheduling_policy="fcfs",
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        stream_interval=1,
        seed=0,
    )
    resolved = engine_args.create_engine_config()
    return engine_args, resolved


def http_get_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_health(port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5.0
            ) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(3)
    raise RuntimeError(f"vLLM health endpoint did not become ready: {last_error}")


def run_smoke_completion(port: int, timeout: float = 120.0) -> dict[str, Any]:
    payload = {
        "model": "Qwen3-14B",
        "prompt": "Hello, write one short sentence about scheduling.",
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_startup_kv_facts(log_text: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    patterns = {
        "gpu_kv_cache_tokens": r"GPU KV cache size:\s*([0-9,]+)\s*tokens",
        "available_kv_cache_gib": r"Available KV cache memory:\s*([0-9.]+)\s*GiB",
        "kv_cache_max_concurrency": (
            r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x"
        ),
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, log_text)
        if match:
            groups = match.groups()
            if key == "kv_cache_max_concurrency":
                facts["kv_cache_max_tokens_per_request"] = int(groups[0].replace(",", ""))
                facts["kv_cache_max_concurrency"] = float(groups[1])
            else:
                facts[key] = (
                    int(groups[0].replace(",", ""))
                    if "tokens" in key or "concurrency" in key
                    else float(groups[0])
                )
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/home/dongj/LLM")
    parser.add_argument(
        "--venv-python",
        default="/home/dongj/LLM/.venv/bin/python",
    )
    parser.add_argument(
        "--model-path",
        default="/home/dongj/models/Qwen3-14B-BF16",
    )
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--completion-timeout", type=float, default=120.0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Unique output directory under results/raw/qwen3_14b_dgx_spark.",
    )
    parser.add_argument("--no-server", action="store_true", help="Resolve config only.")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-g0-stock"
        output_dir = (
            workspace
            / "results/raw/qwen3_14b_dgx_spark"
            / run_id
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve vLLM config without starting GPU work.
    engine_args, resolved = resolve_engine_config(args)
    resolved_payload = {
        "engine_args": _jsonable(engine_args),
        "scheduler_config": _jsonable(resolved.scheduler_config),
        "cache_config": _jsonable(resolved.cache_config),
        "model_config": _jsonable(resolved.model_config),
    }
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as stream:
        json.dump(resolved_payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    host_facts: dict[str, Any] = {}
    try:
        host_facts["hostname"] = subprocess.check_output(["hostname"], text=True).strip()
        host_facts["arch"] = subprocess.check_output(["uname", "-m"], text=True).strip()
        host_facts["python"] = subprocess.check_output(
            [args.venv_python, "--version"], text=True
        ).strip()
        host_facts["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        host_facts["capture_error"] = repr(exc)

    env_facts = {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "PYTHONHASHSEED",
            "PATH",
            "VLLM_CONFIG",
        )
        if key in os.environ
    }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "g0",
        "kind": "qwen3_14b_stock_g0_capture",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "model_path": args.model_path,
        "runtime": {
            "engine": "vllm_v1",
            "dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "scheduler_cls": None,
        },
        "git": {
            "root_commit": git_head(workspace),
            "root_dirty": git_dirty(workspace),
            "vllm_commit": git_head(workspace, "vllm"),
            "vllm_dirty": git_dirty(workspace, "vllm"),
        },
        "host": host_facts,
        "environment": env_facts,
        "resolved_config_file": "resolved_config.json",
        "resolved_config_sha256": sha256_file(output_dir / "resolved_config.json"),
        "server": {
            "started": False,
            "health_ok": False,
            "completion_ok": False,
        },
        "kv_facts": {},
        "startup_log_sha256": None,
        "smoke_completion_file": None,
    }

    if not args.no_server:
        import socket

        for attempt in range(30):
            try:
                with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                    raise RuntimeError(
                        f"port {args.port} is already in use; refusing to start"
                    )
            except OSError:
                break
        else:
            raise RuntimeError(f"port {args.port} remained occupied")

        startup_log = output_dir / "startup.log"
        env = os.environ.copy()
        venv_bin = Path(args.venv_python).parent
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        vllm_cli = venv_bin / "vllm"
        cmd = [
            str(vllm_cli),
            "serve",
            args.model_path,
            "--served-model-name",
            "Qwen3-14B",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "bfloat16",
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--no-async-scheduling",
        ]
        manifest["server"]["command"] = cmd

        process: subprocess.Popen | None = None
        try:
            with startup_log.open("w", encoding="utf-8") as log_stream:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(workspace),
                    env=env,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                wait_for_health(args.port, args.startup_timeout)
                manifest["server"]["started"] = True
                manifest["server"]["health_ok"] = True

                completion = run_smoke_completion(
                    args.port, timeout=args.completion_timeout
                )
                completion_file = output_dir / "smoke_completion.json"
                with completion_file.open("w", encoding="utf-8") as stream:
                    json.dump(completion, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                manifest["server"]["completion_ok"] = True
                manifest["smoke_completion_file"] = completion_file.name
                manifest["smoke_completion_sha256"] = sha256_file(completion_file)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

        log_text = startup_log.read_text(encoding="utf-8", errors="replace")
        manifest["kv_facts"] = parse_startup_kv_facts(log_text)
        manifest["startup_log_sha256"] = sha256_file(startup_log)

    manifest["status"] = "complete" if manifest["server"]["health_ok"] else "config_only"
    manifest_path = output_dir / "g0_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"G0 capture written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
