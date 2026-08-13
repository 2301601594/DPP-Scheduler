from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.dppbench.config import config_hash, workspace_path
from benchmarks.dppbench.io import atomic_write_json, atomic_write_text, sha256_file
from benchmarks.dppbench.matrix import RunSpec


class RunFailure(RuntimeError):
    """Raised after an append-only run directory has captured a failure."""


def shell_command(command: list[str]) -> str:
    return shlex.join(command)


def server_command(config: dict[str, Any], spec: RunSpec) -> list[str]:
    model = config["model"]
    command = [
        config["paths"]["vllm_cli"],
        "serve",
        config["paths"]["model_snapshot"],
        "--served-model-name",
        model["id"],
        "--host",
        str(model["host"]),
        "--port",
        str(model["port"]),
        "--dtype",
        str(model["dtype"]),
        "--max-model-len",
        str(model["max_model_len"]),
        "--gpu-memory-utilization",
        str(model["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(model["max_num_seqs"]),
        "--scheduling-policy",
        str(model["scheduling_policy"]),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--generation-config",
        str(model["generation_config"]),
        "--stream-interval",
        "1",
        "--disable-uvicorn-access-log",
    ]
    if spec.budget is not None:
        command.extend(["--max-num-batched-tokens", str(spec.budget)])
    return command


def client_command(
    config: dict[str, Any], spec: RunSpec, output_path: Path
) -> list[str]:
    model = config["model"]
    request_rate = (
        "inf" if spec.request_rate_rps is None or math.isinf(spec.request_rate_rps) else f"{spec.request_rate_rps:.12g}"
    )
    command = [
        config["paths"]["python"],
        "-m",
        "benchmarks.benchmark_client",
        "--trace",
        spec.trace_path,
        "--output",
        str(output_path),
        "--endpoint",
        f"http://{model['host']}:{model['port']}/v1/completions",
        "--model",
        model["id"],
        "--limit",
        str(
            config["statistics"]["measurement_requests"]
            if spec.request_limit is None
            else spec.request_limit
        ),
        "--request-rate",
        request_rate,
        "--warmups",
        str(
            config["statistics"]["warmup_requests"]
            if spec.warmup_requests is None
            else spec.warmup_requests
        ),
        "--seed",
        str(spec.seed),
        "--server-seed",
        str(model["seed"]),
        "--temperature",
        str(model["temperature"]),
        "--timeout",
        str(config["validity"]["request_timeout_s"]),
        "--drain-timeout",
        str(
            0
            if spec.max_concurrency == 1 and math.isinf(spec.request_rate_rps or math.inf)
            else config["validity"]["drain_timeout_s"]
        ),
        "--percentiles",
        *[str(value) for value in config["statistics"]["percentiles"]],
    ]
    if model["ignore_eos"]:
        command.append("--ignore-eos")
    if spec.self_timed:
        command.append("--self-timed")
    elif spec.burstiness is not None:
        command.extend(["--burstiness", str(spec.burstiness)])
    if spec.max_concurrency is not None:
        command.extend(["--max-concurrency", str(spec.max_concurrency)])
    return command


def resolver_command(
    config: dict[str, Any], spec: RunSpec, output_path: Path
) -> list[str]:
    command = [
        config["paths"]["python"],
        str(Path(config["paths"]["workspace"]) / "benchmarks/resolve_vllm_config.py"),
        "--config",
        config["_config_path"],
        "--output",
        str(output_path),
    ]
    if spec.budget is not None:
        command.extend(["--budget", str(spec.budget)])
    return command


def _run_output(command: list[str], cwd: Path) -> str:
    process = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return process.stdout.strip()


def capture_environment(config: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(config["paths"]["workspace"])
    source = Path(config["paths"]["vllm_source"])
    commands = {
        "uname": ["uname", "-a"],
        "os_release": ["sh", "-c", "sed -n '1,80p' /etc/os-release"],
        "lscpu": ["lscpu"],
        "memory": ["free", "-b"],
        "gpu": [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        "top_git_commit": ["git", "rev-parse", "HEAD"],
        "top_git_dirty": ["git", "status", "--short"],
        "python_runtime": [config["paths"]["python"], "--version"],
        "torch_runtime": [
            config["paths"]["python"],
            "-c",
            "import torch; print(torch.__version__); print(torch.version.cuda)",
        ],
        "vllm_runtime": [
            config["paths"]["python"],
            "-c",
            "import importlib.metadata as m; print(m.version('vllm'))",
        ],
    }
    environment = {}
    for name, command in commands.items():
        try:
            environment[name] = _run_output(command, workspace)
        except OSError as error:
            environment[name] = f"ERROR: {error}"
    if environment.get("top_git_commit", "").startswith("fatal:"):
        environment["top_git_commit"] = "UNBORN (repository initialized, no commit created)"
    environment["vllm_git_commit"] = _run_output(["git", "rev-parse", "HEAD"], source)
    environment["vllm_git_dirty"] = _run_output(["git", "status", "--short"], source)
    environment["configured"] = config["environment"]
    return environment


def code_snapshot(config: dict[str, Any]) -> dict[str, str]:
    workspace = Path(config["paths"]["workspace"])
    paths = list((workspace / "benchmarks").rglob("*.py"))
    paths.extend(
        [
            workspace / "configs/frozen_experiment.yaml",
            workspace / "configs/slo.yaml",
            workspace / "AGENTS.md",
        ]
    )
    return {
        str(path.relative_to(workspace)): sha256_file(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


class GpuMonitor:
    fields = (
        "timestamp",
        "temperature.gpu",
        "power.draw",
        "clocks.sm",
        "memory.used",
        "utilization.gpu",
    )

    def __init__(self, path: Path):
        self.path = path
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.started = False

    def start(self) -> None:
        self.thread.start()
        self.started = True

    def stop(self) -> None:
        self.stop_event.set()
        if self.started:
            self.thread.join(timeout=5)

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.fields)
            while not self.stop_event.is_set():
                command = [
                    "nvidia-smi",
                    f"--query-gpu={','.join(self.fields)}",
                    "--format=csv,noheader,nounits",
                ]
                try:
                    output = subprocess.run(
                        command,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    ).stdout.strip()
                    if output:
                        writer.writerow([part.strip() for part in output.split(",")])
                        stream.flush()
                except (OSError, subprocess.TimeoutExpired):
                    pass
                self.stop_event.wait(1.0)


_SERVER_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


def _scrape_server_metrics(config: dict[str, Any]) -> dict[str, float]:
    model = config["model"]
    url = f"http://{model['host']}:{model['port']}/metrics"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        body = response.read().decode("utf-8", errors="replace")
    values: dict[str, float] = defaultdict(float)
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if name not in _SERVER_METRICS:
            continue
        try:
            values[name] += float(line.rsplit(None, 1)[-1])
        except ValueError:
            continue
    return dict(values)


class ServerMetricsMonitor:
    def __init__(self, config: dict[str, Any], path: Path):
        self.config = config
        self.path = path
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.started = False

    def start(self) -> None:
        self.thread.start()
        self.started = True

    def stop(self) -> None:
        self.stop_event.set()
        if self.started:
            self.thread.join(timeout=5)

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as stream:
            while not self.stop_event.is_set():
                try:
                    sample = {
                        "timestamp_unix_s": time.time(),
                        "metrics": _scrape_server_metrics(self.config),
                    }
                    stream.write(json.dumps(sample, sort_keys=True) + "\n")
                    stream.flush()
                except Exception:
                    pass
                self.stop_event.wait(1.0)


def _wait_for_health(config: dict[str, Any], process: subprocess.Popen, timeout: float = 300) -> None:
    model = config["model"]
    url = f"http://{model['host']}:{model['port']}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunFailure(f"server exited with code {process.returncode}")
        try:
            with opener.open(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RunFailure(f"server health check timed out after {timeout}s")


def _health(config: dict[str, Any]) -> bool:
    model = config["model"]
    url = f"http://{model['host']}:{model['port']}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def server_is_healthy(config: dict[str, Any]) -> bool:
    return _health(config)


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _parse_budget(server_log: str) -> int | None:
    matches = re.findall(r"max_num_batched_tokens[=:](?:\s*)?(\d+)", server_log)
    return int(matches[-1]) if matches else None


def _parse_kv_capacity(server_log: str) -> int | None:
    patterns = (
        r"GPU KV cache size:\s*([\d,]+)\s*tokens",
        r"KV cache.*?([\d,]+)\s*tokens",
    )
    for pattern in patterns:
        matches = re.findall(pattern, server_log, flags=re.IGNORECASE)
        if matches:
            return int(matches[-1].replace(",", ""))
    return None


def _validity(
    config: dict[str, Any],
    result: dict[str, Any],
    server_log: str,
    healthy: bool,
    final_metrics: dict[str, float],
    spec: RunSpec | None = None,
) -> dict[str, Any]:
    summary = result["summary"]
    scheduled = result.get("scheduled_offered_rate_rps")
    actual = result.get("actual_offered_rate_rps")
    rate_error = (
        abs(actual - scheduled) / scheduled
        if actual is not None and scheduled not in (None, 0)
        else None
    )
    checks = {
        "failed_requests": summary["failed"] <= config["validity"]["failed_requests_max"],
        "output_lengths": summary["output_length_mismatches"]
        <= config["validity"]["output_length_mismatch_max"],
        "input_lengths": summary["input_length_mismatches"] == 0,
        "offered_rate": rate_error is None
        or rate_error <= config["validity"]["offered_rate_relative_error_max"],
        "no_oom": "out of memory" not in server_log.lower(),
        "no_startup_conflict": "address already in use" not in server_log.lower(),
        "drained_and_healthy": healthy,
        "queues_drained": final_metrics.get("vllm:num_requests_running", 0.0) == 0
        and final_metrics.get("vllm:num_requests_waiting", 0.0) == 0,
        "single_token_stream_chunks": summary["multi_token_chunks"] == 0,
    }
    saturation_stream_warning = (
        spec is not None
        and spec.stage == "g1"
        and spec.mode == "saturation"
        and not checks["single_token_stream_chunks"]
    )
    hard_checks = {
        name: passed
        for name, passed in checks.items()
        if not (saturation_stream_warning and name == "single_token_stream_chunks")
    }
    warnings = []
    if saturation_stream_warning:
        warnings.append(
            {
                "code": "g1_saturation_multi_token_stream_chunks",
                "multi_token_chunks": int(summary["multi_token_chunks"]),
                "throughput_valid": True,
                "token_timing_exact": False,
            }
        )
    return {
        "valid": all(hard_checks.values()),
        "checks": checks,
        "warnings": warnings,
        "metric_validity": {
            "throughput": all(hard_checks.values()),
            "token_timing_exact": checks["single_token_stream_chunks"],
        },
        "offered_rate_relative_error": rate_error,
        "oom_events": server_log.lower().count("out of memory"),
    }


def execute_run(
    config: dict[str, Any], spec: RunSpec, *, run_attempt: int = 1
) -> Path:
    trace = Path(spec.trace_path)
    if not trace.exists():
        raise FileNotFoundError(trace)
    raw_root = workspace_path(config, "raw_results")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S.%f")
    run_id = f"{timestamp}-{spec.stage}-{spec.run_key}"
    run_dir = raw_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    server_log_path = run_dir / "server.log"
    client_log_path = run_dir / "client.log"
    result_path = run_dir / "client_result.json"
    resolved_config_path = run_dir / "resolved_vllm_config.json"
    server = server_command(config, spec)
    client = client_command(config, spec, result_path)
    atomic_write_text(run_dir / "server_command.txt", shell_command(server) + "\n")
    atomic_write_text(run_dir / "client_command.txt", shell_command(client) + "\n")
    atomic_write_json(run_dir / "environment.json", capture_environment(config))
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "run_key": spec.run_key,
        "run_attempt": run_attempt,
        "status": "starting",
        "config_sha256": config_hash(config),
        "trace_sha256": sha256_file(trace),
        "code_sha256": code_snapshot(config),
        "run_spec": asdict(spec),
        "started_at": datetime.now().astimezone().isoformat(),
    }
    atomic_write_json(run_dir / "metadata.json", metadata)

    environment = os.environ.copy()
    environment.update({key: str(value) for key, value in config["environment"]["required_env"].items()})
    environment["PATH"] = f"{Path(config['paths']['python']).parent}:{environment.get('PATH', '')}"
    environment["PYTHONPATH"] = (
        f"{config['paths']['vllm_source']}:{config['paths']['workspace']}:"
        + environment.get("PYTHONPATH", "")
    ).rstrip(":")
    monitor = GpuMonitor(run_dir / "system_metrics.csv")
    server_monitor = ServerMetricsMonitor(config, run_dir / "server_metrics.jsonl")
    process: subprocess.Popen | None = None
    try:
        with (run_dir / "config_resolution.log").open("w", encoding="utf-8") as log:
            resolution = subprocess.run(
                resolver_command(config, spec, resolved_config_path),
                cwd=config["paths"]["vllm_source"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if resolution.returncode != 0 or not resolved_config_path.exists():
            raise RunFailure(
                f"vLLM config resolution failed with code {resolution.returncode}"
            )
        resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        scheduler_config = resolved_config["scheduler_config"]
        resolved_budget = int(scheduler_config["max_num_batched_tokens"])
        with server_log_path.open("w", encoding="utf-8") as server_log:
            if _health(config):
                raise RunFailure(
                    f"refusing to start: {config['model']['host']}:"
                    f"{config['model']['port']} already serves a health endpoint"
                )
            process = subprocess.Popen(
                server,
                cwd=config["paths"]["vllm_source"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            monitor.start()
            _wait_for_health(config, process)
            time.sleep(1.0)
            if process.poll() is not None:
                raise RunFailure(
                    f"server exited with code {process.returncode} after health check"
                )
            server_monitor.start()
            metadata["status"] = "running"
            atomic_write_json(run_dir / "metadata.json", metadata)
            with client_log_path.open("w", encoding="utf-8") as client_log:
                client_process = subprocess.run(
                    client,
                    cwd=config["paths"]["workspace"],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if not result_path.exists():
                raise RunFailure(f"client exited {client_process.returncode} without result")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            healthy = _health(config)
            try:
                final_metrics = _scrape_server_metrics(config)
            except Exception:
                final_metrics = {}
            server_text = server_log_path.read_text(encoding="utf-8", errors="replace")
            log_budget = _parse_budget(server_text)
            metadata["resolved_max_num_batched_tokens"] = (
                log_budget if log_budget is not None else resolved_budget
            )
            metadata["scheduler_config_source"] = (
                "server_log" if log_budget is not None else "locked_vllm_engine_args"
            )
            metadata["scheduler_config"] = scheduler_config
            metadata["kv_cache_capacity_tokens"] = _parse_kv_capacity(server_text)
            metadata["final_server_metrics"] = final_metrics
            metadata["client_exit_code"] = client_process.returncode
            metadata["validity"] = _validity(
                config, result, server_text, healthy, final_metrics, spec
            )
            if spec.budget is not None:
                budget_matches = resolved_budget == spec.budget
                metadata["validity"]["checks"]["explicit_budget_matches"] = budget_matches
                metadata["validity"]["valid"] &= budget_matches
            if client_process.returncode not in (0, 2):
                metadata["validity"]["valid"] = False
            metadata["status"] = (
                "complete_with_warnings"
                if metadata["validity"]["valid"]
                and metadata["validity"].get("warnings")
                else "complete"
            )
            metadata["finished_at"] = datetime.now().astimezone().isoformat()
            atomic_write_json(run_dir / "metadata.json", metadata)
            return run_dir
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        atomic_write_json(run_dir / "metadata.json", metadata)
        raise
    finally:
        server_monitor.stop()
        if process is not None:
            _stop_server(process)
        monitor.stop()
