"""Shared, config-derived Qwen3-14B runtime validation and launch helpers.

This module is intentionally importable without vLLM.  It makes the sole
active YAML configuration the source of every conclusion-affecting server
argument and keeps the client safety ceiling outside Scheduler-facing state.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ACTIVE_CONFIG_RELATIVE = Path("configs/dgx_spark_experiment.yaml")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE_STATUSES = frozenset({"frozen_g0", "frozen"})


class ActiveConfigError(ValueError):
    """The active Qwen3 configuration is missing or internally inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(mapping: dict[str, Any], key: str, *, where: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ActiveConfigError(f"missing required {where}.{key}")
    return mapping[key]


@dataclass(frozen=True)
class ActiveRuntime:
    config_path: Path
    config_sha256: str
    status: str
    campaign: str
    workspace: Path
    python: Path
    vllm_cli: Path
    model_path: Path
    model_name: str
    model_revision: str
    tokenizer_revision: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    kv_block_size: int
    usable_kv_blocks: int
    raw_results: Path
    active_traces: Path
    client_safety_ceiling_tokens: int
    temperature: float
    top_p: float
    ignore_eos: bool
    seed_source: str
    source_dataset: Path
    source_dataset_sha256: str
    request_pool: Path
    request_pool_manifest: Path
    min_input_tokens: int
    max_input_tokens: int
    pool_size: int
    pool_seed: int
    required_env: tuple[tuple[str, str], ...]

    @property
    def executable(self) -> bool:
        return self.status in EXECUTABLE_STATUSES


def load_active_runtime(config_path: str | Path) -> ActiveRuntime:
    path = Path(config_path).resolve()
    expected_path = (REPOSITORY_ROOT / ACTIVE_CONFIG_RELATIVE).resolve()
    if path != expected_path:
        raise ActiveConfigError(
            f"only {expected_path} may drive Qwen3 execution; got {path}"
        )
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ActiveConfigError("active config must be a mapping")

    paths = _require(config, "paths", where="config")
    model = _require(config, "model", where="config")
    runtime = _require(config, "runtime", where="config")
    generation = _require(config, "generation", where="config")
    trace_source = _require(config, "trace_source", where="config")
    if not all(
        isinstance(item, dict)
        for item in (paths, model, runtime, generation, trace_source)
    ):
        raise ActiveConfigError(
            "paths/model/runtime/generation/trace_source must be mappings"
        )

    expected = {
        "campaign": "qwen3_14b_dgx_spark_modular_dpp",
        "model.requested_name": "Qwen3-14B",
        "model.dtype": "bfloat16",
        "model.kv_cache_dtype": "bfloat16",
        "runtime.engine": "vllm_v1",
        "runtime.enable_chunked_prefill": True,
        "runtime.enable_prefix_caching": False,
        "runtime.speculative_decoding": False,
        "runtime.lora": False,
        "runtime.per_decode_request_tokens_per_iteration": 1,
        "generation.predetermined_output_length": False,
        "generation.scheduler_output_length_state": "forbidden",
    }
    observed = {
        "campaign": config.get("campaign"),
        "model.requested_name": model.get("requested_name"),
        "model.dtype": model.get("dtype"),
        "model.kv_cache_dtype": model.get("kv_cache_dtype"),
        "runtime.engine": runtime.get("engine"),
        "runtime.enable_chunked_prefill": runtime.get("enable_chunked_prefill"),
        "runtime.enable_prefix_caching": runtime.get("enable_prefix_caching"),
        "runtime.speculative_decoding": runtime.get("speculative_decoding"),
        "runtime.lora": runtime.get("lora"),
        "runtime.per_decode_request_tokens_per_iteration": runtime.get(
            "per_decode_request_tokens_per_iteration"
        ),
        "generation.predetermined_output_length": generation.get(
            "predetermined_output_length"
        ),
        "generation.scheduler_output_length_state": generation.get(
            "scheduler_output_length_state"
        ),
    }
    mismatches = [
        f"{key}: expected {value!r}, got {observed[key]!r}"
        for key, value in expected.items()
        if observed[key] != value
    ]
    quantization = model.get("quantization") or {}
    if quantization.get("enabled") is not False or quantization.get("method") != "none":
        mismatches.append("model.quantization must be explicitly disabled")
    if int(model.get("tensor_parallel_size", 0)) != 1:
        mismatches.append("model.tensor_parallel_size must equal 1")
    if int(model.get("pipeline_parallel_size", 0)) != 1:
        mismatches.append("model.pipeline_parallel_size must equal 1")
    if mismatches:
        raise ActiveConfigError("; ".join(mismatches))

    sampling = _require(generation, "sampling_parameters", where="generation")
    if not isinstance(sampling, dict):
        raise ActiveConfigError("generation.sampling_parameters must be a mapping")
    safety_ceiling = int(
        _require(generation, "client_safety_ceiling_tokens", where="generation")
    )
    if safety_ceiling <= 0:
        raise ActiveConfigError("client safety ceiling must be positive")
    if generation.get("client_safety_ceiling_role") != (
        "termination_guard_only_never_scheduler_input"
    ):
        raise ActiveConfigError(
            "client_safety_ceiling_role must explicitly forbid Scheduler use"
        )

    workspace = Path(_require(paths, "workspace", where="paths"))
    raw_results = Path(_require(paths, "raw_results", where="paths"))
    active_traces = Path(_require(paths, "active_traces", where="paths"))
    if not raw_results.is_absolute():
        raw_results = workspace / raw_results
    if not active_traces.is_absolute():
        active_traces = workspace / active_traces
    source_dataset = Path(
        _require(trace_source, "dataset", where="trace_source")
    )
    request_pool = resolve_under(
        active_traces,
        _require(trace_source, "request_pool", where="trace_source"),
        label="request pool",
    )
    request_pool_manifest = resolve_under(
        active_traces,
        _require(trace_source, "request_pool_manifest", where="trace_source"),
        label="request pool manifest",
    )
    min_input_tokens = int(
        _require(trace_source, "min_input_tokens", where="trace_source")
    )
    max_input_tokens = int(
        _require(trace_source, "max_input_tokens", where="trace_source")
    )
    pool_size = int(_require(trace_source, "pool_size", where="trace_source"))
    if min_input_tokens <= 0 or max_input_tokens < min_input_tokens or pool_size <= 0:
        raise ActiveConfigError("invalid trace_source token or pool-size bounds")
    if trace_source.get("enable_thinking") is not False:
        raise ActiveConfigError("trace_source.enable_thinking must be false")

    environment = _require(config, "environment", where="config")
    if not isinstance(environment, dict):
        raise ActiveConfigError("environment must be a mapping")
    required_env = _require(environment, "required_env", where="environment")
    if not isinstance(required_env, dict):
        raise ActiveConfigError("environment.required_env must be a mapping")

    return ActiveRuntime(
        config_path=path,
        config_sha256=sha256_file(path),
        status=str(_require(config, "status", where="config")),
        campaign=str(_require(config, "campaign", where="config")),
        workspace=workspace,
        python=Path(_require(paths, "python", where="paths")),
        vllm_cli=Path(_require(paths, "vllm_cli", where="paths")),
        model_path=Path(_require(paths, "model_snapshot", where="paths")),
        model_name=str(_require(model, "requested_name", where="model")),
        model_revision=str(_require(model, "revision", where="model")),
        tokenizer_revision=str(_require(model, "tokenizer_revision", where="model")),
        max_model_len=int(_require(model, "max_model_len", where="model")),
        gpu_memory_utilization=float(
            _require(model, "gpu_memory_utilization", where="model")
        ),
        max_num_batched_tokens=int(
            _require(runtime, "total_token_budget", where="runtime")
        ),
        max_num_seqs=int(_require(runtime, "sequence_budget", where="runtime")),
        kv_block_size=int(_require(runtime, "kv_block_size", where="runtime")),
        usable_kv_blocks=int(_require(runtime, "usable_kv_blocks", where="runtime")),
        raw_results=raw_results,
        active_traces=active_traces,
        client_safety_ceiling_tokens=safety_ceiling,
        temperature=float(_require(sampling, "temperature", where="sampling_parameters")),
        top_p=float(_require(sampling, "top_p", where="sampling_parameters")),
        ignore_eos=bool(_require(sampling, "ignore_eos", where="sampling_parameters")),
        seed_source=str(_require(sampling, "seed_source", where="sampling_parameters")),
        source_dataset=source_dataset,
        source_dataset_sha256=str(
            _require(trace_source, "dataset_sha256", where="trace_source")
        ),
        request_pool=request_pool,
        request_pool_manifest=request_pool_manifest,
        min_input_tokens=min_input_tokens,
        max_input_tokens=max_input_tokens,
        pool_size=pool_size,
        pool_seed=int(_require(trace_source, "pool_seed", where="trace_source")),
        required_env=tuple(
            sorted((str(key), str(value)) for key, value in required_env.items())
        ),
    )


def require_frozen_for_execution(runtime: ActiveRuntime) -> None:
    if not runtime.executable:
        raise ActiveConfigError(
            f"active config status is {runtime.status!r}; real server execution "
            f"requires one of {sorted(EXECUTABLE_STATUSES)}"
        )
    if sha256_file(runtime.config_path) != runtime.config_sha256:
        raise ActiveConfigError("active config changed after it was resolved")
    if REPOSITORY_ROOT.resolve() != runtime.workspace.resolve():
        raise ActiveConfigError(
            f"execution repository mismatch: {REPOSITORY_ROOT} != {runtime.workspace}"
        )
    user_root = runtime.workspace.parent.resolve()
    for label, path in (
        ("python", runtime.python),
        ("vllm_cli", runtime.vllm_cli),
        ("model_snapshot", runtime.model_path),
    ):
        # Containment and ownership are checked on the configured path itself so
        # that a user-owned venv symlink to the system interpreter is allowed.
        configured = path.absolute()
        try:
            configured.relative_to(user_root)
        except ValueError:
            raise ActiveConfigError(
                f"{label} escapes owned user root: {configured}"
            ) from None
        if not path.exists():
            raise ActiveConfigError(f"{label} does not exist: {configured}")
        if configured.lstat().st_uid != os.getuid():
            raise ActiveConfigError(
                f"{label} is not owned by the current user: {configured}"
            )
    if not runtime.python.is_file() or not os.access(runtime.python, os.X_OK):
        raise ActiveConfigError(f"python is not executable: {runtime.python}")
    if not runtime.vllm_cli.is_file() or not os.access(runtime.vllm_cli, os.X_OK):
        raise ActiveConfigError(f"vllm_cli is not executable: {runtime.vllm_cli}")
    if not runtime.model_path.is_dir():
        raise ActiveConfigError(f"model snapshot is not a directory: {runtime.model_path}")


def build_stock_server_command(runtime: ActiveRuntime, *, port: int) -> list[str]:
    if not 1 <= port <= 65535:
        raise ActiveConfigError(f"invalid TCP port: {port}")
    return [
        str(runtime.vllm_cli),
        "serve",
        str(runtime.model_path),
        "--served-model-name",
        runtime.model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--max-model-len",
        str(runtime.max_model_len),
        "--gpu-memory-utilization",
        str(runtime.gpu_memory_utilization),
        "--max-num-seqs",
        str(runtime.max_num_seqs),
        "--max-num-batched-tokens",
        str(runtime.max_num_batched_tokens),
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--scheduling-policy",
        "fcfs",
        "--generation-config",
        "vllm",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--stream-interval",
        "1",
    ]


def build_stock_profile_server_command(
    runtime: ActiveRuntime, *, port: int
) -> list[str]:
    """Build the locked Stock command with profiling-only observability."""
    command = build_stock_server_command(runtime, port=port)
    command.extend(
        [
            "--scheduler-cls",
            "dpp_scheduler.stock_profile_scheduler.StockProfilingScheduler",
            "--enable-logging-iteration-details",
        ]
    )
    return command


def build_targeted_profile_server_command(
    runtime: ActiveRuntime, *, port: int
) -> list[str]:
    """Build the locked command for exact targeted profiling."""
    command = build_stock_server_command(runtime, port=port)
    command.extend(
        [
            "--scheduler-cls",
            "dpp_scheduler.targeted_profile_scheduler.TargetedProfilingScheduler",
            "--enable-logging-iteration-details",
        ]
    )
    return command


def resolve_under(root: Path, value: str | Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = resolved_root / path
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ActiveConfigError(f"{label} escapes {resolved_root}: {resolved}")
    return resolved
