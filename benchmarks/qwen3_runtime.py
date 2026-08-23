"""Shared, config-derived Qwen3-14B runtime validation and launch helpers.

This module is intentionally importable without vLLM.  It makes the sole
active YAML configuration the source of every conclusion-affecting server
argument and keeps the client safety ceiling outside Scheduler-facing state.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dpp_scheduler.settings import (
    FallbackSettings,
    ObligationSettings,
    SafeSetSettings,
    SchedulerSettings,
)


ACTIVE_CONFIG_RELATIVE = Path("configs/dgx_spark_experiment.yaml")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE_STATUSES = frozenset({"frozen_g0", "frozen"})


class ActiveConfigError(ValueError):
    """The active Qwen3 configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class FrozenPredictor:
    artifact_root: Path
    artifact_manifest_sha256: str
    predictor_version: str


@dataclass(frozen=True)
class FrozenCandidateSettings:
    settings: SchedulerSettings
    manifest_path: Path
    manifest_sha256: str
    runtime_signature_sha256: str


def load_frozen_safe_set_settings(runtime: ActiveRuntime) -> SafeSetSettings:
    """Load Safe-Set parameters, rejecting every unresolved/null value."""
    with runtime.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    safe_set = config.get("safe_set") if isinstance(config, dict) else None
    if not isinstance(safe_set, dict):
        raise ActiveConfigError("active config safe_set section is missing")
    try:
        return SafeSetSettings.from_mapping(safe_set)
    except ValueError as error:
        raise ActiveConfigError(str(error)) from error


def load_fallback_settings(runtime: ActiveRuntime) -> FallbackSettings:
    """Load the deterministic Fallback construction policy."""
    with runtime.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    fallback = config.get("fallback") if isinstance(config, dict) else None
    if not isinstance(fallback, dict):
        raise ActiveConfigError("active config fallback section is missing")
    try:
        return FallbackSettings.from_mapping(fallback)
    except ValueError as error:
        raise ActiveConfigError(str(error)) from error


def load_obligation_settings(runtime: ActiveRuntime) -> ObligationSettings:
    """Load live TTFT/TBT ledger deadlines from the active configuration."""
    with runtime.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    slo = config.get("slo") if isinstance(config, dict) else None
    candidate = config.get("candidate_generator") if isinstance(config, dict) else None
    if not isinstance(slo, dict) or not isinstance(candidate, dict):
        raise ActiveConfigError("active config SLO/Candidate sections are missing")
    try:
        return ObligationSettings(
            ttft_slo_seconds=slo.get("ttft_seconds"),
            tbt_slo_seconds=slo.get("tbt_seconds"),
            recovery_age_threshold_seconds=candidate.get(
                "recovery_age_threshold_seconds"
            ),
        )
    except ValueError as error:
        raise ActiveConfigError(str(error)) from error


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
    processed_results: Path
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
    processed_results = Path(_require(paths, "processed_results", where="paths"))
    active_traces = Path(_require(paths, "active_traces", where="paths"))
    if not raw_results.is_absolute():
        raw_results = workspace / raw_results
    if not processed_results.is_absolute():
        processed_results = workspace / processed_results
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
        processed_results=processed_results,
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


def build_isolated_profile_server_command(
    runtime: ActiveRuntime, *, port: int
) -> list[str]:
    """Build the locked command for clean-baseline exact-batch profiling."""
    command = build_stock_server_command(runtime, port=port)
    command.extend(
        [
            "--scheduler-cls",
            (
                "dpp_scheduler.isolated_profile_scheduler."
                "IsolatedProfilingScheduler"
            ),
            "--enable-logging-iteration-details",
        ]
    )
    return command


def build_predictor_evaluation_server_command(
    runtime: ActiveRuntime, *, port: int
) -> list[str]:
    """Build the locked real-vLLM command for Predictor shadow evaluation."""
    command = build_stock_server_command(runtime, port=port)
    command.extend(
        [
            "--scheduler-cls",
            (
                "dpp_scheduler.predictor_evaluation_scheduler."
                "PredictorEvaluationScheduler"
            ),
            "--enable-logging-iteration-details",
        ]
    )
    return command


def load_frozen_predictor(runtime: ActiveRuntime) -> FrozenPredictor:
    """Resolve the exact Predictor artifact from the sole active config."""
    with runtime.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    predictor = config.get("predictor") if isinstance(config, dict) else None
    if not isinstance(predictor, dict):
        raise ActiveConfigError("active config predictor section is missing")
    value = predictor.get("artifact_path")
    expected_hash = predictor.get("artifact_manifest_sha256")
    version = predictor.get("predictor_version")
    if not value or not expected_hash or not version:
        raise ActiveConfigError("frozen Predictor path/hash/version is incomplete")
    root = Path(str(value))
    if not root.is_absolute():
        root = runtime.workspace / root
    root = root.resolve()
    workspace = runtime.workspace.resolve()
    if root != workspace and workspace not in root.parents:
        raise ActiveConfigError("Predictor artifact escapes the configured workspace")
    manifest = root / "artifact_manifest.json"
    if not root.is_dir() or not manifest.is_file():
        raise ActiveConfigError(f"Predictor artifact is absent: {root}")
    if root.lstat().st_uid != os.getuid():
        raise ActiveConfigError(f"Predictor artifact is not user-owned: {root}")
    observed_hash = sha256_file(manifest)
    if observed_hash != str(expected_hash):
        raise ActiveConfigError(
            f"Predictor artifact manifest hash mismatch: {observed_hash}"
        )
    payload = _require(predictor, "predictor_version", where="predictor")
    with manifest.open("r", encoding="utf-8") as stream:
        artifact_manifest = yaml.safe_load(stream)
    if not isinstance(artifact_manifest, dict) or artifact_manifest.get(
        "predictor_version"
    ) != payload:
        raise ActiveConfigError("Predictor artifact version mismatch")
    return FrozenPredictor(root, observed_hash, str(version))


def candidate_runtime_signature(runtime: ActiveRuntime) -> tuple[dict[str, Any], str]:
    """Hash runtime facts that affect Horizon and Prefill-knee measurements."""
    with runtime.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ActiveConfigError("active config must be a mapping")
    model = config.get("model", {})
    engine = config.get("runtime", {})
    predictor = config.get("predictor", {})
    environment = config.get("environment", {})
    payload = {
        "model_name": runtime.model_name,
        "model_revision": runtime.model_revision,
        "tokenizer_revision": runtime.tokenizer_revision,
        "model_snapshot_sha256": model.get("snapshot_sha256"),
        "dtype": model.get("dtype"),
        "kv_cache_dtype": model.get("kv_cache_dtype"),
        "max_model_len": runtime.max_model_len,
        "gpu_memory_utilization": runtime.gpu_memory_utilization,
        "engine": engine.get("engine"),
        "chunked_prefill": engine.get("enable_chunked_prefill"),
        "prefix_caching": engine.get("enable_prefix_caching"),
        "speculative_decoding": engine.get("speculative_decoding"),
        "token_budget": runtime.max_num_batched_tokens,
        "sequence_budget": runtime.max_num_seqs,
        "kv_block_size": runtime.kv_block_size,
        "usable_kv_blocks": runtime.usable_kv_blocks,
        "vllm_commit": environment.get("vllm_commit"),
        "duration_timing_boundary": predictor.get("duration_timing_boundary"),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return payload, hashlib.sha256(canonical).hexdigest()


def load_frozen_candidate_settings(runtime: ActiveRuntime) -> FrozenCandidateSettings:
    """Load Candidate Generator settings only from a matching frozen artifact."""
    with runtime.config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    candidate = config.get("candidate_generator") if isinstance(config, dict) else None
    if not isinstance(candidate, dict):
        raise ActiveConfigError("active config candidate_generator section is missing")
    try:
        settings = SchedulerSettings.from_mapping(candidate)
    except ValueError as error:
        raise ActiveConfigError(str(error)) from error
    if not settings.frozen:
        raise ActiveConfigError("Candidate Generator parameters are not frozen")
    value = candidate.get("freeze_manifest_path")
    expected_hash = candidate.get("freeze_manifest_sha256")
    expected_signature = candidate.get("runtime_signature_sha256")
    freeze_kind = candidate.get("freeze_kind", "measured")
    if not value or not expected_hash or not expected_signature:
        raise ActiveConfigError("candidate freeze manifest/hash/signature is incomplete")
    if freeze_kind not in {"measured", "user_directed_integration"}:
        raise ActiveConfigError("unknown candidate freeze kind")
    manifest_path = Path(str(value))
    if not manifest_path.is_absolute():
        manifest_path = runtime.workspace / manifest_path
    manifest_path = manifest_path.resolve()
    if freeze_kind == "measured":
        allowed_root = runtime.processed_results.resolve()
        expected_name = "manifest.json"
        escape_error = "candidate freeze manifest escapes processed results"
    else:
        allowed_root = (runtime.workspace / "configs").resolve()
        expected_name = "candidate_generator_integration_freeze.json"
        escape_error = "candidate integration freeze escapes configs"
        if candidate.get("formal_benchmark_eligible") is not False:
            raise ActiveConfigError(
                "integration-frozen Candidate parameters cannot enable formal benchmarks"
            )
    if allowed_root not in manifest_path.parents:
        raise ActiveConfigError(escape_error)
    if not manifest_path.is_file() or manifest_path.name != expected_name:
        raise ActiveConfigError(f"candidate freeze manifest is absent: {manifest_path}")
    if manifest_path.lstat().st_uid != os.getuid():
        raise ActiveConfigError("candidate freeze manifest is not user-owned")
    observed_hash = sha256_file(manifest_path)
    if observed_hash != str(expected_hash):
        raise ActiveConfigError("candidate freeze manifest hash mismatch")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected_identity = (
        (2, "candidate_parameter_freeze_v2", "frozen")
        if freeze_kind == "measured"
        else (
            1,
            "candidate_parameter_integration_freeze_v1",
            "frozen_for_scheduler_integration",
        )
    )
    if (
        manifest.get("schema_version"),
        manifest.get("artifact_id"),
        manifest.get("status"),
    ) != expected_identity:
        raise ActiveConfigError("candidate freeze manifest schema/identity mismatch")
    _, observed_signature = candidate_runtime_signature(runtime)
    if observed_signature != str(expected_signature) or manifest.get(
        "runtime_signature_sha256"
    ) != observed_signature:
        raise ActiveConfigError("candidate runtime signature mismatch")
    if manifest.get("parameters_frozen") is not True:
        raise ActiveConfigError("candidate freeze manifest is not frozen")
    if freeze_kind == "user_directed_integration" and manifest.get(
        "formal_benchmark_eligible"
    ) is not False:
        raise ActiveConfigError(
            "candidate integration freeze manifest must reject formal benchmarks"
        )
    manifest_horizon = manifest.get("critical_horizon_seconds")
    manifest_knee = manifest.get("prefill_knee_tokens")
    if isinstance(manifest_horizon, bool) or not isinstance(
        manifest_horizon, (int, float)
    ):
        raise ActiveConfigError("candidate manifest critical horizon is invalid")
    if isinstance(manifest_knee, bool) or not isinstance(manifest_knee, int):
        raise ActiveConfigError("candidate manifest Prefill knee is invalid")
    if float(manifest_horizon) != settings.critical_horizon_seconds:
        raise ActiveConfigError("critical horizon differs from freeze manifest")
    if manifest_knee != settings.prefill_knee_tokens:
        raise ActiveConfigError("Prefill knee differs from freeze manifest")
    if freeze_kind == "user_directed_integration" and (
        manifest.get("maximum_seed_candidates") != settings.maximum_seed_candidates
        or manifest.get("minimum_prefill_chunk_tokens")
        != settings.minimum_prefill_chunk_tokens
    ):
        raise ActiveConfigError("Candidate integration settings differ from manifest")
    return FrozenCandidateSettings(
        settings=settings,
        manifest_path=manifest_path,
        manifest_sha256=observed_hash,
        runtime_signature_sha256=observed_signature,
    )


def resolve_under(root: Path, value: str | Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = resolved_root / path
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ActiveConfigError(f"{label} escapes {resolved_root}: {resolved}")
    return resolved
