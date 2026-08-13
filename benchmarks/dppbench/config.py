from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the frozen experiment configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ConfigError("configuration root must be a mapping")
    _validate(config)
    config["_config_path"] = str(config_path)
    return config


def _validate(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "status",
        "paths",
        "environment",
        "model",
        "statistics",
        "workloads",
        "datasets",
        "arrivals",
        "budgets",
        "validity",
        "g1",
        "g2",
    }
    missing = required - config.keys()
    if missing:
        raise ConfigError(f"missing configuration sections: {sorted(missing)}")
    if config["status"] != "frozen":
        raise ConfigError("benchmark execution requires status: frozen")

    stats = config["statistics"]
    if int(stats["measurement_requests"]) <= 0:
        raise ConfigError("measurement_requests must be positive")
    for key in (
        "serial_measurement_requests",
        "saturation_measurement_requests",
        "low_load_measurement_requests",
    ):
        if int(stats[key]) <= 0:
            raise ConfigError(f"{key} must be positive")
    seeds = stats["seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ConfigError("seeds must be a non-empty unique list")

    model = config["model"]
    if int(model["max_num_seqs"]) > min(config["budgets"]["fixed"]):
        raise ConfigError("every fixed budget must be >= max_num_seqs")
    if model["enable_prefix_caching"]:
        raise ConfigError("prefix caching must remain disabled for G0-G3")
    if not model["enable_chunked_prefill"]:
        raise ConfigError("chunked prefill must remain enabled for G0-G3")

    max_len = int(model["max_model_len"])
    for name in ("decode_heavy", "balanced", "prefill_heavy", "long_prefill"):
        workload = config["workloads"][name]
        if int(workload["input_tokens"]) + int(workload["output_tokens"]) > max_len:
            raise ConfigError(f"workload {name} exceeds max_model_len")
    supported = {"decode_heavy", "balanced", "prefill_heavy", "long_prefill", "heterogeneous", "sharegpt", "burstgpt_length"}
    scenarios = config["g1"]["scenarios"]
    if not scenarios or len(scenarios) != len(set(scenarios)):
        raise ConfigError("g1.scenarios must be a non-empty unique list")
    if unknown := set(scenarios) - supported:
        raise ConfigError(f"unsupported G1 scenarios: {sorted(unknown)}")
    if config["g1"].get("slo_source", "low_load") not in {
        "serial",
        "low_load",
    }:
        raise ConfigError("g1.slo_source must be serial or low_load")
    start_attempts = config["g1"].get("low_load_start_attempt", {})
    if not isinstance(start_attempts, dict) or set(start_attempts) - set(scenarios):
        raise ConfigError(
            "g1.low_load_start_attempt must map only configured G1 scenarios"
        )
    if any(not 0 <= int(value) < 4 for value in start_attempts.values()):
        raise ConfigError("G1 low-load start attempts must be in [0, 3]")

    g2 = config["g2"]
    if g2.get("reference_policy") != "stock_auto":
        raise ConfigError("G2 reference_policy must be stock_auto")
    if g2.get("slo_tier") not in {"tight", "medium", "loose"}:
        raise ConfigError("g2.slo_tier must be tight, medium, or loose")
    if int(g2.get("measurement_requests", stats["measurement_requests"])) <= 0:
        raise ConfigError("g2.measurement_requests must be positive")
    g2_seeds = g2.get("seeds", seeds)
    if (
        not isinstance(g2_seeds, list)
        or not g2_seeds
        or len(g2_seeds) != len(set(g2_seeds))
        or not set(g2_seeds).issubset(set(seeds))
    ):
        raise ConfigError(
            "g2.seeds must be a non-empty unique subset of statistics.seeds"
        )
    conditions = g2.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ConfigError("g2.conditions must be a non-empty list")
    condition_keys: set[tuple[str, str]] = set()
    allowed_arrivals = {"poisson", "gamma_medium", "gamma_strong"}
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ConfigError("every G2 condition must be a mapping")
        scenario = str(condition.get("scenario", ""))
        arrival = str(condition.get("arrival", ""))
        if scenario not in supported:
            raise ConfigError(f"unsupported G2 scenario: {scenario}")
        if arrival not in allowed_arrivals:
            raise ConfigError(f"unsupported G2 arrival: {arrival}")
        if (scenario, arrival) in condition_keys:
            raise ConfigError(f"duplicate G2 condition: {scenario}:{arrival}")
        condition_keys.add((scenario, arrival))
        burstiness = float(condition.get("burstiness", 0.0))
        if burstiness <= 0:
            raise ConfigError("G2 condition burstiness must be positive")
        if scenario not in scenarios:
            raise ConfigError(
                f"G2 scenario {scenario} requires a G1 saturation baseline"
            )

    validity = config["validity"]
    if int(validity["max_run_attempts"]) < 1:
        raise ConfigError("validity.max_run_attempts must be at least one")
    drift_limit = float(validity["serial_tpot_drift_warning_relative"])
    if not 0 <= drift_limit < 1:
        raise ConfigError(
            "validity.serial_tpot_drift_warning_relative must be in [0, 1)"
        )
    compatibility = config.get("compatibility", {})
    if not isinstance(compatibility, dict):
        raise ConfigError("compatibility must be a mapping")
    for key in (
        "trace_manifest_config_sha256",
        "g1_baseline_config_sha256",
    ):
        hashes = compatibility.get(key, [])
        if not isinstance(hashes, list) or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in hashes
        ):
            raise ConfigError(f"compatibility.{key} must be a list of SHA256 values")


def workspace_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config["paths"][key])
    if value.is_absolute():
        return value
    return Path(config["paths"]["workspace"]) / value


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        public_config(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def compatible_config_hashes(config: dict[str, Any], key: str) -> set[str]:
    return set(config.get("compatibility", {}).get(key, []))


def load_slo_config(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["paths"]["workspace"]) / "configs/slo.yaml"
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ConfigError(f"invalid SLO file: {path}")
    return value


def slo_thresholds_fingerprint(slo: dict[str, Any]) -> str:
    thresholds = slo.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ConfigError("SLO thresholds are not frozen")
    payload = json.dumps(
        {
            "config_sha256": slo.get("config_sha256"),
            "thresholds": thresholds,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
