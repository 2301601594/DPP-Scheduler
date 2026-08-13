#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve locked vLLM EngineArgs without starting a server")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model = config["model"]

    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=config["paths"]["model_snapshot"],
        served_model_name=[model["id"]],
        dtype=model["dtype"],
        max_model_len=int(model["max_model_len"]),
        gpu_memory_utilization=float(model["gpu_memory_utilization"]),
        max_num_seqs=int(model["max_num_seqs"]),
        max_num_batched_tokens=args.budget,
        scheduling_policy=model["scheduling_policy"],
        enable_chunked_prefill=bool(model["enable_chunked_prefill"]),
        enable_prefix_caching=bool(model["enable_prefix_caching"]),
        generation_config=model["generation_config"],
        stream_interval=1,
        seed=int(model["seed"]),
    )
    resolved = engine_args.create_engine_config()
    payload = {
        "engine_args": _jsonable(engine_args),
        "scheduler_config": _jsonable(resolved.scheduler_config),
        "cache_config": _jsonable(resolved.cache_config),
        "model_config": {
            "dtype": str(resolved.model_config.dtype),
            "max_model_len": resolved.model_config.max_model_len,
            "served_model_name": resolved.model_config.served_model_name,
        },
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
