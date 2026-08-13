from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import tempfile
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from benchmarks.dppbench.config import (
    compatible_config_hashes,
    config_hash,
    workspace_path,
)
from benchmarks.dppbench.io import (
    atomic_write_json,
    canonical_json,
    read_jsonl,
    sha256_file,
    write_once_or_verify,
)


TRACE_SCHEMA_VERSION = 1
FIXED_WORKLOADS = (
    "decode_heavy",
    "balanced",
    "prefill_heavy",
    "long_prefill",
)


def _stable_source_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def download_sources(config: dict[str, Any]) -> dict[str, Any]:
    raw_dir = workspace_path(config, "raw_data")
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {}
    for name, dataset in config["datasets"].items():
        destination = raw_dir / dataset["filename"]
        metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
        if destination.exists():
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.exists()
                else {}
            )
        else:
            request = urllib.request.Request(
                dataset["url"], headers={"User-Agent": "dpp-vllm-benchmark/0.1"}
            )
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".part", dir=raw_dir
            )
            downloaded = 0
            try:
                with os.fdopen(descriptor, "wb") as output:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        metadata = {
                            "requested_url": dataset["url"],
                            "final_url": _stable_source_url(response.geturl()),
                            "status": response.status,
                            "etag": response.headers.get("ETag"),
                            "last_modified": response.headers.get("Last-Modified"),
                            "content_type": response.headers.get("Content-Type"),
                            "content_length": response.headers.get("Content-Length"),
                        }
                        print(
                            f"downloading {name}: {metadata['final_url']} "
                            f"({metadata['content_length'] or 'unknown'} bytes)",
                            flush=True,
                        )
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (64 * 1024 * 1024) < len(chunk):
                                print(f"  {name}: {downloaded} bytes", flush=True)
                        output.flush()
                        os.fsync(output.fileno())
                os.replace(temporary_name, destination)
            except BaseException:
                Path(temporary_name).unlink(missing_ok=True)
                raise
        metadata.update(
            {
                "path": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
        atomic_write_json(metadata_path, metadata)
        sources[name] = metadata
    return sources


def _stable_prompt_tokens(
    tokenizer: Any, length: int, seed: int, request_index: int
) -> list[int]:
    vocabulary = getattr(tokenizer, "_dpp_benchmark_vocabulary", None)
    if vocabulary is None:
        special_ids = set(tokenizer.all_special_ids)
        vocabulary = sorted(
            token_id
            for token_id in tokenizer.get_vocab().values()
            if token_id not in special_ids
        )
        setattr(tokenizer, "_dpp_benchmark_vocabulary", vocabulary)
    rng = random.Random((seed << 32) ^ request_index ^ (length << 8))
    return [vocabulary[rng.randrange(len(vocabulary))] for _ in range(length)]


def _trace_row(
    request_id: str,
    workload_class: str,
    output_tokens: int,
    *,
    prompt: str | None = None,
    prompt_token_ids: list[int] | None = None,
    arrival_time_s: float | None = None,
) -> dict[str, Any]:
    if (prompt is None) == (prompt_token_ids is None):
        raise ValueError("exactly one of prompt and prompt_token_ids is required")
    row: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "request_id": request_id,
        "workload_class": workload_class,
        "input_tokens": len(prompt_token_ids) if prompt_token_ids is not None else None,
        "output_tokens": int(output_tokens),
        "arrival_time_s": arrival_time_s,
    }
    if prompt_token_ids is not None:
        row["prompt_token_ids"] = prompt_token_ids
    else:
        row["prompt"] = prompt
    return row


def _serialize_rows(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _write_trace(
    path: Path, rows: list[dict[str, Any]], force: bool
) -> dict[str, Any]:
    text = _serialize_rows(rows)
    write_once_or_verify(path, text, force=force)
    classes: dict[str, int] = {}
    for row in rows:
        name = row["workload_class"]
        classes[name] = classes.get(name, 0) + 1
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "requests": len(rows),
        "classes": classes,
        "first_arrival_s": rows[0].get("arrival_time_s") if rows else None,
        "last_arrival_s": rows[-1].get("arrival_time_s") if rows else None,
    }


def _iter_json_array(path: Path) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open(encoding="utf-8") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            eof = not chunk
            buffer += chunk
            position = 0
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    started = True
                    position += 1
                    continue
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                if position >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield value
                position = end
            buffer = buffer[position:]
            if eof:
                if buffer.strip() not in ("", "]"):
                    raise ValueError(f"truncated JSON array in {path}")
                return


def _prepare_sharegpt(
    config: dict[str, Any], tokenizer: Any, source: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    settings = config["datasets"]["sharegpt"]
    count = int(config["statistics"]["split_requests"])
    accepted: list[dict[str, Any]] = []
    rejected = 0
    for entry in _iter_json_array(source):
        conversations = entry.get("conversations", []) if isinstance(entry, dict) else []
        if len(conversations) < 2:
            rejected += 1
            continue
        prompt = conversations[0].get("value")
        completion = conversations[1].get("value")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            rejected += 1
            continue
        input_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        output_tokens = len(tokenizer(completion, add_special_tokens=False).input_ids)
        valid = (
            input_tokens >= int(settings["min_input_tokens"])
            and output_tokens >= int(settings["min_output_tokens"])
            and input_tokens + output_tokens <= int(settings["max_total_tokens"])
        )
        if not valid:
            rejected += 1
            continue
        accepted.append(
            {
                "prompt": prompt,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        if len(accepted) == 2 * count:
            break
    if len(accepted) < 2 * count:
        raise ValueError(f"ShareGPT has only {len(accepted)} valid rows; need {2 * count}")

    def make_rows(split: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for index, record in enumerate(records):
            row = _trace_row(
                f"sharegpt-{split}-{index:05d}",
                "sharegpt",
                record["output_tokens"],
                prompt=record["prompt"],
            )
            row["input_tokens"] = record["input_tokens"]
            rows.append(row)
        return rows

    return (
        make_rows("validation", accepted[:count]),
        make_rows("test", accepted[count : 2 * count]),
        {"accepted": len(accepted), "rejected_before_quota": rejected},
    )


def _normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_column(fieldnames: list[str], candidates: list[str]) -> str:
    normalized = {_normalized_header(name): name for name in fieldnames}
    for candidate in candidates:
        if _normalized_header(candidate) in normalized:
            return normalized[_normalized_header(candidate)]
    raise ValueError(f"none of {candidates} found in CSV columns {fieldnames}")


def _parse_timestamp(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _prepare_burstgpt(
    config: dict[str, Any], tokenizer: Any, source: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    settings = config["datasets"]["burstgpt"]
    count = int(config["statistics"]["split_requests"])
    records: list[tuple[float, int, int]] = []
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("BurstGPT CSV has no header")
        model_column = _find_column(reader.fieldnames, ["Model"])
        input_column = _find_column(
            reader.fieldnames, ["Request tokens", "Prompt tokens", "Input tokens"]
        )
        output_column = _find_column(
            reader.fieldnames,
            ["Response tokens", "Completion tokens", "Output tokens"],
        )
        timestamp_column = _find_column(
            reader.fieldnames, ["Timestamp", "Time", "Request time"]
        )
        for row in reader:
            if row[model_column] != settings["model_filter"]:
                continue
            input_tokens = int(float(row[input_column]))
            output_tokens = int(float(row[output_column]))
            if (
                input_tokens < int(settings["min_input_tokens"])
                or output_tokens < int(settings["min_output_tokens"])
                or input_tokens + output_tokens > int(settings["max_total_tokens"])
            ):
                continue
            records.append(
                (_parse_timestamp(row[timestamp_column]), input_tokens, output_tokens)
            )
    records.sort(key=lambda item: item[0])
    if len(records) < 2 * count:
        raise ValueError(f"BurstGPT has only {len(records)} valid GPT-4 rows")
    window_size = 2 * count
    max_duration = float(settings["max_window_duration_s"])
    window_start_index = next(
        (
            start
            for start in range(len(records) - window_size + 1)
            if records[start + window_size - 1][0] - records[start][0] <= max_duration
        ),
        None,
    )
    if window_start_index is None:
        shortest = min(
            records[start + window_size - 1][0] - records[start][0]
            for start in range(len(records) - window_size + 1)
        )
        raise ValueError(
            f"no continuous {window_size}-row BurstGPT window fits "
            f"{max_duration}s; shortest is {shortest:.3f}s"
        )
    window = records[window_start_index : window_start_index + window_size]
    full_duration = window[-1][0] - window[0][0]

    def make_rows(
        split: str, subset: list[tuple[float, int, int]], seed: int
    ) -> list[dict[str, Any]]:
        origin = subset[0][0]
        rows = []
        for index, (timestamp, input_tokens, output_tokens) in enumerate(subset):
            rows.append(
                _trace_row(
                    f"burstgpt-{split}-{index:05d}",
                    "burstgpt",
                    output_tokens,
                    prompt_token_ids=_stable_prompt_tokens(
                        tokenizer, input_tokens, seed, index
                    ),
                    arrival_time_s=round(timestamp - origin, 9),
                )
            )
        return rows

    validation = make_rows("validation", window[:count], 271828)
    test = make_rows("test", window[count:], 271829)
    metadata = {
        "valid_rows": len(records),
        "window_start": window[0][0],
        "window_end": window[-1][0],
        "window_duration_s": full_duration,
        "window_start_filtered_row": window_start_index,
        "selection": (
            "first_2000_valid_gpt4_row_window_with_duration_at_most_"
            f"{max_duration:g}s"
        ),
    }
    return validation, test, metadata


def prepare_traces(
    config: dict[str, Any], *, download: bool = True, force: bool = False
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    trace_dir = workspace_path(config, "traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    sources = download_sources(config) if download else _existing_sources(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["paths"]["model_snapshot"], local_files_only=True
    )
    split_count = int(config["statistics"]["split_requests"])
    manifest: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "config_sha256": config_hash(config),
        "tokenizer_model": config["model"]["id"],
        "tokenizer_revision": config["model"]["revision"],
        "sources": sources,
        "traces": {},
        "filters": {},
        "phase_shift_rule": {
            "classes": config["workloads"]["phase_shift"]["classes"],
            "phase_duration_s": config["workloads"]["phase_shift"][
                "phase_duration_s"
            ],
            "load_factor": 0.9,
            "arrival": "poisson_normalized_within_each_phase",
            "seed": 314159,
            "status": "pending_g2_capacity",
        },
    }

    for split_index, split in enumerate(("validation", "test")):
        for workload_name in FIXED_WORKLOADS:
            workload = config["workloads"][workload_name]
            rows = [
                _trace_row(
                    f"{workload_name}-{split}-{index:05d}",
                    workload_name,
                    int(workload["output_tokens"]),
                    prompt_token_ids=_stable_prompt_tokens(
                        tokenizer,
                        int(workload["input_tokens"]),
                        1000 + split_index,
                        index,
                    ),
                )
                for index in range(split_count)
            ]
            path = trace_dir / f"{split}_{workload_name}.jsonl"
            manifest["traces"][f"{split}:{workload_name}"] = _write_trace(
                path, rows, force
            )

        classes = config["workloads"]["heterogeneous"]["classes"]
        rows = []
        for index in range(split_count):
            workload_name = classes[index % len(classes)]
            workload = config["workloads"][workload_name]
            rows.append(
                _trace_row(
                    f"heterogeneous-{split}-{index:05d}",
                    workload_name,
                    int(workload["output_tokens"]),
                    prompt_token_ids=_stable_prompt_tokens(
                        tokenizer,
                        int(workload["input_tokens"]),
                        2000 + split_index,
                        index,
                    ),
                )
            )
        path = trace_dir / f"{split}_heterogeneous.jsonl"
        manifest["traces"][f"{split}:heterogeneous"] = _write_trace(
            path, rows, force
        )

    raw_dir = workspace_path(config, "raw_data")
    share_validation, share_test, share_filter = _prepare_sharegpt(
        config,
        tokenizer,
        raw_dir / config["datasets"]["sharegpt"]["filename"],
    )
    burst_validation, burst_test, burst_filter = _prepare_burstgpt(
        config,
        tokenizer,
        raw_dir / config["datasets"]["burstgpt"]["filename"],
    )
    for split, rows in (("validation", share_validation), ("test", share_test)):
        path = trace_dir / f"{split}_sharegpt.jsonl"
        manifest["traces"][f"{split}:sharegpt"] = _write_trace(path, rows, force)
    for split, rows in (("validation", burst_validation), ("test", burst_test)):
        path = trace_dir / f"{split}_burstgpt.jsonl"
        manifest["traces"][f"{split}:burstgpt"] = _write_trace(path, rows, force)
    manifest["filters"] = {"sharegpt": share_filter, "burstgpt": burst_filter}
    atomic_write_json(trace_dir / "manifest.json", manifest)
    return manifest


def _existing_sources(config: dict[str, Any]) -> dict[str, Any]:
    sources = {}
    raw_dir = workspace_path(config, "raw_data")
    for name, settings in config["datasets"].items():
        path = raw_dir / settings["filename"]
        if not path.exists():
            raise FileNotFoundError(path)
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        sources[name] = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {"requested_url": settings["url"]}
        )
        if "final_url" in sources[name]:
            sources[name]["final_url"] = _stable_source_url(sources[name]["final_url"])
        sources[name].update(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sources


def verify_manifest(config: dict[str, Any]) -> list[str]:
    manifest_path = workspace_path(config, "traces") / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    allowed_hashes = compatible_config_hashes(
        config, "trace_manifest_config_sha256"
    ) | {config_hash(config)}
    if manifest["config_sha256"] not in allowed_hashes:
        errors.append("configuration hash differs from trace manifest")
    for name, trace in manifest["traces"].items():
        path = Path(trace["path"])
        if not path.exists():
            errors.append(f"{name}: missing {path}")
        elif sha256_file(path) != trace["sha256"]:
            errors.append(f"{name}: SHA256 mismatch")
        elif sum(1 for _ in read_jsonl(path)) != int(trace["requests"]):
            errors.append(f"{name}: request count mismatch")
    return errors


def _normalized_poisson_times(count: int, duration: float, seed: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    rng = random.Random(seed)
    delays = [rng.expovariate(1.0) for _ in range(count - 1)]
    total = sum(delays)
    times = [0.0]
    elapsed = 0.0
    for delay in delays:
        elapsed += delay
        times.append(duration * elapsed / total)
    return times


def materialize_phase_trace(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    import yaml
    from transformers import AutoTokenizer

    slo_path = Path(config["paths"]["workspace"]) / "configs/slo.yaml"
    with slo_path.open(encoding="utf-8") as stream:
        slo = yaml.safe_load(stream)
    capacities = slo.get("lambda_cap_rps", {})
    classes = config["workloads"]["phase_shift"]["classes"]
    missing = [name for name in classes if name not in capacities]
    if missing:
        raise ValueError(f"cannot materialize phase trace; missing capacities {missing}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["paths"]["model_snapshot"], local_files_only=True
    )
    duration = float(config["workloads"]["phase_shift"]["phase_duration_s"])
    rows: list[dict[str, Any]] = []
    phase_start = 0.0
    request_index = 0
    phase_metadata = []
    for phase_index, workload_name in enumerate(classes):
        rate = 0.9 * float(capacities[workload_name])
        count = max(1, round(rate * duration))
        arrivals = _normalized_poisson_times(count, duration, 314159 + phase_index)
        workload = config["workloads"][workload_name]
        for local_index, arrival in enumerate(arrivals):
            rows.append(
                _trace_row(
                    f"phase-{phase_index}-{local_index:05d}",
                    workload_name,
                    int(workload["output_tokens"]),
                    prompt_token_ids=_stable_prompt_tokens(
                        tokenizer,
                        int(workload["input_tokens"]),
                        314159 + phase_index,
                        request_index,
                    ),
                    arrival_time_s=round(phase_start + arrival, 9),
                )
            )
            request_index += 1
        phase_metadata.append(
            {
                "workload": workload_name,
                "lambda_cap_rps": capacities[workload_name],
                "offered_rate_rps": rate,
                "requests": count,
                "start_s": phase_start,
                "duration_s": duration,
            }
        )
        phase_start += duration
    trace_dir = workspace_path(config, "traces")
    path = trace_dir / "validation_phase_shift.jsonl"
    trace_metadata = _write_trace(path, rows, force)
    manifest_path = trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["traces"]["validation:phase_shift"] = trace_metadata
    manifest["phase_shift_rule"].update(
        {"status": "materialized", "phases": phase_metadata}
    )
    atomic_write_json(manifest_path, manifest)
    return trace_metadata
