#!/usr/bin/env python3
"""Stage deterministic, length-blind Qwen3-14B Poisson traces for review.

The trace contains prompts, genuine exponential inter-arrival samples, sampling
seeds, and the reviewed client safety ceiling. It never contains a target,
expected, remaining, or eventual output length. Stage under ``results/raw``;
after review, promote the immutable files into ``traces/qwen3_14b``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Sequence

from benchmarks.qwen3_runtime import load_active_runtime, resolve_under, sha256_file


def parse_qps_seed(value: str) -> tuple[float, int]:
    """Parse one explicit ``QPS:SEED`` pair without creating a cross product."""
    try:
        qps_text, seed_text = value.split(":", maxsplit=1)
        qps = float(qps_text)
        seed = int(seed_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"invalid QPS:SEED pair {value!r}"
        ) from error
    if not math.isfinite(qps) or qps <= 0:
        raise argparse.ArgumentTypeError("QPS must be finite and positive")
    return qps, seed


def resolve_trace_pairs(
    *,
    qps_values: Sequence[float] | None,
    seeds: Sequence[int] | None,
    explicit_pairs: Sequence[tuple[float, int]] | None,
) -> tuple[tuple[float, int], ...]:
    """Resolve either the legacy cross product or explicit QPS/seed pairs."""
    if explicit_pairs:
        if qps_values or seeds:
            raise ValueError("--qps-seed cannot be combined with --qps/--seeds")
        pairs = tuple((float(qps), int(seed)) for qps, seed in explicit_pairs)
    else:
        if not qps_values or not seeds:
            raise ValueError("provide either --qps-seed or both --qps and --seeds")
        pairs = tuple(
            (float(qps), int(seed)) for qps in qps_values for seed in seeds
        )

    if len(set(pairs)) != len(pairs):
        raise ValueError("QPS/seed pairs must be unique")
    if any(not math.isfinite(qps) or qps <= 0 for qps, _ in pairs):
        raise ValueError("all qps values must be finite and positive")
    if len({seed for _, seed in pairs}) != len(pairs):
        raise ValueError("trace seeds must be unique")
    return pairs


def generation_seed(trace_seed: int, index: int) -> int:
    payload = f"qwen3-poisson-{trace_seed}-{index}-generation".encode("utf-8")
    # vLLM's OpenAI protocol accepts signed int64. Keep deterministic seeds in
    # the non-negative half of that domain.
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) & ((1 << 63) - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Unique staging directory relative to the active raw-results root.",
    )
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--qps", type=float, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--qps-seed",
        type=parse_qps_seed,
        action="append",
        dest="qps_seed_pairs",
        help="Explicit QPS:SEED pair; repeat this option for multiple traces.",
    )
    args = parser.parse_args()

    runtime = load_active_runtime(args.config)
    if args.num_requests <= 0:
        raise ValueError("num_requests must be positive")
    pairs = resolve_trace_pairs(
        qps_values=args.qps,
        seeds=args.seeds,
        explicit_pairs=args.qps_seed_pairs,
    )

    request_pool_path = runtime.request_pool
    pool_manifest_path = runtime.request_pool_manifest
    with request_pool_path.open("r", encoding="utf-8") as stream:
        pool = [json.loads(line) for line in stream if line.strip()]
    with pool_manifest_path.open("r", encoding="utf-8") as stream:
        pool_manifest = json.load(stream)
    pool_hash = sha256_file(request_pool_path)
    if pool_manifest.get("request_pool_sha256") != pool_hash:
        raise ValueError("request pool SHA256 does not match its manifest")
    if pool_manifest.get("model_revision") != runtime.model_revision:
        raise ValueError("request pool model revision does not match active config")
    if pool_manifest.get("tokenizer_revision") != runtime.tokenizer_revision:
        raise ValueError("request pool tokenizer revision does not match active config")
    expected_pool_facts = {
        "dataset_sha256": runtime.source_dataset_sha256,
        "enable_thinking": False,
        "min_input_tokens": runtime.min_input_tokens,
        "max_input_tokens": runtime.max_input_tokens,
        "max_pool_size": runtime.pool_size,
        "seed": runtime.pool_seed,
        "selected_count": runtime.pool_size,
    }
    for key, expected in expected_pool_facts.items():
        if pool_manifest.get(key) != expected:
            raise ValueError(
                f"request pool {key} mismatch: expected {expected!r}, "
                f"got {pool_manifest.get(key)!r}"
            )
    if args.num_requests > len(pool):
        raise ValueError(
            f"requested {args.num_requests} prompts from a pool of {len(pool)}"
        )

    output_dir = resolve_under(
        runtime.raw_results, args.output_dir, label="trace staging directory"
    )
    if output_dir.exists():
        raise FileExistsError(f"append-only staging directory exists: {output_dir}")
    output_dir.mkdir(parents=True)

    files: list[dict] = []
    for qps, seed in pairs:
        rng = random.Random(seed)
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        selected = indices[: args.num_requests]

        # Do not rescale these samples. Rescaling to an exact finite-sample
        # rate would make the arrival process conditional rather than a
        # genuine sequence of independent exponential inter-arrivals.
        arrivals = [0.0]
        for _ in range(args.num_requests - 1):
            arrivals.append(arrivals[-1] + rng.expovariate(qps))

        rows = []
        for index, pool_index in enumerate(selected):
            record = pool[pool_index]
            rows.append(
                {
                    "request_id": f"poisson_q{qps}_s{seed}_{index:04d}",
                    "prompt_id": record["prompt_id"],
                    "prompt": record["prompt"],
                    "input_tokens": int(record["input_tokens"]),
                    "arrival_time_s": round(arrivals[index], 6),
                    "generation_seed": generation_seed(seed, index),
                    "temperature": runtime.temperature,
                    "top_p": runtime.top_p,
                    "ignore_eos": runtime.ignore_eos,
                    "client_safety_ceiling_tokens": (
                        runtime.client_safety_ceiling_tokens
                    ),
                }
            )

        filename = f"qps_{qps}_seed_{seed}.jsonl"
        path = output_dir / filename
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        arrival_span = arrivals[-1] if len(arrivals) > 1 else 0.0
        realized_qps = (
            (len(arrivals) - 1) / arrival_span if arrival_span > 0 else None
        )
        files.append(
            {
                "file": filename,
                "requested_qps": qps,
                "realized_qps": realized_qps,
                "seed": seed,
                "num_requests": len(rows),
                "arrival_span_s": arrival_span,
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "schema_version": 2,
        "kind": "qwen3_14b_poisson_length_blind_traces",
        "status": "staged_for_review",
        "config_sha256": runtime.config_sha256,
        "model_revision": runtime.model_revision,
        "tokenizer_revision": runtime.tokenizer_revision,
        "request_pool": str(request_pool_path),
        "request_pool_manifest": str(pool_manifest_path),
        "request_pool_sha256": pool_hash,
        "arrival_process": "independent_exponential_interarrivals",
        "pairing": "explicit" if args.qps_seed_pairs else "cross_product",
        "num_requests_per_trace": args.num_requests,
        "client_safety_ceiling_tokens": runtime.client_safety_ceiling_tokens,
        "client_safety_ceiling_role": (
            "termination_guard_only_never_scheduler_input"
        ),
        "predetermined_output_length": False,
        "temperature": runtime.temperature,
        "top_p": runtime.top_p,
        "seed_source": runtime.seed_source,
        "ignore_eos": runtime.ignore_eos,
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
