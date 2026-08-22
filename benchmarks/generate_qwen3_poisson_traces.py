#!/usr/bin/env python3
"""Generate Poisson-arrival Qwen3-14B natural-EOS traces from the request pool.

The generator only creates input/arrival/generation-seed information. It does
not fix output lengths.  max_tokens_safety is only a client-side safety cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generation_seed(trace_seed: int, index: int) -> int:
    payload = f"qwen3-poisson-{trace_seed}-{index}-generation".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-pool", required=True)
    parser.add_argument("--output-dir", default="traces/qwen3_14b/poisson")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--qps", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-tokens-safety", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    with Path(args.request_pool).open("r", encoding="utf-8") as stream:
        pool = [json.loads(line) for line in stream if line.strip()]
    print(f"Loaded {len(pool)} prompts from request pool")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    for qps in args.qps:
        for seed in args.seeds:
            rng = random.Random(seed)
            indices = list(range(len(pool)))
            rng.shuffle(indices)
            selected = indices[: args.num_requests]

            # Poisson arrivals (exponential interarrival times), normalized so
            # that the actual offered QPS is exactly the requested QPS.
            arrivals = [0.0]
            for _ in range(args.num_requests - 1):
                arrivals.append(arrivals[-1] + rng.expovariate(qps))
            if len(arrivals) > 1 and arrivals[-1] > 0:
                target_duration = (args.num_requests - 1) / qps
                scale = target_duration / arrivals[-1]
                arrivals = [value * scale for value in arrivals]

            rows = []
            for idx, pool_idx in enumerate(selected):
                record = pool[pool_idx]
                rows.append(
                    {
                        "request_id": f"poisson_q{qps}_s{seed}_{idx:04d}",
                        "prompt_id": record["prompt_id"],
                        "prompt": record["prompt"],
                        "input_tokens": record["input_tokens"],
                        "arrival_time_s": round(arrivals[idx], 6),
                        "generation_seed": generation_seed(seed, idx),
                        "temperature": args.temperature,
                        "ignore_eos": False,
                        "max_tokens_safety": args.max_tokens_safety,
                    }
                )

            filename = f"qps_{qps}_seed_{seed}.jsonl"
            path = output_dir / filename
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

            files.append(
                {
                    "file": filename,
                    "qps": qps,
                    "seed": seed,
                    "num_requests": len(rows),
                    "sha256": sha256_file(path),
                }
            )
            print(f"Wrote {path} ({len(rows)} requests, qps={qps}, seed={seed})")

    manifest = {
        "schema_version": 1,
        "kind": "qwen3_14b_poisson_natural_eos_traces",
        "request_pool": str(Path(args.request_pool).resolve()),
        "request_pool_sha256": sha256_file(Path(args.request_pool)),
        "arrival": "poisson",
        "num_requests_per_trace": args.num_requests,
        "max_tokens_safety": args.max_tokens_safety,
        "temperature": args.temperature,
        "enable_thinking": False,
        "ignore_eos": False,
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
