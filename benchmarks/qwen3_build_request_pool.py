#!/usr/bin/env python3
"""Build the Stage-1 Qwen3-14B request pool from ShareGPT_V3.

This is intentionally light-weight:
  1. Use the first human turn from each ShareGPT conversation.
  2. Render it through the Qwen3 chat template with enable_thinking=False.
  3. Tokenize with the Qwen3-14B tokenizer.
  4. Filter to [min_input_tokens, max_input_tokens].
  5. Deduplicate by rendered prompt hash.
  6. Deterministically sample a fixed-size pool.
  7. Write request_pool.jsonl and request_pool.meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from benchmarks.qwen3_runtime import load_active_runtime, resolve_under


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_human_message(conversation: dict) -> str | None:
    for message in conversation.get("conversations", []):
        if message.get("from") == "human" and message.get("value"):
            return message["value"].strip()
    return None


def render_prompt(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dgx_spark_experiment.yaml")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Unique staging directory relative to the active raw-results root.",
    )
    args = parser.parse_args()

    runtime = load_active_runtime(args.config)
    dataset_path = runtime.source_dataset.resolve()
    output_dir = resolve_under(
        runtime.raw_results, args.output_dir, label="request-pool staging directory"
    )
    if output_dir.exists():
        raise FileExistsError(f"append-only staging directory exists: {output_dir}")
    output_dir.mkdir(parents=True)

    print(f"Loading dataset: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as stream:
        conversations = json.load(stream)
    source_sha256 = sha256_file(dataset_path)
    if source_sha256 != runtime.source_dataset_sha256:
        raise ValueError("source dataset SHA256 does not match active config")
    print(f"Loaded {len(conversations)} conversations")

    print(f"Loading tokenizer from {runtime.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        runtime.model_path, trust_remote_code=True, local_files_only=True
    )

    records: list[dict] = []
    seen_hashes: set[str] = set()
    skipped_no_human = 0
    skipped_length = 0
    duplicate = 0

    for source_index, conversation in enumerate(conversations):
        user_text = first_human_message(conversation)
        if user_text is None:
            skipped_no_human += 1
            continue

        prompt = render_prompt(tokenizer, user_text)
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tokens = len(input_ids)

        if (
            input_tokens < runtime.min_input_tokens
            or input_tokens > runtime.max_input_tokens
        ):
            skipped_length += 1
            continue

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_hash in seen_hashes:
            duplicate += 1
            continue
        seen_hashes.add(prompt_hash)

        records.append(
            {
                "prompt_id": f"qwen3_sharegpt_{len(records) + 1:06d}",
                "source_id": str(conversation.get("id", source_index)),
                "source_index": source_index,
                "prompt_hash": prompt_hash,
                "prompt": prompt,
                "input_tokens": input_tokens,
            }
        )

    print(
        f"After filter/dedup: {len(records)} (no_human={skipped_no_human}, "
        f"length={skipped_length}, duplicate={duplicate})"
    )

    rng = random.Random(runtime.pool_seed)
    rng.shuffle(records)
    selected = records[: runtime.pool_size]
    if not selected:
        raise ValueError("request-pool filters produced no prompts")
    print(f"Selected pool size: {len(selected)}")

    pool_path = output_dir / "request_pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    token_counts = [record["input_tokens"] for record in selected]
    meta = {
        "schema_version": 1,
        "stage": "stage1_request_pool",
        "dataset": str(dataset_path),
        "dataset_sha256": source_sha256,
        "config_sha256": runtime.config_sha256,
        "model_path": str(runtime.model_path),
        "model_revision": runtime.model_revision,
        "tokenizer_revision": runtime.tokenizer_revision,
        "enable_thinking": False,
        "min_input_tokens": runtime.min_input_tokens,
        "max_input_tokens": runtime.max_input_tokens,
        "max_pool_size": runtime.pool_size,
        "seed": runtime.pool_seed,
        "selected_count": len(selected),
        "input_tokens": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": round(sum(token_counts) / len(token_counts), 2),
        },
        "request_pool_sha256": sha256_file(pool_path),
    }
    meta_path = output_dir / "request_pool.meta.json"
    with meta_path.open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"Wrote {pool_path}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
