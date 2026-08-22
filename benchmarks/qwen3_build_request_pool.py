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
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-path", default="/home/dongj/models/Qwen3-14B-BF16")
    parser.add_argument("--output-dir", default="traces/qwen3_14b")
    parser.add_argument("--min-input-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-pool-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as stream:
        conversations = json.load(stream)
    source_sha256 = sha256_file(dataset_path)
    print(f"Loaded {len(conversations)} conversations")

    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

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

        if input_tokens < args.min_input_tokens or input_tokens > args.max_input_tokens:
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

    rng = random.Random(args.seed)
    rng.shuffle(records)
    selected = records[: args.max_pool_size]
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
        "model_path": args.model_path,
        "enable_thinking": False,
        "min_input_tokens": args.min_input_tokens,
        "max_input_tokens": args.max_input_tokens,
        "max_pool_size": args.max_pool_size,
        "seed": args.seed,
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
