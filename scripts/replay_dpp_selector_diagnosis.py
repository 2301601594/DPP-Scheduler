#!/usr/bin/env python3
"""Replay a two-stage DPP Selector diagnosis JSONL artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dpp_scheduler.selector_diagnosis import counterfactual_replay_file, replay_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnosis", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--counterfactual",
        action="store_true",
        help=(
            "for historical schema 1/2 only, compare Rate and Absolute "
            "TTFT-debt rankings"
        ),
    )
    args = parser.parse_args()
    replay_summary = replay_file(args.diagnosis)
    summary = (
        {
            "source_diagnosis": {
                "path": str(args.diagnosis),
                "sha256": _sha256(args.diagnosis),
            },
            "replay": replay_summary,
            "counterfactual": counterfactual_replay_file(args.diagnosis),
        }
        if args.counterfactual
        else replay_summary
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    mismatches = sum(
        value
        for key, value in replay_summary.items()
        if key.endswith("_mismatch")
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
