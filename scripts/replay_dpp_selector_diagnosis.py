#!/usr/bin/env python3
"""Replay a two-stage DPP Selector diagnosis JSONL artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpp_scheduler.selector_diagnosis import replay_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnosis", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = replay_file(args.diagnosis)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    mismatches = sum(
        value for key, value in summary.items() if key.endswith("_mismatch")
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
