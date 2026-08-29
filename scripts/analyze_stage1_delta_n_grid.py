#!/usr/bin/env python3
"""Summarize the Stage-1 delta-N grid campaign into a TTFT/TBT table.

Reads each main run's run_manifest.json, per_request.jsonl,
dpp_diagnostic_aggregate.json, and selector_diagnosis_replay.json from the
campaign directory and renders one row per run (Stock + one row per N) plus
a guardrail-tagged verdict block. This aggregates real executed runs; it is
development_nonformal evidence and not a formal benchmark claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

DIST_KEYS = ("mean", "p50", "p90", "p95", "p99")
DELTA_N_VALUES = (0, 2, 4, 8, 16)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _slo_ms(config_path: Path) -> tuple[float, float]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    slo = config.get("slo") if isinstance(config, dict) else None
    if not isinstance(slo, dict):
        raise ValueError("active config SLO section is missing")
    ttft = float(slo.get("ttft_seconds", 0))
    tbt = float(slo.get("tbt_seconds", 0))
    if ttft <= 0 or tbt <= 0:
        raise ValueError("active config SLO values must be positive")
    return ttft * 1000.0, tbt * 1000.0


def _dist(value: Any, keys: tuple[str, ...] = DIST_KEYS) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("distribution is missing")
    return {key: value.get(key) for key in keys}


def _run_row(
    run_root: Path, *, ttft_slo_ms: float, tbt_slo_ms: float
) -> dict[str, Any]:
    manifest = _read_json(run_root / "run_manifest.json")
    summary = manifest["summary"]
    rows = _read_jsonl(run_root / "per_request.jsonl")
    slo_success = 0
    for row in rows:
        ttft = row.get("ttft_ms")
        itls = row.get("itls_ms")
        success = bool(
            row.get("completed")
            and row.get("finish_reason") == "stop"
            and row.get("token_timing_exact")
            and ttft is not None
            and float(ttft) <= ttft_slo_ms
            and isinstance(itls, list)
            and all(float(value) <= tbt_slo_ms for value in itls)
        )
        slo_success += int(success)
    elapsed = float(summary["elapsed_s"])

    aggregate = None
    aggregate_path = run_root / "dpp_diagnostic_aggregate.json"
    if aggregate_path.is_file():
        aggregate = _read_json(aggregate_path)

    replay = manifest.get("selector_diagnosis_replay")
    diagnosis_mismatches = None
    if isinstance(replay, dict):
        diagnosis_mismatches = sum(
            value for key, value in replay.items() if key.endswith("_mismatch")
        )

    row: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "policy": manifest["scheduler_policy"],
        "request_count": len(rows),
        "completed": int(summary["completed"]),
        "failed": int(summary["failed"]),
        "elapsed_s": elapsed,
        "ttft_ms": _dist(summary["ttft_ms"]),
        "tbt_ms_exact_requests_only": _dist(summary["tbt_ms_exact_requests_only"]),
        "e2e_ms": _dist(summary["e2e_ms"], ("mean", "p50")),
        "completion_throughput_rps": summary["completion_throughput_rps"],
        "natural_eos_slo_success_requests": slo_success,
        "natural_eos_slo_success_rate": slo_success / len(rows),
        "natural_eos_slo_goodput_rps": (
            slo_success / elapsed if elapsed > 0 else None
        ),
        "finish_reason_counts": summary["finish_reason_counts"],
        "selector_diagnosis_valid": manifest.get("selector_diagnosis_valid"),
        "selector_diagnosis_mismatches": diagnosis_mismatches,
    }
    if aggregate is not None:
        row["selection_histogram"] = aggregate.get("selection_histogram")
        row["tie_frame_count"] = aggregate.get("tie_frame_count")
        row["prefill_backlog_frame_count"] = aggregate.get(
            "prefill_backlog_frame_count"
        )
        row["mixed_iteration_count"] = aggregate.get("mixed_iteration_count")
        row["actual_duration_seconds"] = aggregate.get("actual_duration_seconds")
        row["pipeline_call_counts"] = aggregate.get("pipeline_call_counts")
        row["selector_algorithm"] = aggregate.get("selector_algorithm")
        row["maximum_incremental_tbt_violations"] = aggregate.get(
            "maximum_incremental_tbt_violations"
        )
        row["tbt_delta_seconds"] = aggregate.get("tbt_delta_seconds")
    return row


def analyze(campaign_root: Path, config_path: Path) -> dict[str, Any]:
    runs_dir = campaign_root / "runs"
    if not runs_dir.is_dir():
        raise ValueError(f"campaign runs directory missing: {runs_dir}")
    ttft_slo_ms, tbt_slo_ms = _slo_ms(config_path)

    run_rows: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("main_"):
            continue
        run_rows.append(
            _run_row(run_dir, ttft_slo_ms=ttft_slo_ms, tbt_slo_ms=tbt_slo_ms)
        )

    if not run_rows:
        raise ValueError("no main runs found under the campaign root")

    report = {
        "schema_version": 1,
        "kind": "stage1_delta_n_grid_summary",
        "scope": "development_nonformal_single_seed",
        "formal_benchmark_eligible": False,
        "slo_ms": {"ttft": ttft_slo_ms, "tbt": tbt_slo_ms},
        "delta_n_values": list(DELTA_N_VALUES),
        "runs": run_rows,
        "guardrails": [
            "Single-seed development evidence with no cross-seed confidence "
            "interval; it is not a formal benchmark claim.",
            "DPP runs differ only in DPP_STAGE1_MAX_DELTA_N; Stock is the "
            "same-trace native baseline.",
            "Each DPP run's Selector Diagnosis replay must be zero-mismatch "
            "for the run to be included as valid.",
        ],
        "verdict": None,
    }
    return report


def _render_table(report: dict[str, Any]) -> str:
    lines: list[str] = []
    header = (
        f"{'run':>14} | {'TTFT p50':>8} | {'TTFT p95':>8} | {'TTFT p99':>8} | "
        f"{'TBT p50':>7} | {'TBT p95':>7} | {'TBT p99':>7} | "
        f"{'thr rps':>7} | {'SLO-G rps':>8} | {'fail':>4} | {'diag':>4}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in report["runs"]:
        ttft = row["ttft_ms"]
        tbt = row["tbt_ms_exact_requests_only"]
        lines.append(
            f"{row['run_id']:>14} | "
            f"{ttft['p50']:>8.1f} | {ttft['p95']:>8.1f} | {ttft['p99']:>8.1f} | "
            f"{tbt['p50']:>7.1f} | {tbt['p95']:>7.1f} | {tbt['p99']:>7.1f} | "
            f"{row['completion_throughput_rps']:>7.4f} | "
            f"{row['natural_eos_slo_goodput_rps']:>8.4f} | "
            f"{row['failed']:>4} | "
            f"{str(row['selector_diagnosis_valid']):>4}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/dgx_spark_experiment.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = analyze(args.campaign_root, args.config)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(_render_table(report))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
