#!/usr/bin/env python3
"""Post-run analysis for DPP v2.1 T1 development runs (n=20 smoke / n=50 dev).

Combines the runner's summary.json + per_request.jsonl with the scheduler's
run-end dpp_diagnostic_aggregate.json and startup.log correctness scan into
the plan's Phase B section 29 report fields. Local, dependency-free.

Usage:
    python3 scripts/analyze_dpp_v2p1_t1_run.py <run_dir> [baseline_summary.json]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

TTFT_SLO_MS = 2000.0
TBT_SLO_MS = 250.0


def dist(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": ordered[min(len(ordered) - 1, int(0.5 * len(ordered)))],
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "p99": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
        "max": ordered[-1],
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    run_dir = Path(sys.argv[1])
    summary_path = run_dir / "summary.json"
    per_request_path = run_dir / "per_request.jsonl"
    aggregate_path = run_dir / "dpp_diagnostic_aggregate.json"
    startup_log_path = run_dir / "startup.log"

    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in per_request_path.open()]

    completed = [r for r in rows if r["completed"]]
    exact = [r for r in completed if r.get("token_timing_exact")]
    ttft = [float(r["ttft_ms"]) for r in completed if r.get("ttft_ms") is not None]
    itls: list[float] = [float(v) for r in exact for v in r.get("itls_ms", [])]
    violations = [v for v in itls if v > TBT_SLO_MS]
    severe = Counter(
        label
        for v in itls
        for label, flag in (
            ("gt500ms", v > 500.0),
            ("gt800ms", v > 800.0),
            ("gt1s", v > 1000.0),
        )
        if flag
    )
    ttft_ok = [v for v in ttft if v <= TTFT_SLO_MS]
    # Full SLO success: TTFT <= 2s, every TBT interval <= 250ms, natural stop.
    full_slo = [
        r
        for r in completed
        if r.get("ttft_ms") is not None
        and r["ttft_ms"] <= TTFT_SLO_MS
        and all(v <= TBT_SLO_MS for v in r.get("itls_ms", []))
        and r.get("finish_reason") == "stop"
    ]
    natural_stop = [r for r in completed if r.get("finish_reason") == "stop"]

    report = {
        "run_dir": str(run_dir),
        "summary": summary,
        "derived": {
            "tbt_interval": dist(itls),
            "tbt_violation_count": len(violations),
            "tbt_violation_ratio": len(violations) / len(itls) if itls else None,
            "severe_interval_counts": dict(severe),
            "ttft": dist(ttft),
            "ttft_le_2s_count": len(ttft_ok),
            "ttft_violation_count": len(ttft) - len(ttft_ok),
            "full_slo_success_count": len(full_slo),
            "full_slo_success_ratio": len(full_slo) / len(completed) if completed else None,
            "diagnostic_goodput": len(full_slo) / len(rows) if rows else None,
            "natural_stop_count": len(natural_stop),
        },
    }

    if aggregate_path.exists():
        report["dpp_diagnostic_aggregate"] = json.loads(aggregate_path.read_text())
    else:
        report["dpp_diagnostic_aggregate"] = None

    correctness = {
        "tracebacks": 0,
        "engine_dead_error": 0,
        "oom": 0,
        "nan": 0,
        "stall": 0,
        "liveness_escape": 0,
        "watchdog": 0,
    }
    if startup_log_path.exists():
        text = startup_log_path.read_text(encoding="utf-8", errors="replace")
        correctness["tracebacks"] = text.count("Traceback")
        correctness["engine_dead_error"] = text.count("EngineDeadError")
        correctness["oom"] = text.count("CUDA out of memory")
        correctness["nan"] = text.count("NaN")
        correctness["liveness_escape"] = text.count("liveness_escape")
        correctness["watchdog"] = text.count("watchdog_triggered': True")
    report["correctness_scan"] = correctness

    out_path = run_dir / "t1_run_analysis.json"
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {out_path}")

    d = report["derived"]
    print("\n=== TTFT (ms) ===")
    print(json.dumps(d["ttft"], indent=1))
    print(f"TTFT <= 2s: {d['ttft_le_2s_count']}/{len(ttft)}; violations: {d['ttft_violation_count']}")
    print("\n=== TBT interval (ms) ===")
    print(json.dumps(d["tbt_interval"], indent=1))
    print(
        f">250ms intervals: {d['tbt_violation_count']}/{len(itls)} = {d['tbt_violation_ratio']:.2%}; "
        f"severe: {d['severe_interval_counts']}"
    )
    print(
        f"\n=== SLO ===\nfull SLO success: {d['full_slo_success_count']}/{len(completed)} = "
        f"{d['full_slo_success_ratio']:.2%}; diagnostic Goodput "
        f"{d['diagnostic_goodput']:.2%}; natural stop {d['natural_stop_count']}"
    )
    print(f"\n=== correctness scan ===\n{json.dumps(correctness, indent=1)}")
    agg = report["dpp_diagnostic_aggregate"]
    if agg:
        print("\n=== DPP diagnostic aggregate ===")
        print("selection:", agg["selection_histogram"])
        print("tie frames:", agg["tie_frame_count"], "of backlog", agg["prefill_backlog_frame_count"])
        print("tie selected:", agg["tie_selected_histogram"])
        print("mixed/decode_only/prefill_only iterations:", agg["mixed_iteration_count"],
              agg["decode_only_iteration_count"], agg["prefill_only_iteration_count"])
        print("selected prefill progress:", agg["selected_prefill_progress"])
        print("selected prefill tokens:", agg["selected_prefill_tokens"])
        print("actual mixed duration:", agg["actual_duration_seconds"]["mixed"])
        print("actual decode_only duration:", agg["actual_duration_seconds"]["decode_only"])
        print("MAX audit:", agg["max_audit"])
    else:
        print("\n(dpp_diagnostic_aggregate.json missing)")
    if len(sys.argv) >= 3:
        baseline = json.loads(Path(sys.argv[2]).read_text())
        print("\n=== baseline (T0 n=100, direction only) ===")
        print("TTFT:", {k: round(v, 1) if v else v for k, v in baseline["ttft_ms"].items()})
        print("TBT :", {k: round(v, 1) if v else v for k, v in baseline["tbt_ms_exact_requests_only"].items()})


if __name__ == "__main__":
    main()
