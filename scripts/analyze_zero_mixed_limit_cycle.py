#!/usr/bin/env python3
"""Analyze the DPP ZERO-to-Mixed limit cycle from an existing diagnostic log.

This is an offline, standard-library-only analysis. It joins each Scheduler
diagnostic record to its actual-duration feedback by frame ID, writes the full
per-frame series and cycle statistics, and renders full-run and cycle-detail
SVG plots without running a model or importing project runtime modules.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable


DIAGNOSTIC_MARKER = "ModularDPPScheduler diagnostic="
FEEDBACK_RE = re.compile(
    r"ModularDPPScheduler feedback frame=(?P<frame>\d+).*?"
    r"actual_duration_seconds=(?P<duration>[0-9.eE+-]+)"
)


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[max(0, index)]


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def parse_log(path: Path) -> tuple[list[dict[str, Any]], dict[int, float]]:
    diagnostics: list[dict[str, Any]] = []
    feedback: dict[int, float] = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if DIAGNOSTIC_MARKER in line:
                diagnostics.append(
                    ast.literal_eval(line.split(DIAGNOSTIC_MARKER, 1)[1].strip())
                )
            match = FEEDBACK_RE.search(line)
            if match:
                frame = int(match.group("frame"))
                if frame in feedback:
                    raise ValueError(f"duplicate feedback for frame {frame}")
                feedback[frame] = float(match.group("duration"))
    return diagnostics, feedback


def selected_prediction(frame: dict[str, Any]) -> float | None:
    selected_plan = frame["selected_plan"]
    matches = [
        item
        for item in frame["candidate_scores"]
        if item["plan_id"] == selected_plan or item.get("selected") is True
    ]
    unique = {item["plan_id"]: item for item in matches}
    if len(unique) > 1:
        raise ValueError(
            f"ambiguous selected prediction in frame {frame['frame_id']}: "
            f"{sorted(unique)}"
        )
    if not unique:
        return None
    return float(next(iter(unique.values()))["expected_duration_seconds"])


def classify(frame: dict[str, Any]) -> str:
    prefill_waiting = int(frame["current_prefill_count"])
    decode_active = int(frame["current_decode_count"])
    budget = int(frame["selected_prefill_tokens"])
    if prefill_waiting > 0 and decode_active > 0 and budget == 0:
        return "ZERO"
    if prefill_waiting > 0 and decode_active > 0 and budget > 0:
        return "MIXED"
    if budget > 0 and decode_active == 0:
        return "PREFILL_ONLY"
    if prefill_waiting == 0 and decode_active > 0:
        return "DECODE_ONLY"
    return "OTHER"


def build_rows(
    diagnostics: list[dict[str, Any]], feedback: dict[int, float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    elapsed = 0.0
    seen: set[int] = set()
    for frame in diagnostics:
        frame_id = int(frame["frame_id"])
        if frame_id in seen:
            raise ValueError(f"duplicate diagnostic frame {frame_id}")
        seen.add(frame_id)
        if frame_id not in feedback:
            raise ValueError(f"missing actual-duration feedback for frame {frame_id}")
        actual = feedback[frame_id]
        predicted = selected_prediction(frame)
        row = {
            "frame_id": frame_id,
            "elapsed_start_seconds": elapsed,
            "sum_tbt_debt": float(frame["sum_tbt_debt"]),
            "sum_ttft_debt": float(frame["sum_ttft_debt"]),
            "selected_budget": int(frame["selected_prefill_tokens"]),
            "predicted_duration_seconds": predicted,
            "actual_duration_seconds": actual,
            "prediction_error_seconds": (
                actual - predicted if predicted is not None else None
            ),
            "action": classify(frame),
            "selected_template": str(frame["selected_template"]),
            "current_prefill_count": int(frame["current_prefill_count"]),
            "current_decode_count": int(frame["current_decode_count"]),
        }
        rows.append(row)
        elapsed += actual
    extra_feedback = sorted(set(feedback) - seen)
    if extra_feedback:
        raise ValueError(f"feedback without diagnostics: {extra_feedback[:10]}")
    if [row["frame_id"] for row in rows] != sorted(
        row["frame_id"] for row in rows
    ):
        raise ValueError("diagnostic frames are not monotonically ordered")
    return rows


def find_cycles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cycles: list[dict[str, Any]] = []
    index = 0
    while index + 2 < len(rows):
        if rows[index]["action"] != "MIXED" or rows[index + 1]["action"] != "ZERO":
            index += 1
            continue
        end = index + 1
        while end < len(rows) and rows[end]["action"] == "ZERO":
            end += 1
        if end < len(rows) and rows[end]["action"] == "MIXED":
            first_zero = rows[index + 1]
            start = rows[index]
            next_mixed = rows[end]
            cycles.append(
                {
                    "mixed_start_frame": start["frame_id"],
                    "first_zero_frame": first_zero["frame_id"],
                    "next_mixed_frame": next_mixed["frame_id"],
                    "zero_frame_count": end - index - 1,
                    "tbt_jump_after_mixed": (
                        first_zero["sum_tbt_debt"] - start["sum_tbt_debt"]
                    ),
                    "ttft_rise_during_zero": (
                        next_mixed["sum_ttft_debt"]
                        - first_zero["sum_ttft_debt"]
                    ),
                    "tbt_change_during_zero": (
                        next_mixed["sum_tbt_debt"]
                        - first_zero["sum_tbt_debt"]
                    ),
                    "mixed_predicted_duration_seconds": start[
                        "predicted_duration_seconds"
                    ],
                    "mixed_actual_duration_seconds": start[
                        "actual_duration_seconds"
                    ],
                    "mixed_prediction_error_seconds": start[
                        "prediction_error_seconds"
                    ],
                }
            )
        index = max(index + 1, end)
    return cycles


def svg_plot(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
    frame_min: int | None = None,
    frame_max: int | None = None,
) -> None:
    visible = [
        row
        for row in rows
        if (frame_min is None or row["frame_id"] >= frame_min)
        and (frame_max is None or row["frame_id"] <= frame_max)
    ]
    if not visible:
        raise ValueError("SVG frame window is empty")
    width, height = 1800, 1500
    left, right, top, bottom = 155, 55, 105, 80
    gap = 34
    panel_height = (height - top - bottom - 4 * gap) / 5
    plot_width = width - left - right
    x0 = visible[0]["frame_id"]
    x1 = visible[-1]["frame_id"]

    def x(value: float) -> float:
        return left + (value - x0) / max(1, x1 - x0) * plot_width

    panels: list[tuple[str, list[tuple[str, str, Callable[[dict[str, Any]], float | None]]]]] = [
        ("sum_tbt_debt", [("TBT debt", "#b2182b", lambda r: r["sum_tbt_debt"])]),
        ("sum_ttft_debt", [("TTFT debt", "#2166ac", lambda r: r["sum_ttft_debt"])]),
        ("action", []),
        ("selected Prefill budget (tokens)", [("budget", "#6a3d9a", lambda r: r["selected_budget"])]),
        (
            "iteration duration (seconds)",
            [
                ("predicted", "#1b9e77", lambda r: r["predicted_duration_seconds"]),
                ("actual", "#d95f02", lambda r: r["actual_duration_seconds"]),
            ],
        ),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="42" text-anchor="middle" font-family="sans-serif" font-size="25" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="{width/2}" y="72" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#444">Every diagnostic frame; predicted = selected candidate expected_duration</text>',
    ]
    action_colors = {
        "DECODE_ONLY": "#bdbdbd",
        "ZERO": "#377eb8",
        "MIXED": "#e41a1c",
        "PREFILL_ONLY": "#4daf4a",
        "OTHER": "#984ea3",
    }
    for panel_index, (label, series) in enumerate(panels):
        panel_top = top + panel_index * (panel_height + gap)
        panel_bottom = panel_top + panel_height
        parts.append(
            f'<rect x="{left}" y="{panel_top:.1f}" width="{plot_width}" height="{panel_height:.1f}" fill="#fafafa" stroke="#bbb"/>'
        )
        parts.append(
            f'<text transform="translate(28 {(panel_top + panel_bottom)/2:.1f}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="16">{html.escape(label)}</text>'
        )
        if label == "action":
            bar_width = max(1.0, plot_width / max(1, len(visible)))
            for row in visible:
                parts.append(
                    f'<rect x="{x(row["frame_id"]):.2f}" y="{panel_top + 15:.2f}" width="{bar_width:.2f}" height="{panel_height - 30:.2f}" fill="{action_colors[row["action"]]}"/>'
                )
            legend_x = left + 10
            for action in ("DECODE_ONLY", "ZERO", "MIXED", "PREFILL_ONLY", "OTHER"):
                parts.append(
                    f'<rect x="{legend_x}" y="{panel_top + 8:.1f}" width="13" height="13" fill="{action_colors[action]}"/>'
                    f'<text x="{legend_x + 18}" y="{panel_top + 20:.1f}" font-family="sans-serif" font-size="12">{action}</text>'
                )
                legend_x += 150
            continue
        values = [
            value
            for _, _, getter in series
            for row in visible
            if (value := getter(row)) is not None and math.isfinite(value)
        ]
        ymin = min(0.0, min(values))
        ymax = max(values)
        if math.isclose(ymin, ymax):
            ymax = ymin + 1.0

        def y(value: float) -> float:
            return panel_bottom - (value - ymin) / (ymax - ymin) * panel_height

        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            yy = y(value)
            parts.append(
                f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_width}" y2="{yy:.2f}" stroke="#ddd"/>'
                f'<text x="{left - 10}" y="{yy + 5:.2f}" text-anchor="end" font-family="monospace" font-size="12">{value:.3g}</text>'
            )
        for name, color, getter in series:
            chunks: list[list[tuple[float, float]]] = [[]]
            for row in visible:
                value = getter(row)
                if value is None or not math.isfinite(value):
                    if chunks[-1]:
                        chunks.append([])
                    continue
                chunks[-1].append((x(row["frame_id"]), y(value)))
            for chunk in chunks:
                if len(chunk) >= 2:
                    points = " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in chunk)
                    parts.append(
                        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>'
                    )
            parts.append(
                f'<line x1="{left + 12}" y1="{panel_top + 17}" x2="{left + 42}" y2="{panel_top + 17}" stroke="{color}" stroke-width="3"/>'
                f'<text x="{left + 48}" y="{panel_top + 22}" font-family="sans-serif" font-size="13">{name}</text>'
            )
    axis_y = height - bottom + 28
    for tick in range(6):
        frame = x0 + (x1 - x0) * tick / 5
        xx = x(frame)
        parts.append(
            f'<line x1="{xx:.2f}" y1="{height-bottom}" x2="{xx:.2f}" y2="{height-bottom+6}" stroke="#333"/>'
            f'<text x="{xx:.2f}" y="{axis_y}" text-anchor="middle" font-family="monospace" font-size="13">{frame:.0f}</text>'
        )
    parts.append(
        f'<text x="{left + plot_width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="16">frame_id</text></svg>'
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("startup_log", type=Path)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()

    diagnostics, feedback = parse_log(args.startup_log)
    if not diagnostics:
        raise SystemExit("no diagnostic frames found")
    rows = build_rows(diagnostics, feedback)
    cycles = find_cycles(rows)
    args.processed_dir.mkdir(parents=True, exist_ok=False)
    args.artifact_dir.mkdir(parents=True, exist_ok=False)

    csv_path = args.processed_dir / "frame_timeseries.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tbt_jumps = [cycle["tbt_jump_after_mixed"] for cycle in cycles]
    ttft_rises = [cycle["ttft_rise_during_zero"] for cycle in cycles]
    tbt_zero_changes = [cycle["tbt_change_during_zero"] for cycle in cycles]
    zero_counts = [float(cycle["zero_frame_count"]) for cycle in cycles]
    mixed_rows = [row for row in rows if row["action"] == "MIXED"]
    zero_rows = [row for row in rows if row["action"] == "ZERO"]
    longest = max(cycles, key=lambda item: item["zero_frame_count"]) if cycles else None
    report = {
        "schema_version": 1,
        "analysis": "dpp_zero_to_mixed_limit_cycle",
        "source": {
            "startup_log": str(args.startup_log),
            "startup_log_sha256": hashlib.sha256(
                args.startup_log.read_bytes()
            ).hexdigest(),
        },
        "alignment": {
            "diagnostic_frames": len(diagnostics),
            "feedback_frames": len(feedback),
            "joined_frames": len(rows),
            "frame_id_min": rows[0]["frame_id"],
            "frame_id_max": rows[-1]["frame_id"],
            "selected_prediction_missing_frames": sum(
                row["predicted_duration_seconds"] is None for row in rows
            ),
        },
        "action_counts": dict(Counter(row["action"] for row in rows)),
        "duration_seconds": {
            "mixed_predicted": summary(
                [
                    row["predicted_duration_seconds"]
                    for row in mixed_rows
                    if row["predicted_duration_seconds"] is not None
                ]
            ),
            "mixed_actual": summary(
                [row["actual_duration_seconds"] for row in mixed_rows]
            ),
            "mixed_prediction_error": summary(
                [
                    row["prediction_error_seconds"]
                    for row in mixed_rows
                    if row["prediction_error_seconds"] is not None
                ]
            ),
            "zero_actual": summary(
                [row["actual_duration_seconds"] for row in zero_rows]
            ),
        },
        "complete_mixed_zero_mixed_cycles": {
            "count": len(cycles),
            "with_multiple_zero_frames": sum(
                cycle["zero_frame_count"] >= 2 for cycle in cycles
            ),
            "with_positive_tbt_jump": sum(value > 0 for value in tbt_jumps),
            "with_positive_ttft_rise": sum(value > 0 for value in ttft_rises),
            "matching_full_hypothesis": sum(
                cycle["zero_frame_count"] >= 2
                and cycle["tbt_jump_after_mixed"] > 0
                and cycle["ttft_rise_during_zero"] > 0
                for cycle in cycles
            ),
            "zero_frame_count": summary(zero_counts),
            "tbt_jump_after_mixed": summary(tbt_jumps),
            "ttft_rise_during_zero": summary(ttft_rises),
            "tbt_change_during_zero": summary(tbt_zero_changes),
            "longest_cycle": longest,
            "cycles": cycles,
        },
        "artifacts": {
            "frame_timeseries_csv": str(csv_path),
            "full_timeseries_svg": str(args.artifact_dir / "full_timeseries.svg"),
            "longest_cycle_svg": (
                str(args.artifact_dir / "longest_cycle.svg") if longest else None
            ),
        },
    }
    report_path = args.processed_dir / "limit_cycle_report.json"
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    svg_plot(
        rows,
        args.artifact_dir / "full_timeseries.svg",
        title="DPP QPS=0.25 ZERO-to-Mixed limit-cycle — full run",
    )
    if longest:
        svg_plot(
            rows,
            args.artifact_dir / "longest_cycle.svg",
            title=(
                "DPP QPS=0.25 longest complete Mixed-ZERO-Mixed cycle "
                f"({longest['zero_frame_count']} ZERO frames)"
            ),
            frame_min=max(rows[0]["frame_id"], longest["mixed_start_frame"] - 5),
            frame_max=min(rows[-1]["frame_id"], longest["next_mixed_frame"] + 5),
        )
    print(json.dumps({key: value for key, value in report.items() if key != "complete_mixed_zero_mixed_cycles"}, indent=1))
    print(json.dumps({"cycle_summary": {key: value for key, value in report["complete_mixed_zero_mixed_cycles"].items() if key != "cycles"}}, indent=1))


if __name__ == "__main__":
    main()
