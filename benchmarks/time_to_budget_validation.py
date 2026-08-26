#!/usr/bin/env python3
"""Offline validation of Prefill-budget to predicted-duration inversion.

This experiment-only tool reconstructs immutable project snapshots from exact
target-profile rows.  It does not import or mutate the online Candidate
Generator, Selector, or Predictor implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from dpp_scheduler.candidate_generator import project_kv_blocks, project_sequence_count
from dpp_scheduler.contracts import BatchPlan, DecodeRequest, PrefillRequest, StateSnapshot
from dpp_scheduler.predictor import RidgeDurationPredictor


BUDGETS = (0, 64, 128, 256, 384, 512, 768, 1024, 1536)
TARGETS_MS = (200, 225, 250, 275, 300, 350, 400)
SWEEP_FIELDS = (
    "predictor_version",
    "snapshot_category",
    "prediction_model",
    "decode_count",
    "frame_id",
    "source_run_id",
    "source_iteration_index",
    "snapshot_hash",
    "budget_label",
    "prefill_budget",
    "actual_prefill_tokens",
    "num_prefill_requests",
    "num_decode_requests",
    "predicted_duration",
    "predicted_duration_ms",
    "in_support",
    "prediction_mode",
    "ood_distance",
    "current_p25_budget",
)
INVERSION_FIELDS = (
    "predictor_version",
    "snapshot_category",
    "decode_count",
    "frame_id",
    "target_duration",
    "target_duration_ms",
    "selected_budget",
    "actual_prefill_tokens",
    "predicted_duration",
    "predicted_duration_ms",
    "slack",
    "slack_ms",
    "found",
    "selected_below_current_p25",
    "selected_in_support",
)
ACTUAL_FIELDS = (
    "predictor_version",
    "snapshot_category",
    "prediction_model",
    "decode_count",
    "frame_id",
    "validation_case",
    "prefill_budget",
    "actual_prefill_tokens",
    "num_prefill_requests",
    "num_decode_requests",
    "repeat",
    "predicted_duration",
    "actual_duration",
    "prediction_error",
    "source_run_id",
    "source_iteration_index",
)


def _read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object row: {path}:{line_number}")
                value["_source_path"] = str(path)
                rows.append(value)
    return rows


def _snapshot_category(row: dict[str, Any]) -> str | None:
    if row.get("sample_role") != "target":
        return None
    selected = row.get("selected_requests")
    if not isinstance(selected, list):
        return None
    prefills = [item for item in selected if item.get("phase") == "prefill"]
    decodes = [item for item in selected if item.get("phase") == "decode"]
    prefill_tokens = sum(int(item["scheduled_tokens"]) for item in prefills)
    if not prefills or prefill_tokens < 256:
        return None
    if not decodes:
        return "prefill_only"
    count = len(decodes)
    if 1 <= count <= 4:
        return "mixed_decode_1_4"
    if 5 <= count <= 16:
        return "mixed_decode_5_16"
    if 17 <= count <= 64:
        return "mixed_decode_17_64"
    return None


def _category_from_decode_count(decode_count: int) -> str:
    if decode_count == 0:
        return "prefill_only"
    if decode_count <= 4:
        return "mixed_decode_1_4"
    if decode_count <= 16:
        return "mixed_decode_5_16"
    if decode_count <= 64:
        return "mixed_decode_17_64"
    raise ValueError("Decode count is outside Predictor coverage")


def _stratum(row: dict[str, Any]) -> tuple[int, int, str, str]:
    selected = row["selected_requests"]
    prefills = [item for item in selected if item["phase"] == "prefill"]
    decodes = [item for item in selected if item["phase"] == "decode"]
    contexts = [int(item["current_context_tokens"]) for item in decodes]
    median_context = statistics.median(contexts) if contexts else 0
    context_band = (
        "none" if not contexts else "short" if median_context < 1024 else "long"
    )
    prefill_state = "partial" if any(
        int(item["current_context_tokens"]) > 0 for item in prefills
    ) else "fresh"
    return len(prefills), len(decodes), context_band, prefill_state


def _choose_rows(
    rows: list[dict[str, Any]], count_per_category: int
) -> list[tuple[str, dict[str, Any]]]:
    groups: dict[tuple[str, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = _snapshot_category(row)
        if category is not None:
            groups[(category, *_stratum(row))].append(row)
    if not groups:
        raise ValueError("no eligible Mixed target-profile rows with Prefill work")
    for values in groups.values():
        values.sort(
            key=lambda row: (
                -sum(
                    int(item["scheduled_tokens"])
                    for item in row["selected_requests"]
                    if item["phase"] == "prefill"
                ),
                str(row.get("run_id", "")),
                int(row.get("iteration_index", -1)),
            )
        )
    chosen: list[tuple[str, dict[str, Any]]] = []
    categories = (
        "prefill_only",
        "mixed_decode_1_4",
        "mixed_decode_5_16",
        "mixed_decode_17_64",
    )
    for category in categories:
        keys = sorted(key for key in groups if key[0] == category)
        if not keys:
            raise ValueError(f"no representative rows for Predictor category {category}")
        category_rows: list[dict[str, Any]] = []
        offset = 0
        while len(category_rows) < count_per_category:
            progressed = False
            for key in keys:
                values = groups[key]
                if offset < len(values):
                    category_rows.append(values[offset])
                    progressed = True
                    if len(category_rows) == count_per_category:
                        break
            if not progressed:
                break
            offset += 1
        if len(category_rows) < count_per_category:
            raise ValueError(
                f"insufficient rows for {category}: "
                f"requested={count_per_category}, available={len(category_rows)}"
            )
        chosen.extend((category, row) for row in category_rows)
    return chosen


def _snapshot(row: dict[str, Any], frame_id: int) -> StateSnapshot:
    prefills: list[PrefillRequest] = []
    decodes: list[DecodeRequest] = []
    for ordinal, item in enumerate(row["selected_requests"]):
        context = int(item["current_context_tokens"])
        scheduled = int(item["scheduled_tokens"])
        request_id = f"f{frame_id:04d}-{item['request_id']}"
        if item["phase"] == "prefill":
            prefills.append(
                PrefillRequest(
                    request_id=request_id,
                    arrival_time=float(ordinal),
                    token_count=context + scheduled,
                    prefilled_tokens=context,
                    is_running=context > 0,
                    ordinal=ordinal,
                )
            )
        elif item["phase"] == "decode":
            if scheduled != 1:
                raise ValueError("Decode target scheduled more than one token")
            decodes.append(
                DecodeRequest(
                    request_id=request_id,
                    arrival_time=float(ordinal),
                    kv_context_length=context,
                    ordinal=ordinal,
                )
            )
        else:
            raise ValueError("unknown selected request phase")
    return StateSnapshot.create(
        frame_id=frame_id,
        timestamp=float(frame_id),
        waiting_prefill_requests=tuple(prefills),
        active_decode_requests=tuple(decodes),
        active_ttft_obligations=(),
        active_tbt_obligations=(),
        recovery_requests=(),
        free_kv_blocks=30149,
        kv_block_size=16,
        token_budget=2048,
        sequence_budget=64,
        total_kv_blocks=30149,
        provenance=(
            "time_to_budget_validation:"
            f"{row.get('run_id')}:{row.get('iteration_index')}"
        ),
    )


def _plan(snapshot: StateSnapshot, budget: int, label: str) -> BatchPlan:
    remaining = budget
    items: list[tuple[str, int]] = []
    for request in snapshot.waiting_prefill_requests:
        grant = min(remaining, request.remaining_tokens)
        if grant > 0:
            items.append((request.request_id, grant))
            remaining -= grant
        if remaining == 0:
            break
    prefill_items = tuple(items)
    decode_items = tuple(item.request_id for item in snapshot.active_decode_requests)
    actual = sum(tokens for _, tokens in prefill_items)
    return BatchPlan(
        plan_id=f"time-budget-f{snapshot.frame_id:04d}-{label.lower()}-{budget}",
        snapshot_hash=snapshot.snapshot_hash,
        template_id=f"TIME_BUDGET_SWEEP:{label}:requested_{budget}",
        prefill_items=prefill_items,
        decode_items=decode_items,
        total_prefill_tokens=actual,
        total_decode_tokens=len(decode_items),
        total_sequences=project_sequence_count(snapshot, prefill_items),
        projected_kv_blocks=project_kv_blocks(snapshot, prefill_items, decode_items),
        mandatory_request_ids=(),
    )


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _build_actual_rows(
    path: Path, predictors: list[RidgeDurationPredictor]
) -> list[dict[str, Any]]:
    source = _read_jsonl((path,))
    rows: list[dict[str, Any]] = []
    for ordinal, item in enumerate(source, start=1):
        if item.get("sample_role") != "target":
            continue
        requested = item.get("requested_shape")
        if not isinstance(requested, dict):
            raise ValueError("actual validation row has no requested_shape")
        budget = int(requested["prefill_token_cap"])
        decode_count = int(requested["decode_request_cap"])
        category = _category_from_decode_count(decode_count)
        snapshot = _snapshot(item, 100_000 + ordinal)
        plan = _plan(snapshot, budget, "ACTUAL")
        actual = float(item["actual_duration_seconds"])
        if not math.isfinite(actual) or actual <= 0:
            raise ValueError("actual validation duration is invalid")
        for predictor in predictors:
            prediction = predictor.predict(snapshot, (plan,))[0]
            if prediction.expected_duration is None:
                raise ValueError("actual validation prediction is invalid")
            rows.append(
                {
                    "predictor_version": predictor.predictor_version,
                    "snapshot_category": category,
                    "prediction_model": "decode_only" if budget == 0 else category,
                    "decode_count": decode_count,
                    "frame_id": requested.get("source_family_index"),
                    "validation_case": "GRID",
                    "prefill_budget": budget,
                    "actual_prefill_tokens": plan.total_prefill_tokens,
                    "num_prefill_requests": len(plan.prefill_items),
                    "num_decode_requests": len(plan.decode_items),
                    "repeat": requested.get("repeat_index"),
                    "predicted_duration": prediction.expected_duration,
                    "actual_duration": actual,
                    "prediction_error": prediction.expected_duration - actual,
                    "source_run_id": item.get("run_id"),
                    "source_iteration_index": item.get("iteration_index"),
                }
            )
    grouped: dict[tuple[str, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["predictor_version"], row["frame_id"], row["repeat"])].append(row)
    for values in grouped.values():
        feasible = [
            row for row in values if float(row["predicted_duration"]) <= 0.250
        ]
        if feasible:
            selected = max(feasible, key=lambda row: int(row["prefill_budget"]))
            selected["validation_case"] = "INVERTED_250MS"
    return rows


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _percent(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{100.0 * numerator / denominator:.1f}%"


def _report(
    path: Path,
    sweep: list[dict[str, Any]],
    inversion: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> None:
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in sweep:
        by_frame[(str(row["predictor_version"]), int(row["frame_id"]))].append(row)

    group_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in by_frame.items():
        version = key[0]
        category = str(rows[0]["snapshot_category"])
        stats = group_stats.setdefault(
            (version, category),
            {
                "frames": 0,
                "comparisons": 0,
                "within_comparisons": 0,
                "violations": 0,
                "within_violations": 0,
                "boundary_violations": 0,
                "maximum_reverse_ms": 0.0,
            },
        )
        stats["frames"] += 1
        valid = sorted(
            (row for row in rows if row["predicted_duration"] != ""),
            key=lambda row: int(row["actual_prefill_tokens"]),
        )
        deduped = {int(row["actual_prefill_tokens"]): row for row in valid}
        ordered = [deduped[value] for value in sorted(deduped)]
        for left, right in zip(ordered, ordered[1:]):
            stats["comparisons"] += 1
            same_model = left["prediction_model"] == right["prediction_model"]
            stats["within_comparisons"] += int(same_model)
            reverse_ms = 1000.0 * (
                float(left["predicted_duration"]) - float(right["predicted_duration"])
            )
            if reverse_ms > 1e-9:
                stats["violations"] += 1
                stats["within_violations"] += int(same_model)
                stats["boundary_violations"] += int(not same_model)
                stats["maximum_reverse_ms"] = max(
                    float(stats["maximum_reverse_ms"]), reverse_ms
                )

    inversion_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in inversion:
        inversion_by_group[(str(row["predictor_version"]), str(row["snapshot_category"]))].append(row)
    for group, rows in inversion_by_group.items():
        stats = group_stats[group]
        found = [row for row in rows if row["found"]]
        stats["found"] = len(found)
        stats["targets"] = len(rows)
        stats["selected_supported"] = sum(bool(row["selected_in_support"]) for row in found)
        stats["order_violations"] = 0
        frame_ids = sorted({int(row["frame_id"]) for row in rows})
        for frame_id in frame_ids:
            selected = sorted(
                (
                    row for row in rows
                    if int(row["frame_id"]) == frame_id and row["found"]
                ),
                key=lambda row: float(row["target_duration"]),
            )
            stats["order_violations"] += sum(
                int(right["selected_budget"]) < int(left["selected_budget"])
                for left, right in zip(selected, selected[1:])
            )
        key_rows = [
            row for row in found if float(row["target_duration_ms"]) == 250.0
        ]
        stats["budgets_250"] = sorted({int(row["selected_budget"]) for row in key_rows})
        stats["below_p25_250"] = sum(
            bool(row["selected_below_current_p25"]) for row in key_rows
        )
        stats["rows_250"] = len(key_rows)

    actual_pairs = [
        row for row in actual
        if row.get("actual_duration") not in (None, "")
        and row.get("predicted_duration") not in (None, "")
    ]
    actual_mae_ms = (
        1000.0 * statistics.mean(
            abs(float(row["actual_duration"]) - float(row["predicted_duration"]))
            for row in actual_pairs
        )
        if actual_pairs else None
    )
    actual_250 = [row for row in actual_pairs if row["validation_case"] == "INVERTED_250MS"]
    actual_250_median_ms = (
        1000.0 * statistics.median(float(row["actual_duration"]) for row in actual_250)
        if actual_250 else None
    )
    actual_comparisons = actual_violations = 0
    actual_maximum_reverse_ms = 0.0
    actual_groups: dict[tuple[str, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in actual_pairs:
        actual_groups[(row["predictor_version"], row["frame_id"], row["repeat"])].append(row)
    for rows in actual_groups.values():
        ordered = sorted(rows, key=lambda row: int(row["actual_prefill_tokens"]))
        for left, right in zip(ordered, ordered[1:]):
            actual_comparisons += 1
            reverse_ms = 1000.0 * (
                float(left["actual_duration"]) - float(right["actual_duration"])
            )
            if reverse_ms > 1e-9:
                actual_violations += 1
                actual_maximum_reverse_ms = max(actual_maximum_reverse_ms, reverse_ms)
    actual_250_median_absolute_target_error_ms = (
        statistics.median(
            abs(1000.0 * float(row["actual_duration"]) - 250.0)
            for row in actual_250
        )
        if actual_250 else None
    )

    all_stable = all(
        stats["within_violations"] == 0 and stats.get("order_violations", 0) == 0
        for stats in group_stats.values()
    )
    actual_basic_monotonic = (
        actual_comparisons > 0
        and actual_violations <= max(1, math.ceil(0.05 * actual_comparisons))
    )
    actual_target_near = (
        actual_250_median_absolute_target_error_ms is not None
        and actual_250_median_absolute_target_error_ms <= 50.0
    )
    conclusion = "PENDING_REAL_GPU_VALIDATION" if not actual_pairs else (
        "PASS"
        if all_stable and actual_basic_monotonic and actual_target_near
        and actual_mae_ms is not None and actual_mae_ms <= 50.0
        else "PARTIAL PASS"
        if all_stable and actual_violations <= max(1, math.ceil(0.20 * actual_comparisons))
        else "FAIL"
    )

    summary_rows: list[str] = []
    for (version, category), stats in sorted(group_stats.items()):
        summary_rows.append(
            f"| {version} | {category} | {stats['frames']} | "
            f"{stats['within_violations']}/{stats['within_comparisons']} | "
            f"{stats['boundary_violations']} | {stats['maximum_reverse_ms']:.3f} | "
            f"{stats.get('found', 0)}/{stats.get('targets', 0)} | "
            f"{stats.get('order_violations', 0)} | "
            f"{','.join(map(str, stats.get('budgets_250', []))) or '无'} | "
            f"{_percent(stats.get('selected_supported', 0), stats.get('found', 0))} |"
        )

    curve_rows: list[str] = []
    curve_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in sweep:
        if row["predicted_duration"] != "":
            curve_groups[
                (
                    str(row["predictor_version"]),
                    str(row["snapshot_category"]),
                    int(row["actual_prefill_tokens"]),
                )
            ].append(1000.0 * float(row["predicted_duration"]))
    for (version, category, budget), values in sorted(curve_groups.items()):
        curve_rows.append(
            f"| {version} | {category} | {budget} | {len(values)} | "
            f"{statistics.median(values):.1f} | {min(values):.1f} | {max(values):.1f} |"
        )

    text = f"""# Time → Prefill Budget 可行性验证

本报告只使用实验工具重建 BatchPlan 并调用冻结 Predictor；没有修改线上 Candidate Generator、DPP Selector 或 Predictor。

## 结论摘要

- Predictor artifact 数：{len({str(row['predictor_version']) for row in sweep})}；每个 artifact 分别覆盖 Prefill-only 与三个 Mixed Decode 子模型。Decode-only 只作为 Mixed Snapshot 的 `b=0` 锚点。
- Snapshot-artifact 组合数：{len(by_frame)}。每个 Snapshot 固定全部 Decode 请求和 Prefill 顺序，只改变 Prefill budget。
- 子模型内部 budget → time 是否稳定：{'是' if all_stable else '否'}。跨 `b=0` 的 Decode-only → Mixed 边界违反单独列出，不与子模型内部违反混合。
- time → budget 使用离散扫描，未实现二分搜索；各模型覆盖率和 250 ms budget 见下表。
- 真实运行：{len(actual_pairs)} 行；预测 MAE {_fmt(actual_mae_ms)} ms；反求 250 ms case 的真实时间中位数 {_fmt(actual_250_median_ms)} ms。
- 真实 budget → time：{actual_violations}/{actual_comparisons} 次反转；最大反向差 {actual_maximum_reverse_ms:.3f} ms；250 ms case 的真实绝对目标误差中位数 {_fmt(actual_250_median_absolute_target_error_ms)} ms。
- **最终结论：{conclusion}**

## 分模型结果

| Predictor | Snapshot 类别 | Snapshot 数 | 子模型内部违反/比较 | 跨模型边界违反 | 最大反向 ms | 可反求 target | budget 反向下降 | 250 ms budgets | 反求点 in-support |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
{chr(10).join(summary_rows)}

## Budget → predicted duration 曲线（分模型汇总）

| Predictor | Snapshot 类别 | actual Prefill tokens | Snapshot 数 | 中位预测 ms | 最小 ms | 最大 ms |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(curve_rows)}

这里的 `MAX` 是从原始 exact target-profile 行可重建的 Prefill work 上限；超过该上限的点忽略。`predicted_duration` 使用 expected duration。Prefill-only 没有可执行的空 BatchPlan，因此不生成 `b=0`；Mixed 的 `b=0` 由 Decode-only 模型预测，正 budget 由对应 Decode 分段 Mixed 模型预测。

## 回答

1. budget → time 是否基本单调：{'各子模型内部均单调' if all_stable else '至少一个子模型内部存在反转'}；`b=0` 的跨模型边界单独报告。
2. time → budget 是否可稳定反求：{'是' if all_stable else '存在不稳定模型'}；离散求 `max{{b: tau_hat(b) <= T}}`，不假设跨模型全局单调。
3. 250 ms 附近可得到的 budget：按 Predictor 和 Decode 分段列于分模型结果表。
4. 真实运行是否支持 Predictor：{'尚待小规模 GPU 验证' if not actual_pairs else f'实际 MAE 为 {_fmt(actual_mae_ms)} ms，真实单调反转 {actual_violations}/{actual_comparisons}，250 ms case 中位数为 {_fmt(actual_250_median_ms)} ms'}。
5. 最终结论：**{conclusion}**。只有填入真实 GPU 验证结果后才会给出 PASS / PARTIAL PASS / FAIL。

## 方法与限制

代表性状态从现有 Qwen3-14B DGX exact targeted-profile 原始行按 Prefill-only、Mixed Decode 1–4、5–16、17–64 分层抽取。重建保留全部 Decode 请求及其 pre-iteration KV context、Prefill 顺序和已 Prefill context。active-config segmented v2 与 development cross-feature v3 分开报告；v3 不被描述为线上已采用。历史在线残差窗口不可重放，因此 sweep 使用各 artifact 的同 batch-kind OOF cold-start 校准。
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", type=Path, required=True)
    parser.add_argument("--predictor", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshots-per-model", type=int, default=20)
    parser.add_argument("--actual-csv", type=Path)
    parser.add_argument("--actual-profile", type=Path)
    args = parser.parse_args()
    if args.snapshots_per_model <= 0:
        raise ValueError("snapshots per model must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    source_rows = _read_jsonl(args.profile)
    selected_rows = _choose_rows(source_rows, args.snapshots_per_model)
    predictors = [
        RidgeDurationPredictor.from_artifact(path, ood_uncertainty_coefficient=0.0)
        for path in args.predictor
    ]
    if len({predictor.predictor_version for predictor in predictors}) != len(predictors):
        raise ValueError("duplicate Predictor version")

    sweep: list[dict[str, Any]] = []
    for predictor in predictors:
        for frame_id, (category, source) in enumerate(selected_rows, start=1):
            snapshot = _snapshot(source, frame_id)
            decode_count = len(snapshot.active_decode_requests)
            maximum = min(
                sum(item.remaining_tokens for item in snapshot.waiting_prefill_requests),
                snapshot.token_budget - decode_count,
            )
            p25 = math.floor(maximum * 0.25)
            points = [
                ("GRID", budget) for budget in BUDGETS
                if budget <= maximum and (budget > 0 or decode_count > 0)
            ]
            if maximum not in {budget for _, budget in points}:
                points.append(("MAX", maximum))
            else:
                points = [
                    ("MAX" if budget == maximum else label, budget)
                    for label, budget in points
                ]
            for label, budget in sorted(points, key=lambda item: item[1]):
                plan = _plan(snapshot, budget, label)
                prediction = predictor.predict(snapshot, (plan,))[0]
                duration = prediction.expected_duration
                prediction_model = (
                    "decode_only" if budget == 0 else category
                )
                sweep.append(
                    {
                        "predictor_version": predictor.predictor_version,
                        "snapshot_category": category,
                        "prediction_model": prediction_model,
                        "decode_count": decode_count,
                        "frame_id": frame_id,
                        "source_run_id": source.get("run_id"),
                        "source_iteration_index": source.get("iteration_index"),
                        "snapshot_hash": snapshot.snapshot_hash,
                        "budget_label": label,
                        "prefill_budget": budget,
                        "actual_prefill_tokens": plan.total_prefill_tokens,
                        "num_prefill_requests": len(plan.prefill_items),
                        "num_decode_requests": len(plan.decode_items),
                        "predicted_duration": "" if duration is None else duration,
                        "predicted_duration_ms": "" if duration is None else 1000.0 * duration,
                        "in_support": prediction.in_support,
                        "prediction_mode": prediction.prediction_mode,
                        "ood_distance": prediction.ood_distance,
                        "current_p25_budget": p25,
                    }
                )

    inversion: list[dict[str, Any]] = []
    for predictor in predictors:
      for frame_id in range(1, len(selected_rows) + 1):
        rows = [
            row for row in sweep
            if row["frame_id"] == frame_id
            and row["predictor_version"] == predictor.predictor_version
        ]
        feasible_prediction_rows = [row for row in rows if row["predicted_duration"] != ""]
        for target_ms in TARGETS_MS:
            target = target_ms / 1000.0
            feasible = [
                row for row in feasible_prediction_rows
                if float(row["predicted_duration"]) <= target
            ]
            chosen = max(feasible, key=lambda row: int(row["prefill_budget"])) if feasible else None
            inversion.append(
                {
                    "predictor_version": predictor.predictor_version,
                    "snapshot_category": rows[0]["snapshot_category"],
                    "decode_count": rows[0]["decode_count"],
                    "frame_id": frame_id,
                    "target_duration": target,
                    "target_duration_ms": target_ms,
                    "selected_budget": "" if chosen is None else chosen["prefill_budget"],
                    "actual_prefill_tokens": "" if chosen is None else chosen["actual_prefill_tokens"],
                    "predicted_duration": "" if chosen is None else chosen["predicted_duration"],
                    "predicted_duration_ms": "" if chosen is None else chosen["predicted_duration_ms"],
                    "slack": "" if chosen is None else target - float(chosen["predicted_duration"]),
                    "slack_ms": "" if chosen is None else target_ms - float(chosen["predicted_duration_ms"]),
                    "found": chosen is not None,
                    "selected_below_current_p25": (
                        False if chosen is None else int(chosen["prefill_budget"]) < int(chosen["current_p25_budget"])
                    ),
                    "selected_in_support": False if chosen is None else chosen["in_support"],
                }
            )

    if args.actual_csv is not None and args.actual_profile is not None:
        raise ValueError("choose at most one actual validation input")
    actual: list[dict[str, Any]] = []
    if args.actual_profile is not None:
        actual = _build_actual_rows(args.actual_profile, predictors)
    elif args.actual_csv is not None:
        with args.actual_csv.open("r", encoding="utf-8", newline="") as stream:
            actual = list(csv.DictReader(stream))
    _write_csv(output / "predictor_sweep.csv", SWEEP_FIELDS, sweep)
    _write_csv(output / "inversion_results.csv", INVERSION_FIELDS, inversion)
    _write_csv(output / "actual_validation.csv", ACTUAL_FIELDS, actual)
    _report(output / "report.md", sweep, inversion, actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
