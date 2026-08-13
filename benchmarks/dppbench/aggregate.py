from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from benchmarks.dppbench.config import (
    config_hash,
    load_slo_config,
    slo_thresholds_fingerprint,
    workspace_path,
)
from benchmarks.dppbench.io import atomic_write_json, atomic_write_text
from benchmarks.dppbench.metrics import (
    confidence_interval_95,
    request_derived_metrics,
    slo_attainment,
)
from benchmarks.dppbench.results import (
    has_saturation_stream_warning,
    iter_run_records,
    record_matches_config,
    run_record_usable,
)


def _usable(record: dict[str, Any]) -> bool:
    return run_record_usable(record)


def _stage_records(config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    records = [
        record
        for record in iter_run_records(config, stage)
        if record_matches_config(config, stage, record)
    ]
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["metadata"].get("run_key", record["metadata"].get("run_id"))
        previous = by_key.get(key)
        if previous is None or _usable(record) or not _usable(previous):
            by_key[key] = record
    return list(by_key.values())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    fields = sorted({key for row in rows for key in row})
    lines: list[str] = []
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    atomic_write_text(path, buffer.getvalue())


def _mean(records: Iterable[dict[str, Any]], field: str) -> float:
    values = [float(record["result"]["summary"][field]) for record in records]
    return statistics.fmean(values) if values else math.nan


def _relative_spread(values: list[float]) -> float:
    if not values:
        return math.inf
    mean = statistics.fmean(values)
    return max(abs(value - mean) for value in values) / mean if mean else math.inf


def _max_waiting(record: dict[str, Any]) -> float:
    path = Path(record["run_dir"]) / "server_metrics.jsonl"
    maximum = 0.0
    if not path.exists():
        return math.inf
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                sample = json.loads(line)
                maximum = max(
                    maximum,
                    float(sample.get("metrics", {}).get("vllm:num_requests_waiting", 0.0)),
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return maximum


def _save_slo(config: dict[str, Any], slo: dict[str, Any]) -> None:
    path = Path(config["paths"]["workspace"]) / "configs/slo.yaml"
    atomic_write_text(path, yaml.safe_dump(slo, sort_keys=False))


def _load_slo(config: dict[str, Any]) -> dict[str, Any]:
    return load_slo_config(config)


def serial_tpot_drift(record: dict[str, Any]) -> dict[str, float | bool]:
    requests = [
        request
        for request in record["result"]["requests"]
        if request.get("success")
    ]
    if not requests:
        return {
            "first_mean_tpot_ms": math.nan,
            "last_mean_tpot_ms": math.nan,
            "overall_mean_tpot_ms": math.nan,
            "relative_drift": math.inf,
        }
    width = max(1, math.ceil(len(requests) * 0.10))
    values = [request_derived_metrics(request)["tpot_ms"] for request in requests]
    first = statistics.fmean(values[:width])
    last = statistics.fmean(values[-width:])
    overall = statistics.fmean(values)
    relative = abs(last - first) / overall if overall else math.inf
    return {
        "first_mean_tpot_ms": first,
        "last_mean_tpot_ms": last,
        "overall_mean_tpot_ms": overall,
        "relative_drift": relative,
    }


def aggregate_g1(
    config: dict[str, Any], *, allow_missing_low_load: bool = False
) -> dict[str, Any]:
    records = _stage_records(config, "g1")
    expected_seeds = {int(seed) for seed in config["statistics"]["seeds"]}
    scenarios = tuple(str(name) for name in config["g1"]["scenarios"])
    slo_source = str(config["g1"].get("slo_source", "low_load"))
    by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    expected_limits = {
        "serial": int(config["statistics"]["serial_measurement_requests"]),
        "saturation": int(
            config["statistics"]["saturation_measurement_requests"]
        ),
        "low_load": int(config["statistics"]["low_load_measurement_requests"]),
    }
    for record in records:
        spec = record["metadata"].get("run_spec", {})
        mode = spec.get("mode")
        if (
            spec.get("scenario") in scenarios
            and mode in expected_limits
            and int(spec.get("request_limit", -1)) == expected_limits[mode]
        ):
            by_key[(spec["scenario"], spec["mode"], int(spec.get("attempt", 0)))].append(record)

    saturation_rps: dict[str, float] = {}
    serial: dict[str, Any] = {}
    saturation: dict[str, Any] = {}
    selected_low: dict[str, Any] = {}
    serial_drift: dict[str, dict[str, Any]] = {}
    quality_warnings: list[dict[str, Any]] = []
    missing: list[str] = []
    variation_gate = True
    drift_limit = float(
        config["validity"]["serial_tpot_drift_warning_relative"]
    )
    for scenario in scenarios:
        serial_records = [r for r in by_key[(scenario, "serial", 0)] if _usable(r)]
        saturation_records = [r for r in by_key[(scenario, "saturation", 0)] if _usable(r)]
        for record in saturation_records:
            if has_saturation_stream_warning(record):
                quality_warnings.append(
                    {
                        "code": "g1_saturation_multi_token_stream_chunks",
                        "scenario": scenario,
                        "seed": int(record["metadata"]["run_spec"]["seed"]),
                        "multi_token_chunks": int(
                            record["result"]["summary"]["multi_token_chunks"]
                        ),
                        "throughput_valid": True,
                        "token_timing_exact": False,
                        "run_id": record["metadata"]["run_id"],
                    }
                )
        serial_drift[scenario] = {}
        for record in serial_records:
            seed = str(record["metadata"]["run_spec"]["seed"])
            drift = serial_tpot_drift(record)
            warned = float(drift["relative_drift"]) > drift_limit
            drift["warning"] = warned
            serial_drift[scenario][seed] = drift
            if warned:
                quality_warnings.append(
                    {
                        "code": "serial_tpot_drift",
                        "scenario": scenario,
                        "seed": int(seed),
                        "relative_drift": drift["relative_drift"],
                        "warning_threshold": drift_limit,
                        "run_id": record["metadata"]["run_id"],
                    }
                )
        serial_seeds = {int(r["metadata"]["run_spec"]["seed"]) for r in serial_records}
        saturation_seeds = {
            int(r["metadata"]["run_spec"]["seed"]) for r in saturation_records
        }
        if serial_seeds != expected_seeds or saturation_seeds != expected_seeds:
            missing.append(f"{scenario}:serial_or_saturation")
            continue
        serial[scenario] = {
            "p90_ttft_ms": _mean(serial_records, "p90_ttft_ms"),
            "p90_tpot_ms": _mean(serial_records, "p90_tpot_ms"),
            "request_throughput_rps": _mean(serial_records, "request_throughput_rps"),
        }
        serial_ttft_spread = _relative_spread(
            [float(r["result"]["summary"]["p90_ttft_ms"]) for r in serial_records]
        )
        serial_tpot_spread = _relative_spread(
            [float(r["result"]["summary"]["p90_tpot_ms"]) for r in serial_records]
        )
        rates = [
            float(record["result"]["summary"]["request_throughput_rps"])
            for record in saturation_records
        ]
        spread = _relative_spread(rates)
        saturation_rps[scenario] = statistics.fmean(rates)
        saturation[scenario] = {
            "request_throughput_rps": saturation_rps[scenario],
            "seed_relative_max_deviation": spread,
        }
        serial[scenario]["ttft_seed_relative_max_deviation"] = serial_ttft_spread
        serial[scenario]["tpot_seed_relative_max_deviation"] = serial_tpot_spread
        variation_gate &= max(spread, serial_ttft_spread, serial_tpot_spread) <= float(
            config["validity"]["seed_variation_target"]
        )

        attempts = sorted(
            {
                attempt
                for (name, mode, attempt) in by_key
                if name == scenario and mode == "low_load"
            }
        )
        for attempt in attempts:
            candidates = [r for r in by_key[(scenario, "low_load", attempt)] if _usable(r)]
            seeds = {int(r["metadata"]["run_spec"]["seed"]) for r in candidates}
            if seeds != expected_seeds:
                continue
            p90_ttft = _mean(candidates, "p90_ttft_ms")
            max_waiting = max(_max_waiting(record) for record in candidates)
            if (
                p90_ttft
                <= serial[scenario]["p90_ttft_ms"]
                * (1 + float(config["validity"]["low_load_ttft_increase_max"]))
                and max_waiting == 0
            ):
                p90_tpot = _mean(candidates, "p90_tpot_ms")
                low_ttft_spread = _relative_spread(
                    [
                        float(r["result"]["summary"]["p90_ttft_ms"])
                        for r in candidates
                    ]
                )
                low_tpot_spread = _relative_spread(
                    [
                        float(r["result"]["summary"]["p90_tpot_ms"])
                        for r in candidates
                    ]
                )
                selected_low[scenario] = {
                    "attempt": attempt,
                    "requests_per_seed": int(
                        config["statistics"]["low_load_measurement_requests"]
                    ),
                    "seeds": len(candidates),
                    "request_rate_rps": statistics.fmean(
                        float(r["metadata"]["run_spec"]["request_rate_rps"])
                        for r in candidates
                    ),
                    "p90_ttft_ms": p90_ttft,
                    "p90_tpot_ms": p90_tpot,
                    "max_waiting_requests": max_waiting,
                    "ttft_seed_relative_max_deviation": low_ttft_spread,
                    "tpot_seed_relative_max_deviation": low_tpot_spread,
                }
                if slo_source == "low_load":
                    variation_gate &= max(low_ttft_spread, low_tpot_spread) <= float(
                        config["validity"]["seed_variation_target"]
                    )
                break

    low_load_gate = len(selected_low) == len(scenarios)
    low_load_gate_required = slo_source == "low_load"
    calibration = serial if slo_source == "serial" else selected_low
    calibration_gate = len(calibration) == len(scenarios)
    gate_passed = (
        not missing
        and variation_gate
        and calibration_gate
        and (low_load_gate if low_load_gate_required else True)
    )
    status = (
        "complete_with_warnings"
        if gate_passed and quality_warnings
        else "complete"
        if gate_passed
        else "incomplete"
    )
    derived = {
        "schema_version": 1,
        "stage": "g1",
        "config_sha256": config_hash(config),
        "valid_runs": sum(_usable(record) for record in records),
        "total_runs": len(records),
        "missing": missing,
        "serial": serial,
        "saturation": saturation,
        "saturation_rps": saturation_rps,
        "low_load": selected_low,
        "slo_source": slo_source,
        "serial_tpot_drift": serial_drift,
        "quality_warnings": quality_warnings,
        "quality_status": status,
        "low_load_gate_passed": low_load_gate,
        "low_load_gate_required": low_load_gate_required,
        "seed_variation_gate_passed": variation_gate,
        "gate_passed": gate_passed,
    }
    output = workspace_path(config, "processed_results") / "g1"
    atomic_write_json(output / "derived.json", derived)
    seed_rows: list[dict[str, Any]] = []
    for record in records:
        if not _usable(record):
            continue
        spec = record["metadata"]["run_spec"]
        row = {
            "run_id": record["metadata"]["run_id"],
            "scenario": spec["scenario"],
            "mode": spec["mode"],
            "attempt": spec.get("attempt", 0),
            "seed": spec["seed"],
            "request_rate_rps": spec.get("request_rate_rps"),
            "throughput_valid": True,
            "token_timing_exact": not has_saturation_stream_warning(record),
        }
        if spec["mode"] == "serial":
            drift = serial_drift.get(spec["scenario"], {}).get(str(spec["seed"]))
            if drift:
                row.update(
                    {
                        "serial_tpot_first_mean_ms": drift[
                            "first_mean_tpot_ms"
                        ],
                        "serial_tpot_last_mean_ms": drift["last_mean_tpot_ms"],
                        "serial_tpot_relative_drift": drift["relative_drift"],
                        "serial_tpot_drift_warning": drift["warning"],
                    }
                )
        row.update(record["result"]["summary"])
        seed_rows.append(row)
    _write_csv(output / "per_seed.csv", seed_rows)
    aggregate_rows = _aggregate_generic_rows(
        seed_rows, ("scenario", "mode", "attempt")
    )
    _write_csv(output / "aggregate.csv", aggregate_rows)

    if gate_passed:
        all_base_names = {
            "decode_heavy": "decode_heavy",
            "balanced": "balanced",
            "prefill_heavy": "prefill_heavy",
            "long_prefill": "long_prefill",
            "sharegpt": "sharegpt",
            "burstgpt": "burstgpt_length",
        }
        base_names = {
            output_name: source_name
            for output_name, source_name in all_base_names.items()
            if source_name in scenarios
        }
        factors = {
            "tight": (2.0, 1.5),
            "medium": (4.0, 2.0),
            "loose": (8.0, 3.0),
        }
        thresholds: dict[str, dict[str, dict[str, float]]] = {}
        for tier, (ttft_factor, tpot_factor) in factors.items():
            thresholds[tier] = {
                output_name: {
                    "ttft_ms": calibration[source_name]["p90_ttft_ms"] * ttft_factor,
                    "tpot_ms": calibration[source_name]["p90_tpot_ms"] * tpot_factor,
                }
                for output_name, source_name in base_names.items()
            }
        previous = _load_slo(config)
        slo = {
            "schema_version": 1,
            "status": "frozen_g1",
            "source": (
                "mean of seed-level Stock-Auto serial P90 values"
                if slo_source == "serial"
                else "mean of seed-level Stock-Auto low-load P90 values"
            ),
            "calibration_sample": {
                "mode": slo_source,
                "requests_per_seed": int(
                    config["statistics"][
                        "serial_measurement_requests"
                        if slo_source == "serial"
                        else "low_load_measurement_requests"
                    ]
                ),
                "seeds": len(expected_seeds),
            },
            "config_sha256": config_hash(config),
            "primary_tier": str(config["g2"]["slo_tier"]),
            "thresholds": thresholds,
            "lambda_cap_rps": {},
            "lambda_cap_by_arrival": {},
        }
        fingerprint = slo_thresholds_fingerprint(slo)
        slo["thresholds_fingerprint"] = fingerprint
        if previous.get("thresholds_fingerprint") == fingerprint:
            slo["lambda_cap_rps"] = previous.get("lambda_cap_rps", {})
            slo["lambda_cap_by_arrival"] = previous.get(
                "lambda_cap_by_arrival", {}
            )
            if previous.get("status") == "frozen_g2":
                slo["status"] = "frozen_g2"
        _save_slo(config, slo)
    elif not allow_missing_low_load:
        derived["gate_passed"] = False
    return derived


def _slo_attainment(record: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, float]:
    return slo_attainment(record["result"]["requests"], thresholds)


def g2_point_outcome(config: dict[str, Any], run_key: str) -> bool | None:
    """Return the SLO outcome for one usable G2 RunSpec, if available."""
    slo = _load_slo(config)
    tier = str(config["g2"]["slo_tier"])
    thresholds = slo.get("thresholds", {}).get(tier)
    if not thresholds:
        raise RuntimeError(f"G2 requires frozen {tier} SLO thresholds")
    expected_fingerprint = slo_thresholds_fingerprint(slo)
    for record in _stage_records(config, "g2"):
        metadata = record["metadata"]
        spec = metadata.get("run_spec", {})
        if (
            metadata.get("run_key") != run_key
            or spec.get("slo_fingerprint") != expected_fingerprint
            or not _usable(record)
        ):
            continue
        return _slo_attainment(record, thresholds)["joint_attainment"] >= float(
            config["statistics"]["joint_attainment_target"]
        )
    return None


def g2_brackets(config: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    slo = _load_slo(config)
    tier = str(config["g2"]["slo_tier"])
    thresholds = slo.get("thresholds", {}).get(tier)
    if not thresholds:
        raise RuntimeError(f"G2 requires frozen {tier} SLO thresholds")
    expected_fingerprint = slo_thresholds_fingerprint(slo)
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    from benchmarks.dppbench.matrix import g2_seeds

    first_seed = g2_seeds(config)[0]
    for record in _stage_records(config, "g2"):
        spec = record["metadata"].get("run_spec", {})
        if (
            int(spec.get("seed", -1)) != first_seed
            or spec.get("request_rate_rps") is None
            or spec.get("slo_fingerprint") != expected_fingerprint
        ):
            continue
        # A hard-invalid run is neither an SLO pass nor an SLO fail. The
        # runner retries it according to the frozen attempt limit; treating it
        # as a capacity failure would create a false bracket.
        if not _usable(record):
            continue
        arrival = str(spec.get("mode", "")).removeprefix("capacity_")
        key = f"{spec.get('scenario')}:{arrival}"
        rate = float(spec["request_rate_rps"])
        passed = _slo_attainment(record, thresholds)["joint_attainment"] >= float(
            config["statistics"]["joint_attainment_target"]
        )
        grouped[key].append((rate, passed))

    output: dict[str, dict[str, float | None]] = {}
    from benchmarks.dppbench.matrix import g2_condition_scenarios

    for scenario, arrival, _ in g2_condition_scenarios(config):
        key = f"{scenario}:{arrival}"
        points = grouped.get(key, [])
        rates = [rate for rate, _ in points]
        pass_rates = [rate for rate, passed in points if passed]
        pass_rate = max(pass_rates) if pass_rates else None
        fail_candidates = [
            rate for rate, passed in points if not passed and (pass_rate is None or rate > pass_rate)
        ]
        output[key] = {
            "min_rate": min(rates) if rates else None,
            "max_rate": max(rates) if rates else None,
            "pass_rate": pass_rate,
            "fail_rate": min(fail_candidates) if fail_candidates else None,
        }
    return output


def select_measured_capacities(
    rows: Iterable[dict[str, Any]],
) -> dict[str, float]:
    capacities: dict[str, float] = {}
    for row in rows:
        if not row.get("capacity_pass"):
            continue
        key = f"{row['scenario']}:{row['arrival']}"
        rate = float(row["request_rate_rps"])
        capacities[key] = max(capacities.get(key, 0.0), rate)
    return capacities


def aggregate_g2(config: dict[str, Any]) -> dict[str, Any]:
    slo = _load_slo(config)
    tier = str(config["g2"]["slo_tier"])
    thresholds = slo.get("thresholds", {}).get(tier)
    if not thresholds:
        raise RuntimeError(f"G2 requires frozen {tier} SLO thresholds")
    if slo.get("config_sha256") != config_hash(config):
        raise RuntimeError("G2 SLO was produced by a different experiment config")
    expected_fingerprint = slo_thresholds_fingerprint(slo)
    if slo.get("thresholds_fingerprint") != expected_fingerprint:
        raise RuntimeError("G2 SLO threshold fingerprint does not match its contents")
    from benchmarks.dppbench.matrix import g2_seeds

    expected_seeds = set(g2_seeds(config))
    groups: dict[tuple[str, str, float], dict[int, dict[str, Any]]] = defaultdict(dict)
    all_records = [
        record
        for record in _stage_records(config, "g2")
        if record["metadata"].get("run_spec", {}).get("slo_fingerprint")
        == expected_fingerprint
    ]
    for record in all_records:
        spec = record["metadata"].get("run_spec", {})
        if spec.get("request_rate_rps") is None:
            continue
        arrival = str(spec.get("mode", "")).removeprefix("capacity_")
        key = (spec["scenario"], arrival, round(float(spec["request_rate_rps"]), 9))
        groups[key][int(spec["seed"])] = record

    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for (scenario, arrival, rate), seed_records in sorted(groups.items()):
        if set(seed_records) != expected_seeds:
            continue
        values = []
        achieved_values = []
        throughput_values = []
        goodput_values = []
        valid = True
        for seed in sorted(expected_seeds):
            record = seed_records[seed]
            valid &= _usable(record)
            attainment = _slo_attainment(record, thresholds) if record.get("result") else {
                "joint_attainment": 0.0
            }
            values.append(float(attainment["joint_attainment"]))
            if record.get("result"):
                result = record["result"]
                achieved = record["result"].get("actual_offered_rate_rps")
                if achieved is not None:
                    achieved_values.append(float(achieved))
                throughput = float(result["summary"]["request_throughput_rps"])
                duration = float(result["summary"]["duration_s"])
                goodput = (
                    float(attainment.get("joint_passed", 0.0)) / duration
                    if duration
                    else 0.0
                )
                throughput_values.append(throughput)
                goodput_values.append(goodput)
                seed_rows.append(
                    {
                        "run_id": record["metadata"]["run_id"],
                        "scheduler": "stock_auto",
                        "scenario": scenario,
                        "arrival": arrival,
                        "slo_tier": tier,
                        "seed": seed,
                        "request_rate_rps": rate,
                        "achieved_offered_rate_rps": achieved,
                        "completed_throughput_rps": throughput,
                        "goodput_rps": goodput,
                        "joint_attainment": attainment["joint_attainment"],
                        "ttft_attainment": attainment.get("ttft_attainment"),
                        "tpot_attainment": attainment.get("tpot_attainment"),
                        "valid": _usable(record),
                    }
                )
        mean, low, high = confidence_interval_95(values)
        achieved_mean, achieved_low, achieved_high = confidence_interval_95(
            achieved_values
        )
        throughput_mean, throughput_low, throughput_high = confidence_interval_95(
            throughput_values
        )
        goodput_mean, goodput_low, goodput_high = confidence_interval_95(
            goodput_values
        )
        passed = valid and mean >= float(config["statistics"]["joint_attainment_target"])
        rows.append(
            {
                "scenario": scenario,
                "arrival": arrival,
                "request_rate_rps": rate,
                "scheduler": "stock_auto",
                "slo_tier": tier,
                "achieved_offered_rate_rps_mean": achieved_mean,
                "achieved_offered_rate_rps_ci95_low": achieved_low,
                "achieved_offered_rate_rps_ci95_high": achieved_high,
                "completed_throughput_rps_mean": throughput_mean,
                "completed_throughput_rps_ci95_low": throughput_low,
                "completed_throughput_rps_ci95_high": throughput_high,
                "goodput_rps_mean": goodput_mean,
                "goodput_rps_ci95_low": goodput_low,
                "goodput_rps_ci95_high": goodput_high,
                "joint_attainment_mean": mean,
                "joint_attainment_ci95_low": low,
                "joint_attainment_ci95_high": high,
                "all_runs_valid": valid,
                "capacity_pass": passed,
                "seeds": len(values),
            }
        )
    capacities_by_arrival = select_measured_capacities(rows)

    required = [
        f"{scenario}:{arrival}"
        for scenario, arrival, _ in __import__(
            "benchmarks.dppbench.matrix", fromlist=["g2_condition_scenarios"]
        ).g2_condition_scenarios(config)
    ]
    lambda_cap = {
        key.split(":", 1)[0]: value
        for key, value in capacities_by_arrival.items()
        if key.endswith(":poisson")
    }
    gate_passed = all(key in capacities_by_arrival for key in required)
    derived = {
        "schema_version": 1,
        "stage": "g2",
        "config_sha256": config_hash(config),
        "slo_tier": tier,
        "slo_fingerprint": expected_fingerprint,
        "total_runs": len(all_records),
        "valid_runs": sum(_usable(record) for record in all_records),
        "seeds": sorted(expected_seeds),
        "replication_status": (
            "exploratory_single_seed"
            if len(expected_seeds) == 1
            else "replicated"
        ),
        "quality_warnings": (
            [
                {
                    "code": "g2_single_seed",
                    "message": (
                        "capacity is exploratory; cross-seed confidence intervals "
                        "cannot be estimated"
                    ),
                }
            ]
            if len(expected_seeds) == 1
            else []
        ),
        "lambda_cap_rps": lambda_cap,
        "lambda_cap_by_arrival": capacities_by_arrival,
        "required_conditions": required,
        "gate_passed": gate_passed,
        "points": rows,
    }
    output = workspace_path(config, "processed_results") / "g2"
    atomic_write_json(output / "derived.json", derived)
    _write_csv(output / "per_seed.csv", seed_rows)
    _write_csv(output / "capacity_points.csv", rows)
    _write_g2_svg(workspace_path(config, "artifacts") / "g2_capacity_knee.svg", rows)
    _write_g2_goodput_svg(
        workspace_path(config, "artifacts") / "g2_goodput.svg", rows
    )
    slo["status"] = "frozen_g2" if gate_passed else "g2_incomplete"
    slo["lambda_cap_rps"] = lambda_cap
    slo["lambda_cap_by_arrival"] = capacities_by_arrival
    _save_slo(config, slo)
    write_stock_reference(config, derived)
    return derived


def select_best_fixed(
    scores: dict[int, float], near_tie_relative: float = 0.01
) -> tuple[int, float]:
    if not scores:
        raise ValueError("no fixed-budget scores")
    best_score = max(scores.values())
    tolerance = abs(best_score) * near_tie_relative
    eligible = [budget for budget, score in scores.items() if best_score - score <= tolerance]
    winner = min(eligible)
    return winner, scores[winner]


def _g3_seed_row(record: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    spec = record["metadata"]["run_spec"]
    summary = record["result"]["summary"]
    attainment = _slo_attainment(record, thresholds)
    duration = float(summary["duration_s"])
    offered = record["result"].get("scheduled_offered_rate_rps")
    if offered is None and spec.get("self_timed"):
        requests = record["result"]["requests"]
        scheduled = [float(row["scheduled_arrival_s"]) for row in requests]
        offered = (len(scheduled) - 1) / (max(scheduled) - min(scheduled)) if len(scheduled) > 1 and max(scheduled) > min(scheduled) else math.nan
    row = {
        "run_id": record["metadata"]["run_id"],
        "scenario": spec["scenario"],
        "arrival": spec["mode"],
        "policy": spec["policy"],
        "budget": record["metadata"].get("resolved_max_num_batched_tokens") or spec.get("budget"),
        "seed": int(spec["seed"]),
        "load_factor": spec.get("load_factor"),
        "valid": _usable(record),
        "offered_rate_rps": offered,
        "achieved_rate_rps": record["result"].get("actual_offered_rate_rps"),
        "request_throughput_rps": summary["request_throughput_rps"],
        "goodput_rps": attainment["joint_passed"] / duration if duration else 0.0,
        "joint_attainment": attainment["joint_attainment"],
    }
    for metric in ("ttft_ms", "tpot_ms", "itl_ms", "max_tbt_ms", "e2e_ms"):
        for percentile in (50, 90, 95, 99):
            field = f"p{percentile}_{metric}"
            if field in summary:
                row[field] = summary[field]
    return row


def _aggregate_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario"],
            row["arrival"],
            row["policy"],
            row["budget"],
            row["load_factor"],
        )
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        result = {
            "scenario": key[0],
            "arrival": key[1],
            "policy": key[2],
            "budget": key[3],
            "load_factor": key[4],
            "seeds": len(values),
            "all_runs_valid": all(row["valid"] for row in values),
        }
        numeric = [
            name
            for name, value in values[0].items()
            if name not in {"seed", "budget", "load_factor"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        for name in numeric:
            data = [float(row[name]) for row in values if row.get(name) is not None and math.isfinite(float(row[name]))]
            mean, low, high = confidence_interval_95(data)
            result[f"{name}_mean"] = mean
            result[f"{name}_ci95_low"] = low
            result[f"{name}_ci95_high"] = high
        output.append(result)
    return output


def _aggregate_generic_rows(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        aggregate = {field: value for field, value in zip(group_fields, key)}
        aggregate["seeds"] = len({row.get("seed") for row in values})
        numeric = sorted(
            {
                field
                for row in values
                for field, value in row.items()
                if field not in {*group_fields, "seed"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
        )
        for field in numeric:
            data = [
                float(row[field])
                for row in values
                if field in row
                and row[field] is not None
                and math.isfinite(float(row[field]))
            ]
            mean, low, high = confidence_interval_95(data)
            aggregate[f"{field}_mean"] = mean
            aggregate[f"{field}_ci95_low"] = low
            aggregate[f"{field}_ci95_high"] = high
        output.append(aggregate)
    return output


def _write_g2_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    points = [row for row in rows if row["arrival"] == "poisson"]
    if not points:
        atomic_write_text(path, '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="120"><text x="20" y="60">No G2 data</text></svg>\n')
        return
    width, height, margin = 820, 460, 65
    max_rate = max(float(row["request_rate_rps"]) for row in points)
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#4b5563")
    scenarios = sorted({row["scenario"] for row in points})
    lines = []
    legend = []
    for index, scenario in enumerate(scenarios):
        series = sorted((row for row in points if row["scenario"] == scenario), key=lambda row: row["request_rate_rps"])
        coords = []
        for row in series:
            x = margin + float(row["request_rate_rps"]) / max_rate * (width - 2 * margin)
            y = height - margin - float(row["joint_attainment_mean"]) * (height - 2 * margin)
            coords.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        lines.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend.append(f'<text x="{width-220}" y="{35+index*18}" fill="{color}">{scenario}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{height-margin-0.9*(height-2*margin)}" x2="{width-margin}" y2="{height-margin-0.9*(height-2*margin)}" stroke="#999" stroke-dasharray="5 5"/>
{"".join(lines)}{"".join(legend)}<text x="{width/2}" y="{height-15}" text-anchor="middle">target arrival rate (requests/s)</text><text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">joint SLO attainment</text><text x="{width/2}" y="25" text-anchor="middle">Stock-Auto Poisson capacity knee (seed mean)</text></svg>\n'''
    atomic_write_text(path, svg)


def _write_g2_goodput_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    points = [row for row in rows if row["arrival"] == "poisson"]
    if not points:
        atomic_write_text(
            path,
            '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="120"><text x="20" y="60">No G2 data</text></svg>\n',
        )
        return
    width, height, margin = 820, 460, 65
    max_rate = max(float(row["request_rate_rps"]) for row in points) or 1.0
    max_goodput = max(float(row["goodput_rps_mean"]) for row in points) or 1.0
    colors = (
        "#2563eb",
        "#dc2626",
        "#059669",
        "#7c3aed",
        "#ea580c",
        "#0891b2",
        "#4b5563",
    )
    scenarios = sorted({row["scenario"] for row in points})
    lines = []
    legend = []
    for index, scenario in enumerate(scenarios):
        series = sorted(
            (row for row in points if row["scenario"] == scenario),
            key=lambda row: row["request_rate_rps"],
        )
        coords = []
        for row in series:
            x = margin + float(row["request_rate_rps"]) / max_rate * (
                width - 2 * margin
            )
            y = height - margin - float(row["goodput_rps_mean"]) / max_goodput * (
                height - 2 * margin
            )
            coords.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        lines.append(
            f'<polyline points="{" ".join(coords)}" fill="none" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        legend.append(
            f'<text x="{width-220}" y="{35+index*18}" fill="{color}">{scenario}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>
{"".join(lines)}{"".join(legend)}<text x="{width/2}" y="{height-15}" text-anchor="middle">target arrival rate (requests/s)</text><text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">joint SLO goodput (requests/s)</text><text x="{width/2}" y="25" text-anchor="middle">Stock-Auto Poisson goodput (seed mean)</text></svg>
'''
    atomic_write_text(path, svg)


def write_stock_reference(
    config: dict[str, Any], g2: dict[str, Any]
) -> dict[str, Any]:
    g1_path = workspace_path(config, "processed_results") / "g1" / "derived.json"
    if not g1_path.exists():
        raise RuntimeError("Stock reference requires processed G1 results")
    g1 = json.loads(g1_path.read_text(encoding="utf-8"))
    slo = _load_slo(config)
    tier = str(config["g2"]["slo_tier"])
    thresholds = slo.get("thresholds", {}).get(tier, {})
    warnings_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for warning in g1.get("quality_warnings", []):
        warnings_by_scenario[str(warning.get("scenario"))].append(warning)
    g2_warnings = list(g2.get("quality_warnings", []))

    point_index = {
        (
            str(row["scenario"]),
            str(row["arrival"]),
            round(float(row["request_rate_rps"]), 9),
        ): row
        for row in g2.get("points", [])
    }
    rows: list[dict[str, Any]] = []
    for scenario, arrival, _ in __import__(
        "benchmarks.dppbench.matrix", fromlist=["g2_condition_scenarios"]
    ).g2_condition_scenarios(config):
        capacity_key = f"{scenario}:{arrival}"
        capacity = g2.get("lambda_cap_by_arrival", {}).get(capacity_key)
        threshold = thresholds.get(scenario)
        if capacity is None or threshold is None:
            continue
        point = point_index.get((scenario, arrival, round(float(capacity), 9)))
        if point is None:
            continue
        warnings = warnings_by_scenario.get(scenario, []) + g2_warnings
        rows.append(
            {
                "scheduler": "stock_auto",
                "scenario": scenario,
                "arrival": arrival,
                "slo_tier": tier,
                "ttft_slo_ms": float(threshold["ttft_ms"]),
                "tpot_slo_ms": float(threshold["tpot_ms"]),
                "lambda_cap_target_rps": float(capacity),
                "achieved_offered_rate_rps_mean": point[
                    "achieved_offered_rate_rps_mean"
                ],
                "achieved_offered_rate_rps_ci95_low": point[
                    "achieved_offered_rate_rps_ci95_low"
                ],
                "achieved_offered_rate_rps_ci95_high": point[
                    "achieved_offered_rate_rps_ci95_high"
                ],
                "completed_throughput_rps_mean": point[
                    "completed_throughput_rps_mean"
                ],
                "completed_throughput_rps_ci95_low": point[
                    "completed_throughput_rps_ci95_low"
                ],
                "completed_throughput_rps_ci95_high": point[
                    "completed_throughput_rps_ci95_high"
                ],
                "goodput_rps_mean": point["goodput_rps_mean"],
                "goodput_rps_ci95_low": point["goodput_rps_ci95_low"],
                "goodput_rps_ci95_high": point["goodput_rps_ci95_high"],
                "joint_attainment_mean": point["joint_attainment_mean"],
                "joint_attainment_ci95_low": point[
                    "joint_attainment_ci95_low"
                ],
                "joint_attainment_ci95_high": point[
                    "joint_attainment_ci95_high"
                ],
                "seeds": point["seeds"],
                "replication_status": g2.get(
                    "replication_status", "replicated"
                ),
                "quality_status": "complete_with_warnings"
                if warnings
                else "complete",
                "quality_warning_count": len(warnings),
            }
        )
    status = (
        "incomplete"
        if not g2.get("gate_passed")
        else "complete_with_warnings"
        if g1.get("quality_warnings") or g2_warnings
        else "complete"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "scheduler": "stock_auto",
        "config_sha256": config_hash(config),
        "slo_fingerprint": slo.get("thresholds_fingerprint"),
        "slo_tier": tier,
        "quality_warnings": g1.get("quality_warnings", []) + g2_warnings,
        "rows": rows,
    }
    output = workspace_path(config, "processed_results") / "g1_g2"
    atomic_write_json(output / "stock_reference.json", payload)
    _write_csv(output / "stock_reference.csv", rows)
    return payload


def _equivalence(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    auto = [
        row for row in rows
        if row["scenario"] == "balanced" and row["arrival"] == "poisson"
        and row["policy"] == "stock_auto" and row["load_factor"] == 0.9
    ]
    explicit = [row for row in rows if row["policy"] == "fixed_b2048_explicit"]
    checks = {}
    for metric in ("request_throughput_rps", "p90_ttft_ms", "p90_tpot_ms"):
        left = statistics.fmean(float(row[metric]) for row in auto) if auto else math.nan
        right = statistics.fmean(float(row[metric]) for row in explicit) if explicit else math.nan
        difference = abs(left - right) / left if left and math.isfinite(left) else math.inf
        checks[metric] = {"auto": left, "explicit": right, "relative_difference": difference, "passed": difference <= tolerance}
    return {
        "passed": len(auto) == len(explicit) == 3 and all(item["passed"] for item in checks.values()),
        "tolerance": tolerance,
        "checks": checks,
    }


def _best_fixed(config: dict[str, Any], rows: list[dict[str, Any]], equivalence: dict[str, Any]) -> dict[str, Any]:
    condition_scores: dict[int, dict[tuple[Any, ...], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_workload: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row["valid"] or not row.get("offered_rate_rps") or not math.isfinite(float(row["offered_rate_rps"])):
            continue
        include = (
            (row["arrival"] == "poisson" and row["load_factor"] in (0.9, 1.1))
            or row["arrival"] in ("gamma_medium", "gamma_strong", "self_timed")
        )
        if not include or row["policy"] == "fixed_b2048_explicit":
            continue
        if row["budget"] is None:
            continue
        budget = int(row["budget"])
        if row["policy"] == "stock_auto" and not equivalence["passed"]:
            continue
        normalized = float(row["goodput_rps"]) / float(row["offered_rate_rps"])
        condition = (row["scenario"], row["arrival"], row["load_factor"])
        condition_scores[budget][condition].append(normalized)
        per_workload[row["scenario"]][budget].append(normalized)
    macro: dict[int, float] = {}
    scenario_macro: dict[int, dict[str, float]] = {}
    for budget, conditions in condition_scores.items():
        by_scenario: dict[str, list[float]] = defaultdict(list)
        for condition, seed_values in conditions.items():
            by_scenario[str(condition[0])].append(statistics.fmean(seed_values))
        scenario_macro[budget] = {
            scenario: statistics.fmean(values)
            for scenario, values in by_scenario.items()
        }
        macro[budget] = statistics.fmean(scenario_macro[budget].values())
    winner, score = select_best_fixed(macro, float(config["best_fixed"]["near_tie_relative"]))
    diagnostic = {}
    for scenario, budget_values in per_workload.items():
        means = {budget: statistics.fmean(values) for budget, values in budget_values.items()}
        diagnostic[scenario] = select_best_fixed(means, float(config["best_fixed"]["near_tie_relative"]))[0]
    return {
        "budget": winner,
        "score": score,
        "scores": macro,
        "scenario_macro_scores": scenario_macro,
        "per_workload_budget": diagnostic,
    }


def _write_g3_svg(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    points = [row for row in aggregate_rows if row["scenario"] == "balanced" and row["arrival"] == "poisson" and row["load_factor"] == 0.9 and row["policy"] != "fixed_b2048_explicit"]
    points.sort(key=lambda row: int(row["budget"]))
    width, height = 760, 420
    margin = 60
    if not points:
        atomic_write_text(path, '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="120"><text x="20" y="60">No valid G3 data</text></svg>\n')
        return
    xs = [math.log2(float(row["budget"])) for row in points]
    ys = [float(row["goodput_rps_mean"]) for row in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymax = ymin + 1
    coords = []
    for x, y in zip(xs, ys):
        px = margin + (x - xmin) / max(xmax - xmin, 1) * (width - 2 * margin)
        py = height - margin - (y - ymin) / (ymax - ymin) * (height - 2 * margin)
        coords.append(f"{px:.1f},{py:.1f}")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>
<polyline points="{' '.join(coords)}" fill="none" stroke="#2563eb" stroke-width="3"/>
<text x="{width/2}" y="{height-15}" text-anchor="middle">max_num_batched_tokens (log2)</text><text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">joint SLO goodput (requests/s)</text><text x="{width/2}" y="28" text-anchor="middle">Balanced, 0.9×lambda_cap, mean across seeds</text>
</svg>\n'''
    atomic_write_text(path, svg)


def _write_g3_heatmap(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    points = [
        row
        for row in aggregate_rows
        if row["arrival"] == "poisson"
        and row["load_factor"] == 0.9
        and row["policy"] != "fixed_b2048_explicit"
    ]
    scenarios = sorted({str(row["scenario"]) for row in points})
    budgets = sorted({int(row["budget"]) for row in points if row["budget"] is not None})
    values = {(str(row["scenario"]), int(row["budget"])): float(row["joint_attainment_mean"]) for row in points if row["budget"] is not None}
    cell_w, cell_h = 105, 44
    left, top = 180, 70
    width = left + cell_w * max(len(budgets), 1) + 30
    height = top + cell_h * max(len(scenarios), 1) + 55
    elements = [f'<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="26" text-anchor="middle">Joint SLO attainment heatmap: Poisson, 0.9×lambda_cap</text>']
    for column, budget in enumerate(budgets):
        elements.append(f'<text x="{left+column*cell_w+cell_w/2}" y="55" text-anchor="middle">B{budget}</text>')
    for row_index, scenario in enumerate(scenarios):
        y = top + row_index * cell_h
        elements.append(f'<text x="{left-10}" y="{y+cell_h*0.65}" text-anchor="end">{scenario}</text>')
        for column, budget in enumerate(budgets):
            value = values.get((scenario, budget), math.nan)
            intensity = 0 if not math.isfinite(value) else max(0, min(255, round(value * 255)))
            color = f"rgb({255-intensity},{120+round(intensity*0.45)},{255-intensity})"
            x = left + column * cell_w
            label = "NA" if not math.isfinite(value) else f"{value:.3f}"
            elements.append(f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{color}" stroke="#ddd"/><text x="{x+cell_w/2}" y="{y+cell_h*0.65}" text-anchor="middle">{label}</text>')
    atomic_write_text(path, f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(elements)}</svg>\n')


def aggregate_g3(config: dict[str, Any]) -> dict[str, Any]:
    slo = _load_slo(config)
    thresholds = slo.get("thresholds", {}).get("medium")
    if not thresholds:
        raise RuntimeError("G3 requires frozen medium SLO thresholds")
    records = _stage_records(config, "g3")
    rows = [_g3_seed_row(record, thresholds) for record in records if record.get("result")]
    aggregate_rows = _aggregate_seed_rows(rows)
    equivalence = _equivalence(rows, float(config["budgets"]["equivalence_tolerance"]))
    best_fixed = _best_fixed(config, rows, equivalence) if equivalence["passed"] else None
    from benchmarks.dppbench.matrix import g3_specs

    capacities = slo.get("lambda_cap_rps", {})
    expected_runs = len(g3_specs(config, capacities)) if capacities else 399
    valid_keys = {record["metadata"].get("run_key") for record in records if _usable(record)}
    gate_passed = len(valid_keys) == expected_runs and equivalence["passed"] and best_fixed is not None
    derived = {
        "schema_version": 1,
        "stage": "g3",
        "config_sha256": config_hash(config),
        "total_runs": len(records),
        "valid_unique_runs": len(valid_keys),
        "expected_runs": expected_runs,
        "equivalence_b2048": equivalence,
        "best_fixed": best_fixed,
        "baseline_aliases": {"Stock-B8192": "fixed_b8192", "Fixed-B8192": "fixed_b8192"},
        "gate_passed": gate_passed,
    }
    output = workspace_path(config, "processed_results") / "g3"
    atomic_write_json(output / "derived.json", derived)
    _write_csv(output / "per_seed.csv", rows)
    _write_csv(output / "aggregate.csv", aggregate_rows)
    artifact = workspace_path(config, "artifacts") / "g3_balanced_goodput.svg"
    _write_g3_svg(artifact, aggregate_rows)
    _write_g3_heatmap(
        workspace_path(config, "artifacts") / "g3_budget_attainment_heatmap.svg",
        aggregate_rows,
    )
    return derived


def aggregate_stage(config: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage == "g1":
        return aggregate_g1(config, allow_missing_low_load=True)
    if stage == "g2":
        return aggregate_g2(config)
    if stage == "g3":
        return aggregate_g3(config)
    raise ValueError(stage)
