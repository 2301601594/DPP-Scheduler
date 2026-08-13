#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from benchmarks.dppbench.aggregate import (
    aggregate_g1,
    aggregate_g2,
    aggregate_g3,
    g2_brackets,
    g2_point_outcome,
)
from benchmarks.dppbench.config import (
    load_config,
    load_slo_config,
    slo_thresholds_fingerprint,
    workspace_path,
)
from benchmarks.dppbench.matrix import (
    compilation_warmup_specs,
    g1_initial_specs,
    g1_low_load_specs,
    g1_scenarios,
    g2_coarse_specs,
    g2_coarse_specs_for_condition,
    g2_condition_scenarios,
    g2_measurement_requests,
    g2_point_specs,
    g2_seeds,
    g3_specs,
    planned_request_count,
)
from benchmarks.dppbench.results import (
    completed_run_keys,
    load_processed,
    run_attempt_counts,
)
from benchmarks.dppbench.runner import execute_run
from benchmarks.dppbench.traces import materialize_phase_trace, verify_manifest


def run_specs(config, stage, specs, resume):
    completed = completed_run_keys(config, stage) if resume else set()
    persisted_attempts = run_attempt_counts(config, stage) if resume else {}
    for index, spec in enumerate(specs, start=1):
        if spec.run_key in completed:
            print(f"[{index}/{len(specs)}] SKIP {spec.run_key} {spec.scenario} {spec.mode}")
            continue
        max_attempts = int(config["validity"]["max_run_attempts"])
        prior_attempts = persisted_attempts.get(spec.run_key, 0)
        if prior_attempts >= max_attempts:
            raise RuntimeError(
                f"run {spec.run_key} already exhausted {max_attempts} persisted "
                "attempts; --resume will not retry it indefinitely"
            )
        last_error: Exception | None = None
        for run_attempt in range(prior_attempts + 1, max_attempts + 1):
            print(
                f"[{index}/{len(specs)}] RUN  {spec.run_key} {spec.scenario} "
                f"{spec.mode} attempt={run_attempt}/{max_attempts}"
            )
            try:
                run_dir = execute_run(config, spec, run_attempt=run_attempt)
                persisted_attempts[spec.run_key] = run_attempt
                metadata = json.loads(
                    (run_dir / "metadata.json").read_text(encoding="utf-8")
                )
                if metadata.get("validity", {}).get("valid") is True:
                    completed.add(spec.run_key)
                    print(f"  -> {run_dir}")
                    break
                checks = metadata.get("validity", {}).get("checks", {})
                failed_checks = [name for name, passed in checks.items() if not passed]
                last_error = RuntimeError(
                    f"invalid run {run_dir.name}: {failed_checks or ['unknown']}"
                )
                print(f"  !! {last_error}")
            except Exception as error:
                persisted_attempts[spec.run_key] = run_attempt
                last_error = error
                print(f"  !! {type(error).__name__}: {error}")
            if run_attempt == max_attempts:
                assert last_error is not None
                raise RuntimeError(
                    f"run {spec.run_key} failed after {max_attempts} attempts"
                ) from last_error


def _dry_run(config, stage):
    if stage == "g1":
        specs = g1_initial_specs(config)
        if config["g1"].get("slo_source", "low_load") == "serial":
            print("G1 SLO source is serial; no low-load runs are scheduled.")
        else:
            derived = load_processed(config, "g1")
            if derived and derived.get("saturation_rps"):
                for scenario in g1_scenarios(config):
                    start_attempt = int(
                        config["g1"].get("low_load_start_attempt", {}).get(
                            scenario, 0
                        )
                    )
                    specs += g1_low_load_specs(
                        config,
                        derived["saturation_rps"],
                        attempt=start_attempt,
                        scenarios=[scenario],
                    )
            else:
                pending = len(g1_scenarios(config)) * len(config["statistics"]["seeds"])
                print(f"G1 low-load: {pending} additional derived runs pending saturation")
                print(
                    f"G1 frozen total if the first low-load attempt passes: "
                    f"{len(specs) + pending} runs, "
                    f"{planned_request_count(config, specs) + pending * config['statistics']['low_load_measurement_requests']} "
                    "measurement requests"
                )
    elif stage == "g2":
        g1 = load_processed(config, "g1")
        g1_ready = bool(g1 and g1.get("gate_passed"))
        slo = load_slo_config(config) if g1_ready else None
        fingerprint = slo_thresholds_fingerprint(slo) if slo else None
        specs = (
            g2_coarse_specs(config, g1["saturation_rps"], fingerprint)
            if g1_ready and g1 and fingerprint
            else []
        )
        print("G2 fine scan and bracket extensions are adaptive and not included below.")
        conditions = len(g2_condition_scenarios(config))
        coarse = conditions * len(config["arrivals"]["coarse_saturation_factors"])
        fine = (
            conditions
            * int(config["arrivals"]["fine_points"])
            * len(g2_seeds(config))
        )
        extension = conditions * int(
            config["arrivals"]["max_bracket_extensions"]
        )
        requests = g2_measurement_requests(config)
        minimum = conditions * (2 + int(config["arrivals"]["fine_points"]))
        print("G2 coarse order: descending with per-condition early stop.")
        print(
            f"G2 minimum if each second coarse point passes: {minimum} runs / "
            f"{minimum * requests:,} measurement requests."
        )
        print(
            f"G2 no-extension upper bound: {coarse + fine} runs / "
            f"{(coarse + fine) * requests:,} measurement requests."
        )
        print(
            f"G2 theoretical maximum: {coarse + fine + extension} runs / "
            f"{(coarse + fine + extension) * requests:,} measurement requests."
        )
    else:
        g2 = load_processed(config, "g2")
        specs = g3_specs(config, g2["lambda_cap_rps"]) if g2 else []
        if not g2:
            print("G3 is locked until G2 writes measured lambda_cap_rps.")
    payload = {
        "stage": stage,
        "known_runs": len(specs),
        "known_measurement_requests": planned_request_count(config, specs),
        "result_root": str(workspace_path(config, "raw_results")),
        "runs": [asdict(spec) for spec in specs],
    }
    print(json.dumps(payload, indent=2))


def execute_g1(config, resume):
    initial = g1_initial_specs(config)
    run_specs(
        config,
        "g1",
        [spec for spec in initial if spec.mode == "serial"],
        resume,
    )
    run_specs(
        config,
        "g1",
        [spec for spec in initial if spec.mode == "saturation"],
        resume,
    )
    derived = aggregate_g1(config, allow_missing_low_load=True)
    if not derived["seed_variation_gate_passed"]:
        raise RuntimeError(
            "G1 serial/saturation seed variation exceeds the frozen target; "
            "preserve the runs and investigate noise before continuing"
        )
    if config["g1"].get("slo_source", "low_load") == "serial":
        if not derived["gate_passed"]:
            raise RuntimeError("G1 serial SLO gate did not pass")
        return
    start_attempts = {
        scenario: int(
            config["g1"].get("low_load_start_attempt", {}).get(scenario, 0)
        )
        for scenario in g1_scenarios(config)
    }
    for attempt in range(min(start_attempts.values()), 4):
        pending = [
            scenario
            for scenario in g1_scenarios(config)
            if scenario not in derived["low_load"]
            and attempt >= start_attempts[scenario]
        ]
        if not pending:
            if len(derived["low_load"]) == len(g1_scenarios(config)):
                break
            continue
        specs = g1_low_load_specs(
            config,
            derived["saturation_rps"],
            attempt,
            scenarios=pending,
        )
        run_specs(config, "g1", specs, resume=True)
        derived = aggregate_g1(config, allow_missing_low_load=True)
        if derived["gate_passed"]:
            return
        if not derived["seed_variation_gate_passed"]:
            raise RuntimeError(
                "G1 seed variation exceeds the frozen target; preserve the runs and "
                "investigate noise before rerunning"
            )
        print(f"G1 low-load gate failed at attempt {attempt}; halving request rate")
    derived = aggregate_g1(config, allow_missing_low_load=False)
    if derived["gate_passed"]:
        return
    raise RuntimeError("G1 low-load gate did not pass after four attempts")


def g2_extension_rates(
    bracket: dict[str, float | None], extension_factor: float
) -> list[float]:
    rates: list[float] = []
    if bracket["pass_rate"] is None:
        rates.append(float(bracket["min_rate"]) / 2)
    if bracket["fail_rate"] is None:
        rates.append(float(bracket["max_rate"]) * extension_factor)
    return rates


def g2_fine_rates(
    pass_rate: float, fail_rate: float, count: int
) -> list[float]:
    if count < 2:
        raise ValueError("G2 fine scan requires at least two points")
    return [
        pass_rate + (fail_rate - pass_rate) * index / (count - 1)
        for index in range(count)
    ]


def g2_bracket_complete(bracket: dict[str, float | None]) -> bool:
    return bracket["pass_rate"] is not None and bracket["fail_rate"] is not None


def execute_g2(
    config,
    resume,
    *,
    scenarios: set[str] | None = None,
    coarse_only: bool = False,
):
    g1 = load_processed(config, "g1")
    if not g1 or not g1.get("gate_passed"):
        raise RuntimeError("G2 is locked until the complete G1 gate passes")
    slo = load_slo_config(config)
    fingerprint = slo_thresholds_fingerprint(slo)
    if slo.get("config_sha256") != g1.get("config_sha256"):
        raise RuntimeError("G2 SLO does not belong to the completed G1 config")
    conditions = [
        condition
        for condition in g2_condition_scenarios(config)
        if scenarios is None or condition[0] in scenarios
    ]
    if not conditions:
        raise RuntimeError("no G2 conditions match the requested scenario filter")
    strict_brackets: dict[str, dict[str, float]] = {}
    for scenario, arrival, burstiness in conditions:
        key = f"{scenario}:{arrival}"
        coarse_specs = g2_coarse_specs_for_condition(
            config,
            g1["saturation_rps"],
            fingerprint,
            scenario,
            arrival,
        )
        higher_fail: float | None = None
        pass_rate: float | None = None
        for spec in coarse_specs:
            outcome = g2_point_outcome(config, spec.run_key)
            if outcome is None:
                run_specs(config, "g2", [spec], resume=resume)
                outcome = g2_point_outcome(config, spec.run_key)
            if outcome is None:
                raise RuntimeError(f"G2 point has no usable outcome: {spec.run_key}")
            if outcome:
                pass_rate = float(spec.request_rate_rps)
                if higher_fail is not None:
                    break
            else:
                higher_fail = float(spec.request_rate_rps)

        extension = 0
        while (pass_rate is None or higher_fail is None) and extension < int(
            config["arrivals"]["max_bracket_extensions"]
        ):
            extension += 1
            if pass_rate is None:
                base = min(float(spec.request_rate_rps) for spec in coarse_specs)
                rate = base / (2**extension)
            else:
                base = max(float(spec.request_rate_rps) for spec in coarse_specs)
                rate = base * float(config["arrivals"]["extension_factor"]) ** extension
            extension_spec = g2_point_specs(
                config,
                scenario,
                arrival,
                burstiness,
                [rate],
                seeds=[g2_seeds(config)[0]],
                slo_fingerprint=fingerprint,
                attempt=extension,
            )[0]
            outcome = g2_point_outcome(config, extension_spec.run_key)
            if outcome is None:
                run_specs(config, "g2", [extension_spec], resume=True)
                outcome = g2_point_outcome(config, extension_spec.run_key)
            if outcome is None:
                raise RuntimeError(
                    f"G2 extension has no usable outcome: {extension_spec.run_key}"
                )
            if outcome:
                pass_rate = rate
            else:
                higher_fail = rate

        if pass_rate is None or higher_fail is None:
            raise RuntimeError(f"unable to bracket capacity for {key}")
        strict_brackets[key] = {
            "pass_rate": pass_rate,
            "fail_rate": higher_fail,
        }
        print(
            f"G2 {key}: strict descending bracket complete; "
            f"pass={pass_rate}, fail={higher_fail}"
        )

    if coarse_only:
        print("G2 coarse-only requested; fine scan and aggregation were not run.")
        return

    fine_specs = []
    for scenario, arrival, burstiness in conditions:
        bracket = strict_brackets[f"{scenario}:{arrival}"]
        count = int(config["arrivals"]["fine_points"])
        rates = g2_fine_rates(
            float(bracket["pass_rate"]), float(bracket["fail_rate"]), count
        )
        fine_specs.extend(
            g2_point_specs(
                config,
                scenario,
                arrival,
                burstiness,
                rates,
                seeds=g2_seeds(config),
                slo_fingerprint=fingerprint,
            )
        )
    run_specs(config, "g2", fine_specs, resume=True)
    if scenarios is None:
        aggregate_g2(config)
        materialize_phase_trace(config)
    else:
        print(
            "G2 scenario filter requested; global aggregation remains locked "
            "until every configured condition is complete."
        )


def execute_g3(config, resume):
    g2 = load_processed(config, "g2")
    if not g2 or not g2.get("gate_passed"):
        raise RuntimeError("G3 is locked until G2 capacity gate passes")
    materialize_phase_trace(config)
    run_specs(
        config,
        "compile_warmup",
        compilation_warmup_specs(config),
        resume=True,
    )
    specs = g3_specs(config, g2["lambda_cap_rps"])
    equivalence_specs = [
        spec
        for spec in specs
        if spec.policy == "fixed_b2048_explicit"
        or (
            spec.policy == "stock_auto"
            and spec.scenario == "balanced"
            and spec.mode == "poisson"
            and spec.load_factor == 0.9
        )
    ]
    run_specs(config, "g3", equivalence_specs, resume)
    partial = aggregate_g3(config)
    if not partial["equivalence_b2048"]["passed"]:
        raise RuntimeError("Stock-Auto and explicit B2048 differ by more than 3%; G3 stopped")
    run_specs(config, "g3", specs, resume=True)
    aggregate_g3(config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen G1-G3 matrix")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("g1", "g2", "g3"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=(
            "decode_heavy",
            "balanced",
            "prefill_heavy",
        ),
        help="for G2 only, execute one configured workload",
    )
    parser.add_argument(
        "--coarse-only",
        action="store_true",
        help="for G2 only, stop after finding the strict descending bracket",
    )
    args = parser.parse_args()
    if (args.scenario or args.coarse_only) and args.stage != "g2":
        parser.error("--scenario and --coarse-only are only valid with --stage g2")
    config = load_config(args.config)
    errors = verify_manifest(config)
    if errors:
        raise RuntimeError(f"trace manifest verification failed: {errors}")
    if args.dry_run:
        _dry_run(config, args.stage)
    elif args.stage == "g1":
        execute_g1(config, args.resume)
    elif args.stage == "g2":
        execute_g2(
            config,
            args.resume,
            scenarios={args.scenario} if args.scenario else None,
            coarse_only=args.coarse_only,
        )
    else:
        execute_g3(config, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
