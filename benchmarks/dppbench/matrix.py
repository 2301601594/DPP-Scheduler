from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from benchmarks.dppbench.config import workspace_path


G3_STANDARD_SCENARIOS = (
    "decode_heavy",
    "balanced",
    "prefill_heavy",
    "long_prefill",
    "heterogeneous",
    "sharegpt",
)


def g1_scenarios(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(name) for name in config["g1"]["scenarios"])


def g2_seeds(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(seed)
        for seed in config["g2"].get("seeds", config["statistics"]["seeds"])
    )


def g2_measurement_requests(config: dict[str, Any]) -> int:
    return int(
        config["g2"].get(
            "measurement_requests", config["statistics"]["measurement_requests"]
        )
    )


@dataclass(frozen=True)
class RunSpec:
    stage: str
    scenario: str
    mode: str
    policy: str
    seed: int
    trace_path: str
    budget: int | None = None
    request_rate_rps: float | None = None
    burstiness: float | None = 1.0
    self_timed: bool = False
    max_concurrency: int | None = None
    load_factor: float | None = None
    attempt: int = 0
    # None uses the frozen measurement count; 0 consumes the complete timed trace.
    request_limit: int | None = None
    warmup_requests: int | None = None
    # G2 runs are bound to the frozen threshold set, not to mutable capacities.
    slo_fingerprint: str | None = None

    @property
    def run_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:20]


def trace_for(config: dict[str, Any], scenario: str) -> Path:
    name = "burstgpt" if scenario == "burstgpt_length" else scenario
    return workspace_path(config, "traces") / f"validation_{name}.jsonl"


def g1_initial_specs(config: dict[str, Any]) -> list[RunSpec]:
    specs = []
    for scenario in g1_scenarios(config):
        for seed in config["statistics"]["seeds"]:
            common = {
                "stage": "g1",
                "scenario": scenario,
                "policy": "stock_auto",
                "seed": int(seed),
                "trace_path": str(trace_for(config, scenario)),
            }
            specs.append(
                RunSpec(
                    **common,
                    mode="serial",
                    request_rate_rps=math.inf,
                    max_concurrency=1,
                    request_limit=int(
                        config["statistics"]["serial_measurement_requests"]
                    ),
                )
            )
            specs.append(
                RunSpec(
                    **common,
                    mode="saturation",
                    request_rate_rps=math.inf,
                    request_limit=int(
                        config["statistics"]["saturation_measurement_requests"]
                    ),
                )
            )
    return specs


def g1_low_load_specs(
    config: dict[str, Any],
    saturation_rps: dict[str, float],
    attempt: int = 0,
    scenarios: Iterable[str] | None = None,
) -> list[RunSpec]:
    fraction = float(config["validity"]["low_load_initial_saturation_fraction"])
    specs = []
    selected = tuple(scenarios) if scenarios is not None else g1_scenarios(config)
    for scenario in selected:
        if scenario not in saturation_rps:
            raise KeyError(f"missing G1 saturation rate for {scenario}")
        request_rate = saturation_rps[scenario] * fraction / (2**attempt)
        for seed in config["statistics"]["seeds"]:
            specs.append(
                RunSpec(
                    stage="g1",
                    scenario=scenario,
                    mode="low_load",
                    policy="stock_auto",
                    seed=int(seed),
                    trace_path=str(trace_for(config, scenario)),
                    request_rate_rps=request_rate,
                    burstiness=1.0,
                    attempt=attempt,
                    request_limit=int(
                        config["statistics"]["low_load_measurement_requests"]
                    ),
                )
            )
    return specs


def g2_condition_scenarios(config: dict[str, Any]) -> list[tuple[str, str, float]]:
    return [
        (
            str(condition["scenario"]),
            str(condition["arrival"]),
            float(condition["burstiness"]),
        )
        for condition in config["g2"]["conditions"]
    ]


def g2_coarse_specs(
    config: dict[str, Any],
    saturation_rps: dict[str, float],
    slo_fingerprint: str,
) -> list[RunSpec]:
    # Probe from the largest offered load downward. Run keys depend on the
    # individual spec, not list position, so results from the former ascending
    # traversal remain reusable.
    factors = sorted(
        (float(value) for value in config["arrivals"]["coarse_saturation_factors"]),
        reverse=True,
    )
    seed = g2_seeds(config)[0]
    specs = []
    for scenario, arrival, burstiness in g2_condition_scenarios(config):
        for factor in factors:
            specs.append(
                RunSpec(
                    stage="g2",
                    scenario=scenario,
                    mode=f"capacity_{arrival}",
                    policy="stock_auto",
                    seed=seed,
                    trace_path=str(trace_for(config, scenario)),
                    request_rate_rps=float(saturation_rps[scenario]) * float(factor),
                    burstiness=burstiness,
                    load_factor=float(factor),
                    request_limit=g2_measurement_requests(config),
                    slo_fingerprint=slo_fingerprint,
                )
            )
    return specs


def g2_coarse_specs_for_condition(
    config: dict[str, Any],
    saturation_rps: dict[str, float],
    slo_fingerprint: str,
    scenario: str,
    arrival: str,
) -> list[RunSpec]:
    return [
        spec
        for spec in g2_coarse_specs(config, saturation_rps, slo_fingerprint)
        if spec.scenario == scenario
        and spec.mode == f"capacity_{arrival}"
    ]


def g2_point_specs(
    config: dict[str, Any],
    scenario: str,
    arrival: str,
    burstiness: float,
    rates: Iterable[float],
    *,
    seeds: Iterable[int],
    slo_fingerprint: str,
    attempt: int = 0,
) -> list[RunSpec]:
    return [
        RunSpec(
            stage="g2",
            scenario=scenario,
            mode=f"capacity_{arrival}",
            policy="stock_auto",
            seed=int(seed),
            trace_path=str(trace_for(config, scenario)),
            request_rate_rps=float(rate),
            burstiness=burstiness,
            attempt=attempt,
            request_limit=g2_measurement_requests(config),
            slo_fingerprint=slo_fingerprint,
        )
        for rate in rates
        for seed in seeds
    ]


def preflight_specs(config: dict[str, Any]) -> list[RunSpec]:
    return [
        RunSpec(
            stage="preflight",
            scenario="balanced",
            mode="serial",
            policy="stock_auto",
            seed=int(config["statistics"]["seeds"][0]),
            trace_path=str(trace_for(config, "balanced")),
            request_rate_rps=math.inf,
            max_concurrency=1,
            request_limit=10,
        )
    ]


def policies(config: dict[str, Any]) -> list[tuple[str, int | None]]:
    return [("stock_auto", None)] + [
        (f"fixed_b{int(budget)}", int(budget))
        for budget in config["budgets"]["fixed"]
    ]


def compilation_warmup_specs(config: dict[str, Any]) -> list[RunSpec]:
    return [
        RunSpec(
            stage="compile_warmup",
            scenario="balanced",
            mode="startup_cache",
            policy=policy,
            budget=budget,
            seed=0,
            trace_path=str(trace_for(config, "balanced")),
            request_rate_rps=math.inf,
            max_concurrency=1,
            request_limit=1,
            warmup_requests=0,
        )
        for policy, budget in policies(config)
    ]


def g3_specs(
    config: dict[str, Any], capacities: dict[str, float]
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    seeds = [int(seed) for seed in config["statistics"]["seeds"]]
    base_policies = policies(config)

    def rotated(seed: int) -> list[tuple[str, int | None]]:
        offset = seeds.index(seed) % len(base_policies)
        return base_policies[offset:] + base_policies[:offset]

    for scenario in G3_STANDARD_SCENARIOS:
        for factor in config["arrivals"]["g3_load_factors"]:
            for seed in seeds:
                for policy, budget in rotated(seed):
                    specs.append(
                        RunSpec(
                            stage="g3",
                            scenario=scenario,
                            mode="poisson",
                            policy=policy,
                            budget=budget,
                            seed=seed,
                            trace_path=str(trace_for(config, scenario)),
                            request_rate_rps=float(capacities[scenario]) * float(factor),
                            burstiness=1.0,
                            load_factor=float(factor),
                        )
                    )
    for arrival, burstiness in (("gamma_medium", 0.5), ("gamma_strong", 0.2)):
        for seed in seeds:
            for policy, budget in rotated(seed):
                specs.append(
                    RunSpec(
                        stage="g3",
                        scenario="heterogeneous",
                        mode=arrival,
                        policy=policy,
                        budget=budget,
                        seed=seed,
                        trace_path=str(trace_for(config, "heterogeneous")),
                        request_rate_rps=float(capacities["heterogeneous"]) * 0.9,
                        burstiness=burstiness,
                        load_factor=0.9,
                    )
                )
    for scenario in ("phase_shift", "burstgpt"):
        for seed in seeds:
            for policy, budget in rotated(seed):
                specs.append(
                    RunSpec(
                        stage="g3",
                        scenario=scenario,
                        mode="self_timed",
                        policy=policy,
                        budget=budget,
                        seed=seed,
                        trace_path=str(trace_for(config, scenario)),
                        self_timed=True,
                        burstiness=None,
                        request_limit=0 if scenario == "phase_shift" else None,
                    )
                )
    for seed in seeds:
        specs.append(
            RunSpec(
                stage="g3",
                scenario="balanced",
                mode="equivalence_b2048",
                policy="fixed_b2048_explicit",
                budget=int(config["budgets"]["equivalence_budget"]),
                seed=seed,
                trace_path=str(trace_for(config, "balanced")),
                request_rate_rps=float(capacities["balanced"]) * 0.9,
                burstiness=1.0,
                load_factor=0.9,
            )
        )
    return specs


def planned_request_count(config: dict[str, Any], specs: Iterable[RunSpec]) -> int:
    measurement = int(config["statistics"]["measurement_requests"])
    count = 0
    for spec in specs:
        if spec.request_limit == 0 and Path(spec.trace_path).exists():
            with Path(spec.trace_path).open(encoding="utf-8") as stream:
                count += sum(1 for line in stream if line.strip())
        elif spec.request_limit is not None:
            count += spec.request_limit
        else:
            count += measurement
    return count
