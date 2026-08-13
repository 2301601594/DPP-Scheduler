# Stock Scheduler benchmark: G0-G3

The executable source of truth is `configs/frozen_experiment.yaml`.

The current G1 gate is the minimal SLO-calibration matrix: three fixed-length
classes, 100 serial requests and 300 saturation requests per seed. It is 18
runs and 3600 measured requests. Tight/Medium/Loose SLOs are frozen from the
mean of the three seed-level Stock-Auto serial P90 values. Existing low-load
runs are retained as diagnostics but are no longer scheduled or gated.

The preferred G1-G2 entry point is:

```bash
.venv/bin/python -m benchmarks.run_g1_g2 \
  --config configs/frozen_experiment.yaml --dry-run
.venv/bin/python -m benchmarks.run_g1_g2 \
  --config configs/frozen_experiment.yaml --execute --resume
.venv/bin/python -m benchmarks.run_g1_g2 \
  --config configs/frozen_experiment.yaml --report-only
```

It runs one ten-request Stock-Auto preflight, reuses or completes the G1
serial/saturation gate, freezes the shared Stock-derived serial SLO, and then
performs the configured G2 adaptive scan. The current G2 scope is the three
fixed classes under Poisson arrival with seed 1 only. Each workload is probed
from the largest coarse QPS downward and stops as soon as a pass/fail bracket
exists, before moving to the next workload. The five-point fine scan runs after
all workload brackets exist. A fresh run needs at least 21 runs when each
second coarse point passes, at most 39 without extensions, and at most 48 with
three extensions per workload; every run has 100 measured requests. Medium is
the primary capacity SLO. This reduced single-seed scan locates candidate
capacity knees but cannot estimate cross-seed confidence intervals and has
higher tail uncertainty. `stock_reference.csv/json` is the reference for later
schedulers; it is not a multi-scheduler result by itself.

1. `.venv/bin/python -m benchmarks.prepare_traces --config configs/frozen_experiment.yaml`
   downloads and freezes ShareGPT/BurstGPT, generates validation/test traces,
   and writes a SHA256 manifest.
2. `.venv/bin/python -m benchmarks.smoke_test --config configs/frozen_experiment.yaml --execute`
   runs the four required GPU smoke configurations.
3. Use `benchmarks.run_g1_g2 --dry-run` to review the combined adaptive bounds,
   then replace it with `--execute --resume` after review.
4. `.venv/bin/python -m benchmarks.aggregate_results --config configs/frozen_experiment.yaml --stage g1`
   derives and freezes SLOs from the serial baseline.
5. The combined runner enters G2 only after the complete G1 gate passes. G2
   freezes measured capacity and materializes the phase-shift trace.
6. Repeat run/aggregate for G3. G3 first populates each budget's compilation
   cache, checks Auto-vs-explicit-B2048 equivalence, and selects Best-Fixed from
   validation only.
7. `.venv/bin/python -m benchmarks.aggregate_results --config configs/frozen_experiment.yaml --stage all`
   rebuilds every processed table and artifact from append-only raw results.

Raw results are append-only. Performance and iteration-profile runs are never
mixed; G4 profiling is not implemented here.
