# Benchmark decisions

## 2026-08-12: freeze the installed vLLM build

- Use `vLLM 0.26.1rc1.dev535+g83ad767ee.precompiled` from source commit
  `83ad767eed3be3ee7f2df63be693bfaca5c7c922` for G0-G3.
- Keep `/home/dongj/projects/LLM/vllm` unchanged and record its dirty state for
  every run.
- Use `VLLM_USE_V2_MODEL_RUNNER=0` because CUDA UVA is unavailable under the
  current WSL2 environment. This selects the supported V1 model runner; the V1
  engine and stock scheduler remain in use.
- Treat `configs/frozen_experiment.yaml` as the single frozen source of truth.

## 2026-08-12: development-gate scale

- Run 300 measured requests and 10 warmups per configuration for seeds 1, 2,
  and 3.
- Reserve the 1000-request, 3-5 seed test-trace evaluation for G7.
- Select Best-Fixed using validation data only. Test traces remain untouched.

## 2026-08-12: exact synthetic token lengths

- Send synthetic prompts as token ID lists through the OpenAI-compatible
  completions endpoint.
- Do not decode and re-tokenize synthetic prompts, which could change their
  length and undermine fixed-length comparisons.
- Use request-level streaming timestamps for performance runs. Iteration-level
  profiling remains out of scope until G4.

## 2026-08-12: BurstGPT continuous-window selection

- After the frozen filters, the earliest 2000 GPT-4 requests span 231993
  seconds; no 1800-second window exists, and the shortest possible span is
  4980 seconds. The provisional 1800-second bound was therefore infeasible.
- Before any performance run, freeze the deterministic selection as the first
  time-sorted 2000-row valid GPT-4 window whose span is at most 7200 seconds.
  This preserves a continuous real-arrival interval without choosing a window
  based on scheduler results.

## 2026-08-12: one-time compilation-cache warmup

- The first B8192 server startup compiled a new graph range and exposed 95,296
  KV tokens; the immediately repeated identical startup loaded the cache and
  exposed 115,664 tokens. No measured G1-G3 run may be the first compiler-cache
  population for a budget.
- Before G3, start each of the six unique service configurations once with one
  unmeasured balanced request. Keep these runs under `compile_warmup`; exclude
  them from all performance aggregates. Per-run request warmups remain enabled.

## 2026-08-12: minimal G1 SLO-calibration scope

- Reduce G1 to `decode-heavy (128/1024)`, `balanced (512/512)`, and
  `prefill-heavy (1024/128)`. These are the three classes used by the planned
  heterogeneous mixed workload.
- Use 100 measured requests for serial runs and initially retain 300 for
  saturation and low-load runs. Serial data only establishes the no-queue
  reference. The later 2026-08-13 amendment below supersedes the low-load count
  before any SLO was frozen.
- With three seeds, the nominal matrix is 27 runs and 6300 measured requests.
  If a low-load point fails the predeclared no-queue/TTFT gate, retain it and
  add a lower-rate attempt instead of forcing exactly 27 runs.
- This amendment was frozen before any valid formal G1 run completed. It
  produces SLOs only for the three listed classes; long-prefill, ShareGPT, and
  BurstGPT SLOs remain uncalibrated.

## 2026-08-13: automatic Stock-Auto G1-G2 reference pipeline

- G2 conditions are configuration-driven. The first automated reference run
  covers only decode-heavy, balanced, and prefill-heavy with Poisson arrival.
- Tight/Medium/Loose thresholds are all derived from Stock-Auto low-load data;
  Medium is the capacity tier. Future schedulers must reuse these absolute
  thresholds rather than calibrating scheduler-specific SLOs.
- Serial first/last-decile TPOT drift above 10% is retained as a quality
  warning. Per the user's decision it does not invalidate a run, while the
  frozen 5% cross-seed variation gate remains blocking.
- Every G2 run records a fingerprint of the frozen threshold contents. Raw
  results with another config hash or threshold fingerprint are never resumed
  into the current scan.
- Hard-invalid runs are append-only and retried once. A second invalid attempt
  stops the pipeline; no result is overwritten or silently discarded.

## 2026-08-13: G1 saturation stream-coalescing validity

- The locked vLLM async output collector can coalesce adjacent DELTA outputs
  even with `--stream-interval 1`. In repeated decode-heavy saturation runs,
  all 300 requests completed with exact output lengths, no OOM, and drained
  queues, while the stream-coalescing warning was isolated to the first
  request.
- For G1 saturation only, a failure of `single_token_stream_chunks` is therefore
  a quality warning: completed-request throughput remains valid, while exact
  TTFT/TPOT/ITL/max-TBT from that run must not support latency conclusions.
  Serial, low-load, G2, and later SLO/goodput runs retain the hard-invalid rule.
- Persisted attempts count toward `validity.max_run_attempts` when using
  `--resume`, so restarting the command cannot create an unbounded retry loop.
  Raw records remain append-only; legacy G1 saturation records are classified
  at read time and are not rewritten.

## 2026-08-13: reduce remaining G1 low-load runs to 100 requests

- Before freezing any SLO, reduce G1 low-load calibration from 300 to 100
  measured requests per seed because adaptive low-rate runs are dominated by
  their arrival horizon. Retain all three seeds and continue computing each
  seed's P90 independently before averaging seed-level values.
- This is a development-stage SLO calibration; G7 formal comparisons retain
  the requirement of at least 1000 requests per run. The reduced sample size
  and its larger tail uncertainty must be reported with the frozen SLO.
- Explicitly reuse configuration
  `418a6ec401a72e8608aa80491449d8876ac6344717844a4054e638448e7ccde2`
  only for the unchanged trace manifest and G1 Stock-Auto serial/saturation
  baselines. Its 300-request low-load runs, including incomplete runs, are not
  compatible with the amended calibration and remain append-only diagnostics.
- The prior 300-request attempt-0 measurements rejected `0.1 × saturation`
  for all three workloads because mean seed-level P90 TTFT exceeded the frozen
  `1.25 × serial` bound. Treat those points only as rejection evidence and
  resume the new 100-request search at attempt 1 (`0.05 × saturation`). A
  passing SLO calibration point must still be measured anew with 100 requests
  for each of the three seeds.

## 2026-08-13: freeze serial-derived SLO and use one seed for G2

- Before any G2 or DPP result was inspected, supersede the low-load SLO source
  with the mean of the existing three seed-level Stock-Auto serial P90 values.
  Apply the unchanged Tight/Medium/Loose multipliers. Existing low-load raw
  runs remain append-only diagnostics and are neither scheduled nor gated.
- Keep the three-seed serial and saturation baselines as the G1 quality and
  throughput gate. This makes the executable G1 matrix 18 runs and 3600
  measured requests, all already reusable when valid under the compatibility
  rule.
- Run every G2 coarse, extension and fine capacity point with seed 1 only. The
  no-extension matrix is 39 runs (11,700 measured requests), with a theoretical
  maximum of 57 runs (17,100 requests) before hard-invalid retries.
- Label G2 as an exploratory single-seed capacity scan. Its confidence interval
  fields are unavailable (`NaN`), and its `lambda_cap` is a candidate operating
  point for development rather than a replicated final claim. Later scheduler
  comparisons must reuse the same absolute SLO, trace, QPS and seed; formal G7
  conclusions still require the frozen multi-seed protocol.
- Before accepting any G2 capacity result, reduce every G2 coarse, extension
  and fine run from 300 to 100 measured requests. This changes the G2 config
  hash and SLO fingerprint, so the interrupted 300-request G2 run remains raw
  history and cannot be resumed into the new scan. Under the then-current full
  coarse traversal, the 39-run matrix contained 3,900 measured requests and
  the 57-run bound contained 5,700; the descending early-stop amendment below
  supersedes that execution bound.

## 2026-08-13: descending G2 coarse scan with per-workload early stop

- Traverse the frozen coarse factors from `1.1` down to `0.1` for one workload
  at a time. Once a valid point reaches joint attainment at least 90% and a
  higher valid failure exists, stop that workload's coarse scan and move to the
  next configured workload.
- If all coarse points pass or fail, extend only the missing side of the
  bracket under the existing three-extension limit. Keep the existing
  five-point fine scan after every workload has a bracket.
- Do not change config hashes or individual RunSpec keys for this traversal
  change. Existing raw results with the current config and SLO fingerprint are
  reused by `--resume`, including results produced by the former ascending
  traversal. Hard-invalid runs are ignored by bracket construction rather than
  misclassified as SLO failures.
- With three conditions, one seed, and 100 requests per run, a fresh scan now
  needs at least 21 runs if each second coarse point passes, at most 39 without
  extension, and at most 48 with all three one-sided extensions. Existing raw
  runs may make the observed total larger than the fresh optimized traversal.

## 2026-08-13: freeze nearest sub-90% points for scheduler comparison

- Preserve the existing definition of `lambda_cap` as the largest valid
  measured QPS whose joint Medium-SLO attainment is at least 90%. Do not
  relabel an SLO-failing point as capacity.
- For subsequent development-stage scheduler comparisons, select from each
  workload's valid G2 Poisson measurements the point with joint attainment
  strictly below 90% and closest to 90%. If multiple points have identical
  attainment, prefer the lower target QPS and then the lexicographically
  smaller run ID. Freeze the resulting absolute offered rates in
  `configs/comparison_qps.yaml`.
- The frozen targets are decode-heavy `4.032987082` requests/s (83%), balanced
  `5.782988437` requests/s (84%), and prefill-heavy `3.416571588` requests/s
  (87%). Every compared scheduler must reuse these exact target QPS values,
  the existing Medium SLO thresholds, trace, arrival process, request count,
  and seed; do not normalize by scheduler-specific capacity.
- These targets were selected from one seed with 100 requests per point and
  are therefore exploratory. They intentionally put Stock-Auto just beyond
  its observed SLO knee and are suitable for testing whether another
  scheduler can recover goodput. Formal G7 claims still require the frozen
  multi-seed, larger-sample protocol.
