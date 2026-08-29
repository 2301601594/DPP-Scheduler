# Qwen3-14B modular DPP Scheduler experiment log

## 2026-08-21 — model snapshot identified and recorded

- The Qwen3-14B BF16 snapshot is present on the DGX at
  `/home/dongj/models/Qwen3-14B-BF16` (28G, 8 safetensors shards), acquired on
  2026-08-21 via ModelScope:
  `modelscope download --model Qwen/Qwen3-14B --local-dir $HOME/models/Qwen3-14B-BF16 --max-workers 1`.
- SHA256 of all 19 files was computed on the DGX and compared with the
  HuggingFace `Qwen/Qwen3-14B` `resolve/main` content. All 8 weight shards and
  `tokenizer.json` match the HF LFS sha256 exactly; `config.json`,
  `tokenizer_config.json`, `generation_config.json`,
  `model.safetensors.index.json`, `vocab.json`, `merges.txt`, `LICENSE`, and
  `README.md` match the HF git blob OIDs. The snapshot is therefore
  content-identical to HuggingFace main commit
  `40c069824f4251a91eefaf281ebe4c544efd3e18` (last modified 2025-07-26).
- Known non-content differences: `configuration.json` is a ModelScope-only
  artifact (absent on HF); `.gitattributes` is ModelScope-regenerated and
  differs from the HF blob. Neither affects model or tokenizer content.
- License reviewed: Apache-2.0 (`LICENSE` file and README card).
- Recorded in `configs/qwen3_14b_snapshot_manifest.json`; `model` sections of
  `configs/dgx_spark_experiment.yaml` and `configs/dgx_spark_environment.json`
  updated to verified revision/hashes.
- Remaining G0 blockers: operator approval confirmation of the acquisition, the
  bounded remote Qwen3-14B BF16 smoke, runtime/SchedulerConfig/KV capture,
  stock iteration-event semantics, natural-EOS trace manifest, and
  SLO/Goodput definitions. No GPU workload was run by this entry.

## 2026-08-21 — plan migration

- Adopted
  `docs/Qwen3-14B-DGX-Spark-Modular-DPP-Scheduler.md` as the authoritative
  design.
- Rewrote the repository instructions, active provisional DGX config,
  experiment plan, decisions, and remote model workflow for Qwen3-14B BF16 and
  the planned exact atomic `BatchPlan` design.
- Removed free-form model launchers. A config-generated Qwen3-14B launcher and
  exact `BatchPlan` execution remain future G0/G2 implementation work.
- Reopened the campaign at G0. The exact model revision/snapshot,
  model-specific runtime and KV facts, natural-EOS traces, Stock SLO/Goodput
  definitions, Predictor, and DPP parameters remain pending.
- No model was downloaded, no project Python command or GPU workload was run,
  and no benchmark measurement was created by this migration.
- Historical evidence remains isolated and is not an active input.

## 2026-08-21 — G2 modular scheduler skeleton implemented

- Added the initial `dpp_scheduler/` package with public immutable contracts,
  a deterministic Candidate Generator, a temporary Selector, a Controller, a
  callback-based exact-plan Adapter contract, and G2-safe placeholder modules
  for Predictor, Safe-Set, Observer, StateStore, ConsequenceEstimator, and
  Fallback.
- The Candidate Generator currently emits the 4 Prefill-cap × 3 Decode-profile
  templates, uses stable ordering, canonical deduplication, token/sequence
  limits, and pure KV projection. All prefill cap values and urgent limit remain
  provisional until G0/G1 profiling.
- Added `tests/unit/test_scheduler_g2.py` covering contracts, candidate
  generation, deterministic selection, and exact selected-vs-executed plan
  matching through a fake Adapter.
- Remote unit suite passes: `42` tests total, including the new G2 tests.
- Added a commit-specific `VllmAdapter` implementation that constructs a
  `StateSnapshot` from a live vLLM Scheduler and materializes the selected
  `BatchPlan` as a vLLM `SchedulerOutput` through the exact prefill/decode path.
  It still needs to be validated against the real Qwen3-14B DGX server during
  G0; until then `CallbackVllmAdapter` remains the deterministic test double
  used by the unit suite.

## 2026-08-22 — G0 stock vLLM capture completed

- Added `benchmarks/capture_qwen3_g0.py`, which resolves vLLM EngineArgs,
  starts a stock vLLM server on the DGX, captures the startup log and KV facts,
  sends one bounded model-execution smoke completion, and writes append-only
  raw evidence. The 16-token API cap is a client guard, not natural-EOS
  workload evidence and not a Scheduler input.
- Stock capture run:
  - Model: Qwen3-14B-BF16 at `/home/dongj/models/Qwen3-14B-BF16`
  - vLLM: `83ad767eed3be3ee7f2df63be693bfaca5c7c922`
  - Scheduler: default vLLM Scheduler, not `ModularDPPScheduler`
  - `max_model_len=40960`
  - `max_num_batched_tokens=2048`
  - `max_num_seqs=64`
  - `gpu_memory_utilization=0.84`
  - `kv_cache_dtype=bfloat16`
  - chunked prefill on, prefix caching off, async scheduling off
- Captured KV facts:
  - KV block size: 16
  - GPU KV cache size: 482,384 tokens
  - usable KV blocks: 30,149
  - available KV cache: 73.61 GiB
  - max concurrency at 40,960 tokens/request: 11.78x
- Smoke completion passed with `finish_reason=length`, as expected for its
  deliberately small 16-token guard. It does not validate a natural-completion
  distribution.
- Authoritative facts recorded in:
  - `configs/qwen3_14b_g0_stock_capture.json`
  - `results/raw/qwen3_14b_dgx_spark/g0_stock_capture_084/`
- The final rerun used the corrected capture script with `async_scheduling=False`
  in the resolved EngineArgs, matching the actual server launch command.
- Remaining G0 items are not frozen yet: reviewed length-blind trace manifest,
  Stock iteration semantics, TTFT/TBT SLO and Goodput definitions, and remote
  cleanliness verification after legacy cleanup.

## 2026-08-22 — Stage 1 Qwen3 request pool built

- Added `benchmarks/qwen3_build_request_pool.py`.
- Used local ShareGPT_V3 dataset (`data/raw/ShareGPT_V3_unfiltered_cleaned_split.json`, SHA256
  `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`).
- Rendered the first human turn from each conversation through the Qwen3-14B
  chat template with `enable_thinking=False`.
- Filtered to input token lengths `[128, 8192]`, deduplicated, then
  deterministically sampled 3000 prompts with seed 1001.
- Output:
  - `traces/qwen3_14b/request_pool.jsonl` (3000 records)
  - `traces/qwen3_14b/request_pool.meta.json`
- Pool input-token stats: min=128, max=7938, mean=765.67.
- Request pool SHA256:
  `9d5e57900b9580480c7912a65d449bd608e0edc1293578c6c3e5d73fc71dbab5`.
- Stage 1 is complete. No arrival times, generation seeds, or reference output
  lengths are included yet.

## 2026-08-22 — Fast shared-parameter coarse scan (retired)

- Added:
  - `benchmarks/generate_stock_scan_trace.py`
  - `benchmarks/run_stock_shared_scan.py`
- Used a fast coarse probe: 30 requests, concurrent arrivals, safety cap 64 tokens per request.
- Token-budget scan at `max_num_seqs=64`: [512, 1024, 2048, 4096].
- Sequence scan at `max_num_batched_tokens=2048`: [32, 64, 128].
- All configurations completed 30/30 requests with no failures.
- This probe is retired: its launcher hard-coded
  `gpu_memory_utilization=0.90`, did not honor the recorded generation seed,
  and recorded dispatch timing after completion. Its scripts, draft traces,
  are removed from current tooling and none of its latency observations are
  active evidence. Its ignored raw outputs were removed during the explicit
  2026-08-25 results cleanup and are excluded from all active inputs.

## 2026-08-22 — Shared runtime parameters fixed at 0.84 GPU utilization

- Fixed shared runtime parameters:
  - `max_num_batched_tokens = 2048`
  - `max_num_seqs = 64`
  - `gpu_memory_utilization = 0.84`
- Re-captured G0 KV facts at 0.84 GPU utilization:
  - KV block size: 16
  - GPU KV cache tokens: 482,384
  - usable KV blocks: 30,149
  - available KV cache: 73.61 GiB
  - max concurrency at 40,960 tokens/request: 11.78x
- Updated:
  - `configs/dgx_spark_experiment.yaml`
  - `configs/dgx_spark_environment.json`
  - `configs/qwen3_14b_g0_stock_capture.json`
- Raw evidence:
  - `results/raw/qwen3_14b_dgx_spark/g0_stock_capture_084/`
- These are now fixed as the shared Stock/DPP baseline. No further parameter scan or validation was run per user instruction.

## 2026-08-22 — runner correction and obsolete campaign cleanup

- Replaced the parameter-overridable Stock scan runner with a launcher derived
  exclusively from `configs/dgx_spark_experiment.yaml`. It rejects provisional
  configs for real server execution, records config/trace hashes and actual
  dispatch times, honors per-request seeds, and counts only real streamed token
  events for TTFT/TBT.
- Defined the finite completion cap as a client termination guard only. It is
  absent from every Scheduler contract and both `stop` and `length` terminal
  reasons are retained and stratified.
- Replaced finite-sample-rescaled arrivals with independent exponential
  inter-arrivals. Old generated scan/validation drafts are invalid and excluded
  from current tooling; their trace directories were permanently deleted after
  explicit confirmation.
- Corrected the modular contracts and exact-plan scaffold: running partial
  Prefill stays visible, active sequence count is projected correctly, running
  Prefill uses vLLM cached-request data, all public round contracts use the
  same hash, and provisional ModularDPP launch fails closed.
- Deleted obsolete version-controlled 5070 campaign
  code/configs/traces/results/plots/tests and the standalone Qwen3-8B NVFP4
  compatibility-smoke files. A separately confirmed cleanup deleted the old
  5070 raw results and BurstGPT input. Old model caches and retained Qwen3 raw
  evidence were left untouched. No benchmark or GPU workload was run.

## 2026-08-22 — G0 wrap-up and G1 start

- Completed G0 wrap-up without running a new GPU benchmark.
- Generated a length-blind natural-EOS Poisson calibration trace set remotely:
  - 200 requests per trace
  - seeds 1001, 1002
  - requested QPS: 0.5, 1.0, 2.0, 4.0, 6.0
  - arrival process: independent exponential inter-arrivals
  - no predetermined output length; client safety ceiling remains 16384 tokens
  - staged under `results/raw/qwen3_14b_dgx_spark/g0_trace_manifest_stage1/`
- Pulled and promoted the trace files to `traces/qwen3_14b/` and froze
  `traces/qwen3_14b/manifest.json`.
- Frozen TTFT/TBT event semantics and request-level Goodput definitions in
  `configs/dgx_spark_experiment.yaml`.
- Updated `configs/dgx_spark_experiment.yaml` to `status: frozen_g0`,
  `stage: g1`, with an empty `g0_freeze_blockers` list.
- Updated `configs/dgx_spark_environment.json` and
  `configs/qwen3_14b_g0_stock_capture.json` to frozen status.
- Updated `docs/experiment_plan.md` to reflect G0 complete and G1 in progress.
- Numeric TTFT/TBT SLOs and final test QPS remain to be calibrated from the
  G1 Stock baseline runs.

## 2026-08-22 — G1 Stock matrix trace set with 2048 client cap

- Per user request, changed the client safety ceiling to 2048 tokens for the
  G1 Stock QPS matrix to bound runtime.
- Generated a one-seed (1001) Poisson trace set at QPS 0.5/1/2/4/6 with 200
  requests per trace.
- Promoted as `traces/qwen3_14b/qps_*_seed_1001_cap2048.jsonl` and froze
  `traces/qwen3_14b/manifest_cap2048.json`.
- `configs/dgx_spark_experiment.yaml` now points to
  `manifest_cap2048.json` and `client_safety_ceiling_tokens: 2048`.

## 2026-08-22 — Freeze G1 SLO and candidate test QPS

- Analyzed the full Stock QPS matrix (0.2/0.25/0.3 plus earlier 0.5–6.0).
- Saturation estimate: roughly 0.25 req/s (about 200–216 output tokens/s with
  ~850 mean output tokens).
- QPS 0.20 and 0.25 are stable/near-saturation; QPS 0.30 shows TTFT
  degradation and is retained as an overload candidate.
- Using Stock P95 + margin, froze:
  - TTFT SLO = 2.0s
  - TBT SLO = 0.25s
- Candidate test QPS:
  - 0.20
  - 0.25
  - 0.30
- Updated `configs/dgx_spark_experiment.yaml` and
  `docs/experiment_plan.md`.

## 2026-08-25 — conservative local results cleanup

- Added `results/README.md` to separate active raw evidence, reproducible
  derived datasets, and reviewed processed summaries.
- Removed the retired `shared_scan*` raw namespaces. These records used the
  rejected 0.90 GPU-memory setting, incorrect dispatch timing, or an unfrozen
  seed and were already excluded from every active input.
- Removed the superseded `g0_config_probe`, `g0_stock_capture_20260822`, and
  `g0_stock_capture_final` local namespaces. The authoritative 0.84 capture
  remains `g0_stock_capture_084` and is unchanged.
- Removed the reproducible
  `dpp_v2_phase_a_tie_analysis_v1/phase_a_report.json` and
  `dpp_v2_phase_a1_tiebreak_analysis_v1/phase_a1_frame_changes.jsonl`; their
  source diagnostic run and analysis scripts remain available. The compact
  reviewed Phase A.1 report remains version controlled.
- No Predictor source run, training/split dataset, valid or failed Scheduler
  run, negative result, current DPP v2/v2.1 source evidence, trace manifest, or
  frozen artifact was deleted. No remote result was changed.

## 2026-08-25 — reject and repair mis-scoped 300-request DPP pair

- Campaign
  `20260825T140656Z_scheduler_pair_n300_qps0p25_seed1001` completed its Stock
  run, but the DPP EngineCore failed before serving requests. The runner marked
  a complete development-trace run as `formal`, so the development-only v2
  artifact gate correctly rejected startup.
- Fixed `benchmarks/run_stock_natural_eos.py` to derive DPP execution scope from
  both trace scope and request truncation. A development trace is always
  `development_nonformal`; only a complete active-frozen-trace run requests
  formal artifacts. The artifact validator was not weakened.
- The failed DPP record and completed Stock record remain append-only, but the
  campaign is invalid for comparison. A repaired DPP run cannot be paired with
  the old Stock run because the root source/dirty identity changed; both
  policies must be rerun under one new timestamped campaign.
- Remote regression test `test_qwen3_runner.py` passed 12/12. An exact DPP
  retry dry-run resolved 300 requests, QPS 0.25, seed 1001, and
  `DPP_EXECUTION_SCOPE=development_nonformal` without enabling detailed
  iteration logging.

## 2026-08-28 — online Predictor calibration repair and n=150 diagnostic

- Limited the runtime change to Predictor online residual calibration. The
  online center is now the symmetric 5%-per-tail trimmed mean, the upper tail
  is the untrimmed higher-method residual P95, and conservative duration is
  floored at expected duration. The offline Ridge models, cold-start artifact,
  Candidate Generator, Safe-Set, Selector, Controller fallback, and actual-only
  feedback semantics were not changed.
- Remote model-free checks passed 67/67 across the targeted Predictor,
  artifact/config, runner, Safe-Set, fallback, exact-plan, and evaluation-path
  suites.
- Completed the append-only development run
  `predictor_online_trimmed_calibration_qps0p25_n150_seed1001_v1/runs/`
  `predictor_online_trimmed_calibration_n150_qps0p25_seed1001_attempt01` on the
  unchanged QPS 0.25, seed 1001, n=150 trace with trace SHA-256
  `203e7ed43522f71e44b7ee99a5cf3d5593f2e2d31215f010f67afd5ee2819e31`.
  It completed 150/150 requests with zero request failures and zero token
  accounting mismatches. The run is `development_nonformal`, used detailed
  iteration logging, and is not formal comparison evidence.
- The repair removed the diagnosed failure chain. The pre-repair diagnostic
  had 4,463 frames containing a `PREDICTION_INVALID` rejection, 4,219 empty
  Safe-Sets, and 4,218 `LIVENESS_ESCAPE_DECODE` fallbacks. The repaired run had
  no workload fallback and no workload-empty Safe-Set. Its only invalid
  rejection was the expected empty `plan-ZERO` on frame 1 while ten valid
  candidates remained; its only fallback was the final empty-queue idle frame.
  Across 6,354 logged valid candidate predictions, no
  `conservative_duration < expected_duration` violation occurred.
- Scheduler performance did not pass. TTFT mean/p50/p90 were
  19,093/2,121/59,530 ms versus 716/545/1,496 ms for Stock on the same trace;
  throughput was 0.1723 versus 0.1865 requests/s. TBT mean improved slightly
  from 191.8 to 188.1 ms, but TBT P99 worsened from 648.7 to 821.6 ms.
- Observed mechanism: 1,253 Scheduler frames had a Prefill backlog; 1,087
  (86.75%) selected `ZERO`, every one of those frames admitted only that one
  candidate to Selector Stage 2, and the longest consecutive backlog-`ZERO`
  streak was 452 frames. This supports the inference that the old Predictor
  failure and decode-liveness fallback masked a separate Stage-1/Selector
  Prefill-starvation behavior. No Selector change or gate advance was made.

## 2026-08-29 — full Selector Diagnosis rerun after Predictor repair

- Repeated the unchanged QPS 0.25, seed 1001, n=150 development trace with
  both the per-iteration log and full Selector Diagnosis enabled. The unique
  append-only run is
  `predictor_online_trimmed_calibration_selector_diagnosis_qps0p25_n150_seed1001_v1/runs/selector_diagnosis_n150_qps0p25_seed1001_attempt01`.
- The run completed 150/150 requests with zero request failures and zero token
  accounting mismatches. The 4,573-frame Selector replay reported zero Stage
  1, service-rate invariant, Stage 2 score, debt, tie-break, and winner
  mismatches. `selector_diagnosis_valid` is true. The retained 77,175,969-byte
  JSONL has SHA-256
  `16bd5ca4fe921a4b8a98ab3cbf10b321ce7b086263539d70aca455fa305162d5`.
- Full Stage-1 data confirmed the starvation mechanism. Among 1,611 frames
  with both an active TBT constraint and Prefill backlog, Stock passed in 41
  and was selected in all 41. Stock was filtered in 1,570 frames: `ZERO` won
  1,527 and a partial-Prefill plan won 43. In the `ZERO` cases, Stock effective
  duration was 305.75--988.77 ms (P50 871.19 ms), while the per-frame limit was
  268.77--269.67 ms and `ZERO` duration was 167.63--190.72 ms.
- Across all frames, Stock entered Stage 2 3,002 times and was selected 2,997
  times. Its five losses all occurred with no active TBT obligation and went
  to P70/P80/P90 plans with a strictly higher Prefill service rate. No
  Selector change or gate advance was made; this remains
  `development_nonformal` diagnostic evidence.

## 2026-08-29 — executed-plan Predictor accuracy replay

- Replayed the exact Predictor durations recorded in the preceding n=150
  Selector Diagnosis against the aligned actual batch durations in its
  `startup.log`. Source manifest, Diagnosis, and startup-log hashes matched;
  the prior Selector replay remained zero-mismatch. Of 4,573 frames, 4,572
  non-empty executions joined by frame and exact plan token composition with
  zero alignment mismatch. The excluded final frame was the audited
  `IDLE_EMPTY_QUEUE` zero-token execution, not a Predictor observation.
- Across the 4,572 executed batches, expected-duration MAE/RMSE/bias were
  1.861/6.749/-0.299 ms; P95/P99 absolute error was 3.801/36.040 ms. The
  expected estimate was within 10 ms for 96.74% and within 10% for 98.91% of
  frames. All selected plans were in-support interpolation, so effective
  duration equalled expected duration throughout this run.
- The aggregate is decode-dominated: 4,416 decode-only frames had 0.993 ms
  expected MAE and 2.688 ms P95 absolute error, while 155 mixed frames had
  26.507 ms MAE, 66.537 ms P95 absolute error, and -7.044 ms bias. The sole
  Prefill-only frame is not an estimable stratum. The largest expected
  underprediction was 162.788 ms on mixed frame 2,489.
- Conservative coverage was 93.26% overall (4,264/4,572), 93.46% for
  decode-only (4,127/4,416), and 87.74% for mixed (136/155). Within the 123
  mixed frames after the replayed online window became active, coverage was
  84.55% and expected-duration bias was -17.074 ms. Thus this trace does not
  support a 95% conservative-coverage claim, especially for mixed batches.
- Processed evidence is retained remotely at
  `results/processed/qwen3_14b_dgx_spark/`
  `predictor_accuracy_replay_selector_diagnosis_n150_seed1001_v1/`.
  The replay script SHA-256 is
  `aa7cd5f05d34e168724399c2bf8067852f30c1ace6e7c31c2c390c4cc546b624`.
  `per_frame_accuracy.csv` SHA-256 is
  `5c6031409de915b20dc5180edca86843f264d8014334a49da5f63d9735509e83`,
  `summary.json` SHA-256 is
  `bdd7043abd8945ad4202a990cb4e4789108a459b96c9a177cc60249ddc7141e0`,
  and `artifact_manifest.json` SHA-256 is
  `42e44d49826e9340c49fe2fc3a448f129ad18ccf0e98737072bc5ead2b679fb3`.
- This is selected-plan, single-trace, development/non-formal evidence. The
  Diagnosis does not store Ridge base duration, so the replay cannot isolate
  the online calibration improvement versus the base model. Calibration
  source is reconstructed from the frozen 32-sample threshold, 128-sample
  per-kind window, and prior actual-only in-support executions. No Predictor,
  Selector, gate, or formal claim was changed by this analysis.

## 2026-08-30 — Stage-1 V2-B ZERO-relative ΔN≤N Selector implementation

- Replaced the online min-slack Stage 1 with the ZERO-relative incremental
  TBT-violation guard: per-candidate `ΔN = Σ_j [m_j(c) − m_j(0)]⁺` computed
  with conservative risk durations and `ConsequenceEstimator._misses`
  semantics (`>` when served, `>=` otherwise); a candidate is admitted iff
  `ΔN ≤ maximum_incremental_tbt_violations`. ΔL is recorded, not filtering.
  ZERO reference resolution is deterministic (ZERO template → STOCK identity
  → any zero-service full-decode candidate by ascending plan_id) and
  fail-closes with `ZERO_REFERENCE_MISSING`. Stage 2, tie-break, ZERO
  invariant, Candidate Generator, Predictor, Safe-Set, Fallback, trace, QPS,
  and SLO are unchanged. New algorithm identity
  `two_stage_zero_relative_tbt_prefill_service_rate_v2b`; `tbt_delta_seconds`
  is `legacy_inactive_in_v2b`; Selector Diagnosis is schema v4 with replayable
  per-candidate risk fields, and v1/v2/v3 replay branches remain supported.
- N is grid-configurable via the development-only env `DPP_STAGE1_MAX_DELTA_N`
  (non-negative integer; rejected outside `development_nonformal` at both the
  runner and server gates) and is recorded in the audit, diagnosis, iteration
  log, and aggregate.
- Remote model-free checks pass: 203/203 unit tests (including the new ΔN
  admission/boundary/reference/env tests and schema-v4 diagnosis tests); the
  frozen schema-v3 JSONL (SHA-256 `16bd5ca4…`, 4,573 frames) replays with
  zero mismatches under the new code; the V2-A replay script reproduces its
  frozen output SHA-256 `db6647b2…`.
- Added `scripts/stage1_delta_n_grid_campaign.sh`,
  `benchmarks/run_stage1_delta_n_grid.py`, and
  `scripts/analyze_stage1_delta_n_grid.py` for the smoke-gated N grid over
  the staged n=150 QPS 0.25 seed 1001 trace (SHA-256 `203e7ed4…`).
- Smoke gate passed under campaign
  `stage1_delta_n_grid_qps0p25_n150_seed1001_v1`: Stock 20/20 requests with
  0 failures (TTFT p50/p95 306/1,483 ms) and DPP N=0 20/20 with 0 failures
  (TTFT p50/p95 328/2,240 ms); the DPP smoke manifest has
  `selector_diagnosis_valid: true` with the recorded `DPP_STAGE1_MAX_DELTA_N=0`.
  This is development/non-formal evidence and advances no gate.

## 2026-08-30 — Stage-1 V2-B delta-N grid campaign

- Campaign `stage1_delta_n_grid_qps0p25_n150_seed1001_v1` completed: Stock plus
  DPP N ∈ {0,2,4,8,16}, one n=150 run each on the staged QPS 0.25 seed 1001
  trace (SHA-256 `203e7ed4…`). All six runs are `valid`: 150/150 requests,
  zero failures, and every DPP run has `selector_diagnosis_valid: true` with
  zero replay mismatches (schema v4).
- Selection mechanism (selection_histogram ZERO/STOCK, backlog frames): N=0
  selects ZERO in 1,490 of 1,830 backlog frames; N=2 drops this to 398 of
  625; N≥4 selects ZERO once (1,611→~1 frame) with STOCK winning 4,455-4,811
  frames. The ZERO-relative guard releases the min-slack starvation for
  N ≥ 2 and is Stock-equivalent for N ≥ 4.
- TTFT mean-p50/p95/p99 (ms): Stock 544/1,927/2,767; N=0 44,956/136,385/
  140,736 (severe starvation, as the V2-A replay predicted); N=2 603/14,797/
  30,956 (p50 recovered, heavy tail); N=4 550/1,880/2,908; N=8 541/1,800/
  2,715; N=16 535/1,845/2,727. TBT p50/p95/p99 (ms): Stock 184/203/649;
  N=0 173/226/265 (best TBT tail, bought with TTFT starvation); N=2 184/206/
  650; N=4 183/206/663; N=8 185/201/667; N=16 185/207/664.
- Throughput (completion rps): Stock 0.1916; N=0 0.1685; N=2 0.1838; N=4
  0.1806; N=8 0.1870; N=16 0.1905. Natural-EOS SLO-goodput rps
  (TTFT ≤ 2s ∧ every TBT ≤ 0.25s ∧ stop): Stock 0.0089 (7/150); N=0 0.0135
  (12/150); N=2 0.0172 (14/150); N=4 0.0084 (7/150); N=8 0.0087 (7/150);
  N=16 0.0089 (7/150). E2E mean (ms): Stock 160,056; N=0 193,925; N=2
  162,192; N=4 161,624; N=8 162,466; N=16 162,286.
- Interpretation: N=0 protects TBT at catastrophic TTFT cost and is unusable.
  N=2 is a transition point with the best observed SLO-goodput (14/150) but a
  heavy TTFT tail. N=4/8/16 are Stock-equivalent, with N=8/16 showing slightly
  better TTFT p50/p95/p99 than Stock at 2-4% lower completion throughput.
  Success counts (7-14 per 150) are single-seed development evidence with no
  cross-seed confidence interval; the processed summary is retained at
  `results/processed/qwen3_14b_dgx_spark/stage1_delta_n_grid_qps0p25_n150_seed1001_v1/summary.json`
  (SHA-256 `c1ab2b04…`). No gate advance and no formal claim.

## 2026-08-29 — Stage-1 V2-A ZERO-relative ΔN_TBT=0 offline counterfactual replay

- Added the user-provided V2-A plan as
  `docs/Stage1-V2A-ZERO-Relative-TBT-Replay.md` and its companion script as
  `scripts/replay_stage1_delta_n0.py`. The script is read-only over Selector
  Diagnosis schema v3 and reproduces Stage 1 risk with conservative duration,
  `ConsequenceEstimator._misses` semantics (`>` when the request is served,
  `>=` otherwise), ZERO-relative ΔN/ΔL, and the frozen V1 Stage-2 service-rate
  ranking with its tie-break order.
- Replayed the frozen selector-diagnosis JSONL
  (`selector_diagnosis.jsonl`, SHA-256
  `16bd5ca4fe921a4b8a98ab3cbf10b321ce7b086263539d70aca455fa305162d5`,
  77,175,969 bytes, 4,573 frames; 4,572 OK, 1 `NO_SAFE_CANDIDATES`). Output is
  `results/processed/qwen3_14b_dgx_spark/stage1_v2a_delta_n0_replay_seed1001.json`
  (SHA-256 `db6647b25872cf227ea7ab1029beb4f3913f348ba5bd2bf98281535da151b9ed`),
  pulled append-only and checksum-verified.
- Verdict: plan case B. `R_release = 0.0`: none of the 1,527 old-ZERO-only
  backlog frames gained a non-ZERO V2-A winner, and no old-ZERO-only frame had
  a new ΔN=0 eligible candidate with positive service. In the 1,611
  active-TBT+backlog frames the V2-A winner is ZERO in 1,547 (96.0%) versus
  1,527 under the old min-slack Stage 1; V2-A admits fewer non-ZERO backlog
  winners (64 versus 84). Winner ΔL is identically 0 across all 4,572 OK
  frames.
- Stock ΔN histogram: 2,978×0, 1,565×1, 14×2, 13×3, 2×5; every ΔN≥1 Stock
  frame is a backlog frame. Stock remains eligible in 2,978 frames and wins
  2,973 (versus 2,997 under the old Stage 1). 67 winners changed overall.
- ZERO reference resolution: 1,611 `ZERO_TEMPLATE` (all backlog frames with
  active TBT obligations), 1,764 `ZERO_SERVICE_COUNT_MATCH_FALLBACK`
  (non-backlog frames where canonical dedup preserved the materially identical
  zero-service STOCK over ZERO, per plan §3.3 rule 4), 1,197 `NOT_NEEDED`.
- Supplementary read-only pass over the same frozen JSONL: all 1,527
  old-ZERO-only frames have Stock ΔN=1, so a ΔN≤1 gate would newly admit Stock
  in every one of them. In those frames Stock ΔL (added TBT lateness versus the
  ZERO baseline) is min 0.004 s, P50 0.674 s, P90 0.756 s, max 0.794 s — a
  large lateness extension, supporting a V2-B `ΔN≤K ∧ ΔL≤B` design rather than
  a pure `ΔN≤1` gate.
- Recommendation: do not implement V2-A in the online Selector as-is; ΔN=0 is
  as strict as the min-slack filter and does not relieve the Prefill
  starvation. This is counterfactual replay over recorded predictions, not a
  performance benchmark. No online Selector, Candidate Generator, Predictor,
  Safe-Set, or Fallback change, gate advance, or formal claim was made.
