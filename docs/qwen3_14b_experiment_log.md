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
