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
