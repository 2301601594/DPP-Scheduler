# Qwen3-14B modular DPP Scheduler experiment log

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
