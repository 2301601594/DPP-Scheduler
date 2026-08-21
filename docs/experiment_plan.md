# Qwen3-14B modular DPP Scheduler experiment plan

The active design is
`docs/Qwen3-14B-DGX-Spark-Modular-DPP-Scheduler.md`. This plan turns that
design into scientific gates; it does not freeze values that have not been
measured on the target DGX Spark.

## Current status

The campaign is at **G0**.

Observed DGX software and hardware facts can be reverified from the existing
environment manifests. The Qwen3-14B BF16 snapshot is present on the DGX and
its identity, per-file hashes, source, and acquisition command are recorded in
`configs/qwen3_14b_snapshot_manifest.json` (content-identical to HuggingFace
`Qwen/Qwen3-14B` main `40c069824f4251a91eefaf281ebe4c544efd3e18`). The
following are not yet frozen:

- operator-approval confirmation for the model acquisition and the Qwen3-14B
  smoke;
- model-specific startup parameters, `SchedulerConfig`, and KV capacity;
- natural-EOS request/trace manifest and request-level Goodput definition;
- TTFT/TBT SLOs and obligation event boundaries;
- `C_tok`, `C_seq`, `b_s`, `b_m`, `b_l`, `u`, `H`, `R0`, Top-K,
  Recovery rules, `epsilon^F`, `epsilon^D`, and `V`; and
- profiling dataset, Random-Forest artifact, support domain, and residual
  calibration.

`configs/dgx_spark_experiment.yaml` therefore remains provisional and must
not be used for a benchmark.

## Gate sequence

| Gate | Work | Required evidence |
| --- | --- | --- |
| G0 | Freeze environment, model, runtime, source, and trace identity | exact manifests, remote smoke, startup log, final SchedulerConfig/KV facts |
| G1 | Stock natural-EOS baseline and event telemetry | verified TTFT/TBT event semantics, frozen SLO/Goodput definitions, resource/load calibration |
| G2 | Contracts, Snapshot, exact-plan Adapter, Candidate Generator | immutable same-hash structures; at most 12 deterministic plans; selected plan equals actual execution |
| G3 | Same-configuration profiling and shallow RF | held-out expected error, conservative P95 coverage/underprediction, support/OOD report, CPU overhead |
| G4 | Safe-Set, Rolling KV, and Fallback | physical feasibility, zero-violation/all-risk behavior, deterministic fallback/preemption/idle audit |
| G5 | DPP and SLO ledger freeze | complete equation, units, numeric ranges, obligation boundaries, one owner for fallback, deterministic tie-break |
| G6 | Integrated modular Scheduler | remote unit/integration tests, exact execution, one-time settlement, actual-only state updates |
| G7 | Stock-versus-DPP evaluation | identical frozen model/requests/arrival/runtime settings, append-only raw data, reproducible tables and uncertainty |

No gate is skipped because later code can be scaffolded. Early scaffolding
keeps every unmeasured value explicitly provisional.

## G0 execution checklist

1. Reverify the DGX host, disk, installed project environment, root/vLLM commits,
   and source dirty states.
2. Resolve the exact model repository and immutable revision. Record source,
   license, file sizes, and expected storage before any transfer.
3. Obtain user confirmation that the group/operator approved the model
   acquisition method; then acquire or transfer only the reviewed snapshot.
4. Run a bounded Qwen3-14B BF16 smoke with vLLM V1, one GPU, chunked prefill on,
   Prefix Caching and Speculative Decoding off.
5. Capture the startup log, final `SchedulerConfig`, model/KV dtype, block
   size, usable KV blocks, `C_tok`, `C_seq`, and all environment variables.
6. Define natural-EOS prompts and client safety guards without exposing a fixed
   or expected output length to the Scheduler.
7. Freeze config and manifests only after review; update hashes atomically.

Model acquisition and the smoke are separate authorized tasks. Merely syncing
this plan does not authorize either one.

## Implementation milestones

### G1: Stock semantics

Instrument the locked stock Scheduler without changing its decisions. Identify
the exact Adapter-visible events for Prefill service, first-token return,
non-EOS Decode return, EOS completion, KV release, preemption, and failure.
Calibrate SLOs and actual request-level Goodput from Stock validation before
observing DPP test results.

### G2: Exact BatchPlan path

Implement public immutable contracts and `snapshot_hash` validation. Generate
`{0,b_s,b_m,b_l}` × `{MANDATORY,URGENT(u),ALL}`, canonically deduplicate, and
use a deterministic temporary selector. Adapter execution is atomic: request
IDs and token counts may not be reselected after the decision.

### G3: Predictor

Collect same-model/same-runtime iteration rows with only
`B_P,N_P,N_D,K_D,L_D_max`. Train the shallow RF offline, calibrate residuals,
freeze its support domain and artifact hash, and validate expected versus
conservative duration independently.

### G4: Safety

Apply token, sequence, current KV, Rolling-KV, and Predictor-support hard
filters. Use conservative duration for SLO-risk estimation. Prefer zero-new-
violation plans; otherwise send risk-ranked Top-K to DPP. Keep the independent
Fallback and Preemption/Idle paths deterministic and audited.

### G5–G6: Control and integration

Freeze `Psi_k`, its units/ranges, `Q^P/Z^F/Z^D` ledger, event boundaries, and
tie-break. Integrate Selector, Adapter, Observer, and actual-only updates. Every
TTFT/TBT obligation is created and settled exactly once; natural EOS releases
KV and creates no next obligation.

### G7: Evaluation

Compare Stock and modular DPP at identical absolute offered loads using the
same immutable model, prompts, arrival schedule, seeds, generation settings,
and non-Scheduler vLLM parameters. Profile runs and performance runs use
separate namespaces. Aggregate per seed, retain invalid/negative runs, and
rebuild all tables from append-only raw results.

## Remote workflow

All execution is remote:

```bash
./scripts/remote_dgx.sh check
./scripts/remote_dgx.sh dry-run
./scripts/remote_dgx.sh push
./scripts/remote_dgx.sh verify
```

After source verification, short checks use
`./scripts/remote_dgx.sh run ...`. Long work additionally requires the shared-
host preflight, resource approval, a bounded named session, and a unique output
directory required by `AGENTS.md`.
