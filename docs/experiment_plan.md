# Qwen3-14B modular DPP Scheduler experiment plan

The active design is
`docs/Qwen3-14B-DGX-Spark-Modular-DPP-Scheduler.md`. This plan turns that
design into scientific gates; it does not freeze values that have not been
measured on the target DGX Spark.

## Current status

The campaign is at **G1** after completing **G0**.

G0 is frozen:

- Qwen3-14B BF16 snapshot identity, acquisition approval, bounded model smoke,
  shared runtime (`gpu_memory_utilization=0.84`, `C_tok=2048`, `C_seq=64`), and
  KV capacity (`30149` usable 16-token blocks) are recorded.
- TTFT/TBT event boundaries and request-level Goodput definitions are frozen in
  `configs/dgx_spark_experiment.yaml`.
- `configs/dgx_spark_experiment.yaml` is `frozen_g0` and executable for G1
  Stock baseline runs.

G1 Stock SLO/load calibration is now frozen:

- TTFT SLO = 2.0s, TBT SLO = 0.25s, based on Stock P95 + margin.
- Candidate test QPS = [0.2, 0.25, 0.3].
- Active length-blind Poisson trace manifest:
  `traces/qwen3_14b/manifest_cap2048_lowqps.json`
  (200 requests per trace, seed 1001, client cap 2048).

The validated raw iteration dataset is assembled by batch kind under
`results/processed/qwen3_14b_dgx_spark/predictor_iteration_dataset_v1/`.
`C_tok/C_seq`, the Predictor artifact, and the integration Candidate,
Safe-Set, and DPP values are versioned. The remaining formal-gate inputs are a
Recovery-age rule and measured replacements for the explicitly
integration-only G4/DPP values if they are to support formal benchmark claims.

The active non-formal implementation uses fixed Prefill fractions plus a
Stock-like BatchPlan and the two-stage TBT-constrained TTFT Selector. The
unexecuted nine-run TTFT weight grid is retired without selecting a weight.
The temporary TBT allowance is `delta_D=0.020s`; it is user-directed,
non-formal, and does not complete a gate or establish a Scheduler claim.

## Gate sequence

| Gate | Work | Required evidence |
| --- | --- | --- |
| G0 | Freeze environment, model, runtime, source, and trace identity | exact manifests, remote smoke, startup log, final SchedulerConfig/KV facts |
| G1 | Stock natural-EOS baseline and event telemetry | verified TTFT/TBT event semantics, frozen SLO/Goodput definitions, resource/load calibration |
| G2 | Contracts, Snapshot, exact-plan Adapter, Candidate Generator | immutable same-hash structures; at most 12 deterministic plans; selected plan equals actual execution |
| G3 | Same-configuration profiling and offline model selection | held-out expected error, conservative P95 coverage/underprediction, support/OOD report, CPU overhead |
| G4 | Safe-Set, Rolling KV, and Fallback | physical/Predictor feasibility with risk metadata retained for every candidate, deterministic fallback/liveness-preemption/empty-idle audit |
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
6. Define prompts and a finite client termination guard. Keep that guard out of
   Scheduler contracts, Predictor inputs/labels, and decisions; retain and
   stratify both `stop` and `length` terminal reasons.
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

Implement public immutable contracts, `snapshot_hash` validation, and atomic
exact-plan execution. The active Candidate Generator is defined in
`docs/Candidate-Generator-V3.md`: it emits ZERO, fixed P10 through P100 plans,
and a Stock-like plan, with at most 12 canonically deduplicated candidates.
Fixed-fraction plans retain all active Decode and bind Prefill running-first,
then waiting FCFS. Predictor and Safe-Set evaluate complete plans only after
generation. Request IDs and token counts may not be reselected after the
decision. The retained Predictor-inversion and Horizon/Knee artifacts are
historical and are not active Candidate inputs.

### G3: Predictor

Collect same-model/same-runtime rows containing each executed BatchPlan's
per-request phase, current context, scheduled tokens, and actual duration.
Derive features without using the held-out test split, train the user-selected
three-scenario Ridge baseline, and select each scenario's residual-window size
only from chronological training OOF replay. Freeze the model, feature schema,
support domain, online-calibration policy, and artifact after remote shadow
validation.

The first single-seed 500-request shadow validation completed without request
failures, but failed the predeclared timing-compatibility guard. It is retained
as diagnostic evidence; Predictor effectiveness and G3 remain unvalidated.
The replacement 200-request validation completed without request failures and
matched the locked-vLLM official timing. Online calibration did not meet every
effectiveness criterion: Decode-only and Mixed missed 95% conservative
coverage, while Prefill-only slightly worsened expected MAE. This negative
result is retained; the Predictor is accepted for modular integration.
The later segmented Mixed candidate also missed its predeclared 95%
conservative-coverage criterion. Its use with the explicit `kappa_ood=0`
development default is therefore diagnostic only and does not supersede that
negative result.

### G4: Safety

Apply token, sequence, current KV, Rolling-KV, and Predictor-support hard
filters. Use conservative duration for SLO-risk metadata. Every
physical/Predictor-feasible candidate reaches DPP; the legacy Top-K field is
inactive. Keep the independent Fallback and liveness/preemption/empty-Idle
paths deterministic and audited.

For the current development-only diagnostic, the active config temporarily
uses `H=0` Decode iterations and `R0=0` KV blocks. This disables forward
Rolling-KV and fixed reserve margins while retaining the current projected-KV
hard check. The setting is user-directed, does not supersede the earlier
conservative integration rationale, does not make G4 complete, and is
ineligible for formal DPP results. The legacy `Top-K=3` field is retained only
for schema compatibility and is inactive.
Fallback integration provisionally uses a 6-token minimum Prefill chunk, equal
to the frozen Prefill-only Predictor support-domain lower bound for total
scheduled Prefill tokens. A completion chunk may be smaller, but it must still
pass Predictor support; otherwise a non-empty workload uses the liveness
escape or Preemption path, while Idle is reserved for an empty workload.

The integration implementation now wires the resource filters, risk metadata,
Controller-owned Fallback, and explicit liveness/Preemption-required/empty-Idle
result into the live modular Scheduler. The relevant remote unit tests pass, but the
provisional parameters above still keep G4 ineligible for formal claims.

### G5–G6: Control and integration

Freeze `Psi_k`, its units/ranges, `Q^P/Z^F/Z^D` ledger, event boundaries, and
tie-break. Integrate Selector, Adapter, Observer, and actual-only updates. Every
TTFT/TBT obligation is created and settled exactly once; natural EOS releases
KV and creates no next obligation.

The live obligation ledger is implemented before the DPP score: request
arrival creates TTFT, an actual locked-vLLM `EngineCoreOutput` token settles
TTFT/TBT, a nonterminal token creates the next TBT deadline, and a terminal
event creates none. Snapshot now carries these obligations and Recovery state.
The former combined Prefill/Decode drift Selector is superseded by the
two-stage design: active TBT obligation slack filters candidate duration, then
fixed-reference TTFT drift rate selects the winner. Decode service debt remains
available for actual-feedback diagnostics but no longer participates in
selection. The temporary 20 ms allowance and replayable Diagnosis remain
development-only; remote model-free tests and a real-model integration run are
still required before G5/G6 can be claimed complete.

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
