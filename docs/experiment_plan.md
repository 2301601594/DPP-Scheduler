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

Implement public immutable contracts and `snapshot_hash` validation. Generate
`{ZERO,FINISH,KNEE,BINDABLE_MAX}` × `{MANDATORY,CRITICAL,ALL}`, canonically
deduplicate, and use a deterministic temporary selector. `CRITICAL` uses only
Snapshot slack and a frozen Critical Horizon; Predictor and Safe-Set run only
after complete BatchPlans exist. Adapter execution is atomic: request IDs and
token counts may not be reselected after the decision.

Freeze Candidate parameters with
`benchmarks/freeze_candidate_parameters.py`: Horizon uses only odd-seed Stock
Decode-only development rows and official iteration durations; knee uses the
independent `candidate_knee_profile_isolated_v2` campaign. Each target cell is
prepared in isolation, executed as one Exact BatchPlan, timed at the official
iteration boundary, then aborted and required to restore an empty queue and
the baseline free-KV count before the next cell. The frozen sequence budget
makes 64-Decode mixed cells infeasible, so the largest common mixed count is
48. Likewise, the 2048-token Prefill cap is sampled only with Decode=0 because
adding Decode work would exceed the frozen 2048-token iteration budget. Only
Decode=0 rows select the base knee; mixed rows validate sensitivity.
The measured freeze artifact is append-only. Any sparse Horizon bucket, knee
cell with fewer than 4/5 exact realizations, cleanup failure, or failed knee
rule leaves that artifact ineligible. The retained v1 artifact therefore
remains negative evidence. At the user's explicit direction, no replacement
Knee profiling is run: Scheduler integration instead uses the versioned
`candidate_parameter_integration_freeze_v1` artifact with measured
`critical_horizon_seconds=0.220` and the existing `knee_tokens=768`. Runtime
loading is hash/signature checked, but this user-directed integration freeze is
explicitly ineligible for formal DPP benchmark claims.

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

### G4: Safety

Apply token, sequence, current KV, Rolling-KV, and Predictor-support hard
filters. Use conservative duration for SLO-risk metadata. Every
physical/Predictor-feasible candidate reaches DPP; the legacy Top-K field is
inactive. Keep the independent Fallback and liveness/preemption/empty-Idle
paths deterministic and audited.

For end-to-end Scheduler integration only, the active config provisionally
uses `H=8` Decode iterations and `R0=64` KV blocks. The legacy `Top-K=3` field
is retained only for schema compatibility and is inactive. These are
design-derived scaffolding values: 16-token KV blocks and at most one Decode
token per request make eight iterations half a block period, while 64 reserve
blocks provide one extra block per maximum active sequence and consume about
0.21% of the observed 30,149-block capacity. They are not a measured G4
freeze, do not make G4 complete, and are ineligible for formal DPP results.
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
The normalized DPP Selector and actual-feedback `Q^P/Z^F/Z^D` updates are now
implemented and wired into the live modular Scheduler. For integration,
`epsilon^F=epsilon^D=0.05` is frozen from the retained Stock miss ratios,
token terms are normalized by `C_tok=2048`, obligation terms by `C_seq=64`,
and `weight_v=0.0` disables the obligation-level utility during this integration
repair. The freeze is versioned and runtime hash-checked but explicitly ineligible
for a formal benchmark parameter-optimality claim. Remote model-free
tests establish the deterministic score, ties, one-time feedback, and live
factory wiring; a real model integration run remains required before G6 can be
claimed complete.

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
