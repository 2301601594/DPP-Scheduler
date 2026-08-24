# AGENTS.md — Qwen3-14B modular DPP scheduler research

## 1. Scope and authoritative design

This file applies to the whole repository. A nearer `AGENTS.md` or
`AGENTS.override.md` may add directory-specific rules. **Must** means that a
task cannot be declared complete if the rule is unmet; **should** is the
default when relevant. Requirements do not imply unrelated review,
documentation, remote work, or reporting. Use process proportional to the
requested change.

The active project is the first-version modular DPP Scheduler defined in
`docs/Qwen3-14B-DGX-Spark-Modular-DPP-Scheduler.md`. Its fixed scope is:

- `Qwen3-14B` on one DGX Spark and one GPU;
- BF16 inference with vLLM V1 continuous batching and chunked prefill;
- Prefix Caching and Speculative Decoding disabled;
- normal EOS handling with no output length exposed to the Scheduler, and at
  most one generated token per Decode request in each iteration;
- one SLO class; and
- no PD separation, multi-GPU/model parallelism, LoRA, quantization, online
  predictor training, or output-quality optimization in version 1.

The Scheduler must never store, infer, or consume `remaining_output_tokens`, a
fixed expected output length, or the eventual EOS position. A finite client
termination guard is allowed but must not become Scheduler state, a Predictor
feature/label, or a decision input. Preserve and stratify both `stop` and
`length`; never filter a guard-triggered request merely to manufacture an
all-EOS workload.

Negative and null results must be retained and reported. Never change prompts,
arrival times, SLOs, baselines, filtering, seeds, or failed-run handling to make
DPP appear successful. Current Qwen3-14B raw evidence remains append-only. The
obsolete 5070 campaign's version-controlled files were deleted by explicit
user request. The separately approved 5070 raw results, BurstGPT input, and
invalid Qwen3 scan/validation trace drafts were also deleted. Remaining ignored
historical residue, including old model caches, is intentionally retained and
non-active; no historical campaign configuration, trace, SLO, result, or
conclusion is an active input unless explicitly revalidated and frozen.

When instructions conflict, use this precedence:

1. the user's current explicit request;
2. the frozen Qwen3-14B DGX configuration, model/trace manifests, and predictor
   artifact;
3. the frozen modular Scheduler interface and math specification;
4. `docs/Qwen3-14B-DGX-Spark-Modular-DPP-Scheduler.md`,
   `docs/experiment_plan.md`, and `docs/decisions.md`;
5. this file; and
6. external papers or upstream documentation.

Every conclusion-affecting parameter must be version controlled. A value that
has not been measured or reviewed must be `null`, `pending`, or
`provisional`; never fill it from an older campaign or from an undocumented
default.

## 2. Fixed development and execution architecture

- Local WSL repository: `/home/dongj/projects/LLM`.
- Local role: source editing, review, Git inspection, text processing, and
  dependency-free static/syntax checks only.
- Remote SSH alias: `dgx-spark` (`dongj@10.16.66.191`).
- Remote repository: `/home/dongj/LLM` (`~/LLM`).
- Remote role: **all project execution**, including project Python commands,
  tests, environment creation, dependency installation, vLLM import/build,
  model smoke tests, profiling, and benchmarks.

When a task needs remote execution, source synchronization, or result
retrieval, read the relevant part of `docs/remote_dgx_workflow.md` and use
`scripts/remote_dgx.sh`:

```bash
./scripts/remote_dgx.sh check
./scripts/remote_dgx.sh dry-run
./scripts/remote_dgx.sh push
./scripts/remote_dgx.sh verify
./scripts/remote_dgx.sh run <command> [args...]
./scripts/remote_dgx.sh pull-results
```

Before source synchronization, run `check` and `dry-run`, inspect the exact
target `~/LLM`, then `push` and `verify`. For several related remote commands
against an already verified mirror, one successful preflight is enough; do not
repeat `check`, `dry-run`, or `push` before every command. If no source changed,
do not push merely as ceremony. Local-only documentation, Git inspection, and
dependency-free static checks do not require remote synchronization. Never
synchronize while a benchmark is running. Never use `--delete-excluded` or
apply delete semantics to `$HOME`, `~/`, or another broad directory. SSH keys
and tokens stay outside the repository.

Do not copy the local `.venv`, `.uv-python`, caches, model cache, or compiled
vLLM artifacts to ARM64. Build or verify the environment remotely. A model
snapshot is a separate, reviewed bulk transfer and is never part of routine
source synchronization.

## 3. Shared-host safety

DGX Spark is a multi-user research workstation. Use only the `dongj` account
and paths owned by it unless the user and operator explicitly authorize a
named shared path.

Agents must not:

- use or request an administrator/shared account, `sudo`, privilege
  escalation, account creation, or changes under `/etc`, `/usr`, or `/opt`;
- change SSH, firewall, system services, global environment variables, drivers,
  CUDA, DGX OS, shared software, or another user's environment;
- configure VPNs, tunnels, unapproved proxies, or global networking; disable
  TLS verification; trust an unknown certificate; or bypass security controls;
- inspect, modify, copy, move, delete, or terminate another user's files,
  credentials, processes, jobs, or results;
- expose passwords, private keys, API keys, tokens, or credentials in the
  repository, logs, chat, or command output; or
- start an unbounded job, abandoned session, idle GPU-holding process, or
  storage growth without a declared bound.

Large packages, datasets, model weights, images, or bulk transfers require a
trusted source, immutable revision/checksum, size estimate, `df` check, and
user confirmation that the method is operator-approved. The DGX is not the
only durable copy of source or raw results.

Before a long or resource-intensive job, run read-only GPU, process, memory,
and disk checks; state its name, expected duration, GPU/memory/disk/network
needs, and output path; and obtain confirmation that the group was notified
and resource use was approved. If another user's workload is active or
ownership is unclear, stop and ask the user/operator. Never terminate it.

Long jobs use a bounded, named `tmux` session or another approved
disconnect-safe mechanism. Stop the job and clean its session when complete.
If SSH, permission, package, GPU, disk, service, or networking failures require
a system change, preserve the error and ask the user to contact operator
李文轩. Do not scan the network or work around the control.

## 4. Active configuration and current stage

`configs/dgx_spark_experiment.yaml` is the only active experiment configuration.
It is provisional until G0 is reviewed and frozen. The exact Qwen3-14B model
revision/snapshot, shared runtime limits, and KV capacity have provisional G0
captures. Stock iteration telemetry, the reviewed trace manifest, SLOs,
predictor, and DPP parameters remain pending.

Existing DGX environment manifests may be reused only for observed host,
CUDA, Python, PyTorch, vLLM-installation, and source facts that are reverified.
Model-specific startup behavior, KV capacity, Scheduler defaults, trace
tokenization, SLOs, and performance measurements must be recreated for
Qwen3-14B BF16.

The model and Scheduler redesign reopen the project at **G0**. No later gate
is complete merely because an older campaign reached it. Obsolete campaign
artifacts are not accepted as inputs and must never be recreated or combined
with Qwen3-14B aggregates.

## 5. Task workflow

For routine work, inspect `git status --short` and the directly relevant files,
make the requested change, and run the smallest useful check. Read the active
config/design or identify a G0–G7 gate only when experiment behavior, Scheduler
contracts, runners, manifests, results, readiness, or claims are affected.
Inspect locked vLLM source only for version-specific behavior. Use an explicit
plan or repository-wide audit only when the task is cross-cutting, ambiguous,
risky, resource-intensive, or explicitly asks for one.

Reuse existing infrastructure. Preview long runs with their matrix, resource
needs, and output path. Do not produce run previews, “not run” notices, or
experiment-log/decision/design updates for routine work unless they carry
material information.

Keep changes small and reviewable. Preserve unrelated user edits. Do not
commit, publish, open a PR, change the vLLM commit, install dependencies,
download a model, or start a resource-intensive job unless the user explicitly
authorizes that action. No fabricated measurements.

## 6. Research and implementation gates

The G0–G7 names below are the repository's operational mapping of the seven
implementation steps in the authoritative design document.

| Gate | Scope | Exit condition |
| --- | --- | --- |
| G0 | Freeze DGX environment, exact Qwen3-14B revision, vLLM commit, runtime flags, model/trace manifests, and stock iteration-log semantics | Configuration and commands are reconstructible; model smoke and captured `SchedulerConfig`/KV facts pass remotely |
| G1 | Stock natural-EOS correctness, iteration telemetry, load baseline, and SLO/resource calibration | TTFT/TBT obligation semantics and actual request-level Goodput are defined; required physical and SLO inputs are frozen before DPP results |
| G2 | Public contracts, immutable Snapshot, exact-plan Adapter, Candidate Generator, and deterministic temporary selector | Selected and executed request/token sets agree; candidate construction is pure and deterministic |
| G3 | Same-configuration DGX profiling and offline-selected Predictor | Independent accuracy, conservative coverage, support domain, residual calibration, and inference overhead are reported |
| G4 | Safe-Set, Rolling KV Guard, SLO-risk ranking, Fallback, Preemption/Idle paths | Hard constraints never pass an infeasible plan; every rejection/fallback is deterministic and auditable |
| G5 | Freeze DPP equation, SLO ledger, obligation boundaries, parameter units/ranges, and fallback ownership | All symbols, update times, success/miss semantics, tie-breaks, and unresolved interface names are unambiguous |
| G6 | Integrate Selector, Adapter, Observer, and actual-feedback state updates | Remote unit/integration tests and exact-execution/one-settlement invariants pass |
| G7 | Frozen Stock-versus-DPP experiments, ablations, overhead, and artifact rebuild | Identical model/requests/runtime settings are used; raw-to-table rebuild succeeds and negative results are retained |

Gates govern research readiness and claims, not routine maintenance. Mention or
advance them only when affected. Do not skip gates for research work;
unsupported early scaffolding remains explicitly provisional.

## 7. DGX and model contract

Observed environment facts currently include host `convergence`, ARM64/aarch64,
NVIDIA GB10 compute capability 12.1, driver 580.159.03, CUDA toolkit 13.0,
system Python 3.12.3, and a project-local vLLM environment. The installed vLLM
source is currently commit `83ad767eed3be3ee7f2df63be693bfaca5c7c922`;
reverify and record its clean/dirty state before model execution, profiling,
vLLM-internal integration work, or a gate claim. Do not silently upgrade it.

The G0 manifest must record:

- the exact Qwen3-14B repository, immutable revision, snapshot hash/path,
  tokenizer revision, license/source, and acquisition command;
- DGX OS, kernel, CPU/unified memory, driver, CUDA toolkit/runtime, PyTorch,
  vLLM package/build/source commit, Python environment, and relevant variables;
- BF16 model/KV dtype, V1 engine, single-GPU execution, chunked prefill enabled,
  Prefix Caching and Speculative Decoding disabled, and quantization off;
- `max_model_len`, total token budget `C_tok`, sequence budget `C_seq`, KV
  block size/capacity, memory utilization, startup log, and final
  `SchedulerConfig`;
- natural-EOS generation/request settings and finish-reason behavior; and
- root/vLLM Git commits and dirty states, commands, timestamps, and manifest
  hashes.

Do not inherit a context length, sequence limit, token budget, KV capacity,
generation cap, SLO, or model revision from any other model or platform. A
request API may require a finite safety ceiling, but it must be declared as a
client guard, large enough not to define the workload's normal completion, and
invisible to Scheduler decisions.

Any real server launcher must generate its model path, environment, and full
vLLM argv from the reviewed frozen config and verify the config hash. It must
reject provisional configs, Hub IDs/implicit downloads, duplicate or
caller-overridden conclusion parameters, non-owned/lexically escaping paths,
and missing explicit disables. A free-form launcher is dry-run/help only.

## 8. Modular interfaces and ownership

The decision path is fixed:

```text
vLLM state
  -> StateSnapshot
  -> CandidateGenerator.generate(snapshot) -> BatchPlan[]
  -> DurationPredictor.predict(snapshot, plans) -> Prediction[]
  -> SafeSet.filter(snapshot, plans, predictions) -> SafeSetResult
  -> DPPSelector.select(snapshot, control_state, safe_candidates) -> Decision
  -> vLLM Adapter atomically executes the complete BatchPlan
  -> Observer updates state from actual results only
```

`StateSnapshot`, `BatchPlan`, `Prediction`, and `ControlState` are public,
immutable decision-round contracts. Every plan, prediction, consequence, and
decision must be bound to the same `snapshot_hash`; mismatches fail closed.
Contract fields, units, serialization, numeric ranges, and version fields must
be explicit.

Only `dpp_scheduler/vllm_adapter.py` may import vLLM internal types. Other
modules depend only on public project contracts. Version-specific access and
the commit-specific definition of TTFT completion live in that Adapter.
Finishing the last Prefill chunk must not be assumed to equal producing and
returning the first token unless the locked source proves that boundary.

The Adapter must atomically submit the exact selected `BatchPlan`. It must not
submit only a cap and let the stock Scheduler reselect requests. The selected
Prefill/Decode request IDs, token counts, total sequences, and actual execution
must agree; any mismatch is a counted error and conservative failure path.

Every public decision-round contract carries `snapshot_hash`; the canonical
candidate-set name is `safe_candidates`; and Controller owns Fallback
construction. Before G5 is frozen, resolve the exact Adapter event that settles
each TTFT/TBT obligation.

## 9. Candidate Generator contract

The action is a complete `BatchPlan`, not a scalar Prefill cap. Seed Prefill
breakpoints are `ZERO`, `FINISH`, `KNEE`, and `BINDABLE_MAX`. `FINISH` is
generated only when the highest-priority bindable Prefill request can complete;
`KNEE` is frozen from same-configuration profiling; and `BINDABLE_MAX` is only
a native token/backlog bound, never an SLO-safe bound. Decode profiles are:

- `MANDATORY`: Recovery and otherwise mandatory protected Decode only;
- `CRITICAL`: Mandatory plus live Decode whose current TBT slack is no larger
  than a frozen, snapshot-only Critical Horizon; and
- `ALL`: as many Decode requests as resources permit.

Generate at most 12 profile/cap combinations before canonical deduplication.
Never enumerate arbitrary request subsets.

Critical Horizon and Prefill knee values that have not been measured remain
`null`/provisional and cannot support a formal benchmark. Candidate generation
must not import or call Predictor, Safe-Set, or Selector. If the horizon is
unset, `CRITICAL` conservatively collapses to `MANDATORY`; if the knee is unset,
the `KNEE` seed is omitted. A normal partial Prefill chunk is either zero or at
least the configured minimum; a smaller chunk is allowed only when it completes
the request.

Decode order is: oldest Recovery request whose age threshold is reached;
non-violated requests by TBT deadline using EDF; then remaining Recovery by
first-miss time. Prefill order is: hard-protected TTFT by deadline; partially
prefilled requests; then all others by FCFS. All ties need stable documented
keys.

Every plan must satisfy `total_prefill_tokens + total_decode_tokens <= C_tok`
and `total_sequences <= C_seq`. Project current and added KV blocks as a pure,
side-effect-free calculation using current slots and block size. Candidate
construction must never allocate, free, or mutate the real block manager.

## 10. Predictor contract

Version 1 uses an offline-trained model to predict current-plan iteration
duration. The model family is selected only after comparing multiple suitable
regressors on the development data; it is not fixed by the design. Profiling
must record one row per actually executed
`BatchPlan`: run/plan/snapshot identity, actual iteration duration, and for each
selected request its request ID, Prefill/Decode phase, pre-iteration computed
or KV-context length, and scheduled token count. Identity fields are for joins
and audit only. This raw schema is not the model feature schema and must not
contain remaining output length or future EOS information.

Offline training derives and compares candidate aggregate or nonlinear features
from those raw fields without using the held-out test split. Model-family,
feature, and hyperparameter selection use only training/development data. The
final model family, exact hyperparameters, feature schema, transformations,
support domain, exact seed, and artifact are frozen only after validation.

Calibrate residuals with separate bounded Prefill-only, Decode-only, and Mixed
online windows. `expected_duration` is the frozen model's prediction plus the
window mean residual. `conservative_duration` adds the centered window P95.
The Selector uses expected duration; Safe-Set uses conservative duration. A
window with fewer than 32 completed in-support samples falls back to the same
batch kind's training OOF residual distribution.

Training data must match Qwen3-14B, DGX Spark, BF16, vLLM commit, runtime flags,
and Scheduler-relevant limits exactly. Version 1 trains model weights offline;
online code may update bounded residual windows but must not modify model
weights. Missing, NaN/Inf, schema or
version mismatch, and inputs outside the frozen support domain set
`in_support=false` and fail normal hard admission. Never extrapolate
optimistically.

Report held-out expected-duration error, conservative coverage and
underprediction, per-region support/coverage, out-of-support rate, and Predictor
CPU overhead. Training rows, split, seed, feature ranges, artifact hash, and
predictor version are reproducible manifest fields; the manifest also records
the raw profiling schema and final feature schema.

## 11. Safe-Set, Rolling KV, and Fallback

Safe-Set hard-filters only physical/Predictor feasibility:

1. token budget;
2. sequence budget;
3. current-frame projected KV capacity;
4. finite-horizon Rolling KV Guard; and
5. Predictor support and validity.

Rolling KV reserves only the next `H` Decode iterations plus `R0` safety blocks
and is recomputed every iteration. It must not reserve a request's maximum or
predicted full output length. KV checks remain pure during evaluation; the
native allocator remains the final execution authority.

SLO risk is not an absolute hard constraint. Using conservative duration,
compute for each resource-feasible plan the predicted new violation count
`N_vio` and total predicted lateness `E_vio`. If any plans have `N_vio == 0`,
only those reach DPP. Otherwise sort by `(N_vio, E_vio, stable_plan_key)` and
send the frozen Top-K to DPP. Do not jump directly to Fallback merely because
all resource-feasible plans carry SLO risk.

Fallback is independent of DPP scoring. With active Decode requests, stop
Prefill and build an EDF Decode-only plan. Without Decode requests, build the
smallest physically feasible Prefill chunk. Fallback still must pass token,
sequence, KV/Rolling-KV, and Predictor-support checks. If it cannot execute,
enter the explicit Preemption-or-Idle path. Record every rejection, fallback,
preemption, and idle reason; never catch it silently.

## 12. DPP Selector, ledger, and actual feedback

`ControlState` contains only Prefill backlog `Q_k^P`, TTFT debt `Z_k^F`, and
TBT debt `Z_k^D`; it does not duplicate the physical Decode queue. For each
SafeCandidate, use the authoritative score:

```text
Psi_k(a) = [
    Q_k^P * mu_k^P(a)
  + Z_k^F * (epsilon^F * S_hat_k^F(a) - (1-epsilon^F) * M_hat_k^F(a))
  + Z_k^D * (epsilon^D * S_hat_k^D(a) - (1-epsilon^D) * M_hat_k^D(a))
  + V * U_hat_k(a)
] / expected_duration_k(a)
```

The score's units, normalization, finite numeric ranges, and zero-duration
handling must be frozen before implementation. Select the maximum score. Ties
choose, in order: fewer predicted misses; larger conservative deadline margin;
smaller stable `plan_id`.

The score's Goodput-oriented service utility is not request-level Goodput.
Actual Goodput is counted only after a terminal request event according to the
frozen request-level success definition. `stop` and safety-guard `length`
terminations remain separate strata; the definition may classify them
differently but may not silently discard either one.

Update state only from actual execution and returned-token events:

```text
Q_(k+1)^P = [Q_k^P - mu_k^(P,actual)]+ + A_k^P
Z_(k+1)^F = [Z_k^F + (1-epsilon^F) * M_k^F - epsilon^F * S_k^F]+
Z_(k+1)^D = [Z_k^D + (1-epsilon^D) * M_k^D - epsilon^D * S_k^D]+
```

Each TTFT/TBT obligation is created and settled exactly once at a tested event
boundary. A returned non-EOS token creates the next TBT obligation from its
real return time. EOS completes the request, releases KV through native
semantics, and creates no next obligation. State updates must be deterministic,
non-negative, replayable from the decision/observation log, and immune to
duplicate callbacks.

Parameters that must be measured/reviewed and frozen include `C_tok`, `C_seq`,
Critical Horizon, Prefill knee, `H`, `R0`, Recovery age, hard-TTFT protection,
Top-K, minimum Prefill chunk, TTFT/TBT SLOs, `epsilon^F`, `epsilon^D`, `V`,
residual windows, Predictor model family/artifact/version, and every stable tie
key.

## 13. Implementation and remote tests

Use the modular layout from the design document:

```text
dpp_scheduler/
  contracts.py
  controller.py
  candidate_generator.py
  predictor.py
  consequence_estimator.py
  safe_set.py
  dpp_selector.py
  fallback.py
  state_store.py
  observer.py
  vllm_adapter.py
```

Keep contracts, ordering, deduplication, resource projection, risk estimation,
scoring, tie-breaks, ledger transitions, and fallback as deterministic testable
functions. Do not modify stock Scheduler code to create the baseline.
Instrumentation defaults off in performance runs; decision logging is bounded
and asynchronous.

Remote tests must cover at least:

- immutable contracts, serialization/versioning, and hash mismatches;
- empty, Prefill-only, Decode-only, Mixed, Recovery, and partial-Prefill states;
- all three Decode profiles, four caps, canonical deduplication, stable order,
  and the 12-candidate bound;
- exact token/sequence limits, current and Rolling KV boundaries, no evaluator
  side effects, and native allocation failure;
- Predictor expected/conservative calculations, same-kind OOF cold start, missing/
  NaN/Inf/schema/version/OOD rejection, and zero/non-positive duration;
- zero-violation Safe-Set, all-risk Top-K ordering, every Fallback branch,
  Preemption/Idle, and audited reasons;
- every DPP term, deterministic ties, numeric bounds, and empty candidate input;
- selected-versus-executed plan identity, completion, KV release, and
  commit-specific TTFT completion;
- obligation creation/one-time settlement, duplicate-event rejection, natural
  EOS, and actual-feedback debt non-negativity; and
- pluggability of Predictor, Safe-Set, and Selector behind the public contracts.

Run the smallest relevant test. Use the broader remote suite for cross-module
or shared execution changes, gate claims, and explicit requests—not every
narrow change. Documentation-only changes need only local checks. A
Stock/pass-through diagnostic may validate Adapter observations but cannot
replace exact-plan invariants.

## 14. Experiments, statistics, and reproducibility

Stock and DPP comparisons must use the identical frozen model revision,
tokenizer, prompt/request set, order, arrival times, seeds, API generation
settings, vLLM commit, and all non-Scheduler runtime settings. Policies compare
at the same absolute offered load. Never use a Scheduler-specific capacity to
renormalize load.

The new length-blind natural-completion workload and trace manifest must be
created and frozen for Qwen3-14B. Record source, prompt text or token IDs as appropriate, tokenizer
revision, filtering, request parameters, arrival process, sampling seed, and
SHA256. Do not reuse tokenized or fixed-output traces from another model.

Freeze TTFT/TBT SLOs, the obligation-to-request success definition, actual
request-level Goodput, load points, development/formal run sizes, seed count,
and confidence-interval method from Stock validation before inspecting DPP
test results. Compute tail percentiles and Goodput per seed before aggregating;
never pool requests across seeds to report one tail percentile.

Each run has a unique `run_id`. Raw results are append-only and include command,
resolved config, Git states, model/trace/predictor hashes, seed, arrival target
and achieved rate, warmup/measurement windows, completed/failed/timeout/cancel
counts, finish reasons, per-request events, resource/preemption counters,
iteration composition/duration, Scheduler CPU time, fallback/rejection counts,
and parser schema version. Preserve invalid runs and apply only predeclared
exclusion criteria.

Performance runs disable detailed per-iteration instrumentation. Profile runs
use the same frozen requests in a separate namespace and cannot substitute for
performance measurements. Pull remote raw outputs with
`scripts/remote_dgx.sh pull-results`; rebuild processed tables and charts from
raw data on the DGX, then pull the namespaced derived outputs. Charts state
units, seed count, and uncertainty.

## 15. Completion and reporting

Before formal DPP comparison, require: a frozen G0 environment/model; explained
Stock variability; frozen natural-EOS SLO/Goodput semantics; a validated Predictor
support region and conservative underprediction behavior; exact-plan execution;
one-time obligation settlement; safe fallback behavior; and measured Scheduler
CPU overhead. These are scientific gates, not assumed outcomes.

Routine reports state the outcome, relevant checks, and important limitations.
Do not automatically repeat full file lists, unchanged facts, gates, commands,
or work that was not run.

Use expanded, reproducible reporting for formal experiments, gate transitions,
destructive cleanup, invalid runs, and frozen scientific inputs, or when the
user asks for it.

Separate observed facts, data-supported inferences, and unverified hypotheses
when an analysis actually mixes them; simple factual answers do not need those
labels. Use the locked vLLM source for version-specific interfaces. DPP papers
provide design ideas only; do not claim their proofs or stability results
transfer to mixed continuous batching.
