# Active research decisions

## 2026-08-27: use fixed Prefill fractions, a Stock plan, and one TTFT weight

- This decision supersedes the 2026-08-26 Predictor-inversion multiplier and
  relative-urgency Candidate behavior. The live Candidate Generator no longer
  calls BudgetResolver. It emits ZERO, P10 through P100, and one Stock-like
  BatchPlan, with running Prefill before waiting FCFS and a maximum of 12
  canonical plans.
- The Stock builder mirrors the supported locked-vLLM running-then-waiting
  selection path without mutating Scheduler state. The same builder powers a
  development-only `forced_stock_plan` mode that bypasses Predictor, Safe-Set,
  Selector, and Fallback but preserves exact execution and actual feedback.
- DPP uses `lambda_ttft * normalized_ttft_drift + normalized_tbt_drift`.
  `lambda_ttft=1` is backward-compatible; reference concurrency remains
  unchanged.
- The planned grid is `{0.125, 0.25, 0.5, 1, 2, 4, 8}` at QPS 0.25, seed 1001,
  and 150 requests. Seven weighted DPP runs and one forced-Stock-plan DPP run
  share one trace and one native Stock baseline, for nine runs total.
- The grid is single-seed, development-only, and non-formal. It does not select
  or freeze a winning weight, update a gate, or establish a Scheduler claim.

## 2026-08-26: permit an uncalibrated Predictor default for one development comparison

- At the user's direction, the active segmented Predictor uses
  `kappa_ood=0` without OOD calibration so the Candidate Generator V3 can be
  exercised immediately against Stock.
- The status is `fixed_default_for_development_nonformal_comparison`, not
  frozen or calibrated. The mode requires an explicit acknowledgement in the
  active config, carries no OOD artifact, and is accepted only when
  `DPP_EXECUTION_SCOPE=development_nonformal`.
- Formal execution remains fail-closed. Results produced in this mode are
  single-seed engineering diagnostics and cannot establish Predictor
  effectiveness, G3/G7 completion, or a formal Scheduler claim.

## 2026-08-26: use relative urgency tertiles in Candidate Generator V3

- `COMPLETION_AWARE` no longer compares the dimensional
  `remaining_tokens / TTFT_slack_seconds` score with fixed `0.5/1.0`
  thresholds. The score remains valid for relative ordering, and the current
  Snapshot is split by descending empirical rank into top, middle, and bottom
  urgency thirds.
- Tier boundaries are `ceil(n/3)` and `ceil(2n/3)`. Equal scores stay together
  in the tier of their first descending rank.
- Within a tier, smaller remaining Prefill work precedes running status, then
  arrival time, ordinal, and request ID. Strong running-first behavior remains
  the responsibility of `CONTINUATION`.
- Candidate diagnostics separately report multiplier-budget diversity,
  canonical-plan diversity, and per-multiplier allocation diversity. Synthetic
  diagnostics use next-token TBT deadlines and an explicit non-measured
  Prefill hold interval; they do not establish real-vLLM workload behavior.
- The complete active contract is indexed in `docs/Candidate-Generator-V3.md`.

## 2026-08-21: adopt the Qwen3-14B modular BatchPlan DPP design

- The authoritative Scheduler design is
  `docs/Qwen3-14B-DGX-Spark-Modular-DPP-Scheduler.md`.
- Version 1 uses Qwen3-14B, BF16, one DGX Spark GPU, vLLM V1 continuous
  batching, chunked prefill, natural EOS, and one SLO class. Prefix Caching and
  Speculative Decoding are disabled.
- The action is an exact, atomic `BatchPlan` containing selected Prefill and
  Decode work. A scalar Prefill cap is only one candidate-template dimension;
  vLLM may not reselect the plan after DPP chooses it.
- Candidate Generator, offline Duration Predictor, Safe-Set, DPP Selector,
  Adapter, and Observer communicate through immutable public contracts tied to
  one `snapshot_hash`.
- Safe-Set hard-filters physical feasibility and Predictor support. SLO risk
  remains auditable candidate metadata; every physically/Predictor-feasible
  candidate reaches DPP without zero-risk or Top-K pruning.
- KV safety uses a finite Rolling horizon and reserve blocks. It never reserves
  or predicts a request's complete output length.
- Fallback is independent of DPP scoring: EDF Decode-only when Decode exists,
  otherwise minimum feasible Prefill, followed by explicit Preemption/Idle if
  hard feasibility still fails.
- DPP maintains only Prefill backlog, TTFT debt, and TBT debt. Observer updates
  them from actual events; every obligation is settled once, and actual
  request-level Goodput is counted only after natural EOS.

## 2026-08-24: preserve feasible actions during integration repair

- The Safe-Set now retains every candidate that passes token, sequence,
  current/Rolling KV, and Predictor-support checks. Predicted violation count,
  lateness, and deadline margin remain attached to each SafeCandidate; the
  former zero-risk and all-risk Top-K admission pruning is disabled so the
  Selector owns the SLO-risk trade-off. The legacy top_k_when_all_risky
  config field remains only for schema compatibility.
- The obligation-level ttft_success + tbt_success service utility remains
  available for audit, but its integration contribution is disabled with
  weight_v=0.0. This avoids treating repeated TBT obligations as request-level
  Goodput while leaving the score structure unchanged.
  configs/dpp_selector_integration_freeze.json and its active-config hash
  are updated together.
- candidate_generator.recovery_age_threshold_seconds=0.25 is
  provisional_for_scheduler_integration, sourced from the existing
  slo.tbt_seconds=0.25. It is the smallest conservative existing
  same-scale interval selected for one full TBT-SLO period after the first
  miss; it is not a measured formal freeze.
- The top-level scheduler_diagnostics block is also provisional. It records
  the existing Observer bound of 1024 records, an eight-iteration
  zero-progress watchdog derived from safe_set.rolling_kv_horizon_iterations,
  fail-fast disabled by default, and performance diagnostics disabled unless
  DPP_DIAGNOSTIC_ITERATION_LOG=1 is set.

## 2026-08-23: defer Predictor feature and model selection to offline training

- Profiling records each executed BatchPlan's actual duration and per-selected-
  request identity, phase, current context, and scheduled token count. Identity
  fields are used only for joins and audit.
- Candidate features and suitable regression models are compared offline
  without using the held-out test split for selection. The selected model,
  hyperparameters, feature schema, and support domain are frozen in the
  Predictor artifact before online use.
- Raw or derived data must not include remaining output length or future EOS
  information.

## 2026-08-23: use three bounded online residual windows

- The offline Ridge weights remain immutable. Decode-only, Mixed, and
  Prefill-only keep separate windows of completed in-support residuals.
- Window sizes are selected from 32/64/128 using training OOF replay only;
  fewer than 32 live samples fall back to same-scenario OOF statistics.
- Shadow evaluation records base, expected, conservative, and actual duration
  on real vLLM iterations without changing the executed scheduling policy.

The first 500-request shadow run is retained as timing-incompatible: its P95
timing difference exceeded the predeclared limit. Its error metrics remain
diagnostic, but it does not establish calibration effectiveness and G3 remains
incomplete.

Predictor feedback now uses the locked vLLM official iteration boundary, from
after asynchronous model submission through result collection and sampling;
Scheduler result updates are excluded. The separate append-only 200-request
run validated this boundary without replacing the retained run. Its residual
calibration did not meet every effectiveness criterion, and that negative
result is retained; the Predictor is accepted for modular integration.

## 2026-08-21: reopen G0 and prohibit silent parameter inheritance

- The exact Qwen3-14B repository/revision, model snapshot, tokenizer, runtime
  limits, startup `SchedulerConfig`, KV capacity, natural-EOS traces, Stock
  SLOs, Predictor, and DPP parameters are not frozen.
- `configs/dgx_spark_experiment.yaml` is the sole active configuration and is
  deliberately non-executable while these G0 blockers remain.
- Previously observed DGX host, CUDA, Python, PyTorch, vLLM-installation, and
  source facts may be reused only after verification. No model-specific
  capacity, trace, SLO, output-length assumption, or result transfers into the
  new campaign.
- Obsolete version-controlled 5070 campaign code, configs, traces, results,
  plots, tests, and standalone compatibility-smoke files were removed on
  2026-08-22 at the user's request. A separately confirmed cleanup removed the
  old 5070 raw results, BurstGPT input, and invalid Qwen3 scan/validation trace
  drafts. Old model caches and retained Qwen3 raw evidence were not deleted;
  none of the historical residue is an active input.
- The currently installed vLLM source commit is recorded as an observed G0
  candidate, not silently changed. Compatibility must be verified before it is
  frozen for the new model.

## 2026-08-21: freeze interface ambiguities before DPP implementation

The following interface choices are now fixed:

- every public decision-round contract carries `snapshot_hash`;
- the canonical candidate-set name is `safe_candidates`; and
- Controller owns Fallback construction.

For Scheduler-internal obligation settlement, the locked-vLLM event is the
`EngineCoreOutput` created by `Scheduler.update_from_output`, after actual
model output and sampling. This server-side event drives the live ledger; it
does not replace the client-receives-SSE boundary used by reported TTFT/TBT
metrics.

Remaining formal-freeze work is limited to the Recovery-age rule and replacing
integration-only Safe-Set/DPP values if later evidence selects different formal
benchmark parameters.

## 2026-08-23: freeze normalized DPP Selector parameters for integration

- Retained Stock request events give TTFT miss ratios of 2.0%/2.5% and TBT
  obligation miss ratios of 3.40%/4.62% at 0.20/0.25 req/s. The integration
  freeze uses `epsilon^F=epsilon^D=0.05`; the overload 0.30 req/s evidence is
  retained and exceeds this target.
- Score-time token quantities are normalized by `C_tok=2048`, and debt,
  obligation outcome, and service-utility quantities by `C_seq=64`. With these
  dimensionless numerator terms, `V=1.0`.
- Immediate utility is expected on-time TTFT plus TBT obligation service. It is
  not request-level Goodput. The denominator is expected iteration seconds;
  non-positive or non-finite durations fail closed.
- The exact inputs and raw counts are frozen in
  `configs/dpp_selector_integration_freeze.json`. This is sufficient for the
  first integrated Scheduler, but is not a tuned formal-benchmark optimum.

No implementation may resolve these differences implicitly.

## 2026-08-22: define natural completion as length-blind scheduling

- The objective is not to require every request to finish with
  `finish_reason=stop`; it is to prevent Scheduler access to a future or
  remaining output length.
- The runner sends the reviewed finite `max_tokens` only as a client
  termination guard. The guard is absent from Scheduler contracts, candidates,
  Predictor features/labels, and decisions.
- Both `stop` and `length` observations are retained and reported separately.
  A `length` terminal is never silently filtered or reinterpreted as a natural
  EOS sample.

## 2026-08-21: preserve the remote-only execution and shared-host boundary

- WSL remains the source-edit/review location; all project execution remains on
  `dgx-spark:/home/dongj/LLM` through `scripts/remote_dgx.sh`.
- Routine source synchronization excludes environments, caches, results, logs,
  and model weights.
- Downloading or transferring Qwen3-14B is a separate bulk operation requiring
  an immutable revision, source/size/storage review, and user confirmation of
  operator approval. This documentation migration does not authorize it.

## 2026-08-25: remove only explicitly obsolete local result residue

- Local result cleanup is limited to the retired `shared_scan*` namespaces,
  superseded pre-0.84 G0 captures, and reproducible large Phase A intermediate
  outputs.
- The authoritative `g0_stock_capture_084` record, Predictor datasets and
  source runs, Scheduler runs including failures and negative outcomes, current
  raw DPP diagnostics, frozen artifacts, and all remote copies remain
  unchanged.
- Future cleanup follows `results/README.md`: size, failure, or a negative
  conclusion alone is never a deletion criterion.
