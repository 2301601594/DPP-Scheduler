# Active research decisions

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
- Safe-Set hard-filters physical feasibility and Predictor support. SLO risk is
  ranked: zero-new-violation plans are preferred, otherwise risk-ranked Top-K
  still reaches DPP.
- KV safety uses a finite Rolling horizon and reserve blocks. It never reserves
  or predicts a request's complete output length.
- Fallback is independent of DPP scoring: EDF Decode-only when Decode exists,
  otherwise minimum feasible Prefill, followed by explicit Preemption/Idle if
  hard feasibility still fails.
- DPP maintains only Prefill backlog, TTFT debt, and TBT debt. Observer updates
  them from actual events; every obligation is settled once, and actual
  request-level Goodput is counted only after natural EOS.

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

Before G5, the specification must still choose and test:

- units, numeric ranges, zero-duration handling, tie keys, Recovery rules,
  Top-K, and every ledger update boundary.

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
