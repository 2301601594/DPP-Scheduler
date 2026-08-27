# DPP V3: Candidate absence vs Selector ZERO-bias — diagnosis

Scope: analysis only. No scheduler, Candidate Generator, BudgetResolver, Selector,
Safe-Set, Predictor, or benchmark configuration was modified. No benchmark was
started. All DGX reads and statistics ran through `scripts/remote_dgx.sh run`.

**Verdict: the question cannot be answered from the requested run.** The
discriminating statistic — P(ZERO selected | Prefill backlog ∧ a non-ZERO safe
candidate existed) — is not computable, because the run was launched with
`DPP_DIAGNOSTIC_ITERATION_LOG=0` and therefore never emitted a single per-frame
candidate record. Sections 3–6 and 9–13 of the instruction are unanswerable from
this run. Sections 7 and 10 are partially answerable. What follows separates the
three derivable results from the large set that is not derivable, and states
exactly what a re-collection must capture.

## 1. Run identity

| Field | Value |
| --- | --- |
| campaign_id | `dpp_v3_default_n300_qps0p25_seed1001_v1` |
| run_id | `main_dpp_qps0p25_seed1001_attempt01` |
| window (UTC) | 2026-08-26T15:21:29 → 15:49:06 |
| status | `complete_with_failures` |
| scheduler_policy | `dpp` |
| trace_sha256 | `9da30703ed0fb617e8a02cd3ac871a9a470b745b3405716323f7a9262007a771` |
| config_sha256 | `9589b53c7e90c4206b84931e1e5016d1ca94d6757827e0e0e3e415f6c24e793a` |
| git root | `7e12054`, **dirty** |
| vLLM | `83ad767e`, clean |
| aggregate sha256 | `53d0c02cf93c6f86553d462366f02a85bfa740daafa1a10d82391704a1febf08` |
| scope | `development_nonformal`, `formal_comparison_eligible: false` |

Two provenance limits apply to everything below.

1. The run executed from a **dirty** tree at `7e12054` with
   `dpp_scheduler/candidate_generator.py`, `settings.py`, `fallback.py` and
   `vllm_adapter.py` all modified. The exact Candidate Generator source that
   produced these numbers is therefore not identified by any commit, and is not
   established to equal either the committed `7e12054` or current `HEAD`.
2. One request failed: `poisson_q0.25_s1001_0094`,
   `ServerDisconnectedError`, `e2e_ms=3.09`, at t≈356 s. The server log shows no
   engine error and a clean drain shutdown, so this is a client transport
   failure. It is retained, not filtered.

## 2. Data completeness

`run_manifest.json` records `dpp_diagnostic_iteration_log: false` and
`DPP_DIAGNOSTIC_ITERATION_LOG=0`. Consequences:

- `startup.log` is 551 lines and contains **0** `ModularDPPScheduler diagnostic=`
  records.
- The only surviving diagnostic artifact is `dpp_diagnostic_aggregate.json`,
  which is **pre-reduced by construction**: histograms plus
  count/mean/p50/p90/p95/p99/max reductions.
- `vllm_adapter.py` does hold an in-memory `m150_selected_frames` dict keyed by
  frame id, but `_dpp_write_aggregate` serialises only its reduced statistics.
  **Zero per-frame rows reach disk.**

The per-frame fields the instruction asks for are all produced by
[vllm_adapter.py:1500](../dpp_scheduler/vllm_adapter.py#L1500)
`_dpp_record_schedule_diagnostic` — including `prefill_items`,
`total_prefill_tokens`, per-candidate predictions, and DPP score ranks — but that
function only emits them when the iteration log is enabled.

The machine-readable field-by-field matrix is in
`dpp_v3_candidate_vs_selector_field_availability.csv`. Summary: of the 17 per-frame
fields requested in section 3, **13 are absent**, 2 survive only as order-free
histograms, 1 survives as pooled reduced statistics, and 0 survive per frame.

**No `frames.csv` accompanies this record.** The instruction asked for one, but
zero per-frame rows survive; producing it would require fabrication.

## 3. Backlog-frame population

| Quantity | Value |
| --- | --- |
| Decision frames (`selection_histogram` total) | 7752 |
| `selected_prefill_tokens.count` | 7751 |
| Accounting gap | 1 frame (all derived counts carry ±1) |
| Prefill-backlog frames | 4211 |
| Backlog ∧ active decode | 4209 |
| Backlog ∧ no decode | 2 |
| No backlog (ZERO structurally forced) | 3541 (45.68%) |
| Executed mixed / prefill-only / decode-only | 403 / 2 / 7346 |

Selection histogram: ZERO 7346, M050 250, M075 36, M100 86, M125 11, M150 22,
OTHER 1.

Two counters that look contradictory are not. `mixed_iteration_count` (4209)
buckets by **snapshot** composition, while `actual_duration_seconds.mixed` (403)
buckets by **executed plan** composition
([vllm_adapter.py:1330-1339](../dpp_scheduler/vllm_adapter.py#L1330-L1339) vs
[vllm_adapter.py:1937-1951](../dpp_scheduler/vllm_adapter.py#L1937-L1951)). The
gap between them *is* the starvation measurement.

## 4–6. Candidate budget diversity, canonical plan / policy diversity, clamp analysis

**Not computable for this run.** `raw_candidate_count`,
`deduplicated_candidate_count`, `multiplier_budgets`, and the per-multiplier
allocation sets are all computed into `CandidateGenerator.last_diagnostic` and
surfaced only through the disabled iteration log. Base `P`, per-frame backlog
tokens, and `token_budget − decode_count` are likewise unrecorded, so `P / P_max`
cannot be formed, `P > P_max` cannot be counted, and 5→1/2/3/4/5 multiplier
collapse cannot be measured. Clamp cause attribution
(`BACKLOG_CLAMP` / `TOKEN_BUDGET_CLAMP` / `SEQUENCE_LIMIT` / `MIN_CHUNK` /
`CANONICAL_DEDUP`) is not recorded even when the iteration log *is* enabled, so
that sub-request needs a logging change before it can ever be answered.

## 7. Mixed predicted-duration distribution

**Not computable.** No per-candidate predicted or conservative duration is
retained, for selected or unselected candidates. The light / near-light / medium /
heavy classification cannot be built at any threshold.

## 8. ZERO selection conditional probabilities

Three of the five requested probabilities are derivable. The derivation rests on
one structural fact: a non-ZERO plan can only draw prefill items from
`snapshot.waiting_prefill_requests`
([candidate_generator.py:391](../dpp_scheduler/candidate_generator.py#L391)), so
**every non-ZERO selection is necessarily a backlog frame**.

| Probability | Value | Derivation |
| --- | --- | --- |
| P(ZERO) | **94.76%** | 7346 / 7752 |
| P(ZERO \| backlog) | **90.38%** | (4211 − 405) / 4211 = 3806 / 4211 |
| P(ZERO \| backlog ∧ active decode) | **90.43%** | (4209 − 403) / 4209 = 3806 / 4209 |
| P(ZERO \| backlog ∧ ∃ non-ZERO safe candidate) | **not computable** | per-frame candidate availability unrecorded |
| P(ZERO \| backlog ∧ ∃ mixed candidate τ̂ ≤ 300 ms or ≤ 250 ms) | **not computable** | no per-candidate duration retained |

Of the 7346 ZERO selections, 3541 were structurally forced by an empty backlog and
3805–3806 occurred *despite* a live backlog.

**The identifiability statement that decides this stage.** Among those 3806
backlog frames where ZERO was selected, the number that offered at least one
non-ZERO safe candidate is bounded only by **[0, 3806]**. At 0 the cause is
entirely candidate absence (class A); at 3806 it is entirely selector preference
(class D). The retained data contains **no information** to narrow that interval.
That interval is precisely the question asked, so no honest answer exists from
this run.

## 9. ZERO streak analysis

**Not computable.** Streaks require the ordered per-frame selection sequence; the
aggregate is order-free and retains no frame id or timestamp. For reference, the
existing streak analysis in
`results/processed/qwen3_14b_dgx_spark/dpp_zero_mixed_limit_cycle_qps0p25_v1/`
was built from `dpp_zero_mixed_diag_qps0p25_seed1001_v1`, whose templates are the
**v2 legacy** `ALL_DECODE:P25:requested_106` form, not V3 `SLACK_BUDGET:Mxxx`. It
is a different campaign and a different candidate generator, so it is not an
input here.

## 10. DPP score decomposition

**Not computable.** No per-candidate DPP score, duration denominator, or
TTFT / TBT / Prefill debt contribution is retained.

## 11. Representative frames

**Not computable.** Zero per-frame rows survive; the ten requested frames cannot
be quoted without fabricating them.

## 12. What the run *does* show

Two findings are genuinely derivable and both bear on attribution.

**Multiplier cost ordering is inverted.** M150 is the largest multiplier (1.5 × P),
yet all 22 of its selected frames executed in ≤ 269 ms (mean 233 ms, 0 frames over
300 ms). The pooled mixed population has mean 416 ms, p95 769 ms, max 984 ms, with
141 executions over 500 ms. The heavy executions therefore came from M050–M125
selections. If `P` tracked a per-frame slack target, M150 would be the *heaviest*
bucket; observing it as the *lightest* is the signature expected when `P` varies by
orders of magnitude between frames. This is a data-supported inference from two
separately reduced distributions, not a per-frame join, so it bounds the
population without being checkable frame by frame.

**Prefill is admitted reactively, after debt has already exploded.** At the 22
M150-selected frames, `sum_ttft_debt` averaged 50.5 s and peaked at 423.5 s, and
`max_ttft_debt` peaked at 53.2 s, while TBT debt stayed small
(`sum_tbt_debt` mean 0.10 s). Large prefill allocations coincide with already-
extreme TTFT debt rather than preventing it.

**Counter-evidence against a monotone selector ZERO-bias (weak).** In 119 of the
405 non-ZERO selections the selector chose M100/M125/M150 rather than the smallest
multiplier, and ZERO won **none** of the 36 tie frames. A selector monotonically
biased toward less prefill work would not do that. This is weak because per-frame
availability is unknown: it is not established that a lighter multiplier was on
offer in those 119 frames.

**Structural fact that makes class A the default explanation.**
[candidate_generator.py:411](../dpp_scheduler/candidate_generator.py#L411) enters
the multiplier × policy loop only under `if resolution.base_prefill_budget > 0`.
Any frame whose BudgetResolver returns `P ≤ 0` therefore has a **ZERO-only action
space by construction**, and ZERO selection in such a frame carries zero
information about selector preference.

## 13. External evidence, and why it is not an answer

Two offline reports bear on the action space:

| Report | tracked | frames | raw cand. mean/p50 | dedup mean | `no_feasible` | P p50 | selections |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dpp_v3_candidate_generator_diagnostic_v1` | yes (`7e12054`, 2026-08-26) | 200 | 5.44 / 4 | 2.48 | 11/200 | 2048 | ZERO 168, M050 32 |
| `budget_resolver_v3_after` | **no** (untracked, mtime 2026-08-27 00:08:27) | 200 | 1.165 / 1 | 1.055 | 197/200 | 0 | ZERO 197, M050 3 |

**Neither is evidence about the target run.** Both come from
`scripts/dpp_v3_candidate_generator_diagnostic.py`, whose own docstring states it
drives `CandidateGenerator` against a *synthesized* per-iteration schedule with an
explicit synthetic prefill hold, and that this is "a policy-stress input, not a
vLLM replay or performance measurement". Re-running it would produce
synthetic-schedule statistics, which the instruction explicitly forbids as a
route to a conclusion. They are recorded here only as provenance for the open
question, and are explicitly NOT used to assign a root cause.

**Provenance correction.** The second file, `budget_resolver_v3_after/report.json`,
is **untracked**. Its mtime coincides with in-session modifications to
`dpp_scheduler/budget_resolver.py` and `tests/unit/test_candidate_generator_v3.py`
that the user is making live. It most likely reflects the **current WIP state**
of the BudgetResolver the user is iterating on, not a prior committed V3 harness
run. Per AGENTS.md §6 it is not a version-controlled input and must not be cited
as evidence. The order-of-magnitude disagreement with the committed report
therefore signals that the resolver's feasibility outcome is very sensitive to
its own configuration across edits, not that two independent committed runs
exist. Neither state is established as the code that produced the target run,
which ran from a dirty tree at `7e12054` with `dpp_scheduler/candidate_generator.py`
and `dpp_scheduler/settings.py` modified.

## 14. Root-cause attribution

Per instruction section 15, the branch **cannot be assigned from this run**.
Branch 1 (Candidate Generator / Resolver still the binding constraint) is the
better-supported hypothesis, but it is *not confirmed*:

- Candidate-side support: ZERO-only action space is structurally forced whenever
  `P ≤ 0`; multiplier cost ordering is inverted in the target run; both offline
  resolver reports show a collapsed action space.
- Selector-side support: **none that is specific to the selector.** No retained
  field distinguishes "no candidate was offered" from "a candidate was offered and
  rejected".
- Selector-side counter-evidence: 119/405 non-ZERO selections were not the lightest
  multiplier; ZERO won 0/36 ties.

Stating branch 1 as established would require the quantity bounded to [0, 3806] in
section 8, which this run does not contain.

## 15. What cannot be concluded

- Not concluded: "the Candidate Generator is fixed."
- Not concluded: "the DPP Selector is the sole or primary root cause."
- Not concluded: any share split between the two causes, which section 15 branch 3
  would require.
- Not concluded: that the 90.4% ZERO rate under backlog reflects selector
  preference. It is fully consistent with a ZERO-only action space, and equally
  consistent with candidates existing and being rejected.
- The target run is `development_nonformal`; none of the above may be reported as
  a benchmark result or a gate outcome.

## 16. Recommended next action

1. **Change no algorithm.** Do not modify the Candidate Generator, BudgetResolver,
   Selector, Safe-Set, or Predictor, and do not analyse the Selector objective
   until P(ZERO | backlog ∧ ∃ non-ZERO safe candidate) has actually been measured.
2. **Re-collect with the iteration log on.** Re-run the identical configuration
   and trace with `DPP_DIAGNOSTIC_ITERATION_LOG=1`, then repeat sections 3–13
   against the emitted per-frame records.
3. **A prior attempt already exists and failed for an unrelated reason.**
   `dpp_v3_multiplier_diag_qps0p25_seed1001_v1` was launched on the same trace
   (`9da30703…`) with `dpp_diagnostic_iteration_log: true`, and died during
   EngineCore init with `RuntimeError: live v2 is disabled until held-out OOD
   calibration freezes kappa` at git `06a5d96`, emitting 0 diagnostic frames. That
   is a settings gate, not a defect in the logging path, and the later successful
   run at `7e12054` passed the equivalent gate. Re-collection is therefore
   expected to be viable.
4. **Consider adding clamp-cause fields before that run**, since clamp attribution
   (section 9) is not recorded even with the iteration log enabled. Deciding
   whether to add them is a separate authorised change, not part of this stage.
5. A re-run is a GPU benchmark on a shared host and needs explicit authorisation;
   it was not started.

## Machine-readable outputs

- `results/processed/qwen3_14b_dgx_spark/dpp_v3_candidate_vs_selector_diagnosis_v1/dpp_v3_candidate_vs_selector_diagnosis.json`
- `results/processed/qwen3_14b_dgx_spark/dpp_v3_candidate_vs_selector_diagnosis_v1/dpp_v3_candidate_vs_selector_field_availability.csv`
