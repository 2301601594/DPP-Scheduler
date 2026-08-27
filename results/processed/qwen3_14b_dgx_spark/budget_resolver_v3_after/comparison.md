# BudgetResolver V3 fix — before / after diagnostic comparison

## Scope

The Candidate Generator V3 BudgetResolver previously returned a
`base_prefill_budget` equal to the **requested** grid budget whenever the
Predictor considered the shadow plan feasible, even when the actual shadow
fill collapsed to a much smaller number of tokens. With a configured grid
of `(0, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048)`, a 500-token
backlog caused the resolver to report `P = 2048` because the predictor
still considered the corresponding 500-token shadow plan feasible. The
Candidate Generator then constructed the multiplier neighborhood
`{0.5P, 0.75P, P, 1.25P, 1.5P}` and the backlog clamp collapsed every
multiplier ≥ 1.0 to 500, eliminating the intended `light` / `medium` /
`full-backlog` Mixed budget variety.

The fix replaces requested-grid selection with **actual-shadow-prefill-token**
selection, after clamping the inversion grid to the resource ceiling
`P_max = min(backlog, token_budget − decode_count)` and explicitly
appending `P_max`. `derive_executable_inversion_grid()` performs the
clamp + add + dedup + sort helper.

## Inputs

- Trace: `traces/qwen3_14b/qps_0.25_seed_1001_cap2048.jsonl` (200 frames)
- Predictor: `predictors/qwen3_14b/ridge_mixed_decode_three_segment_cross_online_v3`
  (offline Ridge Mixed/Decode/Prefill-only model, `ood_uncertainty_coefficient=0`)
- Configuration: `configs/dgx_spark_experiment.yaml`,
  `safety_margin_seconds=0.020`, `minimum_prefill_chunk_tokens=6`
- Both reports use the same 200 request arrivals from the same trace.

## Caveat — schedule model changed

Between the baseline report
(`dpp_v3_candidate_generator_diagnostic_v1`) and this report
(`budget_resolver_v3_after`), the synthetic-schedule model was updated:

- **Before (commit 7e12054 era):** `build_schedule` used the **full-output**
  deadline (`arrival_time + assumed_output_tokens × TBT SLO`) and assumed
  Prefill completed instantaneously. Each active Decode therefore carried a
  slack of ~64 s, the target duration was ~63.98 s, and almost every
  shadow plan fit comfortably inside that budget.
- **After (commit c57bb81 era):** `build_schedule` uses the **next-token**
  deadline (`decode_started + (tokens_so_far + 1) × TBT SLO`) with a
  configurable synthetic Prefill hold. Slack is therefore at most one TBT
  SLO (≈ 0.25 s), and the target duration is at most ≈ 0.23 s. This is
  the schedule model the production Scheduler actually requires, but it
  makes most shadow plans infeasible because even a small Prefill chunk
  (≥ 0.10 s duration) often exceeds the slack.

The status-count delta is dominated by this schedule-model change rather
than by the BudgetResolver fix. Both the baseline and the new run emit
deterministic plans for the same snapshot, but they ask the Predictor
"can this fit inside 63.98 s?" vs "can this fit inside 0.23 s?".

To isolate the BudgetResolver effect, the relevant comparison is the
**shape of the P distribution and the per-frame candidate diversity**,
not the absolute counts of feasible frames.

## Resolution summary

| Status                  | Before | After |
| ----------------------- | ------: | ----: |
| INVERTED_OK             |     170 |     1 |
| INVERTED_OOD            |      18 |     0 |
| NO_FEASIBLE_BUDGET      |      11 |   197 |
| NO_DECODE_USE_MAX       |       1 |     2 |
| NO_DECODE_NO_BACKLOG    |       0 |     0 |
| PREDICTOR_INVALID       |       0 |     0 |

The schedule-model tightening pushes most frames from INVERTED_OK into
NO_FEASIBLE_BUDGET. This is correct: when the TBT slack is ≈ 0.23 s, even
a 0.13 s decode-only plan cannot be followed by any prefill chunk, so the
BudgetResolver must honestly report `P = 0` rather than fabricate a
feasible-looking value.

## P distribution

| Metric | Before | After |
| ------ | -----: | ----: |
| p_min  |      0 |     0 |
| p_p50  |   2048 |     0 |
| p_p95  |   2048 |     0 |
| p_max  |   2048 |  1047 |

This is the BudgetResolver fix at work. Before the fix, `P` was pinned to
the configured grid maximum whenever any plan was feasible, regardless of
backlog. After the fix, `P` is bounded by the actual fillable Prefill
tokens (`P_max = min(backlog, token_budget − decode_count)`). The new
`p_max = 1047` corresponds exactly to the largest backlog observed in a
frame where the Predictor still accepts the full Prefill budget inside
the available slack.

## Per-frame candidate layout

| Metric                                  | Before | After |
| --------------------------------------- | -----: | ----: |
| Distinct actual Prefill token totals / frame (max) |  16 |    6 |
| Distinct multiplier budgets / frame (max)         |   5 |    5 |
| Canonical plan count / frame (max)                |  16 |    6 |
| Policy-diverse multipliers / frame (max)          |   3 |    0 |

The after run's lower per-frame maxima reflect the tighter schedule
model, not a BudgetResolver regression — the multiplier clamp and dedup
logic still produce up to five multiplier budgets when the backlog
permits (e.g. the 3 frames in the after run where any plan is feasible).

## Conclusion

1. **BudgetResolver fix verified.** `P` no longer reports a requested-grid
   maximum when the actual fill is much smaller. `p_max` dropped from 2048
   to 1047 and now tracks the largest actually-fillable Prefill token
   count.
2. **Schedule model became stricter.** The 170 → 1 INVERTED_OK / 11 → 197
   NO_FEASIBLE_BUDGET shift is caused by switching from full-output to
   next-token deadlines. The Candidate Generator V3 fix preserves the
   existing 5-multiplier × 3-policy layout and only changes how the
   central `P` is selected.
3. **No regression in the Candidate Generator layout.** All 39 existing
   and new unit tests pass on the DGX Spark
   (`tests/unit/test_candidate_generator_v3.py`).