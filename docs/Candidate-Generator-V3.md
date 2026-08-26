# Candidate Generator V3

## 1. Scope and status

This document defines the current `slack_centered_multiplier_v3` Candidate
Generator. It supersedes the v2 Candidate Generator layout while leaving the
rest of the modular Scheduler design unchanged.

The implementation and configuration are frozen for engineering integration.
The active configuration still marks this component
`formal_benchmark_eligible: false`; implementation completion alone is not
formal performance evidence.

## 2. Candidate construction

For each non-empty `StateSnapshot`:

1. include every active Decode request in every normal candidate;
2. always construct one `ALL_DECODE:ZERO` candidate;
3. obtain a base Prefill budget `P` from the injected `BudgetResolver`;
4. construct the multiplier neighborhood
   `{0.50P, 0.75P, 1.00P, 1.25P, 1.50P}` using floor rounding;
5. clamp each value by visible Prefill backlog and
   `token_budget - active_decode_count`, then deduplicate equal budgets;
6. allocate each positive budget using `URGENCY`, `COMPLETION_AWARE`, and
   `CONTINUATION`; and
7. canonically deduplicate plans by their exact
   `(prefill_items, decode_items)`.

The raw bound is 16 plans: one ZERO plus five budgets times three allocation
policies. Resource projection is pure. Safe-Set remains responsible for final
physical and Predictor feasibility.

## 3. Slack-centered base budget

When active Decode requests have live next-token TBT deadlines, the Resolver
uses the smallest positive slack minus the configured safety margin as the
target iteration duration. It evaluates the configured discrete Prefill budget
grid with shadow `CONTINUATION` plans and selects the largest budget whose
finite positive expected duration does not exceed that target.

`P` is a search center, not a hard safety bound for all three allocation
policies. The formal candidate plans are predicted and filtered again after
generation.

Without Decode or without a live Decode deadline, the Resolver uses the visible
resource-capped Prefill maximum. An invalid Predictor response or no feasible
grid point produces `P=0`.

## 4. Prefill allocation policies

### URGENCY

Sort by `remaining_tokens / positive_ttft_slack` descending. Overdue requests
receive the existing large overdue score. This dimensional value is used only
for relative ordering, never compared with fixed numeric severity thresholds.

### COMPLETION_AWARE

Compute the same urgency score for all waiting Prefill requests in the current
Snapshot and sort scores descending. Divide their empirical ranks into:

- top urgency third;
- middle urgency third; and
- bottom urgency third.

The boundaries are `ceil(n/3)` and `ceil(2n/3)`. Equal urgency scores always
share the tier of their first descending rank, so an unrelated stable tie key
cannot split equal scores across tiers.

Within each tier, order by:

```text
remaining_tokens ascending
-> running before waiting
-> arrival_time
-> ordinal
-> request_id
```

Thus completion-aware behavior prioritizes short remaining Prefill work within
comparable urgency. `CONTINUATION`, rather than `COMPLETION_AWARE`, owns the
strong running-first policy.

### CONTINUATION

Order running Prefill first, then waiting Prefill, using arrival time, ordinal,
and request ID as stable keys.

## 5. Diagnostics

Per-frame Candidate Generator diagnostics distinguish:

- unique actual Prefill token totals;
- distinct positive multiplier budgets;
- canonical plan count after deduplication; and
- distinct canonical Prefill allocations produced by the three policies for
  each multiplier.

These metrics must not be substituted for one another. In particular, a count
of distinct token totals does not establish policy diversity.

The Python diagnostic harness is a synthetic policy-stress tool, not a real
vLLM scheduling replay. It retains requests for an explicit diagnostic-only
Prefill hold interval so a Snapshot may contain multiple waiting requests.
After that interval, each synthetic Decode deadline represents the next token:
`last_synthetic_token_time + TBT SLO`. The hold interval and its
non-measurement status are recorded in the report.

Production aggregate schema version 2 records multiplier selection
(`ZERO`, `M050`, `M075`, `M100`, `M125`, `M150`, `OTHER`) separately from
allocation policy selection (`ZERO`, `URGENCY`, `COMPLETION_AWARE`,
`CONTINUATION`, `OTHER`).
