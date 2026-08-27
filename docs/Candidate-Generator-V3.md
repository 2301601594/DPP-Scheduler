# Candidate Generator — fixed fractions with Stock BatchPlan

Status: active development contract, non-formal.

This document supersedes the former Predictor-inversion/multiplier V3
behavior while retaining the historical filename used by the repository
document index.

## 1. Candidate set

For one immutable `StateSnapshot`, define

```text
D = number of active Decode requests
R = sum of remaining prompt tokens
P_max = min(R, token_budget - D)
```

The normal generator constructs:

```text
ZERO
P10, P20, P30, P40, P50, P60, P70, P80, P90, P100
STOCK
```

`Pxx` uses `floor(P_max * fraction)`. Budgets are clamped, minimum chunk and
sequence constraints are applied, and final plans are canonically deduplicated
by `(prefill_items, decode_items)`. The maximum retained candidate count is 12.

`ZERO` and every `Pxx` plan contain all active Decode requests. The separate
`STOCK` plan is allowed to contain a subset because it follows native Stock
request-selection order.

## 2. Prefill binding order

All fixed-fraction plans use one deterministic ordering:

1. running Prefill requests in live running/ordinal order;
2. waiting Prefill requests by `(arrival_time, ordinal, request_id)`.

Debt, TTFT deadline, Predictor output, and future output length never affect
this ordering. Partial Prefill remains controlled by the active minimum chunk
setting.

## 3. Stock plan

`build_stock_plan(snapshot)` is a side-effect-free implementation of the
locked vLLM Stock request-selection path supported by the active runtime:

1. visit the combined running queue in ordinal order;
2. schedule each running Prefill up to its remaining prompt work and each
   running Decode for one token;
3. admit waiting Prefill in FCFS order;
4. stop or skip work that would exceed token, sequence, or current KV capacity.

The runtime contract keeps prefix caching, speculative decoding, asynchronous
scheduling, and KV connectors disabled. Native preemption is not encoded in a
`BatchPlan`; a non-empty Snapshot that cannot yield Stock work fails closed in
the forced-Stock development mode.

The same builder is used both for the normal `STOCK` candidate and the
`forced_stock_plan` test mode. The latter bypasses Predictor, Safe-Set,
Selector, and Fallback while preserving exact-plan validation, execution, and
actual debt/ledger feedback.

## 4. Predictor boundary and diagnostics

Candidate generation never calls Predictor. Predictor continues to estimate
the duration of the completed normal candidate set before Safe-Set and DPP
selection. `budget_resolver.py` is retained only as historical/compatibility
code and is not wired into the live Scheduler.

Per-run diagnostics use the buckets `ZERO`, `P10` through `P100`, `STOCK`, and
`OTHER`. They record requested fraction budgets, actual candidate budgets,
Stock Prefill budget, canonical candidate count, selection mode, and pipeline
stage call counts.

## 5. Required checks

- fraction budget floor/clamp and canonical deduplication;
- running-first then waiting-FCFS binding;
- Stock running/waiting request selection and capacity bounds;
- at most 12 retained candidates;
- forced-Stock mode never calls Predictor, Safe-Set, Selector, or Fallback.
