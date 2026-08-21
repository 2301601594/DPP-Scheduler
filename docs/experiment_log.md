# Experiment log

> **Historical archive.** These entries predate the Qwen3-14B modular
> `BatchPlan` Scheduler design and are retained only with their append-only
> raw evidence. They are not active G0-G7 status, configuration, traces, SLOs,
> or baselines for the new campaign. Active entries start in
> `docs/qwen3_14b_experiment_log.md`.

## 2026-08-12 — G0 and GPU smoke

- Downloaded ShareGPT (672,837,942 bytes, SHA256
  `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`)
  and BurstGPT (142,376,815 bytes, SHA256
  `56193aa9b2bb26128ded43d2d29a960df6bf5af062bcfc9b005f3fcaa4e6e501`).
- Generated validation/test fixed, heterogeneous, ShareGPT and continuous
  BurstGPT traces. A second deterministic generation produced identical trace
  hashes; manifest verification passed.
- The earliest feasible deterministic BurstGPT window under the corrected
  two-hour bound contains 2000 filtered GPT-4 requests over 7194 seconds.
- An initial smoke attempt was invalid because an older project vLLM service
  occupied port 8000. The append-only failed raw run was retained. The runner
  now refuses to start when the configured health endpoint already exists.
- Final four-run balanced smoke passed: every run completed 10/10 measured
  requests with zero input/output length mismatch, failure, OOM, preemption,
  or residual running/waiting requests.
- Stock-Auto resolved to `max_num_batched_tokens=2048`; its repeated startup
  exposed 128,416 KV tokens. B8192 revealed a one-time compilation-cache effect
  (95,296 KV tokens on first startup, 115,664 on the next), so G3 includes an
  excluded one-time startup-cache warmup for every budget.

G1, G2 and G3 full matrices had not yet produced results at the time of this
entry. No SLO, capacity, budget trade-off or Best-Fixed conclusion is recorded
here.

The full G1 command was started to validate the long-run path, reached the
first decode-heavy serial run, and was intentionally interrupted during the
interactive implementation session. Its append-only metadata is marked
`failed` with `KeyboardInterrupt`; it is excluded by `--resume` and aggregation.

## 2026-08-12 — G1 scope amendment

- Before any valid formal G1 run completed, the user reduced G1 to the three
  fixed classes used by the heterogeneous workload.
- Serial measurement count is now 100; saturation and low-load remain 300.
- Two previously interrupted 300-request decode-heavy serial attempts remain
  in raw storage and are excluded because their configuration hashes and run
  keys do not match the amended frozen configuration.

## 2026-08-13 — G1-G2 automation implementation

- Added a combined dry-run/execute/report-only Stock-Auto pipeline for the
  three-class G1 and Poisson G2 scope.
- The configuration change produced a new experiment hash. Existing raw runs
  remain append-only but are excluded from the new reference.
- Rebuilt the trace manifest from local data with the existing tokenizer. The
  fixed validation/test trace SHA256 values remained deterministic.
- Static tests and dry-run were executed. The full GPU G1-G2 matrix has not
  been run by this entry; no SLO or capacity result is claimed here.
- The real execution preflight correctly refused to start because the earlier
  G1 process (PID 93004) still owned a vLLM child on `127.0.0.1:8000`. That
  process was not stopped or modified; the new ten-request preflight remains
  pending until the port and GPU are idle.
