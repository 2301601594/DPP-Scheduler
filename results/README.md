# Results directory

This directory contains local DGX result copies. Large raw and derived files
are intentionally ignored by Git; frozen manifests, reviewed summaries, and
conclusion-affecting configuration stay version controlled elsewhere in the
repository.

## Layout

- `raw/qwen3_14b_dgx_spark/`: append-only run namespaces pulled from the DGX.
  Keep valid runs, failed runs, negative results, smoke evidence, manifests,
  logs, and source traces together under their original campaign ID.
- `processed/qwen3_14b_dgx_spark/`: reviewed or reproducible aggregates. The
  candidate-parameter freeze and compact DPP analysis summaries are the
  version-controlled records in this tree.
- `dataset/`: reproducible Predictor feature and split datasets. These remain
  local because of their size; their manifests and frozen Predictor reports
  carry the required hashes and provenance.

## Active evidence map

- Environment and trace freeze: `g0_stock_capture_084`,
  `g0_trace_manifest_stage1`, and `g1_stock_matrix_cap2048*`.
- Predictor: `predictor_profile_stock_n500_v1`,
  `predictor_profile_targeted_prefill_mixed_n500_v1`,
  `candidate_knee_profile_n500_v1`, `predictor_online_timing_aligned_n200_v1`,
  `predictor_ood_calibration_v2`, `predictor_ood_validation_v2`, and
  `stock_reference_concurrency_v2_dev_n300`.
- Scheduler development: `scheduler_comparison_dev_n300_seed1001_v1` and the
  namespaced DPP v2/v2.1 diagnostic runs.

Smoke, failed, invalid, and negative runs not listed above are retained for
audit unless a later explicit cleanup records their exact removal. Do not
delete a record merely because it failed, produced a negative result, or is
large. Generated analysis outputs may be removed only when their source run
and rebuilding script remain available.
