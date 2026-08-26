#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SESSION="time-to-budget-actual-v4"
CAMPAIGN_ID="time_to_budget_actual_validation_v4"
RUN_ID="time_to_budget_actual_seed6001_attempt04"
RAW_ROOT="${REPO_ROOT}/results/raw/qwen3_14b_dgx_spark/${CAMPAIGN_ID}"
RUN_ROOT="${RAW_ROOT}/runs/${RUN_ID}"
FINAL_ROOT="${REPO_ROOT}/results/processed/qwen3_14b_dgx_spark/time_to_budget_validation_final_v4"
SMOKE_CAMPAIGN_ID="time_to_budget_actual_validation_smoke_v2"
SMOKE_RUN_ID="time_to_budget_actual_smoke_seed6001_attempt02"
SMOKE_ROOT="${REPO_ROOT}/results/raw/qwen3_14b_dgx_spark/${SMOKE_CAMPAIGN_ID}"
SOURCE_CAMPAIGN="predictor_profile_stock_n500_v1"
SOURCE_TRACE_DIR="traces_attempt_01"
SOURCE_TRACE="qps_0.2_seed_1001.jsonl"

usage() {
  cat <<'EOF'
Usage: scripts/time_to_budget_actual_validation.sh COMMAND [--resource-approved]

Commands:
  preview                    Print the exact 118-batch matrix without GPU work.
  smoke --resource-approved  Run the one-batch admission-wait regression on GPU.
  start --resource-approved  Run preflight and launch the bounded job in tmux.
  status                     Show tmux and run-manifest status.
  validate                   Validate exact execution, timing, and cleanup proofs.
  finalize                   Build the four final validation artifacts.
EOF
}

require_python() {
  if [[ ! -x "${PYTHON}" ]]; then
    printf 'Project interpreter is missing: %s\n' "${PYTHON}" >&2
    exit 2
  fi
}

runner_args() {
  RUNNER_ARGS=(
    --config configs/dgx_spark_experiment.yaml
    --campaign-id "${CAMPAIGN_ID}"
    --source-campaign-id "${SOURCE_CAMPAIGN}"
    --source-trace-dir "${SOURCE_TRACE_DIR}"
    --source-trace "${SOURCE_TRACE}"
    --source-qps 0.2
    --source-seed 1001
    --run-id "${RUN_ID}"
    --recipe-seed 6001
    --recipe-mode time_to_budget_validation
    --startup-timeout 600
    --batch-timeout 300
    --run-timeout 7200
  )
}

resource_preflight() {
  local available_kib owner pid
  local -a gpu_pids=()
  command -v tmux
  command -v nvidia-smi
  printf 'job=%s\n' "${SESSION}"
  printf 'matrix=118 isolated exact-plan target iterations\n'
  printf 'expected_duration=15-45 minutes; hard bound=2 hours\n'
  printf 'gpu=one GPU; disk_budget=2 GiB; network=no downloads\n'
  printf 'output=%s\n' "${RAW_ROOT}"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
  free -h
  df -h -- "${REPO_ROOT}"
  available_kib="$(df -Pk -- "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
  if [[ ! "${available_kib}" =~ ^[0-9]+$ ]] || (( available_kib < 2 * 1024 * 1024 )); then
    printf 'At least 2 GiB free disk is required; available KiB=%s\n' \
      "${available_kib}" >&2
    exit 1
  fi
  mapfile -t gpu_pids < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | awk '$1 ~ /^[0-9]+$/ {print $1}'
  )
  if (( ${#gpu_pids[@]} > 0 )); then
    printf 'Active GPU compute processes detected; refusing to overlap:\n' >&2
    for pid in "${gpu_pids[@]}"; do
      owner="$(ps -o user= -p "${pid}" 2>/dev/null | awk '{print $1}')"
      printf '  pid=%s owner=%s\n' "${pid}" "${owner:-unknown}" >&2
    done
    exit 1
  fi
}

validate_run() {
  "${PYTHON}" -c \
    'from pathlib import Path; from benchmarks.isolated_candidate_profile import validate_isolated_run_directory; print(validate_isolated_run_directory(Path("results/raw/qwen3_14b_dgx_spark/time_to_budget_actual_validation_v3/runs/time_to_budget_actual_seed6001_attempt03"), expected_run_id="time_to_budget_actual_seed6001_attempt03", recipe_seed=6001, recipe_mode="time_to_budget_validation"))'
}

require_python
runner_args
command="${1:-}"
shift || true

case "${command}" in
  preview)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_isolated_candidate_profile "${RUNNER_ARGS[@]}" --dry-run
    ;;
  smoke)
    if [[ "${1:-}" != "--resource-approved" || $# -ne 1 ]]; then
      printf 'smoke requires --resource-approved after group notification.\n' >&2
      exit 2
    fi
    if [[ -e "${SMOKE_ROOT}" ]]; then
      printf 'Append-only smoke output already exists: %s\n' "${SMOKE_ROOT}" >&2
      exit 1
    fi
    resource_preflight
    "${PYTHON}" -m benchmarks.run_isolated_candidate_profile \
      --config configs/dgx_spark_experiment.yaml \
      --campaign-id "${SMOKE_CAMPAIGN_ID}" \
      --source-campaign-id "${SOURCE_CAMPAIGN}" \
      --source-trace-dir "${SOURCE_TRACE_DIR}" \
      --source-trace "${SOURCE_TRACE}" \
      --source-qps 0.2 \
      --source-seed 1001 \
      --run-id "${SMOKE_RUN_ID}" \
      --recipe-seed 6001 \
      --recipe-mode time_to_budget_validation_smoke \
      --startup-timeout 600 \
      --batch-timeout 300 \
      --run-timeout 1200
    "${PYTHON}" -c \
      'from pathlib import Path; from benchmarks.isolated_candidate_profile import validate_isolated_run_directory; print(validate_isolated_run_directory(Path("results/raw/qwen3_14b_dgx_spark/time_to_budget_actual_validation_smoke_v2/runs/time_to_budget_actual_smoke_seed6001_attempt02"), expected_run_id="time_to_budget_actual_smoke_seed6001_attempt02", recipe_seed=6001, recipe_mode="time_to_budget_validation_smoke"))'
    ;;
  start)
    if [[ "${1:-}" != "--resource-approved" || $# -ne 1 ]]; then
      printf 'start requires --resource-approved after group notification.\n' >&2
      exit 2
    fi
    if [[ -e "${RAW_ROOT}" ]]; then
      printf 'Append-only output already exists: %s\n' "${RAW_ROOT}" >&2
      exit 1
    fi
    resource_preflight
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      printf 'tmux session already exists: %s\n' "${SESSION}" >&2
      exit 1
    fi
    printf -v shell_command 'exec %q -m benchmarks.run_isolated_candidate_profile' "${PYTHON}"
    for value in "${RUNNER_ARGS[@]}"; do
      printf -v shell_command '%s %q' "${shell_command}" "${value}"
    done
    tmux new-session -d -s "${SESSION}" -c "${REPO_ROOT}" "${shell_command}"
    printf 'Started tmux session: %s\n' "${SESSION}"
    printf 'Output: %s\n' "${RAW_ROOT}"
    ;;
  status)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      printf 'tmux_status=running session=%s\n' "${SESSION}"
    else
      printf 'tmux_status=not_running session=%s\n' "${SESSION}"
    fi
    if [[ -f "${RUN_ROOT}/run_manifest.json" ]]; then
      "${PYTHON}" -c \
        'import json; from pathlib import Path; p=Path("results/raw/qwen3_14b_dgx_spark/time_to_budget_actual_validation_v3/runs/time_to_budget_actual_seed6001_attempt03/run_manifest.json"); d=json.loads(p.read_text()); print("run_status=" + str(d.get("status"))); print("error=" + str(d.get("error"))) if d.get("error") else None'
    else
      printf 'run_status=not_started\n'
    fi
    ;;
  validate)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    validate_run
    ;;
  finalize)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    validate_run
    if [[ -e "${FINAL_ROOT}" ]]; then
      printf 'Append-only final output already exists: %s\n' "${FINAL_ROOT}" >&2
      exit 1
    fi
    "${PYTHON}" -m benchmarks.time_to_budget_validation \
      --profile results/raw/qwen3_14b_dgx_spark/predictor_profile_targeted_prefill_mixed_n500_v1/runs/source_qps_0p2_seed_1001_recipe_2001_attempt_01/iteration_profile.jsonl \
      --profile results/raw/qwen3_14b_dgx_spark/predictor_profile_targeted_prefill_mixed_n500_v1/runs/source_qps_0p2_seed_1002_recipe_2002_attempt_01/iteration_profile.jsonl \
      --predictor predictors/qwen3_14b/ridge_mixed_decode_three_segment_online_v2 \
      --predictor predictors/qwen3_14b/ridge_mixed_decode_three_segment_cross_online_v3 \
      --actual-profile "${RUN_ROOT}/iteration_profile.jsonl" \
      --output-dir "${FINAL_ROOT}" \
      --snapshots-per-model 20
    printf 'Final artifacts: %s\n' "${FINAL_ROOT}"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
