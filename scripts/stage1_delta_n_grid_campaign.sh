#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SESSION="stage1-delta-n-grid-qps0p25-n150-s1001-v1"
CAMPAIGN_ID="stage1_delta_n_grid_qps0p25_n150_seed1001_v1"
CAMPAIGN_ROOT="${REPO_ROOT}/results/raw/qwen3_14b_dgx_spark/${CAMPAIGN_ID}"

usage() {
  cat <<'EOF'
Usage: scripts/stage1_delta_n_grid_campaign.sh COMMAND [--resource-approved]

Commands:
  preview                    Print the smoke gate and the six-run delta-N matrix.
  smoke --resource-approved  Run/repair the required Stock+DPP(N=0) smoke in detached tmux.
  start --resource-approved  Launch the six main runs only after the smoke checkpoint passed.
  status                     Print tmux and append-only checkpoint status.
  resume --resource-approved Resume failed/pending work in detached tmux.
  validate                   Fail-closed validation of every completed run.

The campaign runs one Stock run and five DPP runs (DPP_STAGE1_MAX_DELTA_N in
{0,2,4,8,16}) over the single staged n=150 QPS 0.25 seed 1001 trace, with
Selector Diagnosis enabled on every DPP run. It is development_nonformal,
single-seed, and not formal-benchmark eligible. SSH disconnection does not
stop the tmux worker.
EOF
}

require_python() {
  if [[ ! -x "${PYTHON}" ]]; then
    printf 'Project interpreter is missing or not executable: %s\n' "${PYTHON}" >&2
    exit 2
  fi
}

require_approval_flag() {
  if [[ "${1:-}" != "--resource-approved" || $# -ne 1 ]]; then
    printf '%s requires exactly --resource-approved after group notification.\n' \
      "${command}" >&2
    exit 2
  fi
}

resource_preflight() {
  local available_kib owner pid
  local -a gpu_pids=()

  command -v tmux
  command -v nvidia-smi
  printf 'job=%s\n' "${SESSION}"
  printf 'scope=development non-formal Stage-1 delta-N grid\n'
  printf 'matrix=1 Stock + 5 DPP runs (N in {0,2,4,8,16}); 150 requests per main run\n'
  printf 'trace=staged n=150 QPS 0.25 seed 1001 (SHA-256 203e7ed4...)\n'
  printf 'smoke_gate=1 Stock + 1 DPP(N=0) run of 20 requests before all main runs\n'
  printf 'expected_duration=3-6 hours; hard internal campaign bound=12 hours\n'
  printf 'gpu=one GPU; unified_memory=model plus runtime; disk_budget=5 GiB\n'
  printf 'network=no downloads; output=%s\n' "${CAMPAIGN_ROOT}"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
  free -h
  df -h -- "${REPO_ROOT}"
  ps -u "$(id -un)" -o pid=,etime=,comm= | head -n 40

  available_kib="$(df -Pk -- "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
  if [[ ! "${available_kib}" =~ ^[0-9]+$ ]] || (( available_kib < 5 * 1024 * 1024 )); then
    printf 'At least 5 GiB free disk is required; available KiB=%s\n' \
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

launch_worker() {
  local worker_command="$1" shell_command
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    printf 'tmux session already exists: %s\n' "${SESSION}" >&2
    exit 1
  fi
  printf -v shell_command 'exec %q -m benchmarks.run_stage1_delta_n_grid %q' \
    "${PYTHON}" "${worker_command}"
  tmux new-session -d -s "${SESSION}" -c "${REPO_ROOT}" "${shell_command}"
  printf 'Started detached tmux session: %s\n' "${SESSION}"
  printf 'Campaign output: %s\n' "${CAMPAIGN_ROOT}"
  printf 'The worker will stop before the six main runs if the smoke fails.\n'
}

require_python
command="${1:-}"
shift || true

case "${command}" in
  preview)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_stage1_delta_n_grid preview
    ;;
  smoke)
    require_approval_flag "$@"
    resource_preflight
    if [[ -f "${CAMPAIGN_ROOT}/campaign_checkpoint.json" ]]; then
      if grep -q '"status": "smoke_passed"' \
        "${CAMPAIGN_ROOT}/campaign_checkpoint.json"; then
        printf 'Smoke gate already passed: %s\n' "${CAMPAIGN_ROOT}" >&2
        exit 1
      fi
      launch_worker resume-smoke
    elif [[ -e "${CAMPAIGN_ROOT}" ]]; then
      printf 'Campaign path exists without a checkpoint: %s\n' \
        "${CAMPAIGN_ROOT}" >&2
      exit 1
    else
      launch_worker smoke
    fi
    ;;
  start)
    require_approval_flag "$@"
    if [[ ! -f "${CAMPAIGN_ROOT}/campaign_checkpoint.json" ]]; then
      printf 'Run and validate the required smoke first: %s\n' \
        "${CAMPAIGN_ROOT}" >&2
      exit 1
    fi
    if ! grep -q '"status": "smoke_passed"' \
      "${CAMPAIGN_ROOT}/campaign_checkpoint.json"; then
      printf 'Smoke checkpoint has not passed; use smoke/status: %s\n' \
        "${CAMPAIGN_ROOT}" >&2
      exit 1
    fi
    resource_preflight
    launch_worker resume
    ;;
  status)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      printf 'tmux_status=running session=%s\n' "${SESSION}"
    else
      printf 'tmux_status=not_running session=%s\n' "${SESSION}"
    fi
    "${PYTHON}" -m benchmarks.run_stage1_delta_n_grid status
    ;;
  resume)
    require_approval_flag "$@"
    if [[ ! -f "${CAMPAIGN_ROOT}/campaign_checkpoint.json" ]]; then
      printf 'Campaign checkpoint does not exist: %s\n' "${CAMPAIGN_ROOT}" >&2
      exit 1
    fi
    if ! grep -q '"status": "passed"\|"status": "complete_with_failures"' \
      "${CAMPAIGN_ROOT}/campaign_checkpoint.json"; then
      printf 'Main campaign cannot resume before smoke passes; use smoke/status.\n' >&2
      exit 1
    fi
    resource_preflight
    launch_worker resume
    ;;
  validate)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_stage1_delta_n_grid validate
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
