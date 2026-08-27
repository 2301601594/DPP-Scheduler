#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SESSION="dpp-ttft-grid-qps0p25-n150-s1001-v1"
CAMPAIGN_ID="dpp_ttft_weight_grid_qps0p25_n150_seed1001_v1"
CAMPAIGN_ROOT="${REPO_ROOT}/results/raw/qwen3_14b_dgx_spark/${CAMPAIGN_ID}"
SMOKE_SESSION="dpp-ttft-grid-smoke-n1-s1001-v1"
SMOKE_CAMPAIGN_ID="dpp_ttft_weight_grid_smoke_n1_seed1001_v1"
SMOKE_ROOT="${REPO_ROOT}/results/raw/qwen3_14b_dgx_spark/${SMOKE_CAMPAIGN_ID}"

usage() {
  cat <<'EOF'
Usage: scripts/dpp_ttft_weight_grid.sh COMMAND [--resource-approved]

Commands:
  preview                    Print the fixed nine-run matrix.
  smoke --resource-approved  Run one request through Stock, forced STOCK, and DPP λ=1.
  start --resource-approved  Launch the campaign in detached tmux.
  status                     Print main/smoke tmux and checkpoint status.
  resume --resource-approved Resume pending/failed append-only work.
  validate-smoke             Validate the isolated three-path smoke campaign.
  validate                   Validate all runs and print the comparison report.
EOF
}

require_python() {
  if [[ ! -x "${PYTHON}" ]]; then
    printf 'Project interpreter is missing: %s\n' "${PYTHON}" >&2
    exit 2
  fi
}

require_approval() {
  if [[ "${1:-}" != "--resource-approved" || $# -ne 1 ]]; then
    printf '%s requires exactly --resource-approved.\n' "${command}" >&2
    exit 2
  fi
}

resource_preflight() {
  local mode="$1" job_session output description workload expected bound
  local available_kib owner pid
  local -a gpu_pids=()
  if [[ "${mode}" == "smoke" ]]; then
    job_session="${SMOKE_SESSION}"
    output="${SMOKE_ROOT}"
    description="1 native Stock + 1 forced Stock-plan DPP + 1 DPP lambda=1"
    workload="three runs; one request per run; isolated one-request trace"
    expected="15-45 minutes"
    bound="90 minutes"
  else
    job_session="${SESSION}"
    output="${CAMPAIGN_ROOT}"
    description="1 native Stock + 1 forced Stock-plan DPP + 7 weighted DPP"
    workload="QPS 0.25; seed 1001; 150 requests; one shared trace"
    expected="several hours"
    bound="12 hours"
  fi
  command -v tmux
  command -v nvidia-smi
  printf 'job=%s\n' "${job_session}"
  printf 'scope=development non-formal TTFT drift-weight grid\n'
  printf 'matrix=%s\n' "${description}"
  printf 'workload=%s\n' "${workload}"
  printf 'expected_duration=%s\n' "${expected}"
  printf 'hard_campaign_bound=%s; disk_budget=5 GiB\n' "${bound}"
  printf 'network=no downloads; gpu=one GPU\n'
  printf 'output=%s\n' "${output}"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
  free -h
  df -h -- "${REPO_ROOT}"
  ps -u "$(id -un)" -o pid=,etime=,comm= | sed -n '1,40p'

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
  local session="$1" output="$2" worker_command="$3" shell_command
  if tmux has-session -t "${session}" 2>/dev/null; then
    printf 'tmux session already exists: %s\n' "${session}" >&2
    exit 1
  fi
  printf -v shell_command 'exec %q -m benchmarks.run_dpp_ttft_weight_grid %q' \
    "${PYTHON}" "${worker_command}"
  tmux new-session -d -s "${session}" -c "${REPO_ROOT}" "${shell_command}"
  printf 'Started detached tmux session: %s\n' "${session}"
  printf 'Campaign output: %s\n' "${output}"
}

require_python
command="${1:-}"
shift || true

case "${command}" in
  preview)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_dpp_ttft_weight_grid preview
    ;;
  smoke)
    require_approval "$@"
    if [[ -f "${SMOKE_ROOT}/campaign_checkpoint.json" ]]; then
      if grep -q '"status": "complete"' \
        "${SMOKE_ROOT}/campaign_checkpoint.json"; then
        printf 'Smoke campaign already passed: %s\n' "${SMOKE_ROOT}" >&2
        exit 1
      fi
      resource_preflight smoke
      launch_worker "${SMOKE_SESSION}" "${SMOKE_ROOT}" smoke-resume
    elif [[ -e "${SMOKE_ROOT}" ]]; then
      printf 'Smoke path exists without a checkpoint: %s\n' "${SMOKE_ROOT}" >&2
      exit 1
    else
      resource_preflight smoke
      launch_worker "${SMOKE_SESSION}" "${SMOKE_ROOT}" smoke-worker
    fi
    ;;
  start)
    require_approval "$@"
    if [[ ! -f "${SMOKE_ROOT}/campaign_checkpoint.json" ]] || \
      ! grep -q '"status": "complete"' \
        "${SMOKE_ROOT}/campaign_checkpoint.json"; then
      printf 'Run and validate the isolated smoke first: %s\n' "${SMOKE_ROOT}" >&2
      exit 1
    fi
    if [[ -e "${CAMPAIGN_ROOT}" ]]; then
      printf 'Append-only campaign already exists: %s\n' "${CAMPAIGN_ROOT}" >&2
      exit 1
    fi
    resource_preflight main
    launch_worker "${SESSION}" "${CAMPAIGN_ROOT}" worker
    ;;
  status)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      printf 'main_tmux_status=running session=%s\n' "${SESSION}"
    else
      printf 'main_tmux_status=not_running session=%s\n' "${SESSION}"
    fi
    "${PYTHON}" -m benchmarks.run_dpp_ttft_weight_grid status
    if tmux has-session -t "${SMOKE_SESSION}" 2>/dev/null; then
      printf 'smoke_tmux_status=running session=%s\n' "${SMOKE_SESSION}"
    else
      printf 'smoke_tmux_status=not_running session=%s\n' "${SMOKE_SESSION}"
    fi
    "${PYTHON}" -m benchmarks.run_dpp_ttft_weight_grid smoke-status
    ;;
  resume)
    require_approval "$@"
    if [[ ! -f "${CAMPAIGN_ROOT}/campaign_checkpoint.json" ]]; then
      printf 'Campaign checkpoint does not exist: %s\n' "${CAMPAIGN_ROOT}" >&2
      exit 1
    fi
    resource_preflight main
    launch_worker "${SESSION}" "${CAMPAIGN_ROOT}" resume
    ;;
  validate-smoke)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_dpp_ttft_weight_grid smoke-validate
    ;;
  validate)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_dpp_ttft_weight_grid validate
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
