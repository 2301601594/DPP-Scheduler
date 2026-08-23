#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SESSION="candidate-knee-profile-isolated-v2"
CAMPAIGN_ROOT="${REPO_ROOT}/results/raw/qwen3_14b_dgx_spark/candidate_knee_profile_isolated_v2"

usage() {
  cat <<'EOF'
Usage: scripts/candidate_knee_profile_campaign.sh COMMAND [--resource-approved]

Commands:
  preview                    Print the fixed one-run, 2160-cell matrix.
  smoke --resource-approved  Run a bounded two-batch isolation smoke.
  start --resource-approved  Launch the knee campaign in detached tmux.
  status                     Print checkpoint and tmux status.
  resume --resource-approved Resume within the two-attempt bound.
  validate                   Validate the completed run.
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
  printf 'expected_duration=4-8 hours; hard campaign bound=10 hours\n'
  printf 'gpu=one GPU; disk_budget=5 GiB; network=no downloads\n'
  printf 'output=%s\n' "${CAMPAIGN_ROOT}"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
  free -h
  df -h -- "${REPO_ROOT}"

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
  printf -v shell_command 'exec %q -m benchmarks.run_candidate_knee_profile_campaign %q' \
    "${PYTHON}" "${worker_command}"
  tmux new-session -d -s "${SESSION}" -c "${REPO_ROOT}" "${shell_command}"
  printf 'Started detached tmux session: %s\n' "${SESSION}"
  printf 'Campaign output: %s\n' "${CAMPAIGN_ROOT}"
}

require_python
command="${1:-}"
shift || true

case "${command}" in
  preview)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_candidate_knee_profile_campaign preview
    ;;
  smoke)
    require_approval_flag "$@"
    resource_preflight
    "${PYTHON}" -m benchmarks.run_candidate_knee_profile_campaign smoke
    ;;
  start)
    require_approval_flag "$@"
    if [[ -e "${CAMPAIGN_ROOT}" ]]; then
      printf 'Append-only campaign already exists; use resume or validate: %s\n' \
        "${CAMPAIGN_ROOT}" >&2
      exit 1
    fi
    resource_preflight
    launch_worker worker
    ;;
  status)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      printf 'tmux_status=running session=%s\n' "${SESSION}"
    else
      printf 'tmux_status=not_running session=%s\n' "${SESSION}"
    fi
    "${PYTHON}" -m benchmarks.run_candidate_knee_profile_campaign status
    ;;
  resume)
    require_approval_flag "$@"
    if [[ ! -f "${CAMPAIGN_ROOT}/campaign_checkpoint.json" ]]; then
      printf 'Campaign checkpoint does not exist: %s\n' "${CAMPAIGN_ROOT}" >&2
      exit 1
    fi
    resource_preflight
    launch_worker resume
    ;;
  validate)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_candidate_knee_profile_campaign validate
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
