#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SESSION="predictor-online-timing-aligned-n200-v1"
CAMPAIGN_ROOT="${REPO_ROOT}/results/raw/qwen3_14b_dgx_spark/predictor_online_timing_aligned_n200_v1"

usage() {
  cat <<'EOF'
Usage: scripts/predictor_online_calibration_eval.sh COMMAND [--resource-approved]

Commands:
  prepare                    Generate the fixed 200-request source trace.
  preview                    Preview artifact, trace, run, and output identity.
  smoke --resource-approved  Run the bounded 10-request shadow smoke.
  start --resource-approved  Launch the one-run evaluation in detached tmux.
  status                     Print tmux and checkpoint status.
  validate                   Validate the completed evaluation.
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
  local available_kib
  local gpu_process_count
  command -v tmux
  command -v nvidia-smi
  printf 'job=%s\n' "${SESSION}"
  printf 'expected_duration=up to 3 hours; hard run bound=3 hours\n'
  printf 'gpu=one GPU; disk_budget=5 GiB; network=no downloads\n'
  printf 'output=%s\n' "${CAMPAIGN_ROOT}"
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
  free -h
  df -h -- "${REPO_ROOT}"
  available_kib="$(df -Pk -- "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
  if [[ ! "${available_kib}" =~ ^[0-9]+$ ]] || (( available_kib < 5 * 1024 * 1024 )); then
    printf 'At least 5 GiB free disk is required.\n' >&2
    exit 1
  fi
  gpu_process_count="$({ nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true; } | awk '$1 ~ /^[0-9]+$/ {n++} END {print n+0}')"
  if (( gpu_process_count > 0 )); then
    printf 'Active GPU compute workload detected; refusing to overlap.\n' >&2
    exit 1
  fi
}

require_python
command="${1:-}"
shift || true

case "${command}" in
  prepare)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_predictor_online_evaluation_campaign prepare
    ;;
  preview)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_predictor_online_evaluation_campaign preview
    ;;
  smoke)
    require_approval "$@"
    resource_preflight
    "${PYTHON}" -m benchmarks.run_predictor_online_evaluation_campaign smoke
    ;;
  start)
    require_approval "$@"
    if [[ -e "${CAMPAIGN_ROOT}/campaign_checkpoint.json" ]]; then
      printf 'Append-only campaign checkpoint already exists: %s\n' \
        "${CAMPAIGN_ROOT}/campaign_checkpoint.json" >&2
      exit 1
    fi
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      printf 'tmux session already exists: %s\n' "${SESSION}" >&2
      exit 1
    fi
    resource_preflight
    printf -v shell_command 'exec %q -m benchmarks.run_predictor_online_evaluation_campaign worker' \
      "${PYTHON}"
    tmux new-session -d -s "${SESSION}" -c "${REPO_ROOT}" "${shell_command}"
    printf 'Started detached tmux session: %s\n' "${SESSION}"
    ;;
  status)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      printf 'tmux_status=running session=%s\n' "${SESSION}"
    else
      printf 'tmux_status=not_running session=%s\n' "${SESSION}"
    fi
    "${PYTHON}" -m benchmarks.run_predictor_online_evaluation_campaign status
    ;;
  validate)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_predictor_online_evaluation_campaign validate
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
