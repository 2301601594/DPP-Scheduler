#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SESSION="predictor-mixed-three-segment-independent-n200-v1"
CAMPAIGN_ROOT="${REPO_ROOT}/results/raw/qwen3_14b_dgx_spark/predictor_mixed_decode_three_segment_independent_n200_v1"

usage() {
  cat <<'EOF'
Usage: scripts/segmented_predictor_validation_campaign.sh COMMAND [--resource-approved]

Commands:
  prepare                    Generate the fixed fresh-seed 200-request trace.
  preview                    Preview artifact, seeds, run, and output identity.
  smoke --resource-approved  Run the bounded 10-request shadow smoke.
  start --resource-approved  Launch the n=200 validation in detached tmux.
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
    printf '%s requires exactly --resource-approved after group notification.\n' \
      "${command}" >&2
    exit 2
  fi
}

resource_preflight() {
  local available_kib
  local owner
  local pid
  local -a gpu_pids=()

  command -v tmux
  command -v nvidia-smi
  printf 'job=%s\n' "${SESSION}"
  printf 'expected_duration=20-40 minutes; hard run bound=3 hours\n'
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

require_python
command="${1:-}"
shift || true

case "${command}" in
  prepare)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_segmented_predictor_validation_campaign prepare
    ;;
  preview)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_segmented_predictor_validation_campaign preview
    ;;
  smoke)
    require_approval "$@"
    resource_preflight
    "${PYTHON}" -m benchmarks.run_segmented_predictor_validation_campaign smoke
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
    printf -v shell_command 'exec %q -m benchmarks.run_segmented_predictor_validation_campaign worker' \
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
    "${PYTHON}" -m benchmarks.run_segmented_predictor_validation_campaign status
    ;;
  validate)
    if (( $# != 0 )); then usage >&2; exit 2; fi
    "${PYTHON}" -m benchmarks.run_segmented_predictor_validation_campaign validate
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
