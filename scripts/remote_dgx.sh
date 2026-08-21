#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DGX_REMOTE_HOST="${DGX_REMOTE_HOST:-dgx-spark}"
DGX_REMOTE_PROJECT="${DGX_REMOTE_PROJECT:-LLM}"

if [[ "${DGX_REMOTE_PROJECT}" != "LLM" ]]; then
  printf 'DGX_REMOTE_PROJECT is fixed to LLM; refusing unsafe target: %s\n' \
    "${DGX_REMOTE_PROJECT}" >&2
  exit 2
fi

usage() {
  cat <<'EOF'
Usage: scripts/remote_dgx.sh COMMAND [ARGS...]

Commands:
  check                Check SSH, remote platform, tools, and repository commits.
  dry-run              Preview the local-to-remote source mirror.
  push                 Mirror source to ~/LLM using the repository filter.
  verify               Check file contents and Git commits without modifying either side.
  run COMMAND [ARGS]   Run a short command from the remote project directory.
  pull-results         Pull active raw/processed/artifact trees append-only and verify them.

Optional overrides:
  DGX_REMOTE_HOST      SSH alias (default: dgx-spark)
  DGX_REMOTE_PROJECT   Must remain exactly LLM
EOF
}

remote_project_shell_path() {
  printf '$HOME/%s' "${DGX_REMOTE_PROJECT}"
}

ensure_remote_project() {
  ssh -o BatchMode=yes "${DGX_REMOTE_HOST}" \
    "mkdir -p -- \"\$HOME/${DGX_REMOTE_PROJECT}\""
}

rsync_source() {
  local mode="$1"
  local -a extra_args
  if [[ "${mode}" == "dry-run" ]]; then
    extra_args=(--dry-run --itemize-changes)
  else
    extra_args=(--info=progress2)
  fi

  ensure_remote_project
  (
    cd -- "${REPO_ROOT}"
    rsync -a \
      --checksum \
      --no-times \
      --partial \
      --human-readable \
      --delete-delay \
      --filter='merge .rsync-filter' \
      "${extra_args[@]}" \
      ./ "${DGX_REMOTE_HOST}:${DGX_REMOTE_PROJECT}/"
  )
}

verify_mirror() {
  local differences
  local local_main_commit
  local local_vllm_commit
  local remote_commits
  local remote_main_commit
  local remote_vllm_commit

  local_main_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  local_vllm_commit="$(git -C "${REPO_ROOT}/vllm" rev-parse HEAD)"
  remote_commits="$(
    ssh -o BatchMode=yes "${DGX_REMOTE_HOST}" \
      "git -C \"\$HOME/${DGX_REMOTE_PROJECT}\" rev-parse HEAD && git -C \"\$HOME/${DGX_REMOTE_PROJECT}/vllm\" rev-parse HEAD"
  )"
  remote_main_commit="$(sed -n '1p' <<<"${remote_commits}")"
  remote_vllm_commit="$(sed -n '2p' <<<"${remote_commits}")"

  differences="$(
    cd -- "${REPO_ROOT}"
    rsync -acnO \
      --no-times \
      --delete-delay \
      --itemize-changes \
      --filter='merge .rsync-filter' \
      --exclude='/.git/' \
      --exclude='/vllm/.git/' \
      ./ "${DGX_REMOTE_HOST}:${DGX_REMOTE_PROJECT}/"
  )"

  if [[ -n "${differences}" ]]; then
    printf 'Mirror differences detected:\n%s\n' "${differences}" >&2
    return 1
  fi
  if [[ "${local_main_commit}" != "${remote_main_commit}" ]]; then
    printf 'Main commit mismatch: local=%s remote=%s\n' \
      "${local_main_commit}" "${remote_main_commit}" >&2
    return 1
  fi
  if [[ "${local_vllm_commit}" != "${remote_vllm_commit}" ]]; then
    printf 'vLLM commit mismatch: local=%s remote=%s\n' \
      "${local_vllm_commit}" "${remote_vllm_commit}" >&2
    return 1
  fi

  printf 'Mirror verified: files and commits match.\n'
  printf '  main: %s\n' "${local_main_commit}"
  printf '  vLLM: %s\n' "${local_vllm_commit}"
}

pull_active_tree() {
  local relative_path="$1"
  local local_path
  local remote_path
  local conflicts
  local remaining

  case "${relative_path}" in
    results/raw/qwen3_14b_dgx_spark|\
    results/processed/qwen3_14b_dgx_spark|\
    artifacts/qwen3_14b_dgx_spark)
      ;;
    *)
      printf 'Refusing unapproved pull path: %s\n' "${relative_path}" >&2
      return 2
      ;;
  esac

  local_path="${REPO_ROOT}/${relative_path}"
  remote_path="$HOME/${DGX_REMOTE_PROJECT}/${relative_path}"
  if ! ssh -o BatchMode=yes "${DGX_REMOTE_HOST}" \
    "test -d \"${remote_path}\""; then
    printf 'Remote active output is absent; skipped: %s\n' "${relative_path}"
    return
  fi

  mkdir -p -- "${local_path}"
  conflicts="$(
    rsync -rcn \
      --existing \
      --no-times \
      --omit-dir-times \
      --itemize-changes \
      "${DGX_REMOTE_HOST}:${DGX_REMOTE_PROJECT}/${relative_path}/" \
      "${local_path}/"
  )"
  if [[ -n "${conflicts}" ]]; then
    printf 'Append-only pull conflict in %s:\n%s\n' \
      "${relative_path}" "${conflicts}" >&2
    return 1
  fi

  rsync -a \
    --ignore-existing \
    --human-readable \
    --info=progress2 \
    "${DGX_REMOTE_HOST}:${DGX_REMOTE_PROJECT}/${relative_path}/" \
    "${local_path}/"

  remaining="$(
    rsync -rcn \
      --no-times \
      --omit-dir-times \
      --itemize-changes \
      "${DGX_REMOTE_HOST}:${DGX_REMOTE_PROJECT}/${relative_path}/" \
      "${local_path}/"
  )"
  if [[ -n "${remaining}" ]]; then
    printf 'Pulled tree still differs for %s:\n%s\n' \
      "${relative_path}" "${remaining}" >&2
    return 1
  fi
  printf 'Pulled and checksum-verified: %s\n' "${relative_path}"
}

command="${1:-}"
case "${command}" in
  check)
    remote_path="$(remote_project_shell_path)"
    ssh -o BatchMode=yes "${DGX_REMOTE_HOST}" \
      "set -eu
       printf 'remote_user=%s\n' \"\$(id -un)\"
       printf 'remote_host=%s\n' \"\$(hostname)\"
       printf 'remote_arch=%s\n' \"\$(uname -m)\"
       printf 'remote_project=%s\n' \"${remote_path}\"
       command -v rsync
       rsync --version | head -n 1
       python3 --version
       nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
       test -d \"\$HOME/${DGX_REMOTE_PROJECT}\"
       printf 'main_commit=%s\n' \"\$(git -C \"\$HOME/${DGX_REMOTE_PROJECT}\" rev-parse HEAD)\"
       printf 'vllm_commit=%s\n' \"\$(git -C \"\$HOME/${DGX_REMOTE_PROJECT}/vllm\" rev-parse HEAD)\""
    ;;
  dry-run)
    rsync_source dry-run
    ;;
  push)
    rsync_source push
    ;;
  verify)
    verify_mirror
    ;;
  run)
    shift
    if (( $# == 0 )); then
      printf 'run requires a command.\n' >&2
      usage >&2
      exit 2
    fi
    printf -v quoted_command '%q ' "$@"
    ssh -o BatchMode=yes "${DGX_REMOTE_HOST}" \
      "cd \"\$HOME/${DGX_REMOTE_PROJECT}\" && ${quoted_command}"
    ;;
  pull-results)
    pull_active_tree results/raw/qwen3_14b_dgx_spark
    pull_active_tree results/processed/qwen3_14b_dgx_spark
    pull_active_tree artifacts/qwen3_14b_dgx_spark
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
