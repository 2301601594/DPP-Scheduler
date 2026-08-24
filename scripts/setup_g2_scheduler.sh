#!/usr/bin/env bash
# Make the local Scheduler and its config loader importable from the EngineCore
# multiprocessing subprocess on the DGX Spark.
#
# Usage (from WSL):
#   ./scripts/remote_dgx.sh run bash scripts/setup_g2_scheduler.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SITE_PACKAGES="$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"

ln -sfnT "${REPO_ROOT}/dpp_scheduler" "${SITE_PACKAGES}/dpp_scheduler"

# vLLM already ships an unrelated namespace-style benchmarks directory. Link
# only the config loader required by ModularDPPScheduler; never replace that
# directory or put the repository root on PYTHONPATH (which shadows editable
# vLLM itself).
BENCHMARKS_DIR="${SITE_PACKAGES}/benchmarks"
mkdir -p "${BENCHMARKS_DIR}"
ACCIDENTAL_LINK="${BENCHMARKS_DIR}/benchmarks"
if [[ -L "${ACCIDENTAL_LINK}" ]]; then
  if [[ "$(readlink -f "${ACCIDENTAL_LINK}")" != "${REPO_ROOT}/benchmarks" ]]; then
    echo "unexpected benchmarks link: ${ACCIDENTAL_LINK}" >&2
    exit 1
  fi
  unlink "${ACCIDENTAL_LINK}"
elif [[ -e "${ACCIDENTAL_LINK}" ]]; then
  echo "refusing to replace existing path: ${ACCIDENTAL_LINK}" >&2
  exit 1
fi
ln -sfnT \
  "${REPO_ROOT}/benchmarks/qwen3_runtime.py" \
  "${BENCHMARKS_DIR}/qwen3_runtime.py"
echo "linked ${REPO_ROOT}/dpp_scheduler -> ${SITE_PACKAGES}/dpp_scheduler"
echo "linked ${REPO_ROOT}/benchmarks/qwen3_runtime.py -> ${BENCHMARKS_DIR}/qwen3_runtime.py"
