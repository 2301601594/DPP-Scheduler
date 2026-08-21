#!/usr/bin/env bash
# Make the local (uninstalled) dpp_scheduler package importable from the
# EngineCore multiprocessing subprocess on the DGX Spark.
#
# Usage (from WSL):
#   ./scripts/remote_dgx.sh run bash scripts/setup_g2_scheduler.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SITE_PACKAGES="$("${PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"

ln -sfn "${REPO_ROOT}/dpp_scheduler" "${SITE_PACKAGES}/dpp_scheduler"
echo "linked ${REPO_ROOT}/dpp_scheduler -> ${SITE_PACKAGES}/dpp_scheduler"
