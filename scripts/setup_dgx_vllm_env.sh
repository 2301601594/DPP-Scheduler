#!/usr/bin/env bash
set -euo pipefail

# This script is intended to run on the remote DGX Spark only. It never uses
# sudo and keeps the executable, cache, virtual environment, and package files
# under the dongj account.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VLLM_SOURCE="${PROJECT_ROOT}/vllm"
VENV_PATH="${PROJECT_ROOT}/.venv"

UV_VERSION="0.11.28"
UV_BIN="/home/dongj/.local/bin/uv"
UV_INSTALL_URL="https://astral.sh/uv/${UV_VERSION}/install.sh"
UV_INSTALL_SHA256="b7b3fe80cad1142a2a5794050b7db7b3291d1bac1423b0732571dd9366e8ca8b"
UV_CACHE_DIR="/home/dongj/.cache/uv"
NVCC_BIN="/usr/local/cuda/bin/nvcc"

VLLM_COMMIT="83ad767eed3be3ee7f2df63be693bfaca5c7c922"
VLLM_VERSION="0.26.1rc1.dev535+g83ad767ee"
VLLM_WHEEL_VARIANT="cu130"
VLLM_WHEEL_URL="https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-${VLLM_VERSION//+/%2B}-cp38-abi3-manylinux_2_28_aarch64.whl"

usage() {
  cat <<'EOF'
Usage: scripts/setup_dgx_vllm_env.sh COMMAND

Commands:
  preflight      Validate DGX architecture, CUDA, source commit, and free space.
  bootstrap-uv   Install pinned uv under /home/dongj/.local/bin if absent.
  create-venv    Create the project-local Python 3.12 environment if absent.
  dry-run        Resolve the pinned ARM64/CUDA 13 vLLM wheel and dependencies.
  install        Install the editable vLLM environment after explicit approval.
  verify         Import torch/vLLM and run a tiny CUDA operation.
  prepare        Run preflight, bootstrap-uv, create-venv, and dry-run.
  all            Run prepare, install, and verify.

The install and all commands require:
  DGX_BULK_INSTALL_CONFIRMED=1

Set it only after confirming the 4-6 GiB trusted-source download and the
roughly 10-15 GiB environment-plus-cache footprint are operator-approved.
EOF
}

require_remote_dgx() {
  local architecture
  local compute_capability
  local source_commit
  local cuda_release
  local available_kib

  architecture="$(uname -m)"
  [[ "${architecture}" == "aarch64" ]] || {
    printf 'Expected aarch64 DGX Spark; found %s.\n' "${architecture}" >&2
    return 1
  }

  [[ "$(id -un)" == "dongj" ]] || {
    printf 'This environment must be owned by the dongj account.\n' >&2
    return 1
  }

  command -v nvidia-smi >/dev/null
  [[ -x "${NVCC_BIN}" ]] || {
    printf 'Required CUDA compiler %s is unavailable.\n' "${NVCC_BIN}" >&2
    return 1
  }
  command -v git >/dev/null
  command -v curl >/dev/null
  command -v sha256sum >/dev/null
  [[ -x /usr/bin/python3.12 ]] || {
    printf 'Required interpreter /usr/bin/python3.12 is unavailable.\n' >&2
    return 1
  }

  compute_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')"
  [[ "${compute_capability}" == "12.1" ]] || {
    printf 'Expected GB10 compute capability 12.1; found %s.\n' \
      "${compute_capability}" >&2
    return 1
  }

  cuda_release="$("${NVCC_BIN}" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
  [[ "${cuda_release}" == 13.* ]] || {
    printf 'Expected CUDA toolkit 13.x; found %s.\n' "${cuda_release}" >&2
    return 1
  }

  source_commit="$(git -C "${VLLM_SOURCE}" rev-parse HEAD)"
  [[ "${source_commit}" == "${VLLM_COMMIT}" ]] || {
    printf 'vLLM source mismatch: expected %s, found %s.\n' \
      "${VLLM_COMMIT}" "${source_commit}" >&2
    return 1
  }

  available_kib="$(df -Pk "${PROJECT_ROOT}" | awk 'NR == 2 {print $4}')"
  if (( available_kib < 20 * 1024 * 1024 )); then
    printf 'At least 20 GiB free is required; df reports %s KiB.\n' \
      "${available_kib}" >&2
    return 1
  fi

  printf 'DGX preflight passed.\n'
  printf '  project=%s\n' "${PROJECT_ROOT}"
  printf '  vllm_commit=%s\n' "${source_commit}"
  printf '  cuda_toolkit=%s\n' "${cuda_release}"
  printf '  compute_capability=%s\n' "${compute_capability}"
  printf '  free_disk_gib=%s\n' "$((available_kib / 1024 / 1024))"
}

bootstrap_uv() {
  local installer
  local installed_uv_version

  if [[ -x "${UV_BIN}" ]]; then
    installed_uv_version="$("${UV_BIN}" --version)"
    [[ "${installed_uv_version}" == "uv ${UV_VERSION}"* ]] || {
      printf 'Expected %s, found %s. Refusing an implicit uv upgrade/downgrade.\n' \
        "uv ${UV_VERSION}" "${installed_uv_version}" >&2
      return 1
    }
    printf 'Pinned uv already installed: %s\n' "${installed_uv_version}"
    return
  fi

  installer="$(mktemp /tmp/uv-install-dongj.XXXXXX.sh)"
  trap 'rm -f -- "${installer}"' RETURN
  curl -LsSf "${UV_INSTALL_URL}" -o "${installer}"
  printf '%s  %s\n' "${UV_INSTALL_SHA256}" "${installer}" | sha256sum -c -
  env UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR=/home/dongj/.local/bin \
    sh "${installer}"
  installed_uv_version="$("${UV_BIN}" --version)"
  [[ "${installed_uv_version}" == "uv ${UV_VERSION}"* ]]
}

create_venv() {
  [[ -x "${UV_BIN}" ]] || {
    printf 'Pinned uv is absent; run bootstrap-uv first.\n' >&2
    return 1
  }

  if [[ -x "${VENV_PATH}/bin/python" ]]; then
    "${VENV_PATH}/bin/python" -c \
      'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
    printf 'Project environment already exists: %s\n' "${VENV_PATH}"
    return
  fi

  "${UV_BIN}" venv --python /usr/bin/python3.12 "${VENV_PATH}"
}

resolve_dependencies() {
  [[ -x "${VENV_PATH}/bin/python" ]] || {
    printf 'Project environment is absent; run create-venv first.\n' >&2
    return 1
  }

  UV_CACHE_DIR="${UV_CACHE_DIR}" "${UV_BIN}" pip install \
    --python "${VENV_PATH}/bin/python" \
    --dry-run \
    --torch-backend=cu130 \
    "${VLLM_WHEEL_URL}"
}

install_environment() {
  [[ "${DGX_BULK_INSTALL_CONFIRMED:-0}" == "1" ]] || {
    printf 'Refusing bulk install without DGX_BULK_INSTALL_CONFIRMED=1.\n' >&2
    printf 'Confirm operator approval after reviewing the size estimate.\n' >&2
    return 2
  }
  [[ -x "${VENV_PATH}/bin/python" ]]

  env \
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_COMMIT}" \
    VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_WHEEL_VARIANT}" \
    "${UV_BIN}" pip install \
      --python "${VENV_PATH}/bin/python" \
      --editable "${VLLM_SOURCE}" \
      --torch-backend=cu130
}

verify_environment() {
  local dependency_check

  [[ -x "${VENV_PATH}/bin/python" ]]
  "${VENV_PATH}/bin/python" - <<'PY'
import ctypes
import platform
from importlib.metadata import distribution, version

import torch
import vllm._C_stable_libtorch  # noqa: F401 -- verifies CUDA extension
import vllm._moe_C_stable_libtorch  # noqa: F401 -- verifies MoE extension

assert platform.machine() == "aarch64", platform.machine()
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert torch.cuda.get_device_capability(0) == (12, 1)
vllm_version = version("vllm")
assert vllm_version.startswith("0.26.1rc1.dev535+g83ad767ee")

# NVIDIA publishes the aarch64 cuSPARSELt wheel with the SBSA platform tag.
# uv 0.11.28 does not consider that tag compatible with manylinux aarch64,
# so validate the installed artifact directly before accepting that one known
# `uv pip check` warning below.
cusparselt = distribution("nvidia-cusparselt-cu13")
wheel_metadata = cusparselt.read_text("WHEEL") or ""
assert "Tag: py3-none-manylinux2014_sbsa" in wheel_metadata
cusparselt_library = cusparselt.locate_file(
    "nvidia/cusparselt/lib/libcusparseLt.so.0"
)
ctypes.CDLL(str(cusparselt_library))
assert torch._C._has_cusparselt

x = torch.tensor([1.0, 2.0], device="cuda")
assert x.sum().item() == 3.0
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"vllm={vllm_version}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"compute_capability={torch.cuda.get_device_capability(0)}")
print("cusparselt_load=passed")
print("cuda_smoke=passed")
PY

  if dependency_check="$(
    "${UV_BIN}" pip check --python "${VENV_PATH}/bin/python" 2>&1
  )"; then
    printf '%s\n' "${dependency_check}"
    printf 'dependency_check=passed\n'
  elif [[ "${dependency_check}" == *"Found 1 incompatibility"* ]] &&
       [[ "${dependency_check}" == *'The package `nvidia-cusparselt-cu13` was built for a different platform'* ]]; then
    printf '%s\n' "${dependency_check}"
    printf 'dependency_check=passed_with_verified_sbsa_tag_exception\n'
  else
    printf '%s\n' "${dependency_check}" >&2
    return 1
  fi
}

command="${1:-}"
case "${command}" in
  preflight)
    require_remote_dgx
    ;;
  bootstrap-uv)
    require_remote_dgx
    bootstrap_uv
    ;;
  create-venv)
    require_remote_dgx
    create_venv
    ;;
  dry-run)
    require_remote_dgx
    create_venv
    resolve_dependencies
    ;;
  install)
    require_remote_dgx
    create_venv
    install_environment
    ;;
  verify)
    require_remote_dgx
    verify_environment
    ;;
  prepare)
    require_remote_dgx
    bootstrap_uv
    create_venv
    resolve_dependencies
    ;;
  all)
    require_remote_dgx
    bootstrap_uv
    create_venv
    resolve_dependencies
    install_environment
    verify_environment
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
