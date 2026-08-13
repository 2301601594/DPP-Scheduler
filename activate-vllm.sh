#!/usr/bin/env bash

_vllm_env_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${_vllm_env_root}/.venv/bin/activate"

# Keep CUDA and FlashAttention compilation within WSL's memory budget.
export MAX_JOBS=2
export NVCC_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=2

# Build only for the installed RTX 5070 (Blackwell, compute capability 12.0).
export TORCH_CUDA_ARCH_LIST=12.0

# Model Runner V2 requires CUDA UVA, which is unavailable in this WSL setup.
export VLLM_USE_V2_MODEL_RUNNER=0

# FlashInfer's JIT build does not escape spaces in this workspace path.
export VLLM_USE_FLASHINFER_SAMPLER=1

# The default user cache is read-only in this workspace.
export UV_CACHE_DIR=/tmp/vllm-uv-cache

unset _vllm_env_root
