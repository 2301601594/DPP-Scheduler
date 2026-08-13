#!/usr/bin/env bash
set -euo pipefail

_qwen_server_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${_qwen_server_root}/activate-vllm.sh"
export HF_HOME="${_qwen_server_root}/.cache/huggingface"

exec "${_qwen_server_root}/.venv/bin/vllm" serve Qwen/Qwen3-0.6B \
    --served-model-name qwen3-0.6b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 \
    "$@"
