# DGX Spark Qwen3.8-27B NVFP4 smoke record

Date: 2026-08-20

Status: passed as standalone compatibility evidence for another model and
precision. The active Qwen3-14B BF16 modular DPP experiment does not consume
this record, and its configuration was not changed by the smoke.

## Outcome

- The local ModelScope snapshot hashes matched the expected revision record.
- vLLM resolved `Qwen3_5ForConditionalGeneration` and auto-detected
  `compressed-tensors` quantization.
- DGX Spark selected `FlashInferCutlassNvFp4LinearKernel`, FlashInfer attention,
  and the Triton/FLA GDN prefill kernel.
- `/v1/models` returned HTTP 200 for `qwen3.8-27b`.
- A real `/v1/chat/completions` request returned HTTP 200 and the content
  `模型运行成功` with 24 prompt tokens and 4 completion tokens.
- The dedicated tmux session was stopped after the request. No compute process
  remained, GPU utilization returned to 0%, and available system memory
  returned to 116 GiB.

The complete resolved parameters and measurements are in
`configs/dgx_spark_qwen38_nvfp4_smoke.yaml`. Remote evidence is retained at:

```text
/home/dongj/LLM/logs/qwen38-nvfp4-smoke-20260820T070822Z.log
/home/dongj/LLM/logs/qwen38-nvfp4-smoke-20260820T070822Z.response.json
```

The evidence was also pulled back to the durable local, append-only path:

```text
results/raw/dgx_spark/compatibility_smoke/qwen38_nvfp4_20260820T070822Z/
```

It is compatibility evidence only and must not be consumed by formal
performance aggregation.

## Required user-space launch environment

The system Python 3.12 installation does not contain `Python.h`. A managed
CPython 3.12.3 was installed under the project solely to provide headers:

```text
/home/dongj/LLM/.uv-python/cpython-3.12.3-linux-aarch64-gnu/include/python3.12
```

Both `.venv/bin` and that include directory must be present in the launch
environment. Omitting `CPATH` causes Triton architecture inspection to fail;
omitting `.venv/bin` from `PATH` prevents FlashInfer from finding the already
installed `ninja` executable.

## Archived resolved command

The command below records the already completed standalone smoke. It is not an
active launcher and must not be reused for the Qwen3-14B project. A future
Qwen3-14B launcher must generate its complete argv from the reviewed frozen
config rather than accept arbitrary caller-supplied model/runtime options.

```bash
./scripts/remote_dgx.sh run timeout --signal=INT --kill-after=30s 20m env \
  PATH=/home/dongj/LLM/.venv/bin:/home/dongj/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  CPATH=/home/dongj/LLM/.uv-python/cpython-3.12.3-linux-aarch64-gnu/include/python3.12 \
  MAX_JOBS=2 \
  NVCC_THREADS=1 \
  /home/dongj/LLM/.venv/bin/vllm serve \
  /home/dongj/models/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.5 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Any authorized reproduction additionally requires the current shared-host
preflight and a bounded named session. Do not leave the server unattended or
holding the GPU after use.

## Non-fatal warnings and performance boundary

The host could not verify the TLS chain for NVIDIA Artifactory. Certificate
verification was not disabled; FlashInfer compiled its kernels locally and
cached them under `/home/dongj/.cache`. The first request triggered additional
Triton JIT and took 51.89 seconds. That number is startup/JIT evidence only and
must not be reported as steady-state latency or throughput.

The vLLM B200 recipe uses tensor parallelism 2 for this Unsloth checkpoint. A
single-GPU DGX Spark instead requires tensor parallelism 1. The 32K context,
50% memory utilization, eager mode, and disabled MTP used here are conservative
smoke settings, not frozen experiment parameters.
