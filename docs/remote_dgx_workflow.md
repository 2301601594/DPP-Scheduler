# WSL Codex to DGX Spark workflow

## Fixed endpoints and ownership

- Local source of truth: `/home/dongj/projects/LLM` in WSL.
- SSH alias: `dgx-spark` (`dongj@10.16.66.191`).
- WSL private key: `~/.ssh/id_ed25519_dgx`; it must remain outside the repository
  with mode `0600` and must never be synchronized.
- Remote source mirror: `/home/dongj/LLM` (`~/LLM`), not `~/work/LLM`.
- Remote platform: DGX Spark, ARM64/aarch64, NVIDIA GB10.
- Local platform: WSL2 x86_64; development/editing only and never a project
  execution or experimental platform.

The two machines cannot share Python environments or compiled vLLM artifacts.
The local repository is authoritative for source. The remote machine is
authoritative for DGX run outputs until they are pulled back append-only.

All project Python commands, tests, dependency setup, vLLM imports/builds,
profiling, and benchmarks run remotely. Local commands are limited to editing,
Git/source inspection, synchronization, and dependency-free static/syntax
checks.

## Agent contract

All agents working in this repository must use `scripts/remote_dgx.sh` instead
of inventing parallel SSH or rsync commands. Before changing remote state, run
`check` and `dry-run`. Never synchronize while a benchmark is running. Never
use `--delete-excluded`, and never mirror a parent directory such as `$HOME`.

Remote commands require the user's normal SSH/network approval when the Codex
sandbox requests it. Authentication must use the SSH key configured outside
the repository. Do not place private keys, Hugging Face tokens, or registry
credentials in tracked files or command output.

## Source synchronization

The repository `.rsync-filter` is the single source of truth for exclusions.
It includes source, reviewed Qwen3-14B frozen traces and Predictor artifacts
when they exist, tests, root Git metadata, and the locked `vllm` source
repository. It excludes:

- local `.venv`, `.uv-python`, caches, and agent runtime directories;
- `data/raw`, which is unnecessary when consuming frozen traces;
- every trace path except the active `traces/qwen3_14b` namespace;
- local `results` and `artifacts`;
- x86_64 vLLM dependencies and compiled objects, including the platform-specific
  `vllm-rs` executable (the remote ARM64 copy is preserved);
- transient caches and Codex-generated Git refs.

Never generate a trace or Predictor directly inside a remotely mirrored source
path. Generate it under a unique remote
`results/raw/qwen3_14b_dgx_spark/<run_id>/` staging directory, pull and review
it, then promote the immutable files locally into `traces/qwen3_14b/` or
`predictors/qwen3_14b/<version>/`. A later source push can then transfer the
reviewed copy without deleting an unpulled remote-only artifact.

Use this sequence:

```bash
./scripts/remote_dgx.sh check
./scripts/remote_dgx.sh dry-run
./scripts/remote_dgx.sh push
./scripts/remote_dgx.sh verify
```

`push` uses `--delete-delay` only inside the exact remote `~/LLM` source
mirror. Excluded remote `.venv` and `results` are protected because the script
does not use `--delete-excluded`.

`verify` performs a checksum dry-run for non-Git files and separately compares
the root and vLLM commits. Volatile internal Git data is intentionally not part
of the checksum comparison. Source synchronization also compares checksums and
ignores modification times: the precompiled editable install re-extracts many
same-content vLLM files with remote timestamps, which must not trigger repeated
transfers. Checksums still detect real content changes, including same-size
changes.

The environment verifier accepts one narrowly checked dependency-audit
exception. NVIDIA's aarch64 `nvidia-cusparselt-cu13` wheel uses the
`manylinux2014_sbsa` tag, which uv 0.11.28 reports as a different platform. The
verifier only accepts this warning after checking the exact SBSA tag, loading
the ARM64 library, and confirming PyTorch's cuSPARSELt support. Any additional
incompatibility still fails verification.

## Qwen3-14B model snapshot

The model cache is never part of daily source synchronization. The snapshot is
present at `/home/dongj/models/Qwen3-14B-BF16` (28G) and its identity is
recorded in `configs/qwen3_14b_snapshot_manifest.json`; never download, copy,
or let `vllm serve` implicitly fetch weights for any other revision.

The acquisition that produced the frozen snapshot was:

```bash
modelscope download --model Qwen/Qwen3-14B \
  --local-dir "$HOME/models/Qwen3-14B-BF16" --max-workers 1
```

It ran inside `~/modelscope-download-env` with the observed local proxy
(`HTTP_PROXY=http://127.0.0.1:17890`). The downloaded content is
byte-identical to HuggingFace `Qwen/Qwen3-14B` main commit
`40c069824f4251a91eefaf281ebe4c544efd3e18` (verified per file in the
manifest). Before transferring or re-acquiring anything else:

1. record the repository, immutable revision, license/source, expected file
   list, and total transfer/storage size in the active manifest;
2. run a remote `df` check and confirm the final path is owned by `dongj`;
3. obtain user confirmation that the group/operator approved the source and
   bulk-transfer method; and
4. use a bounded, resumable transfer that preserves snapshot links/blobs, then
   verify the manifest hashes.

Transfer only that reviewed revision and never unrelated cache contents.

## Remote environment and execution

Create the ARM64 environment on the DGX Spark; never copy the WSL `.venv`.
Use absolute executable paths in automation until the remote environment has
been captured and frozen. Short commands can be run as follows:

The pinned, user-space bootstrap is:

```bash
./scripts/remote_dgx.sh push
./scripts/remote_dgx.sh verify
./scripts/remote_dgx.sh run ./scripts/setup_dgx_vllm_env.sh prepare
```

`prepare` validates ARM64, GB10 compute capability 12.1, CUDA toolkit 13.x,
the locked vLLM commit, and at least 20 GiB free space. It installs `uv
0.11.28` under `/home/dongj/.local/bin` without changing shell startup files,
creates `/home/dongj/LLM/.venv` from `/usr/bin/python3.12`, and performs a
dependency dry-run against the exact CUDA 13 ARM64 wheel for vLLM commit
`83ad767eed3be3ee7f2df63be693bfaca5c7c922`.

The resolved environment contains about 195 packages. Expect roughly 4-6 GiB
of trusted-source downloads and 10-15 GiB for the installed environment plus
the uv cache. The actual install is guarded and must be run only after the user
confirms this bulk installation method is operator-approved:

```bash
./scripts/remote_dgx.sh run env DGX_BULK_INSTALL_CONFIRMED=1 \
  ./scripts/setup_dgx_vllm_env.sh install
./scripts/remote_dgx.sh run ./scripts/setup_dgx_vllm_env.sh verify
```

The installation is precompiled-editable: Python source remains live under
`~/LLM/vllm`, while compiled CUDA/ARM64 artifacts come from the same locked
commit. Do not substitute `nightly`, `latest`, an x86 wheel, or a different
source commit. A C++/CUDA/kernel change invalidates this Python-only install and
requires a separately reviewed full source-build plan.

The standalone compatibility smoke created a user-space CPython header tree
under remote `.uv-python`, which routine synchronization intentionally
excludes and `setup_dgx_vllm_env.sh` does not recreate. The Qwen3-14B G0 smoke
must determine whether those headers are required. If they are, add a reviewed,
pinned, user-space reconstruction step and record its source/hash/size before
freezing G0; do not silently depend on the existing directory.

After verification, use the project interpreter explicitly:

```bash
./scripts/remote_dgx.sh run .venv/bin/python -m pytest tests/unit
```

If `pytest` is not installed, do not install it implicitly; the current unit
suite also runs with the standard library:

```bash
./scripts/remote_dgx.sh run .venv/bin/python -m unittest discover -v -s tests/unit
```

Stage a candidate length-blind trace under a unique raw-results directory. QPS,
seeds, and request count remain review inputs until G0 freezes them:

```bash
./scripts/remote_dgx.sh run .venv/bin/python \
  -m benchmarks.generate_qwen3_poisson_traces \
  --output-dir <unique-trace-stage-run-id> \
  --num-requests <reviewed-count> \
  --qps <reviewed-qps...> \
  --seeds <reviewed-seeds...>
```

After pulling, reviewing, and promoting that trace into
`traces/qwen3_14b/`, update its manifest/config hash and freeze the corresponding
G0 fields. Preview the Stock runner before any real server launch:

```bash
./scripts/remote_dgx.sh run .venv/bin/python \
  -m benchmarks.run_stock_natural_eos \
  --trace <file-relative-to-traces/qwen3_14b> \
  --trace-manifest <manifest-relative-to-traces/qwen3_14b> \
  --run-id <unique-run-id> \
  --dry-run
```

Without `--dry-run`, the runner rejects a provisional active config. A real
run additionally needs the shared-host resource checks and approval below.

Long benchmarks must run under a disconnect-safe mechanism such as `tmux` or
`systemd-run --user`, while still writing unique append-only run directories
under remote `results/raw`. Run the new campaign only from the frozen
`configs/dgx_spark_experiment.yaml`; its current provisional state must be
rejected by launchers.

## Result retrieval

All project aggregation and artifact generation runs remotely. Pull the active
campaign's raw, processed, and report-artifact trees without overwriting local
history:

```bash
./scripts/remote_dgx.sh pull-results
```

`pull-results` handles only these namespaced trees:

```text
results/raw/qwen3_14b_dgx_spark/
results/processed/qwen3_14b_dgx_spark/
artifacts/qwen3_14b_dgx_spark/
```

It checks existing files by checksum, aborts on an append-only conflict, copies
only absent files, and verifies the result. Rebuild processed tables and
artifacts remotely from raw data, then pull them. If a run/build ID conflicts,
investigate it rather than overwriting either copy.

## Current G0 facts

The verified remote project is `/home/dongj/LLM`; capture the root commit and
dirty state for every run rather than copying the value from this document.
The locked vLLM commit is
`83ad767eed3be3ee7f2df63be693bfaca5c7c922`. The host runs DGX OS OTA 7.5.0,
Ubuntu 24.04.4, driver 580.159.03, CUDA toolkit 13.0 (`nvcc` V13.0.88), Python
3.12.3, and user-space `uv 0.11.28`. The 195-package project environment is
installed in `~/LLM/.venv`: PyTorch is `2.13.0+cu130`, vLLM is
`0.26.1rc1.dev535+g83ad767ee.precompiled`, both stable-libtorch extensions
import, and a GB10 CUDA tensor smoke test passes. The environment occupies
7,000,453,170 bytes and the uv cache 156,223,805 bytes. Exact package versions
are recorded in `configs/dgx_spark_environment.freeze.txt`. The Qwen3-14B BF16
snapshot is present at `/home/dongj/models/Qwen3-14B-BF16` and is
content-identical to HuggingFace main
`40c069824f4251a91eefaf281ebe4c544efd3e18`; its per-file hashes, source, and
acquisition command are recorded in
`configs/qwen3_14b_snapshot_manifest.json`. The optional user-space
Python-header tree is an observed compatibility artifact, not yet a
reconstructible Qwen3-14B environment dependency.

The provisional active configuration is `configs/dgx_spark_experiment.yaml`;
captured environment facts and explicit pending checks are in
`configs/dgx_spark_environment.json`. Neither file freezes G0. The obsolete
version-controlled 5070 campaign files were deleted; only the namespaced
Qwen3-14B inputs and outputs may be used by current tooling.
