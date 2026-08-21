# Canonical project instructions

The authoritative instructions for every agent are in `AGENTS.md`; read that
file completely before acting. Do not duplicate or override it here.

The active research design is
`docs/Qwen3-14B-DGX-Spark-Modular-DPP-Scheduler.md`: Qwen3-14B BF16 with a
modular exact-`BatchPlan` DPP Scheduler and natural EOS. Historical campaign
configs, traces, scripts, and results are archival only.

Critical execution rule: develop in the local WSL repository, but execute all
project Python, tests, vLLM, profiling, and benchmarks only on the DGX Spark at
`dgx-spark:~/LLM` through `scripts/remote_dgx.sh`. WSL is never an active
execution or experimental platform.
