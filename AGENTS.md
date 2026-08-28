# AGENTS.md — repository operating rules

## 1. Purpose and scope

This file applies to the whole repository. A nearer `AGENTS.md` or
`AGENTS.override.md` may add directory-specific rules.

Keep this file limited to agent operating rules and navigation. Project scope,
architecture, algorithms, interfaces, experiment parameters, gate criteria, and
current research status belong in the documents indexed below. Do not copy
those details back into this file.

`Must` means a task cannot be declared complete if the rule is unmet. Apply
the workflow proportionally: a documentation edit does not require an
experiment preflight, while a benchmark or scientific claim does.

## 2. Project document index

Read only the documents relevant to the task:

| Task or question | Authoritative source |
| --- | --- |
| Baseline Scheduler scope, architecture, public contracts, algorithms, and acceptance criteria | [`docs/Qwen3-14B-DGX-Spark-Modular-DPP-Scheduler.md`](docs/Qwen3-14B-DGX-Spark-Modular-DPP-Scheduler.md) |
| Request-level Service-Deficit DPP v2 changes, formulas, implementation phases, and tests | [`docs/Request-level-Service-Deficit-DPP-v2-Agent-Modification-Plan.md`](docs/Request-level-Service-Deficit-DPP-v2-Agent-Modification-Plan.md) |
| Current two-stage TBT-constrained Prefill service-rate Selector, diagnosis, replay, and tests | [`docs/Two-Stage-TBT-Constrained-Prefill-Service-Rate-Selector-V1.md`](docs/Two-Stage-TBT-Constrained-Prefill-Service-Rate-Selector-V1.md) |
| Current Candidate Generator V3 budget search, Prefill allocation policies, and diagnostics | [`docs/Candidate-Generator-V3.md`](docs/Candidate-Generator-V3.md) |
| Current stage, G0–G7 gates, evidence requirements, and experiment milestones | [`docs/experiment_plan.md`](docs/experiment_plan.md) |
| Frozen or superseding research decisions and their rationale | [`docs/decisions.md`](docs/decisions.md) |
| Local/remote responsibilities, synchronization, DGX environment, execution, and result retrieval | [`docs/remote_dgx_workflow.md`](docs/remote_dgx_workflow.md) |
| Chronological run history, failures, and observed results | [`docs/qwen3_14b_experiment_log.md`](docs/qwen3_14b_experiment_log.md) |
| Active runtime and conclusion-affecting parameters | [`configs/dgx_spark_experiment.yaml`](configs/dgx_spark_experiment.yaml) and its referenced frozen manifests/artifacts |
| Result namespaces, retention, and cleanup rules | [`results/README.md`](results/README.md) |

The Candidate Generator V3 document takes precedence for that component. The
Prefill service-rate Selector V1 document takes precedence for Selector, Selector settings,
diagnosis, replay, and their tests. The v2 modification plan takes precedence
over the baseline design only for other components it explicitly changes. Use
the baseline design for everything else.
Do not infer current status from a design document; use the experiment plan,
decisions, active configuration, and recorded artifacts.

When instructions conflict, use this order:

1. the user's current explicit request;
2. reviewed frozen configurations, manifests, and artifacts;
3. the Candidate Generator V3 contract and two-stage Selector contract for
   their components, then the applicable v2 modification plan, then the
   unaffected baseline design;
4. active decisions and the experiment plan;
5. this repository-level operating guide; and
6. external papers or upstream documentation.

Never resolve a substantive conflict silently. Record a new research choice in
`docs/decisions.md` and update the affected design or experiment document.

## 3. Routine workflow

- Start with `git status --short` and inspect the directly relevant files.
- Preserve unrelated user changes. Keep edits small and reviewable, and reuse
  existing scripts, contracts, and artifact formats.
- Use `rg` or `rg --files` for repository search.
- Read the active design/configuration only when the change affects runtime
  behavior, Scheduler contracts, runners, manifests, experiments, readiness,
  or claims. Use the locked vLLM source for version-specific behavior.
- Run the smallest useful check. Documentation-only changes need local text or
  diff checks; project Python, tests, vLLM imports, profiling, and benchmarks
  must run remotely as described below.
- Do not commit, publish, open a PR, change the locked vLLM revision, install
  dependencies, acquire models/data, or start a resource-intensive job unless
  the user explicitly authorizes it.
- Do not update decisions, logs, gates, or claims for routine work unless the
  task produces material evidence that changes them.

## 4. Local and remote execution boundary

The local WSL repository `/home/dongj/projects/LLM` is for source editing,
review, Git inspection, text processing, and dependency-free static or syntax
checks. All project execution occurs as user `dongj` on
`dgx-spark:/home/dongj/LLM`.

Before remote execution, synchronization, or result retrieval, read the
relevant section of `docs/remote_dgx_workflow.md` and use
`scripts/remote_dgx.sh`. Do not invent alternate SSH or rsync workflows.
Before a source push, run its `check` and `dry-run` preflight, inspect the exact
target, then `push` and `verify`. Do not push unchanged source as ceremony or
synchronize while a benchmark is running.

Never transfer local environments, caches, compiled artifacts, credentials, or
model caches through routine source synchronization. Model/data acquisition
and bulk transfer are separate reviewed operations.

## 5. Shared-host and destructive-action safety

The DGX Spark is a multi-user research workstation. Operate only through the
`dongj` account and its owned paths unless the user and operator explicitly
authorize a named shared resource.

- Never use `sudo`, administrator/shared accounts, privilege escalation, or
  modify system configuration, drivers, services, global networking, shared
  software, or another user's files or processes.
- Never expose credentials in the repository, logs, chat, or command output;
  bypass security controls; disable TLS verification; or work around an
  operator restriction.
- Resolve exact targets before destructive actions. Never apply recursive or
  synchronization delete semantics to `$HOME`, `~/`, a repository root, or
  another broad directory. Follow `results/README.md` for result cleanup.
- Before a large download or transfer, verify the trusted source, immutable
  revision or checksum, expected size, available disk, destination ownership,
  and user confirmation of operator approval.
- Before a long or resource-intensive job, perform read-only GPU, process,
  memory, and disk checks; report the bound, expected duration, resources, and
  output path; and obtain confirmation that shared-host use is approved. Never
  stop a workload whose ownership is unclear.
- Run long jobs in a bounded, named, disconnect-safe session and clean it up
  when finished. Do not leave an idle GPU-holding or unbounded process.
- If progress would require a system or shared-host change, preserve the error
  and ask the user to contact operator 李文轩.

## 6. Research integrity and artifact handling

- Every conclusion-affecting input must be version controlled or referenced by
  an immutable manifest. Unmeasured or unreviewed values remain `null`,
  `pending`, or `provisional`; never substitute an undocumented default or a
  value from another campaign.
- Preserve append-only raw evidence, unique run identities, failed/invalid
  runs, negative results, null results, and all required outcome strata. Never
  change prompts, traces, arrivals, SLOs, baselines, seeds, filtering, or
  failure handling to improve the apparent result.
- Do not fabricate measurements or present scaffolding, diagnostics, or an
  integration-only parameter as formal evidence.
- Historical or obsolete campaign data is not an active input unless it is
  explicitly revalidated and frozen for the current campaign. Do not combine
  campaigns implicitly.
- Keep Scheduler-visible inputs length-blind. Future EOS position, fixed
  expected output length, or remaining output tokens must not enter Scheduler
  state, Predictor data, or decisions. Preserve client-guard terminations as
  recorded outcomes rather than filtering them away. Detailed semantics live
  in the active design and decisions.
- Build processed results and claims reproducibly from retained raw data.
  Separate observed facts, data-supported inferences, and unverified
  hypotheses whenever an analysis mixes them.

## 7. Completion and reporting

Before declaring completion, inspect the final diff and run the smallest check
appropriate to the change. Broader remote suites and gate evidence are required
only when the indexed experiment plan or design makes them relevant.

Report the outcome, checks performed, and material limitations. Use expanded,
reproducible reporting for experiments, gate transitions, frozen scientific
inputs, invalid runs, or destructive cleanup; keep routine maintenance reports
concise.
