# Two-Stage ZERO-Relative TBT Prefill Service-Rate Selector V2-B

本文档定义当前 Selector、Selector 配置、Diagnosis、Replay 和对应测试的权威
契约。它 supersede `Two-Stage-TBT-Constrained-Prefill-Service-Rate-Selector-V1.md`
的同类内容，但不改写其中保留的 min-slack 研究过程和负面证据，也不改写
`Stage1-V2A-ZERO-Relative-TBT-Replay.md` 的离线 replay 结论。

## 1. 范围与冻结边界

算法标识固定为 `two_stage_zero_relative_tbt_prefill_service_rate_v2b`。
Candidate Generator、Prefill filling order、Predictor 模型与在线 calibration、
Safe-Set、Controller Fallback、obligation ledger 和 actual-only StateStore
更新均不变。trace、QPS、SLO 也不因本次修改改变。

Stage 2 完全冻结，继续使用 V1 的 Prefill service-rate 语义：

\[
\mu_P(c)=\sum_i x_{i,c}=\texttt{BatchPlan.total_prefill_tokens},
\qquad
S(c)=\frac{\mu_P(c)}{\tau^{eff}(c)}.
\]

其中 \(\tau^{eff}\) 继续沿用 `effective_duration`：interpolation 取
`expected_duration`，constrained extrapolation 取 `conservative_duration`。
Tie-break 顺序不变：

```text
1. prefill_service_rate desc
2. effective_duration asc
3. prefill_service_tokens desc
4. prefill_budget asc
5. plan_id asc
```

## 2. Stage 1：ZERO-relative 增量 TBT violation 过滤

Stage 1 不再使用 min-slack duration 过滤，改用保守时长的 ZERO-relative
风险。风险时长统一取 Predictor 的 **conservative duration**：

\[
\bar\tau(c)=\tau^{conservative}(c).
\]

这**不改变** Stage 2 的 denominator。

对每个 active TBT obligation \(j\)（request \(r_j\)、deadline \(d_j\)、
当前时刻 \(t_k\)），miss 语义必须与
`ConsequenceEstimator._misses`（`dpp_scheduler/consequence_estimator.py`）一致：

- 候选服务了该 Decode（\(r_j\in\texttt{plan.decode\_items}\)）：
  \(m_j(c)=\mathbf 1[t_k+\bar\tau(c)>d_j]\)；
- 候选未服务该 Decode：
  \(m_j(c)=\mathbf 1[t_k+\bar\tau(c)\ge d_j]\)。

ZERO 参考 \(c_0\) 是零 Prefill、decode 集合覆盖全部 active Decode 的候选，
解析顺序（按 plan_id 升序、确定性）：

1. `template_id == ZERO` 或前缀 `ZERO:` → `ZERO_TEMPLATE`；
2. `template_id == STOCK` 或前缀 `STOCK:` → `STOCK_IDENTITY`（覆盖 canonical
   dedup 保留 STOCK 身份、删掉物质相同 ZERO 的帧）；
3. 任一零服务全 decode 候选（min plan_id）→ `ZERO_SERVICE_MATCH`；
4. 全部失败 → `raise RuntimeError("ZERO_REFERENCE_MISSING")`，fail-closed，
   禁止静默替代。

对候选 \(c\) 计算：

\[
\Delta N_c^{TBT}
=
\sum_j [m_j(c)-m_j(0)]^+,
\qquad
\Delta L_c^{TBT}
=
\sum_j [\ell_j(c)-\ell_j(0)]^+,
\qquad
\ell_j(c)=[t_k+\bar\tau(c)-d_j]^+.
\]

Stage-1 admissible set：

\[
\mathcal A_k^{V2B}
=
\{c\in\mathcal A_k^{safe}:\Delta N_c^{TBT}\le N\}.
\]

其中 \(N\) 来自配置 `dpp.stage1.maximum_incremental_tbt_violations`
（默认 0）。\(\Delta L\) 只记录用于诊断，不参与过滤。没有 active TBT
obligation 时全部 SafeCandidate 直接进入 Stage 2。ZERO 参考自身恒有
\(\Delta N_0=0\le N\)，因此参考解析成功时 eligible 集合必非空；空
SafeCandidate 仍返回 `NO_SAFE_CANDIDATES`。

旧 Stage 1 的 `min_slack + delta_D` 过滤、`NO_CANDIDATE_WITHIN_SLACK`
min-duration fallback 与 `TBT_NO_CANDIDATE_MIN_DURATION` reason 全部删除。
`min_tbt_slack_seconds` 与 `tbt_request_slacks` 保留为诊断字段；
`tbt_stage.delta_seconds` 保留为 legacy 字段（`delta_seconds_status:
legacy_inactive_in_v2b`），不参与 eligibility。

## 3. ZERO 不变量与确定性排序

Stage-2 的 ZERO 不变量不变：只要 Stage-1 eligible set 中存在
`total_prefill_tokens > 0` 的候选，winner 必须满足
`total_prefill_tokens > 0`；正 service rate 不得与零分进入同一 isclose
tie group。违反即实现错误，fail-closed。

普通选择 reason 保持 `TWO_STAGE_TBT_PREFILL_SERVICE_RATE`；空 Safe-Set 保持
`NO_SAFE_DECISION`。

## 4. 参数 N 与运行时覆盖

配置（`configs/dgx_spark_experiment.yaml`）：

```yaml
dpp:
  algorithm: two_stage_zero_relative_tbt_prefill_service_rate_v2b
  stage1:
    mode: zero_relative_incremental_violation
    duration_source: conservative_duration
    maximum_incremental_tbt_violations: 0
  diagnosis:
    schema_version: 4
```

开发网格通过环境变量 `DPP_STAGE1_MAX_DELTA_N` 覆盖 N：仅
`development_nonformal` scope 接受，取值必须是非负整数（拒绝 bool/负数/
浮点/非法串），在 `resolve_dpp_runtime_overrides` 中经
`dataclasses.replace` 应用到 `DPPSettings.maximum_incremental_tbt_violations`
并随 audit/diagnosis/aggregate 记录。formal scope 下 runner 与 server 双 gate
fail-closed。

## 5. Diagnosis schema v4 与 Replay

Writer 在 v3 基础上新增：

- `selector.maximum_incremental_tbt_violations`；
- `state.active_decode_request_ids`（供 replay 按集合精确解析 ZERO 参考）；
- 每候选 `stage1_tbt`：`risk_duration_seconds`、`violation_count`、
  `zero_violation_count`、`delta_violation_count`、`delta_lateness_seconds`、
  `passed`；
- `stage1`：`reference_plan_id`、`reference_template_id`、
  `reference_risk_duration_seconds`、`reference_violation_count`、
  `zero_reference_resolution`、`eligible_plan_ids`、status
  `DELTA_N_ADMITTED`。

Replay 对 v4 重算 slacks、effective/risk duration、ZERO 参考解析（plan_id
升序确定性顺序）、miss/ΔN/ΔL、admission、status、Stage-2 service-rate 与
ZERO 不变量；任何不一致计入 `stage1_mismatch`。v1/v2/v3 JSONL 的 replay
分支保持可用；`counterfactual_record` 只接受 v1/v2。

## 6. 测试与验证

- `tests/unit/test_dpp_selector.py`：ΔN 准入/N 门限、`>`/`>=` 边界语义、
  `ZERO_REFERENCE_MISSING` fail-fast、`STOCK_IDENTITY` 参考、无 obligation
  全准入、Stage-2 回归（相同 eligible 集与 V1 一致）、ZERO 不变量、N 默认 0、
  env 覆盖 scope 门限。
- `tests/unit/test_selector_diagnosis.py`：v4 往返零 mismatch、ΔN/admission/
  reference-resolution tamper 检测、legacy v3 replay 兼容。
- 远程 model-free 检查：203/203 单测、冻结 v3 JSONL（SHA `16bd5ca4…`）零
  mismatch 回归、V2-A replay 输出 SHA 回归（`db6647b2…`）。

## 7. 研究状态

V2-A 离线 replay（ΔN=0）判为 Case B：0/1,527 旧 ZERO-only 帧被释放；
全部这些帧 Stock ΔN=1，其 ΔL P50 0.674s / P90 0.756s。V2-B 将 N 参数化并
运行网格 {0,2,4,8,16}（每点 n=150、QPS 0.25、seed 1001、staged trace SHA
`203e7ed4…`）+ 同批 Stock，汇总 TTFT/TBT/Goodput 后决定是否值得保留 ΔL
bound。这是 development/non-formal 改动，不推进 G5/G6/G7。
