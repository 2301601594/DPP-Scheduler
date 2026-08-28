# Two-Stage TBT-Constrained Prefill Service-Rate Selector V1

本文档定义当前 Selector、Selector 配置、Diagnosis、Replay 和对应测试的权威
契约。它 supersede `Two-Stage-TBT-Constrained-TTFT-DPP-Selector.md` 的同类
内容，但不改写其中保留的 Rate/Absolute 研究过程和负面证据。

## 1. 范围与冻结边界

算法标识固定为 `two_stage_tbt_prefill_service_rate_v1`。Candidate Generator
继续产生 ZERO、P10–P100 和 STOCK；Prefill filling order、Predictor、
`effective_duration`、Safe-Set、Stage 1、Controller Fallback、obligation ledger
和 actual-only StateStore 更新均不变。TTFT/TBT service debt 继续按实际执行维护，
但不参与 V1 winner 选择。trace、QPS、SLO 和临时
`delta_D=0.020s` 也不因本次修改改变。

## 2. 两阶段选择

Stage 1 仍以 `active_tbt_obligations` 为 deadline 权威来源，使用
`effective_duration <= min_slack + delta_D` 筛选 SafeCandidate。没有有效 TBT
obligation 时全部通过；全部超限时仍按
`(effective_duration, prefill_budget, plan_id)` 选择唯一最短候选；空 Safe-Set
仍返回 `NO_SAFE_DECISION`。

Stage 2 不再预测逐请求 TTFT debt。对每个 Stage-1 eligible candidate (c)：

\[
\mu_P(c)=\sum_i x_{i,c}=\texttt{BatchPlan.total_prefill_tokens},
\qquad
S(c)=\frac{\mu_P(c)}{\tau(c)}.
\]

这里的 token 数必须是候选实际计划执行的 Prefill token 总数，而不是 P10/P20
等模板名义 budget。`tau` 仍是 Predictor 给出的 `effective_duration`：
interpolation 取 `expected_duration`，constrained extrapolation 取
`conservative_duration`。

Stage 2 禁止读取 TTFT/TBT debt、request urgency、完成奖励、剩余工作 utility、
deadline、预测 violation、Goodput utility 或经验权重。`ControlState` 仅保留在公开
接口和 snapshot-hash 契约中。

## 3. ZERO 不变量与确定性排序

只要 Stage-1 eligible set 中存在 `total_prefill_tokens > 0` 的候选，winner
就必须也满足 `total_prefill_tokens > 0`。正 service rate 即使小于固定
`isclose` absolute tolerance，也不得和零分进入同一 tie group。违反该不变量是
实现错误，必须 fail-closed。

ZERO 仅可能在以下情况被选择：没有 Prefill backlog、没有实际执行 Prefill 的
SafeCandidate、Stage 1 筛掉所有非零候选，或
`NO_CANDIDATE_WITHIN_SLACK` 的唯一最短候选 fallback 恰为 ZERO。

候选先按原始 service rate 降序分组；每组以组首 score 使用固定
`isclose(rel_tol=1e-9, abs_tol=1e-12)` 判断，然后按以下顺序排序：

1. effective duration 升序；
2. actual Prefill service tokens 降序；
3. Prefill budget 升序；
4. plan ID 升序。

普通选择 reason 为 `TWO_STAGE_TBT_PREFILL_SERVICE_RATE`；Stage-1 最短候选
fallback 和空 Safe-Set reason 分别保持
`TBT_NO_CANDIDATE_MIN_DURATION`、`NO_SAFE_DECISION`。

## 4. Diagnosis schema v3 与 Replay

详细 Diagnosis 仍默认关闭，沿用 `DPP_SELECTOR_DIAGNOSIS=0|1` 和唯一 JSONL
路径的成对、独占创建、逐 frame flush、forced-Stock 禁用契约。Schema v3 对每个
候选记录 plan、Prediction、effective duration、Stage-1 结果，以及进入 Stage 2
候选的 actual Prefill service tokens、service rate、Decode coverage 和 rank。

每个 frame 还记录 Prefill backlog count/tokens、Stage-1 eligible count、其中
实际 Prefill 非零的候选数、selected plan/tokens/rate、`selected_is_zero` 和
`zero_with_eligible_nonzero`。最后一项必须恒为 false；Writer 和 Replay 都验证
该不变量。

Replay 重算 duration、slack、Stage 1、service tokens、service rate、tie group、
rank、winner 和 ZERO 不变量。新 Writer 只产生 schema v3；旧 schema v1/v2 的
Rate/Absolute JSONL 仍可 replay，且旧 counterfactual 工具只接受 v1/v2，不能
把 V1 产物误解释为 TTFT debt 反事实。

## 5. 研究状态

Rate 版本在已有 n=150 运行的 Prefill-backlog frames 中选择 ZERO 约 82.6%；
Absolute 版本约为 98.5%，并出现严重 TTFT 退化。两者作为负面证据保留。V1 是
针对已定位 starvation 机制的 development/non-formal 改动，不推进 G5/G6/G7，
也不在完成 model-free tests 前构成新的性能结论。正式模型 benchmark 仍需单独
授权。
