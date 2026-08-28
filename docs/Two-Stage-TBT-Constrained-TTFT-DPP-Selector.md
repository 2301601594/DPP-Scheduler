# Two-Stage TBT-Constrained TTFT-DPP Selector

本文档定义当前 DPP Selector 和可重放 Diagnosis 的权威契约。它只取代
`Request-level-Service-Deficit-DPP-v2-Agent-Modification-Plan.md` 与 baseline
设计中的 Selector、Selector 配置、Selector 诊断和相应测试；Candidate
Generator V3、Predictor、Safe-Set、Controller Fallback、obligation ledger 和
actual-only StateStore 更新不变。

## 1. 决策结构

Selector 对同一个 Snapshot 的 SafeCandidate 执行两个严格分离的阶段：

\[
\mathcal A_k^{safe}
\rightarrow \text{TBT time filter}
\rightarrow \mathcal A_k^{TBT}
\rightarrow \text{TTFT DPP rank}
\rightarrow a_k^*.
\]

Safe-Set 只负责物理和 Predictor 数值可行性。TBT 决定候选资格，TTFT
决定最终 winner；Selector 不再计算 Decode drift，也不存在 TTFT/TBT 权重。

## 2. Stage 1：TBT 时间筛选

`StateSnapshot.active_tbt_obligations` 是下一 token deadline 的权威来源。
每条 obligation 必须未结算、kind 为 `TBT`、request 仍是 active Decode，且
与 `DecodeRequest.tbt_deadline` 一一精确一致；不一致、重复或非有限 deadline
均 fail-closed。

`tbt_deadline=None` 且没有 active obligation 的 Decode 不产生虚拟 deadline。
若没有有效 TBT obligation，所有 SafeCandidate 进入 Stage 2，状态为
`NO_ACTIVE_TBT_OBLIGATION`。

对每条有效 obligation：

\[
s_{j,k}^D=d_{j,k}^D-t_k,
\qquad
s_k^{D,\min}=\min_j s_{j,k}^D.
\]

Candidate duration 沿用 Predictor 的 `effective_duration`：interpolation 使用
`expected_duration`，constrained extrapolation 使用
`conservative_duration`。首版 development/non-formal 临时值为
`delta_D=0.020s`，通过条件包含等号：

\[
\hat\tau_k(a)\le s_k^{D,\min}+\delta_D.
\]

若至少一项通过，状态为 `WITHIN_SLACK`。若全部失败，按
`(effective_duration, prefill_budget, plan_id)` 选择唯一最短的物理可行候选，
状态为 `NO_CANDIDATE_WITHIN_SLACK`，继续进入 Stage 2。SafeCandidate 为空
返回 `NO_SAFE_DECISION`，仍由 Controller 所有的 Fallback 处理。

## 3. Stage 2：TTFT DPP

对每个 live Prefill request \(i\)，令当前 debt 为 \(Z_{i,k}^F\)，prompt
总长度为 \(p_i\)，剩余 Prefill 为 \(r_{i,k}^P\)，候选本轮服务为
\(c_{i,k}(a)\)。预测下一 debt 为：

\[
\widehat Z_{i,k+1}^F(a)=
\begin{cases}
0, & c_{i,k}(a)\ge r_{i,k}^P,\\
\left[Z_{i,k}^F+\dfrac{\hat\tau_k(a)}{L_i^F}
-\dfrac{c_{i,k}(a)}{p_i}\right]^+, & \text{otherwise}.
\end{cases}
\]

使用现有 Prefill reference concurrency：

\[
\Delta_F(a)=\frac{1}{2N_F^{ref}}\sum_i
\left[(\widehat Z_{i,k+1}^F(a))^2-(Z_{i,k}^F)^2\right],
\qquad
Score_F(a)=-\Delta_F(a).
\]

因此 Stage 2 等价于在 Stage 1 admissible set 内直接最小化预测的
post-decision debt 平方和。`effective_duration` 没有从 TTFT 预测中删除；它仍
通过 `effective_duration / ttft_slo` 增加所有未完成 Prefill 请求的下一 debt。
2026-08-28 的 n=150 development/non-formal 负面运行显示旧的
`-Delta_F/effective_duration` 与大量 ZERO 选择、Prefill starvation 同时出现；
当前 `two_stage_tbt_ttft_absolute_v1` 是只去除最终 rate normalization 的单变量
消融，不是已验证的性能结论。

Selector 不读取 TBT service debt、Decode reference concurrency、预测 violation、
deadline margin、Goodput utility 或经验权重。TBT debt 可继续由 StateStore
维护，仅用于运行诊断和历史兼容。

## 4. 确定性排序与 Decision

先按原始 score 降序形成分组。每组以该组最高 score 为锚点，使用固定
`isclose(rel_tol=1e-9, abs_tol=1e-12)` 判定成员，再在组内依次使用：

1. completed Prefill count 降序；
2. Prefill progress \(\sum_i c_i/p_i\) 降序；
3. effective duration 升序；
4. Prefill budget 升序；
5. plan ID 升序。

固定 Decision reason 为：

- `TWO_STAGE_TBT_TTFT`；
- `TBT_NO_CANDIDATE_MIN_DURATION`；
- `NO_SAFE_DECISION`。

## 5. Diagnosis 与 Replay

Diagnosis 默认关闭。启用时必须同时提供：

```text
DPP_SELECTOR_DIAGNOSIS=1
DPP_SELECTOR_DIAGNOSIS_PATH=<unique-jsonl-path>
```

缺一项、关闭时仍给路径、forced-Stock 模式启用、或目标文件已存在均
fail-closed。Writer 对每个 normal Selector frame 写入并 flush 一行，直接
序列化实际 Stage 1/Stage 2 audit 对象，不重新计算日志值。

Schema version 2 至少保存：Snapshot/config/Predictor/Selector identity、完整
TBT slack、所有 SafeCandidate 的 plan/Prediction/Stage 1 结果、进入 Stage 2
候选的逐请求 debt 计算、`prefill_drift`、旧 rate score/rank、新 absolute
score/rank、winner tie set、ZERO 专项字段，以及彼此分离的 Selector、
Controller 和实际执行结果。在线决策只使用 absolute score；旧 rate score
只用于 diagnosis counterfactual。Replay 继续读取历史 schema version 1，
但新 writer 只生成 schema version 2。

`scripts/replay_dpp_selector_diagnosis.py` 独立重算 effective duration、slack、
Stage 1、逐请求 debt、score、rank 和 winner，并报告：

```text
frames_replayed
stage1_mismatch
ttft_debt_mismatch
stage2_score_mismatch
winner_mismatch
tie_break_mismatch
```

使用 `--counterfactual` 时还对同一 Stage-2 candidate set 分别执行旧 rate 与
新 absolute 排名，报告 ZERO winner 转移、Stock 平均排名，并按 Prefill
backlog 深度 `1/2-4/5-8/>8` 和最小 TBT slack
`<=0/0-50/50-100/100-200/>200ms` 分层。该 replay 只能回答已有记录中的
选择反事实，不能替代真实执行时间和性能测量。

Diagnosis run 只有在所有 mismatch 为零时才有效。JSONL、replay summary 和
各自 SHA-256 进入 append-only run manifest。正式 benchmark 默认关闭该
诊断，启用前需单独评估其 CPU/I/O 开销。

## 6. 研究状态

`delta_D=20ms` 是用户指定的 development/non-formal 临时值，没有经过目标
DGX profiling，不构成 G5 参数冻结或性能结论。旧 TTFT 权重网格尚未执行，
现已退役；历史设计、决策、失败和负面证据继续保留。
