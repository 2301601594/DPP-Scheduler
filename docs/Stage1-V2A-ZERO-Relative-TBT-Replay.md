# Stage 1 V2-A：ZERO-relative `ΔN_TBT = 0` 实现与离线 Replay 方案

> **Superseded by V2-B:** 本方案的离线 replay 已于 2026-08-29 完成并判为
> Case B（R_release=0.0），后续在线实现按
> `docs/Two-Stage-ZERO-Relative-TBT-Prefill-Service-Rate-Selector-V2B.md`
> 执行（ΔN ≤ N 参数化）。本文件保留为 V2-A 研究记录。

## 1. 本阶段目标

本阶段只修改 Selector 的 Stage 1，不修改：

- Candidate Generator；
- Predictor 模型与在线 calibration；
- Safe-Set；
- Stage 2 Prefill service-rate 评分；
- Stage 2 tie-break；
- trace、SLO、QPS、seed；
- Controller Fallback。

当前 Stage 2 保持：

\[
Score(c)=\frac{\mu^P(c)}{\tau^{eff}(c)}
\]

其中 \(\mu^P(c)\) 是候选 `BatchPlan.total_prefill_tokens`，\(\tau^{eff}(c)\) 继续沿用当前 V1 的 `effective_duration` 语义。

本阶段只把 Stage 1 从：

\[
\tau^{eff}(c)\le \min_j s_j+\delta_D
\]

替换为：

\[
\boxed{\Delta N_c^{TBT}=0}
\]

即：候选相对于 ZERO Decode-only 基线不能新增 TBT violation。

---

## 2. 为什么修改 Stage 1

现有 diagnosis 已确认：

- TBT constraint + Prefill backlog：1611 帧；
- Stock 被旧 Stage 1 淘汰：1570 帧；
- 其中 1527 帧最终只能选 ZERO；
- Stock 一旦进入 Stage 2，3002 次中获选 2997 次。

因此当前主要问题不是 Stage 2，而是旧 Stage 1 的 `min-slack` 绝对过滤。

这一阶段不继续调 `delta_D`，也不修改 Stage 2，避免同时改变多个因素。

---

## 3. ZERO-relative TBT 风险定义

当前帧为 \(k\)，时间为 \(t_k\)。

对每个 active TBT obligation \(j\)：

- 请求 ID：\(r_j\)；
- deadline：\(d_j\)。

### 3.1 Stage 1 使用的时间

Stage 1 风险统一使用 Predictor 的：

\[
\boxed{\bar\tau(c)=\tau^{conservative}(c)}
\]

原因是 Stage 1 负责风险控制。

注意：这不改变 Stage 2 的 denominator。Stage 2 仍使用当前 V1 的 `effective_duration`，保证本轮只改变 Stage 1。

### 3.2 TBT miss 语义

必须与当前 `ConsequenceEstimator._misses()` 保持一致。

若候选 \(c\) 本轮执行请求 \(j\) 的 Decode：

\[
m_j(c)=\mathbf 1[t_k+\bar\tau(c)>d_j]
\]

若候选 \(c\) 没有执行请求 \(j\) 的 Decode：

\[
m_j(c)=\mathbf 1[t_k+\bar\tau(c)\ge d_j]
\]

不能简单统一写成 `duration > slack`，因为 STOCK candidate 可能不覆盖全部 active Decode。

### 3.3 ZERO reference

Stage 1 必须先确定一个 Decode-only baseline \(c_0\)。

在线实现的解析规则：

1. `total_prefill_tokens == 0`；
2. `set(plan.decode_items) == set(snapshot.active_decode_request_ids)`；
3. 优先选择 `template_id == ZERO`；
4. 如果 canonical dedup 删除了 ZERO，但存在物质上完全相同的 zero-service full-decode candidate，可以使用该候选；
5. 如果 active TBT obligation 存在但无法找到上述 baseline，标记 `ZERO_REFERENCE_MISSING`，development 模式 fail-fast；禁止静默拿任意最短 candidate 代替 ZERO。

记 ZERO 风险为：

\[
m_j(0).
\]

---

## 4. `ΔN_TBT` 定义

对 candidate \(c\)：

\[
\boxed{
\Delta N_c^{TBT}
=
\sum_j [m_j(c)-m_j(0)]^+
}
\]

因为 \(m_j\in\{0,1\}\)，它等价于统计：

> ZERO 不会 miss、但 candidate 会新增 miss 的 Decode request 数。

Stage 1 V2-A 的 admissible set：

\[
\boxed{
\mathcal A_k^{V2A}
=
\{c\in\mathcal A_k^{safe}:\Delta N_c^{TBT}=0\}
}
\]

ZERO baseline 本身必然满足：

\[
\Delta N_0^{TBT}=0.
\]

因此只要 ZERO reference 合法，Stage 1 不应该再出现“所有 candidate 被过滤”的状态。

如果当前没有 active TBT obligation，则直接：

\[
\mathcal A_k^{V2A}=\mathcal A_k^{safe}.
\]

---

## 5. 同时计算 `ΔL_TBT`，但暂不参与过滤

为下一阶段诊断定义每个 obligation 的预测 lateness：

\[
\ell_j(c)=[t_k+\bar\tau(c)-d_j]^+
\]

并定义：

\[
\boxed{
\Delta L_c^{TBT}
=
\sum_j [\ell_j(c)-\ell_j(0)]^+
}
\]

V2-A 中：

- `ΔN_TBT` 决定 Stage 1 admission；
- `ΔL_TBT` 只记录，不参与过滤；
- 不引入 `K`、`L_max` 或额外权重。

这样可以先回答：

> `ΔN=0` 是否已经足以释放 Prefill，以及这些候选是否显著扩大已有 violation 的 lateness。

---

## 6. `dpp_selector.py` 修改

### 6.1 新增 Stage-1 risk duration

保留当前 `effective_duration()` 给 Stage 2 使用，新增：

```python
def stage1_risk_duration(candidate: SafeCandidate, maximum: float) -> float:
    return _finite_positive(
        "conservative_duration",
        candidate.prediction.conservative_duration,
        maximum,
    )
```

不要把当前 `effective_duration()` 全局改成 conservative，否则会同时改变 Stage 2。

### 6.2 新增 TBT risk helper

建议新增内部结构：

```python
@dataclass(frozen=True)
class TBTIncrementalRisk:
    plan_id: str
    risk_duration: float
    violation_count: int
    zero_violation_count: int
    delta_violation_count: int
    delta_lateness_seconds: float
```

实现：

```python
def _candidate_tbt_risk(...):
    ...
```

miss 判定必须使用第 3.2 节定义。

### 6.3 替换 `_tbt_stage()`

新的 `_tbt_stage()`：

```text
1. validate SafeCandidates
2. 继续计算当前 tbt_request_slacks，作为 diagnosis 数据
3. 如果 SafeCandidates 为空：
      NO_SAFE_CANDIDATES
4. 如果没有 active TBT obligation：
      所有 SafeCandidates eligible
5. 解析 ZERO reference
6. 用 conservative_duration 计算 ZERO 每个 obligation 的 miss/lateness
7. 对每个 candidate：
      计算 candidate miss/lateness
      计算 ΔN
      计算 ΔL
8. 保留 ΔN == 0 的 candidate
9. 交给现有 Stage 2
```

删除旧 Stage 1 中用于 winner eligibility 的：

```python
duration_limit = min_slack + tbt_delta_seconds
effective_duration <= duration_limit
NO_CANDIDATE_WITHIN_SLACK -> minimum-duration fallback
```

`min_slack` 仍可保留用于 diagnosis，但不再决定 eligibility。

### 6.4 Stage 2 完全冻结

不要修改 `_score_one()` 的核心评分：

```python
service_rate = service_tokens / tau
score = service_rate
```

也不要修改现有 tie-break：

```text
1. prefill_service_rate desc
2. effective_duration asc
3. prefill_service_tokens desc
4. prefill_budget asc
5. plan_id asc
```

---

## 7. Algorithm identity 与配置

建议新算法 ID：

```text
two_stage_zero_relative_tbt_prefill_service_rate_v2a
```

当前：

```yaml
dpp:
  tbt_stage:
    delta_seconds: 0.020
```

在 V2-A 中 `delta_seconds` 不再参与 Stage-1 eligibility。

为避免本轮顺带清理历史配置，可先保留字段，但明确标记：

```yaml
delta_seconds_status: legacy_inactive_in_v2a
stage1:
  mode: zero_relative_incremental_violation
  duration_source: conservative_duration
  maximum_incremental_tbt_violations: 0
```

---

## 8. Diagnosis 修改

在线实现后建议 Selector Diagnosis 升级 schema v4。

每个 Stage-1 candidate 至少记录：

```text
risk_duration_seconds
violation_count
zero_violation_count
delta_violation_count
delta_lateness_seconds
passed
```

Stage 1 记录：

```text
reference_plan_id
reference_template_id
reference_risk_duration_seconds
reference_violation_count
zero_reference_resolution
eligible_plan_ids
```

保留：

```text
min_tbt_slack_seconds
```

仅用于诊断，不再作为 gate。

---

## 9. 单元测试

至少增加：

1. ZERO safe，Mixed 与 ZERO violation 集完全相同：
   - `ΔN=0`
   - Mixed 可以进入 Stage 2。

2. ZERO 不 miss，Mixed 新增一个 miss：
   - `ΔN=1`
   - Mixed 被 Stage 1 拒绝。

3. ZERO 已经 miss 某请求，Mixed 仍然 miss 同一请求但没有新增请求：
   - `ΔN=0`
   - Mixed 允许进入 Stage 2。

4. ZERO lateness 5ms，Mixed lateness 50ms，但 miss request 集相同：
   - `ΔN=0`
   - `ΔL=45ms`
   - V2-A 仍允许候选，`ΔL` 只诊断。

5. candidate 没有服务某 Decode，frame end 恰好等于 deadline：
   - 使用 `>=`
   - 判为 miss。

6. candidate 服务该 Decode，frame end 恰好等于 deadline：
   - 使用 `>`
   - 不判 miss。

7. 无 active TBT obligation：
   - 所有 SafeCandidates 进入 Stage 2。

8. ZERO reference 缺失：
   - development fail-fast / 明确 `ZERO_REFERENCE_MISSING`；
   - 不允许静默用最短 candidate 代替。

9. Stage 2 regression：
   - 在给定相同 Stage-2 candidate set 时，V2-A 与当前 V1 winner 完全一致。

10. ZERO invariant：
    - baseline 自身 `ΔN=0`；
    - Stage 1 正常情况下 eligible set 非空。

---

# 10. 先执行离线 Replay，不立即上线

当前 Selector Diagnosis schema v3 已经包含：

- timestamp；
- TBT request ID/deadline/slack；
- 每个 SafeCandidate 的 `prefill_items/decode_items`；
- expected/conservative duration；
- 当前 Stage-1 结果；
- 当前 Stage-2 score；
- 当前 winner。

因此无需重新跑 GPU，就可以对同一 candidate set counterfactually replay `ΔN=0`。

本方案配套脚本：

```text
replay_stage1_delta_n0.py
```

它不会修改仓库，只读取现有 diagnosis JSONL。

---

## 11. Replay 输入

目标 run：

```text
/home/dongj/LLM/results/raw/qwen3_14b_dgx_spark/
predictor_online_trimmed_calibration_selector_diagnosis_qps0p25_n150_seed1001_v1/
runs/selector_diagnosis_n150_qps0p25_seed1001_attempt01/
```

目标 diagnosis SHA-256：

```text
16bd5ca4fe921a4b8a98ab3cbf10b321ce7b086263539d70aca455fa305162d5
```

先定位 JSONL：

```bash
RUN_DIR="/home/dongj/LLM/results/raw/qwen3_14b_dgx_spark/predictor_online_trimmed_calibration_selector_diagnosis_qps0p25_n150_seed1001_v1/runs/selector_diagnosis_n150_qps0p25_seed1001_attempt01"

find "$RUN_DIR" -maxdepth 2 -type f -name '*diagnosis*.jsonl' -print
```

确认 SHA：

```bash
sha256sum <selector_diagnosis.jsonl>
```

必须等于：

```text
16bd5ca4fe921a4b8a98ab3cbf10b321ce7b086263539d70aca455fa305162d5
```

---

## 12. Replay 命令

把配套脚本放入仓库根目录或 `scripts/` 后：

```bash
cd /home/dongj/projects/LLM

python scripts/replay_stage1_delta_n0.py \
  "<selector_diagnosis.jsonl>" \
  --expected-sha256 16bd5ca4fe921a4b8a98ab3cbf10b321ce7b086263539d70aca455fa305162d5 \
  --output results/processed/qwen3_14b_dgx_spark/stage1_v2a_delta_n0_replay_seed1001.json
```

---

## 13. Replay 必须报告的核心指标

最重要的是旧 ZERO-only frame 的释放率：

\[
\boxed{
R_{release}
=
\frac{
\#(\text{旧 Stage1 ZERO-only 且 V2-A winner 为 non-ZERO})
}{
\#(\text{旧 Stage1 ZERO-only})
}
}
\]

同时报告：

```text
active_tbt_and_backlog_frames
old_zero_only_active_tbt_backlog_frames
old_zero_only_with_new_nonzero_eligible
released_old_zero_only_frames
new_zero_winner_active_tbt_backlog_frames

stock_new_eligible_frames
stock_newly_released_frames
stock_new_winner_frames

new_winner_template_histogram
stock_delta_n_histogram

winner_delta_lateness_seconds:
    min/p50/p90/p95/p99/max/mean

released_frame_winner_delta_lateness_seconds:
    min/p50/p90/p95/p99/max/mean
```

---

## 14. 如何解释 Replay

### 情况 A：大量 ZERO-only frame 被释放

说明：

\[
\Delta N=0
\]

已经能够解除当前 `min-slack` Stage-1 starvation，值得实现 V2-A 并做 n=150 paired diagnostic。

### 情况 B：释放率仍很低

说明即使以 ZERO 为 baseline，大量 Mixed candidate 仍会新增至少一个 TBT miss。

此时不要直接把 `ΔN<=1` 上线，应先看：

```text
stock_delta_n_histogram
ΔL distribution
```

再决定 V2-B 是否采用：

\[
\Delta N\le K
\land
\Delta L\le B.
\]

### 情况 C：`ΔN=0` 能释放很多候选，但 `ΔL` 很大

说明单纯 violation count 不足。

例如 ZERO 和 candidate 都 miss 同一请求，但 candidate 把 lateness 从 5ms 扩大到 200ms。

此时下一版应增加 `ΔL` bound，而不是直接上线 V2-A。

---

## 15. 本阶段停止条件

完成以下内容后停止，不进入 G5/G6/G7：

```text
1. 离线 V2-A replay 完成；
2. 输入 diagnosis SHA 匹配；
3. 输出 replay JSON；
4. 汇报 ZERO-only release rate；
5. 汇报 Stock ΔN 分布；
6. 汇报 winner ΔL 分布；
7. 根据 replay 判断是否值得修改在线 Selector。
```

在离线 replay 结果出来之前，不修改 Candidate Generator，不调 Stage 2，不重训 Predictor，不进行正式 benchmark。
