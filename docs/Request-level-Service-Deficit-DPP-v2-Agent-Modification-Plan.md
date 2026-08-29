# Request-level Service-Deficit DPP v2 修改方案

> **Selector supersession:** 当前 Selector、Selector 配置、Diagnosis、Replay
> 和相应测试以 `Two-Stage-ZERO-Relative-TBT-Prefill-Service-Rate-Selector-V2B.md`
> 为准。本文件中的 Prefill + Decode weighted drift 内容保留为历史设计；其余
> v2 组件仍按本文档执行。

## 0. 修改目标

对当前 `DPP-Scheduler` 做一次结构性重构。

本轮解决四个核心问题：

1. Candidate Generator 过于复杂，同时搜索 Decode 子集和 Prefill Budget；
2. Predictor 超出训练支持域后直接拒绝预测，导致大量 iteration 无法进入 DPP；
3. 当前 TTFT/TBT 使用累计 violation debt，只能在违约发生后产生明显压力；
4. Request-level debt 求和后，Prefill / Decode 请求数量差异可能产生额外的聚合尺度偏差。

本轮采用：

$$
\boxed{\text{Request-level Service-Deficit DPP}}
$$

并通过 profiling 得到固定：

$$
N_F^{ref},\qquad N_D^{ref}
$$

作为 Prefill 和 Decode 两个阶段的 reference concurrency。

本轮明确不实现：

- 不主动牺牲某个 Decode request；
- 不枚举 Decode request 子集；
- 不加入 request sacrifice / Goodput loss penalty；
- 不根据当前 Snapshot 的请求数量动态归一化；
- 不重新设计正式 benchmark；
- 不使用未来输出长度、未来 EOS 等在线不可知信息。

---

# 1. 从已有 Profiling 冻结 Reference Concurrency

不要人为设置：

$$
N_F^{ref},N_D^{ref}.
$$

必须从当前已经完成的 Qwen3-14B / DGX Spark profiling 数据中统计得到。

## 1.1 Profiling 数据选择

Agent 首先检查仓库现有 profiling artifact、manifest 和相关脚本。

数据源必须满足：

- Qwen3-14B；
- DGX Spark；
- 与当前 Scheduler 使用相同的模型部署配置；
- 来自已经冻结的 profiling / predictor profiling 数据；
- 不允许使用之后正式 benchmark 的结果反向调整 reference concurrency；
- 不允许每个 QPS、每个 seed 单独重新计算。

优先使用已经用于 Predictor 建模或验证的、真实 Scheduler Snapshot profiling 数据。

不要使用：

```text
dpp_candidate_diag_*
formal benchmark result
```

作为 reference concurrency 的调参数据。

如果存在多个已有 profiling seed，将它们合并统计。

如果无法从现有 profiling 中恢复 Snapshot 的 Prefill/Decode request count，不允许猜测数值，应明确报告缺失字段。

## 1.2 每个 Snapshot 统计

对每个 profiling frame $k$ 定义：

$$
N_{F,k}
=
|\mathcal P_k|
$$

其中 $\mathcal P_k$ 为当前所有仍存在 Prefill work 的请求：

$$
remaining\_tokens_i > 0.
$$

定义：

$$
N_{D,k}
=
|\mathcal D_k|
$$

其中 $\mathcal D_k$ 为当前全部 Active Decode requests。

分别统计：

```text
P50
P75
P90
mean
max
sample count
```

正式 Scheduler 使用：

$$
\boxed{
N_F^{ref}
=
\max\left(1,
P50\{N_{F,k}\mid N_{F,k}>0\}
\right)
}
$$

和：

$$
\boxed{
N_D^{ref}
=
\max\left(1,
P50\{N_{D,k}\mid N_{D,k}>0\}
\right)
}
$$

即：对该阶段实际处于 active 状态的 frame 使用 median concurrency。

不要把大量 `N=0` 的 frame 加入 median，否则 reference concurrency 会被系统空闲状态污染。

## 1.3 Reference 必须冻结

生成一个可审计 artifact，例如：

```json
{
  "schema_version": 1,
  "model": "Qwen3-14B",
  "hardware": "DGX Spark",

  "prefill_reference_concurrency": 8,
  "decode_reference_concurrency": 32,

  "statistic": "p50_positive_frames",

  "prefill": {
    "p50": 8,
    "p75": 12,
    "p90": 18,
    "mean": 9.4,
    "max": 31,
    "sample_count": 12345
  },

  "decode": {
    "p50": 32,
    "p75": 43,
    "p90": 55,
    "mean": 34.7,
    "max": 64,
    "sample_count": 14122
  },

  "source_files": [],
  "source_sha256": []
}
```

上面的数字只是格式示例，绝对不能直接使用。

实际值必须由已有 profiling 计算。

同时在实验 YAML 中冻结：

```yaml
dpp:
  reference_concurrency:
    statistic: p50_positive_frames
    prefill: <profiling derived>
    decode: <profiling derived>
    artifact: <reference artifact path>
```

后续所有 QPS 和 seed 使用完全相同的：

$$
N_F^{ref},N_D^{ref}.
$$

---

# 2. Candidate Generator 简化

Candidate Generator 不再同时控制：

```text
which Decode requests
+
how much Prefill
```

只控制：

```text
how much Prefill
```

所有正常 Candidate 默认包含：

$$
\boxed{
\mathcal D_k^{all}
=
\text{当前全部 Active Decode requests}
}
$$

删除正常候选中的：

```text
MANDATORY
CRITICAL
ALL
```

三种 Decode profile。

本轮禁止 Candidate Generator 主动跳过任何 Active Decode request。

---

# 3. Prefill Budget Candidate

设：

$$
D_k=|\mathcal D_k^{all}|
$$

总 token budget：

$$
C^{tok}.
$$

当前所有 Prefill remaining tokens：

$$
R_k^P.
$$

则：

$$
P_k^{max}
=
\min
\left(
R_k^P,
C^{tok}-D_k
\right).
$$

第一版固定生成：

$$
\boxed{
\mathcal B_k^P
=
\{
0,\,
0.25P_k^{max},\,
0.5P_k^{max},\,
0.75P_k^{max},\,
P_k^{max}
\}
}
$$

另外加入：

$$
P_k^{finish}
$$

表示刚好完成当前最高优先级 Prefill request 的 budget。

所以正常情况下最多约 6 个 Candidate。

进行：

```text
integer rounding
clamp
canonical deduplicate
minimum_prefill_chunk_tokens check
```

不要继续固定加入：

```text
KNEE = 768
```

后续如果 profiling 能证明稳定 kernel / latency knee，再单独加入。

---

# 4. Prefill Request 的绑定顺序

Candidate Generator 不负责 SLO 决策。

固定使用确定性顺序，例如：

```text
partial/running Prefill
    ↓
其他 waiting Prefill 按 FCFS
    ↓
request_id stable tie-break
```

不要根据：

```text
当前 Debt
TTFT deadline
Predictor duration
predicted violation
```

改变 Prefill request 顺序。

这样 DPP 的控制变量保持为一个清晰的一维变量：

$$
P_k.
$$

---

# 5. Predictor OOD 改为受约束外推

当前行为：

```text
outside support domain
        ↓
Prediction invalid
        ↓
Safe-Set reject
```

需要删除。

改成：

```text
inside support
    -> INTERPOLATION

outside support
    -> CONSTRAINED_EXTRAPOLATION
```

OOD Candidate 仍然必须产生：

```text
expected_duration
conservative_duration
```

并继续进入 DPP。

---

# 6. Predictor 增加外推元数据

Prediction 增加或等价记录：

```text
prediction_mode:
    interpolation
    extrapolation

ood_features
ood_distance
extrapolation_uncertainty_seconds
```

原来的：

```text
in_support
```

可以暂时保留作为兼容字段，但：

```text
in_support == false
```

不得再等价于：

```text
candidate invalid
```

---

# 7. Ridge Predictor 的外推规则

对于每个 active feature：

$$
x_d\in[x_d^{min},x_d^{max}]
$$

先执行：

$$
x_d^{clip}
=
\min
\left(
x_d^{max},
\max(x_d^{min},x_d)
\right).
$$

在训练域边界计算：

$$
\tau_{boundary}
=
f_\theta(\mathbf x^{clip}).
$$

对于超过上界的 workload：

$$
x_d>x_d^{max},
$$

定义 Ridge 对原始特征的 slope：

$$
g_d
=
\frac{\beta_d}{\sigma_d}.
$$

外推 slope 必须满足：

$$
s_d=\max(0,g_d).
$$

然后：

$$
\boxed{
\tau_{base}^{ext}
=
\tau_{boundary}
+
\sum_d
s_d[x_d-x_d^{max}]^+
}
$$

禁止因为负 Ridge coefficient 出现：

```text
workload 增大
duration 反而下降
```

低于训练下界时，第一版直接使用 lower-bound prediction，不做乐观向下外推。

---

# 8. OOD Distance 与不确定性

定义标准化 OOD distance：

$$
d_{OOD}
=
\max_d
\frac{
|x_d-x_d^{clip}|
}{
\sigma_d
}.
$$

在线窗口达到最小样本数后，定义原始 residual 窗口 $r$。Expected 使用双侧
各裁剪 5%（每侧裁剪数为 $\lfloor0.05|r|\rfloor$）的中心估计：

$$
r_c=\operatorname{TrimmedMean}_{5\%}(r),
\qquad
r_{95}=Q_{0.95}^{\mathrm{higher}}(r),
$$

其中 $r_{95}$ 必须从未裁剪的原始窗口计算。于是：

$$
\begin{aligned}
\widehat\tau
&=\tau_{base}^{ext}+r_c,\\
\overline\tau
&=\max\left[
\widehat\tau,
\tau_{base}^{ext}+r_{95}+\kappa_{OOD}d_{OOD}
\right].
\end{aligned}
$$

少于最小在线样本数时，继续使用现有 artifact 的同 batch-kind offline OOF
cold-start mean 和 centered-P95；本次修复不改写历史 artifact。

其中：

$$
\kappa_{OOD}\ge0
$$

必须来自配置；在线 trimmed-mean 比例、quantile 方法和 conservative floor
也必须由 active config 显式固定。

不要写死。

---

# 9. DPP 使用 Effective Duration

为避免 OOD extrapolation 被过度乐观地选择，Selector 定义：

$$
\tau_k^{sel}(\mathbf a)
=
\begin{cases}
\widehat\tau_k(\mathbf a),
&\text{interpolation}\\[4pt]
\overline\tau_k(\mathbf a),
&\text{extrapolation}
\end{cases}
$$

即：

- 支持域内使用 expected duration；
- 支持域外允许执行，但 DPP 使用 conservative duration 评价。

后面的 predicted debt 和 drift 都统一使用：

$$
\tau_k^{sel}.
$$

---

# 10. 真正的 Predictor Failure

只有以下情况才属于 invalid prediction：

```text
NaN
Inf
duration <= 0
conservative < expected
unknown batch kind
artifact/schema invalid
feature construction failure
```

单纯：

```text
outside support domain
```

不属于 failure。

OOD observation 暂时不要直接加入当前 interpolation residual window。

单独记录，用于后续扩大 profiling coverage。

---

# 11. Safe-Set 只负责 Hard Feasibility

Safe-Set 只保留：

```text
token budget
sequence budget
current KV
rolling KV
prediction numerical validity
```

不允许因为：

```text
Predictor OOD
TTFT risk
TBT risk
predicted_violation_count
predicted_total_lateness
deadline margin
```

删除 Candidate。

因此：

$$
\boxed{
\mathcal A_k^{phys}
=
\{
a:
\text{physical constraints satisfied}
\}
}
$$

DPP 直接在：

$$
\mathcal A_k^{phys}
$$

上优化。

`ConsequenceEstimator` 如果暂时保留，只用于 diagnostic。

不得继续影响：

```text
Safe-Set membership
DPP score
tie-break
```

---

# 12. 删除 Aggregate Violation Debt

当前系统级：

```text
prefill_backlog
ttft_debt
tbt_debt
```

不再作为新版 DPP Selector 的控制变量。

其中 `prefill_backlog` 可以保留为 diagnostic，但不能参与新的 DPP 公式。

核心状态改为：

$$
\boxed{
Z_{i,k}^{F}
}
$$

和：

$$
\boxed{
Z_{j,k}^{D}
}
$$

分别表示每个请求的 TTFT/Prefill service deficit 和 TBT/Decode service deficit。

---

# 13. Decode Service-Deficit Queue

对于 Decode request $j$：

TBT SLO：

$$
L_j^D.
$$

所需平均 token 服务频率：

$$
\rho_j^D=\frac1{L_j^D}.
$$

第 $k$ 轮真实 duration：

$$
\tau_k.
$$

如果本轮是否为它产生 token：

$$
x_{j,k}^D\in\{0,1\},
$$

则：

$$
\boxed{
Z_{j,k+1}^D
=
\left[
Z_{j,k}^D
+
\frac{\tau_k}{L_j^D}
-
x_{j,k}^D
\right]^+
}
$$

解释：

$$
\frac{\tau_k}{L_j^D}
$$

表示这段 wall-clock time 中根据 TBT SLO 新增了多少 token 服务需求。

---

# 14. Prefill / TTFT Service-Deficit Queue

对于 Prefill request $i$：

Prompt 总 token：

$$
p_i.
$$

TTFT SLO：

$$
L_i^F.
$$

本轮为它执行：

$$
c_{i,k}^{P}
$$

个 Prefill token。

完整 Prompt 的 normalized service 为 1，因此：

$$
x_{i,k}^{F}
=
\frac{c_{i,k}^{P}}{p_i}.
$$

定义：

$$
\boxed{
Z_{i,k+1}^F
=
\left[
Z_{i,k}^F
+
\frac{\tau_k}{L_i^F}
-
\frac{c_{i,k}^{P}}{p_i}
\right]^+
}
$$

TTFT 和 TBT 统一为：

$$
\boxed{
Z_{next}
=
[
Z
+
\text{required service}
-
\text{actual service}
]^+
}
$$

全部为无量纲量。

---

# 15. Request-level SLO 必须进入 Snapshot

Contracts 中为 request 增加稳定的 SLO 参数：

```text
PrefillRequest:
    ttft_slo_seconds

DecodeRequest:
    tbt_slo_seconds
```

当前单一 SLO class 下可以均来自：

```yaml
ttft_slo_seconds: 2.0
tbt_slo_seconds: 0.25
```

但不要把这些常数直接写死在 StateStore 或 Selector。

不要通过：

```text
deadline - current_time
```

反推出原始 SLO。

---

# 16. 新请求 Debt 初始化

不能简单把所有新请求永远初始化：

$$
Z=0.
$$

否则如果请求在一个很长的 iteration 中途到达，到下一次 Snapshot 才出现，会丢掉已经产生的等待时间。

对于首次进入 DPP StateStore 的 Prefill request：

$$
\boxed{
Z_{i,init}^F
=
\left[
\frac{t_k-a_i}{L_i^F}
-
\frac{prefilled_i}{p_i}
\right]^+
}
$$

其中：

- $a_i$：request arrival time；
- $t_k$：首次看到该 request 的 Snapshot time。

对于正常从已有 DPP 状态持续存在的请求，不重新初始化，继续累计。

Decode debt 的生命周期继续与真实 output 事件同步，不允许因为重新构造 Snapshot 而重置。

---

# 17. ControlState 重构

推荐不要继续把所有 request debt 塞入 immutable Snapshot 本身。

StateStore 内部维护：

```text
ttft_service_debts:
    request_id -> float

tbt_service_debts:
    request_id -> float
```

绑定到当前 Snapshot 后生成只读 ControlState，例如：

```text
snapshot_hash

ttft_service_debts:
    tuple[(request_id, debt), ...]

tbt_service_debts:
    tuple[(request_id, debt), ...]
```

所有 key 必须 deterministic order。

请求结束后删除对应 state，避免无限增长。

---

# 18. Reference-Concurrency Weighted Lyapunov

这是本次 Selector 的核心。

不要使用裸：

$$
\sum_i(Z_i^F)^2+\sum_j(Z_j^D)^2.
$$

也不要使用：

$$
\frac1{N_{F,k}},
\frac1{N_{D,k}}
$$

做动态 normalization。

定义 profiling 冻结的：

$$
N_F^{ref},
\qquad
N_D^{ref}.
$$

Lyapunov function：

$$
\boxed{
\mathcal L_k
=
\frac12
\left[
\frac{1}{N_F^{ref}}
\sum_{i\in\mathcal P_k}
(Z_{i,k}^F)^2
+
\frac{1}{N_D^{ref}}
\sum_{j\in\mathcal D_k}
(Z_{j,k}^D)^2
\right]
}
$$

这等价于使用固定 weighted Lyapunov：

$$
\alpha_F=\frac1{N_F^{ref}},
\qquad
\alpha_D=\frac1{N_D^{ref}}.
$$

这样做的目的不是消除 concurrency 信息，而是：

> 消除 Prefill/Decode 两类请求在正常运行规模上的基础数量级差异，同时仍保留 overload 时请求数量增长形成的真实压力。

例如：

$$
N_D>N_D^{ref}
$$

时 Decode 总压力仍然会自然增大。

绝对不能每一轮除以：

$$
N_{D,k}.
$$

否则 5 个危险 Decode 和 50 个危险 Decode 的系统压力可能被归一化成相近大小。

---

# 19. Candidate 的 Predicted Next Debt

对候选 $\mathbf a$，使用：

$$
\tau_k^{sel}(\mathbf a).
$$

Prefill request：

$$
\boxed{
\widehat Z_{i,k+1}^F(\mathbf a)
=
\left[
Z_{i,k}^F
+
\frac{\tau_k^{sel}(\mathbf a)}{L_i^F}
-
\frac{c_{i,k}^{P}(\mathbf a)}{p_i}
\right]^+
}
$$

Decode request：

因为本轮全部 Active Decode 默认执行：

$$
x_{j,k}^{D}(\mathbf a)=1.
$$

所以：

$$
\boxed{
\widehat Z_{j,k+1}^D(\mathbf a)
=
\left[
Z_{j,k}^D
+
\frac{\tau_k^{sel}(\mathbf a)}{L_j^D}
-
1
\right]^+
}
$$

---

# 20. 分别计算 Prefill / Decode Drift

Prefill：

$$
\boxed{
\widehat\Delta_k^F(\mathbf a)
=
\frac{1}{2N_F^{ref}}
\sum_i
\left[
(\widehat Z_{i,k+1}^F)^2
-
(Z_{i,k}^F)^2
\right]
}
$$

Decode：

$$
\boxed{
\widehat\Delta_k^D(\mathbf a)
=
\frac{1}{2N_D^{ref}}
\sum_j
\left[
(\widehat Z_{j,k+1}^D)^2
-
(Z_{j,k}^D)^2
\right]
}
$$

总 Drift：

$$
\boxed{
\widehat\Delta_k(\mathbf a)
=
\widehat\Delta_k^F(\mathbf a)
+
\widehat\Delta_k^D(\mathbf a)
}
$$

第一版不要再增加额外：

$$
w_F,w_D.
$$

相当于：

$$
w_F=w_D=1.
$$

如果之后 profiling / benchmark 证明仍存在长期系统性偏置，再单独讨论固定权重。

---

# 21. 最终 DPP Score

由于 iteration duration 不固定，使用 Drift rate：

$$
\boxed{
Score_k(\mathbf a)
=
-
\frac{
\widehat\Delta_k(\mathbf a)
}{
\tau_k^{sel}(\mathbf a)
}
}
$$

最终：

$$
\boxed{
\mathbf a_k^*
=
\arg\max_{\mathbf a\in\mathcal A_k^{phys}}
Score_k(\mathbf a)
}
$$

也就是：

> 选择单位 wall-clock 时间内，最能减少 Prefill + Decode normalized service deficit 的 Prefill Budget。

---

# 22. 为什么使用完整 Drift，而不是一阶 Backpressure Term

不要只实现：

$$
\frac{
\sum Zx
}{
\tau
}.
$$

本轮 Candidate 数量最多约 6 个，完整计算 predicted next state 的成本可以忽略。

完整 Drift 有一个重要优点：

即使：

$$
Z_{j,k}=0,
$$

一个过长 Candidate 仍然可能产生：

$$
Z_{j,k+1}>0.
$$

DPP 因而可以提前感知长 Prefill iteration 对 TBT 的影响。

这正是当前 violation-based queue 缺失的能力。

---

# 23. Legacy DPP 参数停止参与评分

以下字段不再进入 Selector：

```text
epsilon_ttft
epsilon_tbt
token_normalization
obligation_normalization
weight_v
service_utility
ttft_success
ttft_miss
tbt_success
tbt_miss
```

可以为了旧配置兼容暂时保留 schema，但必须：

```text
legacy
inactive
not used by selector
```

不能悄悄继续影响排序。

---

# 24. Candidate Tie-Break

新的主排序只有：

$$
Score.
$$

如果 score 在数值容差范围内相等，使用简单 deterministic tie-break：

```text
1. smaller predicted/effective duration
2. smaller Prefill budget
3. stable plan_id
```

不要再用：

```text
predicted_violation_count
deadline_margin
predicted_misses
```

做 tie-break。

否则相当于重新偷偷引入 SLO Risk Gate。

---

# 25. 真实 State 更新必须使用真实执行结果

预测值只用于：

```text
candidate comparison
```

真实 Debt 更新必须使用：

$$
\tau_k^{actual}.
$$

每一轮执行完成后 StateStore 必须收到：

```text
actual_duration_seconds
actual per-request prefill service
actual decode request ids
```

形式例如：

```python
state_store.update_from_actual(
    previous_snapshot=snapshot,
    actual_duration_seconds=actual_tau,
    executed_prefill_items=actual_prefill_items,
    executed_decode_items=actual_decode_items,
)
```

---

# 26. Prefill Feedback 必须改成 Per-Request

当前仅统计：

```text
actual_prefill_tokens
```

总量不够。

必须改成：

```text
request_id -> actual_prefill_tokens
```

因为：

$$
c_i^P/p_i
$$

是 per-request service。

执行后必须验证：

$$
BatchPlan.prefill\_items
=
ActualSchedulerOutput.prefill\_items.
$$

如果不同，fail loudly。

不能拿计划值假装实际执行值。

---

# 27. Decode Feedback

对上一轮 Snapshot 中的每个 active Decode：

如果实际生成一个 token：

$$
x_j^D=1.
$$

否则：

$$
x_j^D=0.
$$

虽然正常 Candidate Generator 要求：

```text
all Decode
```

但真实 feedback 必须从 execution observation 获得。

这样即使以后出现：

```text
preemption
fallback
native scheduling
execution mismatch
```

StateStore 仍能正确更新 debt。

---

# 28. Controller 新流程

目标流程简化为：

```text
Expire / synchronize actual events
        ↓
Build immutable Snapshot
        ↓
Bind request-level Service Debt
        ↓
Candidate Generator
    ALL Decode
    several Prefill Budgets
        ↓
Predictor
    interpolation / extrapolation
        ↓
Hard Safe-Set
    token / seq / KV / prediction validity
        ↓
DPP Selector
    predicted request debt
    fixed reference concurrency
    drift / effective duration
        ↓
Execute exact BatchPlan
        ↓
Actual iteration timing
        ↓
Actual per-request service
        ↓
Update request-level debt
```

`ConsequenceEstimator` 不再位于主决策依赖链中。

如果保留，只用于 diagnostics。

---

# 29. Fallback / Liveness 修改

因为 OOD 不再是 Predictor rejection：

```text
PREDICTOR_OUT_OF_SUPPORT
        ↓
LIVENESS_ESCAPE
```

这一条路径应该自然消失。

Fallback 只负责真正的：

```text
no physically feasible candidate
prediction numerically invalid
native preemption required
```

不要因为正常 extrapolation 触发 fallback。

原有 native progress / zero-progress watchdog 保留。

---

# 30. Diagnostic Log 必须同步更新

每个 Candidate 记录：

```text
plan_id
prefill_budget
prefill_items
decode_count

prediction_mode
ood_distance
ood_features

expected_duration
conservative_duration
selector_effective_duration

raw_prefill_drift
raw_decode_drift

normalized_prefill_drift
normalized_decode_drift

total_drift
drift_per_second
score

selection_rank
selected
```

每轮额外记录：

```text
current_prefill_count
current_decode_count

prefill_reference_concurrency
decode_reference_concurrency

current_prefill_count / N_F_ref
current_decode_count / N_D_ref

sum_ttft_debt
max_ttft_debt
mean_ttft_debt

sum_tbt_debt
max_tbt_debt
mean_tbt_debt

number_of_extrapolated_candidates
max_ood_distance
```

这能够检查：

> 当前 Prefill/Decode term 的数量级差异到底来自请求平均压力，还是 concurrency overload。

---

# 31. Reference Concurrency 的诊断要求

Agent 必须额外输出 profiling 统计报告：

```text
N_F_ref = ?
N_D_ref = ?

Prefill:
P50 / P75 / P90 / max

Decode:
P50 / P75 / P90 / max

source profiling paths
source SHA256
total frames
positive Prefill frames
positive Decode frames
```

并说明为什么最终采用 P50-positive-frames。

禁止直接只把两个数字写进 YAML 而不留下 provenance。

---

# 32. 必须增加的 Unit Tests

## Candidate Generator

验证：

```text
所有正常 Candidate 的 decode_items 完全一致
decode_items == all active decode requests
```

验证不同 Candidate 只主要改变 Prefill Budget。

测试：

```text
ZERO
25%
50%
75%
MAX
FINISH
```

正确 deduplicate。

## Reference Concurrency

构造 profiling records：

```text
Prefill counts:
0, 2, 4, 8, 8, 12

Decode counts:
0, 0, 16, 32, 32, 48
```

验证：

```text
0 不进入 positive-frame median
结果 deterministic
reference >= 1
```

验证 artifact 中包含 source hash。

## Predictor

必须覆盖：

```text
support 内预测与旧模型一致

轻微 OOD
    -> 有效 extrapolation

远距离 OOD
    -> finite positive duration

workload 增大
    -> extrapolated duration 不下降

OOD
    -> 不被 Safe-Set 删除

NaN / Inf
    -> fail closed
```

## Decode Service Debt

验证：

$$
Z'=[Z+\tau/L-x]^+.
$$

例如：

```text
tau 较短 + x=1
    -> debt 降低

tau 较长 + x=1
    -> debt 可增加

x=0
    -> debt 增长

L 更严格
    -> debt 增长更快
```

## Prefill Service Debt

验证：

$$
Z'
=
[Z+\tau/L-c/p]^+.
$$

覆盖：

```text
service 不足
service 充足
partial prefill
不同 prompt length
不同 TTFT SLO
```

## Reference Normalization

如果：

$$
N_F=N_F^{ref}
$$

并且：

$$
N_D=N_D^{ref},
$$

且两边平均单请求 Drift 相同，则 normalized Prefill / Decode contribution 应接近相同。

如果 Decode 数量增加到：

$$
2N_D^{ref},
$$

平均单请求压力不变，则 Decode 总 contribution 应约增加一倍。

即 fixed reference 只能消除基础尺度差：

**不能消除真实 overload concurrency 信号。**

同时验证代码中没有：

```python
divide_by_current_prefill_count
divide_by_current_decode_count
```

## Selector

构造两个 Candidate：

```text
A:
small Prefill
short duration

B:
large Prefill
long duration
```

验证：

```text
Decode pressure 高
    -> 更倾向 A

Prefill pressure 高
    -> 可以选择 B

当前所有 debt = 0
    -> 过长 B 仍可能因 predicted next debt 被惩罚

N_D 增加
    -> Decode pressure 自然增强

改变 current N_D
    -> reference denominator 不变
```

Selector 不允许读取：

```text
predicted_violation_count
ttft_miss
tbt_miss
deadline_margin
```

---

# 33. 配置清理

新版配置建议整理为：

```yaml
candidate_generator:
  prefill_budget_fractions:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    - 1.0
  include_finish_boundary: true

predictor:
  allow_extrapolation: true
  ood_uncertainty_coefficient: ...
  extrapolation_strategy: clipped_monotonic_ridge

safe_set:
  slo_risk_filter: false

dpp:
  algorithm: request_service_deficit_v2

  reference_concurrency:
    prefill: ...
    decode: ...
    statistic: p50_positive_frames
    artifact: ...

  use_variable_frame_rate: true
```

旧字段如果仍存在：

```text
epsilon_ttft
epsilon_tbt
weight_v
token_normalization
obligation_normalization
```

必须明确注释：

```text
legacy compatibility only
inactive under request_service_deficit_v2
```

---

# 34. 推荐实施顺序

严格按下面顺序完成，不要同时大面积修改。

## Phase 1：Reference Concurrency

先写 profiling 分析脚本。

得到并冻结：

$$
N_F^{ref},N_D^{ref}.
$$

输出统计与 SHA。

## Phase 2：Candidate Generator

改为：

```text
ALL Decode
+
1D Prefill Budget
```

先完成单测。

## Phase 3：Predictor Extrapolation

删除 OOD rejection。

实现：

```text
clipping
monotonic extrapolation
OOD distance
uncertainty
```

完成 Predictor 和 Safe-Set 单测。

## Phase 4：Request-level State

重构：

```text
contracts.py
state_store.py
vllm_adapter.py
```

打通：

```text
per-request debt
actual duration
actual per-request service
```

## Phase 5：DPP Selector

实现：

$$
\widehat Z'
$$

然后：

$$
\Delta_F,\Delta_D
$$

然后：

$$
-\frac{\Delta_F+\Delta_D}{\tau}.
$$

## Phase 6：Diagnostics

确认每一个 Candidate 的：

```text
Prefill drift
Decode drift
normalization
duration
score
```

全部可审计。

## Phase 7：Smoke Test

只进行小规模真实运行。

不要直接进入多 seed benchmark。

---

# 35. Smoke Test 顺序

修改完成后建议按：

```text
unit tests
        ↓
synthetic scheduler tests
        ↓
n=20
        ↓
n=50
        ↓
n=100
```

确认没有：

```text
stall
zero-progress loop
OOM
massive fallback
prediction NaN
debt explosion
```

之后再重新执行：

```text
n=300
QPS=0.20
seed=1001
```

该运行仍然首先作为 diagnostic run，而不是正式 benchmark。

---

# 36. n=300 Diagnostic 重点检查

下一次不要只看最终 Goodput。

重点输出：

```text
Candidate Prefill Budget 分布

selected:
0%
25%
50%
75%
100%
FINISH

Prefill normalized drift 分布
Decode normalized drift 分布

|Prefill drift| / |Decode drift|

current N_F / N_F_ref
current N_D / N_D_ref

prediction interpolation ratio
prediction extrapolation ratio

OOD distance P50/P95/max

iteration duration P50/P95/P99/max

TBT P50/P95/P99/max

TTFT P50/P95/P99

Goodput
```

特别检查是否还出现：

```text
Prefill / Decode term 长期相差 1e3 ~ 1e4
```

新版不要求两项每轮完全相等。

但在典型 concurrency 区域：

$$
N_F\approx N_F^{ref},
\qquad
N_D\approx N_D^{ref}
$$

时，不应再存在由于公式 normalization 本身造成的系统性多个数量级差异。

---

# 37. 本轮验收标准

最终必须满足：

1. 正常 Candidate 全部执行所有 Active Decode；
2. Candidate Generator 主要只枚举 Prefill Budget；
3. 候选数量约 5～6 个；
4. OOD Predictor Candidate 不再直接被删除；
5. OOD 使用受约束、单调、带不确定性的 extrapolation；
6. Safe-Set 只负责物理可行性；
7. 不存在 SLO Risk Gate；
8. 不使用累计 violation-ratio TTFT/TBT debt；
9. 每个 live request 有自己的 service-deficit state；
10. TTFT/TBT debt 都是无量纲 service deficit；
11. $N_F^{ref},N_D^{ref}$ 从已有 profiling 自动生成；
12. Reference concurrency 在所有 benchmark 中固定；
13. 不使用当前 Snapshot concurrency 作为 denominator；
14. DPP 使用 fixed-reference weighted Lyapunov；
15. Selector 使用完整 predicted next debt；
16. Selector 使用 variable-frame Drift rate；
17. 真实 State 只根据真实 duration 和真实 service 更新；
18. diagnostic 能解释 Prefill / Decode 两部分贡献；
19. 所有 unit tests 通过；
20. 完成 n=50 或 n=100 smoke 后再决定是否进入 n=300。

---

# 38. Agent 完成后必须汇报

修改完成后不要直接继续参数搜索。

先给出：

```text
1. 从哪些 profiling 文件得到 reference concurrency
2. N_F_ref / N_D_ref 最终值
3. profiling P50/P75/P90/max
4. 修改文件列表
5. Candidate Generator 新旧差异
6. Predictor OOD 新旧差异
7. StateStore 新旧差异
8. DPP 旧公式和新公式
9. unit test 结果
10. smoke test 结果
11. normalized Prefill/Decode drift 数量级
12. extrapolation 使用比例
13. 是否仍存在 fallback/liveness 异常
14. 是否推荐重新运行 n=300 QPS=0.20 seed=1001
```

完成这些内容后停止，等待进一步审核。

---

# 39. 核心公式汇总

首先，由已有 profiling 冻结：

$$
\boxed{
N_F^{ref}=\operatorname{P50}(N_F\mid N_F>0),
\qquad
N_D^{ref}=\operatorname{P50}(N_D\mid N_D>0)
}
$$

然后维护请求级服务债务：

$$
\boxed{
Z_{i,k+1}^F
=
\left[
Z_{i,k}^F+
\frac{\tau_k}{L_i^F}
-
\frac{c_{i,k}^{P}}{p_i}
\right]^+
}
$$

$$
\boxed{
Z_{j,k+1}^D
=
\left[
Z_{j,k}^D+
\frac{\tau_k}{L_j^D}
-
x_{j,k}^{D}
\right]^+
}
$$

最后采用固定 reference concurrency 的 weighted Lyapunov：

$$
\boxed{
\mathcal L_k=
\frac12
\left[
\frac{\sum_i(Z_{i,k}^F)^2}{N_F^{ref}}
+
\frac{\sum_j(Z_{j,k}^D)^2}{N_D^{ref}}
\right]
}
$$

并选择：

$$
\boxed{
a_k^*
=
\arg\max_a
\left(
-\frac{\widehat\Delta_k(a)}
{\tau_k^{sel}(a)}
\right)
}
$$

这版设计同时解决：

- violation-based debt 只能在违约后才产生压力；
- Prefill/Decode 原始量纲不一致；
- 两阶段典型 concurrency 不同造成的基础规模偏差；
- Predictor OOD 导致大量候选无法进入 DPP；
- Candidate Generator 同时搜索 Decode 子集和 Prefill Budget 导致动作空间过于复杂。

同时保留真实高并发情况下的 overload pressure，不通过动态按当前请求数平均的方式把并发压力抹掉。
