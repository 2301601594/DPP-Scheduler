# Qwen3-14B 在 DGX Spark 上的模块化 DPP Scheduler 实现方案

## 1. 实现范围

首版固定条件：

- 模型：`Qwen3-14B`；
- 硬件：单台 `DGX Spark`，单 GPU；
- 推理精度：`BF16`；
- 基于当前 vLLM 源码提交，开发前记录 commit；
- 使用 vLLM V1、continuous batching 和 chunked prefill；
- 关闭 Prefix Caching 和 Speculative Decoding；
- 让模型正常响应 EOS，不把输出长度预先暴露给 Scheduler；
- 每个 Decode 请求每轮最多生成一个 token；
- 单一 SLO 类别。

请求 API 可以携带有限的客户端终止护栏。护栏值只属于 runner，不能进入
Scheduler 快照、候选、Predictor 特征/标签或 DPP 决策。`stop` 和 `length`
终止都必须保留并分层报告；`length` 表示护栏触发，不表示 Scheduler 事先知道
了请求长度。调度器绝不保存、推断或使用 `remaining_output_tokens`、预计固定
输出长度或最终 EOS 位置。

## 2. 总体结构

```text
vLLM 状态
   ↓
StateSnapshot
   ↓
Candidate Generator ──→ BatchPlan[]
   ↓
Predictor ─────────────→ Prediction[]
   ↓
Safe-Set ──────────────→ SafeCandidate[]
   ↓
DPP Selector ──────────→ SelectedPlan
   ↓
vLLM Adapter 执行
   ↓
Observer 使用真实结果更新状态
```

三个主体模块只依赖公共数据结构，不直接调用彼此的内部方法：

```text
CandidateGenerator.generate(snapshot) -> BatchPlan[]
SafeSet.filter(snapshot, plans, predictions) -> SafeSetResult
DPPSelector.select(snapshot, control_state, safe_candidates) -> Decision
```

Predictor 是共享服务：

```text
DurationPredictor.predict(snapshot, plans) -> Prediction[]
```

替换任意实现时，只要保持输入和输出结构不变，其他模块无需修改。

## 3. 公共数据结构

```text
StateSnapshot
  frame_id
  timestamp
  snapshot_hash
  waiting_prefill_requests
  active_decode_requests
  active_ttft_obligations
  active_tbt_obligations
  recovery_requests
  free_kv_blocks
  kv_block_size
  token_budget
  sequence_budget

BatchPlan
  plan_id
  snapshot_hash
  template_id
  prefill_items: [(request_id, token_count)]
  decode_items: [request_id]
  total_prefill_tokens
  total_decode_tokens
  total_sequences
  projected_kv_blocks
  mandatory_request_ids

Prediction
  plan_id
  expected_duration
  conservative_duration
  in_support
  ttft_success
  ttft_miss
  tbt_success
  tbt_miss
  service_utility
  predictor_version

ControlState
  snapshot_hash
  prefill_backlog
  ttft_debt
  tbt_debt
```

所有结构在一轮决策中保持不可变，并携带相同的 `snapshot_hash`。

## 4. Candidate Generator

### 4.1 模板

Prefill seed breakpoint 只保留四类：

$$
\mathcal B^P
=
\left\{
P_{zero},
P_{finish},
P_{knee},
P_{bindable}
\right\}
$$

- `ZERO` 为 0；
- `FINISH` 完整结束当前最高优先级且可绑定的 Prefill 请求，无法完整结束时省略；
- `KNEE` 为同配置 profiling 冻结的效率拐点，并受当前可绑定上限约束；
- `BINDABLE_MAX` 为 token budget 与可见 Prefill backlog 给出的最大尝试值，不代表 SLO-safe。

未冻结的 knee 保持 `null`，此时省略 `KNEE` 候选。

Decode profile 只保留三种：

| Profile | 内容 |
|---|---|
| `MANDATORY` | 仅强制 Recovery 和必须保护的 Decode |
| `CRITICAL` | Mandatory 加当前 TBT slack 不大于 Critical Horizon 的 live Decode |
| `ALL` | 在资源范围内尽可能加入全部 Decode |

因此去重前最多生成 12 个候选，不枚举任意请求子集。
Critical Horizon 是只读取 Snapshot 的冻结阈值，不是候选执行时间，也不读取
Predictor。未冻结时保持 `null`，`CRITICAL` 退化为 `MANDATORY`。

### 4.2 请求排序

Decode 顺序：

1. 达到 Recovery 年龄阈值的最老请求；
2. 尚未违约的请求按 TBT deadline 使用 EDF；
3. 其余 Recovery 请求按首次 miss 时间排序。

Prefill 顺序：

1. hard-protected TTFT 请求按 deadline 排序；
2. 已经执行过部分 Prefill 的请求；
3. 其余请求按 FCFS。

### 4.3 约束

$$
b_k^P(\mathbf a)
+
b_k^D(\mathbf a)
\le
C^{tok}
$$

$$
n_k^{seq}(\mathbf a)
\le
C^{seq}
$$

Prefix Caching 关闭后，可依据每个请求的当前已分配 slot、计划新增 token 数和 block size，纯计算本轮新增 KV block；不得在候选阶段修改真实 block manager。

普通 partial Prefill chunk 必须为 0 或不小于配置的 minimum；仅当请求本身
剩余不足 minimum 且本轮完整完成时允许更小的 chunk。

### 4.4 伪代码

```text
generate(snapshot):
    mandatory = oldest_recovery_if_due(snapshot)
    decode_order = rank_decode_by_edf(snapshot)
    prefill_order = rank_prefill(snapshot)
    plans = []

    for profile in [MANDATORY, CRITICAL, ALL]:
        decode_items = bind_decode(profile, mandatory, decode_order)

        for cap in [ZERO, FINISH, KNEE, BINDABLE_MAX]:
            plan = fill_prefill(decode_items, prefill_order, cap)
            if native_limits_hold(plan):
                plan.projected_kv_blocks = project_kv_without_side_effect(plan)
                plans.append(plan)

    return canonical_deduplicate(plans)
```

完整计划生成后才依次执行 Predictor、Safe-Set 和 DPP Selector；它们的输出
不得反向改变本轮候选构造。

## 5. Predictor

Predictor 只预测当前 `BatchPlan` 的 iteration 时间，不预先固定模型系列。
训练阶段可比较简单基线、树模型或其他满足离线训练和低延迟 CPU 推理约束的
回归器；模型与超参数仅使用训练/开发数据选择，独立测试集只用于最终评估。

Profiling 不预先固定模型特征。每个实际执行的 `BatchPlan` 记录一行基础数据：
run/plan/snapshot 标识、实际 iteration 时间，以及每个选中请求的
request ID、Prefill/Decode 阶段、执行前已计算或 KV-context 长度和本轮 token
数。标识字段只用于关联和审计；基础数据不得包含剩余输出长度或未来 EOS 信息。
实际时间采用锁定 vLLM 的官方 iteration 边界：异步模型执行提交完成之后开始，
到模型结果收集和采样完成为止，不包含 Scheduler 的结果更新。

训练阶段从基础数据离线构造并比较候选聚合或非线性特征，不使用独立测试集
选择特征或模型。验证后冻结最终模型系列、超参数、特征 schema、变换、支持域
和 Predictor artifact；在线特征必须能从当前 `StateSnapshot` 与 `BatchPlan`
直接计算。

基础预测为：

$$
\widetilde{\tau}_k(\mathbf a)
=
f_{\theta}
\left(
\mathbf x_k(\mathbf a)
\right)
$$

残差按 `Prefill-only`、`Decode-only`、`Mixed` 分别维护有界在线窗口。窗口只接收已经完成且位于支持域内的 iteration，残差定义为实际时间减去基础预测；当前 iteration 不得使用自身残差。模型参数保持离线冻结。平均时间与保守时间分别为：

$$
\widehat{\tau}_k(\mathbf a)
=
\widetilde{\tau}_k(\mathbf a)
+
\operatorname{Mean}(e_g)
$$

$$
\overline{\tau}_k(\mathbf a)
=
\widehat{\tau}_k(\mathbf a)
+
Q_{0.95}
\left(
e_g-\operatorname{Mean}(e_g)
\right)
$$

- DPP Selector 使用 `expected_duration`；
- Safe-Set 使用 `conservative_duration`；
- 每类窗口不足 32 条时回退到该类训练集 OOF 残差统计；
- 窗口长度只使用训练集 OOF 顺序回放从 `{32,64,128}` 中选择；
- 输入超出冻结支持域时设置 `in_support=false`；
- 服务重启时清空窗口，不跨 run 或 Predictor 版本继承；
- 第一版不在线更新模型权重。

训练数据必须来自相同的 `Qwen3-14B + DGX Spark + BF16 + vLLM commit + 运行开关`。

# 6. Safe-Set

Safe-Set 只处理当前调度决策是否可以安全执行，不负责优化 Goodput，也不直接控制长期 SLO 违约比例。

整体流程分为两步：

1. 先检查物理资源和 Predictor 是否可用，得到资源可行集合；
2. 再根据保守执行时间估计 SLO 违约风险，从资源可行计划中选择交给 DPP 的候选。



## 6.1 硬资源过滤

对每个候选 BatchPlan `a`，首先检查以下硬约束：

1. token budget 可行；
2. sequence budget 可行；
3. 当前帧 KV Cache 投影不超过物理容量；
4. 通过有限 horizon 的 Rolling KV Guard；
5. Predictor 输入位于训练和支持域内。

只要违反任意一项，候选直接删除，不进入后续 DPP。

资源可行集合记为：

$$
\mathcal A_k^{res}
$$

其中，`k` 表示当前调度 iteration。



## 6.2 Rolling KV Guard

只检查未来少量 Decode iteration 的 KV 增长，不使用请求的最大输出长度做一次性保守预留。

候选执行后，请求 `i` 的 KV 上下文长度记为：

$$
c_{i,k}^{KV,+}(\mathbf a)
$$

KV Cache 的 block 大小记为 $B^{blk}$。

对于当前 Decode 请求，在未来 `H` 个 Decode iteration 内最多新增的 block 数为：

$$
R_{i,k}^{dec}(H,\mathbf a)
=
\left\lceil
\frac{c_{i,k}^{KV,+}(\mathbf a)+H}{B^{blk}}
\right\rceil
-
\left\lceil
\frac{c_{i,k}^{KV,+}(\mathbf a)}{B^{blk}}
\right\rceil
$$

其中，`H` 只是一个较小的 Rolling Horizon，例如未来若干个 Decode iteration，不代表请求的完整输出长度。

候选计划需要满足：

$$
B_k^{proj}(\mathbf a)
+R^0
+\sum_i R_{i,k}^{dec}(H,\mathbf a)
\le C^{KV}
$$

其中：

- `B_k^{proj}(a)`：执行候选计划后的 KV block 投影占用；
- `R^0`：系统预留的安全 block；
- `C^{KV}`：KV Cache 可使用的物理 block 总量。

该条件每个 iteration 重新计算，因此不会因为请求声明了很大的最大输出长度而长期占用大量保守预算。



## 6.3 SLO 违约风险估计

SLO 不再作为绝对硬约束。

对于每个资源可行计划 `a`，Predictor 给出保守执行时间：

$$
\overline{\tau}_k(\mathbf a)
$$

当前时间记为 `t_k`，请求 `i` 当前 obligation 的 deadline 记为 `d_{i,k}`。

Adapter 负责根据当前锁定的 vLLM commit 判断该计划是否会在本轮真正完成对应 obligation，例如 TTFT 是否已经产生并返回首 token，而不能简单把“最后一个 Prefill chunk”等同于 TTFT 完成。

定义：

$$
\nu_{i,k}(\mathbf a)=
\begin{cases}
1, & \text{候选计划预计使 obligation } i \text{发生违约}\\
0, & \text{否则}
\end{cases}
$$

若计划会在本轮完成 obligation，则当

$$
t_k+\overline{\tau}_k(\mathbf a)>d_{i,k}
$$

时认为该 obligation 预计违约。

若计划本轮不会完成 obligation，则当

$$
t_k+\overline{\tau}_k(\mathbf a)\ge d_{i,k}
$$

时认为该计划已经把 obligation 推入违约状态。

候选计划的预计违约数量为：

$$
N_k^{vio}(\mathbf a)
=
\sum_i \nu_{i,k}(\mathbf a)
$$

总预计超时时间为：

$$
E_k^{vio}(\mathbf a)
=
\sum_i
\nu_{i,k}(\mathbf a)
\left[
 t_k+\overline{\tau}_k(\mathbf a)-d_{i,k}
\right]^+
$$

其中：

$$
[x]^+=\max(x,0)
$$



## 6.4 构造 DPP 候选集合

如果存在不会产生新违约的资源可行计划，则只将这些计划交给 DPP：

$$
\mathcal A_k^{DPP}
=
\left\{
\mathbf a\in\mathcal A_k^{res}
:
N_k^{vio}(\mathbf a)=0
\right\}
$$

这样可以避免在有安全选择时主动选择会造成 SLO 违约的计划。

如果不存在零违约计划，则按照以下顺序对资源可行计划排序：

1. `N_k^{vio}(a)` 更小；
2. `E_k^{vio}(a)` 更小。

然后保留前 `K` 个计划：

$$
\mathcal A_k^{DPP}
=
\operatorname{TopK}
\left(\mathcal A_k^{res}\right)
$$

这些计划继续进入 DPP，由 DPP 在 Goodput、队列漂移和长期违约代价之间进行优化。

Safe-Set 本身不负责决定最终选择哪个计划。



## 6.5 FallbackPlan

FallbackPlan 是独立于正常 DPP 候选集合的兜底路径，不参与 DPP 评分。
其构造所有权固定在 Controller；Safe-Set 只返回 `safe_candidates` 和拒绝原因。

当没有可执行的正常候选时，按以下规则构造：

- 当前存在 Decode 请求：停止 Prefill，按照 EDF 顺序构造 Decode-only 计划；
- 当前没有 Decode 请求：执行最小物理可行 Prefill chunk；
- Fallback 仍必须满足 token、sequence、KV Cache 和 Predictor 支持域等硬约束；
- Fallback 的执行时间仍由 Predictor 预测并记录；
- 如果 Fallback 仍然不可执行，非空 workload 进入 liveness escape 或 Preemption；只有空 workload 才进入 Idle。

Fallback 的作用是保证系统在极端负载或全部候选都存在 SLO 风险时仍有明确行为，而不是参与正常性能优化。



## 6.6 实现流程

```python
def filter(snapshot, plans, predictions):
    resource_plans = []
    rejected = []

    # 1. 硬资源过滤
    for plan in plans:
        pred = predictions[plan.plan_id]
        reasons = check_resource_and_predictor(
            snapshot,
            plan,
            pred,
        )

        if reasons:
            rejected.append((plan.plan_id, reasons))
            continue

        resource_plans.append((plan, pred))

    # 2. 没有物理可行计划：交回 Controller，不在 Safe-Set 内构造 Fallback
    if not resource_plans:
        return SafeSetResult(
            safe_candidates=[],
            rejected=rejected,
        )

    # 3. 计算 SLO 违约风险
    evaluated = []

    for plan, pred in resource_plans:
        n_vio, e_vio = estimate_violation(
            snapshot,
            plan,
            pred,
        )

        evaluated.append(
            (plan, pred, n_vio, e_vio)
        )

    # 4. 风险只作为候选元数据保留，不在 Safe-Set 内按零违约或 Top-K 删除
    candidates = evaluated

    return SafeSetResult(
        safe_candidates=candidates,
        rejected=rejected,
    )
```



最终逻辑为：

```text
资源硬约束不可行
    → 删除

资源可行
    → 计算保守 SLO 违约风险并附加到候选元数据
    → 所有物理/Predictor 可行候选进入 DPP（Legacy Top-K inactive）

没有正常可执行候选
    → FallbackPlan

Fallback 仍不可执行且 workload 非空
    → liveness escape 或 Preemption

Fallback 仍不可执行且 workload 为空
    → Idle
```



## 7. DPP Selector

> 本节记录 baseline Selector。当前 Selector 已由
> `Two-Stage-TBT-Constrained-Prefill-Service-Rate-Selector-V1.md` 取代；Candidate Generator、
> Predictor、Safe-Set、Fallback 和 actual-only 更新仍按各自当前权威文档执行。

只维护三个控制量：

$$
\Theta_k
=
\left(
Q_k^P,
Z_k^F,
Z_k^D
\right)
$$

其中不包含物理 Decode 队列。

对每个 SafeCandidate，依据平均时间判断当前 obligation 的预计成功和 miss，并计算 Goodput-oriented 即时服务效用。它不是实际 request-level Goodput；真实 Goodput 只在自然 EOS 后统计。

首版评分为：

$$
\boxed{
\Psi_k(\mathbf a)
=
\frac{
Q_k^P\mu_k^P(\mathbf a)
+
Z_k^F
\left[
\epsilon^F\widehat S_k^F(\mathbf a)
-
\left(1-\epsilon^F\right)\widehat M_k^F(\mathbf a)
\right]
+
Z_k^D
\left[
\epsilon^D\widehat S_k^D(\mathbf a)
-
\left(1-\epsilon^D\right)\widehat M_k^D(\mathbf a)
\right]
+
V\widehat U_k(\mathbf a)
}
{\widehat\tau_k(\mathbf a)}
}
$$

选择最高分计划：

$$
\mathbf a_k^\star
\in
\arg\max_{
\mathbf a\in\mathcal A_k^{safe}
}
\Psi_k(\mathbf a)
$$

平分时依次选择：预计 miss 更少、保守 deadline margin 更大、`plan_id` 更小的计划。

首版 Scheduler 集成冻结以下量纲和参数：

- `epsilon^F = epsilon^D = 0.05`。保留的 Stock 结果在 0.20/0.25
  req/s 下，TTFT miss 比例为 2.0%/2.5%，TBT obligation miss 比例为
  3.40%/4.62%；统一向上取 5%。0.30 req/s 的对应比例为 43.0%/5.51%，
  因而会产生正债务漂移；
- `Q^P` 和 `mu^P` 在评分时分别除以 `C_tok=2048`，`Z^F/Z^D`、
  `S/M` 和即时效用分别除以 `C_seq=64`，使四个分子项无量纲；
- `U_hat = S_hat^F + S_hat^D`，表示本轮预计按时完成的 obligation 数，
  并非 terminal request-level Goodput；当前 integration 令 `weight_v=0.0` 暂时关闭该项，公式保留以便后续单独重做；
- 分数单位为 `1/second`。所有输入和中间项必须是 IEEE-754 binary64
  可表示的有限值，非负量下界为零；`expected_duration <= 0`、NaN、Inf、
  hash/plan 不匹配均 fail-closed，不做截断或乐观外推。

这些值冻结在 `configs/dpp_selector_integration_freeze.json`，用于完成首版
Scheduler 集成。它们没有经过 DPP 参数搜索，因此不构成正式 benchmark 的
最优参数声明。

```text
select(snapshot, control, safe_candidates):
    if safe_candidates is empty:
        return NO_SAFE_DECISION

    scored = []
    for plan, pred in safe_candidates:
        validate_snapshot_and_prediction(plan, pred)
        terms = compute_dpp_terms(plan, pred, control)
        score = sum(terms) / pred.expected_duration
        scored.append((plan, score, terms))

    return deterministic_argmax(scored)
```

## 8. 执行与真实状态更新

vLLM Adapter 必须原子提交完整 BatchPlan，不能只提交 Prefill cap 后再让原调度器重新选择请求。

执行后只用真实结果更新：

$$
Q_{k+1}^P
=
\left[
Q_k^P-\mu_k^{P,actual}
\right]^+
+
A_k^P
$$

$$
Z_{k+1}^F
=
\left[
Z_k^F
+
\left(1-\epsilon^F\right)M_k^F
-
\epsilon^F S_k^F
\right]^+
$$

$$
Z_{k+1}^D
=
\left[
Z_k^D
+
\left(1-\epsilon^D\right)M_k^D
-
\epsilon^D S_k^D
\right]^+
$$

`A_k^P` 来自 Adapter 实际接收的新请求 prompt token 数；正常执行的
`mu_k^{P,actual}` 来自本轮实际 `SchedulerOutput`，并与所选完整 BatchPlan
核对。取消或 Preemption 改变可见 Prefill 工作量时，下一轮仅以实际
Snapshot 进行确定性校正。债务只聚合同一轮 ledger 已一次性结算的真实
TTFT/TBT success 和 miss。

每个 TTFT/TBT obligation 只能结算一次。Decode 返回非 EOS token 时，从真实返回时间创建下一条 TBT obligation；返回 EOS 时完成请求、释放 KV，不创建新 obligation。

## 9. Controller 伪代码

```text
schedule_once(vllm_state):
    snapshot = adapter.make_snapshot(vllm_state)
    control = state_store.current(snapshot)

    plans = candidate_generator.generate(snapshot)
    predictions = predictor.predict(snapshot, plans)
    predictions = consequence_estimator.attach(snapshot, plans, predictions)

    safe_result = safe_set.filter(snapshot, plans, predictions)

    if safe_result.safe_candidates is empty:
        selected = fallback.build(snapshot)
    else:
        selected = dpp_selector.select(
            snapshot,
            control,
            safe_result.safe_candidates,
        )

    execution = adapter.execute_exact_plan(selected)
    observation = observer.collect(execution)
    state_store.update_from_actual(observation)
    decision_log.write_async(snapshot, selected, observation)
```

## 10. 建议代码结构

```text
dpp_scheduler/
  contracts.py
  controller.py
  candidate_generator.py
  predictor.py
  consequence_estimator.py
  safe_set.py
  dpp_selector.py
  fallback.py
  state_store.py
  observer.py
  vllm_adapter.py
```

除 `vllm_adapter.py` 外，其他模块不得直接导入 vLLM 内部类型。这样 vLLM commit 变化时，主要只需修改 Adapter。

## 11. 实施顺序

1. 记录 vLLM commit，建立 stock scheduler 的 iteration 日志；
2. 实现公共数据结构、Snapshot 和精确执行 Adapter；
3. 实现 Candidate Generator，并用固定规则临时选计划；
4. 在 DGX Spark 上采集同配置的逐轮 Batch 基础信息，离线进行特征工程并训练 Predictor；
5. 加入 Safe-Set、Rolling KV 和 Fallback；
6. 加入 DPP Selector、SLO Ledger 和真实反馈更新；
7. 在完全相同的模型、请求集和 vLLM 参数下与 stock scheduler 对比。

第一版需要冻结但不必写死的参数包括：`C_tok`、`C_seq`、三个 Prefill cap、`u`、`H`、`R0`、TTFT/TBT SLO、允许违约比例和 `V`。这些参数应由 stock baseline 与同机 profiling 得到，而不是凭空设定。

## 12. 首版验收条件

- 候选、预测、Safe-Set 和选择结果使用同一 snapshot；
- 选中的请求集合与 vLLM 实际执行集合完全一致；
- Scheduler 中不存在固定输出长度或剩余 Decode token；
- KV 检查从不允许当前物理容量越界；
- Safe-Set 为空时路径确定且可记录；
- SLO obligation 不重复结算；
- 相同状态和配置得到相同选择；
- Predictor、Safe-Set、DPP Selector 均可用同接口的替代实现单独替换。
