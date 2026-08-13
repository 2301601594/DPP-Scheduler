# Prefill-Budget DPP Scheduler：Codex 执行说明

## 0. 文件用途

本文件是第一版 Prefill-Budget Drift-Plus-Penalty（DPP）调度器的工程执行契约。Codex 在目标仓库中工作时，应先完整阅读本文件，再检查仓库状态，并严格按照 M0～M7 的顺序实现、验证和记录结果。

本文件不是一次性设计文档，而是持续更新的执行清单。每完成一个里程碑，Codex 必须：

1. 运行该阶段要求的测试；
2. 记录关键命令、配置和结果；
3. 更新文末“进度记录”；
4. 只有验收条件全部满足，才进入下一阶段。

如果仓库代码、vLLM 版本或实际 API 与本文假设不同，Codex 不得静默猜测。应先定位差异，说明影响，采取最小兼容改动，并把差异记录在文档中。

---

## 1. 总体研究目标

在单 GPU、vLLM V1、混合连续批处理场景下，实现一个闭环调度器：

- 保留默认 FCFS 请求顺序；
- 保留默认 decode 优先级和 decode 服务规则；
- 不改变模型执行器、PagedAttention 和采样逻辑；
- 每个调度 iteration 只动态决定最大 prefill token 数：

\[
b_k^P=\text{第 }k\text{ 轮允许调度的最大 prefill token 数};
\]

- 使用资源约束和 TBT Guard 排除不可行或不安全的候选；
- 使用物理队列、虚拟队列和单位时间 DPP 得分，从安全候选中选择最终 \(b_k^{P*}\)；
- 通过执行反馈更新队列，形成真正的闭环控制器。

完整决策链：

\[
S_k
\rightarrow
\{\Pi_k(b):b\in\mathcal B_0\}
\rightarrow
\mathcal B_k^{\mathrm{res}}
\rightarrow
\mathcal B_k^{\mathrm{safe}}
\rightarrow
\Phi_k(b)
\rightarrow
b_k^{P*}
\rightarrow
\text{实际执行与队列更新}.
\]

---

## 2. 第一版范围

### 2.1 默认实验边界

除非仓库已有明确配置，否则采用以下起始范围，并在 M0 中根据 RTX 5070 实测结果冻结最终数值。

| 项目 | 第一版要求 |
|---|---|
| 推理框架 | vLLM V1，固定 release/tag 和 commit |
| GPU | 单张 RTX 5070 |
| 模型 | 优先 Qwen2.5-3B-Instruct，FP16 或 BF16 |
| 最大上下文 | 从 4096 token 起步 |
| 调度策略 | FCFS |
| Chunked prefill | 开启 |
| Prefix caching | 关闭 |
| Speculative decoding | 关闭 |
| Async scheduling | 关闭 |
| LoRA | 关闭 |
| 多模态输入 | 关闭 |
| Pipeline/Tensor parallel | 关闭，单卡执行 |
| Preemption | 第一版候选检查中禁止 |
| 请求类别 | 第一版统一为 \(C=1\) |
| 总 token budget | 初始建议 2048，M0 实测后冻结 |
| 最大并发序列数 | 初始建议 32，M0 实测后冻结 |
| 动态控制变量 | 仅 \(b_k^P\) |

候选动作集合初始建议：

\[
\mathcal B_0=\{0,128,256,512,1024,1536,2048\}.
\]

如果冻结后的总 token budget 小于 2048，应相应缩小候选集合，并确保：

\[
0\in\mathcal B_0,
\qquad
b\le B^{\mathrm{total}},\ \forall b\in\mathcal B_0.
\]

### 2.2 明确不做的内容

第一版不得主动扩展到以下内容：

- PD 分离；
- 多 GPU 或多副本路由；
- 改变请求 FCFS 顺序；
- 根据 DPP 权重重排单个请求；
- 动态调整总 `max_num_batched_tokens`；
- Prefix cache 感知调度；
- 输出长度预测；
- Speculative decoding；
- 允许 preemption 并对其建模；
- 多请求类别差异化 SLO；
- 复杂神经网络性能预测器；
- 在线训练性能预测器。

这些内容只能在第一版验收完成后作为后续工作加入。

---

## 3. 不可破坏的设计原则

### 3.1 区分总 token budget 和 prefill cap

总 token budget 固定为：

\[
B^{\mathrm{total}}.
\]

动态动作是：

\[
b_k^P.
\]

每轮必须满足：

\[
D_k+\mu_k^P(b_k^P)\le B^{\mathrm{total}},
\]

其中：

- \(D_k\)：本轮实际安排的 decode token 数；
- \(\mu_k^P(b)\)：候选 \(b\) 下实际安排的 prefill token 数；
- 通常 \(\mu_k^P(b)\le b\)。

不得把 \(b\) 当成整轮总 token 数，也不得在每轮修改全局 `max_num_batched_tokens` 来伪装成 prefill cap。

### 3.2 保留默认 Scheduler 的顺序和语义

第一版只控制 prefill token 总量，不使用 DPP 分数改变请求次序。相同状态和相同固定 \(b\) 下，影子计划与实际 Scheduler 应在以下方面一致：

- 被调度的请求；
- 每个请求的 token 数；
- prefill/decode token 分类；
- 总 token 数；
- 是否发生新请求 admission；
- 是否触发 preemption。

### 3.3 适配 vLLM V1 的统一 token 语义

vLLM V1 不一定把请求显式拆成独立 prefill/decode 对象。对于请求 \(i\)，应基于 prompt 长度和已经计算的 token 数识别剩余 prefill：

\[
r_{i,k}^{P}
=
\max\{L_i^{\mathrm{prompt}}-n_{i,k}^{\mathrm{computed}},0\}.
\]

如果 Scheduler 本轮准备给该请求调度 \(n_{i,k}^{\mathrm{new}}\) 个 token，则：

\[
x_{i,k}^{P}
=
\min\{n_{i,k}^{\mathrm{new}},r_{i,k}^{P}\},
\]

\[
d_{i,k}
=
n_{i,k}^{\mathrm{new}}-x_{i,k}^{P}.
\]

实现前必须根据锁定版本的 Request 和 Scheduler 字段验证这个映射，不能仅按字段名称猜测。

### 3.4 候选评估不得污染真实状态

枚举候选时必须使用只读快照或纯函数式影子计划。不得为每个候选直接调用会修改真实状态的 KV allocator、请求队列或 Scheduler 方法。

真实 allocator 和 Scheduler 状态只能对最终选中的 \(b_k^{P*}\) 更新一次。

如果锁定版本没有安全的 dry-run API，第一版使用保守 KV 估算，并在真正执行时继续让 vLLM 原生 allocator 做最终检查。

### 3.5 硬约束先于 DPP 打分

正确顺序是：

1. 构造候选对应的实际影子计划；
2. 进行资源约束检查；
3. 进行执行时间预测和 OOD 检查；
4. 执行 TBT Guard；
5. 只对安全候选进行 DPP 打分。

不得先选择最高 DPP 得分，再检查该候选能否执行。

### 3.6 使用预测量选择，使用实际量更新

选择 \(b_k^{P*}\) 时使用预测量；iteration 执行结束后，物理队列和虚拟队列必须使用实际观测结果更新，不能使用影子计划或预测结果代替真实反馈。

### 3.7 `b=0` 不天然安全

\(b=0\) 只表示不加入额外 prefill 干扰。decode-only iteration 仍可能违反 TBT SLO。

如果没有候选通过 Guard，应进入应急 fallback，同时记录：

```text
fallback_reason=no_safe_candidate
```

如果连 \(b=0\) 都无法满足 TBT Guard，还应记录：

```text
decode_only_already_unsafe=true
```

不得把此情况计为一次正常的“安全选择”。

---

## 4. Codex 工作规则

### 4.1 开始工作时

Codex 必须首先：

1. 阅读仓库中的 `AGENTS.md`、README、开发文档和现有实验脚本；
2. 运行 `git status --short`，识别并保留用户已有修改；
3. 定位 vLLM 版本、commit、V1 Scheduler、Request、KV cache manager 和 benchmark 入口；
4. 搜索现有日志、测试、配置系统和自定义 Scheduler 接口；
5. 输出一个简短执行计划，并说明当前准备完成哪个里程碑；
6. 不得覆盖无关文件，不得使用破坏性 Git 命令。

优先使用 `rg` 和 `rg --files` 搜索仓库。所有关键判断必须以锁定版本代码为准。

### 4.2 每次只推进一个里程碑

除非用户明确要求，否则一次工作应聚焦当前最早的未完成里程碑。不得在 M2 的等价性测试没有通过时直接提交 M6 的 DPP 控制器。

### 4.3 改动策略

- 优先新增独立模块和最小适配层；
- 避免复制整份上游 Scheduler；
- 如果必须修改上游核心循环，保持 diff 最小，并添加清楚注释；
- 所有新功能都应通过配置开关启用；
- 默认关闭实验性 DPP 行为，避免破坏 stock 路径；
- stock 模式不得产生 DPP 决策开销；
- 日志字段和结果格式保持向后兼容。

### 4.4 每轮交付说明

每次实现后必须汇报：

- 完成了哪个里程碑；
- 修改了哪些文件；
- 核心行为变化；
- 运行了哪些测试；
- 测试结果；
- 尚未完成或存在的风险；
- 下一步建议。

---

## 5. 推荐代码结构

根据仓库已有组织方式调整目录位置，但模块职责应保持清晰：

```text
dpp_scheduler/
├── config.py              # 动作集合、SLO、权重和运行模式
├── state.py               # 只读调度状态快照
├── candidate_builder.py   # 候选动作和影子计划
├── predictor.py           # P50/P95预测、置信余量和OOD
├── guards.py              # 资源检查与TBT Guard
├── queues.py              # 物理/虚拟队列
├── scorer.py              # 单位时间DPP打分
├── controller.py          # 完整候选选择流程
├── telemetry.py           # iteration/request/candidate日志
├── vllm_adapter.py        # 与锁定版本vLLM对接
└── tests/
    ├── test_state.py
    ├── test_candidates.py
    ├── test_guards.py
    ├── test_queues.py
    ├── test_scorer.py
    ├── test_fixed_cap.py
    └── test_stock_equivalence.py
```

建议支持以下模式：

```text
scheduler_mode=stock
scheduler_mode=passthrough
scheduler_mode=fixed_cap
scheduler_mode=guard_only
scheduler_mode=dpp
```

各模式定义：

| 模式 | 行为 |
|---|---|
| `stock` | 完全使用原始 vLLM Scheduler |
| `passthrough` | 进入自定义接口，但返回完整可用 prefill budget |
| `fixed_cap` | 每轮使用固定 prefill cap |
| `guard_only` | 从候选中选择最大的安全 cap，不使用 DPP 队列打分 |
| `dpp` | 在安全候选中使用单位时间 DPP 得分选择 |

---

## 6. 核心数据结构

具体字段名可根据仓库风格调整，但必须表达以下信息。

```python
@dataclass(frozen=True)
class SchedulerSnapshot:
    iteration_id: int
    now: float
    running_requests: tuple
    waiting_requests: tuple
    total_token_budget: int
    max_num_seqs: int
    free_kv_blocks: int | None
    physical_prefill_queue: float
    ttft_virtual_queue: float
    tbt_virtual_queue: float
```

```python
@dataclass(frozen=True)
class ShadowPlan:
    prefill_cap: int
    per_request_tokens: tuple
    num_prefill_tokens: int
    num_decode_tokens: int
    num_prefill_requests: int
    num_decode_requests: int
    num_new_requests: int
    estimated_new_kv_blocks: int | None
    would_preempt: bool
```

```python
@dataclass
class CandidateEvaluation:
    prefill_cap: int
    plan: ShadowPlan
    resource_feasible: bool = False
    predictor_in_domain: bool = False
    tbt_safe: bool = False
    tau_p50_ms: float | None = None
    tau_p95_ms: float | None = None
    uncertainty_ms: float | None = None
    score: float | None = None
    rejection_reason: str | None = None
```

所有时间字段必须显式标记单位，内部统一使用秒或毫秒，不能混用。

---

## 7. 分阶段实施计划

## M0：冻结环境与实验契约

### 任务

1. 记录硬件、操作系统、GPU、显存、驱动、CUDA、Python、PyTorch 和 vLLM 版本；
2. 固定 vLLM tag/commit；
3. 验证 V1 引擎和自定义 `scheduler_cls` 接口；
4. 固定模型、dtype、最大上下文和随机种子；
5. 确定能在 RTX 5070 上稳定运行的：
   - `max_num_batched_tokens`；
   - `max_num_seqs`；
   - `gpu_memory_utilization`；
6. 固定候选动作集合 \(\mathcal B_0\)；
7. 定义 TTFT、TBT、TPOT、E2EL、goodput 和 SLO attainment 的计算口径；
8. 建立统一实验配置文件和结果目录；
9. 建立实验元数据自动记录脚本。

### SLO 起始方法

先在低负载下测量基准分布，再设定：

\[
S^{\mathrm{TTFT}}
=
\alpha_F\tau_{\mathrm{TTFT}}^{\mathrm{low-load}},
\]

\[
S^{\mathrm{TBT}}
=
\alpha_D\tau_{\mathrm{TBT}}^{\mathrm{low-load}},
\]

其中 \(\alpha_F,\alpha_D>1\)。必须记录倍数和选择理由。

### 验收条件

- 固定配置下服务能够稳定启动和完成请求；
- 没有 OOM、死锁或意外 preemption；
- 环境和参数能从配置文件完全重建；
- TTFT/TBT 的测量口径已经通过小样例人工核对；
- `EXPERIMENT_CONTRACT.md` 或等价文档已经生成。

### 交付物

- 环境锁定记录；
- 实验配置；
- SLO 定义；
- 一条可复现的启动命令；
- 一条可复现的小规模 benchmark 命令。

---

## M1：建立默认 vLLM 基线

### 任务

1. 使用 stock Scheduler 运行低、中、高负载；
2. 先使用合成长度分布，再使用真实数据集；
3. 每种配置至少使用多个随机种子或重复运行；
4. 保存汇总指标和逐请求明细；
5. 找到系统饱和附近的请求率 \(\lambda_{\max}\)；
6. 建立标准负载档位，例如：

\[
\lambda\in
\{0.25,0.5,0.75,0.9,1.0,1.1\}\lambda_{\max}.
\]

### 合成 workload 起始矩阵

| Prompt token | Output token |
|---:|---:|
| 128 | 128 |
| 512 | 128 |
| 1024 | 128 |
| 2048 | 128 |
| 512 | 512 |
| 2048 | 512 |

根据冻结后的最大上下文删除不合法组合。

### 必须记录的指标

- TTFT P50/P95/P99；
- TBT 或 ITL P50/P95/P99；
- TPOT P50/P95/P99；
- E2EL P50/P95/P99；
- request throughput；
- input/output token throughput；
- TTFT attainment；
- TBT attainment；
- joint goodput；
- preemption 数量；
- waiting/running 请求数；
- KV Cache 使用率；
- 失败请求和错误类型。

### 验收条件

- 基线结果能够稳定复现；
- 重复运行差异有统计记录；
- 已覆盖低负载、饱和附近和过载；
- 能明确指出默认 Scheduler 在什么负载下出现 TTFT/TBT 权衡或违约。

### 交付物

- stock benchmark 脚本；
- 原始 JSON/JSONL/CSV 结果；
- 汇总脚本；
- baseline 报告。

---

## M2：实现 Pass-through Scheduler

### 目标

建立最小自定义入口，但不改变任何调度决策。

### 任务

1. 定位锁定版本的 V1 Scheduler 主循环；
2. 增加可配置的 prefill-budget 选择接口；
3. `passthrough` 模式始终返回完整可用 prefill budget；
4. stock 路径保持原样；
5. 增加逐 iteration 的调度结果对比日志；
6. 建立 stock 与 passthrough 的确定性小型测试；
7. 建立端到端等价性 benchmark。

建议接口：

```python
def select_prefill_budget(self, snapshot: SchedulerSnapshot) -> int:
    return self.max_num_scheduled_tokens
```

具体返回值应根据当前 decode 占用和锁定版本的预算语义调整，但必须保证 passthrough 与 stock 等价。

### 验收条件

在同一输入、配置和种子下：

- 请求顺序一致；
- 每轮被调度的请求一致；
- 每请求 scheduled token 数一致；
- 总 scheduled token 数一致；
- preemption 行为一致；
- 最终输出一致；
- 性能差异处于重复运行噪声范围内；
- stock 模式不承担候选枚举或预测开销。

若无法实现逐轮完全一致，必须记录具体差异及上游非确定性来源，不能只比较最终吞吐量。

### 交付物

- Pass-through Scheduler；
- 等价性单元测试；
- 端到端等价性报告。

---

## M3：实现固定 Prefill Cap

### 目标

先证明调度器可以独立控制 prefill token 上限，再引入预测器和 DPP。

### 任务

1. 增加 `fixed_cap` 配置；
2. 在保留 stock 请求顺序的前提下限制 prefill token；
3. decode token 只消耗总 budget，不消耗 prefill budget；
4. 记录配置 cap 和实际 prefill 服务量；
5. 对候选集合中的每个固定 \(b\) 运行测试；
6. 测试纯 prefill、纯 decode 和 mixed batch；
7. 测试 prefill 请求剩余量小于 cap、等于 cap 和大于 cap；
8. 测试 cap 不能被单请求 chunk 或跨请求累计突破。

必须满足：

\[
\mu_k^P\le b_k^P,
\]

\[
D_k+\mu_k^P\le B^{\mathrm{total}}.
\]

### 必测边界

- \(b=0\) 且存在 decode 请求；
- \(b=0\) 且只有 prefill 请求；
- \(b=B^{\mathrm{total}}\)；
- decode 已占满总 budget；
- 长 prompt 被分成多轮 chunk；
- 多个短 prompt 共同消耗 cap；
- 接近 `max_num_seqs`；
- KV 空间紧张但尚未触发 preemption。

只有 prefill 请求且长期选择 \(b=0\) 时会造成空转；固定 cap 模式可以允许这是用户显式配置造成的行为，但 DPP 模式必须设计进展保障。

### 验收条件

- 所有固定 cap 都能稳定执行；
- 实际 prefill token 从不超过 cap；
- 总 token 从不超过总 budget；
- 请求 FCFS 顺序未被 DPP 逻辑改变；
- \(b=B^{\mathrm{total}}\) 时行为接近 passthrough；
- 日志可以重建每轮 prefill/decode 构成。

### 交付物

- Fixed-cap Scheduler；
- 边界单元测试；
- 固定 cap 扫描结果；
- TTFT、TBT、throughput 随 cap 变化的初步曲线数据。

---

## M4：建立 Telemetry 和执行时间预测器

### 目标

预测候选 mixed batch 的 iteration 完成时间：

\[
\widehat\tau_{50,k}(b),
\qquad
\widehat\tau_{95,k}(b).
\]

### Iteration 日志字段

至少记录：

```text
run_id
iteration_id
schedule_start_time
model_execute_start_time
model_execute_end_time
output_ready_time
actual_iteration_latency_ms
selected_prefill_cap
actual_prefill_tokens
actual_decode_tokens
num_prefill_requests
num_decode_requests
mean_prefill_chunk_tokens
max_prefill_chunk_tokens
mean_decode_context_tokens
max_decode_context_tokens
num_running_requests
num_waiting_requests
kv_cache_usage_ratio
free_kv_blocks
preemption_count
fallback_reason
```

如果某字段在锁定版本无法低成本采集，记录原因和替代字段。

预测目标优先定义为：

\[
\tau_k
=
t_k^{\mathrm{output-ready}}
-
t_k^{\mathrm{schedule-start}}.
\]

不得只预测 CUDA kernel 时间，因为 TBT Guard 保护的是用户可观察的等待间隔。

### 数据采集

使用 `fixed_cap` 模式主动扫描：

- 不同到达率；
- 不同 prompt/output 长度；
- 不同 decode 上下文长度；
- 不同 prefill/decode 混合比例；
- 不同 KV 使用率；
- 不同并发序列数；
- 所有候选 \(b\)。

训练数据必须覆盖非默认 Scheduler 常选的候选，避免 DPP 评估时大面积 OOD。

### 第一版模型

优先使用以下一种：

1. 离线 profiling 查找表/插值；或
2. LightGBM 分位数回归，分别预测 P50 和 P95。

不要第一版直接使用复杂神经网络。

训练集、验证集和测试集应按完整 workload、run 或随机种子划分，不能随机打散相邻 iteration 造成数据泄漏。

### 不确定性校准

在验证集上计算残差并得到安全余量：

\[
\delta^{\mathrm{unc}}
=
Q_q\left(\tau-\widehat\tau_{95}\right),
\]

其中 \(q\) 初始可取 0.95 或 0.99。

要求验证：

\[
\Pr\left[
\tau_k
\le
\widehat\tau_{95,k}
+
\delta_k^{\mathrm{unc}}
\right]
\ge 0.95.
\]

### OOD 处理

必须实现 predictor 支持域检查。出现以下情况时，候选不得被乐观接受：

- 特征超出训练覆盖范围；
- 预测为 NaN/Inf；
- 模型文件缺失或版本不匹配；
- 置信度不足；
- 特征 schema 不匹配。

### 验收条件

- 日志时间点定义明确并经过人工核对；
- 数据集能从原始日志重复生成；
- P50/P95 模型在独立测试集上完成评估；
- P95+余量满足目标覆盖率；
- OOD 样例被保守拒绝；
- 推理开销被测量，并显著小于一次 iteration 时间。

### 交付物

- Telemetry 模块；
- profiling 数据生成脚本；
- predictor 训练和评估脚本；
- 固定 schema 的模型文件；
- 预测准确性与校准报告。

---

## M5：实现候选影子计划、资源检查与 TBT Guard

### 目标

为每个候选 \(b\) 生成真实可解释的影子计划，并得到安全候选集合：

\[
\mathcal B_k^{\mathrm{safe}}.
\]

### 候选构造

先由 decode 占用对候选做总 budget 初筛：

\[
R_k^{\mathrm{tok}}
=
\left[B^{\mathrm{total}}-D_k\right]^+,
\]

\[
\mathcal B_k^{\mathrm{pre}}
=
\{b\in\mathcal B_0:b\le R_k^{\mathrm{tok}}\}.
\]

再按 stock FCFS 顺序生成每个候选的 prefill 计划。对于请求 \(i\)：

\[
x_{i,k}^P(b)
=
\min\{r_{i,k}^P,u_{i,k}^P,h_k(b)\},
\]

其中 \(u_{i,k}^P\) 是锁定版本对该请求的单轮限制，\(h_k(b)\) 是尚未使用的 prefill cap。

### 资源检查顺序

按以下顺序检查并记录首个拒绝原因：

1. `total_token_budget`；
2. `max_num_seqs`；
3. `context_length`；
4. `kv_cache`；
5. `would_preempt`；
6. `predictor_ood`；
7. `tbt_guard`。

第一版硬性要求：

\[
N_k^{\mathrm{preempt}}(b)=0.
\]

### KV 检查

优先复用上游公开、无副作用的容量查询；如果不存在，使用基于 block size 的保守估计。不得在候选循环中调用会实际分配或释放 block 的接口。

真正执行最终候选时，仍由原生 allocator 做最终 admission。若真实执行结果与影子计划不一致，必须记录 mismatch 并保守 fallback，不能静默继续。

### Predictor 特征

第一版至少包含：

\[
\mathbf z_k(b)=
\left(
D_k,
\mu_k^P(b),
N_k^D,
N_k^P(b),
\overline L_k^D,
L_{k,\max}^D,
\overline x_k^P(b),
x_{k,\max}^P(b),
\kappa_k^{\mathrm{KV}}(b)
\right).
\]

### TBT Guard

对每个受保护 decode 请求：

\[
a_{i,k}^D
=
s_k-t_i^{\mathrm{last}},
\]

要求：

\[
a_{i,k}^D
+
\widehat\tau_{95,k}(b)
+
\delta_k^{\mathrm{unc}}(b)
\le
S_i^{\mathrm{TBT}}.
\]

如果当前没有 decode 请求，TBT Guard 直接通过。

还必须确保所有受保护 decode 请求已经包含在固定的 decode 计划中。Prefill cap 无法修复 stock Scheduler 没有服务某个紧迫 decode 请求的问题。

### Guard-only 策略

在进入 DPP 前先实现：

\[
b_k^{*}
=
\max\mathcal B_k^{\mathrm{safe}}.
\]

它用于单独验证 Predictor + Guard 是否有效。

### 验收条件

- Shadow plan 是纯函数或对真实状态无副作用；
- 固定 \(b\) 下影子计划与实际计划高度一致；
- 每个拒绝候选都有唯一、可统计的原因；
- 资源不可行候选不会进入 predictor/DPP；
- Guard-only 能降低 TBT 违约，且不会造成明显死锁；
- `no_safe_candidate` 和 `decode_only_already_unsafe` 被正确区分；
- 候选枚举、预测和打分总开销有独立统计。

### 交付物

- Snapshot 与 ShadowPlanner；
- 资源约束模块；
- TBT Guard；
- Guard-only Scheduler；
- shadow-vs-real 一致性测试；
- Guard-only benchmark 报告。

---

## M6：实现 DPP 队列、打分与闭环更新

### 目标

在安全候选中用单位时间 DPP 得分选择 \(b_k^{P*}\)，并使用真实执行结果更新队列。

第一版只有一个请求类别，因此先省略类别下标 \(c\)。

### Prefill 物理队列

\[
Q_{k+1}^{P}
=
\left[Q_k^P-\mu_k^P\right]^+
+A_k^P.
\]

其中：

- \(A_k^P\)：本轮期间实际新到达的 prompt token；
- \(\mu_k^P\)：本轮实际计算完成的 prefill token。

实现时必须避免把同一请求的 prompt token 重复计入 arrival。

### TTFT 虚拟队列

\[
Z_{k+1}^{F}
=
\left[
Z_k^F
+
\rho^F A_k^R
-
G_k^F
\right]^+.
\]

其中：

- \(A_k^R\)：本轮期间实际到达的新请求数；
- \(G_k^F\)：本轮实际按时产生首 token 的请求数；
- \(\rho^F\)：目标 TTFT attainment。

### TBT 虚拟队列

\[
Z_{k+1}^{D}
=
\left[
Z_k^D
+
V_k^D
-
\epsilon^D N_k^D
\right]^+.
\]

其中：

- \(V_k^D\)：本轮实际 TBT 违约 interval 数；
- \(N_k^D\)：本轮实际产生的 token interval 数；
- \(\epsilon^D\)：允许的最大 TBT violation ratio。

必须使用相邻 token 的实际输出时间计算 TBT，不能用请求平均 TPOT 替代。

### 第一版 DPP 得分

对所有安全候选计算：

\[
\Phi_k(b)
=
\frac{
w_Q\bar Q_k^P\bar\mu_k^P(b)
+
\eta_F w_F\bar Z_k^F\widehat G_k^F(b)
-
\eta_D w_D\bar Z_k^D
\left[
\widehat V_k^D(b)
-
\epsilon^D\widehat N_k^D(b)
\right]
}{
\max\{\widehat\tau_{50,k}(b),\tau_{\min}\}
}.
\]

带横线的量表示归一化量。必须在配置中记录归一化常数，避免 prefill token 队列的数量级完全压倒请求数或违约数。

第一版可以令完成请求 goodput 权重和风险软成本为零，因为 goodput 尚无可靠 predictor，preemption 已通过硬约束禁止：

\[
V_{\mathrm{DPP}}=0,
\qquad
\gamma=0.
\]

### 选择规则

\[
\Phi_k^{\max}
=
\max_{b\in\mathcal B_k^{\mathrm{safe}}}\Phi_k(b),
\]

\[
\mathcal B_k^{\mathrm{tie}}
=
\left\{
b:
\Phi_k(b)\ge\Phi_k^{\max}-\varepsilon_{\mathrm{tie}}
\right\},
\]

\[
b_k^{P*}
=
\min\mathcal B_k^{\mathrm{tie}}.
\]

近似同分时选择更小 cap。

### 进展保障

为了避免无 decode 时 DPP 因异常状态长期选择 \(b=0\)，需要明确处理：

- 如果存在 prefill backlog、没有 decode、且至少一个正 cap 可行，应保证选择正 cap；
- 该规则必须记录触发次数；
- 不能绕过 KV、上下文和最大序列数等硬约束。

### 队列更新时机

必须明确定义 arrival/service 计数窗口，保证每个事件只被记一次。建议在 iteration 完成回调中：

1. 收集本轮实际服务和输出时间；
2. 收集自上次更新以来的实际到达；
3. 更新物理/虚拟队列；
4. 把更新后状态用于下一轮 snapshot。

### 验收条件

- 队列更新单元测试覆盖零到达、零服务、过量服务和违约边界；
- 所有队列始终非负；
- 相同状态下得分和选择确定可复现；
- 单位时间分母不会为零；
- 预测用于选择，实际值用于更新；
- 无 decode 且有 prefill backlog 时不会永久空转；
- DPP 模式可以端到端稳定运行；
- 每轮可从日志重建候选、分数、选择和队列更新。

### 交付物

- QueueManager；
- DPPScorer；
- DPPController；
- 完整候选日志；
- 单元测试和端到端测试；
- 小规模闭环实验报告。

---

## M7：完整 benchmark、消融和结论

### 必须比较的调度器

1. stock vLLM；
2. fixed-cap-small；
3. fixed-cap-medium；
4. fixed-cap-large；
5. guard-only；
6. full DPP；
7. DPP 去掉单位时间分母；
8. DPP 去掉 TTFT/TBT 虚拟队列；
9. 如果实现成本可控，加入使用真实 iteration latency 的 offline Oracle。

固定 cap 的 small/medium/large 必须从 M3 扫描结果中选取，并记录选择方法，不能专门挑选对 DPP 有利的点。

### Workload 维度

- 合成长度分布；
- 真实 prompt/output 长度分布；
- 低负载；
- 中负载；
- 饱和附近；
- 过载；
- prompt-heavy；
- decode-heavy；
- mixed workload；
- 至少多个随机种子。

### 核心评价问题

实验必须能够回答：

1. 动态 prefill cap 是否优于固定 cap？
2. Guard 是否降低 P95/P99 TBT 和 TBT violation ratio？
3. DPP 虚拟队列是否改善长期 TTFT/TBT attainment？
4. 单位时间 DPP 是否优于不除以 iteration 时间的得分？
5. Predictor 误差如何影响安全性和性能？
6. 调度决策开销占 iteration 时间的多少？
7. 取得的 SLO 改善是否以明显吞吐量下降为代价？
8. DPP 是否能在不同负载和长度分布下自适应选择不同 cap？

### 统计要求

- 保存逐请求和逐 iteration 原始数据；
- 报告均值、分位数和重复运行波动；
- 不只报告平均 TPOT，必须报告 TBT/ITL 尾部；
- 所有对照使用相同模型、硬件、数据、请求轨迹和全局配置；
- 对同一请求轨迹尽可能进行配对比较；
- 报告失败请求、OOM、preemption 和 fallback；
- 图表生成脚本必须可重复运行。

### 验收条件

- 所有对照都在相同实验契约下完成；
- 结果目录包含配置、环境、commit、命令和原始数据；
- benchmark 能从干净环境按文档复现；
- 完成消融实验；
- 最终报告明确说明正面结果、负面结果和适用边界；
- 没有只挑选最佳 seed 或最佳负载进行汇报。

### 交付物

- 完整 benchmark runner；
- 汇总与画图脚本；
- 实验结果表格和图；
- predictor 校准报告；
- shadow-plan 一致性报告；
- 最终研究结论。

---

## 8. 调度器主流程伪代码

```python
def schedule_iteration():
    snapshot = snapshot_state()

    decode_plan = build_stock_decode_plan(snapshot)
    decode_tokens = count_decode_tokens(decode_plan)

    candidates = prune_candidates_by_total_budget(
        action_set=config.prefill_caps,
        total_budget=config.total_token_budget,
        decode_tokens=decode_tokens,
    )

    evaluations = []

    for prefill_cap in candidates:
        plan = build_shadow_prefill_plan(
            snapshot=snapshot,
            decode_plan=decode_plan,
            prefill_cap=prefill_cap,
            preserve_stock_order=True,
        )

        rejection_reason = check_resource_constraints(snapshot, plan)
        if rejection_reason is not None:
            evaluations.append(rejected(plan, rejection_reason))
            continue

        features = build_predictor_features(snapshot, plan)
        prediction = latency_predictor.predict(features)

        if prediction.is_ood:
            evaluations.append(rejected(plan, "predictor_ood"))
            continue

        if not tbt_guard_passes(snapshot, decode_plan, prediction):
            evaluations.append(rejected(plan, "tbt_guard"))
            continue

        if config.mode == "guard_only":
            score = float(prefill_cap)
        else:
            score = dpp_score(snapshot, plan, prediction)

        evaluations.append(accepted(plan, prediction, score))

    selected = select_best_candidate(evaluations)

    if selected is None:
        selected = emergency_fallback(prefill_cap=0)

    real_output = execute_stock_scheduler_with_prefill_cap(
        selected.plan.prefill_cap
    )

    observed = collect_actual_iteration_result(real_output)
    compare_shadow_and_actual(selected.plan, observed)
    update_queues_with_actual_observations(observed)
    log_iteration(snapshot, evaluations, selected, observed)

    return real_output
```

如果锁定版本无法在不修改状态的情况下先构造完整 stock decode plan，Codex 应保留上述语义，在 Scheduler 原生遍历顺序中冻结 decode 服务，并用纯函数式适配层构造候选；不得通过多次真实执行 Scheduler 来枚举候选。

---

## 9. 日志与结果格式

### 9.1 Run 级元数据

每次运行至少保存：

```text
run_id
timestamp
git_commit
git_dirty
vllm_version
model
model_revision
dtype
gpu_name
gpu_memory
driver_version
cuda_version
torch_version
python_version
scheduler_mode
random_seed
dataset
request_rate
burstiness
max_model_len
max_num_batched_tokens
max_num_seqs
gpu_memory_utilization
prefill_caps
fixed_prefill_cap
ttft_slo_ms
tbt_slo_ms
target_ttft_attainment
allowed_tbt_violation_ratio
predictor_version
```

### 9.2 Candidate 级日志

每轮每个候选至少保存：

```text
run_id
iteration_id
prefill_cap
planned_prefill_tokens
planned_decode_tokens
planned_prefill_requests
planned_decode_requests
estimated_kv_blocks
tau_p50_ms
tau_p95_ms
uncertainty_ms
resource_feasible
predictor_in_domain
tbt_safe
score
selected
rejection_reason
decision_overhead_us
```

### 9.3 Request 级日志

至少保存：

```text
request_id
arrival_time
prompt_tokens
output_tokens
first_token_time
finish_time
ttft_ms
e2el_ms
token_timestamps
tbt_values_ms
ttft_slo_met
tbt_slo_met
joint_slo_met
preemption_count
```

日志应优先使用 JSONL、Parquet 或结构稳定的 CSV。时间戳与持续时间字段必须区分。

---

## 10. 测试要求

### 10.1 单元测试

至少覆盖：

- prefill/decode token 分类；
- 候选 budget 初筛；
- FCFS 影子 token 分配；
- cap 不被突破；
- 总 token budget；
- `max_num_seqs`；
- 上下文长度；
- KV block 保守估算；
- preemption 硬拒绝；
- predictor OOD；
- TBT Guard 边界等于 SLO；
- 无 decode 请求时 Guard；
- `b=0` fallback；
- DPP tie-break；
- 队列非负性；
- arrival/service 不重复计数；
- 无 decode、有 backlog 时的进展保障。

### 10.2 属性测试或随机测试

对随机生成的小型状态验证：

\[
0\le\mu_k^P(b)\le b,
\]

\[
D_k+\mu_k^P(b)\le B^{\mathrm{total}},
\]

所有队列更新后非负，所有安全候选都通过全部硬约束。

### 10.3 集成测试

- stock vs passthrough；
- passthrough vs full-cap；
- fixed-cap 下 shadow vs real；
- Guard-only 端到端；
- DPP 端到端；
- predictor 缺失、OOD 和 NaN fallback；
- 空队列、单请求、长 prompt、混合批次；
- 高 KV 压力；
- 服务停止和异常退出时日志能够完整刷新。

### 10.4 性能回归

测量：

- snapshot 开销；
- 每候选 shadow plan 开销；
- predictor 开销；
- Guard 开销；
- DPP 打分开销；
- 总决策开销；
- stock 模式开销。

如果决策开销不可忽略，优先减少候选数、缓存特征和批量预测，不能通过删除安全检查掩盖问题。

---

## 11. 常见错误与禁止做法

Codex 必须主动检查以下错误：

1. 把 prefill cap 当成总 token budget；
2. 每轮修改全局 `max_num_batched_tokens`；
3. 使用 DPP 权重改变 FCFS 请求次序；
4. 先打分、后做资源检查；
5. 为每个候选真实调用 allocator 并污染状态；
6. 只预测 kernel 时间，却用它保护端到端 TBT；
7. 使用平均 TPOT 替代每个 token interval；
8. 用预测值更新真实物理/虚拟队列；
9. 默认认为 \(b=0\) 一定安全；
10. 只收集 stock Scheduler 访问过的状态训练 predictor；
11. 随机打散相邻 iteration 造成训练测试泄漏；
12. 忽略 predictor OOD 或 NaN；
13. 不记录候选拒绝原因；
14. 不比较 shadow plan 与真实执行；
15. 在 Pass-through 尚未等价时继续实现 DPP；
16. 只报告平均延迟，不报告 P95/P99 TBT；
17. 只选对 DPP 有利的负载和 seed；
18. 未冻结 vLLM commit 就修改内部 Scheduler；
19. 复制整份上游 Scheduler，导致难以维护；
20. 为修复无关问题扩大本次改动范围。

---

## 12. 参考依据

设计时可参考以下材料，但实现必须以锁定版本代码和实测结果为准：

1. James Pan, Guoliang Li, *A Survey of LLM Inference Systems*：用于理解 continuous batching、chunked prefill、token budget、KV Cache、preemption 和请求调度之间的关系。
2. *Drift-Plus-Penalty Based Queue Management for Edge LLM Inference with Repeated Sampling*：用于借鉴变长 frame 下的单位时间 DPP、队列积压与服务量建模。
3. vLLM 官方文档和锁定 commit 的 V1 Scheduler 源码：用于确认 `scheduler_cls`、`max_num_batched_tokens`、`max_num_seqs`、chunked prefill 和 KV allocator 的真实语义。

不得直接照搬 DPP 论文的“按输出长度分队列、一次选择一个队列和 batch size”模型。本文研究动作是混合连续批处理下的整体 prefill cap，借鉴的是：

\[
\frac{\text{队列压力带来的服务收益}-\text{SLO债务/风险}}{\text{预计执行时间}}
\]

这一基本思想。

---

## 13. 完成定义

只有同时满足以下条件，第一版 Prefill-Budget DPP Scheduler 才算完成：

- M0～M7 全部验收通过；
- stock、passthrough、fixed-cap、guard-only、dpp 五种模式均可运行；
- Pass-through 与 stock 等价性得到验证；
- Fixed-cap 从不突破 prefill cap；
- Predictor 在独立测试集完成校准；
- Shadow plan 与真实计划差异得到量化；
- TBT Guard 的安全收益得到验证；
- DPP 队列使用实际值闭环更新；
- benchmark 和消融完整；
- 所有配置、命令、原始结果和图表可复现；
- 文档明确记录已知限制和失败场景。

---

## 14. 进度记录

Codex 每完成一项后将 `[ ]` 改为 `[x]`，并在其下补充日期、commit、测试命令和结果摘要。

- [ ] M0：冻结环境与实验契约
- [ ] M1：默认 vLLM 基线
- [ ] M2：Pass-through Scheduler 与等价性验证
- [ ] M3：固定 Prefill Cap
- [ ] M4：Telemetry 与执行时间预测器
- [ ] M5：影子计划、资源检查与 TBT Guard
- [ ] M6：DPP 队列、打分与闭环更新
- [ ] M7：完整 benchmark 与消融

### 当前已冻结的决定

- 研究对象：单 GPU 混合连续批处理；
- 基础框架：vLLM V1；
- 第一版只控制 prefill cap；
- 保留 FCFS 和默认 decode 服务；
- 第一版只有一个请求类别；
- 第一版关闭 prefix cache、speculative decoding、async scheduling 和多模态；
- 第一版禁止会触发 preemption 的候选；
- 先实现 stock baseline、passthrough 和 fixed-cap，再实现 predictor、Guard 和 DPP；
- DPP 使用单位时间得分；
- 预测用于选择，实际观测用于队列更新。

### 待 M0 确认的参数

- [ ] vLLM release/tag 和 commit
- [ ] 模型及 revision
- [ ] dtype
- [ ] `max_model_len`
- [ ] `max_num_batched_tokens`
- [ ] `max_num_seqs`
- [ ] `gpu_memory_utilization`
- [ ] 候选动作集合 \(\mathcal B_0\)
- [ ] TTFT/TBT SLO
- [ ] 目标 TTFT attainment \(\rho^F\)
- [ ] 允许 TBT violation ratio \(\epsilon^D\)
- [ ] Predictor 类型与特征 schema
- [ ] DPP 归一化常数和初始权重

---

## 15. 给 Codex 的启动指令

将本文件放到目标仓库后，对 Codex 使用以下指令：

> 完整阅读 `DPP_SCHEDULER_CODEX_EXECUTION_PLAN.md` 和仓库中的 `AGENTS.md`。检查当前仓库、Git 状态和已有实现，确认当前最早尚未完成的里程碑。先给出该里程碑的最小实施计划，然后完成实现、测试和文档记录。除非该里程碑验收通过，否则不要提前进入后续阶段。保留用户已有修改，不做与当前里程碑无关的重构。

