# AGENTS.md - vLLM mixed-batching DPP scheduler research

## 1. 适用范围与目标

本文件适用于整个仓库。Codex 在任何子任务开始前都必须遵守这里的研究约束；若某个子目录存在更具体的 `AGENTS.md` 或 `AGENTS.override.md`，则以更靠近该目录的指令为准。

项目目标是在单张 RTX 5070 12 GB GPU 上，以 vLLM V1 的 mixed continuous batching 为基础，完成以下闭环：

1. 对默认 Scheduler 做可复现 benchmark；
2. 实测 fixed token budget 的 TTFT-TBT-goodput 权衡及 prefill-decode interference；
3. 设计 variable-frame DPP scheduler，动态选择每轮 prefill token budget；
4. 在不改变默认 decode 顺序、请求顺序和 KV Cache 管理语义的前提下实现调度器；
5. 通过等价性、主实验、消融和开销实验验证效果。

第一版只研究单 GPU mixed batching。不要擅自把范围扩展到 PD 分离、多 GPU、模型并行、speculative decoding、prefix caching、LoRA、量化或输出质量优化。

## 2. 指令关键词

- **必须**：若不满足，不得声称任务完成。
- **应当**：默认执行；若不执行，需要说明理由。
- **可以**：仅在有明确收益且不破坏公平性时使用。

研究计划中的数值不是论文结论。若实测结果否定假设，应保留并报告负结果，不得为了证明 DPP 有效而修改 workload、SLO、基线或过滤规则。

## 3. 信息与配置的优先级

出现冲突时按以下顺序处理：

1. 用户在当前任务中的明确要求；
2. 已冻结的实验配置和 trace manifest；
3. `docs/dpp_spec.md` 中已经确认的数学定义；
4. `docs/experiment_plan.md` 和 `docs/decisions.md`；
5. 本文件中的默认建议；
6. 论文、博客或上游文档中的通用设置。

所有会影响实验结论的参数必须写入版本控制中的配置文件，不能只存在于聊天、shell 历史或命令行中。建议以 `configs/frozen_experiment.yaml` 作为单一事实来源。若该文件尚未创建或尚未标记为 frozen，Codex 可以起草，但在用户确认前不得把其中数值称为“已冻结配置”。

## 4. 每个任务开始时的工作方式

Codex 必须先：

1. 查看适用的指令文件、`git status --short`、相关配置和现有实现；
2. 判断当前工作属于 G0-G7 的哪个阶段；
3. 说明本次任务的输入、假设、将修改的文件和验收方式；
4. 优先复用已有脚本、trace 和结果解析器，避免生成平行实现；
5. 对长时间 benchmark 先做 `--dry-run` 或输出实验组合数、预计请求数和结果目录；
6. 在没有真实运行结果时明确写“未运行”，不得构造示例数字冒充结果。

保持修改小而可审查。不要顺手重构无关代码，不要覆盖用户的未提交修改，不要自行提交、推送、开 PR、升级 vLLM 或安装新的生产依赖，除非用户明确要求。

## 5. 推荐的仓库结构

优先沿用仓库已有结构；若需要新建，使用以下职责划分：

```text
configs/                 冻结配置、实验矩阵、SLO 和 predictor 配置
dpp_vllm/                自定义 Scheduler、决策器、状态与 vLLM 适配层
benchmarks/              workload/trace 生成、运行器、结果解析器
scripts/                 薄命令入口，不放核心算法
tests/unit/              纯函数、队列、债务、Guard、预测器测试
tests/integration/       vLLM Scheduler 与端到端请求测试
docs/                    计划、DPP 数学定义、决策记录、实验日志
results/raw/             原始 benchmark 输出，只追加不覆盖，通常不入 Git
results/processed/       可从 raw 重建的聚合表
artifacts/               最终图表和论文用表格
traces/                  固定请求、到达时间、长度与 SHA256 manifest
```

不要把大型模型权重、数据集、原始 trace 或 benchmark 结果提交进 Git。对这些内容保存来源、版本、过滤规则、SHA256 和生成命令。

## 6. 初始实验契约

除非冻结配置明确覆盖，第一版按以下目标环境工作：

| 项目                               | 目标值                               |
| ---------------------------------- | ------------------------------------ |
| GPU                                | RTX 5070 12 GB                       |
| vLLM                               | `0.26.1rc1.dev535+g83ad767ee.precompiled`，commit `83ad767eed3be3ee7f2df63be693bfaca5c7c922`，precompiled editable 安装 |
| 模型                               | `Qwen/Qwen2.5-3B-Instruct`           |
| dtype                              | BF16                                 |
| `max_model_len`                    | 8192                                 |
| `gpu_memory_utilization`           | 0.90                                 |
| `max_num_seqs`                     | 64                                   |
| policy                             | FCFS                                 |
| chunked prefill                    | 开启                                 |
| prefix caching                     | 关闭                                 |
| speculative decoding / LoRA / 量化 | 关闭                                 |
| generation                         | `temperature=0`、`ignore_eos=True`   |

当前 WSL2 环境不提供实验性 V2 Model Runner 所需的 CUDA UVA，因此 G0-G3
必须显式记录并使用 `VLLM_USE_V2_MODEL_RUNNER=0`。该开关只选择 V1 Model
Runner；vLLM V1 engine 与默认 Scheduler 保持不变。所有比较策略必须使用相同
开关。上述版本取代此前的 `0.26.0` 目标值，除非后续冻结配置再次明确覆盖。

实验开始前必须记录 GPU、驱动、CUDA、PyTorch、vLLM、Python、操作系统/WSL、CPU、内存、模型 revision、KV Cache 容量和启动日志中的最终 `SchedulerConfig`。不要仅依赖预期默认值。

所有时间在内部统一使用一个明确单位，推荐毫秒；到达率统一为 requests/s；token 数必须为整数。结果文件必须写明单位。

## 7. 阶段门槛

| 阶段 | 工作                                     | 退出条件                                        |
| ---- | ---------------------------------------- | ----------------------------------------------- |
| G0   | 冻结环境、配置、数据和运行元信息         | 能从 manifest 重建环境与命令                    |
| G1   | Stock Scheduler 正确性、低负载和饱和基线 | 获得基础 TTFT/TPOT/ITL 与饱和吞吐               |
| G2   | 开环 QPS 扫描与 SLO 标定                 | SLO 在查看 DPP 结果前冻结；得到容量 knee        |
| G3   | 默认 Scheduler fixed budget 扫描         | 得到 budget-延迟-goodput 权衡和 Best-Fixed      |
| G4   | mixed iteration profiling                | 得到动作集合、干扰曲面和 predictor 数据         |
| G5   | 冻结 DPP 数学定义与实现接口              | `docs/dpp_spec.md` 的符号、单位、更新时刻无歧义 |
| G6   | pass-through 与 DPP 实现                 | 关闭 DPP 时与 stock 等价，单元/集成测试通过     |
| G7   | 正式主实验、消融和开销分析               | 使用冻结 test trace，结果可从 raw 重建          |

不得因为后续阶段已经可以编码而跳过前面的科学门槛。若用户要求提前搭脚手架，可以实现接口和测试，但必须把未由实测支持的参数标记为 provisional。

## 8. 默认 Scheduler benchmark 规则

### 8.1 必测基线

- `Stock-Auto`：不显式传 `max_num_batched_tokens`，从启动日志记录解析值；
- `Stock-B8192`：显式总 token budget 8192；
- `Fixed-Budget`：至少扫描 `{256, 512, 1024, 2048, 4096, 8192}`；
- `Best-Fixed`：只在 validation trace 上选出并冻结，再用于 test trace；
- 后续公平比较需要 `Constant-Prefill-Cap`，因为 stock 的总 token budget 与 DPP 的纯 prefill cap 并不完全等价。

不要只拿 DPP 与 Stock-B8192 比较。不同策略必须使用相同的绝对 QPS、请求内容、顺序、到达时间、随机种子和非调度配置；不能按各自容量归一化后伪装成同负载比较。

### 8.2 Workload

先运行长度固定的可控 random workload，再加入长度扰动：

| 名称          | 输入/输出 token | 目的                             |
| ------------- | --------------: | -------------------------------- |
| decode-heavy  |      128 / 1024 | 测长 decode 下的 TBT 保护        |
| balanced      |       512 / 512 | 测一般 mixed batching            |
| prefill-heavy |      1024 / 128 | 测 TTFT 与 prefill 吞吐          |
| long-prefill  |      2048 / 128 | 放大 prefill-decode interference |

还必须包含：

- 三类请求各三分之一的 heterogeneous mixed trace；
- 每 60-120 秒在 decode-heavy、prefill-heavy、balanced 间切换的 phase-shift trace；
- ShareGPT 作为真实文本/长度分布；
- BurstGPT 连续时间窗口转换的 timed trace 作为真实突发性证据。

ShareGPT/BurstGPT 的过滤和转换只能执行一次并生成 manifest。不同策略不得重新随机采样。BurstGPT 若验证突发性，必须保留连续窗口的相对时间戳；随机抽行只能用于长度分布实验，不能称为真实到达 trace。

### 8.3 到达与容量

- `request-rate=inf, max-concurrency=1` 只用于串行基础延迟；
- `request-rate=inf` 可以测饱和吞吐，但不能支持在线 SLO 结论；
- 正式容量测试必须使用有限 request rate 的 open-loop 到达；
- 默认测 Poisson，并测试中等和强突发 Gamma 到达；
- 不要用 `max-concurrency` 掩盖服务器过载，除非实验目的就是 closed-loop/concurrency sensitivity。

定义联合 SLO attainment：

$$
A_{joint}=\frac{1}{N}\sum_i \mathbf{1}[TTFT_i\le S^{TTFT}\land TPOT_i\le S^{TPOT}].
$$

定义参考容量：

$$
\lambda_{cap}=\max\{\lambda:A_{joint}(\lambda)\ge 90\%\}.
$$

正式比较优先测试相同绝对到达率对应的 `{0.5, 0.7, 0.9, 1.0, 1.1} * lambda_cap`。若冻结配置另行选择 Stock Scheduler 中 `A_joint < 90%` 且最接近 90% 的实测点作为压力比较 QPS，必须将其命名为 `comparison_qps`，不得覆盖或重定义 `lambda_cap`；所有调度器必须复用该绝对 QPS。容量点和压力比较点都必须检查 achieved rate、失败请求、OOM、积压是否清空和实验时长，不能只看 percentile。

### 8.4 SLO 冻结

SLO 只能根据 Stock Scheduler 的串行/低负载数据和 validation trace 标定，在查看 DPP test 结果前冻结。初始候选为：

| 档位   |                          TTFT |                            TPOT |
| ------ | ----------------------------: | ------------------------------: |
| Tight  | `2 * stock_low_load_P90_TTFT` | `1.5 * stock_low_load_P90_TPOT` |
| Medium | `4 * stock_low_load_P90_TTFT` |   `2 * stock_low_load_P90_TPOT` |
| Loose  | `8 * stock_low_load_P90_TTFT` |   `3 * stock_low_load_P90_TPOT` |

Medium 作为主结果，Tight/Loose 用于敏感性分析。若需要调整，应在 DPP test 运行前写入决策记录，说明调整依据；不得根据 DPP 的表现反向选择阈值。

### 8.5 指标与统计

主指标是联合 SLO goodput：

$$
G=\frac{\#\{i:TTFT_i\le S^{TTFT}\land TPOT_i\le S^{TPOT}\}}{T_{experiment}}.
$$

同时报告：

- TTFT、TPOT、ITL/TBT、E2E 的 P50/P90/P95/P99；
- 每请求 max-TBT 的分布；
- TTFT、TPOT 和联合 attainment；
- requests/s、prompt/output/total tokens/s、completed/failed requests；
- running/waiting requests、KV Cache usage、preemption/recomputation；
- 每轮 decode/prefill token、budget、iteration latency 和 scheduler CPU time。

开发阶段每组可用 300 请求、3 seeds；正式主结果每组至少 1000 请求、3 seeds，关键结论优先 5 seeds。每个 seed 先独立计算 percentile 和 goodput，再对 seed 级统计量计算均值与 95% CI。不得把所有 seed 的请求拼接后计算一个 P99。

### 8.6 Performance run 与 profile run 分离

- `performance` 模式关闭逐 iteration 详细日志和高频 instrumentation，只保留低频系统指标及请求级 detailed result；
- `profile` 模式使用同一 trace 重跑，采集 batch 组成、scheduler 时间、精确 iteration 时间和 predictor 特征；
- profile 结果不能冒充无 instrumentation 的最终性能结果；
- 若关闭 async scheduling、CUDA Graph 或其他优化以获得边界，所有对比策略必须使用相同设置并在限制中说明。

## 9. DPP 第一版设计契约

### 9.1 动作语义

DPP 动作必须定义为：

$$
b_t^P=\text{本轮允许调度的 prefill token 数}.
$$

它不是 `max_num_batched_tokens` 的同义词。总 budget 仍需容纳本轮 decode token。候选集合由冻结配置给出，并包含 `b=0` 的安全退让动作。

默认 Scheduler 仍负责 request order、decode 优先级、KV allocation、preemption、完成和释放。DPP 第一版只改变 prefill budget 决策，不重新实现整套调度循环。

### 9.2 状态与决策时刻

每轮调度前收集至少：

- prefill backlog `Q^P`，按已确认类别分组；
- TTFT virtual debt `Z^TTFT`；
- TBT virtual debt `Z^TBT`；
- running decode 数、上下文长度统计、waiting prefill 数和剩余 prompt token；
- KV Cache 可用量、`max_num_seqs` 余量、preemption 风险；
- 上一轮/滑动窗口的 iteration latency 与 predictor 特征。

队列、债务、服务量、违约量的单位和更新时间必须在 `docs/dpp_spec.md` 明确。不要在代码中临时发明债务更新公式。每轮只能在已定义的边界更新一次，并通过纯函数单元测试验证。

### 9.3 可行性与 TBT Guard

每个候选先通过：

1. latency predictor 的 P95 或保守上界检查；
2. predictor uncertainty/underprediction 风险检查；
3. vLLM KV allocator 可行性；
4. `max_num_seqs`；
5. 总 token budget；
6. full-ISL reservation 或等价机制。

若所有 `b>0` 都不安全，必须选择 `b=0`。Guard 不允许通过事后删除 TBT 违约请求来制造安全性。

### 9.4 Variable-frame DPP-ratio

第一版候选评分为：

$$
\Phi_t(b)=
\frac{
\sum_c Q^P_{c,t}\hat\mu^P_{c,t}(b)
+\eta_F\sum_c Z^{TTFT}_{c,t}\hat g^{TTFT}_{c,t}(b)
-\eta_D\sum_c Z^{TBT}_{c,t}\hat m^{TBT}_{c,t}(b)
+V\hat U_t(b)-\gamma\hat F_t(b)
}{\hat\tau_{50}(S_t,b)}.
$$

最终在安全候选集合上取 `argmax`。必须定义每个符号、符号方向、量纲、归一化、tie-break 和数值范围。分母不得为零；predictor 缺失或输入超出支持域时必须采用保守 fallback，而不是外推乐观时延。

这一 ratio 来自 variable frame 下按单位时间优化的思想，但当前公式是待实验验证的系统设计，不得宣称已经自动继承论文的理论最优性或稳定性证明。

### 9.5 Predictor

第一版优先使用可解释的 mixed-iteration lookup table/interpolation，而不是先引入复杂神经网络。数据集必须由 G4 的独立 profiling trace 构建；validation 用于选择特征、bin 和 safety margin；test 只评估一次。

报告 P50/P95 相对误差、P95 underprediction rate、coverage、候选 regret 和超出支持域比例。若 underprediction 超过冻结门槛，Guard 必须扩大 margin 或退回更小 budget。

## 10. 实现约束

1. 优先通过上游支持的 `--scheduler-cls` 加载 `dpp_vllm.scheduler.DPPBudgetScheduler`；在运行前根据锁定的 vLLM 版本核实类路径和接口。
2. 继承或包装默认 Scheduler，采用最小侵入修改。只有上游扩展点确实不足时才维护 fork，并记录原因和最小 patch。
3. 把版本相关的私有字段访问集中在一个 adapter 中；不要让核心 DPP 逻辑散布 vLLM 内部 API。
4. 把状态快照、候选生成、Guard、评分、tie-break、债务更新写成可独立测试的纯函数。
5. 决策必须确定性；相同状态、配置和 seed 必须选择相同动作。
6. 所有实验开关必须显式写入配置和结果 metadata。禁止依靠隐藏全局变量改变算法。
7. instrumentation 默认关闭，并测量开启/关闭开销；不要在热路径逐轮同步写大日志。
8. 不捕获宽泛异常后静默回退。fallback 必须计数、记录原因并可在结果中审计。
9. 不改变输出 token 数、停止条件或请求过滤方式来改善延迟。
10. 不修改 stock baseline 的代码；需要埋点时优先旁路采集，并用 pass-through 验证其语义和开销。

## 11. Pass-through 验收

`dpp_enabled=false` 时，自定义 Scheduler 必须直接复现默认逻辑。至少验证：

- 相同输入下的调度请求、顺序和 token 数一致；
- 无请求丢失、重复、死锁、状态错误或 KV 泄漏；
- 请求完成数和生成长度一致；
- 吞吐、TTFT、TPOT/ITL 差异通常不超过 2%-3%，若超过必须定位原因；
- instrumentation 关闭后无明显额外开销；
- 相关单元与集成测试通过。

在 pass-through 未通过前，不得把 DPP 与 stock 的性能差异解释为算法收益。

## 12. 测试要求

修改 DPP 代码时至少覆盖：

- 空队列、仅 decode、仅 prefill、mixed batch；
- 候选 `b=0`、最大 budget、budget 不足和 token/KV 不可行；
- predictor 缺失、NaN、零分母、超支持域和保守 fallback；
- TTFT/TBT debt 的边界、非负性、单位和单次更新；
- Guard 全部拒绝与部分拒绝；
- 相同分数下的稳定 tie-break；
- pass-through 与 stock 的决策一致性；
- 所有请求最终完成且资源释放。

优先运行最小相关测试，再运行完整测试。使用仓库已经配置的测试、lint 和 type-check 命令；若没有配置，不要假装命令存在。新增依赖前说明必要性并征得用户同意。

## 13. 结果与可复现性

每次 run 使用唯一 `run_id`，原始结果只能追加，不能覆盖。每个 run 至少保存：

- 完整服务器与 client 命令；
- 解析后的配置、Git commit、dirty 状态和环境快照；
- dataset/trace 名称、SHA256、seed、到达过程和目标/实际 QPS；
- warmup 和 measurement 区间；
- 成功、失败、超时、取消和输出长度不符的请求数；
- 原始请求级结果及聚合脚本版本。

聚合表和图必须能从 `results/raw/` 一条命令重建。图表要包含单位、误差条/95% CI、样本数和策略全名。若剔除异常 run，必须保留原始文件并在决策记录中给出事先定义的客观理由。

实验配置的执行顺序应随机化或轮换，以减小 GPU 温度、功耗和频率漂移。正式实验前做 warmup，并监测热节流、OOM、preemption 和客户端发送不足。

## 14. 公平比较与消融

正式主表至少包含：

- Stock-Auto；
- Stock-B8192；
- validation 选出的 Best-Fixed；
- Constant-Prefill-Cap；
- DPP-Budget。

消融至少包含：

- `DPP-no-ratio`：不除以 iteration time；
- `DPP-no-debt`：保留 predictor/Guard，但去掉虚拟债务；
- `DPP-oracle-predictor`：使用真实或离线 oracle 时延，评估 predictor 损失。

可以增加只使用 Guard 的版本，但不能用消融代替 Best-Fixed。所有超参数只能在 training/validation workload 上选择，test trace 不参与调参。

## 15. Go/No-Go 与完成定义

进入正式 DPP 主实验前，目标检查项为：

- stock 跨 seed 波动约在 3%-5% 内，或已解释主要噪声；
- 出现清晰 SLO capacity knee；
- 不同 workload/负载下最佳 fixed budget 确实移动；
- 大 budget 改善 TTFT 的同时对高分位 ITL/TBT 产生可测代价；
- predictor 的 P95 underprediction rate 达到冻结的安全门槛；
- pass-through 性能差异不超过约 3%；
- DPP scheduler P99 CPU 开销目标低于 1 ms，平均低于 iteration time 的 3%。

最终论文目标可以设为临近容量时相对 Best-Fixed goodput 提升至少 10%、低负载吞吐下降不超过 5%，但这些是 Go/No-Go 目标，不是可以写死或保证得到的结果。

若同一个 fixed budget 在所有冻结场景中稳定占优，应报告“当前硬件、模型和 workload 下动态预算优化空间不足”，并先复查动作语义、负载多样性和测量噪声，而不是修改 test 数据迎合结论。

## 16. Codex 的交付格式

每次完成任务时，简洁报告：

1. 得到的结果或实现内容；
2. 修改的文件；
3. 实际运行的检查/benchmark 及结果；
4. 未运行内容和原因；
5. 当前处于 G0-G7 的哪个阶段；
6. 下一道尚未满足的 gate。

对 benchmark 分析必须区分“观测事实”“由数据支持的推断”“尚待验证的假设”。对代码修改必须给出可复制的验证命令。不要仅汇报平均值，不要隐去失败 run，不要在没有 profile 证据时猜测 GPU kernel 或 Scheduler 行为。

## 17. 常见请求的默认解释

- “benchmark 默认 Scheduler”：只推进 G0-G4；除非用户要求，不改 DPP 决策逻辑。
- “实现 DPP Scheduler”：推进 G5-G6；先读取冻结 spec，先做 pass-through，再启用 DPP。
- “比较 DPP 和 vLLM”：使用冻结 test trace、相同绝对 QPS 和完整基线；不得重新选择 SLO/Best-Fixed。
- “优化结果”：先定位瓶颈和实验有效性；不得通过减少输出长度、改变请求集合或放宽 SLO 获得表面提升。
- “生成论文图表”：只能读取可追溯 raw/processed 数据，图上标明单位、CI、seed 数和 workload。

## 18. 研究依据的使用边界

若仓库中存在以下资料，在涉及相应概念时优先核对原文：

- `project_sources/01-Pan-Li-2025-A-Survey-of-LLM-Inference-Systems.pdf`：用于推理系统术语、系统栈和 goodput/SLO 背景；
- `project_sources/02-Drift-Plus-Penalty_Based_Queue_Management_for_Edge_LLM_Inference_with_Repeated_Sampling-1-.pdf`：用于 variable-frame DPP、队列稳定性和单位时间目标的理论来源；
- 锁定版本的 vLLM 官方文档与源码：用于 Scheduler API、CLI 参数和真实执行语义。

原论文的 repeated sampling、多队列静态 batch 模型与本项目的 vLLM mixed continuous batching 不相同。可以迁移 DPP 思想，但不得直接宣称原定理已经证明本调度器稳定或最优。凡涉及 vLLM 版本相关接口，先检查当前锁定源码或官方版本文档，不从记忆猜测。
