# Time → Prefill Budget 可行性验证

本报告只使用实验工具重建 BatchPlan 并调用冻结 Predictor；没有修改线上 Candidate Generator、DPP Selector 或 Predictor。

## 结论摘要

- Snapshot 数：40（全部为有 Prefill work 且 Decode 集合固定的 Mixed 状态）。
- budget → time 相邻点比较：217 次；单调性违反 23 次（10.6%）；最大反向时间差 87.388 ms。
- time → budget：278/280 个 target 找到离散可行 budget；target 增大时 budget 反向下降 0 次。
- 250 ms 反求 budget：0, 128, 256, 384, 426, 512, 768；其中低于当前同 Snapshot P25 的比例为 12.5%。
- 真实运行：0 行；预测 MAE n/a ms；反求 250 ms case 的真实时间中位数 n/a ms。
- **最终结论：PENDING_REAL_GPU_VALIDATION**

## Budget → predicted duration 曲线（跨 Snapshot 汇总）

| actual Prefill tokens | Snapshot 数 | 中位预测 ms | 最小 ms | 最大 ms |
|---:|---:|---:|---:|---:|
| 0 | 40 | 140.7 | 126.7 | 203.9 |
| 64 | 40 | 124.0 | 39.3 | 451.4 |
| 128 | 40 | 142.6 | 56.4 | 468.8 |
| 256 | 40 | 190.2 | 91.3 | 504.3 |
| 384 | 26 | 196.3 | 127.1 | 540.8 |
| 407 | 1 | 286.1 | 286.1 | 286.1 |
| 426 | 1 | 180.6 | 180.6 | 180.6 |
| 512 | 24 | 241.7 | 164.0 | 578.3 |
| 521 | 1 | 523.4 | 523.4 | 523.4 |
| 655 | 1 | 356.0 | 356.0 | 356.0 |
| 697 | 1 | 369.3 | 369.3 | 369.3 |
| 768 | 21 | 298.0 | 240.7 | 656.3 |
| 1024 | 21 | 369.9 | 321.4 | 738.3 |

这里的 `MAX` 是从原始 exact target-profile 行可重建的 Prefill work 上限；超过该上限的测试点按要求忽略。`predicted_duration` 使用 expected duration，而不是 conservative duration。0-budget 行为 Decode-only，其余行为 Mixed。

## 回答

1. budget → time 是否基本单调：否；见上述违反率和最大反向差。
2. time → budget 是否可稳定反求：是；采用离散集合直接求 `max{b: tau_hat(b) <= T}`，未实现二分搜索。
3. 250 ms 附近可得到的 budget：0, 128, 256, 384, 426, 512, 768 token。
4. 真实运行是否支持 Predictor：尚待小规模 GPU 验证。
5. 最终结论：**PENDING_REAL_GPU_VALIDATION**。只有填入真实 GPU 验证结果后才会给出 PASS / PARTIAL PASS / FAIL。

## 方法与限制

代表性状态从现有 Qwen3-14B DGX exact targeted-profile 原始行分层抽取。重建时保留每行全部 Decode 请求、各自 pre-iteration KV context、Prefill 请求顺序与 prefilled context；只沿固定顺序改变 Prefill budget。运行时在线残差窗口无法从历史 Snapshot 重放，因此离线 sweep 使用当前冻结 artifact 的同 batch-kind OOF cold-start 校准。这是可复现的 current-Predictor cold-start 行为，不代表任一历史在线窗口的瞬时状态。
