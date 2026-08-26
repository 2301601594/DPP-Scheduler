# Time → Prefill Budget 可行性验证

本报告只使用实验工具重建 BatchPlan 并调用冻结 Predictor；没有修改线上 Candidate Generator、DPP Selector 或 Predictor。

## 结论摘要

- Predictor artifact 数：2；每个 artifact 分别覆盖 Prefill-only 与三个 Mixed Decode 子模型。Decode-only 只作为 Mixed Snapshot 的 `b=0` 锚点。
- Snapshot-artifact 组合数：160。每个 Snapshot 固定全部 Decode 请求和 Prefill 顺序，只改变 Prefill budget。
- 子模型内部 budget → time 是否稳定：是。跨 `b=0` 的 Decode-only → Mixed 边界违反单独列出，不与子模型内部违反混合。
- time → budget 使用离散扫描，未实现二分搜索；各模型覆盖率和 250 ms budget 见下表。
- 真实运行：236 行；预测 MAE 33.0 ms；反求 250 ms case 的真实时间中位数 209.7 ms。
- 真实 budget → time：10/196 次反转；最大反向差 42.167 ms；250 ms case 的真实绝对目标误差中位数 40.3 ms。
- **最终结论：PASS**

## 分模型结果

| Predictor | Snapshot 类别 | Snapshot 数 | 子模型内部违反/比较 | 跨模型边界违反 | 最大反向 ms | 可反求 target | budget 反向下降 | 250 ms budgets | 反求点 in-support |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 20 | 0/81 | 6 | 29.565 | 137/140 | 0 | 0,64,128,256,384 | 100.0% |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 20 | 0/99 | 12 | 7.934 | 140/140 | 0 | 256,384,385,420,512 | 95.0% |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 20 | 0/92 | 10 | 21.553 | 140/140 | 0 | 256,384,426,512 | 100.0% |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 20 | 0/148 | 0 | 0.000 | 140/140 | 0 | 384,512 | 100.0% |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 20 | 0/81 | 6 | 32.937 | 137/140 | 0 | 0,64,128,256,384 | 100.0% |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 20 | 0/99 | 19 | 2.751 | 140/140 | 0 | 256,385,420,512,517 | 95.0% |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 20 | 0/92 | 13 | 17.489 | 140/140 | 0 | 256,384,426,512 | 100.0% |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 20 | 0/148 | 0 | 0.000 | 140/140 | 0 | 384,512 | 100.0% |

## Budget → predicted duration 曲线（分模型汇总）

| Predictor | Snapshot 类别 | actual Prefill tokens | Snapshot 数 | 中位预测 ms | 最小 ms | 最大 ms |
|---|---|---:|---:|---:|---:|---:|
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 0 | 20 | 156.6 | 141.8 | 203.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 64 | 20 | 174.3 | 112.3 | 447.2 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 128 | 20 | 196.1 | 148.6 | 465.3 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 256 | 20 | 240.8 | 189.9 | 502.2 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 319 | 1 | 495.9 | 495.9 | 495.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 384 | 11 | 286.4 | 244.9 | 539.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 407 | 1 | 269.9 | 269.9 | 269.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 512 | 10 | 346.9 | 278.3 | 578.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 521 | 1 | 520.9 | 520.9 | 520.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 655 | 1 | 343.6 | 343.6 | 343.6 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 768 | 8 | 428.8 | 377.7 | 658.0 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_17_64 | 1024 | 8 | 513.9 | 447.7 | 740.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 0 | 20 | 129.4 | 126.7 | 131.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 64 | 20 | 125.3 | 121.9 | 128.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 128 | 20 | 140.7 | 140.3 | 149.8 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 256 | 20 | 173.1 | 165.7 | 198.2 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 384 | 16 | 204.9 | 192.8 | 225.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 385 | 1 | 221.8 | 221.8 | 221.8 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 420 | 1 | 237.9 | 237.9 | 237.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 512 | 14 | 234.7 | 219.7 | 257.4 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 517 | 1 | 259.0 | 259.0 | 259.0 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 768 | 13 | 302.0 | 270.7 | 335.0 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 900 | 1 | 322.1 | 322.1 | 322.1 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 920 | 1 | 327.7 | 327.7 | 327.7 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_1_4 | 1024 | 11 | 364.2 | 321.8 | 426.8 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 0 | 20 | 135.6 | 129.8 | 169.1 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 64 | 20 | 130.9 | 109.6 | 205.6 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 128 | 20 | 154.0 | 127.9 | 219.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 256 | 20 | 194.1 | 160.0 | 249.6 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 384 | 14 | 218.0 | 193.8 | 281.1 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 426 | 1 | 220.0 | 220.0 | 220.0 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 512 | 13 | 251.0 | 229.3 | 314.2 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 620 | 1 | 308.1 | 308.1 | 308.1 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 697 | 1 | 331.8 | 331.8 | 331.8 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 768 | 11 | 326.7 | 305.3 | 385.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | mixed_decode_5_16 | 1024 | 11 | 406.5 | 370.8 | 463.6 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 64 | 20 | 124.9 | 122.4 | 124.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 128 | 20 | 137.0 | 136.7 | 137.0 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 256 | 20 | 164.5 | 162.8 | 175.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 384 | 20 | 195.9 | 190.9 | 214.8 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 512 | 20 | 228.9 | 221.4 | 253.7 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 768 | 20 | 301.0 | 289.5 | 331.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 1024 | 20 | 379.9 | 365.9 | 409.3 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 1536 | 14 | 551.5 | 528.8 | 581.7 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 1790 | 1 | 654.7 | 654.7 | 654.7 |
| qwen3-14b-ridge-mixed-decode-three-segment-cross-online-v3 | prefill_only | 2048 | 13 | 742.6 | 690.0 | 807.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 0 | 20 | 156.6 | 141.8 | 203.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 64 | 20 | 172.7 | 108.9 | 446.4 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 128 | 20 | 195.0 | 145.8 | 464.6 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 256 | 20 | 240.1 | 187.5 | 501.6 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 319 | 1 | 495.2 | 495.2 | 495.2 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 384 | 11 | 285.6 | 243.3 | 539.4 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 407 | 1 | 268.6 | 268.6 | 268.6 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 512 | 10 | 346.0 | 276.8 | 578.0 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 521 | 1 | 520.4 | 520.4 | 520.4 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 655 | 1 | 342.8 | 342.8 | 342.8 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 768 | 8 | 427.9 | 376.9 | 657.7 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_17_64 | 1024 | 8 | 513.0 | 447.7 | 740.7 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 0 | 20 | 129.4 | 126.7 | 131.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 64 | 20 | 128.3 | 124.0 | 131.8 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 128 | 20 | 142.1 | 136.7 | 152.1 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 256 | 20 | 170.8 | 163.4 | 199.1 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 384 | 16 | 201.3 | 192.2 | 222.8 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 385 | 1 | 220.9 | 220.9 | 220.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 420 | 1 | 234.3 | 234.3 | 234.3 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 512 | 14 | 230.9 | 222.4 | 243.2 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 517 | 1 | 244.6 | 244.6 | 244.6 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 768 | 13 | 291.2 | 276.3 | 318.3 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 900 | 1 | 329.4 | 329.4 | 329.4 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 920 | 1 | 335.4 | 335.4 | 335.4 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_1_4 | 1024 | 11 | 367.5 | 333.6 | 404.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 0 | 20 | 135.6 | 129.8 | 169.1 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 64 | 20 | 129.8 | 113.7 | 204.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 128 | 20 | 153.1 | 130.7 | 219.1 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 256 | 20 | 195.0 | 160.7 | 249.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 384 | 14 | 216.5 | 192.4 | 281.7 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 426 | 1 | 218.9 | 218.9 | 218.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 512 | 13 | 248.2 | 225.9 | 315.6 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 620 | 1 | 307.2 | 307.2 | 307.2 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 697 | 1 | 330.8 | 330.8 | 330.8 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 768 | 11 | 321.1 | 297.9 | 388.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | mixed_decode_5_16 | 1024 | 11 | 403.7 | 365.7 | 468.2 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 64 | 20 | 124.9 | 122.4 | 124.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 128 | 20 | 137.0 | 136.7 | 137.0 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 256 | 20 | 164.5 | 162.8 | 175.9 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 384 | 20 | 195.9 | 190.9 | 214.8 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 512 | 20 | 228.9 | 221.4 | 253.7 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 768 | 20 | 301.0 | 289.5 | 331.5 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 1024 | 20 | 379.9 | 365.9 | 409.3 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 1536 | 14 | 551.5 | 528.8 | 581.7 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 1790 | 1 | 654.7 | 654.7 | 654.7 |
| qwen3-14b-ridge-mixed-decode-three-segment-online-v2 | prefill_only | 2048 | 13 | 742.6 | 690.0 | 807.5 |

这里的 `MAX` 是从原始 exact target-profile 行可重建的 Prefill work 上限；超过该上限的点忽略。`predicted_duration` 使用 expected duration。Prefill-only 没有可执行的空 BatchPlan，因此不生成 `b=0`；Mixed 的 `b=0` 由 Decode-only 模型预测，正 budget 由对应 Decode 分段 Mixed 模型预测。

## 回答

1. budget → time 是否基本单调：各子模型内部均单调；`b=0` 的跨模型边界单独报告。
2. time → budget 是否可稳定反求：是；离散求 `max{b: tau_hat(b) <= T}`，不假设跨模型全局单调。
3. 250 ms 附近可得到的 budget：按 Predictor 和 Decode 分段列于分模型结果表。
4. 真实运行是否支持 Predictor：实际 MAE 为 33.0 ms，真实单调反转 10/196，250 ms case 中位数为 209.7 ms。
5. 最终结论：**PASS**。只有填入真实 GPU 验证结果后才会给出 PASS / PARTIAL PASS / FAIL。

## 方法与限制

代表性状态从现有 Qwen3-14B DGX exact targeted-profile 原始行按 Prefill-only、Mixed Decode 1–4、5–16、17–64 分层抽取。重建保留全部 Decode 请求及其 pre-iteration KV context、Prefill 顺序和已 Prefill context。active-config segmented v2 与 development cross-feature v3 分开报告；v3 不被描述为线上已采用。历史在线残差窗口不可重放，因此 sweep 使用各 artifact 的同 batch-kind OOF cold-start 校准。
