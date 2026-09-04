# FlowKV 式动态 PD 混合调度仿真

基于 NVIDIA Dynamo 的离散事件仿真引擎 DynaSim，对比静态 PD（disagg）与动态 PD（aggregated）两种调度架构，并给条件分离（conditional disagg）策略补了离线回放支持。目标是量化动态 PD 相对静态 PD 的收益和代价，作为后续落地改造的基线。

## 结论

- 动态 PD 的 TTFT 始终最优，比静态 1:3 低约 3 倍，尾延迟也最好。
- 吞吐上静态 1:1 最高，动态 PD 只低约 6%，明显优于 1:3 / 3:1。
- 代价是动态 PD 的 ITL p99 偏高（prefill 抢占 decode），是下一步要优化的点。
- 真实 trace 开环回放里，重载下动态 PD 相对静态 1:3 基线省 44.8% GPU 时间（等效利用率 +81.3%）。

TTFT mean（8 卡，ms，越低越好）：

```
523.8  1:3  ████████████████
184.9  1:1  ██████
184.4  3:1  ██████
158.9  agg  █████
```

## 方法

不直接改 Dynamo 源码，用 DynaSim（`python -m dynamo.replay`，Rust `lib/mocker` 驱动）在同一负载、同一总卡数下跑静态 PD 和动态 PD 对比。1 卡 = 1 worker（TP=1）。

## 压测结果

### 闭环饱和

8 卡，16000 请求，concurrency 256：

| 配置 | TTFT mean | TTFT p99 | ITL mean | ITL p99 | E2E mean | E2E p99 | rps | 总 tok/s | GPU-hours |
|---|---|---|---|---|---|---|---|---|---|
| 静态 1:3 (2P+6D) | 523.8 | 574.2 | 6.6 | 6.6 | 1357.0 | 1408.2 | 187.4 | 119944 | 0.2 |
| 静态 1:1 (4P+4D) | 184.9 | 349.7 | 7.4 | 7.8 | 1127.9 | 1280.4 | 225.6 | 144380 | 0.2 |
| 静态 3:1 (6P+2D) | 184.4 | 204.8 | 9.2 | 9.8 | 1355.5 | 1374.0 | 187.5 | 119981 | 0.2 |
| 动态 (8 agg) | 158.9 | 175.5 | 8.3 | 39.1 | 1209.7 | 1215.8 | 210.5 | 134701 | 0.2 |

1024 卡，100 万请求，concurrency 32768：

| 配置 | TTFT mean | TTFT p99 | ITL mean | ITL p99 | E2E mean | E2E p99 | rps | 总 tok/s | GPU-hours |
|---|---|---|---|---|---|---|---|---|---|
| 静态 1:3 (256P+768D) | 528.1 | 1024.7 | 6.6 | 6.6 | 1360.9 | 1857.0 | 23746 | 15197474 | 12.0 |
| 静态 1:1 (512P+512D) | 189.0 | 517.1 | 7.4 | 7.8 | 1131.2 | 1462.0 | 28592 | 18298543 | 9.9 |
| 静态 3:1 (768P+256D) | 186.5 | 348.9 | 9.2 | 9.8 | 1355.8 | 1527.6 | 23842 | 15258601 | 11.9 |
| 动态 (1024 agg) | 161.0 | 350.7 | 8.3 | 39.1 | 1209.3 | 1391.3 | 26801 | 17152776 | 10.6 |

单位：TTFT / ITL / E2E 为 ms。

### 真实 trace 开环回放

闭环饱和测不出利用率（供应时间被固化成 worker 数 × duration）。改用真实 trace 开环回放：真实到达时间戳 + 真实长度分布，固定总卡数，比 makespan。

trace 是 `mooncake_trace_1000.jsonl`（1000 请求，ISL 均值 9.8k，偏 prefill 密集），用 `--arrival-speedup-ratio` 扫到达强度。

makespan（秒）：

| 到达强度 | 静态 1:3 | 静态 1:1 | 静态 3:1 | 动态 |
|---|---|---|---|---|
| 1× | 183.03 | 182.41 | 182.55 | 181.84 |
| 2× | 102.01 | 94.27 | 102.40 | 94.18 |
| 4× | 85.08 | 65.16 | 70.53 | 58.47 |
| 8× | 86.81 | 62.57 | 74.26 | 47.89 |

8× 饱和下动态 PD 相对各静态配比：

| 对比 | makespan 缩短 | GPU 时间节省 | 等效利用率提升 |
|---|---|---|---|
| vs 1:3 | 86.81 → 47.89s | 44.8% | +81.3% |
| vs 1:1 | 62.57 → 47.89s | 23.5% | +30.7% |
| vs 3:1 | 74.26 → 47.89s | 35.5% | +55.1% |

轻载收益接近 0，重载收益单调上升。动态 PD 即便对这个 prefill 密集 trace 的最优固定配比（1:1）也快 23.5%。

### 条件分离

`aggregated` 是无 KV 传输的理想上界。条件分离保留 2P+6D 静态骨架，用策略把部分请求的 prefill 挪到 decode worker 本地做（bypass），更贴近 FlowKV 的真实形态。三种策略来自 `ConditionalDisaggPolicyKind`。所有 disagg 配置都开 KV 传输（128KiB/token，带宽 200GB/s）。

makespan（秒）：

| 配置 | 1× | 2× | 4× | 8× |
|---|---|---|---|---|
| static 无 KV | 183.03 | 102.01 | 85.08 | 86.81 |
| static 有 KV | 183.04 | 102.11 | 85.55 | 89.00 |
| isl_bounding | 182.55 | 103.58 | 87.92 | 87.39 |
| prefill_load | 181.48 | 93.09 | 55.88 | 39.67 |
| isl_or_load | 181.48 | 93.06 | 55.43 | 39.10 |

bypass 率（`prefill_worker_idx is None` 判定）：

| 策略 | 1× | 2× | 4× | 8× |
|---|---|---|---|---|
| isl_bounding | 0% | 0% | 0% | 0% |
| prefill_load | 88.2% | 87.2% | 87.7% | 90.2% |
| isl_or_load | 88.2% | 87.3% | 88.2% | 88.1% |

- KV 传输开销在这个 trace 上很小（0–2.5%）。长上下文偏 prefill 密集，prefill 计算和排队占主导，单请求 KV 传输（约 1.25GiB @ 200GB/s ≈ 6.3ms）相对可忽略，短请求才吃这个开销。
- isl_bounding 在这里 0% bypass：它面向短请求 + 热 cache（eff_isl < 2048 且占比 < 0.7），这个 trace 冷启动没有前缀复用，也没有短请求。要验证它得换多轮对话 / 共享前缀的 trace。
- prefill_load / isl_or_load 收益明显：2 个 prefill worker 被长 ISL 长期打满，策略把约 88–90% 请求的 prefill 挪到 6 个相对空闲的 decode worker，消除 prefill 排队瓶颈。相对有 KV 的 static 基线，8× 下 makespan 降 55.4%，TTFT 从 30.8s 降到 52ms。

## 脚本

| 文件 | 用途 |
|---|---|
| run_ablation.py | 批量跑静态/动态 PD 各配置，输出对比表 |
| run_trace_ablation.py | 真实 trace 开环回放，算利用率提升 |
| run_conditional_disagg_ablation.py | 条件分离消融，量化 bypass 收益 |
| verify_ablation.py | 可复现性检查（确定性、种子、Little's law、稳态） |
| trace_inspect.py | 看 trace 的时间戳和长度分布 |

## 源码改动

条件分离的回放支持只动 DynaSim 仿真引擎（`lib/mocker`），没动生产 runtime。生产侧 `ConditionalDisaggPolicyKind` 和配置字段本来就存在，这里只是把它接进离线回放。

| 文件 | 改动 |
|---|---|
| lib/mocker/src/replay/offline/components/engine.rs | has_active_workers / all_workers_have_work，暴露 prefill 池饱和信号 |
| lib/mocker/src/replay/offline/state.rs | DisaggRequestState 加 bypass_remote_prefill 标记 |
| lib/mocker/src/replay/offline/disagg.rs | 本地镜像枚举 + ConditionalDisaggReplayConfig + route_local_prefill（跳过远端 prefill、零 KV 传输） |
| lib/mocker/src/replay/offline/extensions/kv_router/composition_disagg.rs | 从 KvRouterConfig 派生回放配置并接线 |
| lib/mocker/src/replay/offline/extensions/kv_events/mod.rs | 传 None 保持默认禁用 |

bypass 语义：策略判定 bypass 时，请求不再走「prefill worker 做 prefill → 跨 worker KV 传输 → decode worker 接续」，而是直接在选定的 decode worker 上本地 prefill（`prefill_worker_idx = None`，零传输）。

改动以 `conditional_disagg_replay.patch` 提供，基线是 Dynamo `v1.5.0-gemma-4-31b-dev.1`（commit `4645399`）。在官方源码根目录执行：

```bash
git apply --check conditional_disagg_replay.patch   # 预检能否干净应用
git apply conditional_disagg_replay.patch            # 应用
```

应用后重新编译 `dynamo._core`（`maturin develop --release --features mocker-kvbm-offload`），离线回放即可启用条件分离策略。

## 复现

在 WSL 里装好 Rust 工具链和 Python 环境，编译出 `dynamo._core`（`maturin develop --release --features mocker-kvbm-offload`），然后跑上面的脚本。DynaSim 是纯 CPU 仿真，不依赖 GPU；默认多项式性能模型对 batch 不敏感，绝对延迟不代表真实 A100，但静态 vs 动态的相对对比是自洽的。

## 参考

- FlowKV: Weiqing Li et al., *FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling*, Alibaba Cloud, arXiv:2504.03775
- NVIDIA Dynamo: https://github.com/ai-dynamo/dynamo
- LMCache: https://github.com/LMCache/LMCache
