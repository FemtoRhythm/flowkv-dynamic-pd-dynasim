# FlowKV 式动态 Prefill-Decode 混合调度：基于 DynaSim 的仿真评估

[English](README.md)

## 摘要

本文基于 NVIDIA Dynamo 离散事件仿真引擎 DynaSim，对静态 Prefill-Decode 分离与动态混合调度两种架构进行对比评估，并实现条件分离策略的离线回放支持。结果表明，动态调度在首包延迟指标上一致优于静态配置，在真实工作负载重载条件下可节省 44.8% 的 GPU 时间。条件分离策略将部分请求的 prefill 迁移至 decode 节点，进一步消除 prefill 排队瓶颈，在 8× 到达强度下使 makespan 降低 55.4%。

## 1 引言

分离式推理将 prefill 与 decode 部署于不同节点，以隔离两类工作负载对延迟的差异化敏感度。静态配置固定 prefill 与 decode 节点比例，当负载特征偏离预设假设时导致算力浪费。本文复现 FlowKV 提出的负载感知动态混合调度方向，并利用 DynaSim 在相同负载、相同总卡数下量化动态调度相对静态配置的收益与代价。

## 2 方法

本工作不修改 Dynamo 生产运行时，采用 DynaSim（`python -m dynamo.replay`，由 Rust `lib/mocker` 驱动）进行仿真。每个 GPU 对应一个 worker，张量并行度为 1。对比三类调度配置：静态分离、动态混合（aggregated）与条件分离。

<div align="center">
<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/FemtoRhythm/flowkv-dynamic-pd-dynasim/main/figures/fig_ttft.png" width="280"></td>
    <td><img src="https://raw.githubusercontent.com/FemtoRhythm/flowkv-dynamic-pd-dynasim/main/figures/fig_makespan.png" width="280"></td>
    <td><img src="https://raw.githubusercontent.com/FemtoRhythm/flowkv-dynamic-pd-dynasim/main/figures/fig_conditional.png" width="280"></td>
  </tr>
</table>
</div>

## 3 实验设置

### 3.1 闭环饱和测试

在固定请求数与并发度下测量各配置的延迟与吞吐指标。延迟单位均为毫秒。

8 卡，16000 请求，并发度 256：

| 配置 | TTFT mean | TTFT p99 | ITL mean | ITL p99 | E2E mean | E2E p99 | rps | 总 tok/s | GPU-hours |
|---|---|---|---|---|---|---|---|---|---|
| 静态 1:3 (2P+6D) | 523.8 | 574.2 | 6.6 | 6.6 | 1357.0 | 1408.2 | 187.4 | 119944 | 0.2 |
| 静态 1:1 (4P+4D) | 184.9 | 349.7 | 7.4 | 7.8 | 1127.9 | 1280.4 | 225.6 | 144380 | 0.2 |
| 静态 3:1 (6P+2D) | 184.4 | 204.8 | 9.2 | 9.8 | 1355.5 | 1374.0 | 187.5 | 119981 | 0.2 |
| 动态 (8 agg) | 158.9 | 175.5 | 8.3 | 39.1 | 1209.7 | 1215.8 | 210.5 | 134701 | 0.2 |

1024 卡，100 万请求，并发度 32768：

| 配置 | TTFT mean | TTFT p99 | ITL mean | ITL p99 | E2E mean | E2E p99 | rps | 总 tok/s | GPU-hours |
|---|---|---|---|---|---|---|---|---|---|
| 静态 1:3 (256P+768D) | 528.1 | 1024.7 | 6.6 | 6.6 | 1360.9 | 1857.0 | 23746 | 15197474 | 12.0 |
| 静态 1:1 (512P+512D) | 189.0 | 517.1 | 7.4 | 7.8 | 1131.2 | 1462.0 | 28592 | 18298543 | 9.9 |
| 静态 3:1 (768P+256D) | 186.5 | 348.9 | 9.2 | 9.8 | 1355.8 | 1527.6 | 23842 | 15258601 | 11.9 |
| 动态 (1024 agg) | 161.0 | 350.7 | 8.3 | 39.1 | 1209.3 | 1391.3 | 26801 | 17152776 | 10.6 |

### 3.2 真实 trace 开环回放

闭环饱和测试中供应时间被固化为 worker 数与持续时间的乘积，无法反映利用率差异。因此采用真实工作负载开环回放，固定总卡数，以 makespan 为指标。工作负载为 `mooncake_trace_1000.jsonl`，含 1000 请求，输入序列长度均值 9.8k，偏向 prefill 密集，通过 `--arrival-speedup-ratio` 调节到达强度。

makespan（秒）：

| 到达强度 | 静态 1:3 | 静态 1:1 | 静态 3:1 | 动态 |
|---|---|---|---|---|
| 1× | 183.03 | 182.41 | 182.55 | 181.84 |
| 2× | 102.01 | 94.27 | 102.40 | 94.18 |
| 4× | 85.08 | 65.16 | 70.53 | 58.47 |
| 8× | 86.81 | 62.57 | 74.26 | 47.89 |

在 8× 到达强度下，动态调度相对各静态配置的改善：

| 对比 | makespan 缩短 | GPU 时间节省 | 等效利用率提升 |
|---|---|---|---|
| vs 1:3 | 86.81 → 47.89s | 44.8% | +81.3% |
| vs 1:1 | 62.57 → 47.89s | 23.5% | +30.7% |
| vs 3:1 | 74.26 → 47.89s | 35.5% | +55.1% |

轻载条件下各配置差距有限，重载条件下动态调度的优势随到达强度单调上升，且优于该工作负载的最优固定配比。

### 3.3 条件分离

aggregated 配置为无 KV 传输的理想上界。条件分离保留静态骨架，通过策略将部分请求的 prefill 迁移至 decode 节点本地执行，更接近 FlowKV 的实际形态。策略来自 `ConditionalDisaggPolicyKind`，所有分离配置均启用 KV 传输（128KiB/token，带宽 200GB/s）。

makespan（秒）：

| 配置 | 1× | 2× | 4× | 8× |
|---|---|---|---|---|
| static 无 KV | 183.03 | 102.01 | 85.08 | 86.81 |
| static 有 KV | 183.04 | 102.11 | 85.55 | 89.00 |
| isl_bounding | 182.55 | 103.58 | 87.92 | 87.39 |
| prefill_load | 181.48 | 93.09 | 55.88 | 39.67 |
| isl_or_load | 181.48 | 93.06 | 55.43 | 39.10 |

bypass 率（以 `prefill_worker_idx is None` 判定）：

| 策略 | 1× | 2× | 4× | 8× |
|---|---|---|---|---|
| isl_bounding | 0% | 0% | 0% | 0% |
| prefill_load | 88.2% | 87.2% | 87.7% | 90.2% |
| isl_or_load | 88.2% | 87.3% | 88.2% | 88.1% |

## 4 讨论

本工作负载中 KV 传输开销有限，占比 0–2.5%。该负载偏向 prefill 密集，prefill 计算与排队占据主导，单请求 KV 传输约为 1.25GiB，在 200GB/s 带宽下耗时约 6.3ms，相对可忽略。短请求场景下该开销才会显著。

isl_bounding 策略在本负载下 bypass 率为 0%，因其面向短请求与热缓存场景，而该负载为冷启动，缺乏前缀复用。该策略需在多轮对话或共享前缀负载下进一步验证。

prefill_load 与 isl_or_load 策略效果显著。两个 prefill 节点被长输入序列长期占用，策略将约 88–90% 请求的 prefill 迁移至六个相对空闲的 decode 节点，消除 prefill 排队瓶颈。相对启用 KV 传输的静态基线，8× 到达强度下 makespan 降低 55.4%，TTFT 由 30.8s 降至 52ms。

## 5 结论

动态混合调度在延迟指标上一致优于静态配置，重载条件下显著提升算力利用率。条件分离策略在保留静态骨架的同时，有效消除 prefill 排队瓶颈，验证了负载感知调度的有效性。动态调度的 ITL 尾延迟偏高，源于 prefill 对 decode 的抢占，是后续优化的方向。

## 6 复现

### 6.1 依赖与编译

在 WSL 中配置 Rust 工具链与 Python 环境，编译 `dynamo._core`：

```bash
maturin develop --release --features mocker-kvbm-offload
```

DynaSim 为纯 CPU 仿真，不依赖 GPU。默认多项式性能模型对 batch 不敏感，绝对延迟不代表真实硬件，但配置间的相对对比保持自洽。

### 6.2 实验脚本

| 文件 | 用途 |
|---|---|
| run_ablation.py | 批量执行静态与动态调度各配置 |
| run_trace_ablation.py | 真实 trace 开环回放 |
| run_conditional_disagg_ablation.py | 条件分离策略消融 |
| verify_ablation.py | 可复现性校验 |
| trace_inspect.py | 工作负载特征探查 |
| plot_figures.py | 结果图表生成 |

## 7 源码改动

条件分离的离线回放支持仅涉及仿真引擎 `lib/mocker`，未改动生产运行时。

| 文件 | 改动 |
|---|---|
| lib/mocker/src/replay/offline/components/engine.rs | 暴露 prefill 池饱和信号 |
| lib/mocker/src/replay/offline/state.rs | 增加 bypass 状态标记 |
| lib/mocker/src/replay/offline/disagg.rs | 条件分离决策与本地 prefill 路由 |
| lib/mocker/src/replay/offline/extensions/kv_router/composition_disagg.rs | 回放配置派生与接线 |
| lib/mocker/src/replay/offline/extensions/kv_events/mod.rs | 保持默认禁用 |

策略判定 bypass 时，请求不再执行「prefill 节点计算 → 跨节点 KV 传输 → decode 节点接续」，而是直接在选定 decode 节点本地 prefill，零 KV 传输。

改动以 `conditional_disagg_replay.patch` 提供，基线为 Dynamo `v1.5.0-gemma-4-31b-dev.1`。在官方源码根目录执行：

```bash
git apply --check conditional_disagg_replay.patch
git apply conditional_disagg_replay.patch
```

## 参考文献

- Weiqing Li et al. *FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling*. arXiv:2504.03775
- NVIDIA Dynamo: https://github.com/ai-dynamo/dynamo
- LMCache: https://github.com/LMCache/LMCache
