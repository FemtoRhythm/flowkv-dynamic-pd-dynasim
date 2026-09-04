# FlowKV 式动态 PD 混合调度 — DynaSim 仿真项目说明与压测报告

> 版本：v1.2
> 日期：2026-09-04
> 源码基线：NVIDIA Dynamo `v1.5.0-gemma-4-31b-dev.1`（commit `4645399`）

---

## 1. 项目概述

### 1.1 目标

基于 NVIDIA Dynamo + LMCache + NIXL 官方架构，改造实现 **FlowKV 式动态 Prefill-Decode（PD）混合调度**：

- worker 不再固定为 prefill-only / decode-only，运行时可动态切换角色；
- 全局负载感知调度，避免静态 PD 分离比例与真实负载错配导致的算力浪费。

### 1.2 方法

本项目**不直接改动 Dynamo 源码**，而是利用 Dynamo 自带的离散事件仿真引擎 **DynaSim**（`python -m dynamo.replay`，由 Rust `lib/mocker` 驱动），在同一工作负载、同一总卡数下，对**静态 PD（disagg）**与**动态 PD（aggregated）**做消融对比，量化动态 PD 的性能收益与代价，作为后续落地改造的量化基线。

### 1.3 核心结论（一图速览）

| 维度 | 结论 |
|---|---|
| TTFT（首包延迟） | 动态 PD 一致最优，比静态 1:3 低约 3 倍 |
| 吞吐 | 静态 1:1 最高，动态 PD 仅低 ~6%，显著优于 1:3 / 3:1 |
| 尾延迟 | 动态 PD 的 TTFT p99 最优 |
| 代价 | 动态 PD 的 ITL p99 偏高（prefill 抢占 decode），是后续优化点 |

---

## 2. 改动清单

### 2.1 Dynamo 源码改动：**无**

编译与仿真使用的是 **官方原版 v1.5.0 源码**，未修改任何 `.rs` / `.py` 源码文件。DynaSim 已原生支持两种调度模式（见 §5.3），消融对比正是在官方能力边界内完成，保证结果的真实性与可复现性。

### 2.2 方案设计的改造点（源码定位，未落地）

前期方案分析定位了「若要实现 FlowKV 式动态 PD」需改动的源码位置。这些是**设计蓝图**，未在本项目中实际修改：

**调度层（Planner / Router）**

| 文件 | 关键符号 | 说明 |
|---|---|---|
| `lib/kv-router/src/scheduling/selector.rs` | `DefaultWorkerSelector`（L100） | 默认 worker 选择器 |
| 同上 | `worker_logit`（L138–L308） | **已是 load-aware 打分**：`logit = prefill_load_scale·adjusted_prefill_blocks + decode_blocks + active_request_cost_blocks` |
| 同上 | `select_worker`（L312） | worker 选择入口 |
| `lib/kv-router/src/scheduling/config.rs` | `RouterQueuePolicy`（L321） | 队列策略 |
| 同上 | `RouterPrefillLoadModel`（L340） | prefill 负载模型 |
| 同上 | `ConditionalDisaggPolicyKind`（L378） | 条件分离策略：`IslBounding` / `PrefillLoad` / `IslOrLoad` |
| 同上 | `RouterConfigOverride`（L442） | 路由配置覆盖 |

**Worker 管理层**

| 文件 | 关键符号 | 说明 |
|---|---|---|
| `lib/llm/src/worker_type.rs` | `WorkerType` 枚举（L35–L40） | `Prefill` / `Decode` / `Encode` / **`Aggregated`**（通用混合 worker 已在枚举中） |
| `lib/llm/src/kv_router/prefill_router/mod.rs` | `PrefillRouter`（L155） | prefill 路由 |
| 同上 | `generate`（L191–L387） | PD 拆分核心逻辑 |
| 同上 | `prepare_prefill_dispatch`（L391） | prefill 派发准备 |
| 运行时监控 | `ModelWatcher` + `WorkerSet` + `ModelManager` + `KvWorkerMonitor` | worker 生命周期与实时负载采集 |

**KV 传输层**

| 组件 | 说明 |
|---|---|
| `PdConnector`（MultiConnector） | 组合 `LMCacheConnectorV1`（offload/onboard）+ `NixlConnector`（跨 worker 拉取） |
| `kv_role = "kv_both"` | worker 同时承担 KV 生产与消费 |

**术语映射（旧 Python 版 → 新 Rust 版）**：`Planner/Router/Scheduler` → `KvRouter` + `LocalScheduler` + `SchedulerQueue` + `DefaultWorkerSelector` + `SchedulingPolicy`；`WorkerManager` → `ModelWatcher` + `WorkerSet` + `ModelManager` + `KvWorkerMonitor`；`--is-prefill-worker`（已废弃）→ `--disaggregation-mode`（`agg` / `prefill` / `decode` / `encode`）。

### 2.3 新增脚本（本项目交付物）

| 文件 | 用途 |
|---|---|
| `run_ablation.py` | 消融测试驱动：批量跑静态/动态 PD 各配置，解析报告 JSON，输出对比表 |
| `verify_ablation.py` | 可复现性验证：确定性、种子鲁棒性、请求完整性、Little's law 自洽、稳态收敛 |
| `run_trace_ablation.py` | 真实 trace 开环回放：真实到达时间戳 + 真实长度分布，测量静态 vs 动态 PD 的 GPU 利用率提升 |
| `run_conditional_disagg_ablation.py` | 条件分离消融：同一真实 trace 下对比静态 PD（含/不含 KV 传输）与三种 `ConditionalDisaggPolicyKind`，量化 KV 传输开销与 bypass 收益 |
| `trace_inspect.py` | trace 文件探查：时间戳分布与 ISL/OSL 长度统计 |

### 2.4 编译环境改动（WSL 内，非源码）

| 改动 | 说明 |
|---|---|
| 安装 Rust 工具链 1.96.1 | 通过 rustup（rsproxy 镜像），匹配 `rust-toolchain.toml` |
| 创建 Python 3.12.14 venv | `uv venv`，路径 `/home/femto/dynamo-venv` |
| 安装 maturin 1.15.0 | pip（清华镜像） |
| apt 安装系统依赖 | `protobuf-compiler`（velo-transports 需要 protoc）、`libclang-dev` + `clang`（nixl-sys/bindgen 需要 libclang） |
| apt 源切清华镜像 + 删除坏代理 | 原 `95proxy` 指向失效的宿主机 1080 端口；源切 `mirrors.tuna.tsinghua.edu.cn` |
| 源码副本 | rsync 到 WSL 内部 ext4 `/home/femto/dynamo`（避开 9p 挂载的性能损耗） |
| 编译 `dynamo._core` | `maturin develop --release --features mocker-kvbm-offload` |
| 安装 `ai-dynamo` 1.4.1 | 根目录 `pip install -e .`（hatchling） |

### 2.5 条件分离离线回放（v1.2 新增源码改动，仅仿真引擎）

为量化「条件分离（conditional disagg）」——比 `aggregated` 上界更贴近 FlowKV 的动态 PD 形态（见 §8.2）——在 **DynaSim 仿真引擎（`lib/mocker`，非生产 runtime）** 上新增了对 `ConditionalDisaggPolicyKind` 的离线回放支持。生产侧 `lib/kv-router` 的 `ConditionalDisaggPolicyKind`（`IslBounding` / `PrefillLoad` / `IslOrLoad`）与配置字段（`conditional_disagg_enabled/policy/eff_isl_threshold/eff_isl_ratio_threshold`）本已存在，v1.2 只是把它接入离线回放。

| 文件 | 改动 |
|---|---|
| `lib/mocker/src/replay/offline/components/engine.rs` | 新增 `has_active_workers()`、`all_workers_have_work()`，暴露 prefill 池饱和信号 |
| `lib/mocker/src/replay/offline/state.rs` | `DisaggRequestState` 新增 `bypass_remote_prefill` 标记 + 访问器 |
| `lib/mocker/src/replay/offline/disagg.rs` | 新增本地镜像枚举 `ConditionalDisaggPolicyKind`（offline 核心不直接依赖 `dynamo_kv_router`，遵守扩展防火墙）+ `ConditionalDisaggReplayConfig`（同步镜像生产 `ConditionalDisaggPolicy` 决策）+ `route_local_prefill`（跳过远端 prefill、零 KV 传输）+ 条件分离强制 `DestinationFirst` 握手顺序 |
| `lib/mocker/src/replay/offline/extensions/kv_router/composition_disagg.rs` | 从 `KvRouterConfig` 派生条件分离回放配置并接线（`replay_policy()` 把生产枚举转换为本地镜像枚举） |
| `lib/mocker/src/replay/offline/extensions/kv_events/mod.rs` | 传 `None` 保持默认禁用 |

**bypass 语义**：当策略判定 bypass 时，请求不再走「prefill worker 做 prefill → 跨 worker KV 传输 → decode worker 接续 decode」，而是直接在已选定的 decode worker 上本地完成 prefill（`prefill_worker_idx = None`，零传输）。下游用 `per_request` 记录中 `prefill_worker_idx is None` 作为「已 bypass」标记。

---

## 3. 参考资料

| 资料 | 链接 / 出处 | 用途 |
|---|---|---|
| **FlowKV**（核心参考） | Weiqing Li et al., *FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling*, Alibaba Cloud, arXiv:2504.03775, 2025 | 动态 PD 分配 + Load-Aware Scheduler + KV 传输优化（传输延迟降 96%，加速 15.2%–48.9%） |
| NVIDIA Dynamo 源码 | https://github.com/ai-dynamo/dynamo （tag `v1.5.0-gemma-4-31b-dev.1`） | 改造基座 |
| NVIDIA Dynamo 文档 | https://docs.nvidia.com/dynamo/ | 术语与架构 |
| DynaSim | `lib/mocker` + `components/src/dynamo/replay/` | 离散事件仿真引擎与 CLI |
| LMCache | https://github.com/LMCache/LMCache | KV offload/onboard 连接器 |
| NIXL | NVIDIA Inference Xfer Library | 跨 worker KV 拉取连接器 |
| 相关 PD 分离工作 | Splitwise、DistServe、Mooncake（FlowKV 引文） | PD 分离背景 |

---

## 4. 编译与环境

| 项 | 值 |
|---|---|
| 主机 | Windows 11 + WSL2（Ubuntu 26.04 "resolute"） |
| Rust | 1.96.1（`rust-toolchain.toml` 指定） |
| Python | 3.12.14（uv venv） |
| maturin | 1.15.0 |
| 编译 feature | `mocker-kvbm-offload`（KV offload / LMCache 建模） |
| 未启用 feature | `aic-forward-pass`（需 aiconfigurator-core + A100 AIC 数据，本地缺失 → 回退多项式性能模型） |
| 产物 | `ai-dynamo-runtime-1.4.1`（`dynamo._core`，cp310-abi3-linux_x86_64） |

> 说明：DynaSim 是**纯 CPU 离散事件仿真**，不依赖 GPU；本机 RTX 5080 不参与计算。默认多项式性能模型对 batch 不敏感，因此绝对延迟不代表真实 A100 数值，但**相对对比（静态 vs 动态）在自洽性上成立**（见 §7）。

---

## 5. DynaSim 压测配置

### 5.1 工作负载（所有配置保持一致）

| 参数 | 值 | 说明 |
|---|---|---|
| ISL（输入长度） | 512 tokens | 合成回放固定长度 |
| OSL（输出长度） | 128 tokens | 合成回放固定长度 |
| 到达模式 | 闭环并发（`--replay-concurrency`） | 忽略到达时间戳，保持固定 in-flight 数，测饱和吞吐 |
| 到达种子 | `--arrival-seed 42` | 闭环模式下无随机影响（见 §7.2） |
| 请求数 | 8 卡 16000；1024 卡 1000000 | 经稳态收敛验证后取值（见 §7.4） |

### 5.2 资源抽象

- **1 卡 = 1 worker**（TP=1）。消融清晰地把「卡数」直接映射为 worker 数，prefill:decode 比例直接体现在卡数分配上。
- 真实部署的 TP=8 / block size=16 属于目标部署配置，仿真中通过性能模型参数影响延迟，本次用默认多项式模型。

### 5.3 调度配置矩阵（每规模 4 个）

| 配置 | 静态/动态 | worker 分配（8 卡） | worker 分配（1024 卡） | CLI 关键参数 |
|---|---|---|---|---|
| 静态 1:3 | 静态 PD（disagg） | 2P + 6D | 256P + 768D | `--prefill-engine-args {"worker_type":"prefill"}` `--decode-engine-args {"worker_type":"decode"}` `--num-prefill-workers P` `--num-decode-workers D` |
| 静态 1:1 | 静态 PD | 4P + 4D | 512P + 512D | 同上 |
| 静态 3:1 | 静态 PD | 6P + 2D | 768P + 256D | 同上 |
| 动态 | 动态 PD（aggregated） | 8 agg | 1024 agg | `--extra-engine-args {"worker_type":"aggregated"}` `--num-workers N` |

### 5.4 并发参数

| 规模 | 总卡数 | concurrency | 每卡并发 |
|---|---|---|---|
| 8 卡 | 8 | 256 | 32 |
| 1024 卡 | 1024 | 32768 | 32 |

---

## 6. 压测结果

### 6.1 8 卡（16000 请求，concurrency 256）

| 配置 | TTFT mean | TTFT p99 | ITL mean | ITL p99 | E2E mean | E2E p99 | 吞吐(rps) | 总 tok/s | GPU-hours |
|---|---|---|---|---|---|---|---|---|---|
| 静态 1:3 (2P+6D) | 523.8 | 574.2 | 6.6 | 6.6 | 1357.0 | 1408.2 | 187.4 | 119944 | 0.2 |
| 静态 1:1 (4P+4D) | 184.9 | 349.7 | 7.4 | 7.8 | 1127.9 | 1280.4 | **225.6** | 144380 | 0.2 |
| 静态 3:1 (6P+2D) | 184.4 | 204.8 | 9.2 | 9.8 | 1355.5 | 1374.0 | 187.5 | 119981 | 0.2 |
| **动态 (8 agg)** | **158.9** | **175.5** | 8.3 | 39.1 | **1209.7** | **1215.8** | 210.5 | 134701 | 0.2 |

### 6.2 1024 卡（1000000 请求，concurrency 32768）

| 配置 | TTFT mean | TTFT p99 | ITL mean | ITL p99 | E2E mean | E2E p99 | 吞吐(rps) | 总 tok/s | GPU-hours |
|---|---|---|---|---|---|---|---|---|---|
| 静态 1:3 (256P+768D) | 528.1 | 1024.7 | 6.6 | 6.6 | 1360.9 | 1857.0 | 23746 | 15197474 | 12.0 |
| 静态 1:1 (512P+512D) | 189.0 | 517.1 | 7.4 | 7.8 | 1131.2 | 1462.0 | **28592** | 18298543 | **9.9** |
| 静态 3:1 (768P+256D) | 186.5 | 348.9 | 9.2 | 9.8 | 1355.8 | 1527.6 | 23842 | 15258601 | 11.9 |
| **动态 (1024 agg)** | **161.0** | **350.7** | 8.3 | 39.1 | **1209.3** | **1391.3** | 26801 | 17152776 | 10.6 |

> 指标单位：TTFT / ITL / E2E 均为毫秒（ms）；吞吐为每秒请求数（rps）；总 tok/s 为每秒输入+输出 token 总数；GPU-hours 为总 GPU 计算时间。

---

## 6.3 真实 trace 开环回放：GPU 利用率提升测量

针对「GPU 利用率提升多少」这一疑问，§6.1/§6.2 的**闭环饱和**压测无法给出答案——闭环模式下 DynaSim 把供应时间固化为 `worker数 × duration`，busy/idle 无法区分（这也是此前「负载总是刚好均衡」的根因）。因此改走**开环真实 trace 回放**：用真实到达时间戳 + 真实长度分布，在固定总卡数下对比静态 PD 与动态 PD 的 makespan，间接量化利用率。

### 6.3.1 方法与指标定义

- **trace**：`lib/bench/testdata/mooncake_trace_1000.jsonl`（1000 请求，开环真实到达时间戳，跨度 0→177s；长上下文偏 prefill 密集：ISL 均值 9.8k / max 121k，OSL 均值 199 / max 2000）。
- **到达强度**：`--arrival-speedup-ratio`（1×/2×/4×/8×）压缩到达时间线，扫过「未饱和 → 饱和」区间。
- **固定总卡数**：8 卡 = 8 worker（TP=1），与 §5.2 一致。
- **关键前提**：DynaSim 报告只暴露**供应** worker 秒（`prefill/decode_worker_seconds = worker数 × duration`），不暴露真实 busy 时间。故利用率用 makespan 推导——**同一份 trace、同一总卡数下，GPU-hours ∝ makespan**：

$$
\text{GPU 时间节省} = \frac{T_\text{static} - T_\text{dynamic}}{T_\text{static}},\quad
\text{等效利用率提升} = \frac{T_\text{static}}{T_\text{dynamic}} - 1
$$

### 6.3.2 结果（makespan，秒）

| 到达强度 | 静态 1:3 (2P+6D) | 静态 1:1 (4P+4D) | 静态 3:1 (6P+2D) | **动态 (8 agg)** |
|---|---|---|---|---|
| 1×（未饱和） | 183.03 | 182.41 | 182.55 | **181.84** |
| 2× | 102.01 | 94.27 | 102.40 | **94.18** |
| 4× | 85.08 | 65.16 | 70.53 | **58.47** |
| 8×（饱和） | 86.81 | 62.57 | 74.26 | **47.89** |

### 6.3.3 利用率提升（动态为基准，8× 饱和）

| 对比 | makespan 缩短 | GPU 时间节省 | 等效利用率提升 |
|---|---|---|---|
| vs 静态 1:3（基线 2P+6D） | 86.81→47.89s | **44.8%** | **+81.3%** |
| vs 静态 1:1（4P+4D） | 62.57→47.89s | 23.5% | +30.7% |
| vs 静态 3:1（6P+2D） | 74.26→47.89s | 35.5% | +55.1% |

### 6.3.4 结论与边界

1. **轻载下收益为 0，重载下收益显著**：1× 到达时静态与动态 makespan 几乎相同（系统未饱和，无排队）；随到达强度增大，静态 PD 的错配浪费被放大，动态 PD 优势单调上升。
2. **动态 PD 全面优于三个固定配比**：即便对这份 prefill 密集 trace 取「最优」固定配比（1:1），动态仍快 23.5%——因为动态 worker 可在 prefill 突发时把全部 8 卡投入 prefill，静态固定配比做不到。
3. **测量边界（务必注意）**：
   - 「动态 PD」在 DynaSim 里是 `worker_type=aggregated`（每 worker 同时做 prefill+decode，**无跨 worker KV 传输开销**），是 FlowKV 动态 PD 的**理想上界**；真实 FlowKV 还要付 KV transfer 成本，实际收益会略低于此。
   - 绝对延迟仍是多项式性能模型，不代表真实 A100；但相对 makespan 对比在同一模型下自洽。
   - 本 trace 偏 prefill 密集，与静态 1:3（decode 密集配比）天然错配，因此 1:3 基线收益最大（+81%）——这正印证 FlowKV 论点：静态 PD 需提前调准配比，负载一漂移即算力浪费。

---

## 6.4 条件分离消融：KV 传输开销与 bypass 收益

承接 §6.3 的「边界」：`aggregated` 是无 KV 传输的理想上界。这里用 §2.5 实现的**条件分离**（保留 2P+6D 静态骨架 + 策略触发 bypass）逼近更真实的 FlowKV 形态。同一份 mooncake trace（1000 请求，ISL 均值 9.8k），8 worker（2P+6D，TP=1），四个到达强度（1×/2×/4×/8×）。所有 disagg 配置都启用 KV 传输（`kv_bytes_per_token=128KiB/token`，`kv_transfer_bandwidth=200GB/s`），唯一变量是 bypass 策略。

### 6.4.1 makespan（秒）

| 配置 | 1× | 2× | 4× | 8×（饱和） |
|---|---|---|---|---|
| static（无 KV 传输） | 183.03 | 102.01 | 85.08 | 86.81 |
| static（有 KV 传输） | 183.04 | 102.11 | 85.55 | 89.00 |
| `isl_bounding` | 182.55 | 103.58 | 87.92 | 87.39 |
| `prefill_load` | 181.48 | 93.09 | 55.88 | **39.67** |
| `isl_or_load` | 181.48 | 93.06 | 55.43 | **39.10** |

### 6.4.2 bypass 率（`prefill_worker_idx is None` 判定）

| 策略 | 1× | 2× | 4× | 8× |
|---|---|---|---|---|
| `isl_bounding` | 0% | 0% | 0% | 0% |
| `prefill_load` | 88.2% | 87.2% | 87.7% | 90.2% |
| `isl_or_load` | 88.2% | 87.3% | 88.2% | 88.1% |

### 6.4.3 关键结论

1. **KV 传输开销很小（0–2.5%）**：static 无/有 KV 传输 makespan 几乎重合（8× 饱和仅差 2.5%）。原因：本 trace 长上下文偏 prefill 密集，prefill compute + 排队主导，单请求 KV 传输（9.8k×128KiB≈1.25GiB @ 200GB/s ≈ 6.3ms）相对可忽略。KV 传输开销对**短请求**才显著（见下一条）。

2. **`isl_bounding` 在本 trace 无效（0% bypass）**：该策略面向「短请求 + 热 cache」——eff_isl = prompt − decode 命中 < 2048 且占比 < 0.7 才 bypass。但本 trace 冷启动无前缀复用（decode 命中 0），且无 < 2048 的短请求，故永不触发。这是一个**有效的负结果**：isl_bounding 的收益场景（chat 短请求、前缀复用）与本 trace 不匹配，需换多轮对话/共享前缀 trace 才能体现。

3. **`prefill_load` / `isl_or_load` 显著受益**：本 trace 的 2 个 prefill worker 因长 ISL 长期饱和，策略把 ~88–90% 请求的 prefill 从饱和的 prefill 池挪到 6 个相对空闲的 decode worker 本地执行，消除 prefill 排队瓶颈。相对「有 KV 传输」的 static 基线：

   | 到达强度 | makespan 缩短 | GPU-hours 节省 |
   |---|---|---|
   | 2× | −8.8% | 0.020 |
   | 4× | −34.7% | 0.066 |
   | 8×（饱和） | **−55.4%** | **0.110** |

   同时 TTFT 从 19.3s（4×）/ 30.8s（8×）降至 ~40ms / ~52ms（prefill 排队消除的直接体现）。

4. **本质**：`prefill_load` 在 prefill 密集负载下退化为「几乎全部本地 prefill」，等价于动态 PD 把 decode 侧闲置算力动态调给 prefill，与 §6.3 的 `aggregated` 上界方向一致（8× 下 aggregated 47.9s vs 条件分离 39.7s——条件分离甚至更低，因为保留了 2 个专职 prefill worker 处理剩余 10% 未 bypass 的请求）。

5. **测量边界**：
   - 离线 `prefill_pool_busy()` 用「所有 prefill worker 均有活」作为池级饱和近似，对应生产 `PrefillLoadPolicy` 的「选中 prefill worker 越过 busy 线」；长 ISL 下两者几乎等价，故 bypass 率偏高（~88%）是策略语义而非 bug。
   - 本地 prefill 由 decode worker 的 `ActivateDestination` 建模（产出 Stored KV 事件、计入 decode worker 的 prefill pass），**非免费**。
   - 绝对延迟仍为多项式性能模型，相对对比自洽。

---

## 7. 结果验证（真实性与可复现性）

### 7.1 确定性（可复现）

同 seed=42 跑两次，9 项核心指标 **bit-exact 一致**。

### 7.2 种子鲁棒性

seed ∈ {42, 43, 123}，结果 **0.00% 波动**。原因：闭环并发模式忽略到达时间戳，仿真完全确定。

### 7.3 完整性 + Little's law 自洽

- 所有配置 `completed_requests == num_requests`（无请求丢失）。
- `rps × mean_e2e = in-flight ≤ concurrency`，稳态下 in-flight 达 96%+（接近满载）。

### 7.4 稳态收敛（发现并修正了一个真实问题）

1024 卡在 100k 请求时**未达稳态**（in-flight 仅 84%，吞吐被低估 ~15%），原因启动/排空阶段占比过高。增大请求数后收敛：

| request_count | rps | in-flight / 32768 |
|---|---|---|
| 100000 | 22636 | 27455（84%） |
| 400000 | 26022 | 31437（96%） |
| 1000000 | **26801** | 稳态 |

### 7.5 物理自洽性硬证据

1. **延迟 scale-invariant**：8 卡 vs 1024 卡，TTFT mean 差异仅 0.8%–2.2%（相同每卡负载下延迟不随集群规模变化）。
2. **吞吐线性缩放**：1024/8 = 128×，实测 126.7×–127.3×（~99% 达成，偏差来自整数请求边界）。
3. **GPU-hours 反比于吞吐**：静态 1:1 吞吐最高 → GPU-hours 最少（9.9）；动态次之（10.6）。

---

## 8. 结论与后续

### 8.1 结论

1. **动态 PD 在 TTFT 与尾延迟上一致最优**：TTFT mean 158.9–161.0ms、p99 175.5–350.7ms，显著优于静态 PD 的固定配比（尤其 1:3 的 528ms/1025ms），验证了 FlowKV「灵活 PD 分配 + 负载感知」的核心价值。
2. **静态 PD 对分离比例高度敏感**：负载（ISL=512 偏 prefill 密集）与比例错配时性能崩坏——1:3 的 prefill 池成为 TTFT 排队瓶颈，3:1 的 decode 池成为吞吐瓶颈。
3. **动态 PD 吞吐接近最优**：比静态 1:1 低约 6%，但换来 TTFT 降低 15%、p99 降低 47%，是更均衡的权衡。
4. **动态 PD 的代价明确**：ITL p99 从静态的 6.6–9.8ms 升至 39.1ms（prefill burst 抢占 decode step），这是后续 FlowKV 改造需优先优化的点（prefill 分块 / 优先级调度 / decode 抢占保护）。
5. **GPU 利用率提升（真实 trace 开环回放，§6.3）**：闭环饱和测不出利用率，开环真实 trace 下——轻载收益≈0，重载（8× 到达）下动态 PD 相对静态 1:3 基线**节省 44.8% GPU 时间（等效利用率 +81.3%）**，且全面优于三个固定配比（较最优 1:1 仍快 23.5%）。这是 FlowKV「灵活分配避免算力错配浪费」的量化证据；注意该值以 `aggregated`（无 KV transfer 开销）为上界。
6. **条件分离（§6.4）给出更贴近 FlowKV 的量化形态**：保留静态 2P+6D 骨架 + 策略触发 bypass，比 `aggregated` 上界更真实。结论：KV 传输开销在本 prefill 密集 trace 上很小（0–2.5%）；`isl_bounding` 因无短请求而无效；`prefill_load`/`isl_or_load` 在饱和时把 prefill 从饱和的 2 个 prefill worker 动态挪到 6 个空闲 decode worker，**makespan 降 55.4%（8×）、GPU-hours 省 0.11**，印证 FlowKV「负载感知动态 PD」的核心价值，且其收益是「消除 prefill 排队瓶颈」而非「省 KV 传输」。

### 8.2 后续可扩展方向

- ~~用 `ConditionalDisaggPolicyKind` 对真实 trace 建模「条件分离」~~ ✅ **已在 v1.2 完成**（§2.5、§6.4）：三种策略均已接入离线回放并消融；
- 对 `isl_bounding` 换**多轮对话 / 共享前缀 trace**（短请求 + 热 cache）复测——本 mooncake trace 无短请求故 0% bypass，需验证其真实收益场景；
- 增加 `--router-mode kv_router`（KV 感知路由）维度对比；
- 多组 ISL/OSL 负载鲁棒性扫描（prefill 密集 / decode 密集）；
- 启用 `aic-forward-pass` + A100 AIC 数据，获得真实 A100 延迟；
- 开环泊松到达（`--request-rate`）模式下的种子敏感性验证；
- 用 recipes 生产 trace（kimi-k2.6 等，真实长度分布、无时间戳）叠加泊松到达做大规模回放；
- 将方案 §2.2 的改造点落地到源码，实现真正的运行时动态 PD 切换。

---

*本文档由 DynaSim 消融测试自动采集数据整理而成，复现脚本见 `run_ablation.py`、`verify_ablation.py`、`run_trace_ablation.py` 与 `run_conditional_disagg_ablation.py`。*
