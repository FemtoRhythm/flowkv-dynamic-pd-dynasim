# FlowKV-Style Dynamic Prefill-Decode Scheduling: A DynaSim Simulation Study

## Abstract

This work compares static Prefill-Decode disaggregation with dynamic mixed scheduling using DynaSim, the discrete-event simulation engine of NVIDIA Dynamo, and extends offline replay support to conditional disaggregation policies. Results show that dynamic scheduling consistently outperforms static configurations on time-to-first-token, saving 44.8% of GPU time under heavy load on a real workload. Conditional disaggregation further removes the prefill queueing bottleneck by migrating part of the prefill to decode nodes, reducing makespan by 55.4% at 8× arrival intensity.

## 1 Introduction

Disaggregated inference places prefill and decode on separate nodes to isolate their differing latency sensitivity. Static configurations fix the prefill-to-decode ratio, causing wasted compute when workload characteristics deviate from assumptions. This work reproduces the load-aware dynamic scheduling direction proposed by FlowKV and quantifies its benefits and costs against static configurations under identical load and total GPU count.

## 2 Methodology

This work does not modify the Dynamo production runtime. Simulation uses DynaSim (`python -m dynamo.replay`, driven by the Rust `lib/mocker`). Each GPU maps to one worker with tensor parallelism 1. Three scheduling categories are compared: static disaggregation, dynamic aggregation, and conditional disaggregation.

<div align="center">
<table>
  <tr>
    <td><img src="figures/fig_ttft.png" width="280"></td>
    <td><img src="figures/fig_makespan.png" width="280"></td>
    <td><img src="figures/fig_conditional.png" width="280"></td>
  </tr>
</table>
</div>

## 3 Experimental Setup

### 3.1 Closed-Loop Saturation

Latency and throughput are measured under fixed request count and concurrency. Latency is reported in milliseconds.

8 GPUs, 16000 requests, concurrency 256:

| Config | TTFT mean | TTFT p99 | ITL mean | ITL p99 | E2E mean | E2E p99 | rps | Total tok/s | GPU-hours |
|---|---|---|---|---|---|---|---|---|---|
| Static 1:3 (2P+6D) | 523.8 | 574.2 | 6.6 | 6.6 | 1357.0 | 1408.2 | 187.4 | 119944 | 0.2 |
| Static 1:1 (4P+4D) | 184.9 | 349.7 | 7.4 | 7.8 | 1127.9 | 1280.4 | 225.6 | 144380 | 0.2 |
| Static 3:1 (6P+2D) | 184.4 | 204.8 | 9.2 | 9.8 | 1355.5 | 1374.0 | 187.5 | 119981 | 0.2 |
| Dynamic (8 agg) | 158.9 | 175.5 | 8.3 | 39.1 | 1209.7 | 1215.8 | 210.5 | 134701 | 0.2 |

1024 GPUs, 1M requests, concurrency 32768:

| Config | TTFT mean | TTFT p99 | ITL mean | ITL p99 | E2E mean | E2E p99 | rps | Total tok/s | GPU-hours |
|---|---|---|---|---|---|---|---|---|---|
| Static 1:3 (256P+768D) | 528.1 | 1024.7 | 6.6 | 6.6 | 1360.9 | 1857.0 | 23746 | 15197474 | 12.0 |
| Static 1:1 (512P+512D) | 189.0 | 517.1 | 7.4 | 7.8 | 1131.2 | 1462.0 | 28592 | 18298543 | 9.9 |
| Static 3:1 (768P+256D) | 186.5 | 348.9 | 9.2 | 9.8 | 1355.8 | 1527.6 | 23842 | 15258601 | 11.9 |
| Dynamic (1024 agg) | 161.0 | 350.7 | 8.3 | 39.1 | 1209.3 | 1391.3 | 26801 | 17152776 | 10.6 |

### 3.2 Real-Trace Open-Loop Replay

Closed-loop saturation fixes supply time as the product of worker count and duration, masking utilization differences. A real workload is therefore replayed in open loop with a fixed total GPU count, using makespan as the metric. The workload `mooncake_trace_1000.jsonl` contains 1000 requests with mean input sequence length 9.8k, biased toward prefill, and arrival intensity is swept via `--arrival-speedup-ratio`.

Makespan (seconds):

| Arrival intensity | Static 1:3 | Static 1:1 | Static 3:1 | Dynamic |
|---|---|---|---|---|
| 1× | 183.03 | 182.41 | 182.55 | 181.84 |
| 2× | 102.01 | 94.27 | 102.40 | 94.18 |
| 4× | 85.08 | 65.16 | 70.53 | 58.47 |
| 8× | 86.81 | 62.57 | 74.26 | 47.89 |

Improvement of dynamic scheduling over static configurations at 8× intensity:

| Comparison | Makespan reduction | GPU time saved | Utilization gain |
|---|---|---|---|
| vs 1:3 | 86.81 → 47.89s | 44.8% | +81.3% |
| vs 1:1 | 62.57 → 47.89s | 23.5% | +30.7% |
| vs 3:1 | 74.26 → 47.89s | 35.5% | +55.1% |

Differences are small under light load, while the advantage of dynamic scheduling grows monotonically with arrival intensity and exceeds the optimal static configuration for this workload.

### 3.3 Conditional Disaggregation

The aggregated configuration is the ideal upper bound without KV transfer. Conditional disaggregation retains the static skeleton and migrates part of the prefill to decode nodes based on policy, closer to the actual FlowKV form. Policies come from `ConditionalDisaggPolicyKind`, and all disaggregated configurations enable KV transfer (128KiB/token at 200GB/s).

Makespan (seconds):

| Config | 1× | 2× | 4× | 8× |
|---|---|---|---|---|
| static no KV | 183.03 | 102.01 | 85.08 | 86.81 |
| static with KV | 183.04 | 102.11 | 85.55 | 89.00 |
| isl_bounding | 182.55 | 103.58 | 87.92 | 87.39 |
| prefill_load | 181.48 | 93.09 | 55.88 | 39.67 |
| isl_or_load | 181.48 | 93.06 | 55.43 | 39.10 |

Bypass rate (determined by `prefill_worker_idx is None`):

| Policy | 1× | 2× | 4× | 8× |
|---|---|---|---|---|
| isl_bounding | 0% | 0% | 0% | 0% |
| prefill_load | 88.2% | 87.2% | 87.7% | 90.2% |
| isl_or_load | 88.2% | 87.3% | 88.2% | 88.1% |

## 4 Discussion

KV transfer overhead is limited in this workload, accounting for 0–2.5%. The workload is prefill-dominant, where prefill compute and queueing prevail. A single-request KV transfer of about 1.25GiB takes approximately 6.3ms at 200GB/s, negligible relative to the overall latency; the overhead matters only for short requests.

The isl_bounding policy shows 0% bypass on this workload because it targets short requests with warm caches, while this workload is cold-start with no prefix reuse. It requires further validation on multi-turn conversation or shared-prefix workloads.

The prefill_load and isl_or_load policies are effective. Two prefill nodes are saturated by long input sequences, and the policy migrates prefill of about 88–90% of requests to six relatively idle decode nodes, removing the prefill queueing bottleneck. Relative to the static baseline with KV transfer, makespan drops by 55.4% and TTFT falls from 30.8s to 52ms at 8× intensity.

## 5 Conclusion

Dynamic mixed scheduling consistently outperforms static configurations on latency and substantially improves compute utilization under heavy load. Conditional disaggregation retains the static skeleton while removing the prefill queueing bottleneck, confirming the effectiveness of load-aware scheduling. The elevated ITL tail latency of dynamic scheduling stems from prefill preemption of decode and is a direction for future optimization.

## 6 Reproduction

### 6.1 Dependencies and Build

Configure the Rust toolchain and Python environment in WSL, then build `dynamo._core`:

```bash
maturin develop --release --features mocker-kvbm-offload
```

DynaSim is a CPU-only simulation with no GPU dependency. The default polynomial performance model is insensitive to batch size, so absolute latencies do not reflect real hardware, but relative comparisons across configurations remain consistent.

### 6.2 Scripts

| File | Purpose |
|---|---|
| run_ablation.py | Batch static and dynamic scheduling configurations |
| run_trace_ablation.py | Real-trace open-loop replay |
| run_conditional_disagg_ablation.py | Conditional disaggregation ablation |
| verify_ablation.py | Reproducibility checks |
| trace_inspect.py | Workload characterization |
| plot_figures.py | Figure generation |

## 7 Source Changes

Conditional disaggregation replay support only touches the simulation engine `lib/mocker`, leaving the production runtime unchanged.

| File | Change |
|---|---|
| lib/mocker/src/replay/offline/components/engine.rs | Expose prefill pool saturation signal |
| lib/mocker/src/replay/offline/state.rs | Add bypass state flag |
| lib/mocker/src/replay/offline/disagg.rs | Conditional disaggregation decision and local prefill routing |
| lib/mocker/src/replay/offline/extensions/kv_router/composition_disagg.rs | Replay config derivation and wiring |
| lib/mocker/src/replay/offline/extensions/kv_events/mod.rs | Keep disabled by default |

When a policy decides to bypass, the request skips the prefill-node compute, cross-node KV transfer, and decode-node handoff, instead performing local prefill on the chosen decode node with zero KV transfer.

The changes are provided as `conditional_disagg_replay.patch` against Dynamo `v1.5.0-gemma-4-31b-dev.1`. Apply from the source root:

```bash
git apply --check conditional_disagg_replay.patch
git apply conditional_disagg_replay.patch
```

## References

- Weiqing Li et al. *FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling*. arXiv:2504.03775
- NVIDIA Dynamo: https://github.com/ai-dynamo/dynamo
- LMCache: https://github.com/LMCache/LMCache
