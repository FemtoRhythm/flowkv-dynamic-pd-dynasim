# FlowKV-Style Dynamic Prefill-Decode Scheduling: A DynaSim Simulation Study

[中文](README.zh-CN.md)

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

<table width="100%">
  <thead>
    <tr>
      <th>Config</th>
      <th>TTFT mean</th>
      <th>TTFT p99</th>
      <th>ITL mean</th>
      <th>ITL p99</th>
      <th>E2E mean</th>
      <th>E2E p99</th>
      <th>rps</th>
      <th>Total tok/s</th>
      <th>GPU-hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Static 1:3 (2P+6D)</td>
      <td>523.8</td>
      <td>574.2</td>
      <td>6.6</td>
      <td>6.6</td>
      <td>1357.0</td>
      <td>1408.2</td>
      <td>187.4</td>
      <td>119944</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>Static 1:1 (4P+4D)</td>
      <td>184.9</td>
      <td>349.7</td>
      <td>7.4</td>
      <td>7.8</td>
      <td>1127.9</td>
      <td>1280.4</td>
      <td>225.6</td>
      <td>144380</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>Static 3:1 (6P+2D)</td>
      <td>184.4</td>
      <td>204.8</td>
      <td>9.2</td>
      <td>9.8</td>
      <td>1355.5</td>
      <td>1374.0</td>
      <td>187.5</td>
      <td>119981</td>
      <td>0.2</td>
    </tr>
    <tr>
      <td>Dynamic (8 agg)</td>
      <td>158.9</td>
      <td>175.5</td>
      <td>8.3</td>
      <td>39.1</td>
      <td>1209.7</td>
      <td>1215.8</td>
      <td>210.5</td>
      <td>134701</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>

1024 GPUs, 1M requests, concurrency 32768:

<table width="100%">
  <thead>
    <tr>
      <th>Config</th>
      <th>TTFT mean</th>
      <th>TTFT p99</th>
      <th>ITL mean</th>
      <th>ITL p99</th>
      <th>E2E mean</th>
      <th>E2E p99</th>
      <th>rps</th>
      <th>Total tok/s</th>
      <th>GPU-hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Static 1:3 (256P+768D)</td>
      <td>528.1</td>
      <td>1024.7</td>
      <td>6.6</td>
      <td>6.6</td>
      <td>1360.9</td>
      <td>1857.0</td>
      <td>23746</td>
      <td>15197474</td>
      <td>12.0</td>
    </tr>
    <tr>
      <td>Static 1:1 (512P+512D)</td>
      <td>189.0</td>
      <td>517.1</td>
      <td>7.4</td>
      <td>7.8</td>
      <td>1131.2</td>
      <td>1462.0</td>
      <td>28592</td>
      <td>18298543</td>
      <td>9.9</td>
    </tr>
    <tr>
      <td>Static 3:1 (768P+256D)</td>
      <td>186.5</td>
      <td>348.9</td>
      <td>9.2</td>
      <td>9.8</td>
      <td>1355.8</td>
      <td>1527.6</td>
      <td>23842</td>
      <td>15258601</td>
      <td>11.9</td>
    </tr>
    <tr>
      <td>Dynamic (1024 agg)</td>
      <td>161.0</td>
      <td>350.7</td>
      <td>8.3</td>
      <td>39.1</td>
      <td>1209.3</td>
      <td>1391.3</td>
      <td>26801</td>
      <td>17152776</td>
      <td>10.6</td>
    </tr>
  </tbody>
</table>

### 3.2 Real-Trace Open-Loop Replay

Closed-loop saturation fixes supply time as the product of worker count and duration, masking utilization differences. A real workload is therefore replayed in open loop with a fixed total GPU count, using makespan as the metric. The workload `mooncake_trace_1000.jsonl` contains 1000 requests with mean input sequence length 9.8k, biased toward prefill, and arrival intensity is swept via `--arrival-speedup-ratio`.

Makespan (seconds):

<table width="100%">
  <thead>
    <tr>
      <th>Arrival intensity</th>
      <th>Static 1:3</th>
      <th>Static 1:1</th>
      <th>Static 3:1</th>
      <th>Dynamic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1×</td>
      <td>183.03</td>
      <td>182.41</td>
      <td>182.55</td>
      <td>181.84</td>
    </tr>
    <tr>
      <td>2×</td>
      <td>102.01</td>
      <td>94.27</td>
      <td>102.40</td>
      <td>94.18</td>
    </tr>
    <tr>
      <td>4×</td>
      <td>85.08</td>
      <td>65.16</td>
      <td>70.53</td>
      <td>58.47</td>
    </tr>
    <tr>
      <td>8×</td>
      <td>86.81</td>
      <td>62.57</td>
      <td>74.26</td>
      <td>47.89</td>
    </tr>
  </tbody>
</table>

Improvement of dynamic scheduling over static configurations at 8× intensity:

<table width="100%">
  <thead>
    <tr>
      <th>Comparison</th>
      <th>Makespan reduction</th>
      <th>GPU time saved</th>
      <th>Utilization gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>vs 1:3</td>
      <td>86.81 → 47.89s</td>
      <td>44.8%</td>
      <td>+81.3%</td>
    </tr>
    <tr>
      <td>vs 1:1</td>
      <td>62.57 → 47.89s</td>
      <td>23.5%</td>
      <td>+30.7%</td>
    </tr>
    <tr>
      <td>vs 3:1</td>
      <td>74.26 → 47.89s</td>
      <td>35.5%</td>
      <td>+55.1%</td>
    </tr>
  </tbody>
</table>

Differences are small under light load, while the advantage of dynamic scheduling grows monotonically with arrival intensity and exceeds the optimal static configuration for this workload.

### 3.3 Conditional Disaggregation

The aggregated configuration is the ideal upper bound without KV transfer. Conditional disaggregation retains the static skeleton and migrates part of the prefill to decode nodes based on policy, closer to the actual FlowKV form. Policies come from `ConditionalDisaggPolicyKind`, and all disaggregated configurations enable KV transfer (128KiB/token at 200GB/s).

Makespan (seconds):

<table width="100%">
  <thead>
    <tr>
      <th>Config</th>
      <th>1×</th>
      <th>2×</th>
      <th>4×</th>
      <th>8×</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>static no KV</td>
      <td>183.03</td>
      <td>102.01</td>
      <td>85.08</td>
      <td>86.81</td>
    </tr>
    <tr>
      <td>static with KV</td>
      <td>183.04</td>
      <td>102.11</td>
      <td>85.55</td>
      <td>89.00</td>
    </tr>
    <tr>
      <td>isl_bounding</td>
      <td>182.55</td>
      <td>103.58</td>
      <td>87.92</td>
      <td>87.39</td>
    </tr>
    <tr>
      <td>prefill_load</td>
      <td>181.48</td>
      <td>93.09</td>
      <td>55.88</td>
      <td>39.67</td>
    </tr>
    <tr>
      <td>isl_or_load</td>
      <td>181.48</td>
      <td>93.06</td>
      <td>55.43</td>
      <td>39.10</td>
    </tr>
  </tbody>
</table>

Bypass rate (determined by `prefill_worker_idx is None`):

<table width="100%">
  <thead>
    <tr>
      <th>Policy</th>
      <th>1×</th>
      <th>2×</th>
      <th>4×</th>
      <th>8×</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>isl_bounding</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
      <td>0%</td>
    </tr>
    <tr>
      <td>prefill_load</td>
      <td>88.2%</td>
      <td>87.2%</td>
      <td>87.7%</td>
      <td>90.2%</td>
    </tr>
    <tr>
      <td>isl_or_load</td>
      <td>88.2%</td>
      <td>87.3%</td>
      <td>88.2%</td>
      <td>88.1%</td>
    </tr>
  </tbody>
</table>

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

<table width="100%">
  <thead>
    <tr>
      <th>File</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>run_ablation.py</td>
      <td>Batch static and dynamic scheduling configurations</td>
    </tr>
    <tr>
      <td>run_trace_ablation.py</td>
      <td>Real-trace open-loop replay</td>
    </tr>
    <tr>
      <td>run_conditional_disagg_ablation.py</td>
      <td>Conditional disaggregation ablation</td>
    </tr>
    <tr>
      <td>verify_ablation.py</td>
      <td>Reproducibility checks</td>
    </tr>
    <tr>
      <td>trace_inspect.py</td>
      <td>Workload characterization</td>
    </tr>
    <tr>
      <td>plot_figures.py</td>
      <td>Figure generation</td>
    </tr>
  </tbody>
</table>

## 7 Source Changes

Conditional disaggregation replay support only touches the simulation engine `lib/mocker`, leaving the production runtime unchanged.

<table width="100%">
  <thead>
    <tr>
      <th>File</th>
      <th>Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>lib/mocker/src/replay/offline/components/engine.rs</td>
      <td>Expose prefill pool saturation signal</td>
    </tr>
    <tr>
      <td>lib/mocker/src/replay/offline/state.rs</td>
      <td>Add bypass state flag</td>
    </tr>
    <tr>
      <td>lib/mocker/src/replay/offline/disagg.rs</td>
      <td>Conditional disaggregation decision and local prefill routing</td>
    </tr>
    <tr>
      <td>lib/mocker/src/replay/offline/extensions/kv_router/composition_disagg.rs</td>
      <td>Replay config derivation and wiring</td>
    </tr>
    <tr>
      <td>lib/mocker/src/replay/offline/extensions/kv_events/mod.rs</td>
      <td>Keep disabled by default</td>
    </tr>
  </tbody>
</table>

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
