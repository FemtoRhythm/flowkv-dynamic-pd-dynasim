#!/usr/bin/env python3
"""DynaSim real-trace ablation: static-PD vs dynamic-PD utilization.

Replays a real open-loop trace (arrival timestamps + real length distribution)
through static PD (fixed prefill:decode partition) and dynamic PD (aggregated),
then derives GPU-utilization / makespan / GPU-hours-saved comparisons.

Runs inside WSL (dynamo._core is a Linux abi3 extension).
"""
import argparse
import json
import os
import subprocess
import time

VENV_PY = "/home/femto/dynamo-venv/bin/python"
WORKDIR = "/home/femto/dynamo"
REPORT_DIR = "/home/femto/trace_ablation_reports"

TRACE = "/home/femto/dynamo/lib/bench/testdata/mooncake_trace_1000.jsonl"
TRACE_FORMAT = "mooncake"
TRACE_BLOCK_SIZE = 512  # tokens per hash_id in this trace (default 512)


def build_configs(workers):
    """Static partitions summing to `workers`, plus dynamic (all aggregated)."""
    q = workers // 4
    return [
        ("static_1to3", "static", {"prefill": q, "decode": 3 * q}),
        ("static_1to1", "static", {"prefill": 2 * q, "decode": 2 * q}),
        ("static_3to1", "static", {"prefill": 3 * q, "decode": q}),
        ("dynamic", "dynamic", {"workers": workers}),
    ]


def run_one(label, kind, params, speedup):
    cmd = [
        VENV_PY, "-m", "dynamo.replay",
        TRACE,
        "--trace-format", TRACE_FORMAT,
        "--trace-block-size", str(TRACE_BLOCK_SIZE),
        "--arrival-speedup-ratio", str(speedup),
    ]
    if kind == "static":
        cmd += [
            "--prefill-engine-args", json.dumps({"worker_type": "prefill"}),
            "--decode-engine-args", json.dumps({"worker_type": "decode"}),
            "--num-prefill-workers", str(params["prefill"]),
            "--num-decode-workers", str(params["decode"]),
        ]
    else:
        cmd += [
            "--extra-engine-args", json.dumps({"worker_type": "aggregated"}),
            "--num-workers", str(params["workers"]),
        ]
    os.makedirs(REPORT_DIR, exist_ok=True)
    tag = label if speedup == 1.0 else f"{label}_s{speedup:g}"
    report = os.path.join(REPORT_DIR, f"{tag}.json")
    cmd += ["--report-json", report]

    env = dict(os.environ)
    env["HOME"] = "/home/femto"
    t0 = time.time()
    p = subprocess.run(cmd, cwd=WORKDIR, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        return {"label": label, "error": (p.stderr or p.stdout)[-3000:]}, wall
    with open(report) as f:
        rep = json.load(f)
    return rep, wall


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--speedups", type=str, default="1",
                    help="comma-separated arrival-speedup ratios, e.g. 1,2,4,8")
    args = ap.parse_args()

    speedups = [float(x) for x in args.speedups.split(",")]
    configs = build_configs(args.workers)

    for speedup in speedups:
        results = []
        for label, kind, params in configs:
            desc = (
                f"{params['prefill']}P+{params['decode']}D"
                if kind == "static"
                else f"{params['workers']} agg"
            )
            print(f"== [speedup {speedup:g}] {label} ({desc}) ... ", flush=True)
            rep, wall = run_one(label, kind, params, speedup)
            rep["_label"] = label
            rep["_desc"] = desc
            rep["_wall_s"] = wall
            if "error" in rep:
                print(f"   ERROR ({wall:.1f}s):\n{rep['error']}")
            else:
                print(f"   ok ({wall:.1f}s wall)")
            results.append(rep)

        print_report(results, speedup, args.workers)

    print(f"\ntrace={TRACE}  workers={args.workers}  speedups={speedups}")


def print_report(results, speedup, workers):
    fields = [
        ("duration_ms", "Duration(s)"),
        ("gpu_hours", "GPU-hours"),
        ("completed_requests", "Completed"),
        ("total_throughput_tok_s", "Tot tok/s"),
        ("output_throughput_tok_s", "Out tok/s"),
        ("request_throughput_rps", "Req/s"),
        ("mean_ttft_ms", "TTFT(ms)"),
        ("mean_e2e_latency_ms", "E2E(ms)"),
    ]
    print("\n" + "=" * 110)
    print(f"[speedup {speedup:g}]  {'config':<20}" + "".join(f"{f[1]:>14}" for f in fields))
    for rep in results:
        row = f"{rep['_label']} ({rep['_desc']}):"[:20].ljust(20)
        for key, _ in fields:
            v = rep.get(key)
            if key == "duration_ms":
                v = v / 1000.0 if v is not None else None
            row += f"{fmt(v):>14}"
        print(row)
    print("=" * 110)

    dyn = next((r for r in results if r["_label"] == "dynamic"), None)
    print("== GPU-utilization derivation (dynamic vs static, same trace) ==")
    if dyn and "error" not in dyn:
        t_dyn = dyn["duration_ms"]
        print(f"  dynamic duration     = {t_dyn/1000.0:.2f}s (baseline 100% util)")
        for rep in results:
            if rep["_label"] == "dynamic" or "error" in rep:
                continue
            t_static = rep["duration_ms"]
            if t_dyn > 0:
                speedup_v = t_static / t_dyn
                util_gain = (speedup_v - 1.0) * 100.0
                makespan_cut = (t_static - t_dyn) / t_static * 100.0
                gpu_saved = (t_static - t_dyn) / 1000.0 * workers / 3600.0
                print(
                    f"  vs {rep['_desc']:<12} duration {t_static/1000.0:8.2f}s  "
                    f"makespan -{makespan_cut:5.1f}%  "
                    f"util-gain +{util_gain:5.1f}%  "
                    f"GPU-h saved {gpu_saved:6.3f}"
                )
    print()


if __name__ == "__main__":
    main()
