#!/usr/bin/env python3
"""DynaSim ablation: static-PD (disagg) vs dynamic-PD (aggregated).

Holds the workload constant and sweeps the scheduling architecture:
  - static-PD  (disagg, fixed prefill:decode worker ratio)
  - dynamic-PD (aggregated, every worker does prefill + decode)

Runs inside WSL because dynamo._core is a Linux abi3 extension.
"""
import argparse
import json
import os
import subprocess
import sys
import time

VENV_PY = "/home/femto/dynamo-venv/bin/python"
WORKDIR = "/home/femto/dynamo"
REPORT_DIR = "/home/femto/ablation_reports"

ISL = 512
OSL = 128


def build_configs(scale):
    if scale == 8:
        return [
            ("static_1to3", "static", {"prefill": 2, "decode": 6}),
            ("static_1to1", "static", {"prefill": 4, "decode": 4}),
            ("static_3to1", "static", {"prefill": 6, "decode": 2}),
            ("dynamic", "dynamic", {"workers": 8}),
        ]
    if scale == 1024:
        return [
            ("static_1to3", "static", {"prefill": 256, "decode": 768}),
            ("static_1to1", "static", {"prefill": 512, "decode": 512}),
            ("static_3to1", "static", {"prefill": 768, "decode": 256}),
            ("dynamic", "dynamic", {"workers": 1024}),
        ]
    raise ValueError(scale)


def run_one(label, kind, params, concurrency, request_count, scale):
    cmd = [
        VENV_PY, "-m", "dynamo.replay",
        "--input-tokens", str(ISL),
        "--output-tokens", str(OSL),
        "--request-count", str(request_count),
        "--replay-concurrency", str(concurrency),
        "--arrival-seed", "42",
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
    report = os.path.join(REPORT_DIR, f"{scale}c_{label}.json")
    cmd += ["--report-json", report]

    env = dict(os.environ)
    env["HOME"] = "/home/femto"
    t0 = time.time()
    p = subprocess.run(cmd, cwd=WORKDIR, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        return {"label": label, "error": (p.stderr or p.stdout)[-2000:]}, wall
    with open(report) as f:
        rep = json.load(f)
    return rep, wall


METRIC_FIELDS = [
    ("mean_ttft_ms", "TTFT mean (ms)"),
    ("p99_ttft_ms", "TTFT p99 (ms)"),
    ("mean_itl_ms", "ITL mean (ms)"),
    ("p99_itl_ms", "ITL p99 (ms)"),
    ("mean_e2e_latency_ms", "E2E mean (ms)"),
    ("p99_e2e_latency_ms", "E2E p99 (ms)"),
    ("request_throughput_rps", "Req throughput (rps)"),
    ("output_throughput_tok_s", "Out tok/s"),
    ("total_throughput_tok_s", "Total tok/s"),
    ("duration_ms", "Duration (ms)"),
    ("gpu_hours", "GPU-hours"),
    ("prefill_worker_seconds", "Prefill worker-s"),
    ("decode_worker_seconds", "Decode worker-s"),
]


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, required=True, choices=(8, 1024))
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--request-count", type=int, required=True)
    args = ap.parse_args()

    configs = build_configs(args.scale)
    results = []
    for label, kind, params in configs:
        desc = (
            f"{params['prefill']}P+{params['decode']}D"
            if kind == "static"
            else f"{params['workers']} agg"
        )
        print(f"== running {label} ({desc}) ... ", flush=True)
        rep, wall = run_one(label, kind, params, args.concurrency,
                            args.request_count, args.scale)
        rep["_label"] = label
        rep["_desc"] = desc
        rep["_wall_s"] = wall
        if "error" in rep:
            print(f"   ERROR ({wall:.1f}s):\n{rep['error']}")
        else:
            print(f"   ok ({wall:.1f}s wall)")
        results.append(rep)

    print("\n" + "=" * 100)
    header = f"{'config':<24}" + "".join(f"{f[1]:>16}" for f in METRIC_FIELDS)
    print(header)
    for rep in results:
        label = f"{rep['_label']} ({rep['_desc']})"
        row = f"{label:<24}"
        for key, _ in METRIC_FIELDS:
            row += f"{fmt(rep.get(key)):>16}"
        print(row)
    print("=" * 100)
    print(f"ISL={ISL} OSL={OSL} concurrency={args.concurrency} "
          f"request_count={args.request_count}")


if __name__ == "__main__":
    main()
