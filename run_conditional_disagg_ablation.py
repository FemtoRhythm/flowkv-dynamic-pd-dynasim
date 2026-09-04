#!/usr/bin/env python3
"""Conditional-disagg ablation on a real trace.

Compares static PD (round-robin) against KV-router conditional-disagg
(`isl_bounding` / `prefill_load` / `isl_or_load`) on the same trace, worker
split, and KV-transfer parameters, so the only variable is the bypass policy.

KV-transfer overhead is enabled on every disagg config via
`--prefill-engine-args` (kv_bytes_per_token + kv_transfer_bandwidth); a
bypassed request skips that cross-worker transfer and the prefill-pool queue.

Runs inside WSL (dynamo._core is a Linux abi3 extension).
"""
import argparse
import json
import os
import subprocess
import time

VENV_PY = "/home/femto/dynamo-venv/bin/python"
WORKDIR = "/home/femto/dynamo"
REPORT_DIR = "/home/femto/conditional_disagg_reports"

TRACE = "/home/femto/dynamo/lib/bench/testdata/mooncake_trace_1000.jsonl"
TRACE_FORMAT = "mooncake"
TRACE_BLOCK_SIZE = 512  # tokens per hash_id in this trace (default 512)

# KV-transfer parameters applied to every disagg config. `kv_bytes_per_token`
# is the fp8 KV-cache size for a ~31B model (128 KiB/token); bandwidth is a
# conservative NIXL/InfiniBand figure.
KV_BYTES_PER_TOKEN = 131072
KV_TRANSFER_BANDWIDTH = 200.0

POLICIES = ("isl_bounding", "prefill_load", "isl_or_load")


def prefill_args(kv_transfer=True):
    d = {"worker_type": "prefill"}
    if kv_transfer:
        d["kv_bytes_per_token"] = KV_BYTES_PER_TOKEN
        d["kv_transfer_bandwidth"] = KV_TRANSFER_BANDWIDTH
    return json.dumps(d)


def router_config(policy):
    return json.dumps({
        "conditional_disagg_enabled": True,
        "conditional_disagg_policy": policy,
    })


def build_configs(prefill, decode):
    """Static round-robin baselines plus the three conditional-disagg policies."""
    return [
        ("static_nokv", "static", {"prefill": prefill, "decode": decode, "kv": False}),
        ("static_kv", "static", {"prefill": prefill, "decode": decode, "kv": True}),
        ("cond_isl", "kv_router", {"prefill": prefill, "decode": decode, "policy": "isl_bounding"}),
        ("cond_load", "kv_router", {"prefill": prefill, "decode": decode, "policy": "prefill_load"}),
        ("cond_or", "kv_router", {"prefill": prefill, "decode": decode, "policy": "isl_or_load"}),
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
            "--prefill-engine-args", prefill_args(params["kv"]),
            "--decode-engine-args", json.dumps({"worker_type": "decode"}),
            "--num-prefill-workers", str(params["prefill"]),
            "--num-decode-workers", str(params["decode"]),
        ]
    else:  # kv_router
        cmd += [
            "--router-mode", "kv_router",
            "--router-config", router_config(params["policy"]),
            "--prefill-engine-args", prefill_args(True),
            "--decode-engine-args", json.dumps({"worker_type": "decode"}),
            "--num-prefill-workers", str(params["prefill"]),
            "--num-decode-workers", str(params["decode"]),
        ]
    os.makedirs(REPORT_DIR, exist_ok=True)
    tag = label if speedup == 1.0 else f"{label}_s{speedup:g}"
    report = os.path.join(REPORT_DIR, f"{tag}.json")
    report_jsonl = os.path.join(REPORT_DIR, f"{tag}.jsonl")
    cmd += ["--report-json", report, "--report-jsonl", report_jsonl]

    env = dict(os.environ)
    env["HOME"] = "/home/femto"
    t0 = time.time()
    p = subprocess.run(cmd, cwd=WORKDIR, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        return {"label": label, "error": (p.stderr or p.stdout)[-4000:]}, wall
    with open(report) as f:
        rep = json.load(f)
    rep["_bypass_count"], rep["_request_count"] = count_bypasses(report_jsonl)
    return rep, wall


def count_bypasses(jsonl_path):
    """Count requests whose prefill_worker_idx is None in disagg mode.

    Returns (bypass_count, total_count). Bypass is only meaningful for
    conditional-disagg runs; static runs will report bypass_count=0.
    """
    bypass = 0
    total = 0
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                total += 1
                if rec.get("prefill_worker_idx") is None:
                    bypass += 1
    except FileNotFoundError:
        return 0, 0
    return bypass, total


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", type=int, default=2)
    ap.add_argument("--decode", type=int, default=6)
    ap.add_argument("--speedups", type=str, default="1",
                    help="comma-separated arrival-speedup ratios, e.g. 1,2,4")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated config labels to run (default: all)")
    args = ap.parse_args()

    speedups = [float(x) for x in args.speedups.split(",")]
    configs = build_configs(args.prefill, args.decode)
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c[0] in wanted]

    for speedup in speedups:
        results = []
        for label, kind, params in configs:
            desc = (
                f"{params['prefill']}P+{params['decode']}D"
                + (" kv" if kind == "static" and params["kv"] else " nokv")
                if kind == "static"
                else f"{params['prefill']}P+{params['decode']}D {params['policy']}"
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

        print_report(results, speedup, args.prefill + args.decode)

    print(f"\ntrace={TRACE}  workers={args.prefill}+{args.decode}  speedups={speedups}")
    print(f"kv_bytes_per_token={KV_BYTES_PER_TOKEN}  bandwidth={KV_TRANSFER_BANDWIDTH} GB/s")


def print_report(results, speedup, workers):
    fields = [
        ("duration_ms", "Duration(s)"),
        ("gpu_hours", "GPU-hours"),
        ("completed_requests", "Completed"),
        ("output_throughput_tok_s", "Out tok/s"),
        ("request_throughput_rps", "Req/s"),
        ("mean_ttft_ms", "TTFT(ms)"),
        ("mean_itl_ms", "ITL(ms)"),
        ("mean_e2e_latency_ms", "E2E(ms)"),
    ]
    print("\n" + "=" * 116)
    print(f"[speedup {speedup:g}]  {'config':<28}" + "".join(f"{f[1]:>14}" for f in fields) + f"{'Bypass%':>14}")
    for rep in results:
        row = f"{rep['_label']} ({rep['_desc']}):"[:28].ljust(28)
        for key, _ in fields:
            v = rep.get(key)
            if key == "duration_ms":
                v = v / 1000.0 if v is not None else None
            row += f"{fmt(v):>14}"
        b = rep.get("_bypass_count")
        n = rep.get("_request_count") or 0
        row += f"{fmt((b / n * 100.0) if n else None):>14}"
        print(row)
    print("=" * 116)

    # Compare conditional-disagg configs against the KV-transfer static baseline.
    base = next((r for r in results if r["_label"] == "static_kv"), None)
    if base and "error" not in base:
        t_base = base["duration_ms"]
        print("== vs static_kv baseline (KV transfer on) ==")
        for rep in results:
            if rep["_label"] in ("static_kv", "static_nokv") or "error" in rep:
                continue
            t = rep["duration_ms"]
            if t_base > 0:
                makespan_cut = (t_base - t) / t_base * 100.0
                gpu_saved = (t_base - t) / 1000.0 * workers / 3600.0
                ttft = rep.get("mean_ttft_ms")
                ttft_base = base.get("mean_ttft_ms")
                ttft_delta = ""
                if ttft is not None and ttft_base is not None:
                    ttft_delta = f"  TTFT {ttft_base:.1f}->{ttft:.1f}ms"
                print(
                    f"  {rep['_desc']:<24} duration {t/1000.0:8.2f}s  "
                    f"makespan -{makespan_cut:5.1f}%  GPU-h saved {gpu_saved:6.3f}{ttft_delta}"
                )
    print()


if __name__ == "__main__":
    main()
