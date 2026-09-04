#!/usr/bin/env python3
"""Validate DynaSim ablation results: determinism, seed-robustness,
request-completeness, Little's law self-consistency, and throughput scaling."""
import json
import os
import subprocess

VENV_PY = "/home/femto/dynamo-venv/bin/python"
WORKDIR = "/home/femto/dynamo"
REPORT_DIR = "/home/femto/ablation_reports"
ISL = 512
OSL = 128

KEYS = [
    "mean_ttft_ms", "p99_ttft_ms", "mean_e2e_latency_ms", "p99_e2e_latency_ms",
    "mean_itl_ms", "p99_itl_ms", "request_throughput_rps",
    "output_throughput_tok_s", "duration_ms",
]


def run(kind, params, concurrency, request_count, seed, tag):
    cmd = [
        VENV_PY, "-m", "dynamo.replay",
        "--input-tokens", str(ISL), "--output-tokens", str(OSL),
        "--request-count", str(request_count),
        "--replay-concurrency", str(concurrency),
        "--arrival-seed", str(seed),
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
    report = f"/tmp/verify_{tag}.json"
    cmd += ["--report-json", report]
    env = dict(os.environ)
    env["HOME"] = "/home/femto"
    p = subprocess.run(cmd, cwd=WORKDIR, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        return None, (p.stderr or p.stdout)[-1500:]
    with open(report) as f:
        return json.load(f), None


def snap(rep):
    return {k: rep.get(k) for k in KEYS}


def little(rep, concurrency):
    rps = rep.get("request_throughput_rps") or 0
    mean_e2e_s = (rep.get("mean_e2e_latency_ms") or 0) / 1000.0
    return rps * mean_e2e_s


def main():
    print("=" * 90)
    print("V1 确定性（同 seed=42 跑两次，应 bit-exact 一致）")
    print("=" * 90)
    a, e1 = run("dynamic", {"workers": 8}, 256, 4000, 42, "det_a")
    b, e2 = run("dynamic", {"workers": 8}, 256, 4000, 42, "det_b")
    if e1 or e2:
        print("ERROR", e1, e2)
    else:
        same = all(a[k] == b[k] for k in KEYS)
        print(f"两次结果所有 {len(KEYS)} 个指标完全一致: {same}")
        if not same:
            for k in KEYS:
                if a[k] != b[k]:
                    print(f"  DIFF {k}: {a[k]} vs {b[k]}")

    print()
    print("=" * 90)
    print("V2 种子鲁棒性（8卡动态，seed 42/43/123，应统计上稳定）")
    print("=" * 90)
    seeds = [42, 43, 123]
    rows = []
    for s in seeds:
        r, e = run("dynamic", {"workers": 8}, 256, 4000, s, f"seed{s}")
        if e:
            print("ERROR", e)
            return
        rows.append((s, r))
    hdr = "seed  " + "".join(f"{k:>18}" for k in ["p99_ttft", "p99_e2e", "rps", "out_tok_s"])
    print(hdr)
    for s, r in rows:
        print(f"{s:<6}" + f"{r['p99_ttft_ms']:>18.1f}{r['p99_e2e_latency_ms']:>18.1f}"
              f"{r['request_throughput_rps']:>18.1f}{r['output_throughput_tok_s']:>18.1f}")
    rps_vals = [r["request_throughput_rps"] for _, r in rows]
    spread = (max(rps_vals) - min(rps_vals)) / (sum(rps_vals) / len(rps_vals)) * 100
    print(f"吞吐跨种子相对波动: {spread:.2f}%")

    print()
    print("=" * 90)
    print("V3 完整性 + Little's law 自洽（completed==requested；rps*mean_e2e <= 并发）")
    print("=" * 90)
    print(f"{'report':<20}{'completed':>10}{'requested':>10}{'rps':>10}{'mean_e2e':>11}"
          f"{'inflight':>10}{'concurrency':>13}{'OK':>5}")
    configs = {
        "8c_static_1to3": 256, "8c_static_1to1": 256, "8c_static_3to1": 256, "8c_dynamic": 256,
        "1024c_static_1to3": 32768, "1024c_static_1to1": 32768,
        "1024c_static_3to1": 32768, "1024c_dynamic": 32768,
    }
    for name, conc in configs.items():
        path = os.path.join(REPORT_DIR, f"{name}.json")
        if not os.path.exists(path):
            print(f"{name:<20} MISSING")
            continue
        rep = json.load(open(path))
        comp = rep.get("completed_requests")
        req = rep.get("num_requests")
        rps = rep.get("request_throughput_rps") or 0
        me = rep.get("mean_e2e_latency_ms") or 0
        inf = little(rep, conc)
        ok = "Y" if (comp == req and inf <= conc * 1.001) else "N"
        print(f"{name:<20}{comp:>10}{req:>10}{rps:>10.1f}{me:>11.1f}"
              f"{inf:>10.1f}{conc:>13}{ok:>5}")

    print()
    print("=" * 90)
    print("V4 稳态收敛（1024卡动态，请求数 100k -> 400k，吞吐应上升并趋稳）")
    print("=" * 90)
    for rc in [100000, 400000]:
        r, e = run("dynamic", {"workers": 1024}, 32768, rc, 42, f"scale{rc}")
        if e:
            print("ERROR", e)
            return
        inf = little(r, 32768)
        print(f"request_count={rc:<7} rps={r['request_throughput_rps']:.1f}  "
              f"mean_e2e={r['mean_e2e_latency_ms']:.1f}ms  "
              f"inflight={inf:.0f}/{32768}  p99_ttft={r['p99_ttft_ms']:.1f}ms")


if __name__ == "__main__":
    main()
