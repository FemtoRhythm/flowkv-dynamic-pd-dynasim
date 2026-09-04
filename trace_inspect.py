#!/usr/bin/env python3
"""Inspect available real traces: timestamps + length distributions."""
import json
import sys

PATHS = [
    "/home/femto/dynamo/lib/bench/testdata/mooncake_trace_1000.jsonl",
    "/home/femto/dynamo/recipes/kimi-k2.6/perf/traces/8k_1k_70kv_chat_new_noschedule.jsonl",
    "/home/femto/dynamo/recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule.jsonl",
    "/home/femto/dynamo/recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl",
]


def inspect(path):
    ts, il, ol, keys = set(), [], [], set()
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keys |= set(r.keys())
            n += 1
            if "timestamp" in r:
                ts.add(r["timestamp"])
            if "input_length" in r:
                il.append(r["input_length"])
            if "output_length" in r:
                ol.append(r["output_length"])
    print(f"\n=== {path} ===")
    print(f"  lines={n}")
    print(f"  keys={sorted(keys)}")
    if ts:
        s = sorted(ts)
        print(f"  timestamp: unique={len(ts)} min={s[0]} max={s[-1]} first5={s[:5]}")
    else:
        print("  timestamp: NONE")
    if il:
        print(f"  input_length:  min={min(il)} mean={sum(il)//len(il)} max={max(il)}")
    if ol:
        print(f"  output_length: min={min(ol)} mean={sum(ol)//len(ol)} max={max(ol)}")


for p in PATHS:
    try:
        inspect(p)
    except FileNotFoundError:
        print(f"\n=== {p} ===\n  MISSING")
