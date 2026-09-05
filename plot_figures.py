# -*- coding: utf-8 -*-
"""生成项目报告用的 matplotlib 图表（输出到 figures/）。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 注册中文字体：Droid Sans Fallback 只含 CJK、不含 ASCII，需配合 DejaVu 回退
FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/mnt/d/Documents/WorkSpace/python/i2/20260903_4/figures/"
os.makedirs(OUT, exist_ok=True)

# ---- 图 1：8 卡消融 TTFT mean 对比 ----
configs = ["Static 1:3\n(2P+6D)", "Static 1:1\n(4P+4D)", "Static 3:1\n(6P+2D)", "Dynamic\n(8 agg)"]
ttft_mean = [523.8, 184.9, 184.4, 158.9]
colors = ["#c0504d", "#5b9bd5", "#5b9bd5", "#70ad47"]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(configs, ttft_mean, color=colors, width=0.55)
for b, v in zip(bars, ttft_mean):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("TTFT mean (ms)")
ax.set_title("TTFT Mean (8 GPUs)")
ax.set_ylim(0, 600)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + "fig_ttft.png", dpi=150)
plt.close(fig)

# ---- 图 2：真实 trace makespan 对比 ----
import numpy as np
speeds = ["1×", "2×", "4×", "8×"]
s1 = [183.03, 102.01, 85.08, 86.81]
s2 = [182.41, 94.27, 65.16, 62.57]
s3 = [182.55, 102.40, 70.53, 74.26]
dy = [181.84, 94.18, 58.47, 47.89]
x = np.arange(len(speeds))
w = 0.2
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x - 1.5 * w, s1, w, label="Static 1:3")
ax.bar(x - 0.5 * w, s2, w, label="Static 1:1")
ax.bar(x + 0.5 * w, s3, w, label="Static 3:1")
ax.bar(x + 1.5 * w, dy, w, label="Dynamic", color="#70ad47")
ax.set_xticks(x)
ax.set_xticklabels(speeds)
ax.set_xlabel("Arrival intensity")
ax.set_ylabel("makespan (s)")
ax.set_title("Makespan (Real Trace)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + "fig_makespan.png", dpi=150)
plt.close(fig)

# ---- 图 3：条件分离 makespan 对比 ----
static_nokv = [183.03, 102.01, 85.08, 86.81]
static_kv = [183.04, 102.11, 85.55, 89.00]
isl_bound = [182.55, 103.58, 87.92, 87.39]
prefill_load = [181.48, 93.09, 55.88, 39.67]
isl_or_load = [181.48, 93.06, 55.43, 39.10]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(speeds, static_nokv, "o-", label="static (no KV)", color="#7f7f7f")
ax.plot(speeds, static_kv, "s--", label="static (with KV)", color="#c0504d")
ax.plot(speeds, isl_bound, "^:", label="isl_bounding", color="#5b9bd5")
ax.plot(speeds, prefill_load, "D-", label="prefill_load", color="#70ad47")
ax.plot(speeds, isl_or_load, "*-", label="isl_or_load", color="#ed7d31")
ax.set_xlabel("Arrival intensity")
ax.set_ylabel("makespan (s)")
ax.set_title("Makespan (Conditional Disaggregation)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + "fig_conditional.png", dpi=150)
plt.close(fig)

print("saved:", os.listdir(OUT))
