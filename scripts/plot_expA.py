import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

rows = []
with open("results/expA_async.csv", "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Frame Time Breakdown / Total Wall Time
labels = ["Serial (Gfx)", "Async (Depth Overlap)", "Async (G-Buffer Overlap)", "Async (BVH Refit)"]
totals = [float(r["TotalMs"]) for r in rows]
p5s = [float(r["p5Ms"]) for r in rows]
p95s = [float(r["p95Ms"]) for r in rows]
yerr = [
    [totals[i] - p5s[i] for i in range(len(totals))],
    [p95s[i] - totals[i] for i in range(len(totals))]
]

colors = ['#2b5c8f', '#1b9e77', '#d95f02', '#7570b3']
bars = axes[0].bar(labels, totals, yerr=yerr, capsize=5, color=colors, alpha=0.85, edgecolor='black')
axes[0].set_ylabel("Total Frame Time (ms)")
axes[0].set_title("Async Compute Overlap: Total Frame Time (RX 6700 XT)")
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

for bar, total, r in zip(bars, totals, rows):
    fps = float(r["FPS"])
    axes[0].text(bar.get_x() + bar.get_width()/2.0, total + 0.08, f'{total:.2f} ms\n({fps:.0f} FPS)', ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[0].set_ylim(0, max(totals) * 1.3)

# Plot 2: Overlap Efficiency & Contention
eff_labels = ["Depth Overlap\n(Low Texture)", "G-Buffer Overlap\n(Texture-Heavy)", "Dynamic Refit\n(L2/Memory Heavy)"]
efficiencies = [float(rows[1]["EfficiencyPct"]), float(rows[2]["EfficiencyPct"]), float(rows[3]["EfficiencyPct"])]
eff_colors = ['#1b9e77' if e > 0 else '#e41a1c' for e in efficiencies]

bars2 = axes[1].bar(eff_labels, efficiencies, color=eff_colors, alpha=0.85, edgecolor='black', width=0.5)
axes[1].axhline(0, color='black', linewidth=1)
axes[1].set_ylabel("Overlap Efficiency (%)")
axes[1].set_title("Overlap Efficiency: $\\eta = (T_{serial} - T_{overlapped}) / T_{compute}$")
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

for bar, eff in zip(bars2, efficiencies):
    ypos = eff + 3 if eff >= 0 else eff - 8
    axes[1].text(bar.get_x() + bar.get_width()/2.0, ypos, f'{eff:.1f}%', ha='center', va='bottom' if eff >= 0 else 'top', fontsize=10, fontweight='bold')

axes[1].set_ylim(min(-20, min(efficiencies) - 15), 110)

plt.tight_layout()
plt.savefig("results/expA_async.png", dpi=300)
plt.close()
print("Saved results/expA_async.png")
