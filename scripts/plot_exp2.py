import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

rows = []
with open("results/exp2_occlusion.csv", "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            "Scene": r["Scene"],
            "Instances": int(r["Instances"]),
            "Method": r["Method"],
            "CullTime_Median_ms": float(r["CullTime_Median_ms"]),
            "EndToEndTime_Median_ms": float(r["EndToEndTime_Median_ms"]),
            "VisibleCount": int(r["VisibleCount"]),
            "FalsePositiveRate_pct": float(r["FalsePositiveRate_pct"]),
            "FalseNegativeRate_pct": float(r["FalseNegativeRate_pct"])
        })

scenes = sorted(list(set(r["Scene"] for r in rows)))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for s_idx, scene in enumerate(["City_Walls", "Forest_Pillars", "Swiss_Cheese_Grate"]):
    ax = axes[s_idx]
    s_rows = [r for r in rows if r["Scene"] == scene]
    instances = sorted(list(set(r["Instances"] for r in s_rows)))
    
    # False Negative (Popping) Rates across sample counts at 100k instances
    inst_target = 100000
    sub_rows = [r for r in s_rows if r["Instances"] == inst_target]
    
    methods = ["HZB_Depth_Reuse", "RayQuery_1Sample", "RayQuery_3Sample", "RayQuery_5Sample", "RayQuery_9Sample"]
    fn_rates = []
    labels = ["HZB (Reuse)", "RQ 1-Smp", "RQ 3-Smp", "RQ 5-Smp", "RQ 9-Smp"]
    colors = ['#1b9e77', '#d7191c', '#fdae61', '#2b5c8f', '#2b5c8f']
    
    for m in methods:
        matched = [r["FalseNegativeRate_pct"] for r in sub_rows if r["Method"] == m]
        fn_rates.append(matched[0] if matched else 0.0)
        
    x = np.arange(len(methods))
    bars = ax.bar(x, fn_rates, color=colors, width=0.55)
    
    for bar, val in zip(bars, fn_rates):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(f'{scene}\nFalse Negative (Visual Popping) Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('False Negative Rate (%)')
    ax.set_ylim(0, max(65, max(fn_rates) + 8))
    ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("results/exp2_occlusion_culling.png", dpi=300)
plt.close()

# Plot 2: Timing comparison across instance counts (Swiss Cheese Grate)
fig, ax = plt.subplots(figsize=(9, 5))
sc_rows = [r for r in rows if r["Scene"] == "Swiss_Cheese_Grate"]
insts = sorted(list(set(r["Instances"] for r in sc_rows)))

hzb_reuse = [next(r["CullTime_Median_ms"] for r in sc_rows if r["Instances"] == i and r["Method"] == "HZB_Depth_Reuse") for i in insts]
hzb_cold = [next(r["EndToEndTime_Median_ms"] for r in sc_rows if r["Instances"] == i and r["Method"] == "HZB_Cold_Depth") for i in insts]
rq1 = [next(r["CullTime_Median_ms"] for r in sc_rows if r["Instances"] == i and r["Method"] == "RayQuery_1Sample") for i in insts]
rq5 = [next(r["CullTime_Median_ms"] for r in sc_rows if r["Instances"] == i and r["Method"] == "RayQuery_5Sample") for i in insts]

ax.plot(insts, hzb_reuse, 'o-', color='#1b9e77', linewidth=2.5, label='HZB (Two-Pass Depth Reuse)')
ax.plot(insts, hzb_cold, 'o--', color='#1b9e77', linewidth=1.5, alpha=0.7, label='HZB (Cold Depth Prepass)')
ax.plot(insts, rq5, 's-', color='#2b5c8f', linewidth=2.5, label='Ray Query 5-Sample (<1% Popping)')
ax.plot(insts, rq1, '^:', color='#d7191c', linewidth=1.5, label='Ray Query 1-Sample (58% Popping!)')

ax.set_title('Occlusion Culling Time vs Instance Count (Adversarial Swiss Cheese Grate)')
ax.set_xlabel('Instance Count')
ax.set_ylabel('GPU Time (ms)')
ax.set_xticks(insts)
ax.set_xticklabels([f'{i//1000}k' for i in insts])
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

plt.tight_layout()
plt.savefig("results/exp2_occlusion_timing.png", dpi=300)
plt.close()

print("Experiment 2 plots generated in results/")
