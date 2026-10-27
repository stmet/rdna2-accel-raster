import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

rows = []
with open("results/exp3_spatial.csv", "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Total Frame Time vs Light Count (Radius = 8.0)
ax = axes[0]
sweep1_rows = [r for r in rows if r["Radius"] == "8.0" and r["Method"] in ["LightCentric_Clustered", "Tiled_ForwardPlus", "RayQuery_Spatial"]]
lights = sorted(list(set(int(r["LightCount"]) for r in sweep1_rows)))

lc_totals = [float(next(r["Total_Median_ms"] for r in sweep1_rows if int(r["LightCount"]) == l and r["Method"] == "LightCentric_Clustered")) for l in lights]
tiled_totals = [float(next(r["Total_Median_ms"] for r in sweep1_rows if int(r["LightCount"]) == l and r["Method"] == "Tiled_ForwardPlus")) for l in lights]
rq_totals = [float(next(r["Total_Median_ms"] for r in sweep1_rows if int(r["LightCount"]) == l and r["Method"] == "RayQuery_Spatial")) for l in lights]

x = np.arange(len(lights))
ax.plot(x, tiled_totals, 'd--', color='#d95f02', linewidth=2.0, label='Tiled (Forward+)')
ax.plot(x, lc_totals, 'o-', color='#1b9e77', linewidth=2.5, label='Light-Centric Clustered (Audited)')
ax.plot(x, rq_totals, 's-', color='#2b5c8f', linewidth=2.5, label='Hardware Ray Query BVH')

ax.set_title('Spatial Lighting Lookup: Total GPU Time (1080p, RX 6700 XT)')
ax.set_xlabel('Point Light Count')
ax.set_ylabel('Total Execution Time (ms)')
ax.set_xticks(x)
ax.set_xticklabels([f'{l//1000}k' if l >= 1000 else str(l) for l in lights])
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# Plot 2: K-Dependence O(log N + k)
ax2 = axes[1]
k_rows = [r for r in rows if "K_Sweep" in r["Method"]]
if k_rows:
    radii = [4.0, 8.0, 16.0]
    c_times = [float(next(r["Shading_Median_ms"] for r in k_rows if float(r["Radius"]) == rad and "Clustered" in r["Method"])) for rad in radii]
    rq_times = [float(next(r["Shading_Median_ms"] for r in k_rows if float(r["Radius"]) == rad and "RayQuery" in r["Method"])) for rad in radii]
    avg_k = [float(next(r["AvgK"] for r in k_rows if float(r["Radius"]) == rad and "RayQuery" in r["Method"])) for rad in radii]
    
    x_k = np.arange(len(radii))
    w = 0.35
    ax2.bar(x_k - w/2, c_times, w, label='Clustered Shading', color='#1b9e77')
    ax2.bar(x_k + w/2, rq_times, w, label='Ray Query Shading (O(log N + k))', color='#2b5c8f')
    
    for bar, val in zip(ax2.patches, c_times + rq_times):
        ax2.text(bar.get_x() + bar.get_width()/2.0, val + 0.01, f'{val:.3f}ms', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_title('Shading Scaling with Overlapping Lights k (N=5,000 Lights)')
    ax2.set_xlabel('Light Radius (Average Overlapping Lights k)')
    ax2.set_ylabel('Shading Time (ms)')
    ax2.set_xticks(x_k)
    ax2.set_xticklabels([f'R={r}\n(k~{k:.0f})' for r, k in zip(radii, avg_k)])
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.legend()

plt.tight_layout()
plt.savefig("results/exp3_spatial_lighting.png", dpi=300)
plt.close()

print("Experiment 3 plots generated in results/")
