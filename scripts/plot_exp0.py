import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

# 1. Throughput Plot
tp_data = []
with open("results/exp0_throughput.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tp_data.append({
            "Triangles": int(row["Triangles"]),
            "RayPattern": row["RayPattern"],
            "Backend": row["Backend"],
            "Rays": int(row["Rays"]),
            "Time_Median_ms": float(row["Time_Median_ms"]),
            "Throughput_MRays_per_sec": float(row["Throughput_MRays_per_sec"]),
            "Speedup": float(row["Speedup"])
        })

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Coherent
coh_rows = [r for r in tp_data if r["RayPattern"] == "Coherent"]
tris = sorted(list(set(r["Triangles"] for r in coh_rows)))
ra_tp_coh = [next((r["Throughput_MRays_per_sec"] for r in coh_rows if r["Triangles"] == t and r["Backend"] == "Hardware_RA"), 0.0) for t in tris]
sw_bvh4_coh = [next((r["Throughput_MRays_per_sec"] for r in coh_rows if r["Triangles"] == t and r["Backend"] == "Software_ALU_BVH4"), 0.0) for t in tris]
sw_lds_coh = [next((r["Throughput_MRays_per_sec"] for r in coh_rows if r["Triangles"] == t and r["Backend"] == "Software_ALU_LDS"), 0.0) for t in tris]
sw_scratch_coh = [next((r["Throughput_MRays_per_sec"] for r in coh_rows if r["Triangles"] == t and r["Backend"] == "Software_ALU_Scratch"), 0.0) for t in tris]

x = np.arange(len(tris))
width = 0.2

axes[0].bar(x - 1.5*width, ra_tp_coh, width, label='Hardware RA', color='#2b5c8f')
axes[0].bar(x - 0.5*width, sw_bvh4_coh, width, label='Software BVH4 (Ordered+LDS)', color='#1b9e77')
axes[0].bar(x + 0.5*width, sw_lds_coh, width, label='Software BVH2 (LDS)', color='#d95f02')
axes[0].bar(x + 1.5*width, sw_scratch_coh, width, label='Software BVH2 (Scratch)', color='#7570b3')
axes[0].set_ylabel('Throughput (MRays/sec)')
axes[0].set_title('Coherent Rays Throughput (1M Rays)')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{t//1000}k' if t < 1000000 else f'{t//1000000}M' for t in tris])
axes[0].set_xlabel('Scene Complexity (Triangles)')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].legend(fontsize=8)

# Incoherent
incoh_rows = [r for r in tp_data if r["RayPattern"] == "Incoherent"]
ra_tp_incoh = [next((r["Throughput_MRays_per_sec"] for r in incoh_rows if r["Triangles"] == t and r["Backend"] == "Hardware_RA"), 0.0) for t in tris]
sw_bvh4_incoh = [next((r["Throughput_MRays_per_sec"] for r in incoh_rows if r["Triangles"] == t and r["Backend"] == "Software_ALU_BVH4"), 0.0) for t in tris]
sw_lds_incoh = [next((r["Throughput_MRays_per_sec"] for r in incoh_rows if r["Triangles"] == t and r["Backend"] == "Software_ALU_LDS"), 0.0) for t in tris]
sw_scratch_incoh = [next((r["Throughput_MRays_per_sec"] for r in incoh_rows if r["Triangles"] == t and r["Backend"] == "Software_ALU_Scratch"), 0.0) for t in tris]

axes[1].bar(x - 1.5*width, ra_tp_incoh, width, label='Hardware RA', color='#2b5c8f')
axes[1].bar(x - 0.5*width, sw_bvh4_incoh, width, label='Software BVH4 (Ordered+LDS)', color='#1b9e77')
axes[1].bar(x + 0.5*width, sw_lds_incoh, width, label='Software BVH2 (LDS)', color='#d95f02')
axes[1].bar(x + 1.5*width, sw_scratch_incoh, width, label='Software BVH2 (Scratch)', color='#7570b3')
axes[1].set_ylabel('Throughput (MRays/sec)')
axes[1].set_title('Incoherent Rays Throughput (1M Rays)')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'{t//1000}k' if t < 1000000 else f'{t//1000000}M' for t in tris])
axes[1].set_xlabel('Scene Complexity (Triangles)')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/exp0_microbench_throughput.png", dpi=300)
plt.close()

# 2. BVH Build Plot
bvh_rows = []
with open("results/exp0_bvh_build.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        bvh_rows.append({
            "Triangles": int(row["Triangles"]),
            "Build_FastTrace_ms": float(row["Build_FastTrace_ms"]),
            "Build_FastBuild_ms": float(row["Build_FastBuild_ms"]),
            "Update_Refit_ms": float(row["Update_Refit_ms"]),
            "BLAS_Size_MB": float(row["BLAS_Size_MB"])
        })

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(bvh_rows))
width = 0.25

fast_trace = [r["Build_FastTrace_ms"] for r in bvh_rows]
fast_build = [r["Build_FastBuild_ms"] for r in bvh_rows]
update_refit = [r["Update_Refit_ms"] for r in bvh_rows]

ax.bar(x - width, fast_trace, width, label='Build (FastTrace)', color='#7570b3')
ax.bar(x, fast_build, width, label='Build (FastBuild)', color='#1b9e77')
ax.bar(x + width, update_refit, width, label='Update / Refit', color='#e7298a')

ax.set_ylabel('Time (ms)')
ax.set_title('BVH Build and Refit Time vs Triangle Count (RX 6700 XT)')
ax.set_xticks(x)
ax.set_xticklabels([f'{r["Triangles"]//1000}k' if r["Triangles"] < 1000000 else f'{r["Triangles"]//1000000}M' for r in bvh_rows])
ax.set_xlabel('Scene Complexity (Triangles)')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()
plt.savefig("results/exp0_bvh_build.png", dpi=300)
plt.close()

print("Experiment 0 plots successfully generated in results/")
