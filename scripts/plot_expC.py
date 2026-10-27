import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

data = []
with open("results/expC_reflections.csv", "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        data.append({
            "Triangles": int(r["Triangles"]),
            "Coverage": float(r["ScreenCoveragePct"]),
            "RasterTotalMs": float(r["RasterTotalMs"]),
            "RayQueryMs": float(r["RayQueryReflectionMs"]),
            "Speedup": float(r["SpeedupRQvsRaster"])
        })

tri_counts = sorted(list(set(d["Triangles"] for d in data)))
coverages = sorted(list(set(d["Coverage"] for d in data)))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Frame Time vs Screen Coverage %
colors = {100000: '#2b5c8f', 500000: '#1b9e77', 1000000: '#d95f02', 2500000: '#7570b3'}
for t in tri_counts:
    t_data = [d for d in data if d["Triangles"] == t]
    t_label = f'{t//1000}k tris' if t < 1000000 else f'{t/1000000:.1f}M tris'
    rast_times = [d["RasterTotalMs"] for d in t_data]
    rq_times = [d["RayQueryMs"] for d in t_data]
    cov_vals = [d["Coverage"] for d in t_data]
    
    axes[0].plot(cov_vals, rast_times, 'o--', label=f'Raster Pass ({t_label})', color=colors[t], alpha=0.6)
    axes[0].plot(cov_vals, rq_times, 's-', label=f'Ray Query ({t_label})', color=colors[t], linewidth=2)

axes[0].set_xlabel("Reflective Screen Area Coverage (%)")
axes[0].set_ylabel("Reflection Pass Time (ms)")
axes[0].set_title("Planar Reflections: 2nd Raster Pass vs Ray Query")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(fontsize=8, loc='upper left')

# Subplot 2: Speedup vs Screen Coverage
for t in tri_counts:
    t_data = [d for d in data if d["Triangles"] == t]
    t_label = f'{t//1000}k tris' if t < 1000000 else f'{t/1000000:.1f}M tris'
    speedups = [d["Speedup"] for d in t_data]
    cov_vals = [d["Coverage"] for d in t_data]
    axes[1].plot(cov_vals, speedups, 'o-', label=t_label, color=colors[t], linewidth=2)

axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Parity (1.0x)')
axes[1].set_yscale('log')
axes[1].set_xlabel("Reflective Screen Area Coverage (%)")
axes[1].set_ylabel("Speedup (Ray Query vs Raster, log scale)")
axes[1].set_title("Ray Query Reflection Speedup vs Screen Coverage")
axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
axes[1].legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig("results/expC_reflections.png", dpi=300)
plt.close()
print("Saved results/expC_reflections.png")
