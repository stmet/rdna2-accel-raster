import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

data = []
with open("results/expB_shadows.csv", "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        data.append({
            "Triangles": int(r["Triangles"]),
            "LightCount": int(r["LightCount"]),
            "RasterTotalMs": float(r["RasterTotalMs"]),
            "RayQueryShadowsMs": float(r["RayQueryShadowsMs"]),
            "Speedup": float(r["SpeedupRQvsRaster"])
        })

tri_counts = sorted(list(set(d["Triangles"] for d in data)))
lights = sorted(list(set(d["LightCount"] for d in data)))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Frame Time vs Light Count (for 500k and 1M triangles)
colors = {100000: '#2b5c8f', 500000: '#1b9e77', 1000000: '#d95f02', 2500000: '#7570b3'}
for t in [100000, 500000, 1000000]:
    t_data = [d for d in data if d["Triangles"] == t]
    t_label = f'{t//1000}k tris'
    rast_times = [d["RasterTotalMs"] for d in t_data]
    rq_times = [d["RayQueryShadowsMs"] for d in t_data]
    l_counts = [d["LightCount"] for d in t_data]
    
    axes[0].plot(l_counts, rast_times, 'o--', label=f'Raster ({t_label})', color=colors[t], alpha=0.7)
    axes[0].plot(l_counts, rq_times, 's-', label=f'Ray Query ({t_label})', color=colors[t], linewidth=2)

axes[0].set_xscale('log', base=2)
axes[0].set_yscale('log')
axes[0].set_xlabel("Number of Shadow Casters")
axes[0].set_ylabel("Shadow Execution Time (ms, log scale)")
axes[0].set_title("Shadows: Raster Shadow Maps vs Ray Query Shadows")
axes[0].set_xticks(lights)
axes[0].set_xticklabels([str(l) for l in lights])
axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
axes[0].legend(fontsize=8, loc='upper left')

# Subplot 2: Speedup Heatmap (Lights vs Triangles)
speedup_matrix = np.zeros((len(tri_counts), len(lights)))
for i, t in enumerate(tri_counts):
    for j, l in enumerate(lights):
        match = next(d for d in data if d["Triangles"] == t and d["LightCount"] == l)
        speedup_matrix[i, j] = match["Speedup"]

im = axes[1].imshow(speedup_matrix, cmap='viridis', norm=plt.matplotlib.colors.LogNorm(vmin=0.7, vmax=135.0))
cbar = plt.colorbar(im, ax=axes[1])
cbar.set_label("Speedup: RQ Shadows vs Raster (x-fold)", rotation=270, labelpad=15)

axes[1].set_xticks(range(len(lights)))
axes[1].set_xticklabels([str(l) for l in lights])
axes[1].set_yticks(range(len(tri_counts)))
axes[1].set_yticklabels([f'{t//1000}k' if t < 1000000 else f'{t/1000000:.1f}M' for t in tri_counts])
axes[1].set_xlabel("Number of Shadow-Casting Lights")
axes[1].set_ylabel("Scene Triangle Count")
axes[1].set_title("Ray Query Speedup Heatmap (Crossover Frontier)")

for i in range(len(tri_counts)):
    for j in range(len(lights)):
        val = speedup_matrix[i, j]
        color = 'black' if val > 20 else 'white'
        axes[1].text(j, i, f'{val:.1f}x', ha='center', va='center', color=color, fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("results/expB_shadows.png", dpi=300)
plt.close()
print("Saved results/expB_shadows.png")
