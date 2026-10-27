import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

rows = []
with open("results/expE_fullframe.csv", "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

resolutions = ["1080p", "1440p"]
configs = [
    ("Raster_Only_Baseline_Static", "Raster Baseline\n(Static)", "#2b5c8f"),
    ("RA_Assisted_Serial_Static", "RA Serial\n(Static)", "#1b9e77"),
    ("RA_Assisted_Async_Static", "RA Async Overlap\n(Static)", "#33a02c"),
    ("RA_Assisted_Serial_Dynamic", "RA Serial\n(Dynamic)", "#d95f02"),
    ("RA_Assisted_Async_Dynamic", "RA Async Overlap\n(Dynamic)", "#e7298a")
]

for idx, res in enumerate(resolutions):
    res_rows = [r for r in rows if r["Resolution"] == res]
    
    cfg_labels = []
    times = []
    fps_vals = []
    bar_colors = []
    
    for cfg_key, label, col in configs:
        match = next(r for r in res_rows if r["Configuration"] == cfg_key)
        cfg_labels.append(label)
        times.append(float(match["TotalFrameMs"]))
        fps_vals.append(float(match["FPS"]))
        bar_colors.append(col)
        
    x = np.arange(len(cfg_labels))
    width = 0.55
    
    ax = axes[idx]
    bars = ax.bar(x, times, width, color=bar_colors, alpha=0.88, edgecolor='black')
    ax.set_ylabel("Total Frame Time (ms)", fontweight='bold')
    ax.set_title(f"Full-Frame Integration: {res} (RX 6700 XT)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    baseline_time = times[0]
    for b, t, fps in zip(bars, times, fps_vals):
        pct_change = ((baseline_time - t) / baseline_time) * 100.0
        label_text = f'{t:.2f} ms\n({fps:.0f} FPS)'
        if t < baseline_time:
            label_text += f'\n(+{pct_change:.0f}%)'
        ax.text(b.get_x() + b.get_width()/2.0, t + 0.15, label_text, ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    ax.set_ylim(0, max(times) * 1.35)

plt.tight_layout()
plt.savefig("results/expE_fullframe.png", dpi=300)
plt.close()
print("Saved results/expE_fullframe.png")
