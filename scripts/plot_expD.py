import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

rows = []
with open("results/expD_screenspace.csv", "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

effects = ["AmbientOcclusion", "Reflections", "ContactShadows"]
titles = [
    "Ambient Occlusion: SSAO vs RQAO",
    "Reflections: SSR vs Ray Query",
    "Contact Shadows: Screen-Space vs RQ"
]

for idx, eff in enumerate(effects):
    eff_rows = [r for r in rows if r["Effect"] == eff]
    techs = [f'{r["Technique"]}\n({r["SPP_or_Steps"]})' for r in eff_rows]
    times = [float(r["TimeMs"]) for r in eff_rows]
    psnrs = [min(float(r["PSNR_vs_Ref_dB"]), 60.0) for r in eff_rows] # Cap visual display for 100dB exact hits
    
    x = np.arange(len(techs))
    width = 0.4
    
    color_bar = '#2b5c8f'
    color_line = '#d95f02'
    
    ax1 = axes[idx]
    bars = ax1.bar(x, times, width, color=color_bar, alpha=0.85, edgecolor='black', label='Time (ms)')
    ax1.set_ylabel("Execution Time (ms)", color=color_bar, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(techs, fontsize=8)
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.set_title(titles[idx], fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    for b, t in zip(bars, times):
        ax1.text(b.get_x() + b.get_width()/2.0, t + 0.02, f'{t:.3f} ms', ha='center', va='bottom', fontsize=8)
    
    # Second axis for PSNR quality
    ax2 = ax1.twinx()
    line = ax2.plot(x, psnrs, 'o-', color=color_line, linewidth=2.5, markersize=7, label='PSNR (dB)')
    ax2.set_ylabel("Quality: PSNR (dB)", color=color_line, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_line)
    
    for xi, p in zip(x, psnrs):
        ax2.text(xi, p + 1.2, f'{p:.1f} dB', ha='center', va='bottom', color=color_line, fontsize=8, fontweight='bold')
        
    ax2.set_ylim(0, max(psnrs) * 1.25)

plt.tight_layout()
plt.savefig("results/expD_screenspace.png", dpi=300)
plt.close()
print("Saved results/expD_screenspace.png")
