import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

# 1. Full-Screen Dense Visibility Sweep (4-way comparison)
dense_csv = "results/exp1_visbuf_dense.csv"
if os.path.exists(dense_csv):
    rows = []
    with open(dense_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "TargetDensity": float(r["TargetDensity_px_per_tri"]),
                "Triangles": int(r["Triangles"]),
                "ActualDensity": float(r["Actual_px_per_tri"]),
                "HwRaster": float(r["HwRaster_ms"]),
                "MeshShader": float(r["MeshShader_ms"]),
                "ComputeSoft": float(r["ComputeSoftRaster_ms"]),
                "RayQuery": float(r["RayQuery_ms"]),
                "SpeedupHw": float(r["Speedup_RQ_vs_HwRaster"]),
                "SpeedupMesh": float(r["Speedup_RQ_vs_MeshShader"]),
                "SpeedupSoft": float(r["Speedup_RQ_vs_ComputeSoftRaster"])
            })

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(rows))
    densities = [f'{r["TargetDensity"]} px²/tri\n({r["Triangles"]//1000}k tris)' for r in rows]

    # Subplot 1: Absolute Times
    width = 0.2
    axes[0].bar(x - 1.5*width, [r["HwRaster"] for r in rows], width, label='Hardware Raster', color='#d95f02')
    axes[0].bar(x - 0.5*width, [r["MeshShader"] for r in rows], width, label='Mesh Shader (Cluster Culling)', color='#e7298a')
    axes[0].bar(x + 0.5*width, [r["ComputeSoft"] for r in rows], width, label='Compute Software Raster', color='#7570b3')
    axes[0].bar(x + 1.5*width, [r["RayQuery"] for r in rows], width, label='Hardware Ray Query', color='#2b5c8f')

    axes[0].set_title('Dense Full-Screen Visibility Buffer (1080p, 200 Repetitions)')
    axes[0].set_xlabel('Projected Triangle Density')
    axes[0].set_ylabel('Per-Dispatch Execution Time (ms)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(densities)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].legend()

    # Subplot 2: Speedup of Ray Query over Baselines
    axes[1].plot(x, [r["SpeedupHw"] for r in rows], 'o-', color='#d95f02', linewidth=2.5, label='RQ Speedup vs HW Raster')
    axes[1].plot(x, [r["SpeedupMesh"] for r in rows], 's--', color='#e7298a', linewidth=2.5, label='RQ Speedup vs Mesh Shader')
    axes[1].plot(x, [r["SpeedupSoft"] for r in rows], '^-.', color='#7570b3', linewidth=2.5, label='RQ Speedup vs Compute SoftRaster')
    axes[1].axhline(y=1.0, color='gray', linestyle=':', label='Parity (1.0x)')

    axes[1].set_title('Ray Query Speedup vs Baselines Across Triangle Densities')
    axes[1].set_xlabel('Projected Triangle Density')
    axes[1].set_ylabel('Speedup Factor (RQ / Baseline)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(densities)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/exp1_visbuf_dense_showdown.png", dpi=300)
    plt.close()

# 2. Stanford Armadillo Dense Grid (Static BLAS vs TLAS Rigid Instancing)
arm_dense_csv = "results/exp1_visbuf_armadillo_dense.csv"
if os.path.exists(arm_dense_csv):
    arm_rows = []
    with open(arm_dense_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            arm_rows.append({
                "TargetDensity": float(r["TargetDensity_px_per_tri"]),
                "Instances": int(r["Instances"]),
                "TotalTris": int(r["TotalTriangles"]),
                "HwRaster": float(r["HwRaster_ms"]),
                "MeshShader": float(r["MeshShader_ms"]),
                "ComputeSoft": float(r["ComputeSoftRaster_ms"]),
                "RayQueryStatic": float(r["RayQuery_Static_ms"]),
                "RayQueryTLAS": float(r["RayQuery_TLAS_ms"])
            })

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(arm_rows))
    densities = [f'{r["TargetDensity"]} px²/tri\n({r["Instances"]} inst, {r["TotalTris"]//1000}k tris)' for r in arm_rows]

    width = 0.16
    ax.bar(x - 2*width, [r["HwRaster"] for r in arm_rows], width, label='Hardware Raster', color='#d95f02')
    ax.bar(x - width, [r["MeshShader"] for r in arm_rows], width, label='Mesh Shader (Cluster Cull)', color='#e7298a')
    ax.bar(x, [r["ComputeSoft"] for r in arm_rows], width, label='Compute Software Raster', color='#7570b3')
    ax.bar(x + width, [r["RayQueryStatic"] for r in arm_rows], width, label='Ray Query (Static Merged BLAS)', color='#2b5c8f')
    ax.bar(x + 2*width, [r["RayQueryTLAS"] for r in arm_rows], width, label='Ray Query (TLAS Rigid Instancing)', color='#1b9e77')

    ax.set_title('Stanford Armadillo Grid (Up to 6.4M Tris): Static BLAS vs TLAS Instancing')
    ax.set_xlabel('Projected Triangle Density (Instances & Complexity)')
    axes[0].set_ylabel('Execution Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(densities)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/exp1_visbuf_armadillo_dense.png", dpi=300)
    plt.close()

print("Experiment 1 plots generated successfully in results/")
