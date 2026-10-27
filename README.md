# RDNA2 Ray Accelerator Utilization in Rasterized Deferred Rendering & Real-Time GPU BVH Builder

**Author:** stmet  
**Target Architecture:** AMD RDNA2 (Navi 22 / GFX10.3)  
**Implementation:** Vulkan 1.3 (`VK_KHR_ray_query`, `VK_KHR_acceleration_structure`, `VK_KHR_buffer_device_address`), Mesa RADV (ACO Compiler) on Linux

---

## Abstract

Modern graphics processors dedicate substantial die area to hardware ray tracing fixed-function units (Ray Accelerators on AMD RDNA2). In standard rasterized rendering engines, this specialized silicon remains idle across the entire frame. This research project investigates the architectural boundaries of repurposing dedicated RDNA2 Ray Accelerators to accelerate rasterized deferred rendering workloads under Linux (Mesa RADV / ACO). 

Through microbenchmarking and full-pipeline hardware validation on an AMD Radeon RX 6700 XT, we quantify hardware-queue overlap characteristics (94.1% async compute efficiency on TMU-idle raster passes), determine crossover points where inline ray queries outperform raster passes (shadows with $\ge 4$ casters up to 181x faster, planar reflections up to 60x faster, and 1 SPP RQAO providing +19.7 dB PSNR improvement over SSAO at identical frame budget), and resolve the dynamic BVH update bottleneck by implementing an on-chip, single-pass compute BVH builder (`fast_bvh_builder.comp`). Our custom builder constructs valid RDNA2 Box4 and Triangle hierarchies in LDS memory with zero global scratch allocations, achieving a 7.5x speedup over driver builds and lifting dynamic deferred rendering framerates from 204.6 FPS to 231.9 FPS (+27.3 FPS) at 1080p.

---

## Key Hardware Benchmark Results

All measurements were obtained on bare-metal hardware using 64-bit hardware GPU timestamp queries (`vkCmdWriteTimestamp`, 10.0 ns tick resolution) across 200 consecutive measurement iterations following 10 warmup iterations, locked at peak DPM state (2725 MHz engine / 1000 MHz memory).

### 1. Hybrid Technique Crossover Thresholds

| Workload / Technique | Baseline Raster Technique | Ray Query Configuration | Crossover Condition | Peak Speedup Ratio / Fidelity Delta |
| :--- | :--- | :--- | :--- | :---: |
| **Shadows (Multiple Casters)** | 2D Texture Array shadow maps + 16-tap PCF | Tiled LDS inline Ray Query | $\ge 500\text{k}$ triangles, $\ge 4$ shadow lights | **181.6x** (2.5M tris, 16 lights: 0.19 ms vs 33.65 ms) |
| **Planar Reflections** | Mirrored camera raster pass + composite | Inline Ray Query per reflective pixel | $\ge 500\text{k}$ triangles at any screen coverage | **60.4x** (2.5M tris, 2% cov: 0.035 ms vs 2.12 ms) |
| **Ambient Occlusion** | Screen-Space Ambient Occlusion (SSAO 16 samples) | 1 SPP RQAO + Bilateral Blur | Equal frame budget (~0.50 ms) | **+19.7 dB PSNR** (40.2 dB vs 20.5 dB, 0.504 ms vs 0.508 ms) |
| **Subpixel Visibility** | Hardware Rasterizer / Mesh Shaders | 1 SPP Inline Ray Query Visibility | Geometry density $\le 0.25\text{ px}^2/\text{tri}$ | **2.75x** (1.94 ms vs 5.34 ms at 8.3M triangles) |

### 2. Async Compute Overlap Efficiency (Queue Family 1)

On RDNA2, the Ray Accelerator datapath is housed inside the Texture Mapping Unit (TMU). During depth-only and shadow raster passes, fragment shaders do not sample textures and TMUs are idle.

| Queue Scheduling Configuration | Wall Time (ms) | Graphics Queue | Compute Queue | Hardware Overlap | Overlap Efficiency ($\eta$) | Net Framerate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Serialized (Queue Family 0)** | 2.800 ms | 2.800 ms | 0.000 ms | 0.000 ms | 0.0% | 357.1 FPS |
| **Async Overlap: Depth & Shadow Passes** | **1.843 ms** | 1.825 ms | 0.618 ms | **0.574 ms** | **94.1%** | **542.7 FPS (+52.0%)** |
| **Async Overlap: Texture-Heavy G-Buffer** | **1.901 ms** | 1.839 ms | 0.656 ms | **0.545 ms** | **87.8%** | **526.1 FPS (+47.3%)** |

*Architectural finding:* Overlapping compute ray queries with a texture-heavy G-buffer induces a +6.2% contention penalty on compute execution time due to TMU addressing conflict, whereas depth-only rasterization achieves 94.1% concurrency.

### 3. Custom Single-Pass Compute BVH Builder vs. Driver (`vkCmdBuildAccelerationStructuresKHR`)

Standard driver builds introduce synchronization serialization and require up to 553 KB global scratch memory for 4,096 triangles. Our compute builder (`shaders/fast_bvh_builder.comp`) builds an RDNA2-compliant RadvAccelStructHeader, multi-level Box4 internal tree, and 64-byte Triangle leaf nodes in a single workgroup using 32.7 KB Local Data Share (LDS) with **zero scratch memory**.

| Triangle Count ($N$) | Driver Build Time (`vkCmdBuildAccelerationStructuresKHR`) | Custom Builder (`fast_bvh_builder.comp`) | Speedup Ratio | Driver Scratch Memory | Custom Builder Scratch | Hardware Ray Query Hit Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | 377.23 µs | **9.41 µs** | **40.1x** | 70 KB | **0 Bytes** | 64 / 64 (100.0%) |
| **256** | 179.59 µs | **5.14 µs** | **34.9x** | 91 KB | **0 Bytes** | 256 / 256 (100.0%) |
| **1,024** | 190.67 µs | **15.59 µs** | **12.2x** | 172 KB | **0 Bytes** | 1,024 / 1,024 (100.0%) |
| **4,096** | 319.16 µs | **80.35 µs** | **4.0x** | 553 KB | **0 Bytes** | 4,096 / 4,096 (100.0%) |

### 4. Full-Frame Deferred Rendering Pipeline (Dynamic Scene at 1080p)

Workload: 1080p deferred frame with dynamic deforming grid mesh, 16 dynamic shadow-casting lights, 512 tiled point lights (LDS light list culling), 1 SPP RQAO vs SSAO, and planar reflections over 200 consecutive frames.

| Pipeline Configuration | Frame Time (Median) | 5th / 95th Percentile | Framerate (FPS) | Dedicated BVH Duration | PSNR / SSIM vs Raster | Architectural Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Baseline_Raster`** | 9.528 ms | 9.508 ms / 9.542 ms | 104.96 FPS | N/A (Raster) | Baseline | 16 shadow passes + SSAO bottleneck |
| **`RA_Driver_Dynamic`** | 4.887 ms | 4.849 ms / 4.950 ms | 204.63 FPS | 580.68 µs | 24.03 dB / 0.9758 | Driver BVH refit penalty (-27 FPS) |
| **`RA_Custom_Compute_Dynamic`** | **4.312 ms** | **4.290 ms / 4.342 ms** | **231.91 FPS** | **77.56 µs** | 24.03 dB / 0.9758 | **+27.3 FPS gain (7.5x faster BVH build)** |
| **`RA_Custom_Compute_Async`** | **4.325 ms** | **4.308 ms / 4.345 ms** | **231.24 FPS** | 0.00 µs (Overlapped) | 24.03 dB / 0.9758 | Builder + RQAO on Queue Family 1 |

---

## Hardware & Environment Specification

- **GPU:** AMD Radeon RX 6700 XT (Navi 22, GFX10.3)
  - Compute Units: 40 CUs (2560 stream processors, 40 Ray Accelerators)
  - Memory: 12 GB GDDR6 (192-bit bus, 384 GB/s bandwidth)
  - Cache Hierarchy: 96 MB on-chip Infinity Cache (MALL), 3 MB L2, 128 KB L1 per dual-CU
  - Clock Profile: DPM State 1 (Locked Peak: 2725 MHz core sclk / 1000 MHz memory mclk)
- **Host System:**
  - OS: Linux 7.3.0-cachyos (`x86_64`)
  - Vulkan Driver: Mesa RADV 26.2.5 (ACO Compiler, Vulkan 1.4.363)
  - Command Processor: Queue Family 0 (Graphics, Compute, Transfer), Queue Family 1 (Dedicated Async Compute, 4 hardware queues)
- **Vulkan Extensions Utilized:**
  - `VK_KHR_ray_query`
  - `VK_KHR_acceleration_structure`
  - `VK_KHR_buffer_device_address`
  - `VK_KHR_synchronization2`
  - `VK_EXT_scalar_block_layout`

---

## Repository Structure

```text
├── CMakeLists.txt              # Unified CMake configuration
├── README.md                   # Technical specification and benchmark overview
├── REPORT.md                   # Comprehensive empirical research report
├── run.sh                      # Automated build, shader compile, and execution script
├── .gitignore                  # Git exclusions for builds and compilation artifacts
├── include/
│   └── radv_bvh.hpp            # RDNA2 Mesa RADV hardware acceleration structure definitions
├── src/
│   ├── main.cpp                # Hardware BVH reverse engineering & compute builder benchmarks
│   └── fullframe_benchmark.cpp # Integrated full-frame dynamic deferred rendering benchmark
├── shaders/
│   ├── fast_bvh_builder.comp   # Single-pass GPU compute BVH builder (zero scratch memory)
│   ├── verify_rayquery.comp    # Hardware Ray Query intersection verification compute shader
│   ├── tiled_lighting_ra.comp  # Tiled deferred lighting with inline ray query shadows
│   ├── tiled_lighting_raster.comp # Tiled deferred lighting with raster shadow map sampling
│   ├── rq_ao.comp              # 1 SPP Ray Query Ambient Occlusion
│   ├── ssao_gtao.comp          # Screen-Space Ambient Occlusion (16 samples)
│   ├── bilateral_blur.comp     # Cross-bilateral depth/normal reconstruction filter
│   ├── gbuffer_heavy.vert      # Multi-draw G-buffer vertex shader
│   ├── gbuffer_heavy.frag      # Multi-material G-buffer fragment shader
│   ├── raster_depth.vert       # Depth prepass vertex shader
│   └── raster_depth.frag       # Depth prepass fragment shader
├── common/
│   ├── vk_context.hpp/.cpp     # Vulkan 1.3 physical device, queue, and instance setup
│   ├── vk_timer.hpp/.cpp       # 64-bit GPU timestamp query harness (DualQueueTimer)
│   ├── vk_buffer.hpp/.cpp      # Device memory allocations and buffer device addresses
│   ├── vk_bvh.hpp/.cpp         # Driver acceleration structure wrappers
│   ├── vk_shader.hpp/.cpp      # SPIR-V pipeline compilation utilities
│   ├── mesh_gen.hpp            # Dynamic terrain and analytic mesh procedural generators
│   └── image_metrics.hpp       # GPU readback PSNR, RMSE, and SSIM metrics calculation
├── benchmarks/
│   ├── expA_async/             # Dedicated async compute overlap & TMU contention benchmarks
│   ├── expB_shadows/           # Shadow caster sweep benchmarks (1 to 64 lights, 100k to 2.5M tris)
│   ├── expC_reflections/       # Planar reflection sweep benchmarks across mirror screen coverage
│   ├── expD_screenspace/       # Screen-space replacement benchmarks (SSAO, SSR, SSCS vs RA)
│   ├── expE_fullframe/         # Static full-frame realistic integration harness
│   └── exp0_probe/             # Hardware capability probes and microbenchmarks
├── results/
│   ├── dynamic_fullframe_benchmark.csv # 200-frame hardware timestamp logging
│   ├── *.csv                   # Complete CSV measurement logs for all passes
│   └── *.png                   # Rendered benchmark distribution charts
└── scripts/
    └── plot_*.py               # Automated matplotlib figure generation scripts
```

---

## Reproducibility Instructions

### Prerequisites
- Modern Linux distribution (Arch, CachyOS, Fedora, Ubuntu 22.04+)
- AMD RDNA2 GPU (Radeon RX 6000-series, GFX10.3 / Navi 21, Navi 22, Navi 23)
- Mesa RADV 23.0+ with ACO compiler
- Vulkan SDK 1.3+ / 1.4+
- CMake 3.20+, GCC 12+ or Clang 15+
- `glslangValidator` or `glslc` (included in Vulkan SDK)

### Quick Start: Build and Run Integrated Full-Frame Benchmark
```bash
# Clone the repository
git clone https://github.com/stmet/rdna2-accel-raster.git
cd rdna2-accel-raster

# Execute automated compilation and benchmark harness:
./run.sh
```

### Full Benchmark Suite
To execute both the integrated full-frame dynamic deferred pipeline benchmark and the Phase 1/2 hardware BVH unit tests and microbenchmarks:
```bash
./run.sh --all
```

Results are logged directly to standard output and persisted to `results/dynamic_fullframe_benchmark.csv`.
