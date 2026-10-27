#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "================================================================================"
echo " Building AMD RDNA2 Integrated Full-Frame Deferred Rendering Testbench"
echo "================================================================================"

# 1. Compile Shaders
mkdir -p "${SCRIPT_DIR}/shaders/deferred"
if command -v glslangValidator >/dev/null 2>&1; then
    echo "[BUILD] Compiling shaders with glslangValidator..."
    glslangValidator -V --target-env vulkan1.3 shaders/verify_rayquery.comp -o shaders/verify_rayquery.spv
    glslangValidator -V --target-env vulkan1.3 shaders/fast_bvh_builder.comp -o shaders/fast_bvh_builder.spv
    for f in shaders/deferred/*.vert shaders/deferred/*.frag shaders/deferred/*.comp; do
        if [ -f "$f" ]; then
            glslangValidator -V --target-env vulkan1.3 "$f" -o "${f}.spv"
        fi
    done
elif command -v glslc >/dev/null 2>&1; then
    echo "[BUILD] Compiling shaders with glslc..."
    glslc --target-env=vulkan1.3 shaders/verify_rayquery.comp -o shaders/verify_rayquery.spv
    glslc --target-env=vulkan1.3 shaders/fast_bvh_builder.comp -o shaders/fast_bvh_builder.spv
    for f in shaders/deferred/*.vert shaders/deferred/*.frag shaders/deferred/*.comp; do
        if [ -f "$f" ]; then
            glslc --target-env=vulkan1.3 "$f" -o "${f}.spv"
        fi
    done
else
    echo "[ERROR] Neither glslangValidator nor glslc found in PATH!"
    exit 1
fi

# 2. Configure CMake and Build
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

echo "[BUILD] Configuring CMake in ${BUILD_DIR}..."
cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" -DCMAKE_BUILD_TYPE=Release

echo "[BUILD] Compiling targets..."
cmake --build "${BUILD_DIR}" --config Release -j"$(nproc)"

# 3. Execute Integrated Benchmark
cd "${BUILD_DIR}"
if [ "${1:-}" = "--all" ]; then
    echo ""
    echo "================================================================================"
    echo " Running rdna2_bvh_test (Phase 1 & 2 BVH Unit Tests & Microbenchmarks)"
    echo "================================================================================"
    ./rdna2_bvh_test
    echo ""
    echo "================================================================================"
    echo " Running fullframe_benchmark (Experiment E: Full-Frame Deferred Pipeline)"
    echo "================================================================================"
    ./fullframe_benchmark
    if [ -f "${BUILD_DIR}/results/dynamic_fullframe_benchmark.csv" ]; then
        mkdir -p "${SCRIPT_DIR}/results"
        cp -f "${BUILD_DIR}/results/dynamic_fullframe_benchmark.csv" "${SCRIPT_DIR}/results/dynamic_fullframe_benchmark.csv"
    fi
else
    echo ""
    echo "================================================================================"
    echo " Running fullframe_benchmark on AMD Radeon discrete GPU (Navi 22 / GFX10.3)"
    echo "================================================================================"
    ./fullframe_benchmark
    if [ -f "${BUILD_DIR}/results/dynamic_fullframe_benchmark.csv" ]; then
        mkdir -p "${SCRIPT_DIR}/results"
        cp -f "${BUILD_DIR}/results/dynamic_fullframe_benchmark.csv" "${SCRIPT_DIR}/results/dynamic_fullframe_benchmark.csv"
    fi
fi
