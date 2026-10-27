#pragma once

#include "mesh_gen.hpp"
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <iostream>

struct alignas(16) GpuBvhNode {
    float aabbMin[3];
    int32_t leftChild;   // >= 0: inner node (left child index); < 0: leaf (~firstPrimOffset)
    float aabbMax[3];
    int32_t rightChild;  // inner node: right child index; leaf: primitive count
};

struct alignas(16) GpuTriangle {
    float v0[4];
    float v1[4];
    float v2[4];
};

struct SoftwareBVH {
    std::vector<GpuBvhNode> nodes;
    std::vector<GpuTriangle> triangles;

    struct BuildPrimitive {
        uint32_t index;
        float center[3];
        float min[3];
        float max[3];
    };

    static SoftwareBVH build(const SceneMesh& mesh, uint32_t maxPrimsPerLeaf = 4) {
        SoftwareBVH bvh;
        uint32_t triCount = mesh.triangleCount();
        std::vector<BuildPrimitive> prims(triCount);

        for (uint32_t i = 0; i < triCount; i++) {
            uint32_t i0 = mesh.indices[i * 3 + 0];
            uint32_t i1 = mesh.indices[i * 3 + 1];
            uint32_t i2 = mesh.indices[i * 3 + 2];

            const auto& p0 = mesh.vertices[i0].pos;
            const auto& p1 = mesh.vertices[i1].pos;
            const auto& p2 = mesh.vertices[i2].pos;

            prims[i].index = i;
            for (int c = 0; c < 3; c++) {
                prims[i].min[c] = std::min({p0[c], p1[c], p2[c]});
                prims[i].max[c] = std::max({p0[c], p1[c], p2[c]});
                prims[i].center[c] = (p0[c] + p1[c] + p2[c]) / 3.0f;
            }
        }

        bvh.nodes.reserve(triCount * 2);
        bvh.triangles.resize(triCount);

        std::vector<BuildPrimitive> orderedPrims;
        orderedPrims.reserve(triCount);

        recursiveBuild(bvh, prims, 0, triCount, orderedPrims, maxPrimsPerLeaf);

        // Populate triangles in leaf order
        for (uint32_t i = 0; i < triCount; i++) {
            uint32_t origIdx = orderedPrims[i].index;
            uint32_t i0 = mesh.indices[origIdx * 3 + 0];
            uint32_t i1 = mesh.indices[origIdx * 3 + 1];
            uint32_t i2 = mesh.indices[origIdx * 3 + 2];

            const auto& p0 = mesh.vertices[i0].pos;
            const auto& p1 = mesh.vertices[i1].pos;
            const auto& p2 = mesh.vertices[i2].pos;

            for (int c = 0; c < 3; c++) {
                bvh.triangles[i].v0[c] = p0[c];
                bvh.triangles[i].v1[c] = p1[c];
                bvh.triangles[i].v2[c] = p2[c];
            }
            bvh.triangles[i].v0[3] = 0.0f;
            bvh.triangles[i].v1[3] = 0.0f;
            bvh.triangles[i].v2[3] = 0.0f;
        }

        return bvh;
    }

private:
    static int32_t recursiveBuild(
        SoftwareBVH& bvh,
        std::vector<BuildPrimitive>& prims,
        uint32_t start,
        uint32_t end,
        std::vector<BuildPrimitive>& orderedPrims,
        uint32_t maxPrimsPerLeaf
    ) {
        int32_t nodeIdx = static_cast<int32_t>(bvh.nodes.size());
        bvh.nodes.emplace_back();

        // Calculate bounding box
        float bmin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
        float bmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (uint32_t i = start; i < end; i++) {
            for (int c = 0; c < 3; c++) {
                bmin[c] = std::min(bmin[c], prims[i].min[c]);
                bmax[c] = std::max(bmax[c], prims[i].max[c]);
            }
        }
        for (int c = 0; c < 3; c++) {
            bvh.nodes[nodeIdx].aabbMin[c] = bmin[c];
            bvh.nodes[nodeIdx].aabbMax[c] = bmax[c];
        }

        uint32_t count = end - start;
        if (count <= maxPrimsPerLeaf) {
            // Leaf node
            uint32_t firstPrimOffset = static_cast<uint32_t>(orderedPrims.size());
            for (uint32_t i = start; i < end; i++) {
                orderedPrims.push_back(prims[i]);
            }
            bvh.nodes[nodeIdx].leftChild = ~firstPrimOffset; // bitwise NOT
            bvh.nodes[nodeIdx].rightChild = count;
            return nodeIdx;
        }

        // Split on longest axis of centroid bounds
        float cmin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
        float cmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (uint32_t i = start; i < end; i++) {
            for (int c = 0; c < 3; c++) {
                cmin[c] = std::min(cmin[c], prims[i].center[c]);
                cmax[c] = std::max(cmax[c], prims[i].center[c]);
            }
        }
        int axis = 0;
        float maxExtent = cmax[0] - cmin[0];
        if (cmax[1] - cmin[1] > maxExtent) {
            axis = 1;
            maxExtent = cmax[1] - cmin[1];
        }
        if (cmax[2] - cmin[2] > maxExtent) {
            axis = 2;
            maxExtent = cmax[2] - cmin[2];
        }

        if (maxExtent < 1e-6f) {
            // Degenerate bounds, create leaf
            uint32_t firstPrimOffset = static_cast<uint32_t>(orderedPrims.size());
            for (uint32_t i = start; i < end; i++) {
                orderedPrims.push_back(prims[i]);
            }
            bvh.nodes[nodeIdx].leftChild = ~firstPrimOffset;
            bvh.nodes[nodeIdx].rightChild = count;
            return nodeIdx;
        }

        // Midpoint or median split
        uint32_t mid = (start + end) / 2;
        std::nth_element(
            prims.begin() + start,
            prims.begin() + mid,
            prims.begin() + end,
            [axis](const BuildPrimitive& a, const BuildPrimitive& b) {
                return a.center[axis] < b.center[axis];
            }
        );

        int32_t leftChild = recursiveBuild(bvh, prims, start, mid, orderedPrims, maxPrimsPerLeaf);
        int32_t rightChild = recursiveBuild(bvh, prims, mid, end, orderedPrims, maxPrimsPerLeaf);

        bvh.nodes[nodeIdx].leftChild = leftChild;
        bvh.nodes[nodeIdx].rightChild = rightChild;
        return nodeIdx;
    }
};

struct alignas(16) GpuBvh4Node {
    float minX[4];
    float maxX[4];
    float minY[4];
    float maxY[4];
    float minZ[4];
    float maxZ[4];
    int32_t childIndex[4];
    int32_t childCount[4]; // 0: empty, -1: inner node, >0: prim count for leaf
};

struct SoftwareBVH4 {
    std::vector<GpuBvh4Node> nodes;
    std::vector<GpuTriangle> triangles;

    static SoftwareBVH4 fromBVH2(const SoftwareBVH& bvh2) {
        SoftwareBVH4 bvh4;
        bvh4.triangles = bvh2.triangles;
        if (bvh2.nodes.empty()) return bvh4;
        collapse(bvh2, 0, bvh4);
        return bvh4;
    }

private:
    static int32_t collapse(const SoftwareBVH& bvh2, int32_t bvh2NodeIdx, SoftwareBVH4& bvh4) {
        const auto& node = bvh2.nodes[bvh2NodeIdx];

        if (node.leftChild < 0) {
            int32_t outIdx = static_cast<int32_t>(bvh4.nodes.size());
            bvh4.nodes.emplace_back();
            auto& n4 = bvh4.nodes[outIdx];
            for (int k = 0; k < 4; k++) {
                n4.childCount[k] = 0;
                n4.childIndex[k] = 0;
            }
            n4.minX[0] = node.aabbMin[0]; n4.maxX[0] = node.aabbMax[0];
            n4.minY[0] = node.aabbMin[1]; n4.maxY[0] = node.aabbMax[1];
            n4.minZ[0] = node.aabbMin[2]; n4.maxZ[0] = node.aabbMax[2];
            n4.childIndex[0] = node.leftChild;
            n4.childCount[0] = node.rightChild;
            return outIdx;
        }

        std::vector<int32_t> candidates;
        candidates.push_back(node.leftChild);
        candidates.push_back(node.rightChild);

        while (candidates.size() < 4) {
            int bestIdx = -1;
            float maxSurfaceArea = -1.0f;
            for (size_t i = 0; i < candidates.size(); i++) {
                int32_t cIdx = candidates[i];
                if (bvh2.nodes[cIdx].leftChild >= 0) {
                    float dx = bvh2.nodes[cIdx].aabbMax[0] - bvh2.nodes[cIdx].aabbMin[0];
                    float dy = bvh2.nodes[cIdx].aabbMax[1] - bvh2.nodes[cIdx].aabbMin[1];
                    float dz = bvh2.nodes[cIdx].aabbMax[2] - bvh2.nodes[cIdx].aabbMin[2];
                    float sa = 2.0f * (dx * dy + dy * dz + dz * dx);
                    if (sa > maxSurfaceArea) {
                        maxSurfaceArea = sa;
                        bestIdx = static_cast<int>(i);
                    }
                }
            }

            if (bestIdx == -1) break;

            int32_t expandNode = candidates[bestIdx];
            candidates[bestIdx] = bvh2.nodes[expandNode].leftChild;
            candidates.push_back(bvh2.nodes[expandNode].rightChild);
        }

        int32_t outIdx = static_cast<int32_t>(bvh4.nodes.size());
        bvh4.nodes.emplace_back();

        int32_t childIndices[4] = {0, 0, 0, 0};
        int32_t childCounts[4] = {0, 0, 0, 0};
        float minX[4]{}, maxX[4]{}, minY[4]{}, maxY[4]{}, minZ[4]{}, maxZ[4]{};

        for (size_t i = 0; i < candidates.size(); i++) {
            int32_t cIdx = candidates[i];
            const auto& cNode = bvh2.nodes[cIdx];
            minX[i] = cNode.aabbMin[0]; maxX[i] = cNode.aabbMax[0];
            minY[i] = cNode.aabbMin[1]; maxY[i] = cNode.aabbMax[1];
            minZ[i] = cNode.aabbMin[2]; maxZ[i] = cNode.aabbMax[2];

            if (cNode.leftChild < 0) {
                childIndices[i] = cNode.leftChild;
                childCounts[i] = cNode.rightChild;
            } else {
                childCounts[i] = -1;
            }
        }

        for (size_t i = 0; i < candidates.size(); i++) {
            int32_t cIdx = candidates[i];
            const auto& cNode = bvh2.nodes[cIdx];
            if (cNode.leftChild >= 0) {
                childIndices[i] = collapse(bvh2, cIdx, bvh4);
            }
        }

        for (int i = 0; i < 4; i++) {
            bvh4.nodes[outIdx].minX[i] = minX[i];
            bvh4.nodes[outIdx].maxX[i] = maxX[i];
            bvh4.nodes[outIdx].minY[i] = minY[i];
            bvh4.nodes[outIdx].maxY[i] = maxY[i];
            bvh4.nodes[outIdx].minZ[i] = minZ[i];
            bvh4.nodes[outIdx].maxZ[i] = maxZ[i];
            bvh4.nodes[outIdx].childIndex[i] = childIndices[i];
            bvh4.nodes[outIdx].childCount[i] = childCounts[i];
        }

        return outIdx;
    }
};
