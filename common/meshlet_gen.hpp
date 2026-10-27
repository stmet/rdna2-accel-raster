#pragma once

#include "mesh_gen.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>

struct GpuMeshlet {
    glm::vec4 boundingSphere; // xyz = center, w = radius
    uint32_t vertexOffset;    // offset in meshletVertices
    uint32_t triangleOffset;  // offset in meshletTriangles
    uint32_t vertexCount;
    uint32_t triangleCount;
};

struct MeshletData {
    std::vector<GpuMeshlet> meshlets;
    std::vector<uint32_t> meshletVertices;
    std::vector<uint32_t> meshletTriangles; // packed 3 local u8 indices: i0 | (i1<<8) | (i2<<16)
};

class MeshletBuilder {
public:
    static MeshletData build(const SceneMesh& mesh, uint32_t maxTris = 64, uint32_t maxVerts = 64) {
        MeshletData result;
        uint32_t totalTris = mesh.triangleCount();
        if (totalTris == 0) return result;

        result.meshlets.reserve(totalTris / 32);
        result.meshletVertices.reserve(mesh.vertices.size());
        result.meshletTriangles.reserve(totalTris);

        std::unordered_map<uint32_t, uint8_t> localVertexMap;
        std::vector<uint32_t> clusterVertices;
        std::vector<uint32_t> clusterTris;

        auto flushMeshlet = [&]() {
            if (clusterTris.empty()) return;
            GpuMeshlet m{};
            m.vertexOffset = static_cast<uint32_t>(result.meshletVertices.size());
            m.triangleOffset = static_cast<uint32_t>(result.meshletTriangles.size());
            m.vertexCount = static_cast<uint32_t>(clusterVertices.size());
            m.triangleCount = static_cast<uint32_t>(clusterTris.size());

            // Compute tight bounding sphere
            glm::vec3 centroid(0.0f);
            for (uint32_t vIdx : clusterVertices) {
                centroid += glm::vec3(mesh.vertices[vIdx].pos[0], mesh.vertices[vIdx].pos[1], mesh.vertices[vIdx].pos[2]);
            }
            centroid /= static_cast<float>(clusterVertices.size());

            float maxDistSq = 0.0f;
            for (uint32_t vIdx : clusterVertices) {
                glm::vec3 p(mesh.vertices[vIdx].pos[0], mesh.vertices[vIdx].pos[1], mesh.vertices[vIdx].pos[2]);
                maxDistSq = std::max(maxDistSq, glm::dot(p - centroid, p - centroid));
            }
            m.boundingSphere = glm::vec4(centroid, std::sqrt(maxDistSq));

            result.meshlets.push_back(m);
            result.meshletVertices.insert(result.meshletVertices.end(), clusterVertices.begin(), clusterVertices.end());
            result.meshletTriangles.insert(result.meshletTriangles.end(), clusterTris.begin(), clusterTris.end());

            localVertexMap.clear();
            clusterVertices.clear();
            clusterTris.clear();
        };

        for (uint32_t t = 0; t < totalTris; t++) {
            uint32_t i0 = mesh.indices[t * 3 + 0];
            uint32_t i1 = mesh.indices[t * 3 + 1];
            uint32_t i2 = mesh.indices[t * 3 + 2];

            // Count new vertices
            uint32_t newVerts = 0;
            if (localVertexMap.find(i0) == localVertexMap.end()) newVerts++;
            if (localVertexMap.find(i1) == localVertexMap.end()) newVerts++;
            if (localVertexMap.find(i2) == localVertexMap.end()) newVerts++;

            if (clusterTris.size() >= maxTris || clusterVertices.size() + newVerts > maxVerts) {
                flushMeshlet();
            }

            auto getLocalIdx = [&](uint32_t vIdx) -> uint8_t {
                auto it = localVertexMap.find(vIdx);
                if (it != localVertexMap.end()) return it->second;
                uint8_t loc = static_cast<uint8_t>(clusterVertices.size());
                localVertexMap[vIdx] = loc;
                clusterVertices.push_back(vIdx);
                return loc;
            };

            uint8_t l0 = getLocalIdx(i0);
            uint8_t l1 = getLocalIdx(i1);
            uint8_t l2 = getLocalIdx(i2);
            uint32_t packed = static_cast<uint32_t>(l0) | (static_cast<uint32_t>(l1) << 8) | (static_cast<uint32_t>(l2) << 16);
            clusterTris.push_back(packed);
        }

        flushMeshlet();
        return result;
    }
};
