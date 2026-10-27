#pragma once

#include <cstdint>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace rdna2 {

// =============================================================================
// AMD RDNA2 (GFX10.3 / Navi 2x) Hardware Ray Tracing Architecture Definitions
// Reference: Mesa RADV source code (src/amd/vulkan/bvh/bvh_defines.h,
// src/amd/vulkan/radv_acceleration_structure.c, src/amd/vulkan/nir/radv_nir_rt_common.c)
// =============================================================================

// Node types encoded in the lowest 3 bits of 32-bit BVH node IDs
constexpr uint32_t RADV_BVH_NODE_TRIANGLE = 0;
constexpr uint32_t RADV_BVH_NODE_BOX16    = 4;
constexpr uint32_t RADV_BVH_NODE_BOX32    = 5;
constexpr uint32_t RADV_BVH_NODE_INSTANCE = 6;
constexpr uint32_t RADV_BVH_NODE_AABB     = 7;

constexpr uint32_t RADV_BVH_ROOT_NODE     = RADV_BVH_NODE_BOX32; // 5
constexpr uint32_t RADV_BVH_INVALID_NODE  = 0xffffffffu;

// Geometry flags
constexpr uint32_t RADV_GEOMETRY_OPAQUE = (1u << 31);

// Barycentric control id for GFX10.3 (TMU image_bvh_intersect_ray instruction)
// In Mesa src/amd/vulkan/bvh/encode.h: barycentrics_control = 9; if opaque |= 128;
constexpr uint32_t RADV_BARYCENTRICS_CONTROL_DEFAULT = 9;
constexpr uint32_t RADV_BARYCENTRICS_CONTROL_OPAQUE  = 9 | 128;

// Helper packing functions
inline uint32_t pack_node_id(uint32_t offset_from_bvh_base, uint32_t type) {
    return (offset_from_bvh_base >> 3) | (type & 7u);
}

inline uint32_t id_to_offset(uint32_t id) {
    return (id & (~7u)) << 3;
}

inline uint32_t id_to_type(uint32_t id) {
    return id & 7u;
}

inline const char* node_type_to_string(uint32_t type) {
    switch (type) {
        case RADV_BVH_NODE_TRIANGLE: return "Triangle (0)";
        case RADV_BVH_NODE_BOX16:    return "Box16 (4)";
        case RADV_BVH_NODE_BOX32:    return "Box32 (5)";
        case RADV_BVH_NODE_INSTANCE: return "Instance (6)";
        case RADV_BVH_NODE_AABB:     return "AABB (7)";
        default:                     return "Unknown";
    }
}

// Bounding box structures
struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct AABB {
    Vec3 min;
    Vec3 max;
};

// -----------------------------------------------------------------------------
// RADV Acceleration Structure Header
// Matches Mesa's struct radv_accel_struct_header in bvh_defines.h (size: 136 bytes)
// -----------------------------------------------------------------------------
#pragma pack(push, 4)
struct RadvAccelStructHeader {
    uint32_t bvh_offset;                // 0x00: Offset from start of buffer to root Box4 node (0xC0 = 192)
    uint32_t root_flags;                // 0x04: Box flags of root node (0)
    AABB     aabb;                      // 0x08: Total bounding box of the acceleration structure (24 bytes, 0x08-0x20)

    uint32_t update_dispatch_size[3];   // 0x20: Update dispatch dimensions [8, 1, 1]
    uint32_t padding0;                  // 0x2C: Alignment padding

    uint64_t compacted_size;            // 0x30: Compacted size in bytes (384)
    uint64_t serialization_size;        // 0x38: Serialization size in bytes (512)
    uint32_t copy_dispatch_size[3];     // 0x40: Dispatch dimensions for copy shader [1, 1, 1]
    uint32_t padding1;                  // 0x4C: Alignment padding
    uint64_t size;                      // 0x50: Base acceleration structure size (456)

    uint32_t geometry_type;             // 0x58: VK_GEOMETRY_TYPE_TRIANGLES_KHR = 0
    uint32_t geometry_count;            // 0x5C: Number of geometries (1)
    uint64_t leaf_nodes_offset;         // 0x60: Buffer offset to leaf nodes section (0x140 = 320)
    uint64_t primitive_count;           // 0x68: Number of primitives (1)
    uint64_t instance_count;            // 0x70: Number of instances (0 for BLAS, 1 for TLAS)
    uint32_t leaf_node_offsets_offset;  // 0x78: Leaf node offsets offset (0x88)
    uint32_t build_flags;               // 0x7C: VkBuildAccelerationStructureFlagsKHR (4)
    uint32_t type;                      // 0x80: Acceleration structure type
    uint32_t padding2;                  // 0x84: Padding
};
#pragma pack(pop)

static_assert(offsetof(RadvAccelStructHeader, bvh_offset) == 0x00, "bvh_offset offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, aabb) == 0x08, "aabb offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, update_dispatch_size) == 0x20, "update_dispatch_size offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, compacted_size) == 0x30, "compacted_size offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, serialization_size) == 0x38, "serialization_size offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, copy_dispatch_size) == 0x40, "copy_dispatch_size offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, size) == 0x50, "size offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, geometry_type) == 0x58, "geometry_type offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, geometry_count) == 0x5C, "geometry_count offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, leaf_nodes_offset) == 0x60, "leaf_nodes_offset offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, primitive_count) == 0x68, "primitive_count offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, instance_count) == 0x70, "instance_count offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, leaf_node_offsets_offset) == 0x78, "leaf_node_offsets_offset offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, build_flags) == 0x7C, "build_flags offset mismatch");
static_assert(offsetof(RadvAccelStructHeader, type) == 0x80, "type offset mismatch");

// -----------------------------------------------------------------------------
// RADV Box32 Node (Internal BVH4 Node)
// Exactly 128 bytes. Matches Mesa's struct radv_bvh_box32_node in bvh_defines.h.
// Directly decoded by RDNA2 TMU instruction: image_bvh_intersect_ray.
// -----------------------------------------------------------------------------
#pragma pack(push, 4)
struct RadvBvhBox32Node {
    uint32_t children[4];               // 4 Child node IDs: (child_offset_from_bvh_base >> 3) | type
    AABB     coords[4];                 // 4 Child bounding boxes (4 * 24 = 96 bytes)
    uint32_t flags;                     // Box flags (0)
    uint32_t reserved[3];               // Hardware reserved / padding (12 bytes)
};
#pragma pack(pop)
static_assert(sizeof(RadvBvhBox32Node) == 128, "RadvBvhBox32Node must be 128 bytes");

// -----------------------------------------------------------------------------
// RADV Triangle Node (Leaf Node)
// Exactly 64 bytes. Matches Mesa's struct radv_bvh_triangle_node in bvh_defines.h.
// Directly evaluated by RDNA2 TMU instruction: image_bvh_intersect_ray.
// -----------------------------------------------------------------------------
#pragma pack(push, 4)
struct RadvBvhTriangleNode {
    float    coords[3][3];              // Vertices V0, V1, V2 (9 * 4 = 36 bytes)
    uint32_t reserved[3];               // Reserved (12 bytes)
    uint32_t triangle_id;               // Primitive index / triangle ID (4 bytes)
    uint32_t geometry_id_and_flags;     // Flags in upper 4 bits: bit 31 = RADV_GEOMETRY_OPAQUE
    uint32_t reserved2;                 // Reserved (4 bytes)
    uint32_t id;                        // Barycentrics control register value (default: 9)
};
#pragma pack(pop)
static_assert(sizeof(RadvBvhTriangleNode) == 64, "RadvBvhTriangleNode must be 64 bytes");

// =============================================================================
// Phase 2: Fast BVH Builder Math & Push Constants
// =============================================================================

struct FastBvhPushConstants {
    uint64_t vertexBufferAddr;
    uint64_t asBufferAddr;
    uint32_t numTriangles;
    uint32_t totalBoxes;
    uint32_t numLevels;
    uint32_t totalAsSize;
    uint32_t levelOffsets[8];
    uint32_t levelCounts[8];
};

struct FastBvhTreeLayout {
    uint32_t numTriangles = 0;
    uint32_t totalBoxes = 0;
    uint32_t numLevels = 0;
    uint32_t totalAsSize = 0;
    std::vector<uint32_t> levelCounts;
    std::vector<uint32_t> levelOffsets;

    FastBvhPushConstants to_push_constants(uint64_t vbufAddr = 0, uint64_t asbufAddr = 0) const {
        FastBvhPushConstants pc{};
        pc.vertexBufferAddr = vbufAddr;
        pc.asBufferAddr = asbufAddr;
        pc.numTriangles = numTriangles;
        pc.totalBoxes = totalBoxes;
        pc.numLevels = numLevels;
        pc.totalAsSize = totalAsSize;
        for (uint32_t i = 0; i < 8; ++i) {
            pc.levelOffsets[i] = (i < numLevels) ? levelOffsets[i] : 0;
            pc.levelCounts[i] = (i < numLevels) ? levelCounts[i] : 0;
        }
        return pc;
    }
};

inline FastBvhTreeLayout compute_tree_layout(uint32_t N) {
    FastBvhTreeLayout layout;
    layout.numTriangles = N;
    uint32_t leafBoxes = (N + 3) / 4;
    std::vector<uint32_t> counts = {leafBoxes};
    while (counts.back() > 1) {
        counts.push_back((counts.back() + 3) / 4);
    }
    std::reverse(counts.begin(), counts.end());
    layout.numLevels = static_cast<uint32_t>(counts.size());
    layout.levelCounts = counts;
    layout.levelOffsets.resize(layout.numLevels);

    uint32_t curOffset = 0;
    for (uint32_t i = 0; i < layout.numLevels; ++i) {
        layout.levelOffsets[i] = curOffset;
        curOffset += counts[i];
    }
    layout.totalBoxes = curOffset;
    // 192 bytes header + totalBoxes * 128 bytes Box4 + numTriangles * 64 bytes Triangle
    layout.totalAsSize = 192 + layout.totalBoxes * 128 + layout.numTriangles * 64;
    return layout;
}

// Dynamic Deforming Grid Mesh
struct DynamicMesh {
    std::vector<float> vertices; // 9 floats per triangle (3 vertices x 3 coords)
    std::vector<Vec3> centroids; // Analytic centroid per triangle
    uint32_t triangleCount = 0;
};

inline DynamicMesh generate_dynamic_grid_mesh(uint32_t N, float timeVal = 0.0f, float extent = 2.0f) {
    DynamicMesh mesh;
    mesh.triangleCount = N;

    uint32_t numQuads = N / 2;
    uint32_t gridCols = static_cast<uint32_t>(std::sqrt(numQuads));
    while (numQuads % gridCols != 0 && gridCols > 1) {
        gridCols--;
    }
    uint32_t gridRows = numQuads / gridCols;

    float dx = (2.0f * extent) / gridCols;
    float dy = (2.0f * extent) / gridRows;

    auto evalZ = [timeVal](float x, float y) -> float {
        return 5.0f + 0.2f * std::sin(x * 2.0f + timeVal) * std::cos(y * 2.0f + timeVal);
    };

    mesh.vertices.reserve(N * 9);
    mesh.centroids.reserve(N);

    for (uint32_t r = 0; r < gridRows; ++r) {
        for (uint32_t c = 0; c < gridCols; ++c) {
            float x0 = -extent + c * dx;
            float x1 = x0 + dx;
            float y0 = -extent + r * dy;
            float y1 = y0 + dy;

            float z00 = evalZ(x0, y0);
            float z10 = evalZ(x1, y0);
            float z01 = evalZ(x0, y1);
            float z11 = evalZ(x1, y1);

            // Triangle 1: (x0, y0, z00), (x1, y0, z10), (x0, y1, z01)
            mesh.vertices.insert(mesh.vertices.end(), {
                x0, y0, z00,
                x1, y0, z10,
                x0, y1, z01
            });
            mesh.centroids.push_back({
                (x0 + x1 + x0) / 3.0f,
                (y0 + y0 + y1) / 3.0f,
                (z00 + z10 + z01) / 3.0f
            });

            // Triangle 2: (x1, y0, z10), (x1, y1, z11), (x0, y1, z01)
            mesh.vertices.insert(mesh.vertices.end(), {
                x1, y0, z10,
                x1, y1, z11,
                x0, y1, z01
            });
            mesh.centroids.push_back({
                (x1 + x1 + x0) / 3.0f,
                (y0 + y1 + y1) / 3.0f,
                (z10 + z11 + z01) / 3.0f
            });
        }
    }
    return mesh;
}

// =============================================================================
// Diagnostic & Inspection Helpers
// =============================================================================

inline void print_hex_dump(const void* data, size_t size, std::ostream& out = std::cout) {
    const uint8_t* byteData = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; i += 16) {
        out << "  0x" << std::hex << std::setw(4) << std::setfill('0') << i << ": ";
        for (size_t j = 0; j < 16; ++j) {
            if (i + j < size) {
                out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byteData[i + j]) << " ";
            } else {
                out << "   ";
            }
        }
        out << " | ";
        for (size_t j = 0; j < 16; ++j) {
            if (i + j < size) {
                char c = static_cast<char>(byteData[i + j]);
                out << ((c >= 32 && c <= 126) ? c : '.');
            }
        }
        out << std::dec << "\n";
    }
}

inline void dump_bvh_structure(const void* bufferBase, size_t bufferSize, std::ostream& out = std::cout) {
    if (bufferSize < sizeof(RadvAccelStructHeader)) {
        out << "[ERROR] Buffer too small for RADV AS header\n";
        return;
    }

    const uint8_t* raw = static_cast<const uint8_t*>(bufferBase);
    const auto* header = reinterpret_cast<const RadvAccelStructHeader*>(raw);

    out << "================================================================================\n";
    out << "           RDNA2 RADV ACCELERATION STRUCTURE LAYOUT DECODER                    \n";
    out << "================================================================================\n";
    out << "Header Breakdown:\n";
    out << "  [+0x00] bvh_offset:                0x" << std::hex << header->bvh_offset
        << " (" << std::dec << header->bvh_offset << " bytes)\n";
    out << "  [+0x04] root_flags:                0x" << std::hex << header->root_flags << std::dec << "\n";
    out << "  [+0x08] AABB Min:                  (" << header->aabb.min.x << ", " << header->aabb.min.y << ", " << header->aabb.min.z << ")\n";
    out << "  [+0x14] AABB Max:                  (" << header->aabb.max.x << ", " << header->aabb.max.y << ", " << header->aabb.max.z << ")\n";
    out << "  [+0x20] update_dispatch_size:      [" << header->update_dispatch_size[0] << ", "
        << header->update_dispatch_size[1] << ", " << header->update_dispatch_size[2] << "]\n";
    out << "  [+0x30] compacted_size:            " << header->compacted_size << " bytes\n";
    out << "  [+0x38] serialization_size:        " << header->serialization_size << " bytes\n";
    out << "  [+0x40] copy_dispatch_size:        [" << header->copy_dispatch_size[0] << ", "
        << header->copy_dispatch_size[1] << ", " << header->copy_dispatch_size[2] << "]\n";
    out << "  [+0x50] base size:                 " << header->size << " bytes\n";
    out << "  [+0x58] geometry_type:             " << header->geometry_type << " (0 = Triangles)\n";
    out << "  [+0x5C] geometry_count:            " << header->geometry_count << "\n";
    out << "  [+0x60] leaf_nodes_offset:         0x" << std::hex << header->leaf_nodes_offset
        << " (" << std::dec << header->leaf_nodes_offset << " bytes)\n";
    out << "  [+0x68] primitive_count:           " << header->primitive_count << "\n";
    out << "  [+0x70] instance_count:            " << header->instance_count << "\n";
    out << "  [+0x78] leaf_node_offsets_offset:  0x" << std::hex << header->leaf_node_offsets_offset << std::dec << "\n";
    out << "  [+0x7C] build_flags:               0x" << std::hex << header->build_flags << std::dec << "\n";
    out << "  [+0x80] type:                      " << header->type << " (0 = BLAS, 1 = TLAS)\n";

    uint32_t bvhOffset = header->bvh_offset;
    if (bvhOffset + sizeof(RadvBvhBox32Node) > bufferSize) {
        out << "[ERROR] BVH offset 0x" << std::hex << bvhOffset << " exceeds buffer size\n";
        return;
    }

    out << "\nRoot Box4 Node Breakdown (Offset: 0x" << std::hex << bvhOffset << std::dec << "):\n";
    const auto* rootBox = reinterpret_cast<const RadvBvhBox32Node*>(raw + bvhOffset);
    for (int i = 0; i < 4; ++i) {
        uint32_t childRaw = rootBox->children[i];
        out << "  Child [" << i << "]: raw = 0x" << std::hex << childRaw << std::dec;
        if (childRaw == RADV_BVH_INVALID_NODE) {
            out << " [INVALID / EMPTY]\n";
        } else {
            uint32_t childRelOffset = id_to_offset(childRaw);
            uint32_t childType = id_to_type(childRaw);
            uint32_t childAbsOffset = bvhOffset + childRelOffset;
            out << " -> RelOffset = 0x" << std::hex << childRelOffset
                << " (Abs = 0x" << childAbsOffset << "), Type = "
                << node_type_to_string(childType) << std::dec << "\n";
            out << "           AABB Min: (" << rootBox->coords[i].min.x << ", "
                << rootBox->coords[i].min.y << ", " << rootBox->coords[i].min.z << ")\n";
            out << "           AABB Max: (" << rootBox->coords[i].max.x << ", "
                << rootBox->coords[i].max.y << ", " << rootBox->coords[i].max.z << ")\n";
        }
    }
    out << "  Flags:                             0x" << std::hex << rootBox->flags << std::dec << "\n";

    // Inspect child 0 if triangle
    if (rootBox->children[0] != RADV_BVH_INVALID_NODE && id_to_type(rootBox->children[0]) == RADV_BVH_NODE_TRIANGLE) {
        uint32_t triAbsOffset = bvhOffset + id_to_offset(rootBox->children[0]);
        if (triAbsOffset + sizeof(RadvBvhTriangleNode) <= bufferSize) {
            const auto* tri = reinterpret_cast<const RadvBvhTriangleNode*>(raw + triAbsOffset);
            out << "\nTriangle Node Breakdown (Offset: 0x" << std::hex << triAbsOffset << std::dec << "):\n";
            out << "  Vertex V0:                         (" << tri->coords[0][0] << ", " << tri->coords[0][1] << ", " << tri->coords[0][2] << ")\n";
            out << "  Vertex V1:                         (" << tri->coords[1][0] << ", " << tri->coords[1][1] << ", " << tri->coords[1][2] << ")\n";
            out << "  Vertex V2:                         (" << tri->coords[2][0] << ", " << tri->coords[2][1] << ", " << tri->coords[2][2] << ")\n";
            out << "  Primitive Index (triangle_id):     " << tri->triangle_id << "\n";
            out << "  Geometry Flags:                    0x" << std::hex << tri->geometry_id_and_flags << std::dec
                << " (Opaque: " << ((tri->geometry_id_and_flags & RADV_GEOMETRY_OPAQUE) ? "YES" : "NO") << ")\n";
            out << "  Barycentrics Control (id):         0x" << std::hex << tri->id << " (" << std::dec << tri->id << ")\n";
        }
    }
    out << "================================================================================\n";
}

} // namespace rdna2
