#pragma once

#include "vk_context.hpp"
#include "vk_buffer.hpp"
#include <vector>

struct VulkanAS {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VulkanBuffer buffer;
    VkDeviceAddress deviceAddress = 0;
    VkAccelerationStructureTypeKHR type = VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR;

    void destroy();
};

class BVHBuilder {
public:
    static VulkanAS buildTriangleBLAS(
        VkDeviceAddress vertexBufferAddress,
        uint32_t vertexCount,
        VkDeviceSize vertexStride,
        VkFormat vertexFormat,
        VkDeviceAddress indexBufferAddress,
        uint32_t indexCount,
        VkIndexType indexType,
        VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        float* outBuildTimeMs = nullptr,
        VkGeometryFlagsKHR geomFlags = VK_GEOMETRY_OPAQUE_BIT_KHR
    );

    static VulkanAS buildAabbBLAS(
        VkDeviceAddress aabbBufferAddress,
        uint32_t aabbCount,
        VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        float* outBuildTimeMs = nullptr
    );

    static VulkanAS buildTLAS(
        VkDeviceAddress instanceBufferAddress,
        uint32_t instanceCount,
        VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        float* outBuildTimeMs = nullptr
    );

    static void updateTLAS(
        VulkanAS& tlas,
        VkDeviceAddress instanceBufferAddress,
        uint32_t instanceCount,
        VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        float* outUpdateTimeMs = nullptr
    );

    static void updateTriangleBLAS(
        VulkanAS& blas,
        VkDeviceAddress vertexBufferAddress,
        uint32_t vertexCount,
        VkDeviceSize vertexStride,
        VkFormat vertexFormat,
        VkDeviceAddress indexBufferAddress,
        uint32_t indexCount,
        VkIndexType indexType,
        VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        float* outUpdateTimeMs = nullptr,
        VkGeometryFlagsKHR geomFlags = VK_GEOMETRY_OPAQUE_BIT_KHR
    );

    static VkDeviceSize getTriangleBLASUpdateScratchSize(
        uint32_t vertexCount,
        VkDeviceSize vertexStride,
        VkFormat vertexFormat,
        uint32_t indexCount,
        VkIndexType indexType,
        VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
    );

    static void cmdUpdateTriangleBLAS(
        VkCommandBuffer cmd,
        VulkanAS& blas,
        VkDeviceAddress vertexBufferAddress,
        uint32_t vertexCount,
        VkDeviceSize vertexStride,
        VkFormat vertexFormat,
        VkDeviceAddress indexBufferAddress,
        uint32_t indexCount,
        VkIndexType indexType,
        VkDeviceAddress scratchAddress,
        VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        VkGeometryFlagsKHR geomFlags = VK_GEOMETRY_OPAQUE_BIT_KHR
    );
};
