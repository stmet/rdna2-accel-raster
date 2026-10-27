#include "vk_bvh.hpp"
#include <iostream>

void VulkanAS::destroy() {
    if (handle != VK_NULL_HANDLE) {
        g_vk.vkDestroyAccelerationStructureKHR(g_vk.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    buffer.destroy();
    deviceAddress = 0;
}

VulkanAS BVHBuilder::buildTriangleBLAS(
    VkDeviceAddress vertexBufferAddress,
    uint32_t vertexCount,
    VkDeviceSize vertexStride,
    VkFormat vertexFormat,
    VkDeviceAddress indexBufferAddress,
    uint32_t indexCount,
    VkIndexType indexType,
    VkBuildAccelerationStructureFlagsKHR flags,
    float* outBuildTimeMs,
    VkGeometryFlagsKHR geomFlags
) {
    VulkanAS blas;
    blas.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    uint32_t primitiveCount = (indexCount > 0) ? (indexCount / 3) : (vertexCount / 3);

    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = geomFlags;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = vertexFormat;
    geometry.geometry.triangles.vertexData.deviceAddress = vertexBufferAddress;
    geometry.geometry.triangles.vertexStride = vertexStride;
    geometry.geometry.triangles.maxVertex = vertexCount;
    if (indexCount > 0) {
        geometry.geometry.triangles.indexType = indexType;
        geometry.geometry.triangles.indexData.deviceAddress = indexBufferAddress;
    } else {
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = flags;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    g_vk.vkGetAccelerationStructureBuildSizesKHR(
        g_vk.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &primitiveCount,
        &sizeInfo
    );

    // Buffer for AS
    blas.buffer.create(
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = blas.buffer.handle;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VK_CHECK(g_vk.vkCreateAccelerationStructureKHR(g_vk.device, &createInfo, nullptr, &blas.handle));

    // Get AS device address
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = blas.handle;
    blas.deviceAddress = g_vk.vkGetAccelerationStructureDeviceAddressKHR(g_vk.device, &addressInfo);

    // Scratch buffer
    VulkanBuffer scratchBuffer;
    scratchBuffer.create(
        sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    buildInfo.dstAccelerationStructure = blas.handle;
    buildInfo.scratchData.deviceAddress = scratchBuffer.getDeviceAddress();

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex = 0;
    rangeInfo.transformOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos = &rangeInfo;

    // Timestamp query if requested
    VkQueryPool queryPool = VK_NULL_HANDLE;
    if (outBuildTimeMs) {
        VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(g_vk.device, &qpInfo, nullptr, &queryPool));
    }

    VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
    if (outBuildTimeMs) {
        vkCmdResetQueryPool(cmd, queryPool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
    }

    g_vk.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfos);

    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    if (outBuildTimeMs) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
    }
    g_vk.endSingleTimeCommands(cmd);

    if (outBuildTimeMs) {
        uint64_t timestamps[2];
        VK_CHECK(vkGetQueryPoolResults(g_vk.device, queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        *outBuildTimeMs = static_cast<float>(timestamps[1] - timestamps[0]) * g_vk.timestampPeriod / 1e6f;
        vkDestroyQueryPool(g_vk.device, queryPool, nullptr);
    }

    scratchBuffer.destroy();
    return blas;
}

VulkanAS BVHBuilder::buildAabbBLAS(
    VkDeviceAddress aabbBufferAddress,
    uint32_t aabbCount,
    VkBuildAccelerationStructureFlagsKHR flags,
    float* outBuildTimeMs
) {
    VulkanAS blas;
    blas.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geometry.geometry.aabbs.data.deviceAddress = aabbBufferAddress;
    geometry.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = flags;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    g_vk.vkGetAccelerationStructureBuildSizesKHR(
        g_vk.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &aabbCount,
        &sizeInfo
    );

    blas.buffer.create(
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = blas.buffer.handle;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VK_CHECK(g_vk.vkCreateAccelerationStructureKHR(g_vk.device, &createInfo, nullptr, &blas.handle));

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = blas.handle;
    blas.deviceAddress = g_vk.vkGetAccelerationStructureDeviceAddressKHR(g_vk.device, &addressInfo);

    VulkanBuffer scratchBuffer;
    scratchBuffer.create(
        sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    buildInfo.dstAccelerationStructure = blas.handle;
    buildInfo.scratchData.deviceAddress = scratchBuffer.getDeviceAddress();

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = aabbCount;
    rangeInfo.primitiveOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos = &rangeInfo;

    VkQueryPool queryPool = VK_NULL_HANDLE;
    if (outBuildTimeMs) {
        VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(g_vk.device, &qpInfo, nullptr, &queryPool));
    }

    VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
    if (outBuildTimeMs) {
        vkCmdResetQueryPool(cmd, queryPool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
    }

    g_vk.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfos);

    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    if (outBuildTimeMs) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
    }
    g_vk.endSingleTimeCommands(cmd);

    if (outBuildTimeMs) {
        uint64_t timestamps[2];
        VK_CHECK(vkGetQueryPoolResults(g_vk.device, queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        *outBuildTimeMs = static_cast<float>(timestamps[1] - timestamps[0]) * g_vk.timestampPeriod / 1e6f;
        vkDestroyQueryPool(g_vk.device, queryPool, nullptr);
    }

    scratchBuffer.destroy();
    return blas;
}

VulkanAS BVHBuilder::buildTLAS(
    VkDeviceAddress instanceBufferAddress,
    uint32_t instanceCount,
    VkBuildAccelerationStructureFlagsKHR flags,
    float* outBuildTimeMs
) {
    VulkanAS tlas;
    tlas.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = instanceBufferAddress;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = flags;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    g_vk.vkGetAccelerationStructureBuildSizesKHR(
        g_vk.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &instanceCount,
        &sizeInfo
    );

    tlas.buffer.create(
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = tlas.buffer.handle;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    VK_CHECK(g_vk.vkCreateAccelerationStructureKHR(g_vk.device, &createInfo, nullptr, &tlas.handle));

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = tlas.handle;
    tlas.deviceAddress = g_vk.vkGetAccelerationStructureDeviceAddressKHR(g_vk.device, &addressInfo);

    VulkanBuffer scratchBuffer;
    scratchBuffer.create(
        sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    buildInfo.dstAccelerationStructure = tlas.handle;
    buildInfo.scratchData.deviceAddress = scratchBuffer.getDeviceAddress();

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;
    rangeInfo.primitiveOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos = &rangeInfo;

    VkQueryPool queryPool = VK_NULL_HANDLE;
    if (outBuildTimeMs) {
        VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(g_vk.device, &qpInfo, nullptr, &queryPool));
    }

    VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
    if (outBuildTimeMs) {
        vkCmdResetQueryPool(cmd, queryPool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
    }

    g_vk.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfos);

    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    if (outBuildTimeMs) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
    }
    g_vk.endSingleTimeCommands(cmd);

    if (outBuildTimeMs) {
        uint64_t timestamps[2];
        VK_CHECK(vkGetQueryPoolResults(g_vk.device, queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        *outBuildTimeMs = static_cast<float>(timestamps[1] - timestamps[0]) * g_vk.timestampPeriod / 1e6f;
        vkDestroyQueryPool(g_vk.device, queryPool, nullptr);
    }

    scratchBuffer.destroy();
    return tlas;
}

void BVHBuilder::updateTLAS(
    VulkanAS& tlas,
    VkDeviceAddress instanceBufferAddress,
    uint32_t instanceCount,
    VkBuildAccelerationStructureFlagsKHR flags,
    float* outUpdateTimeMs
) {
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = instanceBufferAddress;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = flags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = tlas.handle;
    buildInfo.dstAccelerationStructure = tlas.handle;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    g_vk.vkGetAccelerationStructureBuildSizesKHR(
        g_vk.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &instanceCount,
        &sizeInfo
    );

    VulkanBuffer scratchBuffer;
    scratchBuffer.create(
        sizeInfo.updateScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    buildInfo.scratchData.deviceAddress = scratchBuffer.getDeviceAddress();

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;
    rangeInfo.primitiveOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos = &rangeInfo;

    VkQueryPool queryPool = VK_NULL_HANDLE;
    if (outUpdateTimeMs) {
        VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(g_vk.device, &qpInfo, nullptr, &queryPool));
    }

    VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
    if (outUpdateTimeMs) {
        vkCmdResetQueryPool(cmd, queryPool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
    }

    g_vk.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfos);

    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    if (outUpdateTimeMs) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
    }
    g_vk.endSingleTimeCommands(cmd);

    if (outUpdateTimeMs) {
        uint64_t timestamps[2];
        VK_CHECK(vkGetQueryPoolResults(g_vk.device, queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        *outUpdateTimeMs = static_cast<float>(timestamps[1] - timestamps[0]) * g_vk.timestampPeriod / 1e6f;
        vkDestroyQueryPool(g_vk.device, queryPool, nullptr);
    }

    scratchBuffer.destroy();
}

void BVHBuilder::updateTriangleBLAS(
    VulkanAS& blas,
    VkDeviceAddress vertexBufferAddress,
    uint32_t vertexCount,
    VkDeviceSize vertexStride,
    VkFormat vertexFormat,
    VkDeviceAddress indexBufferAddress,
    uint32_t indexCount,
    VkIndexType indexType,
    VkBuildAccelerationStructureFlagsKHR flags,
    float* outUpdateTimeMs,
    VkGeometryFlagsKHR geomFlags
) {
    uint32_t primitiveCount = (indexCount > 0) ? (indexCount / 3) : (vertexCount / 3);

    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = geomFlags;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = vertexFormat;
    geometry.geometry.triangles.vertexData.deviceAddress = vertexBufferAddress;
    geometry.geometry.triangles.vertexStride = vertexStride;
    geometry.geometry.triangles.maxVertex = vertexCount;
    if (indexCount > 0) {
        geometry.geometry.triangles.indexType = indexType;
        geometry.geometry.triangles.indexData.deviceAddress = indexBufferAddress;
    } else {
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = flags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = blas.handle;
    buildInfo.dstAccelerationStructure = blas.handle;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    g_vk.vkGetAccelerationStructureBuildSizesKHR(
        g_vk.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &primitiveCount,
        &sizeInfo
    );

    VulkanBuffer scratchBuffer;
    scratchBuffer.create(
        sizeInfo.updateScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    buildInfo.scratchData.deviceAddress = scratchBuffer.getDeviceAddress();

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    rangeInfo.primitiveOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos = &rangeInfo;

    VkQueryPool queryPool = VK_NULL_HANDLE;
    if (outUpdateTimeMs) {
        VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(g_vk.device, &qpInfo, nullptr, &queryPool));
    }

    VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
    if (outUpdateTimeMs) {
        vkCmdResetQueryPool(cmd, queryPool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
    }

    g_vk.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfos);

    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    if (outUpdateTimeMs) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
    }
    g_vk.endSingleTimeCommands(cmd);

    if (outUpdateTimeMs) {
        uint64_t timestamps[2];
        VK_CHECK(vkGetQueryPoolResults(g_vk.device, queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        *outUpdateTimeMs = static_cast<float>(timestamps[1] - timestamps[0]) * g_vk.timestampPeriod / 1e6f;
        vkDestroyQueryPool(g_vk.device, queryPool, nullptr);
    }

    scratchBuffer.destroy();
}

VkDeviceSize BVHBuilder::getTriangleBLASUpdateScratchSize(
    uint32_t vertexCount,
    VkDeviceSize vertexStride,
    VkFormat vertexFormat,
    uint32_t indexCount,
    VkIndexType indexType,
    VkBuildAccelerationStructureFlagsKHR flags
) {
    uint32_t primitiveCount = (indexCount > 0) ? (indexCount / 3) : (vertexCount / 3);

    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = vertexFormat;
    geometry.geometry.triangles.vertexData.deviceAddress = 0x1000;
    geometry.geometry.triangles.vertexStride = vertexStride;
    geometry.geometry.triangles.maxVertex = vertexCount;
    geometry.geometry.triangles.indexType = (indexCount > 0) ? indexType : VK_INDEX_TYPE_NONE_KHR;
    geometry.geometry.triangles.indexData.deviceAddress = 0x2000;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = flags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    g_vk.vkGetAccelerationStructureBuildSizesKHR(
        g_vk.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &primitiveCount,
        &sizeInfo
    );

    return sizeInfo.updateScratchSize;
}

void BVHBuilder::cmdUpdateTriangleBLAS(
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
    VkBuildAccelerationStructureFlagsKHR flags,
    VkGeometryFlagsKHR geomFlags
) {
    uint32_t primitiveCount = (indexCount > 0) ? (indexCount / 3) : (vertexCount / 3);

    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = geomFlags;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = vertexFormat;
    geometry.geometry.triangles.vertexData.deviceAddress = vertexBufferAddress;
    geometry.geometry.triangles.vertexStride = vertexStride;
    geometry.geometry.triangles.maxVertex = vertexCount;
    if (indexCount > 0) {
        geometry.geometry.triangles.indexType = indexType;
        geometry.geometry.triangles.indexData.deviceAddress = indexBufferAddress;
    } else {
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = flags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = blas.handle;
    buildInfo.dstAccelerationStructure = blas.handle;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    buildInfo.scratchData.deviceAddress = scratchAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    rangeInfo.primitiveOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos = &rangeInfo;

    g_vk.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfos);

    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}
