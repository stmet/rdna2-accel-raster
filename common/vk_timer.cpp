#include "vk_timer.hpp"

void GpuTimer::init(uint32_t maxQueries) {
    cleanup();
    queryCount = maxQueries;
    VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpInfo.queryCount = maxQueries;
    VK_CHECK(vkCreateQueryPool(g_vk.device, &qpInfo, nullptr, &queryPool));
}

void GpuTimer::cleanup() {
    if (queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(g_vk.device, queryPool, nullptr);
        queryPool = VK_NULL_HANDLE;
    }
}

void GpuTimer::reset(VkCommandBuffer cmd) {
    vkCmdResetQueryPool(cmd, queryPool, 0, queryCount);
}

void GpuTimer::writeStart(VkCommandBuffer cmd, uint32_t queryIndex) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, queryIndex);
}

void GpuTimer::writeEnd(VkCommandBuffer cmd, uint32_t queryIndex) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, queryIndex);
}

float GpuTimer::getElapsedMs(uint32_t startQuery, uint32_t endQuery) {
    uint64_t timestamps[2];
    VK_CHECK(vkGetQueryPoolResults(
        g_vk.device,
        queryPool,
        startQuery,
        2,
        sizeof(timestamps),
        timestamps,
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    ));

    return static_cast<float>(timestamps[1] - timestamps[0]) * g_vk.timestampPeriod / 1e6f;
}

BenchmarkStats GpuTimer::runBenchmark(
    uint32_t warmupIterations,
    uint32_t measurementIterations,
    const std::function<void(VkCommandBuffer cmd)>& recordCommands
) {
    // Warmup
    for (uint32_t i = 0; i < warmupIterations; i++) {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        recordCommands(cmd);
        g_vk.endSingleTimeCommands(cmd);
    }

    // Measurement
    BenchmarkStats stats;
    stats.samples.reserve(measurementIterations);

    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = g_vk.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(g_vk.device, &allocInfo, &cmd));

    for (uint32_t i = 0; i < measurementIterations; i++) {
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

        vkCmdResetQueryPool(cmd, queryPool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

        recordCommands(cmd);

        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        VK_CHECK(vkQueueSubmit(g_vk.queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(g_vk.queue));

        float ms = getElapsedMs(0, 1);
        stats.samples.push_back(ms);
    }

    vkFreeCommandBuffers(g_vk.device, g_vk.commandPool, 1, &cmd);

    // Compute stats
    std::vector<float> sorted = stats.samples;
    std::sort(sorted.begin(), sorted.end());

    stats.runCount = static_cast<uint32_t>(sorted.size());
    stats.minMs = sorted.front();
    stats.maxMs = sorted.back();
    stats.medianMs = sorted[sorted.size() / 2];

    size_t idxP5 = static_cast<size_t>(0.05f * (sorted.size() - 1));
    size_t idxP95 = static_cast<size_t>(0.95f * (sorted.size() - 1));
    stats.p5Ms = sorted[idxP5];
    stats.p95Ms = sorted[idxP95];

    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    stats.avgMs = static_cast<float>(sum / sorted.size());

    double var = 0.0;
    for (float v : sorted) {
        double diff = v - stats.avgMs;
        var += diff * diff;
    }
    stats.stddevMs = static_cast<float>(std::sqrt(var / sorted.size()));

    return stats;
}

BenchmarkStats GpuTimer::runMultiDispatchBenchmark(
    uint32_t warmupIterations,
    uint32_t measurementIterations,
    uint32_t repeatCount,
    const std::function<void(VkCommandBuffer cmd, uint32_t index)>& recordSinglePass
) {
    if (repeatCount == 0) repeatCount = 1;

    // Warmup
    for (uint32_t i = 0; i < warmupIterations; i++) {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        for (uint32_t r = 0; r < repeatCount; r++) {
            recordSinglePass(cmd, r);
        }
        g_vk.endSingleTimeCommands(cmd);
    }

    // Measurement
    BenchmarkStats stats;
    stats.samples.reserve(measurementIterations);

    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = g_vk.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(g_vk.device, &allocInfo, &cmd));

    for (uint32_t i = 0; i < measurementIterations; i++) {
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

        vkCmdResetQueryPool(cmd, queryPool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

        for (uint32_t r = 0; r < repeatCount; r++) {
            recordSinglePass(cmd, r);
        }

        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        VK_CHECK(vkQueueSubmit(g_vk.queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(g_vk.queue));

        float totalMs = getElapsedMs(0, 1);
        float perDispatchMs = totalMs / static_cast<float>(repeatCount);
        stats.samples.push_back(perDispatchMs);
    }

    vkFreeCommandBuffers(g_vk.device, g_vk.commandPool, 1, &cmd);

    // Compute stats
    std::vector<float> sorted = stats.samples;
    std::sort(sorted.begin(), sorted.end());

    stats.runCount = static_cast<uint32_t>(sorted.size());
    stats.minMs = sorted.front();
    stats.maxMs = sorted.back();
    stats.medianMs = sorted[sorted.size() / 2];

    size_t idxP5 = static_cast<size_t>(0.05f * (sorted.size() - 1));
    size_t idxP95 = static_cast<size_t>(0.95f * (sorted.size() - 1));
    stats.p5Ms = sorted[idxP5];
    stats.p95Ms = sorted[idxP95];

    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    stats.avgMs = static_cast<float>(sum / sorted.size());

    double var = 0.0;
    for (float v : sorted) {
        double diff = v - stats.avgMs;
        var += diff * diff;
    }
    stats.stddevMs = static_cast<float>(std::sqrt(var / sorted.size()));

    return stats;
}

void DualQueueTimer::init(uint32_t maxQueriesPerQueue) {
    cleanup();
    queryCount = maxQueriesPerQueue;

    VkQueryPoolCreateInfo qpInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpInfo.queryCount = maxQueriesPerQueue;

    VK_CHECK(vkCreateQueryPool(g_vk.device, &qpInfo, nullptr, &gfxQueryPool));
    VK_CHECK(vkCreateQueryPool(g_vk.device, &qpInfo, nullptr, &computeQueryPool));
}

void DualQueueTimer::cleanup() {
    if (gfxQueryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(g_vk.device, gfxQueryPool, nullptr);
        gfxQueryPool = VK_NULL_HANDLE;
    }
    if (computeQueryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(g_vk.device, computeQueryPool, nullptr);
        computeQueryPool = VK_NULL_HANDLE;
    }
}

void DualQueueTimer::resetGfx(VkCommandBuffer cmdGfx) {
    vkCmdResetQueryPool(cmdGfx, gfxQueryPool, 0, queryCount);
}

void DualQueueTimer::resetCompute(VkCommandBuffer cmdCompute) {
    vkCmdResetQueryPool(cmdCompute, computeQueryPool, 0, queryCount);
}

void DualQueueTimer::recordGfxStart(VkCommandBuffer cmdGfx, uint32_t queryIdx) {
    vkCmdWriteTimestamp(cmdGfx, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, gfxQueryPool, queryIdx);
}

void DualQueueTimer::recordGfxEnd(VkCommandBuffer cmdGfx, uint32_t queryIdx) {
    vkCmdWriteTimestamp(cmdGfx, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, gfxQueryPool, queryIdx);
}

void DualQueueTimer::recordComputeStart(VkCommandBuffer cmdCompute, uint32_t queryIdx) {
    vkCmdWriteTimestamp(cmdCompute, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, computeQueryPool, queryIdx);
}

void DualQueueTimer::recordComputeEnd(VkCommandBuffer cmdCompute, uint32_t queryIdx) {
    vkCmdWriteTimestamp(cmdCompute, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, computeQueryPool, queryIdx);
}

DualQueueTimer::RunTiming DualQueueTimer::getTiming(
    uint32_t gfxStart, uint32_t gfxEnd,
    uint32_t computeStart, uint32_t computeEnd
) {
    uint64_t gfxTimestamps[2] = {0, 0};
    uint64_t computeTimestamps[2] = {0, 0};

    VK_CHECK(vkGetQueryPoolResults(
        g_vk.device, gfxQueryPool, gfxStart, 2,
        sizeof(gfxTimestamps), gfxTimestamps, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    ));

    VK_CHECK(vkGetQueryPoolResults(
        g_vk.device, computeQueryPool, computeStart, 2,
        sizeof(computeTimestamps), computeTimestamps, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    ));

    RunTiming timing{};
    timing.gfxStartTick = gfxTimestamps[0];
    timing.gfxEndTick = gfxTimestamps[1];
    timing.computeStartTick = computeTimestamps[0];
    timing.computeEndTick = computeTimestamps[1];

    float period = g_vk.timestampPeriod; // ns
    timing.gfxMs = static_cast<float>(timing.gfxEndTick - timing.gfxStartTick) * period / 1e6f;
    timing.computeMs = static_cast<float>(timing.computeEndTick - timing.computeStartTick) * period / 1e6f;

    uint64_t overlapStart = std::max(timing.gfxStartTick, timing.computeStartTick);
    uint64_t overlapEnd = std::min(timing.gfxEndTick, timing.computeEndTick);
    if (overlapEnd > overlapStart) {
        timing.overlapMs = static_cast<float>(overlapEnd - overlapStart) * period / 1e6f;
    } else {
        timing.overlapMs = 0.0f;
    }

    uint64_t totalStart = std::min(timing.gfxStartTick, timing.computeStartTick);
    uint64_t totalEnd = std::max(timing.gfxEndTick, timing.computeEndTick);
    timing.totalWallMs = static_cast<float>(totalEnd - totalStart) * period / 1e6f;

    return timing;
}

DualQueueStats DualQueueTimer::runDualQueueBenchmark(
    uint32_t warmupIterations,
    uint32_t measurementIterations,
    float isolatedComputeMs,
    float isolatedGfxMs,
    const std::function<void(VkCommandBuffer cmdGfx, VkCommandBuffer cmdCompute, VkSemaphore semGfxToCompute, VkSemaphore semComputeToGfx)>& recordAndSubmit
) {
    VkSemaphore semG2C = g_vk.createBinarySemaphore();
    VkSemaphore semC2G = g_vk.createBinarySemaphore();

    VkCommandBufferAllocateInfo allocGfx{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocGfx.commandPool = g_vk.graphicsCommandPool;
    allocGfx.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocGfx.commandBufferCount = 1;
    VkCommandBuffer cmdGfx;
    VK_CHECK(vkAllocateCommandBuffers(g_vk.device, &allocGfx, &cmdGfx));

    VkCommandBufferAllocateInfo allocComp{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocComp.commandPool = g_vk.computeCommandPool;
    allocComp.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocComp.commandBufferCount = 1;
    VkCommandBuffer cmdCompute;
    VK_CHECK(vkAllocateCommandBuffers(g_vk.device, &allocComp, &cmdCompute));

    // Warmup
    for (uint32_t i = 0; i < warmupIterations; i++) {
        recordAndSubmit(cmdGfx, cmdCompute, semG2C, semC2G);
        vkQueueWaitIdle(g_vk.graphicsQueue);
        if (g_vk.hasDedicatedCompute) {
            vkQueueWaitIdle(g_vk.computeQueue);
        }
    }

    DualQueueStats stats;
    stats.totalSamples.reserve(measurementIterations);
    std::vector<float> gfxSamples;
    std::vector<float> compSamples;
    std::vector<float> overlapSamples;
    gfxSamples.reserve(measurementIterations);
    compSamples.reserve(measurementIterations);
    overlapSamples.reserve(measurementIterations);

    for (uint32_t i = 0; i < measurementIterations; i++) {
        recordAndSubmit(cmdGfx, cmdCompute, semG2C, semC2G);
        vkQueueWaitIdle(g_vk.graphicsQueue);
        if (g_vk.hasDedicatedCompute) {
            vkQueueWaitIdle(g_vk.computeQueue);
        }

        RunTiming t = getTiming();
        stats.totalSamples.push_back(t.totalWallMs);
        gfxSamples.push_back(t.gfxMs);
        compSamples.push_back(t.computeMs);
        overlapSamples.push_back(t.overlapMs);
    }

    vkFreeCommandBuffers(g_vk.device, g_vk.graphicsCommandPool, 1, &cmdGfx);
    vkFreeCommandBuffers(g_vk.device, g_vk.computeCommandPool, 1, &cmdCompute);
    vkDestroySemaphore(g_vk.device, semG2C, nullptr);
    vkDestroySemaphore(g_vk.device, semC2G, nullptr);

    std::sort(stats.totalSamples.begin(), stats.totalSamples.end());
    std::sort(gfxSamples.begin(), gfxSamples.end());
    std::sort(compSamples.begin(), compSamples.end());
    std::sort(overlapSamples.begin(), overlapSamples.end());

    stats.totalMedianMs = stats.totalSamples[stats.totalSamples.size() / 2];
    stats.gfxMedianMs = gfxSamples[gfxSamples.size() / 2];
    stats.computeMedianMs = compSamples[compSamples.size() / 2];
    stats.overlapMedianMs = overlapSamples[overlapSamples.size() / 2];

    size_t idxP5 = static_cast<size_t>(0.05f * (stats.totalSamples.size() - 1));
    size_t idxP95 = static_cast<size_t>(0.95f * (stats.totalSamples.size() - 1));
    stats.p5Ms = stats.totalSamples[idxP5];
    stats.p95Ms = stats.totalSamples[idxP95];

    float serialMs = isolatedGfxMs + isolatedComputeMs;
    if (isolatedComputeMs > 1e-4f) {
        stats.efficiency = (serialMs - stats.totalMedianMs) / isolatedComputeMs;
        stats.contentionPenaltyMs = stats.computeMedianMs - isolatedComputeMs;
    }

    return stats;
}
