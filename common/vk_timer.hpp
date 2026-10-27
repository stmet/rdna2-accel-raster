#pragma once

#include "vk_context.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

struct BenchmarkStats {
    float medianMs = 0.0f;
    float minMs = 0.0f;
    float maxMs = 0.0f;
    float avgMs = 0.0f;
    float stddevMs = 0.0f;
    float p5Ms = 0.0f;
    float p95Ms = 0.0f;
    uint32_t runCount = 0;
    std::vector<float> samples;
};

struct DualQueueStats {
    float gfxMedianMs = 0.0f;
    float computeMedianMs = 0.0f;
    float totalMedianMs = 0.0f;
    float overlapMedianMs = 0.0f;
    float efficiency = 0.0f; // (serialMs - totalOverlappedMs) / computeMs
    float contentionPenaltyMs = 0.0f; // computeOverlapped - computeIsolated
    float p5Ms = 0.0f;
    float p95Ms = 0.0f;
    std::vector<float> totalSamples;
};

class GpuTimer {
public:
    void init(uint32_t maxQueries = 256);
    void cleanup();

    void reset(VkCommandBuffer cmd);
    void writeStart(VkCommandBuffer cmd, uint32_t queryIndex = 0);
    void writeEnd(VkCommandBuffer cmd, uint32_t queryIndex = 1);

    // Read result between two query indices in milliseconds
    float getElapsedMs(uint32_t startQuery = 0, uint32_t endQuery = 1);

    // Helper to benchmark a command recording function
    // warmupIterations: default 20
    // measurementIterations: default 100
    BenchmarkStats runBenchmark(
        uint32_t warmupIterations,
        uint32_t measurementIterations,
        const std::function<void(VkCommandBuffer cmd)>& recordCommands
    );

    // Multi-dispatch benchmark: records repeatCount dispatches inside a single command buffer
    // between writeStart and writeEnd, then divides elapsed time by repeatCount to obtain
    // the precise per-dispatch execution time without submission overhead.
    BenchmarkStats runMultiDispatchBenchmark(
        uint32_t warmupIterations,
        uint32_t measurementIterations,
        uint32_t repeatCount,
        const std::function<void(VkCommandBuffer cmd, uint32_t index)>& recordSinglePass
    );

private:
    VkQueryPool queryPool = VK_NULL_HANDLE;
    uint32_t queryCount = 0;
};

class DualQueueTimer {
public:
    void init(uint32_t maxQueriesPerQueue = 16);
    void cleanup();

    void resetGfx(VkCommandBuffer cmdGfx);
    void resetCompute(VkCommandBuffer cmdCompute);

    void recordGfxStart(VkCommandBuffer cmdGfx, uint32_t queryIdx = 0);
    void recordGfxEnd(VkCommandBuffer cmdGfx, uint32_t queryIdx = 1);
    void recordComputeStart(VkCommandBuffer cmdCompute, uint32_t queryIdx = 0);
    void recordComputeEnd(VkCommandBuffer cmdCompute, uint32_t queryIdx = 1);

    struct RunTiming {
        float gfxMs = 0.0f;
        float computeMs = 0.0f;
        float totalWallMs = 0.0f;
        float overlapMs = 0.0f;
        uint64_t gfxStartTick = 0;
        uint64_t gfxEndTick = 0;
        uint64_t computeStartTick = 0;
        uint64_t computeEndTick = 0;
    };

    RunTiming getTiming(
        uint32_t gfxStart = 0, uint32_t gfxEnd = 1,
        uint32_t computeStart = 0, uint32_t computeEnd = 1
    );

    DualQueueStats runDualQueueBenchmark(
        uint32_t warmupIterations,
        uint32_t measurementIterations,
        float isolatedComputeMs,
        float isolatedGfxMs,
        const std::function<void(VkCommandBuffer cmdGfx, VkCommandBuffer cmdCompute, VkSemaphore semGfxToCompute, VkSemaphore semComputeToGfx)>& recordAndSubmit
    );

private:
    VkQueryPool gfxQueryPool = VK_NULL_HANDLE;
    VkQueryPool computeQueryPool = VK_NULL_HANDLE;
    uint32_t queryCount = 0;
};
