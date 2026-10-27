#include "vk_context.hpp"
#include "vk_buffer.hpp"
#include "vk_bvh.hpp"
#include "vk_shader.hpp"
#include "vk_timer.hpp"
#include "csv_writer.hpp"
#include "mesh_gen.hpp"
#include "software_bvh.hpp"
#include "gpu_clock.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <filesystem>
#include <cassert>

struct ProceduralPushConstants {
    glm::vec4 camPos;
    glm::vec4 lookAt;
    uint32_t rayCount;
    uint32_t isCoherent;
};

struct RayStats {
    uint32_t hit;
    uint32_t boxTests;
    uint32_t triTests;
    uint32_t pad;
};

int main() {
    std::cout << "====================================================\n";
    std::cout << " PHASE 1 AUDIT: RDNA2 RAY ACCELERATOR MICROBENCHMARK\n";
    std::cout << "        COHERENT/INCOHERENT 1K - 10M TRIANGLES\n";
    std::cout << "====================================================\n";

    g_vk.init(true);
    GpuClockInfo::printStatus();

    std::filesystem::create_directories("results");
    CsvWriter bvhCsv("results/exp0_bvh_build.csv", {
        "Triangles", "Build_FastTrace_ms", "Build_FastBuild_ms", "Update_Refit_ms", "BLAS_Size_MB"
    });

    CsvWriter throughputCsv("results/exp0_throughput.csv", {
        "Triangles", "RayPattern", "Backend", "RaySource", "Rays", 
        "Time_Median_ms", "Time_P5_ms", "Time_P95_ms", "Throughput_MRays_per_sec", "Speedup"
    });

    CsvWriter statsCsv("results/exp0_ray_stats.csv", {
        "Triangles", "RayPattern",
        "HitRate_HW_pct", "HitRate_SW_pct", "HitMatch_pct",
        "AvgBoxTests_PerRay", "AvgBoxTests_HitOnly",
        "AvgTriTests_PerRay", "AvgTriTests_HitOnly",
        "HW_All_MRays_per_sec", "HW_HitOnly_MRays_per_sec",
        "SW_All_MRays_per_sec", "SW_HitOnly_MRays_per_sec",
        "Speedup_AllRays", "Speedup_HitOnly"
    });

    std::vector<uint32_t> triangleCounts = {1000, 100000, 1000000, 10000000};
    const uint32_t rayCount = 1024 * 1024; // 1M rays

    std::cout << "Compiling procedural ray shaders..." << std::endl;
    std::vector<uint32_t> procRqSpv = ShaderUtils::compileGlslFileToSpirv("exp0_probe/shaders/probe_procedural.comp");
    VkShaderModule procRqModule = ShaderUtils::createShaderModule(procRqSpv);

    std::vector<uint32_t> procSwSpv = ShaderUtils::compileGlslFileToSpirv("exp0_probe/shaders/software_bvh_procedural.comp");
    VkShaderModule procSwModule = ShaderUtils::createShaderModule(procSwSpv);

    std::vector<uint32_t> procSwLdsSpv = ShaderUtils::compileGlslFileToSpirv("exp0_probe/shaders/software_bvh_lds.comp");
    VkShaderModule procSwLdsModule = ShaderUtils::createShaderModule(procSwLdsSpv);

    std::vector<uint32_t> procSwBvh4Spv = ShaderUtils::compileGlslFileToSpirv("exp0_probe/shaders/software_bvh_bvh4.comp");
    VkShaderModule procSwBvh4Module = ShaderUtils::createShaderModule(procSwBvh4Spv);

    GpuTimer timer;
    timer.init(64);

    for (uint32_t triTarget : triangleCounts) {
        std::cout << "\n----------------------------------------------------\n";
        std::cout << ">>> BENCHMARKING SCENE: ~" << triTarget << " Triangles <<<\n";
        std::cout << "----------------------------------------------------\n";

        float sceneSize = 100.0f;
        SceneMesh mesh = MeshGenerator::createTessellatedTerrain(triTarget, sceneSize);
        uint32_t actualTris = mesh.triangleCount();
        std::cout << "Generated mesh with " << actualTris << " triangles and " 
                  << mesh.vertices.size() << " vertices.\n";

        VulkanBuffer vBuf, iBuf;
        vBuf.create(
            sizeof(SceneMesh::Vertex) * mesh.vertices.size(),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        vBuf.upload(mesh.vertices.data(), sizeof(SceneMesh::Vertex) * mesh.vertices.size());

        iBuf.create(
            sizeof(uint32_t) * mesh.indices.size(),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        iBuf.upload(mesh.indices.data(), sizeof(uint32_t) * mesh.indices.size());

        // 1. BVH Build Benchmark
        std::cout << "1. Measuring BVH build times..." << std::endl;
        float buildFastTraceTime = 0.0f;
        VulkanAS blasFastTrace = BVHBuilder::buildTriangleBLAS(
            vBuf.getDeviceAddress(), mesh.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            iBuf.getDeviceAddress(), mesh.indices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
            &buildFastTraceTime
        );

        float buildFastBuildTime = 0.0f;
        VulkanAS blasFastBuild = BVHBuilder::buildTriangleBLAS(
            vBuf.getDeviceAddress(), mesh.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            iBuf.getDeviceAddress(), mesh.indices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            &buildFastBuildTime
        );

        float updateTime = 0.0f;
        BVHBuilder::updateTriangleBLAS(
            blasFastTrace,
            vBuf.getDeviceAddress(), mesh.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            iBuf.getDeviceAddress(), mesh.indices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            &updateTime
        );

        float blasSizeMB = static_cast<float>(blasFastTrace.buffer.size) / (1024.0f * 1024.0f);
        std::cout << "  - Build (PREFER_FAST_TRACE): " << buildFastTraceTime << " ms\n";
        std::cout << "  - Build (PREFER_FAST_BUILD): " << buildFastBuildTime << " ms\n";
        std::cout << "  - Refit / Update:           " << updateTime << " ms\n";
        std::cout << "  - BLAS Memory Size:          " << blasSizeMB << " MB\n";

        bvhCsv.writeRow({
            std::to_string(actualTris),
            std::to_string(buildFastTraceTime),
            std::to_string(buildFastBuildTime),
            std::to_string(updateTime),
            std::to_string(blasSizeMB)
        });

        // 2. Build TLAS
        VkTransformMatrixKHR transform = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f
        };
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform = transform;
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blasFastTrace.deviceAddress;

        VulkanBuffer instBuf;
        instBuf.create(sizeof(inst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        instBuf.upload(&inst, sizeof(inst));
        VulkanAS tlas = BVHBuilder::buildTLAS(instBuf.getDeviceAddress(), 1);

        // 3. Build Software BVH (skip for 10M to prevent host memory exhaustion, or build for <= 1M)
        bool hasSwBvh = (actualTris <= 1000000);
        SoftwareBVH swBvh;
        SoftwareBVH4 swBvh4;
        VulkanBuffer swNodeBuf, swTriBuf, swBvh4NodeBuf;
        if (hasSwBvh) {
            std::cout << "2. Building Software BVH on CPU for baseline comparison..." << std::endl;
            swBvh = SoftwareBVH::build(mesh);
            std::cout << "  - Software BVH2 nodes: " << swBvh.nodes.size() << ", prims: " << swBvh.triangles.size() << "\n";

            swNodeBuf.create(sizeof(GpuBvhNode) * swBvh.nodes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            swNodeBuf.upload(swBvh.nodes.data(), sizeof(GpuBvhNode) * swBvh.nodes.size());

            swTriBuf.create(sizeof(GpuTriangle) * swBvh.triangles.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            swTriBuf.upload(swBvh.triangles.data(), sizeof(GpuTriangle) * swBvh.triangles.size());

            // Build Wide BVH4
            swBvh4 = SoftwareBVH4::fromBVH2(swBvh);
            std::cout << "  - Software BVH4 nodes: " << swBvh4.nodes.size() 
                      << " (collapsed " << std::fixed << std::setprecision(2) 
                      << (float)swBvh.nodes.size() / swBvh4.nodes.size() << "x from BVH2)\n";

            swBvh4NodeBuf.create(sizeof(GpuBvh4Node) * swBvh4.nodes.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            swBvh4NodeBuf.upload(swBvh4.nodes.data(), sizeof(GpuBvh4Node) * swBvh4.nodes.size());
        }

        // 4. Output Buffers
        VulkanBuffer rqResultBuf, swResultBuf, swBvh4StatsBuf;
        rqResultBuf.create(sizeof(uint32_t) * rayCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        swResultBuf.create(sizeof(uint32_t) * rayCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (hasSwBvh) {
            swBvh4StatsBuf.create(sizeof(RayStats) * rayCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }

        // 5. Pipelines & Descriptors
        // Procedural Hardware Ray Query Pipeline
        VkDescriptorSetLayoutBinding rqBindings[2]{
            {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
        };
        VkDescriptorSetLayoutCreateInfo rqLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, rqBindings};
        VkDescriptorSetLayout rqSetLayout;
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &rqLayoutInfo, nullptr, &rqSetLayout));

        VkPushConstantRange procPcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ProceduralPushConstants)};
        VkPipelineLayoutCreateInfo rqPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &rqSetLayout, 1, &procPcRange};
        VkPipelineLayout rqPipelineLayout;
        VK_CHECK(vkCreatePipelineLayout(g_vk.device, &rqPipeLayoutInfo, nullptr, &rqPipelineLayout));
        VkPipeline rqPipeline = ShaderUtils::createComputePipeline(procRqModule, rqPipelineLayout);

        // Procedural Software BVH Pipeline
        VkDescriptorSetLayoutBinding swBindings[3]{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
        };
        VkDescriptorSetLayoutCreateInfo swLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, swBindings};
        VkDescriptorSetLayout swSetLayout;
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &swLayoutInfo, nullptr, &swSetLayout));

        VkPipelineLayoutCreateInfo swPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &swSetLayout, 1, &procPcRange};
        VkPipelineLayout swPipelineLayout;
        VK_CHECK(vkCreatePipelineLayout(g_vk.device, &swPipeLayoutInfo, nullptr, &swPipelineLayout));
        VkPipeline swPipeline = ShaderUtils::createComputePipeline(procSwModule, swPipelineLayout);
        VkPipeline swLdsPipeline = ShaderUtils::createComputePipeline(procSwLdsModule, swPipelineLayout);
        VkPipeline swBvh4Pipeline = ShaderUtils::createComputePipeline(procSwBvh4Module, swPipelineLayout);

        // Descriptors
        VkDescriptorPoolSize dPoolSizes[2]{
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10}
        };
        VkDescriptorPoolCreateInfo dPoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 4, 2, dPoolSizes};
        VkDescriptorPool descPool;
        VK_CHECK(vkCreateDescriptorPool(g_vk.device, &dPoolInfo, nullptr, &descPool));

        VkDescriptorSet rqSet, swSet, swBvh4Set;
        VkDescriptorSetAllocateInfo allocSetInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &rqSetLayout};
        VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocSetInfo, &rqSet));

        allocSetInfo.pSetLayouts = &swSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocSetInfo, &swSet));
        VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocSetInfo, &swBvh4Set));

        // Write Descriptors
        VkWriteDescriptorSetAccelerationStructureKHR asWriteInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &tlas.handle};
        VkDescriptorBufferInfo rqResBufInfo{rqResultBuf.handle, 0, VK_WHOLE_SIZE};

        VkWriteDescriptorSet rqWrites[2]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asWriteInfo, rqSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, rqSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &rqResBufInfo, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 2, rqWrites, 0, nullptr);

        if (hasSwBvh) {
            VkDescriptorBufferInfo swNodeInfo{swNodeBuf.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo swTriInfo{swTriBuf.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo swResBufInfo{swResultBuf.handle, 0, VK_WHOLE_SIZE};

            VkWriteDescriptorSet swWrites[3]{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, swSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &swNodeInfo, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, swSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &swTriInfo, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, swSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &swResBufInfo, nullptr}
            };
            vkUpdateDescriptorSets(g_vk.device, 3, swWrites, 0, nullptr);

            VkDescriptorBufferInfo swBvh4NodeInfo{swBvh4NodeBuf.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo swBvh4StatsInfo{swBvh4StatsBuf.handle, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet swBvh4Writes[3]{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, swBvh4Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &swBvh4NodeInfo, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, swBvh4Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &swTriInfo, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, swBvh4Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &swBvh4StatsInfo, nullptr}
            };
            vkUpdateDescriptorSets(g_vk.device, 3, swBvh4Writes, 0, nullptr);
        }

        // 6. Benchmark Procedural Coherent vs Incoherent
        std::vector<std::pair<std::string, uint32_t>> patterns = {{"Coherent", 1}, {"Incoherent", 0}};

        for (const auto& [patternName, isCoherent] : patterns) {
            std::cout << "\n--- Testing Procedural " << patternName << " Rays (1,048,576 rays) ---" << std::endl;

            ProceduralPushConstants pc{};
            pc.camPos = glm::vec4(0.0f, 50.0f, -45.0f, 1.0f);
            pc.lookAt = glm::vec4(0.0f, 0.0f, 5.0f, 1.0f);
            pc.rayCount = rayCount;
            pc.isCoherent = isCoherent;

            // A. Hardware RA Benchmark (100 iterations, 20 warmup)
            BenchmarkStats rqStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rqPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rqPipelineLayout, 0, 1, &rqSet, 0, nullptr);
                vkCmdPushConstants(cmd, rqPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ProceduralPushConstants), &pc);
                vkCmdDispatch(cmd, rayCount / 64, 1, 1);
            });

            float rqThroughput = (rayCount / (rqStats.medianMs * 1e-3f)) / 1e6f;
            std::cout << "  [Hardware RA] Median: " << rqStats.medianMs << " ms (p5: " 
                      << rqStats.p5Ms << ", p95: " << rqStats.p95Ms << ") -> " << rqThroughput << " MRays/sec\n";

            float swThroughput = 0.0f;
            float swLdsThroughput = 0.0f;
            float swBvh4Throughput = 0.0f;
            float speedup = 1.0f;
            float speedupLds = 1.0f;
            float speedupBvh4 = 1.0f;
            BenchmarkStats swStats{};
            BenchmarkStats swLdsStats{};
            BenchmarkStats swBvh4Stats{};

            if (hasSwBvh) {
                // 1. Software ALU with private scratch stack
                swStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swPipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swPipelineLayout, 0, 1, &swSet, 0, nullptr);
                    vkCmdPushConstants(cmd, swPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ProceduralPushConstants), &pc);
                    vkCmdDispatch(cmd, rayCount / 64, 1, 1);
                });
                swThroughput = (rayCount / (swStats.medianMs * 1e-3f)) / 1e6f;
                speedup = rqThroughput / swThroughput;

                // 2. Software ALU with on-chip LDS stack
                swLdsStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swLdsPipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swPipelineLayout, 0, 1, &swSet, 0, nullptr);
                    vkCmdPushConstants(cmd, swPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ProceduralPushConstants), &pc);
                    vkCmdDispatch(cmd, rayCount / 64, 1, 1);
                });
                swLdsThroughput = (rayCount / (swLdsStats.medianMs * 1e-3f)) / 1e6f;
                speedupLds = rqThroughput / swLdsThroughput;

                // 3. Software ALU Wide BVH4 with ordered traversal & LDS stack
                swBvh4Stats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swBvh4Pipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swPipelineLayout, 0, 1, &swBvh4Set, 0, nullptr);
                    vkCmdPushConstants(cmd, swPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ProceduralPushConstants), &pc);
                    vkCmdDispatch(cmd, rayCount / 64, 1, 1);
                });
                swBvh4Throughput = (rayCount / (swBvh4Stats.medianMs * 1e-3f)) / 1e6f;
                speedupBvh4 = rqThroughput / swBvh4Throughput;

                std::cout << "  [Software ALU BVH (Scratch)] Median: " << swStats.medianMs << " ms (p5: " 
                          << swStats.p5Ms << ", p95: " << swStats.p95Ms << ") -> " << swThroughput << " MRays/sec\n";
                std::cout << "  [Software ALU BVH (LDS)]     Median: " << swLdsStats.medianMs << " ms (p5: " 
                          << swLdsStats.p5Ms << ", p95: " << swLdsStats.p95Ms << ") -> " << swLdsThroughput << " MRays/sec\n";
                std::cout << "  [Software ALU BVH (BVH4)]    Median: " << swBvh4Stats.medianMs << " ms (p5: " 
                          << swBvh4Stats.p5Ms << ", p95: " << swBvh4Stats.p95Ms << ") -> " << swBvh4Throughput << " MRays/sec\n";
                std::cout << "  => RA Hardware Speedup over Best SW (BVH4): " << std::fixed << std::setprecision(2) 
                          << speedupBvh4 << "x (over BVH2-LDS: " << speedupLds << "x, over Scratch: " << speedup << "x)\n";

                // Correctness check: verify hit count and ray statistics
                std::vector<uint32_t> rqHits(rayCount);
                std::vector<RayStats> swStatsVec(rayCount);
                rqResultBuf.download(rqHits.data(), sizeof(uint32_t) * rayCount);
                swBvh4StatsBuf.download(swStatsVec.data(), sizeof(RayStats) * rayCount);

                uint32_t hwHits = 0;
                uint32_t swHits = 0;
                uint32_t matches = 0;
                uint64_t totalBoxTests = 0;
                uint64_t totalBoxTestsHit = 0;
                uint64_t totalTriTests = 0;
                uint64_t totalTriTestsHit = 0;

                for (size_t i = 0; i < rayCount; i++) {
                    if (rqHits[i] > 0) hwHits++;
                    if (swStatsVec[i].hit > 0) {
                        swHits++;
                        totalBoxTestsHit += swStatsVec[i].boxTests;
                        totalTriTestsHit += swStatsVec[i].triTests;
                    }
                    totalBoxTests += swStatsVec[i].boxTests;
                    totalTriTests += swStatsVec[i].triTests;
                    if ((rqHits[i] > 0) == (swStatsVec[i].hit > 0)) matches++;
                }

                float hwHitRate = (static_cast<float>(hwHits) / rayCount) * 100.0f;
                float swHitRate = (static_cast<float>(swHits) / rayCount) * 100.0f;
                float matchPct = (static_cast<float>(matches) / rayCount) * 100.0f;

                float avgBoxPerRay = static_cast<float>(totalBoxTests) / rayCount;
                float avgBoxHit = swHits > 0 ? static_cast<float>(totalBoxTestsHit) / swHits : 0.0f;
                float avgTriPerRay = static_cast<float>(totalTriTests) / rayCount;
                float avgTriHit = swHits > 0 ? static_cast<float>(totalTriTestsHit) / swHits : 0.0f;

                float hwAllMRays = rqThroughput;
                float hwHitOnlyMRays = (hwHits / (rqStats.medianMs * 1e-3f)) / 1e6f;
                float swAllMRays = swBvh4Throughput;
                float swHitOnlyMRays = (swHits / (swBvh4Stats.medianMs * 1e-3f)) / 1e6f;

                float speedupAll = hwAllMRays / (swAllMRays > 0.0f ? swAllMRays : 1.0f);
                float speedupHit = hwHitOnlyMRays / (swHitOnlyMRays > 0.0f ? swHitOnlyMRays : 1.0f);

                std::cout << "  [Ray Statistics] Hit Rate: HW=" << std::fixed << std::setprecision(2) << hwHitRate 
                          << "% (" << hwHits << "/" << rayCount << "), SW=" << swHitRate 
                          << "%, Match=" << matchPct << "%\n";
                std::cout << "  [BVH4 Traversal Complexity] Avg Box Tests: " << std::setprecision(1) << avgBoxPerRay 
                          << " (Hit-only: " << avgBoxHit << ") | Avg Tri Tests: " << avgTriPerRay 
                          << " (Hit-only: " << avgTriHit << ")\n";
                std::cout << "  [Throughput Breakdown]\n";
                std::cout << "    - Hardware RA:   " << std::setprecision(2) << hwAllMRays << " MRays/s (All) | " 
                          << hwHitOnlyMRays << " MRays/s (Hit-only)\n";
                std::cout << "    - Software BVH4: " << swAllMRays << " MRays/s (All) | " 
                          << swHitOnlyMRays << " MRays/s (Hit-only)\n";
                std::cout << "    => Speedup HW vs Best SW (BVH4): " << speedupAll << "x (All Rays) | " 
                          << speedupHit << "x (Hit-Only Rays)\n";

                statsCsv.writeRow(std::vector<std::string>{
                    std::to_string(actualTris),
                    patternName,
                    std::to_string(hwHitRate),
                    std::to_string(swHitRate),
                    std::to_string(matchPct),
                    std::to_string(avgBoxPerRay),
                    std::to_string(avgBoxHit),
                    std::to_string(avgTriPerRay),
                    std::to_string(avgTriHit),
                    std::to_string(hwAllMRays),
                    std::to_string(hwHitOnlyMRays),
                    std::to_string(swAllMRays),
                    std::to_string(swHitOnlyMRays),
                    std::to_string(speedupAll),
                    std::to_string(speedupHit)
                });

                if (matchPct < 95.0f) {
                    std::cerr << "AUDIT ERROR: Ray query results diverge significantly from reference BVH!\n";
                    assert(false);
                }
            } else {
                std::cout << "  [Software ALU BVH] Skipped for 10M triangles (ALU memory limit)\n";
                std::vector<uint32_t> rqHits(rayCount);
                rqResultBuf.download(rqHits.data(), sizeof(uint32_t) * rayCount);
                uint32_t hwHits = 0;
                for (size_t i = 0; i < rayCount; i++) {
                    if (rqHits[i] > 0) hwHits++;
                }
                float hwHitRate = (static_cast<float>(hwHits) / rayCount) * 100.0f;
                float hwHitOnlyMRays = (hwHits / (rqStats.medianMs * 1e-3f)) / 1e6f;
                std::cout << "  [Ray Statistics] Hit Rate: HW=" << std::fixed << std::setprecision(2) << hwHitRate 
                          << "% (" << hwHits << "/" << rayCount << ")\n";
                std::cout << "    - Hardware RA: " << rqThroughput << " MRays/s (All) | " 
                          << hwHitOnlyMRays << " MRays/s (Hit-only)\n";
                statsCsv.writeRow(std::vector<std::string>{
                    std::to_string(actualTris),
                    patternName,
                    std::to_string(hwHitRate),
                    "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                    std::to_string(rqThroughput),
                    std::to_string(hwHitOnlyMRays),
                    "N/A", "N/A", "N/A", "N/A"
                });
            }

            throughputCsv.writeRow({
                std::to_string(actualTris),
                patternName,
                "Hardware_RA",
                "In_Shader_Registers",
                std::to_string(rayCount),
                std::to_string(rqStats.medianMs),
                std::to_string(rqStats.p5Ms),
                std::to_string(rqStats.p95Ms),
                std::to_string(rqThroughput),
                std::to_string(speedupLds)
            });

            if (hasSwBvh) {
                throughputCsv.writeRow({
                    std::to_string(actualTris),
                    patternName,
                    "Software_ALU_Scratch",
                    "In_Shader_Registers",
                    std::to_string(rayCount),
                    std::to_string(swStats.medianMs),
                    std::to_string(swStats.p5Ms),
                    std::to_string(swStats.p95Ms),
                    std::to_string(swThroughput),
                    "1.0"
                });

                throughputCsv.writeRow({
                    std::to_string(actualTris),
                    patternName,
                    "Software_ALU_LDS",
                    "In_Shader_Registers",
                    std::to_string(rayCount),
                    std::to_string(swLdsStats.medianMs),
                    std::to_string(swLdsStats.p5Ms),
                    std::to_string(swLdsStats.p95Ms),
                    std::to_string(swLdsThroughput),
                    std::to_string(swLdsThroughput / swThroughput)
                });

                throughputCsv.writeRow(std::vector<std::string>{
                    std::to_string(actualTris),
                    patternName,
                    std::string("Software_ALU_BVH4"),
                    std::string("In_Shader_Registers"),
                    std::to_string(rayCount),
                    std::to_string(swBvh4Stats.medianMs),
                    std::to_string(swBvh4Stats.p5Ms),
                    std::to_string(swBvh4Stats.p95Ms),
                    std::to_string(swBvh4Throughput),
                    std::to_string(swBvh4Throughput / swThroughput)
                });
            }
        }

        // Cleanup per triangle count
        vkDestroyPipeline(g_vk.device, rqPipeline, nullptr);
        vkDestroyPipelineLayout(g_vk.device, rqPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(g_vk.device, rqSetLayout, nullptr);

        vkDestroyPipeline(g_vk.device, swPipeline, nullptr);
        vkDestroyPipeline(g_vk.device, swLdsPipeline, nullptr);
        vkDestroyPipeline(g_vk.device, swBvh4Pipeline, nullptr);
        vkDestroyPipelineLayout(g_vk.device, swPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(g_vk.device, swSetLayout, nullptr);

        vkDestroyDescriptorPool(g_vk.device, descPool, nullptr);

        rqResultBuf.destroy();
        swResultBuf.destroy();
        if (hasSwBvh) {
            swNodeBuf.destroy();
            swTriBuf.destroy();
            swBvh4NodeBuf.destroy();
            swBvh4StatsBuf.destroy();
        }
        tlas.destroy();
        instBuf.destroy();
        blasFastTrace.destroy();
        blasFastBuild.destroy();
        vBuf.destroy();
        iBuf.destroy();
    }

    timer.cleanup();
    vkDestroyShaderModule(g_vk.device, procRqModule, nullptr);
    vkDestroyShaderModule(g_vk.device, procSwModule, nullptr);
    vkDestroyShaderModule(g_vk.device, procSwLdsModule, nullptr);
    vkDestroyShaderModule(g_vk.device, procSwBvh4Module, nullptr);
    g_vk.cleanup();

    std::cout << "\n====================================================\n";
    std::cout << " PHASE 1 AUDIT COMPLETE!\n";
    std::cout << "====================================================\n";

    return 0;
}
