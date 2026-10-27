#include "vk_context.hpp"
#include "vk_buffer.hpp"
#include "vk_image.hpp"
#include "vk_bvh.hpp"
#include "vk_shader.hpp"
#include "vk_timer.hpp"
#include "csv_writer.hpp"
#include "mesh_gen.hpp"
#include "gpu_clock.hpp"
#include "image_metrics.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <filesystem>

struct LightData {
    glm::vec4 posRadius;
    glm::vec4 color;
    glm::mat4 lightViewProj;
};

struct alignas(16) PushConstantsShadowDepth {
    glm::mat4 lightViewProj;
};

struct alignas(16) PushConstantsShading {
    glm::mat4 invViewProj;
    glm::mat4 view;
    glm::uvec2 resolution;
    uint32_t lightCount;
    float pcfFilterRadius;
};

int main(int argc, char** argv) {
    std::cout << "====================================================\n";
    std::cout << " EXPERIMENT B: SHADOWS WITH MANY CASTERS (SWEEP)\n";
    std::cout << "====================================================\n";

    g_vk.init(true);
    GpuClockInfo::printStatus();

    const uint32_t width = 1920;
    const uint32_t height = 1080;
    const uint32_t shadowMapRes = 512; // Standard local light shadow map resolution
    const uint32_t warmupIters = 10;
    const uint32_t benchIters = 200;

    std::vector<uint32_t> triCounts = { 100000, 500000, 1000000, 2500000 };
    std::vector<uint32_t> lightCounts = { 1, 4, 8, 16, 32, 64 };

    // Shaders
    std::cout << "[Exp B] Compiling shaders...\n";
    auto spvDepthV = ShaderUtils::compileGlslFileToSpirv("expB_shadows/shaders/shadow_raster_depth.vert");
    auto spvDepthF = ShaderUtils::compileGlslFileToSpirv("expB_shadows/shaders/shadow_raster_depth.frag");
    auto spvSmShade = ShaderUtils::compileGlslFileToSpirv("expB_shadows/shaders/lighting_shadowmap_tiled.comp");
    auto spvRqShade = ShaderUtils::compileGlslFileToSpirv("expB_shadows/shaders/lighting_rayquery_tiled.comp");

    VkShaderModule modDepthV = ShaderUtils::createShaderModule(spvDepthV);
    VkShaderModule modDepthF = ShaderUtils::createShaderModule(spvDepthF);
    VkShaderModule modSmShade = ShaderUtils::createShaderModule(spvSmShade);
    VkShaderModule modRqShade = ShaderUtils::createShaderModule(spvRqShade);

    // Depth pass pipeline layout
    VkPushConstantRange pcDepthRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsShadowDepth)};
    VkPipelineLayoutCreateInfo plDepthInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plDepthInfo.pushConstantRangeCount = 1;
    plDepthInfo.pPushConstantRanges = &pcDepthRange;
    VkPipelineLayout plDepth;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plDepthInfo, nullptr, &plDepth));

    VkPipelineRenderingCreateInfo depthRenderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkFormat depthFmt = VK_FORMAT_D32_SFLOAT;
    depthRenderingInfo.depthAttachmentFormat = depthFmt;

    VkPipeline pipeDepth = ShaderUtils::createGraphicsPipeline(
        modDepthV, modDepthF, plDepth, depthRenderingInfo, sizeof(SceneMesh::Vertex), true, true, 0
    );

    // Primary depth buffer & output images
    VulkanImage primaryDepth, outRasterColor, outRqColor;
    primaryDepth.create(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    outRasterColor.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    outRqColor.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    primaryDepth.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    outRasterColor.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    outRqColor.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // Shadow Map Array (max 64 layers for up to 64 lights)
    const uint32_t maxShadowLayers = 64;
    VkImage shadowMapArray = VK_NULL_HANDLE;
    VkDeviceMemory shadowMapArrayMem = VK_NULL_HANDLE;
    VkImageView shadowMapArrayView = VK_NULL_HANDLE;
    std::vector<VkImageView> shadowLayerViews(maxShadowLayers);

    {
        VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.extent = { shadowMapRes, shadowMapRes, 1 };
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = maxShadowLayers;
        imgInfo.format = VK_FORMAT_D32_SFLOAT;
        imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateImage(g_vk.device, &imgInfo, nullptr, &shadowMapArray));

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(g_vk.device, shadowMapArray, &memReq);
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = g_vk.findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(g_vk.device, &allocInfo, nullptr, &shadowMapArrayMem));
        VK_CHECK(vkBindImageMemory(g_vk.device, shadowMapArray, shadowMapArrayMem, 0));

        // Array view
        VkImageViewCreateInfo arrViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        arrViewInfo.image = shadowMapArray;
        arrViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        arrViewInfo.format = VK_FORMAT_D32_SFLOAT;
        arrViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, maxShadowLayers };
        VK_CHECK(vkCreateImageView(g_vk.device, &arrViewInfo, nullptr, &shadowMapArrayView));

        // Per-layer views for rendering passes
        for (uint32_t i = 0; i < maxShadowLayers; i++) {
            VkImageViewCreateInfo layerViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            layerViewInfo.image = shadowMapArray;
            layerViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            layerViewInfo.format = VK_FORMAT_D32_SFLOAT;
            layerViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, i, 1 };
            VK_CHECK(vkCreateImageView(g_vk.device, &layerViewInfo, nullptr, &shadowLayerViews[i]));
        }

        // Transition shadow map array to depth attachment optimal
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = shadowMapArray;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, maxShadowLayers };
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        g_vk.endSingleTimeCommands(cmd);
    }

    VkSampler depthSampler = VulkanImage::createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // Compute Pipeline Layouts
    VkPushConstantRange pcShadeRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsShading)};

    // Shadow Map Shading Pipeline
    VkDescriptorSetLayoutBinding smBindings[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslSmInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, smBindings};
    VkDescriptorSetLayout dslSm;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslSmInfo, nullptr, &dslSm));

    VkPipelineLayoutCreateInfo plSmInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plSmInfo.setLayoutCount = 1;
    plSmInfo.pSetLayouts = &dslSm;
    plSmInfo.pushConstantRangeCount = 1;
    plSmInfo.pPushConstantRanges = &pcShadeRange;
    VkPipelineLayout plSm;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plSmInfo, nullptr, &plSm));
    VkPipeline pipeSmShade = ShaderUtils::createComputePipeline(modSmShade, plSm);

    // Ray Query Shading Pipeline
    VkDescriptorSetLayoutBinding rqBindings[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslRqInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, rqBindings};
    VkDescriptorSetLayout dslRq;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslRqInfo, nullptr, &dslRq));

    VkPipelineLayoutCreateInfo plRqInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plRqInfo.setLayoutCount = 1;
    plRqInfo.pSetLayouts = &dslRq;
    plRqInfo.pushConstantRangeCount = 1;
    plRqInfo.pPushConstantRanges = &pcShadeRange;
    VkPipelineLayout plRq;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plRqInfo, nullptr, &plRq));
    VkPipeline pipeRqShade = ShaderUtils::createComputePipeline(modRqShade, plRq);

    // Camera view setup
    glm::vec3 camPos(0.0f, 30.0f, -45.0f);
    glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), float(width) / float(height), 0.1f, 500.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 viewProj = proj * view;
    glm::mat4 invViewProj = glm::inverse(viewProj);

    // Prepare CSV output
    std::filesystem::create_directories("results");
    CsvWriter csv("results/expB_shadows.csv", {
        "Triangles", "LightCount", "RasterShadowMapsMs", "RasterSamplingMs", "RasterTotalMs",
        "RayQueryShadowsMs", "SpeedupRQvsRaster", "Winner", "PSNR_dB", "SSIM", "p5Ms", "p95Ms"
    });

    GpuTimer timer;
    timer.init(64);

    // Main Benchmark Loops over Triangles and Light Counts
    for (uint32_t tris : triCounts) {
        std::cout << "\n----------------------------------------------------\n";
        std::cout << "[Exp B] Generating Scene with " << tris << " triangles...\n";
        std::cout << "----------------------------------------------------\n";

        SceneMesh terrain = MeshGenerator::createTessellatedTerrain(tris, 120.0f);
        uint32_t triCount = terrain.triangleCount();
        uint32_t vertCount = static_cast<uint32_t>(terrain.vertices.size());

        VulkanBuffer vBuf, iBuf;
        vBuf.create(terrain.vertices.size() * sizeof(SceneMesh::Vertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vBuf.upload(terrain.vertices.data(), terrain.vertices.size() * sizeof(SceneMesh::Vertex));

        iBuf.create(terrain.indices.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        iBuf.upload(terrain.indices.data(), terrain.indices.size() * sizeof(uint32_t));

        // Build BLAS & TLAS
        VulkanAS blas = BVHBuilder::buildTriangleBLAS(
            vBuf.getDeviceAddress(), vertCount, sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            iBuf.getDeviceAddress(), terrain.indices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
        );

        VkTransformMatrixKHR transform = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
        VkAccelerationStructureInstanceKHR meshInst{};
        meshInst.transform = transform;
        meshInst.mask = 0xFF;
        meshInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        meshInst.accelerationStructureReference = blas.deviceAddress;

        VulkanBuffer instBuf;
        instBuf.create(sizeof(meshInst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        instBuf.upload(&meshInst, sizeof(meshInst));
        VulkanAS tlas = BVHBuilder::buildTLAS(instBuf.getDeviceAddress(), 1);

        // Render primary depth prepass once for this terrain
        {
            VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
            VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, primaryDepth.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
            dAtt.clearValue.depthStencil.depth = 1.0f;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 0, nullptr, &dAtt, nullptr};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeDepth);
            VkViewport vp{0, 0, (float)width, (float)height, 0, 1};
            VkRect2D sc{{0, 0}, {width, height}};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);

            PushConstantsShadowDepth pc{viewProj};
            vkCmdPushConstants(cmd, plDepth, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsShadowDepth), &pc);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
            vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
            vkCmdEndRendering(cmd);

            primaryDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            g_vk.endSingleTimeCommands(cmd);
        }

        for (uint32_t numLights : lightCounts) {
            std::cout << "[Exp B] Benchmarking " << numLights << " shadow-casting lights against " << tris << " tris...\n";

            // Generate Lights
            std::vector<LightData> lights(numLights);
            std::mt19937 rng(1337 + numLights);
            std::uniform_real_distribution<float> distX(-40.0f, 40.0f);
            std::uniform_real_distribution<float> distY(8.0f, 25.0f);
            std::uniform_real_distribution<float> distZ(-40.0f, 40.0f);
            std::uniform_real_distribution<float> distRadius(25.0f, 45.0f);

            for (uint32_t i = 0; i < numLights; i++) {
                glm::vec3 lPos(distX(rng), distY(rng), distZ(rng));
                float r = distRadius(rng);
                lights[i].posRadius = glm::vec4(lPos, r);
                lights[i].color = glm::vec4(1.0f, 0.85f, 0.7f, 1.0f);

                glm::mat4 lView = glm::lookAt(lPos, lPos + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
                glm::mat4 lProj = glm::perspective(glm::radians(90.0f), 1.0f, 0.5f, r);
                lProj[1][1] *= -1.0f;
                lights[i].lightViewProj = lProj * lView;
            }

            VulkanBuffer lightBuf;
            lightBuf.create(lights.size() * sizeof(LightData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            lightBuf.upload(lights.data(), lights.size() * sizeof(LightData));

            // Allocate descriptor sets
            VkDescriptorPoolSize poolSizes[3] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8 }
            };
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 8, 3, poolSizes};
            VkDescriptorPool descPool;
            VK_CHECK(vkCreateDescriptorPool(g_vk.device, &poolInfo, nullptr, &descPool));

            VkDescriptorSet setSm, setRq;
            VkDescriptorSetAllocateInfo allocSm{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslSm};
            VkDescriptorSetAllocateInfo allocRq{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslRq};
            VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocSm, &setSm));
            VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocRq, &setRq));

            // Update SM set
            VkDescriptorBufferInfo bufInfo{lightBuf.handle, 0, lights.size() * sizeof(LightData)};
            VkDescriptorImageInfo dInfo{depthSampler, primaryDepth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo smArrInfo{depthSampler, shadowMapArrayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo outSmInfo{VK_NULL_HANDLE, outRasterColor.view, VK_IMAGE_LAYOUT_GENERAL};

            VkWriteDescriptorSet smWrites[4]{};
            smWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSm, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufInfo, nullptr};
            smWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSm, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dInfo, nullptr, nullptr};
            smWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSm, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &smArrInfo, nullptr, nullptr};
            smWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSm, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outSmInfo, nullptr, nullptr};
            vkUpdateDescriptorSets(g_vk.device, 4, smWrites, 0, nullptr);

            // Update RQ set
            VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            asInfo.accelerationStructureCount = 1;
            VkAccelerationStructureKHR tHandle = tlas.handle;
            asInfo.pAccelerationStructures = &tHandle;
            VkDescriptorImageInfo outRqInfo{VK_NULL_HANDLE, outRqColor.view, VK_IMAGE_LAYOUT_GENERAL};

            VkWriteDescriptorSet rqWrites[4]{};
            rqWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRq, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufInfo, nullptr};
            rqWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRq, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dInfo, nullptr, nullptr};
            rqWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, setRq, 2, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
            rqWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRq, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outRqInfo, nullptr, nullptr};
            vkUpdateDescriptorSets(g_vk.device, 4, rqWrites, 0, nullptr);

            PushConstantsShading pcShade{invViewProj, view, glm::uvec2(width, height), numLights, 1.0f};

            // 1. Benchmark: Rasterizing N Shadow Maps
            auto recordShadowMaps = [&](VkCommandBuffer cmd) {
                // Transition shadow array to depth attachment optimal
                VkImageMemoryBarrier b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b1.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b1.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b1.image = shadowMapArray;
                b1.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, numLights };
                b1.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                b1.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b1);

                for (uint32_t i = 0; i < numLights; i++) {
                    VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, shadowLayerViews[i], VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
                    dAtt.clearValue.depthStencil.depth = 1.0f;
                    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {shadowMapRes, shadowMapRes}}, 1, 0, 0, nullptr, &dAtt, nullptr};

                    vkCmdBeginRendering(cmd, &ri);
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeDepth);
                    VkViewport vp{0, 0, (float)shadowMapRes, (float)shadowMapRes, 0, 1};
                    VkRect2D sc{{0, 0}, {shadowMapRes, shadowMapRes}};
                    vkCmdSetViewport(cmd, 0, 1, &vp);
                    vkCmdSetScissor(cmd, 0, 1, &sc);

                    PushConstantsShadowDepth pcD{lights[i].lightViewProj};
                    vkCmdPushConstants(cmd, plDepth, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsShadowDepth), &pcD);
                    VkDeviceSize off = 0;
                    vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
                    vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
                    vkCmdEndRendering(cmd);
                }

                // Transition back to shader read
                VkImageMemoryBarrier b2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b2.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b2.image = shadowMapArray;
                b2.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, numLights };
                b2.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b2);
            };

            BenchmarkStats statsSmGen = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
                recordShadowMaps(cmd);
            });

            // 2. Benchmark: Shadow Map PCF Sampling & Lighting
            auto recordSmSample = [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeSmShade);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plSm, 0, 1, &setSm, 0, nullptr);
                vkCmdPushConstants(cmd, plSm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsShading), &pcShade);
                vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
            };

            BenchmarkStats statsSmSample = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
                recordSmSample(cmd);
            });

            float rasterTotalMs = statsSmGen.medianMs + statsSmSample.medianMs;

            // 3. Benchmark: Ray Query Shadows & Lighting
            auto recordRq = [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRqShade);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRq, 0, 1, &setRq, 0, nullptr);
                vkCmdPushConstants(cmd, plRq, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsShading), &pcShade);
                vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
            };

            BenchmarkStats statsRq = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
                recordRq(cmd);
            });

            float speedup = rasterTotalMs / statsRq.medianMs;
            std::string winner = (speedup > 1.0f) ? "RayQuery" : "RasterShadowMaps";

            // 4. Quality Verification (Download and Compute PSNR/SSIM)
            std::vector<uint8_t> smPixels(width * height * 4);
            std::vector<uint8_t> rqPixels(width * height * 4);

            {
                VulkanBuffer readBufSm, readBufRq;
                readBufSm.create(width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                readBufRq.create(width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
                VkBufferImageCopy regionSm{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {width, height, 1}};
                VkBufferImageCopy regionRq{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {width, height, 1}};
                vkCmdCopyImageToBuffer(cmd, outRasterColor.handle, VK_IMAGE_LAYOUT_GENERAL, readBufSm.handle, 1, &regionSm);
                vkCmdCopyImageToBuffer(cmd, outRqColor.handle, VK_IMAGE_LAYOUT_GENERAL, readBufRq.handle, 1, &regionRq);
                g_vk.endSingleTimeCommands(cmd);

                readBufSm.download(smPixels.data(), smPixels.size());
                readBufRq.download(rqPixels.data(), rqPixels.size());
                readBufSm.destroy();
                readBufRq.destroy();
            }

            std::vector<float> smFloats(width * height);
            std::vector<float> rqFloats(width * height);
            for (size_t p = 0; p < width * height; p++) {
                smFloats[p] = static_cast<float>(smPixels[p * 4 + 0]) / 255.0f;
                rqFloats[p] = static_cast<float>(rqPixels[p * 4 + 0]) / 255.0f;
            }
            ImageQualityMetrics qual = ImageMetrics::compute(smFloats, rqFloats, width, height);

            std::cout << "  - Raster (SM Gen: " << statsSmGen.medianMs << " ms, Sample: " << statsSmSample.medianMs << " ms) = Total: " << rasterTotalMs << " ms\n";
            std::cout << "  - Ray Query Shadows: " << statsRq.medianMs << " ms (Speedup: " << std::fixed << std::setprecision(2) << speedup << "x) -> Winner: " << winner << "\n";
            std::cout << "  - Image Quality Match: PSNR = " << qual.psnr << " dB, SSIM = " << qual.ssim << "\n";

            csv.writeRowValues(
                tris, numLights, statsSmGen.medianMs, statsSmSample.medianMs, rasterTotalMs,
                statsRq.medianMs, speedup, winner, qual.psnr, qual.ssim, statsRq.p5Ms, statsRq.p95Ms
            );

            lightBuf.destroy();
            vkDestroyDescriptorPool(g_vk.device, descPool, nullptr);
        }

        blas.destroy();
        tlas.destroy();
        instBuf.destroy();
        vBuf.destroy();
        iBuf.destroy();
    }

    std::cout << "\n[Exp B] Results successfully saved to results/expB_shadows.csv\n";
    GpuClockInfo::printStatus();

    // Cleanup
    for (uint32_t i = 0; i < maxShadowLayers; i++) {
        vkDestroyImageView(g_vk.device, shadowLayerViews[i], nullptr);
    }
    vkDestroyImageView(g_vk.device, shadowMapArrayView, nullptr);
    vkDestroyImage(g_vk.device, shadowMapArray, nullptr);
    vkFreeMemory(g_vk.device, shadowMapArrayMem, nullptr);
    vkDestroySampler(g_vk.device, depthSampler, nullptr);

    primaryDepth.destroy();
    outRasterColor.destroy();
    outRqColor.destroy();

    vkDestroyPipeline(g_vk.device, pipeDepth, nullptr);
    vkDestroyPipeline(g_vk.device, pipeSmShade, nullptr);
    vkDestroyPipeline(g_vk.device, pipeRqShade, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plDepth, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plSm, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plRq, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslSm, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslRq, nullptr);
    vkDestroyShaderModule(g_vk.device, modDepthV, nullptr);
    vkDestroyShaderModule(g_vk.device, modDepthF, nullptr);
    vkDestroyShaderModule(g_vk.device, modSmShade, nullptr);
    vkDestroyShaderModule(g_vk.device, modRqShade, nullptr);

    g_vk.cleanup();
    return 0;
}
