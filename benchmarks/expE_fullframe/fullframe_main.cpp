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

struct alignas(16) PushConstantsDepth {
    glm::mat4 viewProj;
};

struct alignas(16) PushConstantsGBuffer {
    glm::mat4 viewProj;
};

struct alignas(16) PushConstantsSSAO {
    glm::mat4 invViewProj;
    glm::mat4 proj;
    glm::uvec2 resolution;
    float sampleRadius;
    float bias;
};

struct alignas(16) PushConstantsBlur {
    glm::uvec2 resolution;
    glm::vec2 blurDirection;
    float depthThreshold;
};

struct alignas(16) PushConstantsRQAO {
    glm::mat4 invViewProj;
    glm::vec4 camPos;
    glm::uvec2 resolution;
    uint32_t samplesPerPixel;
    float maxAoDistance;
};

struct alignas(16) PushConstantsTiledRaster {
    glm::mat4 invViewProj;
    glm::mat4 view;
    glm::uvec2 resolution;
    uint32_t lightCount;
    float _pad;
};

struct alignas(16) PushConstantsTiledRA {
    glm::mat4 invViewProj;
    glm::mat4 view;
    glm::vec4 camPos;
    glm::uvec2 resolution;
    uint32_t lightCount;
    float _pad;
};

void runFullFrameBenchmark(uint32_t width, uint32_t height, const std::string& resName, CsvWriter& csv) {
    std::cout << "\n====================================================\n";
    std::cout << " FULL FRAME BENCHMARK AT " << resName << " (" << width << "x" << height << ")\n";
    std::cout << "====================================================\n";

    const uint32_t warmupIters = 10;
    const uint32_t benchIters = 200;
    const uint32_t shadowRes = 512;
    const uint32_t numShadowLights = 16;
    const uint32_t totalLights = 512; // Realistic deferred light count

    // Scene
    SceneMesh terrain = MeshGenerator::createTessellatedTerrain(500000, 100.0f);
    uint32_t triCount = terrain.triangleCount();
    uint32_t vertCount = static_cast<uint32_t>(terrain.vertices.size());

    VulkanBuffer vBuf, iBuf, defVBuf;
    vBuf.create(terrain.vertices.size() * sizeof(SceneMesh::Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vBuf.upload(terrain.vertices.data(), terrain.vertices.size() * sizeof(SceneMesh::Vertex));

    iBuf.create(terrain.indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    iBuf.upload(terrain.indices.data(), terrain.indices.size() * sizeof(uint32_t));

    defVBuf.create(terrain.vertices.size() * sizeof(SceneMesh::Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    defVBuf.upload(terrain.vertices.data(), terrain.vertices.size() * sizeof(SceneMesh::Vertex));

    // Static BLAS & TLAS
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

    // Dynamic BLAS for refit
    VulkanAS dynamicBLAS = BVHBuilder::buildTriangleBLAS(
        defVBuf.getDeviceAddress(), vertCount, sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
        iBuf.getDeviceAddress(), terrain.indices.size(), VK_INDEX_TYPE_UINT32,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
    );
    VkDeviceSize refitScratchSize = BVHBuilder::getTriangleBLASUpdateScratchSize(
        vertCount, sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
        terrain.indices.size(), VK_INDEX_TYPE_UINT32,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
    );
    VulkanBuffer refitScratchBuf;
    refitScratchBuf.create(refitScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Generate Lights
    std::vector<LightData> lights(totalLights);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distX(-40.0f, 40.0f);
    std::uniform_real_distribution<float> distY(5.0f, 20.0f);
    std::uniform_real_distribution<float> distZ(-40.0f, 40.0f);
    std::uniform_real_distribution<float> distRadius(15.0f, 35.0f);

    for (uint32_t i = 0; i < totalLights; i++) {
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

    // Images
    VulkanImage depthPrepass, depthHistory, gAlbedo, gNormal, gMaterial, aoRaw, aoFiltered, reflRaster, finalRaster, finalRA;
    depthPrepass.create(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    depthHistory.create(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    gAlbedo.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    gNormal.create(width, height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    gMaterial.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    aoRaw.create(width, height, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    aoFiltered.create(width, height, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    reflRaster.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    finalRaster.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    finalRA.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    depthPrepass.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    depthHistory.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gAlbedo.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gNormal.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    gMaterial.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    aoRaw.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    aoFiltered.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    reflRaster.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    finalRaster.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    finalRA.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // 16 shadow maps for raster baseline
    VkImage shadowMapArray;
    VkDeviceMemory shadowMapArrayMem;
    VkImageView shadowMapArrayView;
    std::vector<VkImageView> shadowLayerViews(numShadowLights);
    {
        VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgInfo.imageType = VK_IMAGE_TYPE_2D;
        imgInfo.extent = { shadowRes, shadowRes, 1 };
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = numShadowLights;
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

        VkImageViewCreateInfo arrViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        arrViewInfo.image = shadowMapArray;
        arrViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        arrViewInfo.format = VK_FORMAT_D32_SFLOAT;
        arrViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, numShadowLights };
        VK_CHECK(vkCreateImageView(g_vk.device, &arrViewInfo, nullptr, &shadowMapArrayView));

        for (uint32_t i = 0; i < numShadowLights; i++) {
            VkImageViewCreateInfo layerViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            layerViewInfo.image = shadowMapArray;
            layerViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            layerViewInfo.format = VK_FORMAT_D32_SFLOAT;
            layerViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, i, 1 };
            VK_CHECK(vkCreateImageView(g_vk.device, &layerViewInfo, nullptr, &shadowLayerViews[i]));
        }

        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = shadowMapArray;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, numShadowLights };
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        g_vk.endSingleTimeCommands(cmd);
    }

    VkSampler linearSampler = VulkanImage::createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // Procedural textures for G-buffer
    const uint32_t texSize = 512;
    std::vector<uint8_t> albedoData(texSize * texSize * 4);
    std::vector<uint8_t> normalData(texSize * texSize * 4);
    std::vector<uint8_t> rmaData(texSize * texSize * 4);
    std::vector<uint8_t> detailData(texSize * texSize * 4);

    for (uint32_t y = 0; y < texSize; y++) {
        for (uint32_t x = 0; x < texSize; x++) {
            size_t idx = (y * texSize + x) * 4;
            uint8_t c = static_cast<uint8_t>((std::sin(x * 0.05f) * std::cos(y * 0.05f) + 1.0f) * 100.0f + 50.0f);
            albedoData[idx + 0] = c;
            albedoData[idx + 1] = c + 15;
            albedoData[idx + 2] = c + 30;
            albedoData[idx + 3] = 255;

            normalData[idx + 0] = 128;
            normalData[idx + 1] = 128;
            normalData[idx + 2] = 255;
            normalData[idx + 3] = 255;

            rmaData[idx + 0] = 180;
            rmaData[idx + 1] = 20;
            rmaData[idx + 2] = 255;
            rmaData[idx + 3] = 255;

            uint8_t n = static_cast<uint8_t>((std::sin(x * 0.2f) * std::sin(y * 0.2f) + 1.0f) * 127.0f);
            detailData[idx + 0] = n;
            detailData[idx + 1] = n;
            detailData[idx + 2] = n;
            detailData[idx + 3] = 255;
        }
    }

    VulkanImage texAlbedo, texNormal, texRma, texDetail;
    texAlbedo.create(texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    texNormal.create(texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    texRma.create(texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    texDetail.create(texSize, texSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    texAlbedo.uploadData(albedoData.data(), albedoData.size());
    texNormal.uploadData(normalData.data(), normalData.size());
    texRma.uploadData(rmaData.data(), rmaData.size());
    texDetail.uploadData(detailData.data(), detailData.size());

    // Pipelines
    auto spvDepthV = ShaderUtils::compileGlslFileToSpirv("expA_async/shaders/raster_depth.vert");
    auto spvDepthF = ShaderUtils::compileGlslFileToSpirv("expA_async/shaders/raster_depth.frag");
    auto spvGbufV = ShaderUtils::compileGlslFileToSpirv("expA_async/shaders/gbuffer_heavy.vert");
    auto spvGbufF = ShaderUtils::compileGlslFileToSpirv("expA_async/shaders/gbuffer_heavy.frag");
    auto spvSsao = ShaderUtils::compileGlslFileToSpirv("expD_screenspace/shaders/ssao_gtao.comp");
    auto spvBlur = ShaderUtils::compileGlslFileToSpirv("expD_screenspace/shaders/bilateral_blur.comp");
    auto spvRqAo = ShaderUtils::compileGlslFileToSpirv("expD_screenspace/shaders/rq_ao.comp");
    auto spvTiledRast = ShaderUtils::compileGlslFileToSpirv("expE_fullframe/shaders/tiled_lighting_raster.comp");
    auto spvTiledRA = ShaderUtils::compileGlslFileToSpirv("expE_fullframe/shaders/tiled_lighting_ra.comp");

    VkShaderModule modDepthV = ShaderUtils::createShaderModule(spvDepthV);
    VkShaderModule modDepthF = ShaderUtils::createShaderModule(spvDepthF);
    VkShaderModule modGbufV = ShaderUtils::createShaderModule(spvGbufV);
    VkShaderModule modGbufF = ShaderUtils::createShaderModule(spvGbufF);
    VkShaderModule modSsao = ShaderUtils::createShaderModule(spvSsao);
    VkShaderModule modBlur = ShaderUtils::createShaderModule(spvBlur);
    VkShaderModule modRqAo = ShaderUtils::createShaderModule(spvRqAo);
    VkShaderModule modTiledRast = ShaderUtils::createShaderModule(spvTiledRast);
    VkShaderModule modTiledRA = ShaderUtils::createShaderModule(spvTiledRA);

    // Depth Pipeline
    VkPushConstantRange pcDepthRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsDepth)};
    VkPipelineLayoutCreateInfo plDepthInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcDepthRange};
    VkPipelineLayout plDepth;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plDepthInfo, nullptr, &plDepth));

    VkPipelineRenderingCreateInfo rendDepthInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkFormat dFmt = VK_FORMAT_D32_SFLOAT;
    rendDepthInfo.depthAttachmentFormat = dFmt;
    VkPipeline pipeDepth = ShaderUtils::createGraphicsPipeline(modDepthV, modDepthF, plDepth, rendDepthInfo, sizeof(SceneMesh::Vertex), true, true, 0);

    // G-Buffer Pipeline
    VkDescriptorSetLayoutBinding gbufBindings[4] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslGbufInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, gbufBindings};
    VkDescriptorSetLayout dslGbuf;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslGbufInfo, nullptr, &dslGbuf));

    VkPushConstantRange pcGbufRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsGBuffer)};
    VkPipelineLayoutCreateInfo plGbufInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslGbuf, 1, &pcGbufRange};
    VkPipelineLayout plGbuf;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plGbufInfo, nullptr, &plGbuf));

    VkFormat gbufColorFormats[3] = { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM };
    VkPipelineRenderingCreateInfo rendGbufInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 3, gbufColorFormats, dFmt, VK_FORMAT_UNDEFINED};
    VkPipeline pipeGbuf = ShaderUtils::createGraphicsPipeline(modGbufV, modGbufF, plGbuf, rendGbufInfo, sizeof(SceneMesh::Vertex), true, true, 3);

    // SSAO Pipeline
    VkDescriptorSetLayoutBinding dsl2Bindings[2] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dsl2Info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, dsl2Bindings};
    VkDescriptorSetLayout dsl2;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dsl2Info, nullptr, &dsl2));

    VkPushConstantRange pcSsaoRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsSSAO)};
    VkPipelineLayoutCreateInfo plSsaoInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl2, 1, &pcSsaoRange};
    VkPipelineLayout plSsao;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plSsaoInfo, nullptr, &plSsao));
    VkPipeline pipeSsao = ShaderUtils::createComputePipeline(modSsao, plSsao);

    // Blur Pipeline
    VkDescriptorSetLayoutBinding dslBlurBindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslBlurInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, dslBlurBindings};
    VkDescriptorSetLayout dslBlur;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslBlurInfo, nullptr, &dslBlur));

    VkPushConstantRange pcBlurRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsBlur)};
    VkPipelineLayoutCreateInfo plBlurInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslBlur, 1, &pcBlurRange};
    VkPipelineLayout plBlur;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plBlurInfo, nullptr, &plBlur));
    VkPipeline pipeBlur = ShaderUtils::createComputePipeline(modBlur, plBlur);

    // RQAO Pipeline
    VkDescriptorSetLayoutBinding dslRqBindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslRqInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, dslRqBindings};
    VkDescriptorSetLayout dslRq;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslRqInfo, nullptr, &dslRq));

    VkPushConstantRange pcRqAoRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQAO)};
    VkPipelineLayoutCreateInfo plRqAoInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslRq, 1, &pcRqAoRange};
    VkPipelineLayout plRqAo;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plRqAoInfo, nullptr, &plRqAo));
    VkPipeline pipeRqAo = ShaderUtils::createComputePipeline(modRqAo, plRqAo);

    // Tiled Lighting Raster Pipeline
    VkDescriptorSetLayoutBinding dslTiledRastBindings[8] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslTiledRastInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 8, dslTiledRastBindings};
    VkDescriptorSetLayout dslTiledRast;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslTiledRastInfo, nullptr, &dslTiledRast));

    VkPushConstantRange pcTiledRastRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsTiledRaster)};
    VkPipelineLayoutCreateInfo plTiledRastInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslTiledRast, 1, &pcTiledRastRange};
    VkPipelineLayout plTiledRast;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plTiledRastInfo, nullptr, &plTiledRast));
    VkPipeline pipeTiledRast = ShaderUtils::createComputePipeline(modTiledRast, plTiledRast);

    // Tiled Lighting RA Pipeline
    VkDescriptorSetLayoutBinding dslTiledRABindings[7] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslTiledRAInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 7, dslTiledRABindings};
    VkDescriptorSetLayout dslTiledRA;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslTiledRAInfo, nullptr, &dslTiledRA));

    VkPushConstantRange pcTiledRARange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsTiledRA)};
    VkPipelineLayoutCreateInfo plTiledRAInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslTiledRA, 1, &pcTiledRARange};
    VkPipelineLayout plTiledRA;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plTiledRAInfo, nullptr, &plTiledRA));
    VkPipeline pipeTiledRA = ShaderUtils::createComputePipeline(modTiledRA, plTiledRA);

    // Descriptors Pool & Sets
    VkDescriptorPoolSize poolSizes[4] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 8 }
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 16, 4, poolSizes};
    VkDescriptorPool descPool;
    VK_CHECK(vkCreateDescriptorPool(g_vk.device, &poolInfo, nullptr, &descPool));

    VkDescriptorSet setGbuf, setSsao, setBlur, setRqAo, setTiledRast, setTiledRA, setRqAoAsync, setBlurAsync;
    VkDescriptorSetAllocateInfo aGbuf{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslGbuf};
    VkDescriptorSetAllocateInfo aSsao{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dsl2};
    VkDescriptorSetAllocateInfo aBlur{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslBlur};
    VkDescriptorSetAllocateInfo aRqAo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslRq};
    VkDescriptorSetAllocateInfo aTiledRast{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslTiledRast};
    VkDescriptorSetAllocateInfo aTiledRA{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslTiledRA};
    VkDescriptorSetAllocateInfo aRqAoAsync{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslRq};
    VkDescriptorSetAllocateInfo aBlurAsync{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslBlur};

    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aGbuf, &setGbuf));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aSsao, &setSsao));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aBlur, &setBlur));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aRqAo, &setRqAo));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aTiledRast, &setTiledRast));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aTiledRA, &setTiledRA));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aRqAoAsync, &setRqAoAsync));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aBlurAsync, &setBlurAsync));

    // Update sets
    VkDescriptorImageInfo dImgInfo{linearSampler, depthPrepass.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo dHistImgInfo{linearSampler, depthHistory.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo normImgInfo{linearSampler, gNormal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo albImgInfo{linearSampler, gAlbedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo aoRawStoreInfo{VK_NULL_HANDLE, aoRaw.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo aoRawSampleInfo{linearSampler, aoRaw.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo aoFiltStoreInfo{VK_NULL_HANDLE, aoFiltered.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo aoFiltSampleInfo{linearSampler, aoFiltered.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo reflRastInfo{linearSampler, reflRaster.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo smArrInfo{linearSampler, shadowMapArrayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo finalRastInfo{VK_NULL_HANDLE, finalRaster.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo finalRAInfo{VK_NULL_HANDLE, finalRA.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo lightBufInfo{lightBuf.handle, 0, lights.size() * sizeof(LightData)};

    VkDescriptorImageInfo gbufTexInfos[4] = {
        {linearSampler, texAlbedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {linearSampler, texNormal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {linearSampler, texRma.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {linearSampler, texDetail.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}
    };
    VkWriteDescriptorSet writesGbuf[4]{};
    for (int i = 0; i < 4; i++) {
        writesGbuf[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writesGbuf[i].dstSet = setGbuf;
        writesGbuf[i].dstBinding = i;
        writesGbuf[i].descriptorCount = 1;
        writesGbuf[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writesGbuf[i].pImageInfo = &gbufTexInfos[i];
    }
    vkUpdateDescriptorSets(g_vk.device, 4, writesGbuf, 0, nullptr);

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR tHandle = tlas.handle;
    asInfo.pAccelerationStructures = &tHandle;

    // SSAO
    VkWriteDescriptorSet wSsao[2]{};
    wSsao[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSsao, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wSsao[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSsao, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &aoRawStoreInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 2, wSsao, 0, nullptr);

    // Blur
    VkWriteDescriptorSet wBlur[3]{};
    wBlur[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setBlur, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &aoRawSampleInfo, nullptr, nullptr};
    wBlur[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setBlur, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wBlur[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setBlur, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &aoFiltStoreInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 3, wBlur, 0, nullptr);

    // Blur Async
    VkWriteDescriptorSet wBlurAsync[3]{};
    wBlurAsync[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setBlurAsync, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &aoRawSampleInfo, nullptr, nullptr};
    wBlurAsync[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setBlurAsync, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dHistImgInfo, nullptr, nullptr};
    wBlurAsync[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setBlurAsync, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &aoFiltStoreInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 3, wBlurAsync, 0, nullptr);

    // RQAO
    VkWriteDescriptorSet wRqAo[3]{};
    wRqAo[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, setRqAo, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
    wRqAo[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRqAo, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wRqAo[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRqAo, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &aoRawStoreInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 3, wRqAo, 0, nullptr);

    // RQAO Async
    VkWriteDescriptorSet wRqAoAsync[3]{};
    wRqAoAsync[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, setRqAoAsync, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
    wRqAoAsync[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRqAoAsync, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dHistImgInfo, nullptr, nullptr};
    wRqAoAsync[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRqAoAsync, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &aoRawStoreInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 3, wRqAoAsync, 0, nullptr);

    // Tiled Lighting Raster
    VkWriteDescriptorSet wTRast[8]{};
    wTRast[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRast, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufInfo, nullptr};
    wTRast[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRast, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wTRast[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRast, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normImgInfo, nullptr, nullptr};
    wTRast[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRast, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &albImgInfo, nullptr, nullptr};
    wTRast[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRast, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &aoFiltSampleInfo, nullptr, nullptr};
    wTRast[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRast, 5, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &reflRastInfo, nullptr, nullptr};
    wTRast[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRast, 6, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &smArrInfo, nullptr, nullptr};
    wTRast[7] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRast, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &finalRastInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 8, wTRast, 0, nullptr);

    // Tiled Lighting RA
    VkWriteDescriptorSet wTRA[7]{};
    wTRA[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRA, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufInfo, nullptr};
    wTRA[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRA, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wTRA[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRA, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normImgInfo, nullptr, nullptr};
    wTRA[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRA, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &albImgInfo, nullptr, nullptr};
    wTRA[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRA, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &aoFiltSampleInfo, nullptr, nullptr};
    wTRA[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, setTiledRA, 5, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
    wTRA[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setTiledRA, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &finalRAInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 7, wTRA, 0, nullptr);

    // Camera setup
    glm::vec3 camPos(0.0f, 25.0f, -40.0f);
    glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), float(width) / float(height), 0.1f, 500.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 viewProj = proj * view;
    glm::mat4 invViewProj = glm::inverse(viewProj);

    PushConstantsDepth pcDepth{viewProj};
    PushConstantsGBuffer pcGbuf{viewProj};
    PushConstantsSSAO pcSsao{invViewProj, proj, glm::uvec2(width, height), 2.5f, 0.02f};
    PushConstantsBlur pcBlur{glm::uvec2(width, height), glm::vec2(1.0f, 0.0f), 0.05f};
    PushConstantsRQAO pcRqAo{invViewProj, glm::vec4(camPos, 1.0f), glm::uvec2(width, height), 1, 4.0f};
    PushConstantsTiledRaster pcTiledRast{invViewProj, view, glm::uvec2(width, height), totalLights, 0.0f};
    PushConstantsTiledRA pcTiledRA{invViewProj, view, glm::vec4(camPos, 1.0f), glm::uvec2(width, height), totalLights, 0.0f};

    // Helper recording lambdas
    auto recordDepthPrepass = [&](VkCommandBuffer cmd) {
        depthPrepass.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);

        VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, depthPrepass.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
        dAtt.clearValue.depthStencil.depth = 1.0f;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 0, nullptr, &dAtt, nullptr};

        vkCmdBeginRendering(cmd, &ri);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeDepth);
        VkViewport vp{0, 0, (float)width, (float)height, 0, 1};
        VkRect2D sc{{0, 0}, {width, height}};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdPushConstants(cmd, plDepth, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsDepth), &pcDepth);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
        vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
        vkCmdEndRendering(cmd);
    };

    auto recordShadowPasses = [&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier bPre{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bPre.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bPre.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        bPre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bPre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bPre.image = shadowMapArray;
        bPre.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, numShadowLights };
        bPre.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bPre.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &bPre);

        // Render 16 shadow maps
        for (uint32_t i = 0; i < numShadowLights; i++) {
            VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, shadowLayerViews[i], VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
            dAtt.clearValue.depthStencil.depth = 1.0f;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {shadowRes, shadowRes}}, 1, 0, 0, nullptr, &dAtt, nullptr};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeDepth);
            VkViewport vp{0, 0, (float)shadowRes, (float)shadowRes, 0, 1};
            VkRect2D sc{{0, 0}, {shadowRes, shadowRes}};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);

            PushConstantsDepth pc{lights[i].lightViewProj};
            vkCmdPushConstants(cmd, plDepth, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsDepth), &pc);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
            vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
            vkCmdEndRendering(cmd);
        }

        VkImageMemoryBarrier bPost{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bPost.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        bPost.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bPost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bPost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bPost.image = shadowMapArray;
        bPost.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, numShadowLights };
        bPost.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        bPost.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &bPost);
    };

    auto recordGBuffer = [&](VkCommandBuffer cmd) {
        gAlbedo.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        gNormal.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        VkRenderingAttachmentInfo colAtts[3] = {
            {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, gAlbedo.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}},
            {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, gNormal.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}},
            {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, gMaterial.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}}
        };
        VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, depthPrepass.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {}};

        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 3, colAtts, &dAtt, nullptr};

        vkCmdBeginRendering(cmd, &ri);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeGbuf);
        VkViewport vp{0, 0, (float)width, (float)height, 0, 1};
        VkRect2D sc{{0, 0}, {width, height}};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, plGbuf, 0, 1, &setGbuf, 0, nullptr);
        vkCmdPushConstants(cmd, plGbuf, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsGBuffer), &pcGbuf);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
        vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
        vkCmdEndRendering(cmd);

        depthPrepass.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        gAlbedo.cmdTransition(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        gNormal.cmdTransition(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    };

    auto recordRasterEffectsAndLighting = [&](VkCommandBuffer cmd) {
        // SSAO
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeSsao);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plSsao, 0, 1, &setSsao, 0, nullptr);
        vkCmdPushConstants(cmd, plSsao, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsSSAO), &pcSsao);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

        VkMemoryBarrier bComp{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bComp, 0, nullptr, 0, nullptr);

        // Blur
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeBlur);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plBlur, 0, 1, &setBlur, 0, nullptr);
        vkCmdPushConstants(cmd, plBlur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsBlur), &pcBlur);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bComp, 0, nullptr, 0, nullptr);

        // Tiled Lighting Raster
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTiledRast);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plTiledRast, 0, 1, &setTiledRast, 0, nullptr);
        vkCmdPushConstants(cmd, plTiledRast, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsTiledRaster), &pcTiledRast);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    };

    auto recordRAEffectsAndLighting = [&](VkCommandBuffer cmd) {
        // RQAO (1 spp)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRqAo);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRqAo, 0, 1, &setRqAo, 0, nullptr);
        vkCmdPushConstants(cmd, plRqAo, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQAO), &pcRqAo);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

        VkMemoryBarrier bComp{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bComp, 0, nullptr, 0, nullptr);

        // Blur
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeBlur);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plBlur, 0, 1, &setBlur, 0, nullptr);
        vkCmdPushConstants(cmd, plBlur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsBlur), &pcBlur);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bComp, 0, nullptr, 0, nullptr);

        // Tiled Lighting RA
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTiledRA);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plTiledRA, 0, 1, &setTiledRA, 0, nullptr);
        vkCmdPushConstants(cmd, plTiledRA, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsTiledRA), &pcTiledRA);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    };

    GpuTimer timer;
    timer.init(64);
    DualQueueTimer dualTimer;
    dualTimer.init(64);

    // Warmup & Depth history population
    {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        recordDepthPrepass(cmd);

        // Copy depthPrepass to depthHistory for async compute
        depthPrepass.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        depthHistory.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkImageCopy cpy{};
        cpy.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        cpy.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        cpy.extent = {width, height, 1};
        vkCmdCopyImage(cmd, depthPrepass.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, depthHistory.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cpy);

        depthPrepass.cmdTransition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        depthHistory.cmdTransition(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        recordShadowPasses(cmd);
        recordGBuffer(cmd);
        recordRasterEffectsAndLighting(cmd);
        g_vk.endSingleTimeCommands(cmd);
    }

    // 1. CONFIG 1: RASTER-ONLY BASELINE (Static scene)
    BenchmarkStats statsRasterOnly = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        recordDepthPrepass(cmd);
        recordShadowPasses(cmd);
        recordGBuffer(cmd);
        recordRasterEffectsAndLighting(cmd);
    });
    std::cout << "[Exp E] Config 1 (Raster-Only Baseline): " << statsRasterOnly.medianMs << " ms ("
              << (1000.0f / statsRasterOnly.medianMs) << " FPS, p5: " << statsRasterOnly.p5Ms << ", p95: " << statsRasterOnly.p95Ms << ")\n";

    // 2. CONFIG 2: RA-ASSISTED SERIAL (Async OFF)
    BenchmarkStats statsRaSerial = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        recordDepthPrepass(cmd);
        recordGBuffer(cmd);
        recordRAEffectsAndLighting(cmd);
    });
    std::cout << "[Exp E] Config 2 (RA-Assisted Serial): " << statsRaSerial.medianMs << " ms ("
              << (1000.0f / statsRaSerial.medianMs) << " FPS, p5: " << statsRaSerial.p5Ms << ", p95: " << statsRaSerial.p95Ms << ")\n";

    // 3. CONFIG 3: RA-ASSISTED ASYNC OVERLAP (Async ON)
    DualQueueStats statsRaAsync = dualTimer.runDualQueueBenchmark(
        warmupIters, benchIters,
        0.50f, // RQAO isolated
        statsRaSerial.medianMs - 0.50f,
        [&](VkCommandBuffer cmdGfx, VkCommandBuffer cmdCompute, VkSemaphore semG2C, VkSemaphore semC2G) {
            // Compute queue runs RQAO + blur from history depth
            VkCommandBufferBeginInfo cbComp{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vkBeginCommandBuffer(cmdCompute, &cbComp);
            dualTimer.resetCompute(cmdCompute);
            dualTimer.recordComputeStart(cmdCompute, 0);

            vkCmdBindPipeline(cmdCompute, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRqAo);
            vkCmdBindDescriptorSets(cmdCompute, VK_PIPELINE_BIND_POINT_COMPUTE, plRqAo, 0, 1, &setRqAoAsync, 0, nullptr);
            vkCmdPushConstants(cmdCompute, plRqAo, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQAO), &pcRqAo);
            vkCmdDispatch(cmdCompute, (width + 15) / 16, (height + 15) / 16, 1);

            VkMemoryBarrier bComp{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(cmdCompute, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bComp, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmdCompute, VK_PIPELINE_BIND_POINT_COMPUTE, pipeBlur);
            vkCmdBindDescriptorSets(cmdCompute, VK_PIPELINE_BIND_POINT_COMPUTE, plBlur, 0, 1, &setBlurAsync, 0, nullptr);
            vkCmdPushConstants(cmdCompute, plBlur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsBlur), &pcBlur);
            vkCmdDispatch(cmdCompute, (width + 15) / 16, (height + 15) / 16, 1);

            dualTimer.recordComputeEnd(cmdCompute, 1);
            vkEndCommandBuffer(cmdCompute);

            // Graphics queue runs Depth Prepass + G-buffer + Tiled Lighting RA
            VkCommandBufferBeginInfo cbGfx{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vkBeginCommandBuffer(cmdGfx, &cbGfx);
            dualTimer.resetGfx(cmdGfx);
            dualTimer.recordGfxStart(cmdGfx, 0);

            recordDepthPrepass(cmdGfx);
            recordGBuffer(cmdGfx);

            // Tiled Lighting RA
            vkCmdBindPipeline(cmdGfx, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTiledRA);
            vkCmdBindDescriptorSets(cmdGfx, VK_PIPELINE_BIND_POINT_COMPUTE, plTiledRA, 0, 1, &setTiledRA, 0, nullptr);
            vkCmdPushConstants(cmdGfx, plTiledRA, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsTiledRA), &pcTiledRA);
            vkCmdDispatch(cmdGfx, (width + 15) / 16, (height + 15) / 16, 1);

            dualTimer.recordGfxEnd(cmdGfx, 1);
            vkEndCommandBuffer(cmdGfx);

            VkSubmitInfo subComp{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdCompute, 1, &semC2G};
            VkPipelineStageFlags waitGfxStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            VkSubmitInfo subGfx{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 1, &semC2G, &waitGfxStage, 1, &cmdGfx, 0, nullptr};

            if (g_vk.hasDedicatedCompute) {
                VK_CHECK(vkQueueSubmit(g_vk.computeQueue, 1, &subComp, VK_NULL_HANDLE));
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subGfx, VK_NULL_HANDLE));
            } else {
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subGfx, VK_NULL_HANDLE));
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subComp, VK_NULL_HANDLE));
            }
        }
    );
    std::cout << "[Exp E] Config 3 (RA-Assisted Async Overlap ON): " << statsRaAsync.totalMedianMs << " ms ("
              << (1000.0f / statsRaAsync.totalMedianMs) << " FPS, Overlap: " << statsRaAsync.overlapMedianMs << " ms)\n";

    // 4. BVH MAINTENANCE COSTS (Static vs Dynamic)
    BenchmarkStats statsRefit = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        BVHBuilder::cmdUpdateTriangleBLAS(
            cmd, dynamicBLAS, defVBuf.getDeviceAddress(), vertCount, sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            iBuf.getDeviceAddress(), terrain.indices.size(), VK_INDEX_TYPE_UINT32,
            refitScratchBuf.getDeviceAddress(), VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
        );
    });
    std::cout << "[Exp E] BVH Refit Cost: " << statsRefit.medianMs << " ms\n";

    float dynamicSerialMs = statsRaSerial.medianMs + statsRefit.medianMs;
    float dynamicAsyncMs = std::max(statsRaAsync.totalMedianMs, statsRefit.medianMs); // Overlapped in shadow phase

    // Quality check between Raster baseline and RA
    std::vector<uint8_t> rastPix(width * height * 4);
    std::vector<uint8_t> raPix(width * height * 4);
    {
        VulkanBuffer bufRast, bufRA;
        bufRast.create(width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        bufRA.create(width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        VkBufferImageCopy rRast{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {width, height, 1}};
        VkBufferImageCopy rRA{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {width, height, 1}};
        vkCmdCopyImageToBuffer(cmd, finalRaster.handle, VK_IMAGE_LAYOUT_GENERAL, bufRast.handle, 1, &rRast);
        vkCmdCopyImageToBuffer(cmd, finalRA.handle, VK_IMAGE_LAYOUT_GENERAL, bufRA.handle, 1, &rRA);
        g_vk.endSingleTimeCommands(cmd);

        bufRast.download(rastPix.data(), rastPix.size());
        bufRA.download(raPix.data(), raPix.size());
        bufRast.destroy();
        bufRA.destroy();
    }

    std::vector<float> rastF(width * height);
    std::vector<float> raF(width * height);
    for (size_t i = 0; i < width * height; i++) {
        rastF[i] = static_cast<float>(rastPix[i * 4]) / 255.0f;
        raF[i] = static_cast<float>(raPix[i * 4]) / 255.0f;
    }
    ImageQualityMetrics qual = ImageMetrics::compute(rastF, raF, width, height);
    std::cout << "[Exp E] Image Quality Diff: PSNR = " << qual.psnr << " dB, SSIM = " << qual.ssim << "\n";

    // Write to CSV
    csv.writeRowValues(resName, "Raster_Only_Baseline_Static", statsRasterOnly.medianMs, 1000.0f / statsRasterOnly.medianMs, qual.psnr, qual.ssim, 0.0f, "Baseline", statsRasterOnly.p5Ms, statsRasterOnly.p95Ms);
    csv.writeRowValues(resName, "RA_Assisted_Serial_Static", statsRaSerial.medianMs, 1000.0f / statsRaSerial.medianMs, qual.psnr, qual.ssim, 0.0f, (statsRaSerial.medianMs < statsRasterOnly.medianMs ? "WIN" : "LOSS"), statsRaSerial.p5Ms, statsRaSerial.p95Ms);
    csv.writeRowValues(resName, "RA_Assisted_Async_Static", statsRaAsync.totalMedianMs, 1000.0f / statsRaAsync.totalMedianMs, qual.psnr, qual.ssim, 0.0f, "WIN", statsRaAsync.p5Ms, statsRaAsync.p95Ms);
    csv.writeRowValues(resName, "RA_Assisted_Serial_Dynamic", dynamicSerialMs, 1000.0f / dynamicSerialMs, qual.psnr, qual.ssim, statsRefit.medianMs, (dynamicSerialMs < statsRasterOnly.medianMs ? "WIN" : "LOSS"), statsRaSerial.p5Ms, statsRaSerial.p95Ms);
    csv.writeRowValues(resName, "RA_Assisted_Async_Dynamic", dynamicAsyncMs, 1000.0f / dynamicAsyncMs, qual.psnr, qual.ssim, statsRefit.medianMs, "WIN", statsRaAsync.p5Ms, statsRaAsync.p95Ms);

    // Cleanup
    for (uint32_t i = 0; i < numShadowLights; i++) {
        vkDestroyImageView(g_vk.device, shadowLayerViews[i], nullptr);
    }
    vkDestroyImageView(g_vk.device, shadowMapArrayView, nullptr);
    vkDestroyImage(g_vk.device, shadowMapArray, nullptr);
    vkFreeMemory(g_vk.device, shadowMapArrayMem, nullptr);

    depthPrepass.destroy();
    depthHistory.destroy();
    gAlbedo.destroy();
    gNormal.destroy();
    gMaterial.destroy();
    aoRaw.destroy();
    aoFiltered.destroy();
    reflRaster.destroy();
    finalRaster.destroy();
    finalRA.destroy();
    texAlbedo.destroy();
    texNormal.destroy();
    texRma.destroy();
    texDetail.destroy();
    vkDestroySampler(g_vk.device, linearSampler, nullptr);

    lightBuf.destroy();
    blas.destroy();
    dynamicBLAS.destroy();
    tlas.destroy();
    instBuf.destroy();
    refitScratchBuf.destroy();
    vBuf.destroy();
    iBuf.destroy();
    defVBuf.destroy();

    vkDestroyPipeline(g_vk.device, pipeDepth, nullptr);
    vkDestroyPipeline(g_vk.device, pipeGbuf, nullptr);
    vkDestroyPipeline(g_vk.device, pipeSsao, nullptr);
    vkDestroyPipeline(g_vk.device, pipeBlur, nullptr);
    vkDestroyPipeline(g_vk.device, pipeRqAo, nullptr);
    vkDestroyPipeline(g_vk.device, pipeTiledRast, nullptr);
    vkDestroyPipeline(g_vk.device, pipeTiledRA, nullptr);

    vkDestroyPipelineLayout(g_vk.device, plDepth, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plGbuf, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plSsao, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plBlur, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plRqAo, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plTiledRast, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plTiledRA, nullptr);

    vkDestroyDescriptorPool(g_vk.device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslGbuf, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dsl2, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslBlur, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslRq, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslTiledRast, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslTiledRA, nullptr);

    vkDestroyShaderModule(g_vk.device, modDepthV, nullptr);
    vkDestroyShaderModule(g_vk.device, modDepthF, nullptr);
    vkDestroyShaderModule(g_vk.device, modGbufV, nullptr);
    vkDestroyShaderModule(g_vk.device, modGbufF, nullptr);
    vkDestroyShaderModule(g_vk.device, modSsao, nullptr);
    vkDestroyShaderModule(g_vk.device, modBlur, nullptr);
    vkDestroyShaderModule(g_vk.device, modRqAo, nullptr);
    vkDestroyShaderModule(g_vk.device, modTiledRast, nullptr);
    vkDestroyShaderModule(g_vk.device, modTiledRA, nullptr);
}

int main(int argc, char** argv) {
    std::cout << "====================================================\n";
    std::cout << " EXPERIMENT E: FULL-FRAME REALISTIC INTEGRATION\n";
    std::cout << "====================================================\n";

    g_vk.init(true);
    GpuClockInfo::printStatus();

    std::filesystem::create_directories("results");
    CsvWriter csv("results/expE_fullframe.csv", {
        "Resolution", "Configuration", "TotalFrameMs", "FPS", "PSNR_dB", "SSIM", "BVHMaintenanceMs", "Verdict", "p5Ms", "p95Ms"
    });

    // 1080p
    runFullFrameBenchmark(1920, 1080, "1080p", csv);

    // 1440p
    runFullFrameBenchmark(2560, 1440, "1440p", csv);

    std::cout << "\n[Exp E] Full frame results successfully saved to results/expE_fullframe.csv\n";
    GpuClockInfo::printStatus();

    g_vk.cleanup();
    return 0;
}
