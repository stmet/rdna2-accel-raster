#include "vk_context.hpp"
#include "vk_buffer.hpp"
#include "vk_image.hpp"
#include "vk_bvh.hpp"
#include "vk_shader.hpp"
#include "vk_timer.hpp"
#include "csv_writer.hpp"
#include "mesh_gen.hpp"
#include "gpu_clock.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <filesystem>
#include <chrono>

struct alignas(16) PushConstantsDepth {
    glm::mat4 viewProj;
};

struct alignas(16) PushConstantsGBuffer {
    glm::mat4 viewProj;
};

struct alignas(16) PushConstantsRA {
    glm::mat4 invViewProj;
    glm::vec4 camPos;
    glm::vec4 lightDir;
    glm::uvec2 resolution;
    uint32_t raysPerPixel;
    float maxAoDistance;
};

struct alignas(16) PushConstantsDeform {
    uint32_t vertexCount;
    float time;
    float amplitude;
    float frequency;
};

int main(int argc, char** argv) {
    std::cout << "====================================================\n";
    std::cout << " EXPERIMENT A: ASYNC COMPUTE OVERLAP & RA CONTENTION\n";
    std::cout << "====================================================\n";

    g_vk.init(true);
    GpuClockInfo::printStatus();

    if (!g_vk.hasDedicatedCompute) {
        std::cerr << "WARNING: No dedicated async compute queue found! Overlap will be emulated on single queue.\n";
    }

    uint32_t width = 1920;
    uint32_t height = 1080;
    uint32_t shadowRes = 2048;
    const uint32_t warmupIters = 20;
    const uint32_t benchIters = 200;

    std::cout << "[Exp A] Frame resolution: " << width << "x" << height 
              << ", Shadow map resolution: " << shadowRes << "x" << shadowRes << "\n";

    // 1. Generate Scene Geometry
    std::cout << "[Exp A] Generating terrain geometry (500k triangles)...\n";
    SceneMesh terrain = MeshGenerator::createTessellatedTerrain(500000, 100.0f);
    uint32_t triCount = terrain.triangleCount();
    uint32_t vertCount = static_cast<uint32_t>(terrain.vertices.size());
    std::cout << "  - Vertices: " << vertCount << ", Triangles: " << triCount << "\n";

    // Vertex and Index buffers
    VulkanBuffer vBuf, iBuf, defVBuf;
    vBuf.create(
        terrain.vertices.size() * sizeof(SceneMesh::Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    vBuf.upload(terrain.vertices.data(), terrain.vertices.size() * sizeof(SceneMesh::Vertex));

    iBuf.create(
        terrain.indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    iBuf.upload(terrain.indices.data(), terrain.indices.size() * sizeof(uint32_t));

    defVBuf.create(
        terrain.vertices.size() * sizeof(SceneMesh::Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    defVBuf.upload(terrain.vertices.data(), terrain.vertices.size() * sizeof(SceneMesh::Vertex));

    // Build static BLAS & TLAS
    std::cout << "[Exp A] Building static BLAS & TLAS...\n";
    VulkanAS staticBLAS = BVHBuilder::buildTriangleBLAS(
        vBuf.getDeviceAddress(), vertCount, sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
        iBuf.getDeviceAddress(), terrain.indices.size(), VK_INDEX_TYPE_UINT32,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
    );

    VkTransformMatrixKHR transform = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    VkAccelerationStructureInstanceKHR meshInst{};
    meshInst.transform = transform;
    meshInst.mask = 0xFF;
    meshInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    meshInst.accelerationStructureReference = staticBLAS.deviceAddress;

    VulkanBuffer instBuf;
    instBuf.create(sizeof(meshInst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    instBuf.upload(&meshInst, sizeof(meshInst));
    VulkanAS staticTLAS = BVHBuilder::buildTLAS(instBuf.getDeviceAddress(), 1);
    std::cout << "  - BLAS/TLAS ready.\n";

    // Dynamic BLAS for refit test
    std::cout << "[Exp A] Building dynamic BLAS for async refit benchmarking...\n";
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

    // 2. Create Render Targets and Textures
    // Depth prepass target
    VulkanImage depthPrepass;
    depthPrepass.create(width, height, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // Shadow map target
    VulkanImage shadowMap;
    shadowMap.create(shadowRes, shadowRes, VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // G-buffer color attachments
    VulkanImage gAlbedo, gNormal, gMaterial;
    gAlbedo.create(width, height, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    gNormal.create(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    gMaterial.create(width, height, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    // RA Output image
    VulkanImage raOut;
    raOut.create(width, height, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    // Initial layouts
    depthPrepass.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    shadowMap.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    gAlbedo.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    gNormal.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    gMaterial.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    raOut.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // Create 4 procedural textures to simulate high-end texture-heavy materials
    std::cout << "[Exp A] Uploading high-res textures for G-buffer TMU contention testing...\n";
    const uint32_t texSize = 1024;
    std::vector<uint8_t> albedoData(texSize * texSize * 4);
    std::vector<uint8_t> normalData(texSize * texSize * 4);
    std::vector<uint8_t> rmaData(texSize * texSize * 4);
    std::vector<uint8_t> detailData(texSize * texSize * 4);

    for (uint32_t y = 0; y < texSize; y++) {
        for (uint32_t x = 0; x < texSize; x++) {
            size_t idx = (y * texSize + x) * 4;
            // Albedo: granite / rock texture pattern
            uint8_t c = static_cast<uint8_t>((std::sin(x * 0.05f) * std::cos(y * 0.05f) + 1.0f) * 100.0f + 50.0f);
            albedoData[idx + 0] = c;
            albedoData[idx + 1] = c + 15;
            albedoData[idx + 2] = c + 30;
            albedoData[idx + 3] = 255;

            // Normal map
            normalData[idx + 0] = 128;
            normalData[idx + 1] = 128;
            normalData[idx + 2] = 255;
            normalData[idx + 3] = 255;

            // Roughness / Metallic / AO
            rmaData[idx + 0] = 180; // Roughness
            rmaData[idx + 1] = 20;  // Metallic
            rmaData[idx + 2] = 255; // AO
            rmaData[idx + 3] = 255;

            // Detail noise
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

    VkSampler linearSampler = VulkanImage::createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    VkSampler depthSampler = VulkanImage::createSampler(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // 3. Create Pipelines
    // (a) Depth pass pipeline
    std::cout << "[Exp A] Compiling depth/shadow pass shaders...\n";
    auto spvDepthV = ShaderUtils::compileGlslFileToSpirv("expA_async/shaders/raster_depth.vert");
    auto spvDepthF = ShaderUtils::compileGlslFileToSpirv("expA_async/shaders/raster_depth.frag");
    VkShaderModule modDepthV = ShaderUtils::createShaderModule(spvDepthV);
    VkShaderModule modDepthF = ShaderUtils::createShaderModule(spvDepthF);

    VkPushConstantRange pcDepthRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsDepth)};
    VkPipelineLayoutCreateInfo plDepthInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plDepthInfo.pushConstantRangeCount = 1;
    plDepthInfo.pPushConstantRanges = &pcDepthRange;
    VkPipelineLayout plDepth;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plDepthInfo, nullptr, &plDepth));

    VkPipelineRenderingCreateInfo depthRenderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkFormat depthFmt = VK_FORMAT_D32_SFLOAT;
    depthRenderingInfo.depthAttachmentFormat = depthFmt;

    VkPipeline pipeDepth = ShaderUtils::createGraphicsPipeline(modDepthV, modDepthF, plDepth, depthRenderingInfo, sizeof(SceneMesh::Vertex), true, true, 0);

    // (b) G-Buffer pipeline
    std::cout << "[Exp A] Compiling texture-heavy G-buffer shaders...\n";
    auto spvGbufV = ShaderUtils::compileGlslFileToSpirv("expA_async/shaders/gbuffer_heavy.vert");
    auto spvGbufF = ShaderUtils::compileGlslFileToSpirv("expA_async/shaders/gbuffer_heavy.frag");
    VkShaderModule modGbufV = ShaderUtils::createShaderModule(spvGbufV);
    VkShaderModule modGbufF = ShaderUtils::createShaderModule(spvGbufF);

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
    VkPipelineLayoutCreateInfo plGbufInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plGbufInfo.setLayoutCount = 1;
    plGbufInfo.pSetLayouts = &dslGbuf;
    plGbufInfo.pushConstantRangeCount = 1;
    plGbufInfo.pPushConstantRanges = &pcGbufRange;
    VkPipelineLayout plGbuf;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plGbufInfo, nullptr, &plGbuf));

    VkFormat gbufColorFormats[3] = { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM };
    VkPipelineRenderingCreateInfo gbufRenderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    gbufRenderingInfo.colorAttachmentCount = 3;
    gbufRenderingInfo.pColorAttachmentFormats = gbufColorFormats;
    gbufRenderingInfo.depthAttachmentFormat = depthFmt;

    VkPipeline pipeGbuf = ShaderUtils::createGraphicsPipeline(modGbufV, modGbufF, plGbuf, gbufRenderingInfo, sizeof(SceneMesh::Vertex), true, false, 3);

    // Allocate & update descriptor set for G-buffer textures
    VkDescriptorPoolSize poolSizes[3] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4 }
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 16;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;
    VkDescriptorPool descPool;
    VK_CHECK(vkCreateDescriptorPool(g_vk.device, &poolInfo, nullptr, &descPool));

    VkDescriptorSetAllocateInfo allocGbuf{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslGbuf};
    VkDescriptorSet setGbuf;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocGbuf, &setGbuf));

    VkDescriptorImageInfo imgInfos[4] = {
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
        writesGbuf[i].pImageInfo = &imgInfos[i];
    }
    vkUpdateDescriptorSets(g_vk.device, 4, writesGbuf, 0, nullptr);

    // (c) RA Compute pipeline
    std::cout << "[Exp A] Compiling Ray Query AO/Shadow compute pipeline...\n";
    auto spvRA = ShaderUtils::compileGlslFileToSpirv("expA_async/shaders/ra_workload.comp");
    VkShaderModule modRA = ShaderUtils::createShaderModule(spvRA);

    VkDescriptorSetLayoutBinding raBindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslRaInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, raBindings};
    VkDescriptorSetLayout dslRA;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslRaInfo, nullptr, &dslRA));

    VkPushConstantRange pcRaRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRA)};
    VkPipelineLayoutCreateInfo plRaInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plRaInfo.setLayoutCount = 1;
    plRaInfo.pSetLayouts = &dslRA;
    plRaInfo.pushConstantRangeCount = 1;
    plRaInfo.pPushConstantRanges = &pcRaRange;
    VkPipelineLayout plRA;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plRaInfo, nullptr, &plRA));

    VkPipeline pipeRA = ShaderUtils::createComputePipeline(modRA, plRA);

    VkDescriptorSetAllocateInfo allocRA{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslRA};
    VkDescriptorSet setRA;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocRA, &setRA));

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR tlasHandle = staticTLAS.handle;
    asInfo.pAccelerationStructures = &tlasHandle;

    VkDescriptorImageInfo dImgInfo{depthSampler, depthPrepass.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo raOutInfo{VK_NULL_HANDLE, raOut.view, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet writesRA[3]{};
    writesRA[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writesRA[0].pNext = &asInfo;
    writesRA[0].dstSet = setRA;
    writesRA[0].dstBinding = 0;
    writesRA[0].descriptorCount = 1;
    writesRA[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    writesRA[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writesRA[1].dstSet = setRA;
    writesRA[1].dstBinding = 1;
    writesRA[1].descriptorCount = 1;
    writesRA[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writesRA[1].pImageInfo = &dImgInfo;

    writesRA[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writesRA[2].dstSet = setRA;
    writesRA[2].dstBinding = 2;
    writesRA[2].descriptorCount = 1;
    writesRA[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writesRA[2].pImageInfo = &raOutInfo;

    vkUpdateDescriptorSets(g_vk.device, 3, writesRA, 0, nullptr);

    // Setup matrices
    glm::vec3 camPos(0.0f, 25.0f, -40.0f);
    glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), float(width) / float(height), 0.1f, 500.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 viewProj = proj * view;
    glm::mat4 invViewProj = glm::inverse(viewProj);

    glm::vec3 lightDir = glm::normalize(glm::vec3(0.4f, 0.8f, 0.3f));
    glm::mat4 lightView = glm::lookAt(lightDir * 60.0f, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 lightProj = glm::ortho(-60.0f, 60.0f, -60.0f, 60.0f, 0.1f, 150.0f);
    lightProj[1][1] *= -1.0f;
    glm::mat4 shadowViewProj = lightProj * lightView;

    PushConstantsDepth pcDepth{viewProj};
    PushConstantsDepth pcShadow{shadowViewProj};
    PushConstantsGBuffer pcGbuf{viewProj};
    PushConstantsRA pcRA{invViewProj, glm::vec4(camPos, 1.0f), glm::vec4(lightDir, 0.0f), glm::uvec2(width, height), 2, 5.0f};

    // Helper recording functions
    auto recordDepthAndShadow = [&](VkCommandBuffer cmd) {
        // Pass 1: Depth prepass
        VkRenderingAttachmentInfo dAttPre{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, depthPrepass.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
        dAttPre.clearValue.depthStencil.depth = 1.0f;
        VkRenderingInfo riPre{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 0, nullptr, &dAttPre, nullptr};

        vkCmdBeginRendering(cmd, &riPre);
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

        // Pass 2: Shadow map pass
        VkRenderingAttachmentInfo dAttShadow{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, shadowMap.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
        dAttShadow.clearValue.depthStencil.depth = 1.0f;
        VkRenderingInfo riShadow{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {shadowRes, shadowRes}}, 1, 0, 0, nullptr, &dAttShadow, nullptr};

        vkCmdBeginRendering(cmd, &riShadow);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeDepth);
        VkViewport vpS{0, 0, (float)shadowRes, (float)shadowRes, 0, 1};
        VkRect2D scS{{0, 0}, {shadowRes, shadowRes}};
        vkCmdSetViewport(cmd, 0, 1, &vpS);
        vkCmdSetScissor(cmd, 0, 1, &scS);
        vkCmdPushConstants(cmd, plDepth, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsDepth), &pcShadow);
        vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
        vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
        vkCmdEndRendering(cmd);
    };

    auto recordGBuffer = [&](VkCommandBuffer cmd) {
        VkRenderingAttachmentInfo colAtts[3] = {
            {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, gAlbedo.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}},
            {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, gNormal.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}},
            {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, gMaterial.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}}
        };
        VkRenderingAttachmentInfo dAttGbuf{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, depthPrepass.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {}};

        VkRenderingInfo riGbuf{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 3, colAtts, &dAttGbuf, nullptr};

        vkCmdBeginRendering(cmd, &riGbuf);
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
    };

    auto recordRAWorkload = [&](VkCommandBuffer cmd) {
        depthPrepass.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRA);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRA, 0, 1, &setRA, 0, nullptr);
        vkCmdPushConstants(cmd, plRA, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRA), &pcRA);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

        depthPrepass.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
    };

    // 4. Benchmarking Configurations
    GpuTimer timer;
    timer.init(64);
    DualQueueTimer dualTimer;
    dualTimer.init(64);

    // Populate depth buffer once
    {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        recordDepthAndShadow(cmd);
        g_vk.endSingleTimeCommands(cmd);
    }

    std::cout << "\n====================================================\n";
    std::cout << " BENCHMARK 1: ISOLATED PASS TIMINGS (BASELINE COSTS)\n";
    std::cout << "====================================================\n";

    BenchmarkStats statsDepthShadow = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        recordDepthAndShadow(cmd);
    });
    std::cout << "  - Depth Prepass + Shadow Map: " << statsDepthShadow.medianMs << " ms (p5: " 
              << statsDepthShadow.p5Ms << ", p95: " << statsDepthShadow.p95Ms << ")\n";

    BenchmarkStats statsGBuffer = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        recordGBuffer(cmd);
    });
    std::cout << "  - Texture-Heavy G-Buffer: " << statsGBuffer.medianMs << " ms (p5: " 
              << statsGBuffer.p5Ms << ", p95: " << statsGBuffer.p95Ms << ")\n";

    BenchmarkStats statsRAIsolated = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        recordRAWorkload(cmd);
    });
    std::cout << "  - RA Pass Isolated (RQAO 2spp + Shadow): " << statsRAIsolated.medianMs << " ms (p5: " 
              << statsRAIsolated.p5Ms << ", p95: " << statsRAIsolated.p95Ms << ")\n";

    // Dynamic mesh refit isolated
    auto recordRefit = [&](VkCommandBuffer cmd) {
        BVHBuilder::cmdUpdateTriangleBLAS(
            cmd, dynamicBLAS, defVBuf.getDeviceAddress(), vertCount, sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            iBuf.getDeviceAddress(), terrain.indices.size(), VK_INDEX_TYPE_UINT32,
            refitScratchBuf.getDeviceAddress(), VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
        );
    };

    BenchmarkStats statsRefitIsolated = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        recordRefit(cmd);
    });
    std::cout << "  - BVH Refit Isolated (500k tris): " << statsRefitIsolated.medianMs << " ms (p5: " 
              << statsRefitIsolated.p5Ms << ", p95: " << statsRefitIsolated.p95Ms << ")\n";

    std::cout << "\n====================================================\n";
    std::cout << " BENCHMARK 2: CONFIG (1) SERIALIZED ON GRAPHICS QUEUE\n";
    std::cout << "====================================================\n";

    BenchmarkStats statsConfig1 = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        recordDepthAndShadow(cmd);
        recordGBuffer(cmd);
        recordRAWorkload(cmd);
    });
    std::cout << "  - Config 1 Serial Frame Time: " << statsConfig1.medianMs << " ms (FPS: "
              << (1000.0f / statsConfig1.medianMs) << ", p5: " << statsConfig1.p5Ms << ", p95: " << statsConfig1.p95Ms << ")\n";

    std::cout << "\n====================================================\n";
    std::cout << " BENCHMARK 3: CONFIG (2) ASYNC OVERLAP WITH DEPTH/SHADOW (LOW TEXTURE)\n";
    std::cout << "====================================================\n";

    DualQueueStats statsConfig2 = dualTimer.runDualQueueBenchmark(
        warmupIters, benchIters,
        statsRAIsolated.medianMs,
        statsDepthShadow.medianMs + statsGBuffer.medianMs,
        [&](VkCommandBuffer cmdGfx, VkCommandBuffer cmdCompute, VkSemaphore semG2C, VkSemaphore semC2G) {
            // Compute queue records RA pass
            VkCommandBufferBeginInfo cbComp{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vkBeginCommandBuffer(cmdCompute, &cbComp);
            dualTimer.resetCompute(cmdCompute);
            dualTimer.recordComputeStart(cmdCompute, 0);
            recordRAWorkload(cmdCompute);
            dualTimer.recordComputeEnd(cmdCompute, 1);
            vkEndCommandBuffer(cmdCompute);

            // Graphics queue records Depth+Shadow, waits for compute to finish before G-buffer/lighting
            VkCommandBufferBeginInfo cbGfx{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vkBeginCommandBuffer(cmdGfx, &cbGfx);
            dualTimer.resetGfx(cmdGfx);
            dualTimer.recordGfxStart(cmdGfx, 0);
            recordDepthAndShadow(cmdGfx);
            recordGBuffer(cmdGfx);
            dualTimer.recordGfxEnd(cmdGfx, 1);
            vkEndCommandBuffer(cmdGfx);

            // Submit Compute queue concurrently with Graphics queue
            VkSubmitInfo subComp{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdCompute, 1, &semC2G};
            VkPipelineStageFlags waitGfxStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo subGfx{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdGfx, 0, nullptr};

            if (g_vk.hasDedicatedCompute) {
                VK_CHECK(vkQueueSubmit(g_vk.computeQueue, 1, &subComp, VK_NULL_HANDLE));
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subGfx, VK_NULL_HANDLE));
            } else {
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subGfx, VK_NULL_HANDLE));
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subComp, VK_NULL_HANDLE));
            }
        }
    );

    std::cout << "  - Config 2 Total Overlapped Time: " << statsConfig2.totalMedianMs << " ms (FPS: " << (1000.0f / statsConfig2.totalMedianMs) << ")\n";
    std::cout << "  - Per-Queue Times: Gfx = " << statsConfig2.gfxMedianMs << " ms, Compute = " << statsConfig2.computeMedianMs << " ms\n";
    std::cout << "  - Hardware Overlap Duration: " << statsConfig2.overlapMedianMs << " ms\n";
    std::cout << "  - Overlap Efficiency: " << (statsConfig2.efficiency * 100.0f) << "%\n";
    std::cout << "  - Contention Penalty (vs Isolated RA): " << statsConfig2.contentionPenaltyMs << " ms\n";

    std::cout << "\n====================================================\n";
    std::cout << " BENCHMARK 4: CONFIG (3) ASYNC OVERLAP WITH TEXTURE-HEAVY G-BUFFER\n";
    std::cout << "====================================================\n";

    DualQueueStats statsConfig3 = dualTimer.runDualQueueBenchmark(
        warmupIters, benchIters,
        statsRAIsolated.medianMs,
        statsDepthShadow.medianMs + statsGBuffer.medianMs,
        [&](VkCommandBuffer cmdGfx, VkCommandBuffer cmdCompute, VkSemaphore semG2C, VkSemaphore semC2G) {
            // Graphics queue runs depth prepass first, signals semG2C
            // Then graphics queue runs G-buffer, while compute runs RA concurrently
            VkCommandBufferBeginInfo cbComp{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vkBeginCommandBuffer(cmdCompute, &cbComp);
            dualTimer.resetCompute(cmdCompute);
            dualTimer.recordComputeStart(cmdCompute, 0);
            recordRAWorkload(cmdCompute);
            dualTimer.recordComputeEnd(cmdCompute, 1);
            vkEndCommandBuffer(cmdCompute);

            VkCommandBufferBeginInfo cbGfx{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vkBeginCommandBuffer(cmdGfx, &cbGfx);
            dualTimer.resetGfx(cmdGfx);
            dualTimer.recordGfxStart(cmdGfx, 0);
            recordDepthAndShadow(cmdGfx);
            recordGBuffer(cmdGfx);
            dualTimer.recordGfxEnd(cmdGfx, 1);
            vkEndCommandBuffer(cmdGfx);

            VkSubmitInfo subComp{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdCompute, 1, &semC2G};
            VkSubmitInfo subGfx{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdGfx, 0, nullptr};

            if (g_vk.hasDedicatedCompute) {
                VK_CHECK(vkQueueSubmit(g_vk.computeQueue, 1, &subComp, VK_NULL_HANDLE));
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subGfx, VK_NULL_HANDLE));
            } else {
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subGfx, VK_NULL_HANDLE));
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subComp, VK_NULL_HANDLE));
            }
        }
    );

    std::cout << "  - Config 3 Total Overlapped Time: " << statsConfig3.totalMedianMs << " ms (FPS: " << (1000.0f / statsConfig3.totalMedianMs) << ")\n";
    std::cout << "  - Per-Queue Times: Gfx = " << statsConfig3.gfxMedianMs << " ms, Compute = " << statsConfig3.computeMedianMs << " ms\n";
    std::cout << "  - Hardware Overlap Duration: " << statsConfig3.overlapMedianMs << " ms\n";
    std::cout << "  - Overlap Efficiency: " << (statsConfig3.efficiency * 100.0f) << "%\n";
    std::cout << "  - Contention Penalty (vs Isolated RA): " << statsConfig3.contentionPenaltyMs << " ms\n";

    std::cout << "\n====================================================\n";
    std::cout << " BENCHMARK 5: DYNAMIC MESH BVH REFIT ON ASYNC COMPUTE QUEUE\n";
    std::cout << "====================================================\n";

    DualQueueStats statsRefitOverlap = dualTimer.runDualQueueBenchmark(
        warmupIters, benchIters,
        statsRefitIsolated.medianMs,
        statsDepthShadow.medianMs,
        [&](VkCommandBuffer cmdGfx, VkCommandBuffer cmdCompute, VkSemaphore semG2C, VkSemaphore semC2G) {
            // Compute queue runs BVH refit
            VkCommandBufferBeginInfo cbComp{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vkBeginCommandBuffer(cmdCompute, &cbComp);
            dualTimer.resetCompute(cmdCompute);
            dualTimer.recordComputeStart(cmdCompute, 0);
            recordRefit(cmdCompute);
            dualTimer.recordComputeEnd(cmdCompute, 1);
            vkEndCommandBuffer(cmdCompute);

            // Graphics queue runs Shadow map rendering
            VkCommandBufferBeginInfo cbGfx{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vkBeginCommandBuffer(cmdGfx, &cbGfx);
            dualTimer.resetGfx(cmdGfx);
            dualTimer.recordGfxStart(cmdGfx, 0);
            recordDepthAndShadow(cmdGfx);
            dualTimer.recordGfxEnd(cmdGfx, 1);
            vkEndCommandBuffer(cmdGfx);

            VkSubmitInfo subComp{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdCompute, 1, &semC2G};
            VkSubmitInfo subGfx{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdGfx, 0, nullptr};

            if (g_vk.hasDedicatedCompute) {
                VK_CHECK(vkQueueSubmit(g_vk.computeQueue, 1, &subComp, VK_NULL_HANDLE));
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subGfx, VK_NULL_HANDLE));
            } else {
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subGfx, VK_NULL_HANDLE));
                VK_CHECK(vkQueueSubmit(g_vk.graphicsQueue, 1, &subComp, VK_NULL_HANDLE));
            }
        }
    );

    std::cout << "  - Shadow + Refit Overlapped Time: " << statsRefitOverlap.totalMedianMs << " ms\n";
    std::cout << "  - Per-Queue Times: Gfx (Shadows) = " << statsRefitOverlap.gfxMedianMs << " ms, Compute (Refit) = " << statsRefitOverlap.computeMedianMs << " ms\n";
    std::cout << "  - Hardware Overlap Duration: " << statsRefitOverlap.overlapMedianMs << " ms\n";
    std::cout << "  - Refit Hiding Efficiency: " << (statsRefitOverlap.efficiency * 100.0f) << "%\n";

    // Write CSV Output
    std::filesystem::create_directories("results");
    CsvWriter csv("results/expA_async.csv", {"Configuration", "SerialMs", "TotalMs", "GfxQueueMs", "ComputeQueueMs", "OverlapMs", "EfficiencyPct", "ContentionPenaltyMs", "FPS", "p5Ms", "p95Ms"});

    float serialConfig1 = statsDepthShadow.medianMs + statsGBuffer.medianMs + statsRAIsolated.medianMs;
    csv.writeRowValues("Config1_Serialized", serialConfig1, statsConfig1.medianMs, statsConfig1.medianMs, 0.0f, 0.0f, 0.0f, 0.0f, 1000.0f / statsConfig1.medianMs, statsConfig1.p5Ms, statsConfig1.p95Ms);

    csv.writeRowValues("Config2_Async_DepthOverlap", serialConfig1, statsConfig2.totalMedianMs, statsConfig2.gfxMedianMs, statsConfig2.computeMedianMs, statsConfig2.overlapMedianMs, statsConfig2.efficiency * 100.0f, statsConfig2.contentionPenaltyMs, 1000.0f / statsConfig2.totalMedianMs, statsConfig2.p5Ms, statsConfig2.p95Ms);

    csv.writeRowValues("Config3_Async_GBufferOverlap", serialConfig1, statsConfig3.totalMedianMs, statsConfig3.gfxMedianMs, statsConfig3.computeMedianMs, statsConfig3.overlapMedianMs, statsConfig3.efficiency * 100.0f, statsConfig3.contentionPenaltyMs, 1000.0f / statsConfig3.totalMedianMs, statsConfig3.p5Ms, statsConfig3.p95Ms);

    float serialRefit = statsDepthShadow.medianMs + statsRefitIsolated.medianMs;
    csv.writeRowValues("Dynamic_BVH_Refit_Overlap", serialRefit, statsRefitOverlap.totalMedianMs, statsRefitOverlap.gfxMedianMs, statsRefitOverlap.computeMedianMs, statsRefitOverlap.overlapMedianMs, statsRefitOverlap.efficiency * 100.0f, statsRefitOverlap.contentionPenaltyMs, 1000.0f / statsRefitOverlap.totalMedianMs, statsRefitOverlap.p5Ms, statsRefitOverlap.p95Ms);

    std::cout << "\n[Exp A] Results successfully saved to results/expA_async.csv\n";
    GpuClockInfo::printStatus();

    // Cleanup
    staticBLAS.destroy();
    dynamicBLAS.destroy();
    staticTLAS.destroy();
    instBuf.destroy();
    refitScratchBuf.destroy();
    vBuf.destroy();
    iBuf.destroy();
    defVBuf.destroy();
    texAlbedo.destroy();
    texNormal.destroy();
    texRma.destroy();
    texDetail.destroy();
    depthPrepass.destroy();
    shadowMap.destroy();
    gAlbedo.destroy();
    gNormal.destroy();
    gMaterial.destroy();
    raOut.destroy();

    vkDestroySampler(g_vk.device, linearSampler, nullptr);
    vkDestroySampler(g_vk.device, depthSampler, nullptr);
    vkDestroyPipeline(g_vk.device, pipeDepth, nullptr);
    vkDestroyPipeline(g_vk.device, pipeGbuf, nullptr);
    vkDestroyPipeline(g_vk.device, pipeRA, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plDepth, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plGbuf, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plRA, nullptr);
    vkDestroyDescriptorPool(g_vk.device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslGbuf, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslRA, nullptr);
    vkDestroyShaderModule(g_vk.device, modDepthV, nullptr);
    vkDestroyShaderModule(g_vk.device, modDepthF, nullptr);
    vkDestroyShaderModule(g_vk.device, modGbufV, nullptr);
    vkDestroyShaderModule(g_vk.device, modGbufF, nullptr);
    vkDestroyShaderModule(g_vk.device, modRA, nullptr);

    g_vk.cleanup();
    return 0;
}
