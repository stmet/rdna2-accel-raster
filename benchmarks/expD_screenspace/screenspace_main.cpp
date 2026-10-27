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
#include <iomanip>
#include <filesystem>

struct alignas(16) PushConstantsDepth {
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

struct alignas(16) PushConstantsSSR {
    glm::mat4 invViewProj;
    glm::mat4 proj;
    glm::vec4 camPos;
    glm::uvec2 resolution;
    float maxDistance;
    uint32_t maxSteps;
};

struct alignas(16) PushConstantsRQR {
    glm::mat4 invViewProj;
    glm::vec4 camPos;
    glm::uvec2 resolution;
    uint32_t samplesPerPixel;
    float maxDistance;
};

struct alignas(16) PushConstantsSSCS {
    glm::mat4 invViewProj;
    glm::mat4 proj;
    glm::vec4 lightDir;
    glm::uvec2 resolution;
    float maxDistance;
    uint32_t maxSteps;
};

struct alignas(16) PushConstantsRQCS {
    glm::mat4 invViewProj;
    glm::vec4 lightDir;
    glm::uvec2 resolution;
    uint32_t samplesPerPixel;
    float maxDistance;
};

int main(int argc, char** argv) {
    std::cout << "====================================================\n";
    std::cout << " EXPERIMENT D: SCREEN-SPACE EFFECTS REPLACEMENT\n";
    std::cout << "====================================================\n";

    g_vk.init(true);
    GpuClockInfo::printStatus();

    const uint32_t width = 1920;
    const uint32_t height = 1080;
    const uint32_t warmupIters = 10;
    const uint32_t benchIters = 200;

    std::cout << "[Exp D] Generating terrain geometry (500k triangles)...\n";
    SceneMesh terrain = MeshGenerator::createTessellatedTerrain(500000, 100.0f);
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

    // Primary Depth & Target Images
    VulkanImage primaryDepth, imgAORaw, imgAOFiltered, imgRQR, imgContactShadow;
    primaryDepth.create(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    imgAORaw.create(width, height, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    imgAOFiltered.create(width, height, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    imgRQR.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    imgContactShadow.create(width, height, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    primaryDepth.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    imgAORaw.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    imgAOFiltered.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    imgRQR.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    imgContactShadow.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    VkSampler linearSampler = VulkanImage::createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // Setup Depth prepass
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

    VkFormat depthFmt = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo rendInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendInfo.depthAttachmentFormat = depthFmt;

    VkPipeline pipeDepth = ShaderUtils::createGraphicsPipeline(
        modDepthV, modDepthF, plDepth, rendInfo, sizeof(SceneMesh::Vertex), true, true, 0
    );

    glm::vec3 camPos(0.0f, 25.0f, -40.0f);
    glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), float(width) / float(height), 0.1f, 500.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 viewProj = proj * view;
    glm::mat4 invViewProj = glm::inverse(viewProj);
    glm::vec3 sunDir = glm::normalize(glm::vec3(0.4f, 0.8f, 0.3f));

    // Render primary depth once
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

        PushConstantsDepth pc{viewProj};
        vkCmdPushConstants(cmd, plDepth, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsDepth), &pc);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
        vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
        vkCmdEndRendering(cmd);

        primaryDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        g_vk.endSingleTimeCommands(cmd);
    }

    // Compile Compute Shaders
    std::cout << "[Exp D] Compiling screen-space and ray query effect shaders...\n";
    auto spvSsao = ShaderUtils::compileGlslFileToSpirv("expD_screenspace/shaders/ssao_gtao.comp");
    auto spvBlur = ShaderUtils::compileGlslFileToSpirv("expD_screenspace/shaders/bilateral_blur.comp");
    auto spvRqAo = ShaderUtils::compileGlslFileToSpirv("expD_screenspace/shaders/rq_ao.comp");
    auto spvSsr = ShaderUtils::compileGlslFileToSpirv("expD_screenspace/shaders/ssr.comp");
    auto spvRqR = ShaderUtils::compileGlslFileToSpirv("expD_screenspace/shaders/rq_reflection.comp");
    auto spvSscs = ShaderUtils::compileGlslFileToSpirv("expD_screenspace/shaders/ss_contact_shadows.comp");
    auto spvRqCs = ShaderUtils::compileGlslFileToSpirv("expD_screenspace/shaders/rq_contact_shadows.comp");

    VkShaderModule modSsao = ShaderUtils::createShaderModule(spvSsao);
    VkShaderModule modBlur = ShaderUtils::createShaderModule(spvBlur);
    VkShaderModule modRqAo = ShaderUtils::createShaderModule(spvRqAo);
    VkShaderModule modSsr = ShaderUtils::createShaderModule(spvSsr);
    VkShaderModule modRqR = ShaderUtils::createShaderModule(spvRqR);
    VkShaderModule modSscs = ShaderUtils::createShaderModule(spvSscs);
    VkShaderModule modRqCs = ShaderUtils::createShaderModule(spvRqCs);

    // Descriptor set layouts
    // 1. Image+Sampler to Image (SSAO, Blur, SSR, SSCS)
    VkDescriptorSetLayoutBinding b2[3] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dsl2Info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, b2};
    VkDescriptorSetLayout dsl2;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dsl2Info, nullptr, &dsl2));

    // Blur Layout (inTex, depthTex, outFiltered)
    VkDescriptorSetLayoutBinding bBlur[3] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslBlurInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, bBlur};
    VkDescriptorSetLayout dslBlur;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslBlurInfo, nullptr, &dslBlur));

    // RQ Layout (TLAS, depthTex, outImage)
    VkDescriptorSetLayoutBinding bRq[3] = {
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslRqInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, bRq};
    VkDescriptorSetLayout dslRq;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslRqInfo, nullptr, &dslRq));

    // Pipeline Layouts
    VkPushConstantRange pcSsaoRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsSSAO)};
    VkPipelineLayoutCreateInfo plSsaoInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl2, 1, &pcSsaoRange};
    VkPipelineLayout plSsao;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plSsaoInfo, nullptr, &plSsao));
    VkPipeline pipeSsao = ShaderUtils::createComputePipeline(modSsao, plSsao);

    VkPushConstantRange pcBlurRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsBlur)};
    VkPipelineLayoutCreateInfo plBlurInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslBlur, 1, &pcBlurRange};
    VkPipelineLayout plBlur;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plBlurInfo, nullptr, &plBlur));
    VkPipeline pipeBlur = ShaderUtils::createComputePipeline(modBlur, plBlur);

    VkPushConstantRange pcRqAoRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQAO)};
    VkPipelineLayoutCreateInfo plRqAoInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslRq, 1, &pcRqAoRange};
    VkPipelineLayout plRqAo;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plRqAoInfo, nullptr, &plRqAo));
    VkPipeline pipeRqAo = ShaderUtils::createComputePipeline(modRqAo, plRqAo);

    VkPushConstantRange pcSsrRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsSSR)};
    VkPipelineLayoutCreateInfo plSsrInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl2, 1, &pcSsrRange};
    VkPipelineLayout plSsr;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plSsrInfo, nullptr, &plSsr));
    VkPipeline pipeSsr = ShaderUtils::createComputePipeline(modSsr, plSsr);

    VkPushConstantRange pcRqRRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQR)};
    VkPipelineLayoutCreateInfo plRqRInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslRq, 1, &pcRqRRange};
    VkPipelineLayout plRqR;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plRqRInfo, nullptr, &plRqR));
    VkPipeline pipeRqR = ShaderUtils::createComputePipeline(modRqR, plRqR);

    VkPushConstantRange pcSscsRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsSSCS)};
    VkPipelineLayoutCreateInfo plSscsInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl2, 1, &pcSscsRange};
    VkPipelineLayout plSscs;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plSscsInfo, nullptr, &plSscs));
    VkPipeline pipeSscs = ShaderUtils::createComputePipeline(modSscs, plSscs);

    VkPushConstantRange pcRqCsRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQCS)};
    VkPipelineLayoutCreateInfo plRqCsInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dslRq, 1, &pcRqCsRange};
    VkPipelineLayout plRqCs;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plRqCsInfo, nullptr, &plRqCs));
    VkPipeline pipeRqCs = ShaderUtils::createComputePipeline(modRqCs, plRqCs);

    // Descriptor Sets Allocation
    VkDescriptorPoolSize poolSizes[3] = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 8 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16 }
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 16, 3, poolSizes};
    VkDescriptorPool descPool;
    VK_CHECK(vkCreateDescriptorPool(g_vk.device, &poolInfo, nullptr, &descPool));

    VkDescriptorSet setSsao, setBlur, setRqAo, setSsr, setRqR, setSscs, setRqCs;
    VkDescriptorSetAllocateInfo aSsao{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dsl2};
    VkDescriptorSetAllocateInfo aBlur{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslBlur};
    VkDescriptorSetAllocateInfo aRq{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslRq};

    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aSsao, &setSsao));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aBlur, &setBlur));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aRq, &setRqAo));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aSsao, &setSsr));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aRq, &setRqR));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aSsao, &setSscs));
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &aRq, &setRqCs));

    VkDescriptorImageInfo dImgInfo{linearSampler, primaryDepth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo aoRawInfo{linearSampler, imgAORaw.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo aoFiltInfo{linearSampler, imgAOFiltered.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo rqrInfo{linearSampler, imgRQR.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo csInfo{linearSampler, imgContactShadow.view, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR tHandle = tlas.handle;
    asInfo.pAccelerationStructures = &tHandle;

    // SSAO Set
    VkWriteDescriptorSet wSsao[2]{};
    wSsao[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSsao, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wSsao[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSsao, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &aoRawInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 2, wSsao, 0, nullptr);

    // Blur Set (in: imgAORaw, depth: primaryDepth, out: imgAOFiltered)
    VkDescriptorImageInfo rawAoSampleInfo{linearSampler, imgAORaw.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wBlur[3]{};
    wBlur[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setBlur, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rawAoSampleInfo, nullptr, nullptr};
    wBlur[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setBlur, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wBlur[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setBlur, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &aoFiltInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 3, wBlur, 0, nullptr);

    // RQAO Set
    VkWriteDescriptorSet wRqAo[3]{};
    wRqAo[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, setRqAo, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
    wRqAo[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRqAo, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wRqAo[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRqAo, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &aoRawInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 3, wRqAo, 0, nullptr);

    // SSR Set
    VkWriteDescriptorSet wSsr[2]{};
    wSsr[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSsr, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wSsr[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSsr, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rqrInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 2, wSsr, 0, nullptr);

    // RQR Set
    VkWriteDescriptorSet wRqR[3]{};
    wRqR[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, setRqR, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
    wRqR[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRqR, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wRqR[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRqR, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rqrInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 3, wRqR, 0, nullptr);

    // SSCS Set
    VkWriteDescriptorSet wSscs[2]{};
    wSscs[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSscs, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wSscs[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setSscs, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &csInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 2, wSscs, 0, nullptr);

    // RQCS Set
    VkWriteDescriptorSet wRqCs[3]{};
    wRqCs[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, setRqCs, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
    wRqCs[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRqCs, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
    wRqCs[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRqCs, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &csInfo, nullptr, nullptr};
    vkUpdateDescriptorSets(g_vk.device, 3, wRqCs, 0, nullptr);

    std::filesystem::create_directories("results");
    CsvWriter csv("results/expD_screenspace.csv", {
        "Effect", "Technique", "SPP_or_Steps", "TimeMs", "FPS", "PSNR_vs_Ref_dB", "SSIM_vs_Ref", "Verdict", "p5Ms", "p95Ms"
    });

    GpuTimer timer;
    timer.init(64);

    // Helper to download single-channel R8 image
    auto downloadR8 = [&](VkImage img) -> std::vector<float> {
        std::vector<uint8_t> bytes(width * height);
        VulkanBuffer readBuf;
        readBuf.create(width * height, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        VkBufferImageCopy region{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {width, height, 1}};
        vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_GENERAL, readBuf.handle, 1, &region);
        g_vk.endSingleTimeCommands(cmd);

        readBuf.download(bytes.data(), bytes.size());
        readBuf.destroy();

        std::vector<float> floats(width * height);
        for (size_t i = 0; i < width * height; i++) {
            floats[i] = static_cast<float>(bytes[i]) / 255.0f;
        }
        return floats;
    };

    // Helper to download RGBA8 image
    auto downloadRGBA8 = [&](VkImage img) -> std::vector<float> {
        std::vector<uint8_t> bytes(width * height * 4);
        VulkanBuffer readBuf;
        readBuf.create(width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        VkBufferImageCopy region{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {width, height, 1}};
        vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_GENERAL, readBuf.handle, 1, &region);
        g_vk.endSingleTimeCommands(cmd);

        readBuf.download(bytes.data(), bytes.size());
        readBuf.destroy();

        std::vector<float> floats(width * height);
        for (size_t i = 0; i < width * height; i++) {
            floats[i] = static_cast<float>(bytes[i * 4 + 0]) / 255.0f;
        }
        return floats;
    };

    // ==========================================
    // EFFECT 1: AMBIENT OCCLUSION (SSAO vs RQAO)
    // ==========================================
    std::cout << "\n----------------------------------------------------\n";
    std::cout << "[Exp D] Evaluating Effect 1: Ambient Occlusion\n";
    std::cout << "----------------------------------------------------\n";

    // 1. Ground Truth Reference: RQAO 64 SPP
    std::cout << "Generating 64-sample ground truth AO reference...\n";
    PushConstantsRQAO pcAoRef{invViewProj, glm::vec4(camPos, 1.0f), glm::uvec2(width, height), 64, 4.0f};
    {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRqAo);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRqAo, 0, 1, &setRqAo, 0, nullptr);
        vkCmdPushConstants(cmd, plRqAo, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQAO), &pcAoRef);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        g_vk.endSingleTimeCommands(cmd);
    }
    std::vector<float> aoRef = downloadR8(imgAORaw.handle);

    // 2. Baseline: SSAO (16 taps) + Bilateral Filter
    PushConstantsSSAO pcSsao{invViewProj, proj, glm::uvec2(width, height), 2.5f, 0.02f};
    PushConstantsBlur pcBlurH{glm::uvec2(width, height), glm::vec2(1.0f, 0.0f), 0.05f};
    PushConstantsBlur pcBlurV{glm::uvec2(width, height), glm::vec2(0.0f, 1.0f), 0.05f};

    BenchmarkStats statsSsao = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        // SSAO pass
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeSsao);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plSsao, 0, 1, &setSsao, 0, nullptr);
        vkCmdPushConstants(cmd, plSsao, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsSSAO), &pcSsao);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

        imgAORaw.cmdTransition(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // Bilateral filter pass
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeBlur);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plBlur, 0, 1, &setBlur, 0, nullptr);
        vkCmdPushConstants(cmd, plBlur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsBlur), &pcBlurH);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

        imgAORaw.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    });

    std::vector<float> ssaoImg = downloadR8(imgAOFiltered.handle);
    ImageQualityMetrics ssaoQual = ImageMetrics::compute(ssaoImg, aoRef, width, height);
    std::cout << "  - SSAO + Bilateral Filter: " << statsSsao.medianMs << " ms (" << (1000.0f / statsSsao.medianMs) 
              << " FPS), PSNR: " << ssaoQual.psnr << " dB, SSIM: " << ssaoQual.ssim << "\n";
    csv.writeRowValues("AmbientOcclusion", "SSAO_Bilateral", "16_samples", statsSsao.medianMs, 1000.0f / statsSsao.medianMs, ssaoQual.psnr, ssaoQual.ssim, "Baseline", statsSsao.p5Ms, statsSsao.p95Ms);

    // 3. Ray Query AO at 1, 2, 4 SPP + Bilateral Filter
    for (uint32_t spp : { 1u, 2u, 4u }) {
        PushConstantsRQAO pcRqAo{invViewProj, glm::vec4(camPos, 1.0f), glm::uvec2(width, height), spp, 4.0f};

        BenchmarkStats statsRqAo = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRqAo);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRqAo, 0, 1, &setRqAo, 0, nullptr);
            vkCmdPushConstants(cmd, plRqAo, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQAO), &pcRqAo);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

            imgAORaw.cmdTransition(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeBlur);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plBlur, 0, 1, &setBlur, 0, nullptr);
            vkCmdPushConstants(cmd, plBlur, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsBlur), &pcBlurH);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

            imgAORaw.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        });

        std::vector<float> rqAoImg = downloadR8(imgAOFiltered.handle);
        ImageQualityMetrics rqQual = ImageMetrics::compute(rqAoImg, aoRef, width, height);

        std::string verdict = (statsRqAo.medianMs <= statsSsao.medianMs && rqQual.psnr >= ssaoQual.psnr) ? "WIN" : 
                              (rqQual.psnr > ssaoQual.psnr ? "QualityGain_ExtraCost" : "Loss");

        std::cout << "  - RQAO " << spp << " SPP + Filter: " << statsRqAo.medianMs << " ms (" << (1000.0f / statsRqAo.medianMs) 
                  << " FPS), PSNR: " << rqQual.psnr << " dB, SSIM: " << rqQual.ssim << " -> Verdict: " << verdict << "\n";
        csv.writeRowValues("AmbientOcclusion", "RQAO_Filtered", std::to_string(spp) + "_spp", statsRqAo.medianMs, 1000.0f / statsRqAo.medianMs, rqQual.psnr, rqQual.ssim, verdict, statsRqAo.p5Ms, statsRqAo.p95Ms);
    }

    // ==========================================
    // EFFECT 2: REFLECTIONS (SSR vs RQR)
    // ==========================================
    std::cout << "\n----------------------------------------------------\n";
    std::cout << "[Exp D] Evaluating Effect 2: Reflections\n";
    std::cout << "----------------------------------------------------\n";

    // 1. Ground Truth Reference: RQR 64 SPP
    std::cout << "Generating 64-sample ground truth reflection reference...\n";
    PushConstantsRQR pcReflRef{invViewProj, glm::vec4(camPos, 1.0f), glm::uvec2(width, height), 64, 200.0f};
    {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRqR);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRqR, 0, 1, &setRqR, 0, nullptr);
        vkCmdPushConstants(cmd, plRqR, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQR), &pcReflRef);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        g_vk.endSingleTimeCommands(cmd);
    }
    std::vector<float> reflRef = downloadRGBA8(imgRQR.handle);

    // 2. Baseline: Screen Space Reflections (32 steps)
    PushConstantsSSR pcSsr{invViewProj, proj, glm::vec4(camPos, 1.0f), glm::uvec2(width, height), 100.0f, 32};
    BenchmarkStats statsSsr = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeSsr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plSsr, 0, 1, &setSsr, 0, nullptr);
        vkCmdPushConstants(cmd, plSsr, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsSSR), &pcSsr);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    });

    std::vector<float> ssrImg = downloadRGBA8(imgRQR.handle);
    ImageQualityMetrics ssrQual = ImageMetrics::compute(ssrImg, reflRef, width, height);
    std::cout << "  - SSR (32 Steps): " << statsSsr.medianMs << " ms (" << (1000.0f / statsSsr.medianMs) 
              << " FPS), PSNR: " << ssrQual.psnr << " dB, SSIM: " << ssrQual.ssim << "\n";
    csv.writeRowValues("Reflections", "SSR_Raymarching", "32_steps", statsSsr.medianMs, 1000.0f / statsSsr.medianMs, ssrQual.psnr, ssrQual.ssim, "Baseline", statsSsr.p5Ms, statsSsr.p95Ms);

    // 3. Ray Query Reflections at 1, 2, 4 SPP
    for (uint32_t spp : { 1u, 2u, 4u }) {
        PushConstantsRQR pcRqR{invViewProj, glm::vec4(camPos, 1.0f), glm::uvec2(width, height), spp, 200.0f};

        BenchmarkStats statsRqR = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRqR);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRqR, 0, 1, &setRqR, 0, nullptr);
            vkCmdPushConstants(cmd, plRqR, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQR), &pcRqR);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        std::vector<float> rqRImg = downloadRGBA8(imgRQR.handle);
        ImageQualityMetrics rqQual = ImageMetrics::compute(rqRImg, reflRef, width, height);

        std::string verdict = (statsRqR.medianMs <= statsSsr.medianMs && rqQual.psnr >= ssrQual.psnr) ? "WIN" : 
                              (rqQual.psnr > ssrQual.psnr ? "QualityGain_ExtraCost" : "Loss");

        std::cout << "  - RQR " << spp << " SPP: " << statsRqR.medianMs << " ms (" << (1000.0f / statsRqR.medianMs) 
                  << " FPS), PSNR: " << rqQual.psnr << " dB, SSIM: " << rqQual.ssim << " -> Verdict: " << verdict << "\n";
        csv.writeRowValues("Reflections", "RQR_Hardware", std::to_string(spp) + "_spp", statsRqR.medianMs, 1000.0f / statsRqR.medianMs, rqQual.psnr, rqQual.ssim, verdict, statsRqR.p5Ms, statsRqR.p95Ms);
    }

    // ==========================================
    // EFFECT 3: CONTACT SHADOWS (SSCS vs RQCS)
    // ==========================================
    std::cout << "\n----------------------------------------------------\n";
    std::cout << "[Exp D] Evaluating Effect 3: Contact Shadows\n";
    std::cout << "----------------------------------------------------\n";

    // 1. Ground Truth Reference: RQCS 32 SPP
    std::cout << "Generating 32-sample ground truth contact shadow reference...\n";
    PushConstantsRQCS pcCsRef{invViewProj, glm::vec4(sunDir, 0.0f), glm::uvec2(width, height), 32, 2.0f};
    {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRqCs);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRqCs, 0, 1, &setRqCs, 0, nullptr);
        vkCmdPushConstants(cmd, plRqCs, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQCS), &pcCsRef);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        g_vk.endSingleTimeCommands(cmd);
    }
    std::vector<float> csRef = downloadR8(imgContactShadow.handle);

    // 2. Baseline: Screen Space Contact Shadows (16 steps)
    PushConstantsSSCS pcSscs{invViewProj, proj, glm::vec4(sunDir, 0.0f), glm::uvec2(width, height), 1.5f, 16};
    BenchmarkStats statsSscs = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeSscs);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plSscs, 0, 1, &setSscs, 0, nullptr);
        vkCmdPushConstants(cmd, plSscs, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsSSCS), &pcSscs);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    });

    std::vector<float> sscsImg = downloadR8(imgContactShadow.handle);
    ImageQualityMetrics sscsQual = ImageMetrics::compute(sscsImg, csRef, width, height);
    std::cout << "  - SSCS (16 Steps): " << statsSscs.medianMs << " ms (" << (1000.0f / statsSscs.medianMs) 
              << " FPS), PSNR: " << sscsQual.psnr << " dB, SSIM: " << sscsQual.ssim << "\n";
    csv.writeRowValues("ContactShadows", "SSCS_Raymarching", "16_steps", statsSscs.medianMs, 1000.0f / statsSscs.medianMs, sscsQual.psnr, sscsQual.ssim, "Baseline", statsSscs.p5Ms, statsSscs.p95Ms);

    // 3. Ray Query Contact Shadows at 1, 2, 4 SPP
    for (uint32_t spp : { 1u, 2u, 4u }) {
        PushConstantsRQCS pcRqCs{invViewProj, glm::vec4(sunDir, 0.0f), glm::uvec2(width, height), spp, 2.0f};

        BenchmarkStats statsRqCs = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRqCs);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRqCs, 0, 1, &setRqCs, 0, nullptr);
            vkCmdPushConstants(cmd, plRqCs, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRQCS), &pcRqCs);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        std::vector<float> rqCsImg = downloadR8(imgContactShadow.handle);
        ImageQualityMetrics rqQual = ImageMetrics::compute(rqCsImg, csRef, width, height);

        std::string verdict = (statsRqCs.medianMs <= statsSscs.medianMs && rqQual.psnr >= sscsQual.psnr) ? "WIN" : 
                              (rqQual.psnr > sscsQual.psnr ? "QualityGain_ExtraCost" : "Loss");

        std::cout << "  - RQCS " << spp << " SPP: " << statsRqCs.medianMs << " ms (" << (1000.0f / statsRqCs.medianMs) 
                  << " FPS), PSNR: " << rqQual.psnr << " dB, SSIM: " << rqQual.ssim << " -> Verdict: " << verdict << "\n";
        csv.writeRowValues("ContactShadows", "RQCS_Hardware", std::to_string(spp) + "_spp", statsRqCs.medianMs, 1000.0f / statsRqCs.medianMs, rqQual.psnr, rqQual.ssim, verdict, statsRqCs.p5Ms, statsRqCs.p95Ms);
    }

    std::cout << "\n[Exp D] Results successfully saved to results/expD_screenspace.csv\n";
    GpuClockInfo::printStatus();

    // Cleanup
    primaryDepth.destroy();
    imgAORaw.destroy();
    imgAOFiltered.destroy();
    imgRQR.destroy();
    imgContactShadow.destroy();
    vkDestroySampler(g_vk.device, linearSampler, nullptr);

    blas.destroy();
    tlas.destroy();
    instBuf.destroy();
    vBuf.destroy();
    iBuf.destroy();

    vkDestroyPipeline(g_vk.device, pipeDepth, nullptr);
    vkDestroyPipeline(g_vk.device, pipeSsao, nullptr);
    vkDestroyPipeline(g_vk.device, pipeBlur, nullptr);
    vkDestroyPipeline(g_vk.device, pipeRqAo, nullptr);
    vkDestroyPipeline(g_vk.device, pipeSsr, nullptr);
    vkDestroyPipeline(g_vk.device, pipeRqR, nullptr);
    vkDestroyPipeline(g_vk.device, pipeSscs, nullptr);
    vkDestroyPipeline(g_vk.device, pipeRqCs, nullptr);

    vkDestroyPipelineLayout(g_vk.device, plDepth, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plSsao, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plBlur, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plRqAo, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plSsr, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plRqR, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plSscs, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plRqCs, nullptr);

    vkDestroyDescriptorPool(g_vk.device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dsl2, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslBlur, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslRq, nullptr);

    vkDestroyShaderModule(g_vk.device, modDepthV, nullptr);
    vkDestroyShaderModule(g_vk.device, modDepthF, nullptr);
    vkDestroyShaderModule(g_vk.device, modSsao, nullptr);
    vkDestroyShaderModule(g_vk.device, modBlur, nullptr);
    vkDestroyShaderModule(g_vk.device, modRqAo, nullptr);
    vkDestroyShaderModule(g_vk.device, modSsr, nullptr);
    vkDestroyShaderModule(g_vk.device, modRqR, nullptr);
    vkDestroyShaderModule(g_vk.device, modSscs, nullptr);
    vkDestroyShaderModule(g_vk.device, modRqCs, nullptr);

    g_vk.cleanup();
    return 0;
}
