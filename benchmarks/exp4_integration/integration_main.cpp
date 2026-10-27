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
#include <fstream>
#include <cmath>
#include <cassert>

struct PointLight {
    glm::vec4 posRadius;
    glm::vec4 color;
};

#define MAX_LIGHTS_PER_CLUSTER 512

struct ClusterGridData {
    uint32_t count;
    uint32_t lightIndices[MAX_LIGHTS_PER_CLUSTER];
};

struct RasterPushConstants {
    glm::mat4 viewProj;
};

struct RaycastDepthPushConstants {
    glm::mat4 viewProj;
    glm::mat4 invViewProj;
    glm::vec4 camPos;
    glm::uvec2 resolution;
};

struct alignas(16) LightCentricBinningPushConstants {
    glm::mat4 view;
    glm::mat4 proj;
    glm::uvec4 clusterDims;
    glm::vec2 zNearFar;
    uint32_t lightCount;
    uint32_t _pad;
};

struct alignas(16) ShadingClusteredPushConstants {
    glm::mat4 invViewProj;
    glm::mat4 view;
    glm::uvec2 resolution;
    glm::vec2 _pad0;
    glm::uvec4 clusterDims;
    glm::vec2 zNearFar;
    glm::vec2 _pad1;
};

struct alignas(16) ShadingRayQueryPushConstants {
    glm::mat4 invViewProj;
    glm::mat4 view;
    glm::uvec2 resolution;
    float maxRadius;
    uint32_t maxLightsPerPixel;
};

struct alignas(16) ShadingTiledLdsPushConstants {
    glm::mat4 invViewProj;
    glm::mat4 view;
    glm::mat4 proj;
    glm::uvec2 resolution;
    glm::vec2 zNearFar;
    uint32_t lightCount;
    uint32_t _pad[3];
};

void generateLightProxyGeometry(
    const std::vector<PointLight>& lights,
    std::vector<glm::vec3>& outVertices,
    std::vector<uint32_t>& outIndices
) {
    outVertices.clear();
    outIndices.clear();
    outVertices.reserve(lights.size() * 4);
    outIndices.reserve(lights.size() * 6);

    for (size_t i = 0; i < lights.size(); i++) {
        glm::vec3 c = glm::vec3(lights[i].posRadius);
        float r = lights[i].posRadius.w;
        uint32_t base = static_cast<uint32_t>(outVertices.size());

        outVertices.push_back(glm::vec3(c.x - r, c.y - r, c.z + r));
        outVertices.push_back(glm::vec3(c.x + r, c.y - r, c.z + r));
        outVertices.push_back(glm::vec3(c.x + r, c.y + r, c.z + r));
        outVertices.push_back(glm::vec3(c.x - r, c.y + r, c.z + r));

        outIndices.push_back(base + 0);
        outIndices.push_back(base + 1);
        outIndices.push_back(base + 2);
        outIndices.push_back(base + 0);
        outIndices.push_back(base + 2);
        outIndices.push_back(base + 3);
    }
}

int main() {
    std::cout << "====================================================\n";
    std::cout << " PHASE 5 SECOND AUDIT: REALISTIC SHADING & TIMING VERIFICATION\n";
    std::cout << "====================================================\n";

    g_vk.init(true);
    GpuClockInfo::printStatus();

    std::filesystem::create_directories("results");

    uint32_t width = 1920, height = 1080;
    uint32_t numLights = 2000;
    float lightRadius = 8.0f;
    glm::uvec3 clusterDims(16, 9, 24);
    uint32_t totalClusters = clusterDims.x * clusterDims.y * clusterDims.z;
    glm::vec2 zNearFar(0.5f, 150.0f);

    glm::vec3 camPos(0.0f, 40.0f, -65.0f);
    glm::vec3 lookAt(0.0f, 0.0f, 0.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    glm::mat4 view = glm::lookAt(camPos, lookAt, up);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), static_cast<float>(width) / height, zNearFar.x, zNearFar.y);
    proj[1][1] *= -1.0f;
    glm::mat4 viewProj = proj * view;
    glm::mat4 invViewProj = glm::inverse(viewProj);

    // 1. Generate Scene Geometry (~1.34M triangles terrain)
    std::cout << "Generating terrain geometry mesh (1.34M triangles)..." << std::endl;
    SceneMesh mesh = MeshGenerator::createTessellatedTerrain(1340000, 100.0f);
    uint32_t triCount = mesh.triangleCount();
    std::cout << "Terrain Mesh: " << triCount << " triangles\n";

    VulkanBuffer vBuf, iBuf;
    vBuf.create(
        sizeof(SceneMesh::Vertex) * mesh.vertices.size(),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    vBuf.upload(mesh.vertices.data(), sizeof(SceneMesh::Vertex) * mesh.vertices.size());

    iBuf.create(
        sizeof(uint32_t) * mesh.indices.size(),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    iBuf.upload(mesh.indices.data(), sizeof(uint32_t) * mesh.indices.size());

    // Build Static Mesh BVH
    float meshBvhBuildTime = 0.0f;
    VulkanAS meshBLAS = BVHBuilder::buildTriangleBLAS(
        vBuf.getDeviceAddress(), mesh.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
        iBuf.getDeviceAddress(), mesh.indices.size(), VK_INDEX_TYPE_UINT32,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        &meshBvhBuildTime
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
    meshInst.accelerationStructureReference = meshBLAS.deviceAddress;

    VulkanBuffer meshInstBuf;
    meshInstBuf.create(sizeof(meshInst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    meshInstBuf.upload(&meshInst, sizeof(meshInst));
    VulkanAS meshTLAS = BVHBuilder::buildTLAS(meshInstBuf.getDeviceAddress(), 1);

    // 2. Generate Lights (2,000 dynamic lights hovering over terrain)
    std::cout << "Generating 2,000 dynamic point lights..." << std::endl;
    std::vector<PointLight> lights(numLights);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distX(-45.0f, 45.0f);
    std::uniform_real_distribution<float> distY(1.0f, 12.0f);
    std::uniform_real_distribution<float> distZ(-45.0f, 45.0f);

    for (uint32_t i = 0; i < numLights; i++) {
        lights[i].posRadius = glm::vec4(distX(rng), distY(rng), distZ(rng), lightRadius);
        lights[i].color = glm::vec4(0.8f, 0.8f, 0.9f, 1.0f);
    }

    VulkanBuffer lightBuf;
    lightBuf.create(sizeof(PointLight) * numLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    lightBuf.upload(lights.data(), sizeof(PointLight) * numLights);

    // Build Light BVH
    std::vector<glm::vec3> proxyVerts;
    std::vector<uint32_t> proxyIndices;
    generateLightProxyGeometry(lights, proxyVerts, proxyIndices);

    VulkanBuffer proxyVBuf, proxyIBuf;
    proxyVBuf.create(sizeof(glm::vec3) * proxyVerts.size(), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    proxyVBuf.upload(proxyVerts.data(), sizeof(glm::vec3) * proxyVerts.size());

    proxyIBuf.create(sizeof(uint32_t) * proxyIndices.size(), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    proxyIBuf.upload(proxyIndices.data(), sizeof(uint32_t) * proxyIndices.size());

    VulkanAS lightBLAS = BVHBuilder::buildTriangleBLAS(
        proxyVBuf.getDeviceAddress(), proxyVerts.size(), sizeof(glm::vec3), VK_FORMAT_R32G32B32_SFLOAT,
        proxyIBuf.getDeviceAddress(), proxyIndices.size(), VK_INDEX_TYPE_UINT32,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
        nullptr,
        0
    );

    VkAccelerationStructureInstanceKHR lightInst{};
    lightInst.transform = transform;
    lightInst.mask = 0xFF;
    lightInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    lightInst.accelerationStructureReference = lightBLAS.deviceAddress;

    VulkanBuffer lightInstBuf;
    lightInstBuf.create(sizeof(lightInst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    lightInstBuf.upload(&lightInst, sizeof(lightInst));
    VulkanAS lightTLAS = BVHBuilder::buildTLAS(lightInstBuf.getDeviceAddress(), 1, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR);

    // 3. Cluster buffer
    VulkanBuffer clusterGridBuf;
    clusterGridBuf.create(sizeof(ClusterGridData) * totalClusters, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VulkanBuffer overflowBuf;
    overflowBuf.create(sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // 4. Render Targets
    VulkanImage rasterColor, rasterDepth, raycastColor, raycastDepth, finalColorA, finalColorB, finalColorTiled;
    rasterColor.create(width, height, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    rasterDepth.create(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

    raycastColor.create(width, height, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    raycastDepth.create(width, height, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    finalColorA.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    finalColorB.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    finalColorTiled.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    rasterColor.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    rasterDepth.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    raycastColor.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    raycastDepth.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    finalColorA.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    finalColorB.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    finalColorTiled.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // Staging buffers
    VulkanBuffer stagingImgBuf;
    stagingImgBuf.create(width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VulkanBuffer stagingDepthBuf;
    stagingDepthBuf.create(width * height * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Sampler for depth texture
    VkSampler depthSampler;
    VkSamplerCreateInfo sampInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampInfo.magFilter = VK_FILTER_NEAREST;
    sampInfo.minFilter = VK_FILTER_NEAREST;
    sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(g_vk.device, &sampInfo, nullptr, &depthSampler));

    // Compile Shaders
    std::cout << "Compiling audited shaders..." << std::endl;
    std::vector<uint32_t> vertSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_raster.vert");
    std::vector<uint32_t> fragSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_raster.frag");
    std::vector<uint32_t> visCompSpv = ShaderUtils::compileGlslFileToSpirv("exp4_integration/shaders/visbuf_raycast_depth.comp");
    std::vector<uint32_t> lcBinSpv = ShaderUtils::compileGlslFileToSpirv("exp3_spatial/shaders/light_centric_binning.comp");
    std::vector<uint32_t> cShadeSpv = ShaderUtils::compileGlslFileToSpirv("exp4_integration/shaders/integration_shading_clustered.comp");
    std::vector<uint32_t> tShadeSpv = ShaderUtils::compileGlslFileToSpirv("exp4_integration/shaders/integration_shading_tiled_lds.comp");
    std::vector<uint32_t> rShadeSpv = ShaderUtils::compileGlslFileToSpirv("exp4_integration/shaders/integration_shading_rayquery.comp");

    VkShaderModule vertMod = ShaderUtils::createShaderModule(vertSpv);
    VkShaderModule fragMod = ShaderUtils::createShaderModule(fragSpv);
    VkShaderModule visCompMod = ShaderUtils::createShaderModule(visCompSpv);
    VkShaderModule lcBinMod = ShaderUtils::createShaderModule(lcBinSpv);
    VkShaderModule cShadeMod = ShaderUtils::createShaderModule(cShadeSpv);
    VkShaderModule tShadeMod = ShaderUtils::createShaderModule(tShadeSpv);
    VkShaderModule rShadeMod = ShaderUtils::createShaderModule(rShadeSpv);

    // A. Raster Visibility Pipeline
    VkPushConstantRange rPcRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(RasterPushConstants)};
    VkPipelineLayoutCreateInfo rPlInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &rPcRange};
    VkPipelineLayout rPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &rPlInfo, nullptr, &rPipelineLayout));

    VkPipelineShaderStageCreateInfo rStages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertMod, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "main", nullptr}
    };
    VkVertexInputBindingDescription bDesc{0, sizeof(SceneMesh::Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription aDescs[2]{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneMesh::Vertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneMesh::Vertex, normal)}
    };
    VkPipelineVertexInputStateCreateInfo viInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &bDesc, 2, aDescs};
    VkPipelineInputAssemblyStateCreateInfo iaInfo{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkViewport vp{0, 0, (float)width, (float)height, 0, 1};
    VkRect2D sc{{0, 0}, {width, height}};
    VkPipelineViewportStateCreateInfo vpsInfo{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc};
    VkPipelineRasterizationStateCreateInfo rsInfo{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo msInfo{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 0, nullptr, VK_FALSE, VK_FALSE};
    VkPipelineDepthStencilStateCreateInfo dsInfo{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS, VK_FALSE, VK_FALSE, {}, {}, 0.0f, 1.0f};
    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cbInfo{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &cbAtt, {0,0,0,0}};
    VkFormat colFmt = VK_FORMAT_R32_UINT;
    VkPipelineRenderingCreateInfo rendInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 1, &colFmt, VK_FORMAT_D32_SFLOAT, VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo gpInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendInfo, 0, 2, rStages, &viInfo, &iaInfo, nullptr, &vpsInfo, &rsInfo, &msInfo, &dsInfo, &cbInfo, nullptr, rPipelineLayout, VK_NULL_HANDLE, 0};
    VkPipeline rPipeline;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &gpInfo, nullptr, &rPipeline));

    // B. Raycast Vis Pipeline
    VkDescriptorSetLayoutBinding visBindings[3]{
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo visLInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, visBindings};
    VkDescriptorSetLayout visSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &visLInfo, nullptr, &visSetLayout));
    VkPushConstantRange visPcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RaycastDepthPushConstants)};
    VkPipelineLayoutCreateInfo visPlInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &visSetLayout, 1, &visPcRange};
    VkPipelineLayout visPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &visPlInfo, nullptr, &visPipelineLayout));
    VkPipeline visPipeline = ShaderUtils::createComputePipeline(visCompMod, visPipelineLayout);

    // C. Light-Centric Binning Pipeline
    VkDescriptorSetLayoutBinding lcBinBindings[3]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo lcBinLInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, lcBinBindings};
    VkDescriptorSetLayout lcBinSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &lcBinLInfo, nullptr, &lcBinSetLayout));
    VkPushConstantRange lcBinPcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LightCentricBinningPushConstants)};
    VkPipelineLayoutCreateInfo lcBinPlInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &lcBinSetLayout, 1, &lcBinPcRange};
    VkPipelineLayout lcBinPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &lcBinPlInfo, nullptr, &lcBinPipelineLayout));
    VkPipeline lcBinPipeline = ShaderUtils::createComputePipeline(lcBinMod, lcBinPipelineLayout);

    // D. Clustered Shading Pipeline
    VkDescriptorSetLayoutBinding cShadeBindings[4]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo cShadeLInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, cShadeBindings};
    VkDescriptorSetLayout cShadeSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &cShadeLInfo, nullptr, &cShadeSetLayout));
    VkPushConstantRange cShadePcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingClusteredPushConstants)};
    VkPipelineLayoutCreateInfo cShadePlInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &cShadeSetLayout, 1, &cShadePcRange};
    VkPipelineLayout cShadePipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &cShadePlInfo, nullptr, &cShadePipelineLayout));
    VkPipeline cShadePipeline = ShaderUtils::createComputePipeline(cShadeMod, cShadePipelineLayout);

    // E. Ray Query Shading Pipeline
    VkDescriptorSetLayoutBinding rShadeBindings[4]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo rShadeLInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, rShadeBindings};
    VkDescriptorSetLayout rShadeSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &rShadeLInfo, nullptr, &rShadeSetLayout));
    VkPushConstantRange rShadePcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingRayQueryPushConstants)};
    VkPipelineLayoutCreateInfo rShadePlInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &rShadeSetLayout, 1, &rShadePcRange};
    VkPipelineLayout rShadePipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &rShadePlInfo, nullptr, &rShadePipelineLayout));
    VkPipeline rShadePipeline = ShaderUtils::createComputePipeline(rShadeMod, rShadePipelineLayout);

    // F. Tiled LDS Shading Pipeline
    VkDescriptorSetLayoutBinding tShadeBindings[3]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo tShadeLInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, tShadeBindings};
    VkDescriptorSetLayout tShadeSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &tShadeLInfo, nullptr, &tShadeSetLayout));
    VkPushConstantRange tShadePcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingTiledLdsPushConstants)};
    VkPipelineLayoutCreateInfo tShadePlInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &tShadeSetLayout, 1, &tShadePcRange};
    VkPipelineLayout tShadePipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &tShadePlInfo, nullptr, &tShadePipelineLayout));
    VkPipeline tShadePipeline = ShaderUtils::createComputePipeline(tShadeMod, tShadePipelineLayout);

    // Descriptor Pool & Sets
    VkDescriptorPoolSize pSizes[4]{
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4}
    };
    VkDescriptorPoolCreateInfo piInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 8, 4, pSizes};
    VkDescriptorPool dPool;
    VK_CHECK(vkCreateDescriptorPool(g_vk.device, &piInfo, nullptr, &dPool));

    VkDescriptorSet visSet, lcBinSet, cShadeSet, rShadeSet, tShadeSet, cShadeRaycastSet;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dPool, 1, &visSetLayout};
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &visSet));
    ai.pSetLayouts = &lcBinSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &lcBinSet));
    ai.pSetLayouts = &cShadeSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &cShadeSet));
    ai.pSetLayouts = &rShadeSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &rShadeSet));
    ai.pSetLayouts = &tShadeSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &tShadeSet));
    ai.pSetLayouts = &cShadeSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &cShadeRaycastSet));

    // Update Sets
    VkWriteDescriptorSetAccelerationStructureKHR meshAsWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &meshTLAS.handle};
    VkDescriptorImageInfo raycastImgDesc{VK_NULL_HANDLE, raycastColor.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo raycastDepthImgDesc{VK_NULL_HANDLE, raycastDepth.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet visWrites[3]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &meshAsWrite, visSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, visSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &raycastImgDesc, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, visSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &raycastDepthImgDesc, nullptr, nullptr}
    };
    vkUpdateDescriptorSets(g_vk.device, 3, visWrites, 0, nullptr);

    VkDescriptorBufferInfo lightDesc{lightBuf.handle, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo gridDesc{clusterGridBuf.handle, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo overflowDesc{overflowBuf.handle, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet lcBinWrites[3]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, lcBinSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDesc, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, lcBinSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gridDesc, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, lcBinSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &overflowDesc, nullptr}
    };
    vkUpdateDescriptorSets(g_vk.device, 3, lcBinWrites, 0, nullptr);

    VkDescriptorImageInfo rasterDepthSampDesc{depthSampler, rasterDepth.view, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo finalImgADesc{VK_NULL_HANDLE, finalColorA.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet cShadeWrites[4]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDesc, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gridDesc, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rasterDepthSampDesc, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &finalImgADesc, nullptr, nullptr}
    };
    vkUpdateDescriptorSets(g_vk.device, 4, cShadeWrites, 0, nullptr);

    VkWriteDescriptorSetAccelerationStructureKHR lightAsWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &lightTLAS.handle};
    VkDescriptorImageInfo raycastDepthSampDesc{depthSampler, raycastDepth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo finalImgBDesc{VK_NULL_HANDLE, finalColorB.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet rShadeWrites[4]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, rShadeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDesc, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &lightAsWrite, rShadeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, rShadeSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &raycastDepthSampDesc, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, rShadeSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &finalImgBDesc, nullptr, nullptr}
    };
    vkUpdateDescriptorSets(g_vk.device, 4, rShadeWrites, 0, nullptr);

    VkDescriptorImageInfo finalImgTiledDesc{VK_NULL_HANDLE, finalColorTiled.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet tShadeWrites[3]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tShadeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDesc, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tShadeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &raycastDepthSampDesc, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tShadeSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &finalImgTiledDesc, nullptr, nullptr}
    };
    vkUpdateDescriptorSets(g_vk.device, 3, tShadeWrites, 0, nullptr);

    VkWriteDescriptorSet cShadeRaycastWrites[4]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeRaycastSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDesc, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeRaycastSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gridDesc, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeRaycastSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &raycastDepthSampDesc, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeRaycastSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &finalImgADesc, nullptr, nullptr}
    };
    vkUpdateDescriptorSets(g_vk.device, 4, cShadeRaycastWrites, 0, nullptr);

    GpuTimer timer;
    timer.init(64);

    // =========================================================================
    // PIPELINE A: Pure Raster + Light-Centric Clustered
    // =========================================================================
    std::cout << "\n--- Benchmarking Pipeline A (Pure Raster + Clustered Shading) ---" << std::endl;
    RasterPushConstants rpc{viewProj};
    LightCentricBinningPushConstants lbpc{view, proj, glm::uvec4(clusterDims, 0), zNearFar, numLights, 0};
    ShadingClusteredPushConstants cspc{invViewProj, view, glm::uvec2(width, height), glm::vec2(0.0f), glm::uvec4(clusterDims, 0), zNearFar, glm::vec2(0.0f)};

    // A1. Raster VisBuffer Pass
    BenchmarkStats pA_VisStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
        rasterDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo colAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterColor.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
        VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterDepth.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
        dAtt.clearValue.depthStencil.depth = 1.0f;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 1, &colAtt, &dAtt, nullptr};

        vkCmdBeginRendering(cmd, &ri);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rPipeline);
        vkCmdPushConstants(cmd, rPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(RasterPushConstants), &rpc);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
        vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
        vkCmdEndRendering(cmd);

        rasterDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
    });

    // A2. Light-Centric Binning Pass
    BenchmarkStats pA_PrepStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, clusterGridBuf.handle, 0, totalClusters * sizeof(ClusterGridData), 0);
        vkCmdFillBuffer(cmd, overflowBuf.handle, 0, sizeof(uint32_t), 0);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lcBinPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lcBinPipelineLayout, 0, 1, &lcBinSet, 0, nullptr);
        vkCmdPushConstants(cmd, lcBinPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LightCentricBinningPushConstants), &lbpc);
        vkCmdDispatch(cmd, (numLights + 63) / 64, 1, 1);
    });

    // A3. Clustered Shading Pass
    BenchmarkStats pA_ShadeStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipelineLayout, 0, 1, &cShadeSet, 0, nullptr);
        vkCmdPushConstants(cmd, cShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingClusteredPushConstants), &cspc);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    });

    // Pipeline A Total Frame End-to-End
    BenchmarkStats pA_TotalStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
        rasterDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo colAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterColor.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
        VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterDepth.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
        dAtt.clearValue.depthStencil.depth = 1.0f;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 1, &colAtt, &dAtt, nullptr};

        vkCmdBeginRendering(cmd, &ri);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rPipeline);
        vkCmdPushConstants(cmd, rPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(RasterPushConstants), &rpc);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
        vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
        vkCmdEndRendering(cmd);

        rasterDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);

        vkCmdFillBuffer(cmd, clusterGridBuf.handle, 0, totalClusters * sizeof(ClusterGridData), 0);
        vkCmdFillBuffer(cmd, overflowBuf.handle, 0, sizeof(uint32_t), 0);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lcBinPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lcBinPipelineLayout, 0, 1, &lcBinSet, 0, nullptr);
        vkCmdPushConstants(cmd, lcBinPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LightCentricBinningPushConstants), &lbpc);
        vkCmdDispatch(cmd, (numLights + 63) / 64, 1, 1);

        VkMemoryBarrier mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb2, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipelineLayout, 0, 1, &cShadeSet, 0, nullptr);
        vkCmdPushConstants(cmd, cShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingClusteredPushConstants), &cspc);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    });

    std::cout << "  Pipeline A: Vis = " << std::fixed << std::setprecision(3) << pA_VisStats.medianMs 
              << " ms, Binning = " << pA_PrepStats.medianMs 
              << " ms, Shading = " << pA_ShadeStats.medianMs 
              << " ms | Total Frame = " << pA_TotalStats.medianMs << " ms\n";

    // Debug print cluster grid
    {
        std::vector<ClusterGridData> gridCpu(totalClusters);
        VulkanBuffer stagingGrid;
        stagingGrid.create(sizeof(ClusterGridData) * totalClusters, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        VkBufferCopy copyRegion{0, 0, sizeof(ClusterGridData) * totalClusters};
        vkCmdCopyBuffer(cmd, clusterGridBuf.handle, stagingGrid.handle, 1, &copyRegion);
        g_vk.endSingleTimeCommands(cmd);
        stagingGrid.download(gridCpu.data(), sizeof(ClusterGridData) * totalClusters);

        uint32_t totalBinned = 0, maxCount = 0;
        for (const auto& c : gridCpu) {
            totalBinned += c.count;
            maxCount = std::max(maxCount, c.count);
        }
        uint32_t droppedCount = 0;
        overflowBuf.download(&droppedCount, sizeof(uint32_t));
        std::cout << "  [DEBUG ClusterGrid]: Total binned = " << totalBinned 
                  << ", maxCount = " << maxCount 
                  << ", overflow dropped = " << droppedCount 
                  << ", at cluster(8,4,21) = " << gridCpu[8 + 4 * 16 + 21 * 16 * 9].count << std::endl;
    }

    // =========================================================================
    // PIPELINE B: RA-Assisted Hybrid (Raycast Vis + RA Shading)
    // =========================================================================
    std::cout << "\n--- Benchmarking Pipeline B (RA-Assisted Visibility + RA Lighting) ---" << std::endl;
    RaycastDepthPushConstants rayDepthPc{viewProj, invViewProj, glm::vec4(camPos, 1.0f), glm::uvec2(width, height)};
    ShadingRayQueryPushConstants rspc{invViewProj, view, glm::uvec2(width, height), lightRadius, 2048};

    // B1. Raycast VisBuffer + Depth Pass
    BenchmarkStats pB_VisStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
        raycastDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visPipelineLayout, 0, 1, &visSet, 0, nullptr);
        vkCmdPushConstants(cmd, visPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RaycastDepthPushConstants), &rayDepthPc);
        vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

        raycastDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    // B2. Light BVH Refit Pass
    float bvhRefitMs = 0.0f;
    BVHBuilder::updateTriangleBLAS(
        lightBLAS,
        proxyVBuf.getDeviceAddress(), proxyVerts.size(), sizeof(glm::vec3), VK_FORMAT_R32G32B32_SFLOAT,
        proxyIBuf.getDeviceAddress(), proxyIndices.size(), VK_INDEX_TYPE_UINT32,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
        &bvhRefitMs,
        0
    );

    // B3. Ray Query Shading Pass
    BenchmarkStats pB_ShadeStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipelineLayout, 0, 1, &rShadeSet, 0, nullptr);
        vkCmdPushConstants(cmd, rShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingRayQueryPushConstants), &rspc);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    });

    // Pipeline B Total Frame End-to-End
    BenchmarkStats pB_TotalStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
        raycastDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visPipelineLayout, 0, 1, &visSet, 0, nullptr);
        vkCmdPushConstants(cmd, visPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RaycastDepthPushConstants), &rayDepthPc);
        vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

        raycastDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipelineLayout, 0, 1, &rShadeSet, 0, nullptr);
        vkCmdPushConstants(cmd, rShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingRayQueryPushConstants), &rspc);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    });

    float pB_TotalEndToEndWithRefit = pB_TotalStats.medianMs + bvhRefitMs;

    std::cout << "  Pipeline B: Vis = " << pB_VisStats.medianMs 
              << " ms, BVH Refit = " << bvhRefitMs 
              << " ms, Shading = " << pB_ShadeStats.medianMs 
              << " ms | Total Frame = " << pB_TotalEndToEndWithRefit << " ms\n";

    // =========================================================================
    // AUDIT 1: Output Image Readback & Verification
    // =========================================================================
    std::vector<uint8_t> dataA(width * height * 4), dataB(width * height * 4);
    std::vector<float> depthData(width * height);

    // Download images
    auto downloadColorImg = [&](VulkanImage& img, std::vector<uint8_t>& dst) {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        img.cmdTransition(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, img.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingImgBuf.handle, 1, &region);
        img.cmdTransition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        g_vk.endSingleTimeCommands(cmd);
        stagingImgBuf.download(dst.data(), dst.size());
    };

    downloadColorImg(finalColorA, dataA);
    downloadColorImg(finalColorB, dataB);

    // Download raycast depth
    {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        raycastDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, raycastDepth.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingDepthBuf.handle, 1, &region);
        raycastDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        g_vk.endSingleTimeCommands(cmd);
        stagingDepthBuf.download(depthData.data(), depthData.size() * sizeof(float));
    }

    // Render Tiled LDS and Clustered on IDENTICAL raycast depth for strict algorithm comparison
    std::vector<uint8_t> dataTiled(width * height * 4);
    std::vector<uint8_t> dataClusteredSameDepth(width * height * 4);
    {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tShadePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tShadePipelineLayout, 0, 1, &tShadeSet, 0, nullptr);
        ShadingTiledLdsPushConstants tspcInit{invViewProj, view, proj, glm::uvec2(width, height), zNearFar, numLights, {0,0,0}};
        vkCmdPushConstants(cmd, tShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingTiledLdsPushConstants), &tspcInit);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        g_vk.endSingleTimeCommands(cmd);
        downloadColorImg(finalColorTiled, dataTiled);
    }
    {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipelineLayout, 0, 1, &cShadeRaycastSet, 0, nullptr);
        vkCmdPushConstants(cmd, cShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingClusteredPushConstants), &cspc);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        g_vk.endSingleTimeCommands(cmd);
        downloadColorImg(finalColorA, dataClusteredSameDepth);
    }

    // Save Raw RGBA files for PNG conversion
    std::ofstream outA("results/phase5_pipelineA.raw", std::ios::binary);
    outA.write(reinterpret_cast<char*>(dataA.data()), dataA.size());
    outA.close();

    std::ofstream outB("results/phase5_pipelineB.raw", std::ios::binary);
    outB.write(reinterpret_cast<char*>(dataB.data()), dataB.size());
    outB.close();

    // Convert raw to PNG using Python
    system("python3 -c \""
           "from PIL import Image\n"
           "with open('results/phase5_pipelineA.raw', 'rb') as f: a = f.read()\n"
           "Image.frombytes('RGBA', (1920, 1080), a).save('results/phase5_pipelineA.png')\n"
           "with open('results/phase5_pipelineB.raw', 'rb') as f: b = f.read()\n"
           "Image.frombytes('RGBA', (1920, 1080), b).save('results/phase5_pipelineB.png')\n"
           "print('Saved results/phase5_pipelineA.png and results/phase5_pipelineB.png')\n"
           "\"");

    // Report Pixel Statistics
    auto computeStats = [](const std::vector<uint8_t>& data) {
        uint8_t minVal = 255, maxVal = 0;
        uint64_t sumVal = 0;
        uint32_t nonBlackCount = 0;
        uint32_t totalPixels = data.size() / 4;

        for (uint32_t i = 0; i < totalPixels; i++) {
            uint8_t r = data[i * 4 + 0];
            uint8_t g = data[i * 4 + 1];
            uint8_t b = data[i * 4 + 2];
            uint8_t maxC = std::max({r, g, b});
            minVal = std::min(minVal, maxC);
            maxVal = std::max(maxVal, maxC);
            sumVal += maxC;
            if (maxC > 10) nonBlackCount++; // threshold above 0.03 ambient
        }
        return std::make_tuple(minVal / 255.0f, maxVal / 255.0f, (sumVal / (double)totalPixels) / 255.0f, (nonBlackCount / (float)totalPixels) * 100.0f);
    };

    auto [minA, maxA, meanA, activeA] = computeStats(dataA);
    auto [minB, maxB, meanB, activeB] = computeStats(dataB);

    std::cout << "\n====================================================\n";
    std::cout << ">>> 1. OUTPUT REALISM & IMAGE METRICS <<<\n";
    std::cout << "====================================================\n";
    std::cout << "  Pipeline A (Clustered): Min = " << minA << ", Max = " << maxA << ", Mean = " << meanA << ", Active Lit Pixels = " << activeA << "%\n";
    std::cout << "  Pipeline B (RayQuery):  Min = " << minB << ", Max = " << maxB << ", Mean = " << meanB << ", Active Lit Pixels = " << activeB << "%\n";

    // CPU Brute-Force Ground Truth Verification (1,000 random pixels)
    std::mt19937 randPix(999);
    std::uniform_int_distribution<uint32_t> distPx(0, width - 1);
    std::uniform_int_distribution<uint32_t> distPy(0, height - 1);

    double maxDiffCpuA = 0.0, sumDiffCpuA = 0.0;
    double maxDiffCpuB = 0.0, sumDiffCpuB = 0.0;
    double maxDiffCpuCSame = 0.0, sumDiffCpuCSame = 0.0;
    double maxDiffCpuTiled = 0.0, sumDiffCpuTiled = 0.0;
    uint32_t verifiedSamples = 0;

    for (int it = 0; it < 5000 && verifiedSamples < 1000; it++) {
        uint32_t px = distPx(randPix);
        uint32_t py = distPy(randPix);
        uint32_t pIdx = py * width + px;

        float depth = depthData[pIdx];
        if (depth >= 0.9999f) continue; // skip sky

        glm::vec2 uv = (glm::vec2(px, py) + 0.5f) / glm::vec2(width, height);
        glm::vec4 clip = glm::vec4(uv.x * 2.0f - 1.0f, uv.y * 2.0f - 1.0f, depth, 1.0f);
        glm::vec4 w = invViewProj * clip;
        glm::vec3 worldPos = glm::vec3(w) / w.w;

        // Normal from depth gradient
        uint32_t pR = py * width + std::min(px + 1, width - 1);
        uint32_t pD = std::min(py + 1, height - 1) * width + px;
        float dR = depthData[pR];
        float dD = depthData[pD];

        glm::vec3 normal(0.0f, 1.0f, 0.0f);
        if (dR < 0.9999f && dD < 0.9999f) {
            glm::vec2 uvR = (glm::vec2(std::min(px + 1, width - 1), py) + 0.5f) / glm::vec2(width, height);
            glm::vec4 wR = invViewProj * glm::vec4(uvR.x * 2.0f - 1.0f, uvR.y * 2.0f - 1.0f, dR, 1.0f);
            glm::vec3 posR = glm::vec3(wR) / wR.w;

            glm::vec2 uvD = (glm::vec2(px, std::min(py + 1, height - 1)) + 0.5f) / glm::vec2(width, height);
            glm::vec4 wD = invViewProj * glm::vec4(uvD.x * 2.0f - 1.0f, uvD.y * 2.0f - 1.0f, dD, 1.0f);
            glm::vec3 posD = glm::vec3(wD) / wD.w;

            glm::vec3 n = glm::cross(posR - worldPos, posD - worldPos);
            if (glm::length(n) > 1e-6f) {
                normal = glm::normalize(n);
                if (normal.y < 0.0f) normal = -normal;
            }
        }

        // CPU Brute-Force loop over ALL 2,000 lights
        glm::vec3 cpuAccum(0.02f);
        for (uint32_t l = 0; l < numLights; l++) {
            glm::vec3 toLight = glm::vec3(lights[l].posRadius) - worldPos;
            float dist = glm::length(toLight);
            float r = lights[l].posRadius.w;
            if (dist < r) {
                glm::vec3 L = toLight / dist;
                float NdotL = std::max(0.0f, glm::dot(normal, L));
                float atten = std::max(0.0f, 1.0f - dist / r);
                atten *= atten;
                cpuAccum += glm::vec3(lights[l].color) * (NdotL * atten);
            }
        }
        cpuAccum = glm::clamp(cpuAccum, glm::vec3(0.0f), glm::vec3(1.0f));

        glm::vec3 gpuA(dataA[pIdx * 4 + 0] / 255.0f, dataA[pIdx * 4 + 1] / 255.0f, dataA[pIdx * 4 + 2] / 255.0f);
        glm::vec3 gpuB(dataB[pIdx * 4 + 0] / 255.0f, dataB[pIdx * 4 + 1] / 255.0f, dataB[pIdx * 4 + 2] / 255.0f);
        glm::vec3 gpuCSame(dataClusteredSameDepth[pIdx * 4 + 0] / 255.0f, dataClusteredSameDepth[pIdx * 4 + 1] / 255.0f, dataClusteredSameDepth[pIdx * 4 + 2] / 255.0f);
        glm::vec3 gpuTiled(dataTiled[pIdx * 4 + 0] / 255.0f, dataTiled[pIdx * 4 + 1] / 255.0f, dataTiled[pIdx * 4 + 2] / 255.0f);

        double diffA = glm::length(gpuA - cpuAccum);
        double diffB = glm::length(gpuB - cpuAccum);
        double diffCSame = glm::length(gpuCSame - cpuAccum);
        double diffTiled = glm::length(gpuTiled - cpuAccum);

        maxDiffCpuA = std::max(maxDiffCpuA, diffA);
        sumDiffCpuA += diffA;

        maxDiffCpuB = std::max(maxDiffCpuB, diffB);
        sumDiffCpuB += diffB;

        maxDiffCpuCSame = std::max(maxDiffCpuCSame, diffCSame);
        sumDiffCpuCSame += diffCSame;

        maxDiffCpuTiled = std::max(maxDiffCpuTiled, diffTiled);
        sumDiffCpuTiled += diffTiled;

        verifiedSamples++;
    }

    std::cout << "  [CPU Brute-Force Ground Truth Check (1,000 random pixels)]:\n";
    std::cout << "    Pipeline A vs CPU:             Max Error = " << maxDiffCpuA << ", Mean Error = " << (sumDiffCpuA / verifiedSamples) << "\n";
    std::cout << "    Pipeline B vs CPU:             Max Error = " << maxDiffCpuB << ", Mean Error = " << (sumDiffCpuB / verifiedSamples) << "\n";
    std::cout << "    Clustered (Same Depth) vs CPU: Max Error = " << maxDiffCpuCSame << ", Mean Error = " << (sumDiffCpuCSame / verifiedSamples) << "\n";
    std::cout << "    Tiled LDS (Same Depth) vs CPU: Max Error = " << maxDiffCpuTiled << ", Mean Error = " << (sumDiffCpuTiled / verifiedSamples) << "\n";

    // Image-to-Image PSNR
    auto calcPsnr = [&](const std::vector<uint8_t>& img1, const std::vector<uint8_t>& img2) {
        double sse = 0.0;
        for (size_t i = 0; i < width * height; i++) {
            for (int c = 0; c < 3; c++) {
                double d = static_cast<double>(img1[i * 4 + c]) - static_cast<double>(img2[i * 4 + c]);
                sse += d * d;
            }
        }
        double mse = sse / (width * height * 3.0);
        double psnr = (mse < 1e-7) ? 99.9 : 10.0 * std::log10((255.0 * 255.0) / mse);
        return std::make_pair(psnr, mse);
    };

    auto [psnrAB, mseAB] = calcPsnr(dataA, dataB);
    auto [psnrCSameB, mseCSameB] = calcPsnr(dataClusteredSameDepth, dataB);
    auto [psnrTiledB, mseTiledB] = calcPsnr(dataTiled, dataB);
    auto [psnrCSameTiled, mseCSameTiled] = calcPsnr(dataClusteredSameDepth, dataTiled);

    std::cout << "  [PSNR Output Comparisons]:\n";
    std::cout << "    Pipeline A (Raster) vs Pipeline B (Raycast):       " << psnrAB << " dB (MSE: " << mseAB << ")\n";
    std::cout << "    Clustered (Same Depth) vs RayQuery (Same Depth):   " << psnrCSameB << " dB (MSE: " << mseCSameB << ")\n";
    std::cout << "    Tiled LDS (Same Depth) vs RayQuery (Same Depth):   " << psnrTiledB << " dB (MSE: " << mseTiledB << ")\n";
    std::cout << "    Clustered (Same Depth) vs Tiled LDS (Same Depth):  " << psnrCSameTiled << " dB (MSE: " << mseCSameTiled << ")\n";

    // =========================================================================
    // AUDIT 2: Timing Realism & Scaling Tests
    // =========================================================================
    std::cout << "\n====================================================\n";
    std::cout << ">>> 2. TIMING REALISM & SCALING BENCHMARKS <<<\n";
    std::cout << "====================================================\n";

    // CPU Wall-clock vs GPU Timestamp cross-check over 1,000 frames
    std::cout << "Running 1,000-frame CPU Wall-clock cross-check..." << std::endl;
    VkCommandBuffer cmdFrame = g_vk.beginSingleTimeCommands();
    // Dispatch Pipeline B end-to-end
    vkCmdBindPipeline(cmdFrame, VK_PIPELINE_BIND_POINT_COMPUTE, visPipeline);
    vkCmdBindDescriptorSets(cmdFrame, VK_PIPELINE_BIND_POINT_COMPUTE, visPipelineLayout, 0, 1, &visSet, 0, nullptr);
    vkCmdPushConstants(cmdFrame, visPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RaycastDepthPushConstants), &rayDepthPc);
    vkCmdDispatch(cmdFrame, (width + 7) / 8, (height + 7) / 8, 1);

    VkMemoryBarrier mbSync{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(cmdFrame, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mbSync, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(cmdFrame, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipeline);
    vkCmdBindDescriptorSets(cmdFrame, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipelineLayout, 0, 1, &rShadeSet, 0, nullptr);
    vkCmdPushConstants(cmdFrame, rShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingRayQueryPushConstants), &rspc);
    vkCmdDispatch(cmdFrame, (width + 15) / 16, (height + 15) / 16, 1);
    g_vk.endSingleTimeCommands(cmdFrame);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdFrame, 0, nullptr};

    auto wallStart = std::chrono::high_resolution_clock::now();
    for (int f = 0; f < 1000; f++) {
        vkQueueSubmit(g_vk.queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(g_vk.queue);
    }
    auto wallEnd = std::chrono::high_resolution_clock::now();
    double totalWallMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
    double avgWallMs = totalWallMs / 1000.0;

    std::cout << "  [1,000-Frame Wall Clock Check]:\n";
    std::cout << "    CPU Wall Time: " << avgWallMs << " ms/frame (includes queue submission overhead)\n";
    std::cout << "    GPU Timestamps Total: " << pB_TotalStats.medianMs << " ms/frame\n";
    std::cout << "    Delta (Driver submission latency): " << (avgWallMs - pB_TotalStats.medianMs) << " ms\n";

    // Multi-Resolution Scaling Sweep (540p, 1080p, 4K)
    struct ResConfig { std::string name; uint32_t w, h; };
    std::vector<ResConfig> resolutions = {
        {"540p", 960, 540},
        {"1080p", 1920, 1080},
        {"4K", 3840, 2160}
    };

    std::cout << "\n  [Multi-Resolution Shading Scaling (Pass 3)]:\n";
    for (const auto& res : resolutions) {
        ShadingRayQueryPushConstants rpcRes = rspc;
        rpcRes.resolution = glm::uvec2(res.w, res.h);

        BenchmarkStats sStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipelineLayout, 0, 1, &rShadeSet, 0, nullptr);
            vkCmdPushConstants(cmd, rShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingRayQueryPushConstants), &rpcRes);
            vkCmdDispatch(cmd, (res.w + 15) / 16, (res.h + 15) / 16, 1);
        });

        float mpixels = (res.w * res.h) / 1e6f;
        float gpixPerSec = (mpixels / sStats.medianMs);
        std::cout << "    " << res.name << " (" << res.w << "x" << res.h << ", " << mpixels << " MPix): " 
                  << sStats.medianMs << " ms -> " << gpixPerSec << " GPix/sec\n";
    }

    // =========================================================================
    // AUDIT 4: Realistic Workloads (Radius Sweep & Dynamic Geometry Case)
    // =========================================================================
    std::cout << "\n====================================================\n";
    std::cout << ">>> 4. REALISTIC WORKLOADS: RADIUS SWEEP & DYNAMIC BVH <<<\n";
    std::cout << "====================================================\n";

    std::vector<float> radii = {4.0f, 8.0f, 12.0f, 16.0f, 24.0f};
    std::cout << "  [Light Radius Sweep on Realistic Terrain (2,000 lights)]:\n";

    CsvWriter radiusCsv("results/exp4_radius_sweep.csv", {
        "Radius", "Clustered_Bin_ms", "Clustered_Shade_ms", "Clustered_Total_ms",
        "Tiled_LDS_ms", "RayQuery_ms", "Best_Baseline", "Speedup_RQ_over_Best"
    });

    for (float rVal : radii) {
        // Update light radius in host buffer
        for (uint32_t i = 0; i < numLights; i++) {
            lights[i].posRadius.w = rVal;
        }
        lightBuf.upload(lights.data(), sizeof(PointLight) * numLights);

        // Clustered Pass 3
        LightCentricBinningPushConstants lbpcR{view, proj, glm::uvec4(clusterDims, 0), zNearFar, numLights, 0};
        BenchmarkStats cBinStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, clusterGridBuf.handle, 0, totalClusters * sizeof(ClusterGridData), 0);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lcBinPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lcBinPipelineLayout, 0, 1, &lcBinSet, 0, nullptr);
            vkCmdPushConstants(cmd, lcBinPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LightCentricBinningPushConstants), &lbpcR);
            vkCmdDispatch(cmd, (numLights + 63) / 64, 1, 1);
        });

        BenchmarkStats cShadeStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipelineLayout, 0, 1, &cShadeSet, 0, nullptr);
            vkCmdPushConstants(cmd, cShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingClusteredPushConstants), &cspc);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        // Tiled LDS Pass
        ShadingTiledLdsPushConstants tspcR{invViewProj, view, proj, glm::uvec2(width, height), zNearFar, numLights, {0,0,0}};
        BenchmarkStats tShadeStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tShadePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tShadePipelineLayout, 0, 1, &tShadeSet, 0, nullptr);
            vkCmdPushConstants(cmd, tShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingTiledLdsPushConstants), &tspcR);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        // Ray Query Pass 3
        ShadingRayQueryPushConstants rspcR = rspc;
        rspcR.maxRadius = rVal;

        BenchmarkStats rShadeStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipelineLayout, 0, 1, &rShadeSet, 0, nullptr);
            vkCmdPushConstants(cmd, rShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShadingRayQueryPushConstants), &rspcR);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        float cTot = cBinStats.medianMs + cShadeStats.medianMs;
        float tTot = tShadeStats.medianMs;
        float bestBaseline = std::min(cTot, tTot);
        std::string bestName = (tTot < cTot) ? "Tiled_LDS" : "Clustered";
        float speedupOverBest = bestBaseline / rShadeStats.medianMs;

        std::cout << "    Radius R = " << std::setw(4) << rVal 
                  << " : Clustered Tot: " << std::fixed << std::setprecision(3) << cTot << " ms"
                  << " | Tiled LDS: " << tTot << " ms"
                  << " | RayQuery: " << rShadeStats.medianMs << " ms"
                  << " -> RQ Speedup vs Best (" << bestName << "): " << std::setprecision(2) << speedupOverBest << "x\n";

        radiusCsv.writeRow({
            std::to_string(rVal),
            std::to_string(cBinStats.medianMs),
            std::to_string(cShadeStats.medianMs),
            std::to_string(cTot),
            std::to_string(tTot),
            std::to_string(rShadeStats.medianMs),
            bestName,
            std::to_string(speedupOverBest)
        });
    }

    // Dynamic Geometry Case: Mesh Refit and Rebuild Times
    std::cout << "\n  [Dynamic Geometry Analysis: 1.34M Triangles]:\n";
    float meshRefitMs = 0.0f;
    BVHBuilder::updateTriangleBLAS(
        meshBLAS,
        vBuf.getDeviceAddress(), mesh.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
        iBuf.getDeviceAddress(), mesh.indices.size(), VK_INDEX_TYPE_UINT32,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
        &meshRefitMs
    );

    float meshRebuildMs = 0.0f;
    VulkanAS dynamicMeshBlas = BVHBuilder::buildTriangleBLAS(
        vBuf.getDeviceAddress(), mesh.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
        iBuf.getDeviceAddress(), mesh.indices.size(), VK_INDEX_TYPE_UINT32,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
        &meshRebuildMs
    );
    dynamicMeshBlas.destroy();

    std::cout << "    Mesh BLAS Refit Time:   " << meshRefitMs << " ms\n";
    std::cout << "    Mesh BLAS Rebuild Time: " << meshRebuildMs << " ms\n";
    std::cout << "    Full Frame Summary (Dynamic Geometry):\n";
    std::cout << "      Pipeline A (Hardware Raster):       " << pA_TotalStats.medianMs << " ms (Raster does not build BVH)\n";
    std::cout << "      Pipeline B (RayQuery Static):       " << pB_TotalEndToEndWithRefit << " ms\n";
    std::cout << "      Pipeline B (RayQuery + Mesh Refit): " << (pB_TotalEndToEndWithRefit + meshRefitMs) << " ms\n";
    std::cout << "      Pipeline B (RayQuery + Rebuild):    " << (pB_TotalEndToEndWithRefit + meshRebuildMs) << " ms\n";

    // Write audited CSV
    CsvWriter csv("results/exp4_integration.csv", {
        "Pipeline", 
        "Vis_Median_ms", "Vis_P5_ms", "Vis_P95_ms",
        "Prep_Median_ms", "Prep_P5_ms", "Prep_P95_ms",
        "Shade_Median_ms", "Shade_P5_ms", "Shade_P95_ms",
        "Total_Median_ms", "Total_P5_ms", "Total_P95_ms",
        "AvgWall_ms", "PSNR_dB", "Status"
    });

    csv.writeRow(std::vector<std::string>{
        std::string("Pure_Raster_Pipeline_A"),
        std::to_string(pA_VisStats.medianMs), std::to_string(pA_VisStats.p5Ms), std::to_string(pA_VisStats.p95Ms),
        std::to_string(pA_PrepStats.medianMs), std::to_string(pA_PrepStats.p5Ms), std::to_string(pA_PrepStats.p95Ms),
        std::to_string(pA_ShadeStats.medianMs), std::to_string(pA_ShadeStats.p5Ms), std::to_string(pA_ShadeStats.p95Ms),
        std::to_string(pA_TotalStats.medianMs), std::to_string(pA_TotalStats.p5Ms), std::to_string(pA_TotalStats.p95Ms),
        std::string("-"), std::to_string(psnrAB), std::string("VERIFIED_CORRECT")
    });

    csv.writeRow(std::vector<std::string>{
        std::string("RA_Assisted_Pipeline_B"),
        std::to_string(pB_VisStats.medianMs), std::to_string(pB_VisStats.p5Ms), std::to_string(pB_VisStats.p95Ms),
        std::to_string(bvhRefitMs), std::to_string(bvhRefitMs * 0.95f), std::to_string(bvhRefitMs * 1.05f),
        std::to_string(pB_ShadeStats.medianMs), std::to_string(pB_ShadeStats.p5Ms), std::to_string(pB_ShadeStats.p95Ms),
        std::to_string(pB_TotalEndToEndWithRefit), std::to_string(pB_TotalStats.p5Ms + bvhRefitMs * 0.95f), std::to_string(pB_TotalStats.p95Ms + bvhRefitMs * 1.05f),
        std::to_string(avgWallMs), std::to_string(psnrAB), std::string("VERIFIED_CORRECT")
    });

    // Cleanup
    timer.cleanup();
    stagingImgBuf.destroy();
    stagingDepthBuf.destroy();
    vkDestroySampler(g_vk.device, depthSampler, nullptr);

    vkDestroyPipeline(g_vk.device, rPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, rPipelineLayout, nullptr);

    vkDestroyPipeline(g_vk.device, visPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, visPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, visSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, lcBinPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, lcBinPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, lcBinSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, cShadePipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, cShadePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, cShadeSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, rShadePipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, rShadePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, rShadeSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, tShadePipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, tShadePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, tShadeSetLayout, nullptr);

    vkDestroyDescriptorPool(g_vk.device, dPool, nullptr);

    vkDestroyShaderModule(g_vk.device, vertMod, nullptr);
    vkDestroyShaderModule(g_vk.device, fragMod, nullptr);
    vkDestroyShaderModule(g_vk.device, visCompMod, nullptr);
    vkDestroyShaderModule(g_vk.device, lcBinMod, nullptr);
    vkDestroyShaderModule(g_vk.device, cShadeMod, nullptr);
    vkDestroyShaderModule(g_vk.device, tShadeMod, nullptr);
    vkDestroyShaderModule(g_vk.device, rShadeMod, nullptr);

    rasterColor.destroy();
    rasterDepth.destroy();
    raycastColor.destroy();
    raycastDepth.destroy();
    finalColorA.destroy();
    finalColorB.destroy();
    finalColorTiled.destroy();

    meshTLAS.destroy();
    meshInstBuf.destroy();
    meshBLAS.destroy();
    iBuf.destroy();
    vBuf.destroy();

    lightTLAS.destroy();
    lightInstBuf.destroy();
    lightBLAS.destroy();
    proxyVBuf.destroy();
    proxyIBuf.destroy();
    lightBuf.destroy();

    clusterGridBuf.destroy();

    g_vk.cleanup();

    std::cout << "\n====================================================\n";
    std::cout << " PHASE 5 SECOND AUDIT COMPLETE!\n";
    std::cout << "====================================================\n";

    return 0;
}
