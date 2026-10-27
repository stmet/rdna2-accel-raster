#include "vk_context.hpp"
#include "vk_buffer.hpp"
#include "vk_image.hpp"
#include "vk_bvh.hpp"
#include "vk_shader.hpp"
#include "vk_timer.hpp"
#include "csv_writer.hpp"
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
#include <cmath>

struct InstanceData {
    glm::vec4 aabbMin;
    glm::vec4 aabbMax;
};

struct HzbPushConstants {
    glm::mat4 viewProj;
    glm::vec4 frustumPlanes[6];
    glm::vec2 hzbSize;
    uint32_t instanceCount;
    uint32_t maxMipLevel;
};

struct RayCullPushConstants {
    glm::vec4 camPos;
    glm::vec4 frustumPlanes[6];
    uint32_t instanceCount;
    uint32_t sampleCount; // 1, 3, 5, or 9
};

void extractFrustumPlanes(const glm::mat4& vp, glm::vec4 planes[6]) {
    planes[0] = glm::vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]);
    planes[1] = glm::vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]);
    planes[2] = glm::vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]);
    planes[3] = glm::vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]);
    planes[4] = glm::vec4(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    planes[5] = glm::vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]);

    for (int i = 0; i < 6; i++) {
        float len = glm::length(glm::vec3(planes[i]));
        planes[i] /= len;
    }
}

struct OccluderMesh {
    struct Vertex {
        float pos[3];
        float normal[3];
    };
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    void addBox(glm::vec3 c, glm::vec3 halfSize) {
        uint32_t base = static_cast<uint32_t>(vertices.size());
        glm::vec3 p[8] = {
            {c.x - halfSize.x, c.y - halfSize.y, c.z - halfSize.z},
            {c.x + halfSize.x, c.y - halfSize.y, c.z - halfSize.z},
            {c.x + halfSize.x, c.y + halfSize.y, c.z - halfSize.z},
            {c.x - halfSize.x, c.y + halfSize.y, c.z - halfSize.z},
            {c.x - halfSize.x, c.y - halfSize.y, c.z + halfSize.z},
            {c.x + halfSize.x, c.y - halfSize.y, c.z + halfSize.z},
            {c.x + halfSize.x, c.y + halfSize.y, c.z + halfSize.z},
            {c.x - halfSize.x, c.y + halfSize.y, c.z + halfSize.z}
        };
        for (int i = 0; i < 8; i++) {
            vertices.push_back({{p[i].x, p[i].y, p[i].z}, {0, 1, 0}});
        }
        uint32_t fi[] = {
            0, 2, 1, 0, 3, 2,
            5, 6, 4, 4, 6, 7,
            4, 7, 0, 0, 7, 3,
            1, 2, 5, 5, 2, 6,
            3, 6, 2, 3, 7, 6,
            0, 1, 4, 4, 1, 5
        };
        for (uint32_t idx : fi) indices.push_back(base + idx);
    }
};

// Scene 1: Macro City Walls
OccluderMesh generateCityWalls() {
    OccluderMesh mesh;
    std::vector<glm::vec3> wallCenters = {
        {-40, 15, 30}, {0, 20, 35}, {40, 15, 30},
        {-50, 25, 70}, {-10, 20, 65}, {30, 25, 75},
        {-30, 30, 110}, {15, 35, 115}, {55, 30, 110},
        {-60, 20, 150}, {-15, 30, 155}, {25, 25, 145}, {65, 20, 150}
    };
    for (const auto& c : wallCenters) {
        mesh.addBox(c, glm::vec3(18.0f, c.y, 5.0f));
    }
    return mesh;
}

// Scene 2: Forest / Columns (Many Small Occluders)
OccluderMesh generateForestPillars() {
    OccluderMesh mesh;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> distX(-60.0f, 60.0f);
    std::uniform_real_distribution<float> distZ(20.0f, 160.0f);

    for (int i = 0; i < 120; i++) {
        float x = distX(rng);
        float z = distZ(rng);
        mesh.addBox(glm::vec3(x, 15.0f, z), glm::vec3(1.2f, 15.0f, 1.2f));
    }
    return mesh;
}

// Scene 3: Swiss Cheese / Grate (Thin Slits & Gaps - Adversarial)
OccluderMesh generateSwissCheeseGrate() {
    OccluderMesh mesh;
    // 3 large walls spanning the view, but perforated with narrow 0.8-unit slits every 2.5 units
    float zPlanes[] = {35.0f, 75.0f, 120.0f};
    for (float z : zPlanes) {
        for (float x = -60.0f; x <= 60.0f; x += 3.0f) {
            // Slat with gap
            mesh.addBox(glm::vec3(x, 15.0f, z), glm::vec3(1.0f, 15.0f, 0.5f));
        }
    }
    return mesh;
}

struct SceneConfig {
    std::string name;
    OccluderMesh mesh;
};

int main() {
    std::cout << "====================================================\n";
    std::cout << " PHASE 3 AUDIT: OCCLUSION CULLING ROBUSTNESS PASS\n";
    std::cout << "        3 SCENES x (HZB vs RQ 1, 3, 5, 9 SAMPLES)\n";
    std::cout << "====================================================\n";

    g_vk.init(true);
    GpuClockInfo::printStatus();

    std::filesystem::create_directories("results");

    CsvWriter csv("results/exp2_occlusion.csv", {
        "Scene", "Instances", "Method", 
        "CullTime_Median_ms", "CullTime_P5_ms", "CullTime_P95_ms",
        "EndToEndTime_Median_ms", "VisibleCount", "FalsePositiveRate_pct", "FalseNegativeRate_pct"
    });

    std::vector<SceneConfig> scenes = {
        {"City_Walls", generateCityWalls()},
        {"Forest_Pillars", generateForestPillars()},
        {"Swiss_Cheese_Grate", generateSwissCheeseGrate()}
    };

    std::vector<uint32_t> instanceCounts = {10000, 50000, 100000, 200000};
    uint32_t fbWidth = 1024, fbHeight = 1024;

    glm::vec3 camPos(0.0f, 15.0f, -20.0f);
    glm::vec3 lookAt(0.0f, 15.0f, 100.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    glm::mat4 view = glm::lookAt(camPos, lookAt, up);
    glm::mat4 proj = glm::perspective(glm::radians(75.0f), static_cast<float>(fbWidth) / fbHeight, 0.1f, 300.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 viewProj = proj * view;

    glm::vec4 frustumPlanes[6];
    extractFrustumPlanes(viewProj, frustumPlanes);

    // Setup Depth Texture & Sampler
    VulkanImage depthImage;
    depthImage.create(fbWidth, fbHeight, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

    VkSampler hzbSampler;
    VkSamplerCreateInfo sampInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampInfo.magFilter = VK_FILTER_LINEAR;
    sampInfo.minFilter = VK_FILTER_LINEAR;
    sampInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.minLod = 0.0f;
    sampInfo.maxLod = 10.0f;
    VK_CHECK(vkCreateSampler(g_vk.device, &sampInfo, nullptr, &hzbSampler));

    // Compile Shaders
    std::vector<uint32_t> vertSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_raster.vert");
    std::vector<uint32_t> fragSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_raster.frag");
    std::vector<uint32_t> hzbSpv = ShaderUtils::compileGlslFileToSpirv("exp2_occlusion/shaders/hzb_cull.comp");
    std::vector<uint32_t> raySpv = ShaderUtils::compileGlslFileToSpirv("exp2_occlusion/shaders/ray_cull.comp");

    VkShaderModule vertModule = ShaderUtils::createShaderModule(vertSpv);
    VkShaderModule fragModule = ShaderUtils::createShaderModule(fragSpv);
    VkShaderModule hzbModule = ShaderUtils::createShaderModule(hzbSpv);
    VkShaderModule rayModule = ShaderUtils::createShaderModule(raySpv);

    // Raster Depth Pipeline Layout
    VkPushConstantRange pcRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)};
    VkPipelineLayoutCreateInfo rLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &pcRange};
    VkPipelineLayout rPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &rLayoutInfo, nullptr, &rPipelineLayout));

    VkPipelineShaderStageCreateInfo rStages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr}
    };
    VkVertexInputBindingDescription bDesc{0, sizeof(OccluderMesh::Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription aDescs[2]{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(OccluderMesh::Vertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(OccluderMesh::Vertex, normal)}
    };
    VkPipelineVertexInputStateCreateInfo viInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &bDesc, 2, aDescs};
    VkPipelineInputAssemblyStateCreateInfo iaInfo{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkViewport vp{0, 0, (float)fbWidth, (float)fbHeight, 0, 1};
    VkRect2D sc{{0, 0}, {fbWidth, fbHeight}};
    VkPipelineViewportStateCreateInfo vpsInfo{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc};
    VkPipelineRasterizationStateCreateInfo rsInfo{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo msInfo{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 0, nullptr, VK_FALSE, VK_FALSE};
    VkPipelineDepthStencilStateCreateInfo dsInfo{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS, VK_FALSE, VK_FALSE, {}, {}, 0.0f, 1.0f};
    VkPipelineRenderingCreateInfo rendInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 0, nullptr, VK_FORMAT_D32_SFLOAT, VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo gpInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendInfo, 0, 2, rStages, &viInfo, &iaInfo, nullptr, &vpsInfo, &rsInfo, &msInfo, &dsInfo, nullptr, nullptr, rPipelineLayout, VK_NULL_HANDLE, 0};
    VkPipeline rPipeline;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &gpInfo, nullptr, &rPipeline));

    // HZB Culling Pipeline Layout
    VkDescriptorSetLayoutBinding hzbBindings[3]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo hzbLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, hzbBindings};
    VkDescriptorSetLayout hzbSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &hzbLayoutInfo, nullptr, &hzbSetLayout));
    VkPushConstantRange hzbPcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HzbPushConstants)};
    VkPipelineLayoutCreateInfo hzbPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &hzbSetLayout, 1, &hzbPcRange};
    VkPipelineLayout hzbPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &hzbPipeLayoutInfo, nullptr, &hzbPipelineLayout));
    VkPipeline hzbPipeline = ShaderUtils::createComputePipeline(hzbModule, hzbPipelineLayout);

    // Ray Cull Pipeline Layout
    VkDescriptorSetLayoutBinding rayBindings[3]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo rayLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, rayBindings};
    VkDescriptorSetLayout raySetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &rayLayoutInfo, nullptr, &raySetLayout));
    VkPushConstantRange rayPcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RayCullPushConstants)};
    VkPipelineLayoutCreateInfo rayPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &raySetLayout, 1, &rayPcRange};
    VkPipelineLayout rayPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &rayPipeLayoutInfo, nullptr, &rayPipelineLayout));
    VkPipeline rayPipeline = ShaderUtils::createComputePipeline(rayModule, rayPipelineLayout);

    // Descriptor Pool
    VkDescriptorPoolSize pSizes[3]{
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2}
    };
    VkDescriptorPoolCreateInfo piInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 5, 3, pSizes};
    VkDescriptorPool dPool;
    VK_CHECK(vkCreateDescriptorPool(g_vk.device, &piInfo, nullptr, &dPool));

    VkDescriptorSet hzbSet, raySet;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dPool, 1, &hzbSetLayout};
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &hzbSet));
    ai.pSetLayouts = &raySetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &raySet));

    GpuTimer timer;
    timer.init(64);

    for (const auto& scene : scenes) {
        std::cout << "\n====================================================\n";
        std::cout << ">>> SCENE: " << scene.name << " (" << scene.mesh.indices.size() / 3 << " occluder tris) <<<\n";
        std::cout << "====================================================\n";

        // Build Occluder Geometry
        VulkanBuffer occVBuf, occIBuf;
        occVBuf.create(sizeof(OccluderMesh::Vertex) * scene.mesh.vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        occVBuf.upload(scene.mesh.vertices.data(), sizeof(OccluderMesh::Vertex) * scene.mesh.vertices.size());

        occIBuf.create(sizeof(uint32_t) * scene.mesh.indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        occIBuf.upload(scene.mesh.indices.data(), sizeof(uint32_t) * scene.mesh.indices.size());

        float occBuildTime = 0.0f;
        VulkanAS occBLAS = BVHBuilder::buildTriangleBLAS(
            occVBuf.getDeviceAddress(), scene.mesh.vertices.size(), sizeof(OccluderMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            occIBuf.getDeviceAddress(), scene.mesh.indices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
            &occBuildTime
        );

        VkTransformMatrixKHR transform = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f
        };
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform = transform;
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = occBLAS.deviceAddress;

        VulkanBuffer instBuf;
        instBuf.create(sizeof(inst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        instBuf.upload(&inst, sizeof(inst));
        VulkanAS occTLAS = BVHBuilder::buildTLAS(instBuf.getDeviceAddress(), 1);

        // Render depth for HZB
        float depthRenderTime = 0.0f;
        {
            depthImage.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
            VkQueryPool qp;
            VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_TIMESTAMP, 2};
            vkCreateQueryPool(g_vk.device, &qpi, nullptr, &qp);

            VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
            vkCmdResetQueryPool(cmd, qp, 0, 2);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);

            VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, depthImage.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
            dAtt.clearValue.depthStencil.depth = 1.0f;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0, 0}, {fbWidth, fbHeight}}, 1, 0, 0, nullptr, &dAtt, nullptr};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rPipeline);
            vkCmdPushConstants(cmd, rPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &viewProj);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &occVBuf.handle, &off);
            vkCmdBindIndexBuffer(cmd, occIBuf.handle, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, static_cast<uint32_t>(scene.mesh.indices.size()), 1, 0, 0, 0);
            vkCmdEndRendering(cmd);

            depthImage.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            g_vk.endSingleTimeCommands(cmd);

            uint64_t ts[2];
            vkGetQueryPoolResults(g_vk.device, qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            depthRenderTime = static_cast<float>(ts[1] - ts[0]) * g_vk.timestampPeriod / 1e6f;
            vkDestroyQueryPool(g_vk.device, qp, nullptr);
        }

        for (uint32_t count : instanceCounts) {
            std::cout << "\n--- Testing " << count << " Instances ---" << std::endl;

            std::vector<InstanceData> instances(count);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> distXZ(-90.0f, 90.0f);
            std::uniform_real_distribution<float> distY(0.0f, 35.0f);
            std::uniform_real_distribution<float> distZ(15.0f, 180.0f);

            for (uint32_t i = 0; i < count; i++) {
                float x = distXZ(rng);
                float y = distY(rng);
                float z = distZ(rng);
                float sz = 1.0f;
                instances[i].aabbMin = glm::vec4(x - sz, y, z - sz, 1.0f);
                instances[i].aabbMax = glm::vec4(x + sz, y + sz * 2, z + sz, 1.0f);
            }

            VulkanBuffer instBufGpu, hzbVisBuf, rqVisBuf[4];
            instBufGpu.create(sizeof(InstanceData) * count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            instBufGpu.upload(instances.data(), sizeof(InstanceData) * count);

            hzbVisBuf.create(sizeof(uint32_t) * count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            for (int k = 0; k < 4; k++) {
                rqVisBuf[k].create(sizeof(uint32_t) * count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }

            // Descriptors
            VkDescriptorBufferInfo instDescInfo{instBufGpu.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo hzbVisDescInfo{hzbVisBuf.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo hzbImgDescInfo{hzbSampler, depthImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet hzbWrites[3]{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, hzbSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &instDescInfo, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, hzbSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hzbVisDescInfo, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, hzbSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hzbImgDescInfo, nullptr, nullptr}
            };
            vkUpdateDescriptorSets(g_vk.device, 3, hzbWrites, 0, nullptr);

            VkWriteDescriptorSetAccelerationStructureKHR asWriteInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &occTLAS.handle};
            VkDescriptorBufferInfo rqVisDescInfo{rqVisBuf[0].handle, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet rayWrites[3]{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, raySet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &instDescInfo, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, raySet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &rqVisDescInfo, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asWriteInfo, raySet, 2, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr}
            };
            vkUpdateDescriptorSets(g_vk.device, 3, rayWrites, 0, nullptr);

            // 1. Benchmark HZB
            HzbPushConstants hpc{viewProj, {}, glm::vec2(fbWidth, fbHeight), count, 10};
            for (int i = 0; i < 6; i++) hpc.frustumPlanes[i] = frustumPlanes[i];

            BenchmarkStats hzbStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hzbPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hzbPipelineLayout, 0, 1, &hzbSet, 0, nullptr);
                vkCmdPushConstants(cmd, hzbPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HzbPushConstants), &hpc);
                vkCmdDispatch(cmd, (count + 63) / 64, 1, 1);
            });

            // 2. Benchmark Ray Query with 1, 3, 5, 9 samples
            uint32_t sampleCounts[] = {1, 3, 5, 9};
            BenchmarkStats rqStats[4];

            for (int sIdx = 0; sIdx < 4; sIdx++) {
                uint32_t samples = sampleCounts[sIdx];
                VkDescriptorBufferInfo bInfo{rqVisBuf[sIdx].handle, 0, VK_WHOLE_SIZE};
                rayWrites[1].pBufferInfo = &bInfo;
                vkUpdateDescriptorSets(g_vk.device, 1, &rayWrites[1], 0, nullptr);

                RayCullPushConstants rpc{glm::vec4(camPos, 1.0f), {}, count, samples};
                for (int i = 0; i < 6; i++) rpc.frustumPlanes[i] = frustumPlanes[i];

                rqStats[sIdx] = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rayPipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rayPipelineLayout, 0, 1, &raySet, 0, nullptr);
                    vkCmdPushConstants(cmd, rayPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RayCullPushConstants), &rpc);
                    vkCmdDispatch(cmd, (count + 63) / 64, 1, 1);
                });
            }

            // Download results & Compute Error Rates
            std::vector<uint32_t> hzbVis(count), rqVis[4];
            hzbVisBuf.download(hzbVis.data(), sizeof(uint32_t) * count);
            for (int k = 0; k < 4; k++) {
                rqVis[k].resize(count);
                rqVisBuf[k].download(rqVis[k].data(), sizeof(uint32_t) * count);
            }

            // Ground truth reference: 9-sample Ray Query
            uint32_t refVisibleCount = 0;
            for (uint32_t i = 0; i < count; i++) {
                if (rqVis[3][i]) refVisibleCount++;
            }
            if (refVisibleCount == 0) refVisibleCount = 1;

            // HZB Stats
            uint32_t hzbVisibleCount = 0;
            uint32_t hzbFp = 0, hzbFn = 0;
            for (uint32_t i = 0; i < count; i++) {
                if (hzbVis[i]) hzbVisibleCount++;
                if (hzbVis[i] == 1 && rqVis[3][i] == 0) hzbFp++;
                if (hzbVis[i] == 0 && rqVis[3][i] == 1) hzbFn++;
            }
            float hzbFpRate = (static_cast<float>(hzbFp) / (count - refVisibleCount)) * 100.0f;
            float hzbFnRate = (static_cast<float>(hzbFn) / refVisibleCount) * 100.0f;

            std::cout << "  [HZB Two-Pass Reuse] Median: " << std::fixed << std::setprecision(3) << hzbStats.medianMs 
                      << " ms (Cold E2E: " << (hzbStats.medianMs + depthRenderTime) << " ms) | FP: " 
                      << std::setprecision(1) << hzbFpRate << "%, FN: " << hzbFnRate << "%\n";

            csv.writeRow({
                scene.name,
                std::to_string(count),
                "HZB_Depth_Reuse",
                std::to_string(hzbStats.medianMs),
                std::to_string(hzbStats.p5Ms),
                std::to_string(hzbStats.p95Ms),
                std::to_string(hzbStats.medianMs),
                std::to_string(hzbVisibleCount),
                std::to_string(hzbFpRate),
                std::to_string(hzbFnRate)
            });

            csv.writeRow({
                scene.name,
                std::to_string(count),
                "HZB_Cold_Depth",
                std::to_string(hzbStats.medianMs),
                std::to_string(hzbStats.p5Ms),
                std::to_string(hzbStats.p95Ms),
                std::to_string(hzbStats.medianMs + depthRenderTime),
                std::to_string(hzbVisibleCount),
                std::to_string(hzbFpRate),
                std::to_string(hzbFnRate)
            });

            for (int sIdx = 0; sIdx < 4; sIdx++) {
                uint32_t samples = sampleCounts[sIdx];
                uint32_t visCount = 0;
                uint32_t fnCount = 0;
                for (uint32_t i = 0; i < count; i++) {
                    if (rqVis[sIdx][i]) visCount++;
                    if (rqVis[3][i] == 1 && rqVis[sIdx][i] == 0) fnCount++;
                }
                float fnRate = (static_cast<float>(fnCount) / refVisibleCount) * 100.0f;

                std::string methodName = "RayQuery_" + std::to_string(samples) + "Sample";
                std::cout << "  [" << methodName << "] Median: " << rqStats[sIdx].medianMs 
                          << " ms (p5: " << rqStats[sIdx].p5Ms << ", p95: " << rqStats[sIdx].p95Ms 
                          << ") | Visible: " << visCount << " | False Negatives (POPPING): " 
                          << std::setprecision(1) << fnRate << "%\n";

                csv.writeRow({
                    scene.name,
                    std::to_string(count),
                    methodName,
                    std::to_string(rqStats[sIdx].medianMs),
                    std::to_string(rqStats[sIdx].p5Ms),
                    std::to_string(rqStats[sIdx].p95Ms),
                    std::to_string(rqStats[sIdx].medianMs),
                    std::to_string(visCount),
                    "0.0",
                    std::to_string(fnRate)
                });
            }

            instBufGpu.destroy();
            hzbVisBuf.destroy();
            for (int k = 0; k < 4; k++) rqVisBuf[k].destroy();
        }

        occTLAS.destroy();
        instBuf.destroy();
        occBLAS.destroy();
        occVBuf.destroy();
        occIBuf.destroy();
    }

    timer.cleanup();
    vkDestroyPipeline(g_vk.device, hzbPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, hzbPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, hzbSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, rayPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, rayPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, raySetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, rPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, rPipelineLayout, nullptr);

    vkDestroySampler(g_vk.device, hzbSampler, nullptr);
    vkDestroyDescriptorPool(g_vk.device, dPool, nullptr);

    depthImage.destroy();
    vkDestroyShaderModule(g_vk.device, vertModule, nullptr);
    vkDestroyShaderModule(g_vk.device, fragModule, nullptr);
    vkDestroyShaderModule(g_vk.device, hzbModule, nullptr);
    vkDestroyShaderModule(g_vk.device, rayModule, nullptr);

    g_vk.cleanup();

    std::cout << "\n====================================================\n";
    std::cout << " PHASE 3 AUDIT COMPLETE!\n";
    std::cout << "====================================================\n";

    return 0;
}
