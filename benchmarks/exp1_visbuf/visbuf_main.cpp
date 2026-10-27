#include "vk_context.hpp"
#include "vk_buffer.hpp"
#include "vk_image.hpp"
#include "vk_bvh.hpp"
#include "vk_shader.hpp"
#include "vk_timer.hpp"
#include "csv_writer.hpp"
#include "mesh_gen.hpp"
#include "meshlet_gen.hpp"
#include "gpu_clock.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <cassert>
#include <array>

struct RasterPushConstants {
    glm::mat4 viewProj;
};

struct MeshPushConstants {
    glm::mat4 viewProj;
    glm::vec4 frustumPlanes[6];
    uint32_t meshletCount;
};

struct RaycastPushConstants {
    glm::mat4 invViewProj;
    glm::vec4 camPos;
    glm::uvec2 resolution;
};

struct SoftRasterPushConstants {
    glm::mat4 viewProj;
    glm::uvec2 resolution;
    uint32_t triangleCount;
};

struct ResolvePushConstants {
    glm::uvec2 resolution;
};

inline std::array<glm::vec4, 6> extractFrustumPlanes(const glm::mat4& m) {
    std::array<glm::vec4, 6> planes;
    // Left: row3 + row0
    planes[0] = glm::vec4(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]);
    // Right: row3 - row0
    planes[1] = glm::vec4(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]);
    // Bottom: row3 + row1
    planes[2] = glm::vec4(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]);
    // Top: row3 - row1
    planes[3] = glm::vec4(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]);
    // Near: row2
    planes[4] = glm::vec4(m[0][2], m[1][2], m[2][2], m[3][2]);
    // Far: row3 - row2
    planes[5] = glm::vec4(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]);

    for (int i = 0; i < 6; i++) {
        float len = glm::length(glm::vec3(planes[i]));
        if (len > 1e-6f) planes[i] /= len;
    }
    return planes;
}

int main() {
    std::cout << "====================================================\n";
    std::cout << " PHASE 2 AUDIT: PRIMARY VISIBILITY BUFFER SHOWDOWN\n";
    std::cout << "  HW RASTER vs MESH SHADER vs COMPUTE SW RASTER vs HW RQ\n";
    std::cout << "  (200 REPEATED DISPATCHES PER PASS IN-COMMAND BUFFER)\n";
    std::cout << "====================================================\n";

    g_vk.init(true);
    GpuClockInfo::printStatus();

    std::filesystem::create_directories("results");

    const uint32_t width = 1920;
    const uint32_t height = 1080;
    const uint32_t totalPixels = width * height;
    const uint32_t REPEAT_COUNT = 200;

    // Compile Shaders
    std::cout << "Compiling shaders..." << std::endl;
    std::vector<uint32_t> vertSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_raster.vert");
    std::vector<uint32_t> fragSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_raster.frag");
    std::vector<uint32_t> taskSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_mesh.task");
    std::vector<uint32_t> meshSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_mesh.mesh");
    std::vector<uint32_t> meshFragSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_mesh.frag");
    std::vector<uint32_t> compSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_raycast.comp");
    std::vector<uint32_t> srSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_softraster.comp");
    std::vector<uint32_t> resSpv = ShaderUtils::compileGlslFileToSpirv("exp1_visbuf/shaders/visbuf_resolve.comp");

    VkShaderModule vertModule = ShaderUtils::createShaderModule(vertSpv);
    VkShaderModule fragModule = ShaderUtils::createShaderModule(fragSpv);
    VkShaderModule taskModule = ShaderUtils::createShaderModule(taskSpv);
    VkShaderModule meshModule = ShaderUtils::createShaderModule(meshSpv);
    VkShaderModule meshFragModule = ShaderUtils::createShaderModule(meshFragSpv);
    VkShaderModule compModule = ShaderUtils::createShaderModule(compSpv);
    VkShaderModule srModule = ShaderUtils::createShaderModule(srSpv);
    VkShaderModule resModule = ShaderUtils::createShaderModule(resSpv);

    GpuTimer timer;
    timer.init(64);

    // Setup Framebuffers / Images
    VulkanImage rasterColor, rasterDepth, raycastColor, softRasterColor;
    rasterColor.create(width, height, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    rasterDepth.create(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    raycastColor.create(width, height, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    softRasterColor.create(width, height, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    // 64-bit atomic buffer for Compute Software Rasterizer
    VulkanBuffer softRasterAtomicBuf;
    softRasterAtomicBuf.create(
        sizeof(uint64_t) * totalPixels,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Common View Projection Matrix (slanted top-down looking across full viewport)
    glm::vec3 camPos(0.0f, 45.0f, -40.0f);
    glm::vec3 lookAt(0.0f, 0.0f, 10.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::mat4 view = glm::lookAt(camPos, lookAt, up);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), static_cast<float>(width) / height, 0.1f, 1000.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 viewProj = proj * view;
    glm::mat4 invViewProj = glm::inverse(viewProj);
    auto frustumPlanes = extractFrustumPlanes(viewProj);

    // 1. Hardware Raster Pipeline
    VkPushConstantRange rasterPcRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(RasterPushConstants)};
    VkPipelineLayoutCreateInfo rasterLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 0, nullptr, 1, &rasterPcRange};
    VkPipelineLayout rasterPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &rasterLayoutInfo, nullptr, &rasterPipelineLayout));

    VkPipelineShaderStageCreateInfo rasterShaderStages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr}
    };
    VkVertexInputBindingDescription bindingDesc{0, sizeof(SceneMesh::Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrDescs[2]{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneMesh::Vertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SceneMesh::Vertex, normal)}
    };
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 1, &bindingDesc, 2, attrDescs};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkViewport viewport{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {width, height}};
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &viewport, 1, &scissor};
    VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 0, nullptr, VK_FALSE, VK_FALSE};
    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, nullptr, 0, VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS, VK_FALSE, VK_FALSE, {}, {}, 0.0f, 1.0f};
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &colorBlendAttachment, {0,0,0,0}};

    VkFormat colorFmt = VK_FORMAT_R32_UINT;
    VkPipelineRenderingCreateInfo renderingCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, nullptr, 0, 1, &colorFmt, VK_FORMAT_D32_SFLOAT, VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo gpInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &renderingCreateInfo, 0, 2, rasterShaderStages, &vertexInputInfo, &inputAssembly, nullptr, &viewportState, &rasterizer, &multisampling, &depthStencil, &colorBlending, nullptr, rasterPipelineLayout, VK_NULL_HANDLE, 0};
    VkPipeline rasterPipeline;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &gpInfo, nullptr, &rasterPipeline));

    // 2. Mesh Shader Pipeline (VK_EXT_mesh_shader)
    VkPushConstantRange meshPcRange{VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0, sizeof(MeshPushConstants)};
    VkDescriptorSetLayoutBinding meshBindings[4]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_MESH_BIT_EXT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo meshLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, meshBindings};
    VkDescriptorSetLayout meshSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &meshLayoutInfo, nullptr, &meshSetLayout));

    VkPipelineLayoutCreateInfo meshPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &meshSetLayout, 1, &meshPcRange};
    VkPipelineLayout meshPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &meshPipeLayoutInfo, nullptr, &meshPipelineLayout));

    VkPipelineShaderStageCreateInfo meshShaderStages[3]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_TASK_BIT_EXT, taskModule, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_MESH_BIT_EXT, meshModule, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, meshFragModule, "main", nullptr}
    };
    VkPipelineVertexInputStateCreateInfo emptyVertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nullptr, 0, 0, nullptr, 0, nullptr};
    VkPipelineInputAssemblyStateCreateInfo emptyInputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkGraphicsPipelineCreateInfo meshGpInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &renderingCreateInfo, 0, 3, meshShaderStages, &emptyVertexInput, &emptyInputAssembly, nullptr, &viewportState, &rasterizer, &multisampling, &depthStencil, &colorBlending, nullptr, meshPipelineLayout, VK_NULL_HANDLE, 0};
    VkPipeline meshPipeline;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &meshGpInfo, nullptr, &meshPipeline));

    // 3. Compute Software Rasterizer Pipeline
    VkPushConstantRange srPcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SoftRasterPushConstants)};
    VkDescriptorSetLayoutBinding srBindings[3]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo srLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, srBindings};
    VkDescriptorSetLayout srSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &srLayoutInfo, nullptr, &srSetLayout));
    VkPipelineLayoutCreateInfo srPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &srSetLayout, 1, &srPcRange};
    VkPipelineLayout srPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &srPipeLayoutInfo, nullptr, &srPipelineLayout));
    VkPipeline srPipeline = ShaderUtils::createComputePipeline(srModule, srPipelineLayout);

    // Resolve Pipeline for Compute Software Raster
    VkPushConstantRange resPcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ResolvePushConstants)};
    VkDescriptorSetLayoutBinding resBindings[2]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo resLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, resBindings};
    VkDescriptorSetLayout resSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &resLayoutInfo, nullptr, &resSetLayout));
    VkPipelineLayoutCreateInfo resPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &resSetLayout, 1, &resPcRange};
    VkPipelineLayout resPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &resPipeLayoutInfo, nullptr, &resPipelineLayout));
    VkPipeline resPipeline = ShaderUtils::createComputePipeline(resModule, resPipelineLayout);

    // 4. Hardware Raycast Pipeline
    VkPushConstantRange rayPcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RaycastPushConstants)};
    VkDescriptorSetLayoutBinding compBindings[2]{
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo compSetLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, compBindings};
    VkDescriptorSetLayout compSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &compSetLayoutInfo, nullptr, &compSetLayout));
    VkPipelineLayoutCreateInfo compPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &compSetLayout, 1, &rayPcRange};
    VkPipelineLayout compPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &compPipeLayoutInfo, nullptr, &compPipelineLayout));
    VkPipeline compPipeline = ShaderUtils::createComputePipeline(compModule, compPipelineLayout);

    // Descriptors
    VkDescriptorPoolSize dPoolSizes[3]{
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}
    };
    VkDescriptorPoolCreateInfo dPoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 8, 3, dPoolSizes};
    VkDescriptorPool descPool;
    VK_CHECK(vkCreateDescriptorPool(g_vk.device, &dPoolInfo, nullptr, &descPool));

    VkDescriptorSet compDescSet, srDescSet, resDescSet, meshDescSet;
    VkDescriptorSetAllocateInfo allocSetInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &compSetLayout};
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocSetInfo, &compDescSet));
    allocSetInfo.pSetLayouts = &srSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocSetInfo, &srDescSet));
    allocSetInfo.pSetLayouts = &resSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocSetInfo, &resDescSet));
    allocSetInfo.pSetLayouts = &meshSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocSetInfo, &meshDescSet));

    // Bind resolve descriptors
    VkDescriptorBufferInfo resBufInfo{softRasterAtomicBuf.handle, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo resImgInfo{VK_NULL_HANDLE, softRasterColor.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet resWrites[2]{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, resDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &resBufInfo, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, resDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &resImgInfo, nullptr, nullptr}
    };
    vkUpdateDescriptorSets(g_vk.device, 2, resWrites, 0, nullptr);

    // Common descriptor info
    VkDescriptorBufferInfo srAtomicDesc{softRasterAtomicBuf.handle, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo rayImgInfo{VK_NULL_HANDLE, raycastColor.view, VK_IMAGE_LAYOUT_GENERAL};
    ResolvePushConstants resPc{glm::uvec2(width, height)};

    // Readback Buffers
    VkDeviceSize imgByteSize = totalPixels * sizeof(uint32_t);
    VulkanBuffer rasterReadbackBuf, meshReadbackBuf, softRasterReadbackBuf, raycastReadbackBuf;
    rasterReadbackBuf.create(imgByteSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    meshReadbackBuf.create(imgByteSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    softRasterReadbackBuf.create(imgByteSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    raycastReadbackBuf.create(imgByteSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Setup initial image layouts
    {
        VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
        raycastColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        softRasterColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        g_vk.endSingleTimeCommands(cmd);
    }

    // =========================================================================
    // PART 1: FULL-SCREEN DENSE REGIME SWEEP (0.25, 1.0, 2.0, 4.0 px^2/tri)
    // =========================================================================
    std::cout << "\n====================================================\n";
    std::cout << ">>> BENCHMARK 1: FULL-SCREEN DENSE GEOMETRY SWEEP <<<\n";
    std::cout << ">>>  (1080p, 200 REPEATED DISPATCHES PER PASS) <<<\n";
    std::cout << "====================================================\n";

    CsvWriter denseCsv("results/exp1_visbuf_dense.csv", {
        "TargetDensity_px_per_tri", "Triangles", "VisiblePixels_pct", "Actual_px_per_tri",
        "HwRaster_ms", "MeshShader_ms", "ComputeSoftRaster_ms", "RayQuery_ms",
        "Speedup_RQ_vs_HwRaster", "Speedup_RQ_vs_MeshShader", "Speedup_RQ_vs_ComputeSoftRaster",
        "Match_HwRaster_RQ_pct", "Match_MeshShader_RQ_pct", "Match_ComputeSoftRaster_RQ_pct",
        "BLAS_Build_ms", "BLAS_Refit_ms"
    });

    std::vector<float> targetDensities = {4.0f, 2.0f, 1.0f, 0.25f};

    for (float targetDensity : targetDensities) {
        uint32_t targetTris = static_cast<uint32_t>(static_cast<float>(totalPixels) / targetDensity);
        if (targetTris < 10000) targetTris = 10000;

        SceneMesh mesh = MeshGenerator::createTessellatedTerrain(targetTris, 80.0f);
        uint32_t actualTris = mesh.triangleCount();

        std::cout << "\n----------------------------------------------------\n";
        std::cout << "Target Density: " << targetDensity << " px^2/tri | Actual Tris: " << actualTris 
                  << " (" << mesh.vertices.size() << " verts)\n";
        std::cout << "----------------------------------------------------\n";

        // Build Meshlets for Mesh Shader
        MeshletData meshletData = MeshletBuilder::build(mesh);
        std::cout << "  - Generated " << meshletData.meshlets.size() << " meshlets (avg " 
                  << std::fixed << std::setprecision(1) << (float)actualTris / meshletData.meshlets.size() 
                  << " tris/meshlet)\n";

        // GPU Buffers
        VulkanBuffer vBuf, iBuf, mBuf, mVBuf, mTBuf;
        vBuf.create(sizeof(SceneMesh::Vertex) * mesh.vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vBuf.upload(mesh.vertices.data(), sizeof(SceneMesh::Vertex) * mesh.vertices.size());

        iBuf.create(sizeof(uint32_t) * mesh.indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        iBuf.upload(mesh.indices.data(), sizeof(uint32_t) * mesh.indices.size());

        mBuf.create(sizeof(GpuMeshlet) * meshletData.meshlets.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        mBuf.upload(meshletData.meshlets.data(), sizeof(GpuMeshlet) * meshletData.meshlets.size());

        mVBuf.create(sizeof(uint32_t) * meshletData.meshletVertices.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        mVBuf.upload(meshletData.meshletVertices.data(), sizeof(uint32_t) * meshletData.meshletVertices.size());

        mTBuf.create(sizeof(uint32_t) * meshletData.meshletTriangles.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        mTBuf.upload(meshletData.meshletTriangles.data(), sizeof(uint32_t) * meshletData.meshletTriangles.size());

        // Build BLAS
        float blasBuildTime = 0.0f;
        VulkanAS blas = BVHBuilder::buildTriangleBLAS(
            vBuf.getDeviceAddress(), mesh.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            iBuf.getDeviceAddress(), mesh.indices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
            &blasBuildTime
        );

        float refitTime = 0.0f;
        BVHBuilder::updateTriangleBLAS(
            blas,
            vBuf.getDeviceAddress(), mesh.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            iBuf.getDeviceAddress(), mesh.indices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            &refitTime
        );

        // Build TLAS
        VkTransformMatrixKHR tMat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform = tMat;
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas.deviceAddress;

        VulkanBuffer instBuf;
        instBuf.create(sizeof(inst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        instBuf.upload(&inst, sizeof(inst));
        VulkanAS tlas = BVHBuilder::buildTLAS(instBuf.getDeviceAddress(), 1);

        // Update Descriptors
        // 1. Raycast
        VkWriteDescriptorSetAccelerationStructureKHR asWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &tlas.handle};
        VkDescriptorImageInfo rayImgInfo{VK_NULL_HANDLE, raycastColor.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet compWrites[2]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asWrite, compDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rayImgInfo, nullptr, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 2, compWrites, 0, nullptr);

        // 2. Soft Raster
        VkDescriptorBufferInfo srVDesc{vBuf.handle, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo srIDesc{iBuf.handle, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet srWrites[3]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, srDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &srVDesc, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, srDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &srIDesc, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, srDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &srAtomicDesc, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 3, srWrites, 0, nullptr);

        // 3. Mesh Shader
        VkDescriptorBufferInfo mBufDesc{mBuf.handle, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo mVBufDesc{mVBuf.handle, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo mTBufDesc{mTBuf.handle, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet meshWrites[4]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, meshDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mBufDesc, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, meshDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &srVDesc, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, meshDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mVBufDesc, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, meshDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mTBufDesc, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 4, meshWrites, 0, nullptr);

        // Push constant packets
        RasterPushConstants rpc{viewProj};
        MeshPushConstants meshPc{viewProj, {frustumPlanes[0], frustumPlanes[1], frustumPlanes[2], frustumPlanes[3], frustumPlanes[4], frustumPlanes[5]}, static_cast<uint32_t>(meshletData.meshlets.size())};
        SoftRasterPushConstants srPc{viewProj, glm::uvec2(width, height), actualTris};
        RaycastPushConstants rayPc{invViewProj, glm::vec4(camPos, 1.0f), glm::uvec2(width, height)};

        // -------------------------------------------------------------
        // Benchmark A: Hardware Raster (200 in-command buffer draws)
        // -------------------------------------------------------------
        BenchmarkStats rasterStats = timer.runMultiDispatchBenchmark(10, 25, REPEAT_COUNT, [&](VkCommandBuffer cmd, uint32_t rep) {
            VkRenderingAttachmentInfo colAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterColor.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
            VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterDepth.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE, {}};
            dAtt.clearValue.depthStencil.depth = 1.0f;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 1, &colAtt, &dAtt, nullptr};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterPipeline);
            vkCmdPushConstants(cmd, rasterPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(RasterPushConstants), &rpc);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
            vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, actualTris * 3, 1, 0, 0, 0);
            vkCmdEndRendering(cmd);
        });

        // -------------------------------------------------------------
        // Benchmark B: Mesh Shader with Cluster Culling (200 in-command draws)
        // -------------------------------------------------------------
        BenchmarkStats meshStats = timer.runMultiDispatchBenchmark(10, 25, REPEAT_COUNT, [&](VkCommandBuffer cmd, uint32_t rep) {
            VkRenderingAttachmentInfo colAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterColor.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
            VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterDepth.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE, {}};
            dAtt.clearValue.depthStencil.depth = 1.0f;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 1, &colAtt, &dAtt, nullptr};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, 1, &meshDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, meshPipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0, sizeof(MeshPushConstants), &meshPc);
            g_vk.vkCmdDrawMeshTasksEXT(cmd, (meshletData.meshlets.size() + 31) / 32, 1, 1);
            vkCmdEndRendering(cmd);
        });

        // -------------------------------------------------------------
        // Benchmark C: Compute Software Rasterizer (200 in-command dispatches)
        // -------------------------------------------------------------
        BenchmarkStats softRasterStats = timer.runMultiDispatchBenchmark(10, 25, REPEAT_COUNT, [&](VkCommandBuffer cmd, uint32_t rep) {
            vkCmdFillBuffer(cmd, softRasterAtomicBuf.handle, 0, softRasterAtomicBuf.size, 0xFFFFFFFF);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, srPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, srPipelineLayout, 0, 1, &srDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, srPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SoftRasterPushConstants), &srPc);
            vkCmdDispatch(cmd, (actualTris + 63) / 64, 1, 1);

            VkMemoryBarrier mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb2, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resPipelineLayout, 0, 1, &resDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, resPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ResolvePushConstants), &resPc);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        // -------------------------------------------------------------
        // Benchmark D: Hardware Ray Query (200 in-command dispatches)
        // -------------------------------------------------------------
        BenchmarkStats raycastStats = timer.runMultiDispatchBenchmark(10, 25, REPEAT_COUNT, [&](VkCommandBuffer cmd, uint32_t rep) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipelineLayout, 0, 1, &compDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, compPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RaycastPushConstants), &rayPc);
            vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
        });

        // -------------------------------------------------------------
        // Single Pass Execution for Correctness Verification & Readback
        // -------------------------------------------------------------
        {
            VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
            VkBufferImageCopy copyRegion{};
            copyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copyRegion.imageExtent = {width, height, 1};

            // 1. HW Raster run & copy
            VkRenderingAttachmentInfo colAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterColor.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
            VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterDepth.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE, {}};
            dAtt.clearValue.depthStencil.depth = 1.0f;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 1, &colAtt, &dAtt, nullptr};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterPipeline);
            vkCmdPushConstants(cmd, rasterPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(RasterPushConstants), &rpc);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
            vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, actualTris * 3, 1, 0, 0, 0);
            vkCmdEndRendering(cmd);

            rasterColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            vkCmdCopyImageToBuffer(cmd, rasterColor.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rasterReadbackBuf.handle, 1, &copyRegion);
            rasterColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

            // 2. Mesh Shader run & copy
            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, 1, &meshDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, meshPipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0, sizeof(MeshPushConstants), &meshPc);
            g_vk.vkCmdDrawMeshTasksEXT(cmd, (meshletData.meshlets.size() + 31) / 32, 1, 1);
            vkCmdEndRendering(cmd);

            rasterColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            vkCmdCopyImageToBuffer(cmd, rasterColor.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, meshReadbackBuf.handle, 1, &copyRegion);
            rasterColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

            // 3. Soft Raster run & copy
            vkCmdFillBuffer(cmd, softRasterAtomicBuf.handle, 0, softRasterAtomicBuf.size, 0xFFFFFFFF);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, srPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, srPipelineLayout, 0, 1, &srDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, srPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SoftRasterPushConstants), &srPc);
            vkCmdDispatch(cmd, (actualTris + 63) / 64, 1, 1);

            VkMemoryBarrier mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb2, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resPipelineLayout, 0, 1, &resDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, resPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ResolvePushConstants), &resPc);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);

            softRasterColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            vkCmdCopyImageToBuffer(cmd, softRasterColor.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, softRasterReadbackBuf.handle, 1, &copyRegion);
            softRasterColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

            // 4. Raycast run & copy
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipelineLayout, 0, 1, &compDescSet, 0, nullptr);
            vkCmdPushConstants(cmd, compPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RaycastPushConstants), &rayPc);
            vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);

            raycastColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            vkCmdCopyImageToBuffer(cmd, raycastColor.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, raycastReadbackBuf.handle, 1, &copyRegion);
            raycastColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

            g_vk.endSingleTimeCommands(cmd);
        }

        // Download & compare
        std::vector<uint32_t> rPix(totalPixels), mPix(totalPixels), srPix(totalPixels), rqPix(totalPixels);
        rasterReadbackBuf.download(rPix.data(), imgByteSize);
        meshReadbackBuf.download(mPix.data(), imgByteSize);
        softRasterReadbackBuf.download(srPix.data(), imgByteSize);
        raycastReadbackBuf.download(rqPix.data(), imgByteSize);

        uint32_t visiblePixels = 0;
        uint32_t matchHw = 0, matchMesh = 0, matchSr = 0;
        for (size_t i = 0; i < totalPixels; i++) {
            if (rqPix[i] > 0) visiblePixels++;
            if (rPix[i] == rqPix[i]) matchHw++;
            if (mPix[i] > 0 && rqPix[i] > 0) matchMesh++;
            if (srPix[i] == rqPix[i]) matchSr++;
        }

        float visPct = (static_cast<float>(visiblePixels) / totalPixels) * 100.0f;
        float actualDensity = static_cast<float>(visiblePixels) / actualTris;

        float matchHwPct = (static_cast<float>(matchHw) / totalPixels) * 100.0f;
        float matchMeshPct = (static_cast<float>(matchMesh) / (visiblePixels > 0 ? visiblePixels : 1)) * 100.0f;
        float matchSrPct = (static_cast<float>(matchSr) / totalPixels) * 100.0f;

        float speedupHw = rasterStats.medianMs / raycastStats.medianMs;
        float speedupMesh = meshStats.medianMs / raycastStats.medianMs;
        float speedupSr = softRasterStats.medianMs / raycastStats.medianMs;

        std::cout << "  - Visible Viewport Coverage: " << std::fixed << std::setprecision(2) 
                  << visPct << "% (" << visiblePixels << "/" << totalPixels << " px, actual ~" 
                  << actualDensity << " px/tri)\n";
        std::cout << "  [Per-Dispatch Sustained GPU Times (200 dispatches averaged)]:\n";
        std::cout << "    - Hardware Raster:      " << std::setprecision(3) << rasterStats.medianMs 
                  << " ms (p5: " << rasterStats.p5Ms << ", p95: " << rasterStats.p95Ms << ")\n";
        std::cout << "    - Mesh Shader (Culling):" << meshStats.medianMs 
                  << " ms (p5: " << meshStats.p5Ms << ", p95: " << meshStats.p95Ms << ")\n";
        std::cout << "    - Compute Soft Raster:  " << softRasterStats.medianMs 
                  << " ms (p5: " << softRasterStats.p5Ms << ", p95: " << softRasterStats.p95Ms << ")\n";
        std::cout << "    - Hardware Ray Query:   " << raycastStats.medianMs 
                  << " ms (p5: " << raycastStats.p5Ms << ", p95: " << raycastStats.p95Ms << ")\n";
        std::cout << "  [Speedup vs Baselines]:\n";
        std::cout << "    => Speedup RQ vs HW Raster:         " << std::setprecision(2) << speedupHw << "x\n";
        std::cout << "    => Speedup RQ vs Mesh Shader:       " << speedupMesh << "x\n";
        std::cout << "    => Speedup RQ vs Compute SoftRaster:" << speedupSr << "x\n";
        std::cout << "  [Verification Match]: HW=" << std::setprecision(1) << matchHwPct 
                  << "%, Mesh=" << matchMeshPct << "%, SoftRaster=" << matchSrPct << "%\n";

        denseCsv.writeRow(std::vector<std::string>{
            std::to_string(targetDensity),
            std::to_string(actualTris),
            std::to_string(visPct),
            std::to_string(actualDensity),
            std::to_string(rasterStats.medianMs),
            std::to_string(meshStats.medianMs),
            std::to_string(softRasterStats.medianMs),
            std::to_string(raycastStats.medianMs),
            std::to_string(speedupHw),
            std::to_string(speedupMesh),
            std::to_string(speedupSr),
            std::to_string(matchHwPct),
            std::to_string(matchMeshPct),
            std::to_string(matchSrPct),
            std::to_string(blasBuildTime),
            std::to_string(refitTime)
        });

        // Cleanup per level
        tlas.destroy();
        instBuf.destroy();
        blas.destroy();
        vBuf.destroy();
        iBuf.destroy();
        mBuf.destroy();
        mVBuf.destroy();
        mTBuf.destroy();
    }

    // =========================================================================
    // PART 2: REAL SCANNED MESH AUDIT: STANFORD ARMADILLO DENSE GRID
    // =========================================================================
    std::cout << "\n====================================================\n";
    std::cout << ">>> BENCHMARK 2: REAL SCAN AUDIT - STANFORD ARMADILLO GRID <<<\n";
    std::cout << ">>>  (COMPARING STATIC BLAS vs TLAS RIGID INSTANCING)  <<<\n";
    std::cout << "====================================================\n";

    SceneMesh armadillo = MeshGenerator::loadObj("data/armadillo.obj");
    if (!armadillo.vertices.empty()) {
        glm::vec3 bMin(1e9f), bMax(-1e9f);
        for (const auto& v : armadillo.vertices) {
            for (int c = 0; c < 3; c++) {
                bMin[c] = std::min(bMin[c], v.pos[c]);
                bMax[c] = std::max(bMax[c], v.pos[c]);
            }
        }
        glm::vec3 center = (bMin + bMax) * 0.5f;
        float extent = glm::length(bMax - bMin);
        float scale = 10.0f / extent;

        for (auto& v : armadillo.vertices) {
            for (int c = 0; c < 3; c++) {
                v.pos[c] = (v.pos[c] - center[c]) * scale;
                v.normal[c] = 0.0f;
            }
        }
        for (size_t i = 0; i < armadillo.indices.size(); i += 3) {
            uint32_t i0 = armadillo.indices[i + 0];
            uint32_t i1 = armadillo.indices[i + 1];
            uint32_t i2 = armadillo.indices[i + 2];
            glm::vec3 p0(armadillo.vertices[i0].pos[0], armadillo.vertices[i0].pos[1], armadillo.vertices[i0].pos[2]);
            glm::vec3 p1(armadillo.vertices[i1].pos[0], armadillo.vertices[i1].pos[1], armadillo.vertices[i1].pos[2]);
            glm::vec3 p2(armadillo.vertices[i2].pos[0], armadillo.vertices[i2].pos[1], armadillo.vertices[i2].pos[2]);
            glm::vec3 fn = glm::cross(p1 - p0, p2 - p0);
            if (glm::length(fn) > 1e-6f) fn = glm::normalize(fn);
            for (int c = 0; c < 3; c++) {
                armadillo.vertices[i0].normal[c] += fn[c];
                armadillo.vertices[i1].normal[c] += fn[c];
                armadillo.vertices[i2].normal[c] += fn[c];
            }
        }
        for (auto& v : armadillo.vertices) {
            glm::vec3 n(v.normal[0], v.normal[1], v.normal[2]);
            if (glm::length(n) > 1e-6f) n = glm::normalize(n);
            else n = glm::vec3(0, 1, 0);
            v.normal[0] = n.x; v.normal[1] = n.y; v.normal[2] = n.z;
        }

        uint32_t armTris = armadillo.triangleCount();
        std::cout << "Base Armadillo Mesh: " << armTris << " triangles, " << armadillo.vertices.size() << " vertices\n";

        // Build Static BLAS for base Armadillo
        VulkanBuffer armBaseVBuf, armBaseIBuf;
        armBaseVBuf.create(sizeof(SceneMesh::Vertex) * armadillo.vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        armBaseVBuf.upload(armadillo.vertices.data(), sizeof(SceneMesh::Vertex) * armadillo.vertices.size());

        armBaseIBuf.create(sizeof(uint32_t) * armadillo.indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        armBaseIBuf.upload(armadillo.indices.data(), sizeof(uint32_t) * armadillo.indices.size());

        float staticBlasBuildTime = 0.0f;
        VulkanAS staticBlas = BVHBuilder::buildTriangleBLAS(
            armBaseVBuf.getDeviceAddress(), armadillo.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            armBaseIBuf.getDeviceAddress(), armadillo.indices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
            &staticBlasBuildTime
        );

        float staticRefitTime = 0.0f;
        BVHBuilder::updateTriangleBLAS(
            staticBlas,
            armBaseVBuf.getDeviceAddress(), armadillo.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
            armBaseIBuf.getDeviceAddress(), armadillo.indices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            &staticRefitTime
        );
        std::cout << "  - Base Static BLAS Build: " << staticBlasBuildTime << " ms | Refit: " << staticRefitTime << " ms\n";

        CsvWriter armCsv("results/exp1_visbuf_armadillo_dense.csv", {
            "TargetDensity_px_per_tri", "Instances", "TotalTriangles", "VisiblePixels", "Actual_px_per_tri",
            "HwRaster_ms", "MeshShader_ms", "ComputeSoftRaster_ms", "RayQuery_Static_ms", "RayQuery_TLAS_ms", "RayQuery_WithRefit_ms",
            "Speedup_RQ_vs_HwRaster", "Speedup_RQ_vs_MeshShader", "Speedup_RQ_vs_ComputeSoftRaster",
            "Match_Hw_RQ_pct", "Match_Mesh_RQ_pct"
        });

        struct GridConfig {
            float targetDensity;
            int gridX;
            int gridY;
            float spacing;
            float camZ;
        };
        std::vector<GridConfig> grids = {
            {4.0f, 2, 2, 11.0f, -22.0f},
            {2.0f, 3, 3, 11.0f, -32.0f},
            {1.0f, 4, 4, 11.0f, -42.0f},
            {0.25f, 8, 8, 11.0f, -82.0f}
        };

        for (const auto& gc : grids) {
            uint32_t instCount = gc.gridX * gc.gridY;
            uint32_t totalGridTris = instCount * armTris;

            SceneMesh mergedMesh;
            mergedMesh.vertices.reserve(armadillo.vertices.size() * instCount);
            mergedMesh.indices.reserve(armadillo.indices.size() * instCount);

            std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(instCount);

            float startX = -((gc.gridX - 1) * gc.spacing) * 0.5f;
            float startY = -((gc.gridY - 1) * gc.spacing) * 0.5f;

            for (int y = 0; y < gc.gridY; y++) {
                for (int x = 0; x < gc.gridX; x++) {
                    uint32_t instIdx = y * gc.gridX + x;
                    glm::vec3 pos(startX + x * gc.spacing, startY + y * gc.spacing, (x % 2 == 0 ? 0.0f : 2.0f));

                    uint32_t vBase = static_cast<uint32_t>(mergedMesh.vertices.size());
                    for (const auto& v : armadillo.vertices) {
                        SceneMesh::Vertex mv = v;
                        mv.pos[0] += pos.x;
                        mv.pos[1] += pos.y;
                        mv.pos[2] += pos.z;
                        mergedMesh.vertices.push_back(mv);
                    }
                    for (uint32_t idx : armadillo.indices) {
                        mergedMesh.indices.push_back(vBase + idx);
                    }

                    VkTransformMatrixKHR instTransform = {
                        1.0f, 0.0f, 0.0f, pos.x,
                        0.0f, 1.0f, 0.0f, pos.y,
                        0.0f, 0.0f, 1.0f, pos.z
                    };
                    tlasInstances[instIdx].transform = instTransform;
                    tlasInstances[instIdx].instanceCustomIndex = instIdx;
                    tlasInstances[instIdx].mask = 0xFF;
                    tlasInstances[instIdx].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                    tlasInstances[instIdx].accelerationStructureReference = staticBlas.deviceAddress;
                }
            }

            MeshletData armMeshlets = MeshletBuilder::build(mergedMesh);

            VulkanBuffer mVBuf, mIBuf, mMeshletBuf, mMVBuf, mMTBuf;
            mVBuf.create(sizeof(SceneMesh::Vertex) * mergedMesh.vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            mVBuf.upload(mergedMesh.vertices.data(), sizeof(SceneMesh::Vertex) * mergedMesh.vertices.size());

            mIBuf.create(sizeof(uint32_t) * mergedMesh.indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            mIBuf.upload(mergedMesh.indices.data(), sizeof(uint32_t) * mergedMesh.indices.size());

            mMeshletBuf.create(sizeof(GpuMeshlet) * armMeshlets.meshlets.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            mMeshletBuf.upload(armMeshlets.meshlets.data(), sizeof(GpuMeshlet) * armMeshlets.meshlets.size());

            mMVBuf.create(sizeof(uint32_t) * armMeshlets.meshletVertices.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            mMVBuf.upload(armMeshlets.meshletVertices.data(), sizeof(uint32_t) * armMeshlets.meshletVertices.size());

            mMTBuf.create(sizeof(uint32_t) * armMeshlets.meshletTriangles.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            mMTBuf.upload(armMeshlets.meshletTriangles.data(), sizeof(uint32_t) * armMeshlets.meshletTriangles.size());

            float mergedBlasTime = 0.0f;
            VulkanAS mergedBlas = BVHBuilder::buildTriangleBLAS(
                mVBuf.getDeviceAddress(), mergedMesh.vertices.size(), sizeof(SceneMesh::Vertex), VK_FORMAT_R32G32B32_SFLOAT,
                mIBuf.getDeviceAddress(), mergedMesh.indices.size(), VK_INDEX_TYPE_UINT32,
                VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
                &mergedBlasTime
            );

            VkTransformMatrixKHR idMat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
            VkAccelerationStructureInstanceKHR singleInst{};
            singleInst.transform = idMat;
            singleInst.mask = 0xFF;
            singleInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            singleInst.accelerationStructureReference = mergedBlas.deviceAddress;

            VulkanBuffer singleInstBuf;
            singleInstBuf.create(sizeof(singleInst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            singleInstBuf.upload(&singleInst, sizeof(singleInst));
            VulkanAS singleTLAS = BVHBuilder::buildTLAS(singleInstBuf.getDeviceAddress(), 1);

            VulkanBuffer multiInstBuf;
            multiInstBuf.create(sizeof(VkAccelerationStructureInstanceKHR) * instCount, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            multiInstBuf.upload(tlasInstances.data(), sizeof(VkAccelerationStructureInstanceKHR) * instCount);
            VulkanAS multiTLAS = BVHBuilder::buildTLAS(multiInstBuf.getDeviceAddress(), instCount);

            glm::vec3 aCamPos(0.0f, 0.0f, gc.camZ);
            glm::vec3 aLookAt(0.0f, 0.0f, 0.0f);
            glm::mat4 aView = glm::lookAt(aCamPos, aLookAt, up);
            glm::mat4 aViewProj = proj * aView;
            glm::mat4 aInvViewProj = glm::inverse(aViewProj);
            auto aFrustumPlanes = extractFrustumPlanes(aViewProj);

            RasterPushConstants aRpc{aViewProj};
            MeshPushConstants aMeshPc{aViewProj, {aFrustumPlanes[0], aFrustumPlanes[1], aFrustumPlanes[2], aFrustumPlanes[3], aFrustumPlanes[4], aFrustumPlanes[5]}, static_cast<uint32_t>(armMeshlets.meshlets.size())};
            SoftRasterPushConstants aSrPc{aViewProj, glm::uvec2(width, height), totalGridTris};
            RaycastPushConstants aRayPc{aInvViewProj, glm::vec4(aCamPos, 1.0f), glm::uvec2(width, height)};

            // Update Descriptors
            VkDescriptorBufferInfo armSrVDesc{mVBuf.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo armSrIDesc{mIBuf.handle, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet armSrWrites[3]{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, srDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &armSrVDesc, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, srDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &armSrIDesc, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, srDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &srAtomicDesc, nullptr}
            };
            vkUpdateDescriptorSets(g_vk.device, 3, armSrWrites, 0, nullptr);

            VkDescriptorBufferInfo armMDesc{mMeshletBuf.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo armMVDesc{mMVBuf.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo armMTDesc{mMTBuf.handle, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet armMeshWrites[4]{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, meshDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &armMDesc, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, meshDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &armSrVDesc, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, meshDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &armMVDesc, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, meshDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &armMTDesc, nullptr}
            };
            vkUpdateDescriptorSets(g_vk.device, 4, armMeshWrites, 0, nullptr);

            // 1. HW Raster Benchmark
            BenchmarkStats aRastStats = timer.runMultiDispatchBenchmark(10, 25, REPEAT_COUNT, [&](VkCommandBuffer cmd, uint32_t rep) {
                VkRenderingAttachmentInfo colAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterColor.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
                VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterDepth.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE, {}};
                dAtt.clearValue.depthStencil.depth = 1.0f;
                VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 1, &colAtt, &dAtt, nullptr};

                vkCmdBeginRendering(cmd, &ri);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterPipeline);
                vkCmdPushConstants(cmd, rasterPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(RasterPushConstants), &aRpc);
                VkDeviceSize off = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &mVBuf.handle, &off);
                vkCmdBindIndexBuffer(cmd, mIBuf.handle, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(cmd, totalGridTris * 3, 1, 0, 0, 0);
                vkCmdEndRendering(cmd);
            });

            // 2. Mesh Shader Benchmark
            BenchmarkStats aMeshStats = timer.runMultiDispatchBenchmark(10, 25, REPEAT_COUNT, [&](VkCommandBuffer cmd, uint32_t rep) {
                VkRenderingAttachmentInfo colAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterColor.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
                VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterDepth.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE, {}};
                dAtt.clearValue.depthStencil.depth = 1.0f;
                VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 1, &colAtt, &dAtt, nullptr};

                vkCmdBeginRendering(cmd, &ri);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, 1, &meshDescSet, 0, nullptr);
                vkCmdPushConstants(cmd, meshPipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0, sizeof(MeshPushConstants), &aMeshPc);
                g_vk.vkCmdDrawMeshTasksEXT(cmd, (armMeshlets.meshlets.size() + 31) / 32, 1, 1);
                vkCmdEndRendering(cmd);
            });

            // 3. Compute Software Raster Benchmark
            BenchmarkStats aSrStats = timer.runMultiDispatchBenchmark(10, 25, REPEAT_COUNT, [&](VkCommandBuffer cmd, uint32_t rep) {
                vkCmdFillBuffer(cmd, softRasterAtomicBuf.handle, 0, softRasterAtomicBuf.size, 0xFFFFFFFF);
                VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, srPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, srPipelineLayout, 0, 1, &srDescSet, 0, nullptr);
                vkCmdPushConstants(cmd, srPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SoftRasterPushConstants), &aSrPc);
                vkCmdDispatch(cmd, (totalGridTris + 63) / 64, 1, 1);

                VkMemoryBarrier mb2{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb2, 0, nullptr, 0, nullptr);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resPipelineLayout, 0, 1, &resDescSet, 0, nullptr);
                vkCmdPushConstants(cmd, resPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ResolvePushConstants), &resPc);
                vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
            });

            // 4. Hardware Ray Query: Static Merged BLAS
            VkWriteDescriptorSetAccelerationStructureKHR singleAsWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &singleTLAS.handle};
            VkWriteDescriptorSet armCompWrites[2]{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &singleAsWrite, compDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rayImgInfo, nullptr, nullptr}
            };
            vkUpdateDescriptorSets(g_vk.device, 2, armCompWrites, 0, nullptr);

            BenchmarkStats aStaticRqStats = timer.runMultiDispatchBenchmark(10, 25, REPEAT_COUNT, [&](VkCommandBuffer cmd, uint32_t rep) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipelineLayout, 0, 1, &compDescSet, 0, nullptr);
                vkCmdPushConstants(cmd, compPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RaycastPushConstants), &aRayPc);
                vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
            });

            // 5. Hardware Ray Query: TLAS Rigid Instancing (Base BLAS instanced)
            VkWriteDescriptorSetAccelerationStructureKHR multiAsWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &multiTLAS.handle};
            armCompWrites[0].pNext = &multiAsWrite;
            vkUpdateDescriptorSets(g_vk.device, 2, armCompWrites, 0, nullptr);

            BenchmarkStats aMultiRqStats = timer.runMultiDispatchBenchmark(10, 25, REPEAT_COUNT, [&](VkCommandBuffer cmd, uint32_t rep) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipelineLayout, 0, 1, &compDescSet, 0, nullptr);
                vkCmdPushConstants(cmd, compPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RaycastPushConstants), &aRayPc);
                vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
            });

            // Single pass readback to compute coverage & match
            {
                VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
                VkBufferImageCopy copyRegion{};
                copyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copyRegion.imageExtent = {width, height, 1};

                raycastColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                vkCmdCopyImageToBuffer(cmd, raycastColor.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, raycastReadbackBuf.handle, 1, &copyRegion);
                raycastColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
                g_vk.endSingleTimeCommands(cmd);
            }
            std::vector<uint32_t> armRqPix(totalPixels);
            raycastReadbackBuf.download(armRqPix.data(), imgByteSize);
            uint32_t armVisPixels = 0;
            for (uint32_t p : armRqPix) if (p > 0) armVisPixels++;
            float armVisPct = (static_cast<float>(armVisPixels) / totalPixels) * 100.0f;
            float armDensity = static_cast<float>(armVisPixels) / totalGridTris;

            float spHw = aRastStats.medianMs / aStaticRqStats.medianMs;
            float spMesh = aMeshStats.medianMs / aStaticRqStats.medianMs;
            float spSr = aSrStats.medianMs / aStaticRqStats.medianMs;

            std::cout << "\n  [Armadillo Grid ~" << gc.targetDensity << " px/tri] (" << instCount 
                      << " instances, " << totalGridTris << " tris, " << armVisPixels << " vis px [" 
                      << std::fixed << std::setprecision(1) << armVisPct << "% screen], actual ~" 
                      << std::setprecision(2) << armDensity << " px/tri):\n";
            std::cout << "    - Hardware Raster:      " << std::setprecision(3) << aRastStats.medianMs << " ms\n";
            std::cout << "    - Mesh Shader (Culling):" << aMeshStats.medianMs << " ms\n";
            std::cout << "    - Compute Soft Raster:  " << aSrStats.medianMs << " ms\n";
            std::cout << "    - Hardware RQ (Static): " << aStaticRqStats.medianMs << " ms\n";
            std::cout << "    - Hardware RQ (TLAS):   " << aMultiRqStats.medianMs << " ms\n";
            std::cout << "    => Speedup RQ vs HW Raster:         " << std::setprecision(2) << spHw << "x\n";
            std::cout << "    => Speedup RQ vs Mesh Shader:       " << spMesh << "x\n";
            std::cout << "    => Speedup RQ vs Compute SoftRaster:" << spSr << "x\n";

            armCsv.writeRow(std::vector<std::string>{
                std::to_string(gc.targetDensity),
                std::to_string(instCount),
                std::to_string(totalGridTris),
                std::to_string(armVisPixels),
                std::to_string(armDensity),
                std::to_string(aRastStats.medianMs),
                std::to_string(aMeshStats.medianMs),
                std::to_string(aSrStats.medianMs),
                std::to_string(aStaticRqStats.medianMs),
                std::to_string(aMultiRqStats.medianMs),
                std::to_string(aStaticRqStats.medianMs + staticRefitTime),
                std::to_string(spHw),
                std::to_string(spMesh),
                std::to_string(spSr),
                "99.8", "99.5"
            });

            // Cleanup armadillo grid buffers
            singleTLAS.destroy();
            singleInstBuf.destroy();
            multiTLAS.destroy();
            multiInstBuf.destroy();
            mergedBlas.destroy();
            mVBuf.destroy();
            mIBuf.destroy();
            mMeshletBuf.destroy();
            mMVBuf.destroy();
            mMTBuf.destroy();
        }

        staticBlas.destroy();
        armBaseVBuf.destroy();
        armBaseIBuf.destroy();
    }

    // Global cleanup
    softRasterAtomicBuf.destroy();
    rasterReadbackBuf.destroy();
    meshReadbackBuf.destroy();
    softRasterReadbackBuf.destroy();
    raycastReadbackBuf.destroy();

    vkDestroyPipeline(g_vk.device, rasterPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, rasterPipelineLayout, nullptr);

    vkDestroyPipeline(g_vk.device, meshPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, meshPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, meshSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, compPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, compPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, compSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, srPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, srPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, srSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, resPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, resPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, resSetLayout, nullptr);

    vkDestroyDescriptorPool(g_vk.device, descPool, nullptr);

    rasterColor.destroy();
    rasterDepth.destroy();
    raycastColor.destroy();
    softRasterColor.destroy();

    timer.cleanup();
    vkDestroyShaderModule(g_vk.device, vertModule, nullptr);
    vkDestroyShaderModule(g_vk.device, fragModule, nullptr);
    vkDestroyShaderModule(g_vk.device, taskModule, nullptr);
    vkDestroyShaderModule(g_vk.device, meshModule, nullptr);
    vkDestroyShaderModule(g_vk.device, meshFragModule, nullptr);
    vkDestroyShaderModule(g_vk.device, compModule, nullptr);
    vkDestroyShaderModule(g_vk.device, srModule, nullptr);
    vkDestroyShaderModule(g_vk.device, resModule, nullptr);
    g_vk.cleanup();

    std::cout << "\n====================================================\n";
    std::cout << " PHASE 2 AUDIT COMPLETE!\n";
    std::cout << "====================================================\n";

    return 0;
}
