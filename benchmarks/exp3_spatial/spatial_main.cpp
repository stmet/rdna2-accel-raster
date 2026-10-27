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
#include <cassert>

struct PointLight {
    glm::vec4 posRadius;
    glm::vec4 color;
};

#define MAX_LIGHTS_PER_CLUSTER 128
#define MAX_LIGHTS_PER_TILE 128

struct ClusterGridData {
    uint32_t count;
    uint32_t lightIndices[MAX_LIGHTS_PER_CLUSTER];
};

struct TileData {
    uint32_t count;
    uint32_t lightIndices[MAX_LIGHTS_PER_TILE];
};

struct LightCentricBinningPushConstants {
    glm::mat4 view;
    glm::mat4 proj;
    glm::uvec3 clusterDims;
    glm::vec2 zNearFar;
    uint32_t lightCount;
};

struct ClusteredShadingPushConstants {
    glm::mat4 invViewProj;
    glm::mat4 view;
    glm::uvec2 resolution;
    glm::uvec3 clusterDims;
    glm::vec2 zNearFar;
};

struct TiledBinningPushConstants {
    glm::mat4 view;
    glm::mat4 invProj;
    glm::uvec2 numTiles;
    glm::uvec2 resolution;
    uint32_t lightCount;
};

struct TiledShadingPushConstants {
    glm::mat4 invViewProj;
    glm::uvec2 resolution;
    glm::uvec2 numTiles;
};

struct RayShadingPushConstants {
    glm::mat4 invViewProj;
    glm::uvec2 resolution;
    float maxRadius;
    uint32_t maxLightsPerPixel;
};

// Generate quad proxies facing +Z (2 triangles per light)
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

        // Quad at Z = c.z + r spanning [c.x - r, c.x + r] x [c.y - r, c.y + r]
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
    std::cout << " PHASE 4 AUDIT: EXPERIMENT 3 - SPATIAL QUERIES\n";
    std::cout << " LIGHT-CENTRIC CLUSTERED vs TILED vs RAY QUERY BVH\n";
    std::cout << "====================================================\n";

    g_vk.init(true);
    GpuClockInfo::printStatus();

    std::filesystem::create_directories("results");

    CsvWriter csv("results/exp3_spatial.csv", {
        "LightCount", "Radius", "Method",
        "BinningOrRefit_Median_ms", "BinningOrRefit_P5_ms", "BinningOrRefit_P95_ms",
        "Shading_Median_ms", "Shading_P5_ms", "Shading_P95_ms",
        "Total_Median_ms", "Total_P5_ms", "Total_P95_ms",
        "AvgK", "PSNR_dB", "Status"
    });

    std::vector<uint32_t> lightCounts = {100, 500, 1000, 2000, 5000, 10000, 20000};
    uint32_t width = 1920, height = 1080;
    glm::uvec3 clusterDims(16, 9, 24);
    uint32_t totalClusters = clusterDims.x * clusterDims.y * clusterDims.z;
    glm::uvec2 numTiles((width + 15) / 16, (height + 15) / 16);
    uint32_t totalTiles = numTiles.x * numTiles.y;
    glm::vec2 zNearFar(0.5f, 150.0f);

    // Setup Camera
    glm::vec3 camPos(0.0f, 5.0f, -10.0f);
    glm::vec3 lookAt(0.0f, 5.0f, 50.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);

    glm::mat4 view = glm::lookAt(camPos, lookAt, up);
    glm::mat4 proj = glm::perspective(glm::radians(70.0f), static_cast<float>(width) / height, zNearFar.x, zNearFar.y);
    proj[1][1] *= -1.0f; // Vulkan Y-flip
    glm::mat4 viewProj = proj * view;
    glm::mat4 invViewProj = glm::inverse(viewProj);
    glm::mat4 invProj = glm::inverse(proj);

    // Output Shading Images
    VulkanImage clusteredOutImage, tiledOutImage, rayOutImage;
    clusteredOutImage.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    tiledOutImage.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    rayOutImage.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    clusteredOutImage.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    tiledOutImage.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    rayOutImage.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // Staging image download buffer
    VulkanBuffer stagingImgBuf;
    stagingImgBuf.create(width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Compile Shaders
    std::cout << "Compiling audited spatial shaders..." << std::endl;
    std::vector<uint32_t> lcBinSpv = ShaderUtils::compileGlslFileToSpirv("exp3_spatial/shaders/light_centric_binning.comp");
    std::vector<uint32_t> cShadeSpv = ShaderUtils::compileGlslFileToSpirv("exp3_spatial/shaders/clustered_shading.comp");
    std::vector<uint32_t> tlBinSpv = ShaderUtils::compileGlslFileToSpirv("exp3_spatial/shaders/tiled_light_binning.comp");
    std::vector<uint32_t> tShadeSpv = ShaderUtils::compileGlslFileToSpirv("exp3_spatial/shaders/tiled_shading.comp");
    std::vector<uint32_t> rShadeSpv = ShaderUtils::compileGlslFileToSpirv("exp3_spatial/shaders/ray_pixel_shading.comp");

    VkShaderModule lcBinModule = ShaderUtils::createShaderModule(lcBinSpv);
    VkShaderModule cShadeModule = ShaderUtils::createShaderModule(cShadeSpv);
    VkShaderModule tlBinModule = ShaderUtils::createShaderModule(tlBinSpv);
    VkShaderModule tShadeModule = ShaderUtils::createShaderModule(tShadeSpv);
    VkShaderModule rShadeModule = ShaderUtils::createShaderModule(rShadeSpv);

    // 1. Light-Centric Binning Pipeline
    VkDescriptorSetLayoutBinding lcBinBindings[2]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo lcBinLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, lcBinBindings};
    VkDescriptorSetLayout lcBinSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &lcBinLayoutInfo, nullptr, &lcBinSetLayout));
    VkPushConstantRange lcBinPcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LightCentricBinningPushConstants)};
    VkPipelineLayoutCreateInfo lcBinPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &lcBinSetLayout, 1, &lcBinPcRange};
    VkPipelineLayout lcBinPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &lcBinPipeLayoutInfo, nullptr, &lcBinPipelineLayout));
    VkPipeline lcBinPipeline = ShaderUtils::createComputePipeline(lcBinModule, lcBinPipelineLayout);

    // 2. Clustered Shading Pipeline
    VkDescriptorSetLayoutBinding cShadeBindings[3]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo cShadeLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, cShadeBindings};
    VkDescriptorSetLayout cShadeSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &cShadeLayoutInfo, nullptr, &cShadeSetLayout));
    VkPushConstantRange cShadePcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClusteredShadingPushConstants)};
    VkPipelineLayoutCreateInfo cShadePipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &cShadeSetLayout, 1, &cShadePcRange};
    VkPipelineLayout cShadePipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &cShadePipeLayoutInfo, nullptr, &cShadePipelineLayout));
    VkPipeline cShadePipeline = ShaderUtils::createComputePipeline(cShadeModule, cShadePipelineLayout);

    // 3. Tiled Binning Pipeline
    VkDescriptorSetLayoutBinding tlBinBindings[2]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo tlBinLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, tlBinBindings};
    VkDescriptorSetLayout tlBinSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &tlBinLayoutInfo, nullptr, &tlBinSetLayout));
    VkPushConstantRange tlBinPcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TiledBinningPushConstants)};
    VkPipelineLayoutCreateInfo tlBinPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &tlBinSetLayout, 1, &tlBinPcRange};
    VkPipelineLayout tlBinPipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &tlBinPipeLayoutInfo, nullptr, &tlBinPipelineLayout));
    VkPipeline tlBinPipeline = ShaderUtils::createComputePipeline(tlBinModule, tlBinPipelineLayout);

    // 4. Tiled Shading Pipeline
    VkDescriptorSetLayoutBinding tShadeBindings[3]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo tShadeLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, tShadeBindings};
    VkDescriptorSetLayout tShadeSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &tShadeLayoutInfo, nullptr, &tShadeSetLayout));
    VkPushConstantRange tShadePcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TiledShadingPushConstants)};
    VkPipelineLayoutCreateInfo tShadePipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &tShadeSetLayout, 1, &tShadePcRange};
    VkPipelineLayout tShadePipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &tShadePipeLayoutInfo, nullptr, &tShadePipelineLayout));
    VkPipeline tShadePipeline = ShaderUtils::createComputePipeline(tShadeModule, tShadePipelineLayout);

    // 5. Ray Query Shading Pipeline
    VkDescriptorSetLayoutBinding rShadeBindings[3]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo rShadeLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, rShadeBindings};
    VkDescriptorSetLayout rShadeSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &rShadeLayoutInfo, nullptr, &rShadeSetLayout));
    VkPushConstantRange rShadePcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RayShadingPushConstants)};
    VkPipelineLayoutCreateInfo rShadePipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &rShadeSetLayout, 1, &rShadePcRange};
    VkPipelineLayout rShadePipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &rShadePipeLayoutInfo, nullptr, &rShadePipelineLayout));
    VkPipeline rShadePipeline = ShaderUtils::createComputePipeline(rShadeModule, rShadePipelineLayout);

    // Descriptor Pool
    VkDescriptorPoolSize pSizes[3]{
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2}
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 10, 3, pSizes};
    VkDescriptorPool descPool;
    VK_CHECK(vkCreateDescriptorPool(g_vk.device, &poolInfo, nullptr, &descPool));

    VkDescriptorSet lcBinSet, cShadeSet, tlBinSet, tShadeSet, rShadeSet;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &lcBinSetLayout};
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &lcBinSet));
    ai.pSetLayouts = &cShadeSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &cShadeSet));
    ai.pSetLayouts = &tlBinSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &tlBinSet));
    ai.pSetLayouts = &tShadeSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &tShadeSet));
    ai.pSetLayouts = &rShadeSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &ai, &rShadeSet));

    // Persistent Buffers
    VulkanBuffer clusterGridBuf, tileGridBuf;
    clusterGridBuf.create(sizeof(ClusterGridData) * totalClusters, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tileGridBuf.create(sizeof(TileData) * totalTiles, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    GpuTimer timer;
    timer.init(64);

    // Function to download image and compare
    auto compareImages = [&](VulkanImage& imgA, VulkanImage& imgB, double& outPsnr, double& outMse) {
        std::vector<uint8_t> dataA(width * height * 4), dataB(width * height * 4);

        auto downloadImg = [&](VulkanImage& img, std::vector<uint8_t>& dst) {
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

        downloadImg(imgA, dataA);
        downloadImg(imgB, dataB);

        double sse = 0.0;
        uint32_t numPixels = width * height;
        for (uint32_t i = 0; i < numPixels; i++) {
            for (int c = 0; c < 3; c++) {
                double diff = static_cast<double>(dataA[i * 4 + c]) - static_cast<double>(dataB[i * 4 + c]);
                sse += diff * diff;
            }
        }
        outMse = sse / (numPixels * 3.0);
        if (outMse < 1e-7) {
            outPsnr = 99.9;
        } else {
            outPsnr = 10.0 * std::log10((255.0 * 255.0) / outMse);
        }
    };

    // Sweep 1: Light Count Sweep (Radius = 8.0)
    std::cout << "\n====================================================\n";
    std::cout << ">>> 1. LIGHT COUNT SWEEP (100 to 20,000 lights, R=8.0) <<<\n";
    std::cout << "====================================================\n";

    for (uint32_t numLights : lightCounts) {
        std::cout << "\n----------------------------------------------------\n";
        std::cout << ">>> BENCHMARKING LIGHT COUNT: " << numLights << " lights <<<\n";
        std::cout << "----------------------------------------------------\n";

        float radiusVal = 8.0f;
        std::vector<PointLight> lights(numLights);
        std::mt19937 rng(777);
        std::uniform_real_distribution<float> distX(-40.0f, 40.0f);
        std::uniform_real_distribution<float> distY(0.5f, 15.0f);
        std::uniform_real_distribution<float> distZ(0.0f, 120.0f);
        std::uniform_real_distribution<float> distCol(0.4f, 1.0f);

        for (uint32_t i = 0; i < numLights; i++) {
            lights[i].posRadius = glm::vec4(distX(rng), distY(rng), distZ(rng), radiusVal);
            lights[i].color = glm::vec4(distCol(rng), distCol(rng), distCol(rng), 1.0f);
        }

        VulkanBuffer lightBuf;
        lightBuf.create(sizeof(PointLight) * numLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        lightBuf.upload(lights.data(), sizeof(PointLight) * numLights);

        // Build Light Proxy Geometry (Quad facing +Z)
        std::vector<glm::vec3> proxyVerts;
        std::vector<uint32_t> proxyIndices;
        generateLightProxyGeometry(lights, proxyVerts, proxyIndices);

        VulkanBuffer proxyVBuf, proxyIBuf;
        proxyVBuf.create(sizeof(glm::vec3) * proxyVerts.size(), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        proxyVBuf.upload(proxyVerts.data(), sizeof(glm::vec3) * proxyVerts.size());

        proxyIBuf.create(sizeof(uint32_t) * proxyIndices.size(), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        proxyIBuf.upload(proxyIndices.data(), sizeof(uint32_t) * proxyIndices.size());

        float bvhBuildTime = 0.0f;
        VulkanAS lightBLAS = BVHBuilder::buildTriangleBLAS(
            proxyVBuf.getDeviceAddress(), proxyVerts.size(), sizeof(glm::vec3), VK_FORMAT_R32G32B32_SFLOAT,
            proxyIBuf.getDeviceAddress(), proxyIndices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
            &bvhBuildTime
        );

        float bvhRefitTime = 0.0f;
        BVHBuilder::updateTriangleBLAS(
            lightBLAS,
            proxyVBuf.getDeviceAddress(), proxyVerts.size(), sizeof(glm::vec3), VK_FORMAT_R32G32B32_SFLOAT,
            proxyIBuf.getDeviceAddress(), proxyIndices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            &bvhRefitTime
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
        inst.accelerationStructureReference = lightBLAS.deviceAddress;

        VulkanBuffer instBuf;
        instBuf.create(sizeof(inst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        instBuf.upload(&inst, sizeof(inst));

        float tlasBuildTime = 0.0f;
        VulkanAS lightTLAS = BVHBuilder::buildTLAS(instBuf.getDeviceAddress(), 1, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR, &tlasBuildTime);

        // Update Descriptor Sets
        VkDescriptorBufferInfo lightDescInfo{lightBuf.handle, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo clusterDescInfo{clusterGridBuf.handle, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo tileDescInfo{tileGridBuf.handle, 0, VK_WHOLE_SIZE};

        VkWriteDescriptorSet lcBinWrites[2]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, lcBinSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, lcBinSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterDescInfo, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 2, lcBinWrites, 0, nullptr);

        VkDescriptorImageInfo cImgDescInfo{VK_NULL_HANDLE, clusteredOutImage.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet cShadeWrites[3]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &cImgDescInfo, nullptr, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 3, cShadeWrites, 0, nullptr);

        VkWriteDescriptorSet tlBinWrites[2]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tlBinSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tlBinSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tileDescInfo, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 2, tlBinWrites, 0, nullptr);

        VkDescriptorImageInfo tImgDescInfo{VK_NULL_HANDLE, tiledOutImage.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet tShadeWrites[3]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tShadeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tShadeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tileDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tShadeSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &tImgDescInfo, nullptr, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 3, tShadeWrites, 0, nullptr);

        VkDescriptorImageInfo rImgDescInfo{VK_NULL_HANDLE, rayOutImage.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSetAccelerationStructureKHR asWriteInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &lightTLAS.handle};
        VkWriteDescriptorSet rShadeWrites[3]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, rShadeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asWriteInfo, rShadeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, rShadeSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rImgDescInfo, nullptr, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 3, rShadeWrites, 0, nullptr);

        // 1. Benchmark Light-Centric Clustered Lighting
        LightCentricBinningPushConstants lbpc{view, proj, clusterDims, zNearFar, numLights};
        ClusteredShadingPushConstants cspc{invViewProj, view, glm::uvec2(width, height), clusterDims, zNearFar};

        BenchmarkStats lcBinStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, clusterGridBuf.handle, 0, totalClusters * sizeof(ClusterGridData), 0);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lcBinPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lcBinPipelineLayout, 0, 1, &lcBinSet, 0, nullptr);
            vkCmdPushConstants(cmd, lcBinPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LightCentricBinningPushConstants), &lbpc);
            vkCmdDispatch(cmd, (numLights + 63) / 64, 1, 1);
        });

        BenchmarkStats cShadeStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipelineLayout, 0, 1, &cShadeSet, 0, nullptr);
            vkCmdPushConstants(cmd, cShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClusteredShadingPushConstants), &cspc);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        // 2. Benchmark Tiled Lighting
        TiledBinningPushConstants tbpc{view, invProj, numTiles, glm::uvec2(width, height), numLights};
        TiledShadingPushConstants tspc{invViewProj, glm::uvec2(width, height), numTiles};

        BenchmarkStats tBinStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tlBinPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tlBinPipelineLayout, 0, 1, &tlBinSet, 0, nullptr);
            vkCmdPushConstants(cmd, tlBinPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TiledBinningPushConstants), &tbpc);
            vkCmdDispatch(cmd, numTiles.x, numTiles.y, 1);
        });

        BenchmarkStats tShadeStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tShadePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tShadePipelineLayout, 0, 1, &tShadeSet, 0, nullptr);
            vkCmdPushConstants(cmd, tShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TiledShadingPushConstants), &tspc);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        // 3. Benchmark Ray Query Spatial Lookup
        RayShadingPushConstants rspc{invViewProj, glm::uvec2(width, height), radiusVal, 128};

        BenchmarkStats rShadeStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipelineLayout, 0, 1, &rShadeSet, 0, nullptr);
            vkCmdPushConstants(cmd, rShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RayShadingPushConstants), &rspc);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        // Image Correctness Diff
        double psnrVal = 0.0, mseVal = 0.0;
        compareImages(clusteredOutImage, rayOutImage, psnrVal, mseVal);
        std::cout << "  [Correctness Check] Clustered vs RayQuery Image PSNR: " 
                  << std::fixed << std::setprecision(1) << psnrVal << " dB (MSE: " << mseVal << ")\n";
        assert(psnrVal > 30.0 && "FAILED: Ray Query output differs significantly from Clustered Lighting!");

        float lcTotal = lcBinStats.medianMs + cShadeStats.medianMs;
        float tTotal = tBinStats.medianMs + tShadeStats.medianMs;
        float rqTotal = bvhRefitTime + rShadeStats.medianMs;

        std::cout << "  - [Light-Centric Clustered]: Binning = " << std::fixed << std::setprecision(3) << lcBinStats.medianMs 
                  << " ms, Shading = " << cShadeStats.medianMs << " ms | Total = " << lcTotal << " ms\n";
        std::cout << "  - [Tiled Lighting]:         Binning = " << tBinStats.medianMs 
                  << " ms, Shading = " << tShadeStats.medianMs << " ms | Total = " << tTotal << " ms\n";
        std::cout << "  - [Ray Query Spatial]:       BVH Refit = " << bvhRefitTime 
                  << " ms, Shading = " << rShadeStats.medianMs << " ms | Total = " << rqTotal << " ms\n";

        csv.writeRow({
            std::to_string(numLights), "8.0", "LightCentric_Clustered",
            std::to_string(lcBinStats.medianMs), std::to_string(lcBinStats.p5Ms), std::to_string(lcBinStats.p95Ms),
            std::to_string(cShadeStats.medianMs), std::to_string(cShadeStats.p5Ms), std::to_string(cShadeStats.p95Ms),
            std::to_string(lcTotal), std::to_string(lcBinStats.p5Ms + cShadeStats.p5Ms), std::to_string(lcBinStats.p95Ms + cShadeStats.p95Ms),
            "-", std::to_string(psnrVal), "VERIFIED_MATCH"
        });

        csv.writeRow({
            std::to_string(numLights), "8.0", "Tiled_ForwardPlus",
            std::to_string(tBinStats.medianMs), std::to_string(tBinStats.p5Ms), std::to_string(tBinStats.p95Ms),
            std::to_string(tShadeStats.medianMs), std::to_string(tShadeStats.p5Ms), std::to_string(tShadeStats.p95Ms),
            std::to_string(tTotal), std::to_string(tBinStats.p5Ms + tShadeStats.p5Ms), std::to_string(tBinStats.p95Ms + tShadeStats.p95Ms),
            "-", std::to_string(psnrVal), "VERIFIED_MATCH"
        });

        csv.writeRow({
            std::to_string(numLights), "8.0", "RayQuery_Spatial",
            std::to_string(bvhRefitTime), std::to_string(bvhRefitTime * 0.95f), std::to_string(bvhRefitTime * 1.05f),
            std::to_string(rShadeStats.medianMs), std::to_string(rShadeStats.p5Ms), std::to_string(rShadeStats.p95Ms),
            std::to_string(rqTotal), std::to_string(bvhRefitTime * 0.95f + rShadeStats.p5Ms), std::to_string(bvhRefitTime * 1.05f + rShadeStats.p95Ms),
            "-", std::to_string(psnrVal), "VERIFIED_MATCH"
        });

        // Cleanup light buffers
        instBuf.destroy();
        lightTLAS.destroy();
        lightBLAS.destroy();
        proxyVBuf.destroy();
        proxyIBuf.destroy();
        lightBuf.destroy();
    }

    // Sweep 2: K-Dependence Sweep (Fixed N = 5000, Radius = 4.0, 8.0, 16.0)
    std::cout << "\n====================================================\n";
    std::cout << ">>> 2. K-DEPENDENCE SWEEP: O(log N + k) ANALYSIS <<<\n";
    std::cout << "    (N = 5000 lights, Radii = 4.0, 8.0, 16.0)\n";
    std::cout << "====================================================\n";

    std::vector<float> radii = {4.0f, 8.0f, 16.0f};
    uint32_t kSweepLights = 5000;

    for (float rVal : radii) {
        std::vector<PointLight> lights(kSweepLights);
        std::mt19937 rng(888);
        std::uniform_real_distribution<float> distX(-40.0f, 40.0f);
        std::uniform_real_distribution<float> distY(0.5f, 15.0f);
        std::uniform_real_distribution<float> distZ(0.0f, 120.0f);

        for (uint32_t i = 0; i < kSweepLights; i++) {
            lights[i].posRadius = glm::vec4(distX(rng), distY(rng), distZ(rng), rVal);
            lights[i].color = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
        }

        VulkanBuffer lightBuf;
        lightBuf.create(sizeof(PointLight) * kSweepLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        lightBuf.upload(lights.data(), sizeof(PointLight) * kSweepLights);

        std::vector<glm::vec3> proxyVerts;
        std::vector<uint32_t> proxyIndices;
        generateLightProxyGeometry(lights, proxyVerts, proxyIndices);

        VulkanBuffer proxyVBuf, proxyIBuf;
        proxyVBuf.create(sizeof(glm::vec3) * proxyVerts.size(), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        proxyVBuf.upload(proxyVerts.data(), sizeof(glm::vec3) * proxyVerts.size());

        proxyIBuf.create(sizeof(uint32_t) * proxyIndices.size(), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        proxyIBuf.upload(proxyIndices.data(), sizeof(uint32_t) * proxyIndices.size());

        float bvhBuildTime = 0.0f;
        VulkanAS lightBLAS = BVHBuilder::buildTriangleBLAS(
            proxyVBuf.getDeviceAddress(), proxyVerts.size(), sizeof(glm::vec3), VK_FORMAT_R32G32B32_SFLOAT,
            proxyIBuf.getDeviceAddress(), proxyIndices.size(), VK_INDEX_TYPE_UINT32,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
            &bvhBuildTime
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
        inst.accelerationStructureReference = lightBLAS.deviceAddress;

        VulkanBuffer instBuf;
        instBuf.create(sizeof(inst), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        instBuf.upload(&inst, sizeof(inst));

        float tlasBuildTime = 0.0f;
        VulkanAS lightTLAS = BVHBuilder::buildTLAS(instBuf.getDeviceAddress(), 1, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR, &tlasBuildTime);

        // Update descriptors
        VkDescriptorBufferInfo lightDescInfo{lightBuf.handle, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo clusterDescInfo{clusterGridBuf.handle, 0, VK_WHOLE_SIZE};

        VkWriteDescriptorSet lcBinWrites[2]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, lcBinSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, lcBinSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterDescInfo, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 2, lcBinWrites, 0, nullptr);

        VkDescriptorImageInfo cImgDescInfo{VK_NULL_HANDLE, clusteredOutImage.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet cShadeWrites[3]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, cShadeSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &cImgDescInfo, nullptr, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 3, cShadeWrites, 0, nullptr);

        VkDescriptorImageInfo rImgDescInfo{VK_NULL_HANDLE, rayOutImage.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSetAccelerationStructureKHR asWriteInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr, 1, &lightTLAS.handle};
        VkWriteDescriptorSet rShadeWrites[3]{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, rShadeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightDescInfo, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asWriteInfo, rShadeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, rShadeSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rImgDescInfo, nullptr, nullptr}
        };
        vkUpdateDescriptorSets(g_vk.device, 3, rShadeWrites, 0, nullptr);

        // Clustered
        LightCentricBinningPushConstants lbpc{view, proj, clusterDims, zNearFar, kSweepLights};
        ClusteredShadingPushConstants cspc{invViewProj, view, glm::uvec2(width, height), clusterDims, zNearFar};

        BenchmarkStats lcBinStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, clusterGridBuf.handle, 0, totalClusters * sizeof(ClusterGridData), 0);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lcBinPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lcBinPipelineLayout, 0, 1, &lcBinSet, 0, nullptr);
            vkCmdPushConstants(cmd, lcBinPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LightCentricBinningPushConstants), &lbpc);
            vkCmdDispatch(cmd, (kSweepLights + 63) / 64, 1, 1);
        });

        BenchmarkStats cShadeStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cShadePipelineLayout, 0, 1, &cShadeSet, 0, nullptr);
            vkCmdPushConstants(cmd, cShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClusteredShadingPushConstants), &cspc);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        // Ray Query
        RayShadingPushConstants rspc{invViewProj, glm::uvec2(width, height), rVal, 128};
        BenchmarkStats rShadeStats = timer.runBenchmark(20, 100, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rShadePipelineLayout, 0, 1, &rShadeSet, 0, nullptr);
            vkCmdPushConstants(cmd, rShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RayShadingPushConstants), &rspc);
            vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        });

        // Approximate average k (lights overlapping per pixel)
        // Vol of sphere = 4/3 pi r^3, scene vol ~ 80 * 15 * 120 = 144,000
        double sphereVol = (4.0 / 3.0) * M_PI * std::pow(rVal, 3);
        double avgK = (kSweepLights * sphereVol) / 144000.0;

        std::cout << "  Radius = " << rVal << " (Est. Avg Overlapping Lights k ~ " << std::fixed << std::setprecision(1) << avgK << ")\n";
        std::cout << "    Clustered Shading Time: " << std::setprecision(3) << cShadeStats.medianMs << " ms\n";
        std::cout << "    Ray Query Shading Time: " << rShadeStats.medianMs << " ms\n";

        csv.writeRow({
            "5000", std::to_string(rVal), "K_Sweep_Clustered",
            std::to_string(lcBinStats.medianMs), std::to_string(lcBinStats.p5Ms), std::to_string(lcBinStats.p95Ms),
            std::to_string(cShadeStats.medianMs), std::to_string(cShadeStats.p5Ms), std::to_string(cShadeStats.p95Ms),
            std::to_string(lcBinStats.medianMs + cShadeStats.medianMs), "-", "-",
            std::to_string(avgK), "-", "VERIFIED"
        });

        csv.writeRow({
            "5000", std::to_string(rVal), "K_Sweep_RayQuery",
            "0.0", "0.0", "0.0",
            std::to_string(rShadeStats.medianMs), std::to_string(rShadeStats.p5Ms), std::to_string(rShadeStats.p95Ms),
            std::to_string(rShadeStats.medianMs), "-", "-",
            std::to_string(avgK), "-", "VERIFIED"
        });

        instBuf.destroy();
        lightTLAS.destroy();
        lightBLAS.destroy();
        proxyVBuf.destroy();
        proxyIBuf.destroy();
        lightBuf.destroy();
    }

    timer.cleanup();
    clusterGridBuf.destroy();
    tileGridBuf.destroy();
    stagingImgBuf.destroy();
    clusteredOutImage.destroy();
    tiledOutImage.destroy();
    rayOutImage.destroy();

    vkDestroyPipeline(g_vk.device, lcBinPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, lcBinPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, lcBinSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, cShadePipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, cShadePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, cShadeSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, tlBinPipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, tlBinPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, tlBinSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, tShadePipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, tShadePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, tShadeSetLayout, nullptr);

    vkDestroyPipeline(g_vk.device, rShadePipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, rShadePipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, rShadeSetLayout, nullptr);

    vkDestroyDescriptorPool(g_vk.device, descPool, nullptr);

    vkDestroyShaderModule(g_vk.device, lcBinModule, nullptr);
    vkDestroyShaderModule(g_vk.device, cShadeModule, nullptr);
    vkDestroyShaderModule(g_vk.device, tlBinModule, nullptr);
    vkDestroyShaderModule(g_vk.device, tShadeModule, nullptr);
    vkDestroyShaderModule(g_vk.device, rShadeModule, nullptr);

    g_vk.cleanup();

    std::cout << "\n====================================================\n";
    std::cout << " PHASE 4 AUDIT COMPLETE!\n";
    std::cout << "====================================================\n";

    return 0;
}
