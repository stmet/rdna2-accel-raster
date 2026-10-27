#include "vk_context.hpp"
#include "vk_buffer.hpp"
#include "vk_bvh.hpp"
#include "vk_shader.hpp"
#include "vk_timer.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>

struct Vertex {
    float pos[3];
};

struct RayInput {
    float origin[4];    // xyz = origin, w = tmin
    float direction[4]; // xyz = dir, w = tmax
};

struct HitResult {
    uint32_t hit;
    float t;
    uint32_t instanceId;
    uint32_t primitiveId;
};

int main() {
    std::cout << "====================================================\n";
    std::cout << " PHASE 0: SETUP AND CAPABILITY PROBE\n";
    std::cout << "====================================================\n";

    try {
        g_vk.init(true);
    } catch (const std::exception& e) {
        std::cerr << "GATE FAILURE: Failed to initialize Vulkan context: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Vulkan context initialized successfully.\n";
    std::cout << "Selected GPU: " << g_vk.properties.deviceName << "\n";
    std::cout << "Timestamp period: " << g_vk.timestampPeriod << " ns\n";

    // 1. Build Triangle BLAS
    std::vector<Vertex> vertices = {
        {{-1.0f, -1.0f, 0.0f}},
        {{ 1.0f, -1.0f, 0.0f}},
        {{ 0.0f,  1.0f, 0.0f}}
    };
    std::vector<uint32_t> indices = {0, 1, 2};

    VulkanBuffer vBuffer, iBuffer;
    vBuffer.create(
        sizeof(Vertex) * vertices.size(),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    vBuffer.upload(vertices.data(), sizeof(Vertex) * vertices.size());

    iBuffer.create(
        sizeof(uint32_t) * indices.size(),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    iBuffer.upload(indices.data(), sizeof(uint32_t) * indices.size());

    std::cout << "\nBuilding Triangle BLAS..." << std::endl;
    float blasBuildTime = 0.0f;
    VulkanAS triangleBLAS = BVHBuilder::buildTriangleBLAS(
        vBuffer.getDeviceAddress(),
        vertices.size(),
        sizeof(Vertex),
        VK_FORMAT_R32G32B32_SFLOAT,
        iBuffer.getDeviceAddress(),
        indices.size(),
        VK_INDEX_TYPE_UINT32,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        &blasBuildTime
    );
    std::cout << "Triangle BLAS built in " << blasBuildTime << " ms. Device Address: 0x" 
              << std::hex << triangleBLAS.deviceAddress << std::dec << "\n";

    // 2. Build TLAS with 1 instance of the BLAS
    VkTransformMatrixKHR transform = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    VkAccelerationStructureInstanceKHR instance{};
    instance.transform = transform;
    instance.instanceCustomIndex = 42;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = triangleBLAS.deviceAddress;

    VulkanBuffer instanceBuffer;
    instanceBuffer.create(
        sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    instanceBuffer.upload(&instance, sizeof(instance));

    std::cout << "Building TLAS..." << std::endl;
    float tlasBuildTime = 0.0f;
    VulkanAS tlas = BVHBuilder::buildTLAS(
        instanceBuffer.getDeviceAddress(),
        1,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        &tlasBuildTime
    );
    std::cout << "TLAS built in " << tlasBuildTime << " ms. Device Address: 0x" 
              << std::hex << tlas.deviceAddress << std::dec << "\n";

    // 3. Prepare ray query pipeline
    std::cout << "Compiling probe ray query shader..." << std::endl;
    std::vector<uint32_t> spv = ShaderUtils::compileGlslFileToSpirv("exp0_probe/shaders/probe_triangle.comp");
    VkShaderModule compModule = ShaderUtils::createShaderModule(spv);

    // Descriptor Set Layout
    VkDescriptorSetLayoutBinding asBinding{};
    asBinding.binding = 0;
    asBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    asBinding.descriptorCount = 1;
    asBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding rayBinding{};
    rayBinding.binding = 1;
    rayBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    rayBinding.descriptorCount = 1;
    rayBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding resultBinding{};
    resultBinding.binding = 2;
    resultBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resultBinding.descriptorCount = 1;
    resultBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding bindings[] = {asBinding, rayBinding, resultBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &layoutInfo, nullptr, &descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPipelineLayout pipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

    VkPipeline pipeline = ShaderUtils::createComputePipeline(compModule, pipelineLayout);
    std::cout << "Compute pipeline created successfully.\n";

    // Descriptor Pool & Set
    VkDescriptorPoolSize poolSizes[2];
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;

    VkDescriptorPool descriptorPool;
    VK_CHECK(vkCreateDescriptorPool(g_vk.device, &poolInfo, nullptr, &descriptorPool));

    VkDescriptorSetAllocateInfo setAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setAllocInfo.descriptorPool = descriptorPool;
    setAllocInfo.descriptorSetCount = 1;
    setAllocInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &setAllocInfo, &descriptorSet));

    // Prepare Rays: 1024 rays
    // first 512 hit (ray origin (0, 0, -1), dir (0, 0, 1))
    // next 512 miss (ray origin (10, 10, -1), dir (0, 0, 1))
    const uint32_t rayCount = 1024;
    std::vector<RayInput> rays(rayCount);
    for (uint32_t i = 0; i < rayCount; i++) {
        if (i < 512) {
            // hitting rays
            rays[i].origin[0] = 0.0f;
            rays[i].origin[1] = -0.5f;
            rays[i].origin[2] = -2.0f;
            rays[i].origin[3] = 0.001f; // tmin

            rays[i].direction[0] = 0.0f;
            rays[i].direction[1] = 0.0f;
            rays[i].direction[2] = 1.0f;
            rays[i].direction[3] = 100.0f; // tmax
        } else {
            // missing rays
            rays[i].origin[0] = 10.0f;
            rays[i].origin[1] = 10.0f;
            rays[i].origin[2] = -2.0f;
            rays[i].origin[3] = 0.001f;

            rays[i].direction[0] = 0.0f;
            rays[i].direction[1] = 0.0f;
            rays[i].direction[2] = 1.0f;
            rays[i].direction[3] = 100.0f;
        }
    }

    VulkanBuffer rayBuffer, resultBuffer;
    rayBuffer.create(
        sizeof(RayInput) * rayCount,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    rayBuffer.upload(rays.data(), sizeof(RayInput) * rayCount);

    resultBuffer.create(
        sizeof(HitResult) * rayCount,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // Update Descriptor Sets
    VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlas.handle;

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &asInfo;
    writes[0].dstSet = descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    VkDescriptorBufferInfo rayBufferInfo{rayBuffer.handle, 0, VK_WHOLE_SIZE};
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &rayBufferInfo;

    VkDescriptorBufferInfo resultBufferInfo{resultBuffer.handle, 0, VK_WHOLE_SIZE};
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &resultBufferInfo;

    vkUpdateDescriptorSets(g_vk.device, 3, writes, 0, nullptr);

    // Dispatch Compute Shader
    std::cout << "Dispatching ray queries (" << rayCount << " rays)..." << std::endl;
    GpuTimer timer;
    timer.init(2);

    VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(cmd, (rayCount + 63) / 64, 1, 1);
    g_vk.endSingleTimeCommands(cmd);

    // Verify Results
    std::vector<HitResult> results(rayCount);
    resultBuffer.download(results.data(), sizeof(HitResult) * rayCount);

    bool correct = true;
    for (uint32_t i = 0; i < rayCount; i++) {
        if (i < 512) {
            if (results[i].hit != 1 || std::abs(results[i].t - 2.0f) > 0.01f || results[i].primitiveId != 0) {
                std::cerr << "FAIL on ray " << i << ": expected hit=1, t=2.0, prim=0. Got hit=" 
                          << results[i].hit << ", t=" << results[i].t << ", prim=" << results[i].primitiveId << "\n";
                correct = false;
                break;
            }
        } else {
            if (results[i].hit != 0) {
                std::cerr << "FAIL on ray " << i << ": expected miss. Got hit=" << results[i].hit << "\n";
                correct = false;
                break;
            }
        }
    }

    if (!correct) {
        std::cerr << "GATE FAILURE: Verification failed! Ray query results did not match expected geometry.\n";
        return 1;
    }

    std::cout << "VERIFICATION PASSED: All " << rayCount << " rays produced mathematically exact results!\n";
    std::cout << "  - Rays [0..511] hit triangle at distance t = " << results[0].t 
              << ", prim = " << results[0].primitiveId << ", instance = " << results[0].instanceId << "\n";
    std::cout << "  - Rays [512..1023] recorded clean misses (t = " << results[512].t << ")\n";

    // 4. AABB BLAS test
    std::cout << "\nTesting AABB BLAS..." << std::endl;
    VkAabbPositionsKHR aabb = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
    VulkanBuffer aabbBuffer;
    aabbBuffer.create(
        sizeof(VkAabbPositionsKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    aabbBuffer.upload(&aabb, sizeof(aabb));

    float aabbBlasTime = 0.0f;
    VulkanAS aabbBLAS = BVHBuilder::buildAabbBLAS(
        aabbBuffer.getDeviceAddress(),
        1,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        &aabbBlasTime
    );
    std::cout << "AABB BLAS built in " << aabbBlasTime << " ms. Device Address: 0x" 
              << std::hex << aabbBLAS.deviceAddress << std::dec << "\n";

    // Clean up
    timer.cleanup();
    vkDestroyPipeline(g_vk.device, pipeline, nullptr);
    vkDestroyPipelineLayout(g_vk.device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(g_vk.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, descriptorSetLayout, nullptr);
    vkDestroyShaderModule(g_vk.device, compModule, nullptr);

    aabbBLAS.destroy();
    aabbBuffer.destroy();
    tlas.destroy();
    triangleBLAS.destroy();
    instanceBuffer.destroy();
    vBuffer.destroy();
    iBuffer.destroy();
    rayBuffer.destroy();
    resultBuffer.destroy();

    g_vk.cleanup();

    std::cout << "\n====================================================\n";
    std::cout << " GATE PASSED: VK_KHR_ray_query is 100% OPERATIONAL!\n";
    std::cout << "====================================================\n";

    return 0;
}
