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

struct alignas(16) PushConstantsReflectionRaster {
    glm::mat4 reflectedViewProj;
    glm::vec4 clipPlane;
};

struct alignas(16) PushConstantsReflectionRQ {
    glm::mat4 invViewProj;
    glm::vec4 camPos;
    glm::vec4 mirrorPlane;
    glm::uvec2 resolution;
    float coverageFraction;
    float maxRayDist;
};

struct alignas(16) PushConstantsComposite {
    glm::uvec2 resolution;
    float coverageFraction;
};

int main(int argc, char** argv) {
    std::cout << "====================================================\n";
    std::cout << " EXPERIMENT C: PLANAR REFLECTIONS / MIRRORS (SWEEP)\n";
    std::cout << "====================================================\n";

    g_vk.init(true);
    GpuClockInfo::printStatus();

    const uint32_t width = 1920;
    const uint32_t height = 1080;
    const uint32_t warmupIters = 10;
    const uint32_t benchIters = 200;

    std::vector<uint32_t> triCounts = { 100000, 500000, 1000000, 2500000 };
    std::vector<float> coverages = { 0.02f, 0.05f, 0.15f, 0.30f, 0.60f, 1.00f };

    std::cout << "[Exp C] Compiling reflection shaders...\n";
    auto spvReflV = ShaderUtils::compileGlslFileToSpirv("expC_reflections/shaders/reflection_raster.vert");
    auto spvReflF = ShaderUtils::compileGlslFileToSpirv("expC_reflections/shaders/reflection_raster.frag");
    auto spvReflRQ = ShaderUtils::compileGlslFileToSpirv("expC_reflections/shaders/reflection_rayquery.comp");
    auto spvComp = ShaderUtils::compileGlslFileToSpirv("expC_reflections/shaders/composite_reflection.comp");

    VkShaderModule modReflV = ShaderUtils::createShaderModule(spvReflV);
    VkShaderModule modReflF = ShaderUtils::createShaderModule(spvReflF);
    VkShaderModule modReflRQ = ShaderUtils::createShaderModule(spvReflRQ);
    VkShaderModule modComp = ShaderUtils::createShaderModule(spvComp);

    // Raster Reflection Pipeline
    VkPushConstantRange pcReflRange{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsReflectionRaster)};
    VkPipelineLayoutCreateInfo plReflInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plReflInfo.pushConstantRangeCount = 1;
    plReflInfo.pPushConstantRanges = &pcReflRange;
    VkPipelineLayout plReflRaster;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plReflInfo, nullptr, &plReflRaster));

    VkFormat colFmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depthFmt = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo rendInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendInfo.colorAttachmentCount = 1;
    rendInfo.pColorAttachmentFormats = &colFmt;
    rendInfo.depthAttachmentFormat = depthFmt;

    VkPipeline pipeReflRaster = ShaderUtils::createGraphicsPipeline(
        modReflV, modReflF, plReflRaster, rendInfo, sizeof(SceneMesh::Vertex), true, true, 1
    );

    // Reflection targets
    VulkanImage rasterReflColor, rasterReflDepth;
    rasterReflColor.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    rasterReflDepth.create(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

    rasterReflColor.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    rasterReflDepth.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    // Primary scene depth buffer & output images
    VulkanImage primaryDepth, outRqColor, outRasterComposite;
    primaryDepth.create(width, height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
    outRqColor.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    outRasterComposite.create(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

    primaryDepth.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    outRqColor.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    outRasterComposite.transitionLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    VkSampler linearSampler = VulkanImage::createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // RQ Reflection Pipeline
    VkDescriptorSetLayoutBinding rqBindings[3] = {
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslRqInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 3, rqBindings};
    VkDescriptorSetLayout dslRq;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslRqInfo, nullptr, &dslRq));

    VkPushConstantRange pcRqRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsReflectionRQ)};
    VkPipelineLayoutCreateInfo plRqInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plRqInfo.setLayoutCount = 1;
    plRqInfo.pSetLayouts = &dslRq;
    plRqInfo.pushConstantRangeCount = 1;
    plRqInfo.pPushConstantRanges = &pcRqRange;
    VkPipelineLayout plRq;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plRqInfo, nullptr, &plRq));
    VkPipeline pipeRq = ShaderUtils::createComputePipeline(modReflRQ, plRq);

    // Raster Composite Pipeline
    VkDescriptorSetLayoutBinding compBindings[2] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };
    VkDescriptorSetLayoutCreateInfo dslCompInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, compBindings};
    VkDescriptorSetLayout dslComp;
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.device, &dslCompInfo, nullptr, &dslComp));

    VkPushConstantRange pcCompRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsComposite)};
    VkPipelineLayoutCreateInfo plCompInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCompInfo.setLayoutCount = 1;
    plCompInfo.pSetLayouts = &dslComp;
    plCompInfo.pushConstantRangeCount = 1;
    plCompInfo.pPushConstantRanges = &pcCompRange;
    VkPipelineLayout plComp;
    VK_CHECK(vkCreatePipelineLayout(g_vk.device, &plCompInfo, nullptr, &plComp));
    VkPipeline pipeComp = ShaderUtils::createComputePipeline(modComp, plComp);

    // Camera setup
    glm::vec3 camPos(0.0f, 20.0f, -40.0f);
    glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), float(width) / float(height), 0.1f, 500.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 viewProj = proj * view;
    glm::mat4 invViewProj = glm::inverse(viewProj);

    // Horizontal mirror plane at y = 0
    glm::vec4 mirrorPlane(0.0f, 1.0f, 0.0f, 0.0f);
    glm::vec3 reflCamPos = camPos - 2.0f * (glm::dot(glm::vec3(mirrorPlane), camPos) + mirrorPlane.w) * glm::vec3(mirrorPlane);
    glm::mat4 reflView = glm::lookAt(reflCamPos, glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(0.0f, -1.0f, 0.0f));
    glm::mat4 reflViewProj = proj * reflView;

    // Output CSV
    std::filesystem::create_directories("results");
    CsvWriter csv("results/expC_reflections.csv", {
        "Triangles", "ScreenCoveragePct", "RasterSecondPassMs", "RasterCompositeMs", "RasterTotalMs",
        "RayQueryReflectionMs", "SpeedupRQvsRaster", "Winner", "PSNR_dB", "SSIM", "p5Ms", "p95Ms"
    });

    GpuTimer timer;
    timer.init(64);

    for (uint32_t tris : triCounts) {
        std::cout << "\n----------------------------------------------------\n";
        std::cout << "[Exp C] Generating Scene with " << tris << " triangles...\n";
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

        // Pre-render depth buffer
        {
            VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
            VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, primaryDepth.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
            dAtt.clearValue.depthStencil.depth = 1.0f;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 0, nullptr, &dAtt, nullptr};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeReflRaster);
            VkViewport vp{0, 0, (float)width, (float)height, 0, 1};
            VkRect2D sc{{0, 0}, {width, height}};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);

            PushConstantsReflectionRaster pc{viewProj, mirrorPlane};
            vkCmdPushConstants(cmd, plReflRaster, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsReflectionRaster), &pc);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
            vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
            vkCmdEndRendering(cmd);

            primaryDepth.cmdTransition(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            g_vk.endSingleTimeCommands(cmd);
        }

        // Measure Second Raster Pass (full scene render from reflected camera)
        auto recordRasterReflection = [&](VkCommandBuffer cmd) {
            rasterReflColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

            VkRenderingAttachmentInfo colAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterReflColor.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
            VkRenderingAttachmentInfo dAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, nullptr, rasterReflDepth.view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_RESOLVE_MODE_NONE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, {}};
            dAtt.clearValue.depthStencil.depth = 1.0f;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO, nullptr, 0, {{0,0}, {width, height}}, 1, 0, 1, &colAtt, &dAtt, nullptr};

            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeReflRaster);
            VkViewport vp{0, 0, (float)width, (float)height, 0, 1};
            VkRect2D sc{{0, 0}, {width, height}};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);

            PushConstantsReflectionRaster pc{reflViewProj, mirrorPlane};
            vkCmdPushConstants(cmd, plReflRaster, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantsReflectionRaster), &pc);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vBuf.handle, &off);
            vkCmdBindIndexBuffer(cmd, iBuf.handle, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, triCount * 3, 1, 0, 0, 0);
            vkCmdEndRendering(cmd);

            rasterReflColor.cmdTransition(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        };

        BenchmarkStats statsRasterPass = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
            recordRasterReflection(cmd);
        });

        // Allocate descriptor set for RQ
        VkDescriptorPoolSize poolSizes[3] = {
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8 }
        };
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 4, 3, poolSizes};
        VkDescriptorPool descPool;
        VK_CHECK(vkCreateDescriptorPool(g_vk.device, &poolInfo, nullptr, &descPool));

        VkDescriptorSet setRq, setComp;
        VkDescriptorSetAllocateInfo allocRq{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslRq};
        VkDescriptorSetAllocateInfo allocComp{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descPool, 1, &dslComp};
        VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocRq, &setRq));
        VK_CHECK(vkAllocateDescriptorSets(g_vk.device, &allocComp, &setComp));

        // Update RQ Set
        VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        asInfo.accelerationStructureCount = 1;
        VkAccelerationStructureKHR tHandle = tlas.handle;
        asInfo.pAccelerationStructures = &tHandle;
        VkDescriptorImageInfo dImgInfo{linearSampler, primaryDepth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo outRqInfo{VK_NULL_HANDLE, outRqColor.view, VK_IMAGE_LAYOUT_GENERAL};

        VkWriteDescriptorSet rqWrites[3]{};
        rqWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, setRq, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr};
        rqWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRq, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dImgInfo, nullptr, nullptr};
        rqWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setRq, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outRqInfo, nullptr, nullptr};
        vkUpdateDescriptorSets(g_vk.device, 3, rqWrites, 0, nullptr);

        // Update Comp Set
        VkDescriptorImageInfo reflTexInfo{linearSampler, rasterReflColor.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo outCompInfo{VK_NULL_HANDLE, outRasterComposite.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet compWrites[2]{};
        compWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setComp, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &reflTexInfo, nullptr, nullptr};
        compWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, setComp, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outCompInfo, nullptr, nullptr};
        vkUpdateDescriptorSets(g_vk.device, 2, compWrites, 0, nullptr);

        // Sweep reflector screen coverage
        for (float cov : coverages) {
            std::cout << "[Exp C] Mirror Screen Coverage: " << (cov * 100.0f) << "% (" << tris << " tris)\n";

            PushConstantsComposite pcComp{glm::uvec2(width, height), cov};
            BenchmarkStats statsComp = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeComp);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plComp, 0, 1, &setComp, 0, nullptr);
                vkCmdPushConstants(cmd, plComp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsComposite), &pcComp);
                vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
            });

            float rasterTotalMs = statsRasterPass.medianMs + statsComp.medianMs;

            PushConstantsReflectionRQ pcRq{invViewProj, glm::vec4(camPos, 1.0f), mirrorPlane, glm::uvec2(width, height), cov, 500.0f};
            BenchmarkStats statsRq = timer.runBenchmark(warmupIters, benchIters, [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeRq);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plRq, 0, 1, &setRq, 0, nullptr);
                vkCmdPushConstants(cmd, plRq, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsReflectionRQ), &pcRq);
                vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
            });

            float speedup = rasterTotalMs / statsRq.medianMs;
            std::string winner = (speedup > 1.0f) ? "RayQuery" : "RasterSecondPass";

            // Image quality diff
            std::vector<uint8_t> compPixels(width * height * 4);
            std::vector<uint8_t> rqPixels(width * height * 4);
            {
                VulkanBuffer readBufComp, readBufRq;
                readBufComp.create(width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                readBufRq.create(width * height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

                VkCommandBuffer cmd = g_vk.beginSingleTimeCommands();
                VkBufferImageCopy regionComp{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {width, height, 1}};
                VkBufferImageCopy regionRq{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {width, height, 1}};
                vkCmdCopyImageToBuffer(cmd, outRasterComposite.handle, VK_IMAGE_LAYOUT_GENERAL, readBufComp.handle, 1, &regionComp);
                vkCmdCopyImageToBuffer(cmd, outRqColor.handle, VK_IMAGE_LAYOUT_GENERAL, readBufRq.handle, 1, &regionRq);
                g_vk.endSingleTimeCommands(cmd);

                readBufComp.download(compPixels.data(), compPixels.size());
                readBufRq.download(rqPixels.data(), rqPixels.size());
                readBufComp.destroy();
                readBufRq.destroy();
            }

            std::vector<float> compFloats(width * height);
            std::vector<float> rqFloats(width * height);
            for (size_t p = 0; p < width * height; p++) {
                compFloats[p] = static_cast<float>(compPixels[p * 4 + 0]) / 255.0f;
                rqFloats[p] = static_cast<float>(rqPixels[p * 4 + 0]) / 255.0f;
            }
            ImageQualityMetrics qual = ImageMetrics::compute(compFloats, rqFloats, width, height);

            std::cout << "  - Raster 2nd Pass: " << statsRasterPass.medianMs << " ms + Comp: " << statsComp.medianMs << " ms = " << rasterTotalMs << " ms\n";
            std::cout << "  - Ray Query Reflection: " << statsRq.medianMs << " ms (Speedup: " << std::fixed << std::setprecision(2) << speedup << "x) -> Winner: " << winner << "\n";
            std::cout << "  - Quality Diff: PSNR = " << qual.psnr << " dB, SSIM = " << qual.ssim << "\n";

            csv.writeRowValues(
                tris, cov * 100.0f, statsRasterPass.medianMs, statsComp.medianMs, rasterTotalMs,
                statsRq.medianMs, speedup, winner, qual.psnr, qual.ssim, statsRq.p5Ms, statsRq.p95Ms
            );
        }

        vkDestroyDescriptorPool(g_vk.device, descPool, nullptr);
        blas.destroy();
        tlas.destroy();
        instBuf.destroy();
        vBuf.destroy();
        iBuf.destroy();
    }

    std::cout << "\n[Exp C] Results successfully saved to results/expC_reflections.csv\n";
    GpuClockInfo::printStatus();

    // Cleanup
    primaryDepth.destroy();
    rasterReflColor.destroy();
    rasterReflDepth.destroy();
    outRqColor.destroy();
    outRasterComposite.destroy();
    vkDestroySampler(g_vk.device, linearSampler, nullptr);

    vkDestroyPipeline(g_vk.device, pipeReflRaster, nullptr);
    vkDestroyPipeline(g_vk.device, pipeRq, nullptr);
    vkDestroyPipeline(g_vk.device, pipeComp, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plReflRaster, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plRq, nullptr);
    vkDestroyPipelineLayout(g_vk.device, plComp, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslRq, nullptr);
    vkDestroyDescriptorSetLayout(g_vk.device, dslComp, nullptr);
    vkDestroyShaderModule(g_vk.device, modReflV, nullptr);
    vkDestroyShaderModule(g_vk.device, modReflF, nullptr);
    vkDestroyShaderModule(g_vk.device, modReflRQ, nullptr);
    vkDestroyShaderModule(g_vk.device, modComp, nullptr);

    g_vk.cleanup();
    return 0;
}
