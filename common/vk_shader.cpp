#include "vk_shader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

std::vector<uint32_t> ShaderUtils::loadSpirvFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V file: " + filepath);
    }

    size_t fileSize = (size_t)file.tellg();
    if (fileSize % 4 != 0) {
        throw std::runtime_error("SPIR-V file size is not a multiple of 4: " + filepath);
    }

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
}

std::vector<uint32_t> ShaderUtils::compileGlslToSpirv(
    const std::string& glslSource,
    const std::string& stage,
    const std::vector<std::string>& definitions
) {
    char tmpSrc[] = "/tmp/shader_XXXXXX";
    int fd = mkstemp(tmpSrc);
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary shader file");
    }
    write(fd, glslSource.c_str(), glslSource.size());
    close(fd);

    std::string renamedSrc = std::string(tmpSrc) + "." + stage;
    rename(tmpSrc, renamedSrc.c_str());

    std::string tmpSpv = renamedSrc + ".spv";

    std::stringstream cmd;
    cmd << "glslangValidator --target-env vulkan1.3 -V ";
    for (const auto& d : definitions) {
        cmd << "-D" << d << " ";
    }
    cmd << renamedSrc << " -o " << tmpSpv << " > /tmp/glslang_out.txt 2>&1";

    int ret = system(cmd.str().c_str());
    if (ret != 0) {
        std::ifstream out("/tmp/glslang_out.txt");
        std::string errStr((std::istreambuf_iterator<char>(out)), std::istreambuf_iterator<char>());
        unlink(renamedSrc.c_str());
        throw std::runtime_error("Shader compilation failed:\n" + errStr);
    }

    std::vector<uint32_t> spirv = loadSpirvFile(tmpSpv);

    unlink(renamedSrc.c_str());
    unlink(tmpSpv.c_str());

    return spirv;
}

std::vector<uint32_t> ShaderUtils::compileGlslFileToSpirv(
    const std::string& glslPath,
    const std::vector<std::string>& definitions
) {
    std::ifstream file(glslPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + glslPath);
    }
    std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Deduce stage from extension
    std::string stage = "comp";
    size_t dotPos = glslPath.rfind('.');
    if (dotPos != std::string::npos) {
        stage = glslPath.substr(dotPos + 1);
    }

    return compileGlslToSpirv(source, stage, definitions);
}

VkShaderModule ShaderUtils::createShaderModule(const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();

    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(g_vk.device, &createInfo, nullptr, &shaderModule));
    return shaderModule;
}

VkPipeline ShaderUtils::createComputePipeline(
    VkShaderModule computeShader,
    VkPipelineLayout layout
) {
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = computeShader;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = layout;

    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(g_vk.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
    return pipeline;
}

VkPipeline ShaderUtils::createGraphicsPipeline(
    VkShaderModule vertShader,
    VkShaderModule fragShader,
    VkPipelineLayout layout,
    const VkPipelineRenderingCreateInfo& renderingInfo,
    uint32_t vertexStride,
    bool depthTest,
    bool depthWrite,
    uint32_t colorAttachmentCount,
    VkCullModeFlags cullMode
) {
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    if (vertShader != VK_NULL_HANDLE) {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage = VK_SHADER_STAGE_VERTEX_BIT;
        s.module = vertShader;
        s.pName = "main";
        stages.push_back(s);
    }
    if (fragShader != VK_NULL_HANDLE) {
        VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        s.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        s.module = fragShader;
        s.pName = "main";
        stages.push_back(s);
    }

    VkVertexInputBindingDescription bDesc{0, vertexStride, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription aDescs[2]{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3}
    };
    VkPipelineVertexInputStateCreateInfo viInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    if (vertexStride > 0) {
        viInfo.vertexBindingDescriptionCount = 1;
        viInfo.pVertexBindingDescriptions = &bDesc;
        viInfo.vertexAttributeDescriptionCount = 2;
        viInfo.pVertexAttributeDescriptions = aDescs;
    }

    VkPipelineInputAssemblyStateCreateInfo iaInfo{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    iaInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vpsInfo{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpsInfo.viewportCount = 1;
    vpsInfo.scissorCount = 1;

    VkDynamicState dynStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynInfo{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynInfo.dynamicStateCount = 2;
    dynInfo.pDynamicStates = dynStates;

    VkPipelineRasterizationStateCreateInfo rsInfo{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rsInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rsInfo.cullMode = cullMode;
    rsInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsInfo.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo msInfo{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dsInfo{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dsInfo.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
    dsInfo.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
    dsInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    std::vector<VkPipelineColorBlendAttachmentState> cbAtts(colorAttachmentCount);
    for (uint32_t i = 0; i < colorAttachmentCount; i++) {
        cbAtts[i].colorWriteMask = 0xF;
        cbAtts[i].blendEnable = VK_FALSE;
    }
    VkPipelineColorBlendStateCreateInfo cbInfo{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbInfo.attachmentCount = colorAttachmentCount;
    cbInfo.pAttachments = cbAtts.data();

    VkGraphicsPipelineCreateInfo gpInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpInfo.pNext = &renderingInfo;
    gpInfo.stageCount = static_cast<uint32_t>(stages.size());
    gpInfo.pStages = stages.data();
    gpInfo.pVertexInputState = &viInfo;
    gpInfo.pInputAssemblyState = &iaInfo;
    gpInfo.pViewportState = &vpsInfo;
    gpInfo.pRasterizationState = &rsInfo;
    gpInfo.pMultisampleState = &msInfo;
    gpInfo.pDepthStencilState = &dsInfo;
    gpInfo.pColorBlendState = &cbInfo;
    gpInfo.pDynamicState = &dynInfo;
    gpInfo.layout = layout;

    VkPipeline pipeline;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &gpInfo, nullptr, &pipeline));
    return pipeline;
}
