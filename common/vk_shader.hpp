#pragma once

#include "vk_context.hpp"
#include <string>
#include <vector>

class ShaderUtils {
public:
    static std::vector<uint32_t> loadSpirvFile(const std::string& filepath);
    static std::vector<uint32_t> compileGlslToSpirv(
        const std::string& glslSource,
        const std::string& stage = "comp",
        const std::vector<std::string>& definitions = {}
    );
    static std::vector<uint32_t> compileGlslFileToSpirv(
        const std::string& glslPath,
        const std::vector<std::string>& definitions = {}
    );
    static VkShaderModule createShaderModule(const std::vector<uint32_t>& spirv);

    static VkPipeline createComputePipeline(
        VkShaderModule computeShader,
        VkPipelineLayout layout
    );

    static VkPipeline createGraphicsPipeline(
        VkShaderModule vertShader,
        VkShaderModule fragShader,
        VkPipelineLayout layout,
        const VkPipelineRenderingCreateInfo& renderingInfo,
        uint32_t vertexStride,
        bool depthTest = true,
        bool depthWrite = true,
        uint32_t colorAttachmentCount = 1,
        VkCullModeFlags cullMode = VK_CULL_MODE_NONE
    );
};
