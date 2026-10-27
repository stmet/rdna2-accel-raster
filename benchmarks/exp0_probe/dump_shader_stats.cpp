#include "vk_context.hpp"
#include "vk_shader.hpp"
#include <iostream>
#include <vector>

int main() {
    g_vk.init(true);

    std::vector<std::pair<std::string, std::string>> shaders = {
        {"Hardware_RA_Procedural", "exp0_probe/shaders/probe_procedural.comp"},
        {"Software_BVH_PreAudit", "exp0_probe/shaders/software_bvh.comp"},
        {"Software_BVH_Procedural_Scratch", "exp0_probe/shaders/software_bvh_procedural.comp"},
        {"Software_BVH_Procedural_LDS", "exp0_probe/shaders/software_bvh_lds.comp"},
        {"Software_BVH_Procedural_BVH4", "exp0_probe/shaders/software_bvh_bvh4.comp"}
    };

    for (const auto& [name, path] : shaders) {
        std::cerr << "\n========================================\n";
        std::cerr << ">>> COMPILING SHADER: " << name << "\n";
        std::cerr << "========================================\n";

        std::vector<uint32_t> spv = ShaderUtils::compileGlslFileToSpirv(path);
        VkShaderModule mod = ShaderUtils::createShaderModule(spv);

        VkDescriptorSetLayoutBinding b[4]{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
        };
        if (name == "Hardware_RA_Procedural") {
            b[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        }

        VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 4, b};
        VkDescriptorSetLayout dsl;
        vkCreateDescriptorSetLayout(g_vk.device, &dslInfo, nullptr, &dsl);

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl, 1, &pcr};
        VkPipelineLayout layout;
        vkCreatePipelineLayout(g_vk.device, &plInfo, nullptr, &layout);

        VkPipeline pipe = ShaderUtils::createComputePipeline(mod, layout);

        vkDestroyPipeline(g_vk.device, pipe, nullptr);
        vkDestroyPipelineLayout(g_vk.device, layout, nullptr);
        vkDestroyDescriptorSetLayout(g_vk.device, dsl, nullptr);
        vkDestroyShaderModule(g_vk.device, mod, nullptr);
    }

    g_vk.cleanup();
    return 0;
}
