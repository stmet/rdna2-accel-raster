#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <iostream>

#define VK_CHECK(x)                                                                           \
    do {                                                                                      \
        VkResult err = (x);                                                                   \
        if (err != VK_SUCCESS) {                                                              \
            std::cerr << "Vulkan error " << err << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            throw std::runtime_error("Vulkan call failed: " #x);                              \
        }                                                                                     \
    } while (0)

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    
    // Primary / Graphics queue
    VkQueue queue = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    uint32_t graphicsQueueFamilyIndex = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandPool graphicsCommandPool = VK_NULL_HANDLE;

    // Async Compute queue
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex = 0;
    VkCommandPool computeCommandPool = VK_NULL_HANDLE;
    bool hasDedicatedCompute = false;

    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{};
    float timestampPeriod = 1.0f; // in nanoseconds

    // Extension functions
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCmdCopyAccelerationStructureKHR vkCmdCopyAccelerationStructureKHR = nullptr;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR vkCmdWriteAccelerationStructuresPropertiesKHR = nullptr;
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR = nullptr;
    PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT = nullptr;

    void init(bool headless = true);
    void cleanup();

    VkCommandBuffer beginSingleTimeCommands() const;
    void endSingleTimeCommands(VkCommandBuffer cmd) const;

    VkCommandBuffer beginSingleTimeComputeCommands() const;
    void endSingleTimeComputeCommands(VkCommandBuffer cmd) const;

    VkSemaphore createTimelineSemaphore(uint64_t initialValue = 0) const;
    VkSemaphore createBinarySemaphore() const;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
};

extern VulkanContext g_vk;
