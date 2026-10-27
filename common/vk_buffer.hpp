#pragma once

#include "vk_context.hpp"
#include <cstring>

struct VulkanBuffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress deviceAddress = 0;
    void* mapped = nullptr;

    void create(VkDeviceSize bufferSize, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    void destroy();
    void* map();
    void unmap();
    void upload(const void* data, VkDeviceSize uploadSize);
    void download(void* data, VkDeviceSize downloadSize);
    VkDeviceAddress getDeviceAddress();
};
