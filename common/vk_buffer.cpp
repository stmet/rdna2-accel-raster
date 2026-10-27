#include "vk_buffer.hpp"
#include <stdexcept>
#include <iostream>

void VulkanBuffer::create(VkDeviceSize bufferSize, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    destroy();
    size = bufferSize;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(g_vk.device, &bufferInfo, nullptr, &handle));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(g_vk.device, handle, &memRequirements);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = g_vk.findMemoryType(memRequirements.memoryTypeBits, properties);

    VkMemoryAllocateFlagsInfo allocFlagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        allocInfo.pNext = &allocFlagsInfo;
    }

    VK_CHECK(vkAllocateMemory(g_vk.device, &allocInfo, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(g_vk.device, handle, memory, 0));
}

void VulkanBuffer::destroy() {
    if (mapped) {
        unmap();
    }
    if (handle != VK_NULL_HANDLE) {
        vkDestroyBuffer(g_vk.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(g_vk.device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
    size = 0;
    deviceAddress = 0;
}

void* VulkanBuffer::map() {
    if (!mapped) {
        VK_CHECK(vkMapMemory(g_vk.device, memory, 0, size, 0, &mapped));
    }
    return mapped;
}

void VulkanBuffer::unmap() {
    if (mapped) {
        vkUnmapMemory(g_vk.device, memory);
        mapped = nullptr;
    }
}

void VulkanBuffer::upload(const void* data, VkDeviceSize uploadSize) {
    if (uploadSize > size) {
        throw std::runtime_error("uploadSize exceeds buffer size!");
    }
    map();
    std::memcpy(mapped, data, uploadSize);
}

void VulkanBuffer::download(void* data, VkDeviceSize downloadSize) {
    if (downloadSize > size) {
        throw std::runtime_error("downloadSize exceeds buffer size!");
    }
    map();
    std::memcpy(data, mapped, downloadSize);
}

VkDeviceAddress VulkanBuffer::getDeviceAddress() {
    if (deviceAddress == 0 && handle != VK_NULL_HANDLE) {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = handle;
        if (g_vk.vkGetBufferDeviceAddressKHR) {
            deviceAddress = g_vk.vkGetBufferDeviceAddressKHR(g_vk.device, &info);
        } else {
            deviceAddress = vkGetBufferDeviceAddress(g_vk.device, &info);
        }
    }
    return deviceAddress;
}
