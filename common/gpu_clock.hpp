#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

struct GpuClockInfo {
    std::string sclkString;
    uint32_t currentMhz = 0;
    uint32_t maxMhz = 0;

    static GpuClockInfo query() {
        GpuClockInfo info;
        std::ifstream file("/sys/class/drm/card1/device/pp_dpm_sclk");
        if (!file.is_open()) {
            file.open("/sys/class/drm/card0/device/pp_dpm_sclk");
        }
        if (!file.is_open()) {
            info.sclkString = "Unavailable";
            return info;
        }

        std::string line;
        while (std::getline(file, line)) {
            info.sclkString += line + " | ";
            size_t mhzPos = line.find("Mhz");
            size_t colonPos = line.find(":");
            if (mhzPos != std::string::npos && colonPos != std::string::npos) {
                std::string numStr = line.substr(colonPos + 1, mhzPos - colonPos - 1);
                try {
                    uint32_t mhz = std::stoul(numStr);
                    info.maxMhz = std::max(info.maxMhz, mhz);
                    if (line.find("*") != std::string::npos) {
                        info.currentMhz = mhz;
                    }
                } catch (...) {}
            }
        }
        return info;
    }

    static void printStatus() {
        GpuClockInfo info = query();
        std::cout << "[GPU Clock] Current: " << info.currentMhz << " MHz (Max DPM: " 
                  << info.maxMhz << " MHz) - " << info.sclkString << "\n";
    }
};
