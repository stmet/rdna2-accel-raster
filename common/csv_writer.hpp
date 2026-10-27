#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

class CsvWriter {
public:
    CsvWriter(const std::string& filepath, const std::vector<std::string>& headers) 
        : path(filepath) {
        std::ofstream file(path, std::ios::out);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open CSV file for writing: " << path << "\n";
            return;
        }
        for (size_t i = 0; i < headers.size(); i++) {
            file << headers[i];
            if (i + 1 < headers.size()) file << ",";
        }
        file << "\n";
    }

    void writeRow(const std::vector<std::string>& values) {
        std::ofstream file(path, std::ios::app);
        if (!file.is_open()) return;
        for (size_t i = 0; i < values.size(); i++) {
            file << values[i];
            if (i + 1 < values.size()) file << ",";
        }
        file << "\n";
    }

    template <typename... Args>
    void writeRowValues(Args&&... args) {
        std::vector<std::string> row;
        auto toStr = [](auto&& val) -> std::string {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, std::string>) {
                return val;
            } else if constexpr (std::is_same_v<T, const char*>) {
                return std::string(val);
            } else {
                return std::to_string(val);
            }
        };
        (row.push_back(toStr(args)), ...);
        writeRow(row);
    }

private:
    std::string path;
};
