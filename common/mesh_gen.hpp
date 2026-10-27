#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

struct SceneMesh {
    struct Vertex {
        float pos[3];
        float normal[3];
    };

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    uint32_t triangleCount() const {
        return static_cast<uint32_t>(indices.size() / 3);
    }
};

class MeshGenerator {
public:
    // Generate a subdivided/displaced sphere or landscape with target triangle count
    static SceneMesh createTessellatedTerrain(uint32_t targetTriangles, float size = 100.0f) {
        SceneMesh mesh;
        // Grid dimension: (res - 1) * (res - 1) * 2 = targetTriangles
        uint32_t quads = targetTriangles / 2;
        uint32_t res = static_cast<uint32_t>(std::sqrt(quads)) + 1;
        if (res < 2) res = 2;

        mesh.vertices.resize(res * res);
        mesh.indices.reserve((res - 1) * (res - 1) * 6);

        float step = size / (res - 1);
        float halfSize = size * 0.5f;

        for (uint32_t y = 0; y < res; y++) {
            for (uint32_t x = 0; x < res; x++) {
                uint32_t idx = y * res + x;
                float px = -halfSize + x * step;
                float pz = -halfSize + y * step;
                // Add some hills to create non-trivial depth complexity
                float fx = px * 0.05f;
                float fz = pz * 0.05f;
                float py = std::sin(fx) * std::cos(fz) * 5.0f + std::sin(fx * 2.5f + fz * 1.7f) * 2.0f;

                mesh.vertices[idx].pos[0] = px;
                mesh.vertices[idx].pos[1] = py;
                mesh.vertices[idx].pos[2] = pz;

                // Approximate normal
                mesh.vertices[idx].normal[0] = 0.0f;
                mesh.vertices[idx].normal[1] = 1.0f;
                mesh.vertices[idx].normal[2] = 0.0f;
            }
        }

        for (uint32_t y = 0; y < res - 1; y++) {
            for (uint32_t x = 0; x < res - 1; x++) {
                uint32_t i0 = y * res + x;
                uint32_t i1 = y * res + (x + 1);
                uint32_t i2 = (y + 1) * res + x;
                uint32_t i3 = (y + 1) * res + (x + 1);

                // Tri 1
                mesh.indices.push_back(i0);
                mesh.indices.push_back(i2);
                mesh.indices.push_back(i1);

                // Tri 2
                mesh.indices.push_back(i1);
                mesh.indices.push_back(i2);
                mesh.indices.push_back(i3);
            }
        }

        return mesh;
    }

    static SceneMesh loadObj(const std::string& filepath) {
        SceneMesh mesh;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open OBJ file: " << filepath << std::endl;
            return mesh;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.size() > 2 && line[0] == 'v' && line[1] == ' ') {
                std::istringstream s(line.substr(2));
                SceneMesh::Vertex v{};
                s >> v.pos[0] >> v.pos[1] >> v.pos[2];
                v.normal[0] = 0.0f; v.normal[1] = 1.0f; v.normal[2] = 0.0f;
                mesh.vertices.push_back(v);
            } else if (line.size() > 2 && line[0] == 'f' && line[1] == ' ') {
                std::istringstream s(line.substr(2));
                std::string token;
                uint32_t fIndices[3];
                int idx = 0;
                while (s >> token && idx < 3) {
                    size_t slash = token.find('/');
                    if (slash != std::string::npos) {
                        token = token.substr(0, slash);
                    }
                    fIndices[idx++] = static_cast<uint32_t>(std::stoul(token) - 1);
                }
                if (idx == 3) {
                    mesh.indices.push_back(fIndices[0]);
                    mesh.indices.push_back(fIndices[1]);
                    mesh.indices.push_back(fIndices[2]);
                }
            }
        }
        return mesh;
    }
};
