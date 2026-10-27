#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

void main() {
    // Simple directional lighting for reflected scene
    vec3 N = normalize(inNormal);
    vec3 L = normalize(vec3(0.5, 0.8, 0.3));
    float diff = max(0.0, dot(N, L)) * 0.8 + 0.2;
    vec3 col = vec3(0.6, 0.65, 0.7) * diff;
    outColor = vec4(col, 1.0);
}
