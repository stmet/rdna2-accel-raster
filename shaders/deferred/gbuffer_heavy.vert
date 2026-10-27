#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUV;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
} pc;

void main() {
    outWorldPos = inPos;
    outNormal = inNormal;
    outUV = inPos.xz * 0.1;
    gl_Position = pc.viewProj * vec4(inPos, 1.0);
}
