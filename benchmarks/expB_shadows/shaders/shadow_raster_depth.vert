#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform PushConstants {
    mat4 lightViewProj;
} pc;

void main() {
    gl_Position = pc.lightViewProj * vec4(inPos, 1.0);
}
