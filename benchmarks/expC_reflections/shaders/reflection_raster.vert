#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;

layout(push_constant) uniform PushConstants {
    mat4 reflectedViewProj;
    vec4 clipPlane;
} pc;

void main() {
    outWorldPos = inPos;
    outNormal = inNormal;
    gl_ClipDistance[0] = dot(vec4(inPos, 1.0), pc.clipPlane);
    gl_Position = pc.reflectedViewProj * vec4(inPos, 1.0);
}
