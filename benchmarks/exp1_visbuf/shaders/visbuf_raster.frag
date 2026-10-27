#version 460

layout(location = 0) out uint outTriangleId;

void main() {
    outTriangleId = uint(gl_PrimitiveID + 1);
}
