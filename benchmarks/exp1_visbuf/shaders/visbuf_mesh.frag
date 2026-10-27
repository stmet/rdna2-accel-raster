#version 460

layout(location = 0) flat in uint inPrimId;
layout(location = 0) out uint outVisBuffer;

void main() {
    outVisBuffer = inPrimId;
}
