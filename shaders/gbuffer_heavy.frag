#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMaterial;

layout(set = 0, binding = 0) uniform sampler2D texAlbedo;
layout(set = 0, binding = 1) uniform sampler2D texNormal;
layout(set = 0, binding = 2) uniform sampler2D texRoughMetal;
layout(set = 0, binding = 3) uniform sampler2D texDetail;

void main() {
    // Texture-heavy material evaluation
    vec4 baseCol = texture(texAlbedo, inUV);
    vec4 normMap = texture(texNormal, inUV * 2.0);
    vec4 rma = texture(texRoughMetal, inUV);
    vec4 detail = texture(texDetail, inUV * 8.0);

    // Detail blend
    vec3 blendedAlbedo = baseCol.rgb * (detail.rgb * 1.5);
    
    // Tangent frame approximation
    vec3 N = normalize(inNormal);
    vec3 dp1 = dFdx(inWorldPos);
    vec3 dp2 = dFdy(inWorldPos);
    vec2 duv1 = dFdx(inUV);
    vec2 duv2 = dFdy(inUV);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T,T), dot(B,B)));
    mat3 TBN = mat3(T * invmax, B * invmax, N);

    vec3 normalSample = normMap.xyz * 2.0 - 1.0;
    vec3 worldNormal = normalize(TBN * normalSample);

    // Multiple dependent texture lookups to represent parallax/heavy layered material
    vec2 offsetUV = inUV + worldNormal.xy * 0.02;
    vec4 secondaryAlbedo = texture(texAlbedo, offsetUV * 3.0);
    vec4 secondaryNorm = texture(texNormal, offsetUV * 4.0);

    blendedAlbedo = mix(blendedAlbedo, secondaryAlbedo.rgb, 0.3);
    worldNormal = normalize(mix(worldNormal, secondaryNorm.xyz * 2.0 - 1.0, 0.2));

    outAlbedo = vec4(blendedAlbedo, 1.0);
    outNormal = vec4(worldNormal * 0.5 + 0.5, 1.0);
    outMaterial = vec4(rma.r, rma.g, 0.0, 1.0); // Roughness, Metallic, Material ID
}
