
#include <metal_stdlib>
using namespace metal;

enum LightType : uint {
    LightTypeAmbient,
    LightTypeDirectional
};

struct Light {
    float4x4 viewProjectionMatrix;
    float3 intensity; // product of color and intensity
    float3 direction;
    LightType type;
};

struct VertexIn {
    float3 position  [[attribute(0)]];
    float3 normal    [[attribute(1)]];
    float2 texCoords [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 viewPosition;
    float3 normal;
    float2 texCoords;
};

struct NodeConstants {
    float4x4 modelMatrix;
};

struct FrameConstants {
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
    uint lightCount;
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant NodeConstants &node [[buffer(2)]],
                             constant FrameConstants &frame [[buffer(3)]])
{
    float4x4 modelMatrix = node.modelMatrix;
    float4x4 modelViewMatrix = frame.viewMatrix * node.modelMatrix;

    float4 worldPosition = modelMatrix * float4(in.position, 1.0);

    float4 viewPosition = frame.viewMatrix * worldPosition;
    float4 viewNormal = modelViewMatrix * float4(in.normal, 0.0);

    VertexOut out;
    out.position = frame.projectionMatrix * viewPosition;
    out.worldPosition = worldPosition.xyz;
    out.viewPosition = viewPosition.xyz;
    out.normal = viewNormal.xyz;
    out.texCoords = in.texCoords;
    return out;
}

static float shadow(float3 worldPosition,
                    depth2d<float, access::sample> depthMap,
                    constant float4x4 &viewProjectionMatrix)
{
    float4 shadowNDC = (viewProjectionMatrix * float4(worldPosition, 1));
    shadowNDC.xyz /= shadowNDC.w;
    float2 shadowCoords = shadowNDC.xy * 0.5 + 0.5;
    shadowCoords.y = 1 - shadowCoords.y;

    constexpr sampler shadowSampler(coord::normalized,
                                    address::clamp_to_edge,
                                    filter::linear,
                                    compare_func::greater_equal);
    float depthBias = 5e-3f;
    float shadowCoverage = depthMap.sample_compare(shadowSampler, shadowCoords, shadowNDC.z - depthBias);
    return shadowCoverage;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              constant NodeConstants &node [[buffer(2)]],
                              constant FrameConstants &frame [[buffer(3)]],
                              constant Light *lights [[buffer(4)]],
                              texture2d<float, access::sample> textureMap [[texture(0)]],
                              sampler textureSampler [[sampler(0)]],
                              depth2d<float, access::sample> shadowMap [[texture(1)]])
{
    float4 baseColor = textureMap.sample(textureSampler, in.texCoords);
    float specularExponent = 50.0;

    float3 N = normalize(in.normal);
    float3 V = normalize(float3(0) - in.viewPosition);

    float3 litColor { 0 };

    for (uint i = 0; i < frame.lightCount; ++i) {
        float ambientFactor = 0;
        float diffuseFactor = 0;
        float specularFactor = 0;

        constant Light &light = lights[i];

        switch(light.type) {
            case LightTypeAmbient:
                ambientFactor = 1;
                break;
            case LightTypeDirectional: {
                float shadowFactor = 1 - shadow(in.worldPosition, shadowMap, light.viewProjectionMatrix);

                float3 L = normalize(-light.direction);
                float3 H = normalize(L + V);
                diffuseFactor = shadowFactor * saturate(dot(N, L));
                specularFactor = shadowFactor * powr(saturate(dot(N, H)), specularExponent);
                break;
            }
        }

        litColor += (ambientFactor + diffuseFactor + specularFactor) * light.intensity * baseColor.rgb;
    }

    return float4(litColor * baseColor.a, baseColor.a);
}

vertex float4 vertex_shadow(VertexIn in [[stage_in]],
                            constant float4x4 &modelViewProjectionMatrix [[buffer(2)]])
{
    return modelViewProjectionMatrix * float4(in.position, 1.0);
}
