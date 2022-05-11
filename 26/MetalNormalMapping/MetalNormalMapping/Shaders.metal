
#include <metal_stdlib>
using namespace metal;

enum LightType : uint {
    LightTypeAmbient,
    LightTypeDirectional,
    LightTypeOmnidirectional,
};

struct Light {
    float3 intensity; // product of color and intensity
    float3 position;  // view-space position
    float3 direction; // view-space direction
    LightType type;
};

struct VertexIn {
    float3 position  [[attribute(0)]];
    float3 normal    [[attribute(1)]];
    float4 tangent   [[attribute(2)]];
    float2 texCoords [[attribute(3)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 viewPosition;
    float3 normal;
    float3 tangent;
    float tangentSign [[flat]];
    float2 texCoords;
};

struct InstanceConstants {
    float4x4 modelMatrix;
    float3x3 normalMatrix; // inverse-transpose of model-view matrix
};

struct FrameConstants {
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
    uint lightCount;
    uint renderMode;
};

enum RenderMode {
    RenderModeGeometricNormals,
    RenderModeNormalMapped
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant InstanceConstants *instances [[buffer(2)]],
                             constant FrameConstants &frame [[buffer(3)]],
                             uint instanceID [[instance_id]])
{
    constant InstanceConstants &instance = instances[instanceID];

    float4 worldPosition = instance.modelMatrix * float4(in.position, 1.0);
    float4 viewPosition = frame.viewMatrix * worldPosition;

    VertexOut out;
    out.position = frame.projectionMatrix * viewPosition;
    out.viewPosition = viewPosition.xyz;
    out.normal = normalize(instance.normalMatrix * in.normal);
    out.tangent = normalize(instance.normalMatrix * in.tangent.xyz);
    out.tangentSign = in.tangent.w;
    out.texCoords = in.texCoords;
    return out;
}

float distanceAttenuation(constant Light &light, float3 toLight) {
    switch (light.type) {
        case LightTypeOmnidirectional: {
            float lightDistSq = dot(toLight, toLight);
            return 1.0f / max(lightDistSq, 1e-4);
            break;
        }
        default:
            return 1.0;
    }
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              constant FrameConstants &frame [[buffer(3)]],
                              constant Light *lights [[buffer(4)]],
                              texture2d<float, access::sample> baseColorTexture [[texture(0)]],
                              texture2d<float, access::sample> normalTexture [[texture(1)]])
{
    constexpr sampler trilinearSampler(coord::normalized,
                                       filter::linear,
                                       mip_filter::linear,
                                       address::repeat);

    float4 baseColor = baseColorTexture.sample(trilinearSampler, in.texCoords);
    float specularExponent = 150.0;

    float3 T = normalize(in.tangent);
    float3 B = cross(in.normal, in.tangent) * in.tangentSign;
    float3 Nv = normalize(in.normal);
    float3x3 TBN = { T, B, Nv };

    float3 Nt { 0.5, 0.5, 1 };
    if (!is_null_texture(normalTexture)) {
        Nt = normalTexture.sample(trilinearSampler, in.texCoords).xyz;
    }
    Nt = Nt * 2.0 - 1.0;

    float3 N = TBN * Nt;
    
    if (frame.renderMode != RenderModeNormalMapped) { N = Nv; }

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
                float3 L = normalize(-light.direction);
                float3 H = normalize(L + V);
                diffuseFactor = saturate(dot(N, L));
                specularFactor = powr(saturate(dot(N, H)), specularExponent);
                break;
            }
            case LightTypeOmnidirectional: {
                float3 toLight = light.position - in.viewPosition;
                float attenuation = distanceAttenuation(light, toLight);

                float3 L = normalize(toLight);
                float3 H = normalize(L + V);
                diffuseFactor = attenuation * saturate(dot(N, L));
                specularFactor = attenuation * powr(saturate(dot(N, H)), specularExponent);
                break;
            }
        }

        litColor += (ambientFactor + diffuseFactor) * light.intensity * baseColor.rgb +
                    specularFactor * light.intensity;
    }

    return float4(litColor * baseColor.a, baseColor.a);
}
