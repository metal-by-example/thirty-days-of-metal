
#include <metal_stdlib>
using namespace metal;

enum class VertexBuffer : int {
    vertexAttributes = 0,
    instanceConstants = 1,
    frameConstants = 2,
    jointMatrices = 3,
};

enum class FragmentBuffer : int {
    frameConstants = 0,
    lights = 1,
    materialConstants = 2,
};

enum class FragmentTexture : int {
    baseColor = 0,
    normal = 1,
};

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
    float2 texCoords [[attribute(2)]];
};

struct SkinnedVertexIn {
    float3 position      [[attribute(0)]];
    float3 normal        [[attribute(1)]];
    float2 texCoords     [[attribute(2)]];
    ushort4 jointIndices [[attribute(3)]];
    float4 jointWeights  [[attribute(4)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 viewPosition;
    float3 normal;
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
};

struct MaterialConstants {
    float4 baseColorFactor;
    float shininess;
};


vertex VertexOut skinned_vertex_main(SkinnedVertexIn in [[stage_in]],
                                     constant InstanceConstants *instances [[buffer(VertexBuffer::instanceConstants)]],
                                     constant FrameConstants &frame [[buffer(VertexBuffer::frameConstants)]],
                                     constant float4x4 *jointMatrices [[buffer(VertexBuffer::jointMatrices)]],
                                     uint instanceID [[instance_id]])
{
    constant InstanceConstants &instance = instances[instanceID];

    float4 modelPosition = float4(in.position, 1.0);
    float4 modelNormal = float4(in.normal, 0.0);

    float4x4 skinningMatrix = in.jointWeights[0] * jointMatrices[in.jointIndices[0]] +
                              in.jointWeights[1] * jointMatrices[in.jointIndices[1]] +
                              in.jointWeights[2] * jointMatrices[in.jointIndices[2]] +
                              in.jointWeights[3] * jointMatrices[in.jointIndices[3]];

    modelPosition = skinningMatrix * modelPosition;

    // Calculating the normal matrix in this way assumes our skinning matrices don't
    // contain any non-uniform scaling components. If this doesn't hold, we would use
    // the inverse transpose of the skinning matrix instead.
    modelNormal = skinningMatrix * modelNormal;

    float4 worldPosition = instance.modelMatrix * modelPosition;
    float4 viewPosition = frame.viewMatrix * worldPosition;

    VertexOut out;
    out.position = frame.projectionMatrix * viewPosition;
    out.viewPosition = viewPosition.xyz;
    out.normal = normalize(instance.normalMatrix * modelNormal.xyz);
    out.texCoords = in.texCoords;
    return out;
}

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant InstanceConstants *instances [[buffer(VertexBuffer::instanceConstants)]],
                             constant FrameConstants &frame [[buffer(VertexBuffer::frameConstants)]],
                             uint instanceID [[instance_id]])
{
    constant InstanceConstants &instance = instances[instanceID];

    float4x4 modelViewMatrix = frame.viewMatrix * instance.modelMatrix;
    float4 viewPosition = modelViewMatrix * float4(in.position, 1.0);

    VertexOut out;
    out.position = frame.projectionMatrix * viewPosition;
    out.viewPosition = viewPosition.xyz;
    out.normal = normalize(instance.normalMatrix * in.normal);
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
                              constant FrameConstants &frame [[buffer(FragmentBuffer::frameConstants)]],
                              constant Light *lights [[buffer(FragmentBuffer::lights)]],
                              constant MaterialConstants &material [[buffer(FragmentBuffer::materialConstants)]],
                              texture2d<float, access::sample> baseColorTexture [[texture(FragmentTexture::baseColor)]],
                              texture2d<float, access::sample> normalTexture [[texture(FragmentTexture::normal)]])
{
    constexpr sampler trilinearSampler(coord::normalized,
                                       filter::linear,
                                       mip_filter::linear,
                                       address::repeat);

    float4 baseColor = material.baseColorFactor;
    if (!is_null_texture(baseColorTexture)) {
        baseColor *= baseColorTexture.sample(trilinearSampler, in.texCoords);
    }

    float specularExponent = max(1.0, min(mix(1.0, 1024.0, material.shininess), 1024.0));

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
