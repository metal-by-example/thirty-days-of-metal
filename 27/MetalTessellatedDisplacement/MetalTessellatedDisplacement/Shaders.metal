
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

struct InstanceConstants {
    float4x4 modelMatrix;
    float3x3 normalMatrix; // inverse-transpose of model-view matrix
};

struct FrameConstants {
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
    float displacementFactor;
    uint lightCount;
};

struct VertexIn {
    float3 position  [[attribute(0)]];
    float3 normal    [[attribute(1)]];
    float4 tangent   [[attribute(2)]];
    float2 texCoords [[attribute(3)]];
};

struct PatchIn {
    patch_control_point<VertexIn> controlPoints;
};

struct VertexOut {
    float4 position [[position]];
    float3 viewPosition;
    float3 normal;
    float3 tangent;
    float  tangentSign;
    float2 texCoords;
};

/// Calculate a value by bilinearly interpolating among four control points.
/// The four values c00, c01, c10, and c11 represent, respectively, the
/// upper-left, upper-right, lower-left, and lower-right points of a quad
/// that is parameterized by a normalized space that runs from (0, 0)
/// in the upper left to (1, 1) in the lower right (similar to Metal's texture
/// space). The vector `uv` contains the influence of the points along the
/// x and y axes.
template <typename T>
T bilerp(T c00, T c01, T c10, T c11, float2 uv) {
    T c0 = mix(c00, c01, T(uv[0]));
    T c1 = mix(c10, c11, T(uv[0]));
    return mix(c0, c1, T(uv[1]));
}

/* unused */
kernel void tess_factors_quad(device MTLQuadTessellationFactorsHalf &factors [[buffer(0)]]
                              /* other parameters */)
{
    factors.edgeTessellationFactor[0] = 1.0;
    factors.edgeTessellationFactor[1] = 1.0;
    factors.edgeTessellationFactor[2] = 1.0;
    factors.edgeTessellationFactor[3] = 1.0;

    factors.insideTessellationFactor[0] = 1.0;
    factors.insideTessellationFactor[1] = 1.0;
}

/* unused */
kernel void tess_factors_tri(device MTLTriangleTessellationFactorsHalf &factors [[buffer(0)]]
                             /* other parameters */)
{
    factors.edgeTessellationFactor[0] = 1.0;
    factors.edgeTessellationFactor[1] = 1.0;
    factors.edgeTessellationFactor[2] = 1.0;

    factors.insideTessellationFactor = 1.0;
}

[[patch(quad, 4)]]
vertex VertexOut vertex_displace_quad(patch_control_point<VertexIn> controlPoints [[stage_in]],
                                      constant InstanceConstants *instances [[buffer(2)]],
                                      constant FrameConstants &frame [[buffer(3)]],
                                      texture2d<float, access::sample> displacementMap [[texture(0)]],
                                      float2 positionInPatch [[position_in_patch]],
                                      uint instanceID [[instance_id]])
{
    constant InstanceConstants &instance = instances[instanceID];

    float3 p00 = controlPoints[0].position;
    float3 p01 = controlPoints[1].position;
    float3 p10 = controlPoints[3].position;
    float3 p11 = controlPoints[2].position;
    float3 position = bilerp(p00, p01, p10, p11, positionInPatch);

    float3 n00 = controlPoints[0].normal;
    float3 n01 = controlPoints[1].normal;
    float3 n10 = controlPoints[3].normal;
    float3 n11 = controlPoints[2].normal;
    float3 normal = bilerp(n00, n01, n10, n11, positionInPatch);

    float3 t00 = controlPoints[0].tangent.xyz;
    float3 t01 = controlPoints[1].tangent.xyz;
    float3 t10 = controlPoints[3].tangent.xyz;
    float3 t11 = controlPoints[2].tangent.xyz;
    float3 tangent = bilerp(t00, t01, t10, t11, positionInPatch);

    float2 uv00 = controlPoints[0].texCoords;
    float2 uv01 = controlPoints[1].texCoords;
    float2 uv10 = controlPoints[3].texCoords;
    float2 uv11 = controlPoints[2].texCoords;
    float2 texCoords = bilerp(uv00, uv01, uv10, uv11, positionInPatch);

    constexpr sampler bilinearSampler(coord::normalized,
                                      filter::linear,
                                      mip_filter::none,
                                      address::repeat);
    float displacement = displacementMap.sample(bilinearSampler, texCoords).r;

    position += normal * displacement * frame.displacementFactor;

    float4 worldPosition = instance.modelMatrix * float4(position, 1.0);
    float4 viewPosition = frame.viewMatrix * worldPosition;

    VertexOut out;
    out.position = frame.projectionMatrix * viewPosition;
    out.viewPosition = viewPosition.xyz;
    out.normal = instance.normalMatrix * normal;
    out.tangent = instance.normalMatrix * tangent;
    out.tangentSign = controlPoints[0].tangent.w;
    out.texCoords = texCoords;
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
