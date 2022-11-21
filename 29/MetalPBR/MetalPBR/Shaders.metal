
#include <metal_stdlib>
using namespace metal;

enum {
    VertexAttributePosition,
    VertexAttributeNormal,
    VertexAttributeTangent,
    VertexAttributeTexCoords,
    VertexAttributeJointIndices,
    VertexAttributeJointWeights
};

enum class VertexBuffer : int {
    vertexAttributes  = 0,
    instanceConstants = 8,
    frameConstants    = 9,
    lights            = 10,
    jointTransforms   = 11
};

enum class FragmentBuffer : int {
    frameConstants,
    lights,
    materialConstants
};

enum class FragmentTexture : int {
    baseColor,
    normal,
    emissive,
    metallic,
    roughness,
    ambientOcclusion,
    IBLDiffuse,
    IBLSpecular,
    IBLBRDFLookup
};

struct FrameConstants {
    float4x4 viewMatrix;
    float4x4 viewProjectionMatrix;
    uint lightCount;
};

struct InstanceConstants {
    float4x4 modelMatrix;
    float3x3 normalMatrix;
};

struct MaterialConstants {
    float4 baseColorFactor;
    float4 emissiveColor;
    float metalnessFactor;
    float roughnessFactor;
    float occlusionWeight;
    float opacity;
};

struct Material {
    // The base color of the material; either the specular color
    // or albedo, depending on the value of metalness.
    float4 baseColor;
    // Degree to which material is dielectric (0) or metal (1)
    float metalness;
    // The authored perceptually linear roughness (0-1)
    float roughness;
    // The extent to which the surface self-occludes ambient light
    float ambientOcclusion;
};

struct Surface {
    float3 reflected { 0 };
    float3 emitted { 0 };
};

struct Light {
    float4 position;  // w is ignored
    float4 direction; // w = 1 means punctual; w = 0 means directional
    float4 intensity; // a is ignored

    /// Computes the (non-normalized) vector toward a point,
    /// based on whether this is a point or directional light
    float3 directionToPoint(float3 p) {
        if (direction.w == 0) {
            return -direction.xyz;
        } else {
            return p - position.xyz;
        }
    }

    /// Evaluates the intensity of this light given a non-normalized
    /// vector from the surface to the light.
    float3 evaluateIntensity(float3 toLight) {
        if (direction.w == 0) {
            return intensity.rgb;
        } else {
            float lightDistSq = dot(toLight, toLight);
            float attenuation = 1.0f / max(lightDistSq, 1e-4);
            return attenuation * intensity.rgb;
        }
    }
};

struct VertexIn {
    float3 position  [[attribute(VertexAttributePosition)]];
    float3 normal    [[attribute(VertexAttributeNormal)]];
    float4 tangent   [[attribute(VertexAttributeTangent)]];
    float2 texCoords [[attribute(VertexAttributeTexCoords)]];
};

struct SkinnedVertexIn {
    float3 position  [[attribute(VertexAttributePosition)]];
    float3 normal    [[attribute(VertexAttributeNormal)]];
    float4 tangent   [[attribute(VertexAttributeTangent)]];
    float2 texCoords [[attribute(VertexAttributeTexCoords)]];
    uchar4 joints    [[attribute(VertexAttributeJointIndices)]];
    float4 weights   [[attribute(VertexAttributeJointWeights)]];
};

struct VertexOut {
    float4 clipPosition [[position]];
    float3 eyePosition;
    float3 eyeNormal;
    float3 eyeTangent;
    float tangentSign [[flat]];
    float2 texCoords;
};

/// Calculates the (monochromatic) specular color at normal incidence
constexpr float3 F0FromIor(float ior) {
    float k = (1.0f - ior) / (1.0f + ior);
    return k * k;
}

// Microfacet Models for Refraction through Rough Surfaces
// Walter, et al. 2007 (eq. 34)
float G1_GGX(float alphaSq, float NdotX) {
    float cosSq = NdotX * NdotX;
    float tanSq = (1.0f - cosSq) / max(cosSq, 1e-4);
    return 2.0f / (1.0f + sqrt(1.0f + alphaSq * tanSq));
}

// Microfacet Models for Refraction through Rough Surfaces
// Walter, et al. 2007 (eq. 23)
float G_JointSmith(float alphaSq, float NdotL, float NdotV) {
    return G1_GGX(alphaSq, NdotL) * G1_GGX(alphaSq, NdotV);
}

// Microfacet Models for Refraction through Rough Surfaces
// Walter, et al. 2007 (eq. 33)
float D_TrowbridgeReitz(float alphaSq, float NdotH) {
    float c = (NdotH * NdotH) * (alphaSq - 1.0f) + 1.0f;
    return step(0.0f, NdotH) * alphaSq / (M_PI_F * (c * c));
}

// An Inexpensive BRDF Model for Physically-based Rendering
// Schlick, 1994 (eq. 15)
float3 F_Schlick(float3 F0, float VdotH) {
    return F0 + (1.0f - F0) * powr(1.0f - abs(VdotH), 5.0f);
}

float3 Lambertian(float3 diffuseColor) {
    return diffuseColor * (1.0f / M_PI_F);
}

float3 BRDF(thread Material &material, float NdotL, float NdotV, float NdotH, float VdotH) {
    float3 baseColor = material.baseColor.rgb;
    float3 diffuseColor = mix(baseColor, float3(0.0f), material.metalness);

    float3 fd = Lambertian(diffuseColor) * material.ambientOcclusion;

    const float3 DielectricF0 = 0.04f; // This results from assuming an IOR of 1.5, the average for common dielectrics
    float3 F0 = mix(DielectricF0, baseColor, material.metalness);
    float alpha = material.roughness * material.roughness;
    float alphaSq = alpha * alpha;

    float D = D_TrowbridgeReitz(alphaSq, NdotH);
    float G = G_JointSmith(alphaSq, NdotL, NdotV);
    float3 F = F_Schlick(F0, VdotH);

    float3 fs = (D * G * F) / (4.0f * abs(NdotL) * abs(NdotV));

    return fd + fs;
}

float remap(float sourceMin, float sourceMax, float destMin, float destMax, float t) {
    float f = (t - sourceMin) / (sourceMax - sourceMin);
    return mix(destMin, destMax, f);
}

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant InstanceConstants &instance [[buffer(VertexBuffer::instanceConstants)]],
                             constant FrameConstants &frame       [[buffer(VertexBuffer::frameConstants)]])
{
    float4 modelPosition = float4(in.position, 1.0f);
    float4 worldPosition = instance.modelMatrix * modelPosition;
    float4 eyePosition = frame.viewMatrix * worldPosition;

    VertexOut out;
    out.clipPosition = frame.viewProjectionMatrix * worldPosition;
    out.eyePosition = eyePosition.xyz;
    out.eyeNormal = normalize(instance.normalMatrix * in.normal);
    out.eyeTangent = normalize(instance.normalMatrix * in.tangent.xyz);
    out.tangentSign = in.tangent.w;
    out.texCoords = in.texCoords;
    return out;
}

vertex VertexOut skinned_vertex_main(SkinnedVertexIn in [[stage_in]],
                                     constant InstanceConstants &instance [[buffer(VertexBuffer::instanceConstants)]],
                                     constant FrameConstants &frame       [[buffer(VertexBuffer::frameConstants)]],
                                     constant float4x4 *jointTransforms   [[buffer(VertexBuffer::jointTransforms)]])
{
    float4x4 skinningMatrix = in.weights[0] * jointTransforms[in.joints[0]] +
    in.weights[1] * jointTransforms[in.joints[1]] +
    in.weights[2] * jointTransforms[in.joints[2]] +
    in.weights[3] * jointTransforms[in.joints[3]];

    // We assume that the skinning matrix doesn't introduce non-uniform scale. If this is not
    // the case, we should take the inverse-transpose of this matrix instead.
    float3x3 skinningOrientationMatrix {
        skinningMatrix[0].xyz,
        skinningMatrix[1].xyz,
        skinningMatrix[2].xyz
    };

    float4 modelPosition = float4(in.position, 1.0f);
    float4 worldPosition = instance.modelMatrix * skinningMatrix * modelPosition;
    float4 eyePosition = frame.viewMatrix * worldPosition;

    VertexOut out;
    out.clipPosition = frame.viewProjectionMatrix * worldPosition;
    out.eyePosition = eyePosition.xyz;
    out.eyeNormal = normalize(instance.normalMatrix * skinningOrientationMatrix * in.normal);
    out.eyeTangent = normalize(instance.normalMatrix * skinningOrientationMatrix * in.tangent.xyz);
    out.tangentSign = in.tangent.w;
    out.texCoords = in.texCoords;
    return out;
}

typedef VertexOut FragmentIn;

fragment float4 fragment_main(FragmentIn in                                     [[stage_in]],
                              constant FrameConstants &frame                    [[buffer(FragmentBuffer::frameConstants)]],
                              constant Light *lights                            [[buffer(FragmentBuffer::lights)]],
                              constant MaterialConstants &materialProperties    [[buffer(FragmentBuffer::materialConstants)]],
                              texture2d<float, access::sample> baseColorTexture [[texture(FragmentTexture::baseColor)]],
                              texture2d<float, access::sample> emissiveTexture  [[texture(FragmentTexture::emissive)]],
                              texture2d<float, access::sample> normalTexture    [[texture(FragmentTexture::normal)]],
                              texture2d<float, access::sample> metalnessTexture [[texture(FragmentTexture::metallic)]],
                              texture2d<float, access::sample> roughnessTexture [[texture(FragmentTexture::roughness)]],
                              texture2d<float, access::sample> occlusionTexture [[texture(FragmentTexture::ambientOcclusion)]])
{
    constexpr sampler repeatSampler(filter::linear, mip_filter::linear, address::repeat);

    float ambientOcclusion = is_null_texture(occlusionTexture) ? 1.0f :
        mix(1.0f, occlusionTexture.sample(repeatSampler, in.texCoords).r, materialProperties.occlusionWeight);
    float4 baseColor = is_null_texture(baseColorTexture) ? materialProperties.baseColorFactor :
        baseColorTexture.sample(repeatSampler, in.texCoords) * materialProperties.baseColorFactor;
    float authoredRoughness = is_null_texture(roughnessTexture) ? materialProperties.roughnessFactor :
        roughnessTexture.sample(repeatSampler, in.texCoords).g * materialProperties.roughnessFactor;
    float metalness = is_null_texture(metalnessTexture) ? materialProperties.metalnessFactor :
        metalnessTexture.sample(repeatSampler, in.texCoords).b * materialProperties.metalnessFactor;

    Material material;
    material.baseColor = baseColor;
    material.roughness = remap(0.0f, 1.0f, 0.045f, 1.0f, authoredRoughness);
    material.metalness = metalness;
    material.ambientOcclusion = ambientOcclusion;

    float3 V = normalize(-in.eyePosition);

    float3 Ng = normalize(in.eyeNormal);
    float3 N;
    if (!is_null_texture(normalTexture)) {
        float3 T = normalize(in.eyeTangent);
        float3 B = cross(in.eyeNormal, in.eyeTangent) * in.tangentSign;
        float3x3 TBN = { T, B, Ng };
        float3 Nt = normalTexture.sample(repeatSampler, in.texCoords).xyz * 2.0f - 1.0f;
        N = TBN * Nt;
    } else {
        N = Ng;
    }

    Surface surface;
    surface.emitted = is_null_texture(emissiveTexture) ? materialProperties.emissiveColor.rgb :
    emissiveTexture.sample(repeatSampler, in.texCoords).rgb;

    for (uint i = 0; i < frame.lightCount; ++i) {
        Light light = lights[i];

        float3 lightToPoint = light.directionToPoint(in.eyePosition);
        float3 intensity = light.evaluateIntensity(lightToPoint);

        float3 L = normalize(-lightToPoint);
        float3 H = normalize(L + V);

        float NdotL = dot(N, L);
        float NdotV = dot(N, V);
        float NdotH = dot(N, H);
        float VdotH = dot(V, H);

        surface.reflected += intensity * saturate(NdotL) * BRDF(material, NdotL, NdotV, NdotH, VdotH);
    }

    float3 color = surface.emitted + surface.reflected;
    float alpha = material.baseColor.a * materialProperties.opacity;

    return float4(color * alpha, alpha);
}
