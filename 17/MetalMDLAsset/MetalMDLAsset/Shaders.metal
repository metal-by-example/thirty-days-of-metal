
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position  [[attribute(0)]];
    float3 normal    [[attribute(1)]];
    float2 texCoords [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 normal;
    float2 texCoords;
};

struct NodeConstants {
    float4x4 modelViewProjectionMatrix;
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant NodeConstants &constants [[buffer(2)]])
{
    VertexOut out;
    out.position = constants.modelViewProjectionMatrix * float4(in.position, 1.0);
    out.normal = in.normal;
    out.texCoords = in.texCoords;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              texture2d<float, access::sample> textureMap [[texture(0)]],
                              sampler textureSampler [[sampler(0)]])
{
    float4 color = textureMap.sample(textureSampler, in.texCoords);
    return color;
}
