
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 normal;
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant float4x4 &transform [[buffer(2)]])
{
    VertexOut out;
    out.position = transform * float4(in.position, 1.0);
    out.normal = in.normal;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    float3 N = normalize(in.normal);
    float3 color = N * float3(0.5) + float3(0.5);
    return float4(color, 1);
}
