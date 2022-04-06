
#include <metal_stdlib>
using namespace metal;

kernel void add_two_values(constant float *inputsA [[buffer(0)]],
                           constant float *inputsB [[buffer(1)]],
                           device float *outputs   [[buffer(2)]],
                           uint index [[thread_position_in_grid]])
{
    outputs[index] = inputsA[index] + inputsB[index];
}

vertex float4 vertex_main(device float2 const* positions [[buffer(0)]],
                          uint vertexID [[vertex_id]])
{
    float2 position = positions[vertexID];
    return float4(position, 0.0, 1.0);
}

fragment float4 fragment_main(float4 position [[stage_in]]) {
    return float4(1.0, 0.0, 0.0, 1.0);
}
