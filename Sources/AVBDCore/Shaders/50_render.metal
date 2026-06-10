#include <metal_stdlib>
using namespace metal;

// Builds per-instance render data (4x4 transform + color) straight from
// solver state, so the renderer never touches body buffers on the CPU.

struct RenderInstance {
    float4x4 model;     // column-major, includes size scaling
    float4 color;       // w = shape type (0 box, 1 sphere, 2 torus)
    float4 params;      // torus: x = major R, y = minor r
};

inline float3x3 q_to_mat(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x*x, yy = y*y, zz = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;
    // columns
    return float3x3(
        float3(1 - 2*(yy + zz), 2*(xy + wz), 2*(xz - wy)),
        float3(2*(xy - wz), 1 - 2*(xx + zz), 2*(yz + wx)),
        float3(2*(xz + wy), 2*(yz - wx), 1 - 2*(xx + yy)));
}

inline float3 palette(uint i) {
    // Golden-ratio hue cycling, decent saturation/brightness
    float h = fract(float(i) * 0.61803398875f);
    float3 k = float3(5.0f, 3.0f, 1.0f);
    float3 p = abs(fract(h + k / 6.0f) * 6.0f - 3.0f) - 1.0f;
    return mix(float3(0.35f), clamp(p, 0.0f, 1.0f), 0.75f);
}

kernel void build_instances(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device const float4* shape      [[buffer(2)]],
    device RenderInstance* out      [[buffer(3)]],
    constant uint& numBodies        [[buffer(4)]],
    constant uint& colorMode        [[buffer(5)]],   // 0 index, 1 graph color
    device const uint* colors       [[buffer(6)]],
    device const uint* shapeType    [[buffer(7)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= numBodies) return;
    float4 pl = posLin[gid];
    float3x3 R = q_to_mat(posAng[gid]);
    uint st = shapeType[gid];
    // torus/capsule: unit-scale model (geometry sized in the vertex shader)
    float3 sz = st >= 2 ? float3(1) : shape[gid].xyz;

    float4x4 m;
    m[0] = float4(R[0] * sz.x, 0);
    m[1] = float4(R[1] * sz.y, 0);
    m[2] = float4(R[2] * sz.z, 0);
    m[3] = float4(pl.xyz, 1);
    out[gid].model = m;

    float3 c = pl.w > 0.0f
        ? palette(colorMode == 1 ? colors[gid] : gid)
        : float3(0.45f, 0.45f, 0.48f);
    out[gid].color = float4(c, float(st));
    out[gid].params = float4(shape[gid].x, shape[gid].y, 0, 0);
}
