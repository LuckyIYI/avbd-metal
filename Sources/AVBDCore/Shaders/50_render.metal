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
    // Golden-ratio hue cycling, cheerful pastel-bright tones
    float h = fract(float(i) * 0.61803398875f);
    float3 p = abs(fract(h + float3(5.0f, 3.0f, 1.0f) / 6.0f) * 6.0f - 3.0f) - 1.0f;
    float3 hue = clamp(p, 0.0f, 1.0f);
    // lift toward white a touch, keep punch
    return mix(float3(0.92f), hue, 0.62f) * 0.98f;
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
    device const uint* surfaced     [[buffer(8)]],   // 1 = body rendered by a surface mesh
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= numBodies) return;
    float4 pl = posLin[gid];
    float3x3 R = q_to_mat(posAng[gid]);
    uint st = shapeType[gid] & SHAPE_KIND_MASK;
    // torus/capsule: unit-scale model (geometry sized in the vertex shader)
    float3 sz = st >= 2 ? float3(1) : shape[gid].xyz;

    float4x4 m;
    m[0] = float4(R[0] * sz.x, 0);
    m[1] = float4(R[1] * sz.y, 0);
    m[2] = float4(R[2] * sz.z, 0);
    m[3] = float4(pl.xyz, 1);
    out[gid].model = m;

    // hide the giant ground slab (the checkerboard floor replaces it)
    if (st == 0 && pl.w <= 0.0f && shape[gid].x > 150.0f) {
        out[gid].model = float4x4(0.0f);
        out[gid].color = float4(0);
        out[gid].params = float4(0);
        return;
    }
    float3 c = pl.w > 0.0f
        ? palette(colorMode == 1 ? colors[gid] : gid)
        : float3(0.58f, 0.60f, 0.66f);
    // Surface-meshed particles (cloth/tet vertices): shape type 9 collapses
    // in every geometry vertex shader, but params survive so the body still
    // casts its blob shadow — dense overlapping blobs read as the soft
    // shadow of the sheet/jelly itself.
    out[gid].color = float4(c, surfaced[gid] != 0 ? 9.0f : float(st));
    // params.z = bounding radius (blob shadow size), w = shadow strength
    // (statics cast none — they're scenery)
    out[gid].params = float4(shape[gid].x, shape[gid].y, shape[gid].w,
                             pl.w > 0.0f ? 1.0f : 0.0f);
}

// Smooth vertex normals, two passes (after the reference cloth renderer):
// face normals once per triangle, then an ANGLE x AREA weighted gather per
// vertex — angle weighting keeps normals honest where incident triangles
// have very different shapes (hems, tet-surface corners).
inline float3 softSafeNormalize(float3 v) {
    float l = length(v);
    return l > 1e-12f ? v / l : float3(0);
}

kernel void soft_face_normals(
    device const float4* posLin     [[buffer(0)]],
    device const uint* surfTris     [[buffer(1)]],   // 3 packed corner ids per tri
    device float4* faceNormals      [[buffer(2)]],   // xyz raw cross (2A * n)
    constant uint& numTris          [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= numTris) return;
    float3 a = posLin[surfTris[3 * gid + 0] & 0x007FFFFFu].xyz;
    float3 b = posLin[surfTris[3 * gid + 1] & 0x007FFFFFu].xyz;
    float3 c = posLin[surfTris[3 * gid + 2] & 0x007FFFFFu].xyz;
    faceNormals[gid] = float4(cross(b - a, c - a), 0);
}

kernel void soft_normals(
    device const float4* posLin     [[buffer(0)]],
    device const uint* surfVerts    [[buffer(1)]],
    device const uint* vtStart      [[buffer(2)]],
    device const uint* vtCount      [[buffer(3)]],
    device const uint* vtList       [[buffer(4)]],
    device const uint* surfTris     [[buffer(5)]],
    device float4* normalsOut       [[buffer(6)]],   // w = render thickness
    constant uint& numSurfVerts     [[buffer(7)]],
    device const float4* faceNormals [[buffer(8)]],
    device const float4* shape      [[buffer(9)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= numSurfVerts) return;
    uint v = surfVerts[gid];
    float3 p = posLin[v].xyz;
    float3 acc = float3(0);
    float wsum = 0.0f;
    uint s = vtStart[gid], e = s + vtCount[gid];
    for (uint k = s; k < e; k++) {
        uint t = vtList[k];
        uint ia = surfTris[3 * t + 0] & 0x001FFFFFu;
        uint ib = surfTris[3 * t + 1] & 0x001FFFFFu;
        uint ic = surfTris[3 * t + 2] & 0x001FFFFFu;
        float3 fn = faceNormals[t].xyz;
        float area = 0.5f * length(fn);
        // angle at THIS vertex inside the triangle
        float3 ea, eb;
        if (ia == v)      { ea = posLin[ib].xyz - p; eb = posLin[ic].xyz - p; }
        else if (ib == v) { ea = posLin[ia].xyz - p; eb = posLin[ic].xyz - p; }
        else              { ea = posLin[ia].xyz - p; eb = posLin[ib].xyz - p; }
        float ang = acos(clamp(dot(softSafeNormalize(ea), softSafeNormalize(eb)),
                               -1.0f, 1.0f));
        float w = area * ang;
        acc += softSafeNormalize(fn) * w;
        wsum += w;
    }
    // Sharp creases can cancel the weighted sum to ~zero (fold-over: the
    // incident faces point opposite ways). Fall back to the first incident
    // face normal instead of shading with a null normal.
    float coherence = wsum > 1e-9f ? length(acc) / wsum : 0.0f;
    if (length(acc) < 1e-6f && vtCount[gid] > 0) {
        acc = faceNormals[vtList[s]].xyz;
    }
    // The render skin THINS where curvature is high (low normal coherence):
    // a +-r extrusion around a hem curled tighter than r pokes the back
    // layer through the front — the black slivers at fold tips.
    float thick = fabs(shape[v].w)
                * mix(0.25f, 1.0f, saturate(coherence * coherence));
    normalsOut[v] = float4(softSafeNormalize(acc), thick);
}
