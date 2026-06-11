#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Cloth element pipeline: triangles (and edges) are first-class collision
// primitives. They are binned into their own spatial grid; detection kernels
// emit unified SoftContactGPU records (V-T, rigid-feature-T, E-E) that get
// the full AVBD treatment downstream: persistent warm-started lambda/penalty,
// bounded duals scaled to participant masses, graph-coloring conflicts across
// the whole stencil, and the block-descent primal.
// Runs at start-of-step poses, before body prediction (like np_collide).
// ============================================================================

#define VT_MAX_PER_VERTEX 4u
#define RT_MAX_PER_TRI 4u
#define EE_MAX_PER_EDGE 2u
#define ELEM_EDGE_BIT 0x80000000u
// E-E contacts only when both closest points are interior (endpoints are
// covered by V-T); standard culling threshold.
#define EE_INTERIOR_EPS 0.02f

// ----------------------------------------------------------------------------
// Closest point on triangle (Ericson, Real-Time Collision Detection 5.1.5)
// ----------------------------------------------------------------------------
inline float3 closestPtTriangle(float3 p, float3 a, float3 b, float3 c,
                                thread float3& bary)
{
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { bary = float3(1, 0, 0); return a; }
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { bary = float3(0, 1, 0); return b; }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / max(d1 - d3, 1e-12f);
        bary = float3(1 - v, v, 0);
        return a + ab * v;
    }
    float3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { bary = float3(0, 0, 1); return c; }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / max(d2 - d6, 1e-12f);
        bary = float3(1 - w, 0, w);
        return a + ac * w;
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-12f);
        bary = float3(0, 1 - w, w);
        return b + (c - b) * w;
    }
    float denom = va + vb + vc;
    if (fabs(denom) < 1e-20f) { bary = float3(1, 0, 0); return a; }
    float v = vb / denom, w = vc / denom;
    bary = float3(1 - v - w, v, w);
    return a + ab * v + ac * w;
}

// Sorted-CSR membership test (per-vertex topological neighborhood)
inline bool nbrContains(device const uint* nbrList, uint s, uint e, uint x) {
    uint lo = s, hi = e;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (nbrList[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo < e && nbrList[lo] == x;
}

// ----------------------------------------------------------------------------
// Soft persistence map (open addressing, CAS on keyA+1). keyA packs the
// contact kind in the top nibble so it is never zero.
// ----------------------------------------------------------------------------
inline uint softKeyHash(uint a, uint b, uint capacity) {
    uint h = (a * 0x9E3779B1u) ^ (b * 0x85EBCA77u);
    h ^= h >> 16;
    h *= 0xC2B2AE3Du;
    h ^= h >> 13;
    return h & (capacity - 1u);
}

inline int softMapFind(device const atomic_uint* keyAs,
                       device const uint* keyBs,
                       device const uint* vals,
                       uint capacity, uint keyA, uint keyB) {
    uint h = softKeyHash(keyA, keyB, capacity);
    for (uint probe = 0; probe < 64; probe++) {
        uint ka = atomic_load_explicit(&keyAs[h], memory_order_relaxed);
        if (ka == 0) return -1;
        if (ka == keyA + 1 && keyBs[h] == keyB) return int(vals[h]);
        h = (h + 1) & (capacity - 1u);
    }
    return -1;
}

kernel void softmap_clear(
    device atomic_uint* keyAs       [[buffer(0)]],
    constant SimParams& P           [[buffer(1)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid < P.softMapCapacity)
        atomic_store_explicit(&keyAs[gid], 0u, memory_order_relaxed);
}

kernel void softmap_insert(
    device const SoftContactGPU* soft [[buffer(0)]],
    device atomic_uint* keyAs       [[buffer(1)]],
    device uint* keyBs              [[buffer(2)]],
    device uint* vals               [[buffer(3)]],
    device const atomic_uint* counters [[buffer(4)]],
    constant SimParams& P           [[buffer(5)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint n = min(atomic_load_explicit(&counters[CTR_SOFT], memory_order_relaxed),
                 P.maxSoft);
    if (gid >= n) return;
    uint keyA = as_type<uint>(soft[gid].lambda.w);
    uint keyB = as_type<uint>(soft[gid].penalty.w);
    uint h = softKeyHash(keyA, keyB, P.softMapCapacity);
    for (uint probe = 0; probe < P.softMapCapacity; probe++) {
        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(&keyAs[h], &expected, keyA + 1,
                                                  memory_order_relaxed, memory_order_relaxed)) {
            keyBs[h] = keyB;
            vals[h] = gid;
            return;
        }
        h = (h + 1) & (P.softMapCapacity - 1u);
    }
}

// ----------------------------------------------------------------------------
// Element binning: triangles + edges hashed at their center into the element
// grid (cell size >= 2x the max element bounding radius, computed at init
// with stretch headroom — inextensible cloth cannot outgrow it).
// ----------------------------------------------------------------------------
kernel void el_count(
    device const float4* posLin     [[buffer(0)]],
    device const uint4* tris        [[buffer(1)]],
    device const uint2* edges       [[buffer(2)]],
    device atomic_uint* cellCount   [[buffer(3)]],
    device uint2* elemSlot          [[buffer(4)]],
    constant SimParams& P           [[buffer(5)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint total = P.numTris + P.numEdges;
    if (gid >= total) return;
    float3 c;
    if (gid < P.numTris) {
        uint4 t = tris[gid];
        c = (posLin[t.x].xyz + posLin[t.y].xyz + posLin[t.z].xyz) / 3.0f;
    } else {
        uint2 e = edges[gid - P.numTris];
        c = (posLin[e.x].xyz + posLin[e.y].xyz) * 0.5f;
    }
    uint h = cellHash(cellCoord(c, P.elemCellSize), P.elemHashSize);
    uint slot = atomic_fetch_add_explicit(&cellCount[h], 1u, memory_order_relaxed);
    elemSlot[gid] = uint2(h, slot);
}

kernel void el_scatter(
    device const uint2* elemSlot    [[buffer(0)]],
    device const uint* cellStart    [[buffer(1)]],
    device uint* cellElems          [[buffer(2)]],
    constant SimParams& P           [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint total = P.numTris + P.numEdges;
    if (gid >= total) return;
    uint2 cs = elemSlot[gid];
    uint entry = gid < P.numTris ? gid : (ELEM_EDGE_BIT | (gid - P.numTris));
    cellElems[cellStart[cs.x] + cs.y] = entry;
}

// ----------------------------------------------------------------------------
// Shared emission helper: warm start from the persistence map and append the
// record. The lambda cap follows the participants — gram-scale particles get
// gram-scale force bounds (memory: wedged contacts stockpile force otherwise).
// ----------------------------------------------------------------------------
inline float softLambdaCap(constant SimParams& P, float minMass) {
    return min(P.lambdaMax, max(10.0f, 1.0e5f * minMass));
}

inline void softEmit(device SoftContactGPU* soft,
                     device atomic_uint* counters,
                     device const SoftContactGPU* prevSoft,
                     device const atomic_uint* mapKeyA,
                     device const uint* mapKeyB,
                     device const uint* mapVal,
                     constant SimParams& P,
                     uint4 ids, float4 weights,
                     float3 n, float friction, float3 anchorA,
                     float C0n, float lamCap,
                     uint kind, bool rigidA, bool roundA, bool sideNeg,
                     uint keyA, uint keyB)
{
    float3 lam = float3(0);
    float3 pen = float3(0);
    int prev = softMapFind(mapKeyA, mapKeyB, mapVal, P.softMapCapacity, keyA, keyB);
    if (prev >= 0) {
        lam = prevSoft[prev].lambda.xyz * (P.alpha * P.gamma);
        pen = prevSoft[prev].penalty.xyz;
    }
    pen = clamp(pen * P.gamma, PENALTY_MIN, PENALTY_MAX);

    uint slot = atomic_fetch_add_explicit(&counters[CTR_SOFT], 1u, memory_order_relaxed);
    if (slot >= P.maxSoft) return;

    uint flags = (rigidA ? SCF_RIGID_A : 0u)
               | (sideNeg ? SCF_SIDE_NEG : 0u)
               | (kind << SCF_KIND_SHIFT)
               | (roundA ? 32u : 0u);
    device SoftContactGPU& sc = soft[slot];
    sc.ids = ids;
    sc.normal = float4(n, friction);
    sc.anchorA = float4(anchorA, as_type<float>(flags));
    sc.weights = weights;
    sc.C0 = float4(C0n, 0, 0, lamCap);
    sc.lambda = float4(lam, as_type<float>(keyA));
    sc.penalty = float4(pen, as_type<float>(keyB));
}

// Resolve sign memory: returns previous side (+1/-1) if the contact persists,
// else the proposed side.
inline float softPrevSide(device const SoftContactGPU* prevSoft,
                          device const atomic_uint* mapKeyA,
                          device const uint* mapKeyB,
                          device const uint* mapVal,
                          constant SimParams& P,
                          uint keyA, uint keyB, float proposed)
{
    int prev = softMapFind(mapKeyA, mapKeyB, mapVal, P.softMapCapacity, keyA, keyB);
    if (prev < 0) return proposed;
    uint flags = as_type<uint>(prevSoft[prev].anchorA.w);
    return (flags & SCF_SIDE_NEG) ? -1.0f : 1.0f;
}

// ----------------------------------------------------------------------------
// V-T: one thread per particle, scanning the element grid for triangles.
// Exclusion is topological: own triangles and triangles touching the 1-ring.
// ----------------------------------------------------------------------------
kernel void vt_emit(
    device const float4* posLin     [[buffer(0)]],
    device const float4* shape      [[buffer(1)]],
    device const float4* props      [[buffer(2)]],
    device const float4* velLin     [[buffer(3)]],
    device const uint* particleIdx  [[buffer(4)]],
    device const uint4* tris        [[buffer(5)]],
    device const uint* elemCellStart [[buffer(6)]],
    device const uint* elemCellCount [[buffer(7)]],
    device const uint* cellElems    [[buffer(8)]],
    device const uint* nbrStart     [[buffer(9)]],
    device const uint* nbrCount     [[buffer(10)]],
    device const uint* nbrList      [[buffer(11)]],
    device SoftContactGPU* soft     [[buffer(12)]],
    device atomic_uint* counters    [[buffer(13)]],
    device const SoftContactGPU* prevSoft [[buffer(14)]],
    device const atomic_uint* mapKeyA [[buffer(15)]],
    device const uint* mapKeyB      [[buffer(16)]],
    device const uint* mapVal       [[buffer(17)]],
    constant SimParams& P           [[buffer(18)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numParticles) return;
    uint v = particleIdx[gid];
    float3 pv = posLin[v].xyz;
    float massV = posLin[v].w;
    float rv = fabs(shape[v].w);
    float3 velV = velLin[v].xyz;
    uint ns = nbrStart[v], ne = ns + nbrCount[v];

    int3 cc = cellCoord(pv, P.elemCellSize);
    // one thread owns each vertex: the cap is thread-local
    uint emitted = 0;

    for (int dz = -1; dz <= 1 && emitted < VT_MAX_PER_VERTEX; dz++)
    for (int dy = -1; dy <= 1 && emitted < VT_MAX_PER_VERTEX; dy++)
    for (int dx = -1; dx <= 1 && emitted < VT_MAX_PER_VERTEX; dx++) {
        uint h = cellHash(cc + int3(dx, dy, dz), P.elemHashSize);
        uint s = elemCellStart[h], e = s + elemCellCount[h];
        for (uint k = s; k < e && emitted < VT_MAX_PER_VERTEX; k++) {
            uint entry = cellElems[k];
            if (entry & ELEM_EDGE_BIT) continue;
            uint t = entry;
            uint4 tid = tris[t];
            if (tid.x == v || tid.y == v || tid.z == v) continue;
            if (nbrContains(nbrList, ns, ne, tid.x) ||
                nbrContains(nbrList, ns, ne, tid.y) ||
                nbrContains(nbrList, ns, ne, tid.z)) continue;

            float3 a = posLin[tid.x].xyz;
            float3 b = posLin[tid.y].xyz;
            float3 c = posLin[tid.z].xyz;
            float3 bary;
            float3 q = closestPtTriangle(pv, a, b, c, bary);
            float3 d = pv - q;
            float dist2 = dot(d, d);

            float rt = max(fabs(shape[tid.x].w),
                           max(fabs(shape[tid.y].w), fabs(shape[tid.z].w)));
            float hSum = rv + rt;
            float3 velT = (velLin[tid.x].xyz + velLin[tid.y].xyz + velLin[tid.z].xyz) / 3.0f;
            float inflate = min(length(velV - velT) * P.dt, 0.3f * P.elemCellSize);
            float detect = hSum + COLLISION_MARGIN + inflate;
            if (dist2 > detect * detect) continue;

            float dist = sqrt(dist2);
            float3 triN = cross(b - a, c - a);
            float tnLen = length(triN);
            if (tnLen < 1e-12f) continue;
            triN /= tnLen;

            float side = dot(d, triN) >= 0.0f ? 1.0f : -1.0f;
            uint keyA = (SC_VT << 28) | v;
            uint keyB = t;
            // Sign memory only when the closest point is INTERIOR: crossing
            // the plane through the face is tunneling (eject back), but
            // sliding laterally around an edge is legitimate — pinning it
            // to the old side snags the cloth on curved surfaces.
            bool interior = bary.x > 0.02f && bary.y > 0.02f && bary.z > 0.02f;
            float sMem = interior
                ? softPrevSide(prevSoft, mapKeyA, mapKeyB, mapVal, P, keyA, keyB, side)
                : side;
            float3 n;
            float g;
            if (dist > 1e-7f && side == sMem) {
                n = d / dist;
                g = dist;
            } else {
                // crossed the plane while persistent: eject back to the
                // remembered side
                n = sMem * triN;
                g = -dist;
            }

            float fricT = (props[tid.x].w * bary.x + props[tid.y].w * bary.y
                           + props[tid.z].w * bary.z);
            float friction = sqrt(max(props[v].w * fricT, 0.0f));

            float minMass = massV > 0.0f ? massV : FLT_MAX;
            float mA = posLin[tid.x].w, mB = posLin[tid.y].w, mC = posLin[tid.z].w;
            if (mA > 0.0f) minMass = min(minMass, mA);
            if (mB > 0.0f) minMass = min(minMass, mB);
            if (mC > 0.0f) minMass = min(minMass, mC);
            if (minMass == FLT_MAX) continue;       // all static: inert

            softEmit(soft, counters, prevSoft, mapKeyA, mapKeyB, mapVal, P,
                     uint4(v, tid.x, tid.y, tid.z),
                     float4(1.0f, -bary.x, -bary.y, -bary.z),
                     n, friction, float3(0),
                     g - hSum + COLLISION_MARGIN,
                     softLambdaCap(P, minMass),
                     SC_VT, false, false, sMem < 0.0f, keyA, keyB);
            emitted++;
        }
    }
}

// ----------------------------------------------------------------------------
// Rigid feature points vs triangle: box corners, sphere center, capsule ends.
// One thread per TRIANGLE scanning the BODY grid (radius widened to cover the
// triangle's own extent) plus the global (oversized/static) list.
// ----------------------------------------------------------------------------
struct RTFeature {
    float3 world;       // feature point, world space
    float radius;       // skin radius carried by the feature (sphere/capsule)
    uint id;            // stable feature index for persistence
};

// Collect candidate feature points of body `o` that could touch a triangle
// with bounding sphere (center m, radius rT). Returns count (<= 4).
inline int rtFeatures(device const float4* posLin, device const float4* posAng,
                      device const float4* shape, device const uint* shapeType,
                      uint o, float3 m, float rT, thread RTFeature* out)
{
    uint st = shapeType[o] & SHAPE_KIND_MASK;
    float3 p = posLin[o].xyz;
    float4 q = posAng[o];
    int n = 0;
    if (st == 1) {                          // sphere
        out[0].world = p;
        out[0].radius = shape[o].x * 0.5f;
        out[0].id = 8;
        return 1;
    }
    if (st == 3) {                          // capsule: 3 spheres along the axis
        float half_ = shape[o].x * 0.5f;
        float r = shape[o].y;
        float3 ax = q_rotate(q, float3(0, 0, 1));
        for (int i = -1; i <= 1; i++) {
            float3 w = p + ax * (half_ * float(i));
            if (distance(w, m) > rT + r + 4.0f * COLLISION_MARGIN) continue;
            out[n].world = w;
            out[n].radius = r;
            out[n].id = uint(9 + i + 1);
            n++;
        }
        return n;
    }
    if (st == 2) return 0;                  // torus: node contacts cover it
    // box: 8 corners, pre-culled by distance to the triangle bound
    float3 h3 = shape[o].xyz * 0.5f;
    for (uint i = 0; i < 8; i++) {
        float3 local = float3(i & 1 ? h3.x : -h3.x,
                              i & 2 ? h3.y : -h3.y,
                              i & 4 ? h3.z : -h3.z);
        float3 w = xform(p, q, local);
        if (distance(w, m) > rT + 4.0f * COLLISION_MARGIN) continue;
        if (n < 4) {
            out[n].world = w;
            out[n].radius = 0.0f;
            out[n].id = i;
            n++;
        }
    }
    return n;
}

kernel void rt_emit(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device const float4* shape      [[buffer(2)]],
    device const float4* props      [[buffer(3)]],
    device const float4* velLin     [[buffer(4)]],
    device const uint* shapeType    [[buffer(5)]],
    device const uint4* tris        [[buffer(6)]],
    device const uint* cellStart    [[buffer(7)]],   // BODY grid
    device const uint* cellCount    [[buffer(8)]],
    device const uint* cellBodies   [[buffer(9)]],
    device const uint* globalIdx    [[buffer(10)]],
    device const uint2* exclusions  [[buffer(11)]],
    constant uint& numExclusions    [[buffer(12)]],
    device SoftContactGPU* soft     [[buffer(13)]],
    device atomic_uint* counters    [[buffer(14)]],
    device const SoftContactGPU* prevSoft [[buffer(15)]],
    device const atomic_uint* mapKeyA [[buffer(16)]],
    device const uint* mapKeyB      [[buffer(17)]],
    device const uint* mapVal       [[buffer(18)]],
    constant SimParams& P           [[buffer(19)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numTris) return;
    uint4 tid = tris[gid];
    float3 a = posLin[tid.x].xyz;
    float3 b = posLin[tid.y].xyz;
    float3 c = posLin[tid.z].xyz;
    float3 m = (a + b + c) / 3.0f;
    float rt_ = max(fabs(shape[tid.x].w),
                    max(fabs(shape[tid.y].w), fabs(shape[tid.z].w)));
    float rT = max(distance(m, a), max(distance(m, b), distance(m, c))) + rt_;
    float3 triN = cross(b - a, c - a);
    float tnLen = length(triN);
    if (tnLen < 1e-12f) return;
    triN /= tnLen;
    float3 velT = (velLin[tid.x].xyz + velLin[tid.y].xyz + velLin[tid.z].xyz) / 3.0f;

    // one thread owns each triangle: the cap is thread-local
    uint emitted = 0;

    // process one candidate rigid body against this triangle
    #define RT_TEST_BODY(o)                                                    \
    {                                                                          \
        float rb = fabs(shape[o].w);                                           \
        if (distance(posLin[o].xyz, m) <= rT + rb + COLLISION_MARGIN) {        \
            bool excl = pairExcluded(exclusions, numExclusions,                \
                                     min(o, tid.x), max(o, tid.x))             \
                     || pairExcluded(exclusions, numExclusions,                \
                                     min(o, tid.y), max(o, tid.y))             \
                     || pairExcluded(exclusions, numExclusions,                \
                                     min(o, tid.z), max(o, tid.z));            \
            if (!excl) {                                                       \
                RTFeature feats[4];                                            \
                int nf = rtFeatures(posLin, posAng, shape, shapeType,          \
                                    o, m, rT, feats);                          \
                for (int fi = 0; fi < nf && emitted < RT_MAX_PER_TRI; fi++) {  \
                    float3 bary;                                               \
                    float3 q = closestPtTriangle(feats[fi].world, a, b, c, bary); \
                    float3 d = feats[fi].world - q;                            \
                    float dist = length(d);                                    \
                    float hSum = rt_ + feats[fi].radius;                       \
                    float inflate = min(length(velLin[o].xyz - velT) * P.dt,   \
                                        0.5f * rT);                            \
                    if (dist - hSum > COLLISION_MARGIN + inflate) continue;    \
                    float side = dot(d, triN) >= 0.0f ? 1.0f : -1.0f;          \
                    uint keyA = (SC_RT << 28) | o;                             \
                    uint keyB = gid * 16u + feats[fi].id;                      \
                    float proposed = dot(posLin[o].xyz - q, triN) >= 0.0f      \
                                   ? 1.0f : -1.0f;                             \
                    bool interior = bary.x > 0.02f && bary.y > 0.02f           \
                                 && bary.z > 0.02f;                            \
                    float sMem = interior                                      \
                        ? softPrevSide(prevSoft, mapKeyA, mapKeyB,             \
                                       mapVal, P, keyA, keyB, proposed)        \
                        : proposed;                                            \
                    float3 n; float g;                                         \
                    if (dist > 1e-7f && side == sMem) { n = d / dist; g = dist; } \
                    else { n = sMem * triN; g = -dist; }                       \
                    float fricT = (props[tid.x].w + props[tid.y].w             \
                                   + props[tid.z].w) / 3.0f;                   \
                    float friction = sqrt(max(props[o].w * fricT, 0.0f));      \
                    float minMass = posLin[o].w > 0.0f ? posLin[o].w : FLT_MAX; \
                    float mA = posLin[tid.x].w, mB = posLin[tid.y].w,          \
                          mC = posLin[tid.z].w;                                \
                    if (mA > 0.0f) minMass = min(minMass, mA);                 \
                    if (mB > 0.0f) minMass = min(minMass, mB);                 \
                    if (mC > 0.0f) minMass = min(minMass, mC);                 \
                    if (minMass == FLT_MAX) continue;                          \
                    uint st = shapeType[o] & SHAPE_KIND_MASK;                  \
                    bool roundA = st != 0;                                     \
                    float3 anchor = roundA ? (feats[fi].world - posLin[o].xyz) \
                        : q_rotate(q_conj(posAng[o]), feats[fi].world - posLin[o].xyz); \
                    softEmit(soft, counters, prevSoft, mapKeyA, mapKeyB, mapVal, P, \
                             uint4(o, tid.x, tid.y, tid.z),                    \
                             float4(1.0f, -bary.x, -bary.y, -bary.z),          \
                             n, friction, anchor,                              \
                             g - hSum + COLLISION_MARGIN,                      \
                             softLambdaCap(P, minMass),                        \
                             SC_RT, true, roundA, sMem < 0.0f, keyA, keyB);    \
                    emitted++;                                                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

    // hashed bodies: widen the scan so big-body centers further than one cell
    // are still reached (cellSize = 2x max hashed radius; triangles add rT)
    int R = int(ceil((rT + 0.5f * P.cellSize) / P.cellSize));
    R = clamp(R, 1, 4);
    int3 cc = cellCoord(m, P.cellSize);
    for (int dz = -R; dz <= R && emitted < RT_MAX_PER_TRI; dz++)
    for (int dy = -R; dy <= R && emitted < RT_MAX_PER_TRI; dy++)
    for (int dx = -R; dx <= R && emitted < RT_MAX_PER_TRI; dx++) {
        uint h = cellHash(cc + int3(dx, dy, dz), P.gridHashSize);
        uint s = cellStart[h], e = s + cellCount[h];
        for (uint k = s; k < e && emitted < RT_MAX_PER_TRI; k++) {
            uint o = cellBodies[k];
            if (shape[o].w < 0.0f) continue;        // particles: V-T covers
            RT_TEST_BODY(o)
        }
    }
    // globals (oversized/static)
    for (uint gI = 0; gI < P.numGlobals && emitted < RT_MAX_PER_TRI; gI++) {
        uint o = globalIdx[gI];
        if (shape[o].w < 0.0f) continue;
        RT_TEST_BODY(o)
    }
    #undef RT_TEST_BODY
}

// ----------------------------------------------------------------------------
// Finalize: clamp the soft count and (re)write the adjacency/dual dispatch
// args so they cover element contacts. Overwrites what bp_finalize_pairs
// wrote — runs only when the scene has cloth elements.
// ----------------------------------------------------------------------------
kernel void soft_finalize(
    device atomic_uint* counters    [[buffer(0)]],
    device uint* dispatchArgs       [[buffer(1)]],
    constant SimParams& P           [[buffer(2)]])
{
    uint n = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    n = min(n, P.maxPairs);
    uint sc = atomic_load_explicit(&counters[CTR_SOFT], memory_order_relaxed);
    sc = min(sc, P.maxSoft);
    atomic_store_explicit(&counters[CTR_SOFT], sc, memory_order_relaxed);
    dispatchArgs[3] = (P.numJoints + P.numSprings + n + P.numTets + sc + 63) / 64;
    dispatchArgs[4] = 1;
    dispatchArgs[5] = 1;
    dispatchArgs[6] = (P.numJoints + P.numSprings + n + sc + 63) / 64;
    dispatchArgs[7] = 1;
    dispatchArgs[8] = 1;
}
