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
#define EE_MAX_PER_EDGE 4u
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
// Element binning: triangles + edges register in EVERY cell their AABB
// overlaps, at a DETECTION-radius cell size (coarse centroid cells made
// every query scan most of the sheet at high resolution: ~1000 candidate
// tests per vertex; AABB multi-cell brings it to ~10-30). Cover loops are
// clamped to 3x3x3 — the cell size is chosen at init so elements span at
// most ~2 cells per axis with stretch headroom.
// ----------------------------------------------------------------------------
inline void elemBounds(device const float4* posLin,
                       device const float4* shape,
                       device const uint4* tris,
                       device const uint2* edges,
                       constant SimParams& P, uint gid,
                       thread float3& lo, thread float3& hi)
{
    if (gid < P.numTris) {
        uint4 t = tris[gid];
        float3 a = posLin[t.x].xyz, b = posLin[t.y].xyz, c = posLin[t.z].xyz;
        float pad = max(fabs(shape[t.x].w),
                        max(fabs(shape[t.y].w), fabs(shape[t.z].w)))
                  + P.elemMargin;
        lo = min(min(a, b), c) - float3(pad);
        hi = max(max(a, b), c) + float3(pad);
    } else {
        uint2 e = edges[gid - P.numTris];
        float3 a = posLin[e.x].xyz, b = posLin[e.y].xyz;
        float pad = max(fabs(shape[e.x].w), fabs(shape[e.y].w)) + P.elemMargin;
        lo = min(a, b) - float3(pad);
        hi = max(a, b) + float3(pad);
    }
}

kernel void el_count(
    device const float4* posLin     [[buffer(0)]],
    device const uint4* tris        [[buffer(1)]],
    device const uint2* edges       [[buffer(2)]],
    device const float4* shape      [[buffer(3)]],
    device atomic_uint* cellCount   [[buffer(4)]],
    constant SimParams& P           [[buffer(5)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint total = P.numTris + P.numEdges;
    if (gid >= total) return;
    float3 lo, hi;
    elemBounds(posLin, shape, tris, edges, P, gid, lo, hi);
    int3 c0 = cellCoord(lo, P.elemCellSize);
    int3 c1 = min(cellCoord(hi, P.elemCellSize), c0 + 2);
    for (int z = c0.z; z <= c1.z; z++)
    for (int y = c0.y; y <= c1.y; y++)
    for (int x = c0.x; x <= c1.x; x++) {
        uint h = cellHash(int3(x, y, z), P.elemHashSize);
        atomic_fetch_add_explicit(&cellCount[h], 1u, memory_order_relaxed);
    }
}

kernel void el_scatter(
    device const float4* posLin     [[buffer(0)]],
    device const uint4* tris        [[buffer(1)]],
    device const uint2* edges       [[buffer(2)]],
    device const float4* shape      [[buffer(3)]],
    device atomic_uint* cellCursor  [[buffer(4)]],
    device uint* cellElems          [[buffer(5)]],
    constant SimParams& P           [[buffer(6)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint total = P.numTris + P.numEdges;
    if (gid >= total) return;
    float3 lo, hi;
    elemBounds(posLin, shape, tris, edges, P, gid, lo, hi);
    uint entry = gid < P.numTris ? gid : (ELEM_EDGE_BIT | (gid - P.numTris));
    int3 c0 = cellCoord(lo, P.elemCellSize);
    int3 c1 = min(cellCoord(hi, P.elemCellSize), c0 + 2);
    for (int z = c0.z; z <= c1.z; z++)
    for (int y = c0.y; y <= c1.y; y++)
    for (int x = c0.x; x <= c1.x; x++) {
        uint h = cellHash(int3(x, y, z), P.elemHashSize);
        uint slot = atomic_fetch_add_explicit(&cellCursor[h], 1u, memory_order_relaxed);
        cellElems[slot] = entry;
    }
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
                     float C0n, float restT, float lamCap, float minMass,
                     uint kind, bool rigidA, bool roundA, bool sideNeg,
                     uint keyA, uint keyB)
{
    float3 lam = float3(0);
    float3 pen = float3(0);
    int prev = softMapFind(mapKeyA, mapKeyB, mapVal, P.softMapCapacity, keyA, keyB);
    if (prev >= 0) {
        lam = prevSoft[prev].lambda.xyz * (P.alpha * P.gamma);
        pen = prevSoft[prev].penalty.xyz;
        // transport the tangential dual between the old and new bases:
        // contact normals rotate as cloth slides/folds, and a stale basis
        // misdirects the carried friction every frame
        float3 nOld = prevSoft[prev].normal.xyz;
        float3 t1o, t2o, t1n, t2n;
        orthonormal(nOld, t1o, t2o);
        orthonormal(n, t1n, t2n);
        float3 lt = t1o * lam.y + t2o * lam.z;
        lam.y = dot(t1n, lt);
        lam.z = dot(t2n, lt);
    }
    // Mass-aware floor: a brand-new contact at PENALTY_MIN=1 is invisible
    // to a gram particle being dragged through in one frame; m/dt^2 makes
    // the first iteration already deflect the approach without shock.
    float penFloor = max(PENALTY_MIN, minMass / (P.dt * P.dt));
    pen = clamp(pen * P.gamma, penFloor, PENALTY_MAX);

    uint slot = atomic_fetch_add_explicit(&counters[CTR_SOFT], 1u, memory_order_relaxed);
    if (slot >= P.maxSoft) return;

    uint flags = (rigidA ? SCF_RIGID_A : 0u)
               | (sideNeg ? SCF_SIDE_NEG : 0u)
               | (kind << SCF_KIND_SHIFT)
               | (roundA ? 32u : 0u);
    device SoftContactGPU& sc = soft[slot];
    sc.ids = ids;
    sc.normal = float4(n, friction);
    // anchorA.xyz is the rigid slot-0 anchor ONLY when SCF_RIGID_A; for
    // particle contacts .x carries the rest-target separation instead (the
    // 2-stage activation's barrier engages below tau = restT/2). C0.yz must
    // stay zero — softContactC consumes C0.xyz as the (n,t1,t2) vector and
    // a nonzero .y injects a phantom tangential rest-offset (constant
    // friction-direction creep force; blocks slid off each other).
    sc.anchorA = float4(rigidA ? anchorA : float3(restT, 0, 0),
                        as_type<float>(flags));
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
// V-T with VORONOI TEMPORAL TRACKING: each particle keeps its 4 closest
// triangles across frames. Per frame it re-evaluates a small candidate
// pool — the tracked set, the triangle-adjacency of the best, and the
// mesh-neighbors' best — and only consults the grid on staggered reseeds
// (1/8 of vertices per frame) or fast approaches. Folds move slowly per
// frame; propagation through topology discovers approaching surfaces the
// way the reference tracker does, at a fraction of a full grid scan.
// ----------------------------------------------------------------------------
struct Best4 {
    uint id[4];
    float d2[4];
};

inline void best4Init(thread Best4& b) {
    for (int i = 0; i < 4; i++) { b.id[i] = 0xFFFFFFFFu; b.d2[i] = FLT_MAX; }
}

inline bool best4Has(thread const Best4& b, uint t) {
    return b.id[0] == t || b.id[1] == t || b.id[2] == t || b.id[3] == t;
}

inline void best4Insert(thread Best4& b, uint t, float d2) {
    if (d2 >= b.d2[3]) return;
    int i = 3;
    for (; i > 0 && b.d2[i - 1] > d2; i--) { b.id[i] = b.id[i - 1]; b.d2[i] = b.d2[i - 1]; }
    b.id[i] = t; b.d2[i] = d2;
}

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
    device uint4* vtTrack           [[buffer(19)]],
    device const uint4* triAdj      [[buffer(20)]],
    device atomic_uint* bounds      [[buffer(21)]],   // d2 bits, atomic-min
    device const uint* nbr2Start    [[buffer(22)]],
    device const uint* nbr2Count    [[buffer(23)]],
    device const uint* nbr2List     [[buffer(24)]],
    device const uint* clothGroup   [[buffer(25)]],
    device const uint* clothVert    [[buffer(26)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numParticles) return;
    uint v = particleIdx[gid];
    float3 pv = posLin[v].xyz;
    float massV = posLin[v].w;
    float rv = fabs(shape[v].w);
    float3 velV = velLin[v].xyz;
    uint ns = nbrStart[v], ne = ns + nbrCount[v];
    uint gv = clothGroup[v];
    bool solidV = gv != 0 && clothVert[v] == 0;

    Best4 best;
    best4Init(best);

    // OGC conservative bounds (paper Eq 21-26) fall out of candidate
    // evaluation: every vertex-facet distance observed atomically mins
    // into the vertex's bound AND the facet's vertices' bounds — but only
    // for pairs OUTSIDE each other's 2-ring (near-geodesic same-sheet
    // elements sit ~2 edge-lengths away forever and would cap every bound
    // at a crawl; the thickness < 0.38*spacing invariant keeps them out of
    // contact range, so excluding them is safe). Non-negative float bits
    // order like uints, so atomic_fetch_min on bit patterns works.
    uint n2s = nbr2Start[v], n2e = n2s + nbr2Count[v];
    float boundsCap2 = P.elemCellSize * P.elemCellSize;
    float ownB2 = FLT_MAX;        // own-vertex bound: one atomic at the end

    // candidate evaluation: closest-point distance, with topological
    // exclusion (own / 1-ring triangles never count)
    #define VT_CONSIDER(T)                                                     \
    {                                                                          \
        uint t_ = (T);                                                         \
        if (t_ < P.numTris && !best4Has(best, t_)) {                           \
            uint4 tid_ = tris[t_];                                             \
            if (tid_.x != v && tid_.y != v && tid_.z != v) {                   \
                bool sameSolid_ = solidV && clothGroup[tid_.x] == gv            \
                    && clothGroup[tid_.y] == gv && clothGroup[tid_.z] == gv     \
                    && clothVert[tid_.x] == 0 && clothVert[tid_.y] == 0         \
                    && clothVert[tid_.z] == 0;                                  \
                if (sameSolid_) {                                               \
                } else {                                                        \
                float3 ba_;                                                    \
                float3 q_ = closestPtTriangle(pv, posLin[tid_.x].xyz,          \
                                              posLin[tid_.y].xyz,              \
                                              posLin[tid_.z].xyz, ba_);        \
                float d2_ = distance_squared(pv, q_);                          \
                if (d2_ < best.d2[3]                                           \
                    && !nbrContains(nbrList, ns, ne, tid_.x)                   \
                    && !nbrContains(nbrList, ns, ne, tid_.y)                   \
                    && !nbrContains(nbrList, ns, ne, tid_.z)) {                \
                    best4Insert(best, t_, d2_);                                \
                }                                                              \
                if (d2_ < boundsCap2                                           \
                    && !nbrContains(nbr2List, n2s, n2e, tid_.x)                \
                    && !nbrContains(nbr2List, n2s, n2e, tid_.y)                \
                    && !nbrContains(nbr2List, n2s, n2e, tid_.z)) {             \
                    uint db_ = as_type<uint>(d2_);                             \
                    ownB2 = min(ownB2, d2_);                                   \
                    atomic_fetch_min_explicit(&bounds[tid_.x], db_,            \
                                              memory_order_relaxed);           \
                    atomic_fetch_min_explicit(&bounds[tid_.y], db_,            \
                                              memory_order_relaxed);           \
                    atomic_fetch_min_explicit(&bounds[tid_.z], db_,            \
                                              memory_order_relaxed);           \
                }                                                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

    uint4 tracked = vtTrack[v];
    for (int i = 0; i < 4; i++) VT_CONSIDER(tracked[i])
    if (tracked.x < P.numTris) {
        uint4 adj = triAdj[tracked.x];
        for (int i = 0; i < 3; i++) VT_CONSIDER(adj[i])
    }
    // propagate from mesh neighbors' best tracked
    for (uint k = ns; k < ne && k < ns + 8; k++) {
        VT_CONSIDER(vtTrack[nbrList[k]].x)
    }

    // Staggered grid reseed. Distance triggers don't work here: on a flat
    // sheet the 2-ring sits ~2 edge-lengths away, so "something is always
    // close" and any d2 heuristic degenerates to a full scan every frame.
    // Absolute speed doesn't either: a sheet in freefall is fast but
    // carries its own element density with it (every scan is dense), with
    // zero collision risk. The rescan PERIOD scales with velocity RELATIVE
    // to the tracked best — freefall tracks its own 2-ring moving along
    // (1/8 rate), while impacts and layer approaches create local relative
    // motion exactly where fast rescans (1/2 rate) are needed.
    // Propagation + persistence carry the manifold between scans.
    float3 relV = velV;
    if (best.id[0] < P.numTris) {
        uint4 bt = tris[best.id[0]];
        relV -= (velLin[bt.x].xyz + velLin[bt.y].xyz + velLin[bt.z].xyz) / 3.0f;
    }
    float vThresh = 0.25f * P.elemCellSize / max(P.dt, 1e-6f);
    uint period = dot(relV, relV) > vThresh * vThresh ? 1u : 7u;
    bool missingTracked = best.id[0] == 0xFFFFFFFFu;
    bool reseed = ((v + P.frame) & period) == 0u
               || (missingTracked && !solidV)
               || (missingTracked && solidV && ((v + P.frame) & 3u) == 0u);
    if (reseed) {
        int3 cc = cellCoord(pv, P.elemCellSize);
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            uint h = cellHash(cc + int3(dx, dy, dz), P.elemHashSize);
            uint cs = elemCellStart[h], ce = cs + elemCellCount[h];
            for (uint k = cs; k < ce; k++) {
                uint entry = cellElems[k];
                if (entry & ELEM_EDGE_BIT) continue;
                VT_CONSIDER(entry)
            }
        }
    }
    vtTrack[v] = uint4(best.id[0], best.id[1], best.id[2], best.id[3]);
    #undef VT_CONSIDER
    if (ownB2 < FLT_MAX)
        atomic_fetch_min_explicit(&bounds[v], as_type<uint>(ownB2),
                                  memory_order_relaxed);

    // ---- Emission: OGC-aligned block semantics (paper Sec 3.2). Face
    // blocks push along ±n_t with a persistent side; boundary closest
    // points use the live radial direction (= the edge/vertex block
    // direction). Per-feature dedup was tried and REVERTED: facets sharing
    // a crease edge double-emit there, and that doubled eject force is
    // part of the equilibrium the gates were tuned against.
    for (int bi = 0; bi < 4; bi++) {
        uint t = best.id[bi];
        if (t >= P.numTris) break;
        uint4 tid = tris[t];
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
        float detect = hSum + P.elemMargin + inflate;
        if (dist2 > detect * detect) continue;
        float dist = sqrt(dist2);

        // F_OGC feature classification (paper Sec 3.2): interior closest
        // points are FACE-block contacts (force along ±n_t, side picked by
        // the persistent memory below); boundary closest points act as the
        // edge/vertex blocks — their live closest-point direction IS the
        // radial block direction, and the tri stencil's barycentric weights
        // already degenerate to the exact feature weights (a zero third
        // component on an edge, a single 1 at a vertex). Canonical
        // per-feature persistence keys were tried and REVERTED: they keep
        // dual/penalty state alive while a contact rotates around a corner,
        // which ratchets weave-tearing forces that facet-keyed contacts
        // shed on every facet flip (combined gate: rods pinned at cap).
        uint4 cids = uint4(v, tid.x, tid.y, tid.z);
        float4 wts = float4(1.0f, -bary.x, -bary.y, -bary.z);
        uint keyB = t;
        uint keyA = (SC_VT << 28) | v;

        // Block semantics (OGC Sec 3.2, matching what this solver already
        // did): deep-interior closest points are FACE-block contacts whose
        // force acts along ±n_t with the side picked by persistent memory
        // (the live direction flips when crossing; the block side does
        // not). Boundary closest points use the live radial direction —
        // exactly the edge/vertex block direction — with memory RELEASED
        // (the sphere-flank lesson: remembered sides snag lateral slides).
        // The TRUE side is stored every frame for ALL contacts: a contact
        // wandering boundary -> interior must inherit its real side, not a
        // default that ejects it through the sheet.
        float3 triN = cross(b - a, c - a);
        float tnLen = length(triN);
        if (tnLen < 1e-12f) continue;
        triN /= tnLen;
        float side = dot(d, triN) >= 0.0f ? 1.0f : -1.0f;
        bool deepInterior = bary.x > 0.02f && bary.y > 0.02f
                         && bary.z > 0.02f;
        float sMem = deepInterior
            ? softPrevSide(prevSoft, mapKeyA, mapKeyB, mapVal, P,
                           keyA, keyB, side)
            : side;
        float3 n;
        float g;
        if (dist > 1e-7f && side == sMem) {
            n = d / dist;
            g = dist;
        } else {
            n = sMem * triN;
            g = -dist;
        }
        bool sideNeg = sMem < 0.0f;

        float fricT = (props[tid.x].w * bary.x + props[tid.y].w * bary.y
                       + props[tid.z].w * bary.z);
        float friction = sqrt(max(props[v].w * fricT, 0.0f));

        float minMass = massV > 0.0f ? massV : FLT_MAX;
        for (int k2 = 1; k2 < 4; k2++) {
            uint bId = cids[k2];
            if (bId == WORLD_BODY) continue;
            float m = posLin[bId].w;
            if (m > 0.0f) minMass = min(minMass, m);
        }
        if (minMass == FLT_MAX) continue;       // all static: inert

        softEmit(soft, counters, prevSoft, mapKeyA, mapKeyB, mapVal, P,
                 cids, wts, n, friction, float3(0),
                 g - hSum + P.elemMargin,
                 hSum - P.elemMargin,
                 softLambdaCap(P, minMass), minMass,
                 SC_VT, false, false, sideNeg, keyA, keyB);
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// E-E with the same Voronoi temporal tracking: each edge keeps its 4
// closest edges, propagates through endpoint-incident edges, reseeds from
// the grid 1/8 per frame. Interior-only (endpoints are V-T territory),
// 1.5-ring exclusion, normal-continuity crossing memory.
// ----------------------------------------------------------------------------
inline void eeClosestSegSeg(float3 p0, float3 p1, float3 q0, float3 q1,
                            thread float& s, thread float& t)
{
    float3 d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
    float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    s = 0.0f; t = 0.0f;
    if (a <= 1e-12f && e <= 1e-12f) return;
    if (a <= 1e-12f) { t = clamp(f / e, 0.0f, 1.0f); return; }
    float c = dot(d1, r);
    if (e <= 1e-12f) { s = clamp(-c / a, 0.0f, 1.0f); return; }
    float b = dot(d1, d2);
    float denom = a * e - b * b;
    if (fabs(denom) > 1e-12f) s = clamp((b * f - c * e) / denom, 0.0f, 1.0f);
    t = (b * s + f) / e;
    if (t < 0.0f) { t = 0.0f; s = clamp(-c / a, 0.0f, 1.0f); }
    else if (t > 1.0f) { t = 1.0f; s = clamp((b - c) / a, 0.0f, 1.0f); }
}

kernel void ee_emit(
    device const float4* posLin     [[buffer(0)]],
    device const float4* shape      [[buffer(1)]],
    device const float4* props      [[buffer(2)]],
    device const float4* velLin     [[buffer(3)]],
    device const uint2* edges       [[buffer(4)]],
    device const uint* elemCellStart [[buffer(5)]],
    device const uint* elemCellCount [[buffer(6)]],
    device const uint* cellElems    [[buffer(7)]],
    device const uint* nbrStart     [[buffer(8)]],
    device const uint* nbrCount     [[buffer(9)]],
    device const uint* nbrList      [[buffer(10)]],
    device SoftContactGPU* soft     [[buffer(11)]],
    device atomic_uint* counters    [[buffer(12)]],
    device const SoftContactGPU* prevSoft [[buffer(13)]],
    device const atomic_uint* mapKeyA [[buffer(14)]],
    device const uint* mapKeyB      [[buffer(15)]],
    device const uint* mapVal       [[buffer(16)]],
    constant SimParams& P           [[buffer(17)]],
    device uint4* eeTrack           [[buffer(18)]],
    device const uint* vertEdgeStart [[buffer(19)]],
    device const uint* vertEdgeCount [[buffer(20)]],
    device const uint* vertEdgeList [[buffer(21)]],
    device atomic_uint* bounds      [[buffer(22)]],   // d2 bits, atomic-min
    device const uint* nbr2Start    [[buffer(23)]],
    device const uint* nbr2Count    [[buffer(24)]],
    device const uint* nbr2List     [[buffer(25)]],
    device const uint* clothGroup   [[buffer(26)]],
    device const uint* clothVert    [[buffer(27)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numEdges) return;
    uint2 eA = edges[gid];
    float3 a0 = posLin[eA.x].xyz, a1 = posLin[eA.y].xyz;
    float rA = max(fabs(shape[eA.x].w), fabs(shape[eA.y].w));
    float3 velA = (velLin[eA.x].xyz + velLin[eA.y].xyz) * 0.5f;
    uint nsx = nbrStart[eA.x], nex = nsx + nbrCount[eA.x];
    uint nsy = nbrStart[eA.y], ney = nsy + nbrCount[eA.y];
    uint gA = clothGroup[eA.x];
    bool solidA = gA != 0 && clothGroup[eA.y] == gA
        && clothVert[eA.x] == 0 && clothVert[eA.y] == 0;

    Best4 best;
    best4Init(best);

    // edge-edge minima feed all four endpoint bounds (2-ring excluded —
    // see vt_emit)
    uint n2sx = nbr2Start[eA.x], n2ex = n2sx + nbr2Count[eA.x];
    float boundsCap2 = P.elemCellSize * P.elemCellSize;
    float ownB2 = FLT_MAX;        // both endpoints' bound: atomics at the end
    #define EE_CONSIDER(E)                                                     \
    {                                                                          \
        uint e_ = (E);                                                         \
        if (e_ < P.numEdges && e_ != gid && !best4Has(best, e_)) {             \
            uint2 eB_ = edges[e_];                                             \
            if (eB_.x != eA.x && eB_.x != eA.y                                 \
                && eB_.y != eA.x && eB_.y != eA.y) {                           \
                bool sameSolid_ = solidA && clothGroup[eB_.x] == gA             \
                    && clothGroup[eB_.y] == gA                                  \
                    && clothVert[eB_.x] == 0 && clothVert[eB_.y] == 0;          \
                if (sameSolid_) {                                               \
                } else {                                                        \
                float s_, t_;                                                  \
                float3 b0_ = posLin[eB_.x].xyz, b1_ = posLin[eB_.y].xyz;       \
                eeClosestSegSeg(a0, a1, b0_, b1_, s_, t_);                     \
                float3 cA_ = a0 + (a1 - a0) * s_;                              \
                float3 cB_ = b0_ + (b1_ - b0_) * t_;                           \
                float d2_ = distance_squared(cA_, cB_);                        \
                if (d2_ < best.d2[3]                                           \
                    && !nbrContains(nbrList, nsx, nex, eB_.x)                  \
                    && !nbrContains(nbrList, nsx, nex, eB_.y)                  \
                    && !nbrContains(nbrList, nsy, ney, eB_.x)                  \
                    && !nbrContains(nbrList, nsy, ney, eB_.y)) {               \
                    best4Insert(best, e_, d2_);                                \
                }                                                              \
                if (d2_ < boundsCap2                                           \
                    && !nbrContains(nbr2List, n2sx, n2ex, eB_.x)               \
                    && !nbrContains(nbr2List, n2sx, n2ex, eB_.y)) {            \
                    uint db_ = as_type<uint>(d2_);                             \
                    ownB2 = min(ownB2, d2_);                                   \
                    atomic_fetch_min_explicit(&bounds[eB_.x], db_,             \
                                              memory_order_relaxed);           \
                    atomic_fetch_min_explicit(&bounds[eB_.y], db_,             \
                                              memory_order_relaxed);           \
                }                                                              \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

    uint4 tracked = eeTrack[gid];
    for (int i = 0; i < 4; i++) EE_CONSIDER(tracked[i])
    // propagate through endpoint-incident edges' best tracked
    for (int side = 0; side < 2; side++) {
        uint vv = side == 0 ? eA.x : eA.y;
        uint es = vertEdgeStart[vv], ec = min(vertEdgeCount[vv], 6u);
        for (uint k = es; k < es + ec; k++) {
            EE_CONSIDER(eeTrack[vertEdgeList[k]].x)
        }
    }

    // relative-velocity staggered reseed (see vt_emit for the reasoning)
    float3 relV = velA;
    if (best.id[0] < P.numEdges) {
        uint2 bt = edges[best.id[0]];
        relV -= (velLin[bt.x].xyz + velLin[bt.y].xyz) * 0.5f;
    }
    float vThresh = 0.25f * P.elemCellSize / max(P.dt, 1e-6f);
    uint period = dot(relV, relV) > vThresh * vThresh ? 1u : 7u;
    bool missingTracked = best.id[0] == 0xFFFFFFFFu;
    bool reseed = ((gid + P.frame) & period) == 0u
               || (missingTracked && !solidA)
               || (missingTracked && solidA && ((gid + P.frame) & 3u) == 0u);
    if (reseed) {
        float3 mid = (a0 + a1) * 0.5f;
        int3 cc = cellCoord(mid, P.elemCellSize);
        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            uint h = cellHash(cc + int3(dx, dy, dz), P.elemHashSize);
            uint s0 = elemCellStart[h], e0 = s0 + elemCellCount[h];
            for (uint k = s0; k < e0; k++) {
                uint entry = cellElems[k];
                if (!(entry & ELEM_EDGE_BIT)) continue;
                EE_CONSIDER(entry & ~ELEM_EDGE_BIT)
            }
        }
    }
    eeTrack[gid] = uint4(best.id[0], best.id[1], best.id[2], best.id[3]);
    #undef EE_CONSIDER
    if (ownB2 < FLT_MAX) {
        uint ob_ = as_type<uint>(ownB2);
        atomic_fetch_min_explicit(&bounds[eA.x], ob_, memory_order_relaxed);
        atomic_fetch_min_explicit(&bounds[eA.y], ob_, memory_order_relaxed);
    }

    // emit (each pair once: smaller index owns it)
    for (int bi = 0; bi < 4; bi++) {
        uint eBi = best.id[bi];
        if (eBi >= P.numEdges) break;
        if (eBi <= gid) continue;
        uint2 eB = edges[eBi];
        float3 b0 = posLin[eB.x].xyz, b1 = posLin[eB.y].xyz;
        float s, t;
        eeClosestSegSeg(a0, a1, b0, b1, s, t);
        if (s < EE_INTERIOR_EPS || s > 1.0f - EE_INTERIOR_EPS ||
            t < EE_INTERIOR_EPS || t > 1.0f - EE_INTERIOR_EPS) continue;

        float3 cA = a0 + (a1 - a0) * s;
        float3 cB = b0 + (b1 - b0) * t;
        float3 d = cA - cB;
        float dist = length(d);
        float rB = max(fabs(shape[eB.x].w), fabs(shape[eB.y].w));
        float hSum = rA + rB;
        float3 velB = (velLin[eB.x].xyz + velLin[eB.y].xyz) * 0.5f;
        float inflate = min(length(velA - velB) * P.dt, 0.3f * P.elemCellSize);
        if (dist - hSum > P.elemMargin + inflate) continue;
        if (dist < 1e-7f) continue;             // degenerate: let V-T handle

        // Edge blocks are radial (live closest-point direction), with
        // normal-continuity memory as crossing protection: the radial
        // direction flips if the edges pass through each other, the
        // remembered normal does not.
        float3 nGeo = d / dist;
        uint keyA = (SC_EE << 28) | gid;
        uint keyB = eBi;
        float3 n = nGeo;
        float g = dist;
        int prevEE = softMapFind(mapKeyA, mapKeyB, mapVal, P.softMapCapacity,
                                 keyA, keyB);
        if (prevEE >= 0 && dot(nGeo, prevSoft[prevEE].normal.xyz) < 0.0f) {
            n = -nGeo;
            g = -dist;
        }

        float fric = sqrt(max(
            (props[eA.x].w + props[eA.y].w) * 0.5f *
            (props[eB.x].w + props[eB.y].w) * 0.5f, 0.0f));
        float minMass = FLT_MAX;
        float m0 = posLin[eA.x].w, m1 = posLin[eA.y].w;
        float m2 = posLin[eB.x].w, m3 = posLin[eB.y].w;
        if (m0 > 0.0f) minMass = min(minMass, m0);
        if (m1 > 0.0f) minMass = min(minMass, m1);
        if (m2 > 0.0f) minMass = min(minMass, m2);
        if (m3 > 0.0f) minMass = min(minMass, m3);
        if (minMass == FLT_MAX) continue;

        softEmit(soft, counters, prevSoft, mapKeyA, mapKeyB, mapVal, P,
                 uint4(eA.x, eA.y, eB.x, eB.y),
                 float4(1.0f - s, s, -(1.0f - t), -t),
                 n, fric, float3(0),
                 g - hSum + P.elemMargin,
                 hSum - P.elemMargin,
                 softLambdaCap(P, minMass), minMass,
                 SC_EE, false, false, false, keyA, keyB);
    }
}

// ----------------------------------------------------------------------------
// OGC mid-step bound refresh (paper Alg 3): when enough vertices have hit
// their conservative bound, re-anchor X^prev at the LIVE positions and
// recompute bounds from the TRACKED sets. Counter-driven and dispatched
// INDIRECTLY, so iterations where nothing is bound-limited cost one tiny
// args kernel and an empty dispatch — and a freely falling sheet earns a
// fresh 0.45*d budget every few iterations instead of crawling.
// ----------------------------------------------------------------------------
kernel void ogc_refresh_args(
    device atomic_uint* counters    [[buffer(0)]],
    device uint* args               [[buffer(1)]],   // 3 uints: threadgroups
    constant SimParams& P           [[buffer(2)]])
{
    uint exceeded = atomic_load_explicit(&counters[CTR_OGC], memory_order_relaxed);
    // the paper's gamma_e = 1%: only refresh when a meaningful share of
    // vertices are bound-limited
    bool fire = exceeded * 100u > P.numParticles;
    args[0] = fire ? (P.numParticles + 63u) / 64u : 0u;
    args[1] = 1u;
    args[2] = 1u;
    atomic_store_explicit(&counters[CTR_OGC], 0u, memory_order_relaxed);
}

kernel void ogc_bounds_refresh(
    device const float4* posLin     [[buffer(0)]],
    device const uint* particleIdx  [[buffer(1)]],
    device const uint4* tris        [[buffer(2)]],
    device const uint2* edges       [[buffer(3)]],
    device const uint4* vtTrack     [[buffer(4)]],
    device const uint4* eeTrack     [[buffer(5)]],
    device const uint* vertEdgeStart [[buffer(6)]],
    device const uint* vertEdgeCount [[buffer(7)]],
    device const uint* vertEdgeList [[buffer(8)]],
    device uint* bounds             [[buffer(9)]],
    device float4* ogcPrev          [[buffer(10)]],
    constant SimParams& P           [[buffer(11)]],
    device const uint* nbr2Start    [[buffer(12)]],
    device const uint* nbr2Count    [[buffer(13)]],
    device const uint* nbr2List     [[buffer(14)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numParticles) return;
    uint v = particleIdx[gid];
    float3 pv = posLin[v].xyz;
    uint n2s = nbr2Start[v], n2e = n2s + nbr2Count[v];
    float dmin2 = 3.0e38f;
    uint4 tracked = vtTrack[v];
    for (int i = 0; i < 4; i++) {
        uint t = tracked[i];
        if (t >= P.numTris) break;
        uint4 tid = tris[t];
        if (nbrContains(nbr2List, n2s, n2e, tid.x)
            || nbrContains(nbr2List, n2s, n2e, tid.y)
            || nbrContains(nbr2List, n2s, n2e, tid.z)) continue;
        float3 ba;
        float3 q = closestPtTriangle(pv, posLin[tid.x].xyz, posLin[tid.y].xyz,
                                     posLin[tid.z].xyz, ba);
        dmin2 = min(dmin2, distance_squared(pv, q));
    }
    uint es = vertEdgeStart[v], ec = min(vertEdgeCount[v], 6u);
    for (uint k = es; k < es + ec; k++) {
        uint myE = vertEdgeList[k];
        uint other = eeTrack[myE].x;
        if (other >= P.numEdges) continue;
        uint2 eo = edges[other];
        if (nbrContains(nbr2List, n2s, n2e, eo.x)
            || nbrContains(nbr2List, n2s, n2e, eo.y)) continue;
        uint2 em = edges[myE];
        float3 a0 = posLin[em.x].xyz, a1 = posLin[em.y].xyz;
        float3 b0 = posLin[eo.x].xyz, b1 = posLin[eo.y].xyz;
        float s, t;
        eeClosestSegSeg(a0, a1, b0, b1, s, t);
        float3 cA = a0 + (a1 - a0) * s;
        float3 cB = b0 + (b1 - b0) * t;
        dmin2 = min(dmin2, distance_squared(cA, cB));
    }
    bounds[v] = as_type<uint>(dmin2);
    ogcPrev[v] = posLin[v];
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
                      uint o, float3 m, float rT, float collisionMargin,
                      thread RTFeature* out)
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
            if (distance(w, m) > rT + r + 4.0f * collisionMargin) continue;
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
        if (distance(w, m) > rT + 4.0f * collisionMargin) continue;
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
    device const uint* hashedRigidIdx [[buffer(20)]],
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
        /* surface-distance gate: center-distance is meaningless for huge   */\
        /* statics (the ground passes always and pays 8 corner tests/tri)   */\
        bool nearO = distance(posLin[o].xyz, m) <= rT + rb + P.collisionMargin; \
        if (nearO && (shapeType[o] & SHAPE_KIND_MASK) == 0u) {                 \
            float3 lm = q_rotate(q_conj(posAng[o]), m - posLin[o].xyz);        \
            float3 dq2 = max(fabs(lm) - shape[o].xyz * 0.5f, 0.0f);            \
            nearO = length(dq2) <= rT + P.collisionMargin + 0.05f;              \
        }                                                                      \
        if (nearO) {                                                           \
            bool excl = pairExcluded(exclusions, numExclusions,                \
                                     min(o, tid.x), max(o, tid.x))             \
                     || pairExcluded(exclusions, numExclusions,                \
                                     min(o, tid.y), max(o, tid.y))             \
                     || pairExcluded(exclusions, numExclusions,                \
                                     min(o, tid.z), max(o, tid.z));            \
            if (!excl) {                                                       \
                RTFeature feats[4];                                            \
                int nf = rtFeatures(posLin, posAng, shape, shapeType,          \
                                    o, m, rT, P.collisionMargin, feats);       \
                for (int fi = 0; fi < nf && emitted < RT_MAX_PER_TRI; fi++) {  \
                    float3 bary;                                               \
                    float3 q = closestPtTriangle(feats[fi].world, a, b, c, bary); \
                    float3 d = feats[fi].world - q;                            \
                    float dist = length(d);                                    \
                    float hSum = rt_ + feats[fi].radius;                       \
                    float inflate = min(length(velLin[o].xyz - velT) * P.dt,   \
                                        0.5f * rT);                            \
                    if (dist - hSum > P.elemMargin + inflate) continue;        \
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
                             g - hSum + P.elemMargin,                          \
                             hSum - P.elemMargin,                              \
                             softLambdaCap(P, minMass), minMass,               \
                             SC_RT, true, roundA, sMem < 0.0f, keyA, keyB);    \
                    emitted++;                                                 \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

    // hashed bodies. With only a handful of rigids hashed, the grid sweep is
    // pure overhead (27+ empty-cell probes per triangle): test them directly.
    if (P.numHashedRigid > 0 && P.numHashedRigid <= 8) {
        for (uint ri = 0; ri < P.numHashedRigid && emitted < RT_MAX_PER_TRI; ri++) {
            uint o = hashedRigidIdx[ri];
            RT_TEST_BODY(o)
        }
    } else if (P.numHashedRigid > 0) {
        // widen the scan so big-body centers further than one cell are still
        // reached (cellSize = 2x max hashed radius; triangles add rT)
        int R = clamp(int(ceil((rT + 0.5f * P.cellSize) / P.cellSize)), 1, 4);
        int3 cc = cellCoord(m, P.cellSize);
        for (int dz = -R; dz <= R && emitted < RT_MAX_PER_TRI; dz++)
        for (int dy = -R; dy <= R && emitted < RT_MAX_PER_TRI; dy++)
        for (int dx = -R; dx <= R && emitted < RT_MAX_PER_TRI; dx++) {
            uint h = cellHash(cc + int3(dx, dy, dz), P.gridHashSize);
            uint s = cellStart[h], e = s + cellCount[h];
            for (uint k = s; k < e && emitted < RT_MAX_PER_TRI; k++) {
                uint o = cellBodies[k];
                if (shape[o].w < 0.0f) continue;    // particles: V-T covers
                RT_TEST_BODY(o)
            }
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
    atomic_store_explicit(&counters[CTR_SOFT_CANDIDATES], sc,
                          memory_order_relaxed);
    sc = min(sc, P.maxSoft);
    atomic_store_explicit(&counters[CTR_SOFT], sc, memory_order_relaxed);
    dispatchArgs[3] = (P.numJoints + P.numSprings + n + P.numTets + sc
                       + P.numMembranes + P.numBends + 63) / 64;
    dispatchArgs[4] = 1;
    dispatchArgs[5] = 1;
    dispatchArgs[6] = (P.numJoints + P.numSprings + n + sc + 63) / 64;
    dispatchArgs[7] = 1;
    dispatchArgs[8] = 1;
}
