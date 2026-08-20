// Direction-aware Planar Divide-and-Truncate for deformable surface
// vertex-triangle and edge-edge motion (arXiv:2604.15513). OGC remains the
// contact force model; this replaces only its isotropic displacement bound.
//
// This is a clean Metal implementation informed by the paper and Newton's
// Apache-2.0 VBD implementation. Important differences from Newton are a
// compact global pair stream and fail-closed overflow handling.

#define DAT_GEOMETRY_EPS 1.0e-8f
#define DAT_TIME_MARGIN 1.0e-3f
#define DAT_QUERY_THREADS 64u

inline bool datCellInRange(int3 c, int3 lo, int3 hi) {
    return all(c >= lo) && all(c <= hi);
}

inline void datAppendPair(device PlanarDATPairGPU* pairs,
                          device atomic_uint* pairCounts,
                          device atomic_uint* counters,
                          constant SimParams& P,
                          uint kind, uint owner, uint other,
                          uint flags) {
    uint slot = atomic_fetch_add_explicit(&pairCounts[0], 1u,
                                           memory_order_relaxed);
    atomic_fetch_add_explicit(&pairCounts[kind + 1u], 1u,
                              memory_order_relaxed);
    if (slot < P.maxPlanarDATPairs)
        pairs[slot].data = uint4(kind, owner, other, flags);
}

// Exact V-T rq query. It runs at both the step-start and accepted predictor
// poses. Between queries each particle moves at most R=0.5*gamma*rq, so a
// pair omitted at the start (gap > rq) cannot cross before the second query.
// Rechecking exact distance rejects hash aliases and keeps the stream compact.
kernel void dat_build_vt_pairs(
    device const float4* posLin      [[buffer(0)]],
    device const uint* particleIdx   [[buffer(1)]],
    device const uint4* tris         [[buffer(2)]],
    device const uint2* edges        [[buffer(19)]],
    device const uint* elemCellStart [[buffer(3)]],
    device const uint* elemCellCount [[buffer(4)]],
    device const uint* cellElems     [[buffer(5)]],
    device const uint* nbrStart      [[buffer(6)]],
    device const uint* nbrCount      [[buffer(7)]],
    device const uint* nbrList       [[buffer(8)]],
    device const float4* shape       [[buffer(9)]],
    device const uint* clothGroup    [[buffer(10)]],
    device const uint* selfCollision [[buffer(11)]],
    device PlanarDATPairGPU* pairs   [[buffer(12)]],
    device atomic_uint* pairCounts   [[buffer(13)]],
    device atomic_uint* counters     [[buffer(14)]],
    constant SimParams& P            [[buffer(15)]],
    uint3 groupPosition              [[threadgroup_position_in_grid]],
    uint lane                        [[thread_index_in_threadgroup]])
{
    uint gid = groupPosition.x;
    if (gid >= P.numParticles) return;
    uint v = particleIdx[gid];
    float3 p = posLin[v].xyz;
    if (!finite3(p)) {
        if (lane == 0u) {
            atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                      memory_order_relaxed);
        }
        return;
    }
    int3 cell = cellCoord(p, P.elemCellSize);
    uint h = cellHash(cell, P.elemHashSize);
    uint s = elemCellStart[h], e = s + elemCellCount[h];
    uint ns = nbrStart[v], ne = ns + nbrCount[v];
    float query2 = P.planarDATQueryRadius * P.planarDATQueryRadius;

    // One threadgroup owns a query vertex. Lanes cooperatively scan its hash
    // bucket instead of serializing every candidate on one GPU thread.
    for (uint k = s + lane; k < e; k += DAT_QUERY_THREADS) {
        uint entry = cellElems[k];
        if ((entry & ELEM_EDGE_BIT) != 0u || entry >= P.numTris) continue;
        uint4 t = tris[entry];
        if (t.x == v || t.y == v || t.z == v) continue;
        uint group = clothGroup[v];
        bool sameComponent = group != 0u
            && clothGroup[t.x] == group && clothGroup[t.y] == group
            && clothGroup[t.z] == group;
        if (sameComponent && selfCollision[v] == 0u) continue;
        if (sameComponent
            && (nbrContains(nbrList, ns, ne, t.x)
                || nbrContains(nbrList, ns, ne, t.y)
                || nbrContains(nbrList, ns, ne, t.z))) continue;
        float3 lo, hi;
        elemBounds(posLin, shape, tris, edges, P, entry, lo, hi);
        int3 tc0 = cellCoord(lo, P.elemCellSize);
        int3 tc1 = min(cellCoord(hi, P.elemCellSize), tc0 + 3);
        if (!datCellInRange(cell, tc0, tc1)) continue;
        float3 bary;
        float3 q = closestPtTriangle(p, posLin[t.x].xyz,
                                     posLin[t.y].xyz,
                                     posLin[t.z].xyz, bary);
        float3 d = p - q;
        if (dot(d, d) > query2) continue;
        uint flags = DAT_PAIR_TRUNCATE | DAT_PAIR_CONTACT;
        datAppendPair(pairs, pairCounts, counters, P,
                      DAT_PAIR_VT, v, entry, flags);
    }
}

// E-E uses the same exact rq neighborhood. A pair is owned by A<B and emitted only from the
// minimum exact cell in tight(A) intersect expanded(B), eliminating multi-cell
// and hash-alias duplicates without a per-thread candidate cap.
kernel void dat_build_ee_pairs(
    device const float4* posLin      [[buffer(0)]],
    device const uint2* edges        [[buffer(1)]],
    device const uint4* tris         [[buffer(2)]],
    device const uint* elemCellStart [[buffer(3)]],
    device const uint* elemCellCount [[buffer(4)]],
    device const uint* cellElems     [[buffer(5)]],
    device const uint* nbrStart      [[buffer(6)]],
    device const uint* nbrCount      [[buffer(7)]],
    device const uint* nbrList       [[buffer(8)]],
    device const float4* shape       [[buffer(9)]],
    device const uint* clothGroup    [[buffer(10)]],
    device const uint* selfCollision [[buffer(11)]],
    device PlanarDATPairGPU* pairs   [[buffer(12)]],
    device atomic_uint* pairCounts   [[buffer(13)]],
    device atomic_uint* counters     [[buffer(14)]],
    constant SimParams& P            [[buffer(15)]],
    uint3 groupPosition              [[threadgroup_position_in_grid]],
    uint lane                        [[thread_index_in_threadgroup]])
{
    uint gid = groupPosition.x;
    if (gid >= P.numEdges) return;
    uint2 a = edges[gid];
    float3 a0 = posLin[a.x].xyz, a1 = posLin[a.y].xyz;
    if (!finite3(a0) || !finite3(a1)) {
        if (lane == 0u) {
            atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                      memory_order_relaxed);
        }
        return;
    }
    int3 ac0 = cellCoord(min(a0, a1), P.elemCellSize);
    int3 rawAC1 = cellCoord(max(a0, a1), P.elemCellSize);
    if (lane == 0u && any(rawAC1 - ac0 > int3(3))) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_GRID_OVERFLOW], 1u,
                                  memory_order_relaxed);
    }
    int3 ac1 = min(rawAC1, ac0 + 3);
    uint ns0 = nbrStart[a.x], ne0 = ns0 + nbrCount[a.x];
    uint ns1 = nbrStart[a.y], ne1 = ns1 + nbrCount[a.y];
    float query2 = P.planarDATQueryRadius * P.planarDATQueryRadius;

    for (int z = ac0.z; z <= ac1.z; ++z)
    for (int y = ac0.y; y <= ac1.y; ++y)
    for (int x = ac0.x; x <= ac1.x; ++x) {
        int3 cell = int3(x, y, z);
        uint h = cellHash(cell, P.elemHashSize);
        uint s = elemCellStart[h], e = s + elemCellCount[h];
        // All lanes visit the same exact A cell and split its candidate
        // bucket. Canonical overlap-cell ownership still emits each pair once.
        for (uint k = s + lane; k < e; k += DAT_QUERY_THREADS) {
            uint entry = cellElems[k];
            if ((entry & ELEM_EDGE_BIT) == 0u) continue;
            uint other = entry & ~ELEM_EDGE_BIT;
            if (other <= gid || other >= P.numEdges) continue;
            uint2 b = edges[other];
            if (b.x == a.x || b.x == a.y || b.y == a.x || b.y == a.y)
                continue;
            uint group = clothGroup[a.x];
            bool sameComponent = group != 0u
                && clothGroup[a.y] == group
                && clothGroup[b.x] == group && clothGroup[b.y] == group;
            if (sameComponent && selfCollision[a.x] == 0u) continue;
            if (sameComponent
                && (nbrContains(nbrList, ns0, ne0, b.x)
                    || nbrContains(nbrList, ns0, ne0, b.y)
                    || nbrContains(nbrList, ns1, ne1, b.x)
                    || nbrContains(nbrList, ns1, ne1, b.y))) continue;
            float3 blo, bhi;
            elemBounds(posLin, shape, tris, edges, P,
                       P.numTris + other, blo, bhi);
            int3 bc0 = cellCoord(blo, P.elemCellSize);
            int3 bc1 = min(cellCoord(bhi, P.elemCellSize), bc0 + 3);
            int3 overlap0 = max(ac0, bc0);
            int3 overlap1 = min(ac1, bc1);
            if (any(overlap0 > overlap1) || any(cell != overlap0)) continue;

            float sb, tb;
            float3 b0 = posLin[b.x].xyz, b1 = posLin[b.y].xyz;
            eeClosestSegSeg(a0, a1, b0, b1, sb, tb);
            float3 ca = mix(a0, a1, sb), cb = mix(b0, b1, tb);
            float3 d = ca - cb;
            if (dot(d, d) > query2) continue;
            uint flags = DAT_PAIR_TRUNCATE;
            if ((gid < P.numSurfaceContactEdges
                 && other < P.numSurfaceContactEdges)
                || (sameComponent && selfCollision[a.x] != 0u)) {
                flags |= DAT_PAIR_CONTACT;
            }
            datAppendPair(pairs, pairCounts, counters, P,
                          DAT_PAIR_EE, gid, other, flags);
        }
    }
}

kernel void dat_finalize_pairs(
    device const atomic_uint* pairCounts [[buffer(0)]],
    device atomic_uint* counters         [[buffer(1)]],
    device uint* args                     [[buffer(2)]],
    constant SimParams& P                [[buffer(3)]],
    uint gid                             [[thread_position_in_grid]])
{
    if (gid != 0u) return;
    uint total = atomic_load_explicit(&pairCounts[0], memory_order_relaxed);
    uint vt = atomic_load_explicit(&pairCounts[1], memory_order_relaxed);
    uint ee = atomic_load_explicit(&pairCounts[2], memory_order_relaxed);
    atomic_fetch_max_explicit(&counters[CTR_DAT_PAIRS], total,
                              memory_order_relaxed);
    atomic_fetch_max_explicit(&counters[CTR_DAT_VT_PAIRS], vt,
                              memory_order_relaxed);
    atomic_fetch_max_explicit(&counters[CTR_DAT_EE_PAIRS], ee,
                              memory_order_relaxed);
    uint stored = min(total, P.maxPlanarDATPairs);
    args[0] = (stored + 63u) / 64u;
    args[1] = 1u;
    args[2] = 1u;
}

inline uint4 datPairBodies(PlanarDATPairGPU pair,
                           device const uint4* tris,
                           device const uint2* edges) {
    if (pair.data.x == DAT_PAIR_VT) {
        uint4 tri = tris[pair.data.z];
        return uint4(pair.data.y, tri.x, tri.y, tri.z);
    }
    uint2 a = edges[pair.data.y], b = edges[pair.data.z];
    return uint4(a.x, a.y, b.x, b.y);
}

// Build one exact accepted-pair CSR over participating particles. Every
// retained V-T/E-E pair has four distinct participants after topology
// filtering, so storage is exactly bounded by 4*pair capacity.
kernel void dat_incidence_count(
    device const PlanarDATPairGPU* pairs [[buffer(0)]],
    device const atomic_uint* pairCounts [[buffer(1)]],
    device atomic_uint* bodyCounts       [[buffer(2)]],
    device const uint4* tris             [[buffer(3)]],
    device const uint2* edges            [[buffer(4)]],
    constant SimParams& P                [[buffer(5)]],
    uint gid                             [[thread_position_in_grid]])
{
    uint count = min(atomic_load_explicit(&pairCounts[0],
                                           memory_order_relaxed),
                     P.maxPlanarDATPairs);
    if (gid >= count) return;
    PlanarDATPairGPU pair = pairs[gid];
    if ((pair.data.w & DAT_PAIR_TRUNCATE) == 0u) return;
    uint4 ids = datPairBodies(pair, tris, edges);
    for (uint i = 0u; i < 4u; ++i) {
        atomic_fetch_add_explicit(&bodyCounts[ids[i]], 1u,
                                  memory_order_relaxed);
    }
}

kernel void dat_incidence_scatter(
    device const PlanarDATPairGPU* pairs [[buffer(0)]],
    device const atomic_uint* pairCounts [[buffer(1)]],
    device atomic_uint* bodyCursor       [[buffer(2)]],
    device uint* bodyPairs               [[buffer(3)]],
    device const uint4* tris             [[buffer(4)]],
    device const uint2* edges            [[buffer(5)]],
    constant SimParams& P                [[buffer(6)]],
    uint gid                             [[thread_position_in_grid]])
{
    uint count = min(atomic_load_explicit(&pairCounts[0],
                                           memory_order_relaxed),
                     P.maxPlanarDATPairs);
    if (gid >= count) return;
    PlanarDATPairGPU pair = pairs[gid];
    if ((pair.data.w & DAT_PAIR_TRUNCATE) == 0u) return;
    uint4 ids = datPairBodies(pair, tris, edges);
    for (uint i = 0u; i < 4u; ++i) {
        uint slot = atomic_fetch_add_explicit(&bodyCursor[ids[i]], 1u,
                                              memory_order_relaxed);
        bodyPairs[slot] = gid;
    }
}

inline float3 datVTFallbackNormal(float3 a, float3 b, float3 c,
                                  float3 relativeMotion,
                                  thread bool& valid) {
    float3 raw = cross(b - a, c - a);
    float magnitude = length(raw);
    valid = finite_bits(magnitude) && magnitude >= DAT_GEOMETRY_EPS;
    if (!valid) return float3(0.0f);
    float3 n = raw / magnitude;
    if (dot(n, relativeMotion) > 0.0f) n = -n;
    return n;
}

inline float3 datEEFallbackNormal(float3 a0, float3 a1,
                                  float3 b0, float3 b1,
                                  float3 relativeMotion,
                                  thread bool& valid) {
    float3 ua = a1 - a0, ub = b1 - b0;
    float3 raw = cross(ua, ub);
    float magnitude = length(raw);
    float3 n;
    if (finite_bits(magnitude) && magnitude >= DAT_GEOMETRY_EPS) {
        n = raw / magnitude;
    } else {
        float la = length(ua), lb = length(ub);
        float3 u = la >= lb ? ua : ub;
        float ul = max(la, lb);
        if (!finite_bits(ul) || ul < DAT_GEOMETRY_EPS) {
            valid = false;
            return float3(0.0f);
        }
        u /= ul;
        float3 perpendicular = relativeMotion
            - u * dot(u, relativeMotion);
        float pl = length(perpendicular);
        if (!finite_bits(pl) || pl < DAT_GEOMETRY_EPS) {
            float3 axis = fabs(u.x) < 0.7f ? float3(1, 0, 0)
                                           : float3(0, 1, 0);
            perpendicular = cross(u, axis);
            pl = length(perpendicular);
        }
        if (!finite_bits(pl) || pl < DAT_GEOMETRY_EPS) {
            valid = false;
            return float3(0.0f);
        }
        n = perpendicular / pl;
    }
    if (dot(n, relativeMotion) > 0.0f) n = -n;
    valid = finite3(n);
    return valid ? n : float3(0.0f);
}

// Pair-parallel OGC contact emission. This uses the exact same full nearby
// pair stream as DAT rather than the old best-four tracker.
kernel void dat_emit_contacts(
    device const PlanarDATPairGPU* pairs [[buffer(0)]],
    device const float4* posLin          [[buffer(1)]],
    device const float4* shape           [[buffer(2)]],
    device const float4* props           [[buffer(3)]],
    device const float4* velLin          [[buffer(4)]],
    device const uint4* tris             [[buffer(5)]],
    device const uint2* edges            [[buffer(6)]],
    device SoftContactGPU* soft          [[buffer(7)]],
    device atomic_uint* counters         [[buffer(8)]],
    device const SoftContactGPU* prevSoft [[buffer(9)]],
    device const atomic_uint* mapKeyA    [[buffer(10)]],
    device const uint* mapKeyB           [[buffer(11)]],
    device const uint* mapVal            [[buffer(12)]],
    constant SimParams& P                [[buffer(13)]],
    device const atomic_uint* pairCounts [[buffer(14)]],
    device const float4* massState        [[buffer(15)]],
    device const float4* referencePos     [[buffer(16)]],
    uint gid                             [[thread_position_in_grid]])
{
    uint count = min(atomic_load_explicit(&pairCounts[0],
                                           memory_order_relaxed),
                     P.maxPlanarDATPairs);
    if (gid >= count) return;
    uint4 pair = pairs[gid].data;
    if ((pair.w & DAT_PAIR_CONTACT) == 0u) return;
    if (pair.x == DAT_PAIR_VT) {
        uint v = pair.y, ti = pair.z;
        if (ti >= P.numTris) return;
        uint4 t = tris[ti];
        float3 p = posLin[v].xyz;
        float3 a = posLin[t.x].xyz, b = posLin[t.y].xyz,
               c = posLin[t.z].xyz;
        float3 bary;
        float3 q = closestPtTriangle(p, a, b, c, bary);
        float3 d = p - q;
        float dist2 = dot(d, d);
        float rv = fabs(shape[v].w);
        float rt = max(fabs(shape[t.x].w),
                       max(fabs(shape[t.y].w), fabs(shape[t.z].w)));
        float hSum = rv + rt;
        float3 velT = (velLin[t.x].xyz + velLin[t.y].xyz
                       + velLin[t.z].xyz) / 3.0f;
        float inflate = min(length(velLin[v].xyz - velT) * P.dt,
                            0.3f * P.surfaceContactCellSize);
        // The shared safety stream is complete only through rq. Never let
        // speculative force inflation claim a wider contact band than the
        // candidate source that feeds it.
        float detect = min(hSum + P.elemMargin + inflate,
                           P.planarDATQueryRadius);
        if (dist2 > detect * detect) return;
        float dist = sqrt(dist2);
        float3 triN = cross(b - a, c - a);
        float nl = length(triN);
        if (nl < DAT_GEOMETRY_EPS) return;
        triN /= nl;
        float side = dot(d, triN) >= 0.0f ? 1.0f : -1.0f;
        uint keyA = (SC_VT << 28) | v;
        uint keyB = ti;
        bool interior = bary.x > 0.02f && bary.y > 0.02f
                     && bary.z > 0.02f;
        float remembered = interior
            ? softPrevSide(prevSoft, mapKeyA, mapKeyB, mapVal, P,
                           keyA, keyB, side) : side;
        float3 n;
        float gap;
        if (dist > 1.0e-7f && side == remembered) {
            n = d / dist;
            gap = dist;
        } else {
            n = remembered * triN;
            gap = -dist;
        }
        float fricT = props[t.x].w * bary.x + props[t.y].w * bary.y
                    + props[t.z].w * bary.z;
        float friction = sqrt(max(props[v].w * fricT, 0.0f));
        float minMass = massState[v].w > 0.0f ? massState[v].w : FLT_MAX;
        uint4 ids = uint4(v, t.x, t.y, t.z);
        for (int j = 1; j < 4; ++j) {
            float m = massState[ids[j]].w;
            if (m > 0.0f) minMass = min(minMass, m);
        }
        if (minMass == FLT_MAX) return;
        float3 dq = (p - referencePos[v].xyz)
            - bary.x * (a - referencePos[t.x].xyz)
            - bary.y * (b - referencePos[t.y].xyz)
            - bary.z * (c - referencePos[t.z].xyz);
        float Cdetect = gap - hSum + P.elemMargin;
        float C0n = Cdetect - dot(n, dq);
        softEmit(soft, counters, prevSoft, mapKeyA, mapKeyB, mapVal, P,
                 ids, float4(1.0f, -bary.x, -bary.y, -bary.z),
                 n, friction, float3(0), C0n,
                 hSum - P.elemMargin, softLambdaCap(P, minMass), minMass,
                 SC_VT, false, false, remembered < 0.0f, keyA, keyB);
        return;
    }

    uint ea = pair.y, eb = pair.z;
    if (ea >= P.numEdges || eb >= P.numEdges) return;
    uint2 aIds = edges[ea], bIds = edges[eb];
    float3 a0 = posLin[aIds.x].xyz, a1 = posLin[aIds.y].xyz;
    float3 b0 = posLin[bIds.x].xyz, b1 = posLin[bIds.y].xyz;
    float s, t;
    eeClosestSegSeg(a0, a1, b0, b1, s, t);
    if (s < EE_INTERIOR_EPS || s > 1.0f - EE_INTERIOR_EPS
        || t < EE_INTERIOR_EPS || t > 1.0f - EE_INTERIOR_EPS) return;
    float3 ca = a0 + (a1 - a0) * s;
    float3 cb = b0 + (b1 - b0) * t;
    float3 d = ca - cb;
    float dist = length(d);
    float rA = max(fabs(shape[aIds.x].w), fabs(shape[aIds.y].w));
    float rB = max(fabs(shape[bIds.x].w), fabs(shape[bIds.y].w));
    float hSum = rA + rB;
    float3 va = (velLin[aIds.x].xyz + velLin[aIds.y].xyz) * 0.5f;
    float3 vb = (velLin[bIds.x].xyz + velLin[bIds.y].xyz) * 0.5f;
    float inflate = min(length(va - vb) * P.dt,
                        0.3f * P.surfaceContactCellSize);
    float detect = min(hSum + P.elemMargin + inflate,
                       P.planarDATQueryRadius);
    if (dist > detect) return;
    uint keyA = (SC_EE << 28) | ea;
    uint keyB = eb;
    int previous = softMapFind(mapKeyA, mapKeyB, mapVal,
                               P.softMapCapacity, keyA, keyB);
    float3 n;
    float gap = dist;
    if (finite_bits(dist) && dist >= 1.0e-7f) {
        float3 nGeo = d / dist;
        n = nGeo;
        if (previous >= 0
            && dot(nGeo, prevSoft[previous].normal.xyz) < 0.0f) {
            n = -nGeo;
            gap = -dist;
        }
    } else if (previous >= 0
               && finite3(prevSoft[previous].normal.xyz)) {
        n = prevSoft[previous].normal.xyz;
        gap = 0.0f;
    } else {
        bool validFallback;
        n = datEEFallbackNormal(a0, a1, b0, b1, va - vb,
                                validFallback);
        if (!validFallback) return;
        gap = 0.0f;
    }
    float friction = sqrt(max(
        (props[aIds.x].w + props[aIds.y].w) * 0.5f
        * (props[bIds.x].w + props[bIds.y].w) * 0.5f, 0.0f));
    float minMass = FLT_MAX;
    uint4 ids = uint4(aIds.x, aIds.y, bIds.x, bIds.y);
    for (int j = 0; j < 4; ++j) {
        float m = massState[ids[j]].w;
        if (m > 0.0f) minMass = min(minMass, m);
    }
    if (minMass == FLT_MAX) return;
    float3 dq = (1.0f - s) * (a0 - referencePos[aIds.x].xyz)
              + s * (a1 - referencePos[aIds.y].xyz)
              - (1.0f - t) * (b0 - referencePos[bIds.x].xyz)
              - t * (b1 - referencePos[bIds.y].xyz);
    float Cdetect = gap - hSum + P.elemMargin;
    float C0n = Cdetect - dot(n, dq);
    softEmit(soft, counters, prevSoft, mapKeyA, mapKeyB, mapVal, P,
             ids, float4(1.0f - s, s, -(1.0f - t), -t), n, friction,
             float3(0), C0n,
             hSum - P.elemMargin, softLambdaCap(P, minMass), minMass,
             SC_EE, false, false, false, keyA, keyB);
}

kernel void dat_clear_pair_counts(device atomic_uint* counts [[buffer(0)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid < 3u) atomic_store_explicit(&counts[gid], 0u,
                                        memory_order_relaxed);
}

kernel void dat_clear_element_grid(device atomic_uint* counts [[buffer(0)]],
                                   constant SimParams& P [[buffer(1)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= P.elemHashSize) return;
    atomic_store_explicit(&counts[gid], 0u, memory_order_relaxed);
}

kernel void dat_reanchor(device const float4* posLin [[buffer(0)]],
                         device const uint* particleIdx [[buffer(1)]],
                         device float4* anchor [[buffer(2)]],
                         constant SimParams& P [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < P.numParticles) {
        uint v = particleIdx[gid];
        anchor[v] = float4(posLin[v].xyz, 0.0f);
    }
}

inline float datPlaneFraction(float3 x, float3 dx, float3 n, float3 plane,
                              float side, float gamma,
                              device atomic_uint* counters,
                              bool reportFailure) {
    float s0 = side * dot(n, x - plane);
    float q = side * dot(n, dx);
    // Tolerances must be translation invariant. Scaling by |x| makes a
    // scene far from the world origin classify millimetre motion as parallel
    // and can miss a genuine crossing.
    if (!finite_bits(s0) || !finite_bits(q)) {
        if (reportFailure) {
            atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                      memory_order_relaxed);
        }
        return 0.0f;
    }
    if (s0 < -DAT_GEOMETRY_EPS) return 0.0f;
    // A generic "parallel" epsilon must not hide an observed sign change:
    // even sub-micron motion is constrained when its endpoint crosses.
    if (q >= 0.0f || s0 + q > 0.0f) return 1.0f;
    float hit = max(s0, 0.0f) / (-q);
    if (hit > 1.0f + DAT_GEOMETRY_EPS) return 1.0f;
    return clamp(min(gamma * hit, hit - DAT_TIME_MARGIN), 0.0f, 1.0f);
}

inline void datAtomicT(device atomic_uint* truncBits, uint vertexID,
                       float3 x, float3 dx, float3 n, float3 plane,
                       float side, float gamma,
                       device atomic_uint* counters) {
    float t = datPlaneFraction(x, dx, n, plane, side, gamma,
                               counters, true);
    if (t < 1.0f) {
        atomic_fetch_min_explicit(&truncBits[vertexID], as_type<uint>(t),
                                  memory_order_relaxed);
    }
}

// Evaluate c0 + c1*t + c2*t^2 + c3*t^3 without relying on a Metal lambda.
inline float datCubicValue(float4 coefficients, float t) {
    return ((coefficients.w * t + coefficients.z) * t
            + coefficients.y) * t + coefficients.x;
}

kernel void dat_reduce_tet_inversion(
    device const float4* posLin       [[buffer(0)]],
    device const float4* anchor       [[buffer(1)]],
    device const TetGPU* tets         [[buffer(2)]],
    device atomic_uint* truncBits     [[buffer(3)]],
    device atomic_uint* counters      [[buffer(4)]],
    constant SimParams& P             [[buffer(5)]],
    uint gid                          [[thread_position_in_grid]])
{
    if (gid >= P.numTets) return;
    TetGPU tet = tets[gid];
    uint4 ids = tet.ids;
    float3 x0 = anchor[ids.x].xyz;
    float3 d0 = anchor[ids.y].xyz - x0;
    float3 d1 = anchor[ids.z].xyz - x0;
    float3 d2 = anchor[ids.w].xyz - x0;
    float restSixVolume = tet.r0.w;
    float currentSixVolume = dot(d0, cross(d1, d2));
    float signedCurrentVolume = (restSixVolume < 0.0f ? -1.0f : 1.0f)
        * currentSixVolume;
    if (!finite_bits(restSixVolume) || fabs(restSixVolume) < 1.0e-20f
        || !finite_bits(signedCurrentVolume)) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                  memory_order_relaxed);
        return;
    }
    if (signedCurrentVolume <= 0.0f) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[CTR_DAT_TET_DEGENERATE], 1u,
                                  memory_order_relaxed);
        return;
    }

    float desiredFloor = fabs(restSixVolume) * 1.0e-4f;
    if (signedCurrentVolume <= desiredFloor) {
        // A strongly compressed but still positive tet is not an invalid
        // collision anchor. Freeze its inertial prediction instead of
        // terminating the simulation; the colored elastic solve may still
        // expand it, while its per-color line guard blocks further collapse.
        uint frozen = as_type<uint>(0.0f);
        atomic_fetch_min_explicit(&truncBits[ids.x], frozen,
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&truncBits[ids.y], frozen,
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&truncBits[ids.z], frozen,
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&truncBits[ids.w], frozen,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[CTR_DAT_TET_DEGENERATE], 1u,
                                  memory_order_relaxed);
        return;
    }

    // With all four vertices moving, signed volume is cubic in the common
    // step fraction. Opposite-face planes alone can miss a flatten-and-
    // recover trajectory whose endpoint has the original sign. Partition
    // [0,1] at the exact derivative roots, then bisect the first monotone
    // interval reaching the positive volume floor.
    float3 y0 = posLin[ids.x].xyz;
    float3 e0 = posLin[ids.y].xyz - y0;
    float3 e1 = posLin[ids.z].xyz - y0;
    float3 e2 = posLin[ids.w].xyz - y0;
    float3 v0 = e0 - d0, v1 = e1 - d1, v2 = e2 - d2;
    float sign = restSixVolume < 0.0f ? -1.0f : 1.0f;
    float minVolume = desiredFloor;
    float c0 = sign * dot(d0, cross(d1, d2)) - minVolume;
    float c1 = sign * (dot(v0, cross(d1, d2))
        + dot(d0, cross(v1, d2) + cross(d1, v2)));
    float c2 = sign * (dot(v0, cross(v1, d2) + cross(d1, v2))
        + dot(d0, cross(v1, v2)));
    float c3 = sign * dot(v0, cross(v1, v2));
    if (!finite_bits(c0) || !finite_bits(c1)
        || !finite_bits(c2) || !finite_bits(c3)) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                  memory_order_relaxed);
        return;
    }
    float4 coefficients = float4(c0, c1, c2, c3);
    float critical[2];
    uint criticalCount = 0u;
    float qa = 3.0f * c3, qb = 2.0f * c2, qc = c1;
    if (fabs(qa) < 1.0e-12f) {
        if (fabs(qb) >= 1.0e-12f) {
            float root = -qc / qb;
            if (finite_bits(root) && root > 0.0f && root < 1.0f)
                critical[criticalCount++] = root;
        }
    } else {
        float discriminant = qb * qb - 4.0f * qa * qc;
        if (discriminant >= 0.0f && finite_bits(discriminant)) {
            float sd = sqrt(discriminant);
            float r0 = (-qb - sd) / (2.0f * qa);
            float r1 = (-qb + sd) / (2.0f * qa);
            if (r0 > r1) { float tmp = r0; r0 = r1; r1 = tmp; }
            if (finite_bits(r0) && r0 > 0.0f && r0 < 1.0f)
                critical[criticalCount++] = r0;
            if (finite_bits(r1) && r1 > 0.0f && r1 < 1.0f
                && (criticalCount == 0u
                    || fabs(r1 - critical[0]) > 1.0e-7f))
                critical[criticalCount++] = r1;
        }
    }
    float left = 0.0f;
    float hit = 2.0f;
    for (uint interval = 0u; interval <= criticalCount; ++interval) {
        float right = interval < criticalCount ? critical[interval] : 1.0f;
        if (datCubicValue(coefficients, right) <= 0.0f) {
            float lo = left, hi = right;
            for (uint iteration = 0u; iteration < 24u; ++iteration) {
                float mid = 0.5f * (lo + hi);
                if (datCubicValue(coefficients, mid) > 0.0f) lo = mid;
                else hi = mid;
            }
            hit = hi;
            break;
        }
        left = right;
    }
    if (hit <= 1.0f) {
        float safe = clamp(min(P.planarDATRelaxation * hit,
                               hit - DAT_TIME_MARGIN), 0.0f, 1.0f);
        atomic_fetch_min_explicit(&truncBits[ids.x], as_type<uint>(safe),
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&truncBits[ids.y], as_type<uint>(safe),
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&truncBits[ids.z], as_type<uint>(safe),
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&truncBits[ids.w], as_type<uint>(safe),
                                  memory_order_relaxed);
    }
}

kernel void dat_apply_tet_inversion(
    device float4* posLin             [[buffer(0)]],
    device const float4* anchor       [[buffer(1)]],
    device const float4* shape        [[buffer(2)]],
    device atomic_uint* truncBits     [[buffer(3)]],
    device atomic_uint* counters      [[buffer(4)]],
    constant SimParams& P             [[buffer(5)]],
    uint gid                          [[thread_position_in_grid]])
{
    if (gid >= P.numBodies || shape[gid].w >= 0.0f) return;
    float4 current = posLin[gid];
    float3 d = current.xyz - anchor[gid].xyz;
    float t = as_type<float>(atomic_exchange_explicit(
        &truncBits[gid], as_type<uint>(1.0f), memory_order_relaxed));
    bool failed = atomic_load_explicit(&counters[CTR_DAT_INVALID_ANCHOR],
                                       memory_order_relaxed) > 0u;
    if (!finite3(d) || !finite_bits(t)) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                  memory_order_relaxed);
        d = float3(0.0f);
        t = 0.0f;
        failed = true;
    }
    if (failed) t = 0.0f;
    if (t < 1.0f) {
        posLin[gid] = float4(anchor[gid].xyz
                             + d * clamp(t, 0.0f, 1.0f), current.w);
        atomic_fetch_add_explicit(&counters[CTR_DAT_TRUNCATIONS], 1u,
                                  memory_order_relaxed);
    }
}

kernel void dat_reduce(
    device const PlanarDATPairGPU* pairs [[buffer(0)]],
    device const float4* posLin          [[buffer(1)]],
    device const float4* anchor          [[buffer(2)]],
    device const uint4* tris             [[buffer(3)]],
    device const uint2* edges            [[buffer(4)]],
    device atomic_uint* truncBits        [[buffer(5)]],
    device const atomic_uint* pairCounts [[buffer(6)]],
    constant SimParams& P                [[buffer(7)]],
    device atomic_uint* counters         [[buffer(8)]],
    device const uint* colors            [[buffer(9)]],
    constant uint& activeColor           [[buffer(10)]],
    uint gid                             [[thread_position_in_grid]])
{
    uint count = min(atomic_load_explicit(&pairCounts[0],
                                           memory_order_relaxed),
                     P.maxPlanarDATPairs);
    if (gid >= count) return;
    uint4 pair = pairs[gid].data;
    if ((pair.w & DAT_PAIR_TRUNCATE) == 0u) return;
    bool fullPass = activeColor == 0xffffffffu;
    if (pair.x == DAT_PAIR_VT) {
        uint v = pair.y;
        uint4 t = tris[pair.z];
        bool activeV = fullPass
            || (posLin[v].w > 0.0f && colors[v] == activeColor);
        bool activeA = fullPass
            || (posLin[t.x].w > 0.0f && colors[t.x] == activeColor);
        bool activeB = fullPass
            || (posLin[t.y].w > 0.0f && colors[t.y] == activeColor);
        bool activeC = fullPass
            || (posLin[t.z].w > 0.0f && colors[t.z] == activeColor);
        if (!activeV && !activeA && !activeB && !activeC) return;
        float3 xv = anchor[v].xyz;
        float3 a = anchor[t.x].xyz, b = anchor[t.y].xyz,
               c = anchor[t.z].xyz;
        float3 bary;
        float3 cp = closestPtTriangle(xv, a, b, c, bary);
        float3 nh = xv - cp;
        float gap = length(nh);
        if (!finite_bits(gap)) {
            atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                      memory_order_relaxed);
            atomic_fetch_min_explicit(&truncBits[v], as_type<uint>(0.0f),
                                      memory_order_relaxed);
            atomic_fetch_min_explicit(&truncBits[t.x], as_type<uint>(0.0f),
                                      memory_order_relaxed);
            atomic_fetch_min_explicit(&truncBits[t.y], as_type<uint>(0.0f),
                                      memory_order_relaxed);
            atomic_fetch_min_explicit(&truncBits[t.z], as_type<uint>(0.0f),
                                      memory_order_relaxed);
            return;
        }
        if (gap > P.planarDATQueryRadius) return;
        float3 dv = posLin[v].xyz - xv;
        float3 da = posLin[t.x].xyz - a;
        float3 db = posLin[t.y].xyz - b;
        float3 dc = posLin[t.z].xyz - c;
        float3 n;
        if (gap >= DAT_GEOMETRY_EPS) {
            n = nh / gap;
        } else {
            bool validFallback;
            n = datVTFallbackNormal(a, b, c,
                                    dv - (da + db + dc) / 3.0f,
                                    validFallback);
            if (!validFallback) {
                atomic_fetch_add_explicit(
                    &counters[CTR_DAT_INVALID_ANCHOR], 1u,
                    memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &counters[CTR_DAT_VT_DEGENERATE], 1u,
                    memory_order_relaxed);
                atomic_fetch_min_explicit(
                    &truncBits[v], as_type<uint>(0.0f),
                    memory_order_relaxed);
                atomic_fetch_min_explicit(
                    &truncBits[t.x], as_type<uint>(0.0f),
                    memory_order_relaxed);
                atomic_fetch_min_explicit(
                    &truncBits[t.y], as_type<uint>(0.0f),
                    memory_order_relaxed);
                atomic_fetch_min_explicit(
                    &truncBits[t.z], as_type<uint>(0.0f),
                    memory_order_relaxed);
                return;
            }
            atomic_fetch_add_explicit(&counters[CTR_DAT_VT_DEGENERATE], 1u,
                                      memory_order_relaxed);
        }
        float approachV = max(-dot(n, dv), 0.0f);
        float approachT = max(max(dot(n, da), dot(n, db)),
                              max(dot(n, dc), 0.0f));
        float lambda = approachV + approachT == 0.0f ? 0.5f
            : clamp(approachT / (approachV + approachT), 0.05f, 0.95f);
        float3 plane = gap >= DAT_GEOMETRY_EPS
            ? cp + lambda * nh : cp;
        if (approachV > 0.0f && activeV)
            datAtomicT(truncBits, v, xv, dv, n, plane, 1.0f,
                       P.planarDATRelaxation, counters);
        if (approachT > 0.0f) {
            if (activeA)
            datAtomicT(truncBits, t.x, a, da, n, plane, -1.0f,
                       P.planarDATRelaxation, counters);
            if (activeB)
            datAtomicT(truncBits, t.y, b, db, n, plane, -1.0f,
                       P.planarDATRelaxation, counters);
            if (activeC)
            datAtomicT(truncBits, t.z, c, dc, n, plane, -1.0f,
                       P.planarDATRelaxation, counters);
        }
        return;
    }

    uint2 ea = edges[pair.y], eb = edges[pair.z];
    bool activeA0 = fullPass
        || (posLin[ea.x].w > 0.0f && colors[ea.x] == activeColor);
    bool activeA1 = fullPass
        || (posLin[ea.y].w > 0.0f && colors[ea.y] == activeColor);
    bool activeB0 = fullPass
        || (posLin[eb.x].w > 0.0f && colors[eb.x] == activeColor);
    bool activeB1 = fullPass
        || (posLin[eb.y].w > 0.0f && colors[eb.y] == activeColor);
    if (!activeA0 && !activeA1 && !activeB0 && !activeB1) return;
    float3 a0 = anchor[ea.x].xyz, a1 = anchor[ea.y].xyz;
    float3 b0 = anchor[eb.x].xyz, b1 = anchor[eb.y].xyz;
    float s, t;
    eeClosestSegSeg(a0, a1, b0, b1, s, t);
    float3 ca = a0 + (a1 - a0) * s;
    float3 cb = b0 + (b1 - b0) * t;
    float3 nh = ca - cb;
    float gap = length(nh);
    if (!finite_bits(gap)) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&truncBits[ea.x], as_type<uint>(0.0f),
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&truncBits[ea.y], as_type<uint>(0.0f),
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&truncBits[eb.x], as_type<uint>(0.0f),
                                  memory_order_relaxed);
        atomic_fetch_min_explicit(&truncBits[eb.y], as_type<uint>(0.0f),
                                  memory_order_relaxed);
        return;
    }
    if (gap > P.planarDATQueryRadius) return;
    float3 da0 = posLin[ea.x].xyz - a0;
    float3 da1 = posLin[ea.y].xyz - a1;
    float3 db0 = posLin[eb.x].xyz - b0;
    float3 db1 = posLin[eb.y].xyz - b1;
    float3 n;
    if (gap > DAT_GEOMETRY_EPS) {
        n = nh / gap;
    } else {
        bool validFallback;
        n = datEEFallbackNormal(
            a0, a1, b0, b1,
            (da0 + da1 - db0 - db1) * 0.5f, validFallback);
        if (!validFallback) {
            atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[CTR_DAT_EE_DEGENERATE], 1u,
                                      memory_order_relaxed);
            atomic_fetch_min_explicit(
                &truncBits[ea.x], as_type<uint>(0.0f), memory_order_relaxed);
            atomic_fetch_min_explicit(
                &truncBits[ea.y], as_type<uint>(0.0f), memory_order_relaxed);
            atomic_fetch_min_explicit(
                &truncBits[eb.x], as_type<uint>(0.0f), memory_order_relaxed);
            atomic_fetch_min_explicit(
                &truncBits[eb.y], as_type<uint>(0.0f), memory_order_relaxed);
            return;
        }
        atomic_fetch_add_explicit(&counters[CTR_DAT_EE_DEGENERATE], 1u,
                                  memory_order_relaxed);
    }
    float approachA = max(max(-dot(n, da0), -dot(n, da1)), 0.0f);
    float approachB = max(max(dot(n, db0), dot(n, db1)), 0.0f);
    float lambda = approachA + approachB == 0.0f ? 0.5f
        : clamp(approachB / (approachA + approachB), 0.05f, 0.95f);
    float3 plane = gap > DAT_GEOMETRY_EPS ? cb + lambda * nh : cb;
    if (approachA > 0.0f) {
        if (activeA0)
        datAtomicT(truncBits, ea.x, a0, da0, n, plane, 1.0f,
                   P.planarDATRelaxation, counters);
        if (activeA1)
        datAtomicT(truncBits, ea.y, a1, da1, n, plane, 1.0f,
                   P.planarDATRelaxation, counters);
    }
    if (approachB > 0.0f) {
        if (activeB0)
        datAtomicT(truncBits, eb.x, b0, db0, n, plane, -1.0f,
                   P.planarDATRelaxation, counters);
        if (activeB1)
        datAtomicT(truncBits, eb.y, b1, db1, n, plane, -1.0f,
                   P.planarDATRelaxation, counters);
    }
}

inline bool datCanonicalDynamicParticipant(uint body, uint4 ids,
                                            device const float4* posLin) {
    uint canonical = 0xffffffffu;
    for (uint i = 0u; i < 4u; ++i) {
        if (posLin[ids[i]].w > 0.0f) canonical = min(canonical, ids[i]);
    }
    return body == canonical;
}

inline float datIncidentPairFraction(
    uint body, PlanarDATPairGPU pair,
    device const float4* posLin, device const float4* anchor,
    device const uint4* tris, device const uint2* edges,
    device atomic_uint* counters, constant SimParams& P)
{
    uint4 ids = datPairBodies(pair, tris, edges);
    bool report = datCanonicalDynamicParticipant(body, ids, posLin);
    if (pair.data.x == DAT_PAIR_VT) {
        uint v = pair.data.y;
        uint4 tri = tris[pair.data.z];
        float3 xv = anchor[v].xyz;
        float3 a = anchor[tri.x].xyz, b = anchor[tri.y].xyz;
        float3 c = anchor[tri.z].xyz;
        float3 bary;
        float3 cp = closestPtTriangle(xv, a, b, c, bary);
        float3 nh = xv - cp;
        float gap = length(nh);
        if (!finite_bits(gap)) {
            if (report) {
                atomic_fetch_add_explicit(
                    &counters[CTR_DAT_INVALID_ANCHOR], 1u,
                    memory_order_relaxed);
                atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                          memory_order_relaxed);
            }
            return 0.0f;
        }
        if (gap > P.planarDATQueryRadius) return 1.0f;
        float3 dv = posLin[v].xyz - xv;
        float3 da = posLin[tri.x].xyz - a;
        float3 db = posLin[tri.y].xyz - b;
        float3 dc = posLin[tri.z].xyz - c;
        if (!finite3(dv) || !finite3(da) || !finite3(db)
            || !finite3(dc)) {
            if (report) {
                atomic_fetch_add_explicit(
                    &counters[CTR_DAT_INVALID_ANCHOR], 1u,
                    memory_order_relaxed);
                atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                          memory_order_relaxed);
            }
            return 0.0f;
        }
        float3 n;
        if (gap >= DAT_GEOMETRY_EPS) {
            n = nh / gap;
        } else {
            bool validFallback;
            n = datVTFallbackNormal(a, b, c,
                                    dv - (da + db + dc) / 3.0f,
                                    validFallback);
            if (!validFallback) {
                if (report) {
                    atomic_fetch_add_explicit(
                        &counters[CTR_DAT_INVALID_ANCHOR], 1u,
                        memory_order_relaxed);
                    atomic_fetch_add_explicit(
                        &counters[CTR_DAT_VT_DEGENERATE], 1u,
                        memory_order_relaxed);
                }
                return 0.0f;
            }
            if (report) {
                atomic_fetch_add_explicit(
                    &counters[CTR_DAT_VT_DEGENERATE], 1u,
                    memory_order_relaxed);
            }
        }
        float approachV = max(-dot(n, dv), 0.0f);
        float approachT = max(max(dot(n, da), dot(n, db)),
                              max(dot(n, dc), 0.0f));
        float lambda = approachV + approachT == 0.0f ? 0.5f
            : clamp(approachT / (approachV + approachT), 0.05f, 0.95f);
        float3 plane = gap >= DAT_GEOMETRY_EPS
            ? cp + lambda * nh : cp;
        if (body == v) {
            return datPlaneFraction(xv, dv, n, plane, 1.0f,
                                    P.planarDATRelaxation,
                                    counters, report);
        }
        float3 x = body == tri.x ? a : (body == tri.y ? b : c);
        float3 d = body == tri.x ? da : (body == tri.y ? db : dc);
        return datPlaneFraction(x, d, n, plane, -1.0f,
                                P.planarDATRelaxation, counters, report);
    }

    uint2 ea = edges[pair.data.y], eb = edges[pair.data.z];
    float3 a0 = anchor[ea.x].xyz, a1 = anchor[ea.y].xyz;
    float3 b0 = anchor[eb.x].xyz, b1 = anchor[eb.y].xyz;
    float s, t;
    eeClosestSegSeg(a0, a1, b0, b1, s, t);
    float3 ca = mix(a0, a1, s), cb = mix(b0, b1, t);
    float3 nh = ca - cb;
    float gap = length(nh);
    if (!finite_bits(gap)) {
        if (report) {
            atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                      memory_order_relaxed);
        }
        return 0.0f;
    }
    if (gap > P.planarDATQueryRadius) return 1.0f;
    float3 da0 = posLin[ea.x].xyz - a0;
    float3 da1 = posLin[ea.y].xyz - a1;
    float3 db0 = posLin[eb.x].xyz - b0;
    float3 db1 = posLin[eb.y].xyz - b1;
    if (!finite3(da0) || !finite3(da1) || !finite3(db0)
        || !finite3(db1)) {
        if (report) {
            atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                      memory_order_relaxed);
        }
        return 0.0f;
    }
    float3 n;
    if (gap > DAT_GEOMETRY_EPS) {
        n = nh / gap;
    } else {
        bool validFallback;
        n = datEEFallbackNormal(
            a0, a1, b0, b1,
            (da0 + da1 - db0 - db1) * 0.5f, validFallback);
        if (!validFallback) {
            if (report) {
                atomic_fetch_add_explicit(
                    &counters[CTR_DAT_INVALID_ANCHOR], 1u,
                    memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &counters[CTR_DAT_EE_DEGENERATE], 1u,
                    memory_order_relaxed);
            }
            return 0.0f;
        }
        if (report) {
            atomic_fetch_add_explicit(&counters[CTR_DAT_EE_DEGENERATE], 1u,
                                      memory_order_relaxed);
        }
    }
    float approachA = max(max(-dot(n, da0), -dot(n, da1)), 0.0f);
    float approachB = max(max(dot(n, db0), dot(n, db1)), 0.0f);
    float lambda = approachA + approachB == 0.0f ? 0.5f
        : clamp(approachB / (approachA + approachB), 0.05f, 0.95f);
    float3 plane = gap > DAT_GEOMETRY_EPS ? cb + lambda * nh : cb;
    bool sideA = body == ea.x || body == ea.y;
    float3 x = body == ea.x ? a0 : (body == ea.y ? a1
        : (body == eb.x ? b0 : b1));
    float3 d = body == ea.x ? da0 : (body == ea.y ? da1
        : (body == eb.x ? db0 : db1));
    return datPlaneFraction(x, d, n, plane, sideA ? 1.0f : -1.0f,
                            P.planarDATRelaxation, counters, report);
}

// Eight neighboring lanes cooperatively reduce each particle's CSR slice.
// Its slice is disjoint, so reduction needs no pair-wide scan and no atomic
// minima; the lane split also prevents a highly contacted fold vertex from
// serializing thousands of plane evaluations on one GPU thread.
// Application remains a separate dispatch, preserving the exact post-color
// snapshot when two retained collision pairs share a static palette color.
kernel void dat_reduce_incident_color(
    device const PlanarDATPairGPU* pairs [[buffer(0)]],
    device const float4* posLin          [[buffer(1)]],
    device const float4* anchor          [[buffer(2)]],
    device const uint4* tris             [[buffer(3)]],
    device const uint2* edges            [[buffer(4)]],
    device atomic_uint* truncBits        [[buffer(5)]],
    device const atomic_uint* pairCounts [[buffer(6)]],
    constant SimParams& P                [[buffer(7)]],
    device atomic_uint* counters         [[buffer(8)]],
    device const uint* colorList         [[buffer(9)]],
    device const uint* colorStart        [[buffer(10)]],
    constant uint& activeColor           [[buffer(11)]],
    device const uint* bodyPairStart     [[buffer(12)]],
    device const uint* bodyPairCount     [[buffer(13)]],
    device const uint* bodyPairs         [[buffer(14)]],
    uint tid                             [[thread_position_in_grid]])
{
    uint start = colorStart[activeColor];
    uint end = colorStart[activeColor + 1u];
    uint slot = tid >> 3;
    uint lane = tid & 7u;
    if (start + slot >= end) return;
    uint body = colorList[start + slot];
    float fraction = 1.0f;
    uint s = bodyPairStart[body];
    uint e = s + bodyPairCount[body];
    uint stored = min(atomic_load_explicit(&pairCounts[0],
                                            memory_order_relaxed),
                      P.maxPlanarDATPairs);
    for (uint i = s + lane; i < e; i += 8u) {
        uint pairIndex = bodyPairs[i];
        if (pairIndex >= stored) {
            fraction = 0.0f;
            break;
        }
        fraction = min(fraction, datIncidentPairFraction(
            body, pairs[pairIndex], posLin, anchor, tris, edges,
            counters, P));
    }
    for (ushort d = 4; d >= 1; d >>= 1) {
        fraction = min(fraction, simd_shuffle_xor(fraction, d));
    }
    if (lane == 0u) {
        atomic_store_explicit(&truncBits[body], as_type<uint>(fraction),
                              memory_order_relaxed);
    }
}

// Per-color apply visits only the bodies that the primal dispatch just
// changed. Fixed division planes are vertex-local, so untouched vertices do
// not need to be rescanned or have their accumulated displacement rescaled.
kernel void dat_apply_color(
    device float4* posLin                  [[buffer(0)]],
    device const float4* anchor            [[buffer(1)]],
    device const uint* colorList           [[buffer(2)]],
    device const uint* colorStart          [[buffer(3)]],
    device const uint* softGroup           [[buffer(4)]],
    device atomic_uint* truncBits          [[buffer(5)]],
    device const atomic_uint* pairCounts   [[buffer(6)]],
    device atomic_uint* counters           [[buffer(7)]],
    constant SimParams& P                  [[buffer(8)]],
    constant uint& activeColor             [[buffer(9)]],
    uint tid                               [[thread_position_in_grid]])
{
    uint start = colorStart[activeColor];
    uint end = colorStart[activeColor + 1u];
    if (start + tid >= end) return;
    uint v = colorList[start + tid];
    if (softGroup[v] == 0u) return;
    float4 current = posLin[v];
    float3 d = current.xyz - anchor[v].xyz;
    uint raw = atomic_load_explicit(&pairCounts[0], memory_order_relaxed);
    bool unsafeCapacity = raw > P.maxPlanarDATPairs
        || atomic_load_explicit(&counters[CTR_DAT_PAIRS],
                                memory_order_relaxed) > P.maxPlanarDATPairs
        || atomic_load_explicit(&counters[CTR_DAT_GRID_OVERFLOW],
                                memory_order_relaxed) > 0u
        || atomic_load_explicit(&counters[CTR_DAT_INVALID_ANCHOR],
                                memory_order_relaxed) > 0u;
    float t = as_type<float>(atomic_exchange_explicit(
        &truncBits[v], as_type<uint>(1.0f), memory_order_relaxed));
    if (unsafeCapacity) t = 0.0f;
    if (!finite3(d) || !finite_bits(t)) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                  memory_order_relaxed);
        d = float3(0.0f);
        t = 0.0f;
    }
    bool truncated = t < 1.0f;
    d *= clamp(t, 0.0f, 1.0f);
    float maxD = 0.5f * P.planarDATRelaxation * P.planarDATQueryRadius;
    float d2 = dot(d, d);
    if (d2 > maxD * maxD) {
        d *= maxD * rsqrt(d2);
        truncated = true;
    }
    if (truncated) {
        posLin[v] = float4(anchor[v].xyz + d, current.w);
        atomic_fetch_add_explicit(&counters[CTR_DAT_TRUNCATIONS], 1u,
                                  memory_order_relaxed);
    }
}

kernel void dat_apply(
    device float4* posLin             [[buffer(0)]],
    device const float4* anchor       [[buffer(1)]],
    device const uint* particleIdx    [[buffer(2)]],
    device atomic_uint* truncBits     [[buffer(3)]],
    device const atomic_uint* pairCounts [[buffer(4)]],
    device atomic_uint* counters      [[buffer(5)]],
    constant SimParams& P             [[buffer(6)]],
    uint gid                          [[thread_position_in_grid]])
{
    if (gid >= P.numParticles) return;
    uint v = particleIdx[gid];
    float4 current = posLin[v];
    float3 d = current.xyz - anchor[v].xyz;
    uint raw = atomic_load_explicit(&pairCounts[0], memory_order_relaxed);
    bool unsafeCapacity = raw > P.maxPlanarDATPairs
        || atomic_load_explicit(&counters[CTR_DAT_PAIRS],
                                memory_order_relaxed) > P.maxPlanarDATPairs
        || atomic_load_explicit(&counters[CTR_DAT_GRID_OVERFLOW],
                                memory_order_relaxed) > 0u
        || atomic_load_explicit(&counters[CTR_DAT_INVALID_ANCHOR],
                                memory_order_relaxed) > 0u;
    // Consume and reset in one atomic operation. Dispatch ordering guarantees
    // that every pair reduction has completed first, and leaving 1.0 behind
    // prepares the next color without a separate full-particle clear pass.
    float reducedT = as_type<float>(atomic_exchange_explicit(
        &truncBits[v], as_type<uint>(1.0f), memory_order_relaxed));
    float t = unsafeCapacity ? 0.0f : reducedT;
    if (!finite3(d) || !finite_bits(t)) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                  memory_order_relaxed);
        d = float3(0.0f);
        t = 0.0f;
    }
    bool truncated = t < 1.0f;
    d *= clamp(t, 0.0f, 1.0f);
    float maxD = 0.5f * P.planarDATRelaxation * P.planarDATQueryRadius;
    float d2 = dot(d, d);
    if (d2 > maxD * maxD) {
        d *= maxD * rsqrt(d2);
        truncated = true;
    }
    if (truncated) {
        posLin[v] = float4(anchor[v].xyz + d, current.w);
        atomic_fetch_add_explicit(&counters[CTR_DAT_TRUNCATIONS], 1u,
                                  memory_order_relaxed);
    }
}

// A soft-contact capacity failure is terminal, but it is discovered on the
// GPU before retirement. Restore the whole deformable surface to the exact
// step-start pose so a clipped contact set is never externally published as a
// partially solved frame. finalize_velocities then produces zero surface
// velocity for the failed step.
kernel void dat_restore_failed_surface(
    device float4* posLin                  [[buffer(0)]],
    device const float4* initLin           [[buffer(1)]],
    device const float4* shape             [[buffer(2)]],
    device const atomic_uint* counters     [[buffer(3)]],
    constant SimParams& P                  [[buffer(4)]],
    uint gid                               [[thread_position_in_grid]])
{
    if (gid >= P.numBodies || shape[gid].w >= 0.0f) return;
    uint rawSoft = atomic_load_explicit(&counters[CTR_SOFT_CANDIDATES],
                                        memory_order_relaxed);
    bool failed = rawSoft > P.maxSoft
        || atomic_load_explicit(&counters[CTR_DAT_PAIRS],
                                memory_order_relaxed) > P.maxPlanarDATPairs
        || atomic_load_explicit(&counters[CTR_DAT_GRID_OVERFLOW],
                                memory_order_relaxed) > 0u
        || atomic_load_explicit(&counters[CTR_DAT_INVALID_ANCHOR],
                                memory_order_relaxed) > 0u;
    if (failed) {
        posLin[gid] = float4(initLin[gid].xyz, posLin[gid].w);
    }
}
