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
    device const uint* clothVert     [[buffer(11)]],
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
        if (nbrContains(nbrList, ns, ne, t.x)
            || nbrContains(nbrList, ns, ne, t.y)
            || nbrContains(nbrList, ns, ne, t.z)) continue;
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
        uint group = clothGroup[v];
        bool sameSolid = group != 0u && clothVert[v] == 0u
            && clothGroup[t.x] == group && clothGroup[t.y] == group
            && clothGroup[t.z] == group
            && clothVert[t.x] == 0u && clothVert[t.y] == 0u
            && clothVert[t.z] == 0u;
        uint flags = DAT_PAIR_TRUNCATE
            | (sameSolid ? 0u : DAT_PAIR_CONTACT);
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
    device const uint* clothVert     [[buffer(11)]],
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
            if (nbrContains(nbrList, ns0, ne0, b.x)
                || nbrContains(nbrList, ns0, ne0, b.y)
                || nbrContains(nbrList, ns1, ne1, b.x)
                || nbrContains(nbrList, ns1, ne1, b.y)) continue;
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
            uint group = clothGroup[a.x];
            bool sameSolid = group != 0u
                && clothGroup[a.y] == group
                && clothGroup[b.x] == group && clothGroup[b.y] == group
                && clothVert[a.x] == 0u && clothVert[a.y] == 0u
                && clothVert[b.x] == 0u && clothVert[b.y] == 0u;
            uint flags = DAT_PAIR_TRUNCATE;
            if (!sameSolid && gid < P.numSurfaceContactEdges
                && other < P.numSurfaceContactEdges) {
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
    if (dist < 1.0e-7f) return;
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
    float3 nGeo = d / dist;
    uint keyA = (SC_EE << 28) | ea;
    uint keyB = eb;
    float3 n = nGeo;
    float gap = dist;
    int previous = softMapFind(mapKeyA, mapKeyB, mapVal,
                               P.softMapCapacity, keyA, keyB);
    if (previous >= 0 && dot(nGeo, prevSoft[previous].normal.xyz) < 0.0f) {
        n = -nGeo;
        gap = -dist;
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

inline void datAtomicT(device atomic_uint* truncBits, uint vertexID,
                       float3 x, float3 dx, float3 n, float3 plane,
                       float side, float gamma) {
    float s0 = side * dot(n, x - plane);
    float q = side * dot(n, dx);
    // Tolerances must be translation invariant. Scaling by |x| makes a
    // scene far from the world origin classify millimetre motion as parallel
    // and can miss a genuine crossing.
    if (!finite_bits(s0) || !finite_bits(q)) return;
    if (s0 < -DAT_GEOMETRY_EPS) {
        atomic_fetch_min_explicit(&truncBits[vertexID], as_type<uint>(0.0f),
                                  memory_order_relaxed);
        return;
    }
    // A generic "parallel" epsilon must not hide an observed sign change:
    // even sub-micron motion is constrained when its endpoint crosses.
    if (q >= 0.0f || s0 + q > 0.0f) return;
    float hit = max(s0, 0.0f) / (-q);
    if (hit > 1.0f + DAT_GEOMETRY_EPS) return;
    float t = clamp(min(gamma * hit, hit - DAT_TIME_MARGIN), 0.0f, 1.0f);
    atomic_fetch_min_explicit(&truncBits[vertexID], as_type<uint>(t),
                              memory_order_relaxed);
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
    uint gid                             [[thread_position_in_grid]])
{
    uint count = min(atomic_load_explicit(&pairCounts[0],
                                           memory_order_relaxed),
                     P.maxPlanarDATPairs);
    if (gid >= count) return;
    uint4 pair = pairs[gid].data;
    if ((pair.w & DAT_PAIR_TRUNCATE) == 0u) return;
    if (pair.x == DAT_PAIR_VT) {
        uint v = pair.y;
        uint4 t = tris[pair.z];
        float3 xv = anchor[v].xyz;
        float3 a = anchor[t.x].xyz, b = anchor[t.y].xyz,
               c = anchor[t.z].xyz;
        float3 bary;
        float3 cp = closestPtTriangle(xv, a, b, c, bary);
        float3 nh = xv - cp;
        float gap = length(nh);
        if (!finite_bits(gap) || gap < DAT_GEOMETRY_EPS) {
            atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
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
        float3 n = nh / gap;
        float3 dv = posLin[v].xyz - xv;
        float3 da = posLin[t.x].xyz - a;
        float3 db = posLin[t.y].xyz - b;
        float3 dc = posLin[t.z].xyz - c;
        float approachV = max(-dot(n, dv), 0.0f);
        float approachT = max(max(dot(n, da), dot(n, db)),
                              max(dot(n, dc), 0.0f));
        float lambda = approachV + approachT == 0.0f ? 0.5f
            : clamp(approachT / (approachV + approachT), 0.05f, 0.95f);
        float3 plane = cp + lambda * nh;
        if (approachV > 0.0f)
            datAtomicT(truncBits, v, xv, dv, n, plane, 1.0f,
                       P.planarDATRelaxation);
        if (approachT > 0.0f) {
            datAtomicT(truncBits, t.x, a, da, n, plane, -1.0f,
                       P.planarDATRelaxation);
            datAtomicT(truncBits, t.y, b, db, n, plane, -1.0f,
                       P.planarDATRelaxation);
            datAtomicT(truncBits, t.z, c, dc, n, plane, -1.0f,
                       P.planarDATRelaxation);
        }
        return;
    }

    uint2 ea = edges[pair.y], eb = edges[pair.z];
    float3 a0 = anchor[ea.x].xyz, a1 = anchor[ea.y].xyz;
    float3 b0 = anchor[eb.x].xyz, b1 = anchor[eb.y].xyz;
    float s, t;
    eeClosestSegSeg(a0, a1, b0, b1, s, t);
    float3 ca = a0 + (a1 - a0) * s;
    float3 cb = b0 + (b1 - b0) * t;
    float3 nh = ca - cb;
    float gap = length(nh);
    if (!finite_bits(gap) || gap <= DAT_GEOMETRY_EPS) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
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
    float3 n = nh / gap;
    float3 da0 = posLin[ea.x].xyz - a0;
    float3 da1 = posLin[ea.y].xyz - a1;
    float3 db0 = posLin[eb.x].xyz - b0;
    float3 db1 = posLin[eb.y].xyz - b1;
    float approachA = max(max(-dot(n, da0), -dot(n, da1)), 0.0f);
    float approachB = max(max(dot(n, db0), dot(n, db1)), 0.0f);
    float lambda = approachA + approachB == 0.0f ? 0.5f
        : clamp(approachB / (approachA + approachB), 0.05f, 0.95f);
    float3 plane = cb + lambda * nh;
    if (approachA > 0.0f) {
        datAtomicT(truncBits, ea.x, a0, da0, n, plane, 1.0f,
                   P.planarDATRelaxation);
        datAtomicT(truncBits, ea.y, a1, da1, n, plane, 1.0f,
                   P.planarDATRelaxation);
    }
    if (approachB > 0.0f) {
        datAtomicT(truncBits, eb.x, b0, db0, n, plane, -1.0f,
                   P.planarDATRelaxation);
        datAtomicT(truncBits, eb.y, b1, db1, n, plane, -1.0f,
                   P.planarDATRelaxation);
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
    device const uint* particleIdx         [[buffer(2)]],
    device const atomic_uint* counters     [[buffer(3)]],
    constant SimParams& P                  [[buffer(4)]],
    uint gid                               [[thread_position_in_grid]])
{
    if (gid >= P.numParticles) return;
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
        uint v = particleIdx[gid];
        posLin[v] = float4(initLin[v].xyz, posLin[v].w);
    }
}
