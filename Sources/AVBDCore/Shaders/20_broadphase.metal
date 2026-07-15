#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Broadphase: counting-sort grouping of bodies into spatial hash cells,
// then pair generation against 27 neighbor cells plus a "global" list of
// oversized/static bodies. Grouping (not full sorting) is sufficient and
// cheaper: count -> exclusive scan -> scatter.
// Hashed body set and global body set are precomputed index lists.
// ============================================================================

inline int3 cellCoord(float3 p, float cellSize) {
    return int3(floor(p / cellSize));
}

inline uint cellHash(int3 c, uint collisionGroup, uint hashSize) {
    // Large-prime XOR plus avalanche finishing. The bare XOR-of-products,
    // masked to a small table, aliases SYSTEMATICALLY on regular lattices
    // (cloth grids): whole families of distant cells share buckets and a
    // 27-cell scan wades through 50x the bodies it should (measured 9.3 ms
    // of a 20k-vertex frame). The mix breaks the structure.
    uint h = uint(c.x * 92837111) ^ uint(c.y * 689287499) ^ uint(c.z * 283923481);
    h ^= collisionGroup * 0x9e3779b9u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    return h & (hashSize - 1u);
}

inline uint cellHash(int3 c, uint hashSize) {
    return cellHash(c, 0u, hashSize);
}

inline bool collisionGroupsCompatible(uint a, uint b) {
    return a == 0u || b == 0u || a == b;
}

inline float3 bpColliderCenter(device const float4* posLin,
                               device const float4* posAng,
                               device const uint* colliderOwner,
                               device const float4* colliderLocalPosition,
                               uint collider) {
    uint body = colliderOwner[collider];
    return posLin[body].xyz
        + q_rotate(posAng[body], colliderLocalPosition[collider].xyz);
}

kernel void bp_clear_cells(
    device atomic_uint* cellCount   [[buffer(0)]],
    constant uint& hashSize         [[buffer(1)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid < hashSize) {
        atomic_store_explicit(&cellCount[gid], 0u, memory_order_relaxed);
    }
}

// Count bodies per cell; remember each body's slot within its cell.
// Cells containing at least one rigid (non-particle) body get flagged so
// soft-surface particles can skip rigid-free cells in pair generation.
kernel void bp_count(
    device const float4* posLin     [[buffer(0)]],
    device const uint* hashedIdx    [[buffer(1)]],   // indices of hashed bodies
    device atomic_uint* cellCount   [[buffer(2)]],
    device uint2* bodyCellSlot      [[buffer(3)]],   // (cellHash, slot) per hashed body
    constant SimParams& P           [[buffer(4)]],
    device const float4* shape      [[buffer(5)]],
    device uint* cellRigid          [[buffer(6)]],
    device const uint* colliderOwner [[buffer(7)]],
    device const float4* colliderLocalPosition [[buffer(8)]],
    device const float4* posAng     [[buffer(9)]],
    device const uint* colliderGroup [[buffer(10)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numHashed) return;
    uint b = hashedIdx[gid];
    float3 center = bpColliderCenter(posLin, posAng, colliderOwner,
                                     colliderLocalPosition, b);
    uint h = cellHash(cellCoord(center, P.cellSize), colliderGroup[b],
                      P.gridHashSize);
    uint slot = atomic_fetch_add_explicit(&cellCount[h], 1u, memory_order_relaxed);
    bodyCellSlot[gid] = uint2(h, slot);
    if (shape[b].w >= 0.0f) cellRigid[h] = 1u;       // racing writes of 1: benign
}

// Scatter hashed bodies into the grouped cell list using scanned offsets.
kernel void bp_scatter(
    device const uint* hashedIdx    [[buffer(0)]],
    device const uint2* bodyCellSlot [[buffer(1)]],
    device const uint* cellStart    [[buffer(2)]],   // exclusive scan of counts
    device uint* cellBodies         [[buffer(3)]],
    constant SimParams& P           [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numHashed) return;
    uint2 cs = bodyCellSlot[gid];
    cellBodies[cellStart[cs.x] + cs.y] = hashedIdx[gid];
}

// Atomic scatter assigns correct cell membership but not a stable order.
// Sort each usually tiny hash bucket by collider id before pair enumeration;
// otherwise identical states can produce different manifold indices and a
// different floating-point accumulation order in the solver.
kernel void bp_sort_cells(
    device const uint* cellStart    [[buffer(0)]],
    device const uint* cellCount    [[buffer(1)]],
    device uint* cellBodies         [[buffer(2)]],
    constant uint& hashSize         [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= hashSize) return;
    uint s = cellStart[gid];
    uint n = cellCount[gid];
    for (uint i = 1; i < n; i++) {
        uint value = cellBodies[s + i];
        uint j = i;
        while (j > 0 && cellBodies[s + j - 1] > value) {
            cellBodies[s + j] = cellBodies[s + j - 1];
            j--;
        }
        cellBodies[s + j] = value;
    }
}

// Binary search in sorted exclusion list of (a,b) pairs, a < b.
inline bool pairExcluded(device const uint2* excl, uint count, uint a, uint b) {
    uint lo = 0, hi = count;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        uint2 e = excl[mid];
        if (e.x < a || (e.x == a && e.y < b)) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= count) return false;
    uint2 e = excl[lo];
    return e.x == a && e.y == b;
}

// Generate candidate pairs: per hashed body, scan 27 neighbor cells (only
// pairs with the other index larger to avoid duplicates), plus all globals.
kernel void bp_gen_pairs(
    device const float4* posLin     [[buffer(0)]],
    device const float4* shape      [[buffer(1)]],   // w = radius
    device const uint* hashedIdx    [[buffer(2)]],
    device const uint* cellStart    [[buffer(3)]],
    device const uint* cellCount    [[buffer(4)]],   // counts (end = start + count)
    device const uint* cellBodies   [[buffer(5)]],
    device const uint* globalIdx    [[buffer(6)]],
    device const uint2* exclusions  [[buffer(7)]],
    constant uint& numExclusions    [[buffer(8)]],
    device atomic_uint* counters    [[buffer(9)]],
    device uint2* pairs             [[buffer(10)]],
    constant SimParams& P           [[buffer(11)]],
    device const uint* clothGroup   [[buffer(12)]],
    device const uint* cellRigid    [[buffer(13)]],
    device const uint* colliderOwner [[buffer(14)]],
    device const float4* colliderLocalPosition [[buffer(15)]],
    device const float4* posAng     [[buffer(16)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numHashed) return;
    uint a = hashedIdx[gid];
    uint ownerA = colliderOwner[a];
    float3 pa = bpColliderCenter(posLin, posAng, colliderOwner,
                                 colliderLocalPosition, a);
    float ra = fabs(shape[a].w);
    bool aDyn = posLin[ownerA].w > 0.0f;
    uint ga = clothGroup[ownerA];

    // Soft-surface particles never sphere-pair with OTHER soft surfaces
    // either (cross-group contact is V-T/E-E territory just like
    // self-collision), so their only legal hashed partners are rigid
    // bodies: skip the scan entirely when none are hashed, and skip
    // rigid-free cells otherwise.
    bool scanCells = !(ga != 0 && P.numHashedRigid == 0);
    int3 cc = cellCoord(pa, P.cellSize);
    if (scanCells) {
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        uint h = cellHash(cc + int3(dx, dy, dz), P.gridHashSize);
        if (ga != 0 && cellRigid[h] == 0) continue;
        uint s = cellStart[h], e = s + cellCount[h];
        for (uint k = s; k < e; k++) {
            uint b = cellBodies[k];
            if (b <= a) continue;   // each pair once
            uint ownerB = colliderOwner[b];
            if (ownerA == ownerB) continue;
            if (!aDyn && posLin[ownerB].w <= 0.0f) continue;
            // soft surface vs any soft surface: V-T/E-E elements own it;
            // sphere pairs would only re-shoulder it at O(n^2) in piles
            if (ga != 0 && clothGroup[ownerB] != 0) continue;
            float rb = fabs(shape[b].w);
            float3 pb = bpColliderCenter(posLin, posAng, colliderOwner,
                                         colliderLocalPosition, b);
            float3 dp = pa - pb;
            float r = ra + rb;
            if (dot(dp, dp) > r * r) continue;
            uint ownerLo = min(ownerA, ownerB), ownerHi = max(ownerA, ownerB);
            if (pairExcluded(exclusions, numExclusions, ownerLo, ownerHi)) continue;
            uint slot = atomic_fetch_add_explicit(&counters[CTR_PAIRS], 1u, memory_order_relaxed);
            if (slot < P.maxPairs) pairs[slot] = uint2(a, b);
        }
    }

    }
    // Globals (statics / oversized)
    for (uint g = 0; g < P.numGlobals; g++) {
        uint b = globalIdx[g];
        uint ownerB = colliderOwner[b];
        if (ownerA == ownerB) continue;
        if (!aDyn && posLin[ownerB].w <= 0.0f) continue;
        if (ga != 0 && clothGroup[ownerB] == ga) continue;
        float rb = fabs(shape[b].w);
        float3 pb = bpColliderCenter(posLin, posAng, colliderOwner,
                                     colliderLocalPosition, b);
        float3 dp = pa - pb;
        float r = ra + rb;
        if (dot(dp, dp) > r * r) continue;
        uint ownerLo = min(ownerA, ownerB), ownerHi = max(ownerA, ownerB);
        if (pairExcluded(exclusions, numExclusions, ownerLo, ownerHi)) continue;
        uint lo = min(a, b), hi = max(a, b);
        uint slot = atomic_fetch_add_explicit(&counters[CTR_PAIRS], 1u, memory_order_relaxed);
        if (slot < P.maxPairs) pairs[slot] = uint2(lo, hi);
    }
}

// Pairs among the global (oversized/static) body set: tiny O(g^2) sweep.
kernel void bp_gen_global_pairs(
    device const float4* posLin     [[buffer(0)]],
    device const float4* shape      [[buffer(1)]],
    device const uint* globalIdx    [[buffer(2)]],
    device const uint2* exclusions  [[buffer(3)]],
    constant uint& numExclusions    [[buffer(4)]],
    device atomic_uint* counters    [[buffer(5)]],
    device uint2* pairs             [[buffer(6)]],
    constant SimParams& P           [[buffer(7)]],
    device const uint* clothGroup   [[buffer(8)]],
    device const uint* colliderOwner [[buffer(9)]],
    device const float4* colliderLocalPosition [[buffer(10)]],
    device const float4* posAng     [[buffer(11)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numGlobals) return;
    uint a = globalIdx[gid];
    uint ownerA = colliderOwner[a];
    float3 pa = bpColliderCenter(posLin, posAng, colliderOwner,
                                 colliderLocalPosition, a);
    float ra = fabs(shape[a].w);
    // Skip static-static early (neither can move)
    bool aDyn = posLin[ownerA].w > 0.0f;

    for (uint g = gid + 1; g < P.numGlobals; g++) {
        uint b = globalIdx[g];
        uint ownerB = colliderOwner[b];
        if (ownerA == ownerB) continue;
        if (!aDyn && posLin[ownerB].w <= 0.0f) continue;
        uint ga2 = clothGroup[ownerA];
        if (ga2 != 0 && clothGroup[ownerB] == ga2) continue;
        float rb = fabs(shape[b].w);
        float3 pb = bpColliderCenter(posLin, posAng, colliderOwner,
                                     colliderLocalPosition, b);
        float3 dp = pa - pb;
        float r = ra + rb;
        if (dot(dp, dp) > r * r) continue;
        uint ownerLo = min(ownerA, ownerB), ownerHi = max(ownerA, ownerB);
        if (pairExcluded(exclusions, numExclusions, ownerLo, ownerHi)) continue;
        uint lo = min(a, b), hi = max(a, b);
        uint slot = atomic_fetch_add_explicit(&counters[CTR_PAIRS], 1u, memory_order_relaxed);
        if (slot < P.maxPairs) pairs[slot] = uint2(lo, hi);
    }
}

// Deterministic pair generation. Each collider producer first counts its
// accepted candidates, an exclusive scan assigns a stable output range, and
// the second pass emits in collider/cell order. This deliberately duplicates
// the inexpensive candidate predicates in count and emit to remove the
// scheduling-dependent atomic append from the physical state transition.
inline uint bpProcessProducer(
    device const float4* posLin,
    device const float4* shape,
    device const uint* hashedIdx,
    device const uint* cellStart,
    device const uint* cellCount,
    device const uint* cellBodies,
    device const uint* globalIdx,
    device const uint2* exclusions,
    uint numExclusions,
    device const uint* clothGroup,
    device const uint* cellRigid,
    device const uint* colliderOwner,
    device const float4* colliderLocalPosition,
    device const float4* posAng,
    device const uint* colliderGroup,
    constant SimParams& P,
    uint producer,
    bool emit,
    uint outputBase,
    device uint2* pairs)
{
    uint count = 0;
    if (producer < P.numHashed) {
        uint a = hashedIdx[producer];
        uint ownerA = colliderOwner[a];
        uint collisionGroupA = colliderGroup[a];
        float3 pa = bpColliderCenter(posLin, posAng, colliderOwner,
                                     colliderLocalPosition, a);
        float ra = fabs(shape[a].w);
        bool aDyn = posLin[ownerA].w > 0.0f;
        uint ga = clothGroup[ownerA];
        bool scanCells = !(ga != 0 && P.numHashedRigid == 0);
        int3 cc = cellCoord(pa, P.cellSize);
        thread uint visited[27];
        uint visitedCount = 0;
        if (scanCells) {
            for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                uint h = cellHash(cc + int3(dx, dy, dz), collisionGroupA,
                                  P.gridHashSize);
                bool duplicateBucket = false;
                for (uint v = 0; v < visitedCount; v++) {
                    if (visited[v] == h) { duplicateBucket = true; break; }
                }
                if (duplicateBucket) continue;
                visited[visitedCount++] = h;
                if (ga != 0 && cellRigid[h] == 0) continue;
                uint s = cellStart[h], e = s + cellCount[h];
                for (uint k = s; k < e; k++) {
                    uint b = cellBodies[k];
                    if (b <= a) continue;
                    if (!collisionGroupsCompatible(collisionGroupA,
                                                   colliderGroup[b])) continue;
                    uint ownerB = colliderOwner[b];
                    if (ownerA == ownerB) continue;
                    if (!aDyn && posLin[ownerB].w <= 0.0f) continue;
                    if (ga != 0 && clothGroup[ownerB] != 0) continue;
                    float rb = fabs(shape[b].w);
                    float3 pb = bpColliderCenter(posLin, posAng, colliderOwner,
                                                 colliderLocalPosition, b);
                    float3 dp = pa - pb;
                    float r = ra + rb;
                    if (dot(dp, dp) > r * r) continue;
                    uint ownerLo = min(ownerA, ownerB);
                    uint ownerHi = max(ownerA, ownerB);
                    if (pairExcluded(exclusions, numExclusions,
                                     ownerLo, ownerHi)) continue;
                    uint slot = outputBase + count;
                    if (emit && slot < P.maxPairs) pairs[slot] = uint2(a, b);
                    count++;
                }
            }
        }
        for (uint g = 0; g < P.numGlobals; g++) {
            uint b = globalIdx[g];
            if (!collisionGroupsCompatible(collisionGroupA,
                                           colliderGroup[b])) continue;
            uint ownerB = colliderOwner[b];
            if (ownerA == ownerB) continue;
            if (!aDyn && posLin[ownerB].w <= 0.0f) continue;
            if (ga != 0 && clothGroup[ownerB] == ga) continue;
            float rb = fabs(shape[b].w);
            float3 pb = bpColliderCenter(posLin, posAng, colliderOwner,
                                         colliderLocalPosition, b);
            float3 dp = pa - pb;
            float r = ra + rb;
            if (dot(dp, dp) > r * r) continue;
            uint ownerLo = min(ownerA, ownerB), ownerHi = max(ownerA, ownerB);
            if (pairExcluded(exclusions, numExclusions, ownerLo, ownerHi)) continue;
            uint slot = outputBase + count;
            if (emit && slot < P.maxPairs)
                pairs[slot] = uint2(min(a, b), max(a, b));
            count++;
        }
    } else {
        uint globalProducer = producer - P.numHashed;
        if (globalProducer >= P.numGlobals) return 0;
        uint a = globalIdx[globalProducer];
        uint ownerA = colliderOwner[a];
        uint collisionGroupA = colliderGroup[a];
        float3 pa = bpColliderCenter(posLin, posAng, colliderOwner,
                                     colliderLocalPosition, a);
        float ra = fabs(shape[a].w);
        bool aDyn = posLin[ownerA].w > 0.0f;
        uint ga = clothGroup[ownerA];
        for (uint g = globalProducer + 1; g < P.numGlobals; g++) {
            uint b = globalIdx[g];
            if (!collisionGroupsCompatible(collisionGroupA,
                                           colliderGroup[b])) continue;
            uint ownerB = colliderOwner[b];
            if (ownerA == ownerB) continue;
            if (!aDyn && posLin[ownerB].w <= 0.0f) continue;
            if (ga != 0 && clothGroup[ownerB] == ga) continue;
            float rb = fabs(shape[b].w);
            float3 pb = bpColliderCenter(posLin, posAng, colliderOwner,
                                         colliderLocalPosition, b);
            float3 dp = pa - pb;
            float r = ra + rb;
            if (dot(dp, dp) > r * r) continue;
            uint ownerLo = min(ownerA, ownerB), ownerHi = max(ownerA, ownerB);
            if (pairExcluded(exclusions, numExclusions, ownerLo, ownerHi)) continue;
            uint slot = outputBase + count;
            if (emit && slot < P.maxPairs)
                pairs[slot] = uint2(min(a, b), max(a, b));
            count++;
        }
    }
    return count;
}

kernel void bp_count_pairs_deterministic(
    device const float4* posLin     [[buffer(0)]],
    device const float4* shape      [[buffer(1)]],
    device const uint* hashedIdx    [[buffer(2)]],
    device const uint* cellStart    [[buffer(3)]],
    device const uint* cellCount    [[buffer(4)]],
    device const uint* cellBodies   [[buffer(5)]],
    device const uint* globalIdx    [[buffer(6)]],
    device const uint2* exclusions  [[buffer(7)]],
    constant uint& numExclusions    [[buffer(8)]],
    device uint* pairCounts         [[buffer(9)]],
    device uint2* pairs             [[buffer(10)]],
    constant SimParams& P           [[buffer(11)]],
    device const uint* clothGroup   [[buffer(12)]],
    device const uint* cellRigid    [[buffer(13)]],
    device const uint* colliderOwner [[buffer(14)]],
    device const float4* colliderLocalPosition [[buffer(15)]],
    device const float4* posAng     [[buffer(16)]],
    device const uint* colliderGroup [[buffer(17)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numHashed + P.numGlobals) return;
    pairCounts[gid] = bpProcessProducer(
        posLin, shape, hashedIdx, cellStart, cellCount, cellBodies,
        globalIdx, exclusions, numExclusions, clothGroup, cellRigid,
        colliderOwner, colliderLocalPosition, posAng, colliderGroup,
        P, gid, false, 0, pairs);
}

kernel void bp_emit_pairs_deterministic(
    device const float4* posLin     [[buffer(0)]],
    device const float4* shape      [[buffer(1)]],
    device const uint* hashedIdx    [[buffer(2)]],
    device const uint* cellStart    [[buffer(3)]],
    device const uint* cellCount    [[buffer(4)]],
    device const uint* cellBodies   [[buffer(5)]],
    device const uint* globalIdx    [[buffer(6)]],
    device const uint2* exclusions  [[buffer(7)]],
    constant uint& numExclusions    [[buffer(8)]],
    device const uint* pairStarts   [[buffer(9)]],
    device uint2* pairs             [[buffer(10)]],
    constant SimParams& P           [[buffer(11)]],
    device const uint* clothGroup   [[buffer(12)]],
    device const uint* cellRigid    [[buffer(13)]],
    device const uint* colliderOwner [[buffer(14)]],
    device const float4* colliderLocalPosition [[buffer(15)]],
    device const float4* posAng     [[buffer(16)]],
    device const uint* colliderGroup [[buffer(17)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numHashed + P.numGlobals) return;
    (void)bpProcessProducer(
        posLin, shape, hashedIdx, cellStart, cellCount, cellBodies,
        globalIdx, exclusions, numExclusions, clothGroup, cellRigid,
        colliderOwner, colliderLocalPosition, posAng, colliderGroup, P, gid, true,
        pairStarts[gid], pairs);
}

kernel void bp_finalize_deterministic_pairs(
    device const uint* pairCounts  [[buffer(0)]],
    device const uint* pairStarts  [[buffer(1)]],
    device atomic_uint* counters   [[buffer(2)]],
    device uint* dispatchArgs      [[buffer(3)]],
    constant SimParams& P          [[buffer(4)]])
{
    uint producers = P.numHashed + P.numGlobals;
    uint n = producers == 0 ? 0
        : pairStarts[producers - 1] + pairCounts[producers - 1];
    n = min(n, P.maxPairs);
    atomic_store_explicit(&counters[CTR_PAIRS], n, memory_order_relaxed);
    dispatchArgs[0] = (n + 63) / 64;
    dispatchArgs[1] = 1;
    dispatchArgs[2] = 1;
    dispatchArgs[3] = (P.numJoints + P.numSprings + n + P.numTets
                       + P.numMembranes + P.numBends + 63) / 64;
    dispatchArgs[4] = 1;
    dispatchArgs[5] = 1;
    dispatchArgs[6] = (P.numJoints + P.numSprings + n + 63) / 64;
    dispatchArgs[7] = 1;
    dispatchArgs[8] = 1;
}

// Clamp pair count to capacity and write indirect dispatch args:
// [0..2]  pairs (narrowphase, dual_manifolds, pm_insert)
// [3..5]  forces = joints + springs + pairs (adjacency kernels)
// [6..8]  diag = joints + pairs
kernel void bp_finalize_pairs(
    device atomic_uint* counters    [[buffer(0)]],
    device uint* dispatchArgs       [[buffer(1)]],   // 9 uints
    constant SimParams& P           [[buffer(2)]])
{
    uint n = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    n = min(n, P.maxPairs);
    atomic_store_explicit(&counters[CTR_PAIRS], n, memory_order_relaxed);
    dispatchArgs[0] = (n + 63) / 64;
    dispatchArgs[1] = 1;
    dispatchArgs[2] = 1;
    dispatchArgs[3] = (P.numJoints + P.numSprings + n + P.numTets
                       + P.numMembranes + P.numBends + 63) / 64;
    dispatchArgs[4] = 1;
    dispatchArgs[5] = 1;
    dispatchArgs[6] = (P.numJoints + P.numSprings + n + 63) / 64;
    dispatchArgs[7] = 1;
    dispatchArgs[8] = 1;
}

// ----------------------------------------------------------------------------
// Collider-pair-keyed manifold persistence map (open addressing, linear
// probing). Distinct shapes attached to the same two links must not overwrite
// each other's dual/contact state.
// keyA buffer: 0 = empty, else colliderA+1 (CAS target). keyB and val written
// by the CAS winner only.
// ----------------------------------------------------------------------------
inline uint pairMapHash(uint a, uint b, uint capacity) {
    uint h = (a * 73856093u) ^ (b * 19349663u);
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    return h & (capacity - 1u);
}

kernel void pm_clear(
    device atomic_uint* mapKeyA     [[buffer(0)]],
    constant uint& capacity         [[buffer(1)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid < capacity) {
        atomic_store_explicit(&mapKeyA[gid], 0u, memory_order_relaxed);
    }
}

// Insert active manifolds of THIS frame into the map (queried next frame).
kernel void pm_insert(
    device const ManifoldGPU* manifolds [[buffer(0)]],
    device atomic_uint* mapKeyA     [[buffer(1)]],
    device uint* mapKeyB            [[buffer(2)]],
    device uint* mapVal             [[buffer(3)]],
    device const atomic_uint* counters [[buffer(4)]],
    constant SimParams& P           [[buffer(5)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    if (gid >= numPairs) return;
    uint4 header = manifolds[gid].header;
    if (header.z == 0) return;      // no contacts

    uint a = manifolds[gid].colliderPair.x;
    uint b = manifolds[gid].colliderPair.y;
    uint h = pairMapHash(a, b, P.mapCapacity);
    for (uint probe = 0; probe < P.mapCapacity; probe++) {
        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(&mapKeyA[h], &expected, a + 1,
                                                  memory_order_relaxed, memory_order_relaxed)) {
            mapKeyB[h] = b;
            mapVal[h] = gid;
            return;
        }
        h = (h + 1) & (P.mapCapacity - 1u);
    }
}

// Query helper used by narrowphase (defined here, used in 30_narrowphase)
inline int pairMapFind(device const atomic_uint* mapKeyA,
                       device const uint* mapKeyB,
                       device const uint* mapVal,
                       uint capacity, uint a, uint b) {
    uint h = pairMapHash(a, b, capacity);
    for (uint probe = 0; probe < capacity; probe++) {
        uint ka = atomic_load_explicit(&mapKeyA[h], memory_order_relaxed);
        if (ka == 0) return -1;
        if (ka == a + 1 && mapKeyB[h] == b) return int(mapVal[h]);
        h = (h + 1) & (capacity - 1u);
    }
    return -1;
}
