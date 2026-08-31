#define OGC_BARRIER 1
#include <metal_stdlib>
using namespace metal;

#define ORDERED_COLORING_BODY_LIMIT 1024u

// ============================================================================
// AVBD solver kernels: adjacency (CSR), coloring, warm start, primal update
// (per color, 6x6 LDL in registers), dual update, velocity finalize.
// ============================================================================

// ----------------------------------------------------------------------------
// CSR adjacency build
// ----------------------------------------------------------------------------

inline bool bodyDynamic(device const float4* posLin, uint b) {
    return b != WORLD_BODY && posLin[b].w > 0.0f;
}

kernel void adj_clear_degrees(
    device atomic_uint* degrees     [[buffer(0)]],
    constant uint& numBodies        [[buffer(1)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid < numBodies) atomic_store_explicit(&degrees[gid], 0u, memory_order_relaxed);
}

// One thread per force across all three kinds, ordered [joints | springs | manifolds].
kernel void adj_count(
    device const float4* posLin     [[buffer(0)]],
    device const JointGPU* joints   [[buffer(1)]],
    device const SpringGPU* springs [[buffer(2)]],
    device const ManifoldGPU* manifolds [[buffer(3)]],
    device atomic_uint* degrees     [[buffer(4)]],
    device const atomic_uint* counters [[buffer(5)]],
    constant SimParams& P           [[buffer(6)]],
    device const TetGPU* tets       [[buffer(7)]],
    device const SoftContactGPU* soft [[buffer(8)]],
    device const MembraneGPU* membranes [[buffer(9)]],
    device const BendGPU* bends     [[buffer(10)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    uint numSoft = min(atomic_load_explicit(&counters[CTR_SOFT], memory_order_relaxed),
                       P.maxSoft);
    uint base = P.numJoints + P.numSprings + numPairs + P.numTets + numSoft;
    uint total = base + P.numMembranes + P.numBends;
    if (gid >= total) return;
    if (gid >= base) {
        // membrane (3 slots) or bend (4 slots): degree per participant
        uint mi = gid - base;
        if (mi < P.numMembranes) {
            uint4 ids = membranes[mi].ids;
            for (uint k = 0; k < 3; k++) {
                if (bodyDynamic(posLin, ids[k]))
                    atomic_fetch_add_explicit(&degrees[ids[k]], 1u, memory_order_relaxed);
            }
        } else {
            uint4 ids = bends[mi - P.numMembranes].ids;
            for (uint k = 0; k < 4; k++) {
                if (bodyDynamic(posLin, ids[k]))
                    atomic_fetch_add_explicit(&degrees[ids[k]], 1u, memory_order_relaxed);
            }
        }
        return;
    }

    uint a = WORLD_BODY, b = WORLD_BODY;
    if (gid < P.numJoints) {
        if (joints[gid].header.z != 0) return;  // broken
        // inert (exclusion-only / inactive drag slots): nothing to stamp
        if (joints[gid].rA.w == 0.0f && joints[gid].rB.w == 0.0f
            && joints[gid].motor.y == 0.0f) return;
        a = joints[gid].header.x;
        b = joints[gid].header.y;
    } else if (gid < P.numJoints + P.numSprings) {
        uint s = gid - P.numJoints;
        a = springs[s].header.x;
        b = springs[s].header.y;
    } else if (gid < P.numJoints + P.numSprings + numPairs) {
        uint m = gid - P.numJoints - P.numSprings;
        if (manifolds[m].header.z == 0) return; // inactive
        a = manifolds[m].header.x;
        b = manifolds[m].header.y;
    } else if (gid < P.numJoints + P.numSprings + numPairs + P.numTets) {
        // tet: one adjacency entry per vertex
        uint t = gid - P.numJoints - P.numSprings - numPairs;
        uint4 ids = tets[t].ids;
        for (uint k = 0; k < 4; k++) {
            uint v = ids[k];
            if (bodyDynamic(posLin, v))
                atomic_fetch_add_explicit(&degrees[v], 1u, memory_order_relaxed);
        }
        return;
    } else {
        // element contact: one adjacency entry per participating body
        uint s = gid - P.numJoints - P.numSprings - numPairs - P.numTets;
        uint4 ids = soft[s].ids;
        for (uint k = 0; k < 4; k++) {
            uint v = ids[k];
            if (v != WORLD_BODY && bodyDynamic(posLin, v))
                atomic_fetch_add_explicit(&degrees[v], 1u, memory_order_relaxed);
        }
        return;
    }
    if (bodyDynamic(posLin, a)) atomic_fetch_add_explicit(&degrees[a], 1u, memory_order_relaxed);
    if (bodyDynamic(posLin, b)) atomic_fetch_add_explicit(&degrees[b], 1u, memory_order_relaxed);
}

kernel void adj_copy_cursor(
    device const uint* adjStart     [[buffer(0)]],
    device atomic_uint* cursor      [[buffer(1)]],
    constant uint& numBodies        [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid < numBodies) {
        atomic_store_explicit(&cursor[gid], adjStart[gid], memory_order_relaxed);
    }
}

kernel void adj_scatter(
    device const float4* posLin     [[buffer(0)]],
    device const JointGPU* joints   [[buffer(1)]],
    device const SpringGPU* springs [[buffer(2)]],
    device const ManifoldGPU* manifolds [[buffer(3)]],
    device atomic_uint* cursor      [[buffer(4)]],
    device uint* adjList            [[buffer(5)]],
    device const atomic_uint* counters [[buffer(6)]],
    constant SimParams& P           [[buffer(7)]],
    device const TetGPU* tets       [[buffer(8)]],
    device const SoftContactGPU* soft [[buffer(9)]],
    device const MembraneGPU* membranes [[buffer(10)]],
    device const BendGPU* bends     [[buffer(11)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    uint numSoft = min(atomic_load_explicit(&counters[CTR_SOFT], memory_order_relaxed),
                       P.maxSoft);
    uint base = P.numJoints + P.numSprings + numPairs + P.numTets + numSoft;
    uint total = base + P.numMembranes + P.numBends;
    if (gid >= total) return;
    if (gid >= base) {
        uint mi = gid - base;
        if (mi < P.numMembranes) {
            uint4 ids = membranes[mi].ids;
            uint entry = (FK_MEMBRANE << ADJ_KIND_SHIFT) | mi;
            for (uint k = 0; k < 3; k++) {
                if (bodyDynamic(posLin, ids[k])) {
                    uint slot = atomic_fetch_add_explicit(&cursor[ids[k]], 1u, memory_order_relaxed);
                    adjList[slot] = entry;
                }
            }
        } else {
            uint bi = mi - P.numMembranes;
            uint4 ids = bends[bi].ids;
            uint entry = (FK_BEND << ADJ_KIND_SHIFT) | bi;
            for (uint k = 0; k < 4; k++) {
                if (bodyDynamic(posLin, ids[k])) {
                    uint slot = atomic_fetch_add_explicit(&cursor[ids[k]], 1u, memory_order_relaxed);
                    adjList[slot] = entry;
                }
            }
        }
        return;
    }

    uint a = WORLD_BODY, b = WORLD_BODY;
    uint entry = 0;
    if (gid < P.numJoints) {
        if (joints[gid].header.z != 0) return;
        if (joints[gid].rA.w == 0.0f && joints[gid].rB.w == 0.0f
            && joints[gid].motor.y == 0.0f) return;
        a = joints[gid].header.x;
        b = joints[gid].header.y;
        entry = (FK_JOINT << ADJ_KIND_SHIFT) | gid;
    } else if (gid < P.numJoints + P.numSprings) {
        uint s = gid - P.numJoints;
        a = springs[s].header.x;
        b = springs[s].header.y;
        entry = (FK_SPRING << ADJ_KIND_SHIFT) | s;
    } else if (gid < P.numJoints + P.numSprings + numPairs) {
        uint m = gid - P.numJoints - P.numSprings;
        if (manifolds[m].header.z == 0) return;
        a = manifolds[m].header.x;
        b = manifolds[m].header.y;
        entry = (FK_MANIFOLD << ADJ_KIND_SHIFT) | m;
    } else if (gid < P.numJoints + P.numSprings + numPairs + P.numTets) {
        uint t = gid - P.numJoints - P.numSprings - numPairs;
        uint4 ids = tets[t].ids;
        entry = (FK_TET << ADJ_KIND_SHIFT) | t;
        for (uint k = 0; k < 4; k++) {
            uint v = ids[k];
            if (bodyDynamic(posLin, v)) {
                uint slot = atomic_fetch_add_explicit(&cursor[v], 1u, memory_order_relaxed);
                adjList[slot] = entry;
            }
        }
        return;
    } else {
        uint s = gid - P.numJoints - P.numSprings - numPairs - P.numTets;
        uint4 ids = soft[s].ids;
        entry = (FK_SOFT << ADJ_KIND_SHIFT) | s;
        for (uint k = 0; k < 4; k++) {
            uint v = ids[k];
            if (v != WORLD_BODY && bodyDynamic(posLin, v)) {
                uint slot = atomic_fetch_add_explicit(&cursor[v], 1u, memory_order_relaxed);
                adjList[slot] = entry;
            }
        }
        return;
    }
    if (bodyDynamic(posLin, a)) {
        uint slot = atomic_fetch_add_explicit(&cursor[a], 1u, memory_order_relaxed);
        adjList[slot] = entry;
    }
    if (bodyDynamic(posLin, b)) {
        uint slot = atomic_fetch_add_explicit(&cursor[b], 1u, memory_order_relaxed);
        adjList[slot] = entry;
    }
}

// Atomic scatter gives each body the correct force set but a scheduling-
// dependent summation order. Once pair/manifold indices are deterministic,
// sorting each (typically short) CSR segment makes every primal accumulation
// bitwise repeatable for a fixed input state.
kernel void adj_sort(
    device const uint* adjStart     [[buffer(0)]],
    device const uint* adjCount     [[buffer(1)]],
    device uint* adjList            [[buffer(2)]],
    constant uint& numBodies        [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= numBodies) return;
    uint s = adjStart[gid];
    uint n = adjCount[gid];
    for (uint i = 1; i < n; i++) {
        uint value = adjList[s + i];
        uint j = i;
        while (j > 0 && adjList[s + j - 1] > value) {
            adjList[s + j] = adjList[s + j - 1];
            j--;
        }
        adjList[s + j] = value;
    }
}

// ----------------------------------------------------------------------------
// Graph coloring: incremental parallel greedy (paper Sec. 4).
// Colors persist across frames; only dynamic-dynamic edges constrain colors.
// ----------------------------------------------------------------------------

inline uint otherBody(device const JointGPU* joints,
                      device const SpringGPU* springs,
                      device const ManifoldGPU* manifolds,
                      uint entry, uint self) {
    uint kind = entry >> ADJ_KIND_SHIFT;
    uint idx = entry & ADJ_INDEX_MASK;
    uint a, b;
    if (kind == FK_JOINT) {
        a = joints[idx].header.x; b = joints[idx].header.y;
    } else if (kind == FK_SPRING) {
        a = springs[idx].header.x; b = springs[idx].header.y;
    } else {
        a = manifolds[idx].header.x; b = manifolds[idx].header.y;
    }
    return a == self ? b : a;
}

// Keep the established small-scene path unchanged. It supports every force
// kind even though dynamic coloring currently runs only for rigid scenes.
inline void neighborColors(device const JointGPU* joints,
                           device const SpringGPU* springs,
                           device const ManifoldGPU* manifolds,
                           device const TetGPU* tets,
                           device const SoftContactGPU* soft,
                           device const MembraneGPU* membranes,
                           device const BendGPU* bends,
                           device const float4* posLin,
                           device const uint* colorsIn,
                           uint entry, uint self, uint myColor,
                           thread uint& maskLo, thread uint& maskHi,
                           thread uint& allLo, thread uint& allHi,
                           thread bool& conflict) {
    uint kind = entry >> ADJ_KIND_SHIFT;
    uint idx = entry & ADJ_INDEX_MASK;
    uint nbs[3];
    uint n = 0;
    if (kind == FK_TET) {
        uint4 ids = tets[idx].ids;
        for (uint k = 0; k < 4; k++) {
            if (ids[k] != self && n < 3) { nbs[n++] = ids[k]; }
        }
    } else if (kind == FK_SOFT) {
        uint4 ids = soft[idx].ids;
        for (uint k = 0; k < 4; k++) {
            if (ids[k] != self && ids[k] != WORLD_BODY && n < 3) {
                nbs[n++] = ids[k];
            }
        }
    } else if (kind == FK_MEMBRANE) {
        uint4 ids = membranes[idx].ids;
        for (uint k = 0; k < 3; k++) {
            if (ids[k] != self && n < 3) { nbs[n++] = ids[k]; }
        }
    } else if (kind == FK_BEND) {
        uint4 ids = bends[idx].ids;
        for (uint k = 0; k < 4; k++) {
            if (ids[k] != self && n < 3) { nbs[n++] = ids[k]; }
        }
    } else {
        nbs[0] = otherBody(joints, springs, manifolds, entry, self);
        n = 1;
    }
    for (uint k = 0; k < n; k++) {
        uint nb = nbs[k];
        if (nb == WORLD_BODY || posLin[nb].w <= 0.0f) continue;
        uint nc = colorsIn[nb];
        uint lo = nc < 32 ? (1u << nc) : 0;
        uint hi = (nc >= 32 && nc < 64) ? (1u << (nc - 32)) : 0;
        allLo |= lo; allHi |= hi;
        if (nb < self) {
            maskLo |= lo; maskHi |= hi;
            if (nc == myColor) conflict = true;
        }
    }
}

// Extract one compact opposite-body id per sorted rigid CSR entry. This pass
// runs only for large rigid scenes, leaving the established adjacency and
// coloring execution of existing scenes untouched.
kernel void adj_extract_neighbors(
    device const JointGPU* joints   [[buffer(0)]],
    device const SpringGPU* springs [[buffer(1)]],
    device const ManifoldGPU* manifolds [[buffer(2)]],
    device const uint* adjStart     [[buffer(3)]],
    device const uint* adjCount     [[buffer(4)]],
    device const uint* adjList      [[buffer(5)]],
    device uint* adjNeighbor        [[buffer(6)]],
    constant uint& numBodies        [[buffer(7)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= numBodies) return;
    uint s = adjStart[gid], e = s + adjCount[gid];
    for (uint k = s; k < e; k++) {
        adjNeighbor[k] = otherBody(joints, springs, manifolds,
                                   adjList[k], gid);
    }
}

// Dynamic coloring is used only for rigid scenes, whose forces all have two
// endpoints. Large scenes read the extracted opposite endpoint stream instead
// of randomly dereferencing 704-byte manifold records on every pass.
inline void neighborColor(device const float4* posLin,
                           device const uint* colorsIn,
                           uint nb, uint self, uint myColor,
                           thread uint& maskLo, thread uint& maskHi,
                           thread uint& allLo, thread uint& allHi,
                           thread bool& conflict) {
    if (nb == WORLD_BODY || posLin[nb].w <= 0.0f) return;
    uint nc = colorsIn[nb];
    uint lo = nc < 32 ? (1u << nc) : 0;
    uint hi = (nc >= 32 && nc < 64) ? (1u << (nc - 32)) : 0;
    allLo |= lo; allHi |= hi;
    if (nb < self) {
        maskLo |= lo; maskHi |= hi;
        if (nc == myColor) conflict = true;
    }
}

kernel void color_iterate(
    device const float4* posLin     [[buffer(0)]],
    device const JointGPU* joints   [[buffer(1)]],
    device const SpringGPU* springs [[buffer(2)]],
    device const ManifoldGPU* manifolds [[buffer(3)]],
    device const uint* adjStart     [[buffer(4)]],
    device const uint* adjCount     [[buffer(5)]],
    device const uint* adjList      [[buffer(6)]],
    device const uint* colorsIn     [[buffer(7)]],
    device uint* colorsOut          [[buffer(8)]],
    device atomic_uint* changedFlag [[buffer(9)]],
    constant SimParams& P           [[buffer(10)]],
    device const TetGPU* tets       [[buffer(11)]],
    device const SoftContactGPU* soft [[buffer(12)]],
    device const MembraneGPU* membranes [[buffer(13)]],
    device const BendGPU* bends     [[buffer(14)]],
    constant uint& passIdx          [[buffer(15)]],
    device const uint* adjNeighbor  [[buffer(16)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numBodies) return;
    // Converged early-out: if the previous pass changed nothing, the
    // coloring is a fixed point and this pass's dst (written two passes
    // ago with the same fixed point) is already correct — skip the
    // adjacency walk. Warm-started colors converge in 1-3 passes on
    // settled scenes; the remaining passes cost launch latency only.
    if (passIdx > 0 &&
        atomic_load_explicit(&changedFlag[passIdx - 1],
                             memory_order_relaxed) == 0u) return;
    if (posLin[gid].w <= 0.0f) { colorsOut[gid] = 0; return; }

    uint myColor = colorsIn[gid];
    uint maskLo = 0, maskHi = 0;        // colors of smaller-index neighbors
    uint allLo = 0, allHi = 0;          // colors of all dynamic neighbors
    bool conflict = false;

    uint s = adjStart[gid], e = s + adjCount[gid];
    bool orderedCompatibility = P.numBodies
        <= ORDERED_COLORING_BODY_LIMIT;
    for (uint k = s; k < e; k++) {
        if (orderedCompatibility) {
            neighborColors(joints, springs, manifolds, tets, soft,
                           membranes, bends, posLin, colorsIn, adjList[k],
                           gid, myColor, maskLo, maskHi, allLo, allHi,
                           conflict);
        } else {
            neighborColor(posLin, colorsIn, adjNeighbor[k], gid, myColor,
                          maskLo, maskHi, allLo, allHi, conflict);
        }
    }

    if (conflict) {
        // Preserve the established small-scene ordering, while large graphs
        // pick a color free across ALL current neighbors. Restricting the
        // latter to smaller neighbors makes convergence travel one graph hop
        // at a time.
        uint freeLo = orderedCompatibility ? ~maskLo : ~allLo;
        uint newColor;
        if (freeLo != 0) newColor = ctz(freeLo);
        else {
            uint freeHi = orderedCompatibility ? ~maskHi : ~allHi;
            newColor = freeHi != 0 ? 32 + ctz(freeHi) : (MAX_COLORS - 1);
        }
        colorsOut[gid] = min(newColor, uint(MAX_COLORS - 1));
        atomic_fetch_or_explicit(&changedFlag[passIdx], 1u, memory_order_relaxed);
    } else if (P.numBodies > ORDERED_COLORING_BODY_LIMIT) {
        // Preserve valid warm-started colors. Recompacting every body every
        // frame made dense contact graphs oscillate for all 20 passes and
        // forced the serial fallback even when no contact changed.
        colorsOut[gid] = myColor;
    } else {
        // The existing demos below the scaling threshold retain their exact
        // palette compaction and therefore their authored solve order.
        uint freeLo = ~allLo;
        uint best = freeLo != 0 ? ctz(freeLo)
                  : (~allHi != 0 ? 32 + ctz(~allHi) : myColor);
        colorsOut[gid] = best < myColor ? best : myColor;
        if (best < myColor)
            atomic_fetch_or_explicit(&changedFlag[passIdx], 1u,
                                     memory_order_relaxed);
    }
}

// Complete an unfinished parallel coloring deterministically. A synchronous
// ordered relaxation can require one pass per graph hop, so the bounded
// parallel phase is only an optimizer. This sweep runs only when the final
// pass still changed; color_validate remains the exact correctness boundary.
kernel void color_repair_greedy(
    device const float4* posLin     [[buffer(0)]],
    device const JointGPU* joints   [[buffer(1)]],
    device const SpringGPU* springs [[buffer(2)]],
    device const ManifoldGPU* manifolds [[buffer(3)]],
    device const uint* adjStart     [[buffer(4)]],
    device const uint* adjCount     [[buffer(5)]],
    device const uint* adjList      [[buffer(6)]],
    device uint* colors             [[buffer(7)]],
    device const atomic_uint* changedFlag [[buffer(8)]],
    constant SimParams& P           [[buffer(9)]],
    device const TetGPU* tets       [[buffer(10)]],
    device const SoftContactGPU* soft [[buffer(11)]],
    device const MembraneGPU* membranes [[buffer(12)]],
    device const BendGPU* bends     [[buffer(13)]],
    constant uint& finalPass        [[buffer(14)]],
    device const uint* adjNeighbor  [[buffer(15)]],
    uint gid                        [[thread_position_in_grid]])
{
    bool needsRepair = atomic_load_explicit(&changedFlag[finalPass],
                                            memory_order_relaxed) != 0u;
    if (gid != 0u || !needsRepair) return;

    for (uint self = 0; self < P.numBodies; self++) {
        if (posLin[self].w <= 0.0f) {
            colors[self] = 0u;
            continue;
        }
        uint maskLo = 0u, maskHi = 0u;
        uint allLo = 0u, allHi = 0u;
        bool conflict = false;
        uint s = adjStart[self], e = s + adjCount[self];
        bool orderedCompatibility = P.numBodies
            <= ORDERED_COLORING_BODY_LIMIT;
        for (uint k = s; k < e; k++) {
            if (orderedCompatibility) {
                neighborColors(joints, springs, manifolds, tets, soft,
                               membranes, bends, posLin, colors, adjList[k],
                               self, colors[self], maskLo, maskHi, allLo,
                               allHi, conflict);
            } else {
                neighborColor(posLin, colors, adjNeighbor[k], self,
                              colors[self], maskLo, maskHi, allLo, allHi,
                              conflict);
            }
        }
        uint freeLo = ~maskLo;
        if (freeLo != 0u) {
            colors[self] = ctz(freeLo);
        } else {
            uint freeHi = ~maskHi;
            // Keeping the last color when all 64 are occupied deliberately
            // leaves a conflict for color_validate to report fail-closed.
            colors[self] = freeHi != 0u
                ? 32u + ctz(freeHi) : uint(MAX_COLORS - 1);
        }
    }
}

// Verify the final dynamic palette instead of trusting a fixed iteration
// count. Each conflicting edge is reported once by its larger body id. A
// conflict means same-color primal updates would race and the frame is not a
// valid AVBD solve.
kernel void color_validate(
    device const float4* posLin     [[buffer(0)]],
    device const JointGPU* joints   [[buffer(1)]],
    device const SpringGPU* springs [[buffer(2)]],
    device const ManifoldGPU* manifolds [[buffer(3)]],
    device const uint* adjStart     [[buffer(4)]],
    device const uint* adjCount     [[buffer(5)]],
    device const uint* adjList      [[buffer(6)]],
    device const uint* colors       [[buffer(7)]],
    device atomic_uint* counters    [[buffer(8)]],
    constant SimParams& P           [[buffer(9)]],
    device const TetGPU* tets       [[buffer(10)]],
    device const SoftContactGPU* soft [[buffer(11)]],
    device const MembraneGPU* membranes [[buffer(12)]],
    device const BendGPU* bends     [[buffer(13)]],
    device const uint* adjNeighbor  [[buffer(14)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numBodies || posLin[gid].w <= 0.0f) return;
    uint maskLo = 0, maskHi = 0, allLo = 0, allHi = 0;
    bool conflict = false;
    uint s = adjStart[gid], e = s + adjCount[gid];
    bool orderedCompatibility = P.numBodies
        <= ORDERED_COLORING_BODY_LIMIT;
    for (uint k = s; k < e; k++) {
        if (orderedCompatibility) {
            neighborColors(joints, springs, manifolds, tets, soft,
                           membranes, bends, posLin, colors, adjList[k], gid,
                           colors[gid], maskLo, maskHi, allLo, allHi,
                           conflict);
        } else {
            neighborColor(posLin, colors, adjNeighbor[k], gid, colors[gid],
                          maskLo, maskHi, allLo, allHi, conflict);
        }
    }
    if (conflict) {
        atomic_fetch_add_explicit(&counters[CTR_COLOR_CONFLICTS], 1u,
                                  memory_order_relaxed);
    }
}

// Count bodies per color & remember slot.
kernel void color_count(
    device const float4* posLin     [[buffer(0)]],
    device const uint* colors       [[buffer(1)]],
    device atomic_uint* counters    [[buffer(2)]],
    device uint* bodySlot           [[buffer(3)]],
    constant SimParams& P           [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numBodies) return;
    if (posLin[gid].w <= 0.0f) return;
    uint c = colors[gid];
    bodySlot[gid] = atomic_fetch_add_explicit(&counters[CTR_COLOR_BASE + c], 1u, memory_order_relaxed);
}

// Single-TG scan over MAX_COLORS counts -> colorStart; fill indirect args.
kernel void color_scan(
    device atomic_uint* counters    [[buffer(0)]],
    device uint* colorStart         [[buffer(1)]],   // MAX_COLORS + 1
    device uint* dispatchArgs       [[buffer(2)]],   // MAX_COLORS * 3 uints
    constant uint& threadsPerBody   [[buffer(3)]],
    uint tid                        [[thread_position_in_threadgroup]])
{
    threadgroup uint counts[MAX_COLORS];
    if (tid < MAX_COLORS) {
        counts[tid] = atomic_load_explicit(&counters[CTR_COLOR_BASE + tid], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint run = 0;
        uint used = 0;
        for (uint c = 0; c < MAX_COLORS; c++) {
            colorStart[c] = run;
            run += counts[c];
            if (counts[c] > 0) used = c + 1;
        }
        colorStart[MAX_COLORS] = run;
        colorStart[MAX_COLORS + 1] = used;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < MAX_COLORS) {
        dispatchArgs[tid * 3 + 0] =
            (counts[tid] * threadsPerBody + 63) / 64;
        dispatchArgs[tid * 3 + 1] = 1;
        dispatchArgs[tid * 3 + 2] = 1;
    }
}

kernel void color_scatter(
    device const float4* posLin     [[buffer(0)]],
    device const uint* colors       [[buffer(1)]],
    device const uint* bodySlot     [[buffer(2)]],
    device const uint* colorStart   [[buffer(3)]],
    device uint* colorList          [[buffer(4)]],
    constant SimParams& P           [[buffer(5)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numBodies) return;
    if (posLin[gid].w <= 0.0f) return;
    colorList[colorStart[colors[gid]] + bodySlot[gid]] = gid;
}

// ----------------------------------------------------------------------------
// Warm start
// ----------------------------------------------------------------------------

kernel void warmstart_joints(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device JointGPU* joints         [[buffer(2)]],
    constant SimParams& P           [[buffer(3)]],
    device SpringGPU* springs       [[buffer(4)]],
    device const float4* velLin     [[buffer(5)]],
    device const float4* velAng     [[buffer(6)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numJoints) {
        uint si = gid - P.numJoints;
        if (si >= P.numSprings) return;
        device SpringGPU& sp = springs[si];
        if (sp.header.z == 0) return;          // soft springs have no dual
        uint a = sp.header.x, b = sp.header.y;
        float3 pA = xform(posLin[a].xyz, posAng[a], sp.rA.xyz);
        float3 pB = xform(posLin[b].xyz, posAng[b], sp.rB.xyz);
        // C0 alpha-shift only for pre-existing STRETCH: rods are tension-
        // only, slack is not an error to stabilize away. Halved: full
        // asymptotic forgiveness legitimizes standing stretch (closure 1%
        // per frame loses to flutter re-stretching); rods spawn exact, so
        // there is little pre-existing error to soften in the first place.
        sp.dual.z = 0.5f * max(length(pA - pB) - sp.rB.w, 0.0f);  // C0
        float warm = P.alpha * P.gamma;
        if (P.rodDecayPow > 0.0f) {
            // The carried dual acted along LAST frame's rod direction; on a
            // rotating rod, re-applying it along the new direction does
            // tangential work each frame (pendulum pump). Decay by the
            // per-frame rotation, reconstructed from the velocities.
            float3 d = pA - pB;
            float3 dPrev = d - (velLin[a].xyz - velLin[b].xyz) * P.dt;
            float dl = length(d), dpl = length(dPrev);
            if (dl > 1e-7f && dpl > 1e-7f) {
                float ct = clamp(dot(d / dl, dPrev / dpl), 0.0f, 1.0f);
                warm *= pow(ct, P.rodDecayPow);
            }
        }
        sp.dual.x *= warm;                                     // lambda
        sp.dual.y = min(clamp(sp.dual.y * P.gamma, PENALTY_MIN, PENALTY_MAX),
                        sp.rA.w);                              // penalty
        return;
    }
    device JointGPU& j = joints[gid];
    if (j.header.z != 0) return;    // broken

    uint a = j.header.x, b = j.header.y;
    float3 pA = a == WORLD_BODY ? j.rA.xyz : xform(posLin[a].xyz, posAng[a], j.rA.xyz);
    float3 pB = xform(posLin[b].xyz, posAng[b], j.rB.xyz);
    float4 qA = a == WORLD_BODY ? float4(0,0,0,1) : posAng[a];
    float torqueArm = j.C0Lin.w;

    j.C0Lin = float4(pA - pB, torqueArm);
    float3 c0a;
    if (j.hingeAxis.w != 0.0f) {
        // hinge: axis-alignment error, invariant to spin about the axis
        float3 aB = q_rotate(posAng[b], j.hingeAxis.xyz);
        float3 aA = q_rotate(q_mul(qA, j.restRel), j.hingeAxis.xyz);
        c0a = cross(aA, aB) * torqueArm;
        if (j.dynamics.x > 0.0f) {
            float4 r = q_mul(q_inv(q_mul(qA, j.restRel)), posAng[b]);
            if (r.w < 0.0f) r = -r;
            float twist = 2.0f * atan2(dot(r.xyz, j.hingeAxis.xyz), r.w);
            float3 omegaA = a == WORLD_BODY ? float3(0) : velAng[a].xyz;
            float twistRate = dot(velAng[b].xyz - omegaA, aB);
            j.dynamics.y = twist + P.dt * twistRate;
        }
    } else {
        c0a = q_sub(q_mul(qA, j.restRel), posAng[b]) * torqueArm;
    }
    j.C0Ang = float4(c0a, j.C0Ang.w);
    if (j.motor.y > 0.0f) {
        // Motor gains are fixed physical scene data, never adaptive
        // augmented-Lagrangian state.
        j.motor.z = 0.0f;
        j.dynamics.z = 0.0f;
        if (motor_uses_explicit_effort(j.header.w)) {
            // Evaluate one bounded effort from the state at the beginning of
            // the physics step and hold it through the nonlinear solve.
            float4 qAr = a == WORLD_BODY ? j.restRel
                                         : q_mul(posAng[a], j.restRel);
            float4 r = q_mul(q_inv(qAr), posAng[b]);
            if (r.w < 0.0f) r = -r;
            float twist = 2.0f * atan2(
                dot(r.xyz, j.hingeAxis.xyz), r.w);
            float3 axisW = q_rotate(posAng[b], j.hingeAxis.xyz);
            float3 omegaA = a == WORLD_BODY ? float3(0)
                                            : velAng[a].xyz;
            float twistRate = dot(velAng[b].xyz - omegaA, axisW);
            if (motor_uses_velocity_feedback(j.header.w)) {
                j.dynamics.z = clamp(
                    j.limits.z * (j.motor.x - twistRate),
                    -j.motor.y, j.motor.y);
            } else {
                j.dynamics.z = clamp(
                    j.motor.w * (j.motor.x - twist)
                        - j.limits.z * twistRate,
                    -j.motor.y, j.motor.y);
            }
        }
    }

    float warm = P.alpha * P.gamma;
    float stiffLin = j.rA.w;
    float stiffAng = j.rB.w;
    j.lambdaLin.xyz *= warm;
    j.lambdaAng.xyz *= warm;
    j.penaltyLin.xyz = min(clamp(j.penaltyLin.xyz * P.gamma, PENALTY_MIN, PENALTY_MAX), stiffLin);
    j.penaltyAng.xyz = min(clamp(j.penaltyAng.xyz * P.gamma, PENALTY_MIN, PENALTY_MAX), stiffAng);
}

kernel void warmstart_bodies(
    device float4* posLin           [[buffer(0)]],
    device float4* posAng           [[buffer(1)]],
    device float4* initLin          [[buffer(2)]],
    device float4* initAng          [[buffer(3)]],
    device float4* inertLin         [[buffer(4)]],
    device float4* inertAng         [[buffer(5)]],
    device const float4* velLin     [[buffer(6)]],
    device const float4* velAng     [[buffer(7)]],
    device const float4* prevVelLin [[buffer(8)]],
    constant SimParams& P           [[buffer(9)]],
    device float4* ogcPrev          [[buffer(10)]],
    device const uint* boundsBits   [[buffer(11)]],
    constant uint& truncationMode   [[buffer(12)]],
    device const float4* shapeW     [[buffer(13)]],
    device const float* gravityScale [[buffer(14)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numBodies) return;
    float4 pl = posLin[gid];
    float4 pa = posAng[gid];
    float mass = pl.w;
    float dt = P.dt;

    float3 vl = velLin[gid].xyz;
    float3 va = velAng[gid].xyz;

    float3 inertial = pl.xyz + vl * dt;
    float bodyGravity = P.gravity * gravityScale[gid];
    if (mass > 0.0f) inertial += float3(0, 0, bodyGravity) * (dt * dt);
    inertLin[gid] = float4(inertial, 0);
    inertAng[gid] = q_addw(pa, va * dt);

    // Adaptive warmstart (original VBD)
    float3 accel = (vl - prevVelLin[gid].xyz) / dt;
    float gsign = P.gravity < 0.0f ? -1.0f : (P.gravity > 0.0f ? 1.0f : 0.0f);
    float accelWeight = clamp(accel.z * gsign / fabs(P.gravity), 0.0f, 1.0f);
    if (!isfinite(accelWeight)) accelWeight = 0.0f;

    initLin[gid] = float4(pl.xyz, 0);
    initAng[gid] = pa;
    // The committed collision-query pose is required for every surface
    // vertex, including mass-zero pins referenced by a moving triangle.
    if (truncationMode != 0u && shapeW[gid].w < 0.0f) {
        ogcPrev[gid] = float4(pl.xyz, 0);
    }
    if (mass > 0.0f) {
        float3 guess = pl.xyz + vl * dt
                     + float3(0, 0, bodyGravity) * (accelWeight * dt * dt);
        // OGC anchor = the detection pose; the warmstart placement itself
        // is truncated within the bound (paper Eq 28) so the step STARTS
        // penetration-free — the in-loop refresh grants further budget
        if (truncationMode == SURFACE_TRUNCATION_ISOTROPIC
            && shapeW[gid].w < 0.0f) {
            float d2 = as_type<float>(boundsBits[gid]);
            if (d2 < 1e37f) {
                float bv = max(0.45f * sqrt(d2), -0.2f * shapeW[gid].w);
                float3 tot = guess - pl.xyz;
                float t2 = dot(tot, tot);
                if (t2 > bv * bv) guess = pl.xyz + tot * (bv * rsqrt(t2));
            }
        }
        posLin[gid] = float4(guess, mass);
        posAng[gid] = q_addw(pa, va * dt);
    }
}

// ----------------------------------------------------------------------------
// Primal update — one thread per body of the current color.
// ----------------------------------------------------------------------------

struct PrimalAccum {
    M3 lhsLin, lhsAng, lhsCross;
    float3 rhsLin, rhsAng;
};

inline M3 geomStiffBallSocket(int k, float3 v) {
    M3 m = m3_diag(float3(-v[k]));
    if (k == 0) { m.r0.x += v.x; m.r1.x += v.y; m.r2.x += v.z; }
    else if (k == 1) { m.r0.y += v.x; m.r1.y += v.y; m.r2.y += v.z; }
    else { m.r0.z += v.x; m.r1.z += v.y; m.r2.z += v.z; }
    return m;
}

inline void stampJoint(device const JointGPU& j, uint self,
                       device const float4* posLin, device const float4* posAng,
                       device const float4* initAng, float alpha, float dt,
                       thread PrimalAccum& acc)
{
    uint a = j.header.x, b = j.header.y;
    bool isA = self == a;
    float torqueArm = j.C0Lin.w;

    // Linear
    float3 penLin = j.penaltyLin.xyz;
    if (dot(penLin, penLin) > 0.0f) {
        float3 pA = a == WORLD_BODY ? j.rA.xyz : xform(posLin[a].xyz, posAng[a], j.rA.xyz);
        float3 pB = xform(posLin[b].xyz, posAng[b], j.rB.xyz);
        float3 C = pA - pB;
        if (j.header.w & 1) C -= j.C0Lin.xyz * alpha;

        float3 F = penLin * C + j.lambdaLin.xyz;
        float jsign = isA ? 1.0f : -1.0f;
        float3 rW = isA ? q_rotate(posAng[a], j.rA.xyz) : q_rotate(posAng[b], j.rB.xyz);
        // jLin = jsign*I ; jAng = skew(-jsign * rW)
        M3 jAng = m3_skew(-jsign * rW);
        M3 jAngT = m3_transpose(jAng);
        M3 K = m3_diag(penLin);
        M3 jAngTk = m3_mulm(jAngT, K);

        acc.lhsLin = m3_add(acc.lhsLin, K);                           // (sI)^T K (sI), s^2=1
        acc.lhsAng = m3_add(acc.lhsAng, m3_mulm(jAngTk, jAng));
        acc.lhsCross = m3_add(acc.lhsCross, m3_scale(jAngTk, jsign)); // jAngT K (sI)

        float3 r = jsign * rW;
        M3 H = m3_add(m3_add(
            m3_scale(geomStiffBallSocket(0, r), F.x),
            m3_scale(geomStiffBallSocket(1, r), F.y)),
            m3_scale(geomStiffBallSocket(2, r), F.z));
        acc.lhsAng = m3_add(acc.lhsAng, m3_diagonalize(H));

        acc.rhsLin += jsign * F;
        acc.rhsAng += m3_mul(jAngT, F);
    }

    // Angular
    float3 penAng = j.penaltyAng.xyz;
    if (dot(penAng, penAng) > 0.0f) {
        float4 qA = a == WORLD_BODY ? float4(0,0,0,1) : posAng[a];
        float3 C = q_sub(q_mul(qA, j.restRel), posAng[b]) * torqueArm;
        float s = (isA ? 1.0f : -1.0f) * torqueArm;
        if (j.hingeAxis.w != 0.0f) {
            // 1-DOF hinge: axis-alignment constraint C = cross(aA, aB).
            // Spin-invariant — unlike quaternion differences, the gradient
            // never degenerates as the wheel revolves. NOTE the Jacobian
            // signs are OPPOSITE to the weld convention:
            // dC/dthetaA = -P, dC/dthetaB = +P.
            float3 aB = q_rotate(posAng[b], j.hingeAxis.xyz);
            float3 aA = q_rotate(q_mul(qA, j.restRel), j.hingeAxis.xyz);
            C = cross(aA, aB) * torqueArm;
            if (j.header.w & 2) C -= j.C0Ang.xyz * alpha;
            float3 F = penAng * C + j.lambdaAng.xyz;
            // Exact Gauss-Newton hinge Jacobian. For C = aA x aB,
            //   dC/dthetaA = aA aB^T - dot(aA,aB) I
            //   dC/dthetaB = dot(aA,aB) I - aB aA^T.
            // At alignment these become -P and +P, where
            // P = I - aa^T has rank two. The previous isotropic `k I`
            // Hessian incorrectly added hard-constraint curvature around
            // the free twist axis, numerically opposing every motor torque.
            float alignment = dot(aA, aB);
            M3 J = isA
                ? m3_add(m3_outer(aA, aB),
                         m3_scale(m3_identity(), -alignment))
                : m3_add(m3_scale(m3_identity(), alignment),
                         m3_scale(m3_outer(aB, aA), -1.0f));
            J = m3_scale(J, torqueArm);
            M3 JT = m3_transpose(J);
            M3 K = m3_diag(penAng);
            acc.lhsAng = m3_add(
                acc.lhsAng, m3_mulm(m3_mulm(JT, K), J));
            acc.rhsAng += m3_mul(JT, F);

            // MOTOR: fixed-gain position PD or velocity feedback with an
            // authored effort bound. This is actuator physics, separate from
            // the adaptive multipliers used by constraints and contacts.
            if (j.motor.y > 0.0f || j.limits.x < j.limits.y) {
                float4 r = q_mul(q_inv(q_mul(qA, j.restRel)), posAng[b]);
                if (r.w < 0.0f) r = -r;
                float twist = 2.0f * atan2(dot(r.xyz, j.hingeAxis.xyz), r.w);

                // Reflected rotor inertia is a joint-space inertial energy:
                //   0.5 * armature / dt^2 * (q - q_hat)^2.
                // q_hat was predicted from the start-of-step relative
                // angular velocity. Stamping its signed gradient on both
                // bodies preserves the coupled inertia under the solver's
                // block-coordinate primal iterations.
                if (j.dynamics.x > 0.0f) {
                    float Ca = twist - j.dynamics.y;
                    Ca = Ca - 6.2831853f
                        * floor((Ca + 3.14159265f) / 6.2831853f);
                    float ka = j.dynamics.x / max(dt * dt, 1e-12f);
                    float sm = (self == a) ? -1.0f : 1.0f;
                    acc.lhsAng = m3_add(
                        acc.lhsAng, m3_scale(m3_outer(aB, aB), ka));
                    acc.rhsAng += aB * (ka * Ca * sm);
                }

                // joint limits: stiff one-sided penalty (rarely active)
                if (j.limits.x < j.limits.y) {
                    float over = max(twist - j.limits.y, 0.0f)
                               + min(twist - j.limits.x, 0.0f);
                    if (over != 0.0f) {
                        float kL = 4.0e4f;
                        float FL = clamp(kL * over, -3000.0f, 3000.0f);
                        float sm = (self == a) ? -1.0f : 1.0f;
                        acc.lhsAng = m3_add(acc.lhsAng,
                                            m3_scale(m3_outer(aB, aB), kL));
                        acc.rhsAng += aB * (FL * sm);
                    }
                }
                if (j.motor.y > 0.0f) {
                    float sm = (self == a) ? -1.0f : 1.0f;
                    float3 axW = aB;
                    if (motor_uses_explicit_effort(j.header.w)) {
                        // `dynamics.z` is positive child-axis torque. The VBD
                        // residual uses the opposite force sign.
                        acc.rhsAng += axW * (-j.dynamics.z * sm);
                    } else {
                        float Cm = twist - j.motor.x;
                        Cm = Cm - 6.2831853f
                            * floor((Cm + 3.14159265f) / 6.2831853f);

                        // Position delta from the beginning of the step.
                        // With kd in torque/(rad/s), its position-level
                        // derivative is kd / dt.
                        float4 qA0 = a == WORLD_BODY
                            ? float4(0,0,0,1) : initAng[a];
                        float4 r0 = q_mul(
                            q_inv(q_mul(qA0, j.restRel)), initAng[b]);
                        if (r0.w < 0.0f) r0 = -r0;
                        float twist0 = 2.0f * atan2(
                            dot(r0.xyz, j.hingeAxis.xyz), r0.w);
                        float dTwist = twist - twist0;
                        dTwist = dTwist - 6.2831853f
                            * floor((dTwist + 3.14159265f) / 6.2831853f);
                        float cD = j.limits.z / max(dt, 1e-6f);
                        float rawEffort = j.motor.w * Cm + cD * dTwist;
                        float effort = clamp(
                            rawEffort, -j.motor.y, j.motor.y);

                        // The derivative of a saturated clamp is zero. Do
                        // not retain an unsaturated spring Hessian at the
                        // effort bound; doing so weakens the actuator and
                        // makes its torque cap iteration-count dependent.
                        if (fabs(rawEffort) < j.motor.y) {
                            acc.lhsAng = m3_add(
                                acc.lhsAng,
                                m3_scale(m3_outer(axW, axW),
                                         j.motor.w + cD));
                        }
                        acc.rhsAng += axW * (effort * sm);
                    }
                }
            }
        } else {
            if (j.header.w & 2) C -= j.C0Ang.xyz * alpha;
            float3 F = penAng * C + j.lambdaAng.xyz;
            acc.lhsAng = m3_add(acc.lhsAng, m3_diag(penAng * (s * s)));
            acc.rhsAng += F * s;
        }
    }
}

inline void stampSpring(device const SpringGPU& sp, uint self,
                        device const float4* posLin, device const float4* posAng,
                        thread PrimalAccum& acc, float alpha)
{
    uint a = sp.header.x, b = sp.header.y;
    float stiffness = sp.rA.w;
    float rest = sp.rB.w;

    float3 pA = xform(posLin[a].xyz, posAng[a], sp.rA.xyz);
    float3 pB = xform(posLin[b].xyz, posAng[b], sp.rB.xyz);
    float3 d = pA - pB;
    float dLen = length(d);
    if (dLen <= 1.0e-6f) return;

    float3 n = d / dLen;
    float f;
    if (sp.header.z != 0) {
        // hard rod (AL): TENSION-ONLY inextensible element. Cloth and rope
        // are inextensible, not incompressible — a compressed rod buckles
        // freely. Enforcing compression makes buckled pairs into perpetual
        // oscillators: the push direction degenerates as dLen -> 0 and the
        // remembered dual keeps firing along whatever direction emerges.
        float C = dLen - rest - sp.dual.z * alpha;
        stiffness = sp.dual.y;                 // ramped penalty
        f = stiffness * C + sp.dual.x;         // + lambda
        if (f <= 0.0f) return;                 // slack: no force, no Hessian
    } else {
        f = stiffness * (dLen - rest);
    }

    bool isA = self == a;
    float3 rW = isA ? q_rotate(posAng[a], sp.rA.xyz) : q_rotate(posAng[b], sp.rB.xyz);
    float3 jLin = isA ? n : -n;
    float3 jAng = isA ? cross(rW, n) : -cross(rW, n);

    acc.lhsLin = m3_add(acc.lhsLin, m3_scale(m3_outer(jLin, jLin), stiffness));
    acc.lhsAng = m3_add(acc.lhsAng, m3_scale(m3_outer(jAng, jAng), stiffness));
    acc.lhsCross = m3_add(acc.lhsCross, m3_scale(m3_outer(jAng, jLin), stiffness));
    acc.rhsLin += jLin * f;
    acc.rhsAng += jAng * f;

    // Geometric (tension) stiffness: a TAUT element resists transverse
    // motion with curvature f/len (I - nn^T). Dropping it (pure rank-1
    // Gauss-Newton) leaves swinging taut networks with force-but-no-
    // Hessian transversely — Newton steps overshoot wherever tension/len
    // exceeds the m/dt^2 mass term, and the sheet pumps forever. Clamped
    // to tension only (compression curvature is negative: skipping it is
    // the safe Gauss-Newton choice).
    float ft = max(f, 0.0f) / dLen;
    if (ft > 0.0f) {
        acc.lhsLin = m3_add(acc.lhsLin,
                            m3_add(m3_diag(float3(ft)),
                                   m3_scale(m3_outer(n, n), -ft)));
    }
}

// Stable Neo-Hookean tetrahedron (Smith et al. 2018):
//   Psi = mu/2 (Ic - 3) + lambda/2 (J - alphaT)^2,  alphaT = 1 + mu/lambda
// PK1: P = mu F + lambda (J - alphaT) dJ/dF.
// Per-vertex SPD Hessian approximation (VBD-style):
//   H_i = vol (mu |w_i|^2 I + lambda g_i g_i^T),  g_i = dJdF w_i,
// dropping the indefinite second derivative of J.
inline void stampTet(device const TetGPU& t, uint self,
                     device const float4* posLin,
                     thread PrimalAccum& acc)
{
    float3 x0 = posLin[t.ids.x].xyz;
    float3 x1 = posLin[t.ids.y].xyz;
    float3 x2 = posLin[t.ids.z].xyz;
    float3 x3 = posLin[t.ids.w].xyz;
    float muV = t.r1.w;
    float lamV = t.r2.w;
    float alphaT = 1.0f + muV / max(lamV, 1e-9f);

    // F = Ds * DmInv  (columns d0..d2; DmInv rows in r0..r2)
    float3 d0 = x1 - x0, d1 = x2 - x0, d2 = x3 - x0;
    float3 m0 = t.r0.xyz, m1 = t.r1.xyz, m2 = t.r2.xyz;   // DmInv rows
    // F column j = d0*DmInv[0][j] + d1*DmInv[1][j] + d2*DmInv[2][j]
    float3 f0 = d0 * m0.x + d1 * m1.x + d2 * m2.x;
    float3 f1 = d0 * m0.y + d1 * m1.y + d2 * m2.y;
    float3 f2 = d0 * m0.z + d1 * m1.z + d2 * m2.z;

    float J = dot(f0, cross(f1, f2));
    float3 j0 = cross(f1, f2);          // dJ/dF columns
    float3 j1 = cross(f2, f0);
    float3 j2 = cross(f0, f1);

    float s2 = lamV * (J - alphaT);
    // PK1 columns (scaled by volume via muV/lamV which premultiply vol)
    float3 p0 = muV * f0 + s2 * j0;
    float3 p1 = muV * f1 + s2 * j1;
    float3 p2 = muV * f2 + s2 * j2;

    // self's weight vector w = DmInv^T column for vertices 1..3, or the
    // negated sum for vertex 0
    float3 w;
    if (self == t.ids.y)      w = float3(m0.x, m0.y, m0.z);
    else if (self == t.ids.z) w = float3(m1.x, m1.y, m1.z);
    else if (self == t.ids.w) w = float3(m2.x, m2.y, m2.z);
    else                      w = -float3(m0.x + m1.x + m2.x,
                                          m0.y + m1.y + m2.y,
                                          m0.z + m1.z + m2.z);

    // gradient on self = P * w (P columns p0..p2)
    float3 grad = p0 * w.x + p1 * w.y + p2 * w.z;
    float3 g = j0 * w.x + j1 * w.y + j2 * w.z;       // dJ/dx_self

    acc.rhsLin += grad;
    acc.lhsLin = m3_add(acc.lhsLin, m3_diag(float3(muV * dot(w, w))));
    acc.lhsLin = m3_add(acc.lhsLin, m3_scale(m3_outer(g, g), lamV));
}

// Exact signed-J line limit for a colored tet update. Graph coloring gives
// every vertex of one tet a distinct color, so only `self` moves during this
// call and J is affine in dx. This is much cheaper than rescanning every tet
// after every color and closes the inversion gap left by surface-only DAT.
inline float tetInversionSafeFraction(
    device const TetGPU& t, uint self,
    device const float4* posLin, float3 dx, float gamma,
    device atomic_uint* counters)
{
    float3 x0 = posLin[t.ids.x].xyz;
    float3 d0 = posLin[t.ids.y].xyz - x0;
    float3 d1 = posLin[t.ids.z].xyz - x0;
    float3 d2 = posLin[t.ids.w].xyz - x0;
    float restSixVolume = t.r0.w;
    float restSign = restSixVolume < 0.0f ? -1.0f : 1.0f;
    float currentSixVolume = dot(d0, cross(d1, d2));
    float3 g1 = cross(d1, d2);
    float3 g2 = cross(d2, d0);
    float3 g3 = cross(d0, d1);
    float3 g;
    if (self == t.ids.y)      g = g1;
    else if (self == t.ids.z) g = g2;
    else if (self == t.ids.w) g = g3;
    else                      g = -(g1 + g2 + g3);
    float signedVolume = restSign * currentSixVolume;
    float q = restSign * dot(g, dx);
    if (!finite_bits(restSixVolume) || abs(restSixVolume) < 1.0e-20f
        || !finite_bits(signedVolume) || !finite3(g) || !finite_bits(q)) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[CTR_DAT_NONFINITE], 1u,
                                  memory_order_relaxed);
        return 0.0f;
    }
    // Match the predictor's positive-volume floor. A shared threshold keeps
    // accepted prediction and colored local solves on the same side of the
    // inversion boundary.
    if (signedVolume <= 0.0f) {
        atomic_fetch_add_explicit(&counters[CTR_DAT_INVALID_ANCHOR], 1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&counters[CTR_DAT_TET_DEGENERATE], 1u,
                                  memory_order_relaxed);
        return 0.0f;
    }
    float minVolume = abs(restSixVolume) * 1.0e-4f;
    if (signedVolume <= minVolume) {
        // Positive, highly compressed tets are recoverable. Permit an
        // expanding color update, but never let this update compress them
        // farther. Predictor telemetry records that the conditioning floor
        // was active without turning it into a terminal collision failure.
        return q > 0.0f ? 1.0f : 0.0f;
    }
    if (q >= 0.0f || signedVolume + q > minVolume) return 1.0f;
    float hit = (signedVolume - minVolume) / (-q);
    return clamp(min(gamma * hit, hit - 1.0e-3f), 0.0f, 1.0f);
}

// StVK triangle membrane with a Gauss-Newton per-vertex Hessian.
//   Psi = mu ||E||_F^2 + lambda/2 tr(E)^2,  E = (F^T F - I)/2  (2x2)
// Forces are exact (P = F S, S = 2 mu E + lambda tr(E) I); the Hessian
// keeps only the strain-gradient outer products (drops the indefinite
// geometric term), which is SPD by construction and captures the in-plane
// directional stiffness that diagonal lumping loses:
//   dE11/dx_i = w.x f1, dE22/dx_i = w.y f2, dE12/dx_i = (w.x f2 + w.y f1)/2
//   H_i = area [2mu(g11 g11^T + g22 g22^T) + 4mu g12 g12^T
//               + lambda (g11+g22)(g11+g22)^T]
inline void stampMembrane(device const MembraneGPU& mb, uint self,
                          device const float4* posLin,
                          thread PrimalAccum& acc)
{
    float3 x0 = posLin[mb.ids.x].xyz;
    float3 x1 = posLin[mb.ids.y].xyz;
    float3 x2 = posLin[mb.ids.z].xyz;
    float muA = mb.mat.x;           // mu * area
    float lamA = mb.mat.y;          // lambda * area

    float3 d0 = x1 - x0, d1 = x2 - x0;
    // DmInv = [a c; b d] column-major in mb.dm = (a, b, c, d)
    float a = mb.dm.x, b = mb.dm.y, c = mb.dm.z, d = mb.dm.w;
    float3 f1 = d0 * a + d1 * b;
    float3 f2 = d0 * c + d1 * d;

    float E11 = 0.5f * (dot(f1, f1) - 1.0f);
    float E22 = 0.5f * (dot(f2, f2) - 1.0f);
    float E12 = 0.5f * dot(f1, f2);
    float trE = E11 + E22;
    float S11 = 2.0f * muA * E11 + lamA * trE;   // premultiplied by area
    float S22 = 2.0f * muA * E22 + lamA * trE;
    float S12 = 2.0f * muA * E12;

    // P = F S columns (area folded into S)
    float3 p1 = f1 * S11 + f2 * S12;
    float3 p2 = f1 * S12 + f2 * S22;

    float2 w;
    if (self == mb.ids.y)      w = float2(a, c);
    else if (self == mb.ids.z) w = float2(b, d);
    else                       w = float2(-a - b, -c - d);

    acc.rhsLin += p1 * w.x + p2 * w.y;

    float3 g11 = f1 * w.x;
    float3 g22 = f2 * w.y;
    float3 g12 = (f2 * w.x + f1 * w.y) * 0.5f;
    float3 gtr = g11 + g22;
    acc.lhsLin = m3_add(acc.lhsLin, m3_scale(m3_outer(g11, g11), 2.0f * muA));
    acc.lhsLin = m3_add(acc.lhsLin, m3_scale(m3_outer(g22, g22), 2.0f * muA));
    acc.lhsLin = m3_add(acc.lhsLin, m3_scale(m3_outer(g12, g12), 4.0f * muA));
    acc.lhsLin = m3_add(acc.lhsLin, m3_scale(m3_outer(gtr, gtr), lamA));
}

// Quadratic (Bergou) bending: E = s/2 |sum_i K_i x_i|^2 over the 4-vertex
// hinge stencil; rest cotangents make K annihilate the flat state. Constant
// rank-1 SPD Hessian per vertex: s K_i^2 I — zero tuning, isotropic.
inline void stampBend(device const BendGPU& bd, uint self,
                      device const float4* posLin,
                      thread PrimalAccum& acc)
{
    float s = bd.mat.x;
    float3 m = posLin[bd.ids.x].xyz * bd.K.x
             + posLin[bd.ids.y].xyz * bd.K.y
             + posLin[bd.ids.z].xyz * bd.K.z
             + posLin[bd.ids.w].xyz * bd.K.w;
    float k;
    if (self == bd.ids.x)      k = bd.K.x;
    else if (self == bd.ids.y) k = bd.K.y;
    else if (self == bd.ids.z) k = bd.K.z;
    else                       k = bd.K.w;
    acc.rhsLin += m * (s * k);
    acc.lhsLin = m3_add(acc.lhsLin, m3_diag(float3(s * k * k)));
}

// Shared contact force computation (Taylor series constraint, Sec 4 + Eq 13-15)
inline float3 contactForceC(device const ManifoldGPU& m, uint ci,
                            float3 dqALin, float3 dqAAng, float3 dqBLin, float3 dqBAng,
                            float3 rAW, float3 rBW, float alpha,
                            thread float3& Cout, thread float& frictionScale, thread float& bounds)
{
    float3 nrm = m.basisN.xyz;
    float3 t1 = m.basisT1.xyz;
    float3 t2 = cross(nrm, t1);
    float dynamicFriction = m.basisN.w;
    float staticFriction = m.basisT1.w;

    device const ContactGPU& c = m.contacts[ci];
    float3 C = c.C0.xyz * (1.0f - alpha);
    C += float3(dot(nrm, dqALin), dot(t1, dqALin), dot(t2, dqALin));
    C -= float3(dot(nrm, dqBLin), dot(t1, dqBLin), dot(t2, dqBLin));
    C += float3(dot(cross(rAW, nrm), dqAAng), dot(cross(rAW, t1), dqAAng), dot(cross(rAW, t2), dqAAng));
    C -= float3(dot(cross(rBW, nrm), dqBAng), dot(cross(rBW, t1), dqBAng), dot(cross(rBW, t2), dqBAng));

    float3 F = c.penalty.xyz * C + c.lambda.xyz;
    F.x = min(F.x, 0.0f);
    bounds = fabs(F.x) * staticFriction;
    frictionScale = length(F.yz);
    if (frictionScale > bounds && frictionScale > 0.0f) {
        float dynamicBounds = fabs(F.x) * dynamicFriction;
        F.yz *= dynamicBounds / frictionScale;
    }
    Cout = C;
    return F;
}

inline float3 solverContactC0(device const SolverContactGPU& c)
{
    return float3(c.rB_C0x.w, c.C0yz_lambdaXY.xy);
}

inline float3 solverContactLambda(device const SolverContactGPU& c)
{
    return float3(c.C0yz_lambdaXY.zw, c.lambdaZ_penalty.x);
}

inline float3 solverContactForceC(
    device const SolverManifoldGPU& m,
    device const SolverContactGPU& c,
    float3 dqALin, float3 dqAAng, float3 dqBLin, float3 dqBAng,
    float3 rAW, float3 rBW, float alpha,
    thread float3& Cout, thread float& frictionScale, thread float& bounds)
{
    float3 nrm = m.basisN.xyz;
    float3 t1 = m.basisT1.xyz;
    float3 t2 = cross(nrm, t1);

    float3 C = solverContactC0(c) * (1.0f - alpha);
    C += float3(dot(nrm, dqALin), dot(t1, dqALin), dot(t2, dqALin));
    C -= float3(dot(nrm, dqBLin), dot(t1, dqBLin), dot(t2, dqBLin));
    C += float3(dot(cross(rAW, nrm), dqAAng), dot(cross(rAW, t1), dqAAng), dot(cross(rAW, t2), dqAAng));
    C -= float3(dot(cross(rBW, nrm), dqBAng), dot(cross(rBW, t1), dqBAng), dot(cross(rBW, t2), dqBAng));

    float3 F = c.lambdaZ_penalty.yzw * C + solverContactLambda(c);
    F.x = min(F.x, 0.0f);
    bounds = fabs(F.x) * m.basisT1.w;
    frictionScale = length(F.yz);
    if (frictionScale > bounds && frictionScale > 0.0f) {
        float dynamicBounds = fabs(F.x) * m.basisN.w;
        F.yz *= dynamicBounds / frictionScale;
    }
    Cout = C;
    return F;
}

inline void stampSolverManifold(
    device const SolverManifoldGPU* solveManifolds,
    device const SolverContactGPU* solveContacts,
    uint manifoldIndex, uint maxManifolds, uint self,
    device const float4* posLin, device const float4* posAng,
    device const float4* initLin, device const float4* initAng,
    float alpha, thread PrimalAccum& acc)
{
    device const SolverManifoldGPU& m = solveManifolds[manifoldIndex];
    uint a = m.header.x, b = m.header.y;
    uint n = m.header.z;
    bool isA = self == a;
    bool sphA = (m.header.w & 2u) != 0;
    bool sphB = (m.header.w & 4u) != 0;

    float3 dqALin = posLin[a].xyz - initLin[a].xyz;
    float3 dqAAng = q_sub(posAng[a], initAng[a]);
    float3 dqBLin = posLin[b].xyz - initLin[b].xyz;
    float3 dqBAng = q_sub(posAng[b], initAng[b]);

    float3 nrm = m.basisN.xyz;
    float3 t1 = m.basisT1.xyz;
    float3 t2 = cross(nrm, t1);

    for (uint i = 0; i < n; ++i) {
        device const SolverContactGPU& c =
            solveContacts[i * maxManifolds + manifoldIndex];
        float3 rAW = sphA ? c.rAStick.xyz
                          : q_rotate(posAng[a], c.rAStick.xyz);
        float3 rBW = sphB ? c.rB_C0x.xyz
                          : q_rotate(posAng[b], c.rB_C0x.xyz);

        float3 C; float fs, bnd;
        float3 F = solverContactForceC(
            m, c, dqALin, dqAAng, dqBLin, dqBAng,
            rAW, rBW, alpha, C, fs, bnd);

        float s = isA ? 1.0f : -1.0f;
        M3 jLin = M3{nrm * s, t1 * s, t2 * s};
        float3 rW = isA ? rAW : rBW;
        M3 jAng = M3{cross(rW, jLin.r0), cross(rW, jLin.r1), cross(rW, jLin.r2)};
        M3 K = m3_diag(c.lambdaZ_penalty.yzw);
        M3 jLinT = m3_transpose(jLin);
        M3 jAngT = m3_transpose(jAng);
        M3 jAngTk = m3_mulm(jAngT, K);

        acc.lhsLin = m3_add(acc.lhsLin, m3_mulm(m3_mulm(jLinT, K), jLin));
        acc.lhsAng = m3_add(acc.lhsAng, m3_mulm(jAngTk, jAng));
        acc.lhsCross = m3_add(acc.lhsCross, m3_mulm(jAngTk, jLin));
        acc.rhsLin += m3_mul(jLinT, F);
        acc.rhsAng += m3_mul(jAngT, F);
    }
}

kernel void manifold_solver_pack(
    device const ManifoldGPU* manifolds [[buffer(0)]],
    device SolverManifoldGPU* solveManifolds [[buffer(1)]],
    device SolverContactGPU* solveContacts [[buffer(2)]],
    device const atomic_uint* counters [[buffer(3)]],
    constant SimParams& P [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(
        &counters[CTR_PAIRS], memory_order_relaxed);
    if (gid >= numPairs) return;
    device const ManifoldGPU& src = manifolds[gid];
    device SolverManifoldGPU& dst = solveManifolds[gid];
    dst.header = src.header;
    dst.basisN = src.basisN;
    dst.basisT1 = src.basisT1;
    for (uint i = 0; i < src.header.z; ++i) {
        device const ContactGPU& sc = src.contacts[i];
        device SolverContactGPU& dc = solveContacts[i * P.maxManifolds + gid];
        dc.rAStick = float4(sc.rA.xyz, sc.rB.w);
        dc.rB_C0x = float4(sc.rB.xyz, sc.C0.x);
        dc.C0yz_lambdaXY = float4(sc.C0.yz, sc.lambda.xy);
        dc.lambdaZ_penalty = float4(sc.lambda.z, sc.penalty.xyz);
    }
}

inline void stampManifold(device const ManifoldGPU& m, uint self,
                          device const float4* posLin, device const float4* posAng,
                          device const float4* initLin, device const float4* initAng,
                          float alpha, thread PrimalAccum& acc)
{
    uint a = m.header.x, b = m.header.y;
    uint n = m.header.z;
    bool isA = self == a;
    bool sphA = (m.header.w & 2u) != 0;
    bool sphB = (m.header.w & 4u) != 0;

    float3 dqALin = posLin[a].xyz - initLin[a].xyz;
    float3 dqAAng = q_sub(posAng[a], initAng[a]);
    float3 dqBLin = posLin[b].xyz - initLin[b].xyz;
    float3 dqBAng = q_sub(posAng[b], initAng[b]);

    float3 nrm = m.basisN.xyz;
    float3 t1 = m.basisT1.xyz;
    float3 t2 = cross(nrm, t1);

    for (uint i = 0; i < n; i++) {
        float3 rAW = sphA ? m.contacts[i].rA.xyz : q_rotate(posAng[a], m.contacts[i].rA.xyz);
        float3 rBW = sphB ? m.contacts[i].rB.xyz : q_rotate(posAng[b], m.contacts[i].rB.xyz);

        float3 C; float fs, bnd;
        float3 F = contactForceC(m, i, dqALin, dqAAng, dqBLin, dqBAng, rAW, rBW, alpha, C, fs, bnd);

        float s = isA ? 1.0f : -1.0f;
        M3 jLin = M3{nrm * s, t1 * s, t2 * s};
        float3 rW = isA ? rAW : rBW;
        M3 jAng = M3{cross(rW, jLin.r0), cross(rW, jLin.r1), cross(rW, jLin.r2)};

        M3 K = m3_diag(m.contacts[i].penalty.xyz);
        M3 jLinT = m3_transpose(jLin);
        M3 jAngT = m3_transpose(jAng);
        M3 jAngTk = m3_mulm(jAngT, K);

        acc.lhsLin = m3_add(acc.lhsLin, m3_mulm(m3_mulm(jLinT, K), jLin));
        acc.lhsAng = m3_add(acc.lhsAng, m3_mulm(jAngTk, jAng));
        acc.lhsCross = m3_add(acc.lhsCross, m3_mulm(jAngTk, jLin));

        acc.rhsLin += m3_mul(jLinT, F);
        acc.rhsAng += m3_mul(jAngT, F);
    }
}

// ----------------------------------------------------------------------------
// Element (soft) contacts: the unified 4-slot stencil. The constraint is the
// rigid-contact Taylor form generalized to weighted point sets:
//   C = C0 (1-alpha) + sum_i w_i [n|t1|t2].dq_i  (+ slot-0 angular terms when
// slot 0 is a rigid body). Shared by primal stamp and dual update.
// ----------------------------------------------------------------------------
inline float3 softContactC(device const SoftContactGPU& sc,
                           device const float4* posLin, device const float4* posAng,
                           device const float4* initLin, device const float4* initAng,
                           float alpha,
                           thread float3& n, thread float3& t1, thread float3& t2,
                           thread float3& rAW, thread bool& rigidA, thread bool& roundA)
{
    uint flags = as_type<uint>(sc.anchorA.w);
    rigidA = (flags & SCF_RIGID_A) != 0;
    roundA = (flags & 32u) != 0;
    n = sc.normal.xyz;
    orthonormal(n, t1, t2);
    rAW = float3(0);

    float3 C = sc.C0.xyz * (1.0f - alpha);
    for (uint k = 0; k < 4; k++) {
        uint b = sc.ids[k];
        if (b == WORLD_BODY) continue;
        float w = sc.weights[k];
        float3 dq = posLin[b].xyz - initLin[b].xyz;
        C += w * float3(dot(n, dq), dot(t1, dq), dot(t2, dq));
        if (k == 0 && rigidA) {
            rAW = roundA ? sc.anchorA.xyz : q_rotate(posAng[b], sc.anchorA.xyz);
            float3 dqa = q_sub(posAng[b], initAng[b]);
            C += w * float3(dot(cross(rAW, n), dqa),
                            dot(cross(rAW, t1), dqa),
                            dot(cross(rAW, t2), dqa));
        }
    }
    return C;
}

inline float3 softContactForce(device const SoftContactGPU& sc, float3 C,
                               thread float& frictionScale, thread float& bounds,
                               thread float& kNormal)
{
    float3 F = sc.penalty.xyz * C + sc.lambda.xyz;
    kNormal = sc.penalty.x;
    // OGC 2-stage activation (paper Eq 18-20): the quadratic stage is the
    // penalty+dual above; below tau = restT/2 the normal force switches to
    // the log-barrier stage k'/d (k' = tau*kc*(r-tau), C1-stitched), which
    // stiffens as the separation closes. CLOTH-CLOTH ONLY (V-T/E-E): those
    // are the pairs the conservative bounds keep penetration-free, so d
    // stays positive. Rigid-vs-cloth penetrates by design (the rigid side
    // has no displacement bound; sign memory ejects it) — a barrier there
    // turned deep impact frames into catapults (drape: 400% stretch, cloth
    // launched). Capped at 4x the full-compression quadratic force: with
    // truncation as the no-cross guarantee, the barrier is a convergence
    // aid, not an infinity.
    // DISABLED pending a mid-step re-detection architecture: the barrier
    // assumes a penetration-free state (paper Sec 3.7 guarantees it via
    // per-iteration conservative-bound refreshes); applied to our deep
    // rigid-pressed pinches it overdrives the pinch instead (boxoncloth
    // gate fails). The data plumbing (restT in anchorA.x) stays live.
    float rT = sc.anchorA.x;            // restT (particle contacts only)
    uint kindB = (as_type<uint>(sc.anchorA.w) >> SCF_KIND_SHIFT) & 3u;
    if (OGC_BARRIER && rT > 0.0f && kindB != SC_RT && C.x < -0.5f * rT) {
        // genuinely divergent (floored at 0.02*rT, not force-capped):
        // under a kinematic squeeze — the twist rope — a capped barrier
        // lets layers grind toward raw distance 0, the conservative
        // bounds collapse with them, and clamped vertices freeze against
        // their driver. The earlier overdrive incident was RT contacts,
        // which are excluded here.
        float dsep = max(C.x + rT, 0.02f * rT);
        float kp = 0.25f * rT * rT * sc.penalty.x;
        F.x = -kp / dsep + sc.lambda.x;
        kNormal = kp / (dsep * dsep);
    }
    F.x = min(F.x, 0.0f);
    bounds = fabs(F.x) * sc.normal.w;
    frictionScale = length(F.yz);
    if (frictionScale > bounds && frictionScale > 0.0f) {
        F.yz *= bounds / frictionScale;
    }
    return F;
}

inline void stampSoft(device const SoftContactGPU& sc, uint self,
                      device const float4* posLin, device const float4* posAng,
                      device const float4* initLin, device const float4* initAng,
                      float alpha, thread PrimalAccum& acc)
{
    float3 n, t1, t2, rAW;
    bool rigidA, roundA;
    float3 C = softContactC(sc, posLin, posAng, initLin, initAng, alpha,
                            n, t1, t2, rAW, rigidA, roundA);
    float fs, bnd, kN;
    float3 F = softContactForce(sc, C, fs, bnd, kN);

    int slot = -1;
    for (uint k = 0; k < 4; k++) {
        if (sc.ids[k] == self) { slot = int(k); break; }
    }
    if (slot < 0) return;
    float w = sc.weights[slot];

    if (slot == 0 && rigidA) {
        // 6-DOF rigid: full linear/angular/cross blocks like stampManifold
        M3 jLin = M3{n * w, t1 * w, t2 * w};
        M3 jAng = M3{cross(rAW, jLin.r0), cross(rAW, jLin.r1), cross(rAW, jLin.r2)};
        M3 K = m3_diag(float3(kN, sc.penalty.y, sc.penalty.z));
        M3 jLinT = m3_transpose(jLin);
        M3 jAngT = m3_transpose(jAng);
        M3 jAngTk = m3_mulm(jAngT, K);
        acc.lhsLin = m3_add(acc.lhsLin, m3_mulm(m3_mulm(jLinT, K), jLin));
        acc.lhsAng = m3_add(acc.lhsAng, m3_mulm(jAngTk, jAng));
        acc.lhsCross = m3_add(acc.lhsCross, m3_mulm(jAngTk, jLin));
        acc.rhsLin += m3_mul(jLinT, F);
        acc.rhsAng += m3_mul(jAngT, F);
    } else {
        // 3-DOF particle: linear block only
        float w2 = w * w;
        acc.lhsLin = m3_add(acc.lhsLin, m3_scale(m3_outer(n, n), kN * w2));
        acc.lhsLin = m3_add(acc.lhsLin, m3_scale(m3_outer(t1, t1), sc.penalty.y * w2));
        acc.lhsLin = m3_add(acc.lhsLin, m3_scale(m3_outer(t2, t2), sc.penalty.z * w2));
        acc.rhsLin += (n * F.x + t1 * F.y + t2 * F.z) * w;
    }
}

// OGC conservative-bound truncation (paper Eq 27): clamp a particle's
// displacement since its bound anchor within gamma_p * (min distance to
// any non-2-ring primitive at anchor time). ABSOLUTE, so the pairwise
// no-cross guarantee composes; exceeders are counted and trigger the
// in-loop anchor/bound refresh (Alg 3) instead of freezing motion.
inline float3 ogcTruncate(float3 dxLin, float3 cur, float3 x0, uint bits,
                          float partR, device atomic_uint* counters)
{
    float d2 = as_type<float>(bits);
    if (d2 >= 1e37f) return dxLin;
    // FLOOR of 0.2*r per anchor window: a vertex wound into a tight rope
    // core (raw layer distance ~ 0) must never FREEZE against a driven
    // clamp — its hard joint would lose by decimeters and the clamp row
    // tears (twist demo: 5x rod stretch). At 0.2*r per refresh, crossing
    // a 2r skin needs ~10 sustained windows against the divergent
    // barrier — practically impossible, and the formal bound holds
    // everywhere clearance exists.
    float bv = max(0.45f * sqrt(d2), 0.2f * partR);
    float3 tot = (cur + dxLin) - x0;
    float t2 = dot(tot, tot);
    if (t2 > bv * bv) {
        tot *= bv * rsqrt(t2);
        dxLin = (x0 + tot) - cur;
        atomic_fetch_add_explicit(&counters[CTR_OGC], 1u, memory_order_relaxed);
    }
    return dxLin;
}

static inline void primal_one(
    device float4* posLin,
    device float4* posAng,
    device const float4* initLin,
    device const float4* initAng,
    device const float4* inertLin,
    device const float4* inertAng,
    device const float4* props,
    device const JointGPU* joints,
    device const SpringGPU* springs,
    device const ManifoldGPU* manifolds,
    device const uint* adjStart,
    device const uint* adjCount,
    device const uint* adjList,
    constant SimParams& P,
    device const float4* shape,
    device const TetGPU* tets,
    device const SoftContactGPU* soft,
    device const MembraneGPU* membranes,
    device const BendGPU* bends,
    device const uint* boundsBits,
    device const float4* ogcPrev,
    device atomic_uint* ogcCounters,
    uint body)
{

    float4 pl = posLin[body];
    float mass = pl.w;
    float dt2 = P.dt * P.dt;
    float3 moment = props[body].xyz;

    PrimalAccum acc;
    acc.lhsLin = m3_diag(float3(mass / dt2));
    M3 inertia = m3_scale(world_inertia(posAng[body], moment), 1.0f / dt2);
    acc.lhsAng = inertia;
    acc.lhsCross = m3_zero();
    acc.rhsLin = (pl.xyz - inertLin[body].xyz) * (mass / dt2);
    acc.rhsAng = m3_mul(inertia, q_sub(posAng[body], inertAng[body]));

    uint as = adjStart[body], ae = as + adjCount[body];
    for (uint k = as; k < ae; k++) {
        uint entry = adjList[k];
        uint kind = entry >> ADJ_KIND_SHIFT;
        uint idx = entry & ADJ_INDEX_MASK;
        if (kind == FK_JOINT) {
            stampJoint(joints[idx], body, posLin, posAng, initAng, P.alpha, P.dt, acc);
        } else if (kind == FK_SPRING) {
            stampSpring(springs[idx], body, posLin, posAng, acc, P.alpha);
        } else if (kind == FK_TET) {
            stampTet(tets[idx], body, posLin, acc);
        } else if (kind == FK_SOFT) {
            stampSoft(soft[idx], body, posLin, posAng, initLin, initAng, P.alpha, acc);
        } else if (kind == FK_MEMBRANE) {
            stampMembrane(membranes[idx], body, posLin, acc);
        } else if (kind == FK_BEND) {
            stampBend(bends[idx], body, posLin, acc);
        } else {
            stampManifold(manifolds[idx], body, posLin, posAng, initLin, initAng, P.alpha, acc);
        }
    }

    // 3-DOF particles (paper: M = mI, 3x3 blocks): no angular DOFs.
    // Encoded as shape.w < 0; zeroing the angular system reduces the
    // 6x6 LDL to the linear 3x3 block exactly.
    if (shape[body].w < 0.0f) {
        acc.lhsAng = m3_identity();
        acc.lhsCross = m3_zero();
        acc.rhsAng = float3(0);
    }

    float3 dxLin = float3(0), dxAng = float3(0);
    solve6x6(acc.lhsLin, acc.lhsAng, acc.lhsCross, -acc.rhsLin, -acc.rhsAng, dxLin, dxAng);

    // Skip the update if the solve produced non-finite values (paper: VBD
    // skips updates with non-PD Hessians; with the pivot clamp this should
    // be unreachable, but one poisoned position would NaN the whole island).
    if (!finite3(dxLin) || !finite3(dxAng)) return;

    // Trust region: VBD with line search guarantees descent (paper Sec 2.2);
    // we cap the step instead, which only engages in violent transients
    // (e.g. a long free-falling chain snapping taut) where a single Newton
    // step on the rotational nonlinearity would overshoot and inject energy.
    // 0.35x bounding radius per iteration: still generous for free motion
    // (iterations multiply), but a single step can no longer carry a body
    // clean through a thin contact (e.g. chainmail tube pull-through)
    float maxLin = 0.35f * fabs(shape[body].w);
    float lin2 = dot(dxLin, dxLin);
    if (lin2 > maxLin * maxLin) dxLin *= maxLin * rsqrt(lin2);
    if (shape[body].w < 0.0f
        && P.surfaceTruncationMode == SURFACE_TRUNCATION_PLANAR_DAT
        && P.tetInversionPreventionEnabled != 0u) {
        float tetT = 1.0f;
        for (uint k = as; k < ae; ++k) {
            uint entry = adjList[k];
            if ((entry >> ADJ_KIND_SHIFT) == FK_TET) {
                tetT = min(tetT, tetInversionSafeFraction(
                    tets[entry & ADJ_INDEX_MASK], body, posLin, dxLin,
                    P.planarDATRelaxation, ogcCounters));
            }
        }
        dxLin *= tetT;
    }
    float ang2 = dot(dxAng, dxAng);
    const float maxAng = 0.5f;              // ~29 deg per iteration
    if (ang2 > maxAng * maxAng) dxAng *= maxAng * rsqrt(ang2);
    // Direct write: bodies within one color are non-adjacent except in the
    // rare same-color-conflict case, where this degrades to a Jacobi update
    // (paper Sec. 4) — aligned 16B stores keep neighbor reads consistent.
    posLin[body] = float4(pl.xyz + dxLin, mass);
    posAng[body] = q_addw(posAng[body], dxAng);
}


// ----------------------------------------------------------------------------
// Dual update — one thread per joint, plus one per manifold.
// ----------------------------------------------------------------------------

kernel void primal_solve(
    device float4* posLin           [[buffer(0)]],
    device float4* posAng           [[buffer(1)]],
    device const float4* initLin    [[buffer(2)]],
    device const float4* initAng    [[buffer(3)]],
    device const float4* inertLin   [[buffer(4)]],
    device const float4* inertAng   [[buffer(5)]],
    device const float4* props      [[buffer(6)]],
    device const JointGPU* joints   [[buffer(7)]],
    device const SpringGPU* springs [[buffer(8)]],
    device const ManifoldGPU* manifolds [[buffer(9)]],
    device const uint* adjStart     [[buffer(10)]],
    device const uint* adjCount     [[buffer(11)]],
    device const uint* adjList      [[buffer(12)]],
    device const uint* colorList    [[buffer(13)]],
    device const uint* colorStart   [[buffer(14)]],
    constant uint& colorIndex       [[buffer(15)]],
    constant SimParams& P           [[buffer(16)]],
    device const float4* shape      [[buffer(17)]],
    device const TetGPU* tets       [[buffer(18)]],
    device const SoftContactGPU* soft [[buffer(19)]],
    device const MembraneGPU* membranes [[buffer(20)]],
    device const BendGPU* bends     [[buffer(21)]],
    device const uint* boundsBits   [[buffer(22)]],
    device const float4* ogcPrev    [[buffer(23)]],
    device atomic_uint* ogcCounters [[buffer(24)]],
    uint tid                        [[thread_position_in_grid]])
{
    uint s = colorStart[colorIndex];
    uint e = colorStart[colorIndex + 1];
    if (s + tid >= e) return;
    primal_one(posLin, posAng, initLin, initAng, inertLin, inertAng, props,
               joints, springs, manifolds, adjStart, adjCount, adjList,
               P, shape, tets, soft, membranes, bends, boundsBits, ogcPrev,
               ogcCounters, colorList[s + tid]);
}

// Lane-split primal: 8 threads/body for established mixed/soft scenes and
// 16 for large compact rigid manifolds. Each lane stamps an adjacency slice,
// the group reduces the system via simd_shuffle_xor, and lane 0 solves.
// Per-body serial stamping (~20-30 scattered gathers) left contact-rich
// dispatches latency-bound; cooperative stamping exposes those gathers.
template <uint splitShift, uint splitWidth>
static inline void primal_split_impl(
    device float4* posLin,
    device float4* posAng,
    device const float4* initLin,
    device const float4* initAng,
    device const float4* inertLin,
    device const float4* inertAng,
    device const float4* props,
    device const JointGPU* joints,
    device const SpringGPU* springs,
    device const ManifoldGPU* manifolds,
    device const uint* adjStart,
    device const uint* adjCount,
    device const uint* adjList,
    device const uint* colorList,
    device const uint* colorStart,
    constant uint& colorIndex,
    constant SimParams& P,
    device const float4* shape,
    device const TetGPU* tets,
    device const SoftContactGPU* soft,
    device const MembraneGPU* membranes,
    device const BendGPU* bends,
    device const uint* boundsBits,
    device const float4* ogcPrev,
    device atomic_uint* ogcCounters,
    device const SolverManifoldGPU* solveManifolds,
    device const SolverContactGPU* solveContacts,
    constant uint& useCompactManifolds,
    uint tid)
{
    uint s = colorStart[colorIndex];
    uint e = colorStart[colorIndex + 1];
    uint slot = tid >> splitShift;
    uint lane = tid & (splitWidth - 1u);
    if (s + slot >= e) return;       // uniform across the cooperative group
    uint body = colorList[s + slot];
    // Rigids stamp cooperatively too: a box resting on high-res cloth
    // gathers THOUSANDS of contacts (rt corner-tri + particle np) in its
    // adjacency — serialized on one lane it gated the whole dispatch
    // (boxoncloth res 90: solve-primal 20.6 ms, 84% of frame).
    bool rigid = shape[body].w >= 0.0f;     // uniform across the group

    PrimalAccum acc;
    acc.lhsLin = m3_zero();
    acc.lhsAng = m3_zero();
    acc.lhsCross = m3_zero();
    acc.rhsLin = float3(0);
    acc.rhsAng = float3(0);

    uint as = adjStart[body];
    uint cnt = adjCount[body];
    for (uint k = lane; k < cnt; k += splitWidth) {
        uint entry = adjList[as + k];
        uint kind = entry >> ADJ_KIND_SHIFT;
        uint idx = entry & ADJ_INDEX_MASK;
        if (kind == FK_JOINT) {
            stampJoint(joints[idx], body, posLin, posAng, initAng, P.alpha, P.dt, acc);
        } else if (kind == FK_SPRING) {
            stampSpring(springs[idx], body, posLin, posAng, acc, P.alpha);
        } else if (kind == FK_TET) {
            stampTet(tets[idx], body, posLin, acc);
        } else if (kind == FK_SOFT) {
            stampSoft(soft[idx], body, posLin, posAng, initLin, initAng, P.alpha, acc);
        } else if (kind == FK_MEMBRANE) {
            stampMembrane(membranes[idx], body, posLin, acc);
        } else if (kind == FK_BEND) {
            stampBend(bends[idx], body, posLin, acc);
        } else if (useCompactManifolds != 0u) {
            stampSolverManifold(
                solveManifolds, solveContacts, idx, P.maxManifolds, body,
                posLin, posAng, initLin, initAng, P.alpha, acc);
        } else {
            stampManifold(manifolds[idx], body, posLin, posAng, initLin, initAng, P.alpha, acc);
        }
    }

    // reduce across the consecutive lanes of one simdgroup:
    // particles need only the linear 3x3, rigids the full 6x6
    for (ushort d = ushort(splitWidth >> 1); d >= 1; d >>= 1) {
        acc.lhsLin.r0 += simd_shuffle_xor(acc.lhsLin.r0, d);
        acc.lhsLin.r1 += simd_shuffle_xor(acc.lhsLin.r1, d);
        acc.lhsLin.r2 += simd_shuffle_xor(acc.lhsLin.r2, d);
        acc.rhsLin    += simd_shuffle_xor(acc.rhsLin, d);
        if (rigid) {
            acc.lhsAng.r0   += simd_shuffle_xor(acc.lhsAng.r0, d);
            acc.lhsAng.r1   += simd_shuffle_xor(acc.lhsAng.r1, d);
            acc.lhsAng.r2   += simd_shuffle_xor(acc.lhsAng.r2, d);
            acc.lhsCross.r0 += simd_shuffle_xor(acc.lhsCross.r0, d);
            acc.lhsCross.r1 += simd_shuffle_xor(acc.lhsCross.r1, d);
            acc.lhsCross.r2 += simd_shuffle_xor(acc.lhsCross.r2, d);
            acc.rhsAng      += simd_shuffle_xor(acc.rhsAng, d);
        }
    }
    if (lane != 0) return;

    float4 pl = posLin[body];
    float mass = pl.w;
    float dt2 = P.dt * P.dt;
    acc.lhsLin = m3_add(acc.lhsLin, m3_diag(float3(mass / dt2)));
    acc.rhsLin += (pl.xyz - inertLin[body].xyz) * (mass / dt2);

    float3 dxLin = float3(0), dxAng = float3(0);
    if (rigid) {
        float3 moment = props[body].xyz;
        M3 inertia = m3_scale(
            world_inertia(posAng[body], moment), 1.0f / dt2);
        acc.lhsAng = m3_add(acc.lhsAng, inertia);
        acc.rhsAng += m3_mul(
            inertia, q_sub(posAng[body], inertAng[body]));
        solve6x6(acc.lhsLin, acc.lhsAng, acc.lhsCross,
                 -acc.rhsLin, -acc.rhsAng, dxLin, dxAng);
        if (!finite3(dxLin) || !finite3(dxAng)) return;
        float ang2 = dot(dxAng, dxAng);
        const float maxAng = 0.5f;          // ~29 deg per iteration
        if (ang2 > maxAng * maxAng) dxAng *= maxAng * rsqrt(ang2);
    } else {
        // direct 3x3 LDL solve (particles carry no angular block)
        solve6x6(acc.lhsLin, m3_identity(), m3_zero(),
                 -acc.rhsLin, float3(0), dxLin, dxAng);
        if (!finite3(dxLin)) return;
    }

    float maxLin = 0.35f * fabs(shape[body].w);
    float lin2 = dot(dxLin, dxLin);
    if (lin2 > maxLin * maxLin) dxLin *= maxLin * rsqrt(lin2);
    if (!rigid
        && P.surfaceTruncationMode == SURFACE_TRUNCATION_PLANAR_DAT
        && P.tetInversionPreventionEnabled != 0u) {
        float tetT = 1.0f;
        for (uint k = 0u; k < cnt; ++k) {
            uint entry = adjList[as + k];
            if ((entry >> ADJ_KIND_SHIFT) == FK_TET) {
                tetT = min(tetT, tetInversionSafeFraction(
                    tets[entry & ADJ_INDEX_MASK], body, posLin, dxLin,
                    P.planarDATRelaxation, ogcCounters));
            }
        }
        dxLin *= tetT;
    }
    if (!rigid && P.numTris > 0u
        && P.surfaceTruncationMode == SURFACE_TRUNCATION_ISOTROPIC) {
        dxLin = ogcTruncate(dxLin, pl.xyz, ogcPrev[body].xyz,
                            boundsBits[body], -shape[body].w, ogcCounters);
    }
    posLin[body] = float4(pl.xyz + dxLin, mass);
    if (rigid) posAng[body] = q_addw(posAng[body], dxAng);
}

#define PRIMAL_SPLIT_KERNEL_ARGS \
    device float4* posLin [[buffer(0)]], \
    device float4* posAng [[buffer(1)]], \
    device const float4* initLin [[buffer(2)]], \
    device const float4* initAng [[buffer(3)]], \
    device const float4* inertLin [[buffer(4)]], \
    device const float4* inertAng [[buffer(5)]], \
    device const float4* props [[buffer(6)]], \
    device const JointGPU* joints [[buffer(7)]], \
    device const SpringGPU* springs [[buffer(8)]], \
    device const ManifoldGPU* manifolds [[buffer(9)]], \
    device const uint* adjStart [[buffer(10)]], \
    device const uint* adjCount [[buffer(11)]], \
    device const uint* adjList [[buffer(12)]], \
    device const uint* colorList [[buffer(13)]], \
    device const uint* colorStart [[buffer(14)]], \
    constant uint& colorIndex [[buffer(15)]], \
    constant SimParams& P [[buffer(16)]], \
    device const float4* shape [[buffer(17)]], \
    device const TetGPU* tets [[buffer(18)]], \
    device const SoftContactGPU* soft [[buffer(19)]], \
    device const MembraneGPU* membranes [[buffer(20)]], \
    device const BendGPU* bends [[buffer(21)]], \
    device const uint* boundsBits [[buffer(22)]], \
    device const float4* ogcPrev [[buffer(23)]], \
    device atomic_uint* ogcCounters [[buffer(24)]], \
    device const SolverManifoldGPU* solveManifolds [[buffer(25)]], \
    device const SolverContactGPU* solveContacts [[buffer(26)]], \
    constant uint& useCompactManifolds [[buffer(27)]], \
    uint tid [[thread_position_in_grid]]

#define CALL_PRIMAL_SPLIT(SHIFT, WIDTH) \
    primal_split_impl<SHIFT, WIDTH>( \
        posLin, posAng, initLin, initAng, inertLin, inertAng, props, \
        joints, springs, manifolds, adjStart, adjCount, adjList, \
        colorList, colorStart, colorIndex, P, shape, tets, soft, \
        membranes, bends, boundsBits, ogcPrev, ogcCounters, \
        solveManifolds, solveContacts, useCompactManifolds, tid)

kernel void primal_particles_split(PRIMAL_SPLIT_KERNEL_ARGS)
{
    CALL_PRIMAL_SPLIT(3u, 8u);
}

kernel void primal_rigid_split(PRIMAL_SPLIT_KERNEL_ARGS)
{
    CALL_PRIMAL_SPLIT(4u, 16u);
}

#undef CALL_PRIMAL_SPLIT
#undef PRIMAL_SPLIT_KERNEL_ARGS

// Dual update for one element contact: bounded normal dual (cap scaled to
// the participating masses at detection), friction-coned tangentials,
// ramped penalties — the rigid-contact treatment on the 4-slot stencil.
static inline void dual_soft_one(
    device const float4* posLin,
    device const float4* posAng,
    device const float4* initLin,
    device const float4* initAng,
    device SoftContactGPU* soft,
    constant SimParams& P,
    uint gid)
{
    device SoftContactGPU& sc = soft[gid];
    float3 n, t1, t2, rAW;
    bool rigidA, roundA;
    float3 C = softContactC(sc, posLin, posAng, initLin, initAng, P.alpha,
                            n, t1, t2, rAW, rigidA, roundA);
    // The dual tracks the QUADRATIC stage only — the OGC log-barrier stage
    // (softContactForce) is dual-free by design: lambda carries steady
    // load, the barrier guards the deep regime.
    float3 F = sc.penalty.xyz * C + sc.lambda.xyz;
    F.x = min(F.x, 0.0f);
    float bnd = fabs(F.x) * sc.normal.w;
    float fs = length(F.yz);
    if (fs > bnd && fs > 0.0f) F.yz *= bnd / fs;

    F.x = max(F.x, -sc.C0.w);           // bounded dual, mass-scaled cap
    sc.lambda.xyz = F;

    // Normal penalty capped at 1e6 here: gram-scale stencils next to 1e10
    // penalties leave Gauss-Seidel a stiffness ratio it cannot relax (the
    // pile jitters forever). 1e6 still holds stacked layers at sub-mm C.
    float3 pen = sc.penalty.xyz;
    if (F.x < 0.0f) {
        pen.x = min(pen.x + P.betaLin * fabs(C.x), PENALTY_MAX_T);
    }
    if (fs <= bnd) {
        pen.y = min(pen.y + P.betaLin * fabs(C.y), PENALTY_MAX_T);
        pen.z = min(pen.z + P.betaLin * fabs(C.z), PENALTY_MAX_T);
    }
    sc.penalty.xyz = pen;
}

static inline void dual_joint_one(
    device const float4* posLin,
    device const float4* posAng,
    device JointGPU* joints,
    constant SimParams& P,
    uint gid)
{
    device JointGPU& j = joints[gid];
    if (j.header.z != 0) return;

    uint a = j.header.x, b = j.header.y;
    float torqueArm = j.C0Lin.w;
    float stiffLin = j.rA.w;
    float stiffAng = j.rB.w;

    float3 penLin = j.penaltyLin.xyz;
    if (dot(penLin, penLin) > 0.0f) {
        float3 pA = a == WORLD_BODY ? j.rA.xyz : xform(posLin[a].xyz, posAng[a], j.rA.xyz);
        float3 pB = xform(posLin[b].xyz, posAng[b], j.rB.xyz);
        float3 C = pA - pB;
        if (j.header.w & 1) {
            C -= j.C0Lin.xyz * P.alpha;
            j.lambdaLin.xyz = clamp(penLin * C + j.lambdaLin.xyz,
                                    -P.lambdaMax, P.lambdaMax);
        }
        float cap = min(stiffLin, PENALTY_MAX);
        j.penaltyLin.xyz = min(penLin + fabs(C) * P.betaLin, cap);
    }

    float3 penAng = j.penaltyAng.xyz;
    if (dot(penAng, penAng) > 0.0f) {
        float4 qA = a == WORLD_BODY ? float4(0,0,0,1) : posAng[a];
        float3 C;
        if (j.hingeAxis.w != 0.0f) {
            float3 aB = q_rotate(posAng[b], j.hingeAxis.xyz);
            float3 aA = q_rotate(q_mul(qA, j.restRel), j.hingeAxis.xyz);
            C = cross(aA, aB) * torqueArm;
        } else {
            C = q_sub(q_mul(qA, j.restRel), posAng[b]) * torqueArm;
        }
        if (j.header.w & 2) {
            C -= j.C0Ang.xyz * P.alpha;
            j.lambdaAng.xyz = clamp(penAng * C + j.lambdaAng.xyz,
                                    -P.lambdaMax, P.lambdaMax);
        }
        float cap = min(stiffAng, PENALTY_MAX);
        j.penaltyAng.xyz = min(penAng + fabs(C) * P.betaAng, cap);
    }

    // Fracture (flagged joints only; avoids inf comparisons under fast math).
    // Angular lambda by default; linear lambda only when bit 8 is set
    // (tension releases like slings) — linear lambda is penalty-scaled and
    // spikes transiently on hard welds.
    float fracture = j.C0Ang.w;
    float3 la = j.lambdaAng.xyz;
    float lin2 = (j.header.w & 8) ? length_squared(j.lambdaLin.xyz) : 0.0f;
    if ((j.header.w & 4) && dot(la, la) + lin2 > fracture * fracture) {
        j.penaltyLin = float4(0);
        j.penaltyAng = float4(0);
        j.lambdaLin = float4(0);
        j.lambdaAng = float4(0);
        j.header.z = 1;
    }
}

static inline void dual_manifold_one(
    device const float4* posLin,
    device const float4* posAng,
    device const float4* initLin,
    device const float4* initAng,
    device ManifoldGPU* manifolds,
    constant SimParams& P,
    uint gid)
{
    device ManifoldGPU& m = manifolds[gid];
    uint n = m.header.z;
    if (n == 0) return;

    uint a = m.header.x, b = m.header.y;
    bool sphA = (m.header.w & 2u) != 0;
    bool sphB = (m.header.w & 4u) != 0;
    float3 dqALin = posLin[a].xyz - initLin[a].xyz;
    float3 dqAAng = q_sub(posAng[a], initAng[a]);
    float3 dqBLin = posLin[b].xyz - initLin[b].xyz;
    float3 dqBAng = q_sub(posAng[b], initAng[b]);

    // Dual bound scaled to the lighter participant: a gram-scale cloth node
    // must not stockpile kilonewtons (paper Sec 4's bound made dimensional;
    // wedged nodes otherwise ratchet lambda to the global cap and explode).
    float mA = posLin[a].w, mB = posLin[b].w;
    float minMass = min(mA > 0.0f ? mA : FLT_MAX, mB > 0.0f ? mB : FLT_MAX);
    float lamCap = min(P.lambdaMax, max(10.0f, 1.0e5f * minMass));

    for (uint i = 0; i < n; i++) {
        float3 rAW = sphA ? m.contacts[i].rA.xyz : q_rotate(posAng[a], m.contacts[i].rA.xyz);
        float3 rBW = sphB ? m.contacts[i].rB.xyz : q_rotate(posAng[b], m.contacts[i].rB.xyz);

        float3 C; float fs, bnd;
        float3 F = contactForceC(m, i, dqALin, dqAAng, dqBLin, dqBAng, rAW, rBW, P.alpha, C, fs, bnd);

        // bound the dual (paper Sec 4): conflicting contacts otherwise ramp
        // each other's lambda without limit until the stored force explodes
        F.x = max(F.x, -lamCap);
        m.contacts[i].lambda = float4(F, 0);

        float3 pen = m.contacts[i].penalty.xyz;
        if (F.x < 0.0f) {
            pen.x = min(pen.x + P.betaLin * fabs(C.x), PENALTY_MAX);
        }
        if (fs <= bnd) {
            pen.y = min(pen.y + P.betaLin * fabs(C.y), PENALTY_MAX_T);
            pen.z = min(pen.z + P.betaLin * fabs(C.z), PENALTY_MAX_T);
            m.contacts[i].rB.w = length(C.yz) < STICK_THRESH ? 1.0f : 0.0f;
        } else {
            // SLIDING DROPS THE ANCHOR. This flag was only ever updated in
            // the in-cone branch above, so a contact that stuck once and
            // then started sliding kept its original anchors for ever: the
            // tangential constraint accumulated the WHOLE slide (12 cm,
            // measured), the applied force stayed cone-capped so pushing
            // still worked - and when the push stopped, that capped force
            // crept the body all the way back to its stale anchor. Every
            // grounded object was elastically tethered to wherever it first
            // touched down. Invisible to any task that LIFTS objects
            // (breaking contact resets anchors); fatal to planar pushing,
            // and to any pushing data collected for training.
            m.contacts[i].rB.w = 0.0f;
        }
        m.contacts[i].penalty = float4(pen, 0);
    }
}

static inline void dual_solver_manifold_one(
    device const float4* posLin,
    device const float4* posAng,
    device const float4* initLin,
    device const float4* initAng,
    device const SolverManifoldGPU* solveManifolds,
    device SolverContactGPU* solveContacts,
    device ManifoldGPU* manifolds,
    bool writeBack,
    constant SimParams& P,
    uint gid)
{
    device const SolverManifoldGPU& m = solveManifolds[gid];
    uint n = m.header.z;
    if (n == 0) return;

    uint a = m.header.x, b = m.header.y;
    bool sphA = (m.header.w & 2u) != 0;
    bool sphB = (m.header.w & 4u) != 0;
    float3 dqALin = posLin[a].xyz - initLin[a].xyz;
    float3 dqAAng = q_sub(posAng[a], initAng[a]);
    float3 dqBLin = posLin[b].xyz - initLin[b].xyz;
    float3 dqBAng = q_sub(posAng[b], initAng[b]);

    float mA = posLin[a].w, mB = posLin[b].w;
    float minMass = min(mA > 0.0f ? mA : FLT_MAX,
                        mB > 0.0f ? mB : FLT_MAX);
    float lamCap = min(P.lambdaMax, max(10.0f, 1.0e5f * minMass));

    for (uint i = 0; i < n; ++i) {
        device SolverContactGPU& c =
            solveContacts[i * P.maxManifolds + gid];
        float3 rAW = sphA ? c.rAStick.xyz
                          : q_rotate(posAng[a], c.rAStick.xyz);
        float3 rBW = sphB ? c.rB_C0x.xyz
                          : q_rotate(posAng[b], c.rB_C0x.xyz);

        float3 C; float fs, bnd;
        float3 F = solverContactForceC(
            m, c, dqALin, dqAAng, dqBLin, dqBAng,
            rAW, rBW, P.alpha, C, fs, bnd);

        F.x = max(F.x, -lamCap);
        c.C0yz_lambdaXY.zw = F.xy;
        c.lambdaZ_penalty.x = F.z;

        float3 pen = c.lambdaZ_penalty.yzw;
        if (F.x < 0.0f) {
            pen.x = min(pen.x + P.betaLin * fabs(C.x), PENALTY_MAX);
        }
        if (fs <= bnd) {
            pen.y = min(pen.y + P.betaLin * fabs(C.y), PENALTY_MAX_T);
            pen.z = min(pen.z + P.betaLin * fabs(C.z), PENALTY_MAX_T);
            c.rAStick.w = length(C.yz) < STICK_THRESH ? 1.0f : 0.0f;
        } else {
            c.rAStick.w = 0.0f;
        }
        c.lambdaZ_penalty.yzw = pen;
        if (writeBack) {
            device ContactGPU& dst = manifolds[gid].contacts[i];
            dst.rB.w = c.rAStick.w;
            dst.lambda.xyz = F;
            dst.penalty.xyz = pen;
        }
    }
}

// Fused dual update: one dispatch covers joints then contact manifolds.
// Small scenes with many iterations are dispatch-bound; merging the two
// kernels halves the per-iteration barrier count.
kernel void dual_all(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device const float4* initLin    [[buffer(2)]],
    device const float4* initAng    [[buffer(3)]],
    device JointGPU* joints         [[buffer(4)]],
    device ManifoldGPU* manifolds   [[buffer(5)]],
    device const atomic_uint* counters [[buffer(6)]],
    constant SimParams& P           [[buffer(7)]],
    device SpringGPU* springs       [[buffer(8)]],
    device SoftContactGPU* soft     [[buffer(9)]],
    device const SolverManifoldGPU* solveManifolds [[buffer(10)]],
    device SolverContactGPU* solveContacts [[buffer(11)]],
    constant uint& useCompactManifolds [[buffer(12)]],
    constant uint& writeBackCompactManifolds [[buffer(13)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid < P.numJoints) {
        dual_joint_one(posLin, posAng, joints, P, gid);
        return;
    }
    uint si = gid - P.numJoints;
    if (si < P.numSprings) {
        device SpringGPU& sp = springs[si];
        if (sp.header.z == 0) return;
        uint a = sp.header.x, b = sp.header.y;
        float3 pA = xform(posLin[a].xyz, posAng[a], sp.rA.xyz);
        float3 pB = xform(posLin[b].xyz, posAng[b], sp.rB.xyz);
        float C = length(pA - pB) - sp.rB.w - sp.dual.z * P.alpha;
        // Tension-only dual (lambda >= 0), mass-scaled cap at HALF the
        // contact scale: when an inextensible weave fights a contact (taut
        // hammock vs box corner) the contact wins — structure may stretch,
        // bodies never tunnel.
        float mA = posLin[a].w, mB = posLin[b].w;
        float mMin = min(mA > 0.0f ? mA : FLT_MAX, mB > 0.0f ? mB : FLT_MAX);
        float cap = min(P.lambdaMax, max(2.0f, 5.0e3f * mMin));
        // leaky dual: under global coupling (a hanging chain cannot reach
        // the alpha-target in one frame of sweeps) a perfect integrator
        // ratchets lambda to the cap and limit-cycles the skirt. The leak
        // pins steady-state lambda to actual need (C_ss ~ lambda*0.02/pen,
        // sub-micron) and drains stockpiles in ~10 frames.
        sp.dual.x = clamp(0.98f * sp.dual.x + sp.dual.y * C, 0.0f, cap);
        if (C > 0.0f) {
            sp.dual.y = min(sp.dual.y + C * P.betaLin,
                            min(sp.rA.w, PENALTY_MAX));
        }
        return;
    }
    uint mi = si - P.numSprings;
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    if (mi < numPairs) {
        if (useCompactManifolds != 0u) {
            dual_solver_manifold_one(
                posLin, posAng, initLin, initAng,
                solveManifolds, solveContacts, manifolds,
                writeBackCompactManifolds != 0u, P, mi);
        } else {
            if (manifolds[mi].header.z == 0) return;
            dual_manifold_one(
                posLin, posAng, initLin, initAng, manifolds, P, mi);
        }
        return;
    }
    uint sci = mi - numPairs;
    uint numSoft = min(atomic_load_explicit(&counters[CTR_SOFT], memory_order_relaxed),
                       P.maxSoft);
    if (sci >= numSoft) return;
    dual_soft_one(posLin, posAng, initLin, initAng, soft, P, sci);
}

// Persistent solver for small scenes: ALL iterations x colors x dual updates
// in a single dispatch of one threadgroup. Scenes with few bodies and many
// iterations are dominated by per-dispatch launch/barrier latency (~40us
// each); this replaces hundreds of dispatches with threadgroup barriers.
kernel void solve_persistent(
    device float4* posLin           [[buffer(0)]],
    device float4* posAng           [[buffer(1)]],
    device const float4* initLin    [[buffer(2)]],
    device const float4* initAng    [[buffer(3)]],
    device const float4* inertLin   [[buffer(4)]],
    device const float4* inertAng   [[buffer(5)]],
    device const float4* props      [[buffer(6)]],
    device JointGPU* joints         [[buffer(7)]],
    device SpringGPU* springs       [[buffer(8)]],
    device ManifoldGPU* manifolds   [[buffer(9)]],
    device const uint* adjStart     [[buffer(10)]],
    device const uint* adjCount     [[buffer(11)]],
    device const uint* adjList      [[buffer(12)]],
    device const uint* colorList    [[buffer(13)]],
    device const uint* colorStart   [[buffer(14)]],
    device const atomic_uint* counters [[buffer(15)]],
    constant SimParams& P           [[buffer(16)]],
    device const float4* shape      [[buffer(17)]],
    device const TetGPU* tets       [[buffer(18)]],
    device SoftContactGPU* soft     [[buffer(19)]],
    device const MembraneGPU* membranes [[buffer(20)]],
    device const BendGPU* bends     [[buffer(21)]],
    device const uint* boundsBits   [[buffer(26)]],
    device const float4* ogcPrev    [[buffer(27)]],
    device atomic_uint* ogcCounters [[buffer(28)]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint tgSize                     [[threads_per_threadgroup]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    uint numSoft = min(atomic_load_explicit(&counters[CTR_SOFT], memory_order_relaxed),
                       P.maxSoft);
    uint dualTotal = P.numJoints + P.numSprings + numPairs + numSoft;
    // empty palette tail costs a threadgroup barrier per color per
    // iteration (64x20 = 1280 barriers dominated small-scene solves)
    uint usedColors = colorStart[MAX_COLORS + 1];
    for (uint iter = 0; iter < P.iterations; iter++) {
        for (uint c = 0; c < usedColors; c++) {
            uint s0 = colorStart[c];
            uint e0 = colorStart[c + 1];
            if (s0 == e0) continue;
            for (uint i = s0 + tid; i < e0; i += tgSize) {
                primal_one(posLin, posAng, initLin, initAng, inertLin, inertAng,
                           props, joints, springs, manifolds,
                           adjStart, adjCount, adjList, P, shape, tets, soft,
                           membranes, bends, boundsBits, ogcPrev, ogcCounters,
                           colorList[i]);
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
        for (uint g = tid; g < dualTotal; g += tgSize) {
            if (g < P.numJoints) {
                if (joints[g].header.z == 0)
                    dual_joint_one(posLin, posAng, joints, P, g);
            } else if (g < P.numJoints + P.numSprings) {
                device SpringGPU& sp = springs[g - P.numJoints];
                if (sp.header.z != 0) {
                    uint a = sp.header.x, b = sp.header.y;
                    float3 pA = xform(posLin[a].xyz, posAng[a], sp.rA.xyz);
                    float3 pB = xform(posLin[b].xyz, posAng[b], sp.rB.xyz);
                    float C = length(pA - pB) - sp.rB.w - sp.dual.z * P.alpha;
                    float mA = posLin[a].w, mB = posLin[b].w;
                    float mMin = min(mA > 0.0f ? mA : FLT_MAX,
                                     mB > 0.0f ? mB : FLT_MAX);
                    float cap = min(P.lambdaMax, max(2.0f, 5.0e3f * mMin));
                    sp.dual.x = clamp(0.98f * sp.dual.x + sp.dual.y * C, 0.0f, cap);
                    if (C > 0.0f) {
                        sp.dual.y = min(sp.dual.y + C * P.betaLin,
                                        min(sp.rA.w, PENALTY_MAX));
                    }
                }
            } else if (g < P.numJoints + P.numSprings + numPairs) {
                uint mi = g - P.numJoints - P.numSprings;
                if (manifolds[mi].header.z != 0)
                    dual_manifold_one(posLin, posAng, initLin, initAng,
                                      manifolds, P, mi);
            } else {
                uint sci = g - P.numJoints - P.numSprings - numPairs;
                dual_soft_one(posLin, posAng, initLin, initAng, soft, P, sci);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Multi-threadgroup persistent solver: the whole iteration loop in ONE
// dispatch of a few co-resident threadgroups (Apple GPUs keep a small grid
// resident), synchronized by a monotonic device-scope arrive-and-spin
// barrier. Fixes both small-scene pathologies at once: the single-group
// kernel ran the solve on one GPU core, and the per-color dispatch path
// paid ~50us launch latency x colors x iterations.
inline void global_barrier(device atomic_uint* bar, uint ntg, uint tid,
                           thread uint& expected)
{
    threadgroup_barrier(mem_flags::mem_device);
    expected += ntg;
    if (tid == 0) {
        atomic_fetch_add_explicit(bar, 1u, memory_order_relaxed);
        while (atomic_load_explicit(bar, memory_order_relaxed) < expected) {}
    }
    threadgroup_barrier(mem_flags::mem_device);
}

kernel void solve_persistent_multi(
    device float4* posLin           [[buffer(0)]],
    device float4* posAng           [[buffer(1)]],
    device const float4* initLin    [[buffer(2)]],
    device const float4* initAng    [[buffer(3)]],
    device const float4* inertLin   [[buffer(4)]],
    device const float4* inertAng   [[buffer(5)]],
    device const float4* props      [[buffer(6)]],
    device JointGPU* joints         [[buffer(7)]],
    device SpringGPU* springs       [[buffer(8)]],
    device ManifoldGPU* manifolds   [[buffer(9)]],
    device const uint* adjStart     [[buffer(10)]],
    device const uint* adjCount     [[buffer(11)]],
    device const uint* adjList      [[buffer(12)]],
    device const uint* colorList    [[buffer(13)]],
    device const uint* colorStart   [[buffer(14)]],
    device atomic_uint* counters    [[buffer(15)]],
    constant SimParams& P           [[buffer(16)]],
    device const float4* shape      [[buffer(17)]],
    device const TetGPU* tets       [[buffer(18)]],
    device SoftContactGPU* soft     [[buffer(19)]],
    device const MembraneGPU* membranes [[buffer(20)]],
    device const BendGPU* bends     [[buffer(21)]],
    constant uint& ntg              [[buffer(22)]],
    device const uint* boundsBits   [[buffer(26)]],
    device const float4* ogcPrev    [[buffer(27)]],
    device atomic_uint* ogcCounters [[buffer(28)]],
    uint tid                        [[thread_position_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tgSize                     [[threads_per_threadgroup]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    uint numSoft = min(atomic_load_explicit(&counters[CTR_SOFT], memory_order_relaxed),
                       P.maxSoft);
    uint dualTotal = P.numJoints + P.numSprings + numPairs + numSoft;
    uint usedColors = colorStart[MAX_COLORS + 1];
    uint gtid = tgid * tgSize + tid;
    uint gsize = ntg * tgSize;
    uint expected = 0;
    device atomic_uint* bar = &counters[CTR_GBAR];

    for (uint iter = 0; iter < P.iterations; iter++) {
        for (uint c = 0; c < usedColors; c++) {
            uint s0 = colorStart[c];
            uint e0 = colorStart[c + 1];
            for (uint i = s0 + gtid; i < e0; i += gsize) {
                primal_one(posLin, posAng, initLin, initAng, inertLin, inertAng,
                           props, joints, springs, manifolds,
                           adjStart, adjCount, adjList, P, shape, tets, soft,
                           membranes, bends, boundsBits, ogcPrev, ogcCounters,
                           colorList[i]);
            }
            global_barrier(bar, ntg, tid, expected);
        }
        for (uint g = gtid; g < dualTotal; g += gsize) {
            if (g < P.numJoints) {
                if (joints[g].header.z == 0)
                    dual_joint_one(posLin, posAng, joints, P, g);
            } else if (g < P.numJoints + P.numSprings) {
                device SpringGPU& sp = springs[g - P.numJoints];
                if (sp.header.z != 0) {
                    uint a = sp.header.x, b = sp.header.y;
                    float3 pA = xform(posLin[a].xyz, posAng[a], sp.rA.xyz);
                    float3 pB = xform(posLin[b].xyz, posAng[b], sp.rB.xyz);
                    float C = length(pA - pB) - sp.rB.w - sp.dual.z * P.alpha;
                    float mA = posLin[a].w, mB = posLin[b].w;
                    float mMin = min(mA > 0.0f ? mA : FLT_MAX,
                                     mB > 0.0f ? mB : FLT_MAX);
                    float cap = min(P.lambdaMax, max(2.0f, 5.0e3f * mMin));
                    sp.dual.x = clamp(0.98f * sp.dual.x + sp.dual.y * C, 0.0f, cap);
                    if (C > 0.0f) {
                        sp.dual.y = min(sp.dual.y + C * P.betaLin,
                                        min(sp.rA.w, PENALTY_MAX));
                    }
                }
            } else if (g < P.numJoints + P.numSprings + numPairs) {
                uint mi = g - P.numJoints - P.numSprings;
                if (manifolds[mi].header.z != 0)
                    dual_manifold_one(posLin, posAng, initLin, initAng,
                                      manifolds, P, mi);
            } else {
                uint sci = g - P.numJoints - P.numSprings - numPairs;
                dual_soft_one(posLin, posAng, initLin, initAng, soft, P, sci);
            }
        }
        global_barrier(bar, ntg, tid, expected);
    }
}

// ----------------------------------------------------------------------------
// Velocity finalize (BDF1)
// ----------------------------------------------------------------------------

kernel void finalize_velocities(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device const float4* initLin    [[buffer(2)]],
    device const float4* initAng    [[buffer(3)]],
    device float4* velLin           [[buffer(4)]],
    device float4* velAng           [[buffer(5)]],
    device float4* prevVelLin       [[buffer(6)]],
    constant SimParams& P           [[buffer(7)]],
    device const float4* shape      [[buffer(8)]],
    device const float* gravityScale [[buffer(9)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numBodies) return;
    prevVelLin[gid] = velLin[gid];
    if (posLin[gid].w > 0.0f) {
        float3 oldV = velLin[gid].xyz;
        float3 v = (posLin[gid].xyz - initLin[gid].xyz) / P.dt;
        // Safety clamp: prevents tunneling of violently flung bodies
        float s2 = dot(v, v);
        if (s2 > P.maxSpeed * P.maxSpeed) v *= P.maxSpeed * rsqrt(s2);
        // Particle damping must not alter uniform ballistic motion: soft
        // bodies and rigids must fall at the same rate in gravity. Damp only
        // solver/contact/internal residual velocity relative to free fall.
        if (shape[gid].w < 0.0f && P.particleDamping > 0.0f) {
            float3 ballistic = oldV
                + float3(0, 0, P.gravity * gravityScale[gid]) * P.dt;
            v = ballistic + (v - ballistic) / (1.0f + P.particleDamping * P.dt);
        }
        if (shape[gid].w >= 0.0f && P.rigidLinearDamping > 0.0f) {
            v *= exp(-P.rigidLinearDamping * P.dt);
        }
        float3 omega = q_sub(posAng[gid], initAng[gid]) / P.dt;
        if (shape[gid].w >= 0.0f && P.rigidAngularDamping > 0.0f) {
            omega *= exp(-P.rigidAngularDamping * P.dt);
        }
        velLin[gid] = float4(v, 0);
        velAng[gid] = float4(omega, 0);
    }
}

// Internal viscosity for thin sheets: blend each particle's velocity toward
// its topological 1-ring average (the reference cloth ships the same pass).
// Damps RELATIVE jitter — the dominant dissipation of real fabric is
// bending-rate viscosity — while bulk translation/rotation pass through.
// Two-buffer read (prevVelLin holds the unsmoothed copy written below).
kernel void smooth_particle_velocities(
    device float4* velLin           [[buffer(0)]],
    device const float4* velRead    [[buffer(1)]],
    device const uint* particleIdx  [[buffer(2)]],
    device const uint* nbrStart     [[buffer(3)]],
    device const uint* nbrCount     [[buffer(4)]],
    device const uint* nbrList      [[buffer(5)]],
    constant SimParams& P           [[buffer(6)]],
    constant float& weight          [[buffer(7)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numParticles) return;
    uint v = particleIdx[gid];
    uint s = nbrStart[v], n = nbrCount[v];
    if (n == 0) return;
    float3 avg = float3(0);
    for (uint k = s; k < s + n; k++) {
        avg += velRead[nbrList[k]].xyz;
    }
    avg /= float(n);
    float3 vel = velRead[v].xyz;
    velLin[v] = float4(vel + (avg - vel) * weight, 0);
}

// ----------------------------------------------------------------------------
// Diagnostics: max constraint error (atomic max on float bits; values >= 0)
// ----------------------------------------------------------------------------

kernel void diag_clear(
    device atomic_uint* diag        [[buffer(0)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid == 0) atomic_store_explicit(&diag[0], 0u, memory_order_relaxed);
}

kernel void diag_error(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device const JointGPU* joints   [[buffer(2)]],
    device const ManifoldGPU* manifolds [[buffer(3)]],
    device const atomic_uint* counters [[buffer(4)]],
    device atomic_uint* diag        [[buffer(5)]],
    constant SimParams& P           [[buffer(6)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    float err = 0.0f;
    if (gid < P.numJoints) {
        device const JointGPU& j = joints[gid];
        // skip broken joints and inert exclusion-only joints (stiffness 0)
        if (j.header.z == 0 && (j.rA.w > 0.0f || j.rB.w > 0.0f)) {
            uint a = j.header.x, b = j.header.y;
            float3 pA = a == WORLD_BODY ? j.rA.xyz : xform(posLin[a].xyz, posAng[a], j.rA.xyz);
            float3 pB = xform(posLin[b].xyz, posAng[b], j.rB.xyz);
            err = length(pA - pB);
        }
    } else if (gid < P.numJoints + numPairs) {
        device const ManifoldGPU& m = manifolds[gid - P.numJoints];
        uint n = m.header.z;
        uint a = m.header.x, b = m.header.y;
        bool sphA = (m.header.w & 2u) != 0;
        bool sphB = (m.header.w & 4u) != 0;
        for (uint i = 0; i < n; i++) {
            float3 xA = sphA ? posLin[a].xyz + m.contacts[i].rA.xyz
                             : xform(posLin[a].xyz, posAng[a], m.contacts[i].rA.xyz);
            float3 xB = sphB ? posLin[b].xyz + m.contacts[i].rB.xyz
                             : xform(posLin[b].xyz, posAng[b], m.contacts[i].rB.xyz);
            float pen = dot(m.basisN.xyz, xA - xB)
                + P.collisionMargin;
            err = max(err, max(0.0f, -pen));
        }
    } else {
        return;
    }
    atomic_fetch_max_explicit(&diag[0], as_type<uint>(err), memory_order_relaxed);
}
