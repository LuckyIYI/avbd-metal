#include <metal_stdlib>
using namespace metal;

// Optional broadphase expansion for rigid bodies with compound colliders.
// This source is compiled into a separate Metal library so merely enabling
// the feature in the product cannot perturb established analytic broadphase
// fast-math code generation in scenes that do not use the hierarchy.
// Each proxy is a balanced subtree of at most 16 leaves (see
// rigidBroadphaseMaxLeavesPerProxy). Splitting one side per level gives a
// maximum paired depth of 8 and at most 9 pending entries. Leave headroom
// while avoiding the original 512-byte private stack per GPU thread.
#define BP_BVH_STACK_CAPACITY 16

struct ColliderBVHNodeGPU {
    float4 centerRadius;
    float4 halfExtent;
    uint4 links;               // left, right, leaf collider, flags
};

inline float3 bpBVHWorldCenter(
    thread const ColliderBVHNodeGPU& node, uint body,
    device const float4* posLin, device const float4* posAng)
{
    return posLin[body].xyz + q_rotate(posAng[body], node.centerRadius.xyz);
}

inline float bpBVHPairRadius(
    float radiusA, float radiusB, uint flagsA, uint flagsB,
    float speedBound, constant SimParams& P)
{
    float result = radiusA + radiusB;
    if (((flagsA | flagsB) & 2u) != 0u) {
        float cap = min(0.25f,
            max(4.0f * P.collisionMargin,
                3.0f * min(radiusA, radiusB)));
        result += P.collisionMargin + min(cap, speedBound * P.dt);
    }
    return result;
}

inline uint bpExpandHierarchyPair(
    uint2 proxyPair,
    device const uint* proxyOwner,
    device const uint* proxyRoot,
    device const ColliderBVHNodeGPU* nodes,
    device const float4* posLin,
    device const float4* posAng,
    device const float4* velLin, device const float4* velAng,
    constant SimParams& P,
    device uchar* leafPairs)
{
    uint bodyA = proxyOwner[proxyPair.x];
    uint bodyB = proxyOwner[proxyPair.y];
    uint2 stack[BP_BVH_STACK_CAPACITY];
    uint stackCount = 1u;
    stack[0] = uint2(proxyRoot[proxyPair.x], proxyRoot[proxyPair.y]);
    uint count = 0u;
    while (stackCount > 0u) {
        uint2 nodePair = stack[--stackCount];
        ColliderBVHNodeGPU nodeA = nodes[nodePair.x];
        ColliderBVHNodeGPU nodeB = nodes[nodePair.y];
        float3 centerA = bpBVHWorldCenter(nodeA, bodyA, posLin, posAng);
        float3 centerB = bpBVHWorldCenter(nodeB, bodyB, posLin, posAng);
        // Bound relative velocity at every descendant collider center.
        // Angular inflation covers the subtree radius; the same velocity
        // buffers feed narrowphase's directional speculative contact gate.
        float3 wa=velAng[bodyA].xyz, wb=velAng[bodyB].xyz;
        float3 va=velLin[bodyA].xyz+cross(wa,centerA-posLin[bodyA].xyz);
        float3 vb=velLin[bodyB].xyz+cross(wb,centerB-posLin[bodyB].xyz);
        float speedBound=length(vb-va)+length(wa)*nodeA.centerRadius.w
            + length(wb)*nodeB.centerRadius.w;
        float radius = bpBVHPairRadius(
            nodeA.centerRadius.w, nodeB.centerRadius.w,
            nodeA.links.w, nodeB.links.w, speedBound, P);
        // Preserve the original sphere envelope. Velocity refines the
        // independent box bound below, not this established traversal gate.
        float sphereRadius=bpBVHPairRadius(nodeA.centerRadius.w,
            nodeB.centerRadius.w,nodeA.links.w,nodeB.links.w,FLT_MAX,P);
        if (distance_squared(centerA, centerB) > sphereRadius * sphereRadius) continue;
        // Tight compound bounds reject distant shelves/walls that overlap a
        // large enclosing sphere. Keep the same speculative hull padding as
        // the sphere test, so no accepted near-contact distance is reduced.
        float3 ha=nodeA.halfExtent.xyz, hb=nodeB.halfExtent.xyz;
        float3 ea=abs(q_rotate(posAng[bodyA],float3(ha.x,0,0)))
            + abs(q_rotate(posAng[bodyA],float3(0,ha.y,0)))
            + abs(q_rotate(posAng[bodyA],float3(0,0,ha.z)));
        float3 eb=abs(q_rotate(posAng[bodyB],float3(hb.x,0,0)))
            + abs(q_rotate(posAng[bodyB],float3(0,hb.y,0)))
            + abs(q_rotate(posAng[bodyB],float3(0,0,hb.z)));
        float padding=radius-nodeA.centerRadius.w-nodeB.centerRadius.w;
        if (any(abs(centerA-centerB)>ea+eb+max(padding,0.0f))) continue;

        bool leafA = (nodeA.links.w & 1u) != 0u;
        bool leafB = (nodeB.links.w & 1u) != 0u;
        if (leafA && leafB) {
            // Leaf spheres are the same conservative bounds used by the
            // collider predicate, including hull contact padding above.
            // Cache their proxy-local ordinals in the original DFS order.
            leafPairs[count] = uchar((nodeA.links.w >> 2)
                | ((nodeB.links.w >> 2) << 4));
            count++;
            continue;
        }

        bool splitB = leafA || (!leafB
            && nodeB.centerRadius.w > nodeA.centerRadius.w);
        if (stackCount + 2u > BP_BVH_STACK_CAPACITY) {
            return P.maxPairs + 1u;
        }
        if (splitB) {
            stack[stackCount++] = uint2(nodePair.x, nodeB.links.y);
            stack[stackCount++] = uint2(nodePair.x, nodeB.links.x);
        } else {
            stack[stackCount++] = uint2(nodeA.links.y, nodePair.y);
            stack[stackCount++] = uint2(nodeA.links.x, nodePair.y);
        }
    }
    return count;
}

kernel void bp_count_hierarchy_pairs(
    device const uint2* proxyPairs [[buffer(0)]],
    device const atomic_uint* counters [[buffer(1)]],
    device uint* pairCounts [[buffer(2)]],
    device const uint* proxyOwner [[buffer(3)]],
    device const uint* proxyRoot [[buffer(4)]],
    device const ColliderBVHNodeGPU* nodes [[buffer(5)]],
    device const float4* posLin [[buffer(6)]],
    device const float4* posAng [[buffer(7)]],
    device uchar* leafPairs [[buffer(8)]],
    constant SimParams& P [[buffer(9)]],
    constant uint& pairCapacity [[buffer(10)]],
    device const float4* velLin [[buffer(11)]],
    device const float4* velAng [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= pairCapacity) return;
    uint proxyCount = atomic_load_explicit(
        &counters[CTR_PAIRS], memory_order_relaxed);
    pairCounts[gid] = gid < proxyCount ? bpExpandHierarchyPair(
        proxyPairs[gid], proxyOwner, proxyRoot, nodes, posLin, posAng, velLin, velAng,
        P, leafPairs + size_t(gid) * 256u) : 0u;
}

kernel void bp_emit_hierarchy_pairs(
    device const uint2* proxyPairs [[buffer(0)]],
    device const atomic_uint* counters [[buffer(1)]],
    device const uint* pairStarts [[buffer(2)]],
    device const uint* pairCounts [[buffer(3)]],
    device const uint* proxyLeaves [[buffer(4)]],
    device const uchar* leafPairs [[buffer(5)]],
    device uint2* pairs [[buffer(6)]],
    constant SimParams& P [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint proxyCount = atomic_load_explicit(
        &counters[CTR_PAIRS], memory_order_relaxed);
    if (gid >= proxyCount) return;
    uint count = pairCounts[gid];
    // Overflow is surfaced by finalize, never decoded as cached leaf data.
    if (count > 256u) return;
    uint base = pairStarts[gid];
    uint2 proxy = proxyPairs[gid];
    for (uint i = 0; i < count && base + i < P.maxPairs; i++) {
        uint code = leafPairs[size_t(gid) * 256u + i];
        uint a = proxyLeaves[size_t(proxy.x) * 16u + (code & 15u)];
        uint b = proxyLeaves[size_t(proxy.y) * 16u + (code >> 4)];
        pairs[base + i] = uint2(min(a, b), max(a, b));
    }
}

kernel void bp_finalize_hierarchy_pairs(
    device const uint* pairCounts [[buffer(0)]],
    device const uint* pairStarts [[buffer(1)]],
    device atomic_uint* counters [[buffer(2)]],
    device uint* dispatchArgs [[buffer(3)]],
    constant SimParams& P [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    // dispatch1D rounds up to a complete threadgroup. This kernel replaces
    // the proxy count with the leaf count; another writer could read that
    // new value as a proxy index and corrupt counts or access past the scan.
    if (gid != 0u) return;
    uint rawProxyCount = atomic_load_explicit(
        &counters[CTR_PAIR_CANDIDATES], memory_order_relaxed);
    uint proxyCount = atomic_load_explicit(
        &counters[CTR_PAIRS], memory_order_relaxed);
    uint n = proxyCount == 0u ? 0u
        : pairStarts[proxyCount - 1u] + pairCounts[proxyCount - 1u];
    if (rawProxyCount > P.maxPairs) n = max(n, P.maxPairs + 1u);
    atomic_store_explicit(&counters[CTR_PAIR_CANDIDATES], n,
                          memory_order_relaxed);
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
