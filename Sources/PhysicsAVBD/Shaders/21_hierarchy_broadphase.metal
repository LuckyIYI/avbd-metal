#include <metal_stdlib>
using namespace metal;

// Optional broadphase expansion for rigid bodies with compound colliders.
// This source is compiled into a separate Metal library so merely enabling
// the feature in the product cannot perturb established analytic broadphase
// fast-math code generation in scenes that do not use the hierarchy.
#define BP_BVH_STACK_CAPACITY 64

struct ColliderBVHNodeGPU {
    float4 centerRadius;
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
    constant SimParams& P)
{
    float result = radiusA + radiusB;
    if (((flagsA | flagsB) & 2u) != 0u) {
        float cap = min(0.25f,
            max(4.0f * P.collisionMargin,
                3.0f * min(radiusA, radiusB)));
        result += P.collisionMargin + cap;
    }
    return result;
}

inline float3 bpHierarchyColliderCenter(
    device const float4* posLin,
    device const float4* posAng,
    device const uint* colliderOwner,
    device const float4* colliderLocalPosition,
    uint collider)
{
    uint body = colliderOwner[collider];
    return posLin[body].xyz
        + q_rotate(posAng[body], colliderLocalPosition[collider].xyz);
}

inline float bpHierarchyLeafPairRadius(
    float signedRadiusA, float signedRadiusB,
    uint shapeTypeA, uint shapeTypeB,
    constant SimParams& P)
{
    float radiusA = fabs(signedRadiusA);
    float radiusB = fabs(signedRadiusB);
    float result = radiusA + radiusB;
    bool hullA = (shapeTypeA & SHAPE_KIND_MASK) == 4u;
    bool hullB = (shapeTypeB & SHAPE_KIND_MASK) == 4u;
    if (hullA || hullB) {
        float cap = min(0.25f,
            max(4.0f * P.collisionMargin,
                3.0f * min(radiusA, radiusB)));
        result += P.collisionMargin + cap;
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
    device const float4* colliderShape,
    device const uint* colliderOwner,
    device const float4* colliderLocalPosition,
    device const uint* colliderShapeType,
    constant SimParams& P,
    bool emit, uint outputBase,
    device uint2* pairs)
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
        float radius = bpBVHPairRadius(
            nodeA.centerRadius.w, nodeB.centerRadius.w,
            nodeA.links.w, nodeB.links.w, P);
        if (distance_squared(centerA, centerB) > radius * radius) continue;

        bool leafA = (nodeA.links.w & 1u) != 0u;
        bool leafB = (nodeB.links.w & 1u) != 0u;
        if (leafA && leafB) {
            uint colliderA = nodeA.links.z;
            uint colliderB = nodeB.links.z;
            float3 leafCenterA = bpHierarchyColliderCenter(
                posLin, posAng, colliderOwner,
                colliderLocalPosition, colliderA);
            float3 leafCenterB = bpHierarchyColliderCenter(
                posLin, posAng, colliderOwner,
                colliderLocalPosition, colliderB);
            float leafRadius = bpHierarchyLeafPairRadius(
                colliderShape[colliderA].w, colliderShape[colliderB].w,
                colliderShapeType[colliderA],
                colliderShapeType[colliderB], P);
            if (distance_squared(leafCenterA, leafCenterB)
                    > leafRadius * leafRadius) continue;
            uint slot = outputBase + count;
            if (emit && slot < P.maxPairs) {
                pairs[slot] = uint2(
                    min(colliderA, colliderB), max(colliderA, colliderB));
            }
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
    device const float4* colliderShape [[buffer(8)]],
    device const uint* colliderOwner [[buffer(9)]],
    device const float4* colliderLocalPosition [[buffer(10)]],
    device const uint* colliderShapeType [[buffer(11)]],
    device uint2* pairs [[buffer(12)]],
    constant SimParams& P [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= P.maxPairs) return;
    uint proxyCount = atomic_load_explicit(
        &counters[CTR_PAIRS], memory_order_relaxed);
    pairCounts[gid] = gid < proxyCount ? bpExpandHierarchyPair(
        proxyPairs[gid], proxyOwner, proxyRoot, nodes, posLin, posAng,
        colliderShape, colliderOwner, colliderLocalPosition,
        colliderShapeType, P, false, 0u, pairs) : 0u;
}

kernel void bp_emit_hierarchy_pairs(
    device const uint2* proxyPairs [[buffer(0)]],
    device const atomic_uint* counters [[buffer(1)]],
    device const uint* pairStarts [[buffer(2)]],
    device const uint* proxyOwner [[buffer(3)]],
    device const uint* proxyRoot [[buffer(4)]],
    device const ColliderBVHNodeGPU* nodes [[buffer(5)]],
    device const float4* posLin [[buffer(6)]],
    device const float4* posAng [[buffer(7)]],
    device const float4* colliderShape [[buffer(8)]],
    device const uint* colliderOwner [[buffer(9)]],
    device const float4* colliderLocalPosition [[buffer(10)]],
    device const uint* colliderShapeType [[buffer(11)]],
    device uint2* pairs [[buffer(12)]],
    constant SimParams& P [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= P.maxPairs) return;
    uint proxyCount = atomic_load_explicit(
        &counters[CTR_PAIRS], memory_order_relaxed);
    if (gid >= proxyCount) return;
    (void)bpExpandHierarchyPair(
        proxyPairs[gid], proxyOwner, proxyRoot, nodes, posLin, posAng,
        colliderShape, colliderOwner, colliderLocalPosition,
        colliderShapeType, P, true, pairStarts[gid], pairs);
}

kernel void bp_finalize_hierarchy_pairs(
    device const uint* pairCounts [[buffer(0)]],
    device const uint* pairStarts [[buffer(1)]],
    device atomic_uint* counters [[buffer(2)]],
    device uint* dispatchArgs [[buffer(3)]],
    constant SimParams& P [[buffer(4)]])
{
    uint rawProxyCount = atomic_load_explicit(
        &counters[CTR_PAIR_CANDIDATES], memory_order_relaxed);
    uint n = pairStarts[P.maxPairs - 1u] + pairCounts[P.maxPairs - 1u];
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
