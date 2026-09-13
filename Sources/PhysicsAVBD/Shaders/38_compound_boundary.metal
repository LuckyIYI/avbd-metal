// Opt-in union-boundary filtering for authored rigid convex compounds.
// Probe in the contact's outward direction: a neighbour containing this point
// proves the chosen normal belongs to an internal partition, not the union.
inline bool compoundBuried(uint collider, float3 bodyPoint,
    device const uint* neighbours, device const float4* localPosition,
    device const float4* localRotation, device const uint* assetIDs,
    device const ConvexHullGPU* hulls, device const ConvexFaceGPU* faces)
{
    for (uint k=neighbours[collider]; k<neighbours[collider+1]; ++k) {
        uint other=neighbours[k];
        float3 p=q_rotate(q_conj(localRotation[other]),bodyPoint-localPosition[other].xyz);
        ConvexHullGPU hull=hulls[assetIDs[other]];
        bool inside=true;
        for (uint f=0;f<hull.verticesFaces.w;++f) {
            float4 plane=faces[hull.verticesFaces.z+f].plane;
            if (dot(plane.xyz,p)>plane.w) {inside=false;break;}
        }
        if (inside) return true;
    }
    return false;
}

kernel void filter_compound_internal_contacts(
    device ManifoldGPU* manifolds [[buffer(0)]],
    device uint2* features [[buffer(1)]],
    device const float4* positions [[buffer(2)]],
    device const float4* rotations [[buffer(3)]],
    device const uint* neighbours [[buffer(4)]],
    device const float4* localPosition [[buffer(5)]],
    device const float4* localRotation [[buffer(6)]],
    device const uint* assetIDs [[buffer(7)]],
    device const ConvexHullGPU* hulls [[buffer(8)]],
    device const ConvexFaceGPU* faces [[buffer(9)]],
    device const atomic_uint* counters [[buffer(10)]],
    constant float& probe [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid>=atomic_load_explicit(&counters[CTR_PAIRS],memory_order_relaxed)) return;
    device ManifoldGPU& m=manifolds[gid];
    if (m.header.z==0u) return;
    uint count=0u;
    float3 outwardA=q_rotate(q_conj(rotations[m.header.x]),-m.basisN.xyz)*probe;
    float3 outwardB=q_rotate(q_conj(rotations[m.header.y]), m.basisN.xyz)*probe;
    for (uint j=0;j<m.header.z;++j) {
        ContactGPU c=m.contacts[j];
        // Filter only hull compounds enabled by the host's neighbour table.
        bool hiddenA=compoundBuried(m.colliderPair.x,c.rA.xyz+outwardA,
            neighbours,localPosition,localRotation,assetIDs,hulls,faces);
        bool hiddenB=compoundBuried(m.colliderPair.y,c.rB.xyz+outwardB,
            neighbours,localPosition,localRotation,assetIDs,hulls,faces);
        if (hiddenA || hiddenB) continue;
        m.contacts[count]=c;
        features[gid*MAX_CONTACTS+count]=features[gid*MAX_CONTACTS+j];
        ++count;
    }
    m.header.z=count;
}
