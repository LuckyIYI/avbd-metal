#include <metal_stdlib>
using namespace metal;

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
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    uint total = P.numJoints + P.numSprings + numPairs;
    if (gid >= total) return;

    uint a = WORLD_BODY, b = WORLD_BODY;
    if (gid < P.numJoints) {
        if (joints[gid].header.z != 0) return;  // broken
        a = joints[gid].header.x;
        b = joints[gid].header.y;
    } else if (gid < P.numJoints + P.numSprings) {
        uint s = gid - P.numJoints;
        a = springs[s].header.x;
        b = springs[s].header.y;
    } else {
        uint m = gid - P.numJoints - P.numSprings;
        if (manifolds[m].header.z == 0) return; // inactive
        a = manifolds[m].header.x;
        b = manifolds[m].header.y;
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
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    uint total = P.numJoints + P.numSprings + numPairs;
    if (gid >= total) return;

    uint a = WORLD_BODY, b = WORLD_BODY;
    uint entry = 0;
    if (gid < P.numJoints) {
        if (joints[gid].header.z != 0) return;
        a = joints[gid].header.x;
        b = joints[gid].header.y;
        entry = (FK_JOINT << ADJ_KIND_SHIFT) | gid;
    } else if (gid < P.numJoints + P.numSprings) {
        uint s = gid - P.numJoints;
        a = springs[s].header.x;
        b = springs[s].header.y;
        entry = (FK_SPRING << ADJ_KIND_SHIFT) | s;
    } else {
        uint m = gid - P.numJoints - P.numSprings;
        if (manifolds[m].header.z == 0) return;
        a = manifolds[m].header.x;
        b = manifolds[m].header.y;
        entry = (FK_MANIFOLD << ADJ_KIND_SHIFT) | m;
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
    if (kind == FK_JOINT) { a = joints[idx].header.x; b = joints[idx].header.y; }
    else if (kind == FK_SPRING) { a = springs[idx].header.x; b = springs[idx].header.y; }
    else { a = manifolds[idx].header.x; b = manifolds[idx].header.y; }
    return a == self ? b : a;
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
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numBodies) return;
    if (posLin[gid].w <= 0.0f) { colorsOut[gid] = 0; return; }

    uint myColor = colorsIn[gid];
    uint maskLo = 0, maskHi = 0;        // colors of smaller-index neighbors
    uint allLo = 0, allHi = 0;          // colors of all dynamic neighbors
    bool conflict = false;

    uint s = adjStart[gid], e = s + adjCount[gid];
    for (uint k = s; k < e; k++) {
        uint nb = otherBody(joints, springs, manifolds, adjList[k], gid);
        if (nb == WORLD_BODY || posLin[nb].w <= 0.0f) continue;
        uint nc = colorsIn[nb];
        uint lo = nc < 32 ? (1u << nc) : 0;
        uint hi = (nc >= 32 && nc < 64) ? (1u << (nc - 32)) : 0;
        allLo |= lo; allHi |= hi;
        if (nb < gid) {
            maskLo |= lo; maskHi |= hi;
            if (nc == myColor) conflict = true;
        }
    }

    if (conflict) {
        // smallest free color among smaller-index neighbors
        uint freeLo = ~maskLo;
        uint newColor;
        if (freeLo != 0) newColor = ctz(freeLo);
        else {
            uint freeHi = ~maskHi;
            newColor = freeHi != 0 ? 32 + ctz(freeHi) : (MAX_COLORS - 1);
        }
        colorsOut[gid] = min(newColor, uint(MAX_COLORS - 1));
        atomic_fetch_or_explicit(&changedFlag[0], 1u, memory_order_relaxed);
    } else {
        // Compaction: take the smallest color free w.r.t. ALL dynamic
        // neighbors when it is below the current one. Keeps the palette
        // dense so the solver loop dispatches few colors; any transient
        // conflict is fixed by the smaller-index rule next round.
        uint freeLo = ~allLo;
        uint best = freeLo != 0 ? ctz(freeLo)
                  : (~allHi != 0 ? 32 + ctz(~allHi) : myColor);
        colorsOut[gid] = best < myColor ? best : myColor;
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
    uint tid                        [[thread_position_in_threadgroup]])
{
    threadgroup uint counts[MAX_COLORS];
    if (tid < MAX_COLORS) {
        counts[tid] = atomic_load_explicit(&counters[CTR_COLOR_BASE + tid], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint run = 0;
        for (uint c = 0; c < MAX_COLORS; c++) {
            colorStart[c] = run;
            run += counts[c];
        }
        colorStart[MAX_COLORS] = run;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < MAX_COLORS) {
        dispatchArgs[tid * 3 + 0] = (counts[tid] + 63) / 64;
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
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numJoints) return;
    device JointGPU& j = joints[gid];
    if (j.header.z != 0) return;    // broken

    uint a = j.header.x, b = j.header.y;
    float3 pA = a == WORLD_BODY ? j.rA.xyz : xform(posLin[a].xyz, posAng[a], j.rA.xyz);
    float3 pB = xform(posLin[b].xyz, posAng[b], j.rB.xyz);
    float4 qA = a == WORLD_BODY ? float4(0,0,0,1) : posAng[a];
    float torqueArm = j.C0Lin.w;

    j.C0Lin = float4(pA - pB, torqueArm);
    j.C0Ang = float4(q_sub(qA, posAng[b]) * torqueArm, j.C0Ang.w);

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
    if (mass > 0.0f) inertial += float3(0, 0, P.gravity) * (dt * dt);
    inertLin[gid] = float4(inertial, 0);
    inertAng[gid] = q_addw(pa, va * dt);

    // Adaptive warmstart (original VBD)
    float3 accel = (vl - prevVelLin[gid].xyz) / dt;
    float gsign = P.gravity < 0.0f ? -1.0f : (P.gravity > 0.0f ? 1.0f : 0.0f);
    float accelWeight = clamp(accel.z * gsign / fabs(P.gravity), 0.0f, 1.0f);
    if (!isfinite(accelWeight)) accelWeight = 0.0f;

    initLin[gid] = float4(pl.xyz, 0);
    initAng[gid] = pa;
    if (mass > 0.0f) {
        posLin[gid] = float4(pl.xyz + vl * dt + float3(0, 0, P.gravity) * (accelWeight * dt * dt), mass);
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
                       float alpha, thread PrimalAccum& acc)
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
        float3 C = q_sub(qA, posAng[b]) * torqueArm;
        if (j.header.w & 2) C -= j.C0Ang.xyz * alpha;

        float3 F = penAng * C + j.lambdaAng.xyz;
        float s = (isA ? 1.0f : -1.0f) * torqueArm;
        acc.lhsAng = m3_add(acc.lhsAng, m3_diag(penAng * (s * s)));
        acc.rhsAng += F * s;
    }
}

inline void stampSpring(device const SpringGPU& sp, uint self,
                        device const float4* posLin, device const float4* posAng,
                        thread PrimalAccum& acc)
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
    float f = stiffness * (dLen - rest);

    bool isA = self == a;
    float3 rW = isA ? q_rotate(posAng[a], sp.rA.xyz) : q_rotate(posAng[b], sp.rB.xyz);
    float3 jLin = isA ? n : -n;
    float3 jAng = isA ? cross(rW, n) : -cross(rW, n);

    acc.lhsLin = m3_add(acc.lhsLin, m3_scale(m3_outer(jLin, jLin), stiffness));
    acc.lhsAng = m3_add(acc.lhsAng, m3_scale(m3_outer(jAng, jAng), stiffness));
    acc.lhsCross = m3_add(acc.lhsCross, m3_scale(m3_outer(jAng, jLin), stiffness));
    acc.rhsLin += jLin * f;
    acc.rhsAng += jAng * f;
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
    float friction = m.basisN.w;

    device const ContactGPU& c = m.contacts[ci];
    float3 C = c.C0.xyz * (1.0f - alpha);
    C += float3(dot(nrm, dqALin), dot(t1, dqALin), dot(t2, dqALin));
    C -= float3(dot(nrm, dqBLin), dot(t1, dqBLin), dot(t2, dqBLin));
    C += float3(dot(cross(rAW, nrm), dqAAng), dot(cross(rAW, t1), dqAAng), dot(cross(rAW, t2), dqAAng));
    C -= float3(dot(cross(rBW, nrm), dqBAng), dot(cross(rBW, t1), dqBAng), dot(cross(rBW, t2), dqBAng));

    float3 F = c.penalty.xyz * C + c.lambda.xyz;
    F.x = min(F.x, 0.0f);
    bounds = fabs(F.x) * friction;
    frictionScale = length(F.yz);
    if (frictionScale > bounds && frictionScale > 0.0f) {
        F.yz *= bounds / frictionScale;
    }
    Cout = C;
    return F;
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
    uint tid                        [[thread_position_in_grid]])
{
    uint s = colorStart[colorIndex];
    uint e = colorStart[colorIndex + 1];
    if (s + tid >= e) return;
    uint body = colorList[s + tid];

    float4 pl = posLin[body];
    float mass = pl.w;
    float dt2 = P.dt * P.dt;
    float3 moment = props[body].xyz;

    PrimalAccum acc;
    acc.lhsLin = m3_diag(float3(mass / dt2));
    acc.lhsAng = m3_diag(moment / dt2);
    acc.lhsCross = m3_zero();
    acc.rhsLin = (pl.xyz - inertLin[body].xyz) * (mass / dt2);
    acc.rhsAng = q_sub(posAng[body], inertAng[body]) * (moment / dt2);

    uint as = adjStart[body], ae = as + adjCount[body];
    for (uint k = as; k < ae; k++) {
        uint entry = adjList[k];
        uint kind = entry >> ADJ_KIND_SHIFT;
        uint idx = entry & ADJ_INDEX_MASK;
        if (kind == FK_JOINT) {
            stampJoint(joints[idx], body, posLin, posAng, P.alpha, acc);
        } else if (kind == FK_SPRING) {
            stampSpring(springs[idx], body, posLin, posAng, acc);
        } else {
            stampManifold(manifolds[idx], body, posLin, posAng, initLin, initAng, P.alpha, acc);
        }
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

kernel void dual_joints(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device JointGPU* joints         [[buffer(2)]],
    constant SimParams& P           [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numJoints) return;
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
        float3 C = q_sub(qA, posAng[b]) * torqueArm;
        if (j.header.w & 2) {
            C -= j.C0Ang.xyz * P.alpha;
            j.lambdaAng.xyz = clamp(penAng * C + j.lambdaAng.xyz,
                                    -P.lambdaMax, P.lambdaMax);
        }
        float cap = min(stiffAng, PENALTY_MAX);
        j.penaltyAng.xyz = min(penAng + fabs(C) * P.betaAng, cap);
    }

    // Fracture (flagged joints only; avoids inf comparisons under fast math)
    float fracture = j.C0Ang.w;
    float3 la = j.lambdaAng.xyz;
    if ((j.header.w & 4) && dot(la, la) > fracture * fracture) {
        j.penaltyLin = float4(0);
        j.penaltyAng = float4(0);
        j.lambdaLin = float4(0);
        j.lambdaAng = float4(0);
        j.header.z = 1;
    }
}

kernel void dual_manifolds(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device const float4* initLin    [[buffer(2)]],
    device const float4* initAng    [[buffer(3)]],
    device ManifoldGPU* manifolds   [[buffer(4)]],
    device const atomic_uint* counters [[buffer(5)]],
    constant SimParams& P           [[buffer(6)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    if (gid >= numPairs) return;
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

    for (uint i = 0; i < n; i++) {
        float3 rAW = sphA ? m.contacts[i].rA.xyz : q_rotate(posAng[a], m.contacts[i].rA.xyz);
        float3 rBW = sphB ? m.contacts[i].rB.xyz : q_rotate(posAng[b], m.contacts[i].rB.xyz);

        float3 C; float fs, bnd;
        float3 F = contactForceC(m, i, dqALin, dqAAng, dqBLin, dqBAng, rAW, rBW, P.alpha, C, fs, bnd);

        // bound the dual (paper Sec 4): conflicting contacts otherwise ramp
        // each other's lambda without limit until the stored force explodes
        F.x = max(F.x, -P.lambdaMax);
        m.contacts[i].lambda = float4(F, 0);

        float3 pen = m.contacts[i].penalty.xyz;
        if (F.x < 0.0f) {
            pen.x = min(pen.x + P.betaLin * fabs(C.x), PENALTY_MAX);
        }
        if (fs <= bnd) {
            pen.y = min(pen.y + P.betaLin * fabs(C.y), PENALTY_MAX_T);
            pen.z = min(pen.z + P.betaLin * fabs(C.z), PENALTY_MAX_T);
            m.contacts[i].rB.w = length(C.yz) < STICK_THRESH ? 1.0f : 0.0f;
        }
        m.contacts[i].penalty = float4(pen, 0);
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
    uint gid                        [[thread_position_in_grid]])
{
    if (gid >= P.numBodies) return;
    prevVelLin[gid] = velLin[gid];
    if (posLin[gid].w > 0.0f) {
        float3 v = (posLin[gid].xyz - initLin[gid].xyz) / P.dt;
        // Safety clamp: prevents tunneling of violently flung bodies
        float s2 = dot(v, v);
        if (s2 > P.maxSpeed * P.maxSpeed) v *= P.maxSpeed * rsqrt(s2);
        velLin[gid] = float4(v, 0);
        velAng[gid] = float4(q_sub(posAng[gid], initAng[gid]) / P.dt, 0);
    }
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
            float pen = dot(m.basisN.xyz, xA - xB) + COLLISION_MARGIN;
            err = max(err, max(0.0f, -pen));
        }
    } else {
        return;
    }
    atomic_fetch_max_explicit(&diag[0], as_type<uint>(err), memory_order_relaxed);
}
