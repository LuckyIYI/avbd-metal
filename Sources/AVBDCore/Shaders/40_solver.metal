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
    device const TetGPU* tets       [[buffer(7)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    uint total = P.numJoints + P.numSprings + numPairs + P.numTets;
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
    } else if (gid < P.numJoints + P.numSprings + numPairs) {
        uint m = gid - P.numJoints - P.numSprings;
        if (manifolds[m].header.z == 0) return; // inactive
        a = manifolds[m].header.x;
        b = manifolds[m].header.y;
    } else {
        // tet: one adjacency entry per vertex
        uint t = gid - P.numJoints - P.numSprings - numPairs;
        uint4 ids = tets[t].ids;
        for (uint k = 0; k < 4; k++) {
            uint v = ids[k];
            if (bodyDynamic(posLin, v))
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
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    uint total = P.numJoints + P.numSprings + numPairs + P.numTets;
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
    } else if (gid < P.numJoints + P.numSprings + numPairs) {
        uint m = gid - P.numJoints - P.numSprings;
        if (manifolds[m].header.z == 0) return;
        a = manifolds[m].header.x;
        b = manifolds[m].header.y;
        entry = (FK_MANIFOLD << ADJ_KIND_SHIFT) | m;
    } else {
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

// Accumulate the neighbor-color masks for one adjacency entry; tets
// conflict with all 3 other vertices.
inline void neighborColors(device const JointGPU* joints,
                           device const SpringGPU* springs,
                           device const ManifoldGPU* manifolds,
                           device const TetGPU* tets,
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
        neighborColors(joints, springs, manifolds, tets, posLin, colorsIn,
                       adjList[k], gid, myColor,
                       maskLo, maskHi, allLo, allHi, conflict);
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
    device SpringGPU* springs       [[buffer(4)]],
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
        sp.dual.z = length(pA - pB) - sp.rB.w;                 // C0
        sp.dual.x *= P.alpha * P.gamma;                        // lambda
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
    } else {
        c0a = q_sub(q_mul(qA, j.restRel), posAng[b]) * torqueArm;
    }
    j.C0Ang = float4(c0a, j.C0Ang.w);
    if (j.motor.w > 0.0f) {
        j.motor.z *= P.alpha * P.gamma;                       // lambda
        j.motor.w = clamp(j.motor.w * P.gamma, 50.0f, 2.0e5f); // penalty
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
            acc.lhsAng = m3_add(acc.lhsAng, m3_diag(penAng * (s * s)));
            acc.rhsAng += F * (-s);

            // MOTOR (servo): drive the twist angle about the hinge axis
            // toward the target, with bounded lambda = torque limit (the
            // paper's bounded-multiplier machinery, like friction).
            // A pure position spring oscillates — real actuators are
            // damped, so add a dissipative term on the twist VELOCITY
            // (measured against the start-of-step pose, implicit in dt).
            if (j.motor.w > 0.0f || j.limits.x < j.limits.y) {
                float4 r = q_mul(q_inv(q_mul(qA, j.restRel)), posAng[b]);
                if (r.w < 0.0f) r = -r;
                float twist = 2.0f * atan2(dot(r.xyz, j.hingeAxis.xyz), r.w);

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
                if (j.motor.w <= 0.0f) { C = C; }
                else {
                float Cm = twist - j.motor.x;
                // wrap to [-pi, pi]
                Cm = Cm - 6.2831853f * floor((Cm + 3.14159265f) / 6.2831853f);

                // twist at start of step (for the damping velocity)
                float4 qA0 = (a == WORLD_BODY) ? float4(0,0,0,1) : initAng[a];
                float4 r0 = q_mul(q_inv(q_mul(qA0, j.restRel)), initAng[b]);
                if (r0.w < 0.0f) r0 = -r0;
                float twist0 = 2.0f * atan2(dot(r0.xyz, j.hingeAxis.xyz), r0.w);
                float dTwist = twist - twist0;
                dTwist = dTwist - 6.2831853f * floor((dTwist + 3.14159265f) / 6.2831853f);

                // damping coefficient: ~2 sqrt(k I) would be critical; a
                // fixed fraction of stiffness over the step works well here
                float cD = 0.30f * j.motor.w * dt * 60.0f;   // scale-stable
                float Fm = clamp(j.motor.w * Cm + j.motor.z + cD * dTwist,
                                 -j.motor.y, j.motor.y);
                float sm = (self == a) ? -1.0f : 1.0f;
                float3 axW = aB;
                acc.lhsAng = m3_add(acc.lhsAng,
                                    m3_scale(m3_outer(axW, axW), j.motor.w + cD));
                acc.rhsAng += axW * (Fm * sm);
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
        // hard rod (AL): inextensible distance element — the same dual
        // machinery as hard joints, 1-D along the current axis
        float C = dLen - rest - sp.dual.z * alpha;
        stiffness = sp.dual.y;                 // ramped penalty
        f = stiffness * C + sp.dual.x;         // + lambda
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
    uint body)
{

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
            stampJoint(joints[idx], body, posLin, posAng, initAng, P.alpha, P.dt, acc);
        } else if (kind == FK_SPRING) {
            stampSpring(springs[idx], body, posLin, posAng, acc, P.alpha);
        } else if (kind == FK_TET) {
            stampTet(tets[idx], body, posLin, acc);
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
    uint tid                        [[thread_position_in_grid]])
{
    uint s = colorStart[colorIndex];
    uint e = colorStart[colorIndex + 1];
    if (s + tid >= e) return;
    primal_one(posLin, posAng, initLin, initAng, inertLin, inertAng, props,
               joints, springs, manifolds, adjStart, adjCount, adjList,
               P, shape, tets, colorList[s + tid]);
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
            if (j.motor.w > 0.0f) {
                float4 qAr = a == WORLD_BODY ? j.restRel : q_mul(posAng[a], j.restRel);
                float4 r = q_mul(q_inv(qAr), posAng[b]);
                if (r.w < 0.0f) r = -r;
                float twist = 2.0f * atan2(dot(r.xyz, j.hingeAxis.xyz), r.w);
                float Cm = twist - j.motor.x;
                Cm = Cm - 6.2831853f * floor((Cm + 3.14159265f) / 6.2831853f);
                j.motor.z = clamp(j.motor.w * Cm + j.motor.z,
                                  -j.motor.y, j.motor.y);
                j.motor.w = min(j.motor.w + fabs(Cm) * P.betaAng, 2.0e5f);
            }
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
        sp.dual.x = clamp(sp.dual.y * C + sp.dual.x, -P.lambdaMax, P.lambdaMax);
        sp.dual.y = min(sp.dual.y + fabs(C) * P.betaLin,
                        min(sp.rA.w, PENALTY_MAX));
        return;
    }
    uint mi = si - P.numSprings;
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    if (mi >= numPairs) return;
    if (manifolds[mi].header.z == 0) return;
    dual_manifold_one(posLin, posAng, initLin, initAng, manifolds, P, mi);
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
    uint tid                        [[thread_position_in_threadgroup]],
    uint tgSize                     [[threads_per_threadgroup]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    uint dualTotal = P.numJoints + P.numSprings + numPairs;
    for (uint iter = 0; iter < P.iterations; iter++) {
        for (uint c = 0; c < MAX_COLORS; c++) {
            uint s0 = colorStart[c];
            uint e0 = colorStart[c + 1];
            if (s0 == e0) continue;
            for (uint i = s0 + tid; i < e0; i += tgSize) {
                primal_one(posLin, posAng, initLin, initAng, inertLin, inertAng,
                           props, joints, springs, manifolds,
                           adjStart, adjCount, adjList, P, shape, tets,
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
                    sp.dual.x = clamp(sp.dual.y * C + sp.dual.x,
                                      -P.lambdaMax, P.lambdaMax);
                    sp.dual.y = min(sp.dual.y + fabs(C) * P.betaLin,
                                    min(sp.rA.w, PENALTY_MAX));
                }
            } else {
                uint mi = g - P.numJoints - P.numSprings;
                if (manifolds[mi].header.z != 0)
                    dual_manifold_one(posLin, posAng, initLin, initAng,
                                      manifolds, P, mi);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
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
