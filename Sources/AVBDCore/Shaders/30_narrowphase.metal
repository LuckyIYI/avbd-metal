#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Narrowphase: OBB-OBB SAT with reference-face clipping and edge-edge
// closest points (port of avbd-demo3d collide.cpp). One thread per pair.
// Warm-starts each contact from the previous frame's manifold (pair hash
// map lookup + feature key match), then initializes C0 / lambda / penalty
// exactly as the AVBD initialize step requires.
// ============================================================================

#define NP_MAX_POLY 16
#define SAT_EPS 1.0e-6f
#define PLANE_EPS 1.0e-5f
#define MERGE_DIST_SQ 1.0e-6f

struct NPBox {
    float3 center;
    float3 half3_;
    float3 ax0, ax1, ax2;
};

inline float3 npAxis(thread const NPBox& b, int i) {
    return i == 0 ? b.ax0 : (i == 1 ? b.ax1 : b.ax2);
}
inline float npHalf(thread const NPBox& b, int i) {
    return i == 0 ? b.half3_.x : (i == 1 ? b.half3_.y : b.half3_.z);
}

inline float3 npSupport(thread const NPBox& box, float3 dir) {
    float sx = dot(dir, box.ax0) >= 0.0f ? 1.0f : -1.0f;
    float sy = dot(dir, box.ax1) >= 0.0f ? 1.0f : -1.0f;
    float sz = dot(dir, box.ax2) >= 0.0f ? 1.0f : -1.0f;
    return box.center + box.ax0 * (box.half3_.x * sx)
                      + box.ax1 * (box.half3_.y * sy)
                      + box.ax2 * (box.half3_.z * sz);
}

struct NPAxisBest {
    int type;       // 0 faceA, 1 faceB, 2 edge
    int indexA;
    int indexB;
    float separation;
    float3 normalAB;
    bool valid;
};

inline bool npTestAxis(thread const NPBox& A, thread const NPBox& B, float3 delta,
                       float3 axis, int type, int ia, int ib,
                       thread NPAxisBest& best) {
    float lenSq = dot(axis, axis);
    if (lenSq < SAT_EPS) return true;

    float3 n = axis * rsqrt(lenSq);
    if (dot(n, delta) < 0.0f) n = -n;

    float distance = fabs(dot(delta, n));
    float rA = A.half3_.x * fabs(dot(n, A.ax0)) + A.half3_.y * fabs(dot(n, A.ax1)) + A.half3_.z * fabs(dot(n, A.ax2));
    float rB = B.half3_.x * fabs(dot(n, B.ax0)) + B.half3_.y * fabs(dot(n, B.ax1)) + B.half3_.z * fabs(dot(n, B.ax2));

    float separation = distance - (rA + rB);
    if (separation > 0.0f) return false;

    if (!best.valid || separation > best.separation) {
        best.valid = true;
        best.type = type;
        best.indexA = ia;
        best.indexB = ib;
        best.separation = separation;
        best.normalAB = n;
    }
    return true;
}

inline void npFaceAxes(thread const NPBox& box, int axisIndex,
                       thread float3& u, thread float3& v,
                       thread float& eu, thread float& ev) {
    if (axisIndex == 0) { u = box.ax1; v = box.ax2; eu = box.half3_.y; ev = box.half3_.z; }
    else if (axisIndex == 1) { u = box.ax0; v = box.ax2; eu = box.half3_.x; ev = box.half3_.z; }
    else { u = box.ax0; v = box.ax1; eu = box.half3_.x; ev = box.half3_.y; }
}

inline int npClip(thread float3* inV, int inCount, float3 n, float offset, thread float3* outV) {
    if (inCount <= 0) return 0;
    int outCount = 0;
    float3 a = inV[inCount - 1];
    float da = dot(n, a) - offset;
    for (int i = 0; i < inCount; i++) {
        float3 b = inV[i];
        float db = dot(n, b) - offset;
        bool aIn = da <= PLANE_EPS;
        bool bIn = db <= PLANE_EPS;
        if (aIn != bIn) {
            float t = 0.0f;
            float denom = da - db;
            if (fabs(denom) > SAT_EPS) t = clamp(da / denom, 0.0f, 1.0f);
            if (outCount < NP_MAX_POLY) outV[outCount++] = a + (b - a) * t;
        }
        if (bIn && outCount < NP_MAX_POLY) outV[outCount++] = b;
        a = b; da = db;
    }
    return outCount;
}

inline void npSupportEdge(thread const NPBox& box, int axisIndex, float3 dir,
                          thread float3& eA, thread float3& eB) {
    int a1 = (axisIndex + 1) % 3;
    int a2 = (axisIndex + 2) % 3;
    float s1 = dot(dir, npAxis(box, a1)) >= 0.0f ? 1.0f : -1.0f;
    float s2 = dot(dir, npAxis(box, a2)) >= 0.0f ? 1.0f : -1.0f;
    float3 c = box.center + npAxis(box, a1) * (npHalf(box, a1) * s1)
                          + npAxis(box, a2) * (npHalf(box, a2) * s2);
    eA = c - npAxis(box, axisIndex) * npHalf(box, axisIndex);
    eB = c + npAxis(box, axisIndex) * npHalf(box, axisIndex);
}

inline void npClosestSegSeg(float3 p0, float3 p1, float3 q0, float3 q1,
                            thread float3& c0, thread float3& c1) {
    float3 d1 = p1 - p0;
    float3 d2 = q1 - q0;
    float3 r = p0 - q0;
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);
    float s = 0.0f, t = 0.0f;

    if (a <= SAT_EPS && e <= SAT_EPS) { c0 = p0; c1 = q0; return; }
    if (a <= SAT_EPS) {
        t = clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = dot(d1, r);
        if (e <= SAT_EPS) {
            s = clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;
            if (fabs(denom) > SAT_EPS) s = clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            t = (b * s + f) / e;
            if (t < 0.0f) { t = 0.0f; s = clamp(-c / a, 0.0f, 1.0f); }
            else if (t > 1.0f) { t = 1.0f; s = clamp((b - c) / a, 0.0f, 1.0f); }
        }
    }
    c0 = p0 + d1 * s;
    c1 = q0 + d2 * t;
}

// Local contact scratch (feature key + world points)
struct NPContact {
    float3 xA;
    float3 xB;
    uint feature;
};

inline bool npAddContact(thread NPContact* contacts, thread int& count,
                         thread float3* midpoints,
                         float3 xA, float3 xB, uint feature) {
    float3 mid = (xA + xB) * 0.5f;
    for (int i = 0; i < count; i++) {
        if (distance_squared(mid, midpoints[i]) < MERGE_DIST_SQ) return false;
    }
    if (count >= MAX_CONTACTS) return false;
    contacts[count].xA = xA;
    contacts[count].xB = xB;
    contacts[count].feature = feature;
    midpoints[count] = mid;
    count++;
    return true;
}

kernel void np_collide(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device const float4* shape      [[buffer(2)]],   // xyz size, w radius
    device const float4* props      [[buffer(3)]],   // xyz moment, w friction
    device const uint2* pairs       [[buffer(4)]],
    device const atomic_uint* counters [[buffer(5)]],
    device ManifoldGPU* manifolds   [[buffer(6)]],
    device const ManifoldGPU* prevManifolds [[buffer(7)]],
    device const atomic_uint* mapKeyA [[buffer(8)]],
    device const uint* mapKeyB      [[buffer(9)]],
    device const uint* mapVal       [[buffer(10)]],
    constant SimParams& P           [[buffer(11)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    if (gid >= numPairs) return;

    uint2 pair = pairs[gid];
    uint ia = pair.x, ib = pair.y;

    float4 pA4 = posLin[ia];
    float4 pB4 = posLin[ib];
    float4 qA = posAng[ia];
    float4 qB = posAng[ib];

    NPBox A, B;
    A.center = pA4.xyz; A.half3_ = shape[ia].xyz * 0.5f;
    A.ax0 = q_rotate(qA, float3(1,0,0)); A.ax1 = q_rotate(qA, float3(0,1,0)); A.ax2 = q_rotate(qA, float3(0,0,1));
    B.center = pB4.xyz; B.half3_ = shape[ib].xyz * 0.5f;
    B.ax0 = q_rotate(qB, float3(1,0,0)); B.ax1 = q_rotate(qB, float3(0,1,0)); B.ax2 = q_rotate(qB, float3(0,0,1));

    float3 delta = B.center - A.center;

    device ManifoldGPU& outM = manifolds[gid];

    // --- Sphere branches (shape.w < 0 marks spheres) ---
    bool sphA = shape[ia].w < 0.0f;
    bool sphB = shape[ib].w < 0.0f;
    NPContact contacts[MAX_CONTACTS];
    int count = 0;
    float3 nrm;     // points from B toward A

    if (sphA || sphB) {
        float rA = shape[ia].x * 0.5f;
        float rB = shape[ib].x * 0.5f;
        if (sphA && sphB) {
            float3 d = A.center - B.center;
            float dist = length(d);
            if (dist > rA + rB + COLLISION_MARGIN) {
                outM.header = uint4(ia, ib, 0, 0);
                return;
            }
            nrm = dist > 1e-9f ? d / dist : float3(0, 0, 1);
            contacts[0].xA = A.center - nrm * rA;
            contacts[0].xB = B.center + nrm * rB;
            contacts[0].feature = 0;
            count = 1;
        } else {
            // one sphere, one box
            bool sIsA = sphA;
            float r = sIsA ? rA : rB;
            thread const NPBox& box = sIsA ? B : A;
            float4 qBox = sIsA ? qB : qA;
            float3 sphereC = sIsA ? A.center : B.center;

            float4 qc = q_conj(qBox);
            float3 local = q_rotate(qc, sphereC - box.center);
            float3 half3_ = box.half3_;
            float3 clamped = clamp(local, -half3_, half3_);

            float3 nLocal;
            float3 qLocal;
            if (all(clamped == local)) {
                float3 pen = half3_ - fabs(local);
                int axis = 0;
                if (pen.y < pen[axis]) axis = 1;
                if (pen.z < pen[axis]) axis = 2;
                float3 n = float3(0);
                n[axis] = local[axis] >= 0.0f ? 1.0f : -1.0f;
                nLocal = n;
                qLocal = local;
                qLocal[axis] = n[axis] * half3_[axis];
            } else {
                float3 d = local - clamped;
                float dist = length(d);
                if (dist > r + COLLISION_MARGIN) {
                    outM.header = uint4(ia, ib, 0, 0);
                    return;
                }
                nLocal = d / max(dist, 1e-9f);
                qLocal = clamped;
            }

            float3 nW = q_rotate(qBox, nLocal);      // box -> sphere
            float3 qW = q_rotate(qBox, qLocal) + box.center;
            float3 xSphere = sphereC - nW * r;

            nrm = sIsA ? nW : -nW;                   // B -> A
            contacts[0].xA = sIsA ? xSphere : qW;
            contacts[0].xB = sIsA ? qW : xSphere;
            contacts[0].feature = 0;
            count = 1;
        }

        // shared tail: warm-start + write. Sphere anchors are stored as
        // WORLD-space offsets (rotation-invariant): a spinning ball would
        // swing a material anchor away from the contact mid-step, breaking
        // the normal constraint. header.w bits: 1 active, 2 A-sphere,
        // 4 B-sphere.
        float3 t1, t2;
        orthonormal(nrm, t1, t2);
        float friction = sqrt(props[ia].w * props[ib].w);
        int prevIdx = pairMapFind(mapKeyA, mapKeyB, mapVal, P.mapCapacity, ia, ib);

        uint flags = 1u | (sphA ? 2u : 0u) | (sphB ? 4u : 0u);
        outM.header = uint4(ia, ib, uint(count), flags);
        outM.basisN = float4(nrm, friction);
        outM.basisT1 = float4(t1, 0);

        float4 qAc = q_conj(qA);
        float4 qBc = q_conj(qB);
        float warm = P.alpha * P.gamma;
        for (int i = 0; i < count; i++) {
            float3 rA_ = sphA ? (contacts[i].xA - pA4.xyz)
                              : q_rotate(qAc, contacts[i].xA - pA4.xyz);
            float3 rB_ = sphB ? (contacts[i].xB - pB4.xyz)
                              : q_rotate(qBc, contacts[i].xB - pB4.xyz);
            float3 lambda = float3(0);
            float3 penalty = float3(0);
            float stick = 0.0f;
            if (prevIdx >= 0) {
                device const ManifoldGPU& pm = prevManifolds[prevIdx];
                uint pn = pm.header.z;
                for (uint j = 0; j < pn; j++) {
                    if (as_type<uint>(pm.contacts[j].rA.w) == contacts[i].feature) {
                        lambda = pm.contacts[j].lambda.xyz;
                        penalty = pm.contacts[j].penalty.xyz;
                        // NOTE: no stick-anchor restoration for spheres —
                        // anchors rotate with the body, which is wrong for
                        // rolling contacts and drags the ball into the floor.
                        break;
                    }
                }
            }
            float3 xAw = sphA ? pA4.xyz + rA_ : xform(pA4.xyz, qA, rA_);
            float3 xBw = sphB ? pB4.xyz + rB_ : xform(pB4.xyz, qB, rB_);
            float3 d = xAw - xBw;
            float3 C0 = float3(dot(nrm, d) + COLLISION_MARGIN, dot(t1, d), dot(t2, d));
            lambda *= warm;
            penalty = clamp(penalty * P.gamma, PENALTY_MIN, PENALTY_MAX);
            outM.contacts[i].rA = float4(rA_, as_type<float>(contacts[i].feature));
            outM.contacts[i].rB = float4(rB_, stick);
            outM.contacts[i].C0 = float4(C0, 0);
            outM.contacts[i].lambda = float4(lambda, 0);
            outM.contacts[i].penalty = float4(penalty, 0);
        }
        return;
    }

    NPAxisBest bestFace; bestFace.valid = false; bestFace.separation = -FLT_MAX;
    NPAxisBest bestEdge; bestEdge.valid = false; bestEdge.separation = -FLT_MAX;

    bool separated = false;
    for (int i = 0; i < 3 && !separated; i++) {
        if (!npTestAxis(A, B, delta, npAxis(A, i), 0, i, -1, bestFace)) separated = true;
    }
    for (int i = 0; i < 3 && !separated; i++) {
        if (!npTestAxis(A, B, delta, npAxis(B, i), 1, -1, i, bestFace)) separated = true;
    }
    for (int i = 0; i < 3 && !separated; i++) {
        for (int j = 0; j < 3 && !separated; j++) {
            float3 axis = cross(npAxis(A, i), npAxis(B, j));
            if (!npTestAxis(A, B, delta, axis, 2, i, j, bestEdge)) separated = true;
        }
    }

    if (separated || !bestFace.valid) {
        outM.header = uint4(ia, ib, 0, 0);
        return;
    }

    NPAxisBest best = bestFace;
    if (bestEdge.valid) {
        if (0.95f * bestEdge.separation > bestFace.separation + 0.01f) best = bestEdge;
    }

    // Build contacts (arrays declared above)
    float3 midpoints[MAX_CONTACTS];

    if (best.type == 2) {
        // Edge-edge
        float3 a0, a1, b0, b1;
        npSupportEdge(A, best.indexA, best.normalAB, a0, a1);
        npSupportEdge(B, best.indexB, -best.normalAB, b0, b1);
        float3 xA, xB;
        npClosestSegSeg(a0, a1, b0, b1, xA, xB);
        uint feature = (2u << 24) | (uint(best.indexA & 0xFF) << 8) | uint(best.indexB & 0xFF);
        npAddContact(contacts, count, midpoints, xA, xB, feature);
        if (count == 0) {
            xA = npSupport(A, best.normalAB);
            xB = npSupport(B, -best.normalAB);
            npAddContact(contacts, count, midpoints, xA, xB, feature);
        }
    } else {
        // Face manifold with clipping
        bool refIsA = best.type == 0;
        int refAxis = refIsA ? best.indexA : best.indexB;
        thread const NPBox& refBox = refIsA ? A : B;
        thread const NPBox& incBox = refIsA ? B : A;
        float3 refOutward = refIsA ? best.normalAB : -best.normalAB;

        float signR = dot(refOutward, npAxis(refBox, refAxis)) >= 0.0f ? 1.0f : -1.0f;
        float3 refNormal = npAxis(refBox, refAxis) * signR;
        float3 refCenter = refBox.center + refNormal * npHalf(refBox, refAxis);
        float3 refU, refV; float refEU, refEV;
        npFaceAxes(refBox, refAxis, refU, refV, refEU, refEV);

        int incAxis = 0;
        float bestD = -FLT_MAX;
        for (int i = 0; i < 3; i++) {
            float d = fabs(dot(npAxis(incBox, i), refNormal));
            if (d > bestD) { bestD = d; incAxis = i; }
        }

        float signI = dot(npAxis(incBox, incAxis), refNormal) > 0.0f ? -1.0f : 1.0f;
        float3 incNormal = npAxis(incBox, incAxis) * signI;
        float3 incCenter = incBox.center + incNormal * npHalf(incBox, incAxis);
        float3 incU, incV; float incEU, incEV;
        npFaceAxes(incBox, incAxis, incU, incV, incEU, incEV);

        float3 poly0[NP_MAX_POLY];
        float3 poly1[NP_MAX_POLY];
        poly0[0] = incCenter + incU * incEU + incV * incEV;
        poly0[1] = incCenter - incU * incEU + incV * incEV;
        poly0[2] = incCenter - incU * incEU - incV * incEV;
        poly0[3] = incCenter + incU * incEU - incV * incEV;
        int n = 4;

        n = npClip(poly0, n, refU, dot(refU, refCenter) + refEU, poly1);
        n = npClip(poly1, n, -refU, dot(-refU, refCenter) + refEU, poly0);
        n = npClip(poly0, n, refV, dot(refV, refCenter) + refEV, poly1);
        n = npClip(poly1, n, -refV, dot(-refV, refCenter) + refEV, poly0);

        uint prefix = (refIsA ? 0u : 1u) << 24;
        prefix |= uint(refAxis & 0xFF) << 16;
        prefix |= uint(incAxis & 0xFF) << 8;

        for (int i = 0; i < n && count < MAX_CONTACTS; i++) {
            float3 pInc = poly0[i];
            float d = dot(pInc - refCenter, refNormal);
            if (d > PLANE_EPS) continue;
            float3 pRef = pInc - refNormal * d;
            float3 xA = refIsA ? pRef : pInc;
            float3 xB = refIsA ? pInc : pRef;
            npAddContact(contacts, count, midpoints, xA, xB, prefix | uint(i & 0xFF));
        }

        if (count == 0) {
            float3 xA = npSupport(A, best.normalAB);
            float3 xB = npSupport(B, -best.normalAB);
            npAddContact(contacts, count, midpoints, xA, xB, prefix);
        }
    }

    // Basis: normal points from B toward A (reference uses -normalAB)
    nrm = -best.normalAB;
    float3 t1, t2;
    orthonormal(nrm, t1, t2);

    float friction = sqrt(props[ia].w * props[ib].w);

    // Previous manifold for warm-start
    int prevIdx = pairMapFind(mapKeyA, mapKeyB, mapVal, P.mapCapacity, ia, ib);

    outM.header = uint4(ia, ib, uint(count), 1);
    outM.basisN = float4(nrm, friction);
    outM.basisT1 = float4(t1, 0);

    float4 qAc = q_conj(qA);
    float4 qBc = q_conj(qB);
    float warm = P.alpha * P.gamma;

    for (int i = 0; i < count; i++) {
        float3 rA = q_rotate(qAc, contacts[i].xA - pA4.xyz);
        float3 rB = q_rotate(qBc, contacts[i].xB - pB4.xyz);
        float3 lambda = float3(0);
        float3 penalty = float3(0);
        float stick = 0.0f;

        if (prevIdx >= 0) {
            device const ManifoldGPU& pm = prevManifolds[prevIdx];
            uint pn = pm.header.z;
            for (uint j = 0; j < pn; j++) {
                if (as_type<uint>(pm.contacts[j].rA.w) == contacts[i].feature) {
                    lambda = pm.contacts[j].lambda.xyz;
                    penalty = pm.contacts[j].penalty.xyz;
                    stick = pm.contacts[j].rB.w;
                    if (stick != 0.0f) {
                        // Sticking: keep previous anchor points for static friction
                        rA = pm.contacts[j].rA.xyz;
                        rB = pm.contacts[j].rB.xyz;
                    }
                    break;
                }
            }
        }

        // C0 at start-of-step pose
        float3 xAw = xform(pA4.xyz, qA, rA);
        float3 xBw = xform(pB4.xyz, qB, rB);
        float3 d = xAw - xBw;
        float3 C0 = float3(dot(nrm, d) + COLLISION_MARGIN, dot(t1, d), dot(t2, d));

        // Warm-start (Eq. 19)
        lambda *= warm;
        penalty = clamp(penalty * P.gamma, PENALTY_MIN, PENALTY_MAX);

        outM.contacts[i].rA = float4(rA, as_type<float>(contacts[i].feature));
        outM.contacts[i].rB = float4(rB, stick);
        outM.contacts[i].C0 = float4(C0, 0);
        outM.contacts[i].lambda = float4(lambda, 0);
        outM.contacts[i].penalty = float4(penalty, 0);
    }
}
