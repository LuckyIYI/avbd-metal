// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// The generic convex support-map, MPR, GJK simplex, and polygon-manifold path
// in this file is an altered Metal implementation inspired by Newton commits
// 37b75a212112cebc35bdbfb521357b8a2900d6be and
// 8e2f385a3d17f27152479e77c5472d14f95ae09f. Newton's implementations derive
// from Jitter Physics 2 (MIT), with MPR based on XenoCollide (zlib).
// See THIRD_PARTY_NOTICES.md.

#include <metal_stdlib>
using namespace metal;

#if defined(AVBD_OPTIMIZED_CONVEX)
#define NPC_MAX_POLY_EDGES 186
#define NPC_MAX_EDGE_PAIR_TESTS 1024
#endif

// ============================================================================
// Narrowphase: OBB-OBB SAT with reference-face clipping and edge-edge
// closest points (port of avbd-demo3d collide.cpp). One thread per pair.
// Warm-starts each contact from the previous frame's manifold (pair hash
// map lookup + feature key match), then initializes C0 / lambda / penalty
// for the AVBD solve.
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
                       float3 relVel, float speculativeCap,
                       constant SimParams& P,
                       thread NPAxisBest& best) {
    float lenSq = dot(axis, axis);
    if (lenSq < SAT_EPS) return true;

    float3 n = axis * rsqrt(lenSq);
    if (dot(n, delta) < 0.0f) n = -n;

    float distance = fabs(dot(delta, n));
    float rA = A.half3_.x * fabs(dot(n, A.ax0)) + A.half3_.y * fabs(dot(n, A.ax1)) + A.half3_.z * fabs(dot(n, A.ax2));
    float rB = B.half3_.x * fabs(dot(n, B.ax0)) + B.half3_.y * fabs(dot(n, B.ax1)) + B.half3_.z * fabs(dot(n, B.ax2));

    float separation = distance - (rA + rB);
    float approach = max(0.0f, -dot(relVel, n));
    float allowed = min(approach * P.dt, speculativeCap);
    if (separation > allowed) return false;

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

inline float npSpeculativeCap(float radiusA, float radiusB,
                              float contactMargin) {
    float r = min(fabs(radiusA), fabs(radiusB));
    return min(0.25f, max(4.0f * contactMargin, 3.0f * r));
}

inline float npDetectMargin(float3 n, float3 relVel, constant SimParams& P,
                            float radiusA, float radiusB) {
    float approach = max(0.0f, -dot(relVel, n));
    float allowed = min(approach * P.dt,
                        npSpeculativeCap(radiusA, radiusB,
                                         P.collisionMargin));
    return P.collisionMargin + allowed;
}

inline float3 npPenaltyFloor(device const float4* posLin, uint ia, uint ib,
                             constant SimParams& P) {
    float mA = posLin[ia].w;
    float mB = posLin[ib].w;
    float minMass = min(mA > 0.0f ? mA : FLT_MAX,
                        mB > 0.0f ? mB : FLT_MAX);
    if (minMass == FLT_MAX) return float3(PENALTY_MIN);
    float k = min(PENALTY_MAX, max(PENALTY_MIN,
                                   minMass / max(P.dt * P.dt, 1e-12f)));
    return float3(k, min(k, PENALTY_MAX_T), min(k, PENALTY_MAX_T));
}

inline float3 npPenaltyCeil() {
    return float3(PENALTY_MAX, PENALTY_MAX_T, PENALTY_MAX_T);
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

// ---------------------------------------------------------------------------
// Generic convex support mapping + Newton-style MPR/GJK
// ---------------------------------------------------------------------------

// The generic path is deliberately entered only when at least one collider is
// a cooked hull. Primitive-only pairs retain their established specialized
// paths below. Torus is non-convex and is rejected during solver construction.
inline float3 npcSafeNormalize(float3 v, float3 fallback) {
    float l2 = dot(v, v);
    if (!finite_bits(l2) || l2 <= 1.0e-20f) return fallback;
    float3 n = v * rsqrt(l2);
    return finite3(n) ? n : fallback;
}

inline NPCShapePoint npcShapeSupport(
    thread const NPCShape& s,
    float3 worldDirection,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices,
    bool centeredBoxTies)
{
    NPCShapePoint result;
    float3 direction = finite3(worldDirection)
        ? worldDirection : float3(1, 0, 0);
    result.feature = 0u;

    // Ephemeral deformable triangle thickened into a finite triangular prism.
    // Kind 5 stores its three authored vertices directly in center,
    // rotation.xyz, and dimensions.xyz. The query computes its interior seed
    // only after transforming those vertices into the small A-local frame;
    // reconstructing C from a world-space centroid loses several Float ULPs.
    if (s.kind == 5u) {
        float3 vertices[3];
        vertices[0] = s.center;
        vertices[1] = s.rotation.xyz;
        vertices[2] = s.dimensions.xyz;
        float3 triNormal = npcSafeNormalize(
            cross(vertices[1] - vertices[0], vertices[2] - vertices[0]),
            float3(0, 0, 1));
        uint best = 0u;
        float bestDot = dot(vertices[0], direction);
        for (uint i = 1u; i < 3u; i++) {
            float candidate = dot(vertices[i], direction);
            if (candidate > bestDot) {
                bestDot = candidate;
                best = i;
            }
        }
        bool positiveSide = dot(direction, triNormal) >= 0.0f;
        result.point = vertices[best] + triNormal
            * ((positiveSide ? 1.0f : -1.0f) * fabs(s.rotation.w));
        result.feature = (5u << 28) | best | (positiveSide ? 4u : 0u);
        return result;
    }

    float3 localDirection = q_rotate(q_conj(s.rotation), direction);

    if (s.kind == 4u) {
        uint2 range = colliderHullRange[s.collider];
        if (range.y == 0u) {
            result.point = s.center;
            result.feature = 0xFFFFFFFFu;
            return result;
        }
        float bestDot = -FLT_MAX;
        uint best = 0u;
        // Full scan is Newton's deterministic correctness path. Canonical
        // vertex order plus the strict comparison gives stable tie-breaking.
        for (uint i = 0u; i < range.y; i++) {
            float3 v = convexHullVertices[range.x + i].xyz;
            float d = dot(v, localDirection);
            if (d > bestDot) {
                bestDot = d;
                best = i;
            }
        }
        result.point = s.center
            + q_rotate(s.rotation, convexHullVertices[range.x + best].xyz);
        result.feature = (4u << 28) | (best & 0x0FFFFFFFu);
        return result;
    }

    if (s.kind == 0u) {
        float directionScale = max(fabs(localDirection.x),
            max(fabs(localDirection.y), fabs(localDirection.z)));
        float deadband = 1.0e-10f * directionScale;
        uint sx = localDirection.x >= -deadband ? 1u : 0u;
        uint sy = localDirection.y >= -deadband ? 1u : 0u;
        uint sz = localDirection.z >= -deadband ? 1u : 0u;
        float3 sign = float3(sx != 0u ? 1.0f : -1.0f,
                             sy != 0u ? 1.0f : -1.0f,
                             sz != 0u ? 1.0f : -1.0f);
        float3 local = sign * (s.dimensions.xyz * 0.5f);
        if (centeredBoxTies) {
            float3 contribution = fabs(localDirection)
                * (s.dimensions.xyz * 0.5f);
            float tie = 1.0e-6f
                * (contribution.x + contribution.y + contribution.z);
            if (contribution.x <= tie) local.x = 0.0f;
            if (contribution.y <= tie) local.y = 0.0f;
            if (contribution.z <= tie) local.z = 0.0f;
        }
        result.point = s.center + q_rotate(s.rotation, local);
        result.feature = (1u << 28) | sx | (sy << 1) | (sz << 2);
        return result;
    }

    float3 normal = npcSafeNormalize(direction, float3(1, 0, 0));
    if (s.kind == 1u) {
        result.point = s.center + normal * (s.dimensions.x * 0.5f);
        result.feature = NPC_FEATURE_SMOOTH | 1u;
        return result;
    }

    // Capsule: authored length is the center-line segment and the local axis
    // is +Z, matching the existing specialized path.
    float3 axis = q_rotate(s.rotation, float3(0, 0, 1));
    bool upper = dot(direction, axis) >= 0.0f;
    result.point = s.center
        + axis * ((upper ? 1.0f : -1.0f) * s.dimensions.x * 0.5f)
        + normal * s.dimensions.y;
    result.feature = NPC_FEATURE_SMOOTH | 2u | (upper ? 0x100u : 0u);
    return result;
}

inline NPCShapePoint npcShapeSupport(
    thread const NPCShape& s,
    float3 worldDirection,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices)
{
    return npcShapeSupport(
        s, worldDirection, colliderHullRange, convexHullVertices, false);
}

// Every generic convex query is solved in one translation- and rotation-local
// coordinate system. Rigid queries use shape A's authored local frame. An
// ephemeral triangle has no quaternion in the shared ABI, so a triangle-first
// swapped retry uses a vertex-A-relative, world-axis frame. In either case all
// portal/simplex arithmetic remains close to zero and independent of the
// absolute world origin.
inline float4 npcAFrameRotation(thread const NPCShape& a) {
    return a.kind == 5u ? float4(0, 0, 0, 1) : a.rotation;
}

inline float3 npcShapeInteriorCenter(thread const NPCShape& shape) {
    if (shape.kind == 5u) {
        // Base-relative summation retains the exact local vertex differences
        // and makes the arithmetic mean a guaranteed point in the prism.
        return shape.center
            + ((shape.rotation.xyz - shape.center)
               + (shape.dimensions.xyz - shape.center)) / 3.0f;
    }
    return shape.center;
}

inline NPCShape npcShapeInFrame(
    thread const NPCShape& shape,
    float3 origin,
    float4 inverseFrameRotation)
{
    NPCShape local = shape;
    if (shape.kind == 5u) {
        // Kind 5 stores three positions rather than a quaternion. Transform
        // each authored vertex independently into the query frame.
        local.center = q_rotate(
            inverseFrameRotation, shape.center - origin);
        local.rotation = float4(
            q_rotate(
                inverseFrameRotation, shape.rotation.xyz - origin),
            shape.rotation.w);
        local.dimensions = float4(
            q_rotate(
                inverseFrameRotation, shape.dimensions.xyz - origin),
            shape.dimensions.w);
    } else {
        local.center = q_rotate(
            inverseFrameRotation, shape.center - origin);
        local.rotation = q_mul(inverseFrameRotation, shape.rotation);
    }
    return local;
}

inline void npcMakeAFrame(
    thread const NPCShape& a,
    thread const NPCShape& b,
    thread NPCShape& localA,
    thread NPCShape& localB,
    thread float3& origin,
    thread float4& frameRotation)
{
    origin = a.center;
    frameRotation = npcAFrameRotation(a);
    float4 inverseFrameRotation = q_conj(frameRotation);
    localA = npcShapeInFrame(a, origin, inverseFrameRotation);
    localB = npcShapeInFrame(b, origin, inverseFrameRotation);
    // Avoid carrying quaternion round-off in the defining A frame.
    localA.center = float3(0);
    if (a.kind != 5u) localA.rotation = float4(0, 0, 0, 1);
}

inline NPCResult npcResultFromAFrame(
    thread const NPCResult& local,
    float3 origin,
    float4 frameRotation)
{
    NPCResult world = local;
    world.pointA = origin + q_rotate(frameRotation, local.pointA);
    world.pointB = origin + q_rotate(frameRotation, local.pointB);
    world.normalAB = q_rotate(frameRotation, local.normalAB);
    return world;
}

inline NPCVertex npcMinkowskiSupport(
    thread const NPCShape& a,
    thread const NPCShape& b,
    float3 direction,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices,
    float extend,
    bool centeredBoxTies)
{
    NPCShapePoint pa = npcShapeSupport(
        a, direction, colliderHullRange, convexHullVertices,
        centeredBoxTies);
    NPCShapePoint pb = npcShapeSupport(
        b, -direction, colliderHullRange, convexHullVertices,
        centeredBoxTies);
    if (extend != 0.0f) {
        float3 offset = npcSafeNormalize(direction, float3(1, 0, 0))
            * (0.5f * extend);
        pa.point += offset;
        pb.point -= offset;
    }
    NPCVertex v;
    v.pointA = pa.point;
    v.pointB = pb.point;
    v.w = pa.point - pb.point;
    v.featureA = pa.feature;
    v.featureB = pb.feature;
    return v;
}

inline NPCVertex npcMinkowskiSupport(
    thread const NPCShape& a,
    thread const NPCShape& b,
    float3 direction,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices)
{
    return npcMinkowskiSupport(
        a, b, direction, colliderHullRange, convexHullVertices, 0.0f, false);
}

inline void npcSwap(thread NPCVertex& a, thread NPCVertex& b) {
    NPCVertex t = a;
    a = b;
    b = t;
}

// Port of Newton/Jitter's XenoCollide portal refinement. It supplies stable
// penetration witnesses directly, avoiding EPA's unbounded face expansion.
inline NPCResult npcMPRWithEnlargeLocal(
    thread const NPCShape& a,
    thread const NPCShape& b,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices,
    float enlarge)
{
    float3 centerA = npcShapeInteriorCenter(a);
    float3 centerB = npcShapeInteriorCenter(b);
    NPCResult out;
    out.overlap = false;
    out.valid = false;
    out.pointA = centerA;
    out.pointB = centerB;
    out.normalAB = npcSafeNormalize(centerB - centerA, float3(1, 0, 0));
    out.signedDistance = 0.0f;
    out.featureA = 0xFFFFFFFFu;
    out.featureB = 0xFFFFFFFFu;

    // These are Newton's absolute portal thresholds. Scaling them by the
    // collider radius makes a 1000-unit hull accept centimeter-scale portal
    // error and materially changes both witnesses and corrected distance.
    const float numericEps = 1.0e-16f;
    const float collideEps = 1.0e-5f;
    // Newton deliberately moves the MPR/GJK switchover away from a resting
    // zero-distance contact. MPR sees a tiny symmetric Minkowski inflation;
    // witnesses and signed distance are corrected back to the true surfaces
    // before returning. GJK remains completely uninflated.
    NPCVertex v0;
    v0.pointA = centerA;
    v0.pointB = centerB;
    v0.w = centerA - centerB;
    v0.featureA = 0xFFFFFFFFu;
    v0.featureB = 0xFFFFFFFFu;
    if (dot(v0.w, v0.w) < numericEps) {
        float bestDot = -FLT_MAX;
        float3 bestDirection = float3(1, 0, 0);
        for (int axis = 0; axis < 3; axis++) {
            float3 direction = float3(0);
            direction[axis] = 1.0f;
            NPCVertex probe = npcMinkowskiSupport(
                a, b, direction, colliderHullRange, convexHullVertices,
                enlarge, true);
            float candidate = dot(probe.w, direction);
            if (candidate > bestDot) {
                bestDot = candidate;
                bestDirection = direction;
            }
        }
        v0.w = bestDirection * 1.0e-5f;
    }

    float3 normal = -v0.w;
    NPCVertex v1 = npcMinkowskiSupport(
        a, b, normal, colliderHullRange, convexHullVertices, enlarge, true);
    if (!finite3(v1.w) || dot(v1.w, normal) <= 0.0f) return out;

    normal = cross(v1.w, v0.w);
    if (dot(normal, normal) < numericEps * numericEps) {
        normal = npcSafeNormalize(v1.w - v0.w, out.normalAB);
        float penetration = dot(v1.w, normal);
        if (!finite_bits(penetration)) return out;
        out.overlap = true;
        out.valid = true;
        out.pointA = v1.pointA;
        out.pointB = v1.pointB;
        out.normalAB = normal;
        out.signedDistance = -penetration + enlarge;
        out.pointA -= normal * (0.5f * enlarge);
        out.pointB += normal * (0.5f * enlarge);
        out.featureA = v1.featureA;
        out.featureB = v1.featureB;
        return out;
    }

    NPCVertex v2 = npcMinkowskiSupport(
        a, b, normal, colliderHullRange, convexHullVertices, enlarge, true);
    if (!finite3(v2.w) || dot(v2.w, normal) <= 0.0f) return out;

    normal = cross(v1.w - v0.w, v2.w - v0.w);
    if (dot(normal, v0.w) > 0.0f) {
        npcSwap(v1, v2);
        normal = -normal;
    }

    NPCVertex v3;
    bool havePortal = false;
    for (int phase1 = 0; phase1 <= NPC_MPR_ITERATIONS; phase1++) {
        if (!finite3(normal) || dot(normal, normal) < numericEps * numericEps)
            return out;
        v3 = npcMinkowskiSupport(
            a, b, normal, colliderHullRange, convexHullVertices, enlarge, true);
        if (!finite3(v3.w) || dot(v3.w, normal) <= 0.0f) return out;

        float3 side = cross(v1.w, v3.w);
        if (dot(side, v0.w) < 0.0f) {
            v2 = v3;
            normal = cross(v1.w - v0.w, v3.w - v0.w);
            continue;
        }
        side = cross(v3.w, v2.w);
        if (dot(side, v0.w) < 0.0f) {
            v1 = v3;
            normal = cross(v3.w - v0.w, v2.w - v0.w);
            continue;
        }
        havePortal = true;
        break;
    }
    if (!havePortal) return out;

    bool hit = false;
    for (int phase2 = 0; phase2 <= NPC_MPR_ITERATIONS; phase2++) {
        float3 e12 = v2.w - v1.w;
        float3 e13 = v3.w - v1.w;
        normal = cross(e12, e13);
        float normalSq = dot(normal, normal);
        if (!finite_bits(normalSq) || normalSq < numericEps * numericEps)
            return out;
        if (!hit) hit = dot(normal, v1.w) >= 0.0f;

        NPCVertex v4 = npcMinkowskiSupport(
            a, b, normal, colliderHullRange, convexHullVertices, enlarge, true);
        if (!finite3(v4.w)) return out;
        float delta = dot(v4.w - v3.w, normal);
        float penetrationRaw = dot(v4.w, normal);
        bool converged = delta * delta
                <= collideEps * collideEps * normalSq
            || penetrationRaw <= 0.0f
            || phase2 == NPC_MPR_ITERATIONS;
        if (converged) {
            if (!hit) return out;
            float invNormal = rsqrt(normalSq);
            float3 unitNormal = normal * invNormal;
            float penetration = penetrationRaw * invNormal;
            if (!finite3(unitNormal) || !finite_bits(penetration)) return out;

            float3 temp = cross(v1.w, e12);
            float gamma = dot(temp, unitNormal) * invNormal;
            temp = cross(e13, v1.w);
            float beta = dot(temp, unitNormal) * invNormal;
            float alpha = 1.0f - gamma - beta;
            if (!finite_bits(alpha) || !finite_bits(beta)
                || !finite_bits(gamma)) return out;

            out.overlap = true;
            out.valid = true;
            out.pointA = alpha * v1.pointA + beta * v2.pointA
                + gamma * v3.pointA;
            out.pointB = alpha * v1.pointB + beta * v2.pointB
                + gamma * v3.pointB;
            out.normalAB = unitNormal;
            out.signedDistance = -penetration + enlarge;
            out.pointA -= unitNormal * (0.5f * enlarge);
            out.pointB += unitNormal * (0.5f * enlarge);
            NPCVertex feature = npcMinkowskiSupport(
                a, b, unitNormal, colliderHullRange, convexHullVertices,
                enlarge, true);
            out.featureA = feature.featureA;
            out.featureB = feature.featureB;
            out.valid = finite3(out.pointA) && finite3(out.pointB)
                && finite3(out.normalAB) && finite_bits(out.signedDistance);
            return out;
        }

        float3 side = cross(v4.w, v0.w);
        if (dot(side, v1.w) >= 0.0f) {
            if (dot(side, v2.w) >= 0.0f) v1 = v4;
            else v3 = v4;
        } else {
            if (dot(side, v3.w) >= 0.0f) v2 = v4;
            else v1 = v4;
        }
    }
    return out;
}

inline bool npcCorrectedMPRInAFrameIsConsistent(
    thread const NPCResult& result)
{
    if (!result.valid || !result.overlap || !finite3(result.pointA)
        || !finite3(result.pointB) || !finite3(result.normalAB)
        || !finite_bits(result.signedDistance)) return false;
    // Five portal convergence epsilons covers the accumulated float error in
    // the barycentric witness reconstruction without admitting a guessed
    // normal/distance pair. This absolute tolerance is valid only here, where
    // both corrected witnesses are still in the query's A-local frame.
    const float tolerance = 5.0e-5f;
    float normalSquared = dot(result.normalAB, result.normalAB);
    if (!finite_bits(normalSquared)
        || fabs(normalSquared - 1.0f) > tolerance) return false;
    float3 delta = result.pointB - result.pointA;
    float projectedResidual = fabs(
        dot(delta, result.normalAB) - result.signedDistance);
    float3 tangentResidual = delta
        - result.normalAB * result.signedDistance;
    return finite_bits(projectedResidual) && finite3(tangentResidual)
        && projectedResidual <= tolerance
        && length(tangentResidual) <= tolerance;
}

inline NPCResult npcMPRWithEnlarge(
    thread const NPCShape& a,
    thread const NPCShape& b,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices,
    float enlarge)
{
    NPCShape localA;
    NPCShape localB;
    float3 origin;
    float4 frameRotation;
    npcMakeAFrame(a, b, localA, localB, origin, frameRotation);
    NPCResult local = npcMPRWithEnlargeLocal(
        localA, localB, colliderHullRange, convexHullVertices, enlarge);
    if (local.valid && local.overlap
        && !npcCorrectedMPRInAFrameIsConsistent(local)) {
        // Never map an uncertified corrected witness to world space. At large
        // origins the two accepted world witnesses may quantize to the same
        // Float value, but signed distance remains the certified local value.
        local.valid = false;
    }
    return npcResultFromAFrame(local, origin, frameRotation);
}

inline NPCResult npcMPR(
    thread const NPCShape& a,
    thread const NPCShape& b,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices)
{
    return npcMPRWithEnlarge(
        a, b, colliderHullRange, convexHullVertices, 1.0e-4f);
}

inline bool npcCorrectedMPRIsConsistent(thread const NPCResult& result) {
    if (!result.valid || !result.overlap || !finite3(result.pointA)
        || !finite3(result.pointB) || !finite3(result.normalAB)
        || !finite_bits(result.signedDistance)) return false;
    // The corrected-witness residual was already certified before mapping out
    // of the A-local query frame. Repeating that subtraction in world space
    // would make validity depend on the world's Float ULP (for example 0.0625
    // at x = 1e6), so mapped callers recheck only finite/unit invariants.
    const float tolerance = 5.0e-5f;
    float normalSquared = dot(result.normalAB, result.normalAB);
    return finite_bits(normalSquared)
        && fabs(normalSquared - 1.0f) <= tolerance;
}

struct NPCSimplex {
    NPCVertex vertices[4];
    float4 barycentric;
    uint mask;
};

inline float npcDeterminant(float3 a, float3 b, float3 c, float3 d) {
    return dot(b - a, cross(c - a, d - a));
}

inline void npcClosestSegment(
    thread const NPCSimplex& simplex, int i0, int i1,
    thread float3& closest, thread float4& barycentric, thread uint& mask)
{
    float3 a = simplex.vertices[i0].w;
    float3 b = simplex.vertices[i1].w;
    float3 edge = b - a;
    float denom = dot(edge, edge);
    bool degenerate = denom < 1.0e-8f;
    float t = -dot(a, edge) / (degenerate ? 1.0e-8f : denom);
    float lambda0 = 1.0f - t;
    float lambda1 = t;
    mask = (1u << uint(i0)) | (1u << uint(i1));
    if (lambda0 < 0.0f || degenerate) {
        lambda0 = 0.0f;
        lambda1 = 1.0f;
        mask = 1u << uint(i1);
    } else if (lambda1 < 0.0f) {
        lambda0 = 1.0f;
        lambda1 = 0.0f;
        mask = 1u << uint(i0);
    }
    barycentric = float4(0);
    barycentric[i0] = lambda0;
    barycentric[i1] = lambda1;
    closest = a * barycentric[i0] + b * barycentric[i1];
}

inline void npcClosestTriangle(
    thread const NPCSimplex& simplex, int i0, int i1, int i2,
    thread float3& closest, thread float4& barycentric, thread uint& mask)
{
    float3 a = simplex.vertices[i0].w;
    float3 b = simplex.vertices[i1].w;
    float3 c = simplex.vertices[i2].w;
    float3 u = a - b;
    float3 w = a - c;
    float3 normal = cross(u, w);
    float denom = dot(normal, normal);
    bool degenerate = denom < 1.0e-8f;
    float inv = degenerate ? 0.0f : 1.0f / denom;
    float lambda2 = dot(cross(u, a), normal) * inv;
    float lambda1 = dot(cross(a, w), normal) * inv;
    float lambda0 = 1.0f - lambda1 - lambda2;

    float bestDistance = FLT_MAX;
    mask = 0u;
    barycentric = float4(0);
    closest = float3(0);
    if (lambda0 < 0.0f || degenerate) {
        float3 p; float4 bc; uint m;
        npcClosestSegment(simplex, i1, i2, p, bc, m);
        float d = dot(p, p);
        if (d < bestDistance) {
            bestDistance = d; closest = p; barycentric = bc; mask = m;
        }
    }
    if (lambda1 < 0.0f || degenerate) {
        float3 p; float4 bc; uint m;
        npcClosestSegment(simplex, i0, i2, p, bc, m);
        float d = dot(p, p);
        if (d < bestDistance) {
            bestDistance = d; closest = p; barycentric = bc; mask = m;
        }
    }
    if (lambda2 < 0.0f || degenerate) {
        float3 p; float4 bc; uint m;
        npcClosestSegment(simplex, i0, i1, p, bc, m);
        float d = dot(p, p);
        if (d < bestDistance) {
            closest = p; barycentric = bc; mask = m;
        }
    }
    if (mask != 0u) return;
    barycentric[i0] = lambda0;
    barycentric[i1] = lambda1;
    barycentric[i2] = lambda2;
    mask = (1u << uint(i0)) | (1u << uint(i1)) | (1u << uint(i2));
    closest = lambda0 * a + lambda1 * b + lambda2 * c;
}

inline void npcClosestTetrahedron(
    thread const NPCSimplex& simplex,
    thread float3& closest, thread float4& barycentric, thread uint& mask)
{
    float3 v0 = simplex.vertices[0].w;
    float3 v1 = simplex.vertices[1].w;
    float3 v2 = simplex.vertices[2].w;
    float3 v3 = simplex.vertices[3].w;
    float det = npcDeterminant(v0, v1, v2, v3);
    bool degenerate = fabs(det) < 1.0e-8f;
    float inv = degenerate ? 0.0f : 1.0f / det;
    float3 zero = float3(0);
    float l0 = npcDeterminant(zero, v1, v2, v3) * inv;
    float l1 = npcDeterminant(v0, zero, v2, v3) * inv;
    float l2 = npcDeterminant(v0, v1, zero, v3) * inv;
    float l3 = 1.0f - l0 - l1 - l2;

    float bestDistance = FLT_MAX;
    mask = 0u;
    barycentric = float4(0);
    closest = zero;
    if (l0 < 0.0f || degenerate) {
        float3 p; float4 bc; uint m;
        npcClosestTriangle(simplex, 1, 2, 3, p, bc, m);
        float d = dot(p, p);
        if (d < bestDistance) {
            bestDistance = d; closest = p; barycentric = bc; mask = m;
        }
    }
    if (l1 < 0.0f || degenerate) {
        float3 p; float4 bc; uint m;
        npcClosestTriangle(simplex, 0, 2, 3, p, bc, m);
        float d = dot(p, p);
        if (d < bestDistance) {
            bestDistance = d; closest = p; barycentric = bc; mask = m;
        }
    }
    if (l2 < 0.0f || degenerate) {
        float3 p; float4 bc; uint m;
        npcClosestTriangle(simplex, 0, 1, 3, p, bc, m);
        float d = dot(p, p);
        if (d < bestDistance) {
            bestDistance = d; closest = p; barycentric = bc; mask = m;
        }
    }
    if (l3 < 0.0f || degenerate) {
        float3 p; float4 bc; uint m;
        npcClosestTriangle(simplex, 0, 1, 2, p, bc, m);
        float d = dot(p, p);
        if (d < bestDistance) {
            closest = p; barycentric = bc; mask = m;
        }
    }
    if (mask != 0u) return;
    barycentric = float4(l0, l1, l2, l3);
    mask = 15u;
    closest = zero;
}

inline void npcSimplexWitness(
    thread const NPCSimplex& simplex,
    thread float3& pointA, thread float3& pointB)
{
    pointA = float3(0);
    pointB = float3(0);
    for (int i = 0; i < 4; i++) {
        if ((simplex.mask & (1u << uint(i))) == 0u) continue;
        pointA += simplex.vertices[i].pointA * simplex.barycentric[i];
        pointB += simplex.vertices[i].pointB * simplex.barycentric[i];
    }
}

// Newton's distance fallback after MPR rejects overlap. The duality-gap
// convergence test and simplex reduction produce accurate speculative
// witnesses without paying an EPA pass for separated pairs.
inline NPCResult npcGJKWithIterationLimitLocal(
    thread const NPCShape& a,
    thread const NPCShape& b,
    float maxDistance,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices,
    int iterationLimit)
{
    float3 centerA = npcShapeInteriorCenter(a);
    float3 centerB = npcShapeInteriorCenter(b);
    NPCResult out;
    out.overlap = false;
    out.valid = false;
    out.pointA = centerA;
    out.pointB = centerB;
    out.normalAB = npcSafeNormalize(centerB - centerA, float3(1, 0, 0));
    out.signedDistance = FLT_MAX;
    out.featureA = 0xFFFFFFFFu;
    out.featureB = 0xFFFFFFFFu;

    NPCSimplex simplex;
    simplex.barycentric = float4(0);
    simplex.mask = 0u;
    float3 v = centerA - centerB;
    // Keep the distance solver's convergence and simplex-degeneracy
    // thresholds aligned with Newton/Jitter and the Swift reference. The
    // previous 1e-5 relative threshold was ten times stricter around
    // unit-scale geometry and could cycle between nearly degenerate simplices
    // until the bounded iteration budget expired.
    float epsilon = 1.0e-4f;
    float distSq = dot(v, v);
    float3 lastDirection = out.normalAB;
    bool converged = false;

    for (int iteration = 0; iteration < iterationLimit; iteration++) {
        if (!finite_bits(distSq)) return out;
        if (distSq < epsilon * epsilon) {
            out.overlap = true;
            converged = true;
            break;
        }
        float3 search = -v;
        lastDirection = search;
        NPCVertex support = npcMinkowskiSupport(
            a, b, search, colliderHullRange, convexHullVertices);
        if (!finite3(support.w)) return out;
        if (maxDistance > 0.0f
            && dot(v, support.w) > maxDistance * sqrt(distSq)) {
            converged = true;
            break;
        }
        float gap = dot(v, v - support.w);
        if (gap <= 0.0f
            || gap * gap < epsilon * epsilon * distSq) {
            converged = true;
            break;
        }

        bool duplicate = false;
        for (int i = 0; i < 4; i++) {
            if ((simplex.mask & (1u << uint(i))) == 0u) continue;
            float3 d = simplex.vertices[i].w - support.w;
            if (dot(d, d) < epsilon * epsilon) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            // Like the stationary reducer case below, a retained support
            // duplicate proves numerical stalling but not separation. The
            // witness upper bound and support-plane lower bound must classify
            // the bounded contact query on the same (positive) side before a
            // finite separated result is trustworthy.
            float duplicateUpper = sqrt(max(distSq, 0.0f));
            float duplicateProjection = dot(v, support.w);
            bool duplicateSeparated = duplicateProjection
                > epsilon * duplicateUpper;
            if (maxDistance > 0.0f && duplicateUpper <= maxDistance
                && duplicateSeparated) {
                converged = true;
                break;
            }
            // Invalid results carry only this conservative simplex-distance
            // upper bound so the caller can size one certified MPR retry.
            out.signedDistance = duplicateUpper;
            return out;
        }

        uint previousMask = simplex.mask;
        float3 previousClosest = v;
        int indices[4];
        int useCount = 0;
        int freeSlot = 0;
        for (int i = 0; i < 4; i++) {
            if ((simplex.mask & (1u << uint(i))) != 0u)
                indices[useCount++] = i;
            else freeSlot = i;
        }
        indices[useCount++] = freeSlot;
        simplex.vertices[freeSlot] = support;

        float3 closest = float3(0);
        float4 barycentric = float4(0);
        uint mask = 0u;
        if (useCount == 1) {
            closest = support.w;
            barycentric[freeSlot] = 1.0f;
            mask = 1u << uint(freeSlot);
        } else if (useCount == 2) {
            npcClosestSegment(simplex, indices[0], indices[1],
                              closest, barycentric, mask);
        } else if (useCount == 3) {
            npcClosestTriangle(simplex, indices[0], indices[1], indices[2],
                               closest, barycentric, mask);
        } else {
            npcClosestTetrahedron(simplex, closest, barycentric, mask);
        }
        simplex.barycentric = barycentric;
        simplex.mask = mask;
        if (mask == 15u) {
            out.overlap = true;
            converged = true;
            break;
        }
        // A support point can be outside the retained simplex and therefore
        // evade the retained-vertex duplicate check. If adding that point is
        // rejected by the simplex reducer and leaves both the simplex and its
        // closest point unchanged, the deterministic query is at a fixed
        // point. This establishes numerical stalling, not separation by
        // itself. Admit only when the current witness is inside the caller's
        // positive contact threshold and the Frank-Wolfe support plane gives
        // a robustly positive lower bound. Threshold-straddling or exact
        // distance queries remain fail-closed.
        float3 closestDelta = closest - previousClosest;
        bool stationary = mask == previousMask
            && dot(closestDelta, closestDelta) <= 1.0e-16f;
        float upperSquared = dot(closest, closest);
        float upper = sqrt(max(upperSquared, 0.0f));
        float supportProjection = dot(previousClosest, support.w);
        bool robustlySeparated = supportProjection > epsilon * upper;
        if (stationary && maxDistance > 0.0f
            && upper <= maxDistance && robustlySeparated) {
            v = closest;
            distSq = upperSquared;
            converged = true;
            break;
        }
        v = closest;
        distSq = dot(v, v);
    }

    // A finite last simplex is not evidence of separation. If the bounded
    // loop used its entire budget without one of the explicit convergence
    // conditions above, return invalid so callers fail the frame closed.
    if (!converged) {
        // Preserve the conservative simplex-distance upper bound for the
        // caller's bounded retry; all witness fields remain invalid.
        out.signedDistance = sqrt(max(distSq, 0.0f));
        return out;
    }

    if (simplex.mask != 0u) {
        npcSimplexWitness(simplex, out.pointA, out.pointB);
    } else {
        NPCVertex seed = npcMinkowskiSupport(
            a, b, out.normalAB, colliderHullRange, convexHullVertices);
        out.pointA = seed.pointA;
        out.pointB = seed.pointB;
    }
    float3 delta = out.pointB - out.pointA;
    float distanceSq = dot(delta, delta);
    if (distanceSq > epsilon * epsilon) {
        out.signedDistance = sqrt(distanceSq);
        out.normalAB = delta / out.signedDistance;
    } else {
        float fallbackSq = dot(lastDirection, lastDirection);
        out.signedDistance = sqrt(max(distSq, 0.0f));
        float3 localNormal = fallbackSq > 1.0e-20f
            ? lastDirection * rsqrt(fallbackSq) : float3(1, 0, 0);
        out.normalAB = localNormal;
    }
    NPCVertex feature = npcMinkowskiSupport(
        a, b, out.normalAB, colliderHullRange, convexHullVertices);
    out.featureA = feature.featureA;
    out.featureB = feature.featureB;
    out.valid = finite3(out.pointA) && finite3(out.pointB)
        && finite3(out.normalAB) && finite_bits(out.signedDistance);
    return out;
}

inline NPCResult npcGJKWithIterationLimit(
    thread const NPCShape& a,
    thread const NPCShape& b,
    float maxDistance,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices,
    int iterationLimit)
{
    NPCShape localA;
    NPCShape localB;
    float3 origin;
    float4 frameRotation;
    npcMakeAFrame(a, b, localA, localB, origin, frameRotation);
    NPCResult local = npcGJKWithIterationLimitLocal(
        localA, localB, maxDistance, colliderHullRange,
        convexHullVertices, iterationLimit);
    return npcResultFromAFrame(local, origin, frameRotation);
}

inline NPCResult npcGJK(
    thread const NPCShape& a,
    thread const NPCShape& b,
    float maxDistance,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices)
{
    return npcGJKWithIterationLimit(
        a, b, maxDistance, colliderHullRange, convexHullVertices,
        NPC_GJK_ITERATIONS);
}

// ---------------------------------------------------------------------------
// Deterministic polyhedral manifold expansion
// ---------------------------------------------------------------------------

// MPR/GJK intentionally return one witness and the collision normal. For
// polyhedra, expand that witness into the clipped reference/incident polygon
// used by Newton/Jitter-style contact generation. The cooked face loops are
// maximal coplanar polygons (the CPU uploader removes triangulation diagonals),
// so a broad quad stays a broad four-point manifold.

#define NPC_FEATURE_BOX_VERTEX  0x10000000u
#define NPC_FEATURE_BOX_FACE    0x20000000u
#define NPC_FEATURE_BOX_EDGE    0x30000000u
#define NPC_FEATURE_HULL_VERTEX 0x40000000u
#define NPC_FEATURE_HULL_FACE   0x50000000u
#define NPC_FEATURE_HULL_EDGE   0x60000000u
#define NPC_FEATURE_CLIPPED     0x70000000u

struct NPCPolyFace {
    bool valid;
    bool hull;
    uint localIndex;
    uint loopStart;
    uint count;
    float3 normal;                  // world-space outward normal
    float distance;                 // dot(normal, world face plane)
};

struct NPCPolyEdge {
    bool valid;
    uint localIndex;
    float3 a;
    float3 b;
};

struct NPCClipVertex {
    float3 point;
    uint incidentFeature;
    uint referenceFeature;
};

struct NPCManifoldContact {
    float3 pointA;
    float3 pointB;
    uint2 features;
};

inline uint npcMixFeature(uint a, uint b) {
    uint lo = min(a, b);
    uint hi = max(a, b);
    uint h = lo * 0x9E3779B9u ^ hi * 0x85EBCA6Bu;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    return h;
}

inline uint npcFeaturePairHash(uint2 pair) {
    uint h = pair.x * 0x9E3779B9u ^ pair.y * 0x85EBCA6Bu;
    h ^= h >> 16;
    return h;
}

inline float3 npcBoxLocalVertex(thread const NPCShape& shape, uint index) {
    float3 halfExtent = shape.dimensions.xyz * 0.5f;
    return float3((index & 1u) != 0u ? halfExtent.x : -halfExtent.x,
                  (index & 2u) != 0u ? halfExtent.y : -halfExtent.y,
                  (index & 4u) != 0u ? halfExtent.z : -halfExtent.z);
}

inline uint npcBoxFaceVertexIndex(uint face, uint index) {
    // Every loop is counter-clockwise when viewed along its outward normal.
    if (face == 0u) { // -X
        uint ids[4] = { 0u, 4u, 6u, 2u };
        return ids[index & 3u];
    }
    if (face == 1u) { // +X
        uint ids[4] = { 1u, 3u, 7u, 5u };
        return ids[index & 3u];
    }
    if (face == 2u) { // -Y
        uint ids[4] = { 0u, 1u, 5u, 4u };
        return ids[index & 3u];
    }
    if (face == 3u) { // +Y
        uint ids[4] = { 2u, 6u, 7u, 3u };
        return ids[index & 3u];
    }
    if (face == 4u) { // -Z
        uint ids[4] = { 0u, 2u, 3u, 1u };
        return ids[index & 3u];
    }
    uint ids[4] = { 4u, 5u, 7u, 6u }; // +Z
    return ids[index & 3u];
}

inline float3 npcBoxFaceLocalNormal(uint face) {
    if (face == 0u) return float3(-1, 0, 0);
    if (face == 1u) return float3(1, 0, 0);
    if (face == 2u) return float3(0, -1, 0);
    if (face == 3u) return float3(0, 1, 0);
    if (face == 4u) return float3(0, 0, -1);
    return float3(0, 0, 1);
}

inline uint npcPolyFaceCount(
    thread const NPCShape& shape,
    device const uint* colliderConvexAssetID,
    device const ConvexHullGPU* convexHulls)
{
    if (shape.kind == 0u) return 6u;
    if (shape.kind != 4u) return 0u;
    uint asset = colliderConvexAssetID[shape.collider];
    return asset == 0xFFFFFFFFu ? 0u : convexHulls[asset].verticesFaces.w;
}

inline NPCPolyFace npcPolyFace(
    thread const NPCShape& shape, uint localFace,
    device const uint* colliderConvexAssetID,
    device const ConvexHullGPU* convexHulls,
    device const ConvexFaceGPU* convexFaces)
{
    NPCPolyFace result;
    result.valid = false;
    result.hull = shape.kind == 4u;
    result.localIndex = localFace;
    result.loopStart = 0u;
    result.count = 0u;
    result.normal = float3(1, 0, 0);
    result.distance = 0.0f;
    if (shape.kind == 0u) {
        if (localFace >= 6u) return result;
        float3 localNormal = npcBoxFaceLocalNormal(localFace);
        uint vertexIndex = npcBoxFaceVertexIndex(localFace, 0u);
        float3 point = shape.center
            + q_rotate(shape.rotation, npcBoxLocalVertex(shape, vertexIndex));
        result.valid = true;
        result.count = 4u;
        result.normal = q_rotate(shape.rotation, localNormal);
        result.distance = dot(result.normal, point);
        return result;
    }
    if (shape.kind != 4u) return result;
    uint asset = colliderConvexAssetID[shape.collider];
    if (asset == 0xFFFFFFFFu) return result;
    ConvexHullGPU hull = convexHulls[asset];
    if (localFace >= hull.verticesFaces.w) return result;
    ConvexFaceGPU face = convexFaces[hull.verticesFaces.z + localFace];
    result.valid = face.loop.y >= 3u
        && face.loop.y <= NPC_MAX_FACE_VERTICES;
    result.loopStart = face.loop.x;
    result.count = face.loop.y;
    result.localIndex = face.loop.z;
    result.normal = q_rotate(shape.rotation, face.plane.xyz);
    result.distance = face.plane.w + dot(result.normal, shape.center);
    return result;
}

inline float3 npcPolyFaceVertex(
    thread const NPCShape& shape, thread const NPCPolyFace& face, uint index,
    device const uint* colliderConvexAssetID,
    device const ConvexHullGPU* convexHulls,
    device const uint* convexFaceVertexIndices,
    device const float4* convexHullVertices,
    thread uint& feature)
{
    if (!face.hull) {
        uint vertexIndex = npcBoxFaceVertexIndex(face.localIndex, index);
        feature = NPC_FEATURE_BOX_VERTEX | vertexIndex;
        return shape.center
            + q_rotate(shape.rotation, npcBoxLocalVertex(shape, vertexIndex));
    }
    uint asset = colliderConvexAssetID[shape.collider];
    ConvexHullGPU hull = convexHulls[asset];
    uint localVertex = convexFaceVertexIndices[face.loopStart + index];
    feature = NPC_FEATURE_HULL_VERTEX | (localVertex & 0x0FFFFFFFu);
    return shape.center + q_rotate(shape.rotation,
        convexHullVertices[hull.verticesFaces.x + localVertex].xyz);
}

inline uint npcFaceFeature(thread const NPCShape& shape, uint face) {
    return (shape.kind == 4u ? NPC_FEATURE_HULL_FACE : NPC_FEATURE_BOX_FACE)
        | (face & 0x0FFFFFFFu);
}

inline uint npcFaceEdgeFeature(
    thread const NPCShape& shape, uint face, uint side)
{
    uint base = shape.kind == 4u ? NPC_FEATURE_HULL_EDGE
        : NPC_FEATURE_BOX_EDGE;
    return base | ((face & 0x3FFFu) << 14) | (side & 0x3FFFu);
}

inline NPCPolyFace npcBestPolyFace(
    thread const NPCShape& shape, float3 direction,
    device const uint* colliderConvexAssetID,
    device const ConvexHullGPU* convexHulls,
    device const ConvexFaceGPU* convexFaces,
    thread float& alignment)
{
    NPCPolyFace best;
    best.valid = false;
    alignment = -FLT_MAX;
    uint count = npcPolyFaceCount(shape, colliderConvexAssetID, convexHulls);
    for (uint index = 0u; index < count; index++) {
        NPCPolyFace face = npcPolyFace(
            shape, index, colliderConvexAssetID, convexHulls, convexFaces);
        if (!face.valid) continue;
        float value = dot(face.normal, direction);
        if (!best.valid || value > alignment + 1.0e-7f
            || (fabs(value - alignment) <= 1.0e-7f
                && face.localIndex < best.localIndex)) {
            best = face;
            alignment = value;
        }
    }
    return best;
}

inline int npcClipPolygon(
    thread const NPCClipVertex* input, int inputCount,
    float3 planeNormal, float planeDistance, uint referenceFeature,
    thread NPCClipVertex* output)
{
    if (inputCount <= 0) return 0;
    int outputCount = 0;
    NPCClipVertex a = input[inputCount - 1];
    float da = dot(planeNormal, a.point) - planeDistance;
    for (int index = 0; index < inputCount; index++) {
        NPCClipVertex b = input[index];
        float db = dot(planeNormal, b.point) - planeDistance;
        bool aInside = da <= PLANE_EPS;
        bool bInside = db <= PLANE_EPS;
        if (aInside != bInside && outputCount < NPC_MAX_FACE_VERTICES) {
            float denominator = da - db;
            float t = fabs(denominator) > SAT_EPS
                ? clamp(da / denominator, 0.0f, 1.0f) : 0.0f;
            NPCClipVertex clipped;
            clipped.point = mix(a.point, b.point, t);
            clipped.incidentFeature = NPC_FEATURE_CLIPPED
                | (npcMixFeature(a.incidentFeature, b.incidentFeature)
                    & 0x0FFFFFFFu);
            clipped.referenceFeature = referenceFeature;
            output[outputCount++] = clipped;
        }
        if (bInside && outputCount < NPC_MAX_FACE_VERTICES) {
            output[outputCount++] = b;
        }
        a = b;
        da = db;
    }
    return outputCount;
}

inline bool npcAddManifoldContact(
    thread NPCManifoldContact* contacts, thread int& count,
    float3 pointA, float3 pointB, uint2 features, float mergeDistanceSquared)
{
    float3 midpoint = (pointA + pointB) * 0.5f;
    for (int index = 0; index < count; index++) {
        float3 other = (contacts[index].pointA + contacts[index].pointB) * 0.5f;
        if (distance_squared(midpoint, other) <= mergeDistanceSquared) {
            // At a geometric duplicate retain the lexicographically smallest
            // exact feature identity, independent of clipping traversal noise.
            uint2 old = contacts[index].features;
            if (features.x < old.x
                || (features.x == old.x && features.y < old.y)) {
                contacts[index].pointA = pointA;
                contacts[index].pointB = pointB;
                contacts[index].features = features;
            }
            return false;
        }
    }
    if (count >= NPC_MAX_FACE_VERTICES) return false;
    contacts[count].pointA = pointA;
    contacts[count].pointB = pointB;
    contacts[count].features = features;
    count++;
    return true;
}

inline int npcReduceManifold(
    thread const NPCManifoldContact* candidates, int candidateCount,
    float3 normalBToA, thread NPCManifoldContact* reduced)
{
    if (candidateCount <= MAX_CONTACTS) {
        for (int index = 0; index < candidateCount; index++)
            reduced[index] = candidates[index];
        return candidateCount;
    }

    bool selected[NPC_MAX_FACE_VERTICES];
    for (int index = 0; index < NPC_MAX_FACE_VERTICES; index++)
        selected[index] = false;
    int first = 0;
    float deepest = FLT_MAX;
    for (int index = 0; index < candidateCount; index++) {
        float separation = dot(normalBToA,
            candidates[index].pointA - candidates[index].pointB);
        uint2 key = candidates[index].features;
        uint2 old = candidates[first].features;
        if (separation < deepest - 1.0e-7f
            || (fabs(separation - deepest) <= 1.0e-7f
                && (key.x < old.x || (key.x == old.x && key.y < old.y)))) {
            first = index;
            deepest = separation;
        }
    }
    reduced[0] = candidates[first];
    selected[first] = true;
    int outputCount = 1;
    while (outputCount < MAX_CONTACTS) {
        int best = -1;
        float bestSpread = -1.0f;
        for (int candidate = 0; candidate < candidateCount; candidate++) {
            if (selected[candidate]) continue;
            float3 midpoint = (candidates[candidate].pointA
                + candidates[candidate].pointB) * 0.5f;
            float minimumSpread = FLT_MAX;
            for (int chosen = 0; chosen < outputCount; chosen++) {
                float3 other = (reduced[chosen].pointA
                    + reduced[chosen].pointB) * 0.5f;
                float3 delta = midpoint - other;
                delta -= normalBToA * dot(normalBToA, delta);
                minimumSpread = min(minimumSpread, dot(delta, delta));
            }
            uint2 key = candidates[candidate].features;
            uint2 old = best >= 0 ? candidates[best].features
                : uint2(0xFFFFFFFFu);
            if (best < 0 || minimumSpread > bestSpread + 1.0e-10f
                || (fabs(minimumSpread - bestSpread) <= 1.0e-10f
                    && (key.x < old.x
                        || (key.x == old.x && key.y < old.y)))) {
                best = candidate;
                bestSpread = minimumSpread;
            }
        }
        if (best < 0) break;
        reduced[outputCount++] = candidates[best];
        selected[best] = true;
    }
    return outputCount;
}

inline uint npcPolyEdgeCount(
    thread const NPCShape& shape,
    device const uint* colliderConvexAssetID,
    device const ConvexHullGPU* convexHulls)
{
    if (shape.kind == 0u) return 12u;
    if (shape.kind != 4u) return 0u;
    uint asset = colliderConvexAssetID[shape.collider];
    return asset == 0xFFFFFFFFu ? 0u : convexHulls[asset].edgesLoops.y;
}

inline uint2 npcBoxEdgeVertices(uint edge) {
    uint pairs[24] = {
        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
        0u, 2u, 1u, 3u, 4u, 6u, 5u, 7u,
        0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u
    };
    uint offset = min(edge, 11u) * 2u;
    return uint2(pairs[offset], pairs[offset + 1u]);
}

inline NPCPolyEdge npcPolyEdge(
    thread const NPCShape& shape, uint localEdge,
    device const uint* colliderConvexAssetID,
    device const ConvexHullGPU* convexHulls,
    device const ConvexEdgeGPU* convexEdges,
    device const float4* convexHullVertices)
{
    NPCPolyEdge result;
    result.valid = false;
    result.localIndex = localEdge;
    result.a = shape.center;
    result.b = shape.center;
    uint2 vertices;
    uint vertexStart = 0u;
    if (shape.kind == 0u) {
        if (localEdge >= 12u) return result;
        vertices = npcBoxEdgeVertices(localEdge);
        result.a = shape.center
            + q_rotate(shape.rotation, npcBoxLocalVertex(shape, vertices.x));
        result.b = shape.center
            + q_rotate(shape.rotation, npcBoxLocalVertex(shape, vertices.y));
        result.valid = true;
        return result;
    }
    if (shape.kind != 4u) return result;
    uint asset = colliderConvexAssetID[shape.collider];
    if (asset == 0xFFFFFFFFu) return result;
    ConvexHullGPU hull = convexHulls[asset];
    if (localEdge >= hull.edgesLoops.y) return result;
    ConvexEdgeGPU edge = convexEdges[hull.edgesLoops.x + localEdge];
    vertices = edge.endpointsFaces.xy;
    vertexStart = hull.verticesFaces.x;
    result.a = shape.center + q_rotate(shape.rotation,
        convexHullVertices[vertexStart + vertices.x].xyz);
    result.b = shape.center + q_rotate(shape.rotation,
        convexHullVertices[vertexStart + vertices.y].xyz);
    result.valid = finite3(result.a) && finite3(result.b)
        && distance_squared(result.a, result.b) > 1.0e-16f;
    return result;
}

inline bool npcBestEdgeContact(
    thread const NPCShape& a, thread const NPCShape& b, float3 normalAtoB,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices,
    device const uint* colliderConvexAssetID,
    device const ConvexHullGPU* convexHulls,
    device const ConvexEdgeGPU* convexEdges,
    thread NPCManifoldContact& contact, thread float& alignment,
    thread uint& edgePairTests)
{
    uint countA = npcPolyEdgeCount(a, colliderConvexAssetID, convexHulls);
    uint countB = npcPolyEdgeCount(b, colliderConvexAssetID, convexHulls);
    if (countA == 0u || countB == 0u) return false;
    float supportA = dot(npcShapeSupport(
        a, normalAtoB, colliderHullRange, convexHullVertices).point,
        normalAtoB);
    float supportB = dot(npcShapeSupport(
        b, -normalAtoB, colliderHullRange, convexHullVertices).point,
        normalAtoB);
    float scale = max(1.0e-3f,
        max(fabs(a.dimensions.w), fabs(b.dimensions.w)));
    float supportTolerance = max(1.0e-5f, 0.005f * scale);
    bool found = false;
    float bestScore = -FLT_MAX;
    uint bestA = 0u;
    uint bestB = 0u;
    NPCPolyEdge chosenA;
    NPCPolyEdge chosenB;
    alignment = -FLT_MAX;

#if defined(AVBD_OPTIMIZED_CONVEX)
    // Compact supporting edges once. The previous nested search revisited all
    // E_b edges for every supporting E_a edge (up to 186² in one thread),
    // although almost all failed the same support-plane predicate. If a very
    // coarse shape still creates an excessive Cartesian product, retain the
    // already-valid face/MPR manifold rather than spending unbounded time on
    // optional edge enrichment.
    ushort supportingA[NPC_MAX_POLY_EDGES];
    ushort supportingB[NPC_MAX_POLY_EDGES];
    uint supportingCountA = 0u;
    uint supportingCountB = 0u;
    for (uint indexA = 0u; indexA < countA; indexA++) {
        NPCPolyEdge edgeA = npcPolyEdge(
            a, indexA, colliderConvexAssetID, convexHulls,
            convexEdges, convexHullVertices);
        if (!edgeA.valid) continue;
        float deficitA = max(fabs(supportA - dot(edgeA.a, normalAtoB)),
                             fabs(supportA - dot(edgeA.b, normalAtoB)));
        if (deficitA <= supportTolerance)
            supportingA[supportingCountA++] = ushort(indexA);
    }
    for (uint indexB = 0u; indexB < countB; indexB++) {
        NPCPolyEdge edgeB = npcPolyEdge(
            b, indexB, colliderConvexAssetID, convexHulls,
            convexEdges, convexHullVertices);
        if (!edgeB.valid) continue;
        float deficitB = max(fabs(dot(edgeB.a, normalAtoB) - supportB),
                             fabs(dot(edgeB.b, normalAtoB) - supportB));
        if (deficitB <= supportTolerance)
            supportingB[supportingCountB++] = ushort(indexB);
    }
    uint supportingPairCount = supportingCountA * supportingCountB;
    if (supportingPairCount == 0u
        || supportingPairCount > NPC_MAX_EDGE_PAIR_TESTS) return false;
    edgePairTests += supportingPairCount;

    for (uint candidateA = 0u; candidateA < supportingCountA; candidateA++) {
        uint indexA = uint(supportingA[candidateA]);
        NPCPolyEdge edgeA = npcPolyEdge(
            a, indexA, colliderConvexAssetID, convexHulls,
            convexEdges, convexHullVertices);
        float deficitA = max(fabs(supportA - dot(edgeA.a, normalAtoB)),
                             fabs(supportA - dot(edgeA.b, normalAtoB)));
        float3 directionA = normalize(edgeA.b - edgeA.a);
        for (uint candidateB = 0u; candidateB < supportingCountB;
             candidateB++) {
            uint indexB = uint(supportingB[candidateB]);
            NPCPolyEdge edgeB = npcPolyEdge(
                b, indexB, colliderConvexAssetID, convexHulls,
                convexEdges, convexHullVertices);
            float deficitB = max(fabs(dot(edgeB.a, normalAtoB) - supportB),
                                 fabs(dot(edgeB.b, normalAtoB) - supportB));
            float3 directionB = normalize(edgeB.b - edgeB.a);
#else
    for (uint indexA = 0u; indexA < countA; indexA++) {
        NPCPolyEdge edgeA = npcPolyEdge(
            a, indexA, colliderConvexAssetID, convexHulls,
            convexEdges, convexHullVertices);
        if (!edgeA.valid) continue;
        float deficitA = max(fabs(supportA - dot(edgeA.a, normalAtoB)),
                             fabs(supportA - dot(edgeA.b, normalAtoB)));
        if (deficitA > supportTolerance) continue;
        float3 directionA = normalize(edgeA.b - edgeA.a);
        for (uint indexB = 0u; indexB < countB; indexB++) {
            NPCPolyEdge edgeB = npcPolyEdge(
                b, indexB, colliderConvexAssetID, convexHulls,
                convexEdges, convexHullVertices);
            if (!edgeB.valid) continue;
            edgePairTests++;
            float deficitB = max(fabs(dot(edgeB.a, normalAtoB) - supportB),
                                 fabs(dot(edgeB.b, normalAtoB) - supportB));
            if (deficitB > supportTolerance) continue;
            float3 directionB = normalize(edgeB.b - edgeB.a);
#endif
            float3 crossDirection = cross(directionA, directionB);
            float crossLengthSquared = dot(crossDirection, crossDirection);
            if (crossLengthSquared <= 1.0e-10f) continue;
            float crossAlignment = fabs(dot(
                crossDirection * rsqrt(crossLengthSquared), normalAtoB));
            float score = crossAlignment
                - (deficitA + deficitB) / max(scale, 1.0e-6f);
            if (!found || score > bestScore + 1.0e-7f
                || (fabs(score - bestScore) <= 1.0e-7f
                    && (indexA < bestA
                        || (indexA == bestA && indexB < bestB)))) {
                found = true;
                bestScore = score;
                bestA = indexA;
                bestB = indexB;
                chosenA = edgeA;
                chosenB = edgeB;
                alignment = crossAlignment;
            }
        }
    }
    if (!found) return false;
    npClosestSegSeg(chosenA.a, chosenA.b, chosenB.a, chosenB.b,
                    contact.pointA, contact.pointB);
    contact.features = uint2(
        (a.kind == 4u ? NPC_FEATURE_HULL_EDGE : NPC_FEATURE_BOX_EDGE)
            | (bestA & 0x0FFFFFFFu),
        (b.kind == 4u ? NPC_FEATURE_HULL_EDGE : NPC_FEATURE_BOX_EDGE)
            | (bestB & 0x0FFFFFFFu));
    return finite3(contact.pointA) && finite3(contact.pointB);
}

inline int npcBuildPolyhedralManifold(
    thread const NPCShape& a, thread const NPCShape& b,
    float3 normalAtoB, float detectDistance,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices,
    device const uint* colliderConvexAssetID,
    device const ConvexHullGPU* convexHulls,
    device const ConvexFaceGPU* convexFaces,
    device const uint* convexFaceVertexIndices,
    device const ConvexEdgeGPU* convexEdges,
    thread NPCManifoldContact* output,
    thread uint& edgePairTests)
{
    float alignmentA, alignmentB;
    NPCPolyFace faceA = npcBestPolyFace(
        a, normalAtoB, colliderConvexAssetID, convexHulls, convexFaces,
        alignmentA);
    NPCPolyFace faceB = npcBestPolyFace(
        b, -normalAtoB, colliderConvexAssetID, convexHulls, convexFaces,
        alignmentB);
    if (!faceA.valid || !faceB.valid) return 0;

    float faceAlignment = max(alignmentA, alignmentB);
    NPCManifoldContact edgeContact;
    float edgeAlignment = -FLT_MAX;
    // A near-parallel face normal cannot lose to an edge cross-axis. Avoid the
    // potentially O(Ea*Eb) supporting-edge search for the overwhelmingly
    // common broad-face/resting case.
    bool haveEdge = false;
    if (faceAlignment < 0.985f) {
        haveEdge = npcBestEdgeContact(
            a, b, normalAtoB, colliderHullRange, convexHullVertices,
            colliderConvexAssetID, convexHulls, convexEdges,
            edgeContact, edgeAlignment, edgePairTests);
    }
    // A face wins ties by design. Edge contact is selected only when the MPR
    // normal is materially better explained by a supporting edge cross-axis.
    if (haveEdge && edgeAlignment > faceAlignment + 0.025f
        && faceAlignment < 0.985f) {
        output[0] = edgeContact;
        return 1;
    }

    bool referenceIsA = alignmentA >= alignmentB - 1.0e-7f;
    NPCShape referenceShape = referenceIsA ? a : b;
    NPCShape incidentShape = referenceIsA ? b : a;
    NPCPolyFace referenceFace = referenceIsA ? faceA : faceB;
    float incidentAlignment;
    NPCPolyFace incidentFace = npcBestPolyFace(
        incidentShape, -referenceFace.normal,
        colliderConvexAssetID, convexHulls, convexFaces, incidentAlignment);
    if (!incidentFace.valid) {
        if (haveEdge) { output[0] = edgeContact; return 1; }
        return 0;
    }

    NPCClipVertex polygonA[NPC_MAX_FACE_VERTICES];
    NPCClipVertex polygonB[NPC_MAX_FACE_VERTICES];
    uint referenceFaceFeature = npcFaceFeature(
        referenceShape, referenceFace.localIndex);
    int polygonCount = int(incidentFace.count);
    for (int index = 0; index < polygonCount; index++) {
        uint vertexFeature;
        polygonA[index].point = npcPolyFaceVertex(
            incidentShape, incidentFace, uint(index),
            colliderConvexAssetID, convexHulls, convexFaceVertexIndices,
            convexHullVertices, vertexFeature);
        polygonA[index].incidentFeature = vertexFeature;
        polygonA[index].referenceFeature = referenceFaceFeature;
    }

    for (uint side = 0u; side < referenceFace.count; side++) {
        uint ignoredFeature;
        float3 p0 = npcPolyFaceVertex(
            referenceShape, referenceFace, side,
            colliderConvexAssetID, convexHulls, convexFaceVertexIndices,
            convexHullVertices, ignoredFeature);
        float3 p1 = npcPolyFaceVertex(
            referenceShape, referenceFace,
            (side + 1u) % referenceFace.count,
            colliderConvexAssetID, convexHulls, convexFaceVertexIndices,
            convexHullVertices, ignoredFeature);
        float3 sideNormal = npcSafeNormalize(
            cross(p1 - p0, referenceFace.normal), float3(1, 0, 0));
        uint edgeFeature = npcFaceEdgeFeature(
            referenceShape, referenceFace.localIndex, side);
        if ((side & 1u) == 0u) {
            polygonCount = npcClipPolygon(
                polygonA, polygonCount, sideNormal, dot(sideNormal, p0),
                edgeFeature, polygonB);
        } else {
            polygonCount = npcClipPolygon(
                polygonB, polygonCount, sideNormal, dot(sideNormal, p0),
                edgeFeature, polygonA);
        }
        if (polygonCount == 0) break;
    }

    thread NPCClipVertex* clipped = (referenceFace.count & 1u) == 0u
        ? polygonA : polygonB;
    NPCManifoldContact candidates[NPC_MAX_FACE_VERTICES];
    int candidateCount = 0;
    float scale = max(1.0e-3f,
        max(fabs(a.dimensions.w), fabs(b.dimensions.w)));
    float mergeDistance = max(1.0e-6f, scale * 2.0e-5f);
    float faceSlop = detectDistance + max(PLANE_EPS, scale * 1.0e-5f);
    for (int index = 0; index < polygonCount; index++) {
        float3 incidentPoint = clipped[index].point;
        float separation = dot(referenceFace.normal, incidentPoint)
            - referenceFace.distance;
        if (separation > faceSlop) continue;
        float3 referencePoint = incidentPoint
            - referenceFace.normal * separation;
        float3 pointA = referenceIsA ? referencePoint : incidentPoint;
        float3 pointB = referenceIsA ? incidentPoint : referencePoint;
        uint2 features = referenceIsA
            ? uint2(clipped[index].referenceFeature,
                    clipped[index].incidentFeature)
            : uint2(clipped[index].incidentFeature,
                    clipped[index].referenceFeature);
        npcAddManifoldContact(
            candidates, candidateCount, pointA, pointB, features,
            mergeDistance * mergeDistance);
    }
    if (candidateCount == 0) {
        if (haveEdge) { output[0] = edgeContact; return 1; }
        return 0;
    }
    return npcReduceManifold(candidates, candidateCount, -normalAtoB, output);
}

// ---------------------------------------------------------------------------
// Torus collision helpers (implicit surface, alternating projection)
// ---------------------------------------------------------------------------


// Kinematic spinner surfaces move between steps but contact anchors are
// re-detected each frame, so friction alone cannot convey. Shift the
// tangential constraint target by the known surface displacement so the
// friction constraint tracks the moving surface.
inline float2 spinSurfaceShift(float4 wA, float4 wB,
                               float3 rAoff, float3 rBoff,
                               float3 t1, float3 t2, float dt, float alpha) {
    float3 sA = wA.w != 0.0f ? cross(wA.xyz, rAoff) * dt : float3(0);
    float3 sB = wB.w != 0.0f ? cross(wB.xyz, rBoff) * dt : float3(0);
    float3 rel = sB - sA;
    // C0 is alpha-discounted by the solver (it treats it as pre-existing
    // error); the surface shift is a TARGET, so pre-compensate.
    float k = 1.0f / max(1.0f - alpha, 0.01f);
    return float2(dot(t1, rel), dot(t2, rel)) * k;
}

inline float3 npClosestOnSegment(float3 p, float3 a, float3 b) {
    float3 ab = b - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12f), 0.0f, 1.0f);
    return a + ab * t;
}

struct TorusFrame {
    float3 center;      // world
    float3 axis;        // world (unit)
    float3 u, v;        // spine plane basis
    float R, r;
};

inline TorusFrame torusFrame(float3 pos, float4 q, float R, float r) {
    TorusFrame t;
    t.center = pos;
    t.axis = q_rotate(q, float3(0, 0, 1));
    t.u = abs(t.axis.x) > abs(t.axis.z) ? float3(-t.axis.y, t.axis.x, 0)
                                        : float3(0, -t.axis.z, t.axis.y);
    t.u = normalize(t.u);
    t.v = cross(t.axis, t.u);
    t.R = R;
    t.r = r;
    return t;
}

// stable contact feature: quantized spine angle in the torus body frame
inline uint torusFeature(TorusFrame t, float3 spinePoint) {
    float3 d = spinePoint - t.center;
    float au = dot(d, t.u);
    float av = dot(d, t.v);
    float ang = atan2(av, au) + M_PI_F;
    return uint(clamp(ang / (2.0f * M_PI_F) * 8.0f, 0.0f, 7.999f));
}

inline float3 torusProjSpine(TorusFrame t, float3 p) {
    float3 d = p - t.center;
    d -= t.axis * dot(t.axis, d);
    float l = length(d);
    return l < 1e-6f ? t.center + t.u * t.R : t.center + d / l * t.R;
}

// number of contacts written into scratch (xA on first shape, xB on second)
struct TorusHit {
    float3 xT;          // point on torus surface
    float3 xO;          // point on other shape
    float3 n;           // other -> torus
    float separation;   // signed surface gap (negative = penetration)
    uint feature;
};

// Compile the analytic and support-mapped paths as different Metal kernels.
// `CONVEX_PASS` is a template constant, so the analytic specialization does
// not retain MPR/GJK, clipping workspaces, or convex-only buffer accesses in
// its generated code. Both passes still consume the same deterministic pair
// stream and address the same manifold slot by `gid`.
template <bool CONVEX_PASS>
inline void npCollidePass(
    device const float4* posLin,
    device const float4* posAng,
    device const float4* shape,      // xyz size, w radius
    device const float4* props,      // xyz moment, w friction
    device const uint2* pairs,
    device atomic_uint* counters,
    device ManifoldGPU* manifolds,
    device const ManifoldGPU* prevManifolds,
    device const atomic_uint* mapKeyA,
    device const uint* mapKeyB,
    device const uint* mapVal,
    constant SimParams& P,
    device const uint* shapeType,
    device const float4* spinVel,
    device const float4* velLin,
    device const float4* velAng,
    device const uint* colliderOwner,
    device const float4* colliderLocalPosition,
    device const float4* colliderLocalRotation,
    device const uint2* colliderHullRange,
    device const float4* convexHullVertices,
    device const float2* colliderFriction,
    device const uint* colliderConvexAssetID,
    device const ConvexHullGPU* convexHulls,
    device const ConvexFaceGPU* convexFaces,
    device const uint* convexFaceVertexIndices,
    device const ConvexEdgeGPU* convexEdges,
    device uint2* contactFeatures,
    device const uint2* prevContactFeatures,
    device atomic_uint* convexQueryPoison,
    uint gid)
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    if (gid >= numPairs) return;

    uint2 pair = pairs[gid];
    uint ia = pair.x, ib = pair.y;

#if defined(AVBD_OPTIMIZED_CONVEX)
    // Both specializations consume the same deterministic pair stream, but
    // only one owns each manifold slot. Reject the other pass before loading
    // owners, body poses, velocities, collider transforms, or constructing
    // OBBs. Mixed scenes therefore pay two tiny kind checks per pair rather
    // than duplicating the full common narrow-phase prologue.
    uint stA = shapeType[ia] & SHAPE_KIND_MASK;
    uint stB = shapeType[ib] & SHAPE_KIND_MASK;
    bool hullA = stA == 4;
    bool hullB = stB == 4;
    bool hullPair = hullA || hullB;
    if (CONVEX_PASS != hullPair) return;
#endif

    uint ba = colliderOwner[ia], bb = colliderOwner[ib];

    float4 bodyPA4 = posLin[ba];
    float4 bodyPB4 = posLin[bb];
    float4 qBodyA = posAng[ba];
    float4 qBodyB = posAng[bb];
    float3 centerA = bodyPA4.xyz
        + q_rotate(qBodyA, colliderLocalPosition[ia].xyz);
    float3 centerB = bodyPB4.xyz
        + q_rotate(qBodyB, colliderLocalPosition[ib].xyz);
    float4 pA4 = float4(centerA, bodyPA4.w);
    float4 pB4 = float4(centerB, bodyPB4.w);
    float4 qA = q_mul(qBodyA, colliderLocalRotation[ia]);
    float4 qB = q_mul(qBodyB, colliderLocalRotation[ib]);
    float3 vA = velLin[ba].xyz + cross(velAng[ba].xyz, centerA - bodyPA4.xyz);
    float3 vB = velLin[bb].xyz + cross(velAng[bb].xyz, centerB - bodyPB4.xyz);
    float3 relVel = vB - vA;

    NPBox A, B;
    A.center = pA4.xyz; A.half3_ = shape[ia].xyz * 0.5f;
    A.ax0 = q_rotate(qA, float3(1,0,0)); A.ax1 = q_rotate(qA, float3(0,1,0)); A.ax2 = q_rotate(qA, float3(0,0,1));
    B.center = pB4.xyz; B.half3_ = shape[ib].xyz * 0.5f;
    B.ax0 = q_rotate(qB, float3(1,0,0)); B.ax1 = q_rotate(qB, float3(0,1,0)); B.ax2 = q_rotate(qB, float3(0,0,1));

    float3 delta = B.center - A.center;

    device ManifoldGPU& outM = manifolds[gid];
    outM.colliderPair = uint4(ia, ib, 0, 0);

    // --- shape type dispatch (0 box, 1 sphere, 2 torus, 3 capsule) ---
#if !defined(AVBD_OPTIMIZED_CONVEX)
    uint stA = shapeType[ia] & SHAPE_KIND_MASK;
    uint stB = shapeType[ib] & SHAPE_KIND_MASK;
#endif
    bool sphA = stA == 1;
    bool sphB = stB == 1;
    bool torA = stA == 2;
    bool torB = stB == 2;
    bool capA = stA == 3;
    bool capB = stB == 3;
#if !defined(AVBD_OPTIMIZED_CONVEX)
    bool hullA = stA == 4;
    bool hullB = stB == 4;

    if (hullA || hullB) {
        if (!CONVEX_PASS) return;
#else
    if (hullPair) {
#endif
        NPCShape convexA;
        convexA.collider = ia;
        convexA.kind = stA;
        convexA.center = centerA;
        convexA.rotation = qA;
        convexA.dimensions = shape[ia];
        NPCShape convexB;
        convexB.collider = ib;
        convexB.kind = stB;
        convexB.center = centerB;
        convexB.rotation = qB;
        convexB.dimensions = shape[ib];

        NPCResult result = npcMPR(
            convexA, convexB, colliderHullRange, convexHullVertices);
        if (result.valid && result.overlap
            && !npcCorrectedMPRIsConsistent(result)) {
            // The bounded portal walk may return its last simplex at the
            // iteration limit. Only admit that standard witness when the
            // corrected undilated points, normal, and signed distance agree;
            // otherwise continue through the certified GJK/retry boundary.
            result.valid = false;
        }
        if (!result.valid || !result.overlap) {
            float maxDetect = P.collisionMargin
                + npSpeculativeCap(shape[ia].w, shape[ib].w,
                                   P.collisionMargin);
            result = npcGJK(convexA, convexB, maxDetect,
                            colliderHullRange, convexHullVertices);
            // GJK is only the separated-distance fallback. It deliberately
            // does not manufacture a penetration witness; if it discovers an
            // overlap after MPR was inconclusive, dropping the pair would let
            // bodies pass through one another. Latch the frame-wide failure so
            // the host retires it as typed commandExecution and the finalizer
            // restores every dynamic pose.
            if (!result.valid || result.overlap) {
                NPCResult failedGJK = result;
                // MPR is geometrically symmetric but its finite portal walk
                // is operand ordered. Retry the same Newton query with the
                // operands reversed before changing its inflation, then map
                // the finite witness back to the caller's A/B convention.
                NPCResult swapped = npcMPR(
                    convexB, convexA, colliderHullRange, convexHullVertices);
                bool recovered = false;
                if (swapped.valid && swapped.overlap) {
                    NPCResult mapped = swapped;
                    mapped.pointA = swapped.pointB;
                    mapped.pointB = swapped.pointA;
                    mapped.normalAB = -swapped.normalAB;
                    mapped.featureA = swapped.featureB;
                    mapped.featureB = swapped.featureA;
                    if (npcCorrectedMPRIsConsistent(mapped)) {
                        result = mapped;
                        recovered = true;
                    }
                }
                if (!recovered) {
                    // The failed simplex distance is a conservative upper
                    // bound on true separation. Twice that bound moves the
                    // origin well inside the dilated CSO instead of starting
                    // another portal walk at the same numerical boundary.
                    // The speculative query band caps this one-shot retry;
                    // witnesses are corrected to the undilated surfaces and
                    // certified below before the normal distance gate runs.
                    float adaptiveEnlarge = min(
                        2.0f * maxDetect + 1.0e-4f,
                        max(1.0e-3f,
                            2.0f * failedGJK.signedDistance + 1.0e-4f));
                    NPCResult retry = npcMPRWithEnlarge(
                        convexA, convexB, colliderHullRange,
                        convexHullVertices, adaptiveEnlarge);
                    if (finite_bits(failedGJK.signedDistance)
                        && failedGJK.signedDistance >= 0.0f
                        && adaptiveEnlarge
                            > failedGJK.signedDistance + 5.0e-5f
                        && npcCorrectedMPRIsConsistent(retry)) {
                        result = retry;
                        recovered = true;
                    }
                }
                if (!recovered) {
                    latchConvexQueryFailure(counters, convexQueryPoison);
                    outM.header = uint4(ba, bb, 0, 0);
                    return;
                }
            }
        }

        float3 normalAtoB = npcSafeNormalize(result.normalAB,
            npcSafeNormalize(centerB - centerA, float3(0, 0, 1)));
        float3 normal = -normalAtoB;            // solver convention B -> A
        normal = npcSafeNormalize(normal,
            npcSafeNormalize(centerA - centerB, float3(0, 0, 1)));
        float detect = npDetectMargin(normal, vA - vB, P,
                                      shape[ia].w, shape[ib].w);
        if (result.signedDistance > detect) {
            outM.header = uint4(ba, bb, 0, 0);
            return;
        }

        NPCManifoldContact convexContacts[MAX_CONTACTS];
        int contactCount = 0;
        uint convexEdgePairTests = 0u;
        bool polyA = stA == 0u || stA == 4u;
        bool polyB = stB == 0u || stB == 4u;
        if (polyA && polyB) {
            contactCount = npcBuildPolyhedralManifold(
                convexA, convexB, normalAtoB, detect,
                colliderHullRange, convexHullVertices,
                colliderConvexAssetID, convexHulls, convexFaces,
                convexFaceVertexIndices, convexEdges, convexContacts,
                convexEdgePairTests);
        }
        if (convexEdgePairTests > 0u) {
            atomic_fetch_add_explicit(
                &counters[CTR_CONVEX_EDGE_PAIRS], convexEdgePairTests,
                memory_order_relaxed);
        }
        if (contactCount == 0) {
            convexContacts[0].pointA = result.pointA;
            convexContacts[0].pointB = result.pointB;
            convexContacts[0].features = uint2(
                result.featureA, result.featureB);
            contactCount = 1;
        }

        float3 t1, t2;
        orthonormal(normal, t1, t2);
        float staticFriction = combine_friction(
            colliderFriction[ia].x, colliderFriction[ib].x,
            P.frictionCombineMode);
        float dynamicFriction = combine_friction(
            colliderFriction[ia].y, colliderFriction[ib].y,
            P.frictionCombineMode);
        int prevIdx = pairMapFind(mapKeyA, mapKeyB, mapVal,
                                  P.mapCapacity, ia, ib);
        bool roundA = (shapeType[ia] & COLLIDER_WORLD_ROUND_ANCHOR) != 0;
        bool roundB = (shapeType[ib] & COLLIDER_WORLD_ROUND_ANCHOR) != 0;
        uint flags = 1u | (roundA ? 2u : 0u) | (roundB ? 4u : 0u);
        outM.header = uint4(ba, bb, uint(contactCount), flags);
        outM.basisN = float4(normal, dynamicFriction);
        outM.basisT1 = float4(t1, staticFriction);

        float4 qAc = q_conj(qBodyA);
        float4 qBc = q_conj(qBodyB);
        uint matchedPrevious = 0u;
        for (int contactIndex = 0; contactIndex < contactCount;
             contactIndex++) {
            float3 pointA = convexContacts[contactIndex].pointA;
            float3 pointB = convexContacts[contactIndex].pointB;
            uint2 featurePair = convexContacts[contactIndex].features;
            float3 rA = roundA ? (pointA - bodyPA4.xyz)
                : q_rotate(qAc, pointA - bodyPA4.xyz);
            float3 rB = roundB ? (pointB - bodyPB4.xyz)
                : q_rotate(qBc, pointB - bodyPB4.xyz);
            float3 lambda = float3(0);
            float3 penalty = float3(0);
            float stick = 0.0f;

            if (prevIdx >= 0) {
                device const ManifoldGPU& previous = prevManifolds[prevIdx];
                int best = -1;
                for (uint j = 0; j < previous.header.z; j++) {
                    if ((matchedPrevious & (1u << j)) == 0u
                        && all(prevContactFeatures[
                            uint(prevIdx) * MAX_CONTACTS + j]
                            == featurePair)) {
                        best = int(j);
                        break;
                    }
                }
                // Smooth shapes and clipped edge transitions may legitimately
                // change exact feature IDs. Use a one-to-one local-anchor
                // fallback only while the manifold normal remains compatible.
                if (best < 0 && dot(previous.basisN.xyz, normal) > 0.95f) {
                    float threshold = 0.08f * max(
                        fabs(shape[ia].w), fabs(shape[ib].w))
                        + 2.0f * P.collisionMargin;
                    float bestDistance = threshold;
                    for (uint j = 0; j < previous.header.z; j++) {
                        if ((matchedPrevious & (1u << j)) != 0u) continue;
                        float distanceA = distance(
                            previous.contacts[j].rA.xyz, rA);
                        float distanceB = distance(
                            previous.contacts[j].rB.xyz, rB);
                        float anchorDistance = max(distanceA, distanceB);
                        if (anchorDistance < bestDistance - 1.0e-7f
                            || (fabs(anchorDistance - bestDistance) <= 1.0e-7f
                                && int(j) < best)) {
                            bestDistance = anchorDistance;
                            best = int(j);
                        }
                    }
                }
                if (best >= 0) {
                    matchedPrevious |= 1u << uint(best);
                    device const ContactGPU& old = previous.contacts[best];
                    lambda = old.lambda.xyz;
                    penalty = old.penalty.xyz;
                    float3 oldT1 = previous.basisT1.xyz;
                    float3 oldT2 = cross(previous.basisN.xyz, oldT1);
                    float3 worldTangent = oldT1 * lambda.y
                        + oldT2 * lambda.z;
                    lambda.y = dot(t1, worldTangent);
                    lambda.z = dot(t2, worldTangent);
                    if (!roundA && !roundB) {
                        stick = old.rB.w;
                        if (stick != 0.0f) {
                            rA = old.rA.xyz;
                            rB = old.rB.xyz;
                        }
                    }
                }
            }

            float3 xAw = roundA ? bodyPA4.xyz + rA
                : xform(bodyPA4.xyz, qBodyA, rA);
            float3 xBw = roundB ? bodyPB4.xyz + rB
                : xform(bodyPB4.xyz, qBodyB, rB);
            float3 d = xAw - xBw;
            float3 C0 = float3(dot(normal, d) + P.collisionMargin,
                               dot(t1, d), dot(t2, d));
            C0.yz -= spinSurfaceShift(spinVel[ba], spinVel[bb],
                                      xAw - bodyPA4.xyz,
                                      xBw - bodyPB4.xyz,
                                      t1, t2, P.dt, P.alpha);
            lambda *= P.alpha * P.gamma;
            penalty = clamp(penalty * P.gamma,
                            npPenaltyFloor(posLin, ba, bb, P),
                            npPenaltyCeil());
            outM.contacts[contactIndex].rA = float4(
                rA, float(npcFeaturePairHash(featurePair) & 0x00FFFFFFu));
            outM.contacts[contactIndex].rB = float4(rB, stick);
            outM.contacts[contactIndex].C0 = float4(C0, 0);
            outM.contacts[contactIndex].lambda = float4(lambda, 0);
            outM.contacts[contactIndex].penalty = float4(penalty, 0);
            contactFeatures[gid * MAX_CONTACTS + uint(contactIndex)]
                = featurePair;
        }
        return;
    }

    // The convex specialization visits the shared pair stream so manifold
    // indices remain stable, but analytic pairs belong exclusively to the
    // legacy specialized kernel above.
#if !defined(AVBD_OPTIMIZED_CONVEX)
    if (CONVEX_PASS) return;
#endif

    // capsule pairs not involving a torus: single-contact closest-point
    if ((capA || capB) && !torA && !torB) {
        bool cIsA = capA;
        uint ic = cIsA ? ia : ib;
        uint io = cIsA ? ib : ia;
        float4 qC = cIsA ? qA : qB;
        float4 qO = cIsA ? qB : qA;
        float3 pC = cIsA ? pA4.xyz : pB4.xyz;
        float3 pO = cIsA ? pB4.xyz : pA4.xyz;
        float half_ = shape[ic].x * 0.5f;
        float rc = shape[ic].y;
        float3 axC = q_rotate(qC, float3(0, 0, 1));
        float3 c0 = pC - axC * half_;
        float3 c1 = pC + axC * half_;
        uint otherType = cIsA ? stB : stA;

        NPContact hits2[3];
        int nh = 0;
        float3 nCO = float3(0, 0, 1);    // capsule -> other

        if (otherType == 1) {
            float rs = shape[io].x * 0.5f;
            float3 q = npClosestOnSegment(pO, c0, c1);
            float3 d = pO - q;
            float dist = length(d);
            float3 nTmp = dist > 1e-6f ? d / dist : float3(0, 0, 1);
            float3 relCO = cIsA ? (vB - vA) : (vA - vB);
            float detectM = npDetectMargin(nTmp, relCO, P,
                                           shape[ic].w, shape[io].w);
            if (dist <= rc + rs + detectM) {
                nCO = nTmp;
                hits2[0].xA = q + nCO * rc;        // on capsule
                hits2[0].xB = pO - nCO * rs;       // on sphere
                hits2[0].feature = 0;
                nh = 1;
            }
        } else if (otherType == 3) {
            float halfO = shape[io].x * 0.5f;
            float ro = shape[io].y;
            float3 axO = q_rotate(qO, float3(0, 0, 1));
            float3 o0 = pO - axO * halfO, o1 = pO + axO * halfO;
            float3 qa, qb;
            npClosestSegSeg(c0, c1, o0, o1, qa, qb);
            float3 d = qb - qa;
            float dist = length(d);
            float3 nTmp = dist > 1e-6f ? d / dist : float3(0, 0, 1);
            float3 relCO = cIsA ? (vB - vA) : (vA - vB);
            float detectM = npDetectMargin(nTmp, relCO, P,
                                           shape[ic].w, shape[io].w);
            if (dist <= rc + ro + detectM) {
                nCO = nTmp;
                hits2[0].xA = qa + nCO * rc;
                hits2[0].xB = qb - nCO * ro;
                hits2[0].feature = 0;
                nh = 1;
            }
        } else {
            // capsule - box: alternating clamp <-> segment, 3 seeds
            float4 qOc = q_conj(qO);
            float3 halfO = shape[io].xyz * 0.5f;
            float3 nAccum = float3(0);
            for (int seedI = 0; seedI < 3 && nh < 3; seedI++) {
                float t = float(seedI) * 0.5f;
                float3 q = c0 + (c1 - c0) * t;
                float3 bw = q;
                for (int it2 = 0; it2 < 5; it2++) {
                    float3 lb = clamp(q_rotate(qOc, q - pO), -halfO, halfO);
                    bw = q_rotate(qO, lb) + pO;
                    q = npClosestOnSegment(bw, c0, c1);
                }
                float3 d = bw - q;
                float dist = length(d);
                float3 n;
                if (dist < 1e-6f) {
                    float3 lb = q_rotate(qOc, q - pO);
                    float3 pen = halfO - fabs(lb);
                    int axisI = 0;
                    if (pen.y < pen[axisI]) axisI = 1;
                    if (pen.z < pen[axisI]) axisI = 2;
                    float3 nl = float3(0);
                    nl[axisI] = lb[axisI] >= 0.0f ? 1.0f : -1.0f;
                    n = -q_rotate(qO, nl);            // capsule -> box
                    bw = q - n * pen[axisI];
                } else {
                    n = d / dist;
                    float3 relCO = cIsA ? (vB - vA) : (vA - vB);
                    float detectM = npDetectMargin(n, relCO, P,
                                                   shape[ic].w, shape[io].w);
                    if (dist - rc > detectM) continue;
                }
                bool dup = false;
                for (int h = 0; h < nh; h++) {
                    // Scale-aware: a flat 5 cm dedup radius is longer than
                    // a short capsule, so every seed collapsed onto one
                    // point and a small capsule rested on a single-point
                    // axle instead of a line.
                    if (distance(hits2[h].xA, q + n * rc)
                        < 0.4f * rc + min(0.05f, 0.25f * half_)) { dup = true; break; }
                }
                if (dup) continue;
                hits2[nh].xA = q + n * rc;
                hits2[nh].xB = bw;
                hits2[nh].feature = uint(seedI);
                nAccum += n;
                nh++;
            }
            if (nh > 0) nCO = normalize(nAccum);
        }

        if (nh == 0) {
            outM.header = uint4(ba, bb, 0, 0);
            return;
        }

        float3 nrmC = cIsA ? -nCO : nCO;          // B -> A
        float3 t1, t2;
        orthonormal(nrmC, t1, t2);
        float staticFriction = combine_friction(
            colliderFriction[ia].x, colliderFriction[ib].x,
            P.frictionCombineMode);
        float dynamicFriction = combine_friction(
            colliderFriction[ia].y, colliderFriction[ib].y,
            P.frictionCombineMode);
        int prevIdx = pairMapFind(mapKeyA, mapKeyB, mapVal, P.mapCapacity, ia, ib);
        bool roundA = (shapeType[ia] & COLLIDER_WORLD_ROUND_ANCHOR) != 0;
        bool roundB = (shapeType[ib] & COLLIDER_WORLD_ROUND_ANCHOR) != 0;
        uint flags = 1u | (roundA ? 2u : 0u) | (roundB ? 4u : 0u);
        outM.header = uint4(ba, bb, uint(nh), flags);
        outM.basisN = float4(nrmC, dynamicFriction);
        outM.basisT1 = float4(t1, staticFriction);
        float4 qAc = q_conj(qBodyA);
        float4 qBc = q_conj(qBodyB);
        float warm = P.alpha * P.gamma;
        for (int i = 0; i < nh; i++) {
            float3 xAw = cIsA ? hits2[i].xA : hits2[i].xB;
            float3 xBw = cIsA ? hits2[i].xB : hits2[i].xA;
            float3 rA_ = roundA ? (xAw - bodyPA4.xyz)
                                : q_rotate(qAc, xAw - bodyPA4.xyz);
            float3 rB_ = roundB ? (xBw - bodyPB4.xyz)
                                : q_rotate(qBc, xBw - bodyPB4.xyz);
            float3 lambda = float3(0);
            float3 penalty = float3(0);
            float stick = 0.0f;
            if (prevIdx >= 0) {
                device const ManifoldGPU& pm = prevManifolds[prevIdx];
                uint pn = pm.header.z;
                float bestD = 0.6f * (shape[ic].y + 0.2f);
                int bestJ = -1;
                for (uint j = 0; j < pn; j++) {
                    float dj = distance(pm.contacts[j].rA.xyz, rA_);
                    if (dj < bestD) { bestD = dj; bestJ = int(j); }
                }
                if (bestJ >= 0) {
                    lambda = pm.contacts[bestJ].lambda.xyz;
                    penalty = pm.contacts[bestJ].penalty.xyz;
                    // transport the tangential dual into the new basis:
                    // proximity-matched contacts slide around round shapes,
                    // and a rotated basis misdirects the carried friction
                    float3 t1o = pm.basisT1.xyz;
                    float3 t2o = cross(pm.basisN.xyz, t1o);
                    float3 lt = t1o * lambda.y + t2o * lambda.z;
                    lambda.y = dot(t1, lt);
                    lambda.z = dot(t2, lt);
                    // A centered standalone round primitive deliberately
                    // uses a world-offset anchor so it can roll. Compound
                    // robot colliders use body-local material anchors: while
                    // the contact is inside the Coulomb cone, retain those
                    // anchors exactly as the box manifold does. Rebuilding
                    // them from the closest points every frame erases the
                    // accumulated tangential displacement and permits creep.
                    if (!roundA && !roundB) {
                        stick = pm.contacts[bestJ].rB.w;
                        if (stick != 0.0f) {
                            rA_ = pm.contacts[bestJ].rA.xyz;
                            rB_ = pm.contacts[bestJ].rB.xyz;
                        }
                    }
                }
            }
            xAw = roundA ? bodyPA4.xyz + rA_
                         : xform(bodyPA4.xyz, qBodyA, rA_);
            xBw = roundB ? bodyPB4.xyz + rB_
                         : xform(bodyPB4.xyz, qBodyB, rB_);
            float3 d = xAw - xBw;
            float3 C0 = float3(dot(nrmC, d) + P.collisionMargin,
                               dot(t1, d), dot(t2, d));
            C0.yz -= spinSurfaceShift(spinVel[ba], spinVel[bb],
                                      xAw - bodyPA4.xyz, xBw - bodyPB4.xyz,
                                      t1, t2, P.dt, P.alpha);
            lambda *= warm;
            penalty = clamp(penalty * P.gamma,
                            npPenaltyFloor(posLin, ba, bb, P), npPenaltyCeil());
            outM.contacts[i].rA = float4(rA_, as_type<float>(hits2[i].feature));
            outM.contacts[i].rB = float4(rB_, stick);
            outM.contacts[i].C0 = float4(C0, 0);
            outM.contacts[i].lambda = float4(lambda, 0);
            outM.contacts[i].penalty = float4(penalty, 0);
        }
        return;
    }

    if (torA || torB) {
        TorusHit hits[4];
        int nHits = 0;
        bool tIsA = torA;
        uint it = tIsA ? ia : ib;       // torus body
        uint io = tIsA ? ib : ia;       // other body
        float4 qT = tIsA ? qA : qB;
        float4 qO = tIsA ? qB : qA;
        float3 pT = tIsA ? pA4.xyz : pB4.xyz;
        float3 pO = tIsA ? pB4.xyz : pA4.xyz;
        TorusFrame T = torusFrame(pT, qT, shape[it].x, shape[it].y);

        uint otherType = tIsA ? stB : stA;
        // detection margin: catch fast approaches before deep penetration
        float detectM = max(P.collisionMargin, 0.8f * T.r);

        if (otherType == 1) {
            // torus - sphere (exact closed form)
            float rs = shape[io].x * 0.5f;
            float3 spine = torusProjSpine(T, pO);
            float3 d = pO - spine;
            float dist = length(d);
            if (dist <= T.r + rs + detectM) {
                float3 n = dist > 1e-6f ? d / dist : float3(0, 0, 1); // torus -> sphere
                hits[0].xT = spine + n * T.r;
                hits[0].xO = pO - n * rs;
                hits[0].n = -n;          // other -> torus
                hits[0].separation = dist - T.r - rs;
                hits[0].feature = 0;
                nHits = 1;
            }
        } else {
            // torus vs torus/box: sample the spine densely; each sample has
            // an EXACT closest point on the other shape. Pick the deepest
            // angularly-spread candidates, then refine with two alternating
            // projection steps. Deterministic and stable.
            #define NSAMP 32
            float dists[NSAMP];
            float3 closest[NSAMP];
            bool isBox = otherType == 0;
            bool isCap = otherType == 3;
            TorusFrame T2;
            float4 qOc = q_conj(qO);
            float3 halfO = shape[io].xyz * 0.5f;
            float otherR = 0.0f;
            float3 cs0 = float3(0), cs1 = float3(0);
            if (isCap) {
                float chalf = shape[io].x * 0.5f;
                float3 axO = q_rotate(qO, float3(0, 0, 1));
                cs0 = pO - axO * chalf;
                cs1 = pO + axO * chalf;
                otherR = shape[io].y;
            } else if (!isBox) {
                T2 = torusFrame(pO, qO, shape[io].x, shape[io].y);
                otherR = T2.r;
            }

            for (int i = 0; i < NSAMP; i++) {
                float ang = float(i) / float(NSAMP) * 2.0f * M_PI_F;
                float3 q = T.center + (T.u * cos(ang) + T.v * sin(ang)) * T.R;
                float3 b;
                if (isBox) {
                    b = q_rotate(qO, clamp(q_rotate(qOc, q - pO), -halfO, halfO)) + pO;
                } else if (isCap) {
                    b = npClosestOnSegment(q, cs0, cs1);
                } else {
                    b = torusProjSpine(T2, q);
                }
                closest[i] = b;
                dists[i] = distance(q, b);
            }

            // greedy: deepest first, then deepest with >=45 deg separation
            bool used[NSAMP];
            for (int i = 0; i < NSAMP; i++) used[i] = false;
            for (int pick = 0; pick < 4; pick++) {
                int best = -1;
                float bd = FLT_MAX;
                for (int i = 0; i < NSAMP; i++) {
                    if (used[i]) continue;
                    if (dists[i] < bd) { bd = dists[i]; best = i; }
                }
                if (best < 0) break;
                if (bd - T.r - otherR > detectM) break;
                // mask neighbors within 45 deg
                for (int k = -3; k <= 3; k++) {
                    used[(best + k + NSAMP) % NSAMP] = true;
                }

                // refine: two alternating projection steps from this sample
                float ang = float(best) / float(NSAMP) * 2.0f * M_PI_F;
                float3 q = T.center + (T.u * cos(ang) + T.v * sin(ang)) * T.R;
                float3 b = closest[best];
                for (int it2 = 0; it2 < 2; it2++) {
                    q = torusProjSpine(T, b);
                    if (isBox) {
                        b = q_rotate(qO, clamp(q_rotate(qOc, q - pO), -halfO, halfO)) + pO;
                    } else if (isCap) {
                        b = npClosestOnSegment(q, cs0, cs1);
                    } else {
                        b = torusProjSpine(T2, q);
                    }
                }
                float3 d = q - b;
                float dist = length(d);
                float3 n;
                float3 xO_ = b;
                if (dist < 1e-6f) {
                    if (isBox) {
                        // spine inside the box: face ejection with true depth
                        float3 lb = q_rotate(qOc, q - pO);
                        float3 pen = halfO - fabs(lb);
                        int axisI = 0;
                        if (pen.y < pen[axisI]) axisI = 1;
                        if (pen.z < pen[axisI]) axisI = 2;
                        float3 nl = float3(0);
                        nl[axisI] = lb[axisI] >= 0.0f ? 1.0f : -1.0f;
                        n = q_rotate(qO, nl);
                        xO_ = q + n * pen[axisI];
                    } else if (isCap) {
                        n = normalize(cross(T.axis, normalize(cs1 - cs0)) + float3(1e-4f, 0, 0));
                    } else {
                        n = normalize(cross(T.axis, T2.axis) + float3(1e-4f, 0, 0));
                    }
                } else {
                    if (dist - T.r - otherR > detectM) continue;
                    n = d / dist;                   // other -> torus
                }
                hits[nHits].xT = q - n * T.r;
                hits[nHits].xO = xO_ + ((isBox) ? float3(0) : n * otherR);
                hits[nHits].n = n;
                hits[nHits].separation = dist - T.r - otherR;
                hits[nHits].feature = uint(best);
                nHits++;
            }
            #undef NSAMP
        }

        if (nHits == 0) {
            outM.header = uint4(ba, bb, 0, 0);
            return;
        }

        // deepest-contact normal defines the basis; nrm points B -> A
        int deep = 0;
        float minSeparation = FLT_MAX;
        for (int h = 0; h < nHits; h++) {
            if (hits[h].separation < minSeparation) {
                minSeparation = hits[h].separation;
                deep = h;
            }
        }
        float3 nTorus = hits[deep].n;                  // other -> torus
        float3 nrmT = tIsA ? nTorus : -nTorus;         // B -> A

        float3 t1, t2;
        orthonormal(nrmT, t1, t2);
        float staticFriction = combine_friction(
            colliderFriction[ia].x, colliderFriction[ib].x,
            P.frictionCombineMode);
        float dynamicFriction = combine_friction(
            colliderFriction[ia].y, colliderFriction[ib].y,
            P.frictionCombineMode);
        int prevIdx = pairMapFind(mapKeyA, mapKeyB, mapVal, P.mapCapacity, ia, ib);

        bool roundA = (shapeType[ia] & COLLIDER_WORLD_ROUND_ANCHOR) != 0;
        bool roundB = (shapeType[ib] & COLLIDER_WORLD_ROUND_ANCHOR) != 0;
        uint flags = 1u | (roundA ? 2u : 0u) | (roundB ? 4u : 0u);
        outM.header = uint4(ba, bb, uint(nHits), flags);
        outM.basisN = float4(nrmT, dynamicFriction);
        outM.basisT1 = float4(t1, staticFriction);

        float4 qAc = q_conj(qBodyA);
        float4 qBc = q_conj(qBodyB);
        float warm = P.alpha * P.gamma;
        for (int i = 0; i < nHits; i++) {
            float3 xAw = tIsA ? hits[i].xT : hits[i].xO;
            float3 xBw = tIsA ? hits[i].xO : hits[i].xT;
            float3 rA_ = roundA ? (xAw - bodyPA4.xyz)
                                : q_rotate(qAc, xAw - bodyPA4.xyz);
            float3 rB_ = roundB ? (xBw - bodyPB4.xyz)
                                : q_rotate(qBc, xBw - bodyPB4.xyz);
            float3 lambda = float3(0);
            float3 penalty = float3(0);
            if (prevIdx >= 0) {
                // proximity match: contacts on round shapes wander around
                // the tube, so exact feature IDs flicker; inherit the
                // nearest previous contact's dual state instead.
                device const ManifoldGPU& pm = prevManifolds[prevIdx];
                uint pn = pm.header.z;
                float bestD = 1.1f * shape[it].x;       // generous: keep dual
                                                        // state through yanks
                int bestJ = -1;
                for (uint j = 0; j < pn; j++) {
                    float dj = distance(pm.contacts[j].rA.xyz, rA_);
                    if (dj < bestD) { bestD = dj; bestJ = int(j); }
                }
                if (bestJ >= 0) {
                    lambda = pm.contacts[bestJ].lambda.xyz;
                    penalty = pm.contacts[bestJ].penalty.xyz;
                    float3 t1o = pm.basisT1.xyz;
                    float3 t2o = cross(pm.basisN.xyz, t1o);
                    float3 lt = t1o * lambda.y + t2o * lambda.z;
                    lambda.y = dot(t1, lt);
                    lambda.z = dot(t2, lt);
                }
            }
            float3 d = xAw - xBw;
            float3 C0 = float3(dot(nrmT, d) + P.collisionMargin,
                               dot(t1, d), dot(t2, d));
            C0.yz -= spinSurfaceShift(spinVel[ba], spinVel[bb],
                                      xAw - bodyPA4.xyz, xBw - bodyPB4.xyz,
                                      t1, t2, P.dt, P.alpha);
            lambda *= warm;
            penalty = clamp(penalty * P.gamma,
                            npPenaltyFloor(posLin, ba, bb, P), npPenaltyCeil());
            outM.contacts[i].rA = float4(rA_, as_type<float>(hits[i].feature));
            outM.contacts[i].rB = float4(rB_, 0.0f);
            outM.contacts[i].C0 = float4(C0, 0);
            outM.contacts[i].lambda = float4(lambda, 0);
            outM.contacts[i].penalty = float4(penalty, 0);
        }
        return;
    }

    NPContact contacts[MAX_CONTACTS];
    int count = 0;
    float3 nrm;     // points from B toward A

    if (sphA || sphB) {
        float rA = shape[ia].x * 0.5f;
        float rB = shape[ib].x * 0.5f;
        if (sphA && sphB) {
            float3 d = A.center - B.center;
            float dist = length(d);
            float3 nTmp = dist > 1e-9f ? d / dist : float3(0, 0, 1);
            float detectM = npDetectMargin(nTmp, vA - vB, P, rA, rB);
            if (dist > rA + rB + detectM) {
                outM.header = uint4(ba, bb, 0, 0);
                return;
            }
            nrm = nTmp;
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
                float3 nLocalTmp = d / max(dist, 1e-9f);
                float3 nWTmp = q_rotate(qBox, nLocalTmp);      // box -> sphere
                float3 vSphere = sIsA ? vA : vB;
                float3 vBox = sIsA ? vB : vA;
                float detectM = npDetectMargin(nWTmp, vSphere - vBox, P,
                                               r, sIsA ? shape[ib].w : shape[ia].w);
                if (dist > r + detectM) {
                    outM.header = uint4(ba, bb, 0, 0);
                    return;
                }
                nLocal = nLocalTmp;
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
        float staticFriction = combine_friction(
            colliderFriction[ia].x, colliderFriction[ib].x,
            P.frictionCombineMode);
        float dynamicFriction = combine_friction(
            colliderFriction[ia].y, colliderFriction[ib].y,
            P.frictionCombineMode);
        int prevIdx = pairMapFind(mapKeyA, mapKeyB, mapVal, P.mapCapacity, ia, ib);

        bool roundA = (shapeType[ia] & COLLIDER_WORLD_ROUND_ANCHOR) != 0;
        bool roundB = (shapeType[ib] & COLLIDER_WORLD_ROUND_ANCHOR) != 0;
        uint flags = 1u | (roundA ? 2u : 0u) | (roundB ? 4u : 0u);
        outM.header = uint4(ba, bb, uint(count), flags);
        outM.basisN = float4(nrm, dynamicFriction);
        outM.basisT1 = float4(t1, staticFriction);

        float4 qAc = q_conj(qBodyA);
        float4 qBc = q_conj(qBodyB);
        float warm = P.alpha * P.gamma;
        for (int i = 0; i < count; i++) {
            float3 rA_ = roundA ? (contacts[i].xA - bodyPA4.xyz)
                                : q_rotate(qAc, contacts[i].xA - bodyPA4.xyz);
            float3 rB_ = roundB ? (contacts[i].xB - bodyPB4.xyz)
                                : q_rotate(qBc, contacts[i].xB - bodyPB4.xyz);
            float3 lambda = float3(0);
            float3 penalty = float3(0);
            float stick = 0.0f;
            if (prevIdx >= 0) {
                // proximity match (see torus tail): inherit nearest previous
                // contact's dual state; no stick-anchor restoration for
                // rolling shapes — EXCEPT particles: they carry no rotation,
                // so their world-offset anchors cannot swing. Without the
                // anchors, particles never latch static friction and creep
                // forever under any standing tangential load (cloth corners
                // pulled by a draped skirt jitter perpetually).
                device const ManifoldGPU& pm = prevManifolds[prevIdx];
                uint pn = pm.header.z;
                float bestD = 0.5f * max(rA, rB);
                int bestJ = -1;
                for (uint j = 0; j < pn; j++) {
                    float dj = distance(pm.contacts[j].rA.xyz, rA_);
                    if (dj < bestD) { bestD = dj; bestJ = int(j); }
                }
                if (bestJ >= 0) {
                    lambda = pm.contacts[bestJ].lambda.xyz;
                    penalty = pm.contacts[bestJ].penalty.xyz;
                    float3 t1o = pm.basisT1.xyz;
                    float3 t2o = cross(pm.basisN.xyz, t1o);
                    float3 lt = t1o * lambda.y + t2o * lambda.z;
                    lambda.y = dot(t1, lt);
                    lambda.z = dot(t2, lt);
                    bool partA = shape[ia].w < 0.0f;
                    bool partB = shape[ib].w < 0.0f;
                    if ((partA || !roundA) && (partB || !roundB)) {
                        stick = pm.contacts[bestJ].rB.w;
                        if (stick != 0.0f) {
                            rA_ = pm.contacts[bestJ].rA.xyz;
                            rB_ = pm.contacts[bestJ].rB.xyz;
                        }
                    }
                }
            }
            float3 xAw = roundA ? bodyPA4.xyz + rA_
                                : xform(bodyPA4.xyz, qBodyA, rA_);
            float3 xBw = roundB ? bodyPB4.xyz + rB_
                                : xform(bodyPB4.xyz, qBodyB, rB_);
            float3 d = xAw - xBw;
            float3 C0 = float3(dot(nrm, d) + P.collisionMargin,
                               dot(t1, d), dot(t2, d));
            C0.yz -= spinSurfaceShift(spinVel[ba], spinVel[bb],
                                      xAw - bodyPA4.xyz, xBw - bodyPB4.xyz,
                                      t1, t2, P.dt, P.alpha);
            lambda *= warm;
            penalty = clamp(penalty * P.gamma,
                            npPenaltyFloor(posLin, ba, bb, P), npPenaltyCeil());
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
        if (!npTestAxis(A, B, delta, npAxis(A, i), 0, i, -1,
                        relVel, npSpeculativeCap(shape[ia].w, shape[ib].w,
                                                 P.collisionMargin),
                        P, bestFace)) separated = true;
    }
    for (int i = 0; i < 3 && !separated; i++) {
        if (!npTestAxis(A, B, delta, npAxis(B, i), 1, -1, i,
                        relVel, npSpeculativeCap(shape[ia].w, shape[ib].w,
                                                 P.collisionMargin),
                        P, bestFace)) separated = true;
    }
    for (int i = 0; i < 3 && !separated; i++) {
        for (int j = 0; j < 3 && !separated; j++) {
            float3 axis = cross(npAxis(A, i), npAxis(B, j));
            if (!npTestAxis(A, B, delta, axis, 2, i, j,
                            relVel, npSpeculativeCap(shape[ia].w, shape[ib].w,
                                                     P.collisionMargin),
                            P, bestEdge)) separated = true;
        }
    }

    if (separated || !bestFace.valid) {
        outM.header = uint4(ba, bb, 0, 0);
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

        float faceClipSlop = PLANE_EPS + max(0.0f, best.separation);
        for (int i = 0; i < n && count < MAX_CONTACTS; i++) {
            float3 pInc = poly0[i];
            float d = dot(pInc - refCenter, refNormal);
            if (d > faceClipSlop) continue;
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

    float staticFriction = combine_friction(
        colliderFriction[ia].x, colliderFriction[ib].x,
        P.frictionCombineMode);
    float dynamicFriction = combine_friction(
        colliderFriction[ia].y, colliderFriction[ib].y,
        P.frictionCombineMode);

    // Previous manifold for warm-start
    int prevIdx = pairMapFind(mapKeyA, mapKeyB, mapVal, P.mapCapacity, ia, ib);

    outM.header = uint4(ba, bb, uint(count), 1);
    outM.basisN = float4(nrm, dynamicFriction);
    outM.basisT1 = float4(t1, staticFriction);

    float4 qAc = q_conj(qBodyA);
    float4 qBc = q_conj(qBodyB);
    float warm = P.alpha * P.gamma;

    for (int i = 0; i < count; i++) {
        float3 rA = q_rotate(qAc, contacts[i].xA - bodyPA4.xyz);
        float3 rB = q_rotate(qBc, contacts[i].xB - bodyPB4.xyz);
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
        float3 xAw = xform(bodyPA4.xyz, qBodyA, rA);
        float3 xBw = xform(bodyPB4.xyz, qBodyB, rB);
        float3 d = xAw - xBw;
        float3 C0 = float3(dot(nrm, d) + P.collisionMargin,
                           dot(t1, d), dot(t2, d));
        C0.yz -= spinSurfaceShift(spinVel[ba], spinVel[bb],
                                  xAw - bodyPA4.xyz, xBw - bodyPB4.xyz,
                                  t1, t2, P.dt, P.alpha);

        // Warm-start (Eq. 19)
        lambda *= warm;
        penalty = clamp(penalty * P.gamma,
                        npPenaltyFloor(posLin, ba, bb, P), npPenaltyCeil());

        outM.contacts[i].rA = float4(rA, as_type<float>(contacts[i].feature));
        outM.contacts[i].rB = float4(rB, stick);
        outM.contacts[i].C0 = float4(C0, 0);
        outM.contacts[i].lambda = float4(lambda, 0);
        outM.contacts[i].penalty = float4(penalty, 0);
    }
}

kernel void np_collide(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device const float4* shape      [[buffer(2)]],
    device const float4* props      [[buffer(3)]],
    device const uint2* pairs       [[buffer(4)]],
    device atomic_uint* counters    [[buffer(5)]],
    device ManifoldGPU* manifolds   [[buffer(6)]],
    device const ManifoldGPU* prevManifolds [[buffer(7)]],
    device const atomic_uint* mapKeyA [[buffer(8)]],
    device const uint* mapKeyB      [[buffer(9)]],
    device const uint* mapVal       [[buffer(10)]],
    constant SimParams& P           [[buffer(11)]],
    device const uint* shapeType    [[buffer(12)]],
    device const float4* spinVel    [[buffer(13)]],
    device const float4* velLin     [[buffer(14)]],
    device const float4* velAng     [[buffer(15)]],
    device const uint* colliderOwner [[buffer(16)]],
    device const float4* colliderLocalPosition [[buffer(17)]],
    device const float4* colliderLocalRotation [[buffer(18)]],
    device const uint2* colliderHullRange [[buffer(19)]],
    device const float4* convexHullVertices [[buffer(20)]],
    device const float2* colliderFriction [[buffer(21)]],
    device const uint* colliderConvexAssetID [[buffer(22)]],
    device const ConvexHullGPU* convexHulls [[buffer(23)]],
    device const ConvexFaceGPU* convexFaces [[buffer(24)]],
    device const uint* convexFaceVertexIndices [[buffer(25)]],
    device const ConvexEdgeGPU* convexEdges [[buffer(26)]],
    device uint2* contactFeatures [[buffer(27)]],
    device const uint2* prevContactFeatures [[buffer(28)]],
    device atomic_uint* convexQueryPoison [[buffer(29)]],
    uint gid                        [[thread_position_in_grid]])
{
    npCollidePass<false>(
        posLin, posAng, shape, props, pairs, counters, manifolds,
        prevManifolds, mapKeyA, mapKeyB, mapVal, P, shapeType, spinVel,
        velLin, velAng, colliderOwner, colliderLocalPosition,
        colliderLocalRotation, colliderHullRange, convexHullVertices,
        colliderFriction, colliderConvexAssetID, convexHulls, convexFaces,
        convexFaceVertexIndices, convexEdges, contactFeatures,
        prevContactFeatures, convexQueryPoison, gid);
}

kernel void np_collide_convex(
    device const float4* posLin     [[buffer(0)]],
    device const float4* posAng     [[buffer(1)]],
    device const float4* shape      [[buffer(2)]],
    device const float4* props      [[buffer(3)]],
    device const uint2* pairs       [[buffer(4)]],
    device atomic_uint* counters    [[buffer(5)]],
    device ManifoldGPU* manifolds   [[buffer(6)]],
    device const ManifoldGPU* prevManifolds [[buffer(7)]],
    device const atomic_uint* mapKeyA [[buffer(8)]],
    device const uint* mapKeyB      [[buffer(9)]],
    device const uint* mapVal       [[buffer(10)]],
    constant SimParams& P           [[buffer(11)]],
    device const uint* shapeType    [[buffer(12)]],
    device const float4* spinVel    [[buffer(13)]],
    device const float4* velLin     [[buffer(14)]],
    device const float4* velAng     [[buffer(15)]],
    device const uint* colliderOwner [[buffer(16)]],
    device const float4* colliderLocalPosition [[buffer(17)]],
    device const float4* colliderLocalRotation [[buffer(18)]],
    device const uint2* colliderHullRange [[buffer(19)]],
    device const float4* convexHullVertices [[buffer(20)]],
    device const float2* colliderFriction [[buffer(21)]],
    device const uint* colliderConvexAssetID [[buffer(22)]],
    device const ConvexHullGPU* convexHulls [[buffer(23)]],
    device const ConvexFaceGPU* convexFaces [[buffer(24)]],
    device const uint* convexFaceVertexIndices [[buffer(25)]],
    device const ConvexEdgeGPU* convexEdges [[buffer(26)]],
    device uint2* contactFeatures [[buffer(27)]],
    device const uint2* prevContactFeatures [[buffer(28)]],
    device atomic_uint* convexQueryPoison [[buffer(29)]],
    uint gid                        [[thread_position_in_grid]])
{
    npCollidePass<true>(
        posLin, posAng, shape, props, pairs, counters, manifolds,
        prevManifolds, mapKeyA, mapKeyB, mapVal, P, shapeType, spinVel,
        velLin, velAng, colliderOwner, colliderLocalPosition,
        colliderLocalRotation, colliderHullRange, convexHullVertices,
        colliderFriction, colliderConvexAssetID, convexHulls, convexFaces,
        convexFaceVertexIndices, convexEdges, contactFeatures,
        prevContactFeatures, convexQueryPoison, gid);
}

// Test-only deterministic bounded-query regression. A zero-iteration budget
// must return invalid (never a guessed finite separation), which then exercises
// the same sticky counter, GPU restoration, and typed retirement path as a real
// exhausted MPR/GJK query.
kernel void convex_query_fail_for_testing(
    device atomic_uint* counters [[buffer(0)]],
    device const uint2* colliderHullRange [[buffer(1)]],
    device const float4* convexHullVertices [[buffer(2)]],
    device atomic_uint* convexQueryPoison [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]])
{
    if (gid == 0u) {
        NPCShape a;
        a.collider = 0u;
        a.kind = 0u;
        a.center = float3(0.0f);
        a.rotation = float4(0, 0, 0, 1);
        a.dimensions = float4(1, 1, 1, 1);
        NPCShape b = a;
        b.center = float3(2, 0, 0);
        NPCResult exhausted = npcGJKWithIterationLimit(
            a, b, 0.1f, colliderHullRange, convexHullVertices, 0);
        if (!exhausted.valid) {
            latchConvexQueryFailure(counters, convexQueryPoison);
        }
    }
}

// A support-query failure is terminal, but the host can only observe it after
// command completion. Restore all dynamic bodies on the GPU before velocity
// finalization so the failed frame is never externally visible in shared pose
// buffers. Static and kinematic authored motion remains untouched.
kernel void convex_restore_failed_frame(
    device float4* posLin                  [[buffer(0)]],
    device float4* posAng                  [[buffer(1)]],
    device const float4* initLin           [[buffer(2)]],
    device const float4* initAng           [[buffer(3)]],
    device const atomic_uint* convexQueryPoison [[buffer(4)]],
    constant SimParams& P                  [[buffer(5)]],
    uint gid                               [[thread_position_in_grid]])
{
    if (gid >= P.numBodies || posLin[gid].w <= 0.0f) return;
    if (atomic_load_explicit(convexQueryPoison,
                             memory_order_relaxed) == 0u) return;
    posLin[gid] = float4(initLin[gid].xyz, posLin[gid].w);
    posAng[gid] = initAng[gid];
}
