// Analytic-only compatibility path.
//
// Keep this kernel body and its 22-buffer ABI byte-for-byte aligned with the
// established origin/main narrow phase. Metal fast-math code generation for
// the gear train changes materially when the same analytic tail is wrapped in
// the generic-convex template (or even when an early hull guard is inserted).
// Hull-free scenes dispatch this direct kernel; mixed scenes dispatch the
// generic split in 30_narrowphase.metal. The gear-clock regression test is the
// executable compatibility boundary for this deliberately isolated copy.
struct NPConvexBoxCandidate {
    float3 xHull;
    float3 xBox;
    float3 boxToHull;
    float separation;
    uint feature;
};

// Exact cooked convex vertex against an oriented terrain/projectile box.
// For a point inside the box, project to its nearest exit face; outside,
// clamp to the box and retain speculative contacts inside the velocity-aware
// detection margin. H1 disables robot self-collision, so its ankle/torso
// hulls only require this path against the flat box terrain (and optional box
// projectiles). General convex-convex GJK/EPA remains a separate extension.
inline bool npConvexVertexBoxCandidate(
    float3 vertexWorld, uint feature,
    thread const NPBox& box, float4 qBox,
    float3 hullVelocityMinusBoxVelocity,
    float hullRadius, float boxRadius,
    constant SimParams& P,
    thread NPConvexBoxCandidate& result)
{
    float4 qInv = q_conj(qBox);
    float3 local = q_rotate(qInv, vertexWorld - box.center);
    float3 clamped = clamp(local, -box.half3_, box.half3_);
    float3 normalLocal;
    float3 boxLocal;
    float separation;
    bool inside = all(local >= -box.half3_) && all(local <= box.half3_);
    if (inside) {
        float3 clearance = box.half3_ - fabs(local);
        int axis = 0;
        if (clearance.y < clearance[axis]) axis = 1;
        if (clearance.z < clearance[axis]) axis = 2;
        normalLocal = float3(0);
        normalLocal[axis] = local[axis] >= 0.0f ? 1.0f : -1.0f;
        boxLocal = local;
        boxLocal[axis] = normalLocal[axis] * box.half3_[axis];
        separation = -clearance[axis];
    } else {
        float3 delta = local - clamped;
        float distance = length(delta);
        if (distance < 1e-8f) return false;
        normalLocal = delta / distance;
        float3 normalWorld = q_rotate(qBox, normalLocal);
        float detect = npDetectMargin(normalWorld,
                                      hullVelocityMinusBoxVelocity, P,
                                      hullRadius, boxRadius);
        if (distance > detect) return false;
        boxLocal = clamped;
        separation = distance;
    }
    result.xHull = vertexWorld;
    result.xBox = q_rotate(qBox, boxLocal) + box.center;
    result.boxToHull = q_rotate(qBox, normalLocal);
    result.separation = separation;
    result.feature = (4u << 24) | (feature & 0x00FFFFFFu);
    return true;
}

kernel void np_collide_analytic_compat(
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
    uint gid                        [[thread_position_in_grid]])
{
    uint numPairs = atomic_load_explicit(&counters[CTR_PAIRS], memory_order_relaxed);
    if (gid >= numPairs) return;

    uint2 pair = pairs[gid];
    uint ia = pair.x, ib = pair.y;
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
    uint stA = shapeType[ia] & SHAPE_KIND_MASK;
    uint stB = shapeType[ib] & SHAPE_KIND_MASK;
    bool sphA = stA == 1;
    bool sphB = stB == 1;
    bool torA = stA == 2;
    bool torB = stB == 2;
    bool capA = stA == 3;
    bool capB = stB == 3;
    bool hullA = stA == 4;
    bool hullB = stB == 4;

    if (hullA || hullB) {
        // The exact H1 contract only needs convex-vs-box. Refuse unsupported
        // pairs explicitly instead of silently treating hull AABBs as boxes.
        if ((hullA && hullB) || (hullA ? stB : stA) != 0) {
            outM.header = uint4(ba, bb, 0, 0);
            return;
        }
        bool hIsA = hullA;
        uint ih = hIsA ? ia : ib;
        float4 qH = hIsA ? qA : qB;
        float3 pH = hIsA ? pA4.xyz : pB4.xyz;
        thread const NPBox& box = hIsA ? B : A;
        float4 qBox = hIsA ? qB : qA;
        float3 relHullBox = hIsA ? (vA - vB) : (vB - vA);
        float hullRadius = fabs(shape[ih].w);
        float boxRadius = fabs(shape[hIsA ? ib : ia].w);
        uint2 range = colliderHullRange[ih];

        NPConvexBoxCandidate best;
        bool haveBest = false;
        for (uint vid = 0; vid < range.y; vid++) {
            float3 localVertex = convexHullVertices[range.x + vid].xyz;
            float3 worldVertex = pH + q_rotate(qH, localVertex);
            NPConvexBoxCandidate candidate;
            if (!npConvexVertexBoxCandidate(
                    worldVertex, vid, box, qBox, relHullBox,
                    hullRadius, boxRadius, P, candidate)) continue;
            if (!haveBest
                || candidate.separation < best.separation - 1e-8f
                || (fabs(candidate.separation - best.separation) <= 1e-8f
                    && candidate.feature < best.feature)) {
                best = candidate;
                haveBest = true;
            }
        }
        if (!haveBest) {
            outM.header = uint4(ba, bb, 0, 0);
            return;
        }

        NPConvexBoxCandidate chosen[MAX_CONTACTS];
        int count = 1;
        chosen[0] = best;
        float separationBand = 0.75f * P.collisionMargin;
        while (count < MAX_CONTACTS) {
            NPConvexBoxCandidate next;
            bool haveNext = false;
            float bestSpread = -1.0f;
            for (uint vid = 0; vid < range.y; vid++) {
                float3 localVertex = convexHullVertices[range.x + vid].xyz;
                float3 worldVertex = pH + q_rotate(qH, localVertex);
                NPConvexBoxCandidate candidate;
                if (!npConvexVertexBoxCandidate(
                        worldVertex, vid, box, qBox, relHullBox,
                        hullRadius, boxRadius, P, candidate)) continue;
                if (candidate.separation > best.separation + separationBand
                    || dot(candidate.boxToHull, best.boxToHull) < 0.98f) continue;
                bool duplicate = false;
                float minSpread = FLT_MAX;
                for (int j = 0; j < count; j++) {
                    if (candidate.feature == chosen[j].feature) {
                        duplicate = true;
                        break;
                    }
                    float3 d = candidate.xHull - chosen[j].xHull;
                    d -= best.boxToHull * dot(best.boxToHull, d);
                    minSpread = min(minSpread, dot(d, d));
                }
                if (duplicate) continue;
                if (!haveNext || minSpread > bestSpread + 1e-10f
                    || (fabs(minSpread - bestSpread) <= 1e-10f
                        && candidate.feature < next.feature)) {
                    next = candidate;
                    bestSpread = minSpread;
                    haveNext = true;
                }
            }
            if (!haveNext || bestSpread < MERGE_DIST_SQ) break;
            chosen[count++] = next;
        }

        float3 normal = hIsA ? best.boxToHull : -best.boxToHull;
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
        outM.header = uint4(ba, bb, uint(count), 1);
        outM.basisN = float4(normal, dynamicFriction);
        outM.basisT1 = float4(t1, staticFriction);
        float4 qAc = q_conj(qBodyA);
        float4 qBc = q_conj(qBodyB);
        float warm = P.alpha * P.gamma;
        for (int i = 0; i < count; i++) {
            float3 xA = hIsA ? chosen[i].xHull : chosen[i].xBox;
            float3 xB = hIsA ? chosen[i].xBox : chosen[i].xHull;
            float3 rA = q_rotate(qAc, xA - bodyPA4.xyz);
            float3 rB = q_rotate(qBc, xB - bodyPB4.xyz);
            float3 lambda = float3(0);
            float3 penalty = float3(0);
            float stick = 0.0f;
            if (prevIdx >= 0) {
                device const ManifoldGPU& previous = prevManifolds[prevIdx];
                for (uint j = 0; j < previous.header.z; j++) {
                    if (as_type<uint>(previous.contacts[j].rA.w)
                        == chosen[i].feature) {
                        lambda = previous.contacts[j].lambda.xyz;
                        penalty = previous.contacts[j].penalty.xyz;
                        stick = previous.contacts[j].rB.w;
                        if (stick != 0.0f) {
                            rA = previous.contacts[j].rA.xyz;
                            rB = previous.contacts[j].rB.xyz;
                        }
                        break;
                    }
                }
            }
            float3 xAw = xform(bodyPA4.xyz, qBodyA, rA);
            float3 xBw = xform(bodyPB4.xyz, qBodyB, rB);
            float3 d = xAw - xBw;
            float3 C0 = float3(dot(normal, d) + P.collisionMargin,
                               dot(t1, d), dot(t2, d));
            C0.yz -= spinSurfaceShift(spinVel[ba], spinVel[bb],
                                      xAw - bodyPA4.xyz,
                                      xBw - bodyPB4.xyz,
                                      t1, t2, P.dt, P.alpha);
            lambda *= warm;
            penalty = clamp(penalty * P.gamma,
                            npPenaltyFloor(posLin, ba, bb, P),
                            npPenaltyCeil());
            outM.contacts[i].rA = float4(
                rA, as_type<float>(chosen[i].feature));
            outM.contacts[i].rB = float4(rB, stick);
            outM.contacts[i].C0 = float4(C0, 0);
            outM.contacts[i].lambda = float4(lambda, 0);
            outM.contacts[i].penalty = float4(penalty, 0);
        }
        return;
    }

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

            // PAD PATCH. A sphere on a face is a mathematical point, and
            // point friction cannot resist rotation about its own normal:
            // a gripped ball spins freely about the grasp axis, so a two-
            // finger grasp behaves like an axle rather than a hold. Real
            // pads flatten into a patch; model it as a small ring of
            // contacts on the face, each paired with the sphere surface
            // point above it, so the ring engages the true curved surface
            // rather than faking a flat.
            int faceAxis = fabs(nLocal.x) > 0.9f ? 0
                         : (fabs(nLocal.y) > 0.9f ? 1 : 2);
            if (P.spherePatchContacts != 0u
                && fabs(nLocal[faceAxis]) > 0.99f) {
                int u = (faceAxis + 1) % 3, v = (faceAxis + 2) % 3;
                float rp = min(0.4f * r, 0.006f);
                const float2 ring[3] = { float2(1.0f, 0.0f),
                                         float2(-0.5f, 0.866f),
                                         float2(-0.5f, -0.866f) };
                for (int k = 0; k < 3; k++) {
                    float3 pl = qLocal;
                    pl[u] = clamp(qLocal[u] + rp * ring[k].x,
                                  -half3_[u], half3_[u]);
                    pl[v] = clamp(qLocal[v] + rp * ring[k].y,
                                  -half3_[v], half3_[v]);
                    float3 pw = q_rotate(qBox, pl) + box.center;
                    float3 dpc = pw - sphereC;
                    float3 lat = dpc - nW * dot(dpc, nW);
                    float rho2 = dot(lat, lat);
                    if (rho2 > 0.64f * r * r || rho2 < 1e-10f) continue;
                    float h = sqrt(max(r * r - rho2, 0.0f));
                    float3 xs = sphereC + lat - nW * h;
                    contacts[count].xA = sIsA ? xs : pw;
                    contacts[count].xB = sIsA ? pw : xs;
                    contacts[count].feature = uint(count);
                    count++;
                }
            }
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
