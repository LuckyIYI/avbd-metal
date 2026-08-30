import simd
import SimCore

// OBB vs OBB SAT collision with face clipping and edge-edge closest points.
// Port of avbd-demo3d/collide.cpp. Produces up to 8 contact points with
// stable feature keys for warm-start persistence.

private let MAX_CONTACTS = 8
private let MAX_POLY_VERTS = 16
private let SAT_AXIS_EPSILON: Float = 1.0e-6
private let PLANE_EPSILON: Float = 1.0e-5
private let CONTACT_MERGE_DIST_SQ: Float = 1.0e-6

private enum AxisType: Int32 {
    case faceA = 0, faceB = 1, edge = 2
}

private struct OBB {
    var center: F3
    var half: F3
    var axis: (F3, F3, F3)

    init(_ body: CPURigid) {
        center = body.positionLin
        half = body.size * 0.5
        axis = (rotate(body.positionAng, F3(1, 0, 0)),
                rotate(body.positionAng, F3(0, 1, 0)),
                rotate(body.positionAng, F3(0, 0, 1)))
    }

    func ax(_ i: Int) -> F3 { i == 0 ? axis.0 : (i == 1 ? axis.1 : axis.2) }
}

private struct SatAxis {
    var type = AxisType.faceA
    var indexA = -1
    var indexB = -1
    var separation = -Float.greatestFiniteMagnitude
    var normalAB = F3.zero
    var valid = false
}

private func supportPoint(_ box: OBB, _ dir: F3) -> F3 {
    let sx: Float = dot(dir, box.axis.0) >= 0 ? 1 : -1
    let sy: Float = dot(dir, box.axis.1) >= 0 ? 1 : -1
    let sz: Float = dot(dir, box.axis.2) >= 0 ? 1 : -1
    return box.center
        + box.axis.0 * (box.half.x * sx)
        + box.axis.1 * (box.half.y * sy)
        + box.axis.2 * (box.half.z * sz)
}

private func getFaceAxes(_ box: OBB, _ axisIndex: Int) -> (u: F3, v: F3, eu: Float, ev: Float) {
    switch axisIndex {
    case 0: return (box.axis.1, box.axis.2, box.half.y, box.half.z)
    case 1: return (box.axis.0, box.axis.2, box.half.x, box.half.z)
    default: return (box.axis.0, box.axis.1, box.half.x, box.half.y)
    }
}

private func testAxis(_ boxA: OBB, _ boxB: OBB, _ delta: F3, _ axis: F3,
                      _ type: AxisType, _ indexA: Int, _ indexB: Int,
                      _ best: inout SatAxis) -> Bool {
    let lenSq = length_squared(axis)
    if lenSq < SAT_AXIS_EPSILON { return true }

    var n = axis / lenSq.squareRoot()
    if dot(n, delta) < 0 { n = -n }

    let distance = abs(dot(delta, n))
    let rA = boxA.half.x * abs(dot(n, boxA.axis.0))
        + boxA.half.y * abs(dot(n, boxA.axis.1))
        + boxA.half.z * abs(dot(n, boxA.axis.2))
    let rB = boxB.half.x * abs(dot(n, boxB.axis.0))
        + boxB.half.y * abs(dot(n, boxB.axis.1))
        + boxB.half.z * abs(dot(n, boxB.axis.2))

    let separation = distance - (rA + rB)
    if separation > 0 { return false }

    if !best.valid || separation > best.separation {
        best.valid = true
        best.type = type
        best.indexA = indexA
        best.indexB = indexB
        best.separation = separation
        best.normalAB = n
    }
    return true
}

private func clipPolygonAgainstPlane(_ inVerts: [F3], _ planeNormal: F3, _ planeOffset: Float) -> [F3] {
    if inVerts.isEmpty { return [] }
    var out: [F3] = []
    out.reserveCapacity(MAX_POLY_VERTS)
    var a = inVerts[inVerts.count - 1]
    var da = dot(planeNormal, a) - planeOffset

    for b in inVerts {
        let db = dot(planeNormal, b) - planeOffset
        let aInside = da <= PLANE_EPSILON
        let bInside = db <= PLANE_EPSILON

        if aInside != bInside {
            var t: Float = 0
            let denom = da - db
            if abs(denom) > SAT_AXIS_EPSILON {
                t = simd_clamp(da / denom, 0, 1)
            }
            if out.count < MAX_POLY_VERTS { out.append(a + (b - a) * t) }
        }
        if bInside && out.count < MAX_POLY_VERTS { out.append(b) }
        a = b
        da = db
    }
    return out
}

private func closestPointsOnSegments(_ p0: F3, _ p1: F3, _ q0: F3, _ q1: F3) -> (F3, F3) {
    let d1 = p1 - p0
    let d2 = q1 - q0
    let r = p0 - q0
    let a = dot(d1, d1)
    let e = dot(d2, d2)
    let f = dot(d2, r)

    var s: Float = 0
    var t: Float = 0

    if a <= SAT_AXIS_EPSILON && e <= SAT_AXIS_EPSILON {
        return (p0, q0)
    }
    if a <= SAT_AXIS_EPSILON {
        t = simd_clamp(f / e, 0, 1)
    } else {
        let c = dot(d1, r)
        if e <= SAT_AXIS_EPSILON {
            s = simd_clamp(-c / a, 0, 1)
        } else {
            let b = dot(d1, d2)
            let denom = a * e - b * b
            if abs(denom) > SAT_AXIS_EPSILON {
                s = simd_clamp((b * f - c * e) / denom, 0, 1)
            }
            t = (b * s + f) / e
            if t < 0 {
                t = 0
                s = simd_clamp(-c / a, 0, 1)
            } else if t > 1 {
                t = 1
                s = simd_clamp((b - c) / a, 0, 1)
            }
        }
    }
    return (p0 + d1 * s, q0 + d2 * t)
}

private func supportEdge(_ box: OBB, _ axisIndex: Int, _ dir: F3) -> (F3, F3) {
    let axis1 = (axisIndex + 1) % 3
    let axis2 = (axisIndex + 2) % 3
    let sign1: Float = dot(dir, box.ax(axis1)) >= 0 ? 1 : -1
    let sign2: Float = dot(dir, box.ax(axis2)) >= 0 ? 1 : -1

    let edgeCenter = box.center
        + box.ax(axis1) * (box.half[axis1] * sign1)
        + box.ax(axis2) * (box.half[axis2] * sign2)

    return (edgeCenter - box.ax(axisIndex) * box.half[axisIndex],
            edgeCenter + box.ax(axisIndex) * box.half[axisIndex])
}

private func addContact(_ bodyA: CPURigid, _ bodyB: CPURigid,
                        _ contacts: inout [ContactPoint], _ midpoints: inout [F3],
                        _ xA: F3, _ xB: F3, _ featureKey: Int32) {
    let midpoint = (xA + xB) * 0.5
    for m in midpoints where length_squared(midpoint - m) < CONTACT_MERGE_DIST_SQ {
        return
    }
    if contacts.count >= MAX_CONTACTS { return }

    var c = ContactPoint()
    c.featureKey = featureKey
    c.rA = rotate(bodyA.positionAng.conjugate, xA - bodyA.positionLin)
    c.rB = rotate(bodyB.positionAng.conjugate, xB - bodyB.positionLin)
    contacts.append(c)
    midpoints.append(midpoint)
}

private func buildFaceManifold(_ bodyA: CPURigid, _ bodyB: CPURigid,
                               _ boxA: OBB, _ boxB: OBB,
                               _ referenceIsA: Bool, _ referenceAxis: Int,
                               _ normalAB: F3, _ contacts: inout [ContactPoint]) {
    let referenceBox = referenceIsA ? boxA : boxB
    let incidentBox = referenceIsA ? boxB : boxA
    let referenceOutward = referenceIsA ? normalAB : -normalAB

    // Reference face frame
    let signR: Float = dot(referenceOutward, referenceBox.ax(referenceAxis)) >= 0 ? 1 : -1
    let refNormal = referenceBox.ax(referenceAxis) * signR
    let refCenter = referenceBox.center + refNormal * referenceBox.half[referenceAxis]
    let (refU, refV, refEU, refEV) = getFaceAxes(referenceBox, referenceAxis)

    // Incident face: most anti-parallel axis on incident box
    var incidentAxis = 0
    var bestD = -Float.greatestFiniteMagnitude
    for i in 0..<3 {
        let d = abs(dot(incidentBox.ax(i), refNormal))
        if d > bestD { bestD = d; incidentAxis = i }
    }

    let signI: Float = dot(incidentBox.ax(incidentAxis), refNormal) > 0 ? -1 : 1
    let incNormal = incidentBox.ax(incidentAxis) * signI
    let incCenter = incidentBox.center + incNormal * incidentBox.half[incidentAxis]
    let (incU, incV, incEU, incEV) = getFaceAxes(incidentBox, incidentAxis)

    var poly: [F3] = [
        incCenter + incU * incEU + incV * incEV,
        incCenter - incU * incEU + incV * incEV,
        incCenter - incU * incEU - incV * incEV,
        incCenter + incU * incEU - incV * incEV,
    ]

    poly = clipPolygonAgainstPlane(poly, refU, dot(refU, refCenter) + refEU)
    if poly.isEmpty { return }
    poly = clipPolygonAgainstPlane(poly, -refU, dot(-refU, refCenter) + refEU)
    if poly.isEmpty { return }
    poly = clipPolygonAgainstPlane(poly, refV, dot(refV, refCenter) + refEV)
    if poly.isEmpty { return }
    poly = clipPolygonAgainstPlane(poly, -refV, dot(-refV, refCenter) + refEV)
    if poly.isEmpty { return }

    var midpoints: [F3] = []
    var featurePrefix: Int32 = Int32((referenceIsA ? AxisType.faceA : AxisType.faceB).rawValue) << 24
    featurePrefix |= Int32(referenceAxis & 0xFF) << 16
    featurePrefix |= Int32(incidentAxis & 0xFF) << 8

    for (i, pIncident) in poly.enumerated() where contacts.count < MAX_CONTACTS {
        let distance = dot(pIncident - refCenter, refNormal)
        if distance > PLANE_EPSILON { continue }

        let pReference = pIncident - refNormal * distance
        let xA = referenceIsA ? pReference : pIncident
        let xB = referenceIsA ? pIncident : pReference
        addContact(bodyA, bodyB, &contacts, &midpoints, xA, xB, featurePrefix | Int32(i & 0xFF))
    }

    if contacts.isEmpty {
        let xA = supportPoint(boxA, normalAB)
        let xB = supportPoint(boxB, -normalAB)
        addContact(bodyA, bodyB, &contacts, &midpoints, xA, xB, featurePrefix)
    }
}

private func buildEdgeContact(_ bodyA: CPURigid, _ bodyB: CPURigid,
                              _ boxA: OBB, _ boxB: OBB,
                              _ axisA: Int, _ axisB: Int,
                              _ normalAB: F3, _ contacts: inout [ContactPoint]) {
    let (a0, a1) = supportEdge(boxA, axisA, normalAB)
    let (b0, b1) = supportEdge(boxB, axisB, -normalAB)
    var (xA, xB) = closestPointsOnSegments(a0, a1, b0, b1)

    var midpoints: [F3] = []
    let featureKey: Int32 = (Int32(AxisType.edge.rawValue) << 24)
        | (Int32(axisA & 0xFF) << 8) | Int32(axisB & 0xFF)
    addContact(bodyA, bodyB, &contacts, &midpoints, xA, xB, featureKey)

    if contacts.isEmpty {
        xA = supportPoint(boxA, normalAB)
        xB = supportPoint(boxB, -normalAB)
        addContact(bodyA, bodyB, &contacts, &midpoints, xA, xB, featureKey)
    }
}

extension CPUManifold {
    /// Collider-aware dispatch. Analytic pairs retain the established CPU
    /// routines through world-pose proxies; any pair containing a cooked hull
    /// uses the isolated support-map MPR/GJK reference implementation.
    static func collideColliders(
        solver: CPUSolver,
        bodyA: CPURigid, colliderA: CPUCollider,
        bodyB: CPURigid, colliderB: CPUCollider,
        margin: Float,
        _ contacts: inout [ContactPoint],
        _ basisOut: inout (F3, F3, F3)
    ) -> Result<Int, ConvexNarrowPhase.Failure> {
        let hullA = colliderHullVertices(solver: solver, collider: colliderA)
        let hullB = colliderHullVertices(solver: solver, collider: colliderB)

        if hullA != nil || hullB != nil {
            let poseA = colliderA.worldPose(bodyA)
            let poseB = colliderB.worldPose(bodyB)
            let shapeA = convexShape(colliderA, hullVertices: hullA)
            let shapeB = convexShape(colliderB, hullVertices: hullB)
            let result: Result<ConvexNarrowPhase.Manifold, ConvexNarrowPhase.Failure>
            if let injected = solver.convexQueryFailureForTesting {
                result = .failure(injected)
            } else {
                result = ConvexNarrowPhase.query(
                    shapeA: shapeA,
                    poseA: .init(position: poseA.position,
                                 orientation: poseA.orientation),
                    shapeB: shapeB,
                    poseB: .init(position: poseB.position,
                                 orientation: poseB.orientation),
                    options: .init(contactThreshold: margin))
            }

            let manifold: ConvexNarrowPhase.Manifold
            switch result {
            case .success(let value):
                manifold = value
            case .failure(let failure):
                return .failure(failure)
            }
            guard manifold.signedDistance <= margin else {
                contacts.removeAll(keepingCapacity: true)
                return .success(0)
            }

            // ConvexNarrowPhase reports A->B; the AVBD constraint basis uses B->A.
            basisOut = orthonormalBasis(-manifold.normalAtoB)
            contacts.removeAll(keepingCapacity: true)
            contacts.reserveCapacity(manifold.contacts.count)
            for source in manifold.contacts {
                var contact = ContactPoint()
                contact.featureKey = Int32(source.sortKey)
                contact.rA = ownerAnchor(
                    worldPoint: source.pointA, body: bodyA,
                    collider: colliderA)
                contact.rB = ownerAnchor(
                    worldPoint: source.pointB, body: bodyB,
                    collider: colliderB)
                contacts.append(contact)
            }
            if solver.spherePatchContacts, contacts.count == 1 {
                appendSphereHullFacePatch(
                    manifold: manifold,
                    bodyA: bodyA, colliderA: colliderA, hullA: hullA,
                    bodyB: bodyB, colliderB: colliderB, hullB: hullB,
                    contacts: &contacts)
            }
            return .success(contacts.count)
        }

        let proxyA = proxyBody(for: colliderA, owner: bodyA)
        let proxyB = proxyBody(for: colliderB, owner: bodyB)
        let count = collide(
            proxyA, proxyB, margin: margin,
            patchContacts: solver.spherePatchContacts,
            &contacts, &basisOut)
        for index in 0..<count {
            let pointA = proxyAnchorWorld(proxyA, contacts[index].rA)
            let pointB = proxyAnchorWorld(proxyB, contacts[index].rB)
            contacts[index].rA = ownerAnchor(
                worldPoint: pointA, body: bodyA, collider: colliderA)
            contacts[index].rB = ownerAnchor(
                worldPoint: pointB, body: bodyB, collider: colliderB)
        }
        return .success(count)
    }

    /// A smooth sphere against a broad convex face has the same torsional-
    /// friction problem as a sphere against an analytic box: one witness has
    /// no lever arm about the face normal. Add a small three-point ring only
    /// when the supporting hull feature is a genuine face and every generated
    /// point remains inside its projected boundary. The ring models a tiny
    /// compliant flattened footprint: all samples retain the exact witness's
    /// normal separation so that they carry load, instead of placing nominal
    /// contacts on the undeformed sphere above the face. Edge and vertex
    /// contacts deliberately retain the exact single support witness.
    private static func appendSphereHullFacePatch(
        manifold: ConvexNarrowPhase.Manifold,
        bodyA: CPURigid, colliderA: CPUCollider, hullA: [F3]?,
        bodyB: CPURigid, colliderB: CPUCollider, hullB: [F3]?,
        contacts: inout [ContactPoint]
    ) {
        let sphereIsA: Bool
        let sphereBody: CPURigid
        let sphereCollider: CPUCollider
        let hullBody: CPURigid
        let hullCollider: CPUCollider
        let hullVertices: [F3]
        if hullA == nil, colliderA.shape == .sphere, let hullB {
            sphereIsA = true
            sphereBody = bodyA
            sphereCollider = colliderA
            hullBody = bodyB
            hullCollider = colliderB
            hullVertices = hullB
        } else if hullB == nil, colliderB.shape == .sphere, let hullA {
            sphereIsA = false
            sphereBody = bodyB
            sphereCollider = colliderB
            hullBody = bodyA
            hullCollider = colliderA
            hullVertices = hullA
        } else {
            return
        }

        let faceNormal = sphereIsA
            ? -manifold.normalAtoB : manifold.normalAtoB // hull -> sphere
        guard faceNormal.x.isFinite, faceNormal.y.isFinite,
              faceNormal.z.isFinite,
              length_squared(faceNormal) > 0.99 else { return }
        let hullPose = hullCollider.worldPose(hullBody)
        let worldVertices = hullVertices.map {
            hullPose.position + hullPose.orientation.act($0)
        }
        guard let support = worldVertices.map({ dot($0, faceNormal) }).max()
        else { return }
        let hullScale = max(hullCollider.boundingRadius, 1.0e-3)
        let faceTolerance = max(1.0e-5, hullScale * 1.0e-4)
        let hullCrossTolerance = max(1.0e-14, hullScale * hullScale * 1.0e-8)
        let insideTolerance = max(1.0e-10, hullScale * hullScale * 1.0e-6)
        let faceVertices = worldVertices.filter {
            support - dot($0, faceNormal) <= faceTolerance
        }
        guard faceVertices.count >= 3 else { return }

        let basis = orthonormalBasis(faceNormal)
        let tangentU = basis.1, tangentV = basis.2
        let origin = faceVertices[0]
        var projected = faceVertices.map {
            SIMD2<Float>(dot($0 - origin, tangentU),
                         dot($0 - origin, tangentV))
        }
        projected.sort {
            $0.x < $1.x || ($0.x == $1.x && $0.y < $1.y)
        }
        func cross2(_ a: SIMD2<Float>, _ b: SIMD2<Float>,
                    _ c: SIMD2<Float>) -> Float {
            let ab = b - a, ac = c - a
            return ab.x * ac.y - ab.y * ac.x
        }
        var lower: [SIMD2<Float>] = []
        for point in projected {
            while lower.count >= 2,
                  cross2(lower[lower.count - 2], lower.last!, point)
                    <= hullCrossTolerance {
                lower.removeLast()
            }
            lower.append(point)
        }
        var upper: [SIMD2<Float>] = []
        for point in projected.reversed() {
            while upper.count >= 2,
                  cross2(upper[upper.count - 2], upper.last!, point)
                    <= hullCrossTolerance {
                upper.removeLast()
            }
            upper.append(point)
        }
        let polygon = Array(lower.dropLast()) + Array(upper.dropLast())
        guard polygon.count >= 3 else { return }

        func isInside(_ point: SIMD2<Float>) -> Bool {
            for index in polygon.indices {
                let next = polygon[(index + 1) % polygon.count]
                if cross2(polygon[index], next, point) < -insideTolerance {
                    return false
                }
            }
            return true
        }

        let sphereCenter = sphereCollider.worldPose(sphereBody).position
        let radius = sphereCollider.size.x * 0.5
        guard radius.isFinite, radius > 0 else { return }
        let hullWitness = sphereIsA
            ? manifold.contacts[0].pointB : manifold.contacts[0].pointA
        let sphereWitness = sphereIsA
            ? manifold.contacts[0].pointA : manifold.contacts[0].pointB
        let normalSeparation = dot(
            sphereWitness - hullWitness, faceNormal)
        let facePoint = hullWitness
            - faceNormal * (dot(hullWitness, faceNormal) - support)
        let facePoint2 = SIMD2<Float>(
            dot(facePoint - origin, tangentU),
            dot(facePoint - origin, tangentV))
        let patchRadius = min(0.4 * radius, 0.006)
        let ring: [SIMD2<Float>] = [
            .init(1, 0), .init(-0.5, 0.866), .init(-0.5, -0.866),
        ]
        for (index, offset) in ring.enumerated() {
            let target2 = facePoint2 + patchRadius * offset
            guard isInside(target2) else { continue }
            let hullPoint = origin + tangentU * target2.x
                + tangentV * target2.y
            let centerToFace = hullPoint - sphereCenter
            let lateral = centerToFace
                - faceNormal * dot(centerToFace, faceNormal)
            let rhoSquared = length_squared(lateral)
            guard rhoSquared > 1.0e-10,
                  rhoSquared <= 0.64 * radius * radius else { continue }
            let spherePoint = hullPoint + faceNormal * normalSeparation
            let pointA = sphereIsA ? spherePoint : hullPoint
            let pointB = sphereIsA ? hullPoint : spherePoint
            var contact = ContactPoint()
            contact.featureKey = Int32(0x5100_0000 | (index + 1))
            contact.rA = ownerAnchor(
                worldPoint: pointA, body: bodyA, collider: colliderA)
            contact.rB = ownerAnchor(
                worldPoint: pointB, body: bodyB, collider: colliderB)
            contacts.append(contact)
        }
    }

    private static func colliderHullVertices(
        solver: CPUSolver, collider: CPUCollider
    ) -> [F3]? {
        if let assetID = collider.convexAssetID {
            precondition(solver.convexAssets.indices.contains(assetID),
                         "CPU convex asset reference is out of range")
            return solver.convexAssets[assetID].vertices
        }
        return collider.convexHullVertices.isEmpty
            ? nil : collider.convexHullVertices
    }

    private static func convexShape(
        _ collider: CPUCollider, hullVertices: [F3]?
    ) -> ConvexNarrowPhase.Shape {
        if let hullVertices { return .convexHull(vertices: hullVertices) }
        switch collider.shape {
        case .box:
            return .box(halfExtents: collider.size * 0.5)
        case .sphere:
            return .sphere(radius: collider.size.x * 0.5)
        case .capsule:
            return .capsule(radius: collider.size.y,
                            halfHeight: collider.size.x * 0.5)
        case .torus:
            return .torus(majorRadius: collider.size.x,
                          minorRadius: collider.size.y)
        }
    }

    private static func proxyBody(
        for collider: CPUCollider, owner: CPURigid
    ) -> CPURigid {
        let pose = collider.worldPose(owner)
        return CPURigid(
            index: owner.index, size: collider.size,
            density: 0, friction: collider.friction,
            dynamicFriction: collider.dynamicFriction,
            position: pose.position, rotation: pose.orientation,
            shape: collider.shape)
    }

    private static func proxyAnchorWorld(_ proxy: CPURigid, _ anchor: F3) -> F3 {
        proxy.shape == .box
            ? transform(proxy.positionLin, proxy.positionAng, anchor)
            : proxy.positionLin + anchor
    }

    private static func ownerAnchor(
        worldPoint: F3, body: CPURigid, collider: CPUCollider
    ) -> F3 {
        if collider.usesWorldSpaceRoundAnchor {
            return worldPoint - body.positionLin
        }
        return body.positionAng.inverse.act(worldPoint - body.positionLin)
    }

    /// Shape-dispatching collision. Returns contact count; fills basis
    /// (rows n, t1, t2) with the normal pointing from B toward A.
    static func collide(_ bodyA: CPURigid, _ bodyB: CPURigid,
                        margin: Float = AVBDConstants.collisionMargin,
                        patchContacts: Bool = false,
                        _ contacts: inout [ContactPoint],
                        _ basisOut: inout (F3, F3, F3)) -> Int {
        switch (bodyA.shape, bodyB.shape) {
        case (.sphere, .sphere):
            return collideSphereSphere(
                bodyA, bodyB, margin: margin, &contacts, &basisOut)
        case (.sphere, .box):
            return collideSphereBox(
                bodyA, bodyB, sphereIsA: true, margin: margin,
                patchContacts: patchContacts, &contacts, &basisOut)
        case (.box, .sphere):
            return collideSphereBox(
                bodyB, bodyA, sphereIsA: false, margin: margin,
                patchContacts: patchContacts, &contacts, &basisOut)
        case (.box, .box):
            return collideBoxBox(bodyA, bodyB, &contacts, &basisOut)
        case (.torus, .sphere):
            return collideTorusSphere(
                bodyA, bodyB, torusIsA: true, margin: margin,
                &contacts, &basisOut)
        case (.sphere, .torus):
            return collideTorusSphere(
                bodyB, bodyA, torusIsA: false, margin: margin,
                &contacts, &basisOut)
        case (.torus, .box):
            return collideTorusBox(
                bodyA, bodyB, torusIsA: true, margin: margin,
                &contacts, &basisOut)
        case (.box, .torus):
            return collideTorusBox(
                bodyB, bodyA, torusIsA: false, margin: margin,
                &contacts, &basisOut)
        case (.torus, .torus):
            return collideTorusTorus(
                bodyA, bodyB, margin: margin, &contacts, &basisOut)
        case (.capsule, .sphere):
            return collideCapsuleSphere(
                bodyA, bodyB, capsuleIsA: true, margin: margin,
                &contacts, &basisOut)
        case (.sphere, .capsule):
            return collideCapsuleSphere(
                bodyB, bodyA, capsuleIsA: false, margin: margin,
                &contacts, &basisOut)
        case (.capsule, .box):
            return collideCapsuleBox(
                bodyA, bodyB, capsuleIsA: true, margin: margin,
                &contacts, &basisOut)
        case (.box, .capsule):
            return collideCapsuleBox(
                bodyB, bodyA, capsuleIsA: false, margin: margin,
                &contacts, &basisOut)
        case (.capsule, .capsule):
            return collideCapsuleCapsule(
                bodyA, bodyB, margin: margin, &contacts, &basisOut)
        case (.torus, .capsule):
            return collideTorusCapsule(
                bodyA, bodyB, torusIsA: true, margin: margin,
                &contacts, &basisOut)
        case (.capsule, .torus):
            return collideTorusCapsule(
                bodyB, bodyA, torusIsA: false, margin: margin,
                &contacts, &basisOut)
        }
    }

    static func collideSphereSphere(_ a: CPURigid, _ b: CPURigid,
                                    margin: Float,
                                    _ contacts: inout [ContactPoint],
                                    _ basisOut: inout (F3, F3, F3)) -> Int {
        contacts.removeAll(keepingCapacity: true)
        let rA = a.size.x / 2, rB = b.size.x / 2
        let d = a.positionLin - b.positionLin
        let dist = length(d)
        if dist > rA + rB + margin { return 0 }
        let n = dist > 1e-9 ? d / dist : F3(0, 0, 1)   // B -> A
        basisOut = orthonormalBasis(n)
        var c = ContactPoint()
        c.featureKey = 0
        let xA = a.positionLin - n * rA
        let xB = b.positionLin + n * rB
        // world-space offsets for spheres (rotation-invariant anchors)
        c.rA = xA - a.positionLin
        c.rB = xB - b.positionLin
        contacts.append(c)
        return 1
    }

    /// sphere vs box; `sphereIsA` says whether the sphere is manifold body A.
    static func collideSphereBox(_ sphere: CPURigid, _ box: CPURigid,
                                 sphereIsA: Bool, margin: Float,
                                 patchContacts: Bool = false,
                                 _ contacts: inout [ContactPoint],
                                 _ basisOut: inout (F3, F3, F3)) -> Int {
        contacts.removeAll(keepingCapacity: true)
        let r = sphere.size.x / 2
        let half = box.size * 0.5
        let qc = box.positionAng.conjugate
        let local = qc.act(sphere.positionLin - box.positionLin)
        let clamped = simd_clamp(local, -half, half)

        var nLocal: F3
        var qLocal: F3
        if clamped == local {
            // center inside the box: push out along min-penetration face
            let pen = half - abs(local)
            var axis = 0
            if pen.y < pen[axis] { axis = 1 }
            if pen.z < pen[axis] { axis = 2 }
            var n = F3.zero
            n[axis] = local[axis] >= 0 ? 1 : -1
            nLocal = n
            qLocal = local
            qLocal[axis] = n[axis] * half[axis]
        } else {
            let d = local - clamped
            let dist = length(d)
            if dist > r + margin { return 0 }
            nLocal = d / max(dist, 1e-9)
            qLocal = clamped
        }

        let nW = box.positionAng.act(nLocal)        // box -> sphere
        let qW = transform(box.positionLin, box.positionAng, qLocal)
        let xSphere = sphere.positionLin - nW * r
        // Manifold normal must point B -> A
        let n = sphereIsA ? nW : -nW
        basisOut = orthonormalBasis(n)
        let bodyA = sphereIsA ? sphere : box
        let bodyB = sphereIsA ? box : sphere
        func append(_ xA: F3, _ xB: F3, _ key: Int32) {
            var c = ContactPoint()
            c.featureKey = key
            // sphere side: world offset; box side: material (local) anchor
            c.rA = bodyA.shape == .sphere ? xA - bodyA.positionLin
                : rotate(bodyA.positionAng.conjugate, xA - bodyA.positionLin)
            c.rB = bodyB.shape == .sphere ? xB - bodyB.positionLin
                : rotate(bodyB.positionAng.conjugate, xB - bodyB.positionLin)
            contacts.append(c)
        }
        append(sphereIsA ? xSphere : qW, sphereIsA ? qW : xSphere, 0)
        // PAD PATCH (mirrors the GPU narrow phase): a sphere on a face is a
        // point, and point friction cannot resist spin about its own
        // normal, so a two-finger grasp holds the ball on an axle. A small
        // ring of contacts on the face models a tiny compliant flattened
        // footprint. Each sample keeps the exact witness's normal separation
        // so it carries load and supplies a real torsional lever arm.
        var faceAxis = 0
        if abs(nLocal.y) > abs(nLocal[faceAxis]) { faceAxis = 1 }
        if abs(nLocal.z) > abs(nLocal[faceAxis]) { faceAxis = 2 }
        if patchContacts, abs(nLocal[faceAxis]) > 0.99 {
            let u = (faceAxis + 1) % 3, v = (faceAxis + 2) % 3
            let rp = min(0.4 * r, 0.006)
            let normalSeparation = dot(xSphere - qW, nW)
            let ring: [(Float, Float)] = [(1, 0), (-0.5, 0.866), (-0.5, -0.866)]
            for (k, off) in ring.enumerated() {
                var pl = qLocal
                pl[u] = min(max(qLocal[u] + rp * off.0, -half[u]), half[u])
                pl[v] = min(max(qLocal[v] + rp * off.1, -half[v]), half[v])
                let pw = transform(box.positionLin, box.positionAng, pl)
                let dpc = pw - sphere.positionLin
                let lat = dpc - nW * dot(dpc, nW)
                let rho2 = length_squared(lat)
                if rho2 > 0.64 * r * r || rho2 < 1e-10 { continue }
                let xs = pw + nW * normalSeparation
                append(sphereIsA ? xs : pw, sphereIsA ? pw : xs, Int32(k + 1))
            }
        }
        return contacts.count
    }

    /// SAT OBB-OBB collision. Returns contact count; fills basis (rows n, t1, t2).
    static func collideBoxBox(_ bodyA: CPURigid, _ bodyB: CPURigid,
                              _ contacts: inout [ContactPoint],
                              _ basisOut: inout (F3, F3, F3)) -> Int {
        let boxA = OBB(bodyA)
        let boxB = OBB(bodyB)
        let delta = boxB.center - boxA.center

        var bestFace = SatAxis()
        var bestEdge = SatAxis()

        for i in 0..<3 {
            if !testAxis(boxA, boxB, delta, boxA.ax(i), .faceA, i, -1, &bestFace) { return 0 }
        }
        for i in 0..<3 {
            if !testAxis(boxA, boxB, delta, boxB.ax(i), .faceB, -1, i, &bestFace) { return 0 }
        }
        for i in 0..<3 {
            for j in 0..<3 {
                let axis = cross(boxA.ax(i), boxB.ax(j))
                if !testAxis(boxA, boxB, delta, axis, .edge, i, j, &bestEdge) { return 0 }
            }
        }

        if !bestFace.valid { return 0 }

        var best = bestFace
        if bestEdge.valid {
            let edgeRelTol: Float = 0.95
            let edgeAbsTol: Float = 0.01
            if edgeRelTol * bestEdge.separation > bestFace.separation + edgeAbsTol {
                best = bestEdge
            }
        }

        basisOut = orthonormalBasis(-best.normalAB)
        contacts.removeAll(keepingCapacity: true)

        if best.type == .edge {
            buildEdgeContact(bodyA, bodyB, boxA, boxB, best.indexA, best.indexB, best.normalAB, &contacts)
        } else if best.type == .faceA {
            buildFaceManifold(bodyA, bodyB, boxA, boxB, true, best.indexA, best.normalAB, &contacts)
        } else {
            buildFaceManifold(bodyA, bodyB, boxA, boxB, false, best.indexB, best.normalAB, &contacts)
        }
        return contacts.count
    }
}

// MARK: - Torus collision (implicit surface, alternating projection)

/// Closest point on the torus SPINE circle (local frame: circle of radius R
/// in the xy-plane) to a local-space point.
@inline(__always)
private func projectToSpine(_ p: F3, _ R: Float) -> F3 {
    var d = F3(p.x, p.y, 0)
    let len = length(d)
    if len < 1e-6 { d = F3(1, 0, 0) } else { d /= len }
    return d * R
}

extension CPUManifold {
    /// Exact torus-sphere: project sphere center onto spine; tube vs sphere.
    static func collideTorusSphere(_ torus: CPURigid, _ sphere: CPURigid,
                                   torusIsA: Bool, margin: Float,
                                   _ contacts: inout [ContactPoint],
                                   _ basisOut: inout (F3, F3, F3)) -> Int {
        contacts.removeAll(keepingCapacity: true)
        let R = torus.size.x, r = torus.size.y
        let rs = sphere.size.x / 2
        let qc = torus.positionAng.conjugate
        let local = qc.act(sphere.positionLin - torus.positionLin)
        let spine = projectToSpine(local, R)
        let d = local - spine
        let dist = length(d)
        if dist > r + rs + margin { return 0 }
        let nLocal = dist > 1e-6 ? d / dist : F3(0, 0, 1)
        let nW = torus.positionAng.act(nLocal)          // torus -> sphere
        let xTorus = torus.positionLin + torus.positionAng.act(spine + nLocal * r)
        let xSphere = sphere.positionLin - nW * rs

        let n = torusIsA ? -nW : nW                     // B -> A
        basisOut = orthonormalBasis(n)
        var c = ContactPoint()
        c.featureKey = 0
        let (bodyA, xA, xB) = torusIsA ? (torus, xTorus, xSphere) : (sphere, xSphere, xTorus)
        let bodyB = torusIsA ? sphere : torus
        // both shapes are round: world-space offsets
        c.rA = xA - bodyA.positionLin
        c.rB = xB - bodyB.positionLin
        contacts.append(c)
        return 1
    }

    /// Torus-box via alternating projection: seed points around the spine,
    /// project box<->spine until converged; keep up to 4 distinct contacts
    /// (a torus resting flat on a box needs a multi-point base).
    static func collideTorusBox(_ torus: CPURigid, _ box: CPURigid,
                                torusIsA: Bool, margin: Float,
                                _ contacts: inout [ContactPoint],
                                _ basisOut: inout (F3, F3, F3)) -> Int {
        contacts.removeAll(keepingCapacity: true)
        let R = torus.size.x, r = torus.size.y
        let half = box.size * 0.5
        // work in box-local space; transform the spine circle there
        let qbc = box.positionAng.conjugate
        let c0 = qbc.act(torus.positionLin - box.positionLin)
        let axis = qbc.act(torus.positionAng.act(F3(0, 0, 1)))
        var u = abs(axis.x) > abs(axis.z) ? F3(-axis.y, axis.x, 0) : F3(0, -axis.z, axis.y)
        u = normalize(u)
        let v = cross(axis, u)

        func spinePoint(_ a: Float) -> F3 { c0 + (u * cos(a) + v * sin(a)) * R }
        func projectToSpineBL(_ p: F3) -> F3 {
            var d = p - c0
            d -= axis * dot(axis, d)
            let len = length(d)
            return len < 1e-6 ? c0 + u * R : c0 + d / len * R
        }

        var found: [(q: F3, b: F3, n: F3, dist: Float)] = []
        var bestSep = Float.greatestFiniteMagnitude
        for seed in 0..<8 {
            var q = spinePoint(Float(seed) / 8 * 2 * .pi)
            var b = F3.zero
            for _ in 0..<5 {
                b = simd_clamp(q, -half, half)
                q = projectToSpineBL(b)
            }
            var d = q - b
            var dist = length(d)
            var n: F3
            if dist < 1e-6 {
                // spine intersects the box: face ejection with TRUE depth
                let pen = half - abs(b)
                var axisI = 0
                if pen.y < pen[axisI] { axisI = 1 }
                if pen.z < pen[axisI] { axisI = 2 }
                n = F3.zero
                n[axisI] = b[axisI] >= 0 ? 1 : -1
                b = q + n * pen[axisI]  // box surface point (true depth)
                dist = 0
            } else {
                n = d / dist
            }
            bestSep = min(bestSep, dist - r)
            if dist - r > margin { continue }
            // dedup by spine point
            if found.contains(where: { length($0.q - q) < 0.25 * R }) { continue }
            if found.count < 4 { found.append((q, b, n, dist)) }
            _ = d
        }
        if found.isEmpty { return 0 }

        // shared normal: average (they're near-parallel for resting contact);
        // per-contact points keep their own geometry
        var nAvg = F3.zero
        for f in found { nAvg += f.n }
        nAvg = normalize(nAvg)
        let nW = box.positionAng.act(nAvg)              // box -> torus
        let n = torusIsA ? -nW : nW                     // B -> A
        basisOut = orthonormalBasis(n)

        for (i, f) in found.enumerated() {
            let xTorusL = f.q - f.n * min(r, max(f.dist, 0))  // torus surface toward box
            let xTorus = box.positionLin + box.positionAng.act(f.q) - nW * r
            _ = xTorusL
            let xBox = box.positionLin + box.positionAng.act(f.b)
            var c = ContactPoint()
            c.featureKey = Int32(i)
            let (bodyA, xA, xB) = torusIsA ? (torus, xTorus, xBox) : (box, xBox, xTorus)
            let bodyB = torusIsA ? box : torus
            c.rA = bodyA.shape == .box ? rotate(bodyA.positionAng.conjugate, xA - bodyA.positionLin)
                                       : xA - bodyA.positionLin
            c.rB = bodyB.shape == .box ? rotate(bodyB.positionAng.conjugate, xB - bodyB.positionLin)
                                       : xB - bodyB.positionLin
            contacts.append(c)
        }
        return contacts.count
    }

    /// Torus-torus: alternating projection between the two spine circles
    /// from several seeds; multiple contacts support interlocked rings.
    static func collideTorusTorus(_ a: CPURigid, _ b: CPURigid,
                                  margin: Float,
                                  _ contacts: inout [ContactPoint],
                                  _ basisOut: inout (F3, F3, F3)) -> Int {
        contacts.removeAll(keepingCapacity: true)
        let Ra = a.size.x, ra = a.size.y
        let Rb = b.size.x, rb = b.size.y
        // world-space circle frames
        let ca = a.positionLin
        let axa = a.positionAng.act(F3(0, 0, 1))
        let cb = b.positionLin
        let axb = b.positionAng.act(F3(0, 0, 1))
        var ua = abs(axa.x) > abs(axa.z) ? F3(-axa.y, axa.x, 0) : F3(0, -axa.z, axa.y)
        ua = normalize(ua)
        let va = cross(axa, ua)

        func projA(_ p: F3) -> F3 {
            var d = p - ca
            d -= axa * dot(axa, d)
            let l = length(d)
            return l < 1e-6 ? ca + ua * Ra : ca + d / l * Ra
        }
        func projB(_ p: F3) -> F3 {
            var d = p - cb
            d -= axb * dot(axb, d)
            let l = length(d)
            return l < 1e-6 ? cb + normalize(cross(axb, axa + F3(0.01, 0, 0))) * Rb
                            : cb + d / l * Rb
        }

        var found: [(pa: F3, pb: F3)] = []
        for seed in 0..<8 {
            let ang = Float(seed) / 8 * 2 * .pi
            var qa = ca + (ua * cos(ang) + va * sin(ang)) * Ra
            var qb = F3.zero
            for _ in 0..<6 {
                qb = projB(qa)
                qa = projA(qb)
            }
            if length(qa - qb) > ra + rb + margin { continue }
            if found.contains(where: { length($0.pa - qa) < 0.25 * Ra }) { continue }
            if found.count < 4 { found.append((qa, qb)) }
        }
        if found.isEmpty { return 0 }

        // basis from the deepest contact's normal (b -> a)
        var deep = found[0]
        var minD = Float.greatestFiniteMagnitude
        for f in found {
            let d = length(f.pa - f.pb)
            if d < minD { minD = d; deep = f }
        }
        var n = deep.pa - deep.pb
        let nl = length(n)
        n = nl > 1e-6 ? n / nl : F3(0, 0, 1)
        basisOut = orthonormalBasis(n)

        for (i, f) in found.enumerated() {
            var nf = f.pa - f.pb
            let l = length(nf)
            nf = l > 1e-6 ? nf / l : n
            var c = ContactPoint()
            c.featureKey = Int32(i)
            let xA = f.pa - nf * ra
            let xB = f.pb + nf * rb
            c.rA = xA - a.positionLin     // round: world offsets
            c.rB = xB - b.positionLin
            contacts.append(c)
        }
        return contacts.count
    }
}

// MARK: - Capsule collision (segment + radius)

@inline(__always)
private func capsuleSegment(_ c: CPURigid) -> (F3, F3, Float) {
    let half = c.size.x / 2
    let axis = c.positionAng.act(F3(0, 0, 1))
    return (c.positionLin - axis * half, c.positionLin + axis * half, c.size.y)
}

@inline(__always)
private func closestOnSegment(_ p: F3, _ a: F3, _ b: F3) -> F3 {
    let ab = b - a
    let t = simd_clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0, 1)
    return a + ab * t
}

extension CPUManifold {
    static func roundAnchor(_ body: CPURigid, _ x: F3) -> F3 {
        body.shape == .box ? rotate(body.positionAng.conjugate, x - body.positionLin)
                           : x - body.positionLin
    }

    static func collideCapsuleSphere(_ cap: CPURigid, _ sph: CPURigid,
                                     capsuleIsA: Bool, margin: Float,
                                     _ contacts: inout [ContactPoint],
                                     _ basisOut: inout (F3, F3, F3)) -> Int {
        contacts.removeAll(keepingCapacity: true)
        let (p0, p1, rc) = capsuleSegment(cap)
        let rs = sph.size.x / 2
        let q = closestOnSegment(sph.positionLin, p0, p1)
        let d = sph.positionLin - q
        let dist = length(d)
        if dist > rc + rs + margin { return 0 }
        let nCS = dist > 1e-6 ? d / dist : F3(0, 0, 1)   // capsule -> sphere
        let n = capsuleIsA ? -nCS : nCS                   // B -> A
        basisOut = orthonormalBasis(n)
        var c = ContactPoint()
        let xCap = q + nCS * rc
        let xSph = sph.positionLin - nCS * rs
        let (bodyA, xA, xB) = capsuleIsA ? (cap, xCap, xSph) : (sph, xSph, xCap)
        let bodyB = capsuleIsA ? sph : cap
        c.rA = roundAnchor(bodyA, xA)
        c.rB = roundAnchor(bodyB, xB)
        contacts.append(c)
        return 1
    }

    static func collideCapsuleBox(_ cap: CPURigid, _ box: CPURigid,
                                  capsuleIsA: Bool, margin: Float,
                                  _ contacts: inout [ContactPoint],
                                  _ basisOut: inout (F3, F3, F3)) -> Int {
        contacts.removeAll(keepingCapacity: true)
        let (p0, p1, rc) = capsuleSegment(cap)
        let half = box.size * 0.5
        let qc = box.positionAng.conjugate

        var found: [(q: F3, b: F3, n: F3)] = []
        for seed in [Float(0), 0.5, 1] {
            var q = p0 + (p1 - p0) * seed
            var bw = F3.zero
            for _ in 0..<5 {
                let lb = simd_clamp(qc.act(q - box.positionLin), -half, half)
                bw = box.positionAng.act(lb) + box.positionLin
                q = closestOnSegment(bw, p0, p1)
            }
            var d = q - bw
            var dist = length(d)
            var n: F3
            if dist < 1e-6 {
                let lb = qc.act(q - box.positionLin)
                let pen = half - abs(lb)
                var axisI = 0
                if pen.y < pen[axisI] { axisI = 1 }
                if pen.z < pen[axisI] { axisI = 2 }
                var nl = F3.zero
                nl[axisI] = lb[axisI] >= 0 ? 1 : -1
                n = box.positionAng.act(nl)
                bw = q + n * pen[axisI]
                dist = 0
            } else {
                if dist - rc > margin { continue }
                n = d / dist
            }
            // Scale-aware dedup: a flat 5 cm radius is longer than a
            // short capsule, collapsing every seed onto one contact point.
            let dedupR = 0.3 * rc + min(0.05, 0.25 * length(p1 - p0) / 2)
            if found.contains(where: { length($0.q - q) < dedupR }) { continue }
            found.append((q, bw, n))
        }
        if found.isEmpty { return 0 }
        let nAvg = normalize(found.reduce(F3.zero) { $0 + $1.n })
        let n = capsuleIsA ? -nAvg : nAvg
        basisOut = orthonormalBasis(n)
        for (i, f) in found.enumerated() {
            var c = ContactPoint()
            c.featureKey = Int32(i)
            let xCap = f.q - f.n * rc
            let (bodyA, xA, xB) = capsuleIsA ? (cap, xCap, f.b) : (box, f.b, xCap)
            let bodyB = capsuleIsA ? box : cap
            c.rA = roundAnchor(bodyA, xA)
            c.rB = roundAnchor(bodyB, xB)
            contacts.append(c)
        }
        return contacts.count
    }

    static func collideCapsuleCapsule(_ a: CPURigid, _ b: CPURigid,
                                      margin: Float,
                                      _ contacts: inout [ContactPoint],
                                      _ basisOut: inout (F3, F3, F3)) -> Int {
        contacts.removeAll(keepingCapacity: true)
        let (a0, a1, ra) = capsuleSegment(a)
        let (b0, b1, rb) = capsuleSegment(b)
        let (pa, pb) = closestPointsOnSegments(a0, a1, b0, b1)
        let d = pa - pb
        let dist = length(d)
        if dist > ra + rb + margin { return 0 }
        let n = dist > 1e-6 ? d / dist : F3(0, 0, 1)   // B -> A
        basisOut = orthonormalBasis(n)
        var c = ContactPoint()
        c.rA = pa - n * ra - a.positionLin
        c.rB = pb + n * rb - b.positionLin
        contacts.append(c)
        return 1
    }

    /// Torus spine sampled against the capsule segment (exact per sample) —
    /// the round-on-round bearing pair.
    static func collideTorusCapsule(_ torus: CPURigid, _ cap: CPURigid,
                                    torusIsA: Bool, margin: Float,
                                    _ contacts: inout [ContactPoint],
                                    _ basisOut: inout (F3, F3, F3)) -> Int {
        contacts.removeAll(keepingCapacity: true)
        let R = torus.size.x, rt = torus.size.y
        let (p0, p1, rc) = capsuleSegment(cap)
        let center = torus.positionLin
        let axis = torus.positionAng.act(F3(0, 0, 1))
        var u = abs(axis.x) > abs(axis.z) ? F3(-axis.y, axis.x, 0) : F3(0, -axis.z, axis.y)
        u = normalize(u)
        let v = cross(axis, u)

        var found: [(q: F3, b: F3)] = []
        var best: (q: F3, b: F3, dist: Float)? = nil
        for k in 0..<32 {
            let a = Float(k) / 32 * 2 * .pi
            let q = center + (u * cos(a) + v * sin(a)) * R
            let b = closestOnSegment(q, p0, p1)
            let dist = length(q - b)
            if best == nil || dist < best!.dist { best = (q, b, dist) }
            if dist - rt - rc > margin { continue }
            if found.contains(where: { length($0.q - q) < 0.35 * R }) { continue }
            if found.count < 4 { found.append((q, b)) }
        }
        if found.isEmpty { return 0 }
        var n = best!.q - best!.b
        let l = length(n)
        n = l > 1e-6 ? n / l : F3(0, 0, 1)               // capsule -> torus
        let nBA = torusIsA ? n : -n                       // B -> A
        basisOut = orthonormalBasis(nBA)
        for (i, f) in found.enumerated() {
            var nf = f.q - f.b
            let lf = length(nf)
            nf = lf > 1e-6 ? nf / lf : n
            var c = ContactPoint()
            c.featureKey = Int32(i)
            let xT = f.q - nf * rt
            let xC = f.b + nf * rc
            let (bodyA, xA, xB) = torusIsA ? (torus, xT, xC) : (cap, xC, xT)
            let bodyB = torusIsA ? cap : torus
            c.rA = xA - bodyA.positionLin
            c.rB = xB - bodyB.positionLin
            contacts.append(c)
        }
        return contacts.count
    }
}
