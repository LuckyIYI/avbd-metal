import simd

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
    /// SAT OBB-OBB collision. Returns contact count; fills basis (rows n, t1, t2).
    static func collide(_ bodyA: CPURigid, _ bodyB: CPURigid,
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
