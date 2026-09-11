import SimCore
import simd

public extension Demos {
    static func cableDemoTitle(_ name: String) -> String? {
        switch name {
        case "cablethreading": return "Thread the Needle"
        case "cabletwisting": return "Twist Laboratory"
        case "cablegrippers": return "Snap-fit Cable Routing"
        case "cableplastic": return "Bend & Keep"
        default: return nil
        }
    }

    static func cableDemoInstructions(_ name: String) -> String? {
        switch name {
        case "cablethreading":
            return "Grab the gold tip and feed it through the three guide rings. Orbit to line up the holes; pull it back to try again."
        case "cabletwisting":
            return "The left chuck winds two striped cables together. Watch the stripes carry the twist. Grab a strand to deflect it, or set Turns / s to zero."
        case "cablegrippers":
            return "Pull the gold end toward you to pop the cable out of the white clips, one at a time. Their narrow lips flex open under contact. Reset to re-seat the cable."
        case "cableplastic":
            return "Bend and release the gold tips. The blue cable springs back; the copper cable keeps bends past its yield threshold. Reset restores both. Drag acts on both cables; damping absorbs internal vibration."
        default: return nil
        }
    }

    static func cableThreading(segments: Int = 40, holeRadius: Float = 0.13,
                                dampingTime: Float = 0.12, drag: Float = 0.8) -> PhysicsScene {
        precondition(segments >= 12 && holeRadius >= 0.08 && holeRadius.isFinite)
        var s = cableBench("Thread the Needle", length: 6.4, drag: drag)
        let tube: Float = 0.045, height: Float = 0.86
        let rotation = Quat(angle: .pi / 2, axis: F3(0, 1, 0))
        for (i, x) in [Float(0), 0.65, 1.3].enumerated() {
            let color = [F3(0.95, 0.42, 0.14), F3(0.32, 0.63, 0.93), F3(0.46, 0.80, 0.55)][i]
            let ring = s.addTorus(major: holeRadius + tube, minor: tube,
                density: 0, friction: 0.25, position: F3(x, 0, height), rotation: rotation)
            paintBody(&s, ring, color)
            for sign: Float in [-1, 1] {
                _ = coloredBox(&s, size: F3(0.12, 0.08, 0.26),
                    position: F3(x, sign * (holeRadius + tube), 0.73), color: color)
            }
            _ = coloredBox(&s, size: F3(0.24, 0.62, 0.045),
                position: F3(x, 0, 0.635), color: F3(0.12, 0.16, 0.22))
        }
        let points = (0...segments).map { i -> F3 in
            let t = Float(i) / Float(segments)
            return F3(-2.65 + 2.35 * t, 0.07 * sin(2 * .pi * t),
                      0.69 + 0.17 * pow(t, 6))
        }
        let cable = try! s.addCable(points: points, radius: 0.03, density: 160,
            material: CableMaterial(stretchRigidity: 2500, shearRigidity: 1500,
                bendRigidity: 0.018, twistRigidity: 0.015, dampingTime: dampingTime), friction: 0.35)
        stripedCable(&s, cable, color: F3(0.06, 0.55, 0.62))
        addGrabBead(&s, cable: cable, atStart: false)
        return s
    }

    static func cableTwisting(segments: Int = 40, turnRate: Float = 0.10,
                               dampingTime: Float = 0.12, drag: Float = 0.8) -> PhysicsScene {
        precondition(segments >= 12 && turnRate >= 0 && turnRate.isFinite)
        var s = cableBench("Twist Laboratory", length: 4.8, drag: drag)
        let q = Quat(angle: .pi / 2, axis: F3(0, 1, 0))
        var chucks: [Int] = []
        for x: Float in [-1.48, 1.48] {
            _ = coloredBox(&s, size: F3(0.22, 0.5, 0.5),
                position: F3(x, 0, 0.87), color: F3(0.12, 0.17, 0.25))
            let chuck = s.addBody(size: F3(0.44, 0.44, 0.12), density: 0,
                friction: 0.3, position: F3(x, 0, 1.2), rotation: q)
            paintBody(&s, chuck, F3(0.87, 0.49, 0.16))
            chucks.append(chuck)
            // A bright radial witness makes the drive visible even at rest.
            _ = s.addCollider(body: chuck, size: F3(0.30, 0.025, 0.015),
                localPosition: F3(0, 0, x < 0 ? 0.065 : -0.065),
                collisionEnabled: false, renderColor: F3(1, 0.86, 0.44))
        }
        s.addSpinner(SceneSpinner(body: chucks[0], axis: F3(1, 0, 0), omega: turnRate * 2 * .pi))
        for strand in 0..<2 {
            let y: Float = strand == 0 ? -0.085 : 0.085
            let points = (0...segments).map { i -> F3 in
                let t = Float(i) / Float(segments)
                return F3(-1.34 + 2.68 * t, y, 1.2 - 0.16 * sin(.pi * t))
            }
            let cable = try! s.addCable(points: points, radius: 0.027, density: 180,
                material: CableMaterial(stretchRigidity: 3500, shearRigidity: 1800,
                    bendRigidity: 0.01, twistRigidity: 0.025, dampingTime: dampingTime), friction: 0.45)
            stripedCable(&s, cable, color: strand == 0 ? F3(0.12, 0.60, 0.79) : F3(0.93, 0.29, 0.17))
            for end in 0..<2 {
                let body = end == 0 ? cable.bodyIDs[0] : cable.bodyIDs.last!
                let anchor = end == 0 ? cable.startAnchor : cable.endAnchor
                let point = s.bodies[body].position + s.bodies[body].rotation.act(anchor)
                let chuck = chucks[end]
                s.addJoint(SceneJoint(bodyA: chuck, bodyB: body,
                    rA: q.inverse.act(point - s.bodies[chuck].position), rB: anchor,
                    stiffnessAng: .infinity))
            }
        }
        return s
    }

    static func cableGrippers(segments: Int = 64, padStiffness: Float = 30000,
                              friction: Float = 0.8, dampingTime: Float = 0.12,
                              drag: Float = 0.8) -> PhysicsScene {
        precondition(segments >= 24 && padStiffness > 0 && padStiffness.isFinite
            && friction >= 0 && friction.isFinite)
        var s = cableBench("Snap-fit Cable Routing", length: 4.4, drag: drag)
        s.settings.dt = 1 / 240
        s.settings.iterations = 24
        s.settings.deterministic = true
        s.settings.particleDamping = 0.8
        s.settings.deformableCollisionMargin = 0.001
        s.settings.cameraTargetZ = 1.35
        s.settings.cameraDistance = 4.5
        // An extrusion frame carries four small split clips. Cable contact
        // is owned by each clip's curved tet boundary, including its lips.
        extrusion(&s, center: F3(-1.3, 0, 1.62), length: 2.0, vertical: true)
        extrusion(&s, center: F3(0.2, 0, 0.90), length: 3.2, vertical: false)
        extrusion(&s, center: F3(1.65, 0, 0.78), length: 0.34, vertical: true)
        for z: Float in [1.55, 2.18] {
            addCableClip(&s, center: F3(-1.3, -0.175, z), axis: F3(0, 0, 1),
                         mu: padStiffness, friction: friction)
        }
        for x: Float in [-0.55, 0.70] {
            addCableClip(&s, center: F3(x, -0.175, 0.90), axis: F3(1, 0, 0),
                         mu: padStiffness, friction: friction)
        }
        // Uniform arclength sampling around the 90-degree routing bend.
        let vertical: Float = 1.38, radius: Float = 0.27, horizontal: Float = 2.53
        let arc = radius * .pi / 2
        let total = vertical + arc + horizontal
        let points = (0...segments).map { i -> F3 in
            let d = total * Float(i) / Float(segments)
            if d <= vertical { return F3(-1.3, -0.175, 2.55 - d) }
            if d < vertical + arc {
                let angle = .pi + (d - vertical) / radius
                return F3(-1.03 + radius * cos(angle), -0.175, 1.17 + radius * sin(angle))
            }
            return F3(-1.03 + d - vertical - arc, -0.175, 0.90)
        }
        let cable = try! s.addCable(points: points, radius: 0.035, density: 30,
            material: CableMaterial(stretchRigidity: 3500, shearRigidity: 1800,
                bendRigidity: 0.12, twistRigidity: 0.02, dampingTime: dampingTime), friction: friction)
        stripedCable(&s, cable, color: F3(0.08, 0.48, 0.82))
        addGrabBead(&s, cable: cable, atStart: false)
        return s
    }


    /// Same initial stiffness and mass, different unloading behavior. Vertical
    /// cantilevers avoid gravity-induced plastic flow before the first grab.
    static func cablePlastic(segments: Int = 24, yieldCurvature: Float = 1.5,
                              dampingTime: Float = 0.12, drag: Float = 0.8) -> PhysicsScene {
        precondition(segments >= 12)
        var s = cableBench("Bend & Keep", length: 4.8, drag: drag)
        s.settings.dt = 1 / 240
        s.settings.iterations = 32
        s.settings.cameraTargetZ = 1.4
        for i in 0..<2 {
            let x: Float = i == 0 ? -0.8 : 0.8
            let color = i == 0 ? F3(0.08, 0.52, 0.85) : F3(0.91, 0.43, 0.17)
            _ = coloredBox(&s, size: F3(0.42, 0.42, 0.16),
                position: F3(x, 0, 0.67), color: F3(0.12, 0.17, 0.25))
            _ = coloredBox(&s, size: F3(0.16, 0.16, 0.06),
                position: F3(x, 0, 0.78), color: color)
            let points = (0...segments).map { j in
                F3(x, 0, 0.79 + 1.5 * Float(j) / Float(segments))
            }
            let cable = try! s.addCable(points: points, radius: 0.025, density: 30,
                material: CableMaterial(stretchRigidity: 3000, shearRigidity: 1600,
                    bendRigidity: 0.8, twistRigidity: 0.08, dampingTime: dampingTime,
                    yieldCurvature: i == 0 ? nil : yieldCurvature),
                friction: 0.4, fixedSegments: [0])
            stripedCable(&s, cable, color: color)
            addGrabBead(&s, cable: cable, atStart: false)
        }
        return s
    }

}

private func cableBench(_ name: String, length: Float, drag: Float = 0.8) -> PhysicsScene {
    precondition(drag >= 0 && drag.isFinite)
    var s = PhysicsScene(name: name)
    s.settings.dt = 1 / 120
    s.settings.iterations = 28
    s.settings.collisionMargin = 0.0015
    s.settings.rigidLinearDamping = drag
    s.settings.rigidAngularDamping = drag
    s.settings.cameraDistance = 5
    s.settings.cameraTargetZ = 0.8
    _ = coloredBox(&s, size: F3(length, 2.0, 0.12),
        position: F3(0, 0, 0.55), color: F3(0.20, 0.26, 0.34))
    _ = coloredBox(&s, size: F3(length + 0.6, 3.0, 0.08),
        position: F3(0, 0, -0.04), color: F3(0.10, 0.13, 0.18))
    for x in [-length * 0.4, length * 0.4] {
        _ = coloredBox(&s, size: F3(0.12, 1.7, 0.5),
            position: F3(x, 0, 0.25), color: F3(0.14, 0.18, 0.24))
    }
    return s
}

private func extrusion(_ s: inout PhysicsScene, center: F3, length: Float, vertical: Bool) {
    let size = vertical ? F3(0.18, 0.18, length) : F3(length, 0.18, 0.18)
    let rail = coloredBox(&s, size: size, position: center, color: F3(0.26, 0.28, 0.30))
    // Inset-looking T-slots are visual instances on the existing static body.
    for offset: Float in [-0.047, 0.047] {
        _ = s.addCollider(body: rail,
            size: vertical ? F3(0.025, 0.004, length - 0.025) : F3(length - 0.025, 0.004, 0.025),
            localPosition: vertical ? F3(offset, -0.092, 0) : F3(0, -0.092, offset),
            collisionEnabled: false, renderColor: F3(0.065, 0.075, 0.085))
    }
}

private func addCableClip(_ s: inout PhysicsScene, center: F3, axis: F3,
                          mu: Float, friction: Float) {
    let front = F3(0, -1, 0), across = cross(axis, front)
    let axial = 3, radial = 3, angular = 17
    let firstNode = s.bodies.count
    func index(_ u: Int, _ r: Int, _ a: Int) -> Int {
        firstNode + (u * radial + r) * angular + a
    }
    // 270-degree annular extrusion: a 56 mm throat retains a 70 mm cable
    // inside an 80 mm bore. Only the outer rear arc is bolted to the frame.
    for u in 0..<axial {
        for r in 0..<radial {
            for a in 0..<angular {
                let angle = Float.pi / 4 + Float(a) * (1.5 * .pi) / Float(angular - 1)
                let radius: Float = 0.04 + 0.0175 * Float(r)
                let position = center + axis * (Float(u) - 1) * 0.06
                    + radius * (front * cos(angle) + across * sin(angle))
                let node = s.addParticle(radius: 0.005, mass: 0.0008,
                                        friction: friction, position: position)
                if r == radial - 1 && cos(angle) < -0.45 { s.bodies[node].density = 0 }
                paintBody(&s, node, F3(0.92, 0.93, 0.91))
            }
        }
    }
    for u in 0..<(axial - 1) {
        for r in 0..<(radial - 1) {
            for a in 0..<(angular - 1) {
                let v = [index(u,r,a), index(u+1,r,a), index(u+1,r+1,a), index(u,r+1,a),
                         index(u,r,a+1), index(u+1,r,a+1), index(u+1,r+1,a+1), index(u,r+1,a+1)]
                // Conforming six-tet split with the same body diagonal in
                // every cell, so no cracks or duplicate contact surfaces.
                for t in [(0,1,2,6), (0,2,3,6), (0,3,7,6), (0,7,4,6), (0,4,5,6), (0,5,1,6)] {
                    s.addTet(SceneTet(ids: (v[t.0], v[t.1], v[t.2], v[t.3]),
                                      mu: mu, lambda: 5 * mu))
                }
            }
        }
    }
    let mount = coloredBox(&s, size: abs(axis) * 0.18 + abs(across) * 0.16 + F3(0, 0.035, 0),
        position: center + F3(0, 0.075, 0), color: F3(0.70, 0.72, 0.73))
    for sign: Float in [-1, 1] {
        _ = s.addCollider(body: mount, size: F3(repeating: 0.022),
            localPosition: axis * sign * 0.071 + F3(0, -0.019, 0), shape: .sphere,
            collisionEnabled: false, renderColor: F3(0.12, 0.13, 0.14))
    }
}

@discardableResult
private func coloredBox(_ s: inout PhysicsScene, size: F3, position: F3, color: F3) -> Int {
    let body = s.addBody(size: size, density: 0, friction: 0.35, position: position)
    paintBody(&s, body, color)
    return body
}

private func paintBody(_ s: inout PhysicsScene, _ body: Int, _ color: F3) {
    for i in s.colliders.indices where s.colliders[i].body == body {
        s.colliders[i].renderColor = color
    }
}

private func addGrabBead(_ s: inout PhysicsScene, cable: SceneCable, atStart: Bool) {
    let body = atStart ? cable.bodyIDs[0] : cable.bodyIDs.last!
    let anchor = atStart ? cable.startAnchor : cable.endAnchor
    // Same body and inertia: an obvious pick target without another solve block.
    _ = s.addCollider(body: body, size: F3(repeating: 0.105), friction: 0.35,
        localPosition: anchor, shape: .sphere, renderColor: F3(1, 0.65, 0.16))
}

private func stripedCable(_ s: inout PhysicsScene, _ cable: SceneCable, color: F3) {
    for (i, body) in cable.bodyIDs.enumerated() {
        paintBody(&s, body, color)
        // A narrow longitudinal surface strip follows the material frame.
        // Four vertices per segment; no additional collision or dynamics.
        let radius = cable.radius * 1.015
        let half = cable.restLengths[i] / 2
        let angles: [Float] = [.pi * 1.25 - 0.16, .pi * 1.25 + 0.16]
        let normals = angles.map { F3(cos($0), sin($0), 0) }
        let vertices = [normals[0] * radius - F3(0, 0, half),
                        normals[1] * radius - F3(0, 0, half),
                        normals[0] * radius + F3(0, 0, half),
                        normals[1] * radius + F3(0, 0, half)]
        s.addRigidMesh(SceneRigidMesh(body: body,
            mesh: SurfaceMesh(vertices: vertices, normals: [normals[0], normals[1], normals[0], normals[1]],
                              triangles: [(0, 1, 2), (1, 3, 2)]),
            color: F3(0.95, 0.94, 0.81), roughness: 0.65))
    }
}
