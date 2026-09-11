import SimCore
import simd

public extension Demos {
    /// Enlarged, compliant RJ45-shaped demonstrator, not a connector tolerance
    /// model. All plug volume, contact ribs and the underside latch are tets.
    /// Only the cable's ribbed strain relief is rigid. The socket is one fixed
    /// compound body with an open rectangular throat and a latch channel.
    static func cableEthernet(segments: Int = 32, plugStiffness: Float = 15000,
                              dampingTime: Float = 0.12, drag: Float = 0.8) -> PhysicsScene {
        precondition(segments >= 12 && plugStiffness.isFinite && plugStiffness > 0)
        var s = cableBench("Ethernet Insertion", length: 4.6, drag: drag)
        s.settings.dt = 1 / 240
        s.settings.iterations = 32
        s.settings.deterministic = true
        s.settings.particleDamping = 1.5
        s.settings.deformableCollisionMargin = 0.001
        s.settings.cameraTargetZ = 0.9
        s.settings.cameraDistance = 3.0
        let blue = F3(0.045, 0.38, 0.76), shell = F3(0.68, 0.84, 0.88)
        let gold = F3(1.0, 0.66, 0.13), metal = F3(0.64, 0.68, 0.72)
        func box(_ size: F3, _ p: F3, _ color: F3) {
            let body = s.addBody(size: size, density: 0, friction: 0.22, position: p)
            paintBody(&s, body, color)
        }
        // Elevated insertion bed with a gap for the underside latch. These
        // support the plug under gravity without pinning it or guiding a joint.
        for y: Float in [-0.119, 0.119] {
            box(F3(1.03, 0.065, 0.20), F3(-0.335, y, 0.717), F3(0.22, 0.27, 0.33))
        }
        // A panel-mounted jack. Its back wall owns all five throat pieces;
        // there is no hidden box filling the insertion cavity.
        let socket = s.addBody(size: F3(0.08, 0.55, 0.43), density: 0,
                               friction: 0.22, position: F3(0.65, 0, 0.855))
        paintBody(&s, socket, F3(0.10, 0.14, 0.19))
        func wall(_ size: F3, _ p: F3, _ color: F3 = F3(0.64, 0.68, 0.72)) {
            _ = s.addCollider(body: socket, size: size, friction: 0.22,
                localPosition: p - s.bodies[socket].position, renderColor: color)
        }
        for sign: Float in [-1, 1] {
            wall(F3(0.29, 0.085, 0.26), F3(0.465, sign * 0.20, 0.90))
            _ = s.addCollider(body: socket, size: F3(0.155, 0.085, 0.26), friction: 0.22,
                localPosition: F3(0.247, sign * 0.221, 0.90) - s.bodies[socket].position,
                localRotation: Quat(angle: -sign * 0.30, axis: F3(0,0,1)), renderColor: metal)
            wall(F3(0.43, 0.10, 0.06), F3(0.395, sign * 0.1075, 0.782))
            // Mounting ears and dark screw recesses dress the same rigid body.
            wall(F3(0.055, 0.14, 0.36), F3(0.205, sign * 0.31, 0.86), metal)
            _ = s.addCollider(body: socket, size: F3(repeating: 0.035),
                localPosition: F3(0.17, sign * 0.32, 0.86) - s.bodies[socket].position,
                shape: .sphere, collisionEnabled: false, renderColor: F3(0.07, 0.08, 0.10))
        }
        wall(F3(0.43, 0.315, 0.06), F3(0.395, 0, 1.025))
        wall(F3(0.43, 0.115, 0.045), F3(0.395, 0, 0.7325), F3(0.17, 0.20, 0.23))
        // Short entry lip in the latch channel, followed by a relief pocket.
        wall(F3(0.055, 0.105, 0.023), F3(0.21, 0, 0.7665))
        for i in 0..<8 {
            wall(F3(0.24, 0.010, 0.006), F3(0.465, -0.119 + Float(i) * 0.034, 0.991), gold)
        }
        // Status lights, separate from the contact geometry.
        for (y, color) in [(Float(-0.20), F3(0.2, 0.9, 0.44)), (Float(0.20), gold)] {
            _ = s.addCollider(body: socket, size: F3(0.009, 0.035, 0.018),
                localPosition: F3(0.171, y, 1.005) - s.bodies[socket].position,
                collisionEnabled: false, renderColor: color)
        }
        let points = (0...segments).map { i -> F3 in
            let t = Float(i) / Float(segments)
            return F3(-2.08 + 1.36 * t, -0.12 * pow(sin(.pi * t), 2),
                      0.655 + 0.245 * pow(t, 5))
        }
        let cable = try! s.addCable(points: points, radius: 0.028, density: 80,
            material: CableMaterial(stretchRigidity: 3500, shearRigidity: 1800,
                bendRigidity: 0.025, twistRigidity: 0.02, dampingTime: dampingTime), friction: 0.3)
        stripedCable(&s, cable, color: blue)
        let end = cable.bodyIDs.last!, endPose = s.bodies[cable.bodyIDs.last!]
        // The boot is a compound collider on the last link. No extra inertia
        // or kinematic drag proxy; pulling it transmits force through the cable.
        let bootCenter = F3(-0.80, 0, 0.90)
        _ = s.addCollider(body: end, size: F3(0.16, 0.35, 0.18), friction: 0.3,
            localPosition: endPose.rotation.inverse.act(bootCenter - endPose.position),
            localRotation: endPose.rotation.inverse, renderColor: blue)
        for i in 0..<5 {
            _ = s.addCollider(body: end, size: F3(0.014, 0.36, 0.19),
                localPosition: endPose.rotation.inverse.act(bootCenter + F3(-0.06 + Float(i) * 0.03, 0, 0) - endPose.position),
                localRotation: endPose.rotation.inverse, collisionEnabled: false,
                renderColor: F3(0.075, 0.49, 0.87))
        }
        // Cable links inside the enlarged strain relief are one physical
        // jacket. Exclude only those buried neighbours, whose artificial
        // capsule/boot contacts would otherwise fight the cable joints.
        for body in cable.bodyIDs.dropLast() where s.bodies[body].position.x > -0.95 {
            s.addJoint(SceneJoint(bodyA: end, bodyB: body, rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }
        // Structured, conforming lattice with eight narrow contact ribs.
        // Shared base vertices bond the ribs/latch to the plug; no overlapping
        // independent soft blocks or visual-only replacement of contact shape.
        var builder = EthernetTetBuilder(scene: s)
        let nx = 9, ny = 17, nz = 3
        var nodes = [Int](repeating: 0, count: nx * ny * nz)
        func offset(_ x: Int, _ y: Int, _ z: Int) -> Int { (x * ny + y) * nz + z }
        for x in 0..<nx { for y in 0..<ny { for z in 0..<nz {
            let taper: Float = x == nx - 1 ? 0.92 : 1
            let p = F3(-0.72 + Float(x) * 0.055, (Float(y) - 8) * 0.017 * taper,
                       0.82 + Float(z) * 0.08 - (x == nx - 1 ? Float(z) * 0.004 : 0))
            nodes[offset(x,y,z)] = builder.node(p, shell)
        } } }
        func n(_ x: Int, _ y: Int, _ z: Int) -> Int { nodes[offset(x,y,z)] }
        for x in 0..<(nx-1) { for y in 0..<(ny-1) { for z in 0..<(nz-1) {
            builder.cell([n(x,y,z), n(x+1,y,z), n(x+1,y+1,z), n(x,y+1,z),
                          n(x,y,z+1), n(x+1,y,z+1), n(x+1,y+1,z+1), n(x,y+1,z+1)], mu: plugStiffness)
        } } }
        for y in stride(from: 0, to: ny-1, by: 2) {
            var upper: [Int: Int] = [:]
            for x in 5..<nx { for j in [y, y+1] {
                let base = n(x,j,nz-1)
                upper[base] = builder.node(builder.scene.bodies[base].position + F3(0,0,0.005), gold)
            } }
            for x in 5..<(nx-1) {
                let base = [n(x,y,2), n(x+1,y,2), n(x+1,y+1,2), n(x,y+1,2)]
                builder.cell(base + base.map { upper[$0]! }, mu: plugStiffness)
            }
        }
        // Underside cantilever latch, bonded along two nose rows. Its root
        // shares every face vertex with the plug, with no overlapping tets.
        var latch: [Int] = []
        for x in 3..<9 { for y in 6...10 {
            let p = builder.scene.bodies[n(x,y,0)].position
            let depth: Float = x >= 7 ? 0 : (x == 6 ? 0.016 : 0.024)
            latch.append(builder.node(p - F3(0,0,depth + 0.012), shell))
            latch.append(x >= 7 ? n(x,y,0) : builder.node(p - F3(0,0,depth), shell))
        } }
        func l(_ x: Int, _ y: Int, _ z: Int) -> Int { latch[(x * 5 + y) * 2 + z] }
        for x in 0..<5 { for y in 0..<4 {
            builder.cell([l(x,y,0),l(x+1,y,0),l(x+1,y+1,0),l(x,y+1,0),
                          l(x,y,1),l(x+1,y,1),l(x+1,y+1,1),l(x,y+1,1)], mu: plugStiffness * 0.35)
        } }
        // Rear cross-section bonded to the boot at distributed material
        // points. All other plug nodes remain dynamic and freely deformable.
        for y in stride(from: 0, to: ny, by: 4) { for z in 0..<nz {
            let node = n(0,y,z), p = builder.scene.bodies[node].position
            builder.scene.addJoint(SceneJoint(bodyA: end, bodyB: node,
                rA: endPose.rotation.inverse.act(p - endPose.position), rB: .zero))
        } }
        return builder.scene
    }
}

/// Local authoring helper. Lump density * rest volume onto the four vertices;
/// changing tessellation does not silently change the plug's mass.
private struct EthernetTetBuilder {
    var scene: PhysicsScene
    mutating func node(_ p: F3, _ color: F3) -> Int {
        let id = scene.addParticle(radius: 0.002, mass: 0.000001, friction: 0.25, position: p)
        paintBody(&scene, id, color)
        return id
    }
    mutating func cell(_ v: [Int], mu: Float) {
        for t in [(0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6),(0,5,1,6)] {
            let ids = [v[t.0],v[t.1],v[t.2],v[t.3]]
            let p = ids.map { scene.bodies[$0].position }
            let mass = 30 * abs(dot(p[1]-p[0], cross(p[2]-p[0], p[3]-p[0]))) / 24
            for id in ids {
                // addParticle uses a sphere density to encode nodal mass.
                let r = scene.bodies[id].size.x * 0.5
                scene.bodies[id].density += mass / (4 * .pi / 3 * r * r * r)
            }
            scene.addTet(SceneTet(ids: (ids[0],ids[1],ids[2],ids[3]), mu: mu, lambda: 5*mu))
        }
    }
}
