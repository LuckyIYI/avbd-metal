import SimCore
import simd

/// SI-unit connector insertion task. The tool is an already-closed, no-slip
/// grasp of the strain relief. A floating tool couples to the commanded wrist
/// through three finite springs. The plug is a tetrahedral solid; its thin
/// latch uses membrane and plate-bending elements with the same PC material.
/// Sources and uncalibrated parameters: Documentation/EthernetInsertion.md.
public struct EthernetInsertionTask {
    public static let plugLength: Float = 0.02248
    public static let plugWidth: Float = 0.01168
    public static let plugHeight: Float = 0.00660
    public static let contactPitch: Float = 0.00102
    public static let seatDepth: Float = 0.0148
    public static let axisHeight: Float = 0.033
    public static let duration: Float = 8
    public static let couplingStiffness: Float = 8000  // each of three springs, N/m
    public static let forceLimit: Float = 20
    public var scene: PhysicsScene
    public var socketBody: Int
    public var remoteConnectorBody: Int
    public var cable: SceneCable
    public var contactWires: [SceneCable]
    public var wristBody: Int
    public var toolBody: Int
    public var noseNodes: [Int]
    public var plugNodes: [Int]
    public var latchNodes: [Int]
    public var couplingAnchors: [F3]
    public var lateralError: Float
    public var yawError: Float
    public var plugRearInTool: Float = 0.008

    public struct Command {
        public var position: F3
        public var rotation: Quat
        public var linearVelocity: F3
        public var angularVelocity: F3
        public var phase: String
    }
    public func command(at time: Float) -> Command {
        func blend(_ t: Float) -> Float {
            let u = min(max(t, 0), 1)
            return u * u * u * (10 + u * (-15 + 6 * u))
        }
        func pose(_ t: Float) -> (F3, Quat, String) {
            let start = F3(-0.058, 0.004, Self.axisHeight + 0.005)
            let aligned = F3(
                -Self.plugLength - plugRearInTool - 0.003,
                lateralError, Self.axisHeight)
            let yaw0: Float = 6 * .pi / 180
            if t < 0.5 { return (start, Quat(angle: yaw0, axis: F3(0, 0, 1)), "Settle") }
            if t < 2.5 {
                let u = blend((t - 0.5) / 2)
                return (
                    mix(start, aligned, t: F3(repeating: u)),
                    Quat(angle: yaw0 + (yawError - yaw0) * u, axis: F3(0, 0, 1)), "Approach & align"
                )
            }
            let seated = F3(
                Self.seatDepth - Self.plugLength - plugRearInTool,
                lateralError, Self.axisHeight)
            let u = blend((t - 2.5) / 4)
            return (
                mix(aligned, seated, t: F3(repeating: u)),
                Quat(angle: yawError, axis: F3(0, 0, 1)), t < 6.5 ? "Insert" : "Seat & hold"
            )
        }
        let (p, q, phase) = pose(time)
        let h: Float = 0.0005
        let (before, qb, _) = pose(max(time - h, 0))
        let (after, qa, _) = pose(time + h)
        return Command(
            position: p, rotation: q, linearVelocity: (after - before) / (2 * h),
            angularVelocity: quatSub(qa, qb) / (2 * h), phase: phase)
    }
    /// Reaction of the physical wrist/tool coupling, not a contact-force guess.
    public func couplingForce(
        wristPosition: F3, wristRotation: Quat,
        toolPosition: F3, toolRotation: Quat
    ) -> F3 {
        couplingAnchors.reduce(F3.zero) { force, a in
            force + Self.couplingStiffness
                * (wristPosition + wristRotation.act(a)
                    - toolPosition - toolRotation.act(a))
        }
    }
}

extension Demos {
    public static func cableEthernet(
        segments: Int = 32, youngModulus: Float = 2.4e9,
        dampingTime: Float = 0.03, drag: Float = 0.1,
        lateralError: Float = 0, yawError: Float = 0
    ) -> PhysicsScene {
        ethernetInsertionTask(
            segments: segments, youngModulus: youngModulus,
            dampingTime: dampingTime, drag: drag, lateralError: lateralError,
            yawError: yawError
        ).scene
    }

    public static func ethernetInsertionTask(
        segments: Int = 32, youngModulus: Float = 2.4e9,
        dampingTime: Float = 0.03, drag: Float = 0.1,
        lateralError: Float = 0, yawError: Float = 0
    ) -> EthernetInsertionTask {
        precondition(segments >= 16 && youngModulus.isFinite && youngModulus > 0)
        let pc = F3(0.79, 0.85, 0.87)
        let blue = F3(0.025, 0.30, 0.67)
        let dark = F3(0.055, 0.065, 0.08)
        let steel = F3(0.57, 0.62, 0.66)
        let gold = F3(0.94, 0.62, 0.16)
        let z = EthernetInsertionTask.axisHeight
        var s = PhysicsScene(name: "Ethernet — Robotic Insertion")
        s.settings.dt = 1 / 480
        s.settings.iterations = 16
        s.settings.gravity = -9.81
        s.settings.collisionMargin = 0.000025
        s.settings.deformableCollisionMargin = 0.000015
        s.settings.deterministic = true
        s.settings.particleDamping = 20
        // Internal velocity smoothing suppresses under-resolved thin-shell
        // vibration. This numerical viscosity is an authored, uncalibrated
        // material damping approximation; the latch keeps every elastic DOF.
        s.settings.clothViscosity = 0.3
        s.settings.clothRenderScale = 1
        s.settings.alpha = 0.9
        // Keep finite tool, boot and contact-root springs at their authored
        // physical stiffness. Gamma decay is an AL convergence heuristic.
        s.settings.gamma = 1
        s.settings.betaLin = 100000
        s.settings.rigidLinearDamping = drag
        s.settings.rigidAngularDamping = drag
        s.settings.cameraDistance = 0.40
        s.settings.cameraTargetZ = 0.020
        s.settings.cameraTargetX = -0.075
        s.settings.cameraTargetY = 0
        s.settings.cameraAzimuth = -2.15
        s.settings.cameraElevation = 0.65
        func box(_ size: F3, _ p: F3, _ color: F3, density: Float = 0) -> Int {
            let b = s.addBody(size: size, density: density, friction: 0.25, position: p)
            paintBody(&s, b, color)
            return b
        }
        func part(
            _ body: Int, _ size: F3, _ local: F3, _ color: F3,
            collision: Bool = true, rotation: Quat = Quat(real: 1, imag: .zero)
        ) {
            _ = s.addCollider(
                body: body, size: size, friction: 0.25,
                localPosition: local, localRotation: rotation, collisionEnabled: collision, renderColor: color
            )
        }
        // Machined fixture, PCB and shielded panel jack. No guide rails or
        // supporting track touches the flying connector during insertion.
        let table = box(F3(0.30, 0.18, 0.006), F3(-0.07, 0, -0.003), F3(0.21, 0.24, 0.28))
        _ = box(F3(0.043, 0.050, 0.020), F3(0.027, 0, 0.010), steel)
        let board = box(F3(0.060, 0.048, 0.0016), F3(0.031, 0, 0.0208), F3(0.025, 0.24, 0.18))
        for y: Float in [-0.019, 0.019] {
            for x: Float in [0.009, 0.052] {
                _ = s.addCollider(
                    body: board, size: F3(repeating: 0.0025), localPosition: F3(x - 0.031, y, 0.0011),
                    shape: .sphere, collisionEnabled: false, renderColor: steel)
            }
        }
        let socket = box(F3(0.002, 0.018, 0.013), F3(0.0162, 0, z - 0.001), dark)
        let socketOrigin = s.bodies[socket].position
        func wall(
            _ size: F3, _ p: F3, _ color: F3 = F3(0.065, 0.075, 0.09),
            rotation: Quat = Quat(real: 1, imag: .zero), collision: Bool = true
        ) {
            part(socket, size, p - socketOrigin, color, collision: collision, rotation: rotation)
        }
        for sign: Float in [-1, 1] {
            // 0.16 mm total lateral clearance; real geometry handles alignment.
            wall(F3(0.014, 0.002, 0.0090), F3(0.008, sign * 0.00692, z))
            wall(
                F3(0.0016, 0.0016, 0.0090), F3(0.00035, sign * 0.00705, z),
                rotation: Quat(angle: -sign * 0.26, axis: F3(0, 0, 1)))
            // Bottom ledges leave a narrow keyway for the compliant latch.
            wall(F3(0.015, 0.0042, 0.0015), F3(0.0075, sign * 0.00382, z - 0.00414))
            wall(F3(0.016, 0.00035, 0.011), F3(0.0075, sign * 0.0091, z - 0.0004), steel)
            wall(F3(0.00035, 0.0021, 0.011), F3(-0.0003, sign * 0.0082, z - 0.0004), steel)
        }
        wall(F3(0.015, 0.01184, 0.0014), F3(0.0075, 0, z + 0.00412))
        wall(F3(0.016, 0.01855, 0.00035), F3(0.0075, 0, z + 0.00528), steel)
        wall(F3(0.016, 0.01855, 0.00035), F3(0.0075, 0, z - 0.0060), steel)
        // Rounded molded entry lip and relief behind it. The soft tab bends
        // over this cam; a sharp leading box face would catch its underside.
        _ = s.addCollider(
            body: socket, size: F3(0.0022, 0.0006, 0), friction: 0.25,
            localPosition: F3(0.0007, 0, z - 0.0047) - socketOrigin,
            localRotation: Quat(angle: .pi / 2, axis: F3(1, 0, 0)), shape: .capsule, renderColor: dark)
        wall(F3(0.011, 0.0034, 0.0006), F3(0.0092, 0, z - 0.0052))
        // Termination pins and PCB traces are display geometry on the jack.
        for i in 0..<8 {
            wall(
                F3(0.002, 0.00042, 0.001), F3(0.018, Float(i) * 0.00102 - 0.00357, 0.0222), gold,
                collision: false)
        }
        let initial = F3(-0.058, 0.004, z + 0.005)
        let rotation = Quat(angle: 6 * .pi / 180, axis: F3(0, 0, 1))
        let wrist = box(F3(0.008, 0.024, 0.024), initial, F3(0.88, 0.40, 0.07))
        s.bodies[wrist].rotation = rotation
        for i in s.colliders.indices where s.colliders[i].body == wrist {
            s.colliders[i].localPosition.z = 0.026
        }
        part(wrist, F3(0.021, 0.016, 0.016), F3(-0.014, 0, 0.026), steel)
        part(wrist, F3(0.055, 0.006, 0.006), F3(-0.034, 0, 0.038), dark)
        part(wrist, F3(0.006, 0.0006, 0.020), F3(0, -0.0122, 0.026), dark, collision: false)
        for x: Float in [-0.002, 0.002] {
            for dz: Float in [0.0175, 0.0345] {
                _ = s.addCollider(
                    body: wrist, size: F3(repeating: 0.0014),
                    localPosition: F3(x, -0.0126, dz), shape: .sphere, collisionEnabled: false,
                    renderColor: steel)
            }
        }
        for y: Float in [-0.0075, 0.0075] {
            _ = s.addCollider(
                body: wrist, size: F3(0.010, 0.0015, 0), localPosition: F3(0, y, 0.014),
                shape: .capsule, collisionEnabled: false, renderColor: steel)
        }
        let tool = box(F3(0.009, 0.027, 0.018), initial, F3(0.14, 0.17, 0.21), density: 15000)
        s.bodies[tool].rotation = rotation
        for i in s.colliders.indices where s.colliders[i].body == tool {
            s.colliders[i].collisionEnabled = false
            s.colliders[i].isRendered = false
        }
        part(tool, F3(0.009, 0.027, 0.006), F3(0, 0, 0.012), dark)
        for sign: Float in [-1, 1] { part(tool, F3(0.009, 0.005, 0.022), F3(0, sign * 0.011, 0.001), dark) }
        // The moving carriage and grip jaws form one inertial tool body.
        for sign: Float in [-1, 1] {
            part(tool, F3(0.012, 0.003, 0.012), F3(0.0015, sign * 0.0085, 0), steel)
            part(tool, F3(0.007, 0.001, 0.009), F3(0.0035, sign * 0.0065, 0), dark)
            for x: Float in [-0.002, 0.005] {
                _ = s.addCollider(
                    body: tool, size: F3(repeating: 0.0017), localPosition: F3(x, sign * 0.0102, 0.002),
                    shape: .sphere, collisionEnabled: false, renderColor: dark)
            }
        }
        let anchors = [F3(0, 0.008, 0), F3(0, -0.004, 0.00693), F3(0, -0.004, -0.00693)]
        for a in anchors {
            s.addJoint(
                SceneJoint(
                    bodyA: wrist, bodyB: tool, rA: a, rB: a,
                    stiffnessLin: EthernetInsertionTask.couplingStiffness))
        }
        // The wrist is compliant in translation but maintains orientation,
        // as a robot insertion tool does. Generic AVBD angular joints square
        // their size-dependent torque arm in the energy; convert Nm/rad here.
        let wristArm2 = length_squared(s.bodies[wrist].size + s.bodies[tool].size)
        s.addJoint(
            SceneJoint(
                bodyA: wrist, bodyB: tool, rA: .zero, rB: .zero,
                stiffnessLin: 0, stiffnessAng: 15 / (wristArm2 * wristArm2)))
        // Chamfered rubber boot in the established grasp, with five fine ribs.
        part(tool, F3(0.0118, 0.0119, 0.0088), F3(0.0019, 0, 0), blue)
        for i in 0..<5 {
            part(
                tool, F3(0.00065, 0.0121, 0.0090), F3(-0.002 + Float(i) * 0.002, 0, 0), blue, collision: false
            )
        }
        let cableEnd = initial + rotation.act(F3(-0.004, 0, 0))
        // A second, already-mated endpoint on the bench. The service loop
        // has surplus arc length for the full 43 mm wrist stroke; attaching
        // the old almost-straight lead would put the insertion under tension.
        let remote = box(F3(0.035, 0.038, 0.026), F3(-0.195, 0.045, 0.013), dark)
        part(remote, F3(0.001, 0.017, 0.014), F3(0.018, 0, 0.005), steel)
        part(remote, F3(0.007, 0.01168, 0.0066), F3(0.021, 0, 0.005), pc)
        part(remote, F3(0.011, 0.0119, 0.0088), F3(0.0265, 0, 0.005), blue)
        for y: Float in [-0.006, 0.006] {
            part(remote, F3(0.001, 0.002, 0.001), F3(0.0181, y, 0.014),
                 F3(0.15, 0.85, 0.25), collision: false)
        }
        let route = [F3(-0.163,0.045,0.018), F3(-0.140,0.045,0.018),
                     F3(-0.125,0.025,0.003), F3(-0.155,-0.045,0.003),
                     F3(-0.125,-0.068,0.003), F3(-0.091,-0.057,0.003),
                     F3(-0.095,-0.015,0.009), cableEnd]
        let points = ethernetServiceLoop(route, segments: segments)
        let cable = try! s.addCable(
            points: points, radius: 0.00273, density: 1450,
            material: CableMaterial(
                stretchRigidity: 3000, shearRigidity: 1200,
                bendRigidity: 0.002, twistRigidity: 0.001, dampingTime: dampingTime), friction: 0.35)
        stripedCable(&s, cable, color: blue)
        let first = cable.bodyIDs[0]
        let rootArm2 = length_squared(s.bodies[remote].size + s.bodies[first].size)
        s.addJoint(SceneJoint(bodyA: remote, bodyB: first,
            rA: points[0] - s.bodies[remote].position, rB: cable.startAnchor,
            stiffnessLin: 1e6, stiffnessAng: 0.1 / (rootArm2 * rootArm2)))
        let end = cable.bodyIDs.last!
        let bootArm2 = length_squared(s.bodies[tool].size + s.bodies[end].size)
        s.addJoint(
            SceneJoint(
                bodyA: tool, bodyB: end, rA: F3(-0.004, 0, 0), rB: cable.endAnchor,
                stiffnessLin: 1e6, stiffnessAng: 50 / (bootArm2 * bootArm2)))
        for body in cable.bodyIDs.dropLast() where length(s.bodies[body].position - cableEnd) < 0.020 {
            s.collisionExclusions.append(SceneCollisionExclusion(bodyA: tool, bodyB: body))
        }
        // Eight round phosphor-bronze spring contacts. Each is a single
        // native rod with one bending mode at its root: k_tip = 3 E I / L^3.
        // Continuous capsule/triangle witnesses cover the entire wire length.
        let leafLength: Float = 0.008
        let wireRadius: Float = 0.00015
        let wireI = Float.pi * pow(wireRadius, 4) / 4
        let leafEI: Float = 110e9 * wireI
        for i in 0..<8 {
            let root = F3(0.006, Float(i) * 0.00102 - 0.00357, z + 0.00365)
            let tip = root + Quat(angle: 0.137, axis: F3(0, 1, 0)).act(F3(leafLength, 0, 0))
            let wire = try! s.addCable(
                points: [root, tip], radius: wireRadius, density: 8800,
                material: CableMaterial(
                    stretchRigidity: 110e9 * Float.pi * wireRadius * wireRadius,
                    shearRigidity: 41e9 * Float.pi * wireRadius * wireRadius, bendRigidity: leafEI,
                    twistRigidity: 82e9 * wireI), friction: 0.25)
            let leaf = wire.bodyIDs[0]
            paintBody(&s, leaf, gold)
            let arm2 = length_squared(s.bodies[socket].size + s.bodies[leaf].size)
            s.addJoint(
                SceneJoint(
                    bodyA: socket, bodyB: leaf, rA: root - socketOrigin,
                    rB: wire.startAnchor, stiffnessLin: 1e6,
                    stiffnessAng: (3 * leafEI / leafLength) / (arm2 * arm2)))
        }
        // Thin-walled PC housing with a rear crimp cavity. More elements are
        // placed near the contact face and latch; mass is volume-lumped.
        let xs: [Float] = [-0.02248, -0.018, -0.013, -0.008, -0.005, -0.002, -0.00035, 0]
        let ys: [Float] = [-0.00584, -0.00514, -0.00134, 0, 0.00134, 0.00514, 0.00584]
        let zs: [Float] = [-0.0033, -0.00265, 0.00265, 0.0033]
        var mesh = EthernetSolid(
            scene: s, origin: initial + rotation.act(F3(0.008 + EthernetInsertionTask.plugLength, 0, 0)),
            rotation: rotation,
            youngModulus: youngModulus)
        let nx = xs.count
        let ny = ys.count
        let nz = zs.count
        var nodes: [Int: Int] = [:]
        func key(_ x: Int, _ y: Int, _ z: Int) -> Int { (x * ny + y) * nz + z }
        func n(_ x: Int, _ y: Int, _ z: Int) -> Int {
            let k = key(x, y, z)
            if let id = nodes[k] { return id }
            let chamfer: Float = x == nx - 1 ? 0.000136 : 0
            let p = F3(xs[x], ys[y] * (1 - chamfer / 0.00584), zs[z] * (1 - chamfer / 0.0033))
            let id = mesh.node(p, pc)
            nodes[k] = id
            return id
        }
        for x in 0..<nx - 1 {
            for y in 0..<ny - 1 {
                for k in 0..<nz - 1 {
                    let hollow = x < 3 && y > 0 && y < ny - 2 && k == 1
                    if hollow { continue }
                    let vertices = [
                        n(x, y, k), n(x + 1, y, k), n(x + 1, y + 1, k), n(x, y + 1, k),
                        n(x, y, k + 1), n(x + 1, y, k + 1), n(x + 1, y + 1, k + 1), n(x, y + 1, k + 1),
                    ]
                    mesh.cell(vertices)
                }
            }
        }
        let latchYs = ys.indices.filter { abs(ys[$0]) <= 0.001341 }
        var latch: [[Int]] = []
        // A 0.47 mm sheet is a bending-dominated member. Use plane-stress
        // shell elements instead of one layer of locking linear tetrahedra.
        // Two rows share housing vertices and clamp the root orientation.
        let latchXs: [Float] = [-0.018, -0.014, -0.010, -0.006, -0.002, 0]
        for x in latchXs.indices {
            var row: [Int] = []
            for y in latchYs {
                if x >= latchXs.count - 2 {
                    row.append(n(x == latchXs.count - 1 ? nx - 1 : nx - 3, y, 0))
                } else {
                    // Contact thickness rounds the free boundary. Inset the
                    // midsurface so its outer envelope stays 3.15 mm wide.
                    let p = F3(
                        latchXs[x], ys[y],
                        -0.0033 + 0.068 * (latchXs[x] + 0.002))
                    row.append(mesh.node(p, pc, radius: 0.000235))
                }
            }
            latch.append(row)
        }
        for x in 0..<latch.count - 1 {
            for y in 0..<latchYs.count - 1 {
                let a = latch[x]
                let b = latch[x + 1]
                mesh.shell((a[y], b[y + 1], b[y]), thickness: 0.00047)
                mesh.shell((a[y], a[y + 1], b[y + 1]), thickness: 0.00047)
            }
        }
        let latchIDs = Array(Set(latch.flatMap { $0 })).sorted()
        // Rear housing lip bonded to the held boot. The rest of the housing
        // and the entire free latch remain deformable under jack contact.
        // Stable attachment order also makes the solver's graph identical
        // across processes with different Swift dictionary hash seeds.
        for k in nodes.keys.sorted() where k / (ny * nz) == 0 {
            let id = nodes[k]!
            let p = mesh.scene.bodies[id].position
            mesh.scene.addJoint(
                SceneJoint(
                    bodyA: tool, bodyB: id, rA: rotation.inverse.act(p - initial), rB: .zero,
                    stiffnessLin: 1e5))
        }
        // Thin gold plating is a material skin, embedded in the actual FEM
        // tetrahedra. It follows deformation without sliver volume elements.
        var platedVertices: [SceneSkinnedVertex] = []
        var platedTris: [(Int, Int, Int)] = []
        let contactXs: [Float] = [-0.005,-0.002,-0.00035,0]
        for i in 0..<8 {
            let c = Float(i) * 0.00102 - 0.00357
            for x in 0..<contactXs.count - 1 {
                let base = platedVertices.count
                for (px, py) in [
                    (contactXs[x], c - 0.00023), (contactXs[x + 1], c - 0.00023),
                    (contactXs[x + 1], c + 0.00023), (contactXs[x], c + 0.00023),
                ] {
                    let top: Float = px == 0 ? 0.003164 : 0.0033
                    let target = mesh.origin + rotation.act(F3(px, py, top + 0.000060))
                    var best: (Float, SceneSkinnedVertex)?
                    for tet in mesh.scene.tets {
                        let ids = [tet.ids.0, tet.ids.1, tet.ids.2, tet.ids.3]
                        let p = ids.map { mesh.scene.bodies[$0].position }
                        let inv = simd_float3x3(columns: (p[1] - p[0], p[2] - p[0], p[3] - p[0])).inverse
                        let v = inv * (target - p[0])
                        let w = SIMD4<Float>(1 - v.x - v.y - v.z, v.x, v.y, v.z)
                        let score = max(0, -w.min())
                        if best == nil || score < best!.0 {
                            best = (
                                score,
                                SceneSkinnedVertex(
                                    ids: tet.ids, weights: w,
                                    restNormal: rotation.act(F3(0, 0, 1)),
                                    restInv0: F3(inv.columns.0.x, inv.columns.1.x, inv.columns.2.x),
                                    restInv1: F3(inv.columns.0.y, inv.columns.1.y, inv.columns.2.y),
                                    restInv2: F3(inv.columns.0.z, inv.columns.1.z, inv.columns.2.z),
                                    color: gold)
                            )
                        }
                    }
                    platedVertices.append(best!.1)
                }
                platedTris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            }
        }
        mesh.scene.addSkinnedMesh(SceneSkinnedMesh(vertices: platedVertices, triangles: platedTris))
        EthernetVisuals.finish(&mesh.scene, bodies: [table, board, wrist, tool, remote], table: table)
        mesh.scene.rigidMotionGroups = [[tool] + mesh.nodes]
        let nose = nodes.filter { $0.key / (ny * nz) == nx - 1 }.map(\.value).sorted()
        return EthernetInsertionTask(
            scene: mesh.scene, socketBody: socket, remoteConnectorBody: remote, cable: cable, contactWires: Array(mesh.scene.cables.dropFirst()),
            wristBody: wrist, toolBody: tool, noseNodes: nose,
            plugNodes: mesh.nodes, latchNodes: latchIDs, couplingAnchors: anchors, lateralError: lateralError,
            yawError: yawError)
    }
}

private struct EthernetSolid {
    var scene: PhysicsScene
    var origin: F3
    var rotation: Quat
    var youngModulus: Float
    var nodes: [Int] = []
    mutating func node(_ p: F3, _ color: F3, radius: Float = 0.000025) -> Int {
        let id = scene.addParticle(
            radius: radius, mass: 0, friction: 0.25, position: origin + rotation.act(p))
        paintBody(&scene, id, color)
        nodes.append(id)
        return id
    }
    mutating func shell(_ ids: (Int, Int, Int), thickness: Float) {
        let poisson: Float = 0.37
        let p = [ids.0, ids.1, ids.2].map { scene.bodies[$0].position }
        let area = length(cross(p[1] - p[0], p[2] - p[0])) / 2
        for id in [ids.0, ids.1, ids.2] {
            let r = scene.bodies[id].size.x / 2
            scene.bodies[id].density += (1200 * area * thickness / 3) / (4 * .pi / 3 * r * r * r)
        }
        scene.addTri(
            SceneTri(
                ids: ids, mu: youngModulus * thickness / (2 * (1 + poisson)),
                lambda: youngModulus * thickness * poisson / (1 - poisson * poisson),
                bend: youngModulus * pow(thickness, 3) / (12 * (1 - poisson * poisson)),
                selfCollisionEnabled: false))
    }
    mutating func cell(_ v: [Int]) {
        let poisson: Float = 0.37
        let mu = youngModulus / (2 * (1 + poisson))
        let lambda = youngModulus * poisson / ((1 + poisson) * (1 - 2 * poisson))
        for t in [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)] {
            let ids = [v[t.0], v[t.1], v[t.2], v[t.3]]
            let p = ids.map { scene.bodies[$0].position }
            let mass = 1200 * abs(dot(p[1] - p[0], cross(p[2] - p[0], p[3] - p[0]))) / 24
            for id in ids {
                let r = scene.bodies[id].size.x / 2
                scene.bodies[id].density += mass / (4 * .pi / 3 * r * r * r)
            }
            scene.addTet(SceneTet(ids: (ids[0], ids[1], ids[2], ids[3]), mu: mu, lambda: lambda))
        }
    }
}

/// Smooth the authored loop, then sample by arc length so short bends do not
/// create tiny segments with a disproportionate stiffness/iteration cost.
private func ethernetServiceLoop(_ points: [F3], segments: Int) -> [F3] {
    var dense: [F3] = [points[0]]
    for i in 0..<points.count-1 {
        let p0 = i > 0 ? points[i-1] : 2*points[i]-points[i+1]
        let p1 = points[i], p2 = points[i+1]
        let p3 = i+2 < points.count ? points[i+2] : 2*p2-p1
        for j in 1...32 {
            let t = Float(j)/32
            let a: F3 = p1 * 2
            let b: F3 = p2 - p0
            let c: F3 = p0 * 2 - p1 * 5 + p2 * 4 - p3
            let d: F3 = -p0 + p1 * 3 - p2 * 3 + p3
            let t2 = t * t
            let t3 = t2 * t
            let curve = a + b * t + c * t2 + d * t3
            var p = curve * 0.5
            p.z = max(0.003, p.z)
            dense.append(p)
        }
    }
    var lengths: [Float] = [0]
    for i in 1..<dense.count { lengths.append(lengths.last! + distance(dense[i],dense[i-1])) }
    var cursor = 1
    return (0...segments).map { i in
        let target = lengths.last!*Float(i)/Float(segments)
        while cursor < lengths.count-1 && lengths[cursor] < target { cursor += 1 }
        let t = (target-lengths[cursor-1])/(lengths[cursor]-lengths[cursor-1])
        return mix(dense[cursor-1],dense[cursor],t:F3(repeating:t))
    }
}
