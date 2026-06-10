import simd
import Metal

public struct PushTEnvGPU {
    public var ids: SIMD4<UInt32> = .zero
    public var frame: SIMD4<Float> = .zero
    public var goal: SIMD4<Float> = .zero
}

// Robotics mode: massively parallel 3D Push-T environments.
//
// Each env: a 3-DOF arm (yaw + shoulder + elbow, servo motors with torque
// limits) with a vertical pusher finger, and a T-shaped block on the floor
// that must be pushed to a goal pose. All envs live in ONE PhysicsScene
// tiled on a grid — a single AVBD GPU solve steps every env simultaneously.
public final class PushTEnv {
    public struct EnvRefs {
        public var motorJoints: [Int]      // yaw, shoulder, elbow
        public var tip: Int                // pusher finger body
        public var blockBar: Int           // T cross-bar body
        public var blockStem: Int          // T stem body
        public var center: F3              // env origin
        public var goalPos: SIMD2<Float>   // goal T position (env-local)
        public var goalYaw: Float
    }

    public let numEnvs: Int
    public private(set) var refs: [EnvRefs] = []
    public let solver: GPUSolver
    public var scene: PhysicsScene
    public let envPitch: Float = 7.0
    /// arm geometry
    static let L1: Float = 1.05            // shoulder link length
    static let L2: Float = 1.05            // elbow link length
    static let tipHeight: Float = 0.42

    public init(numEnvs: Int, seed: UInt64 = 1) throws {
        self.numEnvs = numEnvs
        var s = PhysicsScene(name: "pusht")
        s.settings.iterations = 16
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 4000
        Demos.addGround(&s, friction: 0.9)

        var rng = SplitMix64(seed: seed)
        let side = Int(ceil(Double(numEnvs).squareRoot()))
        for e in 0..<numEnvs {
            let cx = Float(e % side) * envPitch
            let cy = Float(e / side) * envPitch
            let c = F3(cx, cy, 0)
            refs.append(Self.buildOne(&s, center: c, rng: &rng))
        }
        scene = s
        solver = try GPUSolver(scene: s)
    }

    static func buildOne(_ s: inout PhysicsScene, center c: F3,
                         rng: inout SplitMix64) -> EnvRefs {
        // ---- arm: base pillar + yaw hub + two links + finger ----
        let base = s.addBody(size: F3(0.22, 0.22, 1.3), density: 0, friction: 0.1,
                             position: c + F3(0, 0, 0.65))
        let hub = s.addBody(size: F3(0.3, 0.3, 0.22), density: 2, friction: 0.5,
                            position: c + F3(0, 0, 1.41))
        var joints: [Int] = []
        joints.append(s.joints.count)
        s.addJoint(SceneJoint(bodyA: base, bodyB: hub,
                              rA: F3(0, 0, 0.76), rB: .zero,
                              stiffnessLin: .infinity, stiffnessAng: .infinity,
                              hingeAxis: F3(0, 0, 1),
                              motorTarget: 0, motorTorque: 260))
        s.addJoint(SceneJoint(bodyA: base, bodyB: hub, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))

        // shoulder link along +x (box), hinge axis y at the hub
        let l1 = s.addBody(size: F3(L1, 0.16, 0.16), density: 1.2, friction: 0.4,
                           position: c + F3(L1 / 2, 0, 1.41))
        joints.append(s.joints.count)
        s.addJoint(SceneJoint(bodyA: hub, bodyB: l1,
                              rA: F3(0, 0, 0), rB: F3(-L1 / 2, 0, 0),
                              stiffnessLin: .infinity, stiffnessAng: .infinity,
                              hingeAxis: F3(0, 1, 0),
                              motorTarget: 0, motorTorque: 260))
        s.addJoint(SceneJoint(bodyA: hub, bodyB: l1, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))

        let l2 = s.addBody(size: F3(L2, 0.14, 0.14), density: 1.0, friction: 0.4,
                           position: c + F3(L1 + L2 / 2, 0, 1.41))
        joints.append(s.joints.count)
        s.addJoint(SceneJoint(bodyA: l1, bodyB: l2,
                              rA: F3(L1 / 2, 0, 0), rB: F3(-L2 / 2, 0, 0),
                              stiffnessLin: .infinity, stiffnessAng: .infinity,
                              hingeAxis: F3(0, 1, 0),
                              motorTarget: 0, motorTorque: 260))
        s.addJoint(SceneJoint(bodyA: l1, bodyB: l2, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))

        // pusher finger: vertical capsule welded to the elbow link's end
        let tip = s.addCapsule(length: 0.7, radius: 0.09, density: 1.0,
                               friction: 0.35,
                               position: c + F3(L1 + L2, 0, 1.41 - 0.5))
        s.addJoint(SceneJoint(bodyA: l2, bodyB: tip,
                              rA: F3(L2 / 2, 0, 0), rB: F3(0, 0, 0.5),
                              stiffnessLin: .infinity, stiffnessAng: .infinity))
        s.addJoint(SceneJoint(bodyA: l2, bodyB: tip, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))

        // ---- the T block (cross-bar + stem, welded) ----
        // spawn in the arm's WORKING ANNULUS: the tip operates at radius
        // [0.55, 1.95] from the base, so the block must live where the tip
        // can also get BEHIND it (annulus 0.95..1.45, any bearing)
        let ang = rng.nextFloat() * 2 * .pi
        let rad = 0.95 + rng.nextFloat() * 0.5
        let bx = cos(ang) * rad
        let by = sin(ang) * rad
        let byaw = (rng.nextFloat() - 0.5) * 2 * .pi
        let q = Quat(angle: byaw, axis: F3(0, 0, 1))
        let blockC = c + F3(bx, by, 0)
        let bar = s.addBody(size: F3(1.0, 0.25, 0.18), density: 3.0, friction: 0.8,
                            position: blockC + q.act(F3(0, 0.125, 0)) + F3(0, 0, 0.09),
                            rotation: q)
        let stem = s.addBody(size: F3(0.25, 0.65, 0.18), density: 3.0, friction: 0.8,
                             position: blockC + q.act(F3(0, -0.325, 0)) + F3(0, 0, 0.09),
                             rotation: q)
        let mid = (s.bodies[bar].position + s.bodies[stem].position) * 0.5
        s.addJoint(SceneJoint(bodyA: bar, bodyB: stem,
                              rA: s.bodies[bar].rotation.inverse.act(mid - s.bodies[bar].position),
                              rB: s.bodies[stem].rotation.inverse.act(mid - s.bodies[stem].position),
                              stiffnessLin: .infinity, stiffnessAng: .infinity))
        s.addJoint(SceneJoint(bodyA: bar, bodyB: stem, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))

        // containment fence: blocks must never cross into neighbor envs
        for sgn in [Float(-1), 1] {
            _ = s.addBody(size: F3(6.4, 0.12, 0.45), density: 0, friction: 0.4,
                          position: c + F3(0, sgn * 3.2, 0.22))
            _ = s.addBody(size: F3(0.12, 6.4, 0.45), density: 0, friction: 0.4,
                          position: c + F3(sgn * 3.2, 0, 0.22))
        }

        return EnvRefs(motorJoints: joints, tip: tip,
                       blockBar: bar, blockStem: stem, center: c,
                       goalPos: SIMD2(0.85, 0.85), goalYaw: 0)
    }

    // ---- control: action = desired tip position in env plane ----
    /// Analytic IK: yaw to face (x, y); shoulder+elbow as a 2-link chain in
    /// the vertical plane reaching radius r at the finger's working height.
    /// Numeric planar IK for the finger BOTTOM, including the welded
    /// finger's tilt with the arm (the finger pivots with the elbow link).
    /// Solved by damped Newton in the (r, z) plane — 6 iterations suffice.
    static func planarIK(r target: Float) -> (Float, Float) {
        let zT: Float = 0.10 - 1.41          // finger bottom near the floor
        var t1: Float = 0.5, t2: Float = 0.7
        for _ in 0..<8 {
            let p1 = t1, p12 = t1 + t2
            let ex = L1 * cos(p1) + L2 * cos(p12) - 0.85 * sin(p12)
            let ez = -L1 * sin(p1) - L2 * sin(p12) - 0.85 * cos(p12)
            let fx = ex - target, fz = ez - zT
            // Jacobian
            let dex1 = -L1 * sin(p1) - L2 * sin(p12) - 0.85 * cos(p12)
            let dez1 = -L1 * cos(p1) - L2 * cos(p12) + 0.85 * sin(p12)
            let dex2 = -L2 * sin(p12) - 0.85 * cos(p12)
            let dez2 = -L2 * cos(p12) + 0.85 * sin(p12)
            let det = dex1 * dez2 - dex2 * dez1
            if abs(det) < 1e-6 { break }
            t1 -= (fx * dez2 - fz * dex2) / det * 0.8
            t2 -= (fz * dex1 - fx * dez1) / det * 0.8
            t1 = simd_clamp(t1, -0.4, 1.4)
            t2 = simd_clamp(t2, 0.0, 2.2)
        }
        return (t1, t2)
    }

    private var motorCmd: [[Float]] = []

    public func setTipTarget(_ env: Int, _ p: SIMD2<Float>) {
        if motorCmd.isEmpty {
            motorCmd = [[Float]](repeating: [0, 0.5, 0.7], count: numEnvs)
        }
        var r = simd_clamp(length(p), 0.75, 1.95)
        let yaw = atan2(p.y, p.x)
        // turn-then-reach: while the yaw error is large, keep the arm
        // extended so the hanging finger sweeps OUTSIDE the base pillar
        // instead of folding through the center and jamming against it
        var yawErr = yaw - motorCmd[env][0]
        yawErr = yawErr - 2 * .pi * (yawErr / (2 * .pi)).rounded()
        if abs(yawErr) > 0.45 { r = max(r, 1.25) }
        let (t1, t2) = Self.planarIK(r: r)
        // motor-space rate limit: the arm may never whip
        let maxRate: Float = 0.05
        let targets = [yaw, t1, t2]
        let js = refs[env].motorJoints
        for k in 0..<3 {
            var d = targets[k] - motorCmd[env][k]
            if k == 0 { d = d - 2 * .pi * (d / (2 * .pi)).rounded() }
            motorCmd[env][k] += simd_clamp(d, -maxRate, maxRate)
            solver.setMotorTarget(js[k], angle: motorCmd[env][k])
        }
    }

    private var commanded: [SIMD2<Float>] = []

    /// Step all envs: actions are tip-target positions (env-local XY).
    /// Targets are rate-limited (robot speed limit) so the arm pushes
    /// instead of batting.
    public func step(actions: [SIMD2<Float>], substeps: Int = 4,
                     maxStep: Float = 0.16) {
        if commanded.isEmpty {
            commanded = (0..<numEnvs).map { tipPos($0) }
        }
        for e in 0..<numEnvs {
            let d = actions[e] - commanded[e]
            let l = length(d)
            commanded[e] += l > maxStep ? d * (maxStep / l) : d
            setTipTarget(e, commanded[e])
        }
        for _ in 0..<substeps { solver.step() }
    }

    public func tipPos(_ env: Int) -> SIMD2<Float> {
        let p = solver.bodyPosition(refs[env].tip) - refs[env].center
        return SIMD2(p.x, p.y)
    }

    public func blockPose(_ env: Int) -> (SIMD2<Float>, Float) {
        let r = refs[env]
        let pb = solver.bodyPosition(r.blockBar)
        let ps = solver.bodyPosition(r.blockStem)
        let mid = (pb + ps) * 0.5 - r.center
        let q = solver.bodyRotation(r.blockBar)
        let fwd = q.act(F3(1, 0, 0))
        return (SIMD2(mid.x, mid.y), atan2(fwd.y, fwd.x))
    }

    // ---- pixel observations ----
    public let obsRes = 64
    private var envTable: MTLBuffer?
    private var obsBuffer: MTLBuffer?

    /// Render all envs' observations (numEnvs x res x res x 3, uint8).
    public func observations() -> UnsafeBufferPointer<UInt8> {
        if envTable == nil {
            let dev = solver.metalDevice
            envTable = dev.makeBuffer(length: numEnvs * MemoryLayout<PushTEnvGPU>.stride,
                                      options: .storageModeShared)
            obsBuffer = dev.makeBuffer(length: numEnvs * obsRes * obsRes * 3,
                                       options: .storageModeShared)
            let t = envTable!.contents().bindMemory(to: PushTEnvGPU.self, capacity: numEnvs)
            for e in 0..<numEnvs {
                let r = refs[e]
                var g = PushTEnvGPU()
                g.ids = SIMD4(UInt32(r.tip), UInt32(r.blockBar), UInt32(r.blockStem), 0)
                g.frame = SIMD4(r.center.x, r.center.y, r.goalPos.x, r.goalPos.y)
                g.goal = SIMD4(r.goalYaw, 2.6, 0, 0)
                t[e] = g
            }
        }
        solver.sync()
        solver.renderPushTObs(envTable: envTable!, numEnvs: numEnvs,
                              out: obsBuffer!, res: obsRes)
        let ptr = obsBuffer!.contents().bindMemory(to: UInt8.self,
                                                   capacity: numEnvs * obsRes * obsRes * 3)
        return UnsafeBufferPointer(start: ptr, count: numEnvs * obsRes * obsRes * 3)
    }

    public func success(_ env: Int, posTol: Float = 0.22, yawTol: Float = 0.4) -> Bool {
        let (p, yaw) = blockPose(env)
        let r = refs[env]
        var dy = yaw - r.goalYaw
        dy = dy - 2 * .pi * (dy / (2 * .pi)).rounded()
        return length(p - r.goalPos) < posTol && abs(dy) < yawTol
    }
}
