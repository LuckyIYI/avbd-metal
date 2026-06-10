import simd
import Metal

// MARK: - Production env API (Isaac/MuJoCo-shaped)

/// Action/observation specification for vectorized RL/world-model training.
public struct EnvSpec {
    public let numEnvs: Int
    public let actionDim: Int
    public let actionLow: [Float]
    public let actionHigh: [Float]
    public let obsShape: [Int]          // H, W, C (uint8 pixels)
}

/// Vectorized robotics environment: many replicas, one GPU solve.
public protocol RoboticsEnv: AnyObject {
    var spec: EnvSpec { get }
    /// In-place reset of one env (teleport, no scene rebuild).
    func reset(_ env: Int, seed: UInt64)
    func resetAll(seed: UInt64)
    func step(actions: [SIMD2<Float>], substeps: Int, maxStep: Float)
    func observations() -> UnsafeBufferPointer<UInt8>
    func success(_ env: Int, posTol: Float, yawTol: Float) -> Bool
}

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
public final class PushTEnv: RoboticsEnv {
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

    /// spawn poses of every body, for fast in-place resets
    private var spawnPoses: [(F3, Quat)] = []

    public init(numEnvs: Int, seed: UInt64 = 1, goalMarkers: Bool = false) throws {
        self.numEnvs = numEnvs
        var s = PhysicsScene(name: "pusht")
        s.settings.iterations = 16
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 600
        Demos.addGround(&s, friction: 0.9)

        var rng = SplitMix64(seed: seed)
        let side = Int(ceil(Double(numEnvs).squareRoot()))
        for e in 0..<numEnvs {
            let cx = Float(e % side) * envPitch
            let cy = Float(e / side) * envPitch
            let c = F3(cx, cy, 0)
            refs.append(Self.buildOne(&s, center: c, rng: &rng))
            if goalMarkers {
                // visual goal T: static plates sunk into the floor (top
                // sliver visible, no meaningful bump)
                let r = refs[e]
                let gq = Quat(angle: r.goalYaw, axis: F3(0, 0, 1))
                let gc = c + F3(r.goalPos.x, r.goalPos.y, 0)
                _ = s.addBody(size: F3(1.0, 0.25, 0.1), density: 0, friction: 0.9,
                              position: gc + gq.act(F3(0, 0.125, 0)) + F3(0, 0, -0.038),
                              rotation: gq)
                _ = s.addBody(size: F3(0.25, 0.65, 0.1), density: 0, friction: 0.9,
                              position: gc + gq.act(F3(0, -0.325, 0)) + F3(0, 0, -0.038),
                              rotation: gq)
            }
        }
        scene = s
        solver = try GPUSolver(scene: s)
        spawnPoses = s.bodies.map { ($0.position, $0.rotation) }
    }

    public var spec: EnvSpec {
        EnvSpec(numEnvs: numEnvs, actionDim: 2,
                actionLow: [-1.95, -1.95], actionHigh: [1.95, 1.95],
                obsShape: [obsRes, obsRes, 3])
    }

    /// In-place reset: restore the arm to its spawn pose, home the servos,
    /// and re-randomize the block within the working annulus. No rebuild —
    /// safe to call per-episode at scale.
    public func reset(_ e: Int, seed: UInt64) {
        var rng = SplitMix64(seed: seed)
        let r = refs[e]
        // arm chain back to spawn
        for b in [r.tip, r.blockBar, r.blockStem] + armBodies(e) {
            let (p, q) = spawnPoses[b]
            solver.setBodyPose(b, position: p, rotation: q)
        }
        // re-randomized block pose
        let ang = rng.nextFloat() * 2 * .pi
        let rad = 0.95 + rng.nextFloat() * 0.5
        let byaw = (rng.nextFloat() - 0.5) * 2 * .pi
        let q = Quat(angle: byaw, axis: F3(0, 0, 1))
        let bc = r.center + F3(cos(ang) * rad, sin(ang) * rad, 0)
        solver.setBodyPose(r.blockBar,
                           position: bc + q.act(F3(0, 0.125, 0)) + F3(0, 0, 0.09),
                           rotation: q)
        solver.setBodyPose(r.blockStem,
                           position: bc + q.act(F3(0, -0.325, 0)) + F3(0, 0, 0.09),
                           rotation: q)
        // servos to the exact teleported (spawn) pose: any mismatch between
        // the command cache and the true configuration causes post-reset jams
        let js = r.motorJoints
        let spawn: [Float] = [0, 0, 0]      // arm spawns straight
        for (k, j) in js.enumerated() { solver.setMotorTarget(j, angle: spawn[k]) }
        if !motorCmd.isEmpty { motorCmd[e] = spawn }
        if !radialCorr.isEmpty { radialCorr[e] = 0 }
        // settle contacts before the caller resumes
        for _ in 0..<3 { solver.step() }
        if !commanded.isEmpty { commanded[e] = tipPos(e) }
    }

    public func resetAll(seed: UInt64) {
        for e in 0..<numEnvs { reset(e, seed: seed &+ UInt64(e) &* 7919) }
    }

    /// bodies of env e's arm chain (hub + links), derived from the tip index
    private func armBodies(_ e: Int) -> [Int] {
        // build order per env: base(static), hub, l1, l2, tip
        [refs[e].tip - 3, refs[e].tip - 2, refs[e].tip - 1]
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
                              motorTarget: 0, motorTorque: 260,
                              limitLo: -0.55, limitHi: 1.5))
        s.addJoint(SceneJoint(bodyA: hub, bodyB: l1, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))

        let l2 = s.addBody(size: F3(L2, 0.14, 0.14), density: 1.0, friction: 0.4,
                           position: c + F3(L1 + L2 / 2, 0, 1.41))
        joints.append(s.joints.count)
        s.addJoint(SceneJoint(bodyA: l1, bodyB: l2,
                              rA: F3(L1 / 2, 0, 0), rB: F3(-L2 / 2, 0, 0),
                              stiffnessLin: .infinity, stiffnessAng: .infinity,
                              hingeAxis: F3(0, 1, 0),
                              motorTarget: 0, motorTorque: 260,
                              limitLo: -0.1, limitHi: 2.3))
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
    /// closed-loop radial correction (integral of measured radius error):
    /// open-loop IK carries a chronic ~0.2 droop from gravity sag and the
    /// welded finger's tilt — real arms close this loop, so do we
    private var radialCorr: [Float] = []
    /// per-substep joint speed limit (rad); Lab-tunable
    public var jointSpeedLimit: Float = 0.05

    public func setTipTarget(_ env: Int, _ p: SIMD2<Float>) {
        if motorCmd.isEmpty {
            motorCmd = [[Float]](repeating: [0, 0.5, 0.7], count: numEnvs)
        }
        if radialCorr.isEmpty { radialCorr = [Float](repeating: 0, count: numEnvs) }
        let rWant = simd_clamp(length(p), 0.5, 1.95)
        // integrate the measured radial error (slow, stable gain) with
        // anti-windup: freeze while loaded (near the block), or contact
        // resistance winds the integrator and the push lurches on release
        let rMeas = length(tipPos(env))
        let (bp, _) = blockPose(env)
        if length(tipPos(env) - bp) > 0.55 {
            radialCorr[env] = simd_clamp(radialCorr[env] + 0.04 * (rWant - rMeas),
                                         -0.45, 0.45)
        }
        var r = simd_clamp(rWant + radialCorr[env], 0.45, 1.95)
        let yaw = atan2(p.y, p.x)
        // turn-then-reach: while the yaw error is large, keep the arm
        // extended so the hanging finger sweeps OUTSIDE the base pillar
        // instead of folding through the center and jamming against it
        var yawErr = yaw - motorCmd[env][0]
        yawErr = yawErr - 2 * .pi * (yawErr / (2 * .pi)).rounded()
        if abs(yawErr) > 0.45 { r = max(r, 1.25) }
        let (t1, t2) = Self.planarIK(r: r)
        // motor-space rate limit: the arm may never whip
        let maxRate: Float = jointSpeedLimit
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

    /// Greedy geometric push controller (shared by CLI eval and the
    /// Robotics Lab): returns the next tip target for env `e`.
    /// clamp a waypoint into the arm's reachable annulus
    static func annulus(_ p: SIMD2<Float>) -> SIMD2<Float> {
        let l = length(p)
        if l < 1e-5 { return SIMD2(0.55, 0) }
        return p * (simd_clamp(l, 0.55, 1.9) / l)
    }

    public func oracleAction(_ e: Int) -> SIMD2<Float> {
        let r = refs[e]
        let (bp, _) = blockPose(e)
        let tp = tipPos(e)
        let toGoal = r.goalPos - bp
        if length(toGoal) < 0.05 { return tp }
        let dir = normalize(toGoal)
        let behind = bp - dir * 0.42
        let target: SIMD2<Float>
        if length(tp - behind) > 0.20 {
            let toB = behind - tp
            let t = simd_clamp(dot(bp - tp, toB) / max(dot(toB, toB), 1e-6), 0, 1)
            let closest = tp + toB * t
            if length(closest - bp) < 0.33 {
                let side = SIMD2(-dir.y, dir.x)
                let sgn: Float = dot(tp - bp, side) >= 0 ? 1 : -1
                target = bp + side * (sgn * 0.8)
            } else {
                target = behind
            }
        } else {
            let push = min(0.10, length(toGoal) * 0.4)
            target = bp + dir * push
        }
        return Self.annulus(target)
    }

    public func success(_ env: Int, posTol: Float = 0.25, yawTol: Float = .pi) -> Bool {
        let (p, yaw) = blockPose(env)
        let r = refs[env]
        var dy = yaw - r.goalYaw
        dy = dy - 2 * .pi * (dy / (2 * .pi)).rounded()
        return length(p - r.goalPos) < posTol && abs(dy) < yawTol
    }
}
