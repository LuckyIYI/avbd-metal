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

/// Compact state observation used by state-based policies. Pixel policies
/// can continue to use `observations()`; both views come from the same
/// batched simulation state.
public struct PushTState {
    public var tipPosition: SIMD2<Float>
    public var tipVelocity: SIMD2<Float>
    public var blockPosition: SIMD2<Float>
    public var blockYaw: Float
}

/// Complete task-local rigid state needed to branch Push-T futures. Positions
/// are expressed relative to the environment origin so one physical state can
/// be copied to any translated replica without changing its dynamics.
public struct PushTPhysicalState {
    public var tip: GPUSolver.RigidBodyState
    public var blockBar: GPUSolver.RigidBodyState
    public var blockStem: GPUSolver.RigidBodyState
    public var commandedTipTarget: SIMD2<Float>

    public init(tip: GPUSolver.RigidBodyState,
                blockBar: GPUSolver.RigidBodyState,
                blockStem: GPUSolver.RigidBodyState,
                commandedTipTarget: SIMD2<Float>) {
        self.tip = tip
        self.blockBar = blockBar
        self.blockStem = blockStem
        self.commandedTipTarget = commandedTipTarget
    }
}

public struct PushTPhysicalStateUpdate {
    public var environment: Int
    public var state: PushTPhysicalState

    public init(environment: Int, state: PushTPhysicalState) {
        self.environment = environment
        self.state = state
    }
}

/// Grid replicas are independently visible and suit ordinary RL. Speculative
/// replicas use collision domains to overlap at the origin, eliminating
/// coordinate-magnitude differences from counterfactual comparisons.
public enum PushTBatchLayout: Sendable {
    case grid
    case coLocated
}

// Robotics mode: massively parallel 3D Push-T environments.
//
// The robot is a GANTRY PUSHER (industry-standard for tabletop pushing
// benchmarks): a vertical tool head driven by a force-limited Cartesian
// position actuator — a world joint whose anchor we retarget. No IK, no
// kinematic chains, no joint-space pathologies; the force bound keeps the
// physics honest (the pusher cannot smash through anything).
public final class PushTEnv: RoboticsEnv {
    public struct EnvRefs {
        public var dragJoint: Int          // Cartesian actuator joint
        public var tip: Int                // pusher tool head body
        public var blockBar: Int
        public var blockStem: Int
        public var center: F3
        public var goalPos: SIMD2<Float>
        public var goalYaw: Float
    }

    public let numEnvs: Int
    public private(set) var refs: [EnvRefs] = []
    public let groundBody: Int
    public let solver: GPUSolver
    public var scene: PhysicsScene
    public let envPitch: Float = 7.0
    public static let tipHeight: Float = 0.42
    public static let pusherForce: Float = 60.0    // Cartesian force limit
    /// per-step target speed limit (m per control step); Lab-tunable
    public var tipSpeedLimit: Float = 0.16

    private var spawnPoses: [(F3, Quat)] = []
    private var commanded: [SIMD2<Float>] = []

    public init(numEnvs: Int, seed: UInt64 = 1,
                goalMarkers: Bool = false,
                layout: PushTBatchLayout = .grid) throws {
        self.numEnvs = numEnvs
        var s = PhysicsScene(name: "pusht")
        s.settings.iterations = 16
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 600
        let columns = Int(ceil(Double(numEnvs).squareRoot()))
        let rows = Int(ceil(Double(numEnvs) / Double(columns)))
        let halfGridX = layout == .grid
            ? 0.5 * Float(columns - 1) * envPitch : 0
        let halfGridY = layout == .grid
            ? 0.5 * Float(rows - 1) * envPitch : 0
        // A fixed 200 m plane silently left the last row/column of batches
        // larger than 225 environments beyond its edge. Center the replicas
        // to improve Float32 conditioning and derive terrain from batch size.
        groundBody = s.addBody(
            size: F3(2 * (halfGridX + 3.5),
                     2 * (halfGridY + 3.5), 2),
            density: 0, friction: 1.3, position: F3(0, 0, -1))

        var rng = SplitMix64(seed: seed)
        for e in 0..<numEnvs {
            let cx = layout == .grid
                ? Float(e % columns) * envPitch - halfGridX : 0
            let cy = layout == .grid
                ? Float(e / columns) * envPitch - halfGridY : 0
            let c = F3(cx, cy, 0)
            let collisionGroup = UInt32(e + 1)
            refs.append(Self.buildOne(
                &s, center: c, collisionGroup: collisionGroup, rng: &rng))
            if goalMarkers {
                let r = refs[e]
                let gq = Quat(angle: r.goalYaw, axis: F3(0, 0, 1))
                let gc = c + F3(r.goalPos.x, r.goalPos.y, 0)
                // VISUAL-ONLY target: the marker plates must never collide
                // with the block (or the block can't slide onto its own
                // goal!) — excluded pairwise below
                let m1 = s.addBody(size: F3(1.0, 0.25, 0.1), density: 0, friction: 0.9,
                                   position: gc + gq.act(F3(0, 0.125, 0)) + F3(0, 0, -0.038),
                                   rotation: gq,
                                   collisionGroup: collisionGroup)
                let m2 = s.addBody(size: F3(0.25, 0.65, 0.1), density: 0, friction: 0.9,
                                   position: gc + gq.act(F3(0, -0.325, 0)) + F3(0, 0, -0.038),
                                   rotation: gq,
                                   collisionGroup: collisionGroup)
                for m in [m1, m2] {
                    for b in [r.blockBar, r.blockStem, r.tip] {
                        s.addJoint(SceneJoint(bodyA: m, bodyB: b, rA: .zero, rB: .zero,
                                              stiffnessLin: 0, stiffnessAng: 0))
                    }
                }
                // cosmetic gantry frame
                for sx in [Float(-1), 1] {
                    _ = s.addBody(size: F3(0.14, 0.14, 2.6), density: 0, friction: 0.3,
                                  position: c + F3(sx * 2.9, -2.9, 1.3),
                                  collisionGroup: collisionGroup)
                }
                _ = s.addBody(size: F3(6.0, 0.12, 0.12), density: 0, friction: 0.3,
                              position: c + F3(0, -2.9, 2.6),
                              collisionGroup: collisionGroup)
            }
        }
        scene = s
        solver = try GPUSolver(scene: s)
        spawnPoses = s.bodies.map { ($0.position, $0.rotation) }
        commanded = refs.map { reference in
            let p = spawnPoses[reference.tip].0 - reference.center
            return SIMD2(p.x, p.y)
        }
    }

    static func buildOne(_ s: inout PhysicsScene, center c: F3,
                         collisionGroup: UInt32,
                         rng: inout SplitMix64) -> EnvRefs {
        // ---- the pusher tool head ----
        let tip = s.addBody(
            size: F3(0.7, 0.09, 0), density: 1.2, friction: 0.35,
            position: c + F3(1.4, 0, tipHeight), shape: .capsule,
            collisionGroup: collisionGroup)
        // Cartesian actuator: world->tip soft joint with bounded force.
        // Stiffness chosen so force saturates over ~1 tip radius of error.
        let dragJoint = s.joints.count
        s.addJoint(SceneJoint(bodyA: -1, bodyB: tip,
                              rA: c + F3(1.4, 0, tipHeight), rB: .zero,
                              stiffnessLin: pusherForce / 0.08, stiffnessAng: 0))
        // orientation lock: the tool head stays vertical (angular-only weld)
        s.addJoint(SceneJoint(bodyA: -1, bodyB: tip,
                              rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: .infinity))

        // ---- the T block ----
        let ang = rng.nextFloat() * 2 * .pi
        let rad = 0.55 + rng.nextFloat() * 1.0
        let bx = cos(ang) * rad
        let by = sin(ang) * rad
        let byaw = (rng.nextFloat() - 0.5) * 2 * .pi
        let q = Quat(angle: byaw, axis: F3(0, 0, 1))
        let blockC = c + F3(bx, by, 0)
        let bar = s.addBody(size: F3(1.0, 0.25, 0.18), density: 3.0, friction: 1.1,
                            position: blockC + q.act(F3(0, 0.125, 0)) + F3(0, 0, 0.09),
                            rotation: q,
                            collisionGroup: collisionGroup)
        let stem = s.addBody(size: F3(0.25, 0.65, 0.18), density: 3.0, friction: 1.1,
                             position: blockC + q.act(F3(0, -0.325, 0)) + F3(0, 0, 0.09),
                             rotation: q,
                             collisionGroup: collisionGroup)
        let mid = (s.bodies[bar].position + s.bodies[stem].position) * 0.5
        s.addJoint(SceneJoint(bodyA: bar, bodyB: stem,
                              rA: s.bodies[bar].rotation.inverse.act(mid - s.bodies[bar].position),
                              rB: s.bodies[stem].rotation.inverse.act(mid - s.bodies[stem].position),
                              stiffnessLin: .infinity, stiffnessAng: .infinity))
        s.addJoint(SceneJoint(bodyA: bar, bodyB: stem, rA: .zero, rB: .zero,
                              stiffnessLin: 0, stiffnessAng: 0))

        // containment fence
        for sgn in [Float(-1), 1] {
            _ = s.addBody(size: F3(6.4, 0.12, 0.45), density: 0, friction: 0.4,
                          position: c + F3(0, sgn * 3.2, 0.22),
                          collisionGroup: collisionGroup)
            _ = s.addBody(size: F3(0.12, 6.4, 0.45), density: 0, friction: 0.4,
                          position: c + F3(sgn * 3.2, 0, 0.22),
                          collisionGroup: collisionGroup)
        }

        return EnvRefs(dragJoint: dragJoint, tip: tip,
                       blockBar: bar, blockStem: stem, center: c,
                       goalPos: SIMD2(0.85, 0.85), goalYaw: 0)
    }

    public var spec: EnvSpec {
        EnvSpec(numEnvs: numEnvs, actionDim: 2,
                actionLow: [-3.0, -3.0], actionHigh: [3.0, 3.0],
                obsShape: [obsRes, obsRes, 3])
    }

    // ---- control ----
    public func setTipTarget(_ env: Int, _ p: SIMD2<Float>) {
        let pc = simd_clamp(p, SIMD2(-3.02, -3.02), SIMD2(3.02, 3.02))
        let world = refs[env].center + F3(pc.x, pc.y, Self.tipHeight)
        solver.setJointWorldAnchor(refs[env].dragJoint, point: world)
    }

    /// Step all envs; actions are tip-target positions (env-local XY),
    /// rate-limited to a realistic tool-head speed.
    public func step(actions: [SIMD2<Float>], substeps: Int = 4,
                     maxStep: Float = 0.16) {
        do {
            try stepChecked(
                actions: actions, substeps: substeps, maxStep: maxStep)
        } catch {
            fatalError("Push-T simulation failed: \(error.localizedDescription)")
        }
    }

    public func stepChecked(actions: [SIMD2<Float>], substeps: Int = 4,
                            maxStep: Float = 0.16) throws {
        precondition(actions.count == numEnvs, "expected one action per environment")
        try solver.synchronize()
        if commanded.isEmpty {
            commanded = (0..<numEnvs).map { tipPos($0) }
        }
        let limit = min(maxStep, tipSpeedLimit)
        var anchors = [GPUSolver.JointAnchorUpdate]()
        anchors.reserveCapacity(numEnvs)
        for e in 0..<numEnvs {
            let d = actions[e] - commanded[e]
            let l = length(d)
            commanded[e] += l > limit ? d * (limit / l) : d
            let p = commanded[e]
            let world = refs[e].center + F3(p.x, p.y, Self.tipHeight)
            anchors.append(.init(joint: refs[e].dragJoint, point: world))
        }
        solver.setJointWorldAnchors(anchors)
        for _ in 0..<substeps { try solver.submitStep() }
        try solver.synchronize()
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

    /// Read every environment's policy state after one GPU fence.
    public func states() -> [PushTState] {
        var ids = [Int]()
        ids.reserveCapacity(numEnvs * 3)
        for r in refs {
            ids.append(r.tip)
            ids.append(r.blockBar)
            ids.append(r.blockStem)
        }
        let bodies = solver.bodyStates(ids)
        return (0..<numEnvs).map { e in
            let r = refs[e]
            let tip = bodies[e * 3]
            let bar = bodies[e * 3 + 1]
            let stem = bodies[e * 3 + 2]
            let mid = (bar.position + stem.position) * 0.5 - r.center
            let fwd = bar.rotation.act(F3(1, 0, 0))
            return PushTState(
                tipPosition: SIMD2(tip.position.x - r.center.x,
                                   tip.position.y - r.center.y),
                tipVelocity: SIMD2(tip.linearVelocity.x, tip.linearVelocity.y),
                blockPosition: SIMD2(mid.x, mid.y),
                blockYaw: atan2(fwd.y, fwd.x))
        }
    }

    /// Capture every dynamic Push-T body and its actuator target after one GPU
    /// fence. This is deliberately more complete than `states()`: speculative
    /// rollouts must preserve velocities and the rate limiter's target history.
    public func physicalStates() -> [PushTPhysicalState] {
        var ids = [Int]()
        ids.reserveCapacity(numEnvs * 3)
        for reference in refs {
            ids.append(reference.tip)
            ids.append(reference.blockBar)
            ids.append(reference.blockStem)
        }
        let bodies = solver.bodyStates(ids)
        return (0..<numEnvs).map { environment in
            let reference = refs[environment]
            func localized(_ state: GPUSolver.RigidBodyState)
                -> GPUSolver.RigidBodyState {
                GPUSolver.RigidBodyState(
                    position: state.position - reference.center,
                    rotation: state.rotation,
                    linearVelocity: state.linearVelocity,
                    angularVelocity: state.angularVelocity)
            }
            return PushTPhysicalState(
                tip: localized(bodies[environment * 3]),
                blockBar: localized(bodies[environment * 3 + 1]),
                blockStem: localized(bodies[environment * 3 + 2]),
                commandedTipTarget: commanded[environment])
        }
    }

    /// Restore portable task-local physical states into arbitrary replicas.
    /// All bodies are written behind one fence and all actuator anchors behind
    /// one more, making this suitable for batched speculative branching.
    public func setPhysicalStates(_ updates: [PushTPhysicalStateUpdate]) {
        guard !updates.isEmpty else { return }
        var bodyUpdates = [GPUSolver.BodyStateUpdate]()
        var anchorUpdates = [GPUSolver.JointAnchorUpdate]()
        bodyUpdates.reserveCapacity(updates.count * 3)
        anchorUpdates.reserveCapacity(updates.count)
        for update in updates {
            let environment = update.environment
            precondition(environment >= 0 && environment < numEnvs,
                         "environment index out of range")
            let reference = refs[environment]
            let state = update.state
            precondition(Self.isFinite(state),
                         "speculative state must be finite")
            func worldUpdate(_ body: Int,
                             _ local: GPUSolver.RigidBodyState)
                -> GPUSolver.BodyStateUpdate {
                GPUSolver.BodyStateUpdate(
                    body: body,
                    position: local.position + reference.center,
                    rotation: local.rotation,
                    linearVelocity: local.linearVelocity,
                    angularVelocity: local.angularVelocity)
            }
            bodyUpdates.append(worldUpdate(reference.tip, state.tip))
            bodyUpdates.append(worldUpdate(reference.blockBar, state.blockBar))
            bodyUpdates.append(worldUpdate(reference.blockStem, state.blockStem))
            let target = state.commandedTipTarget
            anchorUpdates.append(.init(
                joint: reference.dragJoint,
                point: reference.center
                    + F3(target.x, target.y, Self.tipHeight)))
            commanded[environment] = target
        }
        solver.setBodyStates(bodyUpdates)
        solver.setJointWorldAnchors(anchorUpdates)
    }

    /// Copy one captured future into many replicas for branched rollouts.
    public func fork(_ state: PushTPhysicalState,
                     into environments: [Int]) {
        setPhysicalStates(environments.map {
            PushTPhysicalStateUpdate(environment: $0, state: state)
        })
    }

    private static func isFinite(_ state: PushTPhysicalState) -> Bool {
        func vector(_ value: F3) -> Bool {
            value.x.isFinite && value.y.isFinite && value.z.isFinite
        }
        func quaternion(_ value: Quat) -> Bool {
            value.real.isFinite && vector(value.imag)
        }
        func rigid(_ value: GPUSolver.RigidBodyState) -> Bool {
            vector(value.position) && quaternion(value.rotation)
                && vector(value.linearVelocity)
                && vector(value.angularVelocity)
        }
        return rigid(state.tip) && rigid(state.blockBar)
            && rigid(state.blockStem)
            && state.commandedTipTarget.x.isFinite
            && state.commandedTipTarget.y.isFinite
    }

    /// In-place per-env reset: teleport pusher + re-randomize block.
    public func reset(_ e: Int, seed: UInt64) {
        reset([e], seeds: [seed])
    }

    /// In-place reset for an arbitrary subset, with exactly one solver
    /// synchronization for all teleports and one for all actuator anchors.
    public func reset(_ envs: [Int], seeds: [UInt64]) {
        precondition(envs.count == seeds.count, "one reset seed is required per environment")
        var poses = [GPUSolver.BodyPoseUpdate]()
        var anchors = [GPUSolver.JointAnchorUpdate]()
        poses.reserveCapacity(envs.count * 3)
        anchors.reserveCapacity(envs.count)
        for (i, e) in envs.enumerated() {
            precondition(e >= 0 && e < numEnvs, "environment index out of range")
            var rng = SplitMix64(seed: seeds[i])
            let r = refs[e]
            let (tp, tq) = spawnPoses[r.tip]
            poses.append(.init(body: r.tip, position: tp, rotation: tq))
            anchors.append(.init(joint: r.dragJoint, point: tp))
            let ang = rng.nextFloat() * 2 * .pi
            let rad = 0.55 + rng.nextFloat()
            let byaw = (rng.nextFloat() - 0.5) * 2 * .pi
            let q = Quat(angle: byaw, axis: F3(0, 0, 1))
            let bc = r.center + F3(cos(ang) * rad, sin(ang) * rad, 0)
            poses.append(.init(body: r.blockBar,
                               position: bc + q.act(F3(0, 0.125, 0)) + F3(0, 0, 0.09),
                               rotation: q))
            poses.append(.init(body: r.blockStem,
                               position: bc + q.act(F3(0, -0.325, 0)) + F3(0, 0, 0.09),
                               rotation: q))
            if !commanded.isEmpty {
                commanded[e] = SIMD2(tp.x - r.center.x, tp.y - r.center.y)
            }
        }
        solver.setBodyPoses(poses)
        solver.setJointWorldAnchors(anchors)
    }

    public func resetAll(seed: UInt64) {
        let ids = Array(0..<numEnvs)
        let seeds = ids.map { seed &+ UInt64($0) &* 7919 }
        reset(ids, seeds: seeds)
    }

    /// Geometric two-phase push planner (3D-aware, model-based, non-neural).
    public func oracleAction(_ e: Int) -> SIMD2<Float> {
        let r = refs[e]
        let (bp, _) = blockPose(e)
        let tp = tipPos(e)
        let toGoal = r.goalPos - bp
        if length(toGoal) < 0.05 { return tp }
        var dir = normalize(toGoal)
        var behind = bp - dir * 0.42
        // fence recovery: when the behind-point is inside a wall the direct
        // plan is impossible — slide the block ALONG the fence (tangent
        // component of the goal direction) until it's clear
        if max(abs(behind.x), abs(behind.y)) > 2.95 {
            let tangent: SIMD2<Float> = abs(behind.x) > abs(behind.y)
                ? SIMD2(0, 1) : SIMD2(1, 0)
            var t = tangent * dot(dir, tangent)
            if length(t) < 0.2 {
                // goal is straight off the wall: nudge diagonally inward
                let inward: SIMD2<Float> = abs(behind.x) > abs(behind.y)
                    ? SIMD2(behind.x > 0 ? -1 : 1, 0)
                    : SIMD2(0, behind.y > 0 ? -1 : 1)
                t = normalize(tangent + inward)
            }
            dir = normalize(t)
            behind = bp - dir * 0.42
        }
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
            // push depth shrinks near the goal: overshoot past the goal is
            // expensive (full re-approach from the far side)
            let d = length(toGoal)
            let push = min(0.08, d * 0.30)
            target = bp + dir * push
        }
        return simd_clamp(target, SIMD2(-3.0, -3.0), SIMD2(3.0, 3.0))
    }

    public func success(_ env: Int, posTol: Float = 0.25, yawTol: Float = .pi) -> Bool {
        let (p, yaw) = blockPose(env)
        let r = refs[env]
        var dy = yaw - r.goalYaw
        dy = dy - 2 * .pi * (dy / (2 * .pi)).rounded()
        return length(p - r.goalPos) < posTol && abs(dy) < yawTol
    }

    // ---- pixel observations ----
    public let obsRes = 64
    private var envTable: MTLBuffer?
    private var obsBuffer: MTLBuffer?

    public func observations() -> UnsafeBufferPointer<UInt8> {
        do {
            return try observationsChecked()
        } catch {
            fatalError("Push-T observation failed: \(error.localizedDescription)")
        }
    }

    public func observationsChecked() throws -> UnsafeBufferPointer<UInt8> {
        if envTable == nil {
            let dev = solver.metalDevice
            guard let newTable = dev.makeBuffer(
                    length: numEnvs * MemoryLayout<PushTEnvGPU>.stride,
                    options: .storageModeShared),
                  let newObservations = dev.makeBuffer(
                    length: numEnvs * obsRes * obsRes * 3,
                    options: .storageModeShared) else {
                throw GPUSolver.AVBDError.allocFailed(
                    "Push-T observation buffers")
            }
            envTable = newTable
            obsBuffer = newObservations
            let t = newTable.contents().bindMemory(
                to: PushTEnvGPU.self, capacity: numEnvs)
            for e in 0..<numEnvs {
                let r = refs[e]
                var g = PushTEnvGPU()
                g.ids = SIMD4(UInt32(r.tip), UInt32(r.blockBar), UInt32(r.blockStem), 0)
                g.frame = SIMD4(r.center.x, r.center.y, r.goalPos.x, r.goalPos.y)
                g.goal = SIMD4(r.goalYaw, 3.25, 0, 0)   // window covers the FULL arena
                t[e] = g
            }
        }
        try solver.renderPushTObsChecked(
            envTable: envTable!, numEnvs: numEnvs,
            out: obsBuffer!, res: obsRes)
        let ptr = obsBuffer!.contents().bindMemory(
            to: UInt8.self, capacity: numEnvs * obsRes * obsRes * 3)
        return UnsafeBufferPointer(start: ptr, count: numEnvs * obsRes * obsRes * 3)
    }
}
