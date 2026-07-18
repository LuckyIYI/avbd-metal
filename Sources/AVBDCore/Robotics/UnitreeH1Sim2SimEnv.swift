import Foundation
import simd

/// Source-compatible 10-DoF Unitree H1 plant for unchanged-policy transfer.
/// This is intentionally separate from the native 19-DoF training task: the
/// public Unitree checkpoint was trained/deployed with a rigid upper body.
public final class UnitreeH1Sim2SimEnv {
    public struct Refs {
        public var root: Int
        public var leftFoot: Int
        public var rightFoot: Int
        public var bodies: [Int]
        public var motors: [Int]
        /// Reusable colliding robustness box. It starts below the ground and
        /// is relaunched by updating its ordinary rigid-body state.
        public var projectile: Int
        public var rootFrame: MJCFLinkFrame
        public var leftFootFrame: MJCFLinkFrame
        public var rightFootFrame: MJCFLinkFrame
    }

    public let solver: GPUSolver
    public let scene: PhysicsScene
    public let groundBody: Int
    public let refs: Refs
    public let projectileSize: Float
    public let projectileMass: Float

    private let projectileHiddenPosition = F3(0, 0, -4)
    private let robotBodySet: Set<Int>

    public init(solverIterations: Int? = nil,
                projectileSize: Float = 0.25,
                projectileMass: Float = 8) throws {
        precondition(projectileSize.isFinite && projectileSize > 0
            && projectileMass.isFinite && projectileMass > 0)
        self.projectileSize = projectileSize
        self.projectileMass = projectileMass
        var built = PhysicsScene(name: "unitree-h1-sim2sim")
        built.settings.dt = 0.002
        built.settings.gravity = -9.81
        built.settings.iterations = solverIterations ?? 20
        built.settings.frictionCombineMode = .geometricMean
        built.settings.betaLin = 20_000
        built.settings.betaAng = 400
        built.settings.lambdaMax = 1_200
        built.settings.cameraDistance = 8
        built.settings.cameraTargetX = 2
        built.settings.cameraTargetZ = 0.9
        built.settings.cameraAzimuth = -.pi / 2
        built.settings.cameraElevation = 0.12
        let ground = built.addBody(
            size: F3(80, 40, 2), density: 0, friction: 1,
            position: F3(10, 0, -1))
        let asset = try MJCFAsset.bundledUnitreeRLGymH1()
        let gains: [String: MJCFMotorGain] = [
            "left_hip_yaw": .init(stiffness: 150, damping: 2),
            "left_hip_roll": .init(stiffness: 150, damping: 2),
            "left_hip_pitch": .init(stiffness: 150, damping: 2),
            "left_knee": .init(stiffness: 200, damping: 4),
            "left_ankle": .init(stiffness: 40, damping: 2),
            "right_hip_yaw": .init(stiffness: 150, damping: 2),
            "right_hip_roll": .init(stiffness: 150, damping: 2),
            "right_hip_pitch": .init(stiffness: 150, damping: 2),
            "right_knee": .init(stiffness: 200, damping: 4),
            "right_ankle": .init(stiffness: 40, damping: 2),
        ]
        let imported = try asset.instantiate(
            in: &built,
            options: MJCFInstantiationOptions(
                motorGains: gains,
                collisionGroup: 1,
                selfCollisions: false,
                inertiaFrame: .principal))
        for joint in imported.actuatorJoints {
            built.joints[joint].motorMode = .explicitTorquePD
        }
        let root = imported.bodiesByName["pelvis"]!
        let leftFoot = imported.bodiesByName["left_ankle_link"]!
        let rightFoot = imported.bodiesByName["right_ankle_link"]!
        let bodies = asset.bodyNames.map { imported.bodiesByName[$0]! }
        let projectile = built.addBody(
            size: F3(repeating: projectileSize),
            density: projectileMass / pow(projectileSize, 3),
            friction: 0.7,
            position: projectileHiddenPosition,
            collisionGroup: 1)
        refs = Refs(
            root: root, leftFoot: leftFoot, rightFoot: rightFoot,
            bodies: bodies,
            motors: imported.actuatorJoints,
            projectile: projectile,
            rootFrame: imported.linkFramesInBody["pelvis"]!,
            leftFootFrame: imported.linkFramesInBody["left_ankle_link"]!,
            rightFootFrame: imported.linkFramesInBody["right_ankle_link"]!)
        robotBodySet = Set(bodies)
        precondition(refs.motors.count == 10)
        scene = built
        groundBody = ground
        solver = try GPUSolver(scene: built)
    }

    public func step(jointPositionTargets: ContiguousArray<Float>,
                     decimation: Int) {
        precondition(jointPositionTargets.count == 10)
        precondition(decimation > 0)
        solver.setMotorTargets(zip(refs.motors, jointPositionTargets).map {
            GPUSolver.MotorTargetUpdate(joint: $0.0, angle: $0.1)
        })
        for _ in 0..<decimation { solver.step() }
    }

    /// Launch the reusable physical box toward the measured rigid upper body.
    /// `sideSign` is robot-local left (+) or right (-). The ballistic velocity
    /// leads the torso's current motion and compensates gravity in flight.
    public func throwBoxes(environmentIDs: [Int], sideSigns: [Float],
                           launchDistance: Float = 1.2,
                           speed: Float = 6) {
        precondition(environmentIDs.count == sideSigns.count
            && Set(environmentIDs).count == environmentIDs.count
            && environmentIDs.allSatisfy { $0 == 0 })
        guard !environmentIDs.isEmpty else { return }
        let sideSign = sideSigns[0]
        precondition(sideSign.isFinite && sideSign != 0
            && launchDistance.isFinite && launchDistance > 0
            && speed.isFinite && speed > 0)
        let measured = state()
        let rootForward = measured.root.rotation.act(F3(1, 0, 0))
        let forwardLength = max(
            sqrt(rootForward.x * rootForward.x
                + rootForward.y * rootForward.y),
            1e-6)
        let forward = F3(
            rootForward.x / forwardLength,
            rootForward.y / forwardLength, 0)
        let lateral = F3(-forward.y, forward.x, 0)
        let side: Float = sideSign > 0 ? 1 : -1
        let target = measured.torso.position
        let launch = target + lateral * (launchDistance * side)
        let flightTime = launchDistance / speed
        let gravity = F3(0, 0, scene.settings.gravity)
        let predictedTarget = target
            + measured.torso.linearVelocity * flightTime
        let velocity = (predictedTarget - launch
            - 0.5 * gravity * flightTime * flightTime) / flightTime
        solver.setBodyStates([.init(
            body: refs.projectile,
            position: launch,
            rotation: Quat(real: 1, imag: .zero),
            linearVelocity: velocity,
            angularVelocity: F3(side * 2.5, -side * 1.5, side * 3.5))])
    }

    /// Restore the reusable box to its below-ground parking state, clearing
    /// velocities and incident constraint warm starts through `setBodyStates`.
    public func hideBoxes(environmentIDs: [Int]) {
        precondition(Set(environmentIDs).count == environmentIDs.count
            && environmentIDs.allSatisfy { $0 == 0 })
        guard !environmentIDs.isEmpty else { return }
        solver.setBodyStates([Self.hiddenProjectileState(
            body: refs.projectile, position: projectileHiddenPosition)])
    }

    /// Whether the last completed physics step produced a real box/robot
    /// contact. Ground contact is deliberately excluded.
    public func boxRobotContacts() -> [Bool] {
        for (a, b) in solver.activeRigidContactPairs() {
            if a == refs.projectile, robotBodySet.contains(b) { return [true] }
            if b == refs.projectile, robotBodySet.contains(a) { return [true] }
        }
        return [false]
    }

    public func state() -> HumanoidState {
        let bodies = solver.bodyStates([
            refs.root, refs.leftFoot, refs.rightFoot,
        ])
        let joints = solver.motorStates(refs.motors)
        let root = Self.linkState(bodies[0], refs.rootFrame)
        return HumanoidState(
            root: root, torso: root,
            leftFoot: Self.linkState(bodies[1], refs.leftFootFrame),
            rightFoot: Self.linkState(bodies[2], refs.rightFootFrame),
            jointAngles: joints.map(\.angle),
            jointVelocities: joints.map(\.velocity))
    }

    private static func linkState(_ body: GPUSolver.RigidBodyState,
                                  _ frame: MJCFLinkFrame)
        -> GPUSolver.RigidBodyState {
        let offset = body.rotation.act(frame.position)
        return GPUSolver.RigidBodyState(
            position: body.position + offset,
            rotation: (body.rotation * frame.rotation).normalized,
            linearVelocity: body.linearVelocity
                + cross(body.angularVelocity, offset),
            angularVelocity: body.angularVelocity)
    }

    private static func hiddenProjectileState(
        body: Int, position: F3
    ) -> GPUSolver.BodyStateUpdate {
        .init(
            body: body, position: position,
            rotation: Quat(real: 1, imag: .zero),
            linearVelocity: .zero, angularVelocity: .zero)
    }
}
