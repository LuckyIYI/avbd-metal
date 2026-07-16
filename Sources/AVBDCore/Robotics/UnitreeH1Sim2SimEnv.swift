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
        public var rootFrame: MJCFLinkFrame
        public var leftFootFrame: MJCFLinkFrame
        public var rightFootFrame: MJCFLinkFrame
    }

    public let solver: GPUSolver
    public let scene: PhysicsScene
    public let groundBody: Int
    public let refs: Refs

    public init(solverIterations: Int? = nil) throws {
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
            in: &built, motorGains: gains, selfCollisions: false,
            inertiaFrame: .principal)
        for joint in imported.actuatorJoints {
            built.joints[joint].motorMode = .explicitTorquePD
        }
        let root = imported.bodiesByName["pelvis"]!
        let leftFoot = imported.bodiesByName["left_ankle_link"]!
        let rightFoot = imported.bodiesByName["right_ankle_link"]!
        refs = Refs(
            root: root, leftFoot: leftFoot, rightFoot: rightFoot,
            bodies: asset.bodyNames.map { imported.bodiesByName[$0]! },
            motors: imported.actuatorJoints,
            rootFrame: imported.linkFramesInBody["pelvis"]!,
            leftFootFrame: imported.linkFramesInBody["left_ankle_link"]!,
            rightFootFrame: imported.linkFramesInBody["right_ankle_link"]!)
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
}
