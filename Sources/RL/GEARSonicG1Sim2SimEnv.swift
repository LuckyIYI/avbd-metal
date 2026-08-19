import Foundation
import simd
import SimCore
import PhysicsAVBD
import Robotics

/// Batched physical plant for NVIDIA GEAR-SONIC's 29-DoF Unitree G1 policy.
///
/// The plant itself is imported from NVIDIA's analytic-cylinder training USD.
/// Policy loading stays outside this RL environment: the type only owns source-order
/// joint state, source-position targets, and the shared Metal solve.
public final class GEARSonicG1Sim2SimEnv {
    public struct Configuration: Sendable {
        public var environmentCount: Int
        public var actuatorJointNames: [String]
        public var defaultJointPositions: [Float]
        public var stiffness: [Float]
        public var damping: [Float]
        public var armature: [Float]
        public var effortLimit: [Float]
        public var velocityLimit: [Float]
        public var motorMode: JointMotorMode
        public var physicsTimeStep: Float
        public var controlDecimation: Int
        public var rootHeight: Float
        public var environmentSpacing: Float
        public var groundMargin: Float
        public var solverIterations: Int
        /// Edge length and mass of each environment's reusable physical
        /// robustness box. Boxes start hidden below the ground and remain
        /// ordinary colliding rigid bodies when launched.
        public var projectileSize: Float
        public var projectileMass: Float
        /// Load render-only source meshes when the plant bundle includes
        /// them. They remain absent from the solver's collider buffers.
        public var includeVisuals: Bool

        public init(
            environmentCount: Int = 1,
            actuatorJointNames: [String],
            defaultJointPositions: [Float],
            stiffness: [Float], damping: [Float],
            armature: [Float], effortLimit: [Float],
            velocityLimit: [Float],
            motorMode: JointMotorMode = .implicitPositionPD,
            physicsTimeStep: Float = 0.005,
            controlDecimation: Int = 4,
            rootHeight: Float = 0.76,
            environmentSpacing: Float = 3,
            groundMargin: Float = 10,
            solverIterations: Int = 8,
            projectileSize: Float = 0.25,
            projectileMass: Float = 8,
            includeVisuals: Bool = true
        ) {
            self.environmentCount = environmentCount
            self.actuatorJointNames = actuatorJointNames
            self.defaultJointPositions = defaultJointPositions
            self.stiffness = stiffness
            self.damping = damping
            self.armature = armature
            self.effortLimit = effortLimit
            self.velocityLimit = velocityLimit
            self.motorMode = motorMode
            self.physicsTimeStep = physicsTimeStep
            self.controlDecimation = controlDecimation
            self.rootHeight = rootHeight
            self.environmentSpacing = environmentSpacing
            self.groundMargin = groundMargin
            self.solverIterations = solverIterations
            self.projectileSize = projectileSize
            self.projectileMass = projectileMass
            self.includeVisuals = includeVisuals
        }
    }

    public struct Refs {
        public var root: Int
        public var torso: Int
        public var leftFoot: Int
        public var rightFoot: Int
        public var bodies: [Int]
        public var motors: [Int]
        /// One colliding rigid box owned by this replica. It is never
        /// recreated: reset hides it and a throw updates its rigid state.
        public var projectile: Int
        public var linkFrames: [MJCFLinkFrame]
        public var rootFrame: MJCFLinkFrame
        public var torsoFrame: MJCFLinkFrame
        public var leftFootFrame: MJCFLinkFrame
        public var rightFootFrame: MJCFLinkFrame
    }

    public struct ResetState: Sendable {
        /// Absolute source joint coordinates in actuator/MuJoCo order.
        public var sourceJointPositions: [Float]
        /// Desired pelvis link-frame pose and velocity in world coordinates.
        public var rootPosition: F3
        public var rootRotation: Quat
        public var rootLinearVelocity: F3
        public var rootAngularVelocity: F3

        public init(
            sourceJointPositions: [Float],
            rootPosition: F3, rootRotation: Quat,
            rootLinearVelocity: F3 = .zero,
            rootAngularVelocity: F3 = .zero
        ) {
            self.sourceJointPositions = sourceJointPositions
            self.rootPosition = rootPosition
            self.rootRotation = rootRotation
            self.rootLinearVelocity = rootLinearVelocity
            self.rootAngularVelocity = rootAngularVelocity
        }
    }

    public let configuration: Configuration
    public let solver: GPUSolver
    public let scene: PhysicsScene
    public let plant: MJCFAsset
    public let groundBody: Int
    public let refs: [Refs]
    public let environmentOrigins: [F3]
    private let projectileHiddenPositions: [F3]
    private let projectileOwners: [Int: Int]
    private let robotBodyOwners: [Int: Int]

    public init(plantURL: URL, configuration: Configuration) throws {
        try Self.validate(configuration)
        self.configuration = configuration
        plant = try MJCFAsset.parse(url: plantURL)
        guard plant.bodyNames.count == 30,
              plant.jointNames.count == 29,
              plant.actuatorNames == configuration.actuatorJointNames else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC G1 plant must contain the exact 30-body, "
                    + "29-actuator source ordering")
        }

        var built = PhysicsScene(name: "gear-sonic-g1-sim2sim")
        built.settings.dt = configuration.physicsTimeStep
        built.settings.gravity = -9.81
        built.settings.iterations = configuration.solverIterations
        built.settings.frictionCombineMode = .multiply
        built.settings.cameraDistance = 4.2
        built.settings.cameraTargetZ = 0.8
        built.settings.cameraAzimuth = -.pi / 2
        built.settings.cameraElevation = 0.08

        let columns = max(1, Int(ceil(sqrt(Double(
            configuration.environmentCount)))))
        let rows = max(1, Int(ceil(
            Double(configuration.environmentCount) / Double(columns))))
        let halfGridX = 0.5 * Float(columns - 1)
            * configuration.environmentSpacing
        let halfGridY = 0.5 * Float(rows - 1)
            * configuration.environmentSpacing
        // The shared plane is represented by a finite static box. Centering
        // the grid keeps world coordinates well-conditioned, and deriving
        // its footprint from the batch prevents outer replicas from silently
        // spawning beyond the old fixed 80 m floor.
        let groundSize = F3(
            max(80, 2 * (halfGridX + configuration.groundMargin)),
            max(80, 2 * (halfGridY + configuration.groundMargin)),
            0.2)
        let ground = built.addBody(
            size: groundSize, density: 0, friction: 1,
            position: F3(0, 0, -0.1))

        var origins: [F3] = []
        var replicaRefs: [Refs] = []
        origins.reserveCapacity(configuration.environmentCount)
        replicaRefs.reserveCapacity(configuration.environmentCount)
        let gains = Dictionary(uniqueKeysWithValues: zip(
            configuration.actuatorJointNames,
            zip(configuration.stiffness, configuration.damping).map {
                MJCFMotorGain(stiffness: $0.0, damping: $0.1)
            }))
        let homes = Dictionary(uniqueKeysWithValues: zip(
            configuration.actuatorJointNames,
            configuration.defaultJointPositions))
        let efforts = Dictionary(uniqueKeysWithValues: zip(
            configuration.actuatorJointNames, configuration.effortLimit))
        let armatures = Dictionary(uniqueKeysWithValues: zip(
            configuration.actuatorJointNames, configuration.armature))

        for environment in 0..<configuration.environmentCount {
            let row = environment / columns
            let column = environment % columns
            let origin = F3(
                Float(column) * configuration.environmentSpacing - halfGridX,
                Float(row) * configuration.environmentSpacing - halfGridY,
                configuration.rootHeight)
            origins.append(origin)
            let imported = try plant.instantiate(
                in: &built,
                options: MJCFInstantiationOptions(
                    worldOffset: origin,
                    motorGains: gains,
                    jointHomePositions: homes,
                    motorEffortLimits: efforts,
                    jointArmatures: armatures,
                    collisionGroup: UInt32(environment + 1),
                    selfCollisions: true,
                    collideConnectedBodies: false,
                    inertiaFrame: .principal,
                    includeVisuals: configuration.includeVisuals))
            guard imported.actuatorNames == configuration.actuatorJointNames else {
                throw RLEnvironmentError.invalidConfiguration(
                    "GEAR-SONIC G1 instantiated actuator order changed")
            }
            for joint in imported.actuatorJoints {
                built.joints[joint].motorMode = configuration.motorMode
            }
            func body(_ name: String) throws -> Int {
                guard let value = imported.bodiesByName[name] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "GEAR-SONIC G1 plant is missing body \(name)")
                }
                return value
            }
            func frame(_ name: String) throws -> MJCFLinkFrame {
                guard let value = imported.linkFramesInBody[name] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "GEAR-SONIC G1 plant is missing frame \(name)")
                }
                return value
            }
            let robotBodies = try plant.bodyNames.map(body)
            let hiddenProjectilePosition = F3(origin.x, origin.y, -4)
            let projectileDensity = configuration.projectileMass
                / pow(configuration.projectileSize, 3)
            let projectile = built.addBody(
                size: F3(repeating: configuration.projectileSize),
                density: projectileDensity,
                friction: 0.7,
                position: hiddenProjectilePosition,
                collisionGroup: UInt32(environment + 1))
            replicaRefs.append(Refs(
                root: try body("pelvis"),
                torso: try body("torso_link"),
                leftFoot: try body("left_ankle_roll_link"),
                rightFoot: try body("right_ankle_roll_link"),
                bodies: robotBodies,
                motors: imported.actuatorJoints,
                projectile: projectile,
                linkFrames: try plant.bodyNames.map(frame),
                rootFrame: try frame("pelvis"),
                torsoFrame: try frame("torso_link"),
                leftFootFrame: try frame("left_ankle_roll_link"),
                rightFootFrame: try frame("right_ankle_roll_link")))
        }
        environmentOrigins = origins
        refs = replicaRefs
        projectileHiddenPositions = origins.map { F3($0.x, $0.y, -4) }
        projectileOwners = Dictionary(uniqueKeysWithValues:
            replicaRefs.indices.map { (replicaRefs[$0].projectile, $0) })
        robotBodyOwners = Dictionary(uniqueKeysWithValues:
            replicaRefs.enumerated().flatMap { environment, reference in
                reference.bodies.map { ($0, environment) }
            })
        scene = built
        groundBody = ground
        solver = try GPUSolver(scene: built)
    }

    /// Advance every replica with absolute source-link joint targets.
    /// MJCF home rebasing makes the solver coordinate zero equal the source
    /// default pose, so each target is converted exactly once at this boundary.
    public func step(
        sourceJointPositionTargets targets: ContiguousArray<Float>,
        decimation: Int? = nil
    ) throws {
        let jointCount = 29
        guard targets.count == configuration.environmentCount * jointCount,
              targets.allSatisfy(\.isFinite) else {
            throw RLEnvironmentError.invalidActionShape(
                expected: [configuration.environmentCount, jointCount],
                actual: [targets.count])
        }
        let steps = decimation ?? configuration.controlDecimation
        guard steps > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC control decimation must be positive")
        }
        try solver.synchronize()
        var updates: [GPUSolver.MotorTargetUpdate] = []
        updates.reserveCapacity(targets.count)
        for environment in refs.indices {
            for actuator in 0..<jointCount {
                updates.append(.init(
                    joint: refs[environment].motors[actuator],
                    angle: targets[environment * jointCount + actuator]
                        - configuration.defaultJointPositions[actuator]))
            }
        }
        solver.setMotorTargets(updates)
        for _ in 0..<steps { try solver.submitStep() }
        try solver.synchronize()
    }

    /// Reset complete replicas from absolute source coordinates in one GPU
    /// transaction. Initial link velocities preserve the requested root rigid
    /// motion and use zero relative joint rates; this avoids starting from a
    /// constraint-violating mix of moving and stationary links.
    public func reset(_ resetStates: [ResetState]) throws {
        guard resetStates.count == configuration.environmentCount else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC reset requires one state per environment")
        }
        try solver.synchronize()
        var bodyUpdates: [GPUSolver.BodyStateUpdate] = []
        var motorUpdates: [GPUSolver.MotorTargetUpdate] = []
        bodyUpdates.reserveCapacity(configuration.environmentCount * 31)
        motorUpdates.reserveCapacity(configuration.environmentCount * 29)
        for environment in resetStates.indices {
            let reset = resetStates[environment]
            let quaternionValues = [
                reset.rootRotation.real, reset.rootRotation.imag.x,
                reset.rootRotation.imag.y, reset.rootRotation.imag.z,
            ]
            guard reset.sourceJointPositions.count == 29,
                  reset.sourceJointPositions.allSatisfy(\.isFinite),
                  quaternionValues.allSatisfy(\.isFinite),
                  simd_length_squared(reset.rootRotation.vector) > 1e-12,
                  [reset.rootPosition, reset.rootLinearVelocity,
                   reset.rootAngularVelocity].allSatisfy({ vector in
                      vector.x.isFinite && vector.y.isFinite && vector.z.isFinite
                  }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "GEAR-SONIC reset state must be finite and 29-DoF")
            }
            let jointPositions = Dictionary(uniqueKeysWithValues: zip(
                configuration.actuatorJointNames,
                reset.sourceJointPositions))
            let authored = try plant.bodyPoses(
                worldOffset: environmentOrigins[environment],
                jointPositions: jointPositions,
                inertiaFrame: .principal)
            let rootRotation = reset.rootRotation.normalized
            for (bodyOffset, name) in plant.bodyNames.enumerated() {
                guard let pose = authored[name] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "GEAR-SONIC reset could not resolve body \(name)")
                }
                let body = refs[environment].bodies[bodyOffset]
                let relative = pose.position - environmentOrigins[environment]
                let position = reset.rootPosition + rootRotation.act(relative)
                bodyUpdates.append(.init(
                    body: body,
                    position: position,
                    rotation: (rootRotation * pose.rotation).normalized,
                    linearVelocity: reset.rootLinearVelocity
                        + cross(reset.rootAngularVelocity,
                                position - reset.rootPosition),
                    angularVelocity: reset.rootAngularVelocity))
            }
            for actuator in 0..<29 {
                motorUpdates.append(.init(
                    joint: refs[environment].motors[actuator],
                    angle: reset.sourceJointPositions[actuator]
                        - configuration.defaultJointPositions[actuator]))
            }
            bodyUpdates.append(Self.hiddenProjectileState(
                body: refs[environment].projectile,
                position: projectileHiddenPositions[environment]))
        }
        solver.setBodyStates(bodyUpdates)
        solver.setMotorTargets(motorUpdates)
    }

    /// Launch one reusable, colliding rigid box per selected replica toward
    /// its measured torso. `sideSigns` uses robot-local left (+) or right (-).
    /// The launch leads torso velocity and compensates gravity over the
    /// expected flight time; momentum is transferred only by solver contacts.
    public func throwBoxes(
        environmentIDs: [Int], sideSigns: [Float],
        launchDistance: Float = 1.2, speed: Float = 6
    ) {
        precondition(environmentIDs.count == sideSigns.count)
        precondition(Set(environmentIDs).count == environmentIDs.count
            && environmentIDs.allSatisfy(refs.indices.contains))
        precondition(launchDistance.isFinite && launchDistance > 0
            && speed.isFinite && speed > 0
            && sideSigns.allSatisfy { $0.isFinite && $0 != 0 })
        guard !environmentIDs.isEmpty else { return }

        let measured = states()
        let flightTime = launchDistance / speed
        let gravity = F3(0, 0, scene.settings.gravity)
        var updates: [GPUSolver.BodyStateUpdate] = []
        updates.reserveCapacity(environmentIDs.count)
        for (offset, environment) in environmentIDs.enumerated() {
            let state = measured[environment]
            let rootForward = state.root.rotation.act(F3(1, 0, 0))
            let forwardLength = max(
                sqrt(rootForward.x * rootForward.x
                    + rootForward.y * rootForward.y),
                1e-6)
            let forward = F3(
                rootForward.x / forwardLength,
                rootForward.y / forwardLength, 0)
            let lateral = F3(-forward.y, forward.x, 0)
            let side: Float = sideSigns[offset] > 0 ? 1 : -1
            let target = state.torso.position
            let launch = target + lateral * (launchDistance * side)
            let predictedTarget = target
                + state.torso.linearVelocity * flightTime
            let velocity = (predictedTarget - launch
                - 0.5 * gravity * flightTime * flightTime) / flightTime
            updates.append(.init(
                body: refs[environment].projectile,
                position: launch,
                rotation: Quat(real: 1, imag: .zero),
                linearVelocity: velocity,
                angularVelocity: F3(
                    side * 2.5, -side * 1.5, side * 3.5)))
        }
        solver.setBodyStates(updates)
    }

    /// Return selected boxes to their below-ground parking positions and
    /// clear both linear/angular velocity and incident contact warm starts.
    public func hideBoxes(environmentIDs: [Int]) {
        precondition(Set(environmentIDs).count == environmentIDs.count
            && environmentIDs.allSatisfy(refs.indices.contains))
        solver.setBodyStates(environmentIDs.map { environment in
            Self.hiddenProjectileState(
                body: refs[environment].projectile,
                position: projectileHiddenPositions[environment])
        })
    }

    /// Physical contact state from the last completed solve. Shared terrain
    /// contacts and other replicas are excluded from this robot-hit signal.
    public func boxRobotContacts() -> [Bool] {
        var contacts = [Bool](
            repeating: false, count: configuration.environmentCount)
        for (a, b) in solver.activeRigidContactPairs() {
            if let environment = projectileOwners[a],
               robotBodyOwners[b] == environment {
                contacts[environment] = true
            } else if let environment = projectileOwners[b],
                      robotBodyOwners[a] == environment {
                contacts[environment] = true
            }
        }
        return contacts
    }

    /// Measured semantic-link states and absolute source joint coordinates.
    public func states() -> [HumanoidState] {
        let bodyIDs = refs.flatMap {
            [$0.root, $0.torso, $0.leftFoot, $0.rightFoot]
        }
        let bodyStates = solver.bodyStates(bodyIDs)
        let motorStates = solver.motorStates(refs.flatMap(\.motors))
        return refs.indices.map { environment in
            let bodyOffset = environment * 4
            let motorOffset = environment * 29
            let reference = refs[environment]
            return HumanoidState(
                root: Self.linkState(
                    bodyStates[bodyOffset], reference.rootFrame),
                torso: Self.linkState(
                    bodyStates[bodyOffset + 1], reference.torsoFrame),
                leftFoot: Self.linkState(
                    bodyStates[bodyOffset + 2], reference.leftFootFrame),
                rightFoot: Self.linkState(
                    bodyStates[bodyOffset + 3], reference.rightFootFrame),
                jointAngles: (0..<29).map {
                    motorStates[motorOffset + $0].angle
                        + configuration.defaultJointPositions[$0]
                },
                jointVelocities: (0..<29).map {
                    motorStates[motorOffset + $0].velocity
                })
        }
    }

    /// Read source link-frame states for an ordered subset of the exact
    /// 30-body plant. This is primarily used for source-aligned tracking
    /// metrics; physics and rendering continue to own COM/principal frames.
    public func linkStates(bodyIndices: [Int]) -> [[GPUSolver.RigidBodyState]] {
        precondition(!bodyIndices.isEmpty
            && bodyIndices.allSatisfy(plant.bodyNames.indices.contains),
            "GEAR-SONIC link-state indices must address the 30-body plant")
        let bodyIDs = refs.flatMap { reference in
            bodyIndices.map { reference.bodies[$0] }
        }
        let states = solver.bodyStates(bodyIDs)
        return refs.indices.map { environment in
            bodyIndices.indices.map { offset in
                let bodyIndex = bodyIndices[offset]
                return Self.linkState(
                    states[environment * bodyIndices.count + offset],
                    refs[environment].linkFrames[bodyIndex])
            }
        }
    }

    private static func validate(_ configuration: Configuration) throws {
        let vectors: [[Float]] = [
            configuration.defaultJointPositions, configuration.stiffness,
            configuration.damping, configuration.armature,
            configuration.effortLimit, configuration.velocityLimit,
        ]
        guard configuration.environmentCount > 0,
              configuration.actuatorJointNames.count == 29,
              Set(configuration.actuatorJointNames).count == 29,
              vectors.allSatisfy({ $0.count == 29 }),
              vectors.flatMap({ $0 }).allSatisfy(\.isFinite),
              configuration.stiffness.allSatisfy({ $0 > 0 }),
              configuration.damping.allSatisfy({ $0 > 0 }),
              configuration.armature.allSatisfy({ $0 > 0 }),
              configuration.effortLimit.allSatisfy({ $0 > 0 }),
              configuration.velocityLimit.allSatisfy({ $0 > 0 }),
              abs(configuration.physicsTimeStep - 0.005) < 1e-8,
              configuration.controlDecimation == 4,
              configuration.rootHeight.isFinite,
              configuration.environmentSpacing.isFinite,
              configuration.environmentSpacing > 0,
              configuration.groundMargin.isFinite,
              configuration.groundMargin > 0,
              configuration.projectileSize.isFinite,
              configuration.projectileSize > 0,
              configuration.projectileMass.isFinite,
              configuration.projectileMass > 0,
              configuration.solverIterations > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid GEAR-SONIC G1 plant configuration")
        }
    }

    private static func linkState(
        _ body: GPUSolver.RigidBodyState, _ frame: MJCFLinkFrame
    ) -> GPUSolver.RigidBodyState {
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
