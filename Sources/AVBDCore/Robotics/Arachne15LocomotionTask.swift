import simd

public struct Arachne15LocomotionTaskConfig: Sendable {
    public var numEnvironments: Int
    public var seed: UInt64
    public var maxEpisodeSteps: Int
    public var controlDecimation: Int
    /// Nonlinear AVBD sweeps per 2 ms physics step. Small, light feet need a
    /// higher contact budget than the metre-scale demo default; keeping this
    /// in the serialized task contract prevents silent train/replay drift.
    public var solverIterations: Int
    public var commandResamplingSteps: Int
    public var minimumForwardVelocity: Float
    public var maximumForwardVelocity: Float
    public var maximumLateralVelocity: Float
    public var maximumYawRate: Float
    public var standingCommandProbability: Float
    public var initialRollPitchRange: Float
    public var initialYawRange: Float
    public var observationNoise: Bool
    public var maximumActionLatencySteps: Int
    public var collisionProfile: Arachne15CollisionProfile
    public var domainRandomization: ArticulationDomainRandomization
    public var pointGoal: Bool
    public var minimumGoalDistance: Float
    public var maximumGoalDistance: Float
    public var maximumGoalDirectionAngle: Float
    public var goalRadius: Float
    public var goalSlowdownDistance: Float
    public var goalCommandSpeed: Float
    public var goalBoundaryCommandSpeed: Float
    public var maximumGoalArrivalSpeed: Float
    public var goalDwellSteps: Int
    public var goalProgressRewardWeight: Float
    public var goalStableRewardWeight: Float
    public var goalSuccessBonus: Float
    public var commandProgressRewardWeight: Float
    public var velocityErrorPenaltyWeight: Float
    public var yawErrorPenaltyWeight: Float
    public var autoReset: Bool

    public init(
        numEnvironments: Int, seed: UInt64 = 1,
        maxEpisodeSteps: Int = 1_000, controlDecimation: Int = 10,
        solverIterations: Int = 20,
        commandResamplingSteps: Int = 500,
        minimumForwardVelocity: Float = 0.05,
        maximumForwardVelocity: Float = 0.25,
        maximumLateralVelocity: Float = 0.12,
        maximumYawRate: Float = 0.8,
        standingCommandProbability: Float = 0.10,
        initialRollPitchRange: Float = 0.02,
        initialYawRange: Float = .pi,
        observationNoise: Bool = true,
        maximumActionLatencySteps: Int = 2,
        collisionProfile: Arachne15CollisionProfile = .training,
        domainRandomization: ArticulationDomainRandomization =
            .conservativeSimToReal,
        pointGoal: Bool = false,
        minimumGoalDistance: Float = 0.60,
        maximumGoalDistance: Float = 2.0,
        maximumGoalDirectionAngle: Float = .pi,
        goalRadius: Float = 0.12,
        goalSlowdownDistance: Float = 0.50,
        goalCommandSpeed: Float = 0.15,
        goalBoundaryCommandSpeed: Float = 0.02,
        maximumGoalArrivalSpeed: Float = 0.08,
        goalDwellSteps: Int = 15,
        goalProgressRewardWeight: Float = 4.0,
        goalStableRewardWeight: Float = 1.0,
        goalSuccessBonus: Float = 8.0,
        commandProgressRewardWeight: Float = 20.0,
        velocityErrorPenaltyWeight: Float = 5.0,
        yawErrorPenaltyWeight: Float = 5.0,
        autoReset: Bool = true
    ) {
        self.numEnvironments = numEnvironments
        self.seed = seed
        self.maxEpisodeSteps = maxEpisodeSteps
        self.controlDecimation = controlDecimation
        self.solverIterations = solverIterations
        self.commandResamplingSteps = commandResamplingSteps
        self.minimumForwardVelocity = minimumForwardVelocity
        self.maximumForwardVelocity = maximumForwardVelocity
        self.maximumLateralVelocity = maximumLateralVelocity
        self.maximumYawRate = maximumYawRate
        self.standingCommandProbability = standingCommandProbability
        self.initialRollPitchRange = initialRollPitchRange
        self.initialYawRange = initialYawRange
        self.observationNoise = observationNoise
        self.maximumActionLatencySteps = maximumActionLatencySteps
        self.collisionProfile = collisionProfile
        self.domainRandomization = domainRandomization
        self.pointGoal = pointGoal
        self.minimumGoalDistance = minimumGoalDistance
        self.maximumGoalDistance = maximumGoalDistance
        self.maximumGoalDirectionAngle = maximumGoalDirectionAngle
        self.goalRadius = goalRadius
        self.goalSlowdownDistance = goalSlowdownDistance
        self.goalCommandSpeed = goalCommandSpeed
        self.goalBoundaryCommandSpeed = goalBoundaryCommandSpeed
        self.maximumGoalArrivalSpeed = maximumGoalArrivalSpeed
        self.goalDwellSteps = goalDwellSteps
        self.goalProgressRewardWeight = goalProgressRewardWeight
        self.goalStableRewardWeight = goalStableRewardWeight
        self.goalSuccessBonus = goalSuccessBonus
        self.commandProgressRewardWeight = commandProgressRewardWeight
        self.velocityErrorPenaltyWeight = velocityErrorPenaltyWeight
        self.yawErrorPenaltyWeight = yawErrorPenaltyWeight
        self.autoReset = autoReset
    }
}

public struct Arachne15State {
    public var root: GPUSolver.RigidBodyState
    public var feet: [GPUSolver.RigidBodyState]
    public var jointAngles: [Float]
    public var jointVelocities: [Float]
}

/// Batched Arachne plant. Every replica occupies the same coordinates and is
/// isolated by a collision group, while group-zero terrain is shared. This
/// keeps floating-point conditioning identical between one-view replay and a
/// large training batch.
public final class Arachne15Env {
    public struct EnvRefs {
        public var root: Int
        public var rootFrame: MJCFLinkFrame
        public var bodies: [Int]
        public var motors: [Int]
        public var feet: [Int]
        public var footFrames: [MJCFLinkFrame]
        /// Reusable physical robustness box owned by this replica.
        public var projectile: Int?
        public var startMarker: Int?
        public var goalMarker: Int?
    }

    public static let actionDimension = Arachne15PolicyContract.actionDimension
    public static let footLinkOffset = F3(0.105, 0, 0)
    /// Must match the generated MJCF foot contact box. Keeping this value
    /// explicit here lets runtime diagnostics measure the actual contact
    /// envelope rather than inferring clearance from the foot-link origin.
    public static let footColliderHalfSize = F3(0.00875, 0.007, 0.004)
    public static let actionScales = Arachne15PolicyContract.actionScales

    public let numEnvironments: Int
    public let solver: GPUSolver
    public let scene: PhysicsScene
    public let groundBody: Int
    public let dynamicsScales: [MJCFDynamicsScale]
    public let projectileSize: Float
    public let projectileMass: Float
    public private(set) var refs: [EnvRefs]

    private let spawnPoses: [(F3, Quat)]
    private let groundContactSlots: [Int: Int]
    private let projectileHiddenPositions: [F3]
    private let projectileOwners: [Int: Int]
    private let robotBodyOwners: [Int: Int]

    public init(
        numEnvironments: Int, seed: UInt64 = 1,
        collisionProfile: Arachne15CollisionProfile = .training,
        domainRandomization: ArticulationDomainRandomization = .init(),
        includeGoalMarkers: Bool = false,
        goalMarkerRadius: Float = 0.12,
        solverIterations: Int = 20,
        includeProjectiles: Bool = false,
        projectileSize: Float = 0.05,
        projectileMass: Float = 0.10
    ) throws {
        precondition(numEnvironments > 0 && solverIterations > 0
            && goalMarkerRadius > 0
            && projectileSize.isFinite && projectileSize > 0
            && projectileMass.isFinite && projectileMass > 0)
        self.numEnvironments = numEnvironments
        self.projectileSize = projectileSize
        self.projectileMass = projectileMass
        var built = PhysicsScene(name: "arachne15-locomotion")
        built.settings.dt = 0.002
        built.settings.gravity = -9.80665
        built.settings.iterations = solverIterations
        // The legacy 1 cm rigid-world skin is thicker than the printable
        // 8 mm foot pad. A quarter-millimetre skin retains speculative
        // contact without letting the CAD settle visibly through the floor.
        built.settings.collisionMargin = 0.00025
        built.settings.frictionCombineMode = .geometricMean
        built.settings.betaLin = 20_000
        built.settings.betaAng = 400
        built.settings.lambdaMax = 1_200
        built.settings.cameraDistance = 0.80
        built.settings.cameraTargetX = 0.12
        built.settings.cameraTargetY = 0
        built.settings.cameraTargetZ = 0.07
        built.settings.cameraAzimuth = -.pi / 2
        built.settings.cameraElevation = 0.20

        let ground = built.addBody(
            size: F3(8, 8, 0.10), density: 0, friction: 0.9,
            position: F3(0, 0, -0.05))
        let asset = try MJCFAsset.bundledArachne15(profile: collisionProfile)
        var builtRefs: [EnvRefs] = []
        var scales: [MJCFDynamicsScale] = []
        builtRefs.reserveCapacity(numEnvironments)
        scales.reserveCapacity(numEnvironments)
        let footNames = asset.bodyNames.filter { $0.hasSuffix("_tibia") }
        precondition(footNames.count == 8)
        for e in 0..<numEnvironments {
            let scale = domainRandomization.sample(
                seed: seed &+ UInt64(e) &* 0x9E3779B97F4A7C15)
            let imported = try asset.instantiate(
                in: &built,
                options: MJCFInstantiationOptions(
                    defaultMotorGain: .init(
                        stiffness: 2.0, damping: 0.08),
                    collisionGroup: UInt32(e + 1),
                    selfCollisions: false,
                    dynamicsScale: scale,
                    includeVisuals: numEnvironments <= 4))
            let bodies = asset.bodyNames.map { imported.bodiesByName[$0]! }
            let feet = footNames.map { imported.bodiesByName[$0]! }
            let footFrames = footNames.map { imported.linkFramesInBody[$0]! }
            let projectile: Int?
            if includeProjectiles {
                projectile = built.addBody(
                    size: F3(repeating: projectileSize),
                    density: projectileMass / pow(projectileSize, 3),
                    friction: 0.7,
                    position: F3(0, 0, -4),
                    collisionGroup: UInt32(e + 1))
            } else {
                projectile = nil
            }
            var ref = EnvRefs(
                root: imported.rootBody,
                rootFrame: imported.linkFramesInBody["base"]!,
                bodies: bodies,
                motors: imported.actuatorJoints,
                feet: feet,
                footFrames: footFrames,
                projectile: projectile,
                startMarker: nil,
                goalMarker: nil)
            if includeGoalMarkers && numEnvironments <= 4 {
                // Visual-only navigation landmarks. Static marker colliders
                // are rendered by the normal scene path but deliberately do
                // not participate in broadphase, contacts, or robot state.
                ref.startMarker = built.addBody(
                    size: F3(0.10, 0.10, 0.006), density: 0, friction: 0,
                    position: F3(0, 0, 0.003), collisionEnabled: false)
                built.addCollider(
                    body: ref.startMarker!, size: F3(0.10, 0.10, 0.006),
                    friction: 0, collisionEnabled: false, isRendered: true)
                ref.goalMarker = built.addBody(
                    size: F3(repeating: 0.08), density: 0, friction: 0,
                    position: F3(1, 0, 0.04), shape: .sphere,
                    collisionEnabled: false)
                built.addCollider(
                    body: ref.goalMarker!, size: F3(repeating: 0.08),
                    friction: 0, shape: .sphere,
                    collisionEnabled: false, isRendered: true)
                built.addCollider(
                    body: ref.goalMarker!,
                    size: F3(goalMarkerRadius, 0.012, 0),
                    friction: 0, localPosition: F3(0, 0, -0.028),
                    shape: .torus,
                    collisionEnabled: false, isRendered: true)
            }
            builtRefs.append(ref)
            scales.append(scale)
        }
        refs = builtRefs
        dynamicsScales = scales
        scene = built
        groundBody = ground
        spawnPoses = built.bodies.map { ($0.position, $0.rotation) }
        var slots: [Int: Int] = [:]
        for (e, ref) in builtRefs.enumerated() {
            for (foot, body) in ref.feet.enumerated() {
                slots[body] = e * 8 + foot
            }
        }
        groundContactSlots = slots
        projectileHiddenPositions = [F3](
            repeating: F3(0, 0, -4), count: numEnvironments)
        projectileOwners = Dictionary(uniqueKeysWithValues:
            builtRefs.indices.compactMap { environment in
                builtRefs[environment].projectile.map { ($0, environment) }
            })
        robotBodyOwners = Dictionary(uniqueKeysWithValues:
            builtRefs.enumerated().flatMap { environment, reference in
                reference.bodies.map { ($0, environment) }
            })
        solver = try GPUSolver(scene: built)
    }

    public func step(actions: ContiguousArray<Float>, decimation: Int) {
        do {
            try stepChecked(actions: actions, decimation: decimation)
        } catch {
            fatalError("Arachne simulation failed: \(error.localizedDescription)")
        }
    }

    public func stepChecked(actions: ContiguousArray<Float>,
                            decimation: Int) throws {
        precondition(actions.count == numEnvironments * Self.actionDimension)
        var relativeTargets = ContiguousArray(
            repeating: Float(0), count: actions.count)
        for e in 0..<numEnvironments {
            for j in 0..<Self.actionDimension {
                let requested = simd_clamp(actions[e * Self.actionDimension + j],
                                           -1, 1)
                    * Self.actionScales[j]
                relativeTargets[e * Self.actionDimension + j] = requested
            }
        }
        try stepJointTargetsChecked(relativeTargets, decimation: decimation)
    }

    /// Commissioning/reveal path for targets beyond the learned action scale
    /// but still inside authored mechanical limits. This does not move the
    /// root or link poses directly; it only drives the same torque-limited
    /// motors used by policies.
    public func stepJointTargets(
        _ relativeTargets: ContiguousArray<Float>, decimation: Int
    ) {
        do {
            try stepJointTargetsChecked(
                relativeTargets, decimation: decimation)
        } catch {
            fatalError("Arachne simulation failed: \(error.localizedDescription)")
        }
    }

    public func stepJointTargetsChecked(
        _ relativeTargets: ContiguousArray<Float>, decimation: Int
    ) throws {
        precondition(relativeTargets.count
            == numEnvironments * Self.actionDimension)
        precondition(decimation > 0)
        try solver.synchronize()
        var targets: [GPUSolver.MotorTargetUpdate] = []
        targets.reserveCapacity(relativeTargets.count)
        for e in 0..<numEnvironments {
            for j in 0..<Self.actionDimension {
                let joint = refs[e].motors[j]
                targets.append(.init(
                    joint: joint,
                    angle: simd_clamp(
                        relativeTargets[e * Self.actionDimension + j],
                        scene.joints[joint].limitLo,
                        scene.joints[joint].limitHi)))
            }
        }
        solver.setMotorTargets(targets)
        for _ in 0..<decimation { try solver.submitStep() }
        try solver.synchronize()
    }

    /// Temporarily scale the authored effort budget for short commissioning
    /// maneuvers. A scale of one restores the exact training/runtime torque.
    /// The default reveal scale remains only 40% of the selected servo's
    /// 5 V stall torque (the walking budget is 20%).
    public func setCommissioningTorqueScale(_ scale: Float) {
        precondition(scale.isFinite && scale >= 1 && scale <= 2)
        var updates: [GPUSolver.MotorTorqueUpdate] = []
        updates.reserveCapacity(numEnvironments * Self.actionDimension)
        for ref in refs {
            for joint in ref.motors {
                updates.append(.init(
                    joint: joint,
                    torque: scene.joints[joint].motorTorque * scale))
            }
        }
        solver.setMotorTorques(updates)
    }

    public func reset(_ environmentIDs: [Int], seeds: [UInt64],
                      initialRollPitchRange: Float = 0,
                      initialYawRange: Float = 0) {
        precondition(environmentIDs.count == seeds.count)
        var bodyStates: [GPUSolver.BodyStateUpdate] = []
        var motors: [GPUSolver.MotorTargetUpdate] = []
        for (offset, e) in environmentIDs.enumerated() {
            var rng = SplitMix64(seed: seeds[offset] ^ 0xD1B54A32D192ED03)
            let roll = (2 * rng.nextFloat() - 1) * initialRollPitchRange
            let pitch = (2 * rng.nextFloat() - 1) * initialRollPitchRange
            let yaw = (2 * rng.nextFloat() - 1) * initialYawRange
            let perturbation = (
                Quat(angle: yaw, axis: F3(0, 0, 1))
                    * Quat(angle: pitch, axis: F3(0, 1, 0))
                    * Quat(angle: roll, axis: F3(1, 0, 0))).normalized
            let pivot = spawnPoses[refs[e].root].0
            var transformed: [(body: Int, position: F3, rotation: Quat)] = []
            transformed.reserveCapacity(refs[e].bodies.count)
            var minimumFootClearance = Float.infinity
            for body in refs[e].bodies {
                let spawn = spawnPoses[body]
                let position = pivot + perturbation.act(spawn.0 - pivot)
                let rotation = (perturbation * spawn.1).normalized
                transformed.append((body, position, rotation))
                if refs[e].feet.contains(body) {
                    let half = Self.footColliderHalfSize
                    let verticalSupport =
                        abs(rotation.act(F3(1, 0, 0)).z) * half.x
                        + abs(rotation.act(F3(0, 1, 0)).z) * half.y
                        + abs(rotation.act(F3(0, 0, 1)).z) * half.z
                    minimumFootClearance = min(
                        minimumFootClearance, position.z - verticalSupport)
                }
            }
            // Roll/pitch randomization rotates the complete robot about its
            // root. Without a compensating lift, one side starts millimetres
            // inside the floor and falsely teaches recovery from an impossible
            // assembled pose. Preserve the randomized attitude while placing
            // the lowest authored foot just above the physical ground.
            let verticalLift = max(0, 0.0001 - minimumFootClearance)
            for transformedPose in transformed {
                bodyStates.append(.init(
                    body: transformedPose.body,
                    position: transformedPose.position + F3(0, 0, verticalLift),
                    rotation: transformedPose.rotation))
            }
            if let projectile = refs[e].projectile {
                bodyStates.append(Self.hiddenProjectileState(
                    body: projectile,
                    position: projectileHiddenPositions[e]))
            }
            for joint in refs[e].motors {
                motors.append(.init(joint: joint, angle: 0))
            }
        }
        solver.setBodyStates(bodyStates)
        solver.setMotorTargets(motors)
    }

    /// Reposition render-only start and destination markers. These bodies are
    /// excluded from `EnvRefs.bodies`, so robot resets and policy state cannot
    /// accidentally depend on them.
    public func setGoalMarkers(environmentIDs: [Int], goals: [F3],
                               origins: [F3]) {
        precondition(environmentIDs.count == goals.count
            && environmentIDs.count == origins.count)
        var poses = [GPUSolver.BodyPoseUpdate]()
        poses.reserveCapacity(environmentIDs.count * 2)
        for (offset, e) in environmentIDs.enumerated() {
            if let start = refs[e].startMarker {
                poses.append(.init(
                    body: start,
                    position: F3(origins[offset].x, origins[offset].y, 0.003),
                    rotation: Quat(real: 1, imag: .zero)))
            }
            if let goal = refs[e].goalMarker {
                poses.append(.init(
                    body: goal,
                    position: F3(goals[offset].x, goals[offset].y, 0.04),
                    rotation: Quat(real: 1, imag: .zero)))
            }
        }
        if !poses.isEmpty { solver.setBodyPoses(poses) }
    }

    /// Launch one reusable colliding box per selected replica toward the
    /// measured chassis. `sideSigns` is robot-local left (+) or right (-).
    /// Gravity and current chassis velocity are included in the intercept.
    public func throwBoxes(
        environmentIDs: [Int], sideSigns: [Float],
        launchDistance: Float = 0.35, speed: Float = 2.5
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
            let target = state.root.position
            let launch = target + lateral * (launchDistance * side)
            let predictedTarget = target
                + state.root.linearVelocity * flightTime
            let velocity = (predictedTarget - launch
                - 0.5 * gravity * flightTime * flightTime) / flightTime
            guard let projectile = refs[environment].projectile else {
                continue
            }
            updates.append(.init(
                body: projectile,
                position: launch,
                rotation: Quat(real: 1, imag: .zero),
                linearVelocity: velocity,
                angularVelocity: F3(
                    side * 2.5, -side * 1.5, side * 3.5)))
        }
        solver.setBodyStates(updates)
    }

    /// Return selected boxes below the ground, clearing velocity and contact
    /// warm starts so each launch is independent of earlier impacts.
    public func hideBoxes(environmentIDs: [Int]) {
        precondition(Set(environmentIDs).count == environmentIDs.count
            && environmentIDs.allSatisfy(refs.indices.contains))
        solver.setBodyStates(environmentIDs.compactMap { environment in
            refs[environment].projectile.map { projectile in
                Self.hiddenProjectileState(
                    body: projectile,
                    position: projectileHiddenPositions[environment])
            }
        })
    }

    public func hasProjectile(environment: Int) -> Bool {
        precondition(refs.indices.contains(environment))
        return refs[environment].projectile != nil
    }

    /// Physical box/articulation contacts from the last completed solve.
    /// Shared-ground contacts and other replicas are excluded.
    public func boxRobotContacts() -> [Bool] {
        var contacts = [Bool](repeating: false, count: numEnvironments)
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

    public func groundContacts() -> [[Bool]] {
        var result = [[Bool]](repeating: [Bool](repeating: false, count: 8),
                              count: numEnvironments)
        for (a, b) in solver.activeRigidContactPairs() {
            let other: Int
            if a == groundBody { other = b }
            else if b == groundBody { other = a }
            else { continue }
            guard let slot = groundContactSlots[other] else { continue }
            result[slot / 8][slot % 8] = true
        }
        return result
    }

    public func states() -> [Arachne15State] {
        var bodyIDs: [Int] = []
        var jointIDs: [Int] = []
        bodyIDs.reserveCapacity(numEnvironments * 9)
        jointIDs.reserveCapacity(numEnvironments * Self.actionDimension)
        for ref in refs {
            bodyIDs.append(ref.root)
            bodyIDs.append(contentsOf: ref.feet)
            jointIDs.append(contentsOf: ref.motors)
        }
        let bodies = solver.bodyStates(bodyIDs)
        let joints = solver.motorStates(jointIDs)
        return (0..<numEnvironments).map { e in
            let bodyBase = e * 9
            let jointBase = e * Self.actionDimension
            let ref = refs[e]
            return Arachne15State(
                root: Self.linkState(bodies[bodyBase], frame: ref.rootFrame),
                feet: (0..<8).map { foot in
                    let link = Self.linkState(
                        bodies[bodyBase + 1 + foot],
                        frame: ref.footFrames[foot])
                    return Self.offsetState(link, localOffset: Self.footLinkOffset)
                },
                jointAngles: joints[
                    jointBase..<(jointBase + Self.actionDimension)].map(\.angle),
                jointVelocities: joints[
                    jointBase..<(jointBase + Self.actionDimension)].map(\.velocity))
        }
    }

    private static func linkState(
        _ body: GPUSolver.RigidBodyState, frame: MJCFLinkFrame
    ) -> GPUSolver.RigidBodyState {
        let offset = body.rotation.act(frame.position)
        return GPUSolver.RigidBodyState(
            position: body.position + offset,
            rotation: (body.rotation * frame.rotation).normalized,
            linearVelocity: body.linearVelocity
                + cross(body.angularVelocity, offset),
            angularVelocity: body.angularVelocity)
    }

    private static func offsetState(
        _ state: GPUSolver.RigidBodyState, localOffset: F3
    ) -> GPUSolver.RigidBodyState {
        let offset = state.rotation.act(localOffset)
        return GPUSolver.RigidBodyState(
            position: state.position + offset,
            rotation: state.rotation,
            linearVelocity: state.linearVelocity
                + cross(state.angularVelocity, offset),
            angularVelocity: state.angularVelocity)
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

/// Reference-free velocity-command task for Arachne. The policy observes only
/// quantities available from an IMU, joint encoders, the commanded twist, and
/// its previous applied action. No gait clock or authored trajectory is used.
public final class Arachne15LocomotionTask: VectorizedRLTask,
    RLEvaluationCriteriaProviding, PolicySymmetryProviding,
    ObservationNormalizerTransferProviding
{
    public static let observationDimension =
        Arachne15PolicyContract.observationDimension
    public let spec: RLTaskSpec
    public let environment: Arachne15Env
    public let configuration: Arachne15LocomotionTaskConfig

    public var usesPointGoal: Bool { configuration.pointGoal }

    public var evaluationCriteria: RLEvaluationCriteria {
        if configuration.pointGoal {
            var minimumTaskMetrics: [String: Float] = [
                "episode/survived": 0.95,
                "episode/goal_reached": 0.90,
                "episode/goal_front_success_rate": 0.85,
                "episode/goal_near_success_rate": 0.85,
                "episode/goal_far_success_rate": 0.85,
                // Keep a hard guard against gross tunnelling, then use the
                // time-resolved metrics below to reject sustained contact
                // exploitation without failing on one 2 ms impact sample.
                "episode/minimum_foot_collider_clearance_m": -0.003,
            ]
            if configuration.maximumGoalDirectionAngle > .pi / 4 {
                minimumTaskMetrics["episode/goal_left_success_rate"] = 0.85
                minimumTaskMetrics["episode/goal_right_success_rate"] = 0.85
            }
            if configuration.maximumGoalDirectionAngle > 3 * .pi / 4 {
                minimumTaskMetrics["episode/goal_rear_success_rate"] = 0.85
            }
            return RLEvaluationCriteria(
                minimumSuccessRate: 0.90,
                minimumMeanEpisodeLengthFraction: 0.10,
                minimumTaskMetrics: minimumTaskMetrics,
                maximumTaskMetrics: [
                    "episode/final_goal_distance_m":
                        configuration.goalRadius * 1.25,
                    "episode/minimum_goal_distance_m":
                        configuration.goalRadius,
                    "episode/yaw_rate_rmse_rps": 0.40,
                    "episode/foot_collider_penetration_rmse_m": 0.0005,
                    "episode/foot_collider_penetration_over_1mm_fraction":
                        0.025,
                ])
        }
        return RLEvaluationCriteria(
            minimumSuccessRate: 0.80,
            minimumMeanEpisodeLengthFraction: 0.90,
            minimumTaskMetrics: ["episode/survived": 0.90],
            maximumTaskMetrics: [
                "episode/linear_velocity_rmse_mps": 0.15,
                "episode/yaw_rate_rmse_rps": 0.40,
            ])
    }

    private let actionDimension = Arachne15Env.actionDimension
    private let historyDepth: Int
    private var actionHistory: ContiguousArray<Float>
    private var actionLatencies: [Int]
    private var previousActions: ContiguousArray<Float>
    private var commands: [F3]
    private var commandRNGs: [SplitMix64]
    private var noiseRNGs: [SplitMix64]
    private var episodeLengths: [Int]
    private var episodeReturns: [Float]
    private var linearSquaredErrorSums: [Float]
    private var yawSquaredErrorSums: [Float]
    private var goals: [F3]
    private var goalOrigins: [F3]
    private var goalOverrides: [(direction: F3, distance: Float)?]
    private var previousGoalDistances: [Float]
    private var minimumGoalDistances: [Float]
    private var initialGoalBearings: [Float]
    private var initialGoalDistances: [Float]
    private var goalDwellCounts: [Int]
    private var enteredGoals: [Bool]
    private var previousRootPositions: [F3]
    private var commandProgressSums: [Float]
    private var minimumFootColliderClearances: [Float]
    private var footPenetrationSquaredSums: [Float]
    private var footPenetrationOverOneMillimetreSteps: [Int]

    /// Transfer from the accepted straight walker preserves proprioceptive
    /// statistics, but that source policy saw constant forward and zero
    /// lateral/yaw commands. These floors keep newly introduced navigation
    /// commands in a sane normalized range from the first PPO update.
    public var initializationObservationVarianceFloors: [Int: Double] {
        [9: 0.01, 10: 0.01, 11: 0.16]
    }

    private static let mirroredJointSource = [
        8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7,
    ]
    private static let mirroredJointSign: [Float] = [
        -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1,
    ]

    public var policyActionMirrorSourceIndices: [Int] {
        Self.mirroredJointSource
    }

    public var policyActionMirrorSigns: [Float] {
        Self.mirroredJointSign
    }

    public func mirrorPolicyActions(
        _ actions: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        precondition(actions.count.isMultiple(of: actionDimension))
        var mirrored = actions
        for row in 0..<(actions.count / actionDimension) {
            let base = row * actionDimension
            for j in 0..<actionDimension {
                mirrored[base + j] = Self.mirroredJointSign[j]
                    * actions[base + Self.mirroredJointSource[j]]
            }
        }
        return mirrored
    }

    public func mirrorPolicyObservations(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        precondition(observations.count.isMultiple(
            of: Self.observationDimension))
        var mirrored = observations
        for row in 0..<(observations.count / Self.observationDimension) {
            let base = row * Self.observationDimension
            // Polar vectors: lateral component changes sign. Axial angular
            // velocity additionally flips roll and yaw under reflection.
            mirrored[base + 1] = -observations[base + 1]
            mirrored[base + 3] = -observations[base + 3]
            mirrored[base + 5] = -observations[base + 5]
            mirrored[base + 7] = -observations[base + 7]
            mirrored[base + 10] = -observations[base + 10]
            mirrored[base + 11] = -observations[base + 11]
            for tensorBase in [12, 28, 44] {
                for j in 0..<actionDimension {
                    mirrored[base + tensorBase + j] =
                        Self.mirroredJointSign[j]
                        * observations[base + tensorBase
                            + Self.mirroredJointSource[j]]
                }
            }
        }
        return mirrored
    }

    static func creditedCommandProgress(
        measured: Float, commandSpeed: Float, controlStep: Float
    ) -> Float {
        min(measured, commandSpeed * controlStep)
    }
    private var resetRNG: SplitMix64

    public init(
        configuration: Arachne15LocomotionTaskConfig,
        taskID: String = "arachne15-velocity-v0",
        includeInteractiveRobustnessProbe: Bool = false,
        projectileSize: Float = 0.05,
        projectileMass: Float = 0.10
    ) throws {
        guard (configuration.pointGoal && taskID == "arachne15-goal-v0")
                || (!configuration.pointGoal
                    && taskID == "arachne15-velocity-v0") else {
            throw RLEnvironmentError.invalidConfiguration(
                "Arachne task ID and point-goal mode disagree")
        }
        guard configuration.numEnvironments > 0,
              configuration.maxEpisodeSteps > 0,
              configuration.controlDecimation > 0,
              configuration.solverIterations > 0,
              configuration.commandResamplingSteps > 0,
              configuration.minimumForwardVelocity >= 0,
              configuration.maximumForwardVelocity
                >= configuration.minimumForwardVelocity,
              configuration.maximumLateralVelocity >= 0,
              configuration.maximumYawRate >= 0,
              (0...1).contains(configuration.standingCommandProbability),
              configuration.initialRollPitchRange >= 0,
              configuration.initialYawRange >= 0,
              configuration.initialYawRange <= .pi,
              configuration.maximumActionLatencySteps >= 0,
              !configuration.pointGoal || (
                configuration.minimumGoalDistance > configuration.goalRadius
                && configuration.maximumGoalDistance
                    >= configuration.minimumGoalDistance
                && configuration.maximumGoalDirectionAngle > 0
                && configuration.maximumGoalDirectionAngle <= .pi
                && configuration.goalRadius > 0
                && configuration.goalSlowdownDistance
                    > configuration.goalRadius
                && configuration.goalCommandSpeed > 0
                && configuration.goalBoundaryCommandSpeed >= 0
                && configuration.goalBoundaryCommandSpeed
                    <= configuration.goalCommandSpeed
                && configuration.maximumGoalArrivalSpeed >= 0
                && configuration.goalDwellSteps > 0
                && configuration.goalProgressRewardWeight >= 0
                && configuration.goalStableRewardWeight >= 0
                && configuration.goalSuccessBonus >= 0),
              configuration.commandProgressRewardWeight >= 0,
              configuration.velocityErrorPenaltyWeight >= 0,
              configuration.yawErrorPenaltyWeight >= 0,
              projectileSize.isFinite, projectileSize > 0,
              projectileMass.isFinite, projectileMass > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid Arachne locomotion configuration")
        }
        let env = try Arachne15Env(
            numEnvironments: configuration.numEnvironments,
            seed: configuration.seed,
            collisionProfile: configuration.collisionProfile,
            domainRandomization: configuration.domainRandomization,
            includeGoalMarkers: configuration.pointGoal,
            goalMarkerRadius: configuration.goalRadius,
            solverIterations: configuration.solverIterations,
            includeProjectiles: includeInteractiveRobustnessProbe,
            projectileSize: projectileSize,
            projectileMass: projectileMass)
        environment = env
        self.configuration = configuration
        let d = configuration.domainRandomization
        spec = RLTaskSpec(
            id: taskID,
            revision: RLPhysicsContract.deterministicColorSolveV1(6),
            numEnvironments: configuration.numEnvironments,
            observation: RLTensorSpec(
                name: "policy", shape: [Self.observationDimension]),
            action: RLTensorSpec(
                name: "joint_position_offset", shape: [actionDimension],
                lowerBound: [Float](repeating: -1, count: actionDimension),
                upperBound: [Float](repeating: 1, count: actionDimension)),
            maxEpisodeSteps: configuration.maxEpisodeSteps,
            simulationStep: env.scene.settings.dt,
            controlDecimation: configuration.controlDecimation,
            autoReset: configuration.autoReset,
            configurationValues: [
                "solverIterations": Float(configuration.solverIterations),
                "commandResamplingSteps": Float(configuration.commandResamplingSteps),
                "minimumForwardVelocity": configuration.minimumForwardVelocity,
                "maximumForwardVelocity": configuration.maximumForwardVelocity,
                "maximumLateralVelocity": configuration.maximumLateralVelocity,
                "maximumYawRate": configuration.maximumYawRate,
                "standingCommandProbability": configuration.standingCommandProbability,
                "initialRollPitchRange": configuration.initialRollPitchRange,
                "initialYawRange": configuration.initialYawRange,
                "observationNoise": configuration.observationNoise ? 1 : 0,
                "maximumActionLatencySteps": Float(
                    configuration.maximumActionLatencySteps),
                "validationCollisionProfile":
                    configuration.collisionProfile == .validation ? 1 : 0,
                "pointGoal": configuration.pointGoal ? 1 : 0,
                "minimumGoalDistance": configuration.minimumGoalDistance,
                "maximumGoalDistance": configuration.maximumGoalDistance,
                "maximumGoalDirectionAngle":
                    configuration.maximumGoalDirectionAngle,
                "goalRadius": configuration.goalRadius,
                "goalSlowdownDistance": configuration.goalSlowdownDistance,
                "goalCommandSpeed": configuration.goalCommandSpeed,
                "goalBoundaryCommandSpeed":
                    configuration.goalBoundaryCommandSpeed,
                "maximumGoalArrivalSpeed":
                    configuration.maximumGoalArrivalSpeed,
                "goalDwellSteps": Float(configuration.goalDwellSteps),
                "goalProgressRewardWeight":
                    configuration.goalProgressRewardWeight,
                "goalStableRewardWeight":
                    configuration.goalStableRewardWeight,
                "goalSuccessBonus": configuration.goalSuccessBonus,
                "commandProgressRewardWeight":
                    configuration.commandProgressRewardWeight,
                "velocityErrorPenaltyWeight":
                    configuration.velocityErrorPenaltyWeight,
                "yawErrorPenaltyWeight": configuration.yawErrorPenaltyWeight,
                "massScaleLower": d.mass.lowerBound,
                "massScaleUpper": d.mass.upperBound,
                "inertiaScaleLower": d.inertia.lowerBound,
                "inertiaScaleUpper": d.inertia.upperBound,
                "frictionScaleLower": d.friction.lowerBound,
                "frictionScaleUpper": d.friction.upperBound,
                "motorTorqueScaleLower": d.motorTorque.lowerBound,
                "motorTorqueScaleUpper": d.motorTorque.upperBound,
                "motorStiffnessScaleLower": d.motorStiffness.lowerBound,
                "motorStiffnessScaleUpper": d.motorStiffness.upperBound,
                "motorDampingScaleLower": d.motorDamping.lowerBound,
                "motorDampingScaleUpper": d.motorDamping.upperBound,
                "armatureScaleLower": d.armature.lowerBound,
                "armatureScaleUpper": d.armature.upperBound,
            ])
        let n = configuration.numEnvironments
        historyDepth = configuration.maximumActionLatencySteps + 1
        actionHistory = ContiguousArray(
            repeating: 0, count: n * historyDepth * actionDimension)
        actionLatencies = [Int](repeating: 0, count: n)
        previousActions = ContiguousArray(
            repeating: 0, count: n * actionDimension)
        commands = [F3](repeating: .zero, count: n)
        commandRNGs = (0..<n).map {
            SplitMix64(seed: configuration.seed &+ UInt64($0))
        }
        noiseRNGs = (0..<n).map {
            SplitMix64(seed: configuration.seed ^ UInt64($0)
                ^ 0xA0761D6478BD642F)
        }
        episodeLengths = [Int](repeating: 0, count: n)
        episodeReturns = [Float](repeating: 0, count: n)
        linearSquaredErrorSums = [Float](repeating: 0, count: n)
        yawSquaredErrorSums = [Float](repeating: 0, count: n)
        goals = [F3](repeating: .zero, count: n)
        goalOrigins = [F3](repeating: .zero, count: n)
        goalOverrides = [Optional<(direction: F3, distance: Float)>](
            repeating: nil, count: n)
        previousGoalDistances = [Float](repeating: 0, count: n)
        minimumGoalDistances = [Float](repeating: 0, count: n)
        initialGoalBearings = [Float](repeating: 0, count: n)
        initialGoalDistances = [Float](repeating: 0, count: n)
        goalDwellCounts = [Int](repeating: 0, count: n)
        enteredGoals = [Bool](repeating: false, count: n)
        previousRootPositions = [F3](repeating: .zero, count: n)
        commandProgressSums = [Float](repeating: 0, count: n)
        minimumFootColliderClearances = [Float](repeating: .infinity, count: n)
        footPenetrationSquaredSums = [Float](repeating: 0, count: n)
        footPenetrationOverOneMillimetreSteps = [Int](repeating: 0, count: n)
        resetRNG = SplitMix64(seed: configuration.seed ^ 0xE7037ED1A0B428DB)

        let ids = Array(0..<n)
        let seeds = ids.map { configuration.seed &+ UInt64($0) }
        env.reset(ids, seeds: seeds,
                  initialRollPitchRange: configuration.initialRollPitchRange,
                  initialYawRange: configuration.initialYawRange)
        initializeEpisodes(ids, seeds: seeds, states: env.states())
    }

    public func currentCommand(environment: Int) -> F3 {
        precondition(environment >= 0 && environment < spec.numEnvironments)
        return commands[environment]
    }

    public func currentGoalPosition(environment: Int) -> F3 {
        precondition(configuration.pointGoal)
        precondition((0..<spec.numEnvironments).contains(environment))
        return goals[environment]
    }

    public func currentGoalDistance(environment e: Int) -> Float {
        precondition(configuration.pointGoal)
        precondition((0..<spec.numEnvironments).contains(e))
        let root = environment.states()[e].root.position
        return planarGoalDistance(environment: e, rootPosition: root)
    }

    public func currentGoalDirection(environment e: Int) -> F3 {
        precondition(configuration.pointGoal)
        precondition((0..<spec.numEnvironments).contains(e))
        let root = environment.states()[e].root.position
        let delta = goals[e] - root
        let length = max(sqrt(delta.x * delta.x + delta.y * delta.y), 1e-6)
        return F3(delta.x / length, delta.y / length, 0)
    }

    /// Installs a world-space point goal for replay or deterministic tests.
    /// It changes the command source only; joint actions remain policy-owned.
    public func setGoal(environment e: Int, direction: F3,
                        distance: Float) throws {
        guard configuration.pointGoal,
              (0..<spec.numEnvironments).contains(e),
              direction.x.isFinite, direction.y.isFinite,
              distance.isFinite, distance > configuration.goalRadius else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid Arachne point-goal override")
        }
        let length = sqrt(direction.x * direction.x
            + direction.y * direction.y)
        guard length > 1e-6 else {
            throw RLEnvironmentError.invalidConfiguration(
                "Arachne point-goal direction must be planar and nonzero")
        }
        let normalized = F3(direction.x / length, direction.y / length, 0)
        let state = environment.states()[e]
        goalOverrides[e] = (normalized, distance)
        installGoal(environment: e, rootPosition: state.root.position,
                    direction: normalized, distance: distance)
        recordGoalCohort(environment: e, state: state,
                         direction: normalized, distance: distance)
        previousGoalDistances[e] = distance
        minimumGoalDistances[e] = distance
        goalDwellCounts[e] = 0
        enteredGoals[e] = false
        updateGoalCommand(environment: e, state: state)
        environment.setGoalMarkers(
            environmentIDs: [e], goals: [goals[e]], origins: [goalOrigins[e]])
    }

    public func clearGoalOverride(environment e: Int) {
        precondition(configuration.pointGoal)
        precondition((0..<spec.numEnvironments).contains(e))
        goalOverrides[e] = nil
    }

    /// Rebuild policy-facing history after an explicit commissioning motion
    /// without teleporting the physically settled robot. This is the handoff
    /// boundary from reveal trajectories back to learned or classical control.
    public func resumeAfterCommissioning(
        into observations: inout RLObservationBatch
    ) throws {
        try observations.validate(for: spec)
        let states = environment.states()
        let ids = Array(0..<spec.numEnvironments)
        for e in ids {
            episodeLengths[e] = 0
            episodeReturns[e] = 0
            linearSquaredErrorSums[e] = 0
            yawSquaredErrorSums[e] = 0
            previousRootPositions[e] = states[e].root.position
            commandProgressSums[e] = 0
            minimumFootColliderClearances[e] = .infinity
            footPenetrationSquaredSums[e] = 0
            footPenetrationOverOneMillimetreSteps[e] = 0
            let historyBase = e * historyDepth * actionDimension
            for i in 0..<(historyDepth * actionDimension) {
                actionHistory[historyBase + i] = 0
            }
            for j in 0..<actionDimension {
                previousActions[e * actionDimension + j] = 0
            }
            if configuration.pointGoal {
                let distance = planarGoalDistance(
                    environment: e, rootPosition: states[e].root.position)
                previousGoalDistances[e] = distance
                minimumGoalDistances[e] = distance
                goalDwellCounts[e] = 0
                enteredGoals[e] = distance <= configuration.goalRadius
                updateGoalCommand(environment: e, state: states[e])
            }
        }
        try fillObservations(states, into: &observations.policy)
        try observations.validate(for: spec)
    }

    public func reset(environments ids: [Int]?, seed: UInt64,
                      into observations: inout RLObservationBatch) throws {
        try observations.validate(for: spec)
        let envIDs = try checkedEnvironmentIDs(ids)
        try environment.solver.synchronize()
        let seeds = envIDs.map {
            seed &+ UInt64($0) &* 0x9E3779B97F4A7C15
        }
        environment.reset(
            envIDs, seeds: seeds,
            initialRollPitchRange: configuration.initialRollPitchRange,
            initialYawRange: configuration.initialYawRange)
        let states = environment.states()
        initializeEpisodes(envIDs, seeds: seeds, states: states)
        try fillObservations(states, into: &observations.policy)
        try observations.validate(for: spec)
    }

    public func step(actions: RLActionBatch,
                     into result: inout RLStepBatch) throws {
        try actions.validate(for: spec)
        try result.validate(for: spec)
        try environment.solver.synchronize()
        result.clearSignals()
        let applied = delayedActions(actions.values)
        try environment.stepChecked(
            actions: applied,
            decimation: configuration.controlDecimation)
        var states = environment.states()
        let contacts = environment.groundContacts()
        let n = spec.numEnvironments
        let dt = spec.controlStep
        var trackingLinear = ContiguousArray(repeating: Float(0), count: n)
        var trackingYaw = ContiguousArray(repeating: Float(0), count: n)
        var orientationCost = ContiguousArray(repeating: Float(0), count: n)
        var verticalVelocityCost = ContiguousArray(repeating: Float(0), count: n)
        var angularVelocityCost = ContiguousArray(repeating: Float(0), count: n)
        var yawErrorCost = ContiguousArray(repeating: Float(0), count: n)
        var actionRateCost = ContiguousArray(repeating: Float(0), count: n)
        var jointVelocityCost = ContiguousArray(repeating: Float(0), count: n)
        var footSlipCost = ContiguousArray(repeating: Float(0), count: n)
        var rootHeight = ContiguousArray(repeating: Float(0), count: n)
        var projectedGravityZ = ContiguousArray(repeating: Float(0), count: n)
        var feetInContact = ContiguousArray(repeating: Float(0), count: n)
        var minimumFootColliderClearance = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeReturnMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLengthMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeSurvivedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLinearRMSEMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeYawRMSEMetric = ContiguousArray(repeating: Float(0), count: n)
        var goalProgressReward = ContiguousArray(repeating: Float(0), count: n)
        var goalStableReward = ContiguousArray(repeating: Float(0), count: n)
        var commandProgressReward = ContiguousArray(
            repeating: Float(0), count: n)
        var commandProgressMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var goalDistanceMetric = ContiguousArray(repeating: Float(0), count: n)
        var minimumGoalDistanceMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var goalDwellMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeGoalReachedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalEnteredMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeFinalGoalDistanceMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMinimumGoalDistanceMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalDwellMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeArrivalSpeedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeCommandProgressMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMinimumFootColliderClearanceMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeFootPenetrationRMSEMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeFootPenetrationOverOneMillimetreFractionMetric =
            ContiguousArray(repeating: Float(0), count: n)
        var episodeGoalFrontBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalFrontSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalLeftBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalLeftSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRearBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRearSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRightBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalRightSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalNearBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalNearSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalFarBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeGoalFarSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var resetIDs: [Int] = []
        var resetSeeds: [UInt64] = []

        for e in 0..<n {
            let state = states[e]
            let qInverse = state.root.rotation.conjugate
            let localVelocity = qInverse.act(state.root.linearVelocity)
            let localAngularVelocity = qInverse.act(state.root.angularVelocity)
            let projectedGravity = qInverse.act(F3(0, 0, 1))
            let command = commands[e]
            let linearError = SIMD2<Float>(localVelocity.x - command.x,
                                           localVelocity.y - command.y)
            let linearErrorSquared = simd_length_squared(linearError)
            let yawError = localAngularVelocity.z - command.z
            yawErrorCost[e] = yawError * yawError
            let worldDisplacement = state.root.position
                - previousRootPositions[e]
            let localDisplacement = qInverse.act(worldDisplacement)
            let planarCommand = SIMD2<Float>(command.x, command.y)
            let planarCommandSpeed = simd_length(planarCommand)
            let commandProgress = planarCommandSpeed > 1e-5
                ? simd_dot(
                    SIMD2<Float>(localDisplacement.x, localDisplacement.y),
                    planarCommand / planarCommandSpeed)
                : 0
            previousRootPositions[e] = state.root.position
            commandProgressSums[e] += commandProgress
            commandProgressMetric[e] = commandProgress
            // Credit only the requested displacement for this control step.
            // Raw progress remains observable in metrics, but racing beyond
            // the commanded speed cannot increase return.
            let creditedCommandProgress = Self.creditedCommandProgress(
                measured: commandProgress,
                commandSpeed: planarCommandSpeed,
                controlStep: dt)
            commandProgressReward[e] =
                configuration.commandProgressRewardWeight
                    * creditedCommandProgress
            trackingLinear[e] = exp(-linearErrorSquared / 0.04)
            trackingYaw[e] = exp(-(yawError * yawError) / 0.25)
            orientationCost[e] = projectedGravity.x * projectedGravity.x
                + projectedGravity.y * projectedGravity.y
            verticalVelocityCost[e] = localVelocity.z * localVelocity.z
            angularVelocityCost[e] = localAngularVelocity.x
                * localAngularVelocity.x
                + localAngularVelocity.y * localAngularVelocity.y
            for j in 0..<actionDimension {
                let index = e * actionDimension + j
                let delta = applied[index] - previousActions[index]
                actionRateCost[e] += delta * delta
                jointVelocityCost[e] += state.jointVelocities[j]
                    * state.jointVelocities[j]
                previousActions[index] = applied[index]
            }
            for foot in 0..<8 where contacts[e][foot] {
                let v = state.feet[foot].linearVelocity
                footSlipCost[e] += v.x * v.x + v.y * v.y
            }
            rootHeight[e] = state.root.position.z
            projectedGravityZ[e] = projectedGravity.z
            feetInContact[e] = Float(contacts[e].filter { $0 }.count)
            minimumFootColliderClearance[e] = state.feet.map { foot in
                let half = Arachne15Env.footColliderHalfSize
                let verticalSupport = abs(foot.rotation.act(F3(1, 0, 0)).z)
                        * half.x
                    + abs(foot.rotation.act(F3(0, 1, 0)).z) * half.y
                    + abs(foot.rotation.act(F3(0, 0, 1)).z) * half.z
                return foot.position.z - verticalSupport
            }.min() ?? .infinity
            minimumFootColliderClearances[e] = min(
                minimumFootColliderClearances[e],
                minimumFootColliderClearance[e])
            let footPenetration = max(0, -minimumFootColliderClearance[e])
            footPenetrationSquaredSums[e] += footPenetration * footPenetration
            if footPenetration > 0.001 {
                footPenetrationOverOneMillimetreSteps[e] += 1
            }
            let fallen = state.root.position.z < 0.035
                || projectedGravity.z < 0.25
                || !state.root.position.x.isFinite
                || state.jointAngles.contains(where: { !$0.isFinite })
            let rewardRate = 1.5 * trackingLinear[e]
                + 0.5 * trackingYaw[e]
                + 0.2 * max(projectedGravity.z, 0)
                - 2.0 * orientationCost[e]
                - 0.2 * verticalVelocityCost[e]
                - 0.05 * angularVelocityCost[e]
                - 0.02 * actionRateCost[e]
                - 0.0005 * jointVelocityCost[e]
                - 0.10 * footSlipCost[e]
                - configuration.velocityErrorPenaltyWeight
                    * linearErrorSquared
                - configuration.yawErrorPenaltyWeight * yawErrorCost[e]
            var reachedGoal = false
            var goalDistance: Float = 0
            if configuration.pointGoal {
                goalDistance = planarGoalDistance(
                    environment: e, rootPosition: state.root.position)
                goalDistanceMetric[e] = goalDistance
                minimumGoalDistances[e] = min(
                    minimumGoalDistances[e], goalDistance)
                minimumGoalDistanceMetric[e] = minimumGoalDistances[e]
                goalProgressReward[e] = configuration.goalProgressRewardWeight
                    * (previousGoalDistances[e] - goalDistance)
                if goalDistance <= configuration.goalRadius {
                    enteredGoals[e] = true
                }
                let planarSpeed = sqrt(
                    state.root.linearVelocity.x * state.root.linearVelocity.x
                        + state.root.linearVelocity.y
                            * state.root.linearVelocity.y)
                let stable = goalDistance <= configuration.goalRadius
                    && planarSpeed <= configuration.maximumGoalArrivalSpeed
                goalDwellCounts[e] = stable ? goalDwellCounts[e] + 1 : 0
                goalDwellMetric[e] = Float(goalDwellCounts[e])
                goalStableReward[e] = stable
                    ? configuration.goalStableRewardWeight * dt : 0
                reachedGoal = goalDwellCounts[e]
                    >= configuration.goalDwellSteps
                previousGoalDistances[e] = goalDistance
            }
            result.rewards[e] = rewardRate * dt - (fallen ? 1 : 0)
                + commandProgressReward[e]
                + goalProgressReward[e] + goalStableReward[e]
                + (reachedGoal ? configuration.goalSuccessBonus : 0)
            episodeReturns[e] += result.rewards[e]
            linearSquaredErrorSums[e] += linearErrorSquared
            yawSquaredErrorSums[e] += yawError * yawError
            episodeLengths[e] += 1
            let timedOut = episodeLengths[e] >= configuration.maxEpisodeSteps
            if fallen || timedOut || reachedGoal {
                let inverseLength = 1 / Float(max(episodeLengths[e], 1))
                let linearRMSE = sqrt(linearSquaredErrorSums[e] * inverseLength)
                let yawRMSE = sqrt(yawSquaredErrorSums[e] * inverseLength)
                let success = configuration.pointGoal
                    ? reachedGoal && !fallen
                    : timedOut && !fallen
                        && linearRMSE <= 0.15 && yawRMSE <= 0.40
                result.terminated[e] = fallen || reachedGoal
                result.truncated[e] = !fallen && !reachedGoal && timedOut
                result.successes[e] = success
                result.hasFinalObservation[e] = true
                episodeReturnMetric[e] = episodeReturns[e]
                episodeLengthMetric[e] = Float(episodeLengths[e])
                episodeSurvivedMetric[e] = configuration.pointGoal
                    ? (!fallen ? 1 : 0)
                    : (timedOut && !fallen ? 1 : 0)
                episodeLinearRMSEMetric[e] = linearRMSE
                episodeYawRMSEMetric[e] = yawRMSE
                episodeCommandProgressMetric[e] = commandProgressSums[e]
                episodeMinimumFootColliderClearanceMetric[e] =
                    minimumFootColliderClearances[e]
                episodeFootPenetrationRMSEMetric[e] = sqrt(
                    footPenetrationSquaredSums[e] * inverseLength)
                episodeFootPenetrationOverOneMillimetreFractionMetric[e] =
                    Float(footPenetrationOverOneMillimetreSteps[e])
                        * inverseLength
                if configuration.pointGoal {
                    episodeGoalReachedMetric[e] = success ? 1 : 0
                    episodeGoalEnteredMetric[e] = enteredGoals[e] ? 1 : 0
                    episodeFinalGoalDistanceMetric[e] = goalDistance
                    episodeMinimumGoalDistanceMetric[e] =
                        minimumGoalDistances[e]
                    episodeGoalDwellMetric[e] = Float(goalDwellCounts[e])
                    episodeArrivalSpeedMetric[e] = sqrt(
                        state.root.linearVelocity.x
                            * state.root.linearVelocity.x
                        + state.root.linearVelocity.y
                            * state.root.linearVelocity.y)
                    let bearing = initialGoalBearings[e]
                    let absoluteBearing = abs(bearing)
                    if absoluteBearing <= .pi / 4 {
                        episodeGoalFrontBinMetric[e] = 1
                        episodeGoalFrontSuccessMetric[e] = success ? 1 : 0
                    } else if bearing > 0 && bearing < 3 * .pi / 4 {
                        episodeGoalLeftBinMetric[e] = 1
                        episodeGoalLeftSuccessMetric[e] = success ? 1 : 0
                    } else if bearing < 0 && bearing > -3 * .pi / 4 {
                        episodeGoalRightBinMetric[e] = 1
                        episodeGoalRightSuccessMetric[e] = success ? 1 : 0
                    } else {
                        episodeGoalRearBinMetric[e] = 1
                        episodeGoalRearSuccessMetric[e] = success ? 1 : 0
                    }
                    let distanceMidpoint = 0.5
                        * (configuration.minimumGoalDistance
                            + configuration.maximumGoalDistance)
                    if initialGoalDistances[e] <= distanceMidpoint {
                        episodeGoalNearBinMetric[e] = 1
                        episodeGoalNearSuccessMetric[e] = success ? 1 : 0
                    } else {
                        episodeGoalFarBinMetric[e] = 1
                        episodeGoalFarSuccessMetric[e] = success ? 1 : 0
                    }
                }
                if configuration.autoReset {
                    resetIDs.append(e)
                    resetSeeds.append(resetRNG.next())
                }
            } else if !configuration.pointGoal && episodeLengths[e]
                .isMultiple(of: configuration.commandResamplingSteps) {
                sampleCommand(environment: e)
            }
            if configuration.pointGoal && !fallen && !reachedGoal {
                updateGoalCommand(environment: e, state: state)
            }
        }

        try fillObservations(states, into: &result.observations.policy)
        for e in 0..<n where result.hasFinalObservation[e] {
            let base = e * Self.observationDimension
            for j in 0..<Self.observationDimension {
                result.finalObservations[base + j] =
                    result.observations.policy[base + j]
            }
        }
        result.metrics["reward/tracking_linear_velocity"] = trackingLinear
        result.metrics["reward/tracking_yaw_rate"] = trackingYaw
        result.metrics["reward/command_progress"] = commandProgressReward
        result.metrics["state/command_progress_m"] = commandProgressMetric
        result.metrics["penalty/orientation"] = orientationCost
        result.metrics["penalty/vertical_velocity"] = verticalVelocityCost
        result.metrics["penalty/angular_velocity_xy"] = angularVelocityCost
        result.metrics["penalty/yaw_rate_error"] = yawErrorCost
        result.metrics["penalty/action_rate"] = actionRateCost
        result.metrics["penalty/joint_velocity"] = jointVelocityCost
        result.metrics["penalty/foot_slip"] = footSlipCost
        result.metrics["state/root_height_m"] = rootHeight
        result.metrics["state/projected_gravity_z"] = projectedGravityZ
        result.metrics["state/feet_in_contact"] = feetInContact
        result.metrics["state/minimum_foot_collider_clearance_m"] =
            minimumFootColliderClearance
        result.metrics["episode/return"] = episodeReturnMetric
        result.metrics["episode/length"] = episodeLengthMetric
        result.metrics["episode/survived"] = episodeSurvivedMetric
        result.metrics["episode/linear_velocity_rmse_mps"] =
            episodeLinearRMSEMetric
        result.metrics["episode/yaw_rate_rmse_rps"] = episodeYawRMSEMetric
        result.metrics["episode/command_progress_m"] =
            episodeCommandProgressMetric
        result.metrics["episode/minimum_foot_collider_clearance_m"] =
            episodeMinimumFootColliderClearanceMetric
        result.metrics["episode/foot_collider_penetration_rmse_m"] =
            episodeFootPenetrationRMSEMetric
        result.metrics[
            "episode/foot_collider_penetration_over_1mm_fraction"] =
                episodeFootPenetrationOverOneMillimetreFractionMetric
        if configuration.pointGoal {
            result.metrics["reward/goal_progress"] = goalProgressReward
            result.metrics["reward/goal_stable"] = goalStableReward
            result.metrics["state/goal_distance_m"] = goalDistanceMetric
            result.metrics["state/minimum_goal_distance_m"] =
                minimumGoalDistanceMetric
            result.metrics["state/goal_dwell_steps"] = goalDwellMetric
            result.metrics["episode/goal_reached"] = episodeGoalReachedMetric
            result.metrics["episode/goal_entered"] = episodeGoalEnteredMetric
            result.metrics["episode/final_goal_distance_m"] =
                episodeFinalGoalDistanceMetric
            result.metrics["episode/minimum_goal_distance_m"] =
                episodeMinimumGoalDistanceMetric
            result.metrics["episode/goal_dwell_steps"] =
                episodeGoalDwellMetric
            result.metrics["episode/arrival_speed_mps"] =
                episodeArrivalSpeedMetric
            result.metrics["episode/goal_front_bin"] =
                episodeGoalFrontBinMetric
            result.metrics["episode/goal_front_success"] =
                episodeGoalFrontSuccessMetric
            result.metrics["episode/goal_left_bin"] =
                episodeGoalLeftBinMetric
            result.metrics["episode/goal_left_success"] =
                episodeGoalLeftSuccessMetric
            result.metrics["episode/goal_rear_bin"] =
                episodeGoalRearBinMetric
            result.metrics["episode/goal_rear_success"] =
                episodeGoalRearSuccessMetric
            result.metrics["episode/goal_right_bin"] =
                episodeGoalRightBinMetric
            result.metrics["episode/goal_right_success"] =
                episodeGoalRightSuccessMetric
            result.metrics["episode/goal_near_bin"] =
                episodeGoalNearBinMetric
            result.metrics["episode/goal_near_success"] =
                episodeGoalNearSuccessMetric
            result.metrics["episode/goal_far_bin"] =
                episodeGoalFarBinMetric
            result.metrics["episode/goal_far_success"] =
                episodeGoalFarSuccessMetric
        }

        if !resetIDs.isEmpty {
            environment.reset(
                resetIDs, seeds: resetSeeds,
                initialRollPitchRange: configuration.initialRollPitchRange,
                initialYawRange: configuration.initialYawRange)
            states = environment.states()
            initializeEpisodes(resetIDs, seeds: resetSeeds, states: states)
            try fillObservations(
                states, into: &result.observations.policy,
                environmentIDs: resetIDs)
        }
        try result.observations.validate(for: spec)
    }

    private func delayedActions(
        _ requested: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        var result = ContiguousArray(
            repeating: Float(0), count: requested.count)
        for e in 0..<spec.numEnvironments {
            let envBase = e * historyDepth * actionDimension
            if historyDepth > 1 {
                for age in stride(from: historyDepth - 1, through: 1, by: -1) {
                    for j in 0..<actionDimension {
                        actionHistory[envBase + age * actionDimension + j] =
                            actionHistory[envBase + (age - 1) * actionDimension + j]
                    }
                }
            }
            for j in 0..<actionDimension {
                actionHistory[envBase + j] = simd_clamp(
                    requested[e * actionDimension + j], -1, 1)
                result[e * actionDimension + j] = actionHistory[
                    envBase + actionLatencies[e] * actionDimension + j]
            }
        }
        return result
    }

    private func initializeEpisodes(_ ids: [Int], seeds: [UInt64],
                                    states: [Arachne15State]) {
        var markerGoals = [F3]()
        var markerOrigins = [F3]()
        markerGoals.reserveCapacity(ids.count)
        markerOrigins.reserveCapacity(ids.count)
        for (offset, e) in ids.enumerated() {
            commandRNGs[e] = SplitMix64(seed: seeds[offset]
                ^ 0xA0761D6478BD642F)
            noiseRNGs[e] = SplitMix64(seed: seeds[offset]
                ^ 0xE7037ED1A0B428DB)
            if configuration.maximumActionLatencySteps > 0 {
                actionLatencies[e] = Int(commandRNGs[e].next()
                    % UInt64(configuration.maximumActionLatencySteps + 1))
            } else {
                actionLatencies[e] = 0
            }
            if configuration.pointGoal {
                let direction: F3
                let distance: Float
                if let override = goalOverrides[e] {
                    direction = override.direction
                    distance = override.distance
                } else {
                    let angle = (2 * commandRNGs[e].nextFloat() - 1)
                        * configuration.maximumGoalDirectionAngle
                    let worldDirection = states[e].root.rotation.act(
                        F3(cos(angle), sin(angle), 0))
                    let planarLength = max(sqrt(
                        worldDirection.x * worldDirection.x
                            + worldDirection.y * worldDirection.y), 1e-6)
                    direction = F3(worldDirection.x / planarLength,
                                   worldDirection.y / planarLength, 0)
                    distance = configuration.minimumGoalDistance
                        + (configuration.maximumGoalDistance
                            - configuration.minimumGoalDistance)
                            * commandRNGs[e].nextFloat()
                }
                installGoal(
                    environment: e, rootPosition: states[e].root.position,
                    direction: direction, distance: distance)
                recordGoalCohort(environment: e, state: states[e],
                                 direction: direction, distance: distance)
                previousGoalDistances[e] = distance
                minimumGoalDistances[e] = distance
                goalDwellCounts[e] = 0
                enteredGoals[e] = false
                updateGoalCommand(environment: e, state: states[e])
                markerGoals.append(goals[e])
                markerOrigins.append(goalOrigins[e])
            } else {
                sampleCommand(environment: e)
            }
            episodeLengths[e] = 0
            episodeReturns[e] = 0
            linearSquaredErrorSums[e] = 0
            yawSquaredErrorSums[e] = 0
            previousRootPositions[e] = states[e].root.position
            commandProgressSums[e] = 0
            minimumFootColliderClearances[e] = .infinity
            footPenetrationSquaredSums[e] = 0
            footPenetrationOverOneMillimetreSteps[e] = 0
            let historyBase = e * historyDepth * actionDimension
            for i in 0..<(historyDepth * actionDimension) {
                actionHistory[historyBase + i] = 0
            }
            for j in 0..<actionDimension {
                previousActions[e * actionDimension + j] = 0
            }
        }
        if configuration.pointGoal {
            environment.setGoalMarkers(
                environmentIDs: ids, goals: markerGoals,
                origins: markerOrigins)
        }
    }

    private func sampleCommand(environment e: Int) {
        if commandRNGs[e].nextFloat()
            < configuration.standingCommandProbability {
            commands[e] = .zero
            return
        }
        commands[e] = F3(
            configuration.minimumForwardVelocity
                + (configuration.maximumForwardVelocity
                    - configuration.minimumForwardVelocity)
                    * commandRNGs[e].nextFloat(),
            (2 * commandRNGs[e].nextFloat() - 1)
                * configuration.maximumLateralVelocity,
            (2 * commandRNGs[e].nextFloat() - 1)
                * configuration.maximumYawRate)
    }

    private func installGoal(environment e: Int, rootPosition: F3,
                             direction: F3, distance: Float) {
        goalOrigins[e] = F3(rootPosition.x, rootPosition.y, 0)
        goals[e] = goalOrigins[e] + direction * distance
    }

    private func recordGoalCohort(environment e: Int, state: Arachne15State,
                                  direction: F3, distance: Float) {
        let localDirection = state.root.rotation.conjugate.act(direction)
        initialGoalBearings[e] = atan2(localDirection.y, localDirection.x)
        initialGoalDistances[e] = distance
    }

    private func planarGoalDistance(environment e: Int,
                                    rootPosition: F3) -> Float {
        let delta = goals[e] - rootPosition
        return sqrt(delta.x * delta.x + delta.y * delta.y)
    }

    /// Converts a world point into the same local twist interface used by the
    /// reusable velocity locomotion policy. Future vision can replace goal
    /// acquisition without changing observations 9...11 or the motor action.
    private func updateGoalCommand(environment e: Int,
                                   state: Arachne15State) {
        commands[e] = Arachne15PolicyContract.pointGoalCommand(
            worldGoal: goals[e], rootPosition: state.root.position,
            rootRotation: state.root.rotation,
            goalRadius: configuration.goalRadius,
            slowdownDistance: configuration.goalSlowdownDistance,
            cruiseSpeed: configuration.goalCommandSpeed,
            boundarySpeed: configuration.goalBoundaryCommandSpeed,
            maximumYawRate: configuration.maximumYawRate)
    }

    private func fillObservations(
        _ states: [Arachne15State],
        into output: inout ContiguousArray<Float>,
        environmentIDs: [Int]? = nil
    ) throws {
        func uniformNoise(_ rng: inout SplitMix64, scale: Float) -> Float {
            (2 * rng.nextFloat() - 1) * scale
        }
        for e in environmentIDs ?? Array(0..<spec.numEnvironments) {
            let state = states[e]
            let inverse = state.root.rotation.conjugate
            let localVelocity = inverse.act(state.root.linearVelocity)
            let localAngularVelocity = inverse.act(state.root.angularVelocity)
            let projectedGravity = inverse.act(F3(0, 0, 1))
            let base = e * Self.observationDimension
            var rng = noiseRNGs[e]
            let noisy = configuration.observationNoise
            let encoded = try Arachne15PolicyContract.encode(.init(
                bodyLinearVelocity: localVelocity,
                bodyAngularVelocity: localAngularVelocity,
                projectedGravity: projectedGravity,
                commandedBodyTwist: commands[e],
                jointPositions: state.jointAngles,
                jointVelocities: state.jointVelocities,
                previousActions: Array(previousActions[
                    (e * actionDimension)..<((e + 1) * actionDimension)])))
            for axis in 0..<3 {
                output[base + axis] = encoded[axis]
                    + (noisy ? uniformNoise(&rng, scale: 0.01) : 0)
                output[base + 3 + axis] = encoded[3 + axis]
                    + (noisy ? uniformNoise(&rng, scale: 0.05) : 0)
                output[base + 6 + axis] = encoded[6 + axis]
                    + (noisy ? uniformNoise(&rng, scale: 0.01) : 0)
                output[base + 9 + axis] = encoded[9 + axis]
            }
            for j in 0..<actionDimension {
                output[base + 12 + j] = encoded[12 + j]
                    + (noisy ? uniformNoise(&rng, scale: 0.01) : 0)
                output[base + 28 + j] = encoded[28 + j]
                    + (noisy ? uniformNoise(&rng, scale: 0.10) : 0)
                output[base + 44 + j] = encoded[44 + j]
            }
            noiseRNGs[e] = rng
        }
    }
}
