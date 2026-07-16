import simd

public struct Arachne15LocomotionTaskConfig: Sendable {
    public var numEnvironments: Int
    public var seed: UInt64
    public var maxEpisodeSteps: Int
    public var controlDecimation: Int
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
    public var autoReset: Bool

    public init(
        numEnvironments: Int, seed: UInt64 = 1,
        maxEpisodeSteps: Int = 1_000, controlDecimation: Int = 10,
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
        autoReset: Bool = true
    ) {
        self.numEnvironments = numEnvironments
        self.seed = seed
        self.maxEpisodeSteps = maxEpisodeSteps
        self.controlDecimation = controlDecimation
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
    }

    public static let actionDimension = 16
    public static let footLinkOffset = F3(0.105, 0, 0)
    public static let actionScales: [Float] = (0..<actionDimension).map {
        $0.isMultiple(of: 2) ? 0.35 : 0.45
    }

    public let numEnvironments: Int
    public let solver: GPUSolver
    public let scene: PhysicsScene
    public let groundBody: Int
    public let dynamicsScales: [MJCFDynamicsScale]
    public private(set) var refs: [EnvRefs]

    private let spawnPoses: [(F3, Quat)]
    private let groundContactSlots: [Int: Int]

    public init(
        numEnvironments: Int, seed: UInt64 = 1,
        collisionProfile: Arachne15CollisionProfile = .training,
        domainRandomization: ArticulationDomainRandomization = .init(),
        solverIterations: Int = 20
    ) throws {
        precondition(numEnvironments > 0 && solverIterations > 0)
        self.numEnvironments = numEnvironments
        var built = PhysicsScene(name: "arachne15-locomotion")
        built.settings.dt = 0.002
        built.settings.gravity = -9.80665
        built.settings.iterations = solverIterations
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
                defaultMotorGain: .init(stiffness: 2.0, damping: 0.08),
                selfCollisions: false,
                collisionGroup: UInt32(e + 1),
                dynamicsScale: scale,
                includeVisuals: numEnvironments <= 4)
            let bodies = asset.bodyNames.map { imported.bodiesByName[$0]! }
            let feet = footNames.map { imported.bodiesByName[$0]! }
            let footFrames = footNames.map { imported.linkFramesInBody[$0]! }
            builtRefs.append(EnvRefs(
                root: imported.rootBody,
                rootFrame: imported.linkFramesInBody["base"]!,
                bodies: bodies,
                motors: imported.actuatorJoints,
                feet: feet,
                footFrames: footFrames))
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
        solver = try GPUSolver(scene: built)
    }

    public func step(actions: ContiguousArray<Float>, decimation: Int) {
        precondition(actions.count == numEnvironments * Self.actionDimension)
        var targets: [GPUSolver.MotorTargetUpdate] = []
        targets.reserveCapacity(actions.count)
        for e in 0..<numEnvironments {
            for j in 0..<Self.actionDimension {
                let joint = refs[e].motors[j]
                let requested = simd_clamp(actions[e * Self.actionDimension + j],
                                           -1, 1)
                    * Self.actionScales[j]
                targets.append(.init(
                    joint: joint,
                    angle: simd_clamp(requested,
                                      scene.joints[joint].limitLo,
                                      scene.joints[joint].limitHi)))
            }
        }
        solver.setMotorTargets(targets)
        for _ in 0..<decimation { solver.step() }
    }

    public func reset(_ environmentIDs: [Int], seeds: [UInt64],
                      initialRollPitchRange: Float = 0,
                      initialYawRange: Float = 0) {
        precondition(environmentIDs.count == seeds.count)
        var poses: [GPUSolver.BodyPoseUpdate] = []
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
            for body in refs[e].bodies {
                let spawn = spawnPoses[body]
                poses.append(.init(
                    body: body,
                    position: pivot + perturbation.act(spawn.0 - pivot),
                    rotation: (perturbation * spawn.1).normalized))
            }
            for joint in refs[e].motors {
                motors.append(.init(joint: joint, angle: 0))
            }
        }
        solver.setBodyPoses(poses)
        solver.setMotorTargets(motors)
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
}

/// Reference-free velocity-command task for Arachne. The policy observes only
/// quantities available from an IMU, joint encoders, the commanded twist, and
/// its previous applied action. No gait clock or authored trajectory is used.
public final class Arachne15LocomotionTask: VectorizedRLTask,
    RLEvaluationCriteriaProviding
{
    public static let observationDimension = 60
    public let spec: RLTaskSpec
    public let environment: Arachne15Env
    public let configuration: Arachne15LocomotionTaskConfig

    public var evaluationCriteria: RLEvaluationCriteria {
        RLEvaluationCriteria(
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
    private var resetRNG: SplitMix64

    public init(configuration: Arachne15LocomotionTaskConfig) throws {
        guard configuration.numEnvironments > 0,
              configuration.maxEpisodeSteps > 0,
              configuration.controlDecimation > 0,
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
              configuration.maximumActionLatencySteps >= 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid Arachne locomotion configuration")
        }
        let env = try Arachne15Env(
            numEnvironments: configuration.numEnvironments,
            seed: configuration.seed,
            collisionProfile: configuration.collisionProfile,
            domainRandomization: configuration.domainRandomization)
        environment = env
        self.configuration = configuration
        let d = configuration.domainRandomization
        spec = RLTaskSpec(
            id: "arachne15-velocity-v0", revision: 1,
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
        resetRNG = SplitMix64(seed: configuration.seed ^ 0xE7037ED1A0B428DB)

        let ids = Array(0..<n)
        let seeds = ids.map { configuration.seed &+ UInt64($0) }
        env.reset(ids, seeds: seeds,
                  initialRollPitchRange: configuration.initialRollPitchRange,
                  initialYawRange: configuration.initialYawRange)
        initializeEpisodes(ids, seeds: seeds)
    }

    public func currentCommand(environment: Int) -> F3 {
        precondition(environment >= 0 && environment < spec.numEnvironments)
        return commands[environment]
    }

    public func reset(environments ids: [Int]?, seed: UInt64,
                      into observations: inout RLObservationBatch) throws {
        try observations.validate(for: spec)
        let envIDs = try checkedEnvironmentIDs(ids)
        let seeds = envIDs.map {
            seed &+ UInt64($0) &* 0x9E3779B97F4A7C15
        }
        environment.reset(
            envIDs, seeds: seeds,
            initialRollPitchRange: configuration.initialRollPitchRange,
            initialYawRange: configuration.initialYawRange)
        initializeEpisodes(envIDs, seeds: seeds)
        fillObservations(environment.states(), into: &observations.policy)
        try observations.validate(for: spec)
    }

    public func step(actions: RLActionBatch,
                     into result: inout RLStepBatch) throws {
        try actions.validate(for: spec)
        try result.validate(for: spec)
        result.clearSignals()
        let applied = delayedActions(actions.values)
        environment.step(actions: applied,
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
        var actionRateCost = ContiguousArray(repeating: Float(0), count: n)
        var jointVelocityCost = ContiguousArray(repeating: Float(0), count: n)
        var footSlipCost = ContiguousArray(repeating: Float(0), count: n)
        var rootHeight = ContiguousArray(repeating: Float(0), count: n)
        var projectedGravityZ = ContiguousArray(repeating: Float(0), count: n)
        var feetInContact = ContiguousArray(repeating: Float(0), count: n)
        var episodeReturnMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLengthMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeSurvivedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLinearRMSEMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeYawRMSEMetric = ContiguousArray(repeating: Float(0), count: n)
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
            result.rewards[e] = rewardRate * dt - (fallen ? 1 : 0)
            episodeReturns[e] += result.rewards[e]
            linearSquaredErrorSums[e] += linearErrorSquared
            yawSquaredErrorSums[e] += yawError * yawError
            episodeLengths[e] += 1
            let timedOut = episodeLengths[e] >= configuration.maxEpisodeSteps
            if fallen || timedOut {
                let inverseLength = 1 / Float(max(episodeLengths[e], 1))
                let linearRMSE = sqrt(linearSquaredErrorSums[e] * inverseLength)
                let yawRMSE = sqrt(yawSquaredErrorSums[e] * inverseLength)
                result.terminated[e] = fallen
                result.truncated[e] = !fallen && timedOut
                result.successes[e] = timedOut && !fallen
                    && linearRMSE <= 0.15 && yawRMSE <= 0.40
                result.hasFinalObservation[e] = true
                episodeReturnMetric[e] = episodeReturns[e]
                episodeLengthMetric[e] = Float(episodeLengths[e])
                episodeSurvivedMetric[e] = timedOut && !fallen ? 1 : 0
                episodeLinearRMSEMetric[e] = linearRMSE
                episodeYawRMSEMetric[e] = yawRMSE
                if configuration.autoReset {
                    resetIDs.append(e)
                    resetSeeds.append(resetRNG.next())
                }
            } else if episodeLengths[e]
                .isMultiple(of: configuration.commandResamplingSteps) {
                sampleCommand(environment: e)
            }
        }

        fillObservations(states, into: &result.observations.policy)
        for e in 0..<n where result.hasFinalObservation[e] {
            let base = e * Self.observationDimension
            for j in 0..<Self.observationDimension {
                result.finalObservations[base + j] =
                    result.observations.policy[base + j]
            }
        }
        result.metrics["reward/tracking_linear_velocity"] = trackingLinear
        result.metrics["reward/tracking_yaw_rate"] = trackingYaw
        result.metrics["penalty/orientation"] = orientationCost
        result.metrics["penalty/vertical_velocity"] = verticalVelocityCost
        result.metrics["penalty/angular_velocity_xy"] = angularVelocityCost
        result.metrics["penalty/action_rate"] = actionRateCost
        result.metrics["penalty/joint_velocity"] = jointVelocityCost
        result.metrics["penalty/foot_slip"] = footSlipCost
        result.metrics["state/root_height_m"] = rootHeight
        result.metrics["state/projected_gravity_z"] = projectedGravityZ
        result.metrics["state/feet_in_contact"] = feetInContact
        result.metrics["episode/return"] = episodeReturnMetric
        result.metrics["episode/length"] = episodeLengthMetric
        result.metrics["episode/survived"] = episodeSurvivedMetric
        result.metrics["episode/linear_velocity_rmse_mps"] =
            episodeLinearRMSEMetric
        result.metrics["episode/yaw_rate_rmse_rps"] = episodeYawRMSEMetric

        if !resetIDs.isEmpty {
            environment.reset(
                resetIDs, seeds: resetSeeds,
                initialRollPitchRange: configuration.initialRollPitchRange,
                initialYawRange: configuration.initialYawRange)
            initializeEpisodes(resetIDs, seeds: resetSeeds)
            states = environment.states()
            fillObservations(
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

    private func initializeEpisodes(_ ids: [Int], seeds: [UInt64]) {
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
            sampleCommand(environment: e)
            episodeLengths[e] = 0
            episodeReturns[e] = 0
            linearSquaredErrorSums[e] = 0
            yawSquaredErrorSums[e] = 0
            let historyBase = e * historyDepth * actionDimension
            for i in 0..<(historyDepth * actionDimension) {
                actionHistory[historyBase + i] = 0
            }
            for j in 0..<actionDimension {
                previousActions[e * actionDimension + j] = 0
            }
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

    private func fillObservations(
        _ states: [Arachne15State],
        into output: inout ContiguousArray<Float>,
        environmentIDs: [Int]? = nil
    ) {
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
            for axis in 0..<3 {
                output[base + axis] = localVelocity[axis]
                    + (noisy ? uniformNoise(&rng, scale: 0.01) : 0)
                output[base + 3 + axis] = localAngularVelocity[axis]
                    + (noisy ? uniformNoise(&rng, scale: 0.05) : 0)
                output[base + 6 + axis] = projectedGravity[axis]
                    + (noisy ? uniformNoise(&rng, scale: 0.01) : 0)
                output[base + 9 + axis] = commands[e][axis]
            }
            for j in 0..<actionDimension {
                output[base + 12 + j] = state.jointAngles[j]
                    + (noisy ? uniformNoise(&rng, scale: 0.01) : 0)
                output[base + 28 + j] = state.jointVelocities[j]
                    + (noisy ? uniformNoise(&rng, scale: 0.10) : 0)
                output[base + 44 + j] = previousActions[
                    e * actionDimension + j]
            }
            noiseRNGs[e] = rng
        }
    }
}
