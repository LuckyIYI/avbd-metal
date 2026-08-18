import simd

/// Reference-free H1 joystick locomotion matching the public MuJoCo
/// Playground task boundary: a single actor tracks a body-frame
/// `[forward velocity, lateral velocity, yaw rate]` command. Commands change
/// inside an episode, including an exact-zero standing cohort, so braking and
/// balance are learned transitions rather than separately routed policies.
public struct HumanoidVelocityTaskConfig: Sendable {
    public var numEnvironments: Int
    public var seed: UInt64
    public var maxEpisodeSteps: Int
    public var controlDecimation: Int
    public var minimumForwardVelocity: Float
    public var maximumForwardVelocity: Float
    public var minimumLateralVelocity: Float
    public var maximumLateralVelocity: Float
    public var minimumYawRate: Float
    public var maximumYawRate: Float
    public var commandResamplingSteps: Int
    public var standingCommandProbability: Float
    public var initialRollPitchRange: Float
    public var initialYawRange: Float
    public var autoReset: Bool

    public init(
        numEnvironments: Int, seed: UInt64 = 1,
        maxEpisodeSteps: Int = 1_000, controlDecimation: Int = 5,
        minimumForwardVelocity: Float = -0.6,
        maximumForwardVelocity: Float = 1.5,
        minimumLateralVelocity: Float = -0.8,
        maximumLateralVelocity: Float = 0.8,
        minimumYawRate: Float = -0.7,
        maximumYawRate: Float = 0.7,
        commandResamplingSteps: Int = 500,
        standingCommandProbability: Float = 0.10,
        initialRollPitchRange: Float = 0.015,
        initialYawRange: Float = .pi,
        autoReset: Bool = true
    ) {
        self.numEnvironments = numEnvironments
        self.seed = seed
        self.maxEpisodeSteps = maxEpisodeSteps
        self.controlDecimation = controlDecimation
        self.minimumForwardVelocity = minimumForwardVelocity
        self.maximumForwardVelocity = maximumForwardVelocity
        self.minimumLateralVelocity = minimumLateralVelocity
        self.maximumLateralVelocity = maximumLateralVelocity
        self.minimumYawRate = minimumYawRate
        self.maximumYawRate = maximumYawRate
        self.commandResamplingSteps = commandResamplingSteps
        self.standingCommandProbability = standingCommandProbability
        self.initialRollPitchRange = initialRollPitchRange
        self.initialYawRange = initialYawRange
        self.autoReset = autoReset
    }
}

public final class HumanoidVelocityTask: VectorizedRLTask,
    RLEvaluationCriteriaProviding, PolicySymmetryProviding,
    ObservationNormalizerTransferProviding
{
    /// MuJoCo Playground H1 Joystick uses 15 histories of a 45-value frame:
    /// yaw rate (1), projected gravity (3), twist command (3), joint
    /// positions (19), and previous action (19).
    public static let observationFrameDimension = 45
    public static let observationHistorySteps = 15
    public static let observationDimension = observationFrameDimension
        * observationHistorySteps
    private static let velocityWindowSteps = 5
    private static let trackingVariance: Float = 0.25

    public let spec: RLTaskSpec
    public let environment: HumanoidWalkEnv
    public let configuration: HumanoidVelocityTaskConfig

    public var evaluationCriteria: RLEvaluationCriteria {
        RLEvaluationCriteria(
            minimumSuccessRate: 0.80,
            minimumMeanEpisodeLengthFraction: 0.90,
            minimumTaskMetrics: [
                "episode/survived": 0.90,
                "episode/command_switches": 1,
            ],
            maximumTaskMetrics: [
                "episode/linear_velocity_rmse_mps": 0.35,
                "episode/yaw_rate_rmse_rps": 0.50,
                "episode/standing_speed_mps": 0.15,
            ])
    }

    /// Command curricula intentionally move these channels outside the source
    /// checkpoint's narrow range. Preserve learned proprioceptive statistics
    /// while preventing the imported normalizer from magnifying a new twist
    /// command into an extreme policy input.
    public var initializationObservationVarianceFloors: [Int: Double] {
        var floors = [Int: Double]()
        for history in 0..<Self.observationHistorySteps {
            let base = history * Self.observationFrameDimension
            floors[base + 4] = 0.25
            floors[base + 5] = 0.25
            floors[base + 6] = 0.25
        }
        return floors
    }

    private let actionDimension = HumanoidWalkEnv.jointRanges.count
    private var commands: [F3]
    private var commandRNGs: [SplitMix64]
    private var previousActions: ContiguousArray<Float>
    private var observationHistory: [[Float]]
    private var observationHistoryInitialized: [Bool]
    private var previousRootPositions: [F3]
    private var rootPositionHistory: [[F3]]
    private var rootHistoryIndex = 0
    private var measuredRootVelocities: [F3]
    private var previousFootPositions: [[F3]]
    private var episodeLengths: [Int]
    private var episodeReturns: [Float]
    private var linearSquaredErrorSums: [Float]
    private var yawSquaredErrorSums: [Float]
    private var forwardVelocitySums: [Float]
    private var lateralVelocitySums: [Float]
    private var commandedForwardVelocitySums: [Float]
    private var forwardPathSums: [Float]
    private var standingSpeedSums: [Float]
    private var standingStepCounts: [Int]
    private var commandSwitchCounts: [Int]
    private var resetRNG: SplitMix64

    public init(configuration: HumanoidVelocityTaskConfig) throws {
        guard configuration.numEnvironments > 0,
              configuration.maxEpisodeSteps > 0,
              configuration.controlDecimation > 0,
              configuration.minimumForwardVelocity
                <= configuration.maximumForwardVelocity,
              configuration.minimumLateralVelocity
                <= configuration.maximumLateralVelocity,
              configuration.minimumYawRate <= configuration.maximumYawRate,
              configuration.commandResamplingSteps > 0,
              configuration.commandResamplingSteps
                < configuration.maxEpisodeSteps,
              (0...1).contains(configuration.standingCommandProbability),
              configuration.initialRollPitchRange >= 0,
              configuration.initialRollPitchRange <= 0.10,
              configuration.initialYawRange >= 0,
              configuration.initialYawRange <= .pi else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid humanoid velocity-command configuration")
        }
        let env = try HumanoidWalkEnv(
            numEnvironments: configuration.numEnvironments,
            seed: configuration.seed,
            controlProfile: .mujocoPlayground)
        environment = env
        self.configuration = configuration
        spec = RLTaskSpec(
            id: "humanoid-velocity-v0",
            revision: RLPhysicsContract.deterministicColorSolveV1(2),
            numEnvironments: configuration.numEnvironments,
            observation: RLTensorSpec(
                name: "policy", shape: [Self.observationDimension]),
            action: RLTensorSpec(
                name: "joint_position", shape: [actionDimension],
                lowerBound: [Float](repeating: -1, count: actionDimension),
                upperBound: [Float](repeating: 1, count: actionDimension)),
            maxEpisodeSteps: configuration.maxEpisodeSteps,
            simulationStep: env.scene.settings.dt,
            controlDecimation: configuration.controlDecimation,
            autoReset: configuration.autoReset,
            configurationValues: [
                "minimumForwardVelocity":
                    configuration.minimumForwardVelocity,
                "maximumForwardVelocity":
                    configuration.maximumForwardVelocity,
                "minimumLateralVelocity":
                    configuration.minimumLateralVelocity,
                "maximumLateralVelocity":
                    configuration.maximumLateralVelocity,
                "minimumYawRate": configuration.minimumYawRate,
                "maximumYawRate": configuration.maximumYawRate,
                "commandResamplingSteps":
                    Float(configuration.commandResamplingSteps),
                "standingCommandProbability":
                    configuration.standingCommandProbability,
                "initialRollPitchRange":
                    configuration.initialRollPitchRange,
                "initialYawRange": configuration.initialYawRange,
            ])

        let n = configuration.numEnvironments
        commands = [F3](repeating: .zero, count: n)
        commandRNGs = (0..<n).map {
            SplitMix64(seed: configuration.seed
                &+ UInt64($0) &* 0x9E3779B97F4A7C15)
        }
        previousActions = ContiguousArray(
            repeating: 0, count: n * actionDimension)
        observationHistory = [[Float]](
            repeating: [Float](
                repeating: 0, count: Self.observationDimension),
            count: n)
        observationHistoryInitialized = [Bool](repeating: false, count: n)
        previousRootPositions = [F3](repeating: .zero, count: n)
        rootPositionHistory = [[F3]](
            repeating: [F3](repeating: .zero,
                            count: Self.velocityWindowSteps),
            count: n)
        measuredRootVelocities = [F3](repeating: .zero, count: n)
        previousFootPositions = [[F3]](
            repeating: [.zero, .zero], count: n)
        episodeLengths = [Int](repeating: 0, count: n)
        episodeReturns = [Float](repeating: 0, count: n)
        linearSquaredErrorSums = [Float](repeating: 0, count: n)
        yawSquaredErrorSums = [Float](repeating: 0, count: n)
        forwardVelocitySums = [Float](repeating: 0, count: n)
        lateralVelocitySums = [Float](repeating: 0, count: n)
        commandedForwardVelocitySums = [Float](repeating: 0, count: n)
        forwardPathSums = [Float](repeating: 0, count: n)
        standingSpeedSums = [Float](repeating: 0, count: n)
        standingStepCounts = [Int](repeating: 0, count: n)
        commandSwitchCounts = [Int](repeating: 0, count: n)
        resetRNG = SplitMix64(seed: configuration.seed
            &+ 0xD1B54A32D192ED03)

        let ids = Array(0..<n)
        let initialSeeds = ids.map { configuration.seed &+ UInt64($0) }
        // A newly constructed task must begin through the same reset path as
        // every later episode. Besides enforcing the vector-task contract,
        // this clears solver warm starts and makes the first rollout
        // dynamically identical to all subsequent rollouts.
        env.reset(
            ids, seeds: initialSeeds,
            initialRollPitchRange: configuration.initialRollPitchRange,
            initialYawRange: configuration.initialYawRange)
        let states = env.states()
        initializeEpisodes(
            ids, states: states,
            seeds: initialSeeds)
    }

    public func currentCommand(environment: Int) -> F3 {
        precondition(environment >= 0 && environment < spec.numEnvironments)
        return commands[environment]
    }

    public func reset(
        environments ids: [Int]?, seed: UInt64,
        into observations: inout RLObservationBatch
    ) throws {
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
        initializeEpisodes(envIDs, states: states, seeds: seeds)
        fillObservations(
            states, into: &observations.policy,
            advancingEnvironmentIDs: envIDs)
        try observations.validate(for: spec)
    }

    public func step(
        actions: RLActionBatch, into result: inout RLStepBatch
    ) throws {
        try result.validate(for: spec)
        try actions.validate(for: spec)
        try environment.solver.synchronize()
        result.clearSignals()
        let n = spec.numEnvironments
        let dt = spec.controlStep
        var actionRate = ContiguousArray(repeating: Float(0), count: n)
        for e in 0..<n {
            let base = e * actionDimension
            for j in 0..<actionDimension {
                let action = simd_clamp(actions.values[base + j], -1, 1)
                let delta = action - previousActions[base + j]
                actionRate[e] += delta * delta
            }
        }

        try environment.stepChecked(
            normalizedActions: actions.values,
            decimation: configuration.controlDecimation)
        var states = environment.states()
        updateMeasuredRootVelocities(states)
        let contacts = environment.groundContacts()

        var trackingLinear = ContiguousArray(repeating: Float(0), count: n)
        var trackingYaw = ContiguousArray(repeating: Float(0), count: n)
        var verticalVelocityCost = ContiguousArray(
            repeating: Float(0), count: n)
        var angularVelocityCost = ContiguousArray(
            repeating: Float(0), count: n)
        var orientationCost = ContiguousArray(repeating: Float(0), count: n)
        var standStillCost = ContiguousArray(repeating: Float(0), count: n)
        var feetSlipCost = ContiguousArray(repeating: Float(0), count: n)
        var feetClearanceCost = ContiguousArray(
            repeating: Float(0), count: n)
        var runningReward = ContiguousArray(repeating: Float(0), count: n)
        var torsoHeight = ContiguousArray(repeating: Float(0), count: n)
        var projectedGravityZ = ContiguousArray(repeating: Float(0), count: n)
        var minimumFootClearance = ContiguousArray(
            repeating: Float(0), count: n)
        var feetInContact = ContiguousArray(repeating: Float(0), count: n)
        var minimumJointLimitMargin = ContiguousArray(
            repeating: Float.greatestFiniteMagnitude, count: n)
        var episodeReturnMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeLengthMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeSurvivedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeLinearRMSEMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeYawRMSEMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeForwardVelocityMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeLateralVelocityMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeCommandedForwardVelocityMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeForwardPathMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeStandingSpeedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeCommandSwitchesMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var resetIDs = [Int]()
        var resetSeeds = [UInt64]()

        for e in 0..<n {
            let state = states[e]
            let heading = Self.horizontalHeading(state.torso.rotation)
            let lateral = F3(-heading.y, heading.x, 0)
            let velocity = measuredRootVelocities[e]
            let localVelocity = F3(
                simd_dot(velocity, heading),
                simd_dot(velocity, lateral), velocity.z)
            let localAngularVelocity = state.torso.rotation.conjugate.act(
                state.torso.angularVelocity)
            let projectedGravity = state.torso.rotation.conjugate.act(
                F3(0, 0, 1))
            let command = commands[e]
            let linearErrorSquared =
                (localVelocity.x - command.x)
                    * (localVelocity.x - command.x)
                + (localVelocity.y - command.y)
                    * (localVelocity.y - command.y)
            let yawError = localAngularVelocity.z - command.z
            trackingLinear[e] = exp(
                -linearErrorSquared / Self.trackingVariance)
            trackingYaw[e] = exp(
                -(yawError * yawError) / Self.trackingVariance)
            verticalVelocityCost[e] = localVelocity.z * localVelocity.z
            angularVelocityCost[e] = localAngularVelocity.x
                * localAngularVelocity.x
                + localAngularVelocity.y * localAngularVelocity.y
            orientationCost[e] = projectedGravity.x * projectedGravity.x
                + projectedGravity.y * projectedGravity.y

            let planarCommand = sqrt(
                command.x * command.x + command.y * command.y)
            if planarCommand < 0.10 && abs(command.z) < 0.10 {
                for angle in state.jointAngles {
                    standStillCost[e] += abs(angle)
                }
                standingSpeedSums[e] += sqrt(
                    localVelocity.x * localVelocity.x
                        + localVelocity.y * localVelocity.y)
                standingStepCounts[e] += 1
            }

            let feet = [state.leftFoot, state.rightFoot]
            for foot in 0..<2 {
                let footVelocity = (feet[foot].position
                    - previousFootPositions[e][foot]) / dt
                let horizontalSpeedSquared = footVelocity.x * footVelocity.x
                    + footVelocity.y * footVelocity.y
                if contacts.feet[e][foot] {
                    feetSlipCost[e] += horizontalSpeedSquared
                }
                let clearance = HumanoidWalkTask.footGroundClearance(
                    feet[foot])
                let clearanceError = clearance
                    - HumanoidLocomotionObjective.targetFootClearance
                feetClearanceCost[e] += clearanceError * clearanceError
                    * sqrt(sqrt(max(horizontalSpeedSquared, 0)))
                previousFootPositions[e][foot] = feet[foot].position
            }

            let outsideJointLimits = zip(
                state.jointAngles, environment.refs[e].motors
            ).contains { angle, joint in
                angle < environment.scene.joints[joint].limitLo
                    || angle > environment.scene.joints[joint].limitHi
            }
            torsoHeight[e] = state.torso.position.z
            projectedGravityZ[e] = projectedGravity.z
            minimumFootClearance[e] = min(
                HumanoidWalkTask.footGroundClearance(state.leftFoot),
                HumanoidWalkTask.footGroundClearance(state.rightFoot))
            feetInContact[e] = Float(contacts.feet[e].filter { $0 }.count)
            minimumJointLimitMargin[e] = zip(
                state.jointAngles, environment.refs[e].motors
            ).reduce(Float.greatestFiniteMagnitude) { margin, pair in
                let limits = environment.scene.joints[pair.1]
                return min(margin, min(pair.0 - limits.limitLo,
                                       limits.limitHi - pair.0))
            }
            // MuJoCo Playground H1 termination contract. Its training model
            // enables terrain collision only for the feet, so torso contact
            // is intentionally not a separate termination route.
            let fallen = state.torso.position.z < 0.92
                || projectedGravity.z < 0
                || outsideJointLimits
                || !state.root.position.x.isFinite
            // Coherent MuJoCo Playground H1 reward family. Every scale is
            // integrated by the 20 ms control step and the summed running
            // reward is clipped at zero exactly as in the public task.
            let rewardRate = 1.5 * trackingLinear[e]
                + 0.8 * trackingYaw[e]
                - 2.0 * verticalVelocityCost[e]
                - 0.05 * angularVelocityCost[e]
                - 5.0 * orientationCost[e]
                - 0.2 * actionRate[e]
                - 0.5 * standStillCost[e]
                - 0.1 * feetSlipCost[e]
                - 0.5 * feetClearanceCost[e]
                - (fallen ? 1 : 0)
            runningReward[e] = max(rewardRate * dt, 0)
            result.rewards[e] = runningReward[e]
            episodeReturns[e] += result.rewards[e]
            linearSquaredErrorSums[e] += linearErrorSquared
            yawSquaredErrorSums[e] += yawError * yawError
            forwardVelocitySums[e] += localVelocity.x
            lateralVelocitySums[e] += localVelocity.y
            commandedForwardVelocitySums[e] += command.x
            forwardPathSums[e] += localVelocity.x * dt
            episodeLengths[e] += 1

            let timedOut = episodeLengths[e]
                >= configuration.maxEpisodeSteps
            let completed = fallen || timedOut
            if completed {
                let inverseLength = 1 / Float(max(episodeLengths[e], 1))
                let linearRMSE = sqrt(
                    linearSquaredErrorSums[e] * inverseLength)
                let yawRMSE = sqrt(yawSquaredErrorSums[e] * inverseLength)
                let standingSpeed = standingStepCounts[e] > 0
                    ? standingSpeedSums[e]
                        / Float(standingStepCounts[e])
                    : 0
                let success = timedOut && !fallen
                    && linearRMSE <= 0.35
                    && yawRMSE <= 0.50
                    && (standingStepCounts[e] == 0
                        || standingSpeed <= 0.15)
                result.terminated[e] = fallen
                result.truncated[e] = !fallen && timedOut
                result.successes[e] = success
                result.hasFinalObservation[e] = true
                episodeReturnMetric[e] = episodeReturns[e]
                episodeLengthMetric[e] = Float(episodeLengths[e])
                episodeSurvivedMetric[e] = timedOut && !fallen ? 1 : 0
                episodeLinearRMSEMetric[e] = linearRMSE
                episodeYawRMSEMetric[e] = yawRMSE
                episodeForwardVelocityMetric[e] =
                    forwardVelocitySums[e] * inverseLength
                episodeLateralVelocityMetric[e] =
                    lateralVelocitySums[e] * inverseLength
                episodeCommandedForwardVelocityMetric[e] =
                    commandedForwardVelocitySums[e] * inverseLength
                episodeForwardPathMetric[e] = forwardPathSums[e]
                episodeStandingSpeedMetric[e] = standingSpeed
                episodeCommandSwitchesMetric[e] =
                    Float(commandSwitchCounts[e])
                if configuration.autoReset {
                    resetIDs.append(e)
                    resetSeeds.append(resetRNG.next())
                }
            } else if episodeLengths[e]
                .isMultiple(of: configuration.commandResamplingSteps) {
                sampleCommand(environment: e)
                commandSwitchCounts[e] += 1
            }
        }

        for e in 0..<n {
            let base = e * actionDimension
            for j in 0..<actionDimension {
                previousActions[base + j] = simd_clamp(
                    actions.values[base + j], -1, 1)
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
        result.metrics["reward/running"] = runningReward
        result.metrics["state/torso_height_m"] = torsoHeight
        result.metrics["state/projected_gravity_z"] = projectedGravityZ
        result.metrics["state/minimum_foot_clearance_m"] =
            minimumFootClearance
        result.metrics["state/feet_in_contact"] = feetInContact
        result.metrics["state/minimum_joint_limit_margin_rad"] =
            minimumJointLimitMargin
        result.metrics["penalty/vertical_velocity"] = verticalVelocityCost
        result.metrics["penalty/angular_velocity_xy"] = angularVelocityCost
        result.metrics["penalty/orientation"] = orientationCost
        result.metrics["penalty/action_rate"] = actionRate
        result.metrics["penalty/stand_still"] = standStillCost
        result.metrics["penalty/feet_slip"] = feetSlipCost
        result.metrics["penalty/feet_clearance"] = feetClearanceCost
        result.metrics["episode/return"] = episodeReturnMetric
        result.metrics["episode/length"] = episodeLengthMetric
        result.metrics["episode/survived"] = episodeSurvivedMetric
        result.metrics["episode/linear_velocity_rmse_mps"] =
            episodeLinearRMSEMetric
        result.metrics["episode/yaw_rate_rmse_rps"] = episodeYawRMSEMetric
        result.metrics["episode/mean_local_forward_velocity_mps"] =
            episodeForwardVelocityMetric
        result.metrics["episode/mean_local_lateral_velocity_mps"] =
            episodeLateralVelocityMetric
        result.metrics["episode/mean_commanded_forward_velocity_mps"] =
            episodeCommandedForwardVelocityMetric
        result.metrics["episode/forward_path_m"] = episodeForwardPathMetric
        result.metrics["episode/standing_speed_mps"] =
            episodeStandingSpeedMetric
        result.metrics["episode/command_switches"] =
            episodeCommandSwitchesMetric

        if !resetIDs.isEmpty {
            environment.reset(
                resetIDs, seeds: resetSeeds,
                initialRollPitchRange: configuration.initialRollPitchRange,
                initialYawRange: configuration.initialYawRange)
            states = environment.states()
            initializeEpisodes(resetIDs, states: states, seeds: resetSeeds)
            fillObservations(
                states, into: &result.observations.policy,
                advancingEnvironmentIDs: resetIDs)
        }
        try result.observations.validate(for: spec)
    }

    private func initializeEpisodes(
        _ ids: [Int], states: [HumanoidState], seeds: [UInt64]
    ) {
        precondition(ids.count == seeds.count)
        for (offset, e) in ids.enumerated() {
            commandRNGs[e] = SplitMix64(seed: seeds[offset]
                ^ 0xA0761D6478BD642F)
            sampleCommand(environment: e)
            let root = states[e].root.position
            previousRootPositions[e] = root
            rootPositionHistory[e] = [F3](
                repeating: root, count: Self.velocityWindowSteps)
            measuredRootVelocities[e] = .zero
            previousFootPositions[e] = [
                states[e].leftFoot.position, states[e].rightFoot.position,
            ]
            for j in 0..<actionDimension {
                previousActions[e * actionDimension + j] = 0
            }
            observationHistoryInitialized[e] = false
            episodeLengths[e] = 0
            episodeReturns[e] = 0
            linearSquaredErrorSums[e] = 0
            yawSquaredErrorSums[e] = 0
            forwardVelocitySums[e] = 0
            lateralVelocitySums[e] = 0
            commandedForwardVelocitySums[e] = 0
            forwardPathSums[e] = 0
            standingSpeedSums[e] = 0
            standingStepCounts[e] = 0
            commandSwitchCounts[e] = 0
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
            configuration.minimumLateralVelocity
                + (configuration.maximumLateralVelocity
                    - configuration.minimumLateralVelocity)
                    * commandRNGs[e].nextFloat(),
            configuration.minimumYawRate
                + (configuration.maximumYawRate
                    - configuration.minimumYawRate)
                    * commandRNGs[e].nextFloat())
    }

    private func updateMeasuredRootVelocities(_ states: [HumanoidState]) {
        for e in 0..<spec.numEnvironments {
            let oldPosition = rootPositionHistory[e][rootHistoryIndex]
            measuredRootVelocities[e] =
                (states[e].root.position - oldPosition)
                / (Float(Self.velocityWindowSteps) * spec.controlStep)
            rootPositionHistory[e][rootHistoryIndex] =
                states[e].root.position
            previousRootPositions[e] = states[e].root.position
        }
        rootHistoryIndex = (rootHistoryIndex + 1)
            % Self.velocityWindowSteps
    }

    private func fillObservations(
        _ states: [HumanoidState],
        into output: inout ContiguousArray<Float>,
        advancingEnvironmentIDs: [Int]? = nil
    ) {
        var shouldAdvance = [Bool](
            repeating: advancingEnvironmentIDs == nil,
            count: spec.numEnvironments)
        if let ids = advancingEnvironmentIDs {
            for e in ids { shouldAdvance[e] = true }
        }
        for e in 0..<spec.numEnvironments {
            if shouldAdvance[e] || !observationHistoryInitialized[e] {
                let state = states[e]
                let projectedGravity = state.torso.rotation.conjugate.act(
                    F3(0, 0, 1))
                let localAngularVelocity =
                    state.torso.rotation.conjugate.act(
                        state.torso.angularVelocity)
                var frame = [Float](
                    repeating: 0, count: Self.observationFrameDimension)
                frame[0] = localAngularVelocity.z
                frame[1] = projectedGravity.x
                frame[2] = projectedGravity.y
                frame[3] = projectedGravity.z
                frame[4] = commands[e].x
                frame[5] = commands[e].y
                frame[6] = commands[e].z
                for j in 0..<actionDimension {
                    frame[7 + j] = state.jointAngles[j]
                    frame[26 + j] = previousActions[
                        e * actionDimension + j]
                }
                if observationHistoryInitialized[e] {
                    for history in stride(
                        from: Self.observationHistorySteps - 1,
                        through: 1, by: -1
                    ) {
                        let destination = history
                            * Self.observationFrameDimension
                        let source = (history - 1)
                            * Self.observationFrameDimension
                        for j in 0..<Self.observationFrameDimension {
                            observationHistory[e][destination + j] =
                                observationHistory[e][source + j]
                        }
                    }
                } else {
                    for history in 0..<Self.observationHistorySteps {
                        let base = history
                            * Self.observationFrameDimension
                        for j in 0..<Self.observationFrameDimension {
                            observationHistory[e][base + j] = frame[j]
                        }
                    }
                }
                for j in 0..<Self.observationFrameDimension {
                    observationHistory[e][j] = frame[j]
                }
                observationHistoryInitialized[e] = true
            }
            let outputBase = e * Self.observationDimension
            for j in 0..<Self.observationDimension {
                output[outputBase + j] = observationHistory[e][j]
            }
        }
    }

    private static func horizontalHeading(_ rotation: Quat) -> F3 {
        let forward = rotation.act(F3(1, 0, 0))
        let magnitude = sqrt(forward.x * forward.x
            + forward.y * forward.y)
        guard magnitude > 1e-6 else { return F3(1, 0, 0) }
        return F3(forward.x / magnitude, forward.y / magnitude, 0)
    }

    private static let mirroredJointSource = [
        5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10,
        15, 16, 17, 18, 11, 12, 13, 14,
    ]
    private static let mirroredJointSign: [Float] = [
        -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1,
        1, -1, -1, 1, 1, -1, -1, 1,
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
            let rowBase = row * Self.observationDimension
            for history in 0..<Self.observationHistorySteps {
                let base = rowBase
                    + history * Self.observationFrameDimension
                mirrored[base] = -observations[base]
                mirrored[base + 2] = -observations[base + 2]
                mirrored[base + 5] = -observations[base + 5]
                mirrored[base + 6] = -observations[base + 6]
                for tensorBase in [7, 26] {
                    for j in 0..<actionDimension {
                        mirrored[base + tensorBase + j] =
                            Self.mirroredJointSign[j]
                            * observations[base + tensorBase
                                + Self.mirroredJointSource[j]]
                    }
                }
            }
        }
        return mirrored
    }
}
