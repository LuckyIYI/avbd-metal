import simd
import SimCore
import PhysicsAVBD
import Robotics

/// Maintained, reference-free Isaac Lab H1 flat-velocity contract.
///
/// The task intentionally stays separate from the archived MuJoCo H1
/// joystick port: its 69-value observation, 5 ms / decimation-4 dynamics,
/// actuator profile, reward family, and torso-contact termination are one
/// coherent public baseline rather than a mixture of two simulators.
public struct HumanoidIsaacVelocityTaskConfig: Sendable {
    public var numEnvironments: Int
    public var seed: UInt64
    public var maxEpisodeSteps: Int
    public var commandResamplingSteps: Int
    public var standingCommandProbability: Float
    public var initialYawRange: Float
    public var observationNoise: Bool
    public var solverIterations: Int
    public var autoReset: Bool
    public var pointGoal: Bool
    public var minimumGoalDistance: Float
    public var maximumGoalDistance: Float
    public var goalRadius: Float
    public var goalSlowdownDistance: Float
    public var goalCommandSpeed: Float
    public var goalBoundaryCommandSpeed: Float
    public var goalDwellSteps: Int
    public var maximumGoalArrivalSpeed: Float
    public var goalProgressRewardWeight: Float
    public var goalStableRewardWeight: Float
    public var goalSuccessBonus: Float
    /// Fraction of point-goal episodes receiving one real rigid-box impact.
    /// Zero omits projectile bodies entirely from the scene.
    public var projectileProbability: Float
    public var projectileCurriculumControlSteps: Int
    public var projectileSize: Float
    public var projectileMass: Float
    public var minimumProjectileSpeed: Float
    public var maximumProjectileSpeed: Float
    /// Probability that a scheduled projectile launches from the side
    /// reported as `episode/projectile_left_*`. The default keeps evaluation
    /// balanced; training curricula may oversample a measured weak side.
    public var projectileLeftProbability: Float
    public var minimumProjectileLaunchStep: Int
    public var maximumProjectileLaunchStep: Int
    /// Route post-impact states to an independent recovery actor. The base
    /// locomotion policy remains active before measured contact.
    public var recoveryGatedActor: Bool
    public var freezeBasePolicyExpert: Bool
    /// Append measured impact side and elapsed post-contact time to the goal
    /// observation. These make routed recovery Markov without changing the
    /// transferred base actor before contact.
    public var recoveryContextObservations: Bool
    public var recoveryContextDuration: Float
    /// Zero routes every impact to the recovery expert. -1 or +1 trains and
    /// evaluates a side-specific specialist while the opposite side keeps
    /// using the frozen base actor.
    public var recoveryExpertSide: Float
    public var recoveryExpertGatePeak: Float
    public var recoveryExpertGateDecay: Bool
    public var initializeRecoveryExpertFromBaseOnTransfer: Bool
    public var initializeRecoveryExpertFromMirroredBaseOnTransfer: Bool
    public var postImpactUprightRewardWeight: Float
    public var postImpactAngularVelocityPenaltyWeight: Float
    public var postImpactFallPenalty: Float

    public init(numEnvironments: Int, seed: UInt64 = 1,
                maxEpisodeSteps: Int = 1_000,
                commandResamplingSteps: Int = 500,
                standingCommandProbability: Float = 0.02,
                initialYawRange: Float = .pi,
                observationNoise: Bool = true,
                solverIterations: Int = 20,
                autoReset: Bool = true,
                pointGoal: Bool = false,
                minimumGoalDistance: Float = 4,
                maximumGoalDistance: Float = 8,
                goalRadius: Float = 0.75,
                goalSlowdownDistance: Float = 2.5,
                goalCommandSpeed: Float = 0.55,
                goalBoundaryCommandSpeed: Float = 0.15,
                goalDwellSteps: Int = 25,
                maximumGoalArrivalSpeed: Float = 0.25,
                goalProgressRewardWeight: Float = 1,
                goalStableRewardWeight: Float = 2,
                goalSuccessBonus: Float = 5,
                projectileProbability: Float = 0,
                projectileCurriculumControlSteps: Int = 0,
                projectileSize: Float = 0.25,
                projectileMass: Float = 8,
                minimumProjectileSpeed: Float = 4,
                maximumProjectileSpeed: Float = 6,
                projectileLeftProbability: Float = 0.5,
                minimumProjectileLaunchStep: Int = 100,
                maximumProjectileLaunchStep: Int = 300,
                recoveryGatedActor: Bool = false,
                freezeBasePolicyExpert: Bool = true,
                recoveryContextObservations: Bool = false,
                recoveryContextDuration: Float = 2,
                recoveryExpertSide: Float = 0,
                recoveryExpertGatePeak: Float = 1,
                recoveryExpertGateDecay: Bool = false,
                initializeRecoveryExpertFromBaseOnTransfer: Bool = true,
                initializeRecoveryExpertFromMirroredBaseOnTransfer: Bool = false,
                postImpactUprightRewardWeight: Float = 0,
                postImpactAngularVelocityPenaltyWeight: Float = 0,
                postImpactFallPenalty: Float = 0) {
        self.numEnvironments = numEnvironments
        self.seed = seed
        self.maxEpisodeSteps = maxEpisodeSteps
        self.commandResamplingSteps = commandResamplingSteps
        self.standingCommandProbability = standingCommandProbability
        self.initialYawRange = initialYawRange
        self.observationNoise = observationNoise
        self.solverIterations = solverIterations
        self.autoReset = autoReset
        self.pointGoal = pointGoal
        self.minimumGoalDistance = minimumGoalDistance
        self.maximumGoalDistance = maximumGoalDistance
        self.goalRadius = goalRadius
        self.goalSlowdownDistance = goalSlowdownDistance
        self.goalCommandSpeed = goalCommandSpeed
        self.goalBoundaryCommandSpeed = goalBoundaryCommandSpeed
        self.goalDwellSteps = goalDwellSteps
        self.maximumGoalArrivalSpeed = maximumGoalArrivalSpeed
        self.goalProgressRewardWeight = goalProgressRewardWeight
        self.goalStableRewardWeight = goalStableRewardWeight
        self.goalSuccessBonus = goalSuccessBonus
        self.projectileProbability = projectileProbability
        self.projectileCurriculumControlSteps =
            projectileCurriculumControlSteps
        self.projectileSize = projectileSize
        self.projectileMass = projectileMass
        self.minimumProjectileSpeed = minimumProjectileSpeed
        self.maximumProjectileSpeed = maximumProjectileSpeed
        self.projectileLeftProbability = projectileLeftProbability
        self.minimumProjectileLaunchStep = minimumProjectileLaunchStep
        self.maximumProjectileLaunchStep = maximumProjectileLaunchStep
        self.recoveryGatedActor = recoveryGatedActor
        self.freezeBasePolicyExpert = freezeBasePolicyExpert
        self.recoveryContextObservations = recoveryContextObservations
        self.recoveryContextDuration = recoveryContextDuration
        self.recoveryExpertSide = recoveryExpertSide
        self.recoveryExpertGatePeak = recoveryExpertGatePeak
        self.recoveryExpertGateDecay = recoveryExpertGateDecay
        self.initializeRecoveryExpertFromBaseOnTransfer =
            initializeRecoveryExpertFromBaseOnTransfer
        self.initializeRecoveryExpertFromMirroredBaseOnTransfer =
            initializeRecoveryExpertFromMirroredBaseOnTransfer
        self.postImpactUprightRewardWeight = postImpactUprightRewardWeight
        self.postImpactAngularVelocityPenaltyWeight =
            postImpactAngularVelocityPenaltyWeight
        self.postImpactFallPenalty = postImpactFallPenalty
    }
}

public final class HumanoidIsaacVelocityTask: VectorizedRLTask,
    RLEvaluationCriteriaProviding, PolicySymmetryProviding,
    TrainingModeConfigurable, ObservationSchemaTransferProviding,
    PolicyReferenceRegularizationProviding, PolicyExpertGateProviding
{
    public static let observationDimension = 69
    public static let goalObservationDimension = 71
    public static let recoveryGoalObservationDimension = 73
    private static let trackingVariance: Float = 0.25
    private static let controlDecimation = 4
    private static let footAirTimeThreshold: Float = 0.6

    public let spec: RLTaskSpec
    public let environment: HumanoidWalkEnv
    public let configuration: HumanoidIsaacVelocityTaskConfig

    public var evaluationCriteria: RLEvaluationCriteria {
        if configuration.pointGoal {
            var minimumTaskMetrics: [String: Float] = [
                "episode/goal_reached": 0.80,
                "episode/survived": 0.90,
            ]
            var minimumSuccessRate: Float = 0.80
            if configuration.projectileProbability > 0 {
                minimumSuccessRate = 0.70
                minimumTaskMetrics["episode/goal_reached"] = 0.70
                minimumTaskMetrics["episode/survived"] = 0.80
                minimumTaskMetrics["episode/disturbed_success_rate"] = 0.70
                minimumTaskMetrics["episode/projectile_launched"] =
                    0.90 * configuration.projectileProbability
                minimumTaskMetrics["episode/projectile_impacted"] =
                    0.90 * configuration.projectileProbability
                if configuration.projectileProbability < 1 {
                    minimumTaskMetrics["episode/nominal_success_rate"] = 0.80
                }
            }
            return RLEvaluationCriteria(
                minimumSuccessRate: minimumSuccessRate,
                minimumTaskMetrics: minimumTaskMetrics,
                maximumTaskMetrics: [
                    "episode/final_goal_distance_m":
                        configuration.goalRadius * 1.5,
                    "episode/minimum_goal_distance_m":
                        configuration.goalRadius,
                ])
        }
        return RLEvaluationCriteria(
            minimumSuccessRate: 0.80,
            minimumMeanEpisodeLengthFraction: 0.90,
            minimumTaskMetrics: ["episode/survived": 0.90],
            maximumTaskMetrics: [
                "episode/linear_velocity_rmse_mps": 0.35,
                "episode/yaw_rate_rmse_rps": 0.50,
            ])
    }

    public var usesPointGoal: Bool { configuration.pointGoal }

    private let actionDimension = HumanoidWalkEnv.jointRanges.count
    private var trainingMode = false
    private var trainingControlSteps = 0
    private var commands: [F3]
    private var targetHeadings: [Float]
    private var goals: [F3]
    private var goalOverrides: [(direction: F3, distance: Float)?]
    /// Planar world-space origin of the current velocity-command segment.
    /// This is render-only bookkeeping: it never enters observations,
    /// rewards, termination, or policy inputs.
    private var commandProjectionOrigins: [F3]
    private var commandRNGs: [SplitMix64]
    private var noiseRNGs: [SplitMix64]
    private var previousActions: ContiguousArray<Float>
    private var previousJointVelocities: [[Float]]
    private var jointVelocities: [[Float]]
    private var contactTimes: [[Float]]
    private var airTimes: [[Float]]
    private var episodeLengths: [Int]
    private var episodeReturns: [Float]
    private var linearSquaredErrorSums: [Float]
    private var yawSquaredErrorSums: [Float]
    private var forwardVelocitySums: [Float]
    private var commandedForwardVelocitySums: [Float]
    private var forwardPathSums: [Float]
    private var previousGoalDistances: [Float]
    private var minimumGoalDistances: [Float]
    private var goalDwellCounts: [Int]
    private var enteredGoals: [Bool]
    private var projectileLaunchSteps: [Int]
    private var projectileSpeeds: [Float]
    private var projectileSides: [Float]
    private var projectileLaunched: [Bool]
    private var projectileImpacted: [Bool]
    private var projectileImpactSteps: [Int]
    private var minimumPostImpactUprightCosines: [Float]
    private var minimumPostImpactTorsoHeights: [Float]
    private var disturbedEpisodes: [Bool]
    private var resetRNG: SplitMix64

    public init(
        configuration: HumanoidIsaacVelocityTaskConfig,
        taskID: String = "humanoid-isaac-flat-v0",
        includeInteractiveRobustnessProbe: Bool = false
    ) throws {
        guard configuration.numEnvironments > 0,
              configuration.maxEpisodeSteps > 0,
              configuration.commandResamplingSteps > 0,
              configuration.commandResamplingSteps
                < configuration.maxEpisodeSteps,
              (0...1).contains(configuration.standingCommandProbability),
              configuration.initialYawRange >= 0,
              configuration.initialYawRange <= .pi,
              configuration.solverIterations > 0,
              !configuration.pointGoal || (
                taskID == "humanoid-isaac-goal-v0"
                && configuration.minimumGoalDistance > 0
                && configuration.maximumGoalDistance
                    >= configuration.minimumGoalDistance
                && configuration.goalRadius > 0
                && configuration.goalSlowdownDistance
                    > configuration.goalRadius
                && configuration.goalCommandSpeed > 0
                && configuration.goalBoundaryCommandSpeed >= 0
                && configuration.goalBoundaryCommandSpeed
                    <= configuration.goalCommandSpeed
                && configuration.goalDwellSteps > 0
                && configuration.maximumGoalArrivalSpeed > 0
                && configuration.goalProgressRewardWeight >= 0
                && configuration.goalStableRewardWeight >= 0
                && configuration.goalSuccessBonus >= 0
                && (0...1).contains(configuration.projectileProbability)
                && configuration.projectileCurriculumControlSteps >= 0
                && configuration.projectileSize > 0
                && configuration.projectileMass > 0
                && configuration.minimumProjectileSpeed > 0
                && configuration.maximumProjectileSpeed
                    >= configuration.minimumProjectileSpeed
                && (0...1).contains(configuration.projectileLeftProbability)
                && configuration.minimumProjectileLaunchStep >= 0
                && configuration.maximumProjectileLaunchStep
                    >= configuration.minimumProjectileLaunchStep
                && (configuration.projectileProbability == 0
                    || configuration.maximumProjectileLaunchStep
                        < configuration.maxEpisodeSteps)
                && (!configuration.recoveryGatedActor
                    || configuration.projectileProbability > 0)
                && (!configuration.recoveryContextObservations
                    || (configuration.recoveryGatedActor
                        && configuration.recoveryContextDuration > 0
                        && configuration.recoveryContextDuration.isFinite))
                && [-1, 0, 1].contains(configuration.recoveryExpertSide)
                && (configuration.recoveryExpertSide == 0
                    || configuration.recoveryGatedActor)
                && (0...1).contains(configuration.recoveryExpertGatePeak)
                && configuration.recoveryExpertGatePeak.isFinite
                && (!configuration.recoveryExpertGateDecay
                    || configuration.recoveryContextObservations)
                && (configuration.initializeRecoveryExpertFromBaseOnTransfer
                    || configuration.recoveryGatedActor)
                && (!configuration
                    .initializeRecoveryExpertFromMirroredBaseOnTransfer
                    || configuration.recoveryGatedActor)
                && !(configuration.initializeRecoveryExpertFromBaseOnTransfer
                    && configuration
                        .initializeRecoveryExpertFromMirroredBaseOnTransfer)
                && configuration.postImpactUprightRewardWeight >= 0
                && configuration.postImpactUprightRewardWeight.isFinite
                && configuration.postImpactAngularVelocityPenaltyWeight >= 0
                && configuration.postImpactAngularVelocityPenaltyWeight.isFinite
                && configuration.postImpactFallPenalty >= 0
                && configuration.postImpactFallPenalty.isFinite) else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid Isaac H1 flat-velocity configuration")
        }
        let env = try HumanoidWalkEnv(
            numEnvironments: configuration.numEnvironments,
            seed: configuration.seed,
            includeProjectile: configuration.projectileProbability > 0
                || includeInteractiveRobustnessProbe,
            projectileSize: configuration.projectileSize,
            projectileMass: configuration.projectileMass,
            preserveMinimalTerrainContactProfile:
                includeInteractiveRobustnessProbe
                    && configuration.projectileProbability == 0,
            controlProfile: .isaacLab,
            solverIterations: configuration.solverIterations)
        environment = env
        self.configuration = configuration
        var configurationValues: [String: Float] = [
            "commandResamplingSteps":
                Float(configuration.commandResamplingSteps),
            "standingCommandProbability":
                configuration.standingCommandProbability,
            "initialYawRange": configuration.initialYawRange,
            "observationNoise": configuration.observationNoise ? 1 : 0,
            "solverIterations": Float(configuration.solverIterations),
        ]
        if configuration.pointGoal {
            configurationValues.merge([
                "pointGoal": 1,
                "minimumGoalDistance": configuration.minimumGoalDistance,
                "maximumGoalDistance": configuration.maximumGoalDistance,
                "goalRadius": configuration.goalRadius,
                "goalSlowdownDistance": configuration.goalSlowdownDistance,
                "goalCommandSpeed": configuration.goalCommandSpeed,
                "goalBoundaryCommandSpeed":
                    configuration.goalBoundaryCommandSpeed,
                "goalDwellSteps": Float(configuration.goalDwellSteps),
                "maximumGoalArrivalSpeed":
                    configuration.maximumGoalArrivalSpeed,
                "goalProgressRewardWeight":
                    configuration.goalProgressRewardWeight,
                "goalStableRewardWeight":
                    configuration.goalStableRewardWeight,
                "goalSuccessBonus": configuration.goalSuccessBonus,
            ], uniquingKeysWith: { _, new in new })
            if configuration.projectileProbability > 0 {
                configurationValues.merge([
                    "projectileProbability":
                        configuration.projectileProbability,
                    "projectileCurriculumControlSteps": Float(
                        configuration.projectileCurriculumControlSteps),
                    "projectileSize": configuration.projectileSize,
                    "projectileMass": configuration.projectileMass,
                    "minimumProjectileSpeed":
                        configuration.minimumProjectileSpeed,
                    "maximumProjectileSpeed":
                        configuration.maximumProjectileSpeed,
                    "projectileLeftProbability":
                        configuration.projectileLeftProbability,
                    "minimumProjectileLaunchStep": Float(
                        configuration.minimumProjectileLaunchStep),
                    "maximumProjectileLaunchStep": Float(
                        configuration.maximumProjectileLaunchStep),
                ], uniquingKeysWith: { _, new in new })
                if configuration.recoveryGatedActor {
                    configurationValues["recoveryGatedActor"] = 1
                    configurationValues["freezeBasePolicyExpert"] =
                        configuration.freezeBasePolicyExpert ? 1 : 0
                    if configuration.recoveryContextObservations {
                        configurationValues["recoveryContextObservations"] = 1
                        configurationValues["recoveryContextDuration"] =
                            configuration.recoveryContextDuration
                    }
                    if configuration.recoveryExpertSide != 0 {
                        configurationValues["recoveryExpertSide"] =
                            configuration.recoveryExpertSide
                    }
                    if configuration.recoveryExpertGatePeak != 1 {
                        configurationValues["recoveryExpertGatePeak"] =
                            configuration.recoveryExpertGatePeak
                    }
                    if configuration.recoveryExpertGateDecay {
                        configurationValues["recoveryExpertGateDecay"] = 1
                    }
                    if !configuration.initializeRecoveryExpertFromBaseOnTransfer {
                        configurationValues[
                            "initializeRecoveryExpertFromBaseOnTransfer"] = 0
                    }
                    if configuration
                        .initializeRecoveryExpertFromMirroredBaseOnTransfer {
                        configurationValues[
                            "initializeRecoveryExpertFromMirroredBaseOnTransfer"] = 1
                    }
                }
                // Recovery shaping belongs to the task, not to a particular
                // policy architecture. A single full actor must serialize the
                // same reward contract as a routed recovery expert.
                if configuration.postImpactUprightRewardWeight > 0 {
                    configurationValues["postImpactUprightRewardWeight"] =
                        configuration.postImpactUprightRewardWeight
                }
                if configuration.postImpactAngularVelocityPenaltyWeight > 0 {
                    configurationValues[
                        "postImpactAngularVelocityPenaltyWeight"] =
                            configuration.postImpactAngularVelocityPenaltyWeight
                }
                if configuration.postImpactFallPenalty > 0 {
                    configurationValues["postImpactFallPenalty"] =
                        configuration.postImpactFallPenalty
                }
            }
        }
        let taskRevision: Int
        // The bounded Menagerie support sets replace the former cooked hulls
        // used only by profiles without training projectiles. That changes
        // load-bearing ground contact, so those two revision families must
        // reject legacy checkpoints until they are replay-qualified and
        // republished. Projectile profiles use the full authored primitive
        // model instead; their physics and revisions are unchanged.
        if !configuration.pointGoal {
            taskRevision = 11
        } else if configuration.projectileProbability == 0 {
            taskRevision = 3
        } else if configuration.recoveryExpertSide != 0 {
            taskRevision = 7
        } else if configuration.recoveryContextObservations {
            taskRevision = 6
        } else if configuration.recoveryGatedActor {
            taskRevision = 5
        } else {
            taskRevision = 4
        }
        spec = RLTaskSpec(
            id: taskID,
            revision: RLPhysicsContract.deterministicColorSolveV1(taskRevision),
            numEnvironments: configuration.numEnvironments,
            observation: RLTensorSpec(
                name: "policy", shape: [configuration.recoveryContextObservations
                    ? Self.recoveryGoalObservationDimension
                    : (configuration.pointGoal
                        ? Self.goalObservationDimension
                        : Self.observationDimension)]),
            action: RLTensorSpec(
                name: "joint_position_offset", shape: [actionDimension]),
            maxEpisodeSteps: configuration.maxEpisodeSteps,
            simulationStep: env.scene.settings.dt,
            controlDecimation: Self.controlDecimation,
            autoReset: configuration.autoReset,
            configurationValues: configurationValues)

        let n = configuration.numEnvironments
        commands = [F3](repeating: .zero, count: n)
        targetHeadings = [Float](repeating: 0, count: n)
        goals = [F3](repeating: .zero, count: n)
        goalOverrides = [Optional<(direction: F3, distance: Float)>](
            repeating: nil, count: n)
        commandProjectionOrigins = [F3](repeating: .zero, count: n)
        commandRNGs = (0..<n).map {
            SplitMix64(seed: configuration.seed
                &+ UInt64($0) &* 0x9E3779B97F4A7C15)
        }
        noiseRNGs = (0..<n).map {
            SplitMix64(seed: configuration.seed
                ^ (UInt64($0) &* 0xD1B54A32D192ED03))
        }
        previousActions = ContiguousArray(
            repeating: 0, count: n * actionDimension)
        previousJointVelocities = [[Float]](
            repeating: [Float](repeating: 0, count: actionDimension), count: n)
        jointVelocities = previousJointVelocities
        contactTimes = [[Float]](repeating: [0, 0], count: n)
        airTimes = contactTimes
        episodeLengths = [Int](repeating: 0, count: n)
        episodeReturns = [Float](repeating: 0, count: n)
        linearSquaredErrorSums = [Float](repeating: 0, count: n)
        yawSquaredErrorSums = [Float](repeating: 0, count: n)
        forwardVelocitySums = [Float](repeating: 0, count: n)
        commandedForwardVelocitySums = [Float](repeating: 0, count: n)
        forwardPathSums = [Float](repeating: 0, count: n)
        previousGoalDistances = [Float](repeating: 0, count: n)
        minimumGoalDistances = [Float](repeating: .infinity, count: n)
        goalDwellCounts = [Int](repeating: 0, count: n)
        enteredGoals = [Bool](repeating: false, count: n)
        projectileLaunchSteps = [Int](repeating: .max, count: n)
        projectileSpeeds = [Float](repeating: 0, count: n)
        projectileSides = [Float](repeating: 1, count: n)
        projectileLaunched = [Bool](repeating: false, count: n)
        projectileImpacted = [Bool](repeating: false, count: n)
        projectileImpactSteps = [Int](repeating: .max, count: n)
        minimumPostImpactUprightCosines = [Float](repeating: 1, count: n)
        minimumPostImpactTorsoHeights = [Float](repeating: .infinity, count: n)
        disturbedEpisodes = [Bool](repeating: false, count: n)
        resetRNG = SplitMix64(seed: configuration.seed
            &+ 0xA0761D6478BD642F)

        let ids = Array(0..<n)
        let seeds = ids.map { configuration.seed &+ UInt64($0) }
        env.reset(ids, seeds: seeds, initialRollPitchRange: 0,
                  initialYawRange: configuration.initialYawRange)
        initializeEpisodes(ids, states: env.states(), seeds: seeds)
    }

    public func setTrainingMode(_ enabled: Bool) {
        trainingMode = enabled
        if enabled { trainingControlSteps = 0 }
    }

    public func setTrainingProgress(environmentSteps: Int) {
        precondition(environmentSteps >= 0)
        trainingControlSteps = environmentSteps / spec.numEnvironments
    }

    public var trainingProjectileProbability: Float {
        guard trainingMode,
              configuration.projectileCurriculumControlSteps > 0 else {
            return configuration.projectileProbability
        }
        let progress = simd_clamp(
            Float(trainingControlSteps)
                / Float(configuration.projectileCurriculumControlSteps),
            0, 1)
        return progress * configuration.projectileProbability
    }

    public func hasProjectile(environment e: Int) -> Bool {
        precondition((0..<spec.numEnvironments).contains(e))
        return environment.refs[e].projectile != nil
    }

    /// Inject an interactive physical disturbance while keeping scheduled
    /// projectile and recovery-gate bookkeeping coherent. Marking the body as
    /// launched prevents the task curriculum from silently relaunching the
    /// same rigid body later in the episode.
    public func throwRobustnessBoxes(
        environmentIDs: [Int], sideSigns: [Float],
        launchDistance: Float = 1.2, speed: Float = 6
    ) {
        precondition(environmentIDs.count == sideSigns.count)
        environment.throwBoxes(
            environmentIDs: environmentIDs, sideSigns: sideSigns,
            launchDistance: launchDistance, speed: speed)
        let measured = environment.states()
        for (offset, environmentID) in environmentIDs.enumerated() {
            precondition((0..<spec.numEnvironments).contains(environmentID))
            disturbedEpisodes[environmentID] = true
            projectileLaunched[environmentID] = true
            projectileImpacted[environmentID] = false
            projectileImpactSteps[environmentID] = .max
            projectileLaunchSteps[environmentID] = episodeLengths[environmentID]
            projectileSpeeds[environmentID] = speed
            projectileSides[environmentID] = sideSigns[offset] > 0 ? 1 : -1
            let root = measured[environmentID].root
            minimumPostImpactUprightCosines[environmentID] =
                root.rotation.act(F3(0, 0, 1)).z
            minimumPostImpactTorsoHeights[environmentID] =
                measured[environmentID].torso.position.z
        }
    }

    public func currentCommand(environment: Int) -> F3 {
        precondition((0..<spec.numEnvironments).contains(environment))
        return commands[environment]
    }

    public func currentGoalPosition(environment: Int) -> F3 {
        precondition(configuration.pointGoal)
        precondition((0..<spec.numEnvironments).contains(environment))
        return goals[environment]
    }

    public func currentGoalDirection(environment e: Int) -> F3 {
        precondition(configuration.pointGoal)
        let root = environment.states()[e].root.position
        let delta = goals[e] - root
        let planarLength = max(sqrt(delta.x * delta.x + delta.y * delta.y), 1e-6)
        return F3(delta.x / planarLength, delta.y / planarLength, 0)
    }

    /// Replay-only goal override. It changes the task command source, never
    /// joint targets or policy actions.
    public func setGoal(environment e: Int, direction: F3,
                        distance: Float) throws {
        guard configuration.pointGoal,
              (0..<spec.numEnvironments).contains(e),
              direction.x.isFinite, direction.y.isFinite,
              distance.isFinite, distance > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid H1 point-goal override")
        }
        let planarLength = sqrt(direction.x * direction.x
            + direction.y * direction.y)
        guard planarLength > 1e-6 else {
            throw RLEnvironmentError.invalidConfiguration(
                "H1 point-goal direction must be planar and nonzero")
        }
        let root = environment.states()[e].root.position
        let normalizedDirection = F3(
            direction.x / planarLength, direction.y / planarLength, 0)
        goalOverrides[e] = (normalizedDirection, distance)
        installGoal(environment: e, rootPosition: root,
                    direction: normalizedDirection, distance: distance)
        previousGoalDistances[e] = planarGoalDistance(
            environment: e, rootPosition: root)
        minimumGoalDistances[e] = previousGoalDistances[e]
        goalDwellCounts[e] = 0
        enteredGoals[e] = false
        updateGoalCommand(environment: e, rootPosition: root,
                          rootRotation: environment.states()[e].root.rotation)
        updateCommandMarkers([e])
    }

    public func clearGoalOverride(environment e: Int) {
        precondition(configuration.pointGoal)
        precondition((0..<spec.numEnvironments).contains(e))
        goalOverrides[e] = nil
    }

    /// Absolute world-space direction selected by the task's heading command.
    /// Replay uses this to draw an honest command projection; it is not a
    /// point-goal input and must not be presented as one.
    public func currentCommandDirection(environment: Int) -> F3 {
        precondition((0..<spec.numEnvironments).contains(environment))
        let heading = targetHeadings[environment]
        return F3(cos(heading), sin(heading), 0)
    }

    /// Endpoint reached if the current forward-speed command were tracked
    /// perfectly until its next scheduled resample.
    public func currentCommandProjection(environment: Int) -> F3 {
        precondition((0..<spec.numEnvironments).contains(environment))
        if configuration.pointGoal { return goals[environment] }
        let distance = commands[environment].x
            * Float(configuration.commandResamplingSteps) * spec.controlStep
        return commandProjectionOrigins[environment]
            + currentCommandDirection(environment: environment) * distance
    }

    public func reset(environments ids: [Int]?, seed: UInt64,
                      into observations: inout RLObservationBatch) throws {
        try observations.validate(for: spec)
        let envIDs = try checkedEnvironmentIDs(ids)
        try environment.solver.synchronize()
        let seeds = envIDs.map {
            seed &+ UInt64($0) &* 0x9E3779B97F4A7C15
        }
        environment.reset(envIDs, seeds: seeds, initialRollPitchRange: 0,
                          initialYawRange: configuration.initialYawRange)
        let states = environment.states()
        initializeEpisodes(envIDs, states: states, seeds: seeds)
        fillObservations(states, into: &observations.policy)
        try observations.validate(for: spec)
    }

    public func step(actions: RLActionBatch,
                     into result: inout RLStepBatch) throws {
        try actions.validate(for: spec)
        try result.validate(for: spec)
        try environment.solver.synchronize()
        result.clearSignals()
        let n = spec.numEnvironments
        let dt = spec.controlStep
        var actionRate = ContiguousArray(repeating: Float(0), count: n)
        for e in 0..<n {
            let base = e * actionDimension
            for j in 0..<actionDimension {
                let action = actions.values[base + j]
                let delta = action - previousActions[base + j]
                actionRate[e] += delta * delta
            }
        }

        launchScheduledProjectiles()
        try environment.stepChecked(
            normalizedActions: actions.values,
            decimation: Self.controlDecimation,
            clampActions: false,
            clampTargetsToLimits: false)
        if trainingMode { trainingControlSteps += 1 }
        var states = environment.states()
        let contacts = environment.groundContacts()
        let projectileContacts = configuration.projectileProbability > 0
            ? environment.projectileRobotContacts()
            : [Bool](repeating: false, count: n)
        var projectileContactMetric = ContiguousArray(
            repeating: Float(0), count: n)
        for e in 0..<n where projectileContacts[e] {
            if !projectileImpacted[e] {
                projectileImpactSteps[e] = episodeLengths[e]
            }
            projectileImpacted[e] = true
            projectileContactMetric[e] = 1
        }
        var trackingLinear = ContiguousArray(repeating: Float(0), count: n)
        var trackingYaw = ContiguousArray(repeating: Float(0), count: n)
        var feetAirTimeReward = ContiguousArray(repeating: Float(0), count: n)
        var feetSlideCost = ContiguousArray(repeating: Float(0), count: n)
        var orientationCost = ContiguousArray(repeating: Float(0), count: n)
        var angularVelocityXYCost = ContiguousArray(
            repeating: Float(0), count: n)
        var ankleLimitCost = ContiguousArray(repeating: Float(0), count: n)
        var jointDeviationCost = ContiguousArray(repeating: Float(0), count: n)
        var jointAccelerationCost = ContiguousArray(repeating: Float(0), count: n)
        var runningReward = ContiguousArray(repeating: Float(0), count: n)
        var goalProgressReward = ContiguousArray(repeating: Float(0), count: n)
        var goalStableReward = ContiguousArray(repeating: Float(0), count: n)
        var postImpactRecoveryReward = ContiguousArray(
            repeating: Float(0), count: n)
        var torsoHeight = ContiguousArray(repeating: Float(0), count: n)
        var projectedGravityZ = ContiguousArray(repeating: Float(0), count: n)
        var feetInContact = ContiguousArray(repeating: Float(0), count: n)
        var minimumFootClearance = ContiguousArray(repeating: Float(0), count: n)
        var rootPlanarSpeed = ContiguousArray(repeating: Float(0), count: n)
        var maximumActuatorTorqueRatio = ContiguousArray(
            repeating: Float(0), count: n)
        var saturatedActuatorCount = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeReturnMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLengthMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeSurvivedMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeLinearRMSEMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeYawRMSEMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeForwardVelocityMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeCommandedForwardMetric = ContiguousArray(repeating: Float(0), count: n)
        var episodeForwardPathMetric = ContiguousArray(repeating: Float(0), count: n)
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
        var episodeNominalBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeNominalSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeDisturbedBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeDisturbedSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeProjectileLaunchedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeProjectileImpactedMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeProjectileLeftBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeProjectileLeftSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeProjectileRightBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeProjectileRightSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeProjectileEarlyBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeProjectileEarlySuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeProjectileLateBinMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeProjectileLateSuccessMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeImpactToTerminalMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeFellAfterImpactMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMinimumPostImpactUprightMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var episodeMinimumPostImpactTorsoHeightMetric = ContiguousArray(
            repeating: Float(0), count: n)
        var resetIDs = [Int]()
        var resetSeeds = [UInt64]()
        var resampledCommandIDs = [Int]()

        for e in 0..<n {
            let state = states[e]
            if configuration.pointGoal {
                updateGoalCommand(
                    environment: e, rootPosition: state.root.position,
                    rootRotation: state.root.rotation)
            } else {
                updateCommandYaw(environment: e,
                                 rootRotation: state.root.rotation)
            }
            let heading = Self.horizontalHeading(state.root.rotation)
            let lateral = F3(-heading.y, heading.x, 0)
            let localVelocity = F3(
                simd_dot(state.root.linearVelocity, heading),
                simd_dot(state.root.linearVelocity, lateral),
                state.root.linearVelocity.z)
            let projectedGravity = state.root.rotation.conjugate.act(F3(0, 0, 1))
            let command = commands[e]
            let linearErrorSquared =
                (localVelocity.x - command.x) * (localVelocity.x - command.x)
                + (localVelocity.y - command.y) * (localVelocity.y - command.y)
            let yawError = state.root.angularVelocity.z - command.z
            trackingLinear[e] = exp(-linearErrorSquared / Self.trackingVariance)
            trackingYaw[e] = exp(-(yawError * yawError) / Self.trackingVariance)
            orientationCost[e] = projectedGravity.x * projectedGravity.x
                + projectedGravity.y * projectedGravity.y
            let localAngularVelocity = state.root.rotation.conjugate.act(
                state.root.angularVelocity)
            angularVelocityXYCost[e] =
                localAngularVelocity.x * localAngularVelocity.x
                    + localAngularVelocity.y * localAngularVelocity.y
            projectedGravityZ[e] = projectedGravity.z
            torsoHeight[e] = state.torso.position.z
            if projectileImpacted[e] {
                minimumPostImpactUprightCosines[e] = min(
                    minimumPostImpactUprightCosines[e], projectedGravity.z)
                minimumPostImpactTorsoHeights[e] = min(
                    minimumPostImpactTorsoHeights[e], state.torso.position.z)
            }
            feetInContact[e] = Float(contacts.feet[e].filter { $0 }.count)
            minimumFootClearance[e] = min(
                Self.footHullGroundClearance(
                    state.leftFoot, vertices: UnitreeH1CollisionHulls.leftAnkle),
                Self.footHullGroundClearance(
                    state.rightFoot, vertices: UnitreeH1CollisionHulls.rightAnkle))
            rootPlanarSpeed[e] = sqrt(
                state.root.linearVelocity.x * state.root.linearVelocity.x
                    + state.root.linearVelocity.y * state.root.linearVelocity.y)

            let currentJointVelocities = state.jointVelocities
            for j in 0..<actionDimension {
                let joint = environment.scene.joints[
                    environment.refs[e].motors[j]]
                let requestedTarget = HumanoidWalkEnv.defaultJointPositions[j]
                    + actions.values[e * actionDimension + j]
                    * HumanoidWalkEnv.actionScales[j]
                let rawTorque = joint.motorStiffness
                    * (state.jointAngles[j] - requestedTarget)
                    + joint.motorDamping * currentJointVelocities[j]
                let torqueRatio = abs(rawTorque)
                    / max(joint.motorTorque, Float.leastNormalMagnitude)
                maximumActuatorTorqueRatio[e] = max(
                    maximumActuatorTorqueRatio[e], torqueRatio)
                if torqueRatio >= 1 { saturatedActuatorCount[e] += 1 }
                let acceleration = (currentJointVelocities[j]
                    - previousJointVelocities[e][j]) / dt
                jointAccelerationCost[e] += acceleration * acceleration
            }
            jointVelocities[e] = currentJointVelocities

            for foot in 0..<2 {
                if contacts.feet[e][foot] {
                    contactTimes[e][foot] += dt
                    airTimes[e][foot] = 0
                } else {
                    airTimes[e][foot] += dt
                    contactTimes[e][foot] = 0
                }
            }
            if contacts.feet[e].filter({ $0 }).count == 1,
               sqrt(command.x * command.x + command.y * command.y) > 0.1 {
                let mode0 = contacts.feet[e][0]
                    ? contactTimes[e][0] : airTimes[e][0]
                let mode1 = contacts.feet[e][1]
                    ? contactTimes[e][1] : airTimes[e][1]
                feetAirTimeReward[e] = min(
                    min(mode0, mode1), Self.footAirTimeThreshold)
            }
            let feet = [state.leftFoot, state.rightFoot]
            for foot in 0..<2 where contacts.feet[e][foot] {
                let velocity = feet[foot].linearVelocity
                feetSlideCost[e] += sqrt(velocity.x * velocity.x
                    + velocity.y * velocity.y)
            }

            for ankle in [4, 9] {
                let joint = environment.scene.joints[
                    environment.refs[e].motors[ankle]]
                let angle = state.jointAngles[ankle]
                let center = 0.5 * (joint.limitLo + joint.limitHi)
                let halfRange = 0.45 * (joint.limitHi - joint.limitLo)
                let softLo = center - halfRange
                let softHi = center + halfRange
                ankleLimitCost[e] += max(softLo - angle, 0)
                    + max(angle - softHi, 0)
            }
            for joint in [0, 1, 5, 6] {
                jointDeviationCost[e] += 0.2 * abs(state.jointAngles[joint])
            }
            for joint in 11..<19 {
                jointDeviationCost[e] += 0.2 * abs(state.jointAngles[joint])
            }
            jointDeviationCost[e] += 0.1 * abs(state.jointAngles[10])

            let fallen = contacts.torso[e]
            let rewardRate = trackingLinear[e] + trackingYaw[e]
                + feetAirTimeReward[e]
                - 0.25 * feetSlideCost[e]
                - orientationCost[e]
                - 0.05 * angularVelocityXYCost[e]
                - ankleLimitCost[e]
                - jointDeviationCost[e]
                - 0.005 * actionRate[e]
                - 1.25e-7 * jointAccelerationCost[e]
                - (fallen ? 200 : 0)
            var shapedRewardRate = rewardRate
            if projectileImpacted[e] {
                let recoveryRate =
                    configuration.postImpactUprightRewardWeight
                        * max(projectedGravity.z, 0)
                    - configuration.postImpactAngularVelocityPenaltyWeight
                        * angularVelocityXYCost[e]
                    - (fallen ? configuration.postImpactFallPenalty : 0)
                postImpactRecoveryReward[e] = recoveryRate * dt
                shapedRewardRate += recoveryRate
            }
            runningReward[e] = shapedRewardRate * dt
            var reachedGoal = false
            var goalDistance: Float = 0
            if configuration.pointGoal {
                goalDistance = planarGoalDistance(
                    environment: e, rootPosition: state.root.position)
                let progress = previousGoalDistances[e] - goalDistance
                goalProgressReward[e] =
                    configuration.goalProgressRewardWeight * progress
                minimumGoalDistances[e] = min(
                    minimumGoalDistances[e], goalDistance)
                if goalDistance <= configuration.goalRadius {
                    enteredGoals[e] = true
                }
                let stable = goalDistance <= configuration.goalRadius
                    && rootPlanarSpeed[e]
                        <= configuration.maximumGoalArrivalSpeed
                goalDwellCounts[e] = stable ? goalDwellCounts[e] + 1 : 0
                goalStableReward[e] = stable
                    ? configuration.goalStableRewardWeight * dt : 0
                reachedGoal = goalDwellCounts[e]
                    >= configuration.goalDwellSteps
                previousGoalDistances[e] = goalDistance
                runningReward[e] += goalProgressReward[e]
                    + goalStableReward[e]
                    + (reachedGoal ? configuration.goalSuccessBonus : 0)
            }
            result.rewards[e] = runningReward[e]
            episodeReturns[e] += runningReward[e]
            linearSquaredErrorSums[e] += linearErrorSquared
            yawSquaredErrorSums[e] += yawError * yawError
            forwardVelocitySums[e] += localVelocity.x
            commandedForwardVelocitySums[e] += command.x
            forwardPathSums[e] += localVelocity.x * dt
            episodeLengths[e] += 1

            let timedOut = episodeLengths[e] >= configuration.maxEpisodeSteps
            if fallen || timedOut || reachedGoal {
                let inverseLength = 1 / Float(max(episodeLengths[e], 1))
                let linearRMSE = sqrt(linearSquaredErrorSums[e] * inverseLength)
                let yawRMSE = sqrt(yawSquaredErrorSums[e] * inverseLength)
                let success = configuration.pointGoal
                    ? reachedGoal && !fallen
                    : timedOut && !fallen
                        && linearRMSE <= 0.35 && yawRMSE <= 0.50
                result.terminated[e] = fallen || reachedGoal
                result.truncated[e] = !fallen && !reachedGoal && timedOut
                result.successes[e] = success
                result.hasFinalObservation[e] = true
                episodeReturnMetric[e] = episodeReturns[e]
                episodeLengthMetric[e] = Float(episodeLengths[e])
                // Goal episodes terminate successfully before the horizon.
                // Count those as survived; the flat task retains its stricter
                // full-horizon definition.
                episodeSurvivedMetric[e] = configuration.pointGoal
                    ? (!fallen ? 1 : 0)
                    : (timedOut && !fallen ? 1 : 0)
                episodeLinearRMSEMetric[e] = linearRMSE
                episodeYawRMSEMetric[e] = yawRMSE
                episodeForwardVelocityMetric[e] =
                    forwardVelocitySums[e] * inverseLength
                episodeCommandedForwardMetric[e] =
                    commandedForwardVelocitySums[e] * inverseLength
                episodeForwardPathMetric[e] = forwardPathSums[e]
                if configuration.pointGoal {
                    episodeGoalReachedMetric[e] = success ? 1 : 0
                    episodeGoalEnteredMetric[e] = enteredGoals[e] ? 1 : 0
                    episodeFinalGoalDistanceMetric[e] = goalDistance
                    episodeMinimumGoalDistanceMetric[e] =
                        minimumGoalDistances[e]
                    episodeGoalDwellMetric[e] = Float(goalDwellCounts[e])
                    episodeArrivalSpeedMetric[e] = rootPlanarSpeed[e]
                    episodeProjectileLaunchedMetric[e] =
                        projectileLaunched[e] ? 1 : 0
                    episodeProjectileImpactedMetric[e] =
                        projectileImpacted[e] ? 1 : 0
                    if projectileImpacted[e] {
                        let postImpactSteps = max(
                            episodeLengths[e] - projectileImpactSteps[e], 0)
                        episodeImpactToTerminalMetric[e] = Float(postImpactSteps)
                        episodeFellAfterImpactMetric[e] = fallen ? 1 : 0
                        episodeMinimumPostImpactUprightMetric[e] =
                            minimumPostImpactUprightCosines[e]
                        episodeMinimumPostImpactTorsoHeightMetric[e] =
                            minimumPostImpactTorsoHeights[e]
                        if projectileSides[e] < 0 {
                            episodeProjectileLeftBinMetric[e] = 1
                            episodeProjectileLeftSuccessMetric[e] = success ? 1 : 0
                        } else {
                            episodeProjectileRightBinMetric[e] = 1
                            episodeProjectileRightSuccessMetric[e] = success ? 1 : 0
                        }
                        let launchMidpoint = (
                            configuration.minimumProjectileLaunchStep
                                + configuration.maximumProjectileLaunchStep) / 2
                        if projectileLaunchSteps[e] <= launchMidpoint {
                            episodeProjectileEarlyBinMetric[e] = 1
                            episodeProjectileEarlySuccessMetric[e] = success ? 1 : 0
                        } else {
                            episodeProjectileLateBinMetric[e] = 1
                            episodeProjectileLateSuccessMetric[e] = success ? 1 : 0
                        }
                    }
                    if disturbedEpisodes[e] {
                        episodeDisturbedBinMetric[e] = 1
                        // A scheduled-but-never-launched box does not count
                        // as robust success even if the robot reached early.
                        episodeDisturbedSuccessMetric[e] =
                            success && projectileImpacted[e] ? 1 : 0
                    } else {
                        episodeNominalBinMetric[e] = 1
                        episodeNominalSuccessMetric[e] = success ? 1 : 0
                    }
                }
                if configuration.autoReset {
                    resetIDs.append(e)
                    resetSeeds.append(resetRNG.next())
                }
            } else if !configuration.pointGoal && episodeLengths[e]
                .isMultiple(of: configuration.commandResamplingSteps) {
                sampleCommand(
                    environment: e, rootPosition: state.root.position,
                    rootRotation: state.root.rotation)
                resampledCommandIDs.append(e)
            }
        }

        updateCommandMarkers(resampledCommandIDs)

        for e in 0..<n {
            let base = e * actionDimension
            previousJointVelocities[e] = jointVelocities[e]
            for j in 0..<actionDimension {
                previousActions[base + j] = actions.values[base + j]
            }
        }
        fillObservations(states, into: &result.observations.policy)
        for e in 0..<n where result.hasFinalObservation[e] {
            let dimension = spec.observation.elementCount
            let base = e * dimension
            for j in 0..<dimension {
                result.finalObservations[base + j] =
                    result.observations.policy[base + j]
            }
        }

        result.metrics["reward/tracking_linear_velocity"] = trackingLinear
        result.metrics["reward/tracking_yaw_rate"] = trackingYaw
        result.metrics["reward/feet_air_time"] = feetAirTimeReward
        result.metrics["reward/running"] = runningReward
        result.metrics["reward/goal_progress"] = goalProgressReward
        result.metrics["reward/goal_stable"] = goalStableReward
        result.metrics["reward/post_impact_recovery"] =
            postImpactRecoveryReward
        result.metrics["penalty/feet_slide"] = feetSlideCost
        result.metrics["penalty/orientation"] = orientationCost
        result.metrics["penalty/angular_velocity_xy"] = angularVelocityXYCost
        result.metrics["penalty/action_rate"] = actionRate
        result.metrics["penalty/joint_acceleration"] = jointAccelerationCost
        result.metrics["penalty/joint_deviation"] = jointDeviationCost
        result.metrics["state/torso_height_m"] = torsoHeight
        result.metrics["state/projected_gravity_z"] = projectedGravityZ
        result.metrics["state/feet_in_contact"] = feetInContact
        result.metrics["state/minimum_foot_clearance_m"] =
            minimumFootClearance
        result.metrics["state/root_planar_speed_mps"] = rootPlanarSpeed
        result.metrics["state/maximum_actuator_torque_ratio"] =
            maximumActuatorTorqueRatio
        result.metrics["state/saturated_actuator_count"] =
            saturatedActuatorCount
        result.metrics["state/projectile_robot_contact"] =
            projectileContactMetric
        result.metrics["episode/return"] = episodeReturnMetric
        result.metrics["episode/length"] = episodeLengthMetric
        result.metrics["episode/survived"] = episodeSurvivedMetric
        result.metrics["episode/linear_velocity_rmse_mps"] =
            episodeLinearRMSEMetric
        result.metrics["episode/yaw_rate_rmse_rps"] = episodeYawRMSEMetric
        result.metrics["episode/mean_local_forward_velocity_mps"] =
            episodeForwardVelocityMetric
        result.metrics["episode/mean_commanded_forward_velocity_mps"] =
            episodeCommandedForwardMetric
        result.metrics["episode/forward_path_m"] = episodeForwardPathMetric
        result.metrics["episode/goal_reached"] = episodeGoalReachedMetric
        result.metrics["episode/goal_entered"] = episodeGoalEnteredMetric
        result.metrics["episode/final_goal_distance_m"] =
            episodeFinalGoalDistanceMetric
        result.metrics["episode/minimum_goal_distance_m"] =
            episodeMinimumGoalDistanceMetric
        result.metrics["episode/goal_dwell_steps"] = episodeGoalDwellMetric
        result.metrics["episode/goal_arrival_speed_mps"] =
            episodeArrivalSpeedMetric
        if configuration.pointGoal {
            result.metrics["episode/nominal_bin"] = episodeNominalBinMetric
            result.metrics["episode/nominal_success"] =
                episodeNominalSuccessMetric
            result.metrics["episode/disturbed_bin"] = episodeDisturbedBinMetric
            result.metrics["episode/disturbed_success"] =
                episodeDisturbedSuccessMetric
            result.metrics["episode/projectile_launched"] =
                episodeProjectileLaunchedMetric
            result.metrics["episode/projectile_impacted"] =
                episodeProjectileImpactedMetric
            result.metrics["episode/projectile_left_bin"] =
                episodeProjectileLeftBinMetric
            result.metrics["episode/projectile_left_success"] =
                episodeProjectileLeftSuccessMetric
            result.metrics["episode/projectile_right_bin"] =
                episodeProjectileRightBinMetric
            result.metrics["episode/projectile_right_success"] =
                episodeProjectileRightSuccessMetric
            result.metrics["episode/projectile_early_bin"] =
                episodeProjectileEarlyBinMetric
            result.metrics["episode/projectile_early_success"] =
                episodeProjectileEarlySuccessMetric
            result.metrics["episode/projectile_late_bin"] =
                episodeProjectileLateBinMetric
            result.metrics["episode/projectile_late_success"] =
                episodeProjectileLateSuccessMetric
            result.metrics["episode/impact_to_terminal_steps"] =
                episodeImpactToTerminalMetric
            result.metrics["episode/fell_after_impact"] =
                episodeFellAfterImpactMetric
            result.metrics["episode/minimum_post_impact_upright_cosine"] =
                episodeMinimumPostImpactUprightMetric
            result.metrics["episode/minimum_post_impact_torso_height_m"] =
                episodeMinimumPostImpactTorsoHeightMetric
        }

        if !resetIDs.isEmpty {
            environment.reset(resetIDs, seeds: resetSeeds,
                              initialRollPitchRange: 0,
                              initialYawRange: configuration.initialYawRange)
            states = environment.states()
            initializeEpisodes(resetIDs, states: states, seeds: resetSeeds)
            fillObservations(states, into: &result.observations.policy)
        }
        try result.observations.validate(for: spec)
    }

    /// Exact lowest point of an authored Isaac collision hull. The incoming
    /// state is already expressed in the USD/MJCF link frame, and the decoded
    /// cooked vertices are authored in that same frame.
    static func footHullGroundClearance(
        _ foot: GPUSolver.RigidBodyState, vertices: [F3]
    ) -> Float {
        vertices.reduce(Float.infinity) {
            min($0, (foot.position + foot.rotation.act($1)).z)
        }
    }

    private func initializeEpisodes(_ ids: [Int], states: [HumanoidState],
                                    seeds: [UInt64]) {
        precondition(ids.count == seeds.count)
        for (offset, e) in ids.enumerated() {
            commandRNGs[e] = SplitMix64(seed: seeds[offset]
                ^ 0xA0761D6478BD642F)
            noiseRNGs[e] = SplitMix64(seed: seeds[offset]
                ^ 0xE7037ED1A0B428DB)
            sampleCommand(
                environment: e, rootPosition: states[e].root.position,
                rootRotation: states[e].root.rotation)
            if configuration.pointGoal {
                let distance = planarGoalDistance(
                    environment: e, rootPosition: states[e].root.position)
                previousGoalDistances[e] = distance
                minimumGoalDistances[e] = distance
                goalDwellCounts[e] = 0
                enteredGoals[e] = false
                if configuration.projectileProbability > 0 {
                    disturbedEpisodes[e] = commandRNGs[e].nextFloat()
                        < trainingProjectileProbability
                } else {
                    disturbedEpisodes[e] = false
                }
                if disturbedEpisodes[e] {
                    let span = configuration.maximumProjectileLaunchStep
                        - configuration.minimumProjectileLaunchStep + 1
                    projectileLaunchSteps[e] =
                        configuration.minimumProjectileLaunchStep
                        + Int(commandRNGs[e].next() % UInt64(span))
                    projectileSpeeds[e] = configuration.minimumProjectileSpeed
                        + (configuration.maximumProjectileSpeed
                            - configuration.minimumProjectileSpeed)
                            * commandRNGs[e].nextFloat()
                    projectileSides[e] = commandRNGs[e].nextFloat()
                        < configuration.projectileLeftProbability
                        ? -1 : 1
                } else {
                    projectileLaunchSteps[e] = .max
                    projectileSpeeds[e] = 0
                    projectileSides[e] = 1
                }
                projectileLaunched[e] = false
                projectileImpacted[e] = false
                projectileImpactSteps[e] = .max
                minimumPostImpactUprightCosines[e] = 1
                minimumPostImpactTorsoHeights[e] = .infinity
            }
            previousJointVelocities[e] = states[e].jointVelocities
            jointVelocities[e] = states[e].jointVelocities
            contactTimes[e] = [0, 0]
            airTimes[e] = [0, 0]
            let base = e * actionDimension
            for j in 0..<actionDimension { previousActions[base + j] = 0 }
            episodeLengths[e] = 0
            episodeReturns[e] = 0
            linearSquaredErrorSums[e] = 0
            yawSquaredErrorSums[e] = 0
            forwardVelocitySums[e] = 0
            commandedForwardVelocitySums[e] = 0
            forwardPathSums[e] = 0
        }
        updateCommandMarkers(ids)
    }

    /// Launches one colliding rigid box from a randomized side toward the H1
    /// torso. Every replica owns its projectile through collision groups, so
    /// this remains a genuinely batched physical perturbation.
    private func launchScheduledProjectiles() {
        guard configuration.projectileProbability > 0 else { return }
        let states = environment.states()
        var ids = [Int]()
        var positions = [F3]()
        var velocities = [F3]()
        var angularVelocities = [F3]()
        for e in 0..<spec.numEnvironments
            where disturbedEpisodes[e] && !projectileLaunched[e]
                && episodeLengths[e] >= projectileLaunchSteps[e] {
            let root = states[e].root.position
            let delta = configuration.pointGoal
                ? goals[e] - root
                : currentCommandDirection(environment: e)
            let planarLength = max(
                sqrt(delta.x * delta.x + delta.y * delta.y), 1e-6)
            let forward = F3(
                delta.x / planarLength, delta.y / planarLength, 0)
            let lateral = F3(-forward.y, forward.x, 0)
            let target = states[e].torso.position
            let launchDistance: Float = 1.2
            let launch = F3(target.x, target.y, target.z)
                + lateral * (launchDistance * projectileSides[e])
                + forward * 0.15
            // Lead the moving torso and compensate gravity. A straight
            // line-of-sight launch from 1--2 m drops below the authored torso
            // hull before arrival and silently turns most "disturbances" into
            // misses. `projectileSpeeds` is the intended lateral closing
            // speed; the vertical component is whatever the ballistic
            // intercept requires.
            let flightTime = launchDistance / projectileSpeeds[e]
            let gravity = F3(0, 0, environment.scene.settings.gravity)
            let predictedTarget = target
                + states[e].torso.linearVelocity * flightTime
            let velocity = (predictedTarget - launch
                - 0.5 * gravity * flightTime * flightTime) / flightTime
            ids.append(e)
            positions.append(launch)
            velocities.append(velocity)
            angularVelocities.append(F3(
                projectileSides[e] * 2.5,
                -projectileSides[e] * 1.5,
                projectileSides[e] * 3.5))
            projectileLaunched[e] = true
        }
        if !ids.isEmpty {
            environment.throwProjectiles(
                environmentIDs: ids, positions: positions,
                velocities: velocities,
                angularVelocities: angularVelocities)
        }
    }

    private func sampleCommand(environment e: Int, rootPosition: F3,
                               rootRotation: Quat) {
        commandProjectionOrigins[e] = F3(
            rootPosition.x, rootPosition.y, environment.refs[e].center.z)
        let currentHeading = Self.headingAngle(rootRotation)
        if configuration.pointGoal {
            let direction: F3
            let distance: Float
            if let override = goalOverrides[e] {
                direction = override.direction
                distance = override.distance
            } else {
                distance = configuration.minimumGoalDistance
                    + commandRNGs[e].nextFloat()
                        * (configuration.maximumGoalDistance
                            - configuration.minimumGoalDistance)
                let angle = (2 * commandRNGs[e].nextFloat() - 1) * .pi
                direction = F3(cos(angle), sin(angle), 0)
            }
            installGoal(environment: e, rootPosition: rootPosition,
                        direction: direction, distance: distance)
            updateGoalCommand(
                environment: e, rootPosition: rootPosition,
                rootRotation: rootRotation)
            return
        }
        if commandRNGs[e].nextFloat()
            < configuration.standingCommandProbability {
            commands[e] = .zero
            targetHeadings[e] = currentHeading
            return
        }
        commands[e] = F3(commandRNGs[e].nextFloat(), 0, 0)
        targetHeadings[e] = (2 * commandRNGs[e].nextFloat() - 1) * .pi
        updateCommandYaw(environment: e, rootRotation: rootRotation)
    }

    private func installGoal(environment e: Int, rootPosition: F3,
                             direction: F3, distance: Float) {
        goals[e] = F3(
            rootPosition.x + direction.x * distance,
            rootPosition.y + direction.y * distance,
            environment.refs[e].center.z)
        targetHeadings[e] = atan2(direction.y, direction.x)
    }

    private func updateCommandMarkers(_ ids: [Int]) {
        guard !ids.isEmpty else { return }
        if configuration.pointGoal {
            let states = environment.states()
            environment.setGoalMarkers(
                environmentIDs: ids,
                directions: ids.map { e in
                    let delta = goals[e] - states[e].root.position
                    let planar = max(
                        sqrt(delta.x * delta.x + delta.y * delta.y), 1e-6)
                    return F3(delta.x / planar, delta.y / planar, 0)
                },
                distances: ids.map { e in
                    planarGoalDistance(
                        environment: e,
                        rootPosition: states[e].root.position)
                },
                origins: ids.map { e in
                    let root = states[e].root.position
                    return F3(root.x, root.y, environment.refs[e].center.z)
                })
            return
        }
        environment.setGoalMarkers(
            environmentIDs: ids,
            directions: ids.map { currentCommandDirection(environment: $0) },
            distances: ids.map {
                commands[$0].x * Float(configuration.commandResamplingSteps)
                    * spec.controlStep
            },
            origins: ids.map { commandProjectionOrigins[$0] })
    }

    private func updateCommandYaw(environment e: Int, rootRotation: Quat) {
        guard commands[e].x != 0 else { commands[e].z = 0; return }
        var error = targetHeadings[e] - Self.headingAngle(rootRotation)
        error -= 2 * .pi * floor((error + .pi) / (2 * .pi))
        commands[e].z = simd_clamp(0.5 * error, -1, 1)
    }

    private func planarGoalDistance(environment e: Int,
                                    rootPosition: F3) -> Float {
        let delta = goals[e] - rootPosition
        return sqrt(delta.x * delta.x + delta.y * delta.y)
    }

    private func updateGoalCommand(environment e: Int, rootPosition: F3,
                                   rootRotation: Quat) {
        let navigation = PointGoalNavigator.command(
            worldGoal: goals[e],
            bodyPosition: rootPosition,
            bodyRotation: rootRotation,
            parameters: PointGoalNavigationParameters(
                goalRadius: configuration.goalRadius,
                slowdownDistance: configuration.goalSlowdownDistance,
                cruiseSpeed: configuration.goalCommandSpeed,
                boundarySpeed: configuration.goalBoundaryCommandSpeed,
                yawGain: 0.5,
                maximumYawRate: 1,
                mode: .forwardOnlyYaw))
        commands[e] = navigation.bodyTwist
        targetHeadings[e] = navigation.desiredWorldHeading
    }

    private func fillObservations(_ states: [HumanoidState],
                                  into output: inout ContiguousArray<Float>) {
        let dimension = spec.observation.elementCount
        for e in 0..<spec.numEnvironments {
            let state = states[e]
            let localLinear = state.root.rotation.conjugate.act(
                state.root.linearVelocity)
            let localAngular = state.root.rotation.conjugate.act(
                state.root.angularVelocity)
            // Isaac Lab's `projected_gravity` observation rotates the
            // normalized world gravity vector into the root frame. World
            // gravity points down, so an upright robot observes (0, 0, -1),
            // not +Z. The sign is policy-critical even though the flatness
            // reward below only squares its horizontal components.
            let gravity = state.root.rotation.conjugate.act(F3(0, 0, -1))
            let base = e * dimension
            let noiseEnabled = trainingMode && configuration.observationNoise
            func noise(_ amplitude: Float) -> Float {
                guard noiseEnabled else { return 0 }
                return (2 * noiseRNGs[e].nextFloat() - 1) * amplitude
            }
            output[base] = localLinear.x + noise(0.1)
            output[base + 1] = localLinear.y + noise(0.1)
            output[base + 2] = localLinear.z + noise(0.1)
            output[base + 3] = localAngular.x + noise(0.2)
            output[base + 4] = localAngular.y + noise(0.2)
            output[base + 5] = localAngular.z + noise(0.2)
            output[base + 6] = gravity.x + noise(0.05)
            output[base + 7] = gravity.y + noise(0.05)
            output[base + 8] = gravity.z + noise(0.05)
            output[base + 9] = commands[e].x
            output[base + 10] = commands[e].y
            output[base + 11] = commands[e].z
            for j in 0..<actionDimension {
                output[base + 12 + j] = state.jointAngles[j] + noise(0.01)
                output[base + 31 + j] = jointVelocities[e][j] + noise(1.5)
                output[base + 50 + j] = previousActions[
                    e * actionDimension + j]
            }
            if configuration.pointGoal {
                let localGoal = state.root.rotation.conjugate.act(
                    goals[e] - state.root.position)
                let scale = max(configuration.maximumGoalDistance, 1)
                output[base + 69] = localGoal.x / scale
                output[base + 70] = localGoal.y / scale
                if configuration.recoveryContextObservations {
                    output[base + 71] = projectileImpacted[e]
                        ? projectileSides[e] : 0
                    let elapsedSteps = projectileImpactSteps[e] == .max
                        ? 0 : max(episodeLengths[e] - projectileImpactSteps[e], 0)
                    output[base + 72] = min(
                        Float(elapsedSteps) * spec.controlStep
                            / configuration.recoveryContextDuration,
                        1)
                }
            }
        }
    }

    public func initializationObservationSourceIndices(
        sourceDimension: Int
    ) -> [Int?]? {
        guard configuration.pointGoal else { return nil }
        let destinationDimension = spec.observation.elementCount
        if sourceDimension == Self.observationDimension {
            return (0..<Self.observationDimension).map(Optional.some)
                + [Int?](repeating: nil,
                         count: destinationDimension - Self.observationDimension)
        }
        if configuration.recoveryContextObservations,
           sourceDimension == Self.goalObservationDimension {
            return (0..<Self.goalObservationDimension).map(Optional.some)
                + [nil, nil]
        }
        return nil
    }

    public func policyReferenceRegularizationWeights(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        precondition(observations.count
            == spec.numEnvironments * spec.observation.elementCount)
        guard configuration.pointGoal,
              configuration.projectileProbability > 0 else {
            return ContiguousArray(
                repeating: 1, count: spec.numEnvironments)
        }
        // Preserve the transferred locomotion/steering behavior up to real
        // contact. Once a physical impact has occurred, PPO is free to learn
        // a recovery action without being pulled toward the policy that just
        // failed in that disturbed state.
        return ContiguousArray(projectileImpacted.map { $0 ? 0 : 1 })
    }

    public var usesPolicyExpertGate: Bool {
        configuration.recoveryGatedActor
    }

    public var freezesBasePolicyExpert: Bool {
        configuration.recoveryGatedActor
            && configuration.freezeBasePolicyExpert
    }

    public var initializesPolicyExpertFromBaseOnTransfer: Bool {
        configuration.recoveryGatedActor
            && configuration.initializeRecoveryExpertFromBaseOnTransfer
    }

    public var initializesPolicyExpertFromMirroredBaseOnTransfer: Bool {
        configuration.recoveryGatedActor
            && configuration
                .initializeRecoveryExpertFromMirroredBaseOnTransfer
    }

    public func policyExpertGates(
        _ observations: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        precondition(observations.count
            == spec.numEnvironments * spec.observation.elementCount)
        guard configuration.recoveryGatedActor else {
            return ContiguousArray(
                repeating: 0, count: spec.numEnvironments)
        }
        if configuration.recoveryContextObservations {
            let dimension = spec.observation.elementCount
            return ContiguousArray((0..<spec.numEnvironments).map { e in
                let side = observations[e * dimension + 71]
                guard side != 0 else { return 0 }
                guard configuration.recoveryExpertSide == 0
                    || side * configuration.recoveryExpertSide > 0 else {
                    return 0
                }
                let peak = configuration.recoveryExpertGatePeak
                guard configuration.recoveryExpertGateDecay else { return peak }
                let elapsed = simd_clamp(
                    observations[e * dimension + 72], 0, 1)
                return peak * (1 - elapsed)
            })
        }
        return ContiguousArray((0..<spec.numEnvironments).map { e in
            guard projectileImpacted[e] else { return 0 }
            return configuration.recoveryExpertSide == 0
                || projectileSides[e] * configuration.recoveryExpertSide > 0
                ? configuration.recoveryExpertGatePeak : 0
        })
    }

    private static func headingAngle(_ rotation: Quat) -> Float {
        let forward = rotation.act(F3(1, 0, 0))
        return atan2(forward.y, forward.x)
    }

    private static func horizontalHeading(_ rotation: Quat) -> F3 {
        let angle = headingAngle(rotation)
        return F3(cos(angle), sin(angle), 0)
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
    public var policyActionMirrorSigns: [Float] { Self.mirroredJointSign }

    public func mirrorPolicyActions(_ actions: ContiguousArray<Float>)
        -> ContiguousArray<Float> {
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

    public func mirrorPolicyObservations(_ observations: ContiguousArray<Float>)
        -> ContiguousArray<Float> {
        let dimension = spec.observation.elementCount
        precondition(observations.count.isMultiple(of: dimension))
        var mirrored = observations
        for row in 0..<(observations.count / dimension) {
            let b = row * dimension
            mirrored[b + 1] = -observations[b + 1]
            mirrored[b + 3] = -observations[b + 3]
            mirrored[b + 5] = -observations[b + 5]
            mirrored[b + 7] = -observations[b + 7]
            mirrored[b + 10] = -observations[b + 10]
            mirrored[b + 11] = -observations[b + 11]
            for tensorBase in [12, 31, 50] {
                for j in 0..<actionDimension {
                    mirrored[b + tensorBase + j] =
                        Self.mirroredJointSign[j]
                        * observations[b + tensorBase
                            + Self.mirroredJointSource[j]]
                }
            }
            if configuration.pointGoal {
                mirrored[b + 70] = -observations[b + 70]
                if configuration.recoveryContextObservations {
                    mirrored[b + 71] = -observations[b + 71]
                }
            }
        }
        return mirrored
    }
}
