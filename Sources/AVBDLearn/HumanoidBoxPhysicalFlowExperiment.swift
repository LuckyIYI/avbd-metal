import AVBDCore
import Foundation
import simd

public struct HumanoidBoxPhysicalFlowConfiguration: Sendable {
    public var populationSize: Int
    public var proposalProbeSize: Int
    public var generations: Int
    public var maximumWarmupSteps: Int
    public var contactDwellSteps: Int
    public var targetGenerationSteps: Int
    public var targetDiscoveryPopulationSize: Int
    public var targetDiscoveryGenerations: Int
    public var targetDiscoveryInitialStandardDeviation: Float
    public var carryBaseLegActionFractionOverride: Float?
    public var legBlendKnotCount: Int
    public var minimumTargetCarryDistanceMeters: Float
    public var trajectoryKnotCount: Int
    public var robustReplayCount: Int
    public var initialStandardDeviation: Float
    public var eliteFraction: Float
    public var seed: UInt64

    public init(
        populationSize: Int = 128,
        proposalProbeSize: Int = 32,
        generations: Int = 6,
        maximumWarmupSteps: Int = 600,
        contactDwellSteps: Int = 8,
        targetGenerationSteps: Int = 400,
        targetDiscoveryPopulationSize: Int = 0,
        targetDiscoveryGenerations: Int = 0,
        targetDiscoveryInitialStandardDeviation: Float = 0.25,
        carryBaseLegActionFractionOverride: Float? = nil,
        legBlendKnotCount: Int = 0,
        minimumTargetCarryDistanceMeters: Float = 0,
        trajectoryKnotCount: Int = 5,
        robustReplayCount: Int = 32,
        initialStandardDeviation: Float = 0.25,
        eliteFraction: Float = 0.05,
        seed: UInt64 = 1
    ) {
        self.populationSize = populationSize
        self.proposalProbeSize = proposalProbeSize
        self.generations = generations
        self.maximumWarmupSteps = maximumWarmupSteps
        self.contactDwellSteps = contactDwellSteps
        self.targetGenerationSteps = targetGenerationSteps
        self.targetDiscoveryPopulationSize = targetDiscoveryPopulationSize
        self.targetDiscoveryGenerations = targetDiscoveryGenerations
        self.targetDiscoveryInitialStandardDeviation =
            targetDiscoveryInitialStandardDeviation
        self.carryBaseLegActionFractionOverride =
            carryBaseLegActionFractionOverride
        self.legBlendKnotCount = legBlendKnotCount
        self.minimumTargetCarryDistanceMeters =
            minimumTargetCarryDistanceMeters
        self.trajectoryKnotCount = trajectoryKnotCount
        self.robustReplayCount = robustReplayCount
        self.initialStandardDeviation = initialStandardDeviation
        self.eliteFraction = eliteFraction
        self.seed = seed
    }

    func validate() throws {
        guard populationSize >= 8,
              proposalProbeSize >= 2,
              proposalProbeSize <= populationSize,
              generations > 0,
              maximumWarmupSteps > 0,
              contactDwellSteps > 0,
              targetGenerationSteps >= 8,
              ((targetDiscoveryPopulationSize == 0
                    && targetDiscoveryGenerations == 0)
                || (targetDiscoveryPopulationSize >= 8
                    && targetDiscoveryGenerations > 0)),
              targetDiscoveryInitialStandardDeviation.isFinite,
              targetDiscoveryInitialStandardDeviation > 0,
              (carryBaseLegActionFractionOverride.map {
                  $0.isFinite && (0...1).contains($0)
              } ?? true),
              legBlendKnotCount >= 0,
              minimumTargetCarryDistanceMeters.isFinite,
              minimumTargetCarryDistanceMeters >= 0,
              trajectoryKnotCount > 0,
              robustReplayCount > 0,
              initialStandardDeviation.isFinite,
              initialStandardDeviation > 0,
              eliteFraction > 0,
              eliteFraction <= 0.5 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid humanoid-box physical-flow configuration")
        }
    }
}

/// One simulator-executed segment in a composable physical-state flow.
/// Keeping the complete ordered lineage in every derived artifact prevents a
/// continuation from being mistaken for a standalone action trajectory.
public struct HumanoidBoxPhysicalFlowStage: Codable, Sendable {
    public var trajectory: [Float]
    public var controlSteps: Int
    public var trajectoryDurationSteps: Int?

    public init(
        trajectory: [Float], controlSteps: Int,
        trajectoryDurationSteps: Int? = nil
    ) {
        self.trajectory = trajectory
        self.controlSteps = controlSteps
        self.trajectoryDurationSteps = trajectoryDurationSteps
    }
}

public struct HumanoidBoxPhysicalFlowMetrics: Codable, Sendable {
    public var loss: Float
    public var maximumNormalizedError: Float
    public var rootPositionErrorMeters: Float
    public var rootRotationErrorRadians: Float
    public var rootLinearVelocityErrorMPS: Float
    public var rootAngularVelocityErrorRadPS: Float
    public var jointAngleRMSErrorRadians: Float
    public var jointVelocityRMSErrorRadPS: Float
    public var maximumFootPositionErrorMeters: Float
    public var maximumFootVelocityErrorMPS: Float
    public var boxPositionErrorMeters: Float
    public var boxRotationErrorRadians: Float
    public var boxLinearVelocityErrorMPS: Float
    public var boxAngularVelocityErrorRadPS: Float
    public var maximumHandPositionErrorMeters: Float
    public var maximumHandVelocityErrorMPS: Float
    public var carryDistanceErrorMeters: Float
    public var bilateralHandContact: Bool
    public var unsupported: Bool
    public var physicallyLifted: Bool
    public var robotUpright: Bool
    public var boxUpright: Bool
    public var failed: Bool
    public var minimumCarryDistanceAchieved: Bool

    public var endpointPassed: Bool {
        maximumNormalizedError < 1 && minimumCarryDistanceAchieved
    }
}

public struct HumanoidBoxPhysicalFlowGeneration: Codable, Sendable {
    public var generation: Int
    public var bestLoss: Float
    public var medianLoss: Float
    public var bestMaximumNormalizedError: Float
    public var meanStandardDeviation: Float
    public var warmupControlSteps: Int
}

public struct HumanoidBoxPhysicalFlowReport: Codable, Sendable {
    public var experiment: String
    public var checkpointDirectory: String
    public var seed: UInt64
    public var populationSize: Int
    public var generations: Int
    public var targetTrajectoryWithheldFromSearch: Bool
    public var targetGenerationSteps: Int
    public var targetDiscoveryPopulationSize: Int
    public var targetDiscoveryGenerations: Int
    public var targetDiscoveryCandidateRollouts: Int
    public var targetGeneratingTrajectory: [Float]
    public var carryBaseLegActionFractionOverride: Float?
    public var legBlendKnotCount: Int
    public var sourceTrajectorySteps: Int
    public var sourceStages: [HumanoidBoxPhysicalFlowStage]
    public var sourceReplaySuccessFraction: Float
    public var selectedTargetStep: Int
    public var targetClearanceMeters: Float
    public var targetBoxUprightAlignment: Float
    public var targetRobotUprightAlignment: Float
    public var targetCarryDistanceMeters: Float
    public var targetCloneSuccessFraction: Float
    public var targetReplayMaximumNormalizedError: Float
    public var providedProposal: HumanoidBoxPhysicalFlowMetrics?
    public var zeroProposal: HumanoidBoxPhysicalFlowMetrics
    public var generationZeroSelectedProposal:
        PhysicalFlowProposalSelection
    public var providedProposalProbeBestLoss: Float?
    public var zeroProposalProbeBestLoss: Float?
    public var optimized: HumanoidBoxPhysicalFlowMetrics
    public var optimizedToZeroLossRatio: Float
    public var robustReplaySuccessFraction: Float
    public var robustReplayMedianMaximumNormalizedError: Float
    public var robustReplayWorstMaximumNormalizedError: Float
    public var selectedReplayMaximumNormalizedStateError: Float
    public var bestTrajectory: [Float]
    public var generationHistory: [HumanoidBoxPhysicalFlowGeneration]
    public var candidateRollouts: Int
    public var simulatedEnvironmentControlSteps: Int
    public var elapsedSeconds: Double
    public var infrastructureGatePassed: Bool
    public var targetGatePassed: Bool
    public var targetPlanningGatePassed: Bool
    public var reconstructionGatePassed: Bool
    public var robustReplayGatePassed: Bool
    public var goGatePassed: Bool
}

/// First humanoid scaling bridge for the physical-flow controller. A hidden
/// trajectory creates a real, unsupported H1+box target state. Search receives
/// only that state plus an unrelated stored trajectory as a proposal; every
/// candidate and final replay is executed by the simulator.
public enum HumanoidBoxPhysicalFlowExperiment {
    private static let firstArmAction = 11
    private static let armActionCount = 8

    private struct State {
        var humanoid: HumanoidState
        var manipulation: HumanoidManipulationState
    }

    private struct Flags {
        var bilateral: Bool
        var unsupported: Bool
        var physicallyLifted: Bool
        var robotUpright: Bool
        var boxUpright: Bool
        var failed: Bool
    }

    private struct Candidate {
        var parameters: [Float]
        var metrics: HumanoidBoxPhysicalFlowMetrics
        var terminal: State
        var carryDistance: Float
        var warmupSteps: Int
    }

    private struct TargetDiscoveryCandidate {
        var parameters: [Float]
        var loss: Float
        var maximumNormalizedError: Float
        var bestStep: Int
    }

    private struct Target {
        var state: State
        var flags: Flags
        var step: Int
        var clearance: Float
        var boxUpright: Float
        var robotUpright: Float
        var carryDistance: Float
        var replicas: [State]
        var replicaFlags: [Flags]
        var replicaCarryDistances: [Float]
        var warmupSteps: Int
        var sourceReplaySuccessFraction: Float
    }

    private struct WarmTask {
        var task: HumanoidBoxCarryTask
        var observation: RLObservationBatch
        var result: RLStepBatch
        var warmupSteps: Int
        var sourceReplaySuccessFraction: Float
    }

    public static func run(
        checkpointDirectory: String,
        targetTrajectory: [Float],
        proposalTrajectory: [Float]?,
        sourceStages: [HumanoidBoxPhysicalFlowStage] = [],
        configuration: HumanoidBoxPhysicalFlowConfiguration = .init()
    ) throws -> HumanoidBoxPhysicalFlowReport {
        try configuration.validate()
        let armParameterCount = 4 * configuration.trajectoryKnotCount
        let parameterCount = armParameterCount
            + configuration.legBlendKnotCount

        func normalizedTrajectory(_ values: [Float]) -> [Float]? {
            guard values.allSatisfy(\.isFinite) else { return nil }
            if values.count == parameterCount { return values }
            if values.count == armParameterCount,
               configuration.legBlendKnotCount > 0 {
                return values + [Float](
                    repeating: 0,
                    count: configuration.legBlendKnotCount)
            }
            return nil
        }

        guard let targetTrajectory = normalizedTrajectory(targetTrajectory),
              proposalTrajectory == nil
                || normalizedTrajectory(proposalTrajectory!) != nil,
              sourceStages.allSatisfy({
                  $0.controlSteps > 0
                      && ($0.trajectoryDurationSteps ?? $0.controlSteps) > 0
                      && normalizedTrajectory($0.trajectory) != nil
              }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "humanoid-box flow trajectories do not match the knot schema")
        }
        let proposalTrajectory = proposalTrajectory.flatMap(
            normalizedTrajectory)
        let sourceStages = sourceStages.map {
            HumanoidBoxPhysicalFlowStage(
                trajectory: normalizedTrajectory($0.trajectory)!,
                controlSteps: $0.controlSteps,
                trajectoryDurationSteps: $0.trajectoryDurationSteps)
        }
        if let proposalTrajectory, proposalTrajectory == targetTrajectory {
            throw RLEnvironmentError.invalidConfiguration(
                "target-generating trajectory must be withheld from search")
        }
        if sourceStages.contains(where: {
            $0.trajectory == targetTrajectory
        }) {
            throw RLEnvironmentError.invalidConfiguration(
                "carried target trajectory must differ from the source flow")
        }
        let sourceControlSteps = sourceStages.reduce(0) {
            $0 + $1.controlSteps
        }

        let startTime = Date()
        let runner = try VectorPolicyRunner(
            checkpointDirectory: checkpointDirectory)
        guard runner.metadata.task == "humanoid-box-carry-v0",
              let semanticOptions = runner.metadata.taskConfiguration else {
            throw RLEnvironmentError.invalidConfiguration(
                "humanoid-box flow requires a configured box-carry checkpoint")
        }
        var replayOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: runner.metadata.task,
            semanticOptions: semanticOptions,
            maxEpisodeSteps: runner.metadata.maxEpisodeSteps,
            controlDecimation: runner.metadata.controlDecimation)
        if let fraction = configuration
                .carryBaseLegActionFractionOverride {
            replayOptions["carryBaseLegActionFraction"] = fraction
        }

        func policyActions(
            task: HumanoidBoxCarryTask, observation: RLObservationBatch
        ) throws -> RLActionBatch {
            try runner.actions(
                for: observation,
                expertGates: task.policyExpertGates(observation.policy),
                expertActionMask: task.policyExpertActionMask,
                standExpertGates:
                    task.policyStandExpertGates(observation.policy),
                standExpertActionMask: task.policyStandExpertActionMask,
                auxiliaryExpertGates:
                    task.policyAuxiliaryExpertGates(observation.policy),
                auxiliaryExpertActionMask:
                    task.policyAuxiliaryExpertActionMask)
        }

        func baseLegPolicyActions(
            task: HumanoidBoxCarryTask, observation: RLObservationBatch
        ) throws -> RLActionBatch {
            try runner.actions(
                for: observation,
                expertGates: task.policyExpertGates(observation.policy),
                expertActionMask: task.policyExpertActionMask,
                standExpertGates:
                    task.policyStandExpertGates(observation.policy),
                standExpertActionMask: task.policyStandExpertActionMask,
                auxiliaryExpertGates: ContiguousArray(
                    repeating: 0, count: task.spec.numEnvironments),
                auxiliaryExpertActionMask:
                    task.policyAuxiliaryExpertActionMask)
        }

        func makeWarmTask(count: Int) throws -> WarmTask {
            let anyTask = try BuiltInRLTasks.registry.make(
                runner.metadata.task,
                configuration: RLTaskConfiguration(
                    numEnvironments: count,
                    seed: configuration.seed,
                    autoReset: false,
                    options: replayOptions))
            guard let task = anyTask as? HumanoidBoxCarryTask else {
                throw RLEnvironmentError.invalidConfiguration(
                    "registered box-carry task has an unexpected implementation")
            }
            var compatibilityMetadata = runner.metadata
            if let fraction = configuration
                    .carryBaseLegActionFractionOverride {
                compatibilityMetadata.taskConfiguration?[
                    "carryBaseLegActionFraction"] = fraction
            }
            let compatibility = compatibilityMetadata
                .compatibilityMismatches(with: task.spec)
            guard compatibility.isEmpty else {
                throw RLEnvironmentError.invalidConfiguration(
                    "humanoid-box flow checkpoint/task mismatch: "
                        + compatibility.joined(separator: "; "))
            }
            var observation = try task.reset(seed: configuration.seed)
            var result = RLStepBatch(spec: task.spec)
            var contactStreak = 0
            var warmupSteps = 0
            while warmupSteps < configuration.maximumWarmupSteps,
                  contactStreak < configuration.contactDwellSteps {
                let actions = try policyActions(
                    task: task, observation: observation)
                try task.step(actions: actions, into: &result)
                warmupSteps += 1
                observation = result.observations
                let left = result.metrics["state/left_hand_contact"]!
                let right = result.metrics["state/right_hand_contact"]!
                let allBilateral = zip(left, right).allSatisfy {
                    $0.0 > 0.5 && $0.1 > 0.5
                }
                contactStreak = allBilateral ? contactStreak + 1 : 0
                if result.terminated.contains(true)
                    || result.truncated.contains(true) {
                    break
                }
            }
            guard contactStreak >= configuration.contactDwellSteps else {
                throw RLEnvironmentError.invalidConfiguration(
                    "checkpoint failed to establish the physical-flow source grasp")
            }
            return WarmTask(
                task: task, observation: observation, result: result,
                warmupSteps: warmupSteps,
                sourceReplaySuccessFraction: 1)
        }

        func applyTrajectory(
            _ parameters: [[Float]], step: Int, denominator: Int,
            task: HumanoidBoxCarryTask, observation: RLObservationBatch
        ) throws -> RLActionBatch {
            var actions = try policyActions(task: task, observation: observation)
            let baseLegActions = configuration.legBlendKnotCount > 0
                ? try baseLegPolicyActions(
                    task: task, observation: observation) : nil
            let progress = Float(step + 1) / Float(denominator)
            for environment in parameters.indices {
                let delta = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
                    Array(parameters[environment].prefix(armParameterCount)),
                    knotCount: configuration.trajectoryKnotCount,
                    progress: progress)
                let base = environment * task.spec.action.elementCount
                if let baseLegActions {
                    let blend = legBlendFraction(
                        parameters[environment], progress: progress,
                        armParameterCount: armParameterCount,
                        knotCount: configuration.legBlendKnotCount)
                    for action in 0..<10 {
                        let index = base + action
                        actions.values[index] = (1 - blend)
                            * actions.values[index]
                            + blend * baseLegActions.values[index]
                    }
                }
                for arm in 0..<armActionCount {
                    let index = base + firstArmAction + arm
                    actions.values[index] = simd_clamp(
                        actions.values[index] + delta[arm], -0.999, 0.999)
                }
            }
            return actions
        }

        func states(_ task: HumanoidBoxCarryTask) -> [State] {
            zip(task.environment.states(),
                task.environment.manipulationStates()).map {
                    State(humanoid: $0.0, manipulation: $0.1)
                }
        }

        func flags(
            task: HumanoidBoxCarryTask, observation: RLObservationBatch,
            result: RLStepBatch
        ) -> [Flags] {
            let humanoids = task.environment.states()
            let manipulation = task.environment.manipulationStates()
            let left = result.metrics["state/left_hand_contact"]!
            let right = result.metrics["state/right_hand_contact"]!
            let pedestal = result.metrics["state/box_pedestal_contact"]!
            return (0..<task.spec.numEnvironments).map { environment in
                let rootUp = humanoids[environment].root.rotation
                    .act(F3(0, 0, 1)).z
                let boxUp = manipulation[environment].object.rotation
                    .act(F3(0, 0, 1)).z
                return Flags(
                    bilateral: left[environment] > 0.5
                        && right[environment] > 0.5,
                    unsupported: pedestal[environment] < 0.5,
                    physicallyLifted: observation.policy[
                        environment
                            * HumanoidBoxCarryTask.observationDimension + 89] > 0.5,
                    robotUpright: rootUp > 0.75,
                    boxUpright: boxUp > 0.75,
                    failed: result.terminated[environment]
                        || result.truncated[environment])
            }
        }

        func prepareSource(count: Int) throws -> WarmTask {
            var runtime = try makeWarmTask(count: count)
            for stage in sourceStages {
                let repeated = [[Float]](
                    repeating: stage.trajectory, count: count)
                for step in 0..<stage.controlSteps {
                    let actions = try applyTrajectory(
                        repeated, step: step,
                        denominator: stage.trajectoryDurationSteps
                            ?? stage.controlSteps,
                        task: runtime.task,
                        observation: runtime.observation)
                    try runtime.task.step(
                        actions: actions, into: &runtime.result)
                    runtime.observation = runtime.result.observations
                }
            }
            guard !sourceStages.isEmpty else { return runtime }
            let sourceFlags = flags(
                task: runtime.task, observation: runtime.observation,
                result: runtime.result)
            let successes = sourceFlags.filter {
                $0.bilateral && $0.unsupported && $0.physicallyLifted
                    && $0.robotUpright && $0.boxUpright && !$0.failed
            }.count
            runtime.sourceReplaySuccessFraction = Float(successes)
                / Float(sourceFlags.count)
            guard runtime.sourceReplaySuccessFraction >= 0.8 else {
                throw RLEnvironmentError.invalidConfiguration(
                    String(format:
                        "source physical flow replay passed only %.1f%% of replicas",
                        100 * runtime.sourceReplaySuccessFraction))
            }
            return runtime
        }

        func generateTarget(using trajectory: [Float]) throws -> Target {
            let count = min(32, configuration.populationSize)
            var runtime = try prepareSource(count: count)
            var best: Target?
            var maximumClearance: Float = -.infinity
            var maximumCarryDistance: Float = 0
            var maximumStableCarryDistance: Float = 0
            var everLifted = false
            let repeated = [[Float]](
                repeating: trajectory, count: count)
            for step in 0..<configuration.targetGenerationSteps {
                let actions = try applyTrajectory(
                    repeated, step: step,
                    denominator: configuration.targetGenerationSteps,
                    task: runtime.task, observation: runtime.observation)
                try runtime.task.step(actions: actions, into: &runtime.result)
                runtime.observation = runtime.result.observations
                let allStates = states(runtime.task)
                let allFlags = flags(
                    task: runtime.task, observation: runtime.observation,
                    result: runtime.result)
                let clearances = runtime.result.metrics[
                    "state/box_clearance_m"]!
                let carryDistances = runtime.result.metrics[
                    "state/carry_distance_m"]!
                maximumClearance = max(maximumClearance, clearances[0])
                maximumCarryDistance = max(
                    maximumCarryDistance, carryDistances[0])
                everLifted = everLifted || allFlags[0].physicallyLifted
                let boxUp = allStates[0].manipulation.object.rotation
                    .act(F3(0, 0, 1)).z
                let robotUp = allStates[0].humanoid.root.rotation
                    .act(F3(0, 0, 1)).z
                let stableCarryManifold = allFlags[0].bilateral
                    && allFlags[0].unsupported
                    && allFlags[0].physicallyLifted
                    // The bridge target should be visually and mechanically
                    // useful, not merely below the looser fall guard used for
                    // candidate rejection.
                    && robotUp > 0.9
                    && boxUp > 0.9
                    && !allFlags[0].failed
                    && clearances[0] >= 0.01
                if stableCarryManifold {
                    maximumStableCarryDistance = max(
                        maximumStableCarryDistance, carryDistances[0])
                }
                let feasible = stableCarryManifold
                    && carryDistances[0]
                        >= configuration.minimumTargetCarryDistanceMeters
                guard feasible else { continue }
                let object = allStates[0].manipulation.object
                let root = allStates[0].humanoid.root
                let selection = PhysicalFlowBalancedObjective.evaluate(
                    normalizedErrors: [
                        max(0.04 - clearances[0], 0) / 0.04,
                        length(object.linearVelocity) / 0.15,
                        length(object.angularVelocity) / 0.50,
                        length(root.linearVelocity) / 0.30,
                        length(root.angularVelocity) / 0.60,
                        max(0.9 - boxUp, 0) / 0.15,
                        max(0.9 - robotUp, 0) / 0.15,
                        max(configuration.minimumTargetCarryDistanceMeters
                            - carryDistances[0], 0)
                            / max(configuration.minimumTargetCarryDistanceMeters,
                                  0.05),
                    ])
                if best == nil || selection.bottleneckLoss
                    < targetSelectionLoss(best!) {
                    best = Target(
                        state: allStates[0], flags: allFlags[0],
                        step: step + 1, clearance: clearances[0],
                        boxUpright: boxUp, robotUpright: robotUp,
                        carryDistance: carryDistances[0],
                        replicas: allStates, replicaFlags: allFlags,
                        replicaCarryDistances: Array(carryDistances),
                        warmupSteps: runtime.warmupSteps,
                        sourceReplaySuccessFraction:
                            runtime.sourceReplaySuccessFraction)
                }
            }
            guard let best else {
                throw RLEnvironmentError.invalidConfiguration(
                    String(format:
                        "hidden trajectory produced no upright unsupported target (lifted %d, max clearance %.4f m, max carry %.4f m, max stable carry %.4f m, required carry %.4f m)",
                        everLifted ? 1 : 0, maximumClearance,
                        maximumCarryDistance, maximumStableCarryDistance,
                        configuration.minimumTargetCarryDistanceMeters))
            }
            return best
        }

        func targetSelectionLoss(_ target: Target) -> Float {
            let object = target.state.manipulation.object
            let root = target.state.humanoid.root
            return PhysicalFlowBalancedObjective.evaluate(
                normalizedErrors: [
                    max(0.04 - target.clearance, 0) / 0.04,
                    length(object.linearVelocity) / 0.15,
                    length(object.angularVelocity) / 0.50,
                    length(root.linearVelocity) / 0.30,
                    length(root.angularVelocity) / 0.60,
                    max(0.9 - target.boxUpright, 0) / 0.15,
                    max(0.9 - target.robotUpright, 0) / 0.15,
                    max(configuration.minimumTargetCarryDistanceMeters
                        - target.carryDistance, 0)
                        / max(configuration.minimumTargetCarryDistanceMeters,
                              0.05),
                ]).bottleneckLoss
        }

        var simulatedEnvironmentControlSteps = 0
        var targetDiscoveryCandidateRollouts = 0
        var selectedTargetTrajectory = targetTrajectory

        if configuration.targetDiscoveryPopulationSize > 0 {
            let count = configuration.targetDiscoveryPopulationSize
            var generator = ProbeRandomNumberGenerator(
                seed: configuration.seed &+ 0xD15C_0A3E)
            var mean = targetTrajectory
            var covariance = [[Float]](
                repeating: [Float](repeating: 0, count: parameterCount),
                count: parameterCount)
            for index in 0..<parameterCount {
                covariance[index][index] =
                    configuration.targetDiscoveryInitialStandardDeviation
                    * configuration.targetDiscoveryInitialStandardDeviation
            }
            var overallBest: TargetDiscoveryCandidate?

            func discoverySample(
                around center: [Float], transform: [[Float]]
            ) -> [Float] {
                let noise = center.map { _ in generator.normal() }
                return center.indices.map { row in
                    let delta = (0...row).reduce(Float(0)) {
                        $0 + transform[row][$1] * noise[$1]
                    }
                    return simd_clamp(center[row] + delta, -0.999, 0.999)
                }
            }

            func evaluateDiscovery(
                _ parameters: [[Float]]
            ) throws -> [TargetDiscoveryCandidate] {
                var runtime = try prepareSource(count: parameters.count)
                var bestLosses = [Float](
                    repeating: .infinity, count: parameters.count)
                var bestErrors = [Float](
                    repeating: .infinity, count: parameters.count)
                var bestSteps = [Int](repeating: 0, count: parameters.count)
                for step in 0..<configuration.targetGenerationSteps {
                    let actions = try applyTrajectory(
                        parameters, step: step,
                        denominator: configuration.targetGenerationSteps,
                        task: runtime.task,
                        observation: runtime.observation)
                    try runtime.task.step(
                        actions: actions, into: &runtime.result)
                    runtime.observation = runtime.result.observations
                    let allStates = states(runtime.task)
                    let allFlags = flags(
                        task: runtime.task,
                        observation: runtime.observation,
                        result: runtime.result)
                    let clearances = runtime.result.metrics[
                        "state/box_clearance_m"]!
                    let carryDistances = runtime.result.metrics[
                        "state/carry_distance_m"]!
                    for environment in parameters.indices {
                        let evaluation = frontierEvaluation(
                            state: allStates[environment],
                            flags: allFlags[environment],
                            clearance: clearances[environment],
                            carryDistance: carryDistances[environment],
                            minimumCarryDistance:
                                configuration.minimumTargetCarryDistanceMeters)
                        if evaluation.bottleneckLoss
                            < bestLosses[environment] {
                            bestLosses[environment] =
                                evaluation.bottleneckLoss
                            bestErrors[environment] =
                                evaluation.maximumNormalizedError
                            bestSteps[environment] = step + 1
                        }
                    }
                }
                simulatedEnvironmentControlSteps += parameters.count
                    * (runtime.warmupSteps + sourceControlSteps
                        + configuration.targetGenerationSteps)
                targetDiscoveryCandidateRollouts += parameters.count
                return parameters.indices.map {
                    TargetDiscoveryCandidate(
                        parameters: parameters[$0],
                        loss: bestLosses[$0],
                        maximumNormalizedError: bestErrors[$0],
                        bestStep: bestSteps[$0])
                }
            }

            for _ in 0..<configuration.targetDiscoveryGenerations {
                let transform = cholesky(covariance)
                var parameters = [[Float]]()
                parameters.reserveCapacity(count)
                for index in 0..<count {
                    if index == 0 {
                        parameters.append(mean)
                    } else if index == 1, let overallBest {
                        parameters.append(overallBest.parameters)
                    } else {
                        parameters.append(discoverySample(
                            around: mean, transform: transform))
                    }
                }
                let candidates = try evaluateDiscovery(parameters).sorted {
                    $0.loss < $1.loss
                }
                guard let generationBest = candidates.first,
                      generationBest.loss.isFinite else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "non-finite humanoid-box target discovery")
                }
                if overallBest == nil
                    || generationBest.loss < overallBest!.loss {
                    overallBest = generationBest
                }
                let eliteCount = max(
                    2, Int(Float(count) * configuration.eliteFraction))
                let elites = candidates.prefix(eliteCount)
                var nextMean = [Float](
                    repeating: 0, count: parameterCount)
                for elite in elites {
                    for index in 0..<parameterCount {
                        nextMean[index] += elite.parameters[index]
                    }
                }
                for index in 0..<parameterCount {
                    nextMean[index] /= Float(eliteCount)
                }
                var nextCovariance = [[Float]](
                    repeating: [Float](
                        repeating: 0, count: parameterCount),
                    count: parameterCount)
                for elite in elites {
                    for row in 0..<parameterCount {
                        for column in 0..<parameterCount {
                            nextCovariance[row][column] +=
                                (elite.parameters[row] - nextMean[row])
                                * (elite.parameters[column]
                                    - nextMean[column])
                        }
                    }
                }
                for row in 0..<parameterCount {
                    for column in 0..<parameterCount {
                        nextCovariance[row][column] /= Float(eliteCount)
                    }
                    nextCovariance[row][row] = max(
                        nextCovariance[row][row], 0.02 * 0.02)
                }
                mean = zip(mean, nextMean).map {
                    0.25 * $0.0 + 0.75 * $0.1
                }
                for row in 0..<parameterCount {
                    for column in 0..<parameterCount {
                        covariance[row][column] = 0.25
                            * covariance[row][column]
                            + 0.75 * nextCovariance[row][column]
                    }
                }
            }
            guard let discovered = overallBest else {
                throw RLEnvironmentError.invalidConfiguration(
                    "humanoid-box target discovery produced no candidate")
            }
            selectedTargetTrajectory = discovered.parameters
        }

        let target = try generateTarget(using: selectedTargetTrajectory)
        simulatedEnvironmentControlSteps += target.replicas.count
            * (target.warmupSteps + sourceControlSteps
                + configuration.targetGenerationSteps)

        func evaluate(_ parameters: [[Float]]) throws -> [Candidate] {
            precondition(!parameters.isEmpty)
            var runtime = try prepareSource(count: parameters.count)
            for step in 0..<target.step {
                let actions = try applyTrajectory(
                    parameters, step: step, denominator: target.step,
                    task: runtime.task, observation: runtime.observation)
                try runtime.task.step(actions: actions, into: &runtime.result)
                runtime.observation = runtime.result.observations
            }
            simulatedEnvironmentControlSteps += parameters.count
                * (runtime.warmupSteps + sourceControlSteps
                    + target.step)
            let terminals = states(runtime.task)
            let terminalFlags = flags(
                task: runtime.task, observation: runtime.observation,
                result: runtime.result)
            let carryDistances = runtime.result.metrics[
                "state/carry_distance_m"]!
            return parameters.indices.map { index in
                Candidate(
                    parameters: parameters[index],
                    metrics: metrics(
                        from: terminals[index], to: target.state,
                        flags: terminalFlags[index],
                        carryDistance: carryDistances[index],
                        targetCarryDistance: target.carryDistance,
                        minimumCarryDistance:
                            configuration.minimumTargetCarryDistanceMeters),
                    terminal: terminals[index],
                    carryDistance: carryDistances[index],
                    warmupSteps: runtime.warmupSteps)
            }
        }

        // Replay the hidden target action only for an infrastructure audit. It
        // remains excluded from every search center, sample, and elite update.
        var targetReplayRuntime = try prepareSource(
            count: target.replicas.count)
        let repeatedTarget = [[Float]](
            repeating: selectedTargetTrajectory,
            count: target.replicas.count)
        for step in 0..<target.step {
            let actions = try applyTrajectory(
                repeatedTarget, step: step,
                denominator: configuration.targetGenerationSteps,
                task: targetReplayRuntime.task,
                observation: targetReplayRuntime.observation)
            try targetReplayRuntime.task.step(
                actions: actions, into: &targetReplayRuntime.result)
            targetReplayRuntime.observation =
                targetReplayRuntime.result.observations
        }
        simulatedEnvironmentControlSteps += target.replicas.count
            * (targetReplayRuntime.warmupSteps
                + sourceControlSteps + target.step)
        let targetReplayStates = states(targetReplayRuntime.task)
        let targetReplayFlags = flags(
            task: targetReplayRuntime.task,
            observation: targetReplayRuntime.observation,
            result: targetReplayRuntime.result)
        let targetReplayCarryDistances = targetReplayRuntime.result.metrics[
            "state/carry_distance_m"]!
        let targetReplayMetrics = metrics(
            from: targetReplayStates[0], to: target.state,
            flags: targetReplayFlags[0],
            carryDistance: targetReplayCarryDistances[0],
            targetCarryDistance: target.carryDistance,
            minimumCarryDistance:
                configuration.minimumTargetCarryDistanceMeters)
        let targetCloneMetrics = target.replicas.indices.map { index in
            metrics(
                from: target.replicas[index], to: target.state,
                flags: target.replicaFlags[index],
                carryDistance: target.replicaCarryDistances[index],
                targetCarryDistance: target.carryDistance,
                minimumCarryDistance:
                    configuration.minimumTargetCarryDistanceMeters)
        }
        let targetCloneSuccessFraction = Float(targetCloneMetrics.filter {
            $0.endpointPassed
        }.count) / Float(targetCloneMetrics.count)

        var generator = ProbeRandomNumberGenerator(
            seed: configuration.seed &+ 0xF10A)
        let zero = [Float](repeating: 0, count: parameterCount)
        var searchMean = proposalTrajectory ?? zero
        var covariance = [[Float]](
            repeating: [Float](repeating: 0, count: parameterCount),
            count: parameterCount)
        for index in 0..<parameterCount {
            covariance[index][index] = configuration.initialStandardDeviation
                * configuration.initialStandardDeviation
        }
        var overallBest: Candidate?
        var initialProvidedMetrics: HumanoidBoxPhysicalFlowMetrics?
        var zeroMetrics: HumanoidBoxPhysicalFlowMetrics?
        var selectedProposal: PhysicalFlowProposalSelection =
            proposalTrajectory == nil ? .notApplicable : .provided
        var providedProbeBestLoss: Float?
        var zeroProbeBestLoss: Float?
        var histories = [HumanoidBoxPhysicalFlowGeneration]()
        var candidateRollouts = 0

        func sample(around center: [Float], transform: [[Float]]) -> [Float] {
            let noise = center.map { _ in generator.normal() }
            return center.indices.map { row in
                let delta = (0...row).reduce(Float(0)) {
                    $0 + transform[row][$1] * noise[$1]
                }
                return simd_clamp(center[row] + delta, -0.999, 0.999)
            }
        }

        for generation in 0..<configuration.generations {
            let transform = cholesky(covariance)
            let candidates: [Candidate]
            if generation == 0, let proposalTrajectory {
                let maximumProbe = min(
                    configuration.proposalProbeSize,
                    configuration.populationSize)
                let probeCount = max(2, maximumProbe - maximumProbe % 2)
                var modes = [PhysicalFlowProposalSelection]()
                let probeParameters = (0..<probeCount).map { index -> [Float] in
                    let mode: PhysicalFlowProposalSelection =
                        index.isMultiple(of: 2) ? .provided : .geometric
                    modes.append(mode)
                    let center = mode == .provided ? proposalTrajectory : zero
                    return index < 2 ? center
                        : sample(around: center, transform: transform)
                }
                let probe = try evaluate(probeParameters)
                candidateRollouts += probe.count
                initialProvidedMetrics = probe[0].metrics
                zeroMetrics = probe[1].metrics
                var bestProvided = probe[0]
                var bestZero = probe[1]
                for index in probe.indices {
                    if modes[index] == .provided,
                       probe[index].metrics.loss < bestProvided.metrics.loss {
                        bestProvided = probe[index]
                    } else if modes[index] == .geometric,
                              probe[index].metrics.loss
                                < bestZero.metrics.loss {
                        bestZero = probe[index]
                    }
                }
                providedProbeBestLoss = bestProvided.metrics.loss
                zeroProbeBestLoss = bestZero.metrics.loss
                let winner: Candidate
                if bestProvided.metrics.loss <= bestZero.metrics.loss {
                    selectedProposal = .provided
                    winner = bestProvided
                } else {
                    selectedProposal = .geometric
                    winner = bestZero
                }
                let remaining = configuration.populationSize - probeCount
                if remaining > 0 {
                    let exploitation = (0..<remaining).map { _ in
                        sample(around: winner.parameters, transform: transform)
                    }
                    let evaluated = try evaluate(exploitation)
                    candidateRollouts += evaluated.count
                    candidates = probe + evaluated
                } else {
                    candidates = probe
                }
            } else {
                var parameters = [[Float]]()
                parameters.reserveCapacity(configuration.populationSize)
                for index in 0..<configuration.populationSize {
                    if index == 0 {
                        parameters.append(searchMean)
                    } else if index == 1, let overallBest {
                        parameters.append(overallBest.parameters)
                    } else {
                        parameters.append(sample(
                            around: searchMean, transform: transform))
                    }
                }
                candidates = try evaluate(parameters)
                candidateRollouts += candidates.count
                if generation == 0 {
                    zeroMetrics = candidates[0].metrics
                }
            }
            guard candidates.allSatisfy({ $0.metrics.loss.isFinite }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "non-finite humanoid-box physical-flow candidate")
            }
            let sorted = candidates.sorted {
                $0.metrics.loss < $1.metrics.loss
            }
            if overallBest == nil
                || sorted[0].metrics.loss < overallBest!.metrics.loss {
                overallBest = sorted[0]
            }
            let eliteCount = max(
                2, Int(Float(configuration.populationSize)
                    * configuration.eliteFraction))
            let elites = sorted.prefix(eliteCount)
            var nextMean = [Float](repeating: 0, count: parameterCount)
            for elite in elites {
                for index in 0..<parameterCount {
                    nextMean[index] += elite.parameters[index]
                }
            }
            for index in 0..<parameterCount {
                nextMean[index] /= Float(eliteCount)
            }
            var nextCovariance = [[Float]](
                repeating: [Float](repeating: 0, count: parameterCount),
                count: parameterCount)
            for elite in elites {
                for row in 0..<parameterCount {
                    for column in 0..<parameterCount {
                        nextCovariance[row][column] +=
                            (elite.parameters[row] - nextMean[row])
                            * (elite.parameters[column] - nextMean[column])
                    }
                }
            }
            for row in 0..<parameterCount {
                for column in 0..<parameterCount {
                    nextCovariance[row][column] /= Float(eliteCount)
                }
                nextCovariance[row][row] = max(
                    nextCovariance[row][row], 0.02 * 0.02)
            }
            searchMean = zip(searchMean, nextMean).map {
                0.25 * $0.0 + 0.75 * $0.1
            }
            for row in 0..<parameterCount {
                for column in 0..<parameterCount {
                    covariance[row][column] = 0.25
                        * covariance[row][column]
                        + 0.75 * nextCovariance[row][column]
                }
            }
            let losses = candidates.map(\.metrics.loss).sorted()
            histories.append(.init(
                generation: generation,
                bestLoss: sorted[0].metrics.loss,
                medianLoss: losses[losses.count / 2],
                bestMaximumNormalizedError:
                    sorted[0].metrics.maximumNormalizedError,
                meanStandardDeviation: (0..<parameterCount).reduce(0) {
                    $0 + sqrt(max(covariance[$1][$1], 0))
                } / Float(parameterCount),
                warmupControlSteps: sorted[0].warmupSteps))
        }

        let best = overallBest!
        let robustParameters = [[Float]](
            repeating: best.parameters,
            count: configuration.robustReplayCount)
        let robustCandidates = try evaluate(robustParameters)
        let robustErrors = robustCandidates.map {
            $0.metrics.maximumNormalizedError
        }.sorted()
        let robustSuccessFraction = Float(robustCandidates.filter {
            $0.metrics.endpointPassed
        }.count) / Float(robustCandidates.count)
        let selectedReplayStateError = stateEndpointEvaluation(
            from: robustCandidates[0].terminal, to: best.terminal)
            .maximumNormalizedError
        let infrastructureGatePassed =
            targetReplayMetrics.maximumNormalizedError < 0.02
                && selectedReplayStateError < 0.02
        let targetGatePassed = target.flags.bilateral
            && target.flags.unsupported && target.flags.physicallyLifted
            && target.flags.robotUpright && target.flags.boxUpright
            && target.robotUpright > 0.9 && target.boxUpright > 0.9
            && target.clearance >= 0.01
            && target.carryDistance
                >= configuration.minimumTargetCarryDistanceMeters
        let reconstructionGatePassed = best.metrics.endpointPassed
        let robustReplayGatePassed = robustSuccessFraction >= 0.8
        let targetPlanningGatePassed = targetGatePassed
            && targetCloneSuccessFraction >= 0.8
            && targetReplayMetrics.maximumNormalizedError < 0.02
        let zeroLoss = max(zeroMetrics!.loss, 1e-12)
        let targetTrajectoryWasWithheld = selectedTargetTrajectory != zero
            && selectedTargetTrajectory != proposalTrajectory

        return HumanoidBoxPhysicalFlowReport(
            experiment: "humanoid-box-state-to-state-physical-flow-v0",
            checkpointDirectory: checkpointDirectory,
            seed: configuration.seed,
            populationSize: configuration.populationSize,
            generations: configuration.generations,
            targetTrajectoryWithheldFromSearch:
                targetTrajectoryWasWithheld,
            targetGenerationSteps: configuration.targetGenerationSteps,
            targetDiscoveryPopulationSize:
                configuration.targetDiscoveryPopulationSize,
            targetDiscoveryGenerations:
                configuration.targetDiscoveryGenerations,
            targetDiscoveryCandidateRollouts:
                targetDiscoveryCandidateRollouts,
            targetGeneratingTrajectory: selectedTargetTrajectory,
            carryBaseLegActionFractionOverride:
                configuration.carryBaseLegActionFractionOverride,
            legBlendKnotCount: configuration.legBlendKnotCount,
            sourceTrajectorySteps: sourceControlSteps,
            sourceStages: sourceStages,
            sourceReplaySuccessFraction:
                target.sourceReplaySuccessFraction,
            selectedTargetStep: target.step,
            targetClearanceMeters: target.clearance,
            targetBoxUprightAlignment: target.boxUpright,
            targetRobotUprightAlignment: target.robotUpright,
            targetCarryDistanceMeters: target.carryDistance,
            targetCloneSuccessFraction: targetCloneSuccessFraction,
            targetReplayMaximumNormalizedError:
                targetReplayMetrics.maximumNormalizedError,
            providedProposal: initialProvidedMetrics,
            zeroProposal: zeroMetrics!,
            generationZeroSelectedProposal: selectedProposal,
            providedProposalProbeBestLoss: providedProbeBestLoss,
            zeroProposalProbeBestLoss: zeroProbeBestLoss,
            optimized: best.metrics,
            optimizedToZeroLossRatio: best.metrics.loss / zeroLoss,
            robustReplaySuccessFraction: robustSuccessFraction,
            robustReplayMedianMaximumNormalizedError:
                robustErrors[robustErrors.count / 2],
            robustReplayWorstMaximumNormalizedError: robustErrors.last!,
            selectedReplayMaximumNormalizedStateError:
                selectedReplayStateError,
            bestTrajectory: best.parameters,
            generationHistory: histories,
            candidateRollouts: candidateRollouts
                + configuration.robustReplayCount,
            simulatedEnvironmentControlSteps:
                simulatedEnvironmentControlSteps,
            elapsedSeconds: Date().timeIntervalSince(startTime),
            infrastructureGatePassed: infrastructureGatePassed,
            targetGatePassed: targetGatePassed,
            targetPlanningGatePassed: targetPlanningGatePassed,
            reconstructionGatePassed: reconstructionGatePassed,
            robustReplayGatePassed: robustReplayGatePassed,
            goGatePassed: infrastructureGatePassed && targetGatePassed
                && reconstructionGatePassed && robustReplayGatePassed)
    }

    static func legBlendFraction(
        _ parameters: [Float], progress: Float,
        armParameterCount: Int, knotCount: Int
    ) -> Float {
        precondition(knotCount > 0)
        precondition(parameters.count >= armParameterCount + knotCount)
        let scaled = simd_clamp(progress, 0, 1) * Float(knotCount)
        let lower = min(Int(floor(scaled)), knotCount - 1)
        let fraction = scaled - Float(lower)
        func knot(_ index: Int) -> Float {
            guard index > 0 else { return 0 }
            return simd_clamp(
                parameters[armParameterCount + index - 1], 0, 1)
        }
        return (1 - fraction) * knot(lower)
            + fraction * knot(min(lower + 1, knotCount))
    }

    /// Goal-set score used only to discover a new physically valid frontier
    /// state. The subsequent reconstruction still targets the exact selected
    /// simulator state and never receives its generating trajectory.
    private static func frontierEvaluation(
        state: State, flags: Flags, clearance: Float,
        carryDistance: Float, minimumCarryDistance: Float
    ) -> PhysicalFlowEndpointEvaluation {
        let rootUp = state.humanoid.root.rotation
            .act(F3(0, 0, 1)).z
        let boxUp = state.manipulation.object.rotation
            .act(F3(0, 0, 1)).z
        return PhysicalFlowBalancedObjective.evaluate(normalizedErrors: [
            max(minimumCarryDistance - carryDistance, 0) / 0.02,
            max(0.025 - clearance, 0) / 0.02,
            length(state.manipulation.object.linearVelocity) / 0.20,
            length(state.manipulation.object.angularVelocity) / 0.60,
            length(state.humanoid.root.linearVelocity) / 0.40,
            length(state.humanoid.root.angularVelocity) / 0.80,
            max(0.9 - rootUp, 0) / 0.10,
            max(0.9 - boxUp, 0) / 0.10,
            // These are feasibility barriers, not tradeable rewards. A box
            // flying toward the goal after contact loss must never outrank a
            // shorter but valid carry state.
            flags.bilateral ? 0 : 10,
            flags.unsupported ? 0 : 10,
            flags.physicallyLifted ? 0 : 10,
            rootUp > 0.9 ? 0 : 10,
            boxUp > 0.9 ? 0 : 10,
            clearance >= 0.01 ? 0 : 10,
            flags.failed ? 20 : 0,
        ])
    }

    private static func metrics(
        from state: State, to target: State, flags: Flags,
        carryDistance: Float, targetCarryDistance: Float,
        minimumCarryDistance: Float
    ) -> HumanoidBoxPhysicalFlowMetrics {
        let endpoint = stateEndpointEvaluation(from: state, to: target)
        var errors = endpoint.normalizedErrors
        let carryDistanceError = abs(carryDistance - targetCarryDistance)
        errors.append(carryDistanceError / 0.05)
        errors.append(flags.bilateral ? 0 : 2)
        errors.append(flags.unsupported ? 0 : 2)
        errors.append(flags.physicallyLifted ? 0 : 2)
        errors.append(flags.robotUpright ? 0 : 2)
        errors.append(flags.boxUpright ? 0 : 2)
        errors.append(flags.failed ? 4 : 0)
        let minimumCarryAchieved = carryDistance >= minimumCarryDistance
        // Keep a smooth search signal up to the exact task boundary. The
        // endpoint classifier below still requires the unrelaxed Boolean
        // minimum, so a near miss can guide CEM but can never pass replay.
        errors.append(max(minimumCarryDistance - carryDistance, 0) / 0.01)
        let constrained = PhysicalFlowBalancedObjective.evaluate(
            normalizedErrors: errors)
        let components = stateErrors(from: state, to: target)
        return HumanoidBoxPhysicalFlowMetrics(
            loss: constrained.bottleneckLoss,
            maximumNormalizedError: constrained.maximumNormalizedError,
            rootPositionErrorMeters: components[0],
            rootRotationErrorRadians: components[1],
            rootLinearVelocityErrorMPS: components[2],
            rootAngularVelocityErrorRadPS: components[3],
            jointAngleRMSErrorRadians: components[4],
            jointVelocityRMSErrorRadPS: components[5],
            maximumFootPositionErrorMeters: components[6],
            maximumFootVelocityErrorMPS: components[7],
            boxPositionErrorMeters: components[8],
            boxRotationErrorRadians: components[9],
            boxLinearVelocityErrorMPS: components[10],
            boxAngularVelocityErrorRadPS: components[11],
            maximumHandPositionErrorMeters: components[12],
            maximumHandVelocityErrorMPS: components[13],
            carryDistanceErrorMeters: carryDistanceError,
            bilateralHandContact: flags.bilateral,
            unsupported: flags.unsupported,
            physicallyLifted: flags.physicallyLifted,
            robotUpright: flags.robotUpright,
            boxUpright: flags.boxUpright,
            failed: flags.failed,
            minimumCarryDistanceAchieved: minimumCarryAchieved)
    }

    private static func stateEndpointEvaluation(
        from state: State, to target: State
    ) -> PhysicalFlowEndpointEvaluation {
        let e = stateErrors(from: state, to: target)
        let thresholds: [Float] = [
            0.05, 0.12, 0.15, 0.30,
            0.12, 0.50, 0.06, 0.25,
            0.04, 0.12, 0.12, 0.40,
            0.05, 0.25,
        ]
        return PhysicalFlowBalancedObjective.evaluate(
            normalizedErrors: zip(e, thresholds).map { $0 / $1 })
    }

    private static func stateErrors(
        from state: State, to target: State
    ) -> [Float] {
        func rmse(_ lhs: [Float], _ rhs: [Float]) -> Float {
            sqrt(zip(lhs, rhs).reduce(0) {
                let d = $1.0 - $1.1
                return $0 + d * d
            } / Float(lhs.count))
        }
        func maximum(_ values: Float...) -> Float { values.max() ?? 0 }
        let h = state.humanoid, t = target.humanoid
        let m = state.manipulation, tm = target.manipulation
        return [
            length(h.root.position - t.root.position),
            rotationError(h.root.rotation, t.root.rotation),
            length(h.root.linearVelocity - t.root.linearVelocity),
            length(h.root.angularVelocity - t.root.angularVelocity),
            rmse(h.jointAngles, t.jointAngles),
            rmse(h.jointVelocities, t.jointVelocities),
            maximum(
                length(h.leftFoot.position - t.leftFoot.position),
                length(h.rightFoot.position - t.rightFoot.position)),
            maximum(
                length(h.leftFoot.linearVelocity
                    - t.leftFoot.linearVelocity),
                length(h.rightFoot.linearVelocity
                    - t.rightFoot.linearVelocity)),
            length(m.object.position - tm.object.position),
            rotationError(m.object.rotation, tm.object.rotation),
            length(m.object.linearVelocity - tm.object.linearVelocity),
            length(m.object.angularVelocity - tm.object.angularVelocity),
            maximum(
                length(m.leftHand.position - tm.leftHand.position),
                length(m.rightHand.position - tm.rightHand.position)),
            maximum(
                length(m.leftHand.linearVelocity
                    - tm.leftHand.linearVelocity),
                length(m.rightHand.linearVelocity
                    - tm.rightHand.linearVelocity)),
        ]
    }

    private static func rotationError(
        _ lhs: simd_quatf, _ rhs: simd_quatf
    ) -> Float {
        let cosine = simd_clamp(abs(simd_dot(lhs.vector, rhs.vector)), 0, 1)
        return 2 * acos(cosine)
    }

    private static func cholesky(_ covariance: [[Float]]) -> [[Float]] {
        let count = covariance.count
        var lower = [[Float]](
            repeating: [Float](repeating: 0, count: count), count: count)
        for row in 0..<count {
            for column in 0...row {
                var value = covariance[row][column]
                for k in 0..<column {
                    value -= lower[row][k] * lower[column][k]
                }
                if row == column {
                    lower[row][column] = sqrt(max(value, 1e-8))
                } else {
                    lower[row][column] = value
                        / max(lower[column][column], 1e-8)
                }
            }
        }
        return lower
    }
}
