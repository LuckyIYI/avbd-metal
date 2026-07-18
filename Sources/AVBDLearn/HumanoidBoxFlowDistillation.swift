import AVBDCore
import Foundation
import MLX
import MLXNN
import simd

public struct HumanoidBoxFlowDistillationConfiguration: Sendable {
    public var collectionEnvironments: Int
    public var contactDwellSteps: Int
    public var epochs: Int
    public var learningRate: Float
    public var maximumGradientNorm: Float
    public var targetRowWeight: Float
    public var policySourceRowWeight: Float
    public var aggregationRounds: Int
    public var initialTeacherMix: Float
    public var finalTeacherMix: Float
    public var rolloutActionNoiseStandardDeviation: Float
    public var stateAlignmentLookahead: Int
    public var validationActionNoiseStandardDeviation: Float
    public var validationNoiseIncludesSource: Bool
    public var robustReplayCount: Int
    public var seed: UInt64

    public init(
        collectionEnvironments: Int = 256,
        contactDwellSteps: Int = 8,
        epochs: Int = 200,
        learningRate: Float = 1e-4,
        maximumGradientNorm: Float = 1,
        targetRowWeight: Float = 4,
        policySourceRowWeight: Float = 4,
        aggregationRounds: Int = 4,
        initialTeacherMix: Float = 0.75,
        finalTeacherMix: Float = 0,
        rolloutActionNoiseStandardDeviation: Float = 0.005,
        stateAlignmentLookahead: Int = 12,
        validationActionNoiseStandardDeviation: Float = 0.001,
        validationNoiseIncludesSource: Bool = false,
        robustReplayCount: Int = 32,
        seed: UInt64 = 1
    ) {
        self.collectionEnvironments = collectionEnvironments
        self.contactDwellSteps = contactDwellSteps
        self.epochs = epochs
        self.learningRate = learningRate
        self.maximumGradientNorm = maximumGradientNorm
        self.targetRowWeight = targetRowWeight
        self.policySourceRowWeight = policySourceRowWeight
        self.aggregationRounds = aggregationRounds
        self.initialTeacherMix = initialTeacherMix
        self.finalTeacherMix = finalTeacherMix
        self.rolloutActionNoiseStandardDeviation =
            rolloutActionNoiseStandardDeviation
        self.stateAlignmentLookahead = stateAlignmentLookahead
        self.validationActionNoiseStandardDeviation =
            validationActionNoiseStandardDeviation
        self.validationNoiseIncludesSource = validationNoiseIncludesSource
        self.robustReplayCount = robustReplayCount
        self.seed = seed
    }

    func validate() throws {
        guard collectionEnvironments > 0, contactDwellSteps > 0,
              epochs > 0, learningRate.isFinite, learningRate > 0,
              maximumGradientNorm.isFinite, maximumGradientNorm > 0,
              targetRowWeight.isFinite, targetRowWeight >= 1,
              policySourceRowWeight.isFinite,
              policySourceRowWeight >= 1,
              aggregationRounds > 0, aggregationRounds <= epochs,
              initialTeacherMix.isFinite,
              finalTeacherMix.isFinite,
              (0...1).contains(initialTeacherMix),
              (0...1).contains(finalTeacherMix),
              finalTeacherMix <= initialTeacherMix,
              rolloutActionNoiseStandardDeviation.isFinite,
              rolloutActionNoiseStandardDeviation >= 0,
              stateAlignmentLookahead > 0,
              validationActionNoiseStandardDeviation.isFinite,
              validationActionNoiseStandardDeviation >= 0,
              robustReplayCount > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid humanoid-box flow distillation configuration")
        }
    }
}

public struct HumanoidBoxFlowDistillationValidation: Codable, Sendable {
    public var epoch: Int
    public var teacherMix: Float?
    public var datasetRows: Int
    public var maximumStableCarryDistanceMeters: Float
    public var finalSuccessFraction: Float
    public var robustMaximumStableCarryDistanceMeters: Float
    public var robustFinalSuccessFraction: Float
    public var checkpoint: String
}

public struct HumanoidBoxFlowDistillationReport: Codable, Sendable {
    public var sourceCheckpoint: String
    public var flowReport: String
    public var teacherRows: Int
    public var aggregatedRows: Int
    public var aggregationRounds: Int
    public var stateAlignedTeacherRows: Int
    public var onPolicySourceRows: Int
    public var rejectedPostFailureRows: Int
    public var meanStateAlignmentDistance: Float
    public var maximumStateAlignmentDistance: Float
    public var sourceControlSteps: Int
    public var targetControlSteps: Int
    public var targetTrajectoryDurationSteps: Int
    public var legBlendKnotCount: Int
    public var legResidualKnotCount: Int
    public var initialActionMSE: Float
    public var finalActionMSE: Float
    public var teacherCarryDistanceMeters: Float
    public var learnerMaximumStableCarryDistanceMeters: Float
    public var learnerDeterministicFinalSuccessFraction: Float
    public var learnerRobustMaximumStableCarryDistanceMeters: Float
    public var learnerRobustFinalSuccessFraction: Float
    public var validationActionNoiseStandardDeviation: Float
    public var validationNoiseIncludesSource: Bool
    public var bestEpoch: Int
    public var validations: [HumanoidBoxFlowDistillationValidation]
    public var outputCheckpoint: String
    public var distillationGatePassed: Bool
}

public struct HumanoidBoxFlowReplayReport: Codable, Sendable {
    public var checkpoint: String
    public var flowReport: String
    public var requiredCarryDistanceMeters: Float
    public var deterministicMaximumStableCarryDistanceMeters: Float
    public var deterministicFinalSuccessFraction: Float
    public var actionNoiseStandardDeviation: Float
    public var actionNoiseScope: String
    public var robustMaximumStableCarryDistanceMeters: Float
    public var robustFinalSuccessFraction: Float
    public var deterministicGatePassed: Bool
    public var robustGatePassed: Bool
}

struct HumanoidBoxStableCarrySample: Sendable {
    var terminated: Bool
    var truncated: Bool
    var leftContact: Float
    var rightContact: Float
    var sourceSupportContact: Float
    var lifted: Bool
    var rootUprightAlignment: Float
    var boxUprightAlignment: Float
    var clearanceMeters: Float
    var carryDistanceMeters: Float
}

enum HumanoidBoxStableCarryVerifier {
    static func failed(previouslyFailed: Bool,
                       sample: HumanoidBoxStableCarrySample) -> Bool {
        previouslyFailed || sample.terminated || sample.truncated
    }

    static func isStable(
        previouslyFailed: Bool, sample: HumanoidBoxStableCarrySample
    ) -> Bool {
        !failed(previouslyFailed: previouslyFailed, sample: sample)
            && sample.leftContact > 0.5 && sample.rightContact > 0.5
            && sample.sourceSupportContact < 0.5 && sample.lifted
            && sample.rootUprightAlignment > 0.9
            && sample.boxUprightAlignment > 0.9
            && sample.clearanceMeters >= 0.01
    }

    static func isFinalSuccess(
        previouslyFailed: Bool, sample: HumanoidBoxStableCarrySample,
        maximumStableCarryDistanceMeters: Float,
        requiredCarryDistanceMeters: Float
    ) -> Bool {
        isStable(previouslyFailed: previouslyFailed, sample: sample)
            && sample.carryDistanceMeters >= requiredCarryDistanceMeters
            && maximumStableCarryDistanceMeters
                >= requiredCarryDistanceMeters
    }
}

/// Amortizes the simulator-selected carry segment into the ordinary routed
/// policy. Base locomotion and the pre-lift grasp expert are stop-gradient
/// frozen. The verified source lift flow remains an explicit, reported
/// bootstrap for this experiment; after that boundary the learned carry branch
/// receives only observations and uses no target trajectory or hidden phase.
public enum HumanoidBoxFlowDistillation {
    private static let firstArmAction = 11
    private static let armActionCount = 8
    private static let legActionCount = 10

    private struct Artifact {
        var sourceStages: [HumanoidBoxPhysicalFlowStage]
        var targetTrajectory: [Float]
        var targetSteps: Int
        var targetDuration: Int
        var legBlendKnotCount: Int
        var legResidualKnotCount: Int
        var maximumLegResidualAction: Float
        var teacherCarryDistance: Float
        var requiredCarryDistance: Float
    }

    public static func run(
        checkpointDirectory: String,
        flowReportPath: String,
        outputDirectory: String,
        configuration: HumanoidBoxFlowDistillationConfiguration
    ) throws -> HumanoidBoxFlowDistillationReport {
        try configuration.validate()
        let metadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self,
            from: Data(contentsOf: URL(fileURLWithPath:
                "\(checkpointDirectory)/metadata.json")))
        guard metadata.task == "humanoid-box-carry-v0",
              let semanticOptions = metadata.taskConfiguration else {
            throw RLEnvironmentError.invalidConfiguration(
                "flow distillation requires box-carry checkpoint metadata")
        }
        let artifact = try loadArtifact(flowReportPath)
        let runner = try VectorPolicyRunner(
            checkpointDirectory: checkpointDirectory)
        let replayOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: metadata.task, semanticOptions: semanticOptions,
            maxEpisodeSteps: metadata.maxEpisodeSteps,
            controlDecimation: metadata.controlDecimation)

        func makeTask(_ count: Int, seed: UInt64) throws -> HumanoidBoxCarryTask {
            let anyTask = try BuiltInRLTasks.registry.make(
                metadata.task, configuration: RLTaskConfiguration(
                    numEnvironments: count, seed: seed, autoReset: false,
                    options: replayOptions))
            guard let task = anyTask as? HumanoidBoxCarryTask,
                  metadata.compatibilityMismatches(with: task.spec).isEmpty else {
                throw RLEnvironmentError.invalidConfiguration(
                    "flow distillation task/checkpoint mismatch")
            }
            return task
        }

        func runnerActions(
            _ task: HumanoidBoxCarryTask, _ observation: RLObservationBatch,
            disableAuxiliary: Bool = false
        ) throws -> RLActionBatch {
            try runner.actions(
                for: observation,
                expertGates: task.policyExpertGates(observation.policy),
                expertActionMask: task.policyExpertActionMask,
                standExpertGates:
                    task.policyStandExpertGates(observation.policy),
                standExpertActionMask: task.policyStandExpertActionMask,
                auxiliaryExpertGates: disableAuxiliary
                    ? ContiguousArray(
                        repeating: 0, count: task.spec.numEnvironments)
                    : task.policyAuxiliaryExpertGates(observation.policy),
                auxiliaryExpertActionMask:
                    task.policyAuxiliaryExpertActionMask)
        }

        let armParameterCount = artifact.targetTrajectory.count
            - artifact.legBlendKnotCount
            - 10 * artifact.legResidualKnotCount
        guard armParameterCount > 0,
              armParameterCount.isMultiple(of: 4) else {
            throw RLEnvironmentError.invalidConfiguration(
                "flow distillation trajectory schema is invalid")
        }
        let armKnotCount = armParameterCount / 4

        func flowActions(
            task: HumanoidBoxCarryTask, observation: RLObservationBatch,
            trajectory: [Float], step: Int, duration: Int
        ) throws -> RLActionBatch {
            try flowActions(
                task: task, observation: observation,
                trajectory: trajectory,
                steps: [Int](repeating: step,
                             count: task.spec.numEnvironments),
                duration: duration)
        }

        func flowActions(
            task: HumanoidBoxCarryTask, observation: RLObservationBatch,
            trajectory: [Float], steps: [Int], duration: Int
        ) throws -> RLActionBatch {
            precondition(steps.count == task.spec.numEnvironments)
            var loaded = try runnerActions(task, observation)
            let base = artifact.legBlendKnotCount > 0
                ? try runnerActions(task, observation, disableAuxiliary: true)
                : nil
            for environment in 0..<task.spec.numEnvironments {
                let progress = Float(steps[environment] + 1)
                    / Float(duration)
                if let base {
                    let blend = HumanoidBoxPhysicalFlowExperiment
                        .legBlendFraction(
                            trajectory, progress: progress,
                            armParameterCount: armParameterCount,
                            knotCount: artifact.legBlendKnotCount)
                    let row = environment * metadata.actionDimension
                    for action in 0..<legActionCount {
                        let index = row + action
                        loaded.values[index] = (1 - blend)
                            * loaded.values[index] + blend * base.values[index]
                    }
                }
                if artifact.legResidualKnotCount > 0 {
                    for action in 0..<legActionCount {
                        let index = environment * metadata.actionDimension
                            + action
                        loaded.values[index] = simd_clamp(
                            loaded.values[index]
                                + HumanoidBoxPhysicalFlowExperiment
                                    .legResidualAction(
                                        trajectory, action: action,
                                        progress: progress,
                                        armParameterCount: armParameterCount,
                                        blendKnotCount:
                                            artifact.legBlendKnotCount,
                                        residualKnotCount:
                                            artifact.legResidualKnotCount,
                                        maximumAction:
                                            artifact.maximumLegResidualAction),
                            -0.999, 0.999)
                    }
                }
                let arm = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
                    Array(trajectory.prefix(armParameterCount)),
                    knotCount: armKnotCount, progress: progress)
                let row = environment * metadata.actionDimension
                    + firstArmAction
                for action in 0..<armActionCount {
                    loaded.values[row + action] = simd_clamp(
                        loaded.values[row + action] + arm[action],
                        -0.999, 0.999)
                }
            }
            return loaded
        }

        let task = try makeTask(
            configuration.collectionEnvironments, seed: configuration.seed)
        var observation = try task.reset(seed: configuration.seed)
        var result = RLStepBatch(spec: task.spec)
        var contactStreak = 0
        var warmupSteps = 0
        while warmupSteps < metadata.maxEpisodeSteps,
              contactStreak < configuration.contactDwellSteps {
            let action = try runnerActions(task, observation)
            try task.step(actions: action, into: &result)
            observation = result.observations
            warmupSteps += 1
            contactStreak = zip(
                result.metrics["state/left_hand_contact"]!,
                result.metrics["state/right_hand_contact"]!).allSatisfy {
                    $0.0 > 0.5 && $0.1 > 0.5
                } ? contactStreak + 1 : 0
        }
        guard contactStreak >= configuration.contactDwellSteps else {
            throw RLEnvironmentError.invalidConfiguration(
                "source checkpoint did not establish the flow grasp")
        }

        let observationDimension = metadata.observationDimension
        let actionDimension = metadata.actionDimension
        var observations = [Float]()
        var targets = [Float]()
        var expertGates = [Float]()
        var standGates = [Float]()
        var auxiliaryGates = [Float]()
        var rowWeights = [Float]()

        func appendRows(
            task: HumanoidBoxCarryTask,
            observation: RLObservationBatch,
            action: RLActionBatch,
            weight: Float,
            environments: [Int]? = nil
        ) {
            let expert = task.policyExpertGates(observation.policy)
            let stand = task.policyStandExpertGates(observation.policy)
            let auxiliary = task.policyAuxiliaryExpertGates(observation.policy)
            let rows = environments
                ?? Array(0..<task.spec.numEnvironments)
            for environment in rows {
                let o = environment * observationDimension
                observations.append(contentsOf:
                    observation.policy[o..<(o + observationDimension)])
                let a = environment * actionDimension
                targets.append(contentsOf: action.values[a..<(a + actionDimension)])
                expertGates.append(expert[environment])
                standGates.append(stand[environment])
                auxiliaryGates.append(auxiliary[environment])
                rowWeights.append(weight)
            }
        }

        for stage in artifact.sourceStages {
            for step in 0..<stage.controlSteps {
                let action = stage.policyOnly == true
                    ? try runnerActions(task, observation)
                    : try flowActions(
                        task: task, observation: observation,
                        trajectory: stage.trajectory, step: step,
                        duration: stage.trajectoryDurationSteps
                            ?? stage.controlSteps)
                appendRows(
                    task: task, observation: observation,
                    action: action,
                    weight: stage.policyOnly == true
                        ? configuration.policySourceRowWeight : 1)
                try task.step(actions: action, into: &result)
                observation = result.observations
            }
        }
        var referenceTargetObservations = [[Float]]()
        for step in 0..<artifact.targetSteps {
            let action = try flowActions(
                task: task, observation: observation,
                trajectory: artifact.targetTrajectory, step: step,
                duration: artifact.targetDuration)
            referenceTargetObservations.append(Array(
                observation.policy.prefix(observationDimension)))
            appendRows(
                task: task, observation: observation,
                action: action, weight: configuration.targetRowWeight)
            try task.step(actions: action, into: &result)
            observation = result.observations
        }
        guard result.metrics["state/carry_distance_m"]![0]
                >= artifact.requiredCarryDistance,
              result.metrics["state/left_hand_contact"]![0] > 0.5,
              result.metrics["state/right_hand_contact"]![0] > 0.5,
              result.metrics["state/box_pedestal_contact"]![0] < 0.5 else {
            throw RLEnvironmentError.invalidConfiguration(
                "flow teacher did not reproduce its physical carry state")
        }

        let policy = VectorActorCritic(
            observationDimension: observationDimension,
            actionDimension: actionDimension,
            hiddenSize: metadata.ppo.hiddenSize,
            hiddenDimensions: metadata.ppo.hiddenDimensions,
            initialActionStd: metadata.ppo.initialActionStd,
            activation: metadata.ppo.resolvedActivation)
        let sourceArrays = try VectorActorCritic.compatibleWeights(
            try loadArrays(url: URL(fileURLWithPath:
                "\(checkpointDirectory)/policy.safetensors")),
            architectureVersion: metadata.architectureVersion)
        try policy.update(
            parameters: ModuleParameters.unflattened(sourceArrays),
            verify: [.all])
        eval(policy)

        let normalizer = RunningObservationNormalizer(
            snapshot: metadata.normalizer)
        let expertMask = task.policyExpertActionMask.map {
            MLXArray(Array($0)).reshaped([1, actionDimension])
        }
        let standMask = task.policyStandExpertActionMask.map {
            MLXArray(Array($0)).reshaped([1, actionDimension])
        }
        let auxiliaryMask = task.policyAuxiliaryExpertActionMask.map {
            MLXArray(Array($0)).reshaped([1, actionDimension])
        }
        let distribution = metadata.ppo.resolvedActionDistribution

        func datasetArrays() -> [MLXArray] {
            let rowCount = expertGates.count
            let normalized = metadata.ppo.normalizeObservations
                ? Array(normalizer.normalize(ContiguousArray(observations)))
                : observations
            return [
                MLXArray(normalized).reshaped(
                    [rowCount, observationDimension]),
                MLXArray(targets).reshaped([rowCount, actionDimension]),
                MLXArray(expertGates).reshaped([rowCount, 1]),
                MLXArray(standGates).reshaped([rowCount, 1]),
                MLXArray(auxiliaryGates).reshaped([rowCount, 1]),
                MLXArray(rowWeights),
            ]
        }

        func predicted(
            _ model: VectorActorCritic, _ arrays: [MLXArray]
        ) -> MLXArray {
            let mean = model.forward(
                arrays[0], expertGate: arrays[2],
                expertActionMask: expertMask,
                standExpertGate: arrays[3],
                standExpertActionMask: standMask,
                auxiliaryExpertGate: arrays[4],
                auxiliaryExpertActionMask: auxiliaryMask,
                freezeBaseActor: true, freezeExpertActor: true,
                freezeStandActor: false, freezeAuxiliaryActor: false).mean
            return distribution == .squashedGaussian ? tanh(mean) : mean
        }

        func loss(
            _ model: VectorActorCritic, _ arrays: [MLXArray]
        ) -> MLXArray {
            let row = sum(
                (predicted(model, arrays) - arrays[1]).square(), axis: -1)
                / Float(actionDimension)
            return sum(row * arrays[5]) / clip(
                sum(arrays[5]), min: 1,
                max: Float.greatestFiniteMagnitude)
        }

        func learnerActions(
            task: HumanoidBoxCarryTask,
            observation: RLObservationBatch
        ) throws -> RLActionBatch {
            let rowCount = task.spec.numEnvironments
            let inferenceRows = max(
                rowCount, metadata.inferenceBatchSize ?? rowCount)
            var policyObservations = observation.policy
            let finalObservation = policyObservations.suffix(
                observationDimension)
            for _ in rowCount..<inferenceRows {
                policyObservations.append(contentsOf: finalObservation)
            }
            if metadata.ppo.normalizeObservations {
                policyObservations = normalizer.normalize(policyObservations)
            }
            func padded(_ values: ContiguousArray<Float>) -> [Float] {
                var result = Array(values)
                let final = result.last ?? 0
                for _ in rowCount..<inferenceRows { result.append(final) }
                return result
            }
            let output = policy.forward(
                MLXArray(Array(policyObservations)).reshaped(
                    [inferenceRows, observationDimension]),
                expertGate: MLXArray(padded(
                    task.policyExpertGates(observation.policy))).reshaped(
                        [inferenceRows, 1]),
                expertActionMask: expertMask,
                standExpertGate: MLXArray(padded(
                    task.policyStandExpertGates(observation.policy))).reshaped(
                        [inferenceRows, 1]),
                standExpertActionMask: standMask,
                auxiliaryExpertGate: MLXArray(padded(
                    task.policyAuxiliaryExpertGates(
                        observation.policy))).reshaped([inferenceRows, 1]),
                auxiliaryExpertActionMask: auxiliaryMask).mean
            let bounded = distribution == .squashedGaussian
                ? tanh(output) : output
            eval(bounded)
            return try RLActionBatch(
                numEnvironments: rowCount,
                actionDimension: actionDimension,
                values: ContiguousArray(
                    bounded.asArray(Float.self).prefix(
                        rowCount * actionDimension)))
        }

        let initialTeacherRows = expertGates.count
        var arrays = datasetArrays()
        let initialLossArray = loss(policy, arrays)
        eval(initialLossArray)
        let initialLoss = initialLossArray.item(Float.self)
        let lossAndGradient = valueAndGrad(model: policy) {
            model, arguments in [loss(model, arguments)]
        }
        let optimizer = CheckpointableAdam(
            learningRate: configuration.learningRate)
        var finalLoss = initialLoss
        var aggregationGenerator = ProbeRandomNumberGenerator(
            seed: configuration.seed &+ 0xDA66_EA11)
        var stateAlignedTeacherRows = 0
        var onPolicySourceRows = 0
        var rejectedPostFailureRows = 0
        var alignmentDistanceSum: Float = 0
        var maximumAlignmentDistance: Float = 0

        func collectOnPolicyRows(
            round: Int, teacherMix: Float
        ) throws {
            let rolloutTask = try makeTask(
                configuration.collectionEnvironments,
                seed: configuration.seed
                    &+ UInt64(round + 1) * 10_007)
            var rolloutObservation = try rolloutTask.reset(
                seed: configuration.seed
                    &+ UInt64(round + 1) * 10_007)
            var rolloutResult = RLStepBatch(spec: rolloutTask.spec)
            var streak = 0
            var warmup = 0
            while warmup < metadata.maxEpisodeSteps,
                  streak < configuration.contactDwellSteps {
                let action = try runnerActions(
                    rolloutTask, rolloutObservation)
                try rolloutTask.step(
                    actions: action, into: &rolloutResult)
                rolloutObservation = rolloutResult.observations
                warmup += 1
                streak = zip(
                    rolloutResult.metrics["state/left_hand_contact"]!,
                    rolloutResult.metrics["state/right_hand_contact"]!)
                    .allSatisfy { $0.0 > 0.5 && $0.1 > 0.5 }
                    ? streak + 1 : 0
                if rolloutResult.terminated.contains(true)
                    || rolloutResult.truncated.contains(true) {
                    break
                }
            }
            guard streak >= configuration.contactDwellSteps else {
                throw RLEnvironmentError.invalidConfiguration(
                    "flow DAgger could not reproduce the source grasp")
            }
            var active = [Bool](
                repeating: true,
                count: rolloutTask.spec.numEnvironments)
            for stage in artifact.sourceStages {
                for step in 0..<stage.controlSteps {
                    if stage.policyOnly == true {
                        let teacher = try runnerActions(
                            rolloutTask, rolloutObservation)
                        let learner = try learnerActions(
                            task: rolloutTask,
                            observation: rolloutObservation)
                        let activeEnvironments = active.indices.filter {
                            active[$0]
                        }
                        appendRows(
                            task: rolloutTask,
                            observation: rolloutObservation,
                            action: teacher,
                            weight: configuration.policySourceRowWeight,
                            environments: activeEnvironments)
                        onPolicySourceRows += activeEnvironments.count
                        rejectedPostFailureRows += active.count
                            - activeEnvironments.count
                        var executed = learner
                        for environment in 0..<rolloutTask.spec
                                .numEnvironments {
                            let base = environment * actionDimension
                            for action in 0..<actionDimension {
                                let index = base + action
                                var value = teacherMix
                                    * teacher.values[index]
                                    + (1 - teacherMix)
                                        * learner.values[index]
                                if active[environment], configuration
                                        .rolloutActionNoiseStandardDeviation
                                    > 0 {
                                    value += (1 - teacherMix)
                                        * configuration
                                            .rolloutActionNoiseStandardDeviation
                                        * aggregationGenerator.normal()
                                }
                                executed.values[index] = simd_clamp(
                                    value, -0.999, 0.999)
                            }
                        }
                        try rolloutTask.step(
                            actions: executed, into: &rolloutResult)
                    } else {
                        let teacher = try flowActions(
                            task: rolloutTask,
                            observation: rolloutObservation,
                            trajectory: stage.trajectory, step: step,
                            duration: stage.trajectoryDurationSteps
                                ?? stage.controlSteps)
                        try rolloutTask.step(
                            actions: teacher, into: &rolloutResult)
                    }
                    rolloutObservation = rolloutResult.observations
                    for environment in active.indices
                        where active[environment]
                            && (rolloutResult.terminated[environment]
                                || rolloutResult.truncated[environment]) {
                        active[environment] = false
                    }
                }
            }
            let sourceLeft = rolloutResult.metrics[
                "state/left_hand_contact"]!
            let sourceRight = rolloutResult.metrics[
                "state/right_hand_contact"]!
            for environment in active.indices where active[environment] {
                if sourceLeft[environment] <= 0.5
                    || sourceRight[environment] <= 0.5 {
                    active[environment] = false
                }
            }
            guard active.contains(true) else { return }

            var phases = [Int](
                repeating: 0,
                count: rolloutTask.spec.numEnvironments)
            for _ in 0..<artifact.targetSteps {
                for environment in 0..<rolloutTask.spec.numEnvironments
                    where active[environment] {
                    let aligned = alignedTargetPhase(
                        observation: rolloutObservation.policy,
                        environment: environment,
                        observationDimension: observationDimension,
                        references: referenceTargetObservations,
                        previousPhase: phases[environment],
                        lookahead: configuration.stateAlignmentLookahead)
                    phases[environment] = aligned.phase
                    alignmentDistanceSum += aligned.distance
                    maximumAlignmentDistance = max(
                        maximumAlignmentDistance, aligned.distance)
                }
                let teacher = try flowActions(
                    task: rolloutTask,
                    observation: rolloutObservation,
                    trajectory: artifact.targetTrajectory,
                    steps: phases, duration: artifact.targetDuration)
                let learner = try learnerActions(
                    task: rolloutTask,
                    observation: rolloutObservation)
                let activeEnvironments = active.indices.filter { active[$0] }
                appendRows(
                    task: rolloutTask,
                    observation: rolloutObservation,
                    action: teacher,
                    weight: configuration.targetRowWeight,
                    environments: activeEnvironments)
                stateAlignedTeacherRows += activeEnvironments.count
                rejectedPostFailureRows += active.count
                    - activeEnvironments.count
                var executed = learner
                for environment in 0..<rolloutTask.spec.numEnvironments {
                    let base = environment * actionDimension
                    for action in 0..<actionDimension {
                        let index = base + action
                        var value = teacherMix * teacher.values[index]
                            + (1 - teacherMix) * learner.values[index]
                        if active[environment],
                           configuration.rolloutActionNoiseStandardDeviation
                                > 0 {
                            value += (1 - teacherMix)
                                * configuration
                                    .rolloutActionNoiseStandardDeviation
                                * aggregationGenerator.normal()
                        }
                        executed.values[index] = simd_clamp(
                            value, -0.999, 0.999)
                    }
                }
                try rolloutTask.step(
                    actions: executed, into: &rolloutResult)
                rolloutObservation = rolloutResult.observations
                for environment in active.indices where active[environment] {
                    if rolloutResult.terminated[environment]
                        || rolloutResult.truncated[environment] {
                        active[environment] = false
                    }
                }
                if !active.contains(true) { break }
            }
        }

        var validations = [HumanoidBoxFlowDistillationValidation]()
        let sourceReplay = try replayLearner(
            checkpointDirectory: checkpointDirectory,
            artifact: artifact, configuration: configuration,
            actionNoiseStandardDeviation: 0,
            noiseIncludesSource: false)
        let sourceRobustReplay = try replayLearner(
            checkpointDirectory: checkpointDirectory,
            artifact: artifact, configuration: configuration,
            actionNoiseStandardDeviation:
                configuration.validationActionNoiseStandardDeviation,
            noiseIncludesSource:
                configuration.validationNoiseIncludesSource)
        validations.append(.init(
            epoch: 0, teacherMix: nil,
            datasetRows: expertGates.count,
            maximumStableCarryDistanceMeters: sourceReplay.maximumCarry,
            finalSuccessFraction: sourceReplay.successFraction,
            robustMaximumStableCarryDistanceMeters:
                sourceRobustReplay.maximumCarry,
            robustFinalSuccessFraction:
                sourceRobustReplay.successFraction,
            checkpoint: checkpointDirectory))
        var bestEpoch = 0
        var bestCheckpoint = checkpointDirectory
        var bestReplay = sourceReplay
        var bestRobustReplay = sourceRobustReplay
        var precedingTeacherMix: Float?

        for round in 0..<configuration.aggregationRounds {
            let firstEpoch = round * configuration.epochs
                / configuration.aggregationRounds + 1
            let finalEpoch = (round + 1) * configuration.epochs
                / configuration.aggregationRounds
            for epoch in firstEpoch...finalEpoch {
                let (losses, gradients) = lossAndGradient(policy, arrays)
                let clipped = clippedGradients(
                    gradients, maximum: configuration.maximumGradientNorm)
                eval(losses + [clipped.squaredNorm])
                finalLoss = losses[0].item(Float.self)
                guard finalLoss.isFinite,
                      clipped.squaredNorm.item(Float.self).isFinite else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "non-finite flow distillation at epoch \(epoch)")
                }
                try optimizer.update(
                    model: policy, gradients: clipped.gradients)
                eval(policy)
                optimizer.evaluate()
            }

            let checkpoint = outputDirectory + "/checkpoints/"
                + String(format: "update-%06d", finalEpoch)
            try saveCheckpoint(
                policy: policy, metadata: metadata, optimizer: optimizer,
                completedEpochs: finalEpoch,
                teacherRows: expertGates.count, directory: checkpoint)
            let replay = try replayLearner(
                checkpointDirectory: checkpoint,
                artifact: artifact, configuration: configuration,
                actionNoiseStandardDeviation: 0,
                noiseIncludesSource: false)
            let robustReplay = try replayLearner(
                checkpointDirectory: checkpoint,
                artifact: artifact, configuration: configuration,
                actionNoiseStandardDeviation:
                    configuration.validationActionNoiseStandardDeviation,
                noiseIncludesSource:
                    configuration.validationNoiseIncludesSource)
            validations.append(.init(
                epoch: finalEpoch,
                teacherMix: precedingTeacherMix,
                datasetRows: expertGates.count,
                maximumStableCarryDistanceMeters: replay.maximumCarry,
                finalSuccessFraction: replay.successFraction,
                robustMaximumStableCarryDistanceMeters:
                    robustReplay.maximumCarry,
                robustFinalSuccessFraction: robustReplay.successFraction,
                checkpoint: checkpoint))
            let currentRank = (
                robustReplay.successFraction,
                replay.successFraction,
                min(robustReplay.maximumCarry, replay.maximumCarry))
            let bestRank = (
                bestRobustReplay.successFraction,
                bestReplay.successFraction,
                min(bestRobustReplay.maximumCarry, bestReplay.maximumCarry))
            if currentRank.0 > bestRank.0
                || (currentRank.0 == bestRank.0
                    && (currentRank.1 > bestRank.1
                        || (currentRank.1 == bestRank.1
                            && currentRank.2 > bestRank.2))) {
                bestEpoch = finalEpoch
                bestCheckpoint = checkpoint
                bestReplay = replay
                bestRobustReplay = robustReplay
            }

            guard round + 1 < configuration.aggregationRounds else {
                continue
            }
            let collectionCount = configuration.aggregationRounds - 1
            let progress = collectionCount <= 1 ? Float(1)
                : Float(round) / Float(collectionCount - 1)
            let teacherMix = configuration.initialTeacherMix
                + progress * (configuration.finalTeacherMix
                    - configuration.initialTeacherMix)
            try collectOnPolicyRows(round: round, teacherMix: teacherMix)
            precedingTeacherMix = teacherMix
            arrays = datasetArrays()
        }

        let finalLossArray = loss(policy, arrays)
        eval(finalLossArray)
        finalLoss = finalLossArray.item(Float.self)
        let meanAlignmentDistance = stateAlignedTeacherRows > 0
            ? alignmentDistanceSum / Float(stateAlignedTeacherRows) : 0
        let report = HumanoidBoxFlowDistillationReport(
            sourceCheckpoint: checkpointDirectory,
            flowReport: flowReportPath,
            teacherRows: initialTeacherRows,
            aggregatedRows: expertGates.count,
            aggregationRounds: configuration.aggregationRounds,
            stateAlignedTeacherRows: stateAlignedTeacherRows,
            onPolicySourceRows: onPolicySourceRows,
            rejectedPostFailureRows: rejectedPostFailureRows,
            meanStateAlignmentDistance: meanAlignmentDistance,
            maximumStateAlignmentDistance: maximumAlignmentDistance,
            sourceControlSteps: artifact.sourceStages.reduce(0) {
                $0 + $1.controlSteps
            }, targetControlSteps: artifact.targetSteps,
            targetTrajectoryDurationSteps: artifact.targetDuration,
            legBlendKnotCount: artifact.legBlendKnotCount,
            legResidualKnotCount: artifact.legResidualKnotCount,
            initialActionMSE: initialLoss,
            finalActionMSE: finalLoss,
            teacherCarryDistanceMeters: artifact.teacherCarryDistance,
            learnerMaximumStableCarryDistanceMeters: bestReplay.maximumCarry,
            learnerDeterministicFinalSuccessFraction:
                bestReplay.successFraction,
            learnerRobustMaximumStableCarryDistanceMeters:
                bestRobustReplay.maximumCarry,
            learnerRobustFinalSuccessFraction:
                bestRobustReplay.successFraction,
            validationActionNoiseStandardDeviation:
                configuration.validationActionNoiseStandardDeviation,
            validationNoiseIncludesSource:
                configuration.validationNoiseIncludesSource,
            bestEpoch: bestEpoch,
            validations: validations,
            outputCheckpoint: bestCheckpoint,
            distillationGatePassed: bestReplay.successFraction >= 0.8
                && bestRobustReplay.successFraction >= 0.8
                && min(bestReplay.maximumCarry,
                       bestRobustReplay.maximumCarry)
                    >= artifact.requiredCarryDistance)
        try FileManager.default.createDirectory(
            atPath: outputDirectory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: URL(fileURLWithPath:
            "\(outputDirectory)/distillation.json"), options: .atomic)
        return report
    }

    public static func evaluate(
        checkpointDirectory: String,
        flowReportPath: String,
        configuration: HumanoidBoxFlowDistillationConfiguration
    ) throws -> HumanoidBoxFlowReplayReport {
        try configuration.validate()
        let artifact = try loadArtifact(flowReportPath)
        let deterministic = try replayLearner(
            checkpointDirectory: checkpointDirectory,
            artifact: artifact, configuration: configuration,
            actionNoiseStandardDeviation: 0,
            noiseIncludesSource: false)
        let robust = try replayLearner(
            checkpointDirectory: checkpointDirectory,
            artifact: artifact, configuration: configuration,
            actionNoiseStandardDeviation:
                configuration.validationActionNoiseStandardDeviation,
            noiseIncludesSource:
                configuration.validationNoiseIncludesSource)
        let deterministicGate = deterministic.successFraction >= 0.8
            && deterministic.maximumCarry >= artifact.requiredCarryDistance
        let robustGate = robust.successFraction >= 0.8
            && robust.maximumCarry >= artifact.requiredCarryDistance
        return HumanoidBoxFlowReplayReport(
            checkpoint: checkpointDirectory,
            flowReport: flowReportPath,
            requiredCarryDistanceMeters: artifact.requiredCarryDistance,
            deterministicMaximumStableCarryDistanceMeters:
                deterministic.maximumCarry,
            deterministicFinalSuccessFraction:
                deterministic.successFraction,
            actionNoiseStandardDeviation:
                configuration.validationActionNoiseStandardDeviation,
            actionNoiseScope: configuration.validationNoiseIncludesSource
                ? "full-policy-sequence" : "target-policy-only",
            robustMaximumStableCarryDistanceMeters: robust.maximumCarry,
            robustFinalSuccessFraction: robust.successFraction,
            deterministicGatePassed: deterministicGate,
            robustGatePassed: robustGate)
    }

    private static func loadArtifact(_ path: String) throws -> Artifact {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let object = try JSONSerialization.jsonObject(with: data)
                as? [String: Any],
              (object["targetGatePassed"] as? Bool) == true,
              ((object["targetCloneSuccessFraction"]
                as? NSNumber)?.floatValue ?? 0) >= 0.8,
              ((object["targetReplayMaximumNormalizedError"]
                as? NSNumber)?.floatValue ?? .infinity) < 0.02,
              let targetValues = object["targetGeneratingTrajectory"]
                as? [NSNumber],
              let targetSteps = (object["selectedTargetStep"]
                as? NSNumber)?.intValue,
              let targetDuration = (object["targetGenerationSteps"]
                as? NSNumber)?.intValue,
              let legKnots = (object["legBlendKnotCount"]
                as? NSNumber)?.intValue,
              let carry = (object["targetCarryDistanceMeters"]
                as? NSNumber)?.floatValue,
              targetSteps > 0, targetDuration >= targetSteps,
              legKnots > 0,
              let encodedStages = object["sourceStages"]
                as? [[String: Any]] else {
            throw RLEnvironmentError.invalidConfiguration(
                "distillation requires a planner-verified dynamic flow")
        }
        let stages = try encodedStages.map { encoded in
            guard let steps = (encoded["controlSteps"]
                    as? NSNumber)?.intValue,
                  steps > 0 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "flow source stage is invalid")
            }
            if encoded["policyOnly"] as? Bool == true {
                return HumanoidBoxPhysicalFlowStage(
                    trajectory: [], controlSteps: steps,
                    policyOnly: true)
            }
            guard let values = encoded["trajectory"] as? [NSNumber] else {
                throw RLEnvironmentError.invalidConfiguration(
                    "flow source trajectory is missing")
            }
            return HumanoidBoxPhysicalFlowStage(
                trajectory: values.map(\.floatValue), controlSteps: steps,
                trajectoryDurationSteps: (encoded[
                    "trajectoryDurationSteps"] as? NSNumber)?.intValue)
        }
        return Artifact(
            sourceStages: stages,
            targetTrajectory: targetValues.map(\.floatValue),
            targetSteps: targetSteps, targetDuration: targetDuration,
            legBlendKnotCount: legKnots,
            legResidualKnotCount: (object[
                "legResidualKnotCount"] as? NSNumber)?.intValue ?? 0,
            maximumLegResidualAction: (object[
                "maximumLegResidualAction"] as? NSNumber)?.floatValue
                ?? 0.25,
            teacherCarryDistance: carry,
            requiredCarryDistance: (object[
                "minimumTargetCarryDistanceMeters"] as? NSNumber)?
                .floatValue ?? min(carry, 0.35))
    }

    static func alignedTargetPhase(
        observation: ContiguousArray<Float>, environment: Int,
        observationDimension: Int, references: [[Float]],
        previousPhase: Int, lookahead: Int
    ) -> (phase: Int, distance: Float) {
        precondition(!references.isEmpty)
        precondition(observationDimension > 104)
        precondition(observation.count
            >= (environment + 1) * observationDimension)
        let first = min(max(previousPhase, 0), references.count - 1)
        let last = min(first + lookahead, references.count - 1)
        var bestPhase = first
        var bestDistance = Float.infinity
        let row = environment * observationDimension
        let weightedRanges: [(Range<Int>, Float)] = [
            (0..<9, 1),
            (12..<31, 0.25),
            (31..<50, 0.01),
            (50..<69, 0.10),
            (69..<84, 4),
            (84..<86, 8),
            (89..<93, 10),
            (96..<99, 2),
            (102..<105, 2),
        ]
        for phase in first...last {
            let reference = references[phase]
            precondition(reference.count == observationDimension)
            var squaredDistance: Float = 0
            var totalWeight: Float = 0
            for (range, weight) in weightedRanges {
                for channel in range {
                    let delta = observation[row + channel]
                        - reference[channel]
                    squaredDistance += weight * delta * delta
                    totalWeight += weight
                }
            }
            let distance = sqrt(
                squaredDistance / max(totalWeight, 1e-6))
            if distance < bestDistance {
                bestDistance = distance
                bestPhase = phase
            }
        }
        return (bestPhase, bestDistance)
    }

    private static func replayLearner(
        checkpointDirectory: String, artifact: Artifact,
        configuration: HumanoidBoxFlowDistillationConfiguration,
        actionNoiseStandardDeviation: Float,
        noiseIncludesSource: Bool
    ) throws -> (maximumCarry: Float, successFraction: Float) {
        precondition(actionNoiseStandardDeviation >= 0)
        let runner = try VectorPolicyRunner(
            checkpointDirectory: checkpointDirectory)
        guard let options = runner.metadata.taskConfiguration else {
            throw RLEnvironmentError.invalidConfiguration(
                "distilled checkpoint has no task configuration")
        }
        let replayOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: runner.metadata.task, semanticOptions: options,
            maxEpisodeSteps: runner.metadata.maxEpisodeSteps,
            controlDecimation: runner.metadata.controlDecimation)
        let anyTask = try BuiltInRLTasks.registry.make(
            runner.metadata.task, configuration: RLTaskConfiguration(
                numEnvironments: configuration.robustReplayCount,
                seed: configuration.seed, autoReset: false,
                options: replayOptions))
        guard let task = anyTask as? HumanoidBoxCarryTask else {
            throw RLEnvironmentError.invalidConfiguration(
                "distilled replay task mismatch")
        }
        func actions(
            _ observation: RLObservationBatch,
            disableAuxiliary: Bool = false
        ) throws -> RLActionBatch {
            try runner.actions(
                for: observation,
                expertGates: task.policyExpertGates(observation.policy),
                expertActionMask: task.policyExpertActionMask,
                standExpertGates:
                    task.policyStandExpertGates(observation.policy),
                standExpertActionMask: task.policyStandExpertActionMask,
                auxiliaryExpertGates: disableAuxiliary
                    ? ContiguousArray(
                        repeating: 0,
                        count: task.spec.numEnvironments)
                    : task.policyAuxiliaryExpertGates(observation.policy),
                auxiliaryExpertActionMask:
                    task.policyAuxiliaryExpertActionMask)
        }
        var replayGenerator = ProbeRandomNumberGenerator(
            seed: configuration.seed &+ 0xA551_5EED)
        func perturb(_ action: inout RLActionBatch) {
            guard actionNoiseStandardDeviation > 0 else { return }
            for index in action.values.indices {
                action.values[index] = simd_clamp(
                    action.values[index]
                        + actionNoiseStandardDeviation
                            * replayGenerator.normal(),
                    -0.999, 0.999)
            }
        }
        var observation = try task.reset(seed: configuration.seed)
        var result = RLStepBatch(spec: task.spec)
        var streak = 0
        var warmup = 0
        while warmup < runner.metadata.maxEpisodeSteps,
              streak < configuration.contactDwellSteps {
            var action = try actions(observation)
            if noiseIncludesSource { perturb(&action) }
            try task.step(actions: action, into: &result)
            observation = result.observations
            warmup += 1
            streak = zip(
                result.metrics["state/left_hand_contact"]!,
                result.metrics["state/right_hand_contact"]!).allSatisfy {
                    $0.0 > 0.5 && $0.1 > 0.5
                } ? streak + 1 : 0
        }
        guard streak >= configuration.contactDwellSteps else {
            return (0, 0)
        }
        func samples() -> [HumanoidBoxStableCarrySample] {
            let states = task.environment.states()
            let manipulation = task.environment.manipulationStates()
            let left = result.metrics["state/left_hand_contact"]!
            let right = result.metrics["state/right_hand_contact"]!
            let support = result.metrics["state/box_pedestal_contact"]!
            let carry = result.metrics["state/carry_distance_m"]!
            let clearance = result.metrics["state/box_clearance_m"]!
            return (0..<task.spec.numEnvironments).map { environment in
                HumanoidBoxStableCarrySample(
                    terminated: result.terminated[environment],
                    truncated: result.truncated[environment],
                    leftContact: left[environment],
                    rightContact: right[environment],
                    sourceSupportContact: support[environment],
                    lifted: observation.policy[
                        environment
                            * HumanoidBoxCarryTask.observationDimension + 89]
                        > 0.5,
                    rootUprightAlignment: states[environment].root.rotation
                        .act(F3(0, 0, 1)).z,
                    boxUprightAlignment: manipulation[environment]
                        .object.rotation.act(F3(0, 0, 1)).z,
                    clearanceMeters: clearance[environment],
                    carryDistanceMeters: carry[environment])
            }
        }

        let armCount = artifact.targetTrajectory.count
            - artifact.legBlendKnotCount
            - 10 * artifact.legResidualKnotCount
        let armKnots = armCount / 4
        var failedEver = [Bool](
            repeating: false, count: task.spec.numEnvironments)
        for stage in artifact.sourceStages {
            for step in 0..<stage.controlSteps {
                var action = try actions(observation)
                if stage.policyOnly == true {
                    if noiseIncludesSource { perturb(&action) }
                    try task.step(actions: action, into: &result)
                    observation = result.observations
                    for (environment, sample) in samples().enumerated() {
                        failedEver[environment] =
                            HumanoidBoxStableCarryVerifier.failed(
                                previouslyFailed: failedEver[environment],
                                sample: sample)
                    }
                    continue
                }
                let baseLegAction = artifact.legBlendKnotCount > 0
                    ? try actions(observation, disableAuxiliary: true)
                    : nil
                let progress = Float(step + 1)
                    / Float(stage.trajectoryDurationSteps ?? stage.controlSteps)
                let arm = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
                    Array(stage.trajectory.prefix(armCount)),
                    knotCount: armKnots, progress: progress)
                for environment in 0..<task.spec.numEnvironments {
                    let row = environment * runner.metadata.actionDimension
                    if let baseLegAction {
                        let blend = HumanoidBoxPhysicalFlowExperiment
                            .legBlendFraction(
                                stage.trajectory, progress: progress,
                                armParameterCount: armCount,
                                knotCount: artifact.legBlendKnotCount)
                        for index in 0..<legActionCount {
                            action.values[row + index] = (1 - blend)
                                * action.values[row + index]
                                + blend * baseLegAction.values[row + index]
                        }
                    }
                    if artifact.legResidualKnotCount > 0 {
                        for index in 0..<legActionCount {
                            action.values[row + index] = simd_clamp(
                                action.values[row + index]
                                    + HumanoidBoxPhysicalFlowExperiment
                                        .legResidualAction(
                                            stage.trajectory,
                                            action: index,
                                            progress: progress,
                                            armParameterCount: armCount,
                                            blendKnotCount:
                                                artifact.legBlendKnotCount,
                                            residualKnotCount:
                                                artifact.legResidualKnotCount,
                                            maximumAction: artifact
                                                .maximumLegResidualAction),
                                -0.999, 0.999)
                        }
                    }
                    let base = row + firstArmAction
                    for index in 0..<armActionCount {
                        action.values[base + index] = simd_clamp(
                            action.values[base + index] + arm[index],
                            -0.999, 0.999)
                    }
                }
                if noiseIncludesSource { perturb(&action) }
                try task.step(actions: action, into: &result)
                observation = result.observations
                for (environment, sample) in samples().enumerated() {
                    failedEver[environment] = HumanoidBoxStableCarryVerifier
                        .failed(
                            previouslyFailed: failedEver[environment],
                            sample: sample)
                }
            }
        }
        var maximumStableCarry = [Float](
            repeating: 0, count: task.spec.numEnvironments)
        for _ in 0..<artifact.targetSteps {
            var action = try actions(observation)
            perturb(&action)
            try task.step(actions: action, into: &result)
            observation = result.observations
            for (environment, sample) in samples().enumerated() {
                let previouslyFailed = failedEver[environment]
                failedEver[environment] = HumanoidBoxStableCarryVerifier
                    .failed(previouslyFailed: previouslyFailed,
                            sample: sample)
                if HumanoidBoxStableCarryVerifier.isStable(
                    previouslyFailed: previouslyFailed, sample: sample
                ) {
                    maximumStableCarry[environment] = max(
                        maximumStableCarry[environment],
                        sample.carryDistanceMeters)
                }
            }
        }
        let finalSamples = samples()
        let successes = (0..<task.spec.numEnvironments).filter {
            HumanoidBoxStableCarryVerifier.isFinalSuccess(
                previouslyFailed: failedEver[$0],
                sample: finalSamples[$0],
                maximumStableCarryDistanceMeters: maximumStableCarry[$0],
                requiredCarryDistanceMeters: artifact.requiredCarryDistance)
        }.count
        return (maximumStableCarry.max() ?? 0, Float(successes)
            / Float(task.spec.numEnvironments))
    }

    private static func clippedGradients(
        _ gradients: ModuleParameters, maximum: Float
    ) -> (gradients: ModuleParameters, squaredNorm: MLXArray) {
        let flat = gradients.flattened()
        var squaredNorm = MLXArray(Float(0))
        for (_, gradient) in flat {
            squaredNorm = squaredNorm + gradient.square().sum()
        }
        let scale = minimum(
            MLXArray(maximum) / (sqrt(squaredNorm) + 1e-6),
            MLXArray(Float(1)))
        return (
            ModuleParameters.unflattened(flat.map { ($0.0, $0.1 * scale) }),
            squaredNorm)
    }

    private static func saveCheckpoint(
        policy: VectorActorCritic, metadata: VectorPolicyMetadata,
        optimizer: CheckpointableAdam, completedEpochs: Int,
        teacherRows: Int, directory: String
    ) throws {
        try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
        let arrays = Dictionary(uniqueKeysWithValues:
            policy.parameters().flattened().map { ($0.0, $0.1) })
        try MLX.saveToData(arrays: arrays).write(
            to: URL(fileURLWithPath: "\(directory)/policy.safetensors"),
            options: .atomic)
        try JSONEncoder().encode(metadata).write(
            to: URL(fileURLWithPath: "\(directory)/metadata.json"),
            options: .atomic)
        try MLX.saveToData(arrays: optimizer.checkpointArrays()).write(
            to: URL(fileURLWithPath: "\(directory)/optimizer.safetensors"),
            options: .atomic)
        try JSONEncoder().encode(VectorPPOTrainingState(
            completedUpdates: completedEpochs,
            environmentSteps: teacherRows,
            optimizerSteps: completedEpochs,
            adaptiveLearningRate: nil)).write(to: URL(fileURLWithPath:
                "\(directory)/training-state.json"), options: .atomic)
    }
}
