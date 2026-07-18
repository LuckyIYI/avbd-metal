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
    public var robustReplayCount: Int
    public var seed: UInt64

    public init(
        collectionEnvironments: Int = 256,
        contactDwellSteps: Int = 8,
        epochs: Int = 200,
        learningRate: Float = 1e-4,
        maximumGradientNorm: Float = 1,
        targetRowWeight: Float = 4,
        robustReplayCount: Int = 32,
        seed: UInt64 = 1
    ) {
        self.collectionEnvironments = collectionEnvironments
        self.contactDwellSteps = contactDwellSteps
        self.epochs = epochs
        self.learningRate = learningRate
        self.maximumGradientNorm = maximumGradientNorm
        self.targetRowWeight = targetRowWeight
        self.robustReplayCount = robustReplayCount
        self.seed = seed
    }

    func validate() throws {
        guard collectionEnvironments > 0, contactDwellSteps > 0,
              epochs > 0, learningRate.isFinite, learningRate > 0,
              maximumGradientNorm.isFinite, maximumGradientNorm > 0,
              targetRowWeight.isFinite, targetRowWeight >= 1,
              robustReplayCount > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid humanoid-box flow distillation configuration")
        }
    }
}

public struct HumanoidBoxFlowDistillationReport: Codable, Sendable {
    public var sourceCheckpoint: String
    public var flowReport: String
    public var teacherRows: Int
    public var sourceControlSteps: Int
    public var targetControlSteps: Int
    public var targetTrajectoryDurationSteps: Int
    public var legBlendKnotCount: Int
    public var initialActionMSE: Float
    public var finalActionMSE: Float
    public var teacherCarryDistanceMeters: Float
    public var learnerMaximumStableCarryDistanceMeters: Float
    public var learnerRobustFinalSuccessFraction: Float
    public var outputCheckpoint: String
    public var distillationGatePassed: Bool
}

/// Amortizes a simulator-selected physical flow into the routed carry branches.
/// Base locomotion and the pre-lift grasp expert are stop-gradient frozen;
/// exported inference uses the ordinary policy with no trajectory or hidden
/// task state.
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
        var teacherCarryDistance: Float
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
            var loaded = try runnerActions(task, observation)
            let progress = Float(step + 1) / Float(duration)
            if artifact.legBlendKnotCount > 0 {
                let base = try runnerActions(
                    task, observation, disableAuxiliary: true)
                let blend = HumanoidBoxPhysicalFlowExperiment.legBlendFraction(
                    trajectory, progress: progress,
                    armParameterCount: armParameterCount,
                    knotCount: artifact.legBlendKnotCount)
                for environment in 0..<task.spec.numEnvironments {
                    let row = environment * metadata.actionDimension
                    for action in 0..<legActionCount {
                        let index = row + action
                        loaded.values[index] = (1 - blend)
                            * loaded.values[index] + blend * base.values[index]
                    }
                }
            }
            let arm = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
                Array(trajectory.prefix(armParameterCount)),
                knotCount: armKnotCount, progress: progress)
            for environment in 0..<task.spec.numEnvironments {
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

        func appendRows(_ action: RLActionBatch, weight: Float) {
            let expert = task.policyExpertGates(observation.policy)
            let stand = task.policyStandExpertGates(observation.policy)
            let auxiliary = task.policyAuxiliaryExpertGates(observation.policy)
            for environment in 0..<task.spec.numEnvironments {
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
                let action = try flowActions(
                    task: task, observation: observation,
                    trajectory: stage.trajectory, step: step,
                    duration: stage.trajectoryDurationSteps
                        ?? stage.controlSteps)
                appendRows(action, weight: 1)
                try task.step(actions: action, into: &result)
                observation = result.observations
            }
        }
        for step in 0..<artifact.targetSteps {
            let action = try flowActions(
                task: task, observation: observation,
                trajectory: artifact.targetTrajectory, step: step,
                duration: artifact.targetDuration)
            appendRows(action, weight: configuration.targetRowWeight)
            try task.step(actions: action, into: &result)
            observation = result.observations
        }
        guard result.metrics["state/carry_distance_m"]![0] >= 0.35,
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
        let normalized = metadata.ppo.normalizeObservations
            ? Array(normalizer.normalize(ContiguousArray(observations)))
            : observations
        let rowCount = expertGates.count
        let x = MLXArray(normalized).reshaped(
            [rowCount, observationDimension])
        let y = MLXArray(targets).reshaped([rowCount, actionDimension])
        let expert = MLXArray(expertGates).reshaped([rowCount, 1])
        let stand = MLXArray(standGates).reshaped([rowCount, 1])
        let auxiliary = MLXArray(auxiliaryGates).reshaped([rowCount, 1])
        let weights = MLXArray(rowWeights)
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

        func predicted(_ model: VectorActorCritic) -> MLXArray {
            let mean = model.forward(
                x, expertGate: expert, expertActionMask: expertMask,
                standExpertGate: stand, standExpertActionMask: standMask,
                auxiliaryExpertGate: auxiliary,
                auxiliaryExpertActionMask: auxiliaryMask,
                freezeBaseActor: true, freezeExpertActor: true,
                freezeStandActor: false, freezeAuxiliaryActor: false).mean
            return distribution == .squashedGaussian ? tanh(mean) : mean
        }
        func loss(_ model: VectorActorCritic) -> MLXArray {
            let row = sum((predicted(model) - y).square(), axis: -1)
                / Float(actionDimension)
            return sum(row * weights) / clip(
                sum(weights), min: 1, max: Float.greatestFiniteMagnitude)
        }
        let initialLossArray = loss(policy)
        eval(initialLossArray)
        let initialLoss = initialLossArray.item(Float.self)
        let lossAndGradient = valueAndGrad(model: policy) {
            model, _ in [loss(model)]
        }
        let optimizer = CheckpointableAdam(
            learningRate: configuration.learningRate)
        var finalLoss = initialLoss
        for epoch in 1...configuration.epochs {
            let (losses, gradients) = lossAndGradient(policy, [])
            let clipped = clippedGradients(
                gradients, maximum: configuration.maximumGradientNorm)
            eval(losses + [clipped.squaredNorm])
            finalLoss = losses[0].item(Float.self)
            guard finalLoss.isFinite,
                  clipped.squaredNorm.item(Float.self).isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "non-finite flow distillation at epoch \(epoch)")
            }
            try optimizer.update(model: policy, gradients: clipped.gradients)
            eval(policy)
            optimizer.evaluate()
        }

        let outputCheckpoint = outputDirectory + "/checkpoints/"
            + String(format: "update-%06d", configuration.epochs)
        try saveCheckpoint(
            policy: policy, metadata: metadata, optimizer: optimizer,
            completedEpochs: configuration.epochs, teacherRows: rowCount,
            directory: outputCheckpoint)
        let replay = try replayLearner(
            checkpointDirectory: outputCheckpoint,
            artifact: artifact, configuration: configuration)
        let report = HumanoidBoxFlowDistillationReport(
            sourceCheckpoint: checkpointDirectory,
            flowReport: flowReportPath, teacherRows: rowCount,
            sourceControlSteps: artifact.sourceStages.reduce(0) {
                $0 + $1.controlSteps
            }, targetControlSteps: artifact.targetSteps,
            targetTrajectoryDurationSteps: artifact.targetDuration,
            legBlendKnotCount: artifact.legBlendKnotCount,
            initialActionMSE: initialLoss,
            finalActionMSE: finalLoss,
            teacherCarryDistanceMeters: artifact.teacherCarryDistance,
            learnerMaximumStableCarryDistanceMeters: replay.maximumCarry,
            learnerRobustFinalSuccessFraction: replay.successFraction,
            outputCheckpoint: outputCheckpoint,
            distillationGatePassed: replay.successFraction >= 0.8
                && replay.maximumCarry >= 0.35)
        try FileManager.default.createDirectory(
            atPath: outputDirectory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: URL(fileURLWithPath:
            "\(outputDirectory)/distillation.json"), options: .atomic)
        return report
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
            guard let values = encoded["trajectory"] as? [NSNumber],
                  let steps = (encoded["controlSteps"]
                    as? NSNumber)?.intValue,
                  steps > 0 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "flow source stage is invalid")
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
            teacherCarryDistance: carry)
    }

    private static func replayLearner(
        checkpointDirectory: String, artifact: Artifact,
        configuration: HumanoidBoxFlowDistillationConfiguration
    ) throws -> (maximumCarry: Float, successFraction: Float) {
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
        func actions(_ observation: RLObservationBatch) throws -> RLActionBatch {
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
        var observation = try task.reset(seed: configuration.seed)
        var result = RLStepBatch(spec: task.spec)
        var streak = 0
        var warmup = 0
        while warmup < runner.metadata.maxEpisodeSteps,
              streak < configuration.contactDwellSteps {
            let action = try actions(observation)
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
        // Source arm flows establish the lifted state; their leg blend is zero.
        let armCount = artifact.targetTrajectory.count
            - artifact.legBlendKnotCount
        let armKnots = armCount / 4
        for stage in artifact.sourceStages {
            for step in 0..<stage.controlSteps {
                var action = try actions(observation)
                let progress = Float(step + 1)
                    / Float(stage.trajectoryDurationSteps ?? stage.controlSteps)
                let arm = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
                    Array(stage.trajectory.prefix(armCount)),
                    knotCount: armKnots, progress: progress)
                for environment in 0..<task.spec.numEnvironments {
                    let base = environment * runner.metadata.actionDimension
                        + firstArmAction
                    for index in 0..<armActionCount {
                        action.values[base + index] = simd_clamp(
                            action.values[base + index] + arm[index],
                            -0.999, 0.999)
                    }
                }
                try task.step(actions: action, into: &result)
                observation = result.observations
            }
        }
        var maximumStableCarry = [Float](
            repeating: 0, count: task.spec.numEnvironments)
        var failedEver = [Bool](
            repeating: false, count: task.spec.numEnvironments)
        for _ in 0..<artifact.targetSteps {
            let action = try actions(observation)
            try task.step(actions: action, into: &result)
            observation = result.observations
            let states = task.environment.states()
            let manipulation = task.environment.manipulationStates()
            let left = result.metrics["state/left_hand_contact"]!
            let right = result.metrics["state/right_hand_contact"]!
            let support = result.metrics["state/box_pedestal_contact"]!
            let carry = result.metrics["state/carry_distance_m"]!
            let clearance = result.metrics["state/box_clearance_m"]!
            for environment in 0..<task.spec.numEnvironments {
                failedEver[environment] = failedEver[environment]
                    || result.terminated[environment]
                    || result.truncated[environment]
                let rootUp = states[environment].root.rotation
                    .act(F3(0, 0, 1)).z
                let boxUp = manipulation[environment].object.rotation
                    .act(F3(0, 0, 1)).z
                let lifted = observation.policy[
                    environment * HumanoidBoxCarryTask.observationDimension
                        + 89] > 0.5
                let stable = !failedEver[environment]
                    && left[environment] > 0.5
                    && right[environment] > 0.5
                    && support[environment] < 0.5
                    && lifted && rootUp > 0.9 && boxUp > 0.9
                    && clearance[environment] >= 0.01
                if stable {
                    maximumStableCarry[environment] = max(
                        maximumStableCarry[environment], carry[environment])
                }
            }
        }
        let left = result.metrics["state/left_hand_contact"]!
        let right = result.metrics["state/right_hand_contact"]!
        let support = result.metrics["state/box_pedestal_contact"]!
        let carry = result.metrics["state/carry_distance_m"]!
        let clearance = result.metrics["state/box_clearance_m"]!
        let finalStates = task.environment.states()
        let finalManipulation = task.environment.manipulationStates()
        let successes = (0..<task.spec.numEnvironments).filter {
            let rootUp = finalStates[$0].root.rotation
                .act(F3(0, 0, 1)).z
            let boxUp = finalManipulation[$0].object.rotation
                .act(F3(0, 0, 1)).z
            let lifted = observation.policy[
                $0 * HumanoidBoxCarryTask.observationDimension + 89] > 0.5
            return !failedEver[$0]
                && left[$0] > 0.5 && right[$0] > 0.5 && support[$0] < 0.5
                && carry[$0] >= 0.35
                && maximumStableCarry[$0] >= 0.35
                && lifted && rootUp > 0.9 && boxUp > 0.9
                && clearance[$0] >= 0.01
                && !result.terminated[$0] && !result.truncated[$0]
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
