import SimCore
import PhysicsAVBD
import Robotics
import RL
import Foundation
import MLX
import MLXNN
import simd

public struct HumanoidBoxCarryDistillationConfiguration: Sendable {
    public var collectionEnvironments: Int
    public var contactDwellSteps: Int
    public var probeSteps: Int
    public var epochs: Int
    public var learningRate: Float
    public var maximumGradientNorm: Float
    public var aggregationRounds: Int
    public var initialTeacherMix: Float
    public var finalTeacherMix: Float
    public var rolloutArmNoiseStandardDeviation: Float
    public var stateAlignedTeacherQueries: Bool
    public var physicalStateWeighting: Bool
    public var armActionWeight: Float
    public var probeSeed: UInt64

    public init(
        collectionEnvironments: Int = 512,
        contactDwellSteps: Int = 12,
        probeSteps: Int = 200,
        epochs: Int = 200,
        learningRate: Float = 1e-4,
        maximumGradientNorm: Float = 1,
        aggregationRounds: Int = 1,
        initialTeacherMix: Float = 0.75,
        finalTeacherMix: Float = 0,
        rolloutArmNoiseStandardDeviation: Float = 0.01,
        stateAlignedTeacherQueries: Bool = false,
        physicalStateWeighting: Bool = false,
        armActionWeight: Float = 1,
        probeSeed: UInt64 = 1
    ) {
        self.collectionEnvironments = collectionEnvironments
        self.contactDwellSteps = contactDwellSteps
        self.probeSteps = probeSteps
        self.epochs = epochs
        self.learningRate = learningRate
        self.maximumGradientNorm = maximumGradientNorm
        self.aggregationRounds = aggregationRounds
        self.initialTeacherMix = initialTeacherMix
        self.finalTeacherMix = finalTeacherMix
        self.rolloutArmNoiseStandardDeviation =
            rolloutArmNoiseStandardDeviation
        self.stateAlignedTeacherQueries = stateAlignedTeacherQueries
        self.physicalStateWeighting = physicalStateWeighting
        self.armActionWeight = armActionWeight
        self.probeSeed = probeSeed
    }

    func validate() throws {
        guard collectionEnvironments > 0,
              contactDwellSteps > 0,
              probeSteps > 0,
              epochs > 0,
              learningRate > 0,
              maximumGradientNorm > 0,
              aggregationRounds > 0,
              aggregationRounds <= epochs,
              (0...1).contains(initialTeacherMix),
              (0...1).contains(finalTeacherMix),
              finalTeacherMix <= initialTeacherMix,
              rolloutArmNoiseStandardDeviation >= 0,
              armActionWeight >= 1 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid humanoid box-carry distillation configuration")
        }
    }
}

public struct HumanoidBoxCarryDistillationReport: Codable, Sendable {
    public var sourceCheckpoint: String
    public var probeReport: String
    public var teacherRows: Int
    public var aggregatedRows: Int
    public var aggregationRounds: Int
    public var stateAlignedTeacherQueries: Bool
    public var physicalStateWeighting: Bool
    public var armActionWeight: Float
    public var warmupSteps: Int
    public var teacherMaximumClearanceMeters: Float
    public var teacherMaximumCarryDistanceMeters: Float
    public var teacherMaximumStableUnsupportedSteps: Int
    public var teacherBilateralContactFraction: Float
    public var teacherPhysicallyLifted: Bool
    public var initialMeanSquaredActionError: Float
    public var finalMeanSquaredActionError: Float
    public var snapshots: [String]
}

/// Distills a simulator-verified, optimizer-discovered contact trajectory into
/// the routed MLX actor. The optimizer is used only to label observations;
/// exported checkpoints contain an ordinary observation-to-action policy and
/// replay without a scripted controller or hidden task state.
public enum HumanoidBoxCarryDistillation {
    private static let firstArmAction = 11
    private static let armActionCount = 8

    public static func run(
        checkpointDirectory: String,
        probeReportPath: String,
        outputDirectory: String,
        configuration: HumanoidBoxCarryDistillationConfiguration
    ) throws -> HumanoidBoxCarryDistillationReport {
        try configuration.validate()
        let metadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self,
            from: Data(contentsOf: URL(fileURLWithPath:
                "\(checkpointDirectory)/metadata.json")))
        guard metadata.task == "humanoid-box-carry-v0",
              let semanticOptions = metadata.taskConfiguration else {
            throw RLEnvironmentError.invalidConfiguration(
                "box-carry distillation requires exact checkpoint metadata")
        }
        let probe = try JSONDecoder().decode(
            HumanoidBoxCarryActuationProbeReport.self,
            from: Data(contentsOf: URL(fileURLWithPath: probeReportPath)))
        guard probe.checkpointDirectory == checkpointDirectory,
              probe.physicallyLifted,
              !probe.bestArmTarget.isEmpty,
              probe.bestArmTarget.count.isMultiple(of: 4) else {
            throw RLEnvironmentError.invalidConfiguration(
                "distillation requires a physically successful trajectory probe")
        }
        let runner = try VectorPolicyRunner(
            checkpointDirectory: checkpointDirectory)
        let replayOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: metadata.task,
            semanticOptions: semanticOptions,
            maxEpisodeSteps: metadata.maxEpisodeSteps,
            controlDecimation: metadata.controlDecimation)
        let anyTask = try BuiltInRLTasks.registry.make(
            metadata.task,
            configuration: RLTaskConfiguration(
                numEnvironments: configuration.collectionEnvironments,
                seed: configuration.probeSeed,
                autoReset: false,
                options: replayOptions))
        guard let task = anyTask as? HumanoidBoxCarryTask,
              metadata.compatibilityMismatches(with: task.spec).isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "distillation task does not match its source checkpoint")
        }

        let demonstrationSeed = configuration.probeSeed
            &+ UInt64(probe.bestGeneration)
        var observation = try task.reset(seed: demonstrationSeed)
        var result = RLStepBatch(spec: task.spec)
        let observationDimension = metadata.observationDimension
        let actionDimension = metadata.actionDimension
        var observations = [Float]()
        var actions = [Float]()
        var expertGates = [Float]()
        var standExpertGates = [Float]()

        func appendRows(
            observation: RLObservationBatch,
            action: RLActionBatch,
            environments: [Int]? = nil
        ) {
            let rows = environments ?? Array(0..<task.spec.numEnvironments)
            let gates = task.policyExpertGates(observation.policy)
            let standGates = task.policyStandExpertGates(observation.policy)
            for environment in rows {
                let observationBase = environment * observationDimension
                observations.append(contentsOf: observation.policy[
                    observationBase..<(observationBase + observationDimension)])
                let actionBase = environment * actionDimension
                actions.append(contentsOf: action.values[
                    actionBase..<(actionBase + actionDimension)])
                expertGates.append(gates[environment])
                standExpertGates.append(standGates[environment])
            }
        }

        var contactStreak = 0
        var warmupSteps = 0
        while warmupSteps < metadata.maxEpisodeSteps,
              contactStreak < configuration.contactDwellSteps {
            let teacher = try policyActions(
                runner: runner, task: task, observation: observation)
            appendRows(observation: observation, action: teacher)
            try task.step(actions: teacher, into: &result)
            observation = result.observations
            warmupSteps += 1
            let left = result.metrics["state/left_hand_contact"]!
            let right = result.metrics["state/right_hand_contact"]!
            contactStreak = zip(left, right).allSatisfy {
                $0.0 > 0.5 && $0.1 > 0.5
            } ? contactStreak + 1 : 0
        }
        guard contactStreak >= configuration.contactDwellSteps else {
            throw RLEnvironmentError.invalidConfiguration(
                "source policy did not reproduce the probe's bilateral grasp")
        }

        let knotCount = probe.bestArmTarget.count / 4
        var referenceProbeObservations = [[Float]]()
        var referenceProbeDeltas = [[Float]]()
        var maximumClearance = -Float.infinity
        var maximumCarryDistance: Float = 0
        var bilateralSteps = 0
        var executedProbeSteps = 0
        var consecutiveStableUnsupportedSteps = 0
        var maximumStableUnsupportedSteps = 0
        var physicallyLifted = false
        for step in 0..<configuration.probeSteps {
            var teacher = try policyActions(
                runner: runner, task: task, observation: observation)
            let delta = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
                probe.bestArmTarget,
                knotCount: knotCount,
                progress: Float(step + 1) / Float(configuration.probeSteps))
            referenceProbeObservations.append(Array(
                observation.policy.prefix(observationDimension)))
            referenceProbeDeltas.append(delta)
            for environment in 0..<task.spec.numEnvironments {
                let base = environment * actionDimension + firstArmAction
                for arm in 0..<armActionCount {
                    teacher.values[base + arm] = simd_clamp(
                        teacher.values[base + arm] + delta[arm],
                        -0.999, 0.999)
                }
            }
            appendRows(observation: observation, action: teacher)
            try task.step(actions: teacher, into: &result)
            executedProbeSteps += 1
            observation = result.observations
            maximumClearance = max(
                maximumClearance,
                result.metrics["state/box_clearance_m"]![0])
            let bilateral = result.metrics["state/left_hand_contact"]![0] > 0.5
                && result.metrics["state/right_hand_contact"]![0] > 0.5
            if bilateral { bilateralSteps += 1 }
            physicallyLifted = physicallyLifted
                || observation.policy[89] > 0.5
            let unsupported = physicallyLifted && bilateral
                && result.metrics["state/box_pedestal_contact"]![0] < 0.5
            if unsupported {
                consecutiveStableUnsupportedSteps += 1
                maximumStableUnsupportedSteps = max(
                    maximumStableUnsupportedSteps,
                    consecutiveStableUnsupportedSteps)
                maximumCarryDistance = max(
                    maximumCarryDistance,
                    result.metrics["state/carry_distance_m"]![0])
            } else {
                consecutiveStableUnsupportedSteps = 0
            }
            let failed = result.truncated[0]
                || (result.terminated[0] && !result.successes[0])
            if failed {
                // The action just appended caused the terminal failure. It is
                // valuable diagnostic evidence but a harmful imitation label:
                // remove that single row so BC learns the verified prefix,
                // not the drop/fall transition at its end.
                observations.removeLast(
                    observationDimension * task.spec.numEnvironments)
                actions.removeLast(
                    actionDimension * task.spec.numEnvironments)
                expertGates.removeLast(task.spec.numEnvironments)
                standExpertGates.removeLast(task.spec.numEnvironments)
                referenceProbeObservations.removeLast()
                referenceProbeDeltas.removeLast()
                break
            }
            // A successful task terminal is only the evaluation threshold,
            // not a physics boundary. Keep collecting the verified teacher
            // trajectory until its grasp genuinely fails (or the requested
            // probe horizon ends), so imitation sees the robust post-success
            // hold instead of only the first qualifying dwell window.
        }
        guard physicallyLifted else {
            throw RLEnvironmentError.invalidConfiguration(
                "optimizer trajectory did not reproduce a physical lift")
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

        let initialTeacherRows = expertGates.count
        let expertMaskArray = task.policyExpertActionMask.map {
            MLXArray(Array($0)).reshaped([1, actionDimension])
        }
        let standMaskArray = task.policyStandExpertActionMask.map {
            MLXArray(Array($0)).reshaped([1, actionDimension])
        }
        let actionDistribution = metadata.ppo.resolvedActionDistribution
        let normalizer = RunningObservationNormalizer(
            snapshot: metadata.normalizer)

        func datasetArrays() -> (
            observations: MLXArray, targets: MLXArray,
            expertGates: MLXArray, standGates: MLXArray,
            trainingWeights: MLXArray
        ) {
            let rowCount = expertGates.count
            let policyObservations = metadata.ppo.normalizeObservations
                ? Array(normalizer.normalize(ContiguousArray(observations)))
                : observations
            // A ~200-step approach contains only a few late load-bearing
            // carry rows. Uniform BC lets already-solved approach actions
            // dominate the gradient, so weight only measured physical state:
            // current grip quality, unsupported lift, and observed handoff.
            let trainingWeights = (0..<rowCount).map { row -> Float in
                guard configuration.physicalStateWeighting else { return 1 }
                let base = row * observationDimension
                return 1
                    + 3 * max(observations[base + 91], 0)
                    + 8 * max(observations[base + 89], 0)
                    + 4 * max(standExpertGates[row], 0)
            }
            return (
                MLXArray(policyObservations)
                    .reshaped([rowCount, observationDimension]),
                MLXArray(actions).reshaped([rowCount, actionDimension]),
                MLXArray(expertGates).reshaped([rowCount, 1]),
                MLXArray(standExpertGates).reshaped([rowCount, 1]),
                MLXArray(trainingWeights))
        }

        func learnerActions(
            observation: RLObservationBatch
        ) throws -> RLActionBatch {
            let rowCount = task.spec.numEnvironments
            let inferenceRows = max(
                rowCount, metadata.inferenceBatchSize ?? rowCount)
            var policyObservations = observation.policy
            let lastObservation = policyObservations.suffix(
                observationDimension)
            for _ in rowCount..<inferenceRows {
                policyObservations.append(contentsOf: lastObservation)
            }
            if metadata.ppo.normalizeObservations {
                policyObservations = normalizer.normalize(policyObservations)
            }
            var gates = task.policyExpertGates(observation.policy)
            var standGates = task.policyStandExpertGates(observation.policy)
            let lastGate = gates.last ?? 0
            let lastStandGate = standGates.last ?? 0
            for _ in rowCount..<inferenceRows {
                gates.append(lastGate)
                standGates.append(lastStandGate)
            }
            let unbounded = policy.forward(
                MLXArray(Array(policyObservations)).reshaped(
                    [inferenceRows, observationDimension]),
                expertGate: MLXArray(Array(gates)).reshaped(
                    [inferenceRows, 1]),
                expertActionMask: expertMaskArray,
                standExpertGate: MLXArray(Array(standGates)).reshaped(
                    [inferenceRows, 1]),
                standExpertActionMask: standMaskArray).mean
            let output: MLXArray
            switch actionDistribution {
            case .squashedGaussian: output = tanh(unbounded)
            case .gaussian: output = unbounded
            }
            eval(output)
            return try RLActionBatch(
                numEnvironments: rowCount,
                actionDimension: actionDimension,
                values: ContiguousArray(output.asArray(Float.self).prefix(
                    rowCount * actionDimension)))
        }

        func alignedReferencePhase(
            observation: ContiguousArray<Float>, environment: Int,
            previousPhase: Int
        ) -> Int {
            precondition(!referenceProbeObservations.isEmpty)
            let rowBase = environment * observationDimension
            let first = min(max(previousPhase, 0),
                            referenceProbeObservations.count - 1)
            let last = min(first + 16,
                           referenceProbeObservations.count - 1)
            var bestPhase = first
            var bestDistance = Float.infinity
            for phase in first...last {
                let reference = referenceProbeObservations[phase]
                var distance: Float = 0
                // Joint configuration disambiguates the pre-lift oscillation.
                for channel in 12..<31 {
                    let delta = observation[rowBase + channel]
                        - reference[channel]
                    distance += 0.25 * delta * delta
                }
                // Object pose/velocity, both hand poses, and object attitude
                // are the directly measured manipulation state.
                for channel in 69..<84 {
                    let delta = observation[rowBase + channel]
                        - reference[channel]
                    distance += 4 * delta * delta
                }
                // Lift, handoff, current load-bearing grip and exact clearance
                // prevent matching a geometrically similar supported phase.
                for channel in 89..<93 {
                    let delta = observation[rowBase + channel]
                        - reference[channel]
                    distance += 10 * delta * delta
                }
                if distance < bestDistance {
                    bestDistance = distance
                    bestPhase = phase
                }
            }
            return bestPhase
        }

        func collectOnPolicyRows(
            round: Int, teacherMix: Float,
            generator: inout ProbeRandomNumberGenerator
        ) throws {
            var rolloutObservation = try task.reset(
                seed: demonstrationSeed &+ UInt64(round + 1) * 10_007)
            var rolloutResult = RLStepBatch(spec: task.spec)
            var rolloutContactStreak = 0
            var rolloutWarmupSteps = 0
            while rolloutWarmupSteps < metadata.maxEpisodeSteps,
                  rolloutContactStreak < configuration.contactDwellSteps {
                let teacher = try policyActions(
                    runner: runner, task: task,
                    observation: rolloutObservation)
                try task.step(actions: teacher, into: &rolloutResult)
                rolloutObservation = rolloutResult.observations
                rolloutWarmupSteps += 1
                let left = rolloutResult.metrics["state/left_hand_contact"]!
                let right = rolloutResult.metrics["state/right_hand_contact"]!
                rolloutContactStreak = zip(left, right).allSatisfy {
                    $0.0 > 0.5 && $0.1 > 0.5
                } ? rolloutContactStreak + 1 : 0
                if rolloutResult.terminated.contains(true)
                    || rolloutResult.truncated.contains(true) {
                    break
                }
            }
            guard rolloutContactStreak >= configuration.contactDwellSteps else {
                throw RLEnvironmentError.invalidConfiguration(
                    "source policy lost its bilateral grasp during dataset "
                        + "aggregation round \(round)")
            }

            var referencePhases = [Int](
                repeating: 0, count: task.spec.numEnvironments)
            var active = [Bool](
                repeating: true, count: task.spec.numEnvironments)
            for probeStep in 0..<configuration.probeSteps {
                var teacher = try policyActions(
                    runner: runner, task: task,
                    observation: rolloutObservation)
                for environment in 0..<task.spec.numEnvironments {
                    let phase = configuration.stateAlignedTeacherQueries
                        ? alignedReferencePhase(
                            observation: rolloutObservation.policy,
                            environment: environment,
                            previousPhase: referencePhases[environment])
                        : min(probeStep, referenceProbeDeltas.count - 1)
                    referencePhases[environment] = phase
                    let delta = referenceProbeDeltas[phase]
                    let base = environment * actionDimension + firstArmAction
                    for arm in 0..<armActionCount {
                        teacher.values[base + arm] = simd_clamp(
                            teacher.values[base + arm] + delta[arm],
                            -0.999, 0.999)
                    }
                }
                let learner = try learnerActions(
                    observation: rolloutObservation)
                appendRows(
                    observation: rolloutObservation, action: teacher,
                    environments: active.indices.filter { active[$0] })
                var executed = learner
                for environment in 0..<task.spec.numEnvironments {
                    let base = environment * actionDimension
                    for action in 0..<actionDimension {
                        let index = base + action
                        var value = teacherMix * teacher.values[index]
                            + (1 - teacherMix) * learner.values[index]
                        if action >= firstArmAction,
                           configuration.rolloutArmNoiseStandardDeviation > 0 {
                            value += (1 - teacherMix)
                                * configuration.rolloutArmNoiseStandardDeviation
                                * generator.normal()
                        }
                        executed.values[index] = simd_clamp(
                            value, -0.999, 0.999)
                    }
                }
                try task.step(actions: executed, into: &rolloutResult)
                rolloutObservation = rolloutResult.observations
                for environment in active.indices where active[environment] {
                    let failed = rolloutResult.truncated[environment]
                        || (rolloutResult.terminated[environment]
                            && !rolloutResult.successes[environment])
                    if failed {
                        active[environment] = false
                    }
                }
                if !active.contains(true) { break }
            }
        }

        let imitationActionWeights = MLXArray(
            [Float](repeating: 1, count: firstArmAction)
                + [Float](repeating: configuration.armActionWeight,
                          count: armActionCount))
            .reshaped([1, actionDimension])
        let lossAndGradient = valueAndGrad(model: policy) {
            (model: VectorActorCritic, args: [MLXArray]) -> [MLXArray] in
            let output = model.forward(
                args[0], expertGate: args[2],
                expertActionMask: expertMaskArray,
                standExpertGate: args[3],
                standExpertActionMask: standMaskArray,
                freezeBaseActor: true, freezeExpertActor: false)
            let predicted: MLXArray
            switch actionDistribution {
            case .squashedGaussian: predicted = tanh(output.mean)
            case .gaussian: predicted = output.mean
            }
            let rowLoss = sum(
                (predicted - args[1]).square() * imitationActionWeights,
                axis: -1) / sum(imitationActionWeights)
            let loss = sum(rowLoss * args[4])
                / clip(sum(args[4]), min: 1,
                       max: Float.greatestFiniteMagnitude)
            return [loss]
        }
        let optimizer = CheckpointableAdam(
            learningRate: configuration.learningRate)
        var arrays = datasetArrays()
        let initialLoss = try imitationLoss(
            policy: policy,
            observations: arrays.observations,
            targets: arrays.targets,
            expertGates: arrays.expertGates,
            standExpertGates: arrays.standGates,
            expertMask: expertMaskArray,
            standMask: standMaskArray,
            actionDistribution: actionDistribution)
        let snapshotEpochs = Set([
            1, 5, 10, 25, 50, 100, 200, 300, 400, 500, 750, 1_000,
            configuration.epochs,
        ].filter { $0 <= configuration.epochs })
        var snapshots = [String]()
        var finalLoss = initialLoss
        var aggregationGenerator = ProbeRandomNumberGenerator(
            seed: configuration.probeSeed &+ 0xDABB_EA6)
        for round in 0..<configuration.aggregationRounds {
            let firstEpoch = round * configuration.epochs
                / configuration.aggregationRounds + 1
            let finalEpoch = (round + 1) * configuration.epochs
                / configuration.aggregationRounds
            for epoch in firstEpoch...finalEpoch {
                let (losses, gradients) = lossAndGradient(policy, [
                    arrays.observations, arrays.targets,
                    arrays.expertGates, arrays.standGates,
                    arrays.trainingWeights,
                ])
                let clipped = clippedGradients(
                    gradients, maximum: configuration.maximumGradientNorm)
                eval(losses + [clipped.squaredNorm])
                finalLoss = losses[0].item(Float.self)
                guard finalLoss.isFinite,
                      clipped.squaredNorm.item(Float.self).isFinite else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "non-finite box-carry distillation at epoch \(epoch)")
                }
                try optimizer.update(
                    model: policy, gradients: clipped.gradients)
                eval(policy)
                optimizer.evaluate()
                if snapshotEpochs.contains(epoch)
                    || epoch == finalEpoch {
                    let directory = outputDirectory + "/checkpoints/"
                        + String(format: "update-%06d", epoch)
                    try saveCheckpoint(
                        policy: policy, metadata: metadata,
                        optimizer: optimizer, completedEpochs: epoch,
                        teacherRows: expertGates.count, directory: directory)
                    if !snapshots.contains(directory) {
                        snapshots.append(directory)
                    }
                }
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
            try collectOnPolicyRows(
                round: round, teacherMix: teacherMix,
                generator: &aggregationGenerator)
            arrays = datasetArrays()
        }
        finalLoss = try imitationLoss(
            policy: policy,
            observations: arrays.observations,
            targets: arrays.targets,
            expertGates: arrays.expertGates,
            standExpertGates: arrays.standGates,
            expertMask: expertMaskArray,
            standMask: standMaskArray,
            actionDistribution: actionDistribution)
        let report = HumanoidBoxCarryDistillationReport(
            sourceCheckpoint: checkpointDirectory,
            probeReport: probeReportPath,
            teacherRows: initialTeacherRows,
            aggregatedRows: expertGates.count,
            aggregationRounds: configuration.aggregationRounds,
            stateAlignedTeacherQueries:
                configuration.stateAlignedTeacherQueries,
            physicalStateWeighting: configuration.physicalStateWeighting,
            armActionWeight: configuration.armActionWeight,
            warmupSteps: warmupSteps,
            teacherMaximumClearanceMeters: maximumClearance,
            teacherMaximumCarryDistanceMeters: maximumCarryDistance,
            teacherMaximumStableUnsupportedSteps:
                maximumStableUnsupportedSteps,
            teacherBilateralContactFraction:
                Float(bilateralSteps) / Float(max(executedProbeSteps, 1)),
            teacherPhysicallyLifted: physicallyLifted,
            initialMeanSquaredActionError: initialLoss,
            finalMeanSquaredActionError: finalLoss,
            snapshots: snapshots)
        try FileManager.default.createDirectory(
            atPath: outputDirectory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: URL(fileURLWithPath:
            "\(outputDirectory)/distillation.json"), options: .atomic)
        return report
    }

    private static func policyActions(
        runner: VectorPolicyRunner,
        task: HumanoidBoxCarryTask,
        observation: RLObservationBatch
    ) throws -> RLActionBatch {
        try runner.actions(
            for: observation,
            expertGates: task.policyExpertGates(observation.policy),
            expertActionMask: task.policyExpertActionMask,
            standExpertGates: task.policyStandExpertGates(observation.policy),
            standExpertActionMask: task.policyStandExpertActionMask,
            auxiliaryExpertGates:
                task.policyAuxiliaryExpertGates(observation.policy),
            auxiliaryExpertActionMask:
                task.policyAuxiliaryExpertActionMask)
    }

    private static func imitationLoss(
        policy: VectorActorCritic,
        observations: MLXArray,
        targets: MLXArray,
        expertGates: MLXArray,
        standExpertGates: MLXArray,
        expertMask: MLXArray?,
        standMask: MLXArray?,
        actionDistribution: PPOActionDistribution
    ) throws -> Float {
        let output = policy.forward(
            observations, expertGate: expertGates,
            expertActionMask: expertMask,
            standExpertGate: standExpertGates,
            standExpertActionMask: standMask).mean
        let actions: MLXArray
        switch actionDistribution {
        case .squashedGaussian: actions = tanh(output)
        case .gaussian: actions = output
        }
        let loss = mean((actions - targets).square())
        eval(loss)
        let value = loss.item(Float.self)
        guard value.isFinite else {
            throw RLEnvironmentError.invalidConfiguration(
                "non-finite box-carry imitation loss")
        }
        return value
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
        policy: VectorActorCritic,
        metadata: VectorPolicyMetadata,
        optimizer: CheckpointableAdam,
        completedEpochs: Int,
        teacherRows: Int,
        directory: String
    ) throws {
        try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
        let weights = Dictionary(uniqueKeysWithValues:
            policy.parameters().flattened().map { ($0.0, $0.1) })
        try MLX.saveToData(arrays: weights).write(
            to: URL(fileURLWithPath: "\(directory)/policy.safetensors"),
            options: .atomic)
        try JSONEncoder().encode(metadata).write(
            to: URL(fileURLWithPath: "\(directory)/metadata.json"),
            options: .atomic)
        try MLX.saveToData(arrays: optimizer.checkpointArrays()).write(
            to: URL(fileURLWithPath: "\(directory)/optimizer.safetensors"),
            options: .atomic)
        // Write progress last: checkpoint discovery treats this as the atomic
        // completeness marker, exactly like PPO snapshots.
        try JSONEncoder().encode(VectorPPOTrainingState(
            completedUpdates: completedEpochs,
            environmentSteps: teacherRows,
            optimizerSteps: completedEpochs,
            adaptiveLearningRate: nil)).write(
                to: URL(fileURLWithPath:
                    "\(directory)/training-state.json"),
                options: .atomic)
    }
}
