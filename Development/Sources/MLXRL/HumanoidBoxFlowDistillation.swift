import SimCore
import PhysicsAVBD
import Robotics
import RL
import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXRandom
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
    /// Preserve the verified state representation and learn only the final
    /// routed action maps. Source rows then anchor a zero correction while
    /// physical-flow rows teach the correction beyond the source frontier.
    public var trainOutputHeadsOnly: Bool
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
        trainOutputHeadsOnly: Bool = true,
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
        self.trainOutputHeadsOnly = trainOutputHeadsOnly
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
    public var trainedOutputHeadsOnly: Bool
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

public struct HumanoidBoxFlowResidualScaleValidation: Codable, Sendable {
    public var scale: Float
    public var checkpoint: String
    public var deterministicMaximumStableCarryDistanceMeters: Float
    public var deterministicFinalSuccessFraction: Float
    public var robustMaximumStableCarryDistanceMeters: Float
    public var robustFinalSuccessFraction: Float
}

public struct HumanoidBoxFlowResidualScaleReport: Codable, Sendable {
    public var baseCheckpoint: String
    public var candidateCheckpoint: String
    public var flowReport: String
    public var validationActionNoiseStandardDeviation: Float
    public var validations: [HumanoidBoxFlowResidualScaleValidation]
    public var selectedScale: Float
    public var selectedCheckpoint: String
    public var selectedDeterministicSuccessFraction: Float
    public var selectedRobustSuccessFraction: Float
    public var selectedMaximumStableCarryDistanceMeters: Float
    public var gatePassed: Bool
}

public struct HumanoidBoxFlowActionChunkConfiguration:
    Codable, Equatable, Sendable
{
    public var horizon: Int
    public var hiddenDimension: Int
    public var epochs: Int
    public var learningRate: Float
    public var maximumGradientNorm: Float
    public var maximumResidualAction: Float
    public var contactDwellSteps: Int
    public var trainingReplayCount: Int
    public var trainingActionNoiseStandardDeviation: Float
    public var exactReplayRowWeight: Float
    public var exactFineTuneEpochs: Int
    public var exactFineTuneLearningRate: Float
    public var exactActionBiasWeight: Float
    public var robustReplayCount: Int
    public var validationActionNoiseStandardDeviation: Float
    public var seed: UInt64

    public init(
        horizon: Int = 8,
        hiddenDimension: Int = 128,
        epochs: Int = 2_000,
        learningRate: Float = 3e-4,
        maximumGradientNorm: Float = 1,
        maximumResidualAction: Float = 2,
        contactDwellSteps: Int = 8,
        trainingReplayCount: Int = 1,
        trainingActionNoiseStandardDeviation: Float = 0,
        exactReplayRowWeight: Float = 1,
        exactFineTuneEpochs: Int = 0,
        exactFineTuneLearningRate: Float = 1e-5,
        exactActionBiasWeight: Float = 0,
        robustReplayCount: Int = 32,
        validationActionNoiseStandardDeviation: Float = 0.001,
        seed: UInt64 = 1
    ) {
        self.horizon = horizon
        self.hiddenDimension = hiddenDimension
        self.epochs = epochs
        self.learningRate = learningRate
        self.maximumGradientNorm = maximumGradientNorm
        self.maximumResidualAction = maximumResidualAction
        self.contactDwellSteps = contactDwellSteps
        self.trainingReplayCount = trainingReplayCount
        self.trainingActionNoiseStandardDeviation =
            trainingActionNoiseStandardDeviation
        self.exactReplayRowWeight = exactReplayRowWeight
        self.exactFineTuneEpochs = exactFineTuneEpochs
        self.exactFineTuneLearningRate = exactFineTuneLearningRate
        self.exactActionBiasWeight = exactActionBiasWeight
        self.robustReplayCount = robustReplayCount
        self.validationActionNoiseStandardDeviation =
            validationActionNoiseStandardDeviation
        self.seed = seed
    }

    func validate() throws {
        guard horizon > 0, horizon <= 32,
              hiddenDimension > 0, epochs > 0,
              learningRate.isFinite, learningRate > 0,
              maximumGradientNorm.isFinite, maximumGradientNorm > 0,
              maximumResidualAction.isFinite,
              maximumResidualAction > 0, maximumResidualAction <= 2,
              contactDwellSteps > 0, trainingReplayCount > 0,
              trainingActionNoiseStandardDeviation.isFinite,
              trainingActionNoiseStandardDeviation >= 0,
              exactReplayRowWeight.isFinite,
              exactReplayRowWeight >= 1,
              exactFineTuneEpochs >= 0,
              exactFineTuneLearningRate.isFinite,
              exactFineTuneLearningRate > 0,
              exactActionBiasWeight.isFinite,
              exactActionBiasWeight >= 0,
              robustReplayCount > 0,
              validationActionNoiseStandardDeviation.isFinite,
              validationActionNoiseStandardDeviation >= 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid humanoid-box action-chunk configuration")
        }
    }
}

public struct HumanoidBoxFlowActionChunkReport: Codable, Sendable {
    public var sourceCheckpoint: String
    public var flowReport: String
    public var checkpoint: String
    public var sourceCheckpointFingerprint: String? = nil
    public var flowReportFingerprint: String? = nil
    public var configuration: HumanoidBoxFlowActionChunkConfiguration? = nil
    public var trainingRows: Int
    public var horizon: Int
    public var epochs: Int
    public var trainingReplayCount: Int
    public var trainingActionNoiseStandardDeviation: Float
    public var exactReplayRowWeight: Float
    public var exactFineTuneEpochs: Int
    public var exactFineTuneLearningRate: Float
    public var exactActionBiasWeight: Float
    public var initialMeanSquaredResidualError: Float
    public var tubeMeanSquaredResidualError: Float
    public var finalMeanSquaredResidualError: Float
    public var maximumTrainingResidualError: Float
    public var maximumExactFirstActionResidualError: Float
    public var maximumAbsoluteExactActionBias: Float
    public var maximumTeacherResidualAction: Float
    public var zeroResidualMaximumStableCarryDistanceMeters: Float
    public var zeroResidualFinalSuccessFraction: Float
    public var teacherMaximumStableCarryDistanceMeters: Float
    public var teacherMaximumStableDestinationProgressMeters: Float? = nil
    public var teacherFinalSuccessFraction: Float
    public var teacherRobustMaximumStableCarryDistanceMeters: Float
    public var teacherRobustMaximumStableDestinationProgressMeters: Float? = nil
    public var teacherRobustFinalSuccessFraction: Float
    public var learnedMaximumStableCarryDistanceMeters: Float
    public var learnedMaximumStableDestinationProgressMeters: Float? = nil
    public var learnedFinalSuccessFraction: Float
    public var learnedRobustMaximumStableCarryDistanceMeters: Float
    public var learnedRobustMaximumStableDestinationProgressMeters: Float? = nil
    public var learnedRobustFinalSuccessFraction: Float
    public var validationActionNoiseStandardDeviation: Float
    public var sourcePrefixIsArchitecturallyBypassed: Bool
    public var checkpointMaximumResidualParityError: Float
    public var gatePassed: Bool
}

public struct HumanoidBoxFlowActionChunkMetadata: Codable, Sendable {
    public var schemaVersion: Int
    public var observationDimension: Int
    public var actionDimension: Int
    public var horizon: Int
    public var hiddenDimension: Int
    public var maximumResidualAction: Float
    public var normalizeObservations: Bool?
    public var sourceCheckpoint: String
    public var flowReport: String
    public var sourceCheckpointFingerprint: String? = nil
    public var flowReportFingerprint: String? = nil
    public var trainingConfiguration:
        HumanoidBoxFlowActionChunkConfiguration? = nil
    public var normalizer: RunningNormalizerSnapshot
}

extension HumanoidBoxFlowActionChunkMetadata {
    static func canonicalPath(_ path: String) -> String {
        URL(fileURLWithPath: path).standardizedFileURL
            .resolvingSymlinksInPath().path
    }

    static func recordedPath(_ recorded: String, matches supplied: String)
        -> Bool
    {
        if canonicalPath(recorded) == canonicalPath(supplied) { return true }
        guard !recorded.hasPrefix("/") else { return false }
        let components = recorded.split(separator: "/")
        guard components.count >= 2,
              !components.contains("..") else { return false }
        let normalizedRelative = String(URL(fileURLWithPath: "/" + recorded)
            .standardizedFileURL.path.dropFirst())
        return canonicalPath(supplied).hasSuffix("/" + normalizedRelative)
    }

    static func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { byte in
            (48...57).contains(byte) || (97...102).contains(byte)
        }
    }

    func validateForLoading() throws {
        guard (1...2).contains(schemaVersion),
              observationDimension > 0,
              actionDimension > 0,
              horizon > 0,
              hiddenDimension > 0,
              maximumResidualAction.isFinite,
              maximumResidualAction > 0,
              normalizer.count.isFinite,
              normalizer.count >= 0,
              normalizer.mean.count == observationDimension,
              normalizer.variance.count == observationDimension,
              normalizer.mean.allSatisfy(\.isFinite),
              normalizer.variance.allSatisfy({
                  $0.isFinite && $0 >= 0
              }),
              !sourceCheckpoint.isEmpty,
              !flowReport.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "unsupported action-chunk checkpoint metadata")
        }
        guard schemaVersion >= 2 else { return }
        guard let sourceCheckpointFingerprint,
              Self.isSHA256(sourceCheckpointFingerprint),
              let flowReportFingerprint,
              Self.isSHA256(flowReportFingerprint),
              let trainingConfiguration,
              normalizeObservations != nil else {
            throw RLEnvironmentError.invalidConfiguration(
                "action-chunk schema v2 requires source/artifact fingerprints "
                    + "and its complete training configuration")
        }
        try trainingConfiguration.validate()
        guard trainingConfiguration.horizon == horizon,
              trainingConfiguration.hiddenDimension == hiddenDimension,
              trainingConfiguration.maximumResidualAction
                == maximumResidualAction else {
            throw RLEnvironmentError.invalidConfiguration(
                "action-chunk training provenance does not match its architecture")
        }
    }

    func isBound(
        toSourceCheckpoint sourcePath: String,
        sourceFingerprint: String,
        flowReportPath: String,
        flowFingerprint: String
    ) -> Bool {
        if schemaVersion >= 2 {
            return sourceCheckpointFingerprint == sourceFingerprint
                && flowReportFingerprint == flowFingerprint
        }
        // Schema v1 predates content fingerprints. It is safe to replay only
        // against the exact paths recorded at training time; copied or
        // substituted dependencies must be upgraded to schema v2.
        return Self.recordedPath(sourceCheckpoint, matches: sourcePath)
            && Self.recordedPath(flowReport, matches: flowReportPath)
    }
}

public struct HumanoidBoxFlowActionChunkEvaluationReport:
    Codable, Sendable
{
    public var sourceCheckpoint: String
    public var flowReport: String
    public var actionChunkCheckpoint: String
    public var actionChunkFingerprint: String
    public var sourceCheckpointFingerprint: String? = nil
    public var flowReportFingerprint: String? = nil
    public var evaluationSeed: UInt64
    public var replayCount: Int
    public var targetControlSteps: Int
    public var actionNoiseStandardDeviation: Float
    public var actionNoiseScope: String
    public var teacherMaximumStableCarryDistanceMeters: Float
    public var teacherMaximumStableDestinationProgressMeters: Float? = nil
    public var teacherSuccessFraction: Float
    public var learnedMaximumStableCarryDistanceMeters: Float
    public var learnedMaximumStableDestinationProgressMeters: Float? = nil
    public var learnedSuccessFraction: Float
    public var sourcePrefixIsArchitecturallyBypassed: Bool
    public var gatePassed: Bool
}

final class HumanoidBoxFlowActionChunkPolicy: Module {
    @ModuleInfo var hidden1: Linear
    @ModuleInfo var hidden2: Linear
    @ModuleInfo var output: Linear

    let horizon: Int
    let actionDimension: Int
    let maximumResidualAction: Float

    init(
        observationDimension: Int, actionDimension: Int, horizon: Int,
        hiddenDimension: Int, maximumResidualAction: Float
    ) {
        self.horizon = horizon
        self.actionDimension = actionDimension
        self.maximumResidualAction = maximumResidualAction
        hidden1 = Linear(observationDimension, hiddenDimension)
        hidden2 = Linear(hiddenDimension, hiddenDimension)
        // The exact zero output is the safety contract: before any update the
        // model is the commissioned controller, not merely close to it.
        output = Linear(
            weight: MLXArray.zeros(
                [horizon * actionDimension, hiddenDimension]),
            bias: MLXArray.zeros([horizon * actionDimension]))
    }

    func callAsFunction(_ observations: MLXArray) -> MLXArray {
        let first = tanh(hidden1(observations))
        let second = tanh(hidden2(first))
        return maximumResidualAction * tanh(output(second))
    }
}

/// Immutable MLX inference wrapper for a learned frontier action chunk. It
/// consumes the same raw observation rows as the source policy and returns
/// row-major `[environment, horizon, action]` residuals.
public final class HumanoidBoxFlowActionChunkRunner {
    public let metadata: HumanoidBoxFlowActionChunkMetadata
    private let policy: HumanoidBoxFlowActionChunkPolicy
    private let normalizer: RunningObservationNormalizer

    public init(checkpointDirectory: String) throws {
        metadata = try JSONDecoder().decode(
            HumanoidBoxFlowActionChunkMetadata.self,
            from: Data(contentsOf: URL(fileURLWithPath:
                "\(checkpointDirectory)/metadata.json")))
        try metadata.validateForLoading()
        policy = HumanoidBoxFlowActionChunkPolicy(
            observationDimension: metadata.observationDimension,
            actionDimension: metadata.actionDimension,
            horizon: metadata.horizon,
            hiddenDimension: metadata.hiddenDimension,
            maximumResidualAction: metadata.maximumResidualAction)
        let weights = try loadArrays(url: URL(fileURLWithPath:
            "\(checkpointDirectory)/policy.safetensors"))
        try policy.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(policy)
        normalizer = RunningObservationNormalizer(
            snapshot: metadata.normalizer)
    }

    public func residualChunks(
        for policyObservations: ContiguousArray<Float>
    ) throws -> ContiguousArray<Float> {
        guard policyObservations.count.isMultiple(
                of: metadata.observationDimension),
              !policyObservations.isEmpty else {
            throw RLEnvironmentError.invalidObservationCount(
                expected: metadata.observationDimension,
                actual: policyObservations.count)
        }
        if let index = policyObservations.firstIndex(where: { !$0.isFinite }) {
            throw RLEnvironmentError.invalidConfiguration(
                "action-chunk observation at flat index \(index) is not finite")
        }
        let rows = policyObservations.count / metadata.observationDimension
        let normalized = metadata.normalizeObservations == false
            ? policyObservations : normalizer.normalize(policyObservations)
        let output = policy(MLXArray(Array(normalized)).reshaped(
            [rows, metadata.observationDimension]))
        eval(output)
        let values = ContiguousArray(output.asArray(Float.self))
        if let index = values.firstIndex(where: { !$0.isFinite }) {
            throw RLEnvironmentError.nonFiniteAction(index: index)
        }
        return values
    }
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
    var graspQuality: Float = 1
}

enum HumanoidBoxStableCarryVerifier {
    static func failed(previouslyFailed: Bool,
                       sample: HumanoidBoxStableCarrySample) -> Bool {
        previouslyFailed || sample.terminated || sample.truncated
    }

    static func isStable(
        previouslyFailed: Bool, sample: HumanoidBoxStableCarrySample,
        requiredGraspQuality: Float = 0
    ) -> Bool {
        !failed(previouslyFailed: previouslyFailed, sample: sample)
            && sample.leftContact > 0.5 && sample.rightContact > 0.5
            && sample.sourceSupportContact < 0.5 && sample.lifted
            && sample.rootUprightAlignment > 0.9
            && sample.boxUprightAlignment > 0.9
            && sample.clearanceMeters >= 0.01
            && sample.graspQuality >= requiredGraspQuality
    }

    static func isFinalSuccess(
        previouslyFailed: Bool, sample: HumanoidBoxStableCarrySample,
        maximumStableCarryDistanceMeters: Float,
        requiredCarryDistanceMeters: Float,
        maximumStableDestinationProgressMeters: Float = .infinity,
        requiredDestinationProgressMeters: Float = 0,
        requiredGraspQuality: Float = 0
    ) -> Bool {
        isStable(
            previouslyFailed: previouslyFailed, sample: sample,
            requiredGraspQuality: requiredGraspQuality)
            && sample.carryDistanceMeters >= requiredCarryDistanceMeters
            && maximumStableCarryDistanceMeters
                >= requiredCarryDistanceMeters
            && (requiredDestinationProgressMeters <= 0
                || maximumStableDestinationProgressMeters
                    >= requiredDestinationProgressMeters)
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

    static func fileFingerprint(_ path: String) throws -> String {
        let digest = SHA256.hash(data: try Data(
            contentsOf: URL(fileURLWithPath: path)))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    static func actionChunkFingerprint(_ directory: String) throws -> String {
        var hasher = SHA256()
        for name in ["metadata.json", "policy.safetensors"] {
            let data = try Data(contentsOf: URL(fileURLWithPath: directory)
                .appendingPathComponent(name))
            hasher.update(data: Data(name.utf8))
            hasher.update(data: Data([0]))
            hasher.update(data: data)
            hasher.update(data: Data([0]))
        }
        return hasher.finalize().map {
            String(format: "%02x", $0)
        }.joined()
    }

    struct Artifact {
        var sourceStages: [HumanoidBoxPhysicalFlowStage]
        var sourceWarmupAppliedActions: [[Float]]?
        var sourceAppliedActions: [[Float]]?
        var targetTrajectory: [Float]
        var targetTrajectorySequence: [[Float]]?
        var targetTrajectorySequencePhaseSteps: [Int]?
        /// Exact full-body actions committed by the simulator-verified target.
        /// When present these supersede a lossy reconstruction from trajectory
        /// parameters, including state-dependent grasp feedback.
        var targetAppliedActions: [[Float]]?
        var targetSteps: Int
        var targetDuration: Int
        var targetLocomotionCheckpointDirectory: String?
        var targetLocomotionCommandSpeed: Float?
        var targetForwardOnlyBaseCommand: Bool
        var targetHolonomicBaseCommand: Bool
        var canonicalizeReplicasBeforeTarget: Bool
        var legBlendKnotCount: Int
        var legResidualKnotCount: Int
        var maximumLegResidualAction: Float
        var torsoResidualKnotCount: Int
        var maximumTorsoResidualAction: Float
        var armAsymmetryKnotCount: Int
        var maximumArmAsymmetryAction: Float
        var teacherCarryDistance: Float
        var requiredCarryDistance: Float
        var requiredDestinationProgress: Float
        var requiredGraspQuality: Float
        var physicalBalanceOnly: Bool
        var reusableFrontier: Bool
        /// The artifact carries exact actions or controller semantics that the
        /// original whole-policy DAgger path cannot reproduce losslessly.
        var hasModernExecutionSemantics: Bool
    }

    private static func requireLosslessLegacyReplay(_ artifact: Artifact)
        throws
    {
        guard !artifact.hasModernExecutionSemantics else {
            throw RLEnvironmentError.invalidConfiguration(
                "legacy whole-policy flow distillation cannot replay this "
                    + "exact/structured artifact without changing its physical "
                    + "semantics; use trainActionChunk/evaluateActionChunk or "
                    + "ReplayController")
        }
    }

    /// Interactive one-environment replay of the exact commissioned prefix
    /// followed by either its simulator teacher or a learned action chunk.
    /// The controller pauses cleanly at the measured endpoint and separately
    /// reports whether that endpoint is a reusable, recovery-safe frontier.
    public final class ReplayController: RLActionProvider {
        public let actionProviderID = "humanoid-box-flow-replay-v1"
        public private(set) var isComplete = false
        public private(set) var phaseDescription = "waiting for grasp"
        public var targetPhaseActive: Bool {
            contactStreak >= contactDwellSteps
                && sourceStage >= artifact.sourceStages.count
                && targetStep < artifact.targetSteps
                && !isComplete
        }
        public var targetControlSteps: Int { artifact.targetSteps }
        public var isBalanceOnly: Bool { artifact.physicalBalanceOnly }
        public var isReusableFrontier: Bool { artifact.reusableFrontier }

        private let runner: VectorPolicyRunner
        private let locomotionRunners: [String: VectorPolicyRunner]
        private let chunkRunner: HumanoidBoxFlowActionChunkRunner?
        private let artifact: Artifact
        private let actionDimension: Int
        private let armParameterCount: Int
        private let armKnotCount: Int
        private let contactDwellSteps: Int
        private var contactStreak = 0
        private var warmupActionOffset = 0
        private var sourceStage = 0
        private var sourceStep = 0
        private var sourceActionOffset = 0
        private var targetStep = 0
        private var targetBoundaryCanonicalized = false

        public init(
            checkpointDirectory: String,
            flowReportPath: String,
            actionChunkCheckpointDirectory: String? = nil,
            contactDwellSteps: Int = 8
        ) throws {
            guard contactDwellSteps > 0 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "flow replay contact dwell must be positive")
            }
            runner = try VectorPolicyRunner(
                checkpointDirectory: checkpointDirectory)
            chunkRunner = try actionChunkCheckpointDirectory.map {
                try HumanoidBoxFlowActionChunkRunner(
                    checkpointDirectory: $0)
            }
            artifact = try HumanoidBoxFlowDistillation.loadArtifact(
                flowReportPath, allowPhysicalBalance: true)
            let sourceFingerprint = try VectorPPOTrainer
                .checkpointFingerprint(directory: checkpointDirectory)
            let flowFingerprint = try HumanoidBoxFlowDistillation
                .fileFingerprint(flowReportPath)
            let locomotionDirectories = Set(
                artifact.sourceStages.compactMap(
                    \.locomotionCheckpointDirectory)
                    + [artifact.targetLocomotionCheckpointDirectory]
                        .compactMap { $0 })
            var loadedLocomotionRunners = [String: VectorPolicyRunner]()
            for directory in locomotionDirectories.sorted() {
                let locomotion = try VectorPolicyRunner(
                    checkpointDirectory: directory)
                guard locomotion.metadata.task == "humanoid-isaac-flat-v0",
                      locomotion.metadata.observationDimension
                        == HumanoidIsaacVelocityTask.observationDimension,
                      locomotion.metadata.actionDimension
                        == runner.metadata.actionDimension else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "flow replay locomotion checkpoint/schema mismatch")
                }
                loadedLocomotionRunners[directory] = locomotion
            }
            locomotionRunners = loadedLocomotionRunners
            actionDimension = runner.metadata.actionDimension
            armParameterCount = artifact.targetTrajectory.count
                - artifact.legBlendKnotCount
                - HumanoidBoxFlowDistillation.legActionCount
                    * artifact.legResidualKnotCount
                - artifact.torsoResidualKnotCount
                - 4 * artifact.armAsymmetryKnotCount
            armKnotCount = max(armParameterCount / 4, 0)
            self.contactDwellSteps = contactDwellSteps
            let sourceControlSteps = artifact.sourceStages.reduce(0) {
                $0 + $1.controlSteps
            }
            guard runner.metadata.task == "humanoid-box-carry-v0",
                  actionDimension
                    == HumanoidBoxFlowDistillation.firstArmAction
                        + HumanoidBoxFlowDistillation.armActionCount,
                  artifact.sourceWarmupAppliedActions?.allSatisfy({
                      $0.count == actionDimension
                  }) ?? true,
                  artifact.sourceAppliedActions?.allSatisfy({
                      $0.count == actionDimension
                  }) ?? true,
                  artifact.sourceAppliedActions.map({
                      $0.count == sourceControlSteps
                  }) ?? true,
                  artifact.targetAppliedActions?.allSatisfy({
                      $0.count == actionDimension
                  }) ?? true,
                  artifact.targetAppliedActions.map({
                      $0.count == artifact.targetSteps
                  }) ?? true,
                  artifact.sourceStages.allSatisfy({ stage in
                      stage.appliedNormalizedActions.map { actions in
                          actions.count == stage.controlSteps
                              && actions.allSatisfy {
                                  $0.count == actionDimension
                              }
                      } ?? true
                  }),
                  armParameterCount > 0,
                  armParameterCount.isMultiple(of: 4),
                  chunkRunner.map({ chunk in
                      chunk.metadata.observationDimension
                          == runner.metadata.observationDimension
                          && chunk.metadata.actionDimension == actionDimension
                          && chunk.metadata.isBound(
                              toSourceCheckpoint: checkpointDirectory,
                              sourceFingerprint: sourceFingerprint,
                              flowReportPath: flowReportPath,
                              flowFingerprint: flowFingerprint)
                  }) ?? true else {
                throw RLEnvironmentError.invalidConfiguration(
                    "flow replay checkpoint/schema mismatch")
            }
        }

        public func reset(
            for task: any VectorizedRLTask,
            environments: [Int]?, observation: RLObservationBatch
        ) throws {
            guard task is HumanoidBoxCarryTask else {
                throw RLEnvironmentError.invalidConfiguration(
                    "box-flow replay requires the carry task")
            }
            contactStreak = 0
            warmupActionOffset = 0
            sourceStage = 0
            sourceStep = 0
            sourceActionOffset = 0
            targetStep = 0
            targetBoundaryCanonicalized = false
            isComplete = false
            phaseDescription = "waiting for grasp"
        }

        public func actions(
            for observation: RLObservationBatch,
            task anyTask: any VectorizedRLTask
        ) throws -> RLActionBatch {
            guard let task = anyTask as? HumanoidBoxCarryTask else {
                throw RLEnvironmentError.invalidConfiguration(
                    "box-flow replay task type mismatch")
            }
            if let warmup = artifact.sourceWarmupAppliedActions,
               warmupActionOffset < warmup.count {
                phaseDescription =
                    "verified grasp warm-up \(warmupActionOffset + 1)/\(warmup.count)"
                let action = try repeatedAction(
                    warmup[warmupActionOffset], for: task)
                warmupActionOffset += 1
                if warmupActionOffset == warmup.count {
                    contactStreak = contactDwellSteps
                }
                return action
            }
            if artifact.sourceWarmupAppliedActions?.isEmpty ?? true,
               sourceStage == 0, sourceStep == 0, targetStep == 0,
               !isComplete {
                let allContact = (0..<task.spec.numEnvironments)
                    .allSatisfy { environment in
                        let row = environment
                            * task.spec.observation.elementCount
                        return observation.policy[row + 84] > 0.5
                            && observation.policy[row + 85] > 0.5
                    }
                contactStreak = allContact ? contactStreak + 1 : 0
                if contactStreak < contactDwellSteps {
                    phaseDescription = "establishing bilateral grasp \(contactStreak)/\(contactDwellSteps)"
                    return try baseActions(task, observation)
                }
            }
            if sourceStage < artifact.sourceStages.count {
                let stage = artifact.sourceStages[sourceStage]
                var sourceObservation = observation
                if sourceStep == 0,
                   stage.canonicalizeReplicasBeforeExecution == true {
                    task.canonicalizeSpeculationReplicas(
                        observation: &sourceObservation,
                        sourceEnvironment: 0)
                }
                phaseDescription = "verified source \(sourceStage + 1)/\(artifact.sourceStages.count) · \(sourceStep + 1)/\(stage.controlSteps)"
                let action: RLActionBatch
                if let committed =
                        artifact.sourceAppliedActions?[sourceActionOffset] {
                    action = try repeatedAction(
                        committed, for: task)
                } else if let committed =
                            stage.appliedNormalizedActions?[sourceStep] {
                    action = try repeatedAction(committed, for: task)
                } else {
                    action = stage.policyOnly == true
                        ? try baseActions(task, sourceObservation)
                        : try flowActions(
                            task, sourceObservation,
                            trajectory: stage.trajectorySequence?[sourceStep]
                                ?? stage.trajectory,
                            step: stage.trajectoryEvaluationStep(
                                at: sourceStep),
                            duration: stage.trajectorySequence == nil
                                ? (stage.trajectoryDurationSteps
                                    ?? stage.controlSteps)
                                : stage.trajectorySequenceStepDenominator!,
                            forwardOnlyBaseCommand:
                                stage.forwardOnlyBaseCommand == true,
                            holonomicBaseCommand:
                                stage.holonomicBaseCommand == true,
                            locomotionCheckpointDirectory:
                                stage.locomotionCheckpointDirectory,
                            locomotionCommandSpeed:
                                stage.locomotionCommandSpeed)
                }
                sourceActionOffset += 1
                sourceStep += 1
                if sourceStep == stage.controlSteps {
                    sourceStage += 1
                    sourceStep = 0
                }
                return action
            }
            guard targetStep < artifact.targetSteps else {
                isComplete = true
                phaseDescription = artifact.reusableFrontier
                    ? "reusable recovery-safe frontier"
                    : "measured finite-prefix endpoint"
                return try baseActions(task, observation)
            }
            var targetObservation = observation
            if !targetBoundaryCanonicalized {
                if artifact.canonicalizeReplicasBeforeTarget {
                    task.canonicalizeSpeculationReplicas(
                        observation: &targetObservation,
                        sourceEnvironment: 0)
                }
                targetBoundaryCanonicalized = true
            }
            phaseDescription = chunkRunner == nil
                ? "exact feedback teacher \(targetStep + 1)/\(artifact.targetSteps)"
                : "learned MLX action chunk \(targetStep + 1)/\(artifact.targetSteps)"
            let action: RLActionBatch
            if let chunkRunner {
                var loaded = try baseActions(task, targetObservation)
                let residual = try chunkRunner.residualChunks(
                    for: targetObservation.policy)
                for environment in 0..<task.spec.numEnvironments {
                    let actionRow = environment * actionDimension
                    let residualRow = environment
                        * chunkRunner.metadata.horizon * actionDimension
                    for component in 0..<actionDimension {
                        loaded.values[actionRow + component] = simd_clamp(
                            loaded.values[actionRow + component]
                                + residual[residualRow + component],
                            -0.999, 0.999)
                    }
                }
                action = loaded
            } else {
                if let committed = artifact.targetAppliedActions?[targetStep] {
                    action = try repeatedAction(committed, for: task)
                } else {
                    let trajectory = artifact
                        .targetTrajectorySequence?[targetStep]
                        ?? artifact.targetTrajectory
                    action = try flowActions(
                        task, targetObservation, trajectory: trajectory,
                        step: artifact.targetTrajectorySequence == nil
                            ? targetStep
                            : artifact.targetTrajectorySequencePhaseSteps?[
                                targetStep] ?? 0,
                        duration: artifact.targetDuration,
                        forwardOnlyBaseCommand:
                            artifact.targetForwardOnlyBaseCommand,
                        holonomicBaseCommand:
                            artifact.targetHolonomicBaseCommand,
                        locomotionCheckpointDirectory:
                            artifact.targetLocomotionCheckpointDirectory,
                        locomotionCommandSpeed:
                            artifact.targetLocomotionCommandSpeed)
                }
            }
            targetStep += 1
            if targetStep == artifact.targetSteps { isComplete = true }
            return action
        }

        private func repeatedAction(
            _ action: [Float], for task: HumanoidBoxCarryTask
        ) throws -> RLActionBatch {
            var values = ContiguousArray<Float>()
            values.reserveCapacity(
                task.spec.numEnvironments * actionDimension)
            for _ in 0..<task.spec.numEnvironments {
                values.append(contentsOf: action)
            }
            return try RLActionBatch(
                numEnvironments: task.spec.numEnvironments,
                actionDimension: actionDimension, values: values)
        }

        private func baseActions(
            _ task: HumanoidBoxCarryTask,
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

        private func flowActions(
            _ task: HumanoidBoxCarryTask,
            _ observation: RLObservationBatch,
            trajectory: [Float], step: Int, duration: Int,
            forwardOnlyBaseCommand: Bool = false,
            holonomicBaseCommand: Bool = false,
            locomotionCheckpointDirectory: String? = nil,
            locomotionCommandSpeed: Float? = nil
        ) throws -> RLActionBatch {
            var loaded = try baseActions(task, observation)
            let base = artifact.legBlendKnotCount > 0
                ? try baseLegActions(
                    task, observation,
                    forwardOnlyBaseCommand: forwardOnlyBaseCommand,
                    holonomicBaseCommand: holonomicBaseCommand,
                    locomotionCheckpointDirectory:
                        locomotionCheckpointDirectory,
                    locomotionCommandSpeed: locomotionCommandSpeed)
                : nil
            let progress = Float(step + 1) / Float(duration)
            for environment in 0..<task.spec.numEnvironments {
                let actionRow = environment * actionDimension
                if let base {
                    let blend = HumanoidBoxPhysicalFlowExperiment
                        .legBlendFraction(
                            trajectory, progress: progress,
                            armParameterCount: armParameterCount,
                            knotCount: artifact.legBlendKnotCount)
                    for component in 0..<HumanoidBoxFlowDistillation
                            .legActionCount {
                        let index = actionRow + component
                        loaded.values[index] = (1 - blend)
                            * loaded.values[index]
                            + blend * base.values[index]
                    }
                }
                if artifact.legResidualKnotCount > 0 {
                    for component in 0..<HumanoidBoxFlowDistillation
                            .legActionCount {
                        let index = actionRow + component
                        loaded.values[index] = simd_clamp(
                            loaded.values[index]
                                + HumanoidBoxPhysicalFlowExperiment
                                    .legResidualAction(
                                        trajectory, action: component,
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
                if artifact.torsoResidualKnotCount > 0 {
                    let index = actionRow + 10
                    loaded.values[index] = simd_clamp(
                        loaded.values[index]
                            + HumanoidBoxPhysicalFlowExperiment
                                .torsoResidualAction(
                                    trajectory, progress: progress,
                                    armParameterCount: armParameterCount,
                                    blendKnotCount:
                                        artifact.legBlendKnotCount,
                                    legResidualKnotCount:
                                        artifact.legResidualKnotCount,
                                    torsoResidualKnotCount:
                                        artifact.torsoResidualKnotCount,
                                    maximumAction:
                                        artifact.maximumTorsoResidualAction),
                        -0.999, 0.999)
                }
                let arm = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
                    Array(trajectory.prefix(armParameterCount)),
                    knotCount: armKnotCount, progress: progress)
                for component in 0..<HumanoidBoxFlowDistillation
                        .armActionCount {
                    let index = actionRow
                        + HumanoidBoxFlowDistillation.firstArmAction
                        + component
                    loaded.values[index] = simd_clamp(
                        loaded.values[index] + arm[component],
                        -0.999, 0.999)
                }
                if artifact.armAsymmetryKnotCount > 0 {
                    for component in 0..<4 {
                        let correction = HumanoidBoxPhysicalFlowExperiment
                            .armAsymmetryAction(
                            trajectory, action: component,
                            progress: progress,
                            armParameterCount: armParameterCount,
                            blendKnotCount: artifact.legBlendKnotCount,
                            legResidualKnotCount:
                                artifact.legResidualKnotCount,
                            torsoResidualKnotCount:
                                artifact.torsoResidualKnotCount,
                            asymmetryKnotCount:
                                artifact.armAsymmetryKnotCount,
                            maximumAction:
                                artifact.maximumArmAsymmetryAction)
                        let left = actionRow
                            + HumanoidBoxFlowDistillation.firstArmAction
                            + component
                        let right = left + 4
                        loaded.values[left] = simd_clamp(
                            loaded.values[left] + correction,
                            -0.999, 0.999)
                        loaded.values[right] = simd_clamp(
                            loaded.values[right] - correction,
                            -0.999, 0.999)
                    }
                }
            }
            return loaded
        }

        private func baseLegActions(
            _ task: HumanoidBoxCarryTask,
            _ observation: RLObservationBatch,
            forwardOnlyBaseCommand: Bool,
            holonomicBaseCommand: Bool,
            locomotionCheckpointDirectory: String?,
            locomotionCommandSpeed: Float?
        ) throws -> RLActionBatch {
            var baseObservation = observation
            HumanoidBoxPhysicalFlowExperiment.overrideLocomotionCommand(
                &baseObservation.policy,
                numEnvironments: task.spec.numEnvironments,
                forwardOnly: forwardOnlyBaseCommand,
                holonomic: holonomicBaseCommand,
                speed: locomotionCommandSpeed)
            if let locomotionCheckpointDirectory {
                guard let locomotion = locomotionRunners[
                        locomotionCheckpointDirectory] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "flow replay locomotion checkpoint was not loaded")
                }
                let projected = HumanoidBoxPhysicalFlowExperiment
                    .isolatedLegPolicyObservations(
                        baseObservation.policy,
                        numEnvironments: task.spec.numEnvironments)
                return try RLActionBatch(
                    numEnvironments: task.spec.numEnvironments,
                    actionDimension: actionDimension,
                    values: locomotion.actions(for: projected))
            }
            return try baseActions(
                task, baseObservation, disableAuxiliary: true)
        }
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
        try requireLosslessLegacyReplay(artifact)
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
            - artifact.torsoResidualKnotCount
            - 4 * artifact.armAsymmetryKnotCount
        guard armParameterCount > 0,
              armParameterCount.isMultiple(of: 4) else {
            throw RLEnvironmentError.invalidConfiguration(
                "flow distillation trajectory schema is invalid")
        }
        let armKnotCount = armParameterCount / 4

        func targetTrajectory(at phase: Int) -> (trajectory: [Float],
                                                  step: Int) {
            if let sequence = artifact.targetTrajectorySequence {
                return (
                    sequence[phase],
                    artifact.targetTrajectorySequencePhaseSteps?[phase] ?? 0)
            }
            return (artifact.targetTrajectory, phase)
        }

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
            try flowActions(
                task: task, observation: observation,
                trajectories: [[Float]](
                    repeating: trajectory,
                    count: task.spec.numEnvironments),
                steps: steps, duration: duration)
        }

        func flowActions(
            task: HumanoidBoxCarryTask, observation: RLObservationBatch,
            trajectories: [[Float]], steps: [Int], duration: Int
        ) throws -> RLActionBatch {
            precondition(steps.count == task.spec.numEnvironments)
            precondition(trajectories.count == task.spec.numEnvironments)
            var loaded = try runnerActions(task, observation)
            let base = artifact.legBlendKnotCount > 0
                ? try runnerActions(task, observation, disableAuxiliary: true)
                : nil
            for environment in 0..<task.spec.numEnvironments {
                let trajectory = trajectories[environment]
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
                if artifact.torsoResidualKnotCount > 0 {
                    let index = environment * metadata.actionDimension + 10
                    loaded.values[index] = simd_clamp(
                        loaded.values[index]
                            + HumanoidBoxPhysicalFlowExperiment
                                .torsoResidualAction(
                                    trajectory, progress: progress,
                                    armParameterCount: armParameterCount,
                                    blendKnotCount:
                                        artifact.legBlendKnotCount,
                                    legResidualKnotCount:
                                        artifact.legResidualKnotCount,
                                    torsoResidualKnotCount:
                                        artifact.torsoResidualKnotCount,
                                    maximumAction:
                                        artifact.maximumTorsoResidualAction),
                        -0.999, 0.999)
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
                if artifact.armAsymmetryKnotCount > 0 {
                    for action in 0..<4 {
                        let correction = HumanoidBoxPhysicalFlowExperiment
                            .armAsymmetryAction(
                                trajectory, action: action,
                                progress: progress,
                                armParameterCount: armParameterCount,
                                blendKnotCount:
                                    artifact.legBlendKnotCount,
                                legResidualKnotCount:
                                    artifact.legResidualKnotCount,
                                torsoResidualKnotCount:
                                    artifact.torsoResidualKnotCount,
                                asymmetryKnotCount:
                                    artifact.armAsymmetryKnotCount,
                                maximumAction:
                                    artifact.maximumArmAsymmetryAction)
                        loaded.values[row + action] = simd_clamp(
                            loaded.values[row + action] + correction,
                            -0.999, 0.999)
                        loaded.values[row + 4 + action] = simd_clamp(
                            loaded.values[row + 4 + action] - correction,
                            -0.999, 0.999)
                    }
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
        guard artifact.sourceWarmupAppliedActions?.allSatisfy({
            $0.count == actionDimension
        }) ?? true else {
            throw RLEnvironmentError.invalidConfiguration(
                "committed source warm-up action schema does not match checkpoint")
        }
        guard artifact.sourceAppliedActions?.allSatisfy({
            $0.count == actionDimension
        }) ?? true else {
            throw RLEnvironmentError.invalidConfiguration(
                "committed source action schema does not match checkpoint")
        }
        guard artifact.targetAppliedActions?.allSatisfy({
            $0.count == actionDimension
        }) ?? true else {
            throw RLEnvironmentError.invalidConfiguration(
                "committed target action schema does not match checkpoint")
        }
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
            if stage.canonicalizeReplicasBeforeExecution == true {
                task.canonicalizeSpeculationReplicas(
                    observation: &observation, result: &result)
            }
            for step in 0..<stage.controlSteps {
                let action = stage.policyOnly == true
                    ? try runnerActions(task, observation)
                    : try flowActions(
                        task: task, observation: observation,
                        trajectory: stage.trajectorySequence?[step]
                            ?? stage.trajectory,
                        step: stage.trajectoryEvaluationStep(at: step),
                        duration: stage.trajectorySequence == nil
                            ? (stage.trajectoryDurationSteps
                                ?? stage.controlSteps)
                            : stage.trajectorySequenceStepDenominator!)
                appendRows(
                    task: task, observation: observation,
                    action: action,
                    weight: stage.policyOnly == true
                        ? configuration.policySourceRowWeight : 1)
                try task.step(actions: action, into: &result)
                observation = result.observations
            }
        }
        if artifact.canonicalizeReplicasBeforeTarget {
            task.canonicalizeSpeculationReplicas(
                observation: &observation, result: &result)
        }
        var referenceTargetObservations = [[Float]]()
        for step in 0..<artifact.targetSteps {
            let target = targetTrajectory(at: step)
            let action = try flowActions(
                task: task, observation: observation,
                trajectory: target.trajectory, step: target.step,
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
              result.metrics["state/box_pedestal_contact"]![0] < 0.5,
              result.metrics["state/grasp_quality"]![0]
                >= artifact.requiredGraspQuality else {
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
                freezeStandActor: false, freezeAuxiliaryActor: false,
                freezeStandActorBackbone:
                    configuration.trainOutputHeadsOnly,
                freezeAuxiliaryActorBackbone:
                    configuration.trainOutputHeadsOnly).mean
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
                if stage.canonicalizeReplicasBeforeExecution == true {
                    rolloutTask.canonicalizeSpeculationReplicas(
                        observation: &rolloutObservation,
                        result: &rolloutResult)
                }
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
                            trajectory: stage.trajectorySequence?[step]
                                ?? stage.trajectory,
                            step: stage.trajectoryEvaluationStep(at: step),
                            duration: stage.trajectorySequence == nil
                                ? (stage.trajectoryDurationSteps
                                    ?? stage.controlSteps)
                                : stage
                                    .trajectorySequenceStepDenominator!)
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
            if artifact.canonicalizeReplicasBeforeTarget {
                rolloutTask.canonicalizeSpeculationReplicas(
                    observation: &rolloutObservation,
                    result: &rolloutResult)
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
                let teacherTrajectories = phases.map {
                    targetTrajectory(at: $0).trajectory
                }
                let teacherSteps = phases.map {
                    targetTrajectory(at: $0).step
                }
                let teacher = try flowActions(
                    task: rolloutTask,
                    observation: rolloutObservation,
                    trajectories: teacherTrajectories,
                    steps: teacherSteps, duration: artifact.targetDuration)
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
            trainedOutputHeadsOnly: configuration.trainOutputHeadsOnly,
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

    /// Interpret a learned checkpoint as a residual direction from a
    /// commissioned base policy and select its scale by physical replay. At
    /// scale zero the saved policy is bit-for-bit behavior-equivalent to the
    /// base; no supervised metric can override the deterministic and noisy
    /// simulator ranking.
    public static func evaluateResidualScales(
        baseCheckpointDirectory: String,
        candidateCheckpointDirectory: String,
        flowReportPath: String,
        outputDirectory: String,
        configuration: HumanoidBoxFlowDistillationConfiguration,
        scales: [Float] = [
            0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1,
        ]
    ) throws -> HumanoidBoxFlowResidualScaleReport {
        try configuration.validate()
        guard !scales.isEmpty,
              scales.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
              Set(scales).count == scales.count else {
            throw RLEnvironmentError.invalidConfiguration(
                "residual policy scales must be unique values in zero through one")
        }
        let artifact = try loadArtifact(flowReportPath)
        try requireLosslessLegacyReplay(artifact)
        let decoder = JSONDecoder()
        let baseMetadata = try decoder.decode(
            VectorPolicyMetadata.self,
            from: Data(contentsOf: URL(fileURLWithPath:
                "\(baseCheckpointDirectory)/metadata.json")))
        let candidateMetadata = try decoder.decode(
            VectorPolicyMetadata.self,
            from: Data(contentsOf: URL(fileURLWithPath:
                "\(candidateCheckpointDirectory)/metadata.json")))
        guard baseMetadata.task == candidateMetadata.task,
              baseMetadata.taskRevision == candidateMetadata.taskRevision,
              baseMetadata.observationDimension
                == candidateMetadata.observationDimension,
              baseMetadata.actionDimension
                == candidateMetadata.actionDimension,
              baseMetadata.architectureVersion
                == candidateMetadata.architectureVersion else {
            throw RLEnvironmentError.invalidConfiguration(
                "residual candidate is not architecture-compatible with its base")
        }
        let base = try VectorActorCritic.compatibleWeights(
            try loadArrays(url: URL(fileURLWithPath:
                "\(baseCheckpointDirectory)/policy.safetensors")),
            architectureVersion: baseMetadata.architectureVersion)
        let candidate = try VectorActorCritic.compatibleWeights(
            try loadArrays(url: URL(fileURLWithPath:
                "\(candidateCheckpointDirectory)/policy.safetensors")),
            architectureVersion: candidateMetadata.architectureVersion)
        guard Set(base.keys) == Set(candidate.keys),
              base.allSatisfy({ entry in
                  candidate[entry.key]?.shape == entry.value.shape
              }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "residual candidate tensors do not match the base policy")
        }

        let policy = VectorActorCritic(
            observationDimension: baseMetadata.observationDimension,
            actionDimension: baseMetadata.actionDimension,
            hiddenSize: baseMetadata.ppo.hiddenSize,
            hiddenDimensions: baseMetadata.ppo.hiddenDimensions,
            initialActionStd: baseMetadata.ppo.initialActionStd,
            activation: baseMetadata.ppo.resolvedActivation)
        var validations = [HumanoidBoxFlowResidualScaleValidation]()
        for scale in scales.sorted() {
            let blended = Dictionary(uniqueKeysWithValues: base.map { entry in
                (entry.key, entry.value
                    + scale * (candidate[entry.key]! - entry.value))
            })
            try policy.update(
                parameters: ModuleParameters.unflattened(blended),
                verify: [.all])
            eval(policy)
            let checkpoint = outputDirectory + "/checkpoints/"
                + String(format: "scale-%07d", Int(scale * 1_000_000))
            try saveCheckpoint(
                policy: policy, metadata: baseMetadata,
                optimizer: CheckpointableAdam(learningRate: 0),
                completedEpochs: 0, teacherRows: 0,
                directory: checkpoint)
            let deterministic = try replayLearner(
                checkpointDirectory: checkpoint,
                artifact: artifact, configuration: configuration,
                actionNoiseStandardDeviation: 0,
                noiseIncludesSource: false)
            let robust = try replayLearner(
                checkpointDirectory: checkpoint,
                artifact: artifact, configuration: configuration,
                actionNoiseStandardDeviation: configuration
                    .validationActionNoiseStandardDeviation,
                noiseIncludesSource:
                    configuration.validationNoiseIncludesSource)
            validations.append(.init(
                scale: scale, checkpoint: checkpoint,
                deterministicMaximumStableCarryDistanceMeters:
                    deterministic.maximumCarry,
                deterministicFinalSuccessFraction:
                    deterministic.successFraction,
                robustMaximumStableCarryDistanceMeters:
                    robust.maximumCarry,
                robustFinalSuccessFraction: robust.successFraction))
        }
        let selected = validations.max { lhs, rhs in
            let left = (
                lhs.robustFinalSuccessFraction,
                lhs.deterministicFinalSuccessFraction,
                min(lhs.robustMaximumStableCarryDistanceMeters,
                    lhs.deterministicMaximumStableCarryDistanceMeters),
                -lhs.scale)
            let right = (
                rhs.robustFinalSuccessFraction,
                rhs.deterministicFinalSuccessFraction,
                min(rhs.robustMaximumStableCarryDistanceMeters,
                    rhs.deterministicMaximumStableCarryDistanceMeters),
                -rhs.scale)
            return left < right
        }!
        let selectedCarry = min(
            selected.deterministicMaximumStableCarryDistanceMeters,
            selected.robustMaximumStableCarryDistanceMeters)
        let gatePassed = selected.deterministicFinalSuccessFraction >= 0.8
            && selected.robustFinalSuccessFraction >= 0.8
            && selectedCarry >= artifact.requiredCarryDistance
        let report = HumanoidBoxFlowResidualScaleReport(
            baseCheckpoint: baseCheckpointDirectory,
            candidateCheckpoint: candidateCheckpointDirectory,
            flowReport: flowReportPath,
            validationActionNoiseStandardDeviation:
                configuration.validationActionNoiseStandardDeviation,
            validations: validations,
            selectedScale: selected.scale,
            selectedCheckpoint: selected.checkpoint,
            selectedDeterministicSuccessFraction:
                selected.deterministicFinalSuccessFraction,
            selectedRobustSuccessFraction:
                selected.robustFinalSuccessFraction,
            selectedMaximumStableCarryDistanceMeters: selectedCarry,
            gatePassed: gatePassed)
        try FileManager.default.createDirectory(
            atPath: outputDirectory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: URL(fileURLWithPath:
            "\(outputDirectory)/residual-scale.json"), options: .atomic)
        return report
    }

    public static func evaluate(
        checkpointDirectory: String,
        flowReportPath: String,
        configuration: HumanoidBoxFlowDistillationConfiguration
    ) throws -> HumanoidBoxFlowReplayReport {
        try configuration.validate()
        let artifact = try loadArtifact(flowReportPath)
        try requireLosslessLegacyReplay(artifact)
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

    /// Fits a single observation-to-action-chunk residual only after the
    /// simulator-verified source frontier. Source execution never calls this
    /// model, so its complete grasp/lift/carry prefix is invariant under every
    /// optimizer update. The zero output layer also makes epoch zero exactly
    /// the commissioned policy throughout the target phase.
    public static func trainActionChunk(
        checkpointDirectory: String,
        flowReportPath: String,
        outputDirectory: String,
        configuration: HumanoidBoxFlowActionChunkConfiguration
    ) throws -> HumanoidBoxFlowActionChunkReport {
        try configuration.validate()
        let sourceCheckpointFingerprint = try VectorPPOTrainer
            .checkpointFingerprint(directory: checkpointDirectory)
        let flowReportFingerprint = try fileFingerprint(flowReportPath)
        let metadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self,
            from: Data(contentsOf: URL(fileURLWithPath:
                "\(checkpointDirectory)/metadata.json")))
        guard metadata.task == "humanoid-box-carry-v0",
              let semanticOptions = metadata.taskConfiguration else {
            throw RLEnvironmentError.invalidConfiguration(
                "action-chunk training requires box-carry checkpoint metadata")
        }
        let artifact = try loadArtifact(
            flowReportPath, allowPhysicalBalance: true)
        let runner = try VectorPolicyRunner(
            checkpointDirectory: checkpointDirectory)
        let replayOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: metadata.task, semanticOptions: semanticOptions,
            maxEpisodeSteps: metadata.maxEpisodeSteps,
            controlDecimation: metadata.controlDecimation)
        let observationDimension = metadata.observationDimension
        let actionDimension = metadata.actionDimension
        let armParameterCount = artifact.targetTrajectory.count
            - artifact.legBlendKnotCount
            - legActionCount * artifact.legResidualKnotCount
            - artifact.torsoResidualKnotCount
            - 4 * artifact.armAsymmetryKnotCount
        guard armParameterCount > 0,
              armParameterCount.isMultiple(of: 4) else {
            throw RLEnvironmentError.invalidConfiguration(
                "action-chunk flow trajectory schema is invalid")
        }
        let armKnotCount = armParameterCount / 4

        func makeTask(_ count: Int, seed: UInt64) throws
            -> HumanoidBoxCarryTask {
            let anyTask = try BuiltInRLTasks.registry.make(
                metadata.task, configuration: RLTaskConfiguration(
                    numEnvironments: count, seed: seed, autoReset: false,
                    options: replayOptions))
            guard let task = anyTask as? HumanoidBoxCarryTask,
                  metadata.compatibilityMismatches(with: task.spec).isEmpty
            else {
                throw RLEnvironmentError.invalidConfiguration(
                    "action-chunk task/checkpoint mismatch")
            }
            return task
        }

        func runnerActions(
            _ task: HumanoidBoxCarryTask,
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
                        repeating: 0, count: task.spec.numEnvironments)
                    : task.policyAuxiliaryExpertGates(observation.policy),
                auxiliaryExpertActionMask:
                    task.policyAuxiliaryExpertActionMask)
        }

        func targetTrajectory(_ phase: Int) -> (values: [Float], step: Int) {
            if let sequence = artifact.targetTrajectorySequence {
                return (
                    sequence[phase],
                    artifact.targetTrajectorySequencePhaseSteps?[phase] ?? 0)
            }
            return (artifact.targetTrajectory, phase)
        }

        func targetTeacherActions(
            _ task: HumanoidBoxCarryTask,
            _ observation: RLObservationBatch,
            phase: Int
        ) throws -> RLActionBatch {
            if let committed = artifact.targetAppliedActions?[phase] {
                var values = ContiguousArray<Float>()
                values.reserveCapacity(
                    task.spec.numEnvironments * actionDimension)
                for _ in 0..<task.spec.numEnvironments {
                    values.append(contentsOf: committed)
                }
                return try RLActionBatch(
                    numEnvironments: task.spec.numEnvironments,
                    actionDimension: actionDimension, values: values)
            }
            let target = targetTrajectory(phase)
            return try flowActions(
                task, observation, trajectory: target.values,
                step: target.step, duration: artifact.targetDuration)
        }

        func flowActions(
            _ task: HumanoidBoxCarryTask,
            _ observation: RLObservationBatch,
            trajectory: [Float], step: Int, duration: Int
        ) throws -> RLActionBatch {
            var loaded = try runnerActions(task, observation)
            let base = artifact.legBlendKnotCount > 0
                ? try runnerActions(
                    task, observation, disableAuxiliary: true)
                : nil
            let progress = Float(step + 1) / Float(duration)
            for environment in 0..<task.spec.numEnvironments {
                let actionRow = environment * actionDimension
                if let base {
                    let blend = HumanoidBoxPhysicalFlowExperiment
                        .legBlendFraction(
                            trajectory, progress: progress,
                            armParameterCount: armParameterCount,
                            knotCount: artifact.legBlendKnotCount)
                    for action in 0..<legActionCount {
                        let index = actionRow + action
                        loaded.values[index] = (1 - blend)
                            * loaded.values[index] + blend * base.values[index]
                    }
                }
                if artifact.legResidualKnotCount > 0 {
                    for action in 0..<legActionCount {
                        let index = actionRow + action
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
                                        maximumAction: artifact
                                            .maximumLegResidualAction),
                            -0.999, 0.999)
                    }
                }
                if artifact.torsoResidualKnotCount > 0 {
                    let index = actionRow + 10
                    loaded.values[index] = simd_clamp(
                        loaded.values[index]
                            + HumanoidBoxPhysicalFlowExperiment
                                .torsoResidualAction(
                                    trajectory, progress: progress,
                                    armParameterCount: armParameterCount,
                                    blendKnotCount:
                                        artifact.legBlendKnotCount,
                                    legResidualKnotCount:
                                        artifact.legResidualKnotCount,
                                    torsoResidualKnotCount:
                                        artifact.torsoResidualKnotCount,
                                    maximumAction: artifact
                                        .maximumTorsoResidualAction),
                        -0.999, 0.999)
                }
                let arm = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
                    Array(trajectory.prefix(armParameterCount)),
                    knotCount: armKnotCount, progress: progress)
                let armRow = actionRow + firstArmAction
                for action in 0..<armActionCount {
                    loaded.values[armRow + action] = simd_clamp(
                        loaded.values[armRow + action] + arm[action],
                        -0.999, 0.999)
                }
                if artifact.armAsymmetryKnotCount > 0 {
                    for action in 0..<4 {
                        let correction = HumanoidBoxPhysicalFlowExperiment
                            .armAsymmetryAction(
                                trajectory, action: action,
                                progress: progress,
                                armParameterCount: armParameterCount,
                                blendKnotCount:
                                    artifact.legBlendKnotCount,
                                legResidualKnotCount:
                                    artifact.legResidualKnotCount,
                                torsoResidualKnotCount:
                                    artifact.torsoResidualKnotCount,
                                asymmetryKnotCount:
                                    artifact.armAsymmetryKnotCount,
                                maximumAction: artifact
                                    .maximumArmAsymmetryAction)
                        loaded.values[armRow + action] = simd_clamp(
                            loaded.values[armRow + action] + correction,
                            -0.999, 0.999)
                        loaded.values[armRow + 4 + action] = simd_clamp(
                            loaded.values[armRow + 4 + action] - correction,
                            -0.999, 0.999)
                    }
                }
            }
            return loaded
        }

        func advanceToFrontier(_ task: HumanoidBoxCarryTask) throws
            -> (RLObservationBatch, RLStepBatch, [Bool]) {
            var observation = try task.reset(seed: configuration.seed)
            var result = RLStepBatch(spec: task.spec)
            if let warmupActions = artifact.sourceWarmupAppliedActions {
                for committed in warmupActions {
                    var values = ContiguousArray<Float>()
                    values.reserveCapacity(
                        task.spec.numEnvironments * actionDimension)
                    for _ in 0..<task.spec.numEnvironments {
                        values.append(contentsOf: committed)
                    }
                    let action = try RLActionBatch(
                        numEnvironments: task.spec.numEnvironments,
                        actionDimension: actionDimension, values: values)
                    try task.step(actions: action, into: &result)
                    observation = result.observations
                }
            } else {
                var streak = 0
                var warmup = 0
                while warmup < metadata.maxEpisodeSteps,
                      streak < configuration.contactDwellSteps {
                    let action = try runnerActions(task, observation)
                    try task.step(actions: action, into: &result)
                    observation = result.observations
                    warmup += 1
                    streak = zip(
                        result.metrics["state/left_hand_contact"]!,
                        result.metrics["state/right_hand_contact"]!)
                        .allSatisfy { $0.0 > 0.5 && $0.1 > 0.5 }
                        ? streak + 1 : 0
                }
                guard streak >= configuration.contactDwellSteps else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "action-chunk source checkpoint did not establish grasp")
                }
            }
            guard result.metrics["state/left_hand_contact"]!
                    .allSatisfy({ $0 > 0.5 }),
                  result.metrics["state/right_hand_contact"]!
                    .allSatisfy({ $0 > 0.5 }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "action-chunk source warm-up did not end in bilateral contact")
            }
            var failed = [Bool](
                repeating: false, count: task.spec.numEnvironments)
            var sourceActionOffset = 0
            for stage in artifact.sourceStages {
                if stage.canonicalizeReplicasBeforeExecution == true {
                    task.canonicalizeSpeculationReplicas(
                        observation: &observation, result: &result)
                }
                for step in 0..<stage.controlSteps {
                    let action: RLActionBatch
                    if let committed =
                            artifact.sourceAppliedActions?[sourceActionOffset] {
                        var values = ContiguousArray<Float>()
                        values.reserveCapacity(
                            task.spec.numEnvironments * actionDimension)
                        for _ in 0..<task.spec.numEnvironments {
                            values.append(contentsOf: committed)
                        }
                        action = try RLActionBatch(
                            numEnvironments: task.spec.numEnvironments,
                            actionDimension: actionDimension, values: values)
                    } else if let committed =
                                stage.appliedNormalizedActions?[step] {
                        var values = ContiguousArray<Float>()
                        values.reserveCapacity(
                            task.spec.numEnvironments * actionDimension)
                        for _ in 0..<task.spec.numEnvironments {
                            values.append(contentsOf: committed)
                        }
                        action = try RLActionBatch(
                            numEnvironments: task.spec.numEnvironments,
                            actionDimension: actionDimension, values: values)
                    } else if stage.policyOnly == true {
                        action = try runnerActions(task, observation)
                    } else {
                        action = try flowActions(
                            task, observation,
                            trajectory: stage.trajectorySequence?[step]
                                ?? stage.trajectory,
                            step: stage.trajectoryEvaluationStep(at: step),
                            duration: stage.trajectorySequence == nil
                                ? (stage.trajectoryDurationSteps
                                    ?? stage.controlSteps)
                                : stage.trajectorySequenceStepDenominator!)
                    }
                    sourceActionOffset += 1
                    try task.step(actions: action, into: &result)
                    observation = result.observations
                    for environment in failed.indices {
                        failed[environment] = failed[environment]
                            || result.terminated[environment]
                            || result.truncated[environment]
                    }
                }
            }
            if artifact.canonicalizeReplicasBeforeTarget {
                task.canonicalizeSpeculationReplicas(
                    observation: &observation, result: &result)
            }
            return (observation, result, failed)
        }

        // Record a tube around the feedback teacher with batched physical
        // rollouts. Replica zero is deliberately noise-free and remains the
        // exact reconstruction invariant. Other replicas receive independent
        // action perturbations after their noiseless teacher label is
        // captured, so every added row is a measured nearby simulator state
        // rather than synthetic observation jitter.
        let teacherTask = try makeTask(
            configuration.trainingReplayCount, seed: configuration.seed)
        var (teacherObservation, teacherResult, _) = try advanceToFrontier(
            teacherTask)
        var teacherObservations = [[Float]]()
        var baseActions = [[Float]]()
        var teacherActions = [[Float]]()
        let trainingRows = artifact.targetSteps
            * configuration.trainingReplayCount
        teacherObservations.reserveCapacity(trainingRows)
        baseActions.reserveCapacity(trainingRows)
        teacherActions.reserveCapacity(trainingRows)
        var trainingNoiseGenerator = ProbeRandomNumberGenerator(
            seed: configuration.seed &+ 0x7A11_5EED)
        for phase in 0..<artifact.targetSteps {
            let base = try runnerActions(teacherTask, teacherObservation)
            var teacher = try targetTeacherActions(
                teacherTask, teacherObservation, phase: phase)
            for environment in 0..<configuration.trainingReplayCount {
                let observationRow = environment * observationDimension
                let actionRow = environment * actionDimension
                teacherObservations.append(Array(
                    teacherObservation.policy[
                        observationRow..<(observationRow
                            + observationDimension)]))
                baseActions.append(Array(
                    base.values[actionRow..<(actionRow + actionDimension)]))
                teacherActions.append(Array(
                    teacher.values[
                        actionRow..<(actionRow + actionDimension)]))
                // Replica zero proves exact teacher reconstruction. It must
                // never receive augmentation noise.
                if environment > 0,
                   configuration.trainingActionNoiseStandardDeviation > 0 {
                    for action in 0..<actionDimension {
                        teacher.values[actionRow + action] = simd_clamp(
                            teacher.values[actionRow + action]
                                + configuration
                                    .trainingActionNoiseStandardDeviation
                                    * trainingNoiseGenerator.normal(),
                            -0.999, 0.999)
                    }
                }
            }
            try teacherTask.step(actions: teacher, into: &teacherResult)
            teacherObservation = teacherResult.observations
        }
        guard teacherResult.metrics["state/carry_distance_m"]![0]
                >= artifact.requiredCarryDistance,
              teacherResult.metrics["state/left_hand_contact"]![0] > 0.5,
              teacherResult.metrics["state/right_hand_contact"]![0] > 0.5,
              teacherResult.metrics["state/box_pedestal_contact"]![0] < 0.5,
              teacherResult.metrics["state/grasp_quality"]![0]
                >= artifact.requiredGraspQuality
        else {
            throw RLEnvironmentError.invalidConfiguration(
                "action-chunk teacher did not reproduce verified frontier")
        }

        let normalizer = RunningObservationNormalizer(
            snapshot: metadata.normalizer)
        let flattenedObservations = ContiguousArray(
            teacherObservations.flatMap { $0 })
        let normalizedObservations = metadata.ppo.normalizeObservations
            ? normalizer.normalize(flattenedObservations)
            : flattenedObservations
        var residualTargets = [Float]()
        residualTargets.reserveCapacity(
            trainingRows * configuration.horizon * actionDimension)
        var maximumTeacherResidual: Float = 0
        for phase in 0..<artifact.targetSteps {
            for environment in 0..<configuration.trainingReplayCount {
                for offset in 0..<configuration.horizon {
                    let targetPhase = min(
                        phase + offset, artifact.targetSteps - 1)
                    let targetRow = targetPhase
                        * configuration.trainingReplayCount + environment
                    for action in 0..<actionDimension {
                        let residual = teacherActions[targetRow][action]
                            - baseActions[targetRow][action]
                        residualTargets.append(residual)
                        maximumTeacherResidual = max(
                            maximumTeacherResidual, abs(residual))
                    }
                }
            }
        }
        guard maximumTeacherResidual <= configuration.maximumResidualAction
                + 1e-6 else {
            throw RLEnvironmentError.invalidConfiguration(
                "teacher residual exceeds configured action-chunk bound")
        }

        MLXRandom.seed(configuration.seed)
        let model = HumanoidBoxFlowActionChunkPolicy(
            observationDimension: observationDimension,
            actionDimension: actionDimension,
            horizon: configuration.horizon,
            hiddenDimension: configuration.hiddenDimension,
            maximumResidualAction: configuration.maximumResidualAction)
        let inputs = MLXArray(Array(normalizedObservations)).reshaped(
            [trainingRows, observationDimension])
        let targets = MLXArray(residualTargets).reshaped(
            [trainingRows,
             configuration.horizon * actionDimension])
        let headWeights = MLXArray((0..<configuration.horizon).flatMap {
            offset in
            [Float](
                repeating: 1 / sqrt(Float(offset + 1)),
                count: actionDimension)
        }).reshaped([1, configuration.horizon * actionDimension])
        let rowWeights = MLXArray((0..<artifact.targetSteps).flatMap {
            _ in
            (0..<configuration.trainingReplayCount).map { environment in
                environment == 0
                    ? configuration.exactReplayRowWeight : 1
            }
        }).reshaped([trainingRows, 1])
        let weightSum = headWeights.sum() * rowWeights.sum()
        func loss(
            _ policy: HumanoidBoxFlowActionChunkPolicy,
            _ arrays: [MLXArray]
        ) -> MLXArray {
            sum((policy(arrays[0]) - arrays[1]).square()
                * headWeights * rowWeights)
                / weightSum
        }
        let arrays = [inputs, targets]
        let initialLossArray = loss(model, arrays)
        eval(initialLossArray)
        let initialLoss = initialLossArray.item(Float.self)
        let lossAndGradient = valueAndGrad(model: model) {
            policy, arguments in [loss(policy, arguments)]
        }
        let optimizer = CheckpointableAdam(
            learningRate: configuration.learningRate)
        var finalLoss = initialLoss
        for epoch in 1...configuration.epochs {
            let (losses, gradients) = lossAndGradient(model, arrays)
            let clipped = clippedGradients(
                gradients, maximum: configuration.maximumGradientNorm)
            eval(losses + [clipped.squaredNorm])
            finalLoss = losses[0].item(Float.self)
            guard finalLoss.isFinite,
                  clipped.squaredNorm.item(Float.self).isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "non-finite action-chunk update at epoch \(epoch)")
            }
            try optimizer.update(model: model, gradients: clipped.gradients)
            eval(model)
            optimizer.evaluate()
        }
        let tubeLoss = finalLoss

        if configuration.exactFineTuneEpochs > 0 {
            let outputDimension = configuration.horizon * actionDimension
            var exactObservations = ContiguousArray<Float>()
            var exactTargets = [Float]()
            exactObservations.reserveCapacity(
                artifact.targetSteps * observationDimension)
            exactTargets.reserveCapacity(
                artifact.targetSteps * outputDimension)
            for phase in 0..<artifact.targetSteps {
                let row = phase * configuration.trainingReplayCount
                exactObservations.append(
                    contentsOf: teacherObservations[row])
                let targetStart = row * outputDimension
                exactTargets.append(contentsOf:
                    residualTargets[
                        targetStart..<(targetStart + outputDimension)])
            }
            let normalizedExact = metadata.ppo.normalizeObservations
                ? normalizer.normalize(exactObservations)
                : exactObservations
            let exactArrays = [
                MLXArray(Array(normalizedExact)).reshaped(
                    [artifact.targetSteps, observationDimension]),
                MLXArray(exactTargets).reshaped(
                    [artifact.targetSteps, outputDimension]),
            ]
            let exactWeightSum = headWeights.sum()
                * Float(artifact.targetSteps)
            func exactLoss(
                _ policy: HumanoidBoxFlowActionChunkPolicy,
                _ arguments: [MLXArray]
            ) -> MLXArray {
                let error = policy(arguments[0]) - arguments[1]
                let regression = sum(error.square() * headWeights)
                    / exactWeightSum
                let actionBias = mean(
                    mean(error, axis: 0).square())
                return regression
                    + configuration.exactActionBiasWeight * actionBias
            }
            let exactLossAndGradient = valueAndGrad(model: model) {
                policy, arguments in
                [exactLoss(policy, arguments)]
            }
            let exactOptimizer = CheckpointableAdam(
                learningRate: configuration.exactFineTuneLearningRate)
            for epoch in 1...configuration.exactFineTuneEpochs {
                let (losses, gradients) = exactLossAndGradient(
                    model, exactArrays)
                let clipped = clippedGradients(
                    gradients, maximum: configuration.maximumGradientNorm)
                eval(losses + [clipped.squaredNorm])
                finalLoss = losses[0].item(Float.self)
                guard finalLoss.isFinite,
                      clipped.squaredNorm.item(Float.self).isFinite else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "non-finite exact action-chunk consolidation at epoch \(epoch)")
                }
                try exactOptimizer.update(
                    model: model, gradients: clipped.gradients)
                eval(model)
                exactOptimizer.evaluate()
            }
        }

        func replay(
            learned: Bool, teacher: Bool = false,
            actionNoiseStandardDeviation: Float
        ) throws -> (
            maximumCarry: Float,
            maximumDestinationProgress: Float,
            successFraction: Float
        ) {
            precondition(!(learned && teacher))
            let task = try makeTask(
                configuration.robustReplayCount, seed: configuration.seed)
            var (observation, result, failed) = try advanceToFrontier(task)
            var maximumCarry = [Float](
                repeating: 0, count: task.spec.numEnvironments)
            let initialPlacement = Array(
                result.metrics["state/placement_distance_m"]!)
            var maximumDestinationProgress = [Float](
                repeating: -.infinity,
                count: task.spec.numEnvironments)
            var stablePath = [Bool](
                repeating: true, count: task.spec.numEnvironments)
            var generator = ProbeRandomNumberGenerator(
                seed: configuration.seed &+ 0xC817_A11C)
            for phase in 0..<artifact.targetSteps {
                var action = try runnerActions(task, observation)
                if teacher {
                    action = try targetTeacherActions(
                        task, observation, phase: phase)
                } else if learned {
                    var values = observation.policy
                    if metadata.ppo.normalizeObservations {
                        values = normalizer.normalize(values)
                    }
                    let residual = model(
                        MLXArray(Array(values)).reshaped([
                            task.spec.numEnvironments,
                            observationDimension,
                        ]))
                    eval(residual)
                    let corrections = residual.asArray(Float.self)
                    for environment in 0..<task.spec.numEnvironments {
                        let actionRow = environment * actionDimension
                        let correctionRow = environment
                            * configuration.horizon * actionDimension
                        for component in 0..<actionDimension {
                            action.values[actionRow + component] = simd_clamp(
                                action.values[actionRow + component]
                                    + corrections[
                                        correctionRow + component],
                                -0.999, 0.999)
                        }
                    }
                }
                if actionNoiseStandardDeviation > 0 {
                    for index in action.values.indices {
                        action.values[index] = simd_clamp(
                            action.values[index]
                                + actionNoiseStandardDeviation
                                    * generator.normal(),
                            -0.999, 0.999)
                    }
                }
                try task.step(actions: action, into: &result)
                observation = result.observations
                let states = task.environment.states()
                let manipulation = task.environment.manipulationStates()
                let left = result.metrics["state/left_hand_contact"]!
                let right = result.metrics["state/right_hand_contact"]!
                let support = result.metrics["state/box_pedestal_contact"]!
                let carry = result.metrics["state/carry_distance_m"]!
                let clearance = result.metrics["state/box_clearance_m"]!
                let placement = result.metrics[
                    "state/placement_distance_m"]!
                for environment in 0..<task.spec.numEnvironments {
                    let sample = HumanoidBoxStableCarrySample(
                        terminated: result.terminated[environment],
                        truncated: result.truncated[environment],
                        leftContact: left[environment],
                        rightContact: right[environment],
                        sourceSupportContact: support[environment],
                        lifted: observation.policy[
                            environment * observationDimension + 89] > 0.5,
                        rootUprightAlignment: states[environment].root
                            .rotation.act(F3(0, 0, 1)).z,
                        boxUprightAlignment: manipulation[environment]
                            .object.rotation.act(F3(0, 0, 1)).z,
                        clearanceMeters: clearance[environment],
                        carryDistanceMeters: carry[environment],
                        graspQuality: result.metrics[
                            "state/grasp_quality"]![environment])
                    let previouslyFailed = failed[environment]
                    failed[environment] = HumanoidBoxStableCarryVerifier
                        .failed(
                            previouslyFailed: previouslyFailed,
                            sample: sample)
                    let stable = HumanoidBoxStableCarryVerifier.isStable(
                        previouslyFailed: previouslyFailed, sample: sample,
                        requiredGraspQuality:
                            artifact.requiredGraspQuality)
                    stablePath[environment] = stablePath[environment]
                        && stable
                    if stable {
                        maximumCarry[environment] = max(
                            maximumCarry[environment], carry[environment])
                        maximumDestinationProgress[environment] = max(
                            maximumDestinationProgress[environment],
                            initialPlacement[environment]
                                - placement[environment])
                    }
                }
            }
            let states = task.environment.states()
            let manipulation = task.environment.manipulationStates()
            let left = result.metrics["state/left_hand_contact"]!
            let right = result.metrics["state/right_hand_contact"]!
            let support = result.metrics["state/box_pedestal_contact"]!
            let carry = result.metrics["state/carry_distance_m"]!
            let clearance = result.metrics["state/box_clearance_m"]!
            var successes = 0
            for environment in 0..<task.spec.numEnvironments {
                let sample = HumanoidBoxStableCarrySample(
                    terminated: result.terminated[environment],
                    truncated: result.truncated[environment],
                    leftContact: left[environment],
                    rightContact: right[environment],
                    sourceSupportContact: support[environment],
                    lifted: observation.policy[
                        environment * observationDimension + 89] > 0.5,
                    rootUprightAlignment: states[environment].root.rotation
                        .act(F3(0, 0, 1)).z,
                    boxUprightAlignment: manipulation[environment].object
                        .rotation.act(F3(0, 0, 1)).z,
                    clearanceMeters: clearance[environment],
                    carryDistanceMeters: carry[environment],
                    graspQuality: result.metrics[
                        "state/grasp_quality"]![environment])
                let success = artifact.physicalBalanceOnly
                    ? stablePath[environment]
                        && HumanoidBoxStableCarryVerifier.isStable(
                            previouslyFailed: failed[environment],
                            sample: sample,
                            requiredGraspQuality:
                                artifact.requiredGraspQuality)
                    : HumanoidBoxStableCarryVerifier.isFinalSuccess(
                        previouslyFailed: failed[environment], sample: sample,
                        maximumStableCarryDistanceMeters:
                            maximumCarry[environment],
                        requiredCarryDistanceMeters:
                            artifact.requiredCarryDistance,
                        maximumStableDestinationProgressMeters:
                            maximumDestinationProgress[environment],
                        requiredDestinationProgressMeters:
                            artifact.requiredDestinationProgress,
                        requiredGraspQuality:
                            artifact.requiredGraspQuality)
                if success {
                    successes += 1
                }
            }
            return (
                maximumCarry.max() ?? 0,
                maximumDestinationProgress.filter(\.isFinite).max() ?? 0,
                Float(successes) / Float(task.spec.numEnvironments))
        }

        let zero = try replay(
            learned: false, actionNoiseStandardDeviation: 0)
        let teacher = try replay(
            learned: false, teacher: true,
            actionNoiseStandardDeviation: 0)
        let teacherRobust = try replay(
            learned: false, teacher: true,
            actionNoiseStandardDeviation:
                configuration.validationActionNoiseStandardDeviation)
        let learned = try replay(
            learned: true, actionNoiseStandardDeviation: 0)
        let robust = try replay(
            learned: true,
            actionNoiseStandardDeviation:
                configuration.validationActionNoiseStandardDeviation)
        let checkpoint = outputDirectory + "/checkpoint"
        guard try VectorPPOTrainer.checkpointFingerprint(
                directory: checkpointDirectory)
                == sourceCheckpointFingerprint,
              try fileFingerprint(flowReportPath) == flowReportFingerprint else {
            throw RLEnvironmentError.invalidConfiguration(
                "action-chunk source checkpoint or flow artifact changed during training")
        }
        try FileManager.default.createDirectory(
            atPath: checkpoint, withIntermediateDirectories: true)
        let weights = Dictionary(uniqueKeysWithValues:
            model.parameters().flattened().map { ($0.0, $0.1) })
        try MLX.saveToData(arrays: weights).write(
            to: URL(fileURLWithPath: "\(checkpoint)/policy.safetensors"),
            options: .atomic)
        let chunkMetadata = HumanoidBoxFlowActionChunkMetadata(
            schemaVersion: 2,
            observationDimension: observationDimension,
            actionDimension: actionDimension,
            horizon: configuration.horizon,
            hiddenDimension: configuration.hiddenDimension,
            maximumResidualAction: configuration.maximumResidualAction,
            normalizeObservations: metadata.ppo.normalizeObservations,
            sourceCheckpoint: checkpointDirectory,
            flowReport: flowReportPath,
            sourceCheckpointFingerprint: sourceCheckpointFingerprint,
            flowReportFingerprint: flowReportFingerprint,
            trainingConfiguration: configuration,
            normalizer: metadata.normalizer)
        try JSONEncoder().encode(chunkMetadata).write(
            to: URL(fileURLWithPath: "\(checkpoint)/metadata.json"),
            options: .atomic)
        let originalResidual = model(inputs)
        eval(originalResidual)
        let originalResidualValues = originalResidual.asArray(Float.self)
        let maximumTrainingResidualError = zip(
            originalResidualValues, residualTargets).reduce(Float(0)) {
                maximum, pair in
                max(maximum, abs(pair.0 - pair.1))
            }
        let outputDimension = configuration.horizon * actionDimension
        var maximumExactFirstActionResidualError: Float = 0
        var exactActionBiases = [Float](
            repeating: 0, count: actionDimension)
        for phase in 0..<artifact.targetSteps {
            let row = phase * configuration.trainingReplayCount
            let offset = row * outputDimension
            for action in 0..<actionDimension {
                let error = originalResidualValues[offset + action]
                    - residualTargets[offset + action]
                maximumExactFirstActionResidualError = max(
                    maximumExactFirstActionResidualError, abs(error))
                exactActionBiases[action] += error
                    / Float(artifact.targetSteps)
            }
        }
        let maximumAbsoluteExactActionBias = exactActionBiases
            .map(abs).max() ?? 0
        let loadedRunner = try HumanoidBoxFlowActionChunkRunner(
            checkpointDirectory: checkpoint)
        let loadedResidualValues = try loadedRunner.residualChunks(
            for: flattenedObservations)
        let checkpointParityError = zip(
            originalResidualValues, loadedResidualValues).reduce(Float(0)) {
                maximum, pair in
                max(maximum, abs(pair.0 - pair.1))
            }
        let gatePassed = teacher.successFraction >= 0.8
            && teacherRobust.successFraction >= 0.8
            && learned.successFraction >= 0.8
            && robust.successFraction >= 0.8
            && learned.maximumCarry >= artifact.requiredCarryDistance
            && robust.maximumCarry >= artifact.requiredCarryDistance
            && learned.successFraction + 0.01 >= teacher.successFraction
            && robust.successFraction + 0.01
                >= teacherRobust.successFraction
            && learned.maximumCarry + 0.01 >= teacher.maximumCarry
            && robust.maximumCarry + 0.01 >= teacherRobust.maximumCarry
            && learned.maximumDestinationProgress + 0.01
                >= teacher.maximumDestinationProgress
            && robust.maximumDestinationProgress + 0.01
                >= teacherRobust.maximumDestinationProgress
            && checkpointParityError <= 1e-6
        let report = HumanoidBoxFlowActionChunkReport(
            sourceCheckpoint: checkpointDirectory,
            flowReport: flowReportPath,
            checkpoint: checkpoint,
            sourceCheckpointFingerprint: sourceCheckpointFingerprint,
            flowReportFingerprint: flowReportFingerprint,
            configuration: configuration,
            trainingRows: trainingRows,
            horizon: configuration.horizon,
            epochs: configuration.epochs,
            trainingReplayCount: configuration.trainingReplayCount,
            trainingActionNoiseStandardDeviation:
                configuration.trainingActionNoiseStandardDeviation,
            exactReplayRowWeight: configuration.exactReplayRowWeight,
            exactFineTuneEpochs: configuration.exactFineTuneEpochs,
            exactFineTuneLearningRate:
                configuration.exactFineTuneLearningRate,
            exactActionBiasWeight:
                configuration.exactActionBiasWeight,
            initialMeanSquaredResidualError: initialLoss,
            tubeMeanSquaredResidualError: tubeLoss,
            finalMeanSquaredResidualError: finalLoss,
            maximumTrainingResidualError:
                maximumTrainingResidualError,
            maximumExactFirstActionResidualError:
                maximumExactFirstActionResidualError,
            maximumAbsoluteExactActionBias:
                maximumAbsoluteExactActionBias,
            maximumTeacherResidualAction: maximumTeacherResidual,
            zeroResidualMaximumStableCarryDistanceMeters: zero.maximumCarry,
            zeroResidualFinalSuccessFraction: zero.successFraction,
            teacherMaximumStableCarryDistanceMeters:
                teacher.maximumCarry,
            teacherMaximumStableDestinationProgressMeters:
                teacher.maximumDestinationProgress,
            teacherFinalSuccessFraction: teacher.successFraction,
            teacherRobustMaximumStableCarryDistanceMeters:
                teacherRobust.maximumCarry,
            teacherRobustMaximumStableDestinationProgressMeters:
                teacherRobust.maximumDestinationProgress,
            teacherRobustFinalSuccessFraction:
                teacherRobust.successFraction,
            learnedMaximumStableCarryDistanceMeters: learned.maximumCarry,
            learnedMaximumStableDestinationProgressMeters:
                learned.maximumDestinationProgress,
            learnedFinalSuccessFraction: learned.successFraction,
            learnedRobustMaximumStableCarryDistanceMeters:
                robust.maximumCarry,
            learnedRobustMaximumStableDestinationProgressMeters:
                robust.maximumDestinationProgress,
            learnedRobustFinalSuccessFraction: robust.successFraction,
            validationActionNoiseStandardDeviation:
                configuration.validationActionNoiseStandardDeviation,
            sourcePrefixIsArchitecturallyBypassed: true,
            checkpointMaximumResidualParityError: checkpointParityError,
            gatePassed: gatePassed)
        try FileManager.default.createDirectory(
            atPath: outputDirectory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: URL(fileURLWithPath:
            "\(outputDirectory)/action-chunk.json"), options: .atomic)
        return report
    }

    /// Replays one immutable action-chunk checkpoint without optimizer state
    /// or retraining. The exact teacher is evaluated under the identical
    /// batch layout and target-only perturbation stream, making the
    /// amortization gap explicit rather than conflating policy failure with a
    /// fragile physical target.
    public static func evaluateActionChunk(
        checkpointDirectory: String,
        flowReportPath: String,
        actionChunkCheckpointDirectory: String,
        replayCount: Int,
        actionNoiseStandardDeviation: Float,
        seed: UInt64
    ) throws -> HumanoidBoxFlowActionChunkEvaluationReport {
        guard replayCount > 0,
              actionNoiseStandardDeviation.isFinite,
              actionNoiseStandardDeviation >= 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid action-chunk evaluation configuration")
        }
        let metadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self,
            from: Data(contentsOf: URL(fileURLWithPath:
                "\(checkpointDirectory)/metadata.json")))
        guard metadata.task == "humanoid-box-carry-v0",
              let semanticOptions = metadata.taskConfiguration else {
            throw RLEnvironmentError.invalidConfiguration(
                "action-chunk evaluation requires box-carry metadata")
        }
        let artifact = try loadArtifact(
            flowReportPath, allowPhysicalBalance: true)
        let sourceCheckpointFingerprint = try VectorPPOTrainer
            .checkpointFingerprint(directory: checkpointDirectory)
        let flowReportFingerprint = try fileFingerprint(flowReportPath)
        let immutableChunkFingerprint = try actionChunkFingerprint(
            actionChunkCheckpointDirectory)
        let immutableChunk = try HumanoidBoxFlowActionChunkRunner(
            checkpointDirectory: actionChunkCheckpointDirectory)
        guard immutableChunk.metadata.observationDimension
                == metadata.observationDimension,
              immutableChunk.metadata.actionDimension
                == metadata.actionDimension,
              immutableChunk.metadata.isBound(
                toSourceCheckpoint: checkpointDirectory,
                sourceFingerprint: sourceCheckpointFingerprint,
                flowReportPath: flowReportPath,
                flowFingerprint: flowReportFingerprint) else {
            throw RLEnvironmentError.invalidConfiguration(
                "action-chunk evaluation checkpoint/schema binding mismatch")
        }
        let replayOptions = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: metadata.task, semanticOptions: semanticOptions,
            maxEpisodeSteps: metadata.maxEpisodeSteps,
            controlDecimation: metadata.controlDecimation)

        func run(
            actionChunkCheckpoint: String?
        ) throws -> (
            maximumCarry: Float, maximumDestinationProgress: Float,
            successFraction: Float
        ) {
            let anyTask = try BuiltInRLTasks.registry.make(
                metadata.task, configuration: RLTaskConfiguration(
                    numEnvironments: replayCount, seed: seed,
                    autoReset: false, options: replayOptions))
            guard let task = anyTask as? HumanoidBoxCarryTask,
                  metadata.compatibilityMismatches(
                    with: task.spec).isEmpty else {
                throw RLEnvironmentError.invalidConfiguration(
                    "action-chunk evaluation task/checkpoint mismatch")
            }
            let controller = try ReplayController(
                checkpointDirectory: checkpointDirectory,
                flowReportPath: flowReportPath,
                actionChunkCheckpointDirectory: actionChunkCheckpoint)
            var observation = try task.reset(seed: seed)
            try controller.reset(
                for: task, environments: nil, observation: observation)
            var result = RLStepBatch(spec: task.spec)
            var failed = [Bool](repeating: false, count: replayCount)
            var stablePath = [Bool](repeating: true, count: replayCount)
            var maximumCarry = [Float](repeating: 0, count: replayCount)
            var initialPlacement: [Float]?
            var maximumDestinationProgress = [Float](
                repeating: -.infinity, count: replayCount)
            var targetSteps = 0
            var generator = ProbeRandomNumberGenerator(
                seed: seed &+ 0xE7A1_5EED)
            let stepLimit = metadata.maxEpisodeSteps
                + artifact.sourceStages.reduce(0) { $0 + $1.controlSteps }
                + artifact.targetSteps
            var totalSteps = 0
            while !controller.isComplete, totalSteps < stepLimit {
                let targetPhase = controller.targetPhaseActive
                if targetPhase, initialPlacement == nil {
                    initialPlacement = Array(
                        result.metrics["state/placement_distance_m"]!)
                }
                var action = try controller.actions(
                    for: observation, task: task)
                if targetPhase && actionNoiseStandardDeviation > 0 {
                    for index in action.values.indices {
                        action.values[index] = simd_clamp(
                            action.values[index]
                                + actionNoiseStandardDeviation
                                    * generator.normal(),
                            -0.999, 0.999)
                    }
                }
                try task.step(actions: action, into: &result)
                observation = result.observations
                totalSteps += 1

                if !targetPhase {
                    for environment in 0..<replayCount {
                        failed[environment] = failed[environment]
                            || result.terminated[environment]
                            || result.truncated[environment]
                    }
                    continue
                }
                targetSteps += 1
                let states = task.environment.states()
                let manipulation = task.environment.manipulationStates()
                let left = result.metrics["state/left_hand_contact"]!
                let right = result.metrics["state/right_hand_contact"]!
                let support = result.metrics[
                    "state/box_pedestal_contact"]!
                let carry = result.metrics["state/carry_distance_m"]!
                let clearance = result.metrics["state/box_clearance_m"]!
                let placement = result.metrics[
                    "state/placement_distance_m"]!
                for environment in 0..<replayCount {
                    let sample = HumanoidBoxStableCarrySample(
                        terminated: result.terminated[environment],
                        truncated: result.truncated[environment],
                        leftContact: left[environment],
                        rightContact: right[environment],
                        sourceSupportContact: support[environment],
                        lifted: observation.policy[
                            environment * metadata.observationDimension
                                + 89] > 0.5,
                        rootUprightAlignment: states[environment].root
                            .rotation.act(F3(0, 0, 1)).z,
                        boxUprightAlignment: manipulation[environment]
                            .object.rotation.act(F3(0, 0, 1)).z,
                        clearanceMeters: clearance[environment],
                        carryDistanceMeters: carry[environment],
                        graspQuality: result.metrics[
                            "state/grasp_quality"]![environment])
                    let previouslyFailed = failed[environment]
                    failed[environment] = HumanoidBoxStableCarryVerifier
                        .failed(
                            previouslyFailed: previouslyFailed,
                            sample: sample)
                    let stable = HumanoidBoxStableCarryVerifier.isStable(
                        previouslyFailed: previouslyFailed, sample: sample,
                        requiredGraspQuality:
                            artifact.requiredGraspQuality)
                    stablePath[environment] = stablePath[environment]
                        && stable
                    if stable {
                        maximumCarry[environment] = max(
                            maximumCarry[environment], carry[environment])
                        maximumDestinationProgress[environment] = max(
                            maximumDestinationProgress[environment],
                            initialPlacement![environment]
                                - placement[environment])
                    }
                }
            }
            guard controller.isComplete,
                  targetSteps == artifact.targetSteps else {
                throw RLEnvironmentError.invalidConfiguration(
                    "action-chunk evaluation did not reach its measured "
                        + "endpoint (phase=\(controller.phaseDescription), "
                        + "totalSteps=\(totalSteps)/\(stepLimit), "
                        + "targetSteps=\(targetSteps)/\(artifact.targetSteps), "
                        + "complete=\(controller.isComplete))")
            }

            let states = task.environment.states()
            let manipulation = task.environment.manipulationStates()
            let left = result.metrics["state/left_hand_contact"]!
            let right = result.metrics["state/right_hand_contact"]!
            let support = result.metrics["state/box_pedestal_contact"]!
            let carry = result.metrics["state/carry_distance_m"]!
            let clearance = result.metrics["state/box_clearance_m"]!
            var successes = 0
            for environment in 0..<replayCount {
                let sample = HumanoidBoxStableCarrySample(
                    terminated: result.terminated[environment],
                    truncated: result.truncated[environment],
                    leftContact: left[environment],
                    rightContact: right[environment],
                    sourceSupportContact: support[environment],
                    lifted: observation.policy[
                        environment * metadata.observationDimension + 89]
                        > 0.5,
                    rootUprightAlignment: states[environment].root.rotation
                        .act(F3(0, 0, 1)).z,
                    boxUprightAlignment: manipulation[environment].object
                        .rotation.act(F3(0, 0, 1)).z,
                    clearanceMeters: clearance[environment],
                    carryDistanceMeters: carry[environment],
                    graspQuality: result.metrics[
                        "state/grasp_quality"]![environment])
                let success = artifact.physicalBalanceOnly
                    ? stablePath[environment]
                        && HumanoidBoxStableCarryVerifier.isStable(
                            previouslyFailed: failed[environment],
                            sample: sample,
                            requiredGraspQuality:
                                artifact.requiredGraspQuality)
                    : HumanoidBoxStableCarryVerifier.isFinalSuccess(
                        previouslyFailed: failed[environment],
                        sample: sample,
                        maximumStableCarryDistanceMeters:
                            maximumCarry[environment],
                        requiredCarryDistanceMeters:
                            artifact.requiredCarryDistance,
                        maximumStableDestinationProgressMeters:
                            maximumDestinationProgress[environment],
                        requiredDestinationProgressMeters:
                            artifact.requiredDestinationProgress,
                        requiredGraspQuality:
                            artifact.requiredGraspQuality)
                if success { successes += 1 }
            }
            return (
                maximumCarry.max() ?? 0,
                maximumDestinationProgress.filter(\.isFinite).max() ?? 0,
                Float(successes) / Float(replayCount))
        }

        let teacher = try run(actionChunkCheckpoint: nil)
        let learned = try run(
            actionChunkCheckpoint: actionChunkCheckpointDirectory)
        guard try VectorPPOTrainer.checkpointFingerprint(
                directory: checkpointDirectory)
                == sourceCheckpointFingerprint,
              try fileFingerprint(flowReportPath) == flowReportFingerprint,
              try actionChunkFingerprint(actionChunkCheckpointDirectory)
                == immutableChunkFingerprint else {
            throw RLEnvironmentError.invalidConfiguration(
                "action-chunk evaluation inputs changed during replay")
        }
        return HumanoidBoxFlowActionChunkEvaluationReport(
            sourceCheckpoint: checkpointDirectory,
            flowReport: flowReportPath,
            actionChunkCheckpoint: actionChunkCheckpointDirectory,
            actionChunkFingerprint: immutableChunkFingerprint,
            sourceCheckpointFingerprint: sourceCheckpointFingerprint,
            flowReportFingerprint: flowReportFingerprint,
            evaluationSeed: seed,
            replayCount: replayCount,
            targetControlSteps: artifact.targetSteps,
            actionNoiseStandardDeviation:
                actionNoiseStandardDeviation,
            actionNoiseScope: "target-policy-only",
            teacherMaximumStableCarryDistanceMeters:
                teacher.maximumCarry,
            teacherMaximumStableDestinationProgressMeters:
                teacher.maximumDestinationProgress,
            teacherSuccessFraction: teacher.successFraction,
            learnedMaximumStableCarryDistanceMeters:
                learned.maximumCarry,
            learnedMaximumStableDestinationProgressMeters:
                learned.maximumDestinationProgress,
            learnedSuccessFraction: learned.successFraction,
            sourcePrefixIsArchitecturallyBypassed: true,
            gatePassed: teacher.successFraction >= 0.8
                && learned.successFraction >= 0.8
                && learned.successFraction + 0.01
                    >= teacher.successFraction
                && learned.maximumCarry + 0.01 >= teacher.maximumCarry
                && learned.maximumDestinationProgress + 0.01
                    >= teacher.maximumDestinationProgress)
    }

    static func loadArtifact(
        _ path: String, allowPhysicalBalance: Bool = false
    ) throws -> Artifact {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let object = try JSONSerialization.jsonObject(with: data)
                as? [String: Any] else {
            throw RLEnvironmentError.invalidConfiguration(
                "distillation requires a planner-verified dynamic flow")
        }
        let targetSequence = (object[
            "targetGeneratingTrajectorySequence"] as? [[NSNumber]])?.map {
                $0.map(\.floatValue)
            }
        let targetSequencePhaseSteps = (object[
            "targetGeneratingTrajectorySequencePhaseSteps"]
            as? [NSNumber])?.map(\.intValue)
        let encodedTargetTrace = (object["targetCommittedTrace"]
            ?? object["committedTrace"]) as? [[String: Any]]
        let encodedSourceActions = object["sourceAppliedActions"]
            as? [[NSNumber]]
        let encodedSourceWarmupActions = object[
            "sourceWarmupAppliedActions"] as? [[NSNumber]]
        let finitePrefixGate = (object[
            "targetFinitePrefixGatePassed"] as? Bool)
            ?? (object["targetPlanningGatePassed"] as? Bool)
            ?? ((object["targetGatePassed"] as? Bool) == true
            && ((object["targetCloneSuccessFraction"]
                as? NSNumber)?.floatValue ?? 0) >= 0.8
            && ((object["targetReplayMaximumNormalizedError"]
                as? NSNumber)?.floatValue ?? .infinity) < 0.02)
        let recoverySafe = (object[
            "targetPredictedRecoveryPathSafe"] as? Bool)
            ?? (object["recedingHorizonSteps"] == nil)
        let transportGate = (object[
            "targetReusableFrontierGatePassed"] as? Bool)
            ?? (finitePrefixGate && recoverySafe)
        let targetSteps = ((object["selectedTargetStep"]
            ?? object["targetExecutionSteps"]) as? NSNumber)?.intValue
        let requiredGraspQuality = ((object[
            "minimumTargetGraspQuality"]
            ?? object["requiredGraspQuality"])
            as? NSNumber)?.floatValue ?? 0
        let measuredGraspDwell = (object[
            "maximumGraspQualityThresholdDwellSteps"]
            as? NSNumber)?.intValue ?? 0
        let balanceGate = allowPhysicalBalance
            && (object["physicalBalanceGatePassed"] as? Bool) == true
            && (requiredGraspQuality <= 0
                || (targetSteps.map { measuredGraspDwell >= $0 } ?? false))
        // Replay/evaluation may inspect a rigorously reconstructed finite
        // prefix even when it is not safe continuation lineage. The explicit
        // caller opt-in keeps it out of legacy whole-policy distillation, and
        // `reusableFrontier` remains false so it cannot become a new source.
        let diagnosticFinitePrefixGate = allowPhysicalBalance
            && finitePrefixGate
        let carry = ((object["targetCarryDistanceMeters"]
            ?? object["finalCarryDistanceMeters"]
            ?? object["maximumStableCarryDistanceMeters"])
            as? NSNumber)?.floatValue
        guard
              transportGate || balanceGate || diagnosticFinitePrefixGate,
              let targetValues = object["targetGeneratingTrajectory"]
                as? [NSNumber],
              let targetSteps,
              let openLoopTargetDuration = (object[
                "targetGenerationSteps"] as? NSNumber)?.intValue,
              let legKnots = (object["legBlendKnotCount"]
                as? NSNumber)?.intValue,
              let carry,
              targetSteps > 0,
              legKnots > 0,
              let encodedStages = object["sourceStages"]
                as? [[String: Any]] else {
            throw RLEnvironmentError.invalidConfiguration(
                "distillation requires a planner-verified dynamic flow")
        }
        let targetDuration: Int
        if let targetSequence {
            guard targetSequence.count == targetSteps,
                  (targetSequencePhaseSteps.map {
                      $0.count == targetSteps
                  } ?? true),
                  targetSequence.allSatisfy({ !$0.isEmpty
                    && $0.allSatisfy(\.isFinite) }),
                  // Successful flow reports encode this explicitly. A
                  // physically certified balance frontier is emitted as a
                  // target-failure artifact, where `targetGenerationSteps`
                  // is the receding planning horizon. Both artifacts carry
                  // the same per-control trajectory sequence.
                  let horizon = ((object["recedingHorizonSteps"]
                    ?? object["targetGenerationSteps"])
                    as? NSNumber)?.intValue,
                  horizon > 0 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "feedback flow has an invalid trajectory sequence")
            }
            guard targetSequencePhaseSteps?.allSatisfy({
                $0 >= 0 && $0 < horizon
            }) ?? true else {
                throw RLEnvironmentError.invalidConfiguration(
                    "feedback flow has invalid trajectory phase steps")
            }
            targetDuration = horizon
        } else {
            guard openLoopTargetDuration >= targetSteps else {
                throw RLEnvironmentError.invalidConfiguration(
                    "open-loop flow duration is shorter than its target")
            }
            targetDuration = openLoopTargetDuration
        }
        let targetAppliedActions: [[Float]]?
        if let encodedTargetTrace {
            let actionSamples = encodedTargetTrace.compactMap { sample
                -> (step: Int, actions: [Float])? in
                guard let step = (sample["step"] as? NSNumber)?.intValue,
                      let values = sample["appliedNormalizedActions"]
                        as? [NSNumber] else { return nil }
                return (step, values.map(\.floatValue))
            }.sorted { $0.step < $1.step }
            if encodedTargetTrace.isEmpty {
                targetAppliedActions = nil
            } else {
                guard actionSamples.count == targetSteps,
                      actionSamples.enumerated().allSatisfy({
                          index, sample in
                          sample.step == index + 1
                              && sample.actions.count
                                == firstArmAction + armActionCount
                              && sample.actions.allSatisfy(\.isFinite)
                      }) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "committed target actions are incomplete or invalid")
                }
                targetAppliedActions = actionSamples.map(\.actions)
            }
        } else {
            targetAppliedActions = nil
        }
        let stages = try encodedStages.map { encoded in
            guard let steps = (encoded["controlSteps"]
                    as? NSNumber)?.intValue,
                  steps > 0 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "flow source stage is invalid")
            }
            let minimumCarry = (encoded["minimumCarryDistanceMeters"]
                as? NSNumber)?.floatValue
            let minimumDestinationProgress = (encoded[
                "minimumDestinationProgressMeters"]
                as? NSNumber)?.floatValue
            let minimumRootDestinationProgress = (encoded[
                "minimumRootDestinationProgressMeters"]
                as? NSNumber)?.floatValue
            let minimumTouchdowns = (encoded["minimumTouchdowns"]
                as? NSNumber)?.intValue
            let minimumAlternatingSteps = (encoded[
                "minimumAlternatingSteps"] as? NSNumber)?.intValue
            let minimumSwingFootLift = (encoded[
                "minimumSwingFootLiftMeters"] as? NSNumber)?.floatValue
            let minimumFootAirTime = (encoded[
                "minimumFootAirTimeSeconds"] as? NSNumber)?.floatValue
            let minimumFootUnloading = (encoded[
                "minimumFootUnloadingFraction"] as? NSNumber)?.floatValue
            let minimumTerminalFootUnloading = (encoded[
                "minimumTerminalFootUnloadingFraction"]
                as? NSNumber)?.floatValue
            let minimumClearance = (encoded["minimumClearanceMeters"]
                as? NSNumber)?.floatValue
            let minimumGraspQuality = (encoded["minimumGraspQuality"]
                as? NSNumber)?.floatValue
            let certificationDwell = (encoded["certificationDwellSteps"]
                as? NSNumber)?.intValue
            let canonicalizeReplicasBeforeExecution = encoded[
                "canonicalizeReplicasBeforeExecution"] as? Bool == true
            let appliedNormalizedActions = (encoded[
                "appliedNormalizedActions"] as? [[NSNumber]])?.map {
                    $0.map(\.floatValue)
                }
            guard appliedNormalizedActions.map({
                $0.count == steps
                    && $0.allSatisfy {
                        $0.count == firstArmAction + armActionCount
                            && $0.allSatisfy(\.isFinite)
                    }
            }) ?? true else {
                throw RLEnvironmentError.invalidConfiguration(
                    "flow source stage has invalid committed actions")
            }
            if encoded["policyOnly"] as? Bool == true {
                return HumanoidBoxPhysicalFlowStage(
                    trajectory: [], controlSteps: steps,
                    policyOnly: true,
                    appliedNormalizedActions:
                        appliedNormalizedActions,
                    canonicalizeReplicasBeforeExecution:
                        canonicalizeReplicasBeforeExecution,
                    minimumCarryDistanceMeters: minimumCarry,
                    minimumDestinationProgressMeters:
                        minimumDestinationProgress,
                    minimumRootDestinationProgressMeters:
                        minimumRootDestinationProgress,
                    minimumTouchdowns: minimumTouchdowns,
                    minimumAlternatingSteps: minimumAlternatingSteps,
                    minimumSwingFootLiftMeters: minimumSwingFootLift,
                    minimumFootAirTimeSeconds: minimumFootAirTime,
                    minimumFootUnloadingFraction: minimumFootUnloading,
                    minimumTerminalFootUnloadingFraction:
                        minimumTerminalFootUnloading,
                    minimumClearanceMeters: minimumClearance,
                    minimumGraspQuality: minimumGraspQuality,
                    certificationDwellSteps: certificationDwell)
            }
            if let encodedSequence = encoded[
                    "trajectorySequence"] as? [[NSNumber]] {
                let sequence = encodedSequence.map { $0.map(\.floatValue) }
                let phaseSteps = (encoded[
                    "trajectorySequencePhaseSteps"] as? [NSNumber])?
                    .map(\.intValue)
                guard sequence.count == steps,
                      let denominator = (encoded[
                        "trajectorySequenceStepDenominator"]
                        as? NSNumber)?.intValue,
                      denominator > 0,
                      (phaseSteps.map {
                          $0.count == steps
                            && $0.allSatisfy {
                                $0 >= 0 && $0 < denominator
                            }
                      } ?? true),
                      sequence.allSatisfy({ !$0.isEmpty
                        && $0.allSatisfy(\.isFinite) }) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "flow source feedback sequence is invalid")
                }
                return HumanoidBoxPhysicalFlowStage(
                    trajectory: [], controlSteps: steps,
                    trajectorySequence: sequence,
                    trajectorySequencePhaseSteps: phaseSteps,
                    trajectorySequenceStepDenominator: denominator,
                    forwardOnlyBaseCommand: encoded[
                        "forwardOnlyBaseCommand"] as? Bool == true,
                    holonomicBaseCommand: encoded[
                        "holonomicBaseCommand"] as? Bool == true,
                    locomotionCheckpointDirectory: encoded[
                        "locomotionCheckpointDirectory"] as? String,
                    locomotionCommandSpeed: (encoded[
                        "locomotionCommandSpeed"] as? NSNumber)?.floatValue,
                    appliedNormalizedActions:
                        appliedNormalizedActions,
                    canonicalizeReplicasBeforeExecution:
                        canonicalizeReplicasBeforeExecution,
                    minimumCarryDistanceMeters: minimumCarry,
                    minimumDestinationProgressMeters:
                        minimumDestinationProgress,
                    minimumRootDestinationProgressMeters:
                        minimumRootDestinationProgress,
                    minimumTouchdowns: minimumTouchdowns,
                    minimumAlternatingSteps: minimumAlternatingSteps,
                    minimumSwingFootLiftMeters: minimumSwingFootLift,
                    minimumFootAirTimeSeconds: minimumFootAirTime,
                    minimumFootUnloadingFraction: minimumFootUnloading,
                    minimumTerminalFootUnloadingFraction:
                        minimumTerminalFootUnloading,
                    minimumClearanceMeters: minimumClearance,
                    minimumGraspQuality: minimumGraspQuality,
                    certificationDwellSteps: certificationDwell)
            }
            guard let values = encoded["trajectory"] as? [NSNumber] else {
                throw RLEnvironmentError.invalidConfiguration(
                    "flow source trajectory is missing")
            }
            return HumanoidBoxPhysicalFlowStage(
                trajectory: values.map(\.floatValue), controlSteps: steps,
                trajectoryDurationSteps: (encoded[
                    "trajectoryDurationSteps"] as? NSNumber)?.intValue,
                forwardOnlyBaseCommand: encoded[
                    "forwardOnlyBaseCommand"] as? Bool == true,
                holonomicBaseCommand: encoded[
                    "holonomicBaseCommand"] as? Bool == true,
                locomotionCheckpointDirectory: encoded[
                    "locomotionCheckpointDirectory"] as? String,
                locomotionCommandSpeed: (encoded[
                    "locomotionCommandSpeed"] as? NSNumber)?.floatValue,
                appliedNormalizedActions:
                    appliedNormalizedActions,
                canonicalizeReplicasBeforeExecution:
                    canonicalizeReplicasBeforeExecution,
                minimumCarryDistanceMeters: minimumCarry,
                minimumDestinationProgressMeters:
                    minimumDestinationProgress,
                minimumRootDestinationProgressMeters:
                    minimumRootDestinationProgress,
                minimumTouchdowns: minimumTouchdowns,
                minimumAlternatingSteps: minimumAlternatingSteps,
                minimumSwingFootLiftMeters: minimumSwingFootLift,
                minimumFootAirTimeSeconds: minimumFootAirTime,
                minimumFootUnloadingFraction: minimumFootUnloading,
                minimumTerminalFootUnloadingFraction:
                    minimumTerminalFootUnloading,
                minimumClearanceMeters: minimumClearance,
                minimumGraspQuality: minimumGraspQuality,
                certificationDwellSteps: certificationDwell)
        }
        let sourceAppliedActions: [[Float]]?
        if let encodedSourceActions {
            let actions = encodedSourceActions.map { $0.map(\.floatValue) }
            guard actions.count == stages.reduce(0, {
                      $0 + $1.controlSteps
                  }),
                  actions.allSatisfy({
                      $0.count == firstArmAction + armActionCount
                          && $0.allSatisfy(\.isFinite)
                  }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "committed source actions are incomplete or invalid")
            }
            sourceAppliedActions = actions
        } else {
            sourceAppliedActions = nil
        }
        let sourceWarmupAppliedActions: [[Float]]?
        if let encodedSourceWarmupActions {
            let actions = encodedSourceWarmupActions.map {
                $0.map(\.floatValue)
            }
            guard !actions.isEmpty,
                  actions.allSatisfy({
                      $0.count == firstArmAction + armActionCount
                          && $0.allSatisfy(\.isFinite)
                  }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "committed source warm-up actions are invalid")
            }
            sourceWarmupAppliedActions = actions
        } else {
            sourceWarmupAppliedActions = nil
        }
        let requiredCarryDistance = balanceGate ? 0 : (object[
            "minimumTargetCarryDistanceMeters"] as? NSNumber)?
            .floatValue ?? min(carry, 0.35)
        let requiredDestinationProgress = balanceGate ? 0 : (object[
            "minimumTargetDestinationProgressMeters"] as? NSNumber)?
            .floatValue ?? 0
        let structuredStageKeys: Set<String> = [
            "trajectorySequence",
            "trajectorySequencePhaseSteps",
            "appliedNormalizedActions",
            "continueFromPreviousTrajectoryTerminal",
            "locomotionCheckpointDirectory",
            "locomotionCommandSpeed",
            "forwardOnlyBaseCommand",
            "holonomicBaseCommand",
            "graspAnchorFeedbackBlend",
            "graspAnchorFeedbackVelocityHorizonSeconds",
            "graspAnchorFeedbackMaximumActionCorrection",
            "graspAnchorFeedbackInwardPreloadMeters",
            "leftGraspAnchorBoxLocalMeters",
            "rightGraspAnchorBoxLocalMeters",
            "graspAnchorBoxHeightMeters",
        ]
        let structuredTargetKeys: Set<String> = [
            "targetGeneratingTrajectorySequence",
            "targetGeneratingTrajectorySequencePhaseSteps",
            "targetCommittedTrace",
            "committedTrace",
            "derivedStageContinuesFromSourceTerminal",
            "recedingLocomotionCheckpointDirectory",
            "recedingLocomotionCommandSpeed",
            "recedingForwardOnlyBaseCommand",
            "recedingHolonomicBaseCommand",
            "graspAnchorFeedbackBlend",
            "graspAnchorFeedbackVelocityHorizonSeconds",
            "graspAnchorFeedbackMaximumActionCorrection",
            "graspAnchorFeedbackInwardPreloadMeters",
            "leftGraspAnchorBoxLocalMeters",
            "rightGraspAnchorBoxLocalMeters",
            "graspAnchorBoxHeightMeters",
        ]
        let exactStageCount = stages.filter {
            $0.appliedNormalizedActions != nil
        }.count
        guard sourceAppliedActions != nil
                || exactStageCount == 0
                || exactStageCount == stages.count else {
            throw RLEnvironmentError.invalidConfiguration(
                "source stages mix exact actions with reconstructed actions")
        }
        guard sourceWarmupAppliedActions == nil
                || sourceAppliedActions != nil
                || exactStageCount == stages.count else {
            throw RLEnvironmentError.invalidConfiguration(
                "exact source warm-up is missing a complete exact source lineage")
        }
        let stageSemanticsRequiringExactActions = structuredStageKeys
            .subtracting([
                "trajectorySequence",
                "trajectorySequencePhaseSteps",
                "appliedNormalizedActions",
            ])
        for (encoded, stage) in zip(encodedStages, stages) {
            let requiresExact = stageSemanticsRequiringExactActions
                .contains(where: { encoded[$0] != nil })
            guard !requiresExact
                    || sourceAppliedActions != nil
                    || stage.appliedNormalizedActions != nil else {
                throw RLEnvironmentError.invalidConfiguration(
                    "structured source controller semantics require exact actions")
            }
        }
        let targetSemanticsRequiringExactActions = structuredTargetKeys
            .subtracting([
                "targetGeneratingTrajectorySequence",
                "targetGeneratingTrajectorySequencePhaseSteps",
            ])
        guard !targetSemanticsRequiringExactActions.contains(where: {
                  object[$0] != nil
              }) || targetAppliedActions != nil else {
            throw RLEnvironmentError.invalidConfiguration(
                "structured target controller semantics require exact actions")
        }
        let hasModernExecutionSemantics = encodedSourceWarmupActions != nil
            || encodedSourceActions != nil
            || structuredTargetKeys.contains(where: { object[$0] != nil })
            || encodedStages.contains { encoded in
                structuredStageKeys.contains(where: { encoded[$0] != nil })
            }
        return Artifact(
            sourceStages: stages,
            sourceWarmupAppliedActions: sourceWarmupAppliedActions,
            sourceAppliedActions: sourceAppliedActions,
            targetTrajectory: targetValues.map(\.floatValue),
            targetTrajectorySequence: targetSequence,
            targetTrajectorySequencePhaseSteps:
                targetSequencePhaseSteps,
            targetAppliedActions: targetAppliedActions,
            targetSteps: targetSteps, targetDuration: targetDuration,
            targetLocomotionCheckpointDirectory: object[
                "recedingLocomotionCheckpointDirectory"] as? String,
            targetLocomotionCommandSpeed: (object[
                "recedingLocomotionCommandSpeed"] as? NSNumber)?.floatValue,
            targetForwardOnlyBaseCommand: object[
                "recedingForwardOnlyBaseCommand"] as? Bool == true,
            targetHolonomicBaseCommand: object[
                "recedingHolonomicBaseCommand"] as? Bool == true,
            canonicalizeReplicasBeforeTarget: (object[
                "derivedStageSourceCanonicalized"] as? Bool)
                ?? (targetSequencePhaseSteps != nil),
            legBlendKnotCount: legKnots,
            legResidualKnotCount: (object[
                "legResidualKnotCount"] as? NSNumber)?.intValue ?? 0,
            maximumLegResidualAction: (object[
                "maximumLegResidualAction"] as? NSNumber)?.floatValue
                ?? 0.25,
            torsoResidualKnotCount: (object[
                "torsoResidualKnotCount"] as? NSNumber)?.intValue ?? 0,
            maximumTorsoResidualAction: (object[
                "maximumTorsoResidualAction"] as? NSNumber)?.floatValue
                ?? 0.25,
            armAsymmetryKnotCount: (object[
                "armAsymmetryKnotCount"] as? NSNumber)?.intValue ?? 0,
            maximumArmAsymmetryAction: (object[
                "maximumArmAsymmetryAction"] as? NSNumber)?.floatValue
                ?? 0.25,
            teacherCarryDistance: carry,
            requiredCarryDistance: requiredCarryDistance,
            requiredDestinationProgress: requiredDestinationProgress,
            requiredGraspQuality: requiredGraspQuality,
            physicalBalanceOnly: balanceGate
                || (requiredCarryDistance <= 0
                    && requiredDestinationProgress <= 0),
            reusableFrontier: transportGate,
            hasModernExecutionSemantics: hasModernExecutionSemantics)
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
        try requireLosslessLegacyReplay(artifact)
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
                    carryDistanceMeters: carry[environment],
                    graspQuality: result.metrics[
                        "state/grasp_quality"]![environment])
            }
        }

        let armCount = artifact.targetTrajectory.count
            - artifact.legBlendKnotCount
            - 10 * artifact.legResidualKnotCount
            - artifact.torsoResidualKnotCount
            - 4 * artifact.armAsymmetryKnotCount
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
                let trajectory = stage.trajectorySequence?[step]
                    ?? stage.trajectory
                let progress = stage.trajectorySequence == nil
                    ? Float(step + 1) / Float(
                        stage.trajectoryDurationSteps ?? stage.controlSteps)
                    : Float(stage.trajectoryEvaluationStep(at: step) + 1)
                        / Float(stage.trajectorySequenceStepDenominator!)
                let arm = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
                    Array(trajectory.prefix(armCount)),
                    knotCount: armKnots, progress: progress)
                for environment in 0..<task.spec.numEnvironments {
                    let row = environment * runner.metadata.actionDimension
                    if let baseLegAction {
                        let blend = HumanoidBoxPhysicalFlowExperiment
                            .legBlendFraction(
                                trajectory, progress: progress,
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
                                            trajectory,
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
                    if artifact.torsoResidualKnotCount > 0 {
                        let index = row + 10
                        action.values[index] = simd_clamp(
                            action.values[index]
                                + HumanoidBoxPhysicalFlowExperiment
                                    .torsoResidualAction(
                                        trajectory, progress: progress,
                                        armParameterCount: armCount,
                                        blendKnotCount:
                                            artifact.legBlendKnotCount,
                                        legResidualKnotCount:
                                            artifact.legResidualKnotCount,
                                        torsoResidualKnotCount:
                                            artifact.torsoResidualKnotCount,
                                        maximumAction: artifact
                                            .maximumTorsoResidualAction),
                            -0.999, 0.999)
                    }
                    let base = row + firstArmAction
                    for index in 0..<armActionCount {
                        action.values[base + index] = simd_clamp(
                            action.values[base + index] + arm[index],
                            -0.999, 0.999)
                    }
                    if artifact.armAsymmetryKnotCount > 0 {
                        for index in 0..<4 {
                            let correction =
                                HumanoidBoxPhysicalFlowExperiment
                                    .armAsymmetryAction(
                                        trajectory, action: index,
                                        progress: progress,
                                        armParameterCount: armCount,
                                        blendKnotCount:
                                            artifact.legBlendKnotCount,
                                        legResidualKnotCount:
                                            artifact.legResidualKnotCount,
                                        torsoResidualKnotCount:
                                            artifact.torsoResidualKnotCount,
                                        asymmetryKnotCount:
                                            artifact.armAsymmetryKnotCount,
                                        maximumAction: artifact
                                            .maximumArmAsymmetryAction)
                            action.values[base + index] = simd_clamp(
                                action.values[base + index] + correction,
                                -0.999, 0.999)
                            action.values[base + 4 + index] = simd_clamp(
                                action.values[base + 4 + index] - correction,
                                -0.999, 0.999)
                        }
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
                    previouslyFailed: previouslyFailed, sample: sample,
                    requiredGraspQuality: artifact.requiredGraspQuality
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
                requiredCarryDistanceMeters: artifact.requiredCarryDistance,
                requiredGraspQuality: artifact.requiredGraspQuality)
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
