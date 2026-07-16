import Foundation
import CryptoKit
import MLX
import MLXNN
import MLXRandom
import AVBDCore

public enum PPOActionDistribution: String, Codable, Sendable {
    /// Diagonal Gaussian followed by tanh. This remains the default for
    /// bounded normalized-action tasks and historical AVBD checkpoints.
    case squashedGaussian = "squashed-gaussian"
    /// Unbounded diagonal Gaussian used by RSL-RL/Isaac Lab locomotion.
    case gaussian

    fileprivate func environmentAction(_ unbounded: MLXArray) -> MLXArray {
        switch self {
        case .squashedGaussian: tanh(unbounded)
        case .gaussian: unbounded
        }
    }
}

public struct VectorPPOConfig: Codable, Sendable {
    public var updates: Int
    public var rolloutSteps: Int
    public var updateEpochs: Int
    public var minibatchSize: Int
    public var learningRate: Float
    public var gamma: Float
    public var gaeLambda: Float
    /// Multiplier applied to task rewards before value targets and GAE. Nil
    /// preserves the historical scale of one for checkpoint compatibility.
    /// This is optimizer conditioning only; task metrics and evaluation
    /// rewards remain in the environment's native units.
    public var rewardScale: Float?
    public var policyClip: Float
    public var valueClip: Float
    public var valueCoefficient: Float
    public var entropyCoefficient: Float
    public var maxGradientNorm: Float
    public var targetKL: Float
    public var hiddenSize: Int
    /// Exact three-layer actor/critic widths. Nil preserves AVBD's historical
    /// `[hiddenSize, max(hiddenSize/2,64), max(hiddenSize/4,64)]` topology.
    public var hiddenDimensions: [Int]?
    /// Optional for backward-compatible decoding; nil means the historical
    /// squashed Gaussian policy.
    public var actionDistribution: PPOActionDistribution?
    public var initialActionStd: Float
    /// Optional lower bound on the Gaussian exploration standard deviation.
    /// This is applied to the distribution used for both rollout sampling and
    /// PPO likelihoods, while leaving the learned parameter untouched. It is
    /// useful for sparse-contact tasks where an unconstrained policy can
    /// collapse its variance before it has observed a successful transition.
    /// `nil` preserves the ordinary learned-standard-deviation behavior.
    public var minimumActionStd: Float?
    /// Optional upper bound paired with `minimumActionStd`. A fixed floor and
    /// ceiling are equal; changing them between explicit resume stages gives
    /// a checkpointed exploration-to-consolidation curriculum without
    /// modifying task rewards or policy actions.
    public var maximumActionStd: Float?
    /// Optional policy/normalizer checkpoint used to initialize a fresh run.
    /// Unlike `resume`, this intentionally resets update counters, Adam
    /// moments, and the KL scheduler, and permits a task revision change when
    /// tensor dimensions and network architecture remain compatible.
    public var initializationCheckpoint: String?
    /// Optional checkpoint supplying its base actor as the destination's
    /// routed policy-expert branch during an explicit policy transfer. The
    /// source actor is reparameterized into the destination observation
    /// normalizer, preserving its raw-observation function exactly. This is
    /// useful for independently training approach, recovery, or precision
    /// specialists and composing them without task-specific trainer code.
    public var policyExpertInitializationCheckpoint: String?
    /// Optional checkpoint supplying only the exact-stand actor branch during
    /// an explicit policy transfer. The source actor's first layer is
    /// reparameterized from its observation normalizer into the destination
    /// normalizer, so composition preserves its raw-observation function
    /// instead of silently feeding it inputs in the wrong coordinate system.
    public var standExpertInitializationCheckpoint: String?
    /// Optional effective sample count assigned to imported observation
    /// statistics during policy transfer. The saved mean and variance remain
    /// the initial prior, but capping their count lets a task whose command
    /// distribution or observation semantics changed adapt those statistics
    /// instead of treating millions of source-task samples as immutable.
    /// `nil` preserves exact legacy transfer behavior.
    public var initializationNormalizerPriorCount: Double?
    /// Use an exact task-provided policy symmetry. Depending on
    /// `symmetryMirrorLossCoefficient`, this enables actor mirror consistency
    /// or the legacy PPO data-augmentation path. Optional for backward-
    /// compatible checkpoint decoding; `nil` means on.
    public var useTaskSymmetryAugmentation: Bool?
    /// Positive values use a differentiable actor-mean mirror-consistency
    /// loss, which is safe when fine-tuning an asymmetric checkpoint. Zero or
    /// nil preserves legacy mirrored-transition PPO augmentation for explicit
    /// from-scratch ablations. Optional for old checkpoint decoding.
    public var symmetryMirrorLossCoefficient: Float?
    /// Optional actor-mean regression on transitions from successful episodes
    /// in the current rollout. Failed and timed-out behavior has zero weight.
    public var successImitationCoefficient: Float?
    /// Actor-mean regression toward the immutable policy present immediately
    /// after `initializationCheckpoint` transfer. Tasks may provide a
    /// per-transition mask to relax retention in an adaptation regime. Nil or
    /// zero disables the frozen reference actor and preserves old checkpoints.
    public var referencePolicyCoefficient: Float?
    public var normalizeObservations: Bool
    /// Whether rollout observations update running statistics. `nil` means
    /// true for legacy checkpoints. Transfer runs can freeze a well-trained
    /// normalizer while continuing to use it, preventing representation drift
    /// during very small policy polishes.
    public var updateObservationNormalizer: Bool?
    public var checkpointInterval: Int
    public var seed: UInt64

    public init(updates: Int = 3_000, rolloutSteps: Int = 24,
                updateEpochs: Int = 5, minibatchSize: Int = 1_024,
                learningRate: Float = 3e-4, gamma: Float = 0.99,
                gaeLambda: Float = 0.95, rewardScale: Float? = nil,
                policyClip: Float = 0.2,
                valueClip: Float = 0.2, valueCoefficient: Float = 1,
                entropyCoefficient: Float = 0.01, maxGradientNorm: Float = 1,
                targetKL: Float = 0.01, hiddenSize: Int = 512,
                hiddenDimensions: [Int]? = nil,
                actionDistribution: PPOActionDistribution? = nil,
                initialActionStd: Float = 1.0,
                minimumActionStd: Float? = nil,
                maximumActionStd: Float? = nil,
                initializationCheckpoint: String? = nil,
                policyExpertInitializationCheckpoint: String? = nil,
                standExpertInitializationCheckpoint: String? = nil,
                initializationNormalizerPriorCount: Double? = nil,
                useTaskSymmetryAugmentation: Bool? = true,
                symmetryMirrorLossCoefficient: Float? = 0.01,
                successImitationCoefficient: Float? = nil,
                referencePolicyCoefficient: Float? = nil,
                normalizeObservations: Bool = true,
                updateObservationNormalizer: Bool? = true,
                checkpointInterval: Int = 50, seed: UInt64 = 1) {
        self.updates = updates
        self.rolloutSteps = rolloutSteps
        self.updateEpochs = updateEpochs
        self.minibatchSize = minibatchSize
        self.learningRate = learningRate
        self.gamma = gamma
        self.gaeLambda = gaeLambda
        self.rewardScale = rewardScale
        self.policyClip = policyClip
        self.valueClip = valueClip
        self.valueCoefficient = valueCoefficient
        self.entropyCoefficient = entropyCoefficient
        self.maxGradientNorm = maxGradientNorm
        self.targetKL = targetKL
        self.hiddenSize = hiddenSize
        self.hiddenDimensions = hiddenDimensions
        self.actionDistribution = actionDistribution
        self.initialActionStd = initialActionStd
        self.minimumActionStd = minimumActionStd
        self.maximumActionStd = maximumActionStd
        self.initializationCheckpoint = initializationCheckpoint
        self.policyExpertInitializationCheckpoint =
            policyExpertInitializationCheckpoint
        self.standExpertInitializationCheckpoint =
            standExpertInitializationCheckpoint
        self.initializationNormalizerPriorCount =
            initializationNormalizerPriorCount
        self.useTaskSymmetryAugmentation = useTaskSymmetryAugmentation
        self.symmetryMirrorLossCoefficient = symmetryMirrorLossCoefficient
        self.successImitationCoefficient = successImitationCoefficient
        self.referencePolicyCoefficient = referencePolicyCoefficient
        self.normalizeObservations = normalizeObservations
        self.updateObservationNormalizer = updateObservationNormalizer
        self.checkpointInterval = checkpointInterval
        self.seed = seed
    }

    public func validate(batchSize: Int) throws {
        guard updates > 0, rolloutSteps > 0, updateEpochs > 0 else {
            throw RLEnvironmentError.invalidConfiguration("PPO loop counts must be positive")
        }
        guard minibatchSize > 0, minibatchSize <= batchSize else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO minibatch must be in 1...rollout batch (\(batchSize))")
        }
        guard learningRate > 0, gamma > 0, gamma <= 1,
              gaeLambda >= 0, gaeLambda <= 1 else {
            throw RLEnvironmentError.invalidConfiguration("invalid PPO optimizer/discount values")
        }
        if let rewardScale {
            guard rewardScale > 0, rewardScale.isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "PPO reward scale must be finite and positive")
            }
        }
        guard hiddenSize > 0, initialActionStd > 0, checkpointInterval > 0,
              policyClip > 0, valueClip > 0, maxGradientNorm > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO network, distribution, clipping, and checkpoint values must be positive")
        }
        if let hiddenDimensions {
            guard hiddenDimensions.count == 3,
                  hiddenDimensions.allSatisfy({ $0 > 0 }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "PPO hidden dimensions must contain three positive widths")
            }
        }
        if let minimumActionStd {
            guard minimumActionStd > 0, minimumActionStd <= exp(Float(1)) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "minimum PPO action standard deviation must be in (0, e]")
            }
        }
        if let maximumActionStd {
            guard maximumActionStd > 0, maximumActionStd <= exp(Float(1)) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "maximum PPO action standard deviation must be in (0, e]")
            }
        }
        if let symmetryMirrorLossCoefficient {
            guard symmetryMirrorLossCoefficient >= 0,
                  symmetryMirrorLossCoefficient.isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "symmetry mirror-loss coefficient must be finite and non-negative")
            }
        }
        if let successImitationCoefficient {
            guard successImitationCoefficient >= 0,
                  successImitationCoefficient.isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "success-imitation coefficient must be finite and non-negative")
            }
        }
        if let referencePolicyCoefficient {
            guard referencePolicyCoefficient >= 0,
                  referencePolicyCoefficient.isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "reference-policy coefficient must be finite and non-negative")
            }
        }
        if let minimumActionStd, let maximumActionStd,
           minimumActionStd > maximumActionStd {
            throw RLEnvironmentError.invalidConfiguration(
                "minimum PPO action standard deviation exceeds its maximum")
        }
        if let initializationNormalizerPriorCount,
           (!initializationNormalizerPriorCount.isFinite
            || initializationNormalizerPriorCount < 2) {
            throw RLEnvironmentError.invalidConfiguration(
                "initialization normalizer prior count must be finite and at least 2")
        }
        if (policyExpertInitializationCheckpoint != nil
            || standExpertInitializationCheckpoint != nil),
           initializationCheckpoint == nil {
            throw RLEnvironmentError.invalidConfiguration(
                "expert composition requires an initialization checkpoint")
        }
    }

    /// A true resume restores Adam moments and the adaptive KL scheduler, so
    /// every setting that determines their update trajectory must remain
    /// identical. Extending the update count or changing snapshot frequency
    /// is safe; changing optimization, sampling, normalization, symmetry, or
    /// seed is an explicit transfer and must use `--initialize-from`.
    func resumeIncompatibilities(with checkpoint: Self) -> [String] {
        var fields = [String]()
        func require<T: Equatable>(_ name: String, _ current: T, _ saved: T) {
            if current != saved { fields.append(name) }
        }
        require("rolloutSteps", rolloutSteps, checkpoint.rolloutSteps)
        require("updateEpochs", updateEpochs, checkpoint.updateEpochs)
        require("minibatchSize", minibatchSize, checkpoint.minibatchSize)
        require("learningRate", learningRate, checkpoint.learningRate)
        require("gamma", gamma, checkpoint.gamma)
        require("gaeLambda", gaeLambda, checkpoint.gaeLambda)
        require("rewardScale", rewardScale ?? 1,
                checkpoint.rewardScale ?? 1)
        require("policyClip", policyClip, checkpoint.policyClip)
        require("valueClip", valueClip, checkpoint.valueClip)
        require("valueCoefficient", valueCoefficient,
                checkpoint.valueCoefficient)
        require("entropyCoefficient", entropyCoefficient,
                checkpoint.entropyCoefficient)
        require("maxGradientNorm", maxGradientNorm,
                checkpoint.maxGradientNorm)
        require("targetKL", targetKL, checkpoint.targetKL)
        require("hiddenSize", hiddenSize, checkpoint.hiddenSize)
        require("hiddenDimensions", resolvedHiddenDimensions,
                checkpoint.resolvedHiddenDimensions)
        require("actionDistribution", resolvedActionDistribution,
                checkpoint.resolvedActionDistribution)
        require("initialActionStd", initialActionStd,
                checkpoint.initialActionStd)
        require("minimumActionStd", minimumActionStd,
                checkpoint.minimumActionStd)
        require("maximumActionStd", maximumActionStd,
                checkpoint.maximumActionStd)
        require("useTaskSymmetryAugmentation",
                useTaskSymmetryAugmentation != false,
                checkpoint.useTaskSymmetryAugmentation != false)
        require("symmetryMirrorLossCoefficient",
                symmetryMirrorLossCoefficient ?? 0,
                checkpoint.symmetryMirrorLossCoefficient ?? 0)
        require("successImitationCoefficient",
                successImitationCoefficient ?? 0,
                checkpoint.successImitationCoefficient ?? 0)
        require("referencePolicyCoefficient",
                referencePolicyCoefficient ?? 0,
                checkpoint.referencePolicyCoefficient ?? 0)
        require("normalizeObservations", normalizeObservations,
                checkpoint.normalizeObservations)
        require("updateObservationNormalizer",
                updateObservationNormalizer != false,
                checkpoint.updateObservationNormalizer != false)
        require("initializationNormalizerPriorCount",
                initializationNormalizerPriorCount,
                checkpoint.initializationNormalizerPriorCount)
        require("seed", seed, checkpoint.seed)
        return fields
    }

    /// A resume command deliberately omits initialization checkpoints: those
    /// are transfer operations and cannot be performed again while restoring
    /// Adam. They are nevertheless immutable experiment provenance. Carry the
    /// saved parent paths into new snapshots so a stop/resume cycle cannot
    /// silently make a transferred policy look like a from-scratch run.
    func preservingInitializationProvenance(from checkpoint: Self) -> Self {
        var persisted = self
        persisted.initializationCheckpoint = checkpoint.initializationCheckpoint
        persisted.policyExpertInitializationCheckpoint =
            checkpoint.policyExpertInitializationCheckpoint
        persisted.standExpertInitializationCheckpoint =
            checkpoint.standExpertInitializationCheckpoint
        return persisted
    }

    public var resolvedHiddenDimensions: [Int] {
        hiddenDimensions ?? [
            hiddenSize, max(hiddenSize / 2, 64), max(hiddenSize / 4, 64),
        ]
    }

    public var resolvedActionDistribution: PPOActionDistribution {
        actionDistribution ?? .squashedGaussian
    }

    public var resolvedRewardScale: Float { rewardScale ?? 1 }
}

public enum GeneralizedAdvantageEstimator {
    /// Inputs use [time, environment] row-major layout. `dones` includes both
    /// true terminations and timeouts after timeout rewards have received the
    /// gamma * V(final_observation) correction.
    public static func compute(rewards: [Float], values: [Float], dones: [Bool],
                               lastValues: [Float], numEnvironments: Int,
                               horizon: Int, gamma: Float, lambda: Float)
        -> (advantages: [Float], returns: [Float]) {
        precondition(rewards.count == horizon * numEnvironments)
        precondition(values.count == rewards.count && dones.count == rewards.count)
        precondition(lastValues.count == numEnvironments)
        var advantages = [Float](repeating: 0, count: rewards.count)
        var lastGAE = [Float](repeating: 0, count: numEnvironments)
        for t in stride(from: horizon - 1, through: 0, by: -1) {
            for e in 0..<numEnvironments {
                let i = t * numEnvironments + e
                let nonTerminal: Float = dones[i] ? 0 : 1
                let nextValue = t == horizon - 1
                    ? lastValues[e]
                    : values[(t + 1) * numEnvironments + e]
                let delta = rewards[i] + gamma * nextValue * nonTerminal - values[i]
                lastGAE[e] = delta + gamma * lambda * nonTerminal * lastGAE[e]
                advantages[i] = lastGAE[e]
            }
        }
        return (advantages, zip(advantages, values).map(+))
    }
}

public struct RunningNormalizerSnapshot: Codable, Sendable {
    public var count: Double
    public var mean: [Double]
    public var variance: [Double]

    public func limitingPriorCount(to maximum: Double?)
        -> RunningNormalizerSnapshot {
        guard let maximum else { return self }
        return RunningNormalizerSnapshot(
            count: min(count, maximum), mean: mean, variance: variance)
    }

    public func applyingVarianceFloors(_ floors: [Int: Double]) throws
        -> RunningNormalizerSnapshot {
        var adjusted = self
        for (index, floor) in floors {
            guard variance.indices.contains(index), floor.isFinite, floor > 0 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "observation normalizer transfer variance floor is invalid")
            }
            adjusted.variance[index] = max(adjusted.variance[index], floor)
        }
        return adjusted
    }

    /// Apply the same destination-to-source observation mapping used for the
    /// policy's input layers. Newly introduced channels begin as ordinary
    /// zero-mean, unit-variance inputs; inherited channels preserve their
    /// exact statistics and effective sample count.
    public func remappingObservationChannels(
        sourceIndices: [Int?]
    ) throws -> RunningNormalizerSnapshot {
        guard !sourceIndices.isEmpty,
              !mean.isEmpty, mean.count == variance.count,
              sourceIndices.allSatisfy({ index in
                  guard let index else { return true }
                  return mean.indices.contains(index)
              }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "observation normalizer schema mapping is invalid")
        }
        return RunningNormalizerSnapshot(
            count: count,
            mean: sourceIndices.map { $0.map { mean[$0] } ?? 0 },
            variance: sourceIndices.map { $0.map { variance[$0] } ?? 1 })
    }
}

/// Numerically stable parallel Welford statistics for policy observations.
public struct RunningObservationNormalizer: Sendable {
    public let dimension: Int
    public private(set) var count: Double = 0
    private var mean: [Double]
    private var m2: [Double]

    public init(dimension: Int) {
        precondition(dimension > 0)
        self.dimension = dimension
        mean = [Double](repeating: 0, count: dimension)
        m2 = [Double](repeating: 0, count: dimension)
    }

    public init(snapshot: RunningNormalizerSnapshot) {
        precondition(!snapshot.mean.isEmpty)
        precondition(snapshot.mean.count == snapshot.variance.count)
        dimension = snapshot.mean.count
        count = snapshot.count
        mean = snapshot.mean
        m2 = snapshot.variance.map { $0 * max(snapshot.count - 1, 0) }
    }

    public mutating func update(_ values: ContiguousArray<Float>, rows: Int) {
        precondition(values.count == rows * dimension)
        guard rows > 0 else { return }
        for row in 0..<rows {
            count += 1
            for j in 0..<dimension {
                let x = Double(values[row * dimension + j])
                let delta = x - mean[j]
                mean[j] += delta / count
                m2[j] += delta * (x - mean[j])
            }
        }
    }

    public func normalize(_ values: ContiguousArray<Float>, clip limit: Float = 10)
        -> ContiguousArray<Float> {
        precondition(values.count.isMultiple(of: dimension))
        var output = values
        for i in output.indices {
            let j = i % dimension
            let variance = count > 1 ? m2[j] / (count - 1) : 1
            let z = Float((Double(values[i]) - mean[j]) / sqrt(max(variance, 1e-8)))
            output[i] = min(max(z, -limit), limit)
        }
        return output
    }

    public var snapshot: RunningNormalizerSnapshot {
        RunningNormalizerSnapshot(
            count: count,
            mean: mean,
            variance: m2.map { count > 1 ? $0 / (count - 1) : 1 })
    }
}

/// Separate actor and critic MLPs, matching current robotics PPO practice.
/// The trainer owns the action-distribution transform: historical normalized
/// tasks use a tanh-squashed Gaussian, while Isaac/RSL tasks pass an unbounded
/// diagonal Gaussian directly to their position-offset controller.
public final class VectorActorCritic: Module {
    @ModuleInfo var actor1: Linear
    @ModuleInfo var actor2: Linear
    @ModuleInfo var actor3: Linear
    @ModuleInfo var actorOutput: Linear
    @ModuleInfo var expertActor1: Linear
    @ModuleInfo var expertActor2: Linear
    @ModuleInfo var expertActor3: Linear
    @ModuleInfo var expertActorOutput: Linear
    @ModuleInfo var standActor1: Linear
    @ModuleInfo var standActor2: Linear
    @ModuleInfo var standActor3: Linear
    @ModuleInfo var standActorOutput: Linear
    @ModuleInfo var critic1: Linear
    @ModuleInfo var critic2: Linear
    @ModuleInfo var critic3: Linear
    @ModuleInfo var criticOutput: Linear
    @ParameterInfo var logStandardDeviation: MLXArray

    public let actionDimension: Int
    // Version 5 adds an independent task-routed stand expert. Versions 3 and
    // 4 remain inference/transfer compatible: the loader clones an existing
    // actor into every missing branch, so expansion is behavior-identical.
    public static let architectureVersion = 5
    public static let compatibleArchitectureVersions: Set<Int> = [3, 4, 5]

    public init(observationDimension: Int, actionDimension: Int,
                hiddenSize: Int = 512, hiddenDimensions: [Int]? = nil,
                initialActionStd: Float = 1.0) {
        self.actionDimension = actionDimension
        let dimensions = hiddenDimensions ?? [
            hiddenSize, max(hiddenSize / 2, 64), max(hiddenSize / 4, 64),
        ]
        precondition(dimensions.count == 3
            && dimensions.allSatisfy { $0 > 0 })
        let firstSize = dimensions[0]
        let middleSize = dimensions[1]
        let finalSize = dimensions[2]
        actor1 = Linear(observationDimension, firstSize)
        actor2 = Linear(firstSize, middleSize)
        actor3 = Linear(middleSize, finalSize)
        actorOutput = Linear(weight: MLXArray.zeros([actionDimension, finalSize]),
                        bias: MLXArray.zeros([actionDimension]))
        expertActor1 = Linear(observationDimension, firstSize)
        expertActor2 = Linear(firstSize, middleSize)
        expertActor3 = Linear(middleSize, finalSize)
        expertActorOutput = Linear(
            weight: MLXArray.zeros([actionDimension, finalSize]),
            bias: MLXArray.zeros([actionDimension]))
        standActor1 = Linear(observationDimension, firstSize)
        standActor2 = Linear(firstSize, middleSize)
        standActor3 = Linear(middleSize, finalSize)
        standActorOutput = Linear(
            weight: MLXArray.zeros([actionDimension, finalSize]),
            bias: MLXArray.zeros([actionDimension]))
        critic1 = Linear(observationDimension, firstSize)
        critic2 = Linear(firstSize, middleSize)
        critic3 = Linear(middleSize, finalSize)
        criticOutput = Linear(weight: MLXArray.zeros([1, finalSize]),
                         bias: MLXArray.zeros([1]))
        logStandardDeviation = MLXArray(
            [Float](repeating: log(initialActionStd), count: actionDimension))
    }

    public func forward(_ observations: MLXArray,
                        expertGate: MLXArray? = nil,
                        standExpertGate: MLXArray? = nil,
                        freezeBaseActor: Bool = false,
                        freezeExpertActor: Bool = false)
        -> (mean: MLXArray, value: MLXArray, logStandardDeviation: MLXArray) {
        var baseMean = actorOutput(Self.stableELU(actor3(Self.stableELU(
            actor2(Self.stableELU(actor1(observations)))))))
        if freezeBaseActor { baseMean = stopGradient(baseMean) }
        var expertMean = expertActorOutput(Self.stableELU(expertActor3(
            Self.stableELU(expertActor2(Self.stableELU(
                expertActor1(observations)))))))
        if freezeExpertActor { expertMean = stopGradient(expertMean) }
        let standMean = standActorOutput(Self.stableELU(standActor3(
            Self.stableELU(standActor2(Self.stableELU(
                standActor1(observations)))))))
        let expert = expertGate ?? MLXArray.zeros([observations.shape[0], 1])
        let stand = standExpertGate
            ?? MLXArray.zeros([observations.shape[0], 1])
        let mean = (1 - expert - stand) * baseMean
            + expert * expertMean + stand * standMean
        let value = criticOutput(
            Self.stableELU(critic3(Self.stableELU(
                critic2(Self.stableELU(critic1(observations)))))))
            .squeezed(axis: -1)
        return (mean, value, clip(logStandardDeviation, min: -5, max: 1))
    }

    /// Expand a legacy checkpoint by cloning existing actors into missing
    /// routed branches. Existing inference is unchanged and a gated transfer
    /// begins from identical policies rather than a random specialist.
    public static func compatibleWeights(
        _ source: [String: MLXArray], architectureVersion: Int?
    ) throws -> [String: MLXArray] {
        let version = architectureVersion ?? 1
        guard compatibleArchitectureVersions.contains(version) else {
            throw RLEnvironmentError.invalidConfiguration(
                "legacy policy architecture is not compatible")
        }
        guard version < Self.architectureVersion else { return source }
        var expanded = source
        func cloneActor(from sourcePrefix: String, to destinationPrefix: String)
            throws {
            let sourceStem = sourcePrefix.isEmpty ? "actor" : "\(sourcePrefix)Actor"
            let destinationStem = destinationPrefix.isEmpty
                ? "actor" : "\(destinationPrefix)Actor"
            for suffix in ["weight", "bias"] {
                for layer in ["1", "2", "3"] {
                    let input = "\(sourceStem)\(layer).\(suffix)"
                    let output = "\(destinationStem)\(layer).\(suffix)"
                    guard let value = expanded[input] else {
                        throw RLEnvironmentError.invalidConfiguration(
                            "legacy actor checkpoint is missing \(input)")
                    }
                    expanded[output] = value
                }
                let input = "\(sourceStem)Output.\(suffix)"
                let output = "\(destinationStem)Output.\(suffix)"
                guard let value = expanded[input] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "legacy actor checkpoint is missing \(input)")
                }
                expanded[output] = value
            }
        }
        if version == 3 {
            try cloneActor(from: "", to: "expert")
        }
        // A v4 expert was conventionally the exact-standing specialist, so
        // preserve it in the new stand branch. A v3 policy has just cloned
        // its base into that same expert and therefore remains identical too.
        try cloneActor(from: "expert", to: "stand")
        return expanded
    }

    /// Remap every actor and critic input matrix to a destination observation
    /// schema. A nil source index creates a zero column, making appended
    /// sensors behavior-identical until gradient updates learn to use them.
    public static func remappingObservationInputs(
        _ source: [String: MLXArray], sourceIndices: [Int?]
    ) throws -> [String: MLXArray] {
        guard !sourceIndices.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "observation input schema mapping is empty")
        }
        var remapped = source
        for name in ["actor1.weight", "expertActor1.weight",
                     "standActor1.weight", "critic1.weight"] {
            guard let input = source[name], input.shape.count == 2 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "policy checkpoint is missing compatible \(name)")
            }
            let rows = input.shape[0]
            let sourceDimension = input.shape[1]
            guard sourceIndices.allSatisfy({ index in
                guard let index else { return true }
                return (0..<sourceDimension).contains(index)
            }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "policy observation schema mapping exceeds \(name)")
            }
            let inputValues = input.asArray(Float.self)
            var outputValues = [Float](
                repeating: 0, count: rows * sourceIndices.count)
            for row in 0..<rows {
                for destination in sourceIndices.indices {
                    guard let sourceIndex = sourceIndices[destination] else {
                        continue
                    }
                    outputValues[row * sourceIndices.count + destination] =
                        inputValues[row * sourceDimension + sourceIndex]
                }
            }
            remapped[name] = MLXArray(outputValues)
                .reshaped([rows, sourceIndices.count])
        }
        return remapped
    }

    /// A version-4 checkpoint has a verified base actor plus one specialist,
    /// conventionally trained for exact standing. When a three-mode task is
    /// explicitly initialized from it, the legacy specialist becomes the new
    /// stand branch and the newly introduced low-speed branch should begin
    /// from the closer cruise gait rather than from a stationary policy.
    public static func initializingLowSpeedExpertFromBase(
        _ source: [String: MLXArray]
    ) throws -> [String: MLXArray] {
        var initialized = source
        for suffix in ["weight", "bias"] {
            for layer in ["1", "2", "3"] {
                let base = "actor\(layer).\(suffix)"
                let expert = "expertActor\(layer).\(suffix)"
                guard let value = source[base] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "expanded actor checkpoint is missing \(base)")
                }
                initialized[expert] = value
            }
            let base = "actorOutput.\(suffix)"
            let expert = "expertActorOutput.\(suffix)"
            guard let value = source[base] else {
                throw RLEnvironmentError.invalidConfiguration(
                    "expanded actor checkpoint is missing \(base)")
            }
            initialized[expert] = value
        }
        return initialized
    }

    /// Initialize the routed expert as an exact symmetry conjugate of the
    /// base actor. The observation transform is a signed permutation in raw
    /// task space. When normalization is active, its affine mean/variance
    /// correction is folded into the expert's first layer so the identity
    /// remains exact in policy-input space as well.
    public static func initializingPolicyExpertAsMirroredBase(
        _ source: [String: MLXArray],
        observationSources: [Int], observationSigns: [Float],
        actionSources: [Int], actionSigns: [Float],
        normalizer: RunningNormalizerSnapshot,
        normalizesObservations: Bool
    ) throws -> [String: MLXArray] {
        let observationDimension = observationSources.count
        let actionDimension = actionSources.count
        guard observationDimension > 0, actionDimension > 0,
              observationSigns.count == observationDimension,
              actionSigns.count == actionDimension,
              Set(observationSources).count == observationDimension,
              Set(actionSources).count == actionDimension,
              observationSources.allSatisfy({ 0..<observationDimension ~= $0 }),
              actionSources.allSatisfy({ 0..<actionDimension ~= $0 }),
              observationSigns.allSatisfy({ abs(abs($0) - 1) < 1e-6 }),
              actionSigns.allSatisfy({ abs(abs($0) - 1) < 1e-6 }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy mirror initialization requires signed permutations")
        }
        guard let baseFirstWeight = source["actor1.weight"],
              let baseFirstBias = source["actor1.bias"],
              baseFirstWeight.shape.count == 2,
              baseFirstWeight.shape[1] == observationDimension,
              baseFirstBias.shape == [baseFirstWeight.shape[0]],
              let baseOutputWeight = source["actorOutput.weight"],
              let baseOutputBias = source["actorOutput.bias"],
              baseOutputWeight.shape.count == 2,
              baseOutputWeight.shape[0] == actionDimension,
              baseOutputBias.shape == [actionDimension] else {
            throw RLEnvironmentError.invalidConfiguration(
                "base actor is incompatible with policy mirror initialization")
        }
        if normalizesObservations {
            guard normalizer.mean.count == observationDimension,
                  normalizer.variance.count == observationDimension else {
                throw RLEnvironmentError.invalidConfiguration(
                    "policy mirror normalizer has an incompatible dimension")
            }
        }

        let means = normalizesObservations
            ? normalizer.mean.map(Float.init)
            : [Float](repeating: 0, count: observationDimension)
        let standardDeviations = normalizesObservations
            ? normalizer.variance.map { sqrt(max(Float($0), 1e-8)) }
            : [Float](repeating: 1, count: observationDimension)
        guard means.allSatisfy(\.isFinite),
              standardDeviations.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy mirror normalizer contains invalid statistics")
        }

        var initialized = source
        let hidden = baseFirstWeight.shape[0]
        let baseFirstWeights = baseFirstWeight.asArray(Float.self)
        var mirroredFirstWeights = [Float](
            repeating: 0, count: hidden * observationDimension)
        var mirroredFirstBias = baseFirstBias.asArray(Float.self)
        for mirroredInput in 0..<observationDimension {
            let sourceInput = observationSources[mirroredInput]
            let sign = observationSigns[mirroredInput]
            let scale = sign * standardDeviations[sourceInput]
                / standardDeviations[mirroredInput]
            let offset = (sign * means[sourceInput] - means[mirroredInput])
                / standardDeviations[mirroredInput]
            for row in 0..<hidden {
                let weight = baseFirstWeights[
                    row * observationDimension + mirroredInput]
                mirroredFirstWeights[row * observationDimension + sourceInput]
                    += weight * scale
                mirroredFirstBias[row] += weight * offset
            }
        }
        initialized["expertActor1.weight"] = MLXArray(mirroredFirstWeights)
            .reshaped([hidden, observationDimension])
        initialized["expertActor1.bias"] = MLXArray(mirroredFirstBias)

        for layer in ["2", "3"] {
            for suffix in ["weight", "bias"] {
                let base = "actor\(layer).\(suffix)"
                guard let value = source[base] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "base actor is missing \(base)")
                }
                initialized["expertActor\(layer).\(suffix)"] = value
            }
        }
        let outputWidth = baseOutputWeight.shape[1]
        let baseOutputWeights = baseOutputWeight.asArray(Float.self)
        let baseOutputBiases = baseOutputBias.asArray(Float.self)
        var mirroredOutputWeights = [Float](
            repeating: 0, count: actionDimension * outputWidth)
        var mirroredOutputBiases = [Float](repeating: 0, count: actionDimension)
        for output in 0..<actionDimension {
            let sourceOutput = actionSources[output]
            let sign = actionSigns[output]
            for column in 0..<outputWidth {
                mirroredOutputWeights[output * outputWidth + column] = sign
                    * baseOutputWeights[sourceOutput * outputWidth + column]
            }
            mirroredOutputBiases[output] = sign * baseOutputBiases[sourceOutput]
        }
        initialized["expertActorOutput.weight"] = MLXArray(
            mirroredOutputWeights).reshaped([actionDimension, outputWidth])
        initialized["expertActorOutput.bias"] = MLXArray(mirroredOutputBiases)
        return initialized
    }

    /// Compose a separately trained stand actor into a destination policy.
    /// Both policies normally consume normalized observations. Rewriting the
    /// source first layer makes the composed branch evaluate the same affine
    /// preactivation from raw observations under the destination statistics:
    ///
    ///   W_t = W_s diag(sigma_t / sigma_s)
    ///   b_t = b_s + W_s ((mu_t - mu_s) / sigma_s)
    ///
    /// Deeper layers can then be copied byte-for-byte. This is a generic
    /// mixture-of-experts transfer primitive, not a humanoid-specific action.
    public static func initializingStandExpert(
        _ destination: [String: MLXArray],
        from source: [String: MLXArray],
        sourceNormalizer: RunningNormalizerSnapshot,
        destinationNormalizer: RunningNormalizerSnapshot
    ) throws -> [String: MLXArray] {
        let dimension = sourceNormalizer.mean.count
        guard dimension > 0,
              sourceNormalizer.variance.count == dimension,
              destinationNormalizer.mean.count == dimension,
              destinationNormalizer.variance.count == dimension else {
            throw RLEnvironmentError.invalidConfiguration(
                "stand-expert normalizers have incompatible dimensions")
        }
        let firstWeightName = "standActor1.weight"
        let firstBiasName = "standActor1.bias"
        guard let sourceWeightArray = source[firstWeightName],
              let sourceBiasArray = source[firstBiasName],
              sourceWeightArray.shape.count == 2,
              sourceWeightArray.shape[1] == dimension,
              sourceBiasArray.shape == [sourceWeightArray.shape[0]] else {
            throw RLEnvironmentError.invalidConfiguration(
                "stand-expert checkpoint has an incompatible first layer")
        }
        let rows = sourceWeightArray.shape[0]
        let sourceWeights = sourceWeightArray.asArray(Float.self)
        var destinationWeights = sourceWeights
        var destinationBias = sourceBiasArray.asArray(Float.self)
        var scales = [Float](repeating: 0, count: dimension)
        var offsets = [Float](repeating: 0, count: dimension)
        for j in 0..<dimension {
            let sourceVariance = max(sourceNormalizer.variance[j], 1e-8)
            let destinationVariance = max(
                destinationNormalizer.variance[j], 1e-8)
            guard sourceVariance.isFinite, destinationVariance.isFinite,
                  sourceNormalizer.mean[j].isFinite,
                  destinationNormalizer.mean[j].isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "stand-expert normalizer contains non-finite statistics")
            }
            let sourceScale = sqrt(sourceVariance)
            scales[j] = Float(sqrt(destinationVariance) / sourceScale)
            offsets[j] = Float((destinationNormalizer.mean[j]
                - sourceNormalizer.mean[j]) / sourceScale)
        }
        for row in 0..<rows {
            var biasOffset: Float = 0
            for j in 0..<dimension {
                let index = row * dimension + j
                biasOffset += sourceWeights[index] * offsets[j]
                destinationWeights[index] = sourceWeights[index] * scales[j]
            }
            destinationBias[row] += biasOffset
        }
        var composed = destination
        composed[firstWeightName] = MLXArray(destinationWeights)
            .reshaped(sourceWeightArray.shape)
        composed[firstBiasName] = MLXArray(destinationBias)
        for layer in ["standActor2", "standActor3", "standActorOutput"] {
            for suffix in ["weight", "bias"] {
                let name = "\(layer).\(suffix)"
                guard let value = source[name] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "stand-expert checkpoint is missing \(name)")
                }
                composed[name] = value
            }
        }
        return composed
    }

    /// Compose the source checkpoint's base actor into the destination's
    /// routed expert branch. Source and destination may use different running
    /// observation statistics (or either may consume raw observations); the
    /// first layer is rewritten so the composed expert remains the exact same
    /// function of raw task observations.
    public static func initializingPolicyExpert(
        _ destination: [String: MLXArray],
        from source: [String: MLXArray],
        sourceNormalizer: RunningNormalizerSnapshot,
        destinationNormalizer: RunningNormalizerSnapshot,
        sourceNormalizesObservations: Bool = true,
        destinationNormalizesObservations: Bool = true
    ) throws -> [String: MLXArray] {
        let dimension = sourceNormalizer.mean.count
        guard dimension > 0,
              sourceNormalizer.variance.count == dimension,
              destinationNormalizer.mean.count == dimension,
              destinationNormalizer.variance.count == dimension else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy-expert normalizers have incompatible dimensions")
        }
        let sourceWeightName = "actor1.weight"
        let sourceBiasName = "actor1.bias"
        guard let sourceWeightArray = source[sourceWeightName],
              let sourceBiasArray = source[sourceBiasName],
              sourceWeightArray.shape.count == 2,
              sourceWeightArray.shape[1] == dimension,
              sourceBiasArray.shape == [sourceWeightArray.shape[0]] else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy-expert checkpoint has an incompatible first layer")
        }
        let rows = sourceWeightArray.shape[0]
        let sourceWeights = sourceWeightArray.asArray(Float.self)
        var destinationWeights = sourceWeights
        var destinationBias = sourceBiasArray.asArray(Float.self)
        var scales = [Float](repeating: 0, count: dimension)
        var offsets = [Float](repeating: 0, count: dimension)
        for j in 0..<dimension {
            let sourceVariance = max(sourceNormalizer.variance[j], 1e-8)
            let destinationVariance = max(
                destinationNormalizer.variance[j], 1e-8)
            guard sourceVariance.isFinite, destinationVariance.isFinite,
                  sourceNormalizer.mean[j].isFinite,
                  destinationNormalizer.mean[j].isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "policy-expert normalizer contains non-finite statistics")
            }
            let sourceScale = sourceNormalizesObservations
                ? sqrt(sourceVariance) : 1
            let destinationScale = destinationNormalizesObservations
                ? sqrt(destinationVariance) : 1
            let sourceMean = sourceNormalizesObservations
                ? sourceNormalizer.mean[j] : 0
            let destinationMean = destinationNormalizesObservations
                ? destinationNormalizer.mean[j] : 0
            scales[j] = Float(destinationScale / sourceScale)
            offsets[j] = Float((destinationMean - sourceMean) / sourceScale)
        }
        for row in 0..<rows {
            var biasOffset: Float = 0
            for j in 0..<dimension {
                let index = row * dimension + j
                biasOffset += sourceWeights[index] * offsets[j]
                destinationWeights[index] = sourceWeights[index] * scales[j]
            }
            destinationBias[row] += biasOffset
        }
        var composed = destination
        composed["expertActor1.weight"] = MLXArray(destinationWeights)
            .reshaped(sourceWeightArray.shape)
        composed["expertActor1.bias"] = MLXArray(destinationBias)
        for layer in ["2", "3"] {
            for suffix in ["weight", "bias"] {
                let sourceName = "actor\(layer).\(suffix)"
                guard let value = source[sourceName] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "policy-expert checkpoint is missing \(sourceName)")
                }
                composed["expertActor\(layer).\(suffix)"] = value
            }
        }
        for suffix in ["weight", "bias"] {
            let sourceName = "actorOutput.\(suffix)"
            guard let value = source[sourceName] else {
                throw RLEnvironmentError.invalidConfiguration(
                    "policy-expert checkpoint is missing \(sourceName)")
            }
            composed["expertActorOutput.\(suffix)"] = value
        }
        return composed
    }

    /// Version 4 used one routed expert for two different purposes. Standing
    /// curricula trained it only at zero command, while point-goal curricula
    /// froze the cruise actor and trained the same expert throughout braking
    /// and arrival. Only the former should be replaced by the cruise actor
    /// when it becomes the v5 low-speed branch.
    public static func legacyExpertIsStandOnly(
        taskConfiguration: [String: Float]
    ) -> Bool {
        let standingOnly =
            (taskConfiguration["standingCommandProbability"] ?? 0) >= 0.999
        let trainedBaseInstead =
            (taskConfiguration["trainBasePolicyExpert"] ?? 0) > 0
            && (taskConfiguration["freezeBasePolicyExpert"] ?? 0) == 0
        return standingOnly || trainedBaseInstead
    }

    /// MLXNN's compiled ELU evaluates `exp(x)` in the unselected branch of
    /// `which(x > 0, x, exp(x) - 1)`.  A finite, large positive preactivation
    /// can therefore create an infinity that becomes `0 * infinity` during
    /// backward, even though the selected forward value is just `x`.  Clamp
    /// the exponential branch at zero.  This is exactly ELU for every input
    /// (and avoids overflow in both forward and reverse mode).
    private static func stableELU(_ x: MLXArray) -> MLXArray {
        which(x .> 0, x, exp(minimum(x, 0)) - 1)
    }
}

public struct PPOUpdateMetrics: Codable, Sendable {
    public var update: Int
    public var environmentSteps: Int
    public var policyLoss: Float
    public var valueLoss: Float
    public var learningRate: Float
    public var entropy: Float
    public var symmetryLoss: Float
    public var approximateKL: Float
    public var explainedVariance: Float
    public var stepsPerSecond: Float
    public var completedEpisodes: Int
    public var successRate: Float
    public var meanEpisodeReturn: Float
    public var meanEpisodeLength: Float
    public var meanEpisodeForwardDistance: Float
    public var meanActionStandardDeviation: Float
    /// Per-transition reward/penalty terms and per-completed-episode task
    /// metrics. Keeping these in the training log makes reward exploits and
    /// gait collapse auditable without relying on a rendered replay.
    public var taskMetrics: [String: Float]
}

public struct PPOEvaluationMetrics: Codable, Sendable {
    /// Version 1 records immutable checkpoint identity and training lineage;
    /// version 2 also records replica count because a batched simulator must
    /// not silently compare validation and test under different layouts.
    /// Optional only so historical reports remain readable; publication and
    /// checkpoint-selection gates reject reports without this provenance.
    public var provenanceVersion: Int? = nil
    public var task: String
    public var taskRevision: Int? = nil
    /// Exact serialized task contracts on the two sides of evaluation.
    /// These are intentionally part of every new report: collision-profile
    /// and domain-randomization validation must never masquerade as an
    /// in-distribution checkpoint replay.
    public var checkpointTaskConfiguration: [String: Float]? = nil
    public var evaluationTaskConfiguration: [String: Float]? = nil
    public var taskConfigurationTransferred: Bool? = nil
    public var checkpointDirectory: String
    public var checkpointFingerprint: String? = nil
    public var initializationCheckpoint: String? = nil
    public var trainingSeed: UInt64
    public var evaluationSeed: UInt64
    public var evaluationEnvironments: Int? = nil
    public var trainingUpdates: Int
    public var trainingEnvironmentSteps: Int
    public var episodes: Int
    public var successes: Int
    public var successRate: Float
    public var meanReturn: Float
    public var meanEpisodeLength: Float
    public var taskMetrics: [String: Float]
    public var acceptance: PPOEvaluationAcceptance?
}

public struct PPOEvaluationAcceptance: Codable, Sendable {
    public var passed: Bool
    public var failures: [String]
}

public struct PPOScalarSummary: Codable, Sendable, Equatable {
    public var median: Float
    public var firstQuartile: Float
    public var thirdQuartile: Float
    public var minimum: Float
    public var maximum: Float
}

public struct PPOEvaluationAggregate: Codable, Sendable {
    public var task: String
    public var evaluationTaskConfiguration: [String: Float]?
    public var taskConfigurationTransferred: Bool?
    public var trainingSeeds: [UInt64]
    public var evaluationSeeds: [UInt64]
    public var evaluationEnvironments: Int?
    public var totalEpisodes: Int
    public var requiredRuns: Int
    public var requiredEpisodesPerRun: Int
    public var acceptedRuns: Int
    public var allRunsPassed: Bool
    public var hasRequiredRunCount: Bool
    public var allRunsHaveRequiredEpisodes: Bool
    public var provenanceComplete: Bool
    public var allRunsFromScratch: Bool
    public var initializationCheckpoints: [String]
    public var checkpointFingerprints: [String]
    public var publishable: Bool
    public var successRate: PPOScalarSummary
    public var trainingUpdates: PPOScalarSummary
    public var trainingEnvironmentSteps: PPOScalarSummary
    public var meanReturn: PPOScalarSummary
    public var meanEpisodeLength: PPOScalarSummary
    public var taskMetrics: [String: PPOScalarSummary]

    public static func make(_ evaluations: [PPOEvaluationMetrics],
                            requiredRuns: Int = 5,
                            requiredEpisodesPerRun: Int = 512) throws -> Self {
        guard requiredRuns > 0, requiredEpisodesPerRun > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "required run and episode counts must be positive")
        }
        guard let first = evaluations.first else {
            throw RLEnvironmentError.invalidConfiguration(
                "at least one evaluation report is required")
        }
        guard evaluations.allSatisfy({ $0.task == first.task }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation reports must all belong to the same task")
        }
        guard evaluations.allSatisfy({
            $0.evaluationEnvironments == first.evaluationEnvironments
                && $0.evaluationTaskConfiguration
                    == first.evaluationTaskConfiguration
                && $0.taskConfigurationTransferred
                    == first.taskConfigurationTransferred
        }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation reports must use the same replica count and task contract")
        }
        guard Set(evaluations.map(\.trainingSeed)).count == evaluations.count else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation reports must come from distinct training seeds")
        }
        let metricNames = Set(evaluations.flatMap { $0.taskMetrics.keys })
        var summarizedMetrics = [String: PPOScalarSummary]()
        for name in metricNames {
            guard evaluations.allSatisfy({ $0.taskMetrics[name] != nil }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task metric '\(name)' is missing from one or more reports")
            }
            summarizedMetrics[name] = summarize(evaluations.map {
                $0.taskMetrics[name]!
            })
        }
        let accepted = evaluations.filter { $0.acceptance?.passed == true }.count
        let hasRequiredRuns = evaluations.count >= requiredRuns
        let hasRequiredEpisodes = evaluations.allSatisfy {
            $0.episodes >= requiredEpisodesPerRun
        }
        let allPassed = accepted == evaluations.count
        let provenanceComplete = evaluations.allSatisfy {
            ($0.provenanceVersion ?? 0) >= 2
                && $0.taskRevision != nil
                && ($0.evaluationEnvironments ?? 0) > 0
                && !($0.checkpointFingerprint ?? "").isEmpty
                && (($0.provenanceVersion ?? 0) < 3
                    || ($0.checkpointTaskConfiguration != nil
                        && $0.evaluationTaskConfiguration != nil
                        && $0.taskConfigurationTransferred != nil))
        }
        let allRunsFromScratch = evaluations.allSatisfy {
            $0.initializationCheckpoint == nil
        }
        return Self(
            task: first.task,
            evaluationTaskConfiguration: first.evaluationTaskConfiguration,
            taskConfigurationTransferred: first.taskConfigurationTransferred,
            trainingSeeds: evaluations.map(\.trainingSeed).sorted(),
            evaluationSeeds: evaluations.map(\.evaluationSeed).sorted(),
            evaluationEnvironments: first.evaluationEnvironments,
            totalEpisodes: evaluations.reduce(0) { $0 + $1.episodes },
            requiredRuns: requiredRuns,
            requiredEpisodesPerRun: requiredEpisodesPerRun,
            acceptedRuns: accepted,
            allRunsPassed: allPassed,
            hasRequiredRunCount: hasRequiredRuns,
            allRunsHaveRequiredEpisodes: hasRequiredEpisodes,
            provenanceComplete: provenanceComplete,
            allRunsFromScratch: allRunsFromScratch,
            initializationCheckpoints: evaluations.compactMap(
                \.initializationCheckpoint).sorted(),
            checkpointFingerprints: evaluations.compactMap(
                \.checkpointFingerprint).sorted(),
            publishable: hasRequiredRuns && hasRequiredEpisodes && allPassed
                && provenanceComplete && allRunsFromScratch,
            successRate: summarize(evaluations.map(\.successRate)),
            trainingUpdates: summarize(evaluations.map { Float($0.trainingUpdates) }),
            trainingEnvironmentSteps: summarize(evaluations.map {
                Float($0.trainingEnvironmentSteps)
            }),
            meanReturn: summarize(evaluations.map(\.meanReturn)),
            meanEpisodeLength: summarize(evaluations.map(\.meanEpisodeLength)),
            taskMetrics: summarizedMetrics)
    }

    static func summarize(_ values: [Float]) -> PPOScalarSummary {
        let sorted = values.sorted()
        return PPOScalarSummary(
            median: quantile(sorted, probability: 0.5),
            firstQuartile: quantile(sorted, probability: 0.25),
            thirdQuartile: quantile(sorted, probability: 0.75),
            minimum: sorted.first!, maximum: sorted.last!)
    }

    /// Linear interpolation at index p*(n-1), a deterministic definition that
    /// remains meaningful for the required five-run suite.
    private static func quantile(_ sorted: [Float], probability: Float) -> Float {
        let index = probability * Float(sorted.count - 1)
        let lower = Int(floor(index))
        let upper = Int(ceil(index))
        let fraction = index - Float(lower)
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction
    }
}

/// Robustness report for one immutable checkpoint evaluated over independent
/// reset seeds. This is deliberately separate from `PPOEvaluationAggregate`,
/// whose publication gate requires independently trained policies.
public struct PPOCheckpointEvaluationAggregate: Codable, Sendable {
    public var scope: String
    public var task: String
    public var evaluationTaskConfiguration: [String: Float]?
    public var taskConfigurationTransferred: Bool?
    public var checkpointDirectory: String
    public var trainingSeed: UInt64
    public var checkpointFingerprint: String?
    public var initializationCheckpoint: String?
    public var evaluationSeeds: [UInt64]
    public var evaluationEnvironments: Int?
    public var runs: Int
    public var requiredRuns: Int
    public var requiredEpisodesPerRun: Int
    public var totalEpisodes: Int
    public var totalSuccesses: Int
    public var pooledSuccessRate: Float
    public var acceptedRuns: Int
    public var allRunsPassed: Bool
    public var hasRequiredRunCount: Bool
    public var allRunsHaveRequiredEpisodes: Bool
    public var provenanceComplete: Bool
    public var robustAcrossEvaluationSeeds: Bool
    public var successRate: PPOScalarSummary
    public var meanReturn: PPOScalarSummary
    public var meanEpisodeLength: PPOScalarSummary
    public var taskMetrics: [String: PPOScalarSummary]

    public static func make(_ evaluations: [PPOEvaluationMetrics],
                            requiredRuns: Int = 4,
                            requiredEpisodesPerRun: Int = 512) throws -> Self {
        guard requiredRuns > 0, requiredEpisodesPerRun > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "required run and episode counts must be positive")
        }
        guard let first = evaluations.first else {
            throw RLEnvironmentError.invalidConfiguration(
                "at least one evaluation report is required")
        }
        guard evaluations.allSatisfy({
            $0.task == first.task
                && $0.checkpointDirectory == first.checkpointDirectory
                && $0.checkpointFingerprint == first.checkpointFingerprint
                && $0.initializationCheckpoint == first.initializationCheckpoint
                && $0.trainingSeed == first.trainingSeed
                && $0.trainingUpdates == first.trainingUpdates
                && $0.trainingEnvironmentSteps == first.trainingEnvironmentSteps
                && $0.evaluationEnvironments == first.evaluationEnvironments
                && $0.checkpointTaskConfiguration
                    == first.checkpointTaskConfiguration
                && $0.evaluationTaskConfiguration
                    == first.evaluationTaskConfiguration
                && $0.taskConfigurationTransferred
                    == first.taskConfigurationTransferred
        }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint robustness reports must describe one immutable checkpoint")
        }
        guard Set(evaluations.map(\.evaluationSeed)).count == evaluations.count else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint robustness reports require distinct evaluation seeds")
        }
        let metricNames = Set(evaluations.flatMap { $0.taskMetrics.keys })
        var summarizedMetrics = [String: PPOScalarSummary]()
        for name in metricNames {
            guard evaluations.allSatisfy({ $0.taskMetrics[name] != nil }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task metric '\(name)' is missing from one or more reports")
            }
            summarizedMetrics[name] = PPOEvaluationAggregate.summarize(
                evaluations.map { $0.taskMetrics[name]! })
        }
        let totalEpisodes = evaluations.reduce(0) { $0 + $1.episodes }
        let totalSuccesses = evaluations.reduce(0) { $0 + $1.successes }
        let accepted = evaluations.filter { $0.acceptance?.passed == true }.count
        let hasRequiredRuns = evaluations.count >= requiredRuns
        let hasRequiredEpisodes = evaluations.allSatisfy {
            $0.episodes >= requiredEpisodesPerRun
        }
        let allPassed = accepted == evaluations.count
        let provenanceComplete = evaluations.allSatisfy {
            ($0.provenanceVersion ?? 0) >= 2
                && $0.taskRevision != nil
                && ($0.evaluationEnvironments ?? 0) > 0
                && !($0.checkpointFingerprint ?? "").isEmpty
                && (($0.provenanceVersion ?? 0) < 3
                    || ($0.checkpointTaskConfiguration != nil
                        && $0.evaluationTaskConfiguration != nil
                        && $0.taskConfigurationTransferred != nil))
        }
        return Self(
            scope: "single_checkpoint_across_evaluation_seeds",
            task: first.task,
            evaluationTaskConfiguration: first.evaluationTaskConfiguration,
            taskConfigurationTransferred: first.taskConfigurationTransferred,
            checkpointDirectory: first.checkpointDirectory,
            trainingSeed: first.trainingSeed,
            checkpointFingerprint: first.checkpointFingerprint,
            initializationCheckpoint: first.initializationCheckpoint,
            evaluationSeeds: evaluations.map(\.evaluationSeed).sorted(),
            evaluationEnvironments: first.evaluationEnvironments,
            runs: evaluations.count,
            requiredRuns: requiredRuns,
            requiredEpisodesPerRun: requiredEpisodesPerRun,
            totalEpisodes: totalEpisodes,
            totalSuccesses: totalSuccesses,
            pooledSuccessRate: Float(totalSuccesses) / Float(totalEpisodes),
            acceptedRuns: accepted,
            allRunsPassed: allPassed,
            hasRequiredRunCount: hasRequiredRuns,
            allRunsHaveRequiredEpisodes: hasRequiredEpisodes,
            provenanceComplete: provenanceComplete,
            robustAcrossEvaluationSeeds: hasRequiredRuns
                && hasRequiredEpisodes && allPassed && provenanceComplete,
            successRate: PPOEvaluationAggregate.summarize(
                evaluations.map(\.successRate)),
            meanReturn: PPOEvaluationAggregate.summarize(
                evaluations.map(\.meanReturn)),
            meanEpisodeLength: PPOEvaluationAggregate.summarize(
                evaluations.map(\.meanEpisodeLength)),
            taskMetrics: summarizedMetrics)
    }
}

public struct PPOCheckpointValidationCandidate: Codable, Sendable, Equatable {
    public var checkpointDirectory: String
    public var checkpointFingerprint: String
    public var trainingUpdates: Int
    /// Validation resets represented by this candidate. Optional only for
    /// decoding schema-v1 selection manifests written before multi-seed
    /// checkpoint selection was introduced.
    public var evaluationSeeds: [UInt64]? = nil
    public var validationRuns: Int? = nil
    public var totalEpisodes: Int? = nil
    public var successes: Int
    public var successRate: Float
    public var minimumSeedSuccessRate: Float? = nil
    public var meanReturn: Float
    public var acceptancePassed: Bool
}

/// Auditable checkpoint selection on validation-only reset seeds. Every
/// candidate must be evaluated on the exact same seed/episode matrix. Ranking
/// begins with the worst seed rather than mean return, which prevents a lucky
/// reset stream from selecting an otherwise fragile locomotion policy. Test
/// reports are checked against the complete validation seed set, and the
/// immutable fingerprint prevents a mutable path from changing after choice.
public struct PPOCheckpointSelection: Codable, Sendable {
    public var schemaVersion: Int
    public var task: String
    public var taskRevision: Int
    public var trainingSeed: UInt64
    public var initializationCheckpoint: String?
    /// First validation seed, retained for schema-v1 compatibility.
    public var validationSeed: UInt64
    public var validationSeeds: [UInt64]? = nil
    public var validationEnvironments: Int? = nil
    public var evaluationTaskConfiguration: [String: Float]? = nil
    public var taskConfigurationTransferred: Bool? = nil
    public var validationEpisodesPerCandidate: Int
    public var validationEpisodesPerSeed: Int? = nil
    public var selectionRule: String
    public var candidates: [PPOCheckpointValidationCandidate]
    public var selectedCheckpointDirectory: String
    public var selectedCheckpointFingerprint: String
    public var selectedTrainingUpdates: Int

    public static func make(_ evaluations: [PPOEvaluationMetrics]) throws -> Self {
        guard let first = evaluations.first else {
            throw RLEnvironmentError.invalidConfiguration(
                "at least one checkpoint validation report is required")
        }
        guard (first.provenanceVersion ?? 0) >= 2,
              let taskRevision = first.taskRevision,
              let evaluationEnvironments = first.evaluationEnvironments,
              evaluationEnvironments > 0,
              let firstFingerprint = first.checkpointFingerprint,
              !firstFingerprint.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint selection requires provenance-version-1 reports")
        }
        guard evaluations.allSatisfy({
            ($0.provenanceVersion ?? 0) >= 2
                && $0.task == first.task
                && $0.taskRevision == taskRevision
                && $0.trainingSeed == first.trainingSeed
                && $0.initializationCheckpoint == first.initializationCheckpoint
                && $0.evaluationEnvironments == evaluationEnvironments
                && $0.evaluationTaskConfiguration
                    == first.evaluationTaskConfiguration
                && $0.taskConfigurationTransferred
                    == first.taskConfigurationTransferred
                && !($0.checkpointFingerprint ?? "").isEmpty
        }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint validation reports must share task and training lineage")
        }

        let grouped = Dictionary(grouping: evaluations) {
            $0.checkpointFingerprint!
        }
        guard grouped.count >= 1,
              Set(grouped.values.compactMap { $0.first?.checkpointDirectory }).count
                == grouped.count else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint validation candidates must have distinct paths and fingerprints")
        }

        let firstGroup = grouped[firstFingerprint]!
        let expectedSeeds = firstGroup.map(\.evaluationSeed).sorted()
        guard Set(expectedSeeds).count == expectedSeeds.count else {
            throw RLEnvironmentError.invalidConfiguration(
                "a checkpoint has duplicate validation evaluation seeds")
        }
        let expectedEpisodes = Dictionary(uniqueKeysWithValues:
            firstGroup.map { ($0.evaluationSeed, $0.episodes) })
        guard grouped.values.allSatisfy({ reports in
            guard reports.map(\.evaluationSeed).sorted() == expectedSeeds,
                  Set(reports.map(\.checkpointDirectory)).count == 1,
                  Set(reports.map(\.trainingUpdates)).count == 1 else {
                return false
            }
            return reports.allSatisfy {
                expectedEpisodes[$0.evaluationSeed] == $0.episodes
            }
        }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "every checkpoint must use the same distinct validation seeds and episode counts")
        }

        let candidates = grouped.values.map { reports in
            let candidate = reports[0]
            let totalEpisodes = reports.reduce(0) { $0 + $1.episodes }
            let totalSuccesses = reports.reduce(0) { $0 + $1.successes }
            let weightedReturn = reports.reduce(Float(0)) {
                $0 + $1.meanReturn * Float($1.episodes)
            } / Float(totalEpisodes)
            return PPOCheckpointValidationCandidate(
                checkpointDirectory: candidate.checkpointDirectory,
                checkpointFingerprint: candidate.checkpointFingerprint!,
                trainingUpdates: candidate.trainingUpdates,
                evaluationSeeds: expectedSeeds,
                validationRuns: reports.count,
                totalEpisodes: totalEpisodes,
                successes: totalSuccesses,
                successRate: Float(totalSuccesses) / Float(totalEpisodes),
                minimumSeedSuccessRate: reports.map(\.successRate).min()!,
                meanReturn: weightedReturn,
                acceptancePassed: reports.allSatisfy {
                    $0.acceptance?.passed ?? true
                })
        }.sorted {
            if $0.trainingUpdates != $1.trainingUpdates {
                return $0.trainingUpdates < $1.trainingUpdates
            }
            return $0.checkpointFingerprint < $1.checkpointFingerprint
        }
        let selected = candidates.max {
            if $0.acceptancePassed != $1.acceptancePassed {
                return !$0.acceptancePassed && $1.acceptancePassed
            }
            if $0.minimumSeedSuccessRate != $1.minimumSeedSuccessRate {
                return ($0.minimumSeedSuccessRate ?? $0.successRate)
                    < ($1.minimumSeedSuccessRate ?? $1.successRate)
            }
            if $0.successRate != $1.successRate {
                return $0.successRate < $1.successRate
            }
            if $0.meanReturn != $1.meanReturn {
                return $0.meanReturn < $1.meanReturn
            }
            if $0.trainingUpdates != $1.trainingUpdates {
                return $0.trainingUpdates > $1.trainingUpdates
            }
            return $0.checkpointFingerprint > $1.checkpointFingerprint
        }!
        let uniformEpisodesPerSeed = Set(expectedEpisodes.values).count == 1
            ? expectedEpisodes.values.first : nil
        return Self(
            schemaVersion: 3,
            task: first.task,
            taskRevision: taskRevision,
            trainingSeed: first.trainingSeed,
            initializationCheckpoint: first.initializationCheckpoint,
            validationSeed: expectedSeeds[0],
            validationSeeds: expectedSeeds,
            validationEnvironments: evaluationEnvironments,
            evaluationTaskConfiguration: first.evaluationTaskConfiguration,
            taskConfigurationTransferred: first.taskConfigurationTransferred,
            validationEpisodesPerCandidate: expectedEpisodes.values.reduce(0, +),
            validationEpisodesPerSeed: uniformEpisodesPerSeed,
            selectionRule: "all_validation_acceptance_pass,worst_seed_success_rate,pooled_success_rate,mean_return,earlier_update,fingerprint",
            candidates: candidates,
            selectedCheckpointDirectory: selected.checkpointDirectory,
            selectedCheckpointFingerprint: selected.checkpointFingerprint,
            selectedTrainingUpdates: selected.trainingUpdates)
    }

    public func validateTestReport(_ evaluation: PPOEvaluationMetrics) throws {
        guard evaluation.task == task,
              evaluation.taskRevision == taskRevision,
              evaluation.trainingSeed == trainingSeed,
              evaluation.initializationCheckpoint == initializationCheckpoint,
              evaluation.checkpointDirectory == selectedCheckpointDirectory,
              evaluation.checkpointFingerprint == selectedCheckpointFingerprint,
              evaluationTaskConfiguration == nil
                || evaluation.evaluationTaskConfiguration
                    == evaluationTaskConfiguration,
              taskConfigurationTransferred == nil
                || evaluation.taskConfigurationTransferred
                    == taskConfigurationTransferred,
              validationEnvironments == nil
                || evaluation.evaluationEnvironments == validationEnvironments else {
            throw RLEnvironmentError.invalidConfiguration(
                "test report does not match the selected immutable checkpoint")
        }
        let seeds = Set(validationSeeds ?? [validationSeed])
        guard !seeds.contains(evaluation.evaluationSeed) else {
            throw RLEnvironmentError.invalidConfiguration(
                "test evaluation seed must differ from every checkpoint validation seed")
        }
    }
}

public struct VectorPolicyMetadata: Codable, Sendable {
    public var architectureVersion: Int?
    public var task: String
    public var taskRevision: Int?
    /// Exact semantic task configuration. Optional only so checkpoints from
    /// before metadata v2 can still be inspected or explicitly transferred;
    /// such checkpoints are never eligible for an in-place training resume.
    public var taskConfiguration: [String: Float]?
    public var observationDimension: Int
    public var actionDimension: Int
    public var simulationStep: Float
    public var controlDecimation: Int
    public var maxEpisodeSteps: Int
    public var ppo: VectorPPOConfig
    public var normalizer: RunningNormalizerSnapshot
}

public struct VectorPPOTrainingState: Codable, Sendable {
    public var completedUpdates: Int
    public var environmentSteps: Int
    /// Minibatch-level Adam step. Optional for pre-checkpoint-v2 runs.
    public var optimizerSteps: Int?
    /// Adaptive KL scheduler state. Optional for pre-checkpoint-v2 runs.
    public var adaptiveLearningRate: Float?

    public init(completedUpdates: Int, environmentSteps: Int,
                optimizerSteps: Int? = nil,
                adaptiveLearningRate: Float? = nil) {
        self.completedUpdates = completedUpdates
        self.environmentSteps = environmentSteps
        self.optimizerSteps = optimizerSteps
        self.adaptiveLearningRate = adaptiveLearningRate
    }
}

/// A checkpoint snapshot that is safe for another process to open. Training
/// writes each snapshot into a new `checkpoints/update-NNNNNN` directory and
/// writes `training-state.json` last. Discovering snapshots (rather than the
/// mutable run root) prevents replay from observing a mixture of old metadata
/// and newly replaced weights while an atomic per-file save is in progress.
public struct VectorPolicyCheckpointCandidate: Sendable, Equatable {
    public var directory: String
    public var completedUpdates: Int
    public var environmentSteps: Int

    public init(directory: String, completedUpdates: Int,
                environmentSteps: Int) {
        self.directory = directory
        self.completedUpdates = completedUpdates
        self.environmentSteps = environmentSteps
    }
}

public enum VectorPolicyCheckpointDiscovery {
    private struct Identity: Decodable {
        var task: String
        var taskRevision: Int?
    }

    private struct Progress: Decodable {
        var completedUpdates: Int
        var environmentSteps: Int
    }

    /// Return the highest numbered complete, task-compatible immutable
    /// snapshot. Malformed and half-written directories are ignored so a live
    /// trainer can be polled without transient replay failures.
    public static func latestCompleteCheckpoint(
        inRunDirectory runDirectory: String,
        task: String,
        taskRevision: Int
    ) -> VectorPolicyCheckpointCandidate? {
        let manager = FileManager.default
        let snapshots = URL(fileURLWithPath: runDirectory, isDirectory: true)
            .appendingPathComponent("checkpoints", isDirectory: true)
        guard let children = try? manager.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]) else { return nil }

        var latest: VectorPolicyCheckpointCandidate?
        for directory in children {
            let name = directory.lastPathComponent
            guard name.hasPrefix("update-"),
                  let numberedUpdate = Int(name.dropFirst("update-".count)),
                  (try? directory.resourceValues(forKeys: [.isDirectoryKey])
                    .isDirectory) == true else { continue }

            let required = [
                "policy.safetensors", "optimizer.safetensors",
                "metadata.json", "training-state.json",
            ]
            guard required.allSatisfy({ fileName in
                let path = directory.appendingPathComponent(fileName).path
                guard manager.isReadableFile(atPath: path),
                      let size = (try? manager.attributesOfItem(atPath: path)[.size])
                        as? NSNumber else { return false }
                return size.intValue > 0
            }) else { continue }

            guard let identity = try? JSONDecoder().decode(
                    Identity.self,
                    from: Data(contentsOf: directory
                        .appendingPathComponent("metadata.json"))),
                  identity.task == task,
                  (identity.taskRevision ?? 1) == taskRevision,
                  let progress = try? JSONDecoder().decode(
                    Progress.self,
                    from: Data(contentsOf: directory
                        .appendingPathComponent("training-state.json"))),
                  progress.completedUpdates == numberedUpdate else { continue }

            let candidate = VectorPolicyCheckpointCandidate(
                directory: directory.path,
                completedUpdates: progress.completedUpdates,
                environmentSteps: progress.environmentSteps)
            if latest == nil
                || candidate.completedUpdates > latest!.completedUpdates {
                latest = candidate
            }
        }
        return latest
    }
}

/// Adam with explicitly named, serializable moment buffers. MLX Swift's Adam
/// exposes its state only as an ordered array, which cannot be restored safely
/// against a named module tree. PPO checkpoints need optimizer state to make a
/// continuation mathematically meaningful, so the trainer owns this small
/// name-preserving implementation.
final class CheckpointableAdam {
    var learningRate: Float
    let betas: (Float, Float)
    let epsilon: Float
    private(set) var step: Int
    private var firstMoments: [String: MLXArray]
    private var secondMoments: [String: MLXArray]

    init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999),
         epsilon: Float = 1e-8, step: Int = 0) {
        self.learningRate = learningRate
        self.betas = betas
        self.epsilon = epsilon
        self.step = step
        firstMoments = [:]
        secondMoments = [:]
    }

    func update(model: Module, gradients: ModuleParameters) throws {
        let parameterPairs = model.parameters().flattened()
        let gradientMap = Dictionary(uniqueKeysWithValues: gradients.flattened())
        step += 1
        let beta1Correction = 1 - Float(pow(Double(betas.0), Double(step)))
        let beta2Correction = 1 - Float(pow(Double(betas.1), Double(step)))
        var updated = [(String, MLXArray)]()
        updated.reserveCapacity(parameterPairs.count)
        for (name, parameter) in parameterPairs {
            guard let gradient = gradientMap[name] else {
                throw RLEnvironmentError.invalidConfiguration(
                    "optimizer gradient is missing parameter \(name)")
            }
            let oldFirst = firstMoments[name] ?? MLXArray.zeros(like: parameter)
            let oldSecond = secondMoments[name] ?? MLXArray.zeros(like: parameter)
            let first = betas.0 * oldFirst + (1 - betas.0) * gradient
            let second = betas.1 * oldSecond + (1 - betas.1) * gradient.square()
            firstMoments[name] = first
            secondMoments[name] = second
            let firstHat = first / beta1Correction
            let secondHat = second / beta2Correction
            updated.append((name, parameter
                - learningRate * firstHat / (sqrt(secondHat) + epsilon)))
        }
        try model.update(parameters: ModuleParameters.unflattened(updated), verify: [.all])
    }

    func evaluate() {
        eval(Array(firstMoments.values) + Array(secondMoments.values))
    }

    func checkpointArrays() -> [String: MLXArray] {
        var arrays = [String: MLXArray]()
        arrays.reserveCapacity(firstMoments.count + secondMoments.count)
        for (name, value) in firstMoments { arrays["first.\(name)"] = value }
        for (name, value) in secondMoments { arrays["second.\(name)"] = value }
        return arrays
    }

    func restore(arrays: [String: MLXArray], step: Int,
                 parameters: ModuleParameters) throws {
        guard step >= 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "optimizer checkpoint has a negative step")
        }
        var restoredFirst = [String: MLXArray]()
        var restoredSecond = [String: MLXArray]()
        for (name, parameter) in parameters.flattened() {
            guard let first = arrays["first.\(name)"],
                  let second = arrays["second.\(name)"],
                  first.shape == parameter.shape,
                  second.shape == parameter.shape else {
                throw RLEnvironmentError.invalidConfiguration(
                    "optimizer checkpoint is missing or mismatches parameter \(name)")
            }
            restoredFirst[name] = first
            restoredSecond[name] = second
        }
        self.step = step
        firstMoments = restoredFirst
        secondMoments = restoredSecond
        evaluate()
    }
}

/// Stateful deterministic checkpoint runner shared by evaluation and the UI.
/// It owns only MLX inference state; the simulator remains task-owned.
public final class VectorPolicyRunner {
    public let metadata: VectorPolicyMetadata
    private let policy: VectorActorCritic
    private let normalizer: RunningObservationNormalizer

    public init(checkpointDirectory: String) throws {
        let metadataURL = URL(fileURLWithPath: "\(checkpointDirectory)/metadata.json")
        metadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self, from: Data(contentsOf: metadataURL))
        guard VectorActorCritic.compatibleArchitectureVersions.contains(
            metadata.architectureVersion ?? 1) else {
            throw RLEnvironmentError.invalidConfiguration(
                "legacy policy architecture is not replayable; train a fresh checkpoint")
        }
        policy = VectorActorCritic(
            observationDimension: metadata.observationDimension,
            actionDimension: metadata.actionDimension,
            hiddenSize: metadata.ppo.hiddenSize,
            hiddenDimensions: metadata.ppo.hiddenDimensions,
            initialActionStd: metadata.ppo.initialActionStd)
        let sourceWeights = try loadArrays(
            url: URL(fileURLWithPath: "\(checkpointDirectory)/policy.safetensors"))
        let weights = try VectorActorCritic.compatibleWeights(
            sourceWeights, architectureVersion: metadata.architectureVersion)
        try policy.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(policy)
        normalizer = RunningObservationNormalizer(snapshot: metadata.normalizer)
    }

    public func actions(
        for observation: RLObservationBatch,
        expertGates: ContiguousArray<Float>? = nil,
        standExpertGates: ContiguousArray<Float>? = nil
    ) throws -> RLActionBatch {
        let values = try actions(
            for: observation.policy, expertGates: expertGates,
            standExpertGates: standExpertGates)
        return try RLActionBatch(
            numEnvironments: values.count / metadata.actionDimension,
            actionDimension: metadata.actionDimension, values: values)
    }

    /// Deterministic inference over row-major policy observations without a
    /// simulator-owned observation wrapper. This is the deployment entry
    /// point used by an iPhone or other hardware controller.
    public func actions(
        for policyObservations: ContiguousArray<Float>,
        expertGates: ContiguousArray<Float>? = nil,
        standExpertGates: ContiguousArray<Float>? = nil
    ) throws -> ContiguousArray<Float> {
        let n = policyObservations.count / metadata.observationDimension
        guard n > 0,
              policyObservations.count == n * metadata.observationDimension else {
            throw RLEnvironmentError.invalidObservationCount(
                expected: metadata.observationDimension,
                actual: policyObservations.count)
        }
        if let index = policyObservations.firstIndex(where: { !$0.isFinite }) {
            throw RLEnvironmentError.invalidConfiguration(
                "observation value at flat index \(index) is not finite")
        }
        let normalized = metadata.ppo.normalizeObservations
            ? normalizer.normalize(policyObservations) : policyObservations
        let input = MLXArray(Array(normalized)).reshaped(
            [n, metadata.observationDimension])
        if let expertGates, expertGates.count != n {
            throw RLEnvironmentError.invalidConfiguration(
                "policy expert gate count must match observation rows")
        }
        if let standExpertGates, standExpertGates.count != n {
            throw RLEnvironmentError.invalidConfiguration(
                "policy stand expert gate count must match observation rows")
        }
        let gate = expertGates.map {
            MLXArray(Array($0)).reshaped([n, 1])
        }
        let standGate = standExpertGates.map {
            MLXArray(Array($0)).reshaped([n, 1])
        }
        let output = metadata.ppo.resolvedActionDistribution.environmentAction(
            policy.forward(
                input, expertGate: gate, standExpertGate: standGate).mean)
        eval(output)
        let values = ContiguousArray(output.asArray(Float.self))
        if let index = values.firstIndex(where: { !$0.isFinite }) {
            throw RLEnvironmentError.nonFiniteAction(index: index)
        }
        return values
    }
}

public final class VectorPPOTrainer {
    public let configuration: VectorPPOConfig
    public var onUpdate: ((PPOUpdateMetrics) -> Void)?

    private static let logSqrt2Pi: Float = 0.9189385332
    /// A mathematically equivalent PPO ratio can overflow before clipping
    /// when a minibatch contains an action that became very unlikely under
    /// the updated policy. Bounding the *log* ratio keeps both the loss and
    /// its diagnostics finite without changing the ordinary PPO trust region
    /// (exp(+-20) is already far outside the configured 0.8...1.2 clip).
    static let maximumImportanceLogRatio: Float = 20

    public init(configuration: VectorPPOConfig = VectorPPOConfig()) {
        self.configuration = configuration
    }

    static func markSuccessfulEpisodeSegment(
        mask: inout [Float], environment: Int, numEnvironments: Int,
        startStep: Int, endStep: Int
    ) {
        precondition(numEnvironments > 0)
        precondition((0..<numEnvironments).contains(environment))
        precondition(startStep >= 0 && endStep >= startStep)
        precondition(endStep * numEnvironments + environment < mask.count)
        for step in startStep...endStep {
            mask[step * numEnvironments + environment] = 1
        }
    }

    /// Recover the signed permutation represented by a task's row-wise
    /// tensor mirror. Evaluating one-hot rows keeps the task as the single
    /// owner of observation semantics and avoids duplicating index tables in
    /// the learner.
    static func signedPermutation(
        dimension: Int,
        transform: (ContiguousArray<Float>) -> ContiguousArray<Float>
    ) throws -> (sources: [Int], signs: [Float]) {
        guard dimension > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy mirror dimension must be positive")
        }
        var basis = ContiguousArray(
            repeating: Float(0), count: dimension * dimension)
        for row in 0..<dimension { basis[row * dimension + row] = 1 }
        let transformed = transform(basis)
        guard transformed.count == basis.count else {
            throw RLEnvironmentError.invalidConfiguration(
                "task observation mirror changed tensor shape")
        }
        var sources = [Int](repeating: -1, count: dimension)
        var signs = [Float](repeating: 0, count: dimension)
        for output in 0..<dimension {
            for input in 0..<dimension {
                let coefficient = transformed[input * dimension + output]
                if abs(coefficient) > 1e-6 {
                    guard sources[output] == -1,
                          abs(abs(coefficient) - 1) < 1e-6 else {
                        throw RLEnvironmentError.invalidConfiguration(
                            "task observation mirror is not a signed permutation")
                    }
                    sources[output] = input
                    signs[output] = coefficient
                }
            }
        }
        guard Set(sources).count == dimension,
              sources.allSatisfy({ 0..<dimension ~= $0 }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "task observation mirror is not a complete permutation")
        }
        return (sources, signs)
    }

    /// Fraction of each routed row whose actor output can receive gradients.
    /// Critic training still uses every row; this mask keeps frozen actor
    /// branches out of policy normalization, loss denominators, and KL
    /// scheduling instead of silently diluting specialist updates.
    static func actorTrainingWeights(
        expertGates: [Float], standExpertGates: [Float],
        freezesBaseActor: Bool, freezesExpertActor: Bool
    ) -> [Float] {
        precondition(expertGates.count == standExpertGates.count)
        return zip(expertGates, standExpertGates).map { expert, stand in
            let base = max(1 - expert - stand, 0)
            return (freezesBaseActor ? 0 : base)
                + (freezesExpertActor ? 0 : expert) + stand
        }
    }

    static func weightedNormalizedAdvantages(
        _ advantages: [Float], weights: [Float]
    ) -> (values: [Float], mean: Float, variance: Float, scale: Float) {
        precondition(advantages.count == weights.count)
        let denominator = weights.reduce(0, +)
        guard denominator > 1e-8 else {
            return ([Float](repeating: 0, count: advantages.count), 0, 0, 0)
        }
        let mean = zip(advantages, weights).reduce(Float(0)) {
            $0 + $1.0 * $1.1
        } / denominator
        let variance = zip(advantages, weights).reduce(Float(0)) {
            let delta = $1.0 - mean
            return $0 + $1.1 * delta * delta
        } / denominator
        let scale = 1 / max(sqrt(variance), 1e-8)
        return (advantages.map { ($0 - mean) * scale }, mean, variance, scale)
    }

    public func train(task: any VectorizedRLTask, outputDirectory: String) throws {
        try train(task: task, outputDirectory: outputDirectory, resume: false)
    }

    public func train(task: any VectorizedRLTask, outputDirectory: String,
                      resume: Bool) throws {
        if resume && (configuration.initializationCheckpoint != nil
            || configuration.policyExpertInitializationCheckpoint != nil
            || configuration.standExpertInitializationCheckpoint != nil) {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO resume and checkpoint initialization are mutually exclusive")
        }
        (task as? any TrainingModeConfigurable)?.setTrainingMode(true)
        defer { (task as? any TrainingModeConfigurable)?.setTrainingMode(false) }
        let spec = task.spec
        let n = spec.numEnvironments
        let obsDim = spec.observation.elementCount
        let actionDim = spec.action.elementCount
        let taskPolicySymmetry = task as? any PolicySymmetryProviding
        let policySymmetry = configuration.useTaskSymmetryAugmentation != false
            ? taskPolicySymmetry : nil
        let policyExpertGate = task as? any PolicyExpertGateProviding
        let usesPolicyExpertGate = policyExpertGate?.usesPolicyExpertGate == true
        let policyStandExpertGate = task as? any PolicyStandExpertGateProviding
        let usesPolicyStandExpertGate =
            policyStandExpertGate?.usesPolicyStandExpertGate == true
        let freezesBasePolicyExpert = usesPolicyExpertGate
            && policyExpertGate?.freezesBasePolicyExpert == true
        let freezesLowSpeedPolicyExpert = usesPolicyStandExpertGate
            && policyStandExpertGate?.freezesLowSpeedPolicyExpert == true
        let symmetryMirrorLossCoefficient =
            configuration.symmetryMirrorLossCoefficient ?? 0
        let successImitationCoefficient =
            configuration.successImitationCoefficient ?? 0
        let referencePolicyCoefficient =
            configuration.referencePolicyCoefficient ?? 0
        let referenceRegularizationProvider = task as?
            any PolicyReferenceRegularizationProviding
        let actionDistribution = configuration.resolvedActionDistribution
        let usesSymmetryMirrorLoss = policySymmetry != nil
            && symmetryMirrorLossCoefficient > 0
        let usesSymmetryDataAugmentation = policySymmetry != nil
            && !usesSymmetryMirrorLoss
        if let policySymmetry {
            guard policySymmetry.policyActionMirrorSourceIndices.count == actionDim,
                  policySymmetry.policyActionMirrorSigns.count == actionDim,
                  policySymmetry.policyActionMirrorSourceIndices.allSatisfy({
                      0..<actionDim ~= $0
                  }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task action symmetry is not a valid signed permutation")
            }
        }
        let batchSize = n * configuration.rolloutSteps
        try configuration.validate(batchSize: batchSize)
        if referencePolicyCoefficient > 0, !resume,
           configuration.initializationCheckpoint == nil {
            throw RLEnvironmentError.invalidConfiguration(
                "reference-policy regularization requires --initialize-from")
        }
        try FileManager.default.createDirectory(atPath: outputDirectory,
                                                withIntermediateDirectories: true)
        let metricsURL = URL(fileURLWithPath: "\(outputDirectory)/metrics.jsonl")
        if !resume {
            _ = FileManager.default.createFile(atPath: metricsURL.path, contents: nil)
        } else if !FileManager.default.fileExists(atPath: metricsURL.path) {
            _ = FileManager.default.createFile(atPath: metricsURL.path, contents: nil)
        }
        let metricsFile = try FileHandle(forWritingTo: metricsURL)
        if resume { try metricsFile.seekToEnd() }
        defer { try? metricsFile.close() }
        MLXRandom.seed(configuration.seed)
        if usesSymmetryMirrorLoss {
            Swift.print("using task-provided actor mirror loss with coefficient "
                + "\(symmetryMirrorLossCoefficient)")
        } else if usesSymmetryDataAugmentation {
            Swift.print("using legacy task-provided policy symmetry augmentation "
                + "for half of every PPO rollout")
        }

        let policy = VectorActorCritic(observationDimension: obsDim,
                                       actionDimension: actionDim,
                                       hiddenSize: configuration.hiddenSize,
                                       hiddenDimensions:
                                           configuration.hiddenDimensions,
                                       initialActionStd: configuration.initialActionStd)
        var normalizer = RunningObservationNormalizer(dimension: obsDim)
        var startingUpdate = 0
        var totalSteps = 0
        var restoredTrainingState: VectorPPOTrainingState?
        var persistedConfiguration = configuration
        if resume {
            let metadataURL = URL(fileURLWithPath: "\(outputDirectory)/metadata.json")
            let metadata = try JSONDecoder().decode(
                VectorPolicyMetadata.self, from: Data(contentsOf: metadataURL))
            guard let checkpointTaskConfiguration = metadata.taskConfiguration else {
                throw RLEnvironmentError.invalidConfiguration(
                    "resume checkpoint predates serialized task configuration; "
                    + "use --initialize-from for an explicit policy transfer")
            }
            let ppoIncompatibilities = configuration.resumeIncompatibilities(
                with: metadata.ppo)
            guard metadata.task == spec.id,
                  (metadata.taskRevision ?? 1) == spec.revision,
                  checkpointTaskConfiguration == spec.configurationValues,
                  metadata.observationDimension == obsDim,
                  metadata.actionDimension == actionDim,
                  metadata.simulationStep == spec.simulationStep,
                  metadata.controlDecimation == spec.controlDecimation,
                  metadata.maxEpisodeSteps == spec.maxEpisodeSteps,
                  metadata.architectureVersion == VectorActorCritic.architectureVersion,
                  ppoIncompatibilities.isEmpty else {
                throw RLEnvironmentError.invalidConfiguration(
                    "resume checkpoint is incompatible with the exact task, "
                    + "dynamics, or PPO trajectory; changed PPO fields: "
                    + (ppoIncompatibilities.isEmpty
                        ? "none" : ppoIncompatibilities.joined(separator: ", "))
                    + ". Use --initialize-from for an explicit fine-tune.")
            }
            persistedConfiguration = configuration
                .preservingInitializationProvenance(from: metadata.ppo)
            let sourceWeights = try loadArrays(url: URL(
                fileURLWithPath: "\(outputDirectory)/policy.safetensors"))
            let weights = try VectorActorCritic.compatibleWeights(
                sourceWeights, architectureVersion: metadata.architectureVersion)
            try policy.update(parameters: ModuleParameters.unflattened(weights),
                              verify: [.all])
            normalizer = RunningObservationNormalizer(snapshot: metadata.normalizer)
            let stateURL = URL(fileURLWithPath: "\(outputDirectory)/training-state.json")
            if FileManager.default.fileExists(atPath: stateURL.path) {
                let state = try JSONDecoder().decode(
                    VectorPPOTrainingState.self, from: Data(contentsOf: stateURL))
                startingUpdate = state.completedUpdates
                totalSteps = state.environmentSteps
                restoredTrainingState = state
            } else {
                // Backward-compatible inference for checkpoints written before
                // explicit trainer state existed. Observation count is exact
                // for ordinary PPO rollouts; demonstration pretraining users
                // should start a fresh run rather than rely on this fallback.
                totalSteps = Int(normalizer.snapshot.count)
                startingUpdate = totalSteps / batchSize
            }
        } else if let checkpointDirectory = configuration.initializationCheckpoint {
            let metadataURL = URL(
                fileURLWithPath: "\(checkpointDirectory)/metadata.json")
            let metadata = try JSONDecoder().decode(
                VectorPolicyMetadata.self, from: Data(contentsOf: metadataURL))
            // `--initialize-from` is an explicit transfer operation, unlike
            // exact resume. Permit a different task id when the complete
            // network interface and architecture match; this supports a
            // locomotion backbone moving from straight walking to goal
            // steering without weakening resume/evaluation provenance.
            let observationSourceIndices = metadata.observationDimension == obsDim
                ? nil
                : (task as? any ObservationSchemaTransferProviding)?
                    .initializationObservationSourceIndices(
                        sourceDimension: metadata.observationDimension)
            guard (metadata.observationDimension == obsDim
                    || observationSourceIndices?.count == obsDim),
                  metadata.actionDimension == actionDim,
                  VectorActorCritic.compatibleArchitectureVersions.contains(
                      metadata.architectureVersion ?? 1),
                  metadata.ppo.resolvedHiddenDimensions
                    == configuration.resolvedHiddenDimensions else {
                throw RLEnvironmentError.invalidConfiguration(
                    "initialization checkpoint has an incompatible network interface")
            }
            let sourceWeights = try loadArrays(url: URL(
                fileURLWithPath: "\(checkpointDirectory)/policy.safetensors"))
            var weights = try VectorActorCritic.compatibleWeights(
                sourceWeights, architectureVersion: metadata.architectureVersion)
            if let observationSourceIndices {
                weights = try VectorActorCritic.remappingObservationInputs(
                    weights, sourceIndices: observationSourceIndices)
            }
            if usesPolicyStandExpertGate,
               (metadata.architectureVersion ?? 1) == 4,
               VectorActorCritic.legacyExpertIsStandOnly(
                    taskConfiguration: metadata.taskConfiguration ?? [:]) {
                weights = try VectorActorCritic
                    .initializingLowSpeedExpertFromBase(weights)
                Swift.print("mapped legacy expert to stand branch and initialized "
                    + "new low-speed expert from frozen base actor")
            } else if usesPolicyStandExpertGate,
                      (metadata.architectureVersion ?? 1) == 4 {
                Swift.print("mapped shared legacy braking expert to both low-speed "
                    + "and stand branches")
            }
            // Policy transfer keeps the learned mean/value features but uses
            // the new experiment's exploration schedule. Replace the serialized
            // module parameter before `update`; assigning the Swift property
            // afterward does not mutate MLX's registered parameter tree.
            guard weights["logStandardDeviation"] != nil else {
                throw RLEnvironmentError.invalidConfiguration(
                    "initialization checkpoint has no policy log standard deviation")
            }
            weights["logStandardDeviation"] = MLXArray([Float](
                repeating: log(configuration.initialActionStd),
                count: actionDim))
            let varianceFloors = (task as?
                any ObservationNormalizerTransferProviding)?
                .initializationObservationVarianceFloors ?? [:]
            var importedNormalizer = metadata.normalizer.limitingPriorCount(
                to: configuration.initializationNormalizerPriorCount)
            if let observationSourceIndices {
                importedNormalizer = try importedNormalizer
                    .remappingObservationChannels(
                        sourceIndices: observationSourceIndices)
            }
            importedNormalizer = try importedNormalizer
                .applyingVarianceFloors(varianceFloors)
            if usesPolicyExpertGate,
               policyExpertGate?
                .initializesPolicyExpertFromMirroredBaseOnTransfer == true {
                guard let taskPolicySymmetry else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "mirrored expert initialization requires task symmetry")
                }
                let observationMirror = try Self.signedPermutation(
                    dimension: obsDim,
                    transform: taskPolicySymmetry.mirrorPolicyObservations)
                weights = try VectorActorCritic
                    .initializingPolicyExpertAsMirroredBase(
                        weights,
                        observationSources: observationMirror.sources,
                        observationSigns: observationMirror.signs,
                        actionSources:
                            taskPolicySymmetry.policyActionMirrorSourceIndices,
                        actionSigns: taskPolicySymmetry.policyActionMirrorSigns,
                        normalizer: importedNormalizer,
                        normalizesObservations:
                            configuration.normalizeObservations)
                Swift.print("initialized routed policy expert as the exact "
                    + "symmetry transform of the transferred base actor")
            } else if usesPolicyExpertGate,
                      policyExpertGate?
                        .initializesPolicyExpertFromBaseOnTransfer == true {
                weights = try VectorActorCritic
                    .initializingLowSpeedExpertFromBase(weights)
                Swift.print("initialized routed policy expert as an exact copy "
                    + "of the transferred base actor")
            }
            if let expertCheckpoint =
                configuration.policyExpertInitializationCheckpoint {
                guard usesPolicyExpertGate else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "policy-expert composition requires a routed-expert task")
                }
                let expertMetadataURL = URL(
                    fileURLWithPath: "\(expertCheckpoint)/metadata.json")
                let expertMetadata = try JSONDecoder().decode(
                    VectorPolicyMetadata.self,
                    from: Data(contentsOf: expertMetadataURL))
                guard expertMetadata.observationDimension == obsDim,
                      expertMetadata.actionDimension == actionDim,
                      VectorActorCritic.compatibleArchitectureVersions
                        .contains(expertMetadata.architectureVersion ?? 1),
                      expertMetadata.ppo.resolvedHiddenDimensions
                        == configuration.resolvedHiddenDimensions,
                      expertMetadata.ppo.resolvedActionDistribution
                        == configuration.resolvedActionDistribution else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "policy-expert checkpoint has an incompatible network interface")
                }
                let expertSourceWeights = try loadArrays(url: URL(
                    fileURLWithPath: "\(expertCheckpoint)/policy.safetensors"))
                let expertWeights = try VectorActorCritic.compatibleWeights(
                    expertSourceWeights,
                    architectureVersion: expertMetadata.architectureVersion)
                weights = try VectorActorCritic.initializingPolicyExpert(
                    weights, from: expertWeights,
                    sourceNormalizer: expertMetadata.normalizer,
                    destinationNormalizer: importedNormalizer,
                    sourceNormalizesObservations:
                        expertMetadata.ppo.normalizeObservations,
                    destinationNormalizesObservations:
                        configuration.normalizeObservations)
                Swift.print("composed routed policy expert from \(expertCheckpoint) "
                    + "with exact observation-normalizer reparameterization")
            }
            if let standCheckpoint =
                configuration.standExpertInitializationCheckpoint {
                guard usesPolicyStandExpertGate else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "stand-expert composition requires a three-mode task")
                }
                let standMetadataURL = URL(
                    fileURLWithPath: "\(standCheckpoint)/metadata.json")
                let standMetadata = try JSONDecoder().decode(
                    VectorPolicyMetadata.self,
                    from: Data(contentsOf: standMetadataURL))
                guard standMetadata.observationDimension == obsDim,
                      standMetadata.actionDimension == actionDim,
                      VectorActorCritic.compatibleArchitectureVersions
                        .contains(standMetadata.architectureVersion ?? 1),
                      standMetadata.ppo.resolvedHiddenDimensions
                        == configuration.resolvedHiddenDimensions,
                      standMetadata.ppo.resolvedActionDistribution
                        == configuration.resolvedActionDistribution else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "stand-expert checkpoint has an incompatible network interface")
                }
                let standSourceWeights = try loadArrays(url: URL(
                    fileURLWithPath: "\(standCheckpoint)/policy.safetensors"))
                let standWeights = try VectorActorCritic.compatibleWeights(
                    standSourceWeights,
                    architectureVersion: standMetadata.architectureVersion)
                weights = try VectorActorCritic.initializingStandExpert(
                    weights, from: standWeights,
                    sourceNormalizer: standMetadata.normalizer,
                    destinationNormalizer: importedNormalizer)
                Swift.print("composed stand expert from \(standCheckpoint) "
                    + "with exact observation-normalizer reparameterization")
            }
            try policy.update(parameters: ModuleParameters.unflattened(weights),
                              verify: [.all])
            normalizer = RunningObservationNormalizer(snapshot: importedNormalizer)
            let priorDescription = configuration.initializationNormalizerPriorCount
                .map { ", normalizer prior count capped at \(Int($0))" } ?? ""
            let floorDescription = varianceFloors.isEmpty
                ? "" : ", \(varianceFloors.count) task-declared variance floors applied"
            let schemaDescription = observationSourceIndices == nil
                ? "" : ", observation schema remapped "
                    + "\(metadata.observationDimension)->\(obsDim)"
            let actionTransformDescription =
                metadata.ppo.resolvedActionDistribution
                    == configuration.resolvedActionDistribution
                ? ""
                : ", action transform "
                    + "\(metadata.ppo.resolvedActionDistribution.rawValue)->"
                    + configuration.resolvedActionDistribution.rawValue
            Swift.print("initialized policy and observation statistics from "
                + "\(checkpointDirectory)\(priorDescription)\(floorDescription)"
                + "\(schemaDescription)\(actionTransformDescription); "
                + "Adam, KL scheduler, "
                + "and progress start fresh")
        }
        (task as? any TrainingModeConfigurable)?.setTrainingProgress(
            environmentSteps: totalSteps)
        eval(policy)
        let referencePolicy: VectorActorCritic? = try {
            guard referencePolicyCoefficient > 0 else { return nil }
            let frozen = VectorActorCritic(
                observationDimension: obsDim,
                actionDimension: actionDim,
                hiddenSize: configuration.hiddenSize,
                hiddenDimensions: configuration.hiddenDimensions,
                initialActionStd: configuration.initialActionStd)
            let weights: [String: MLXArray]
            if resume {
                let url = URL(fileURLWithPath:
                    "\(outputDirectory)/reference-policy.safetensors")
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "resume checkpoint has reference-policy regularization "
                            + "but no frozen reference-policy.safetensors")
                }
                weights = try loadArrays(url: url)
            } else {
                weights = Dictionary(uniqueKeysWithValues:
                    policy.parameters().flattened().map { ($0.0, $0.1) })
            }
            try frozen.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: [.all])
            eval(frozen)
            Swift.print("using frozen initialization-policy retention with "
                + "coefficient \(referencePolicyCoefficient)")
            return frozen
        }()
        // Robotics PPO implementations (RSL-RL, Brax, CleanRL) use Adam
        // without decoupled weight decay. Bias correction matters most after
        // a fresh start or policy-only resume.
        let optimizer = CheckpointableAdam(learningRate: configuration.learningRate)
        var adaptiveLearningRate = restoredTrainingState?.adaptiveLearningRate
            ?? configuration.learningRate
        if resume {
            let optimizerURL = URL(
                fileURLWithPath: "\(outputDirectory)/optimizer.safetensors")
            if let optimizerSteps = restoredTrainingState?.optimizerSteps,
               FileManager.default.fileExists(atPath: optimizerURL.path) {
                let arrays = try loadArrays(url: optimizerURL)
                try optimizer.restore(arrays: arrays, step: optimizerSteps,
                                      parameters: policy.parameters())
                Swift.print("resumed policy, normalizer, Adam moments, and KL scheduler "
                    + "from \(outputDirectory) at update \(startingUpdate), "
                    + "\(totalSteps) environment steps")
            } else {
                Swift.print("resumed legacy policy and observation statistics from "
                    + "\(outputDirectory) at update \(startingUpdate), \(totalSteps) "
                    + "environment steps; Adam moments restart")
            }
        }
        var observation = try task.reset(seed: configuration.seed)
        // A task's success signal describes the current transition. Keep the
        // episode reduction here so fixed-horizon tasks can distinguish
        // success-once from success-at-end without changing the generic API.
        var episodeSucceeded = [Bool](repeating: false, count: n)
        var stepResult = RLStepBatch(spec: spec)
        let actionMirrorSources = MLXArray(
            (policySymmetry?.policyActionMirrorSourceIndices
                ?? Array(0..<actionDim)).map(Int32.init))
        let actionMirrorSigns = MLXArray(
            policySymmetry?.policyActionMirrorSigns
                ?? [Float](repeating: 1, count: actionDim))

        let lossAndGradient = valueAndGrad(model: policy) {
            (model: VectorActorCritic, args: [MLXArray]) -> [MLXArray] in
            let observations = args[0], preTanhActions = args[1]
            let oldLogProb = args[2], advantages = args[3]
            let returns = args[4], oldValues = args[5]
            let entropyNoise = args[6]
            let klWeights = args[7]
            let mirroredObservations = args[8]
            let expertGates = args[9]
            let standExpertGates = args[10]
            let successImitationMask = args[11]
            let referencePolicyWeights = args[12]
            let actorTrainingWeights = args[13]
            let out = model.forward(
                observations, expertGate: expertGates,
                standExpertGate: standExpertGates,
                freezeBaseActor: freezesBasePolicyExpert,
                freezeExpertActor: freezesLowSpeedPolicyExpert)
            let effectiveLogStandardDeviation = clip(
                out.logStandardDeviation,
                min: self.configuration.minimumActionStd.map(log) ?? -5,
                max: self.configuration.maximumActionStd.map(log) ?? 1)
            let logProb = Self.gaussianLogProbability(
                preTanhActions, mean: out.mean,
                logStandardDeviation: effectiveLogStandardDeviation)
            let (ratio, logRatio) = Self.stableImportanceRatio(
                newLogProbability: logProb,
                oldLogProbability: oldLogProb)
            let loss1 = -advantages * ratio
            let loss2 = -advantages * clip(ratio,
                                            min: 1 - self.configuration.policyClip,
                                            max: 1 + self.configuration.policyClip)
            let actorTrainingDenominator = clip(
                sum(actorTrainingWeights), min: 1,
                max: Float.greatestFiniteMagnitude)
            let policyLoss = sum(
                maximum(loss1, loss2) * actorTrainingWeights)
                / actorTrainingDenominator
            let clippedValue = oldValues + clip(out.value - oldValues,
                                                min: -self.configuration.valueClip,
                                                max: self.configuration.valueClip)
            let valueLoss = 0.5 * mean(maximum((out.value - returns).square(),
                                               (clippedValue - returns).square()))
            let entropy: MLXArray
            switch actionDistribution {
            case .gaussian:
                // RSL-RL uses the analytic entropy of its unbounded diagonal
                // Gaussian action distribution.
                entropy = sum(effectiveLogStandardDeviation
                    + Float(1.4189385332))
            case .squashedGaussian:
                // Entropy must belong to the policy the environment actually
                // receives. A reparameterized Monte-Carlo estimate includes
                // the tanh Jacobian and does not reward saturated ±1 actions.
                let entropyPreTanh = out.mean + entropyNoise
                    * exp(effectiveLogStandardDeviation)
                let entropyActions = tanh(entropyPreTanh)
                let entropyLogProbability = Self.gaussianLogProbability(
                    entropyPreTanh, mean: out.mean,
                    logStandardDeviation: effectiveLogStandardDeviation)
                    - sum(log(clip(1 - entropyActions.square(),
                                   min: 1e-6, max: 1)), axis: -1)
                entropy = -mean(entropyLogProbability)
            }
            // Symmetry-transformed actions were not sampled from the policy
            // at their transformed observations. They are valid supervised
            // PPO augmentation rows, but not Monte-Carlo samples for the
            // behavior-policy KL estimator. Match RSL-RL's trust-region
            // scheduling semantics by measuring KL only on original rollout
            // rows; otherwise an asymmetric transferred policy reports a huge
            // false KL on update one and collapses the adaptive learning rate.
            let trainableKLWeights = klWeights * actorTrainingWeights
            let klDenominator = clip(sum(trainableKLWeights), min: 1,
                                     max: Float.greatestFiniteMagnitude)
            let approximateKL = sum(
                ((ratio - 1) - logRatio) * trainableKLWeights)
                / klDenominator
            var symmetryLoss = MLXArray(Float(0))
            if usesSymmetryMirrorLoss {
                let mirroredMean = model.forward(
                    mirroredObservations, expertGate: expertGates,
                    standExpertGate: standExpertGates,
                    freezeBaseActor: freezesBasePolicyExpert,
                    freezeExpertActor: freezesLowSpeedPolicyExpert).mean
                // Compare the actual deterministic environment action. For a
                // squashed policy this avoids over-penalizing saturated
                // logits; for RSL's Gaussian policy it compares raw means.
                let expectedMirroredMean = stopGradient(
                    actionDistribution.environmentAction(out.mean)
                        .take(actionMirrorSources, axis: 1)
                        * actionMirrorSigns)
                let rowError = mean(
                    (actionDistribution.environmentAction(mirroredMean)
                        - expectedMirroredMean).square(), axis: -1)
                symmetryLoss = sum(rowError * actorTrainingWeights)
                    / actorTrainingDenominator
            }
            let imitationActionError = sum(
                (actionDistribution.environmentAction(out.mean)
                    - actionDistribution.environmentAction(preTanhActions))
                    .square(), axis: -1)
            let trainableImitationMask = successImitationMask
                * actorTrainingWeights
            let imitationDenominator = clip(
                sum(trainableImitationMask), min: 1,
                max: Float.greatestFiniteMagnitude)
            let successImitationLoss = sum(
                imitationActionError * trainableImitationMask)
                / imitationDenominator
            var referencePolicyLoss = MLXArray(Float(0))
            if let referencePolicy {
                let referenceMean = stopGradient(referencePolicy.forward(
                    observations, expertGate: expertGates,
                    standExpertGate: standExpertGates).mean)
                let referenceActionError = sum(
                    (actionDistribution.environmentAction(out.mean)
                        - actionDistribution.environmentAction(referenceMean))
                        .square(), axis: -1)
                let trainableReferenceWeights = referencePolicyWeights
                    * actorTrainingWeights
                let referenceDenominator = clip(
                    sum(trainableReferenceWeights), min: 1,
                    max: Float.greatestFiniteMagnitude)
                referencePolicyLoss = sum(
                    referenceActionError * trainableReferenceWeights)
                    / referenceDenominator
            }
            let total = policyLoss
                + self.configuration.valueCoefficient * valueLoss
                - self.configuration.entropyCoefficient * entropy
                + symmetryMirrorLossCoefficient * symmetryLoss
                + successImitationCoefficient * successImitationLoss
                + referencePolicyCoefficient * referencePolicyLoss
            return [total, policyLoss, valueLoss, entropy, approximateKL,
                    symmetryLoss, successImitationLoss, referencePolicyLoss]
        }

        for localUpdate in 0..<configuration.updates {
            let update = startingUpdate + localUpdate
            let startTime = Date()
            optimizer.learningRate = adaptiveLearningRate

            var storedObservations = [Float](); storedObservations.reserveCapacity(batchSize * obsDim)
            var storedMirroredObservations = [Float]()
            storedMirroredObservations.reserveCapacity(batchSize * obsDim)
            var storedActions = [Float](); storedActions.reserveCapacity(batchSize * actionDim)
            var storedLogProb = [Float](); storedLogProb.reserveCapacity(batchSize)
            var storedValues = [Float](); storedValues.reserveCapacity(batchSize)
            var storedRewards = [Float](); storedRewards.reserveCapacity(batchSize)
            var storedDones = [Bool](); storedDones.reserveCapacity(batchSize)
            var storedKLWeights = [Float](); storedKLWeights.reserveCapacity(batchSize)
            var storedSuccessImitationMask = [Float](
                repeating: 0, count: batchSize)
            var storedReferencePolicyWeights = [Float]()
            storedReferencePolicyWeights.reserveCapacity(batchSize)
            var episodeStartSteps = [Int](repeating: 0, count: n)
            var storedExpertGates = [Float]();
            storedExpertGates.reserveCapacity(batchSize)
            var storedStandExpertGates = [Float]();
            storedStandExpertGates.reserveCapacity(batchSize)
            var storedActorTrainingWeights = [Float]();
            storedActorTrainingWeights.reserveCapacity(batchSize)
            var completedEpisodes = 0, successfulEpisodes = 0
            var completedReturnSum: Float = 0, completedLengthSum: Float = 0
            var completedDistanceSum: Float = 0
            var taskMetricSums = [String: Float]()
            var taskMetricCounts = [String: Int]()

            for rolloutStep in 0..<configuration.rolloutSteps {
                let mirroredRawObservation = policySymmetry?.mirrorPolicyObservations(
                    observation.policy)
                if configuration.normalizeObservations
                    && configuration.updateObservationNormalizer != false {
                    normalizer.update(observation.policy, rows: n)
                    // Feeding both exact symmetry partners to the running
                    // statistics prevents a one-sided behavior policy from
                    // baking left/right bias into normalization itself.
                    if let mirroredRawObservation {
                        normalizer.update(mirroredRawObservation, rows: n)
                    }
                }
                let normalized = configuration.normalizeObservations
                    ? normalizer.normalize(observation.policy)
                    : observation.policy
                var trainingNormalized = normalized
                if usesSymmetryDataAugmentation,
                   let mirroredRawObservation {
                    let augmentedRaw = Self.alternatingSymmetryRows(
                        original: observation.policy,
                        mirrored: mirroredRawObservation,
                        rowDimension: obsDim, mirroredParity: update & 1)
                    trainingNormalized = configuration.normalizeObservations
                        ? normalizer.normalize(augmentedRaw) : augmentedRaw
                }
                let mirroredNormalized: ContiguousArray<Float>
                if usesSymmetryMirrorLoss, let mirroredRawObservation {
                    mirroredNormalized = configuration.normalizeObservations
                        ? normalizer.normalize(mirroredRawObservation)
                        : mirroredRawObservation
                } else {
                    mirroredNormalized = trainingNormalized
                }
                guard normalized.allSatisfy(\.isFinite),
                      trainingNormalized.allSatisfy(\.isFinite),
                      mirroredNormalized.allSatisfy(\.isFinite) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "non-finite normalized observation before PPO update \(update + 1)")
                }
                storedObservations.append(contentsOf: trainingNormalized)
                storedMirroredObservations.append(
                    contentsOf: mirroredNormalized)
                let rolloutReferencePolicyWeights: ContiguousArray<Float>
                if referencePolicyCoefficient > 0,
                   let referenceRegularizationProvider {
                    rolloutReferencePolicyWeights =
                        referenceRegularizationProvider
                            .policyReferenceRegularizationWeights(
                                observation.policy)
                } else {
                    rolloutReferencePolicyWeights = ContiguousArray(
                        repeating: 1, count: n)
                }
                guard rolloutReferencePolicyWeights.count == n,
                      rolloutReferencePolicyWeights.allSatisfy({
                          $0.isFinite && (0...1).contains($0)
                      }) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "task returned invalid reference-policy weights")
                }
                storedReferencePolicyWeights.append(
                    contentsOf: rolloutReferencePolicyWeights)
                let rolloutExpertGates = usesPolicyExpertGate
                    ? policyExpertGate!.policyExpertGates(observation.policy)
                    : ContiguousArray(repeating: Float(0), count: n)
                let rolloutStandExpertGates = usesPolicyStandExpertGate
                    ? policyStandExpertGate!.policyStandExpertGates(
                        observation.policy)
                    : ContiguousArray(repeating: Float(0), count: n)
                guard rolloutExpertGates.count == n,
                      rolloutStandExpertGates.count == n,
                      rolloutExpertGates.allSatisfy({ (0...1).contains($0) }),
                      rolloutStandExpertGates.allSatisfy({ (0...1).contains($0) }),
                      zip(rolloutExpertGates, rolloutStandExpertGates)
                        .allSatisfy({ $0.0 + $0.1 <= 1 }) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "task returned invalid or overlapping policy expert gates")
                }
                storedExpertGates.append(contentsOf: rolloutExpertGates)
                storedStandExpertGates.append(
                    contentsOf: rolloutStandExpertGates)
                storedActorTrainingWeights.append(contentsOf:
                    Self.actorTrainingWeights(
                        expertGates: Array(rolloutExpertGates),
                        standExpertGates: Array(rolloutStandExpertGates),
                        freezesBaseActor: freezesBasePolicyExpert,
                        freezesExpertActor: freezesLowSpeedPolicyExpert))
                let obsArray = MLXArray(Array(normalized)).reshaped([n, obsDim])
                let rolloutGateArray = MLXArray(Array(rolloutExpertGates))
                    .reshaped([n, 1])
                let rolloutStandGateArray = MLXArray(
                    Array(rolloutStandExpertGates)).reshaped([n, 1])
                let out = policy.forward(
                    obsArray, expertGate: rolloutGateArray,
                    standExpertGate: rolloutStandGateArray,
                    freezeBaseActor: freezesBasePolicyExpert)
                let effectiveLogStandardDeviation = clip(
                    out.logStandardDeviation,
                    min: configuration.minimumActionStd.map(log) ?? -5,
                    max: configuration.maximumActionStd.map(log) ?? 1)
                // An explicit update/step key makes action sampling stable
                // across process restarts without depending on global PRNG
                // state consumed during model construction.
                let randomKey = MLXRandom.key(
                    configuration.seed
                        &+ UInt64(update) &* 0x9E3779B97F4A7C15
                        &+ UInt64(rolloutStep) &* 0xD1B54A32D192ED03)
                let preTanh = out.mean
                    + MLXRandom.normal([n, actionDim], key: randomKey)
                        * exp(effectiveLogStandardDeviation)
                let environmentActions = actionDistribution.environmentAction(
                    preTanh)
                let logProb = Self.gaussianLogProbability(
                    preTanh, mean: out.mean,
                    logStandardDeviation: effectiveLogStandardDeviation)
                eval(preTanh, environmentActions, logProb, out.value)
                let actionHost = environmentActions.asArray(Float.self)
                let preTanhHost = preTanh.asArray(Float.self)
                let logProbHost = logProb.asArray(Float.self)
                let valueHost = out.value.asArray(Float.self)
                guard actionHost.allSatisfy(\.isFinite),
                      preTanhHost.allSatisfy(\.isFinite),
                      logProbHost.allSatisfy(\.isFinite),
                      valueHost.allSatisfy(\.isFinite) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "non-finite policy rollout before PPO update \(update + 1): "
                        + "action \(Self.hostSummary(actionHost)), "
                        + "pre_tanh \(Self.hostSummary(preTanhHost)), "
                        + "log_prob \(Self.hostSummary(logProbHost)), "
                        + "value \(Self.hostSummary(valueHost))")
                }
                var trainingPreTanh = ContiguousArray(preTanhHost)
                var trainingLogProb = ContiguousArray(logProbHost)
                var trainingValues = ContiguousArray(valueHost)
                if usesSymmetryDataAugmentation, let policySymmetry {
                    let mirroredPreTanh = policySymmetry.mirrorPolicyActions(
                        trainingPreTanh)
                    trainingPreTanh = Self.alternatingSymmetryRows(
                        original: trainingPreTanh, mirrored: mirroredPreTanh,
                        rowDimension: actionDim, mirroredParity: update & 1)
                    // The transformed transition is evaluated under the same
                    // frozen behavior policy before any PPO update.  This
                    // supplies the exact old likelihood/value for the
                    // augmented sample rather than assuming the still-learning
                    // network is already equivariant.
                    let trainingInput = MLXArray(Array(trainingNormalized))
                        .reshaped([n, obsDim])
                    let trainingOut = policy.forward(
                        trainingInput, expertGate: rolloutGateArray,
                        standExpertGate: rolloutStandGateArray,
                        freezeBaseActor: freezesBasePolicyExpert)
                    let trainingLogStandardDeviation = clip(
                        trainingOut.logStandardDeviation,
                        min: configuration.minimumActionStd.map(log) ?? -5,
                        max: configuration.maximumActionStd.map(log) ?? 1)
                    let trainingPreTanhArray = MLXArray(Array(trainingPreTanh))
                        .reshaped([n, actionDim])
                    let oldTrainingLogProbability = Self.gaussianLogProbability(
                        trainingPreTanhArray, mean: trainingOut.mean,
                        logStandardDeviation: trainingLogStandardDeviation)
                    eval(oldTrainingLogProbability, trainingOut.value)
                    trainingLogProb = ContiguousArray(
                        oldTrainingLogProbability.asArray(Float.self))
                    trainingValues = ContiguousArray(
                        trainingOut.value.asArray(Float.self))
                }
                guard trainingPreTanh.allSatisfy(\.isFinite),
                      trainingLogProb.allSatisfy(\.isFinite),
                      trainingValues.allSatisfy(\.isFinite) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "non-finite symmetry-augmented PPO sample before update "
                        + "\(update + 1)")
                }
                storedActions.append(contentsOf: trainingPreTanh)
                storedLogProb.append(contentsOf: trainingLogProb)
                storedValues.append(contentsOf: trainingValues)
                if usesSymmetryDataAugmentation {
                    let mirroredParity = update & 1
                    for e in 0..<n {
                        storedKLWeights.append(e & 1 == mirroredParity ? 0 : 1)
                    }
                } else {
                    storedKLWeights.append(contentsOf:
                        repeatElement(Float(1), count: n))
                }
                let actionBatch = try RLActionBatch(
                    numEnvironments: n, actionDimension: actionDim,
                    values: ContiguousArray(actionHost))
                try task.step(actions: actionBatch, into: &stepResult)
                for (name, values) in stepResult.metrics
                    where name.hasPrefix("reward/")
                        || name.hasPrefix("penalty/")
                        || name.hasPrefix("gait/") {
                    taskMetricSums[name, default: 0] += values.reduce(0, +)
                    taskMetricCounts[name, default: 0] += values.count
                }

                var adjustedRewards = stepResult.rewards.map {
                    $0 * configuration.resolvedRewardScale
                }
                if stepResult.truncated.contains(true) {
                    var final = stepResult.finalObservations
                    if usesSymmetryDataAugmentation, let policySymmetry {
                        let mirroredFinal = policySymmetry
                            .mirrorPolicyObservations(final)
                        final = Self.alternatingSymmetryRows(
                            original: final, mirrored: mirroredFinal,
                            rowDimension: obsDim, mirroredParity: update & 1)
                    }
                    if configuration.normalizeObservations {
                        final = normalizer.normalize(final)
                    }
                    let finalArray = MLXArray(Array(final)).reshaped([n, obsDim])
                    let finalValues = policy.forward(finalArray).value
                    eval(finalValues)
                    let host = finalValues.asArray(Float.self)
                    for e in 0..<n where stepResult.truncated[e] {
                        adjustedRewards[e] += configuration.gamma * host[e]
                    }
                }
                storedRewards.append(contentsOf: adjustedRewards)
                guard adjustedRewards.allSatisfy(\.isFinite),
                      stepResult.observations.policy.allSatisfy(\.isFinite) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "non-finite task transition before PPO update \(update + 1): "
                        + "reward \(Self.hostSummary(adjustedRewards)), "
                        + "observation \(Self.hostSummary(Array(stepResult.observations.policy)))")
                }
                for e in 0..<n {
                    let done = stepResult.terminated[e] || stepResult.truncated[e]
                    storedDones.append(done)
                    if stepResult.successes[e] && !episodeSucceeded[e] {
                        episodeSucceeded[e] = true
                        if successImitationCoefficient > 0 {
                            Self.markSuccessfulEpisodeSegment(
                                mask: &storedSuccessImitationMask,
                                environment: e, numEnvironments: n,
                                startStep: episodeStartSteps[e],
                                endStep: rolloutStep)
                        }
                    }
                    if done {
                        completedEpisodes += 1
                        if episodeSucceeded[e] { successfulEpisodes += 1 }
                        episodeSucceeded[e] = false
                        episodeStartSteps[e] = rolloutStep + 1
                        if let episodeReturns = stepResult.metrics["episode/return"] {
                            completedReturnSum += episodeReturns[e]
                        }
                        if let episodeLengths = stepResult.metrics["episode/length"] {
                            completedLengthSum += episodeLengths[e]
                        }
                        if let distances = stepResult.metrics["episode/forward_distance_m"] {
                            completedDistanceSum += distances[e]
                        }
                        for (name, values) in stepResult.metrics
                            where name.hasPrefix("episode/") && e < values.count {
                            taskMetricSums[name, default: 0] += values[e]
                            taskMetricCounts[name, default: 0] += 1
                        }
                    }
                }
                observation = stepResult.observations
            }

            var lastRaw = observation.policy
            if usesSymmetryDataAugmentation, let policySymmetry {
                let mirroredLast = policySymmetry.mirrorPolicyObservations(lastRaw)
                lastRaw = Self.alternatingSymmetryRows(
                    original: lastRaw, mirrored: mirroredLast,
                    rowDimension: obsDim, mirroredParity: update & 1)
            }
            let lastNormalized = configuration.normalizeObservations
                ? normalizer.normalize(lastRaw) : lastRaw
            let lastObs = MLXArray(Array(lastNormalized)).reshaped([n, obsDim])
            let lastValueArray = policy.forward(lastObs).value
            eval(lastValueArray)
            let gae = GeneralizedAdvantageEstimator.compute(
                rewards: storedRewards, values: storedValues, dones: storedDones,
                lastValues: lastValueArray.asArray(Float.self), numEnvironments: n,
                horizon: configuration.rolloutSteps, gamma: configuration.gamma,
                lambda: configuration.gaeLambda)
            let advantageNormalization = Self.weightedNormalizedAdvantages(
                gae.advantages, weights: storedActorTrainingWeights)
            let advantageMean = advantageNormalization.mean
            let advantageVariance = advantageNormalization.variance
            let advantageScale = advantageNormalization.scale
            let normalizedAdvantages = advantageNormalization.values
            guard advantageMean.isFinite, advantageVariance.isFinite,
                  advantageScale.isFinite,
                  normalizedAdvantages.allSatisfy(\.isFinite),
                  gae.returns.allSatisfy(\.isFinite) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "non-finite GAE before PPO update \(update + 1): "
                    + "advantage \(Self.hostSummary(gae.advantages)), "
                    + "return \(Self.hostSummary(gae.returns))")
            }

            var order = Array(0..<batchSize)
            var lastPolicyLoss: Float = 0, lastValueLoss: Float = 0
            var lastEntropy: Float = 0, lastKL: Float = 0
            var lastSymmetryLoss: Float = 0
            var lastSuccessImitationLoss: Float = 0
            var lastReferencePolicyLoss: Float = 0
            var klSum: Float = 0
            var klCount = 0
            var shuffleRNG = SplitMix64(
                seed: configuration.seed
                    &+ UInt64(update) &* 0xA24BAED4963EE407
                    &+ 0x8EBC6AF09C88C6E3)
            epochLoop: for epoch in 0..<configuration.updateEpochs {
                Self.shuffle(&order, using: &shuffleRNG)
                for start in stride(from: 0, to: batchSize,
                                    by: configuration.minibatchSize) {
                    let end = min(start + configuration.minibatchSize, batchSize)
                    let indices = order[start..<end]
                    let m = indices.count
                    var mbObs = [Float](); mbObs.reserveCapacity(m * obsDim)
                    var mbMirroredObs = [Float]()
                    mbMirroredObs.reserveCapacity(m * obsDim)
                    var mbActions = [Float](); mbActions.reserveCapacity(m * actionDim)
                    var mbLogProb = [Float](); mbLogProb.reserveCapacity(m)
                    var mbAdvantages = [Float](); mbAdvantages.reserveCapacity(m)
                    var mbReturns = [Float](); mbReturns.reserveCapacity(m)
                    var mbValues = [Float](); mbValues.reserveCapacity(m)
                    var mbKLWeights = [Float](); mbKLWeights.reserveCapacity(m)
                    var mbExpertGates = [Float]();
                    mbExpertGates.reserveCapacity(m)
                    var mbStandExpertGates = [Float]();
                    mbStandExpertGates.reserveCapacity(m)
                    var mbSuccessImitationMask = [Float]()
                    mbSuccessImitationMask.reserveCapacity(m)
                    var mbReferencePolicyWeights = [Float]()
                    mbReferencePolicyWeights.reserveCapacity(m)
                    var mbActorTrainingWeights = [Float]()
                    mbActorTrainingWeights.reserveCapacity(m)
                    for i in indices {
                        mbObs.append(contentsOf:
                            storedObservations[(i * obsDim)..<((i + 1) * obsDim)])
                        mbMirroredObs.append(contentsOf:
                            storedMirroredObservations[
                                (i * obsDim)..<((i + 1) * obsDim)])
                        mbActions.append(contentsOf:
                            storedActions[(i * actionDim)..<((i + 1) * actionDim)])
                        mbLogProb.append(storedLogProb[i])
                        mbAdvantages.append(normalizedAdvantages[i])
                        mbReturns.append(gae.returns[i])
                        mbValues.append(storedValues[i])
                        mbKLWeights.append(storedKLWeights[i])
                        mbExpertGates.append(storedExpertGates[i])
                        mbStandExpertGates.append(storedStandExpertGates[i])
                        mbSuccessImitationMask.append(
                            storedSuccessImitationMask[i])
                        mbReferencePolicyWeights.append(
                            storedReferencePolicyWeights[i])
                        mbActorTrainingWeights.append(
                            storedActorTrainingWeights[i])
                    }
                    let entropyKey = MLXRandom.key(
                        configuration.seed
                            &+ UInt64(update) &* 0xD1B54A32D192ED03
                            &+ UInt64(epoch) &* 0x94D049BB133111EB
                            &+ UInt64(start))
                    let entropyNoise = MLXRandom.normal(
                        [m, actionDim], key: entropyKey)
                    let (losses, gradients) = lossAndGradient(policy, [
                        MLXArray(mbObs).reshaped([m, obsDim]),
                        MLXArray(mbActions).reshaped([m, actionDim]),
                        MLXArray(mbLogProb), MLXArray(mbAdvantages),
                        MLXArray(mbReturns), MLXArray(mbValues), entropyNoise,
                        MLXArray(mbKLWeights),
                        MLXArray(mbMirroredObs).reshaped([m, obsDim]),
                        MLXArray(mbExpertGates).reshaped([m, 1]),
                        MLXArray(mbStandExpertGates).reshaped([m, 1]),
                        MLXArray(mbSuccessImitationMask),
                        MLXArray(mbReferencePolicyWeights),
                        MLXArray(mbActorTrainingWeights),
                    ])
                    let (clippedGradients, squaredGradientNorm) =
                        Self.clipGradientNorm(
                            gradients, maximum: configuration.maxGradientNorm)
                    // Evaluate the loss and norm before mutating parameters.
                    // If a future task or model produces invalid arithmetic,
                    // fail while the last atomic checkpoint is still valid
                    // instead of writing a silently corrupted policy.
                    eval(losses + [squaredGradientNorm])
                    let scalarLosses = losses.map { $0.item(Float.self) }
                    let normSquared = squaredGradientNorm.item(Float.self)
                    guard scalarLosses.allSatisfy(\.isFinite),
                          normSquared.isFinite else {
                        let gradientSummary = Self.gradientSummary(gradients)
                        throw RLEnvironmentError.invalidConfiguration(
                            "non-finite PPO minibatch at update \(update + 1), "
                            + "epoch \(epoch + 1), rows \(start)..<\(end): "
                            + "losses \(scalarLosses), grad_norm2 \(normSquared), "
                            + "gradients \(gradientSummary), "
                            + "obs \(Self.hostSummary(mbObs)), "
                            + "actions \(Self.hostSummary(mbActions)), "
                            + "old_log_prob \(Self.hostSummary(mbLogProb)), "
                            + "advantages \(Self.hostSummary(mbAdvantages)), "
                            + "returns \(Self.hostSummary(mbReturns)); "
                            + "last checkpoint remains valid")
                    }
                    try optimizer.update(model: policy, gradients: clippedGradients)
                    eval(policy)
                    optimizer.evaluate()
                    lastPolicyLoss = scalarLosses[1]
                    lastValueLoss = scalarLosses[2]
                    lastEntropy = scalarLosses[3]
                    lastKL = scalarLosses[4]
                    lastSymmetryLoss = scalarLosses[5]
                    lastSuccessImitationLoss = scalarLosses[6]
                    lastReferencePolicyLoss = scalarLosses[7]
                    klSum += lastKL
                    klCount += 1
                    // Adaptive scheduling handles ordinary KL drift; stop the
                    // epoch only on a genuine trust-region overshoot.
                    if configuration.targetKL > 0
                        && lastKL > 4 * configuration.targetKL {
                        break epochLoop
                    }
                }
            }
            let meanKL = klCount > 0 ? klSum / Float(klCount) : 0
            if configuration.targetKL > 0 {
                if meanKL > 2 * configuration.targetKL {
                    // Keep the floor relative to the requested rate.  A
                    // hard 1e-5 floor used to *increase* carefully calibrated
                    // transfer runs whose initial rate was below 1e-5 after
                    // a KL overshoot—the opposite of a trust-region response.
                    adaptiveLearningRate = max(
                        configuration.learningRate / 100,
                        adaptiveLearningRate / 1.5)
                } else if meanKL < 0.5 * configuration.targetKL {
                    adaptiveLearningRate = min(configuration.learningRate,
                                               adaptiveLearningRate * 1.5)
                }
            }

            totalSteps += batchSize
            let elapsed = max(Float(-startTime.timeIntervalSinceNow), 1e-6)
            let returnMean = gae.returns.reduce(0, +) / Float(batchSize)
            let returnVariance = gae.returns.reduce(Float(0)) {
                $0 + ($1 - returnMean) * ($1 - returnMean)
            }
            let residual = zip(gae.returns, storedValues).reduce(Float(0)) {
                $0 + ($1.0 - $1.1) * ($1.0 - $1.1)
            }
            let explainedVariance = returnVariance > 1e-8
                ? 1 - residual / returnVariance : 0
            var meanTaskMetrics = Dictionary(uniqueKeysWithValues:
                taskMetricSums.map { name, sum in
                    let count = taskMetricCounts[name] ?? 0
                    return (name, count > 0 ? sum / Float(count) : 0)
                })
            meanTaskMetrics["training/success_imitation_loss"] =
                lastSuccessImitationLoss
            meanTaskMetrics["training/success_imitation_fraction"] =
                storedSuccessImitationMask.reduce(0, +) / Float(batchSize)
            meanTaskMetrics["training/reference_policy_loss"] =
                lastReferencePolicyLoss
            meanTaskMetrics["training/reference_policy_fraction"] =
                storedReferencePolicyWeights.reduce(0, +) / Float(batchSize)
            meanTaskMetrics["training/actor_training_fraction"] =
                storedActorTrainingWeights.reduce(0, +) / Float(batchSize)
            let metrics = PPOUpdateMetrics(
                update: update, environmentSteps: totalSteps,
                policyLoss: lastPolicyLoss,
                valueLoss: lastValueLoss, learningRate: optimizer.learningRate,
                entropy: lastEntropy,
                symmetryLoss: lastSymmetryLoss,
                approximateKL: meanKL, explainedVariance: explainedVariance,
                stepsPerSecond: Float(batchSize) / elapsed,
                completedEpisodes: completedEpisodes,
                successRate: completedEpisodes > 0
                    ? Float(successfulEpisodes) / Float(completedEpisodes) : 0,
                meanEpisodeReturn: completedEpisodes > 0
                    ? completedReturnSum / Float(completedEpisodes) : 0,
                meanEpisodeLength: completedEpisodes > 0
                    ? completedLengthSum / Float(completedEpisodes) : 0,
                meanEpisodeForwardDistance: completedEpisodes > 0
                    ? completedDistanceSum / Float(completedEpisodes) : 0,
                meanActionStandardDeviation: {
                    let value = mean(exp(clip(policy.logStandardDeviation,
                        min: configuration.minimumActionStd.map(log) ?? -5,
                        max: configuration.maximumActionStd.map(log) ?? 1)))
                    eval(value)
                    return value.item(Float.self)
                }(), taskMetrics: meanTaskMetrics)
            var metricData = try JSONEncoder().encode(metrics)
            metricData.append(0x0A)
            try metricsFile.write(contentsOf: metricData)
            if let onUpdate { onUpdate(metrics) } else { Self.print(metrics) }

            if (localUpdate + 1).isMultiple(of: configuration.checkpointInterval)
                || localUpdate == configuration.updates - 1 {
                let trainingState = VectorPPOTrainingState(
                    completedUpdates: update + 1, environmentSteps: totalSteps,
                    optimizerSteps: optimizer.step,
                    adaptiveLearningRate: adaptiveLearningRate)
                try Self.save(policy: policy, normalizer: normalizer,
                              optimizer: optimizer,
                              referencePolicy: referencePolicy,
                              taskSpec: spec,
                              configuration: persistedConfiguration,
                              trainingState: trainingState,
                              outputDirectory: outputDirectory)
                // Preserve the policy trajectory for deterministic selection.
                // A noisy on-policy update can regress after a good checkpoint;
                // overwriting the only weights makes that impossible to audit.
                let snapshotDirectory = String(
                    format: "%@/checkpoints/update-%06d",
                    outputDirectory, update + 1)
                try Self.save(policy: policy, normalizer: normalizer,
                              optimizer: optimizer,
                              referencePolicy: referencePolicy,
                              taskSpec: spec,
                              configuration: persistedConfiguration,
                              trainingState: trainingState,
                              outputDirectory: snapshotDirectory)
            }
        }
    }

    /// Deterministic mean-action evaluation using the exact observation
    /// statistics and network dimensions stored with the checkpoint.
    public static func evaluate(task: any VectorizedRLTask,
                                checkpointDirectory: String,
                                episodes requestedEpisodes: Int,
                                seed: UInt64 = 10_001,
                                allowTaskConfigurationTransfer: Bool = false) throws
        -> PPOEvaluationMetrics {
        guard requestedEpisodes > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation episode count must be positive")
        }
        let metadataURL = URL(fileURLWithPath: "\(checkpointDirectory)/metadata.json")
        let metadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self, from: Data(contentsOf: metadataURL))
        guard VectorActorCritic.compatibleArchitectureVersions.contains(
            metadata.architectureVersion ?? 1) else {
            throw RLEnvironmentError.invalidConfiguration(
                "legacy policy architecture is not evaluable; train a fresh checkpoint")
        }
        let trainingStateURL = URL(
            fileURLWithPath: "\(checkpointDirectory)/training-state.json")
        let trainingState = try JSONDecoder().decode(
            VectorPPOTrainingState.self, from: Data(contentsOf: trainingStateURL))
        let checkpointFingerprint = try Self.checkpointFingerprint(
            directory: checkpointDirectory)
        (task as? any TrainingModeConfigurable)?.setTrainingMode(false)
        let spec = task.spec
        guard !spec.autoReset else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation requires a task created with autoReset=false so "
                + "each environment contributes a fixed episode quota")
        }
        let taskConfigurationMatches = metadata.taskConfiguration.map {
            $0 == spec.configurationValues
        } ?? true
        guard metadata.task == spec.id,
              (metadata.taskRevision ?? 1) == spec.revision,
              (taskConfigurationMatches || allowTaskConfigurationTransfer),
              metadata.observationDimension == spec.observation.elementCount,
              metadata.actionDimension == spec.action.elementCount,
              metadata.simulationStep == spec.simulationStep,
              metadata.controlDecimation == spec.controlDecimation,
              metadata.maxEpisodeSteps == spec.maxEpisodeSteps else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint/task mismatch: checkpoint \(metadata.task) "
                + "[\(metadata.observationDimension), \(metadata.actionDimension)], "
                + "task \(spec.id) [\(spec.observation.elementCount), "
                + "\(spec.action.elementCount)]")
        }
        let policy = VectorActorCritic(
            observationDimension: metadata.observationDimension,
            actionDimension: metadata.actionDimension,
            hiddenSize: metadata.ppo.hiddenSize,
            hiddenDimensions: metadata.ppo.hiddenDimensions,
            initialActionStd: metadata.ppo.initialActionStd)
        let sourceWeights = try loadArrays(
            url: URL(fileURLWithPath: "\(checkpointDirectory)/policy.safetensors"))
        let weights = try VectorActorCritic.compatibleWeights(
            sourceWeights, architectureVersion: metadata.architectureVersion)
        try policy.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(policy)
        let normalizer = RunningObservationNormalizer(snapshot: metadata.normalizer)
        let policyExpertGate = task as? any PolicyExpertGateProviding
        let usesPolicyExpertGate = policyExpertGate?.usesPolicyExpertGate == true
        let policyStandExpertGate = task as? any PolicyStandExpertGateProviding
        let usesPolicyStandExpertGate =
            policyStandExpertGate?.usesPolicyStandExpertGate == true
        var observation = try task.reset(seed: seed)
        var stepResult = RLStepBatch(spec: spec)
        var runningReturns = [Float](repeating: 0, count: spec.numEnvironments)
        var runningLengths = [Int](repeating: 0, count: spec.numEnvironments)
        var runningSuccessOnce = [Bool](
            repeating: false, count: spec.numEnvironments)
        var completedReturns = [Float](), completedLengths = [Int]()
        var successes = 0
        var taskMetricSums = [String: Float]()
        var taskMetricCounts = [String: Int]()
        let episodeQuotas = Self.evaluationEpisodeQuotas(
            requestedEpisodes: requestedEpisodes,
            numEnvironments: spec.numEnvironments)
        var completedPerEnvironment = [Int](
            repeating: 0, count: spec.numEnvironments)
        var active = episodeQuotas.map { $0 > 0 }
        let episodeWaves = episodeQuotas.max() ?? 0
        let maximumControlSteps = max(episodeWaves, 1)
            * spec.maxEpisodeSteps * 4
        var controlSteps = 0
        while completedReturns.count < requestedEpisodes
                && controlSteps < maximumControlSteps {
            let normalized = metadata.ppo.normalizeObservations
                ? normalizer.normalize(observation.policy)
                : observation.policy
            let input = MLXArray(Array(normalized)).reshaped(
                [spec.numEnvironments, spec.observation.elementCount])
            let expertGate = usesPolicyExpertGate
                ? MLXArray(Array(policyExpertGate!.policyExpertGates(
                    observation.policy))).reshaped([spec.numEnvironments, 1])
                : nil
            let standExpertGate = usesPolicyStandExpertGate
                ? MLXArray(Array(policyStandExpertGate!.policyStandExpertGates(
                    observation.policy))).reshaped([spec.numEnvironments, 1])
                : nil
            let action = metadata.ppo.resolvedActionDistribution
                .environmentAction(policy.forward(
                    input, expertGate: expertGate,
                    standExpertGate: standExpertGate,
                    freezeBaseActor: usesPolicyExpertGate
                        && policyExpertGate!.freezesBasePolicyExpert).mean)
            eval(action)
            var actionValues = ContiguousArray(action.asArray(Float.self))
            for e in 0..<spec.numEnvironments where !active[e] {
                let base = e * spec.action.elementCount
                for j in 0..<spec.action.elementCount {
                    actionValues[base + j] = 0
                }
            }
            let actionBatch = try RLActionBatch(
                numEnvironments: spec.numEnvironments,
                actionDimension: spec.action.elementCount,
                values: actionValues)
            try task.step(actions: actionBatch, into: &stepResult)
            var resetGroups = [Int: [Int]]()
            for e in 0..<spec.numEnvironments {
                guard active[e] else { continue }
                runningReturns[e] += stepResult.rewards[e]
                runningLengths[e] += 1
                runningSuccessOnce[e] = runningSuccessOnce[e]
                    || stepResult.successes[e]
                if stepResult.terminated[e] || stepResult.truncated[e] {
                    completedReturns.append(runningReturns[e])
                    completedLengths.append(runningLengths[e])
                    if runningSuccessOnce[e] { successes += 1 }
                    for (name, values) in stepResult.metrics
                        where name.hasPrefix("episode/")
                            && name != "episode/return" && name != "episode/length" {
                        taskMetricSums[name, default: 0] += values[e]
                        taskMetricCounts[name, default: 0] += 1
                    }
                    runningReturns[e] = 0
                    runningLengths[e] = 0
                    runningSuccessOnce[e] = false
                    completedPerEnvironment[e] += 1
                    if completedPerEnvironment[e] < episodeQuotas[e] {
                        resetGroups[completedPerEnvironment[e], default: []].append(e)
                    } else {
                        active[e] = false
                    }
                }
            }
            observation = stepResult.observations
            // Reset by episode index, rather than completion order. Thus
            // environment e always receives the same initial state for wave
            // k even if GPU contact scheduling changes which peer finishes
            // first. Environments that have met their quota stay inactive and
            // cannot over-sample short failures while other episodes survive.
            for episodeIndex in resetGroups.keys.sorted() {
                try task.reset(
                    environments: resetGroups[episodeIndex],
                    seed: Self.evaluationEpisodeSeed(
                        base: seed, episodeIndex: episodeIndex),
                    into: &observation)
            }
            controlSteps += 1
        }
        guard completedReturns.count == requestedEpisodes else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation did not finish \(requestedEpisodes) episodes within safety limit")
        }
        var averagedTaskMetrics = Dictionary(uniqueKeysWithValues: taskMetricSums.map {
            name, value in
            (name, value / Float(max(taskMetricCounts[name] ?? 1, 1)))
        })
        // Tasks can publish mutually exclusive episode cohorts as
        // `episode/<cohort>_bin` plus `episode/<cohort>_success`. Derive the
        // conditional rate centrally so nominal and robustness performance
        // are not diluted by the other cohort's zero entries.
        for (name, binFraction) in Array(averagedTaskMetrics)
            where name.hasPrefix("episode/") && name.hasSuffix("_bin")
                && binFraction > 0 {
            let stem = String(name.dropLast(4))
            let successName = "\(stem)_success"
            if let successFraction = averagedTaskMetrics[successName] {
                averagedTaskMetrics["\(stem)_success_rate"] =
                    successFraction / binFraction
            }
        }
        let meanEpisodeLength = Float(completedLengths.reduce(0, +))
            / Float(requestedEpisodes)
        let successRate = Float(successes) / Float(requestedEpisodes)
        let acceptance: PPOEvaluationAcceptance?
        if let provider = task as? any RLEvaluationCriteriaProviding {
            let failures = provider.evaluationCriteria.failures(
                successRate: successRate,
                meanEpisodeLength: meanEpisodeLength,
                maxEpisodeSteps: spec.maxEpisodeSteps,
                taskMetrics: averagedTaskMetrics)
            acceptance = PPOEvaluationAcceptance(
                passed: failures.isEmpty, failures: failures)
        } else {
            acceptance = nil
        }
        return PPOEvaluationMetrics(
            provenanceVersion: 3,
            task: spec.id, taskRevision: spec.revision,
            checkpointTaskConfiguration: metadata.taskConfiguration,
            evaluationTaskConfiguration: spec.configurationValues,
            taskConfigurationTransferred: !taskConfigurationMatches,
            checkpointDirectory: checkpointDirectory,
            checkpointFingerprint: checkpointFingerprint,
            initializationCheckpoint: metadata.ppo.initializationCheckpoint,
            trainingSeed: metadata.ppo.seed, evaluationSeed: seed,
            evaluationEnvironments: spec.numEnvironments,
            trainingUpdates: trainingState.completedUpdates,
            trainingEnvironmentSteps: trainingState.environmentSteps,
            episodes: requestedEpisodes, successes: successes,
            successRate: successRate,
            meanReturn: completedReturns.reduce(0, +) / Float(requestedEpisodes),
            meanEpisodeLength: meanEpisodeLength,
            taskMetrics: averagedTaskMetrics,
            acceptance: acceptance)
    }

    /// Content identity for every file that changes deterministic evaluation
    /// semantics. Optimizer moments are intentionally excluded because they
    /// cannot affect policy replay.
    public static func checkpointFingerprint(directory: String) throws -> String {
        var hasher = SHA256()
        for name in ["metadata.json", "policy.safetensors", "training-state.json"] {
            let url = URL(fileURLWithPath: directory).appendingPathComponent(name)
            let data = try Data(contentsOf: url)
            hasher.update(data: Data(name.utf8))
            hasher.update(data: Data([0]))
            hasher.update(data: data)
            hasher.update(data: Data([0]))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Distribute an exact episode request across environment identities. The
    /// first remainder environments receive one additional episode, making
    /// the assignment independent of which simulations terminate first.
    static func evaluationEpisodeQuotas(requestedEpisodes: Int,
                                        numEnvironments: Int) -> [Int] {
        precondition(requestedEpisodes > 0 && numEnvironments > 0)
        let completeWaves = requestedEpisodes / numEnvironments
        let remainder = requestedEpisodes % numEnvironments
        return (0..<numEnvironments).map {
            completeWaves + ($0 < remainder ? 1 : 0)
        }
    }

    static func evaluationEpisodeSeed(base: UInt64,
                                      episodeIndex: Int) -> UInt64 {
        precondition(episodeIndex >= 0)
        return base &+ UInt64(episodeIndex) &* 0xD1B54A32D192ED03
    }

    public static func gaussianLogProbability(_ action: MLXArray, mean: MLXArray,
                                              logStandardDeviation: MLXArray)
        -> MLXArray {
        let z = (action - mean) / exp(logStandardDeviation)
        return sum(-0.5 * z.square() - logStandardDeviation - logSqrt2Pi,
                   axis: -1)
    }

    /// Numerically stable importance ratio used by PPO's clipped surrogate.
    /// Internal visibility intentionally permits a focused regression test.
    static func stableImportanceRatio(newLogProbability: MLXArray,
                                      oldLogProbability: MLXArray)
        -> (ratio: MLXArray, logRatio: MLXArray) {
        let logRatio = clip(newLogProbability - oldLogProbability,
                            min: -maximumImportanceLogRatio,
                            max: maximumImportanceLogRatio)
        return (exp(logRatio), logRatio)
    }

    /// Scalar mirror of the array operation for platform-independent unit
    /// testing (SwiftPM test bundles do not embed MLX's default metallib).
    static func stableImportanceRatio(logProbabilityDifference: Float)
        -> (ratio: Float, logRatio: Float) {
        let logRatio = min(max(logProbabilityDifference,
                               -maximumImportanceLogRatio),
                           maximumImportanceLogRatio)
        return (exp(logRatio), logRatio)
    }

    /// Host mirror of the KL estimator used to verify that augmentation-only
    /// rows cannot steer the adaptive trust-region schedule.
    static func weightedApproximateKL(
        logProbabilityDifferences: [Float], weights: [Float]
    ) -> Float {
        precondition(logProbabilityDifferences.count == weights.count)
        var weightedSum: Float = 0
        var weightSum: Float = 0
        for (difference, weight) in zip(logProbabilityDifferences, weights) {
            precondition(weight >= 0 && weight.isFinite)
            let stable = stableImportanceRatio(
                logProbabilityDifference: difference)
            weightedSum += weight
                * ((stable.ratio - 1) - stable.logRatio)
            weightSum += weight
        }
        return weightSum > 0 ? weightedSum / weightSum : 0
    }

    private static func clipGradientNorm(_ gradients: ModuleParameters,
                                         maximum: Float)
        -> (gradients: ModuleParameters, squaredNorm: MLXArray) {
        let flat = gradients.flattened()
        var squaredNorm = MLXArray(Float(0))
        for (_, gradient) in flat { squaredNorm = squaredNorm + gradient.square().sum() }
        let scale = minimum(MLXArray(maximum) / (sqrt(squaredNorm) + 1e-6),
                            MLXArray(Float(1)))
        return (ModuleParameters.unflattened(flat.map { ($0.0, $0.1 * scale) }),
                squaredNorm)
    }

    /// Compact diagnostics used only on an invalid training path. Keeping the
    /// full failing tensors out of logs makes long unattended runs auditable
    /// without producing multi-megabyte error messages.
    private static func hostSummary(_ values: [Float]) -> String {
        var finiteCount = 0
        var minimumValue = Float.infinity
        var maximumValue = -Float.infinity
        var maximumAbsolute: Float = 0
        for value in values where value.isFinite {
            finiteCount += 1
            minimumValue = min(minimumValue, value)
            maximumValue = max(maximumValue, value)
            maximumAbsolute = max(maximumAbsolute, abs(value))
        }
        if finiteCount == 0 {
            return "count=\(values.count), finite=0"
        }
        return "count=\(values.count), finite=\(finiteCount), "
            + "range=[\(minimumValue),\(maximumValue)], max_abs=\(maximumAbsolute)"
    }

    private static func gradientSummary(_ gradients: ModuleParameters) -> String {
        var invalid = [String]()
        var largestName = "none"
        var largestMagnitude: Float = 0
        for (name, gradient) in gradients.flattened() {
            eval(gradient)
            let values = gradient.asArray(Float.self)
            if values.contains(where: { !$0.isFinite }) {
                invalid.append(name)
            }
            if let localMaximum = values.lazy.filter(\.isFinite).map(abs).max(),
               localMaximum > largestMagnitude {
                largestMagnitude = localMaximum
                largestName = name
            }
        }
        let invalidDescription = invalid.isEmpty
            ? "none" : invalid.prefix(8).joined(separator: ",")
        return "invalid=[\(invalidDescription)], "
            + "largest=\(largestName):\(largestMagnitude)"
    }

    /// Replace alternating complete rows with their exact symmetry partner.
    /// The parity flips between PPO updates, so even a one-environment task
    /// observes both orientations over training while each GAE rollout keeps
    /// a consistent state representation.
    private static func alternatingSymmetryRows(
        original: ContiguousArray<Float>, mirrored: ContiguousArray<Float>,
        rowDimension: Int, mirroredParity: Int
    ) -> ContiguousArray<Float> {
        precondition(rowDimension > 0)
        precondition(original.count == mirrored.count)
        precondition(original.count.isMultiple(of: rowDimension))
        precondition(mirroredParity == 0 || mirroredParity == 1)
        var augmented = original
        let rows = original.count / rowDimension
        for row in 0..<rows where row & 1 == mirroredParity {
            let base = row * rowDimension
            for column in 0..<rowDimension {
                augmented[base + column] = mirrored[base + column]
            }
        }
        return augmented
    }

    private static func shuffle(_ values: inout [Int], using rng: inout SplitMix64) {
        guard values.count > 1 else { return }
        for i in stride(from: values.count - 1, through: 1, by: -1) {
            let j = Int(rng.next() % UInt64(i + 1))
            values.swapAt(i, j)
        }
    }

    private static func save(policy: VectorActorCritic,
                             normalizer: RunningObservationNormalizer,
                             optimizer: CheckpointableAdam,
                             referencePolicy: VectorActorCritic?,
                             taskSpec: RLTaskSpec, configuration: VectorPPOConfig,
                             trainingState: VectorPPOTrainingState,
                             outputDirectory: String) throws {
        try FileManager.default.createDirectory(atPath: outputDirectory,
                                                withIntermediateDirectories: true)
        let weights = Dictionary(uniqueKeysWithValues:
            policy.parameters().flattened().map { ($0.0, $0.1) })
        let weightData = try MLX.saveToData(arrays: weights)
        try weightData.write(to: URL(
            fileURLWithPath: "\(outputDirectory)/policy.safetensors"),
            options: .atomic)
        let optimizerData = try MLX.saveToData(arrays: optimizer.checkpointArrays())
        try optimizerData.write(to: URL(
            fileURLWithPath: "\(outputDirectory)/optimizer.safetensors"),
            options: .atomic)
        if let referencePolicy {
            let referenceWeights = Dictionary(uniqueKeysWithValues:
                referencePolicy.parameters().flattened().map { ($0.0, $0.1) })
            let referenceData = try MLX.saveToData(arrays: referenceWeights)
            try referenceData.write(to: URL(fileURLWithPath:
                "\(outputDirectory)/reference-policy.safetensors"),
                options: .atomic)
        }
        let metadata = VectorPolicyMetadata(
            architectureVersion: VectorActorCritic.architectureVersion,
            task: taskSpec.id,
            taskRevision: taskSpec.revision,
            taskConfiguration: taskSpec.configurationValues,
            observationDimension: taskSpec.observation.elementCount,
            actionDimension: taskSpec.action.elementCount,
            simulationStep: taskSpec.simulationStep,
            controlDecimation: taskSpec.controlDecimation,
            maxEpisodeSteps: taskSpec.maxEpisodeSteps,
            ppo: configuration,
            normalizer: normalizer.snapshot)
        let data = try JSONEncoder().encode(metadata)
        try data.write(to: URL(fileURLWithPath: "\(outputDirectory)/metadata.json"),
                       options: .atomic)
        let stateData = try JSONEncoder().encode(trainingState)
        try stateData.write(to: URL(
            fileURLWithPath: "\(outputDirectory)/training-state.json"),
            options: .atomic)
    }

    private static func print(_ metrics: PPOUpdateMetrics) {
        Swift.print(String(format:
            "ppo %4d  steps %9d  %.0f step/s  pg %+.4f  vf %.4f  "
            + "sym %.4f  kl %.4f  lr %.2e  std %.3f  ev %+.2f  eps %d  ret %+.1f  "
            + "len %.1f  dense %.4f/%.0f%%  dx %+.3f  alt %.2f  "
            + "support %.2f  success %.1f%%",
            metrics.update + 1, metrics.environmentSteps, metrics.stepsPerSecond,
            metrics.policyLoss, metrics.valueLoss, metrics.symmetryLoss,
            metrics.approximateKL,
            metrics.learningRate, metrics.meanActionStandardDeviation,
            metrics.explainedVariance,
            metrics.completedEpisodes,
            metrics.meanEpisodeReturn, metrics.meanEpisodeLength,
            metrics.taskMetrics["reward/running"] ?? 0,
            (metrics.taskMetrics["reward/running_positive_fraction"] ?? 0) * 100,
            metrics.meanEpisodeForwardDistance,
            metrics.taskMetrics["episode/alternating_steps"] ?? 0,
            metrics.taskMetrics["episode/single_support_fraction"] ?? 0,
            metrics.successRate * 100))
    }
}

/// Small algorithm registry used by experiment entry points. The protocol is
/// deliberately task-agnostic: adding SAC, Dreamer, AMP, or distillation does
/// not change any scene code.
public protocol VectorRLAlgorithm: AnyObject {
    var id: String { get }
    func train(task: any VectorizedRLTask, outputDirectory: String) throws
}

extension VectorPPOTrainer: VectorRLAlgorithm {
    public var id: String { "ppo" }
}

public final class VectorRLAlgorithmRegistry {
    public typealias Factory = () -> any VectorRLAlgorithm
    private var factories: [String: Factory] = [:]

    public init() {}

    public func register(_ id: String, factory: @escaping Factory) {
        precondition(factories[id] == nil, "algorithm already registered")
        factories[id] = factory
    }

    public var algorithmIDs: [String] { factories.keys.sorted() }

    public func make(_ id: String) throws -> any VectorRLAlgorithm {
        guard let factory = factories[id] else {
            throw RLEnvironmentError.invalidConfiguration(
                "unknown algorithm '\(id)'; available: \(algorithmIDs.joined(separator: ", "))")
        }
        return factory()
    }

    public static let builtIn: VectorRLAlgorithmRegistry = {
        let registry = VectorRLAlgorithmRegistry()
        registry.register("ppo") { VectorPPOTrainer() }
        return registry
    }()
}
