import Foundation
import CryptoKit
import Darwin
import MLX
import MLXNN
import MLXRandom
import AVBDCore

// Darwin's Swift module exposes the `struct flock` name but not the colliding
// BSD `flock(2)` function. Bind that stable POSIX symbol explicitly.
@_silgen_name("flock")
private func avbdPOSIXFlock(_ descriptor: Int32, _ operation: Int32) -> Int32

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

public enum PPOActivation: String, Codable, Sendable {
    case elu
    case tanh
}

/// How PPO responds when a minibatch leaves the requested KL trust region.
/// Different reference implementations attach materially different semantics
/// to the same `target_kl` flag, so this must be explicit experiment state.
public enum PPOKLSchedule: String, Codable, Sendable {
    /// RSL-RL style adaptive learning rate with a large emergency stop.
    case adaptive
    /// CleanRL/ManiSkill style: keep the configured rate and stop the update
    /// as soon as one minibatch exceeds the target.
    case earlyStop = "early-stop"
    /// Record KL without changing the optimizer or number of epochs.
    case none
}

public struct VectorPPOConfig: Codable, Sendable {
    public var updates: Int
    public var rolloutSteps: Int
    public var updateEpochs: Int
    public var minibatchSize: Int
    public var learningRate: Float
    /// Adam denominator epsilon. Optional so historical checkpoints retain
    /// the former 1e-8 default; ManiSkill's CleanRL baseline uses 1e-5.
    public var optimizerEpsilon: Float?
    public var gamma: Float
    public var gaeLambda: Float
    /// Multiplier applied to task rewards before value targets and GAE. Nil
    /// preserves the historical scale of one for checkpoint compatibility.
    /// This is optimizer conditioning only; task metrics and evaluation
    /// rewards remain in the environment's native units.
    public var rewardScale: Float?
    public var policyClip: Float
    public var valueClip: Float
    /// Nil preserves historical clipped value loss. ManiSkill's published
    /// PPO baseline uses the ordinary, unclipped squared value error.
    public var clipValueLoss: Bool?
    public var valueCoefficient: Float
    public var entropyCoefficient: Float
    public var maxGradientNorm: Float
    public var targetKL: Float
    /// Optional for checkpoint compatibility; nil is the historical adaptive
    /// schedule. Reference presets set their implementation's exact behavior.
    public var klSchedule: PPOKLSchedule?
    public var hiddenSize: Int
    /// Exact three-layer actor/critic widths. Nil preserves AVBD's historical
    /// `[hiddenSize, max(hiddenSize/2,64), max(hiddenSize/4,64)]` topology.
    public var hiddenDimensions: [Int]?
    /// Optional so historical checkpoints decode as AVBD's ELU network.
    public var activation: PPOActivation?
    /// Reproduce CleanRL/ManiSkill's orthogonal layer initialization when
    /// requested. Nil/false preserves MLXNN defaults for old experiments.
    public var orthogonalInitialization: Bool?
    /// Gain for the orthogonally initialized actor output. ManiSkill uses
    /// 0.01 * sqrt(2); nil resolves to that value when orthogonal init is on.
    public var actorOutputGain: Float?
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
    /// Optional final exploration standard deviation for a checkpointed,
    /// deterministic curriculum. When all three annealing fields are set, PPO
    /// uses `initialActionStd` through `actionStdAnnealStartUpdate`, then
    /// interpolates exponentially to this value at
    /// `actionStdAnnealEndUpdate` and holds it there. The same effective value
    /// is used by rollout sampling, likelihoods, KL, and reported metrics.
    public var finalActionStd: Float?
    public var actionStdAnnealStartUpdate: Int?
    public var actionStdAnnealEndUpdate: Int?
    /// Versioned rollout semantics for routed/frozen actors. Version 2 masks
    /// exploration out of dimensions controlled exclusively by frozen
    /// branches. Version 3 also scales each dimension's Gaussian by its
    /// trainable routed contribution and uses that exact standard deviation
    /// in sampling, likelihoods, entropy, and KL. Version 4 excludes exactly
    /// frozen (Dirac) action dimensions from joint likelihoods, entropy, and
    /// KL so a shared learned standard deviation cannot create enormous PPO
    /// ratios through actuators whose policy mean and noise are both frozen.
    /// Optional so historical
    /// metadata decodes as version 1 and is rejected for exact resume rather
    /// than silently changing trajectory.
    public var routedExplorationMaskVersion: Int?
    /// Versioned task-curriculum clock semantics. Version 2 advances every
    /// `TrainingModeConfigurable` task after each completed rollout batch;
    /// historical version 1 only initialized progress at process start.
    public var trainingProgressUpdateVersion: Int?
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
    /// Optional checkpoint supplying its already-routed policy-expert branch
    /// as the destination routed branch. Unlike
    /// `policyExpertInitializationCheckpoint`, this preserves a specialist
    /// that was trained inside a multi-expert policy instead of copying that
    /// checkpoint's base actor.
    public var policyExpertBranchInitializationCheckpoint: String?
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
    /// Optional bounded replay of successful actor transitions across PPO
    /// updates. Zero/nil preserves rollout-local self-imitation. Replay stores
    /// only observations, sampled actions, and generic policy-routing gates.
    public var successReplayCapacity: Int?
    /// Number of replay rows sampled per PPO minibatch. Must be positive when
    /// replay is enabled and no larger than `successReplayCapacity`.
    public var successReplayBatchSize: Int?
    /// Maximum recent causal window retained when a success or task-certified
    /// imitation milestone occurs. Nil preserves full-episode behavior within
    /// the current rollout; a positive bound prevents long settle/approach
    /// prefixes from overwhelming the rare actions that produced the milestone.
    public var successImitationHistorySteps: Int?
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
                learningRate: Float = 3e-4, optimizerEpsilon: Float? = nil,
                gamma: Float = 0.99,
                gaeLambda: Float = 0.95, rewardScale: Float? = nil,
                policyClip: Float = 0.2,
                valueClip: Float = 0.2, clipValueLoss: Bool? = true,
                valueCoefficient: Float = 1,
                entropyCoefficient: Float = 0.01, maxGradientNorm: Float = 1,
                targetKL: Float = 0.01, klSchedule: PPOKLSchedule? = nil,
                hiddenSize: Int = 512,
                hiddenDimensions: [Int]? = nil,
                activation: PPOActivation? = nil,
                orthogonalInitialization: Bool? = nil,
                actorOutputGain: Float? = nil,
                actionDistribution: PPOActionDistribution? = nil,
                initialActionStd: Float = 1.0,
                minimumActionStd: Float? = nil,
                maximumActionStd: Float? = nil,
                finalActionStd: Float? = nil,
                actionStdAnnealStartUpdate: Int? = nil,
                actionStdAnnealEndUpdate: Int? = nil,
                routedExplorationMaskVersion: Int? = 4,
                trainingProgressUpdateVersion: Int? = 2,
                initializationCheckpoint: String? = nil,
                policyExpertInitializationCheckpoint: String? = nil,
                policyExpertBranchInitializationCheckpoint: String? = nil,
                standExpertInitializationCheckpoint: String? = nil,
                initializationNormalizerPriorCount: Double? = nil,
                useTaskSymmetryAugmentation: Bool? = true,
                symmetryMirrorLossCoefficient: Float? = 0.01,
                successImitationCoefficient: Float? = nil,
                successReplayCapacity: Int? = nil,
                successReplayBatchSize: Int? = nil,
                successImitationHistorySteps: Int? = nil,
                referencePolicyCoefficient: Float? = nil,
                normalizeObservations: Bool = true,
                updateObservationNormalizer: Bool? = true,
                checkpointInterval: Int = 50, seed: UInt64 = 1) {
        self.updates = updates
        self.rolloutSteps = rolloutSteps
        self.updateEpochs = updateEpochs
        self.minibatchSize = minibatchSize
        self.learningRate = learningRate
        self.optimizerEpsilon = optimizerEpsilon
        self.gamma = gamma
        self.gaeLambda = gaeLambda
        self.rewardScale = rewardScale
        self.policyClip = policyClip
        self.valueClip = valueClip
        self.clipValueLoss = clipValueLoss
        self.valueCoefficient = valueCoefficient
        self.entropyCoefficient = entropyCoefficient
        self.maxGradientNorm = maxGradientNorm
        self.targetKL = targetKL
        self.klSchedule = klSchedule
        self.hiddenSize = hiddenSize
        self.hiddenDimensions = hiddenDimensions
        self.activation = activation
        self.orthogonalInitialization = orthogonalInitialization
        self.actorOutputGain = actorOutputGain
        self.actionDistribution = actionDistribution
        self.initialActionStd = initialActionStd
        self.minimumActionStd = minimumActionStd
        self.maximumActionStd = maximumActionStd
        self.finalActionStd = finalActionStd
        self.actionStdAnnealStartUpdate = actionStdAnnealStartUpdate
        self.actionStdAnnealEndUpdate = actionStdAnnealEndUpdate
        self.routedExplorationMaskVersion = routedExplorationMaskVersion
        self.trainingProgressUpdateVersion = trainingProgressUpdateVersion
        self.initializationCheckpoint = initializationCheckpoint
        self.policyExpertInitializationCheckpoint =
            policyExpertInitializationCheckpoint
        self.policyExpertBranchInitializationCheckpoint =
            policyExpertBranchInitializationCheckpoint
        self.standExpertInitializationCheckpoint =
            standExpertInitializationCheckpoint
        self.initializationNormalizerPriorCount =
            initializationNormalizerPriorCount
        self.useTaskSymmetryAugmentation = useTaskSymmetryAugmentation
        self.symmetryMirrorLossCoefficient = symmetryMirrorLossCoefficient
        self.successImitationCoefficient = successImitationCoefficient
        self.successReplayCapacity = successReplayCapacity
        self.successReplayBatchSize = successReplayBatchSize
        self.successImitationHistorySteps = successImitationHistorySteps
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
        guard [
            learningRate, gamma, gaeLambda, policyClip, valueClip,
            valueCoefficient, entropyCoefficient, maxGradientNorm, targetKL,
            initialActionStd,
        ].allSatisfy(\.isFinite) else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO floating-point configuration must be finite")
        }
        guard minibatchSize > 0, minibatchSize <= batchSize else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO minibatch must be in 1...rollout batch (\(batchSize))")
        }
        guard batchSize.isMultiple(of: minibatchSize) else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO rollout batch must be divisible by minibatch size so "
                    + "behavior and update inference use one stable Metal "
                    + "matrix shape")
        }
        guard learningRate > 0, resolvedOptimizerEpsilon > 0,
              resolvedOptimizerEpsilon.isFinite, gamma > 0, gamma <= 1,
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
        if let actorOutputGain {
            guard actorOutputGain > 0, actorOutputGain.isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "PPO actor output initialization gain must be finite and positive")
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
        let annealingFieldsPresent = [
            finalActionStd != nil,
            actionStdAnnealStartUpdate != nil,
            actionStdAnnealEndUpdate != nil,
        ]
        if annealingFieldsPresent.contains(true) {
            guard annealingFieldsPresent.allSatisfy({ $0 }),
                  let finalActionStd,
                  let actionStdAnnealStartUpdate,
                  let actionStdAnnealEndUpdate,
                  finalActionStd > 0,
                  finalActionStd <= exp(Float(1)),
                  actionStdAnnealStartUpdate >= 0,
                  actionStdAnnealEndUpdate > actionStdAnnealStartUpdate,
                  minimumActionStd == nil,
                  maximumActionStd == nil else {
                throw RLEnvironmentError.invalidConfiguration(
                    "action-std annealing requires a valid final std, a "
                        + "non-negative start before its end, and no fixed "
                        + "minimum/maximum std bounds")
            }
        }
        guard (1...4).contains(resolvedRoutedExplorationMaskVersion) else {
            throw RLEnvironmentError.invalidConfiguration(
                "unsupported routed exploration-mask version")
        }
        guard (1...2).contains(resolvedTrainingProgressUpdateVersion) else {
            throw RLEnvironmentError.invalidConfiguration(
                "unsupported task training-progress update version")
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
        let replayCapacity = successReplayCapacity ?? 0
        let replayBatchSize = successReplayBatchSize ?? 0
        guard replayCapacity >= 0, replayBatchSize >= 0,
              (replayCapacity == 0 && replayBatchSize == 0)
                || (replayCapacity > 0 && replayBatchSize > 0
                    && replayBatchSize <= replayCapacity) else {
            throw RLEnvironmentError.invalidConfiguration(
                "success replay requires 0/0 or a positive batch no larger "
                    + "than its capacity")
        }
        if replayCapacity > 0
            && (successImitationCoefficient ?? 0) <= 0 {
            throw RLEnvironmentError.invalidConfiguration(
                "success replay requires a positive imitation coefficient")
        }
        if replayCapacity > 0, normalizeObservations,
           updateObservationNormalizer != false {
            throw RLEnvironmentError.invalidConfiguration(
                "success replay requires raw observations or a frozen "
                    + "observation normalizer")
        }
        if let successImitationHistorySteps,
           successImitationHistorySteps <= 0 {
            throw RLEnvironmentError.invalidConfiguration(
                "success-imitation history must be positive when specified")
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
            || policyExpertBranchInitializationCheckpoint != nil
            || standExpertInitializationCheckpoint != nil),
           initializationCheckpoint == nil {
            throw RLEnvironmentError.invalidConfiguration(
                "expert composition requires an initialization checkpoint")
        }
        if policyExpertInitializationCheckpoint != nil,
           policyExpertBranchInitializationCheckpoint != nil {
            throw RLEnvironmentError.invalidConfiguration(
                "base-to-expert and routed-branch composition are mutually exclusive")
        }
    }

    /// A resume restores policy, normalizer, Adam, scheduler, and progress at
    /// an update boundary, so every setting that determines later updates must
    /// remain identical. Simulator/task state is intentionally not serialized:
    /// environments restart from the configured seed, so this is not an exact
    /// mid-trajectory continuation. Extending the update count or changing
    /// snapshot frequency is safe; changing optimization, sampling,
    /// normalization, symmetry, or seed is an explicit transfer and must use
    /// `--initialize-from`.
    func resumeIncompatibilities(with checkpoint: Self) -> [String] {
        var fields = [String]()
        func require<T: Equatable>(_ name: String, _ current: T, _ saved: T) {
            if current != saved { fields.append(name) }
        }
        require("rolloutSteps", rolloutSteps, checkpoint.rolloutSteps)
        require("updateEpochs", updateEpochs, checkpoint.updateEpochs)
        require("minibatchSize", minibatchSize, checkpoint.minibatchSize)
        require("learningRate", learningRate, checkpoint.learningRate)
        require("optimizerEpsilon", resolvedOptimizerEpsilon,
                checkpoint.resolvedOptimizerEpsilon)
        require("gamma", gamma, checkpoint.gamma)
        require("gaeLambda", gaeLambda, checkpoint.gaeLambda)
        require("rewardScale", rewardScale ?? 1,
                checkpoint.rewardScale ?? 1)
        require("policyClip", policyClip, checkpoint.policyClip)
        require("valueClip", valueClip, checkpoint.valueClip)
        require("clipValueLoss", resolvedClipValueLoss,
                checkpoint.resolvedClipValueLoss)
        require("valueCoefficient", valueCoefficient,
                checkpoint.valueCoefficient)
        require("entropyCoefficient", entropyCoefficient,
                checkpoint.entropyCoefficient)
        require("maxGradientNorm", maxGradientNorm,
                checkpoint.maxGradientNorm)
        require("targetKL", targetKL, checkpoint.targetKL)
        require("klSchedule", resolvedKLSchedule,
                checkpoint.resolvedKLSchedule)
        require("hiddenSize", hiddenSize, checkpoint.hiddenSize)
        require("hiddenDimensions", resolvedHiddenDimensions,
                checkpoint.resolvedHiddenDimensions)
        require("activation", resolvedActivation,
                checkpoint.resolvedActivation)
        require("orthogonalInitialization", resolvedOrthogonalInitialization,
                checkpoint.resolvedOrthogonalInitialization)
        require("actorOutputGain", resolvedActorOutputGain,
                checkpoint.resolvedActorOutputGain)
        require("actionDistribution", resolvedActionDistribution,
                checkpoint.resolvedActionDistribution)
        require("initialActionStd", initialActionStd,
                checkpoint.initialActionStd)
        require("minimumActionStd", minimumActionStd,
                checkpoint.minimumActionStd)
        require("maximumActionStd", maximumActionStd,
                checkpoint.maximumActionStd)
        require("finalActionStd", finalActionStd,
                checkpoint.finalActionStd)
        require("actionStdAnnealStartUpdate", actionStdAnnealStartUpdate,
                checkpoint.actionStdAnnealStartUpdate)
        require("actionStdAnnealEndUpdate", actionStdAnnealEndUpdate,
                checkpoint.actionStdAnnealEndUpdate)
        require("routedExplorationMaskVersion",
                resolvedRoutedExplorationMaskVersion,
                checkpoint.resolvedRoutedExplorationMaskVersion)
        require("trainingProgressUpdateVersion",
                resolvedTrainingProgressUpdateVersion,
                checkpoint.resolvedTrainingProgressUpdateVersion)
        require("useTaskSymmetryAugmentation",
                useTaskSymmetryAugmentation != false,
                checkpoint.useTaskSymmetryAugmentation != false)
        require("symmetryMirrorLossCoefficient",
                symmetryMirrorLossCoefficient ?? 0,
                checkpoint.symmetryMirrorLossCoefficient ?? 0)
        require("successImitationCoefficient",
                successImitationCoefficient ?? 0,
                checkpoint.successImitationCoefficient ?? 0)
        require("successReplayCapacity", successReplayCapacity ?? 0,
                checkpoint.successReplayCapacity ?? 0)
        require("successReplayBatchSize", successReplayBatchSize ?? 0,
                checkpoint.successReplayBatchSize ?? 0)
        require("successImitationHistorySteps", successImitationHistorySteps,
                checkpoint.successImitationHistorySteps)
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

    /// Log-space bounds for the exact distribution used by one update. A
    /// fixed equal bound makes annealing independent of the learned log-std
    /// parameter and therefore exactly reproducible across stop/resume.
    func actionLogStandardDeviationBounds(
        completedUpdate: Int
    ) -> (minimum: Float, maximum: Float) {
        if let finalActionStd,
           let start = actionStdAnnealStartUpdate,
           let end = actionStdAnnealEndUpdate {
            let progress = min(max(
                Float(completedUpdate - start) / Float(end - start), 0), 1)
            let value = log(initialActionStd)
                + progress * (log(finalActionStd) - log(initialActionStd))
            return (value, value)
        }
        return (minimumActionStd.map(log) ?? -5,
                maximumActionStd.map(log) ?? 1)
    }

    var resolvedRoutedExplorationMaskVersion: Int {
        routedExplorationMaskVersion ?? 1
    }

    var resolvedTrainingProgressUpdateVersion: Int {
        trainingProgressUpdateVersion ?? 1
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
        persisted.policyExpertBranchInitializationCheckpoint =
            checkpoint.policyExpertBranchInitializationCheckpoint
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

    public var resolvedActivation: PPOActivation { activation ?? .elu }
    public var resolvedOrthogonalInitialization: Bool {
        orthogonalInitialization == true
    }
    public var resolvedActorOutputGain: Float {
        actorOutputGain ?? (0.01 * sqrt(2))
    }
    public var resolvedClipValueLoss: Bool { clipValueLoss != false }
    public var resolvedOptimizerEpsilon: Float { optimizerEpsilon ?? 1e-8 }
    public var resolvedKLSchedule: PPOKLSchedule { klSchedule ?? .adaptive }

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
    @ModuleInfo var auxiliaryActor1: Linear
    @ModuleInfo var auxiliaryActor2: Linear
    @ModuleInfo var auxiliaryActor3: Linear
    @ModuleInfo var auxiliaryActorOutput: Linear
    @ModuleInfo var critic1: Linear
    @ModuleInfo var critic2: Linear
    @ModuleInfo var critic3: Linear
    @ModuleInfo var criticOutput: Linear
    @ParameterInfo var logStandardDeviation: MLXArray

    public let actionDimension: Int
    public let activation: PPOActivation
    // Version 6 adds a fourth, independently routed actor. Earlier versions
    // remain inference/transfer compatible: the loader clones the verified
    // base actor into the missing branch, so expansion is behavior-identical.
    public static let architectureVersion = 6
    public static let compatibleArchitectureVersions: Set<Int> = [3, 4, 5, 6]

    public init(observationDimension: Int, actionDimension: Int,
                hiddenSize: Int = 512, hiddenDimensions: [Int]? = nil,
                initialActionStd: Float = 1.0,
                activation: PPOActivation = .elu,
                orthogonalInitialization: Bool = false,
                actorOutputGain: Float = 0.01 * sqrt(2),
                initializationSeed: UInt64 = 1) {
        self.actionDimension = actionDimension
        self.activation = activation
        let dimensions = hiddenDimensions ?? [
            hiddenSize, max(hiddenSize / 2, 64), max(hiddenSize / 4, 64),
        ]
        precondition(dimensions.count == 3
            && dimensions.allSatisfy { $0 > 0 })
        let firstSize = dimensions[0]
        let middleSize = dimensions[1]
        let finalSize = dimensions[2]
        if orthogonalInitialization {
            var rng = SplitMix64(seed: initializationSeed)
            let initializedActor1 = Self.orthogonalLinear(
                input: observationDimension, output: firstSize,
                gain: sqrt(2), rng: &rng)
            let initializedActor2 = Self.orthogonalLinear(
                input: firstSize, output: middleSize,
                gain: sqrt(2), rng: &rng)
            let initializedActor3 = Self.orthogonalLinear(
                input: middleSize, output: finalSize,
                gain: sqrt(2), rng: &rng)
            let initializedActorOutput = Self.orthogonalLinear(
                input: finalSize, output: actionDimension,
                gain: actorOutputGain, rng: &rng)
            actor1 = initializedActor1
            actor2 = initializedActor2
            actor3 = initializedActor3
            actorOutput = initializedActorOutput
            // Routed branches are dormant for ordinary tasks. Give them exact
            // copies of the base actor so turning a gate on starts as a
            // behavior-preserving operation rather than arbitrary routing.
            expertActor1 = Self.clonedLinear(initializedActor1)
            expertActor2 = Self.clonedLinear(initializedActor2)
            expertActor3 = Self.clonedLinear(initializedActor3)
            expertActorOutput = Self.clonedLinear(initializedActorOutput)
            standActor1 = Self.clonedLinear(initializedActor1)
            standActor2 = Self.clonedLinear(initializedActor2)
            standActor3 = Self.clonedLinear(initializedActor3)
            standActorOutput = Self.clonedLinear(initializedActorOutput)
            auxiliaryActor1 = Self.clonedLinear(initializedActor1)
            auxiliaryActor2 = Self.clonedLinear(initializedActor2)
            auxiliaryActor3 = Self.clonedLinear(initializedActor3)
            auxiliaryActorOutput = Self.clonedLinear(initializedActorOutput)
            critic1 = Self.orthogonalLinear(
                input: observationDimension, output: firstSize,
                gain: sqrt(2), rng: &rng)
            critic2 = Self.orthogonalLinear(
                input: firstSize, output: middleSize,
                gain: sqrt(2), rng: &rng)
            critic3 = Self.orthogonalLinear(
                input: middleSize, output: finalSize,
                gain: sqrt(2), rng: &rng)
            criticOutput = Self.orthogonalLinear(
                input: finalSize, output: 1,
                gain: sqrt(2), rng: &rng)
        } else {
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
            auxiliaryActor1 = Linear(observationDimension, firstSize)
            auxiliaryActor2 = Linear(firstSize, middleSize)
            auxiliaryActor3 = Linear(middleSize, finalSize)
            auxiliaryActorOutput = Linear(
                weight: MLXArray.zeros([actionDimension, finalSize]),
                bias: MLXArray.zeros([actionDimension]))
            critic1 = Linear(observationDimension, firstSize)
            critic2 = Linear(firstSize, middleSize)
            critic3 = Linear(middleSize, finalSize)
            criticOutput = Linear(weight: MLXArray.zeros([1, finalSize]),
                             bias: MLXArray.zeros([1]))
        }
        logStandardDeviation = MLXArray(
            [Float](repeating: log(initialActionStd), count: actionDimension))
    }

    public func forward(_ observations: MLXArray,
                        expertGate: MLXArray? = nil,
                        expertActionMask: MLXArray? = nil,
                        standExpertGate: MLXArray? = nil,
                        standExpertActionMask: MLXArray? = nil,
                        auxiliaryExpertGate: MLXArray? = nil,
                        auxiliaryExpertActionMask: MLXArray? = nil,
                        freezeBaseActor: Bool = false,
                        freezeExpertActor: Bool = false,
                        freezeStandActor: Bool = false,
                        freezeAuxiliaryActor: Bool = false,
                        freezeStandActorBackbone: Bool = false,
                        freezeAuxiliaryActorBackbone: Bool = false)
        -> (mean: MLXArray, value: MLXArray, logStandardDeviation: MLXArray) {
        var baseMean = actorOutput(activate(actor3(activate(
            actor2(activate(actor1(observations)))))))
        if freezeBaseActor { baseMean = stopGradient(baseMean) }
        var expertMean = expertActorOutput(activate(expertActor3(
            activate(expertActor2(activate(expertActor1(observations)))))))
        if freezeExpertActor { expertMean = stopGradient(expertMean) }
        var standHidden = activate(standActor3(
            activate(standActor2(activate(standActor1(observations))))))
        if freezeStandActorBackbone {
            standHidden = stopGradient(standHidden)
        }
        var standMean = standActorOutput(standHidden)
        if freezeStandActor { standMean = stopGradient(standMean) }
        var auxiliaryHidden = activate(auxiliaryActor3(
            activate(auxiliaryActor2(activate(auxiliaryActor1(observations))))))
        if freezeAuxiliaryActorBackbone {
            auxiliaryHidden = stopGradient(auxiliaryHidden)
        }
        var auxiliaryMean = auxiliaryActorOutput(auxiliaryHidden)
        if freezeAuxiliaryActor { auxiliaryMean = stopGradient(auxiliaryMean) }
        let expert = expertGate ?? MLXArray.zeros([observations.shape[0], 1])
        let stand = standExpertGate
            ?? MLXArray.zeros([observations.shape[0], 1])
        let auxiliary = auxiliaryExpertGate
            ?? MLXArray.zeros([observations.shape[0], 1])
        // A task may route only selected action dimensions through the third
        // expert. The remaining dimensions stay on the verified base actor,
        // enabling compositional policies such as base locomotion plus learned
        // upper-body manipulation. A nil mask preserves historical whole-row
        // routing exactly.
        let expertActionGate = expertActionMask.map { expert * $0 } ?? expert
        let standActionGate = standExpertActionMask.map { stand * $0 } ?? stand
        let auxiliaryActionGate = auxiliaryExpertActionMask.map {
            auxiliary * $0
        } ?? auxiliary
        let mean = (1 - expertActionGate - standActionGate
            - auxiliaryActionGate) * baseMean
            + expertActionGate * expertMean + standActionGate * standMean
            + auxiliaryActionGate * auxiliaryMean
        let criticHidden = activate(critic3(activate(
            critic2(activate(critic1(observations))))))
        let value = criticOutput(criticHidden).squeezed(axis: -1)
        return (mean, value, clip(logStandardDeviation, min: -5, max: 1))
    }

    private func activate(_ value: MLXArray) -> MLXArray {
        switch activation {
        case .elu: Self.stableELU(value)
        case .tanh: MLX.tanh(value)
        }
    }

    private static func clonedLinear(_ source: Linear) -> Linear {
        Linear(weight: source.weight, bias: source.bias)
    }

    /// CPU-side modified Gram-Schmidt runs only once at model creation and
    /// produces the same orthogonal-weight contract as the published CleanRL
    /// baseline without adding a runtime dependency to the training graph.
    private static func orthogonalLinear(
        input: Int, output: Int, gain: Float,
        rng: inout SplitMix64
    ) -> Linear {
        let vectorCount = min(input, output)
        let vectorLength = max(input, output)
        var basis = [[Float]]()
        basis.reserveCapacity(vectorCount)
        for _ in 0..<vectorCount {
            var vector = [Float](repeating: 0, count: vectorLength)
            var index = 0
            while index < vectorLength {
                let u1 = max(rng.nextFloat(), 1e-7)
                let u2 = rng.nextFloat()
                let radius = sqrt(-2 * log(u1))
                vector[index] = radius * cos(2 * .pi * u2)
                if index + 1 < vectorLength {
                    vector[index + 1] = radius * sin(2 * .pi * u2)
                }
                index += 2
            }
            // Two passes keep the square 256x256 layers well conditioned.
            for _ in 0..<2 {
                for previous in basis {
                    var projection: Float = 0
                    for i in 0..<vectorLength {
                        projection += vector[i] * previous[i]
                    }
                    for i in 0..<vectorLength {
                        vector[i] -= projection * previous[i]
                    }
                }
            }
            let norm = sqrt(max(vector.reduce(0) { $0 + $1 * $1 }, 1e-12))
            basis.append(vector.map { $0 / norm })
        }
        var weights = [Float](repeating: 0, count: output * input)
        if output <= input {
            for row in 0..<output {
                for column in 0..<input {
                    weights[row * input + column] = gain * basis[row][column]
                }
            }
        } else {
            for row in 0..<output {
                for column in 0..<input {
                    weights[row * input + column] = gain * basis[column][row]
                }
            }
        }
        return Linear(
            weight: MLXArray(weights).reshaped([output, input]),
            bias: MLXArray.zeros([output]))
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
        if version < 5 {
            // A v4 expert was conventionally the exact-standing specialist,
            // so preserve it in the new stand branch. A v3 policy has just
            // cloned its base into that same expert and remains identical.
            try cloneActor(from: "expert", to: "stand")
        }
        try cloneActor(from: "", to: "auxiliary")
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
                     "standActor1.weight", "auxiliaryActor1.weight",
                     "critic1.weight"] {
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

    /// Clone the first routed expert into the independent third branch. This
    /// is useful when a task splits one learned behavior into a preserved
    /// precondition skill and a newly trainable continuation skill.
    public static func initializingStandExpertFromPolicyExpert(
        _ source: [String: MLXArray]
    ) throws -> [String: MLXArray] {
        var initialized = source
        for suffix in ["weight", "bias"] {
            for layer in ["1", "2", "3"] {
                let expert = "expertActor\(layer).\(suffix)"
                let stand = "standActor\(layer).\(suffix)"
                guard let value = source[expert] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "expanded actor checkpoint is missing \(expert)")
                }
                initialized[stand] = value
            }
            let expert = "expertActorOutput.\(suffix)"
            let stand = "standActorOutput.\(suffix)"
            guard let value = source[expert] else {
                throw RLEnvironmentError.invalidConfiguration(
                    "expanded actor checkpoint is missing \(expert)")
            }
            initialized[stand] = value
        }
        return initialized
    }

    /// Clone the verified base actor into the third routed branch. The task can
    /// then route only lower-body actions through this branch while a disjoint
    /// frozen expert supplies upper-body manipulation.
    public static func initializingStandExpertFromBase(
        _ source: [String: MLXArray]
    ) throws -> [String: MLXArray] {
        var initialized = source
        for suffix in ["weight", "bias"] {
            for layer in ["1", "2", "3"] {
                let base = "actor\(layer).\(suffix)"
                let stand = "standActor\(layer).\(suffix)"
                guard let value = source[base] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "expanded actor checkpoint is missing \(base)")
                }
                initialized[stand] = value
            }
            let base = "actorOutput.\(suffix)"
            let stand = "standActorOutput.\(suffix)"
            guard let value = source[base] else {
                throw RLEnvironmentError.invalidConfiguration(
                    "expanded actor checkpoint is missing \(base)")
            }
            initialized[stand] = value
        }
        return initialized
    }

    /// Clone the verified base actor into the fourth routed branch, optionally
    /// projecting task-declared first-layer inputs out of the initialization.
    /// Unlike a fixed action blend, the result remains a complete trainable
    /// state-feedback policy while the source actor stays frozen.
    public static func initializingAuxiliaryExpertFromBase(
        _ source: [String: MLXArray],
        zeroedObservationIndices: [Int] = []
    ) throws -> [String: MLXArray] {
        var initialized = source
        for suffix in ["weight", "bias"] {
            for layer in ["1", "2", "3"] {
                let base = "actor\(layer).\(suffix)"
                let auxiliary = "auxiliaryActor\(layer).\(suffix)"
                guard let value = source[base] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "expanded actor checkpoint is missing \(base)")
                }
                initialized[auxiliary] = value
            }
            let base = "actorOutput.\(suffix)"
            let auxiliary = "auxiliaryActorOutput.\(suffix)"
            guard let value = source[base] else {
                throw RLEnvironmentError.invalidConfiguration(
                    "expanded actor checkpoint is missing \(base)")
            }
            initialized[auxiliary] = value
        }
        if !zeroedObservationIndices.isEmpty {
            guard Set(zeroedObservationIndices).count
                    == zeroedObservationIndices.count,
                  let firstLayer = initialized["auxiliaryActor1.weight"],
                  firstLayer.shape.count == 2 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "auxiliary observation projection is invalid")
            }
            let rows = firstLayer.shape[0]
            let columns = firstLayer.shape[1]
            guard zeroedObservationIndices.allSatisfy({
                (0..<columns).contains($0)
            }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "auxiliary observation projection exceeds actor input")
            }
            var values = firstLayer.asArray(Float.self)
            for row in 0..<rows {
                for column in zeroedObservationIndices {
                    values[row * columns + column] = 0
                }
            }
            initialized["auxiliaryActor1.weight"] = MLXArray(values)
                .reshaped([rows, columns])
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

    /// Compose an already-trained routed expert into another policy. This is
    /// deliberately separate from `initializingPolicyExpert`, whose contract
    /// is to promote a source *base* actor into the routed slot. Keeping both
    /// operations explicit prevents a multi-expert checkpoint from silently
    /// losing the specialist the caller intended to preserve.
    public static func initializingPolicyExpertFromExpert(
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
                "routed-expert normalizers have incompatible dimensions")
        }
        let firstWeightName = "expertActor1.weight"
        let firstBiasName = "expertActor1.bias"
        guard let sourceWeightArray = source[firstWeightName],
              let sourceBiasArray = source[firstBiasName],
              sourceWeightArray.shape.count == 2,
              sourceWeightArray.shape[1] == dimension,
              sourceBiasArray.shape == [sourceWeightArray.shape[0]] else {
            throw RLEnvironmentError.invalidConfiguration(
                "routed-expert checkpoint has an incompatible first layer")
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
                    "routed-expert normalizer contains non-finite statistics")
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
        composed[firstWeightName] = MLXArray(destinationWeights)
            .reshaped(sourceWeightArray.shape)
        composed[firstBiasName] = MLXArray(destinationBias)
        for layer in ["expertActor2", "expertActor3", "expertActorOutput"] {
            for suffix in ["weight", "bias"] {
                let name = "\(layer).\(suffix)"
                guard let value = source[name] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "routed-expert checkpoint is missing \(name)")
                }
                composed[name] = value
            }
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

public extension PPOEvaluationMetrics {
    /// Validate the self-contained facts in an evaluation report before it is
    /// ranked, aggregated, or used as publication evidence. Acceptance is
    /// optional because historical/development tasks may not define a gate;
    /// absence is never interpreted as a pass by downstream selection.
    func validateStructure() throws {
        guard !task.isEmpty, !checkpointDirectory.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation task and checkpoint directory must be non-empty")
        }
        guard trainingUpdates >= 0, trainingEnvironmentSteps >= 0,
              episodes > 0, successes >= 0, successes <= episodes else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation counts are invalid")
        }
        if let evaluationEnvironments {
            guard evaluationEnvironments > 0 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "evaluation replica count must be positive")
            }
        }
        let derivedSuccessRate = Float(successes) / Float(episodes)
        guard successRate.isFinite, (0...1).contains(successRate),
              abs(successRate - derivedSuccessRate) <= 1e-6 else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation success count and rate are inconsistent")
        }
        guard meanReturn.isFinite, meanEpisodeLength.isFinite,
              meanEpisodeLength >= 0,
              taskMetrics.values.allSatisfy(\.isFinite),
              checkpointTaskConfiguration?.values.allSatisfy(\.isFinite)
                ?? true,
              evaluationTaskConfiguration?.values.allSatisfy(\.isFinite)
                ?? true else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation outputs and task contracts must be finite")
        }
        if let acceptance {
            guard acceptance.passed == acceptance.failures.isEmpty,
                  acceptance.failures.allSatisfy({
                      !$0.trimmingCharacters(in: .whitespacesAndNewlines)
                        .isEmpty
                  }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "evaluation acceptance disagrees with its failures")
            }
        }
    }
}

private func ppoCheckedSum(_ values: some Sequence<Int>,
                           label: String) throws -> Int {
    var total = 0
    for value in values {
        let result = total.addingReportingOverflow(value)
        guard !result.overflow else {
            throw RLEnvironmentError.invalidConfiguration(
                "\(label) overflows Int")
        }
        total = result.partialValue
    }
    return total
}

public struct PPOScalarSummary: Codable, Sendable, Equatable {
    public var median: Float
    public var firstQuartile: Float
    public var thirdQuartile: Float
    public var minimum: Float
    public var maximum: Float
}

private extension PPOScalarSummary {
    func validateStructure(label: String, range: ClosedRange<Float>? = nil,
                           minimumAllowed: Float? = nil) throws {
        let values = [self.minimum, firstQuartile, median, thirdQuartile,
                      maximum]
        guard values.allSatisfy(\.isFinite),
              self.minimum <= firstQuartile,
              firstQuartile <= median,
              median <= thirdQuartile,
              thirdQuartile <= maximum,
              minimumAllowed.map({ self.minimum >= $0 }) ?? true,
              range.map({ values.allSatisfy($0.contains) }) ?? true else {
            throw RLEnvironmentError.invalidConfiguration(
                "aggregate summary '\(label)' is invalid")
        }
    }
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
        try evaluations.forEach { try $0.validateStructure() }
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
        let totalEpisodes = try ppoCheckedSum(
            evaluations.lazy.map(\.episodes), label: "total evaluation episodes")
        let aggregate = Self(
            task: first.task,
            evaluationTaskConfiguration: first.evaluationTaskConfiguration,
            taskConfigurationTransferred: first.taskConfigurationTransferred,
            trainingSeeds: evaluations.map(\.trainingSeed).sorted(),
            evaluationSeeds: evaluations.map(\.evaluationSeed).sorted(),
            evaluationEnvironments: first.evaluationEnvironments,
            totalEpisodes: totalEpisodes,
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
        try aggregate.validateStructure()
        return aggregate
    }

    /// Validate invariants that remain independently checkable after the
    /// source reports have been reduced into an aggregate.
    public func validateStructure() throws {
        let runs = trainingSeeds.count
        guard !task.isEmpty,
              evaluationTaskConfiguration?.values.allSatisfy(\.isFinite)
                ?? true,
              evaluationEnvironments.map({ $0 > 0 }) ?? true,
              initializationCheckpoints.allSatisfy({ !$0.isEmpty }),
              checkpointFingerprints.allSatisfy({ !$0.isEmpty }),
              runs > 0, requiredRuns > 0, requiredEpisodesPerRun > 0,
              totalEpisodes > 0,
              Set(trainingSeeds).count == runs,
              evaluationSeeds.count == runs,
              acceptedRuns >= 0, acceptedRuns <= runs,
              allRunsPassed == (acceptedRuns == runs),
              hasRequiredRunCount == (runs >= requiredRuns),
              allRunsFromScratch == initializationCheckpoints.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation aggregate counts or gates are inconsistent")
        }
        let requiredEpisodeResult = runs.multipliedReportingOverflow(
            by: requiredEpisodesPerRun)
        guard !requiredEpisodeResult.overflow,
              !allRunsHaveRequiredEpisodes
                || totalEpisodes >= requiredEpisodeResult.partialValue else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation aggregate episode gate is inconsistent")
        }
        let expectedPublishable = hasRequiredRunCount
            && allRunsHaveRequiredEpisodes && allRunsPassed
            && provenanceComplete && allRunsFromScratch
        guard publishable == expectedPublishable,
              !provenanceComplete
                || ((evaluationEnvironments ?? 0) > 0
                    && checkpointFingerprints.count == runs) else {
            throw RLEnvironmentError.invalidConfiguration(
                "evaluation aggregate publication gate is inconsistent")
        }
        try successRate.validateStructure(
            label: "successRate", range: 0...1)
        try trainingUpdates.validateStructure(
            label: "trainingUpdates", minimumAllowed: 0)
        try trainingEnvironmentSteps.validateStructure(
            label: "trainingEnvironmentSteps", minimumAllowed: 0)
        try meanReturn.validateStructure(label: "meanReturn")
        try meanEpisodeLength.validateStructure(
            label: "meanEpisodeLength", minimumAllowed: 0)
        for (name, summary) in taskMetrics {
            try summary.validateStructure(label: "taskMetrics.\(name)")
        }
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
    /// Optional only so aggregate reports written before this field existed
    /// remain decodable. Newly produced, provenance-complete aggregates always
    /// persist the exact task physics revision.
    public var taskRevision: Int?
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
        try evaluations.forEach { try $0.validateStructure() }
        guard first.taskRevision != nil,
              evaluations.allSatisfy({ $0.taskRevision != nil }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint robustness reports require an explicit task revision")
        }
        guard evaluations.allSatisfy({
            $0.task == first.task
                && $0.taskRevision == first.taskRevision
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
        let totalEpisodes = try ppoCheckedSum(
            evaluations.lazy.map(\.episodes), label: "total evaluation episodes")
        let totalSuccesses = try ppoCheckedSum(
            evaluations.lazy.map(\.successes), label: "total evaluation successes")
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
        let aggregate = Self(
            scope: "single_checkpoint_across_evaluation_seeds",
            task: first.task,
            taskRevision: first.taskRevision,
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
        try aggregate.validateStructure()
        return aggregate
    }

    /// Validate the arithmetic and Boolean gates of a decoded checkpoint
    /// aggregate without trusting fields such as `pooledSuccessRate` or
    /// `robustAcrossEvaluationSeeds` merely because they were serialized.
    public func validateStructure() throws {
        guard scope == "single_checkpoint_across_evaluation_seeds",
              !task.isEmpty, !checkpointDirectory.isEmpty,
              evaluationTaskConfiguration?.values.allSatisfy(\.isFinite)
                ?? true,
              evaluationEnvironments.map({ $0 > 0 }) ?? true,
              initializationCheckpoint.map({ !$0.isEmpty }) ?? true,
              checkpointFingerprint.map({ !$0.isEmpty }) ?? true,
              runs > 0, requiredRuns > 0, requiredEpisodesPerRun > 0,
              evaluationSeeds.count == runs,
              Set(evaluationSeeds).count == runs,
              totalEpisodes > 0,
              totalSuccesses >= 0, totalSuccesses <= totalEpisodes,
              acceptedRuns >= 0, acceptedRuns <= runs,
              allRunsPassed == (acceptedRuns == runs),
              hasRequiredRunCount == (runs >= requiredRuns) else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint aggregate counts or gates are inconsistent")
        }
        let derivedRate = Float(totalSuccesses) / Float(totalEpisodes)
        guard pooledSuccessRate.isFinite,
              (0...1).contains(pooledSuccessRate),
              abs(pooledSuccessRate - derivedRate) <= 1e-6 else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint aggregate success count and rate are inconsistent")
        }
        let requiredEpisodeResult = runs.multipliedReportingOverflow(
            by: requiredEpisodesPerRun)
        guard !requiredEpisodeResult.overflow,
              !allRunsHaveRequiredEpisodes
                || totalEpisodes >= requiredEpisodeResult.partialValue else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint aggregate episode gate is inconsistent")
        }
        let expectedRobust = hasRequiredRunCount
            && allRunsHaveRequiredEpisodes && allRunsPassed
            && provenanceComplete
        guard robustAcrossEvaluationSeeds == expectedRobust,
              !provenanceComplete
                || (taskRevision != nil
                    && (evaluationEnvironments ?? 0) > 0
                    && !(checkpointFingerprint ?? "").isEmpty) else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint aggregate robustness gate is inconsistent")
        }
        try successRate.validateStructure(
            label: "successRate", range: 0...1)
        try meanReturn.validateStructure(label: "meanReturn")
        try meanEpisodeLength.validateStructure(
            label: "meanEpisodeLength", minimumAllowed: 0)
        for (name, summary) in taskMetrics {
            try summary.validateStructure(label: "taskMetrics.\(name)")
        }
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
        try evaluations.forEach { try $0.validateStructure() }
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

        let candidates = try grouped.values.map { reports in
            let candidate = reports[0]
            let totalEpisodes = try ppoCheckedSum(
                reports.lazy.map(\.episodes),
                label: "checkpoint validation episodes")
            let totalSuccesses = try ppoCheckedSum(
                reports.lazy.map(\.successes),
                label: "checkpoint validation successes")
            let weightedReturn = Float(reports.reduce(Double(0)) {
                $0 + Double($1.meanReturn) * Double($1.episodes)
            } / Double(totalEpisodes))
            guard weightedReturn.isFinite else {
                throw RLEnvironmentError.invalidConfiguration(
                    "checkpoint validation return is not finite")
            }
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
                    $0.acceptance?.passed == true
                })
        }.sorted {
            if $0.trainingUpdates != $1.trainingUpdates {
                return $0.trainingUpdates < $1.trainingUpdates
            }
            return $0.checkpointFingerprint < $1.checkpointFingerprint
        }
        let acceptedCandidates = candidates.filter(\.acceptancePassed)
        guard !acceptedCandidates.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "no checkpoint candidate passed every validation acceptance gate")
        }
        let selected = acceptedCandidates.max {
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
        let validationEpisodesPerCandidate = try ppoCheckedSum(
            expectedEpisodes.values,
            label: "validation episodes per checkpoint candidate")
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
            validationEpisodesPerCandidate: validationEpisodesPerCandidate,
            validationEpisodesPerSeed: uniformEpisodesPerSeed,
            selectionRule: "all_validation_acceptance_pass,worst_seed_success_rate,pooled_success_rate,mean_return,earlier_update,fingerprint",
            candidates: candidates,
            selectedCheckpointDirectory: selected.checkpointDirectory,
            selectedCheckpointFingerprint: selected.checkpointFingerprint,
            selectedTrainingUpdates: selected.trainingUpdates)
    }

    public func validateTestReport(_ evaluation: PPOEvaluationMetrics) throws {
        try evaluation.validateStructure()
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
    /// Row count used for policy inference during training. Some Metal
    /// matrix-multiplication kernels select a different reduction path for a
    /// one-row UI replay than for a batched rollout. Recording the validated
    /// geometry lets deployment reproduce the checkpoint's numerical path.
    /// Nil identifies checkpoints written before this contract existed.
    public var inferenceBatchSize: Int? = nil
    public var ppo: VectorPPOConfig
    public var normalizer: RunningNormalizerSnapshot

    /// Structural validation used before a live snapshot is considered for
    /// replay or exact resume. Decoding only task identity is insufficient: a
    /// truncated newest metadata file would otherwise hide an older complete
    /// generation and fail later after selection.
    func validateReplayCheckpointStructure() throws {
        guard let architectureVersion,
              VectorActorCritic.compatibleArchitectureVersions.contains(
                architectureVersion),
              !task.isEmpty,
              let taskRevision, taskRevision > 0,
              observationDimension > 0,
              actionDimension > 0,
              simulationStep.isFinite, simulationStep > 0,
              controlDecimation > 0,
              maxEpisodeSteps > 0,
              inferenceBatchSize.map({ $0 > 0 }) ?? true,
              normalizer.count.isFinite, normalizer.count >= 0,
              normalizer.mean.count == observationDimension,
              normalizer.variance.count == observationDimension,
              normalizer.mean.allSatisfy(\.isFinite),
              normalizer.variance.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO checkpoint metadata structure is invalid")
        }
    }

    /// Exact in-place replay compatibility. Explicit transfer paths may relax
    /// selected fields, but a generic action provider must always fail closed.
    public func compatibilityMismatches(with spec: RLTaskSpec) -> [String] {
        var mismatches = [String]()
        if task != spec.id {
            mismatches.append("task \(task) != \(spec.id)")
        }
        if (taskRevision ?? 1) != spec.revision {
            mismatches.append(
                "revision \(taskRevision ?? 1) != \(spec.revision)")
        }
        if taskConfiguration == nil {
            mismatches.append("configuration metadata is missing")
        } else if let taskConfiguration,
                  taskConfiguration != spec.configurationValues {
            let keys = Set(taskConfiguration.keys)
                .union(spec.configurationValues.keys)
            let differences = keys.sorted().compactMap { key -> String? in
                let saved = taskConfiguration[key]
                let runtime = spec.configurationValues[key]
                guard saved != runtime else { return nil }
                return "\(key)=\(saved.map { String($0) } ?? "missing")/"
                    + "\(runtime.map { String($0) } ?? "missing")"
            }
            mismatches.append(
                "configuration [\(differences.joined(separator: ", "))]")
        }
        if observationDimension != spec.observation.elementCount {
            mismatches.append(
                "observations \(observationDimension) != "
                    + "\(spec.observation.elementCount)")
        }
        if actionDimension != spec.action.elementCount {
            mismatches.append(
                "actions \(actionDimension) != \(spec.action.elementCount)")
        }
        if simulationStep != spec.simulationStep {
            mismatches.append(
                "simulation step \(simulationStep) != \(spec.simulationStep)")
        }
        if controlDecimation != spec.controlDecimation {
            mismatches.append(
                "control decimation \(controlDecimation) != "
                    + "\(spec.controlDecimation)")
        }
        if maxEpisodeSteps != spec.maxEpisodeSteps {
            mismatches.append(
                "episode steps \(maxEpisodeSteps) != \(spec.maxEpisodeSteps)")
        }
        return mismatches
    }
}

public struct VectorPPOTrainingState: Codable, Sendable {
    public static let currentResumableSnapshotVersion = 2

    public var completedUpdates: Int
    public var environmentSteps: Int
    /// Explicit opt-in to the exact-resume contract. Nil is retained for
    /// historical, evaluation-only, distillation, and requalification files.
    public var resumableSnapshotVersion: Int?
    /// Simulator rows advanced by each rollout step in this run.
    public var rolloutEnvironmentCount: Int?
    /// Minibatch-level Adam step. Optional for pre-checkpoint-v2 runs.
    public var optimizerSteps: Int?
    /// Adaptive KL scheduler state. Optional for pre-checkpoint-v2 runs.
    public var adaptiveLearningRate: Float?
    /// Number of chronological rows in success-replay.safetensors. Optional
    /// so evaluation and requalification artifacts outside the resumable PPO
    /// snapshot format retain source-compatible JSON.
    public var successReplayCount: Int?

    public init(completedUpdates: Int, environmentSteps: Int,
                resumableSnapshotVersion: Int? = nil,
                rolloutEnvironmentCount: Int? = nil,
                optimizerSteps: Int? = nil,
                adaptiveLearningRate: Float? = nil,
                successReplayCount: Int? = nil) {
        self.completedUpdates = completedUpdates
        self.environmentSteps = environmentSteps
        self.resumableSnapshotVersion = resumableSnapshotVersion
        self.rolloutEnvironmentCount = rolloutEnvironmentCount
        self.optimizerSteps = optimizerSteps
        self.adaptiveLearningRate = adaptiveLearningRate
        self.successReplayCount = successReplayCount
    }

    enum OptimizerResumeState: Equatable {
        /// A current checkpoint taken before the first optimizer update has
        /// no moment tensors to restore.
        case fresh(adaptiveLearningRate: Float)
        /// A current checkpoint with an advanced Adam trajectory must carry
        /// its named first- and second-moment tensors.
        case checkpointed(steps: Int, adaptiveLearningRate: Float)
    }

    func validatedOptimizerResumeState(
        configuration: VectorPPOConfig,
        maximumOptimizerSteps: Int? = nil
    ) throws -> OptimizerResumeState {
        guard resumableSnapshotVersion
                == Self.currentResumableSnapshotVersion,
              completedUpdates >= 0, environmentSteps >= 0,
              let optimizerSteps, let adaptiveLearningRate else {
            throw RLEnvironmentError.invalidConfiguration(
                "exact PPO resume requires current optimizer counters and "
                    + "adaptive learning-rate state")
        }
        guard adaptiveLearningRate.isFinite, adaptiveLearningRate > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO checkpoint adaptive learning rate is invalid")
        }
        let configuredLearningRate = configuration.learningRate
        guard configuredLearningRate.isFinite, configuredLearningRate > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO checkpoint configuration learning rate is invalid")
        }
        switch configuration.resolvedKLSchedule {
        case .adaptive:
            guard configuration.targetKL > 0 else {
                guard adaptiveLearningRate == configuredLearningRate else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "disabled adaptive PPO scheduler must retain its "
                            + "configured learning rate")
                }
                break
            }
            let minimumLearningRate = configuredLearningRate / 100
            guard minimumLearningRate.isFinite,
                  adaptiveLearningRate >= minimumLearningRate,
                  adaptiveLearningRate <= configuredLearningRate else {
                throw RLEnvironmentError.invalidConfiguration(
                    "adaptive PPO checkpoint learning rate is outside its "
                        + "configured scheduler range")
            }
        case .earlyStop, .none:
            guard adaptiveLearningRate == configuredLearningRate else {
                throw RLEnvironmentError.invalidConfiguration(
                    "fixed-rate PPO checkpoint learning rate differs from "
                        + "its configuration")
            }
        }
        if completedUpdates == 0 || environmentSteps == 0
            || optimizerSteps == 0 {
            guard completedUpdates == 0, environmentSteps == 0,
                  optimizerSteps == 0,
                  adaptiveLearningRate == configuredLearningRate else {
                throw RLEnvironmentError.invalidConfiguration(
                    "a fresh PPO checkpoint must have zero update, environment, "
                        + "and optimizer counters and the configured learning "
                        + "rate")
            }
            return .fresh(adaptiveLearningRate: adaptiveLearningRate)
        }
        guard optimizerSteps >= completedUpdates else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO optimizer steps cannot trail completed updates")
        }
        if let maximumOptimizerSteps,
           optimizerSteps > maximumOptimizerSteps {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO optimizer steps exceed the rollout/update geometry")
        }
        return .checkpointed(
            steps: optimizerSteps,
            adaptiveLearningRate: adaptiveLearningRate)
    }

    func validatedRolloutEnvironmentCount(rolloutSteps: Int) throws -> Int {
        guard resumableSnapshotVersion
                == Self.currentResumableSnapshotVersion,
              let rolloutEnvironmentCount, rolloutEnvironmentCount > 0,
              rolloutSteps > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "exact PPO resume requires v2 rollout geometry")
        }
        let updateRows = completedUpdates.multipliedReportingOverflow(
            by: rolloutEnvironmentCount)
        guard !updateRows.overflow else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO checkpoint rollout progress overflows Int")
        }
        let expectedSteps = updateRows.partialValue
            .multipliedReportingOverflow(by: rolloutSteps)
        guard !expectedSteps.overflow,
              environmentSteps == expectedSteps.partialValue else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO checkpoint environment steps do not match rollout geometry")
        }
        return rolloutEnvironmentCount
    }

    func maximumOptimizerSteps(
        rolloutSteps: Int,
        updateEpochs: Int,
        minibatchSize: Int
    ) throws -> Int {
        let environments = try validatedRolloutEnvironmentCount(
            rolloutSteps: rolloutSteps)
        guard updateEpochs > 0, minibatchSize > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO optimizer geometry must be positive")
        }
        let batch = environments.multipliedReportingOverflow(by: rolloutSteps)
        guard !batch.overflow else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO rollout batch overflows Int")
        }
        let minibatches = batch.partialValue / minibatchSize
            + (batch.partialValue % minibatchSize == 0 ? 0 : 1)
        let perUpdate = minibatches.multipliedReportingOverflow(by: updateEpochs)
        guard !perUpdate.overflow else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO optimizer epoch geometry overflows Int")
        }
        let maximum = completedUpdates.multipliedReportingOverflow(
            by: perUpdate.partialValue)
        guard !maximum.overflow else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO optimizer step ceiling overflows Int")
        }
        return maximum.partialValue
    }

    func validatedSuccessReplayCount(capacity: Int) throws -> Int {
        guard capacity >= 0, let successReplayCount,
              (0...capacity).contains(successReplayCount) else {
            throw RLEnvironmentError.invalidConfiguration(
                "exact PPO resume requires a valid success replay row count")
        }
        return successReplayCount
    }
}

/// A checkpoint snapshot that is safe for another process to open. Training
/// completes each generation in a hidden staging directory, writes
/// `training-state.json` last, and atomically renames it to
/// `checkpoints/update-NNNNNN`. Discovering immutable snapshots (rather than
/// the mutable run root) prevents replay or resume from observing mixed files.
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
    private static func isNonemptyReadableFile(_ url: URL) -> Bool {
        let manager = FileManager.default
        guard manager.isReadableFile(atPath: url.path),
              let values = try? url.resourceValues(forKeys: [
                .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
              ]),
              values.isRegularFile == true,
              values.isSymbolicLink != true else { return false }
        return (values.fileSize ?? 0) > 0
    }

    static func checkpointDirectoryName(completedUpdates: Int) throws -> String {
        guard completedUpdates >= 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO checkpoint update must be non-negative")
        }
        let digits = String(completedUpdates)
        return "update-"
            + String(repeating: "0", count: max(6 - digits.count, 0))
            + digits
    }

    /// Read only the bounded JSON header. Live checkpoint discovery should
    /// not initialize an MLX device or map a potentially large replay payload
    /// merely to verify tensor row counts.
    private static func safetensorShapes(at url: URL) -> [String: [Int]]? {
        guard isNonemptyReadableFile(url),
              let fileSize = try? url.resourceValues(forKeys: [.fileSizeKey])
                .fileSize,
              fileSize >= 8,
              let handle = try? FileHandle(forReadingFrom: url) else {
            return nil
        }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8),
              lengthData.count == 8 else { return nil }
        let headerLength = lengthData.enumerated().reduce(UInt64(0)) {
            $0 | UInt64($1.element) << UInt64(8 * $1.offset)
        }
        let maximumHeaderBytes = UInt64(16 * 1_024 * 1_024)
        guard headerLength > 0, headerLength <= maximumHeaderBytes,
              headerLength <= UInt64(fileSize - 8),
              headerLength <= UInt64(Int.max),
              let header = try? handle.read(upToCount: Int(headerLength)),
              header.count == Int(headerLength),
              let object = try? JSONSerialization.jsonObject(with: header),
              let root = object as? [String: Any] else { return nil }
        let payloadBytes = fileSize - 8 - Int(headerLength)
        var shapes = [String: [Int]]()
        var spans = [(Int, Int)]()
        for (name, rawDescriptor) in root where name != "__metadata__" {
            guard let descriptor = rawDescriptor as? [String: Any],
                  descriptor["dtype"] as? String == "F32",
                  let rawShape = descriptor["shape"] as? [NSNumber],
                  let rawOffsets = descriptor["data_offsets"] as? [NSNumber],
                  rawOffsets.count == 2 else { return nil }
            var shape = [Int]()
            shape.reserveCapacity(rawShape.count)
            for number in rawShape {
                let dimension = number.intValue
                guard dimension >= 0,
                      number.doubleValue == Double(dimension) else { return nil }
                shape.append(dimension)
            }
            let start = rawOffsets[0].intValue
            let end = rawOffsets[1].intValue
            guard start >= 0, end >= start, end <= payloadBytes,
                  rawOffsets[0].doubleValue == Double(start),
                  rawOffsets[1].doubleValue == Double(end) else { return nil }
            var elementCount = 1
            for dimension in shape {
                let product = elementCount.multipliedReportingOverflow(
                    by: dimension)
                guard !product.overflow else { return nil }
                elementCount = product.partialValue
            }
            let byteCount = elementCount.multipliedReportingOverflow(by: 4)
            guard !byteCount.overflow,
                  end - start == byteCount.partialValue else { return nil }
            shapes[name] = shape
            spans.append((start, end))
        }
        spans.sort { $0.0 < $1.0 }
        guard spans.first?.0 == 0,
              spans.last?.1 == payloadBytes else { return nil }
        for index in 1..<spans.count
            where spans[index].0 != spans[index - 1].1 {
            return nil
        }
        return shapes
    }

    private static func successReplayMatches(
        directory: URL,
        expectedCount: Int,
        capacity: Int,
        observationDimension: Int,
        actionDimension: Int
    ) -> Bool {
        guard capacity >= 0, (0...capacity).contains(expectedCount) else {
            return false
        }
        let url = directory.appendingPathComponent(
            VectorPPOTrainer.successReplayFileName)
        if expectedCount == 0 {
            return !FileManager.default.fileExists(atPath: url.path)
        }
        guard capacity > 0, isNonemptyReadableFile(url),
              let shapes = safetensorShapes(at: url),
              Set(shapes.keys) == Set([
                "observations", "actions", "expertGates",
                "standExpertGates", "auxiliaryExpertGates",
              ]),
              shapes["observations"] == [
                expectedCount, observationDimension,
              ],
              shapes["actions"] == [expectedCount, actionDimension],
              shapes["expertGates"] == [expectedCount],
              shapes["standExpertGates"] == [expectedCount],
              shapes["auxiliaryExpertGates"] == [expectedCount] else {
            return false
        }
        return true
    }

    private static func replayCandidate(
        at directory: URL,
        numberedUpdate: Int,
        task: String,
        taskRevision: Int
    ) -> VectorPolicyCheckpointCandidate? {
        guard numberedUpdate >= 0,
              isNonemptyReadableFile(directory
                .appendingPathComponent("policy.safetensors")),
              isNonemptyReadableFile(directory
                .appendingPathComponent("metadata.json")),
              isNonemptyReadableFile(directory
                .appendingPathComponent("training-state.json")),
              let metadata = try? JSONDecoder().decode(
                VectorPolicyMetadata.self,
                from: Data(contentsOf: directory
                    .appendingPathComponent("metadata.json"))),
              (try? metadata.validateReplayCheckpointStructure()) != nil,
              (try? metadata.ppo.validate(
                batchSize: metadata.ppo.minibatchSize)) != nil,
              metadata.task == task,
              (metadata.taskRevision ?? 1) == taskRevision,
              let progress = try? JSONDecoder().decode(
                VectorPPOTrainingState.self,
                from: Data(contentsOf: directory
                    .appendingPathComponent("training-state.json"))),
              progress.completedUpdates == numberedUpdate,
              progress.completedUpdates >= 0,
              progress.environmentSteps >= 0 else { return nil }
        return VectorPolicyCheckpointCandidate(
            directory: directory.resolvingSymlinksInPath().path,
            completedUpdates: progress.completedUpdates,
            environmentSteps: progress.environmentSteps)
    }

    private static func resumableCandidate(
        at directory: URL,
        numberedUpdate: Int,
        task: String,
        taskRevision: Int,
        requiredEnvironmentCount: Int?
    ) -> VectorPolicyCheckpointCandidate? {
        guard let candidate = replayCandidate(
                at: directory, numberedUpdate: numberedUpdate,
                task: task, taskRevision: taskRevision),
              let metadata = try? JSONDecoder().decode(
                VectorPolicyMetadata.self,
                from: Data(contentsOf: directory
                    .appendingPathComponent("metadata.json"))),
              let taskConfiguration = metadata.taskConfiguration,
              taskConfiguration.keys.allSatisfy({ !$0.isEmpty }),
              taskConfiguration.values.allSatisfy(\.isFinite),
              let inferenceBatchSize = metadata.inferenceBatchSize,
              inferenceBatchSize > 0,
              (try? metadata.ppo.validate(
                batchSize: metadata.ppo.minibatchSize)) != nil,
              let progress = try? JSONDecoder().decode(
                VectorPPOTrainingState.self,
                from: Data(contentsOf: directory
                    .appendingPathComponent("training-state.json"))),
              let rolloutEnvironmentCount = try? progress
                .validatedRolloutEnvironmentCount(
                    rolloutSteps: metadata.ppo.rolloutSteps),
              requiredEnvironmentCount.map({
                $0 == rolloutEnvironmentCount
              }) ?? true,
              let maximumOptimizerSteps = try? progress.maximumOptimizerSteps(
                rolloutSteps: metadata.ppo.rolloutSteps,
                updateEpochs: metadata.ppo.updateEpochs,
                minibatchSize: metadata.ppo.minibatchSize),
              let optimizerState = try? progress
                .validatedOptimizerResumeState(
                    configuration: metadata.ppo,
                    maximumOptimizerSteps: maximumOptimizerSteps),
              let replayCount = try? progress.validatedSuccessReplayCount(
                capacity: metadata.ppo.successReplayCapacity ?? 0),
              successReplayMatches(
                directory: directory,
                expectedCount: replayCount,
                capacity: metadata.ppo.successReplayCapacity ?? 0,
                observationDimension: metadata.observationDimension,
                actionDimension: metadata.actionDimension) else { return nil }

        if case .checkpointed = optimizerState,
           !isNonemptyReadableFile(directory
            .appendingPathComponent("optimizer.safetensors")) {
            return nil
        }
        if (metadata.ppo.referencePolicyCoefficient ?? 0) > 0,
           !isNonemptyReadableFile(directory
            .appendingPathComponent("reference-policy.safetensors")) {
            return nil
        }
        return candidate
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
            let values = try? directory.resourceValues(forKeys: [
                .isDirectoryKey, .isSymbolicLinkKey,
            ])
            guard name.hasPrefix("update-"),
                  let numberedUpdate = Int(name.dropFirst("update-".count)),
                  (try? checkpointDirectoryName(
                    completedUpdates: numberedUpdate)) == name,
                  values?.isDirectory == true,
                  values?.isSymbolicLink != true else { continue }

            guard let candidate = replayCandidate(
                at: directory, numberedUpdate: numberedUpdate,
                task: task, taskRevision: taskRevision) else { continue }
            if latest == nil
                || candidate.completedUpdates > latest!.completedUpdates {
                latest = candidate
            }
        }
        return latest
    }

    /// Resolve the only checkpoint source accepted by exact PPO resume. The
    /// mutable run root remains a convenience mirror for tools that expect it,
    /// but it has no generation marker and can contain files from two saves
    /// after interruption. A policy-only root is therefore an explicit
    /// transfer source, never an in-place resume source.
    static func checkpointForResume(
        inRunDirectory runDirectory: String,
        task: String,
        taskRevision: Int,
        numEnvironments: Int
    ) throws -> VectorPolicyCheckpointCandidate {
        guard numEnvironments > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO resume environment count must be positive")
        }
        let manager = FileManager.default
        let snapshots = URL(fileURLWithPath: runDirectory, isDirectory: true)
            .appendingPathComponent("checkpoints", isDirectory: true)
        let children = (try? manager.contentsOfDirectory(
            at: snapshots,
            includingPropertiesForKeys: [
                .isDirectoryKey, .isSymbolicLinkKey,
            ], options: [.skipsHiddenFiles])) ?? []
        var newestGeneration: (directory: URL, update: Int)?
        for entry in children {
            let name = entry.lastPathComponent
            guard name.hasPrefix("update-"),
                  let update = Int(name.dropFirst("update-".count)),
                  (try? checkpointDirectoryName(completedUpdates: update))
                    == name else { continue }
            if newestGeneration == nil || update > newestGeneration!.update {
                newestGeneration = (entry, update)
            }
        }
        guard let newestGeneration else {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO resume requires a complete resumable-format-v2 immutable "
                    + "checkpoint under checkpoints/update-NNNNNN; legacy "
                    + "snapshots and the mutable run root are replay-only. "
                    + "Use --initialize-from for policy transfer.")
        }
        let values = try? newestGeneration.directory.resourceValues(forKeys: [
            .isDirectoryKey, .isSymbolicLinkKey,
        ])
        guard values?.isDirectory == true,
              values?.isSymbolicLink != true,
              let candidate = resumableCandidate(
                at: newestGeneration.directory,
                numberedUpdate: newestGeneration.update,
                task: task, taskRevision: taskRevision,
                requiredEnvironmentCount: numEnvironments) else {
            throw RLEnvironmentError.invalidConfiguration(
                "newest immutable PPO generation update-"
                    + "\(newestGeneration.update) is not resumable format v2; "
                    + "quarantine/remove it or use --initialize-from. Exact "
                    + "resume will not fork behind a visible newer generation.")
        }
        return candidate
    }

    private static func synchronize(_ url: URL) throws {
        let descriptor = Darwin.open(url.path, O_RDONLY)
        guard descriptor >= 0 else {
            throw NSError(
                domain: NSPOSIXErrorDomain, code: Int(errno),
                userInfo: [NSFilePathErrorKey: url.path])
        }
        let syncResult = Darwin.fsync(descriptor)
        let syncError = errno
        let closeResult = Darwin.close(descriptor)
        if syncResult != 0 {
            throw NSError(
                domain: NSPOSIXErrorDomain, code: Int(syncError),
                userInfo: [NSFilePathErrorKey: url.path])
        }
        if closeResult != 0 {
            throw NSError(
                domain: NSPOSIXErrorDomain, code: Int(errno),
                userInfo: [NSFilePathErrorKey: url.path])
        }
    }

    private static func synchronizeSnapshot(at directory: URL) throws {
        let children = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [
                .isRegularFileKey, .isSymbolicLinkKey,
            ])
        for child in children {
            let values = try child.resourceValues(forKeys: [
                .isRegularFileKey, .isSymbolicLinkKey,
            ])
            guard values.isRegularFile == true,
                  values.isSymbolicLink != true else {
                throw RLEnvironmentError.invalidConfiguration(
                    "PPO checkpoint staging contains a non-regular file")
            }
            try synchronize(child)
        }
        try synchronize(directory)
    }

    /// Write a complete snapshot in a hidden sibling directory, validate its
    /// commit marker and required state, then publish it with one same-volume
    /// rename. Readers can observe either no final directory or the complete
    /// immutable generation, never an intermediate set of files.
    static func publishCompleteCheckpoint(
        inRunDirectory runDirectory: String,
        completedUpdates: Int,
        task: String,
        taskRevision: Int,
        write: (URL) throws -> Void
    ) throws -> VectorPolicyCheckpointCandidate {
        let manager = FileManager.default
        let run = URL(fileURLWithPath: runDirectory, isDirectory: true)
        let snapshots = run
            .appendingPathComponent("checkpoints", isDirectory: true)
        try manager.createDirectory(
            at: snapshots, withIntermediateDirectories: true)
        try synchronize(run)
        let name = try checkpointDirectoryName(
            completedUpdates: completedUpdates)
        let destination = snapshots.appendingPathComponent(
            name, isDirectory: true)
        guard !manager.fileExists(atPath: destination.path) else {
            throw RLEnvironmentError.invalidConfiguration(
                "refusing to replace immutable PPO checkpoint \(destination.path)")
        }
        let staging = snapshots.appendingPathComponent(
            ".\(name)-\(UUID().uuidString).staging", isDirectory: true)
        try manager.createDirectory(at: staging, withIntermediateDirectories: false)
        do {
            try write(staging)
            guard let candidate = resumableCandidate(
                at: staging, numberedUpdate: completedUpdates,
                task: task, taskRevision: taskRevision,
                requiredEnvironmentCount: nil) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "refusing to publish incomplete PPO checkpoint \(name)")
            }
            try synchronizeSnapshot(at: staging)
            let renameResult = staging.path.withCString { source in
                destination.path.withCString { target in
                    Darwin.renameatx_np(
                        AT_FDCWD, source, AT_FDCWD, target,
                        UInt32(RENAME_EXCL))
                }
            }
            guard renameResult == 0 else {
                let renameError = errno
                if renameError == EEXIST {
                    throw RLEnvironmentError.invalidConfiguration(
                        "refusing to replace immutable PPO checkpoint "
                            + destination.path)
                }
                throw NSError(
                    domain: NSPOSIXErrorDomain, code: Int(renameError),
                    userInfo: [NSFilePathErrorKey: destination.path])
            }
            try synchronize(snapshots)
            try synchronize(run)
            return VectorPolicyCheckpointCandidate(
                directory: destination.resolvingSymlinksInPath().path,
                completedUpdates: candidate.completedUpdates,
                environmentSteps: candidate.environmentSteps)
        } catch {
            try? manager.removeItem(at: staging)
            throw error
        }
    }
}

/// Advisory single-writer lease for a PPO run directory. `flock` is attached
/// to the open file description, so the kernel releases it on ordinary close
/// and on process termination; the persistent lock file is never deleted and
/// therefore cannot split contenders across different inodes.
final class VectorPPORunLock {
    static let fileName = ".ppo-trainer.lock"
    private var descriptor: Int32?

    init(runDirectory: String) throws {
        let url = URL(fileURLWithPath: runDirectory, isDirectory: true)
            .appendingPathComponent(Self.fileName)
        let opened = Darwin.open(
            url.path, O_CREAT | O_RDWR | O_CLOEXEC,
            mode_t(S_IRUSR | S_IWUSR))
        guard opened >= 0 else {
            throw NSError(
                domain: NSPOSIXErrorDomain, code: Int(errno),
                userInfo: [NSFilePathErrorKey: url.path])
        }
        guard avbdPOSIXFlock(opened, LOCK_EX | LOCK_NB) == 0 else {
            let lockError = errno
            _ = Darwin.close(opened)
            if lockError == EWOULDBLOCK || lockError == EAGAIN {
                throw RLEnvironmentError.invalidConfiguration(
                    "another PPO trainer already owns run directory "
                        + runDirectory)
            }
            throw NSError(
                domain: NSPOSIXErrorDomain, code: Int(lockError),
                userInfo: [NSFilePathErrorKey: url.path])
        }
        descriptor = opened
    }

    func unlock() {
        guard let descriptor else { return }
        _ = avbdPOSIXFlock(descriptor, LOCK_UN)
        _ = Darwin.close(descriptor)
        self.descriptor = nil
    }

    deinit { unlock() }
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

    static func validateCheckpointKeys(
        actual: Set<String>, expected: Set<String>
    ) throws {
        guard actual == expected else {
            let missing = expected.subtracting(actual).sorted()
            let unexpected = actual.subtracting(expected).sorted()
            throw RLEnvironmentError.invalidConfiguration(
                "optimizer checkpoint tensor keys mismatch; missing="
                    + "\(missing), unexpected=\(unexpected)")
        }
    }

    static func validateFirstMomentValues(_ values: [Float]) throws {
        guard values.allSatisfy(\.isFinite) else {
            throw RLEnvironmentError.invalidConfiguration(
                "optimizer checkpoint first moments must be finite")
        }
    }

    static func validateSecondMomentValues(_ values: [Float]) throws {
        guard values.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "optimizer checkpoint second moments must be finite and "
                    + "non-negative")
        }
    }

    func restore(arrays: [String: MLXArray], step: Int,
                 parameters: ModuleParameters) throws {
        guard step >= 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "optimizer checkpoint has a negative step")
        }
        let parameterPairs = parameters.flattened()
        let expectedKeys = Set(parameterPairs.flatMap { name, _ in
            ["first.\(name)", "second.\(name)"]
        })
        try Self.validateCheckpointKeys(
            actual: Set(arrays.keys), expected: expectedKeys)
        var restoredFirst = [String: MLXArray]()
        var restoredSecond = [String: MLXArray]()
        for (name, parameter) in parameterPairs {
            guard let first = arrays["first.\(name)"],
                  let second = arrays["second.\(name)"],
                  first.dtype == .float32,
                  second.dtype == .float32,
                  first.shape == parameter.shape,
                  second.shape == parameter.shape else {
                throw RLEnvironmentError.invalidConfiguration(
                    "optimizer checkpoint is missing or mismatches parameter \(name)")
            }
            try Self.validateFirstMomentValues(first.asArray(Float.self))
            try Self.validateSecondMomentValues(second.asArray(Float.self))
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

    public convenience init(checkpointDirectory: String) throws {
        let root = URL(fileURLWithPath: checkpointDirectory, isDirectory: true)
        let metadataData = try Data(contentsOf: root.appendingPathComponent(
            "metadata.json"))
        let policyData = try Data(contentsOf: root.appendingPathComponent(
            "policy.safetensors"))
        try self.init(metadataData: metadataData, policyData: policyData)
    }

    /// Construct from the exact bytes already authenticated by a deployment
    /// manifest. Keeping this initializer internal prevents callers from
    /// bypassing the public checkpoint contract while allowing deployment to
    /// close the path re-read/TOCTOU window.
    init(metadataData: Data, policyData: Data) throws {
        let decodedMetadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self, from: metadataData)
        try Self.validateMetadata(decodedMetadata)

        let minimumPayloadBytes = try Self.expectedPolicyPayloadByteCount(
            metadata: decodedMetadata)
        guard policyData.count >= minimumPayloadBytes else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy checkpoint is smaller than its declared tensor geometry")
        }

        // Metadata and all checked integer geometry are validated before MLX
        // parses tensors or allocates a model. The Data-backed API also makes
        // the weights below exactly the bytes authenticated by deployment.
        let sourceWeights = try loadArrays(data: policyData)
        try Self.validatePolicyTensors(sourceWeights)
        let weights = try VectorActorCritic.compatibleWeights(
            sourceWeights,
            architectureVersion: decodedMetadata.architectureVersion)

        let loadedPolicy = VectorActorCritic(
            observationDimension: decodedMetadata.observationDimension,
            actionDimension: decodedMetadata.actionDimension,
            hiddenSize: decodedMetadata.ppo.hiddenSize,
            hiddenDimensions: decodedMetadata.ppo.hiddenDimensions,
            initialActionStd: decodedMetadata.ppo.initialActionStd,
            activation: decodedMetadata.ppo.resolvedActivation)
        try loadedPolicy.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(loadedPolicy)

        metadata = decodedMetadata
        policy = loadedPolicy
        normalizer = RunningObservationNormalizer(
            snapshot: decodedMetadata.normalizer)
    }

    /// Validate every host-side value used to size or initialize inference.
    /// Training-only fields such as `updates` may legitimately be zero in an
    /// optimizer-free deployment export, so this is deliberately narrower
    /// than `VectorPPOConfig.validate(batchSize:)` while remaining strict for
    /// every PPO field that can affect replay.
    static func validateMetadata(_ metadata: VectorPolicyMetadata) throws {
        guard let architectureVersion = metadata.architectureVersion,
              VectorActorCritic.compatibleArchitectureVersions.contains(
                architectureVersion),
              !metadata.task.isEmpty,
              let taskRevision = metadata.taskRevision, taskRevision > 0,
              let taskConfiguration = metadata.taskConfiguration,
              taskConfiguration.keys.allSatisfy({ !$0.isEmpty }),
              taskConfiguration.values.allSatisfy(\.isFinite),
              metadata.observationDimension > 0,
              metadata.actionDimension > 0,
              metadata.simulationStep.isFinite,
              metadata.simulationStep > 0,
              metadata.controlDecimation > 0,
              metadata.maxEpisodeSteps > 0,
              metadata.inferenceBatchSize.map({ $0 > 0 }) ?? true else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy checkpoint metadata structure is invalid")
        }
        let controlPeriod = metadata.simulationStep
            * Float(metadata.controlDecimation)
        let controlFrequency = 1 / controlPeriod
        guard controlPeriod.isFinite, controlPeriod > 0,
              controlFrequency.isFinite, controlFrequency > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy checkpoint control timing is invalid")
        }

        let ppo = metadata.ppo
        let hiddenDimensions = ppo.resolvedHiddenDimensions
        guard ppo.hiddenSize > 0,
              hiddenDimensions.count == 3,
              hiddenDimensions.allSatisfy({ $0 > 0 }),
              ppo.initialActionStd.isFinite,
              ppo.initialActionStd > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy checkpoint PPO inference configuration is invalid")
        }

        let normalizer = metadata.normalizer
        let priorWeight = max(normalizer.count - 1, 0)
        guard normalizer.count.isFinite, normalizer.count >= 0,
              normalizer.mean.count == metadata.observationDimension,
              normalizer.variance.count == metadata.observationDimension,
              normalizer.mean.allSatisfy(\.isFinite),
              normalizer.variance.allSatisfy({
                $0.isFinite && $0 >= 0
                    && ($0 * priorWeight).isFinite
              }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy checkpoint observation normalizer is invalid")
        }

        // Validate both the recorded payload and the runtime-expanded model
        // geometry before their dimensions reach MLX allocation or reshape.
        _ = try policyElementCount(
            metadata: metadata,
            actorBranchCount: try recordedActorBranchCount(metadata))
        _ = try policyElementCount(metadata: metadata, actorBranchCount: 4)
        let inferenceRows = metadata.inferenceBatchSize ?? 1
        _ = try checkedProduct(
            inferenceRows, metadata.observationDimension,
            context: "policy inference observation geometry")
        _ = try checkedProduct(
            inferenceRows, metadata.actionDimension,
            context: "policy inference action geometry")
    }

    private static func checkedProduct(
        _ lhs: Int, _ rhs: Int, context: String
    ) throws -> Int {
        guard lhs >= 0, rhs >= 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "\(context) contains a negative dimension")
        }
        let result = lhs.multipliedReportingOverflow(by: rhs)
        guard !result.overflow else {
            throw RLEnvironmentError.invalidConfiguration(
                "\(context) overflows Int")
        }
        return result.partialValue
    }

    private static func checkedSum(
        _ lhs: Int, _ rhs: Int, context: String
    ) throws -> Int {
        let result = lhs.addingReportingOverflow(rhs)
        guard !result.overflow else {
            throw RLEnvironmentError.invalidConfiguration(
                "\(context) overflows Int")
        }
        return result.partialValue
    }

    private static func recordedActorBranchCount(
        _ metadata: VectorPolicyMetadata
    ) throws -> Int {
        guard let architectureVersion = metadata.architectureVersion else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy checkpoint architecture version is missing")
        }
        switch architectureVersion {
        case 3: return 1
        case 4: return 2
        case 5: return 3
        case VectorActorCritic.architectureVersion: return 4
        default:
            throw RLEnvironmentError.invalidConfiguration(
                "policy checkpoint architecture is not replayable")
        }
    }

    private static func policyElementCount(
        metadata: VectorPolicyMetadata, actorBranchCount: Int
    ) throws -> Int {
        let hidden = metadata.ppo.resolvedHiddenDimensions
        func networkElementCount(outputDimension: Int) throws -> Int {
            let products = [
                try checkedProduct(
                    hidden[0], metadata.observationDimension,
                    context: "policy input-layer geometry"),
                hidden[0],
                try checkedProduct(
                    hidden[1], hidden[0],
                    context: "policy middle-layer geometry"),
                hidden[1],
                try checkedProduct(
                    hidden[2], hidden[1],
                    context: "policy final-layer geometry"),
                hidden[2],
                try checkedProduct(
                    outputDimension, hidden[2],
                    context: "policy output-layer geometry"),
                outputDimension,
            ]
            var result = 0
            for elements in products {
                result = try checkedSum(
                    result, elements,
                    context: "policy network element count")
            }
            return result
        }

        let actorElements = try networkElementCount(
            outputDimension: metadata.actionDimension)
        var total = try checkedProduct(
            actorElements, actorBranchCount,
            context: "policy actor element count")
        total = try checkedSum(
            total, try networkElementCount(outputDimension: 1),
            context: "policy actor/critic element count")
        return try checkedSum(
            total, metadata.actionDimension,
            context: "policy total element count")
    }

    private static func expectedPolicyPayloadByteCount(
        metadata: VectorPolicyMetadata
    ) throws -> Int {
        let totalElements = try policyElementCount(
            metadata: metadata,
            actorBranchCount: try recordedActorBranchCount(metadata))
        return try checkedProduct(
            totalElements, MemoryLayout<Float>.stride,
            context: "policy tensor payload byte count")
    }

    private static func validatePolicyTensors(
        _ tensors: [String: MLXArray]
    ) throws {
        guard !tensors.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy checkpoint contains no tensors")
        }
        for (name, tensor) in tensors.sorted(by: { $0.key < $1.key }) {
            guard tensor.dtype == .float32 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "policy checkpoint tensor \(name) must use float32")
            }
            guard tensor.asArray(Float.self).allSatisfy(\.isFinite) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "policy checkpoint tensor \(name) contains non-finite values")
            }
        }
    }

    /// Validate task-provided routing entirely on the host before constructing
    /// any MLX gate or mask arrays. The base actor coefficient must remain
    /// non-negative for every row/action pair, making the routed actor mean a
    /// convex composition rather than an accidental extrapolation.
    static func validateRouting(
        rowCount: Int,
        actionDimension: Int,
        expertGates: ContiguousArray<Float>?,
        expertActionMask: ContiguousArray<Float>?,
        standExpertGates: ContiguousArray<Float>?,
        standExpertActionMask: ContiguousArray<Float>?,
        auxiliaryExpertGates: ContiguousArray<Float>?,
        auxiliaryExpertActionMask: ContiguousArray<Float>?
    ) throws {
        guard rowCount > 0, actionDimension > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "policy routing dimensions must be positive")
        }
        _ = try checkedProduct(
            rowCount, actionDimension,
            context: "policy routing geometry")
        let routes: [(
            name: String,
            gates: ContiguousArray<Float>?,
            mask: ContiguousArray<Float>?
        )] = [
            ("expert", expertGates, expertActionMask),
            ("stand expert", standExpertGates, standExpertActionMask),
            ("auxiliary expert", auxiliaryExpertGates,
             auxiliaryExpertActionMask),
        ]
        for route in routes {
            if let gates = route.gates {
                guard gates.count == rowCount,
                      gates.allSatisfy({
                        $0.isFinite && (0...1).contains($0)
                      }) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "policy \(route.name) gates must match observation "
                            + "rows and contain finite values in [0, 1]")
                }
            }
            if let mask = route.mask {
                guard mask.count == actionDimension,
                      mask.allSatisfy({
                        $0.isFinite && (0...1).contains($0)
                      }) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "policy \(route.name) action mask must match action "
                            + "dimensions and contain finite values in [0, 1]")
                }
            }
        }

        guard routes.contains(where: { $0.gates != nil }) else { return }
        for row in 0..<rowCount {
            for action in 0..<actionDimension {
                var routedWeight: Float = 0
                for route in routes {
                    let gate = route.gates?[row] ?? 0
                    let mask = route.mask?[action] ?? 1
                    routedWeight += gate * mask
                }
                guard routedWeight.isFinite, routedWeight <= 1 else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "policy expert routing is not a convex actor "
                            + "composition at row \(row), action \(action)")
                }
            }
        }
    }

    public func actions(
        for observation: RLObservationBatch,
        expertGates: ContiguousArray<Float>? = nil,
        expertActionMask: ContiguousArray<Float>? = nil,
        standExpertGates: ContiguousArray<Float>? = nil,
        standExpertActionMask: ContiguousArray<Float>? = nil,
        auxiliaryExpertGates: ContiguousArray<Float>? = nil,
        auxiliaryExpertActionMask: ContiguousArray<Float>? = nil
    ) throws -> RLActionBatch {
        let values = try actions(
            for: observation.policy, expertGates: expertGates,
            expertActionMask: expertActionMask,
            standExpertGates: standExpertGates,
            standExpertActionMask: standExpertActionMask,
            auxiliaryExpertGates: auxiliaryExpertGates,
            auxiliaryExpertActionMask: auxiliaryExpertActionMask)
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
        expertActionMask: ContiguousArray<Float>? = nil,
        standExpertGates: ContiguousArray<Float>? = nil,
        standExpertActionMask: ContiguousArray<Float>? = nil,
        auxiliaryExpertGates: ContiguousArray<Float>? = nil,
        auxiliaryExpertActionMask: ContiguousArray<Float>? = nil
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
        let actionValueCount = try Self.checkedProduct(
            n, metadata.actionDimension,
            context: "policy output geometry")
        try Self.validateRouting(
            rowCount: n, actionDimension: metadata.actionDimension,
            expertGates: expertGates, expertActionMask: expertActionMask,
            standExpertGates: standExpertGates,
            standExpertActionMask: standExpertActionMask,
            auxiliaryExpertGates: auxiliaryExpertGates,
            auxiliaryExpertActionMask: auxiliaryExpertActionMask)
        // A checkpoint records the Metal matrix row geometry validated during
        // training. Padding smaller calls handles UI/single-environment replay;
        // split larger calls into the same exact row geometry as well. Running
        // an arbitrary larger matrix can select a different reduction kernel
        // and, for contact-sensitive policies, change the physical trajectory.
        if let recordedRows = metadata.inferenceBatchSize,
           recordedRows > 0, n > recordedRows {
            var combined = ContiguousArray<Float>()
            combined.reserveCapacity(actionValueCount)
            for range in Self.inferenceBatchRanges(
                rowCount: n, recordedBatchSize: recordedRows) {
                let observationStart = range.lowerBound
                    * metadata.observationDimension
                let observationEnd = range.upperBound
                    * metadata.observationDimension
                let chunkObservations = ContiguousArray(
                    policyObservations[observationStart..<observationEnd])
                let chunkExpertGates = expertGates.map {
                    ContiguousArray($0[range])
                }
                let chunkStandGates = standExpertGates.map {
                    ContiguousArray($0[range])
                }
                let chunkAuxiliaryGates = auxiliaryExpertGates.map {
                    ContiguousArray($0[range])
                }
                combined.append(contentsOf: try actions(
                    for: chunkObservations,
                    expertGates: chunkExpertGates,
                    expertActionMask: expertActionMask,
                    standExpertGates: chunkStandGates,
                    standExpertActionMask: standExpertActionMask,
                    auxiliaryExpertGates: chunkAuxiliaryGates,
                    auxiliaryExpertActionMask: auxiliaryExpertActionMask))
            }
            return combined
        }
        let inferenceRows = max(n, metadata.inferenceBatchSize ?? n)
        let inferenceObservationCount = try Self.checkedProduct(
            inferenceRows, metadata.observationDimension,
            context: "policy inference observation geometry")
        _ = try Self.checkedProduct(
            inferenceRows, metadata.actionDimension,
            context: "policy inference action geometry")
        var inferenceObservations = policyObservations
        if inferenceRows > n {
            let lastRow = policyObservations.suffix(metadata.observationDimension)
            inferenceObservations.reserveCapacity(inferenceObservationCount)
            for _ in n..<inferenceRows {
                inferenceObservations.append(contentsOf: lastRow)
            }
        }
        let normalized = metadata.ppo.normalizeObservations
            ? normalizer.normalize(inferenceObservations) : inferenceObservations
        let input = MLXArray(Array(normalized)).reshaped(
            [inferenceRows, metadata.observationDimension])
        let gate = expertGates.map {
            let values = Self.paddedGate($0, rows: inferenceRows)
            return MLXArray(Array(values)).reshaped([inferenceRows, 1])
        }
        let expertMask = expertActionMask.map {
            MLXArray(Array($0)).reshaped([1, metadata.actionDimension])
        }
        let standGate = standExpertGates.map {
            let values = Self.paddedGate($0, rows: inferenceRows)
            return MLXArray(Array(values)).reshaped([inferenceRows, 1])
        }
        let standMask = standExpertActionMask.map {
            MLXArray(Array($0)).reshaped([1, metadata.actionDimension])
        }
        let auxiliaryGate = auxiliaryExpertGates.map {
            let values = Self.paddedGate($0, rows: inferenceRows)
            return MLXArray(Array(values)).reshaped([inferenceRows, 1])
        }
        let auxiliaryMask = auxiliaryExpertActionMask.map {
            MLXArray(Array($0)).reshaped([1, metadata.actionDimension])
        }
        let output = metadata.ppo.resolvedActionDistribution.environmentAction(
            policy.forward(
                input, expertGate: gate, expertActionMask: expertMask,
                standExpertGate: standGate,
                standExpertActionMask: standMask,
                auxiliaryExpertGate: auxiliaryGate,
                auxiliaryExpertActionMask: auxiliaryMask).mean)
        eval(output)
        let allValues = output.asArray(Float.self)
        let values = ContiguousArray(
            allValues.prefix(actionValueCount))
        if let index = values.firstIndex(where: { !$0.isFinite }) {
            throw RLEnvironmentError.nonFiniteAction(index: index)
        }
        return values
    }

    static func inferenceBatchRanges(
        rowCount: Int, recordedBatchSize: Int
    ) -> [Range<Int>] {
        precondition(rowCount > 0 && recordedBatchSize > 0)
        var ranges = [Range<Int>]()
        for start in stride(from: 0, to: rowCount, by: recordedBatchSize) {
            ranges.append(start..<min(start + recordedBatchSize, rowCount))
        }
        return ranges
    }

    private static func paddedGate(
        _ values: ContiguousArray<Float>, rows: Int
    ) -> ContiguousArray<Float> {
        guard values.count < rows, let last = values.last else { return values }
        var padded = values
        padded.reserveCapacity(rows)
        padded.append(contentsOf: repeatElement(last, count: rows - values.count))
        return padded
    }
}

extension VectorPolicyRunner: RLActionProvider {
    public var actionProviderID: String { "mlx-vector-policy" }

    public func actions(
        for observation: RLObservationBatch,
        task: any VectorizedRLTask
    ) throws -> RLActionBatch {
        try observation.validate(for: task.spec)
        let mismatches = metadata.compatibilityMismatches(with: task.spec)
        guard mismatches.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "checkpoint/task mismatch: \(mismatches.joined(separator: "; "))")
        }
        let expertGates =
            (task as? any PolicyExpertGateProviding)?
                .policyExpertGates(observation.policy)
        let expertActionMask =
            (task as? any PolicyExpertGateProviding)?
                .policyExpertActionMask
        let standExpertGates =
            (task as? any PolicyStandExpertGateProviding)?
                .policyStandExpertGates(observation.policy)
        let standExpertActionMask =
            (task as? any PolicyStandExpertGateProviding)?
                .policyStandExpertActionMask
        let auxiliaryExpertGates =
            (task as? any PolicyAuxiliaryExpertGateProviding)?
                .policyAuxiliaryExpertGates(observation.policy)
        let auxiliaryExpertActionMask =
            (task as? any PolicyAuxiliaryExpertGateProviding)?
                .policyAuxiliaryExpertActionMask
        let batch = try actions(
            for: observation, expertGates: expertGates,
            expertActionMask: expertActionMask,
            standExpertGates: standExpertGates,
            standExpertActionMask: standExpertActionMask,
            auxiliaryExpertGates: auxiliaryExpertGates,
            auxiliaryExpertActionMask: auxiliaryExpertActionMask)
        try batch.validate(for: task.spec)
        return batch
    }
}

public final class VectorPPOTrainer {
    public let configuration: VectorPPOConfig
    public var onUpdate: ((PPOUpdateMetrics) -> Void)?

    private static let logSqrt2Pi: Float = 0.9189385332
    static let successReplayFileName = "success-replay.safetensors"

    /// Validate untrusted host values before they enter the in-memory replay
    /// ring. Shape-only safetensors discovery cannot detect NaNs or invalid
    /// routing gates without reading the payload.
    static func validateSuccessReplayCheckpointValues(
        observations: [Float],
        actions: [Float],
        expertGates: [Float],
        standExpertGates: [Float],
        auxiliaryExpertGates: [Float],
        rowCount: Int,
        observationDimension: Int,
        actionDimension: Int
    ) throws {
        guard rowCount >= 0, observationDimension > 0,
              actionDimension > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "success replay checkpoint dimensions are invalid")
        }
        let observationCount = rowCount.multipliedReportingOverflow(
            by: observationDimension)
        let actionCount = rowCount.multipliedReportingOverflow(
            by: actionDimension)
        guard !observationCount.overflow, !actionCount.overflow,
              observations.count == observationCount.partialValue,
              actions.count == actionCount.partialValue,
              expertGates.count == rowCount,
              standExpertGates.count == rowCount,
              auxiliaryExpertGates.count == rowCount else {
            throw RLEnvironmentError.invalidConfiguration(
                "success replay checkpoint payload sizes are invalid")
        }
        guard observations.allSatisfy(\.isFinite),
              actions.allSatisfy(\.isFinite) else {
            throw RLEnvironmentError.invalidConfiguration(
                "success replay observations and actions must be finite")
        }
        let validGate: (Float) -> Bool = {
            $0.isFinite && (0...1).contains($0)
        }
        guard expertGates.allSatisfy(validGate),
              standExpertGates.allSatisfy(validGate),
              auxiliaryExpertGates.allSatisfy(validGate) else {
            throw RLEnvironmentError.invalidConfiguration(
                "success replay routing gates must be finite and in [0, 1]")
        }
    }
    /// A mathematically equivalent PPO ratio can overflow before clipping
    /// when a minibatch contains an action that became very unlikely under
    /// the updated policy. Bounding the *log* ratio keeps both the loss and
    /// its diagnostics finite without changing the ordinary PPO trust region
    /// (exp(+-20) is already far outside the configured 0.8...1.2 clip).
    static let maximumImportanceLogRatio: Float = 20

    /// Host-side trust-region policy used by the training loop and regression
    /// tests. Keeping the reference semantics here prevents presets from
    /// silently sharing a numeric target while doing different updates.
    static func shouldStopForKL(
        minibatchKL: Float, targetKL: Float, schedule: PPOKLSchedule
    ) -> Bool {
        guard targetKL > 0, minibatchKL.isFinite else { return false }
        switch schedule {
        case .adaptive: return minibatchKL > 4 * targetKL
        case .earlyStop: return minibatchKL > targetKL
        case .none: return false
        }
    }

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

    static func imitationSegmentStart(
        episodeStart: Int, milestoneStep: Int, historySteps: Int?
    ) -> Int {
        precondition(episodeStart >= 0 && milestoneStep >= episodeStart)
        guard let historySteps else { return episodeStart }
        precondition(historySteps > 0)
        return max(episodeStart, milestoneStep - historySteps + 1)
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
    static func hasValidActorComposition(
        expertGates: [Float], standExpertGates: [Float],
        actionDimension: Int,
        expertActionMask: [Float]?, standExpertActionMask: [Float]?,
        auxiliaryExpertGates: [Float]? = nil,
        auxiliaryExpertActionMask: [Float]? = nil
    ) -> Bool {
        let auxiliaryGates = auxiliaryExpertGates
            ?? [Float](repeating: 0, count: expertGates.count)
        guard expertGates.count == standExpertGates.count,
              expertGates.count == auxiliaryGates.count,
              actionDimension > 0,
              expertActionMask == nil
                || expertActionMask?.count == actionDimension,
              standExpertActionMask == nil
                || standExpertActionMask?.count == actionDimension,
              auxiliaryExpertActionMask == nil
                || auxiliaryExpertActionMask?.count == actionDimension,
              expertGates.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
              standExpertGates.allSatisfy({
                  $0.isFinite && (0...1).contains($0)
              }), auxiliaryGates.allSatisfy({
                  $0.isFinite && (0...1).contains($0)
              }) else { return false }
        for row in expertGates.indices {
            for action in 0..<actionDimension {
                let expert = expertGates[row]
                    * (expertActionMask?[action] ?? 1)
                let stand = standExpertGates[row]
                    * (standExpertActionMask?[action] ?? 1)
                let auxiliary = auxiliaryGates[row]
                    * (auxiliaryExpertActionMask?[action] ?? 1)
                if expert + stand + auxiliary > 1 + 1e-6 { return false }
            }
        }
        return true
    }

    static func actorTrainingWeights(
        expertGates: [Float], standExpertGates: [Float],
        actionDimension: Int = 1,
        expertActionMask: [Float]? = nil,
        standExpertActionMask: [Float]? = nil,
        freezesBaseActor: Bool, freezesExpertActor: Bool,
        auxiliaryExpertGates: [Float]? = nil,
        auxiliaryExpertActionMask: [Float]? = nil,
        freezesStandActor: Bool = false,
        freezesAuxiliaryActor: Bool = false
    ) -> [Float] {
        let auxiliaryGates = auxiliaryExpertGates
            ?? [Float](repeating: 0, count: expertGates.count)
        precondition(expertGates.count == standExpertGates.count)
        precondition(expertGates.count == auxiliaryGates.count)
        precondition(actionDimension > 0)
        precondition(expertActionMask == nil
            || expertActionMask?.count == actionDimension)
        precondition(standExpertActionMask == nil
            || standExpertActionMask?.count == actionDimension)
        precondition(auxiliaryExpertActionMask == nil
            || auxiliaryExpertActionMask?.count == actionDimension)
        return expertGates.indices.map { row in
            let expert = expertGates[row]
            let stand = standExpertGates[row]
            let auxiliary = auxiliaryGates[row]
            var trainable: Float = 0
            for action in 0..<actionDimension {
                let expertContribution = expert
                    * (expertActionMask?[action] ?? 1)
                let standContribution = stand
                    * (standExpertActionMask?[action] ?? 1)
                let auxiliaryContribution = auxiliary
                    * (auxiliaryExpertActionMask?[action] ?? 1)
                let baseContribution = max(
                    1 - expertContribution - standContribution
                        - auxiliaryContribution, 0)
                trainable += (freezesBaseActor ? 0 : baseContribution)
                    + (freezesExpertActor ? 0 : expertContribution)
                    + (freezesStandActor ? 0 : standContribution)
                    + (freezesAuxiliaryActor ? 0 : auxiliaryContribution)
            }
            return min(max(trainable / Float(actionDimension), 0), 1)
        }
    }

    /// Per-action exploration scale for a routed actor composition. Frozen
    /// experts must be frozen in the physical rollout as well as in the
    /// gradient graph: injecting motor noise into their exclusive dimensions
    /// changes the state distribution even though no branch can learn from
    /// those rows. Partial actor blends receive proportional standard
    /// deviation, matching their proportional influence on the composed mean.
    static func actorExplorationActionScales(
        expertGates: [Float], standExpertGates: [Float],
        actionDimension: Int, expertActionMask: [Float]? = nil,
        standExpertActionMask: [Float]?,
        freezesBaseActor: Bool, freezesExpertActor: Bool,
        auxiliaryExpertGates: [Float]? = nil,
        auxiliaryExpertActionMask: [Float]? = nil,
        freezesStandActor: Bool = false,
        freezesAuxiliaryActor: Bool = false
    ) -> [Float] {
        let auxiliaryGates = auxiliaryExpertGates
            ?? [Float](repeating: 0, count: expertGates.count)
        precondition(expertGates.count == standExpertGates.count)
        precondition(expertGates.count == auxiliaryGates.count)
        precondition(actionDimension > 0)
        precondition(expertActionMask == nil
            || expertActionMask?.count == actionDimension)
        precondition(standExpertActionMask == nil
            || standExpertActionMask?.count == actionDimension)
        precondition(auxiliaryExpertActionMask == nil
            || auxiliaryExpertActionMask?.count == actionDimension)
        var scales = [Float](
            repeating: 0, count: expertGates.count * actionDimension)
        for row in expertGates.indices {
            let expert = expertGates[row]
            let stand = standExpertGates[row]
            let auxiliary = auxiliaryGates[row]
            for action in 0..<actionDimension {
                let standMask = standExpertActionMask?[action] ?? 1
                let expertMask = expertActionMask?[action] ?? 1
                let expertContribution = expert * expertMask
                let standContribution = stand * standMask
                let auxiliaryContribution = auxiliary
                    * (auxiliaryExpertActionMask?[action] ?? 1)
                let baseContribution = max(
                    1 - expertContribution - standContribution
                        - auxiliaryContribution, 0)
                let trainableContribution =
                    (freezesBaseActor ? 0 : baseContribution)
                    + (freezesExpertActor ? 0 : expertContribution)
                    + (freezesStandActor ? 0 : standContribution)
                    + (freezesAuxiliaryActor ? 0 : auxiliaryContribution)
                scales[row * actionDimension + action] = min(max(
                    trainableContribution, 0), 1)
            }
        }
        return scales
    }

    /// Exactly frozen routed dimensions are Dirac actions, not narrow
    /// Gaussians. They must not participate in a joint PPO likelihood or a
    /// shared log-standard-deviation update can dominate the importance ratio
    /// despite having no effect on the environment action.
    static func likelihoodActionMask(
        explorationActionScales: [Float]
    ) -> [Float] {
        explorationActionScales.map { $0 > 0 ? 1 : 0 }
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

    /// Make the append-only training log agree with the last durable
    /// checkpoint before an exact resume. Metrics are emitted before a
    /// checkpoint is saved, so an interrupt can legitimately leave rows for
    /// updates whose policy and optimizer state were never committed. Keeping
    /// those rows would create duplicate update numbers on the next resume and
    /// make experiment selection ambiguous.
    static func reconcileMetricsLog(
        at url: URL, completedUpdates: Int
    ) throws {
        precondition(completedUpdates >= 0)
        struct MetricIdentity: Decodable { var update: Int }

        guard FileManager.default.fileExists(atPath: url.path) else {
            try Data().write(to: url, options: .atomic)
            return
        }
        let original = try Data(contentsOf: url)
        var reconciled = Data()
        var previousUpdate = -1
        for rawLine in original.split(separator: 0x0A,
                                      omittingEmptySubsequences: true) {
            let line = Data(rawLine)
            guard let identity = try? JSONDecoder().decode(
                    MetricIdentity.self, from: line),
                  identity.update >= 0,
                  identity.update < completedUpdates,
                  identity.update > previousUpdate else { continue }
            reconciled.append(line)
            reconciled.append(0x0A)
            previousUpdate = identity.update
        }
        if reconciled != original {
            try reconciled.write(to: url, options: .atomic)
        }
    }

    /// Serialize the ring in chronological order. Keeping the file independent
    /// of the in-memory write cursor lets a transfer retain the newest rows
    /// even when its replay capacity differs from the source run.
    private static func successReplayCheckpointArrays(
        observations: [Float], actions: [Float], expertGates: [Float],
        standExpertGates: [Float], auxiliaryExpertGates: [Float],
        count: Int, next: Int, capacity: Int,
        observationDimension: Int, actionDimension: Int
    ) -> [String: MLXArray]? {
        guard count > 0, capacity > 0 else { return nil }
        precondition(count <= capacity && (0..<capacity).contains(next))
        let oldest = count == capacity ? next : 0
        var orderedObservations = [Float]()
        var orderedActions = [Float]()
        var orderedExpertGates = [Float]()
        var orderedStandExpertGates = [Float]()
        var orderedAuxiliaryExpertGates = [Float]()
        orderedObservations.reserveCapacity(count * observationDimension)
        orderedActions.reserveCapacity(count * actionDimension)
        orderedExpertGates.reserveCapacity(count)
        orderedStandExpertGates.reserveCapacity(count)
        orderedAuxiliaryExpertGates.reserveCapacity(count)
        for offset in 0..<count {
            let slot = (oldest + offset) % capacity
            orderedObservations.append(contentsOf: observations[
                (slot * observationDimension)..<((slot + 1) * observationDimension)])
            orderedActions.append(contentsOf: actions[
                (slot * actionDimension)..<((slot + 1) * actionDimension)])
            orderedExpertGates.append(expertGates[slot])
            orderedStandExpertGates.append(standExpertGates[slot])
            orderedAuxiliaryExpertGates.append(auxiliaryExpertGates[slot])
        }
        return [
            "observations": MLXArray(orderedObservations)
                .reshaped([count, observationDimension]),
            "actions": MLXArray(orderedActions)
                .reshaped([count, actionDimension]),
            "expertGates": MLXArray(orderedExpertGates),
            "standExpertGates": MLXArray(orderedStandExpertGates),
            "auxiliaryExpertGates": MLXArray(orderedAuxiliaryExpertGates),
        ]
    }

    public func train(task: any VectorizedRLTask, outputDirectory: String) throws {
        try train(task: task, outputDirectory: outputDirectory, resume: false)
    }

    public func train(task: any VectorizedRLTask, outputDirectory: String,
                      resume: Bool) throws {
        try FileManager.default.createDirectory(
            atPath: outputDirectory, withIntermediateDirectories: true)
        let runLock = try VectorPPORunLock(runDirectory: outputDirectory)
        defer { runLock.unlock() }
        if resume && (configuration.initializationCheckpoint != nil
            || configuration.policyExpertInitializationCheckpoint != nil
            || configuration.policyExpertBranchInitializationCheckpoint != nil
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
        let expertActionMask = usesPolicyExpertGate
            ? policyExpertGate?.policyExpertActionMask : nil
        if let expertActionMask {
            guard expertActionMask.count == actionDim,
                  expertActionMask.allSatisfy({ (0...1).contains($0) }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task expert action mask must match action dimensions")
            }
        }
        let expertActionMaskArray = expertActionMask.map {
            MLXArray(Array($0)).reshaped([1, actionDim])
        }
        let policyStandExpertGate = task as? any PolicyStandExpertGateProviding
        let usesPolicyStandExpertGate =
            policyStandExpertGate?.usesPolicyStandExpertGate == true
        let standExpertActionMask = usesPolicyStandExpertGate
            ? policyStandExpertGate?.policyStandExpertActionMask : nil
        if let standExpertActionMask {
            guard standExpertActionMask.count == actionDim,
                  standExpertActionMask.allSatisfy({ (0...1).contains($0) }) else {
                throw RLEnvironmentError.invalidConfiguration(
                "task stand-expert action mask must match action dimensions")
            }
        }
        let policyAuxiliaryExpertGate = task as?
            any PolicyAuxiliaryExpertGateProviding
        let usesPolicyAuxiliaryExpertGate =
            policyAuxiliaryExpertGate?.usesPolicyAuxiliaryExpertGate == true
        let auxiliaryExpertActionMask = usesPolicyAuxiliaryExpertGate
            ? policyAuxiliaryExpertGate?.policyAuxiliaryExpertActionMask : nil
        if let auxiliaryExpertActionMask {
            guard auxiliaryExpertActionMask.count == spec.action.elementCount,
                  auxiliaryExpertActionMask.allSatisfy({
                      (0...1).contains($0)
                  }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task auxiliary-expert action mask must match action dimensions")
            }
        }
        if let expertMask = expertActionMask,
           let standMask = standExpertActionMask {
            guard zip(expertMask, standMask).allSatisfy({
                $0.0 + $0.1 <= 1 + 1e-6
            }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task expert action masks overlap")
            }
        }
        let standExpertActionMaskArray = standExpertActionMask.map {
            MLXArray(Array($0)).reshaped([1, actionDim])
        }
        if let standMask = standExpertActionMask,
           let auxiliaryMask = auxiliaryExpertActionMask {
            guard zip(standMask, auxiliaryMask).allSatisfy({
                $0.0 + $0.1 <= 1 + 1e-6
            }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task stand and auxiliary expert action masks overlap")
            }
        }
        let auxiliaryExpertActionMaskArray = auxiliaryExpertActionMask.map {
            MLXArray(Array($0)).reshaped([1, actionDim])
        }
        let freezesBasePolicyExpert = usesPolicyExpertGate
            && policyExpertGate?.freezesBasePolicyExpert == true
        let freezesLowSpeedPolicyExpert = usesPolicyStandExpertGate
            && policyStandExpertGate?.freezesLowSpeedPolicyExpert == true
        let freezesStandPolicyExpert = usesPolicyAuxiliaryExpertGate
            && policyAuxiliaryExpertGate?.freezesStandPolicyExpert == true
        let symmetryMirrorLossCoefficient =
            configuration.symmetryMirrorLossCoefficient ?? 0
        let successImitationCoefficient =
            configuration.successImitationCoefficient ?? 0
        let successReplayCapacity = configuration.successReplayCapacity ?? 0
        let successReplayBatchSize = configuration.successReplayBatchSize ?? 0
        let referencePolicyCoefficient =
            configuration.referencePolicyCoefficient ?? 0
        let referenceRegularizationProvider = task as?
            any PolicyReferenceRegularizationProviding
        let referenceActionRegularizationProvider = task as?
            any PolicyReferenceActionRegularizationProviding
        let actorTrainingWeightProvider = task as?
            any PolicyActorTrainingWeightProviding
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
        let metricsURL = URL(fileURLWithPath: "\(outputDirectory)/metrics.jsonl")
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
                                       initialActionStd: configuration.initialActionStd,
                                       activation: configuration.resolvedActivation,
                                       orthogonalInitialization: configuration
                                           .resolvedOrthogonalInitialization,
                                       actorOutputGain: configuration
                                           .resolvedActorOutputGain,
                                       initializationSeed: configuration.seed)
        var normalizer = RunningObservationNormalizer(dimension: obsDim)
        var startingUpdate = 0
        var totalSteps = 0
        var restoredTrainingState: VectorPPOTrainingState?
        var restoredSuccessReplayCount: Int?
        var persistedConfiguration = configuration
        var replayInitializationDirectory: String?
        var resumeCheckpointDirectory: String?
        // A transferred policy may be contact-sensitive to the Metal matrix
        // row geometry used when its behavior was certified. Preserve that
        // geometry during rollout instead of silently changing it to the new
        // simulator batch size. PPO minibatches remain independently batched.
        var policyInferenceBatchSize = n
        if resume {
            let resumeCheckpoint = try VectorPolicyCheckpointDiscovery
                .checkpointForResume(
                    inRunDirectory: outputDirectory, task: spec.id,
                    taskRevision: spec.revision,
                    numEnvironments: n)
            let checkpointDirectory = resumeCheckpoint.directory
            resumeCheckpointDirectory = checkpointDirectory
            let metadataURL = URL(
                fileURLWithPath: "\(checkpointDirectory)/metadata.json")
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
            if let recordedRows = metadata.inferenceBatchSize,
               recordedRows > 0 {
                policyInferenceBatchSize = recordedRows
            }
            replayInitializationDirectory = checkpointDirectory
            let sourceWeights = try loadArrays(url: URL(
                fileURLWithPath: "\(checkpointDirectory)/policy.safetensors"))
            let weights = try VectorActorCritic.compatibleWeights(
                sourceWeights, architectureVersion: metadata.architectureVersion)
            try policy.update(parameters: ModuleParameters.unflattened(weights),
                              verify: [.all])
            normalizer = RunningObservationNormalizer(snapshot: metadata.normalizer)
            let stateURL = URL(
                fileURLWithPath: "\(checkpointDirectory)/training-state.json")
            let state = try JSONDecoder().decode(
                VectorPPOTrainingState.self, from: Data(contentsOf: stateURL))
            let restoredEnvironmentCount = try state
                .validatedRolloutEnvironmentCount(
                    rolloutSteps: configuration.rolloutSteps)
            guard restoredEnvironmentCount == n else {
                throw RLEnvironmentError.invalidConfiguration(
                    "PPO resume environment count differs from its checkpoint")
            }
            let maximumOptimizerSteps = try state.maximumOptimizerSteps(
                rolloutSteps: configuration.rolloutSteps,
                updateEpochs: configuration.updateEpochs,
                minibatchSize: configuration.minibatchSize)
            _ = try state.validatedOptimizerResumeState(
                configuration: configuration,
                maximumOptimizerSteps: maximumOptimizerSteps)
            restoredSuccessReplayCount = try state
                .validatedSuccessReplayCount(capacity: successReplayCapacity)
            startingUpdate = state.completedUpdates
            totalSteps = state.environmentSteps
            restoredTrainingState = state
        } else if let checkpointDirectory = configuration.initializationCheckpoint {
            let metadataURL = URL(
                fileURLWithPath: "\(checkpointDirectory)/metadata.json")
            let metadata = try JSONDecoder().decode(
                VectorPolicyMetadata.self, from: Data(contentsOf: metadataURL))
            if let recordedRows = metadata.inferenceBatchSize,
               recordedRows > 0 {
                policyInferenceBatchSize = recordedRows
            }
            // Successful-transition replay is valid only for the exact task
            // contract that produced it. `--initialize-from` may deliberately
            // transfer network weights across curricula or expert-routing
            // changes, but reusing old replay rows would silently attach old
            // rewards, gates, and success semantics to the new experiment.
            if metadata.task == spec.id,
               (metadata.taskRevision ?? 1) == spec.revision,
               metadata.taskConfiguration == spec.configurationValues {
                replayInitializationDirectory = checkpointDirectory
            }
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
                let expertObservationSourceIndices =
                    expertMetadata.observationDimension == obsDim
                    ? nil
                    : (task as? any ObservationSchemaTransferProviding)?
                        .initializationObservationSourceIndices(
                            sourceDimension:
                                expertMetadata.observationDimension)
                guard (expertMetadata.observationDimension == obsDim
                        || expertObservationSourceIndices?.count == obsDim),
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
                var expertWeights = try VectorActorCritic.compatibleWeights(
                    expertSourceWeights,
                    architectureVersion: expertMetadata.architectureVersion)
                var expertNormalizer = expertMetadata.normalizer
                if let expertObservationSourceIndices {
                    expertWeights = try VectorActorCritic
                        .remappingObservationInputs(
                            expertWeights,
                            sourceIndices: expertObservationSourceIndices)
                    expertNormalizer = try expertNormalizer
                        .remappingObservationChannels(
                            sourceIndices: expertObservationSourceIndices)
                }
                weights = try VectorActorCritic.initializingPolicyExpert(
                    weights, from: expertWeights,
                    sourceNormalizer: expertNormalizer,
                    destinationNormalizer: importedNormalizer,
                    sourceNormalizesObservations:
                        expertMetadata.ppo.normalizeObservations,
                    destinationNormalizesObservations:
                        configuration.normalizeObservations)
                Swift.print("composed routed policy expert from \(expertCheckpoint) "
                    + "with exact observation-normalizer reparameterization"
                    + (expertObservationSourceIndices == nil ? ""
                        : " and schema remapping"))
            }
            if let expertCheckpoint =
                configuration.policyExpertBranchInitializationCheckpoint {
                guard usesPolicyExpertGate else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "routed-expert branch composition requires a routed task")
                }
                let expertMetadataURL = URL(
                    fileURLWithPath: "\(expertCheckpoint)/metadata.json")
                let expertMetadata = try JSONDecoder().decode(
                    VectorPolicyMetadata.self,
                    from: Data(contentsOf: expertMetadataURL))
                let expertObservationSourceIndices =
                    expertMetadata.observationDimension == obsDim
                    ? nil
                    : (task as? any ObservationSchemaTransferProviding)?
                        .initializationObservationSourceIndices(
                            sourceDimension:
                                expertMetadata.observationDimension)
                guard (expertMetadata.observationDimension == obsDim
                        || expertObservationSourceIndices?.count == obsDim),
                      expertMetadata.actionDimension == actionDim,
                      VectorActorCritic.compatibleArchitectureVersions
                        .contains(expertMetadata.architectureVersion ?? 1),
                      expertMetadata.ppo.resolvedHiddenDimensions
                        == configuration.resolvedHiddenDimensions,
                      expertMetadata.ppo.resolvedActionDistribution
                        == configuration.resolvedActionDistribution else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "routed-expert checkpoint has an incompatible network interface")
                }
                let sourceArrays = try loadArrays(url: URL(
                    fileURLWithPath: "\(expertCheckpoint)/policy.safetensors"))
                var expertWeights = try VectorActorCritic.compatibleWeights(
                    sourceArrays,
                    architectureVersion: expertMetadata.architectureVersion)
                var expertNormalizer = expertMetadata.normalizer
                if let expertObservationSourceIndices {
                    expertWeights = try VectorActorCritic
                        .remappingObservationInputs(
                            expertWeights,
                            sourceIndices: expertObservationSourceIndices)
                    expertNormalizer = try expertNormalizer
                        .remappingObservationChannels(
                            sourceIndices: expertObservationSourceIndices)
                }
                weights = try VectorActorCritic
                    .initializingPolicyExpertFromExpert(
                        weights, from: expertWeights,
                        sourceNormalizer: expertNormalizer,
                        destinationNormalizer: importedNormalizer,
                        sourceNormalizesObservations:
                            expertMetadata.ppo.normalizeObservations,
                        destinationNormalizesObservations:
                            configuration.normalizeObservations)
                Swift.print("composed existing routed expert branch from "
                    + "\(expertCheckpoint) with exact observation-normalizer "
                    + "reparameterization"
                    + (expertObservationSourceIndices == nil ? ""
                        : " and schema remapping"))
            }
            let initializesStandFromPolicyExpert = usesPolicyStandExpertGate
                && policyStandExpertGate?
                    .initializesPolicyStandExpertFromPolicyExpertOnTransfer == true
            let initializesStandFromBase = usesPolicyStandExpertGate
                && policyStandExpertGate?
                    .initializesPolicyStandExpertFromBaseOnTransfer == true
            guard !initializesStandFromPolicyExpert || !initializesStandFromBase else {
                throw RLEnvironmentError.invalidConfiguration(
                    "stand expert cannot initialize from both base and policy expert")
            }
            if initializesStandFromPolicyExpert {
                weights = try VectorActorCritic
                    .initializingStandExpertFromPolicyExpert(weights)
                Swift.print("initialized stand policy expert as an exact copy "
                    + "of the transferred routed expert")
            } else if initializesStandFromBase {
                weights = try VectorActorCritic
                    .initializingStandExpertFromBase(weights)
                Swift.print("initialized stand policy expert as an exact copy "
                    + "of the transferred base actor")
            }
            if usesPolicyAuxiliaryExpertGate,
               policyAuxiliaryExpertGate?
                    .initializesPolicyAuxiliaryExpertFromBaseOnTransfer == true {
                let projectedInputs = policyAuxiliaryExpertGate?
                    .policyAuxiliaryExpertZeroedObservationIndicesOnTransfer
                    ?? []
                weights = try VectorActorCritic
                    .initializingAuxiliaryExpertFromBase(
                        weights,
                        zeroedObservationIndices: projectedInputs)
                Swift.print("initialized auxiliary policy expert from the "
                    + "transferred base actor"
                    + (projectedInputs.isEmpty
                        ? " as an exact copy"
                        : " with \(projectedInputs.count) task-declared "
                            + "observation inputs projected out"))
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
                let standObservationSourceIndices =
                    standMetadata.observationDimension == obsDim
                    ? nil
                    : (task as? any ObservationSchemaTransferProviding)?
                        .initializationObservationSourceIndices(
                            sourceDimension: standMetadata.observationDimension)
                guard (standMetadata.observationDimension == obsDim
                        || standObservationSourceIndices?.count == obsDim),
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
                var standWeights = try VectorActorCritic.compatibleWeights(
                    standSourceWeights,
                    architectureVersion: standMetadata.architectureVersion)
                var standNormalizer = standMetadata.normalizer
                if let standObservationSourceIndices {
                    standWeights = try VectorActorCritic
                        .remappingObservationInputs(
                            standWeights,
                            sourceIndices: standObservationSourceIndices)
                    standNormalizer = try standNormalizer
                        .remappingObservationChannels(
                            sourceIndices: standObservationSourceIndices)
                }
                weights = try VectorActorCritic.initializingStandExpert(
                    weights, from: standWeights,
                    sourceNormalizer: standNormalizer,
                    destinationNormalizer: importedNormalizer)
                Swift.print("composed stand expert from \(standCheckpoint) "
                    + "with exact observation-normalizer reparameterization"
                    + (standObservationSourceIndices == nil ? ""
                        : " and schema remapping"))
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
        if policyInferenceBatchSize != n {
            Swift.print("preserving transferred policy inference geometry at "
                + "\(policyInferenceBatchSize) rows across \(n) simulator "
                + "environments")
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
                initialActionStd: configuration.initialActionStd,
                activation: configuration.resolvedActivation)
            let weights: [String: MLXArray]
            if resume {
                guard let resumeCheckpointDirectory else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "PPO resume checkpoint was not resolved")
                }
                let url = URL(fileURLWithPath:
                    "\(resumeCheckpointDirectory)/reference-policy.safetensors")
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
        let optimizer = CheckpointableAdam(
            learningRate: configuration.learningRate,
            epsilon: configuration.resolvedOptimizerEpsilon)
        var adaptiveLearningRate = restoredTrainingState?.adaptiveLearningRate
            ?? configuration.learningRate
        if resume {
            guard let resumeCheckpointDirectory,
                  let restoredTrainingState else {
                throw RLEnvironmentError.invalidConfiguration(
                    "PPO resume checkpoint state was not resolved")
            }
            let optimizerURL = URL(
                fileURLWithPath:
                    "\(resumeCheckpointDirectory)/optimizer.safetensors")
            switch try restoredTrainingState.validatedOptimizerResumeState(
                configuration: configuration) {
            case .checkpointed(let optimizerSteps, _):
                guard FileManager.default.fileExists(atPath: optimizerURL.path)
                else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "current PPO checkpoint has \(optimizerSteps) optimizer "
                            + "steps but no optimizer.safetensors")
                }
                let arrays = try loadArrays(url: optimizerURL)
                try optimizer.restore(arrays: arrays, step: optimizerSteps,
                                      parameters: policy.parameters())
                Swift.print("resumed policy, normalizer, Adam moments, and KL scheduler "
                    + "from \(resumeCheckpointDirectory) at update \(startingUpdate), "
                    + "\(totalSteps) environment steps")
            case .fresh:
                Swift.print("resumed current policy and scheduler before the first "
                    + "Adam step from \(resumeCheckpointDirectory)")
            }
            Swift.print("resume restarts simulator/task environments from seed "
                + "\(configuration.seed); task state is not an exact trajectory "
                + "checkpoint")
        }
        var observation = try task.reset(seed: configuration.seed)
        // A task's success signal describes the current transition. Keep the
        // episode reduction here so fixed-horizon tasks can distinguish
        // success-once from success-at-end without changing the generic API.
        var episodeSucceeded = [Bool](repeating: false, count: n)
        var episodeSuccessImitationReached = [Bool](
            repeating: false, count: n)
        var stepResult = RLStepBatch(spec: spec)
        var successReplayObservations = [Float](
            repeating: 0, count: successReplayCapacity * obsDim)
        var successReplayActions = [Float](
            repeating: 0, count: successReplayCapacity * actionDim)
        var successReplayExpertGates = [Float](
            repeating: 0, count: successReplayCapacity)
        var successReplayStandExpertGates = [Float](
            repeating: 0, count: successReplayCapacity)
        var successReplayAuxiliaryExpertGates = [Float](
            repeating: 0, count: successReplayCapacity)
        var successReplayCount = 0
        var successReplayNext = 0
        if resume, successReplayCapacity > 0,
           replayInitializationDirectory == nil {
            throw RLEnvironmentError.invalidConfiguration(
                "PPO resume did not resolve its success replay checkpoint")
        }
        if successReplayCapacity > 0, let replayInitializationDirectory {
            let replayURL = URL(fileURLWithPath: replayInitializationDirectory)
                .appendingPathComponent(Self.successReplayFileName)
            let replayExists = FileManager.default.fileExists(
                atPath: replayURL.path)
            if resume {
                guard let restoredSuccessReplayCount else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "PPO resume checkpoint is missing success replay count")
                }
                guard (restoredSuccessReplayCount == 0) != replayExists else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "PPO resume success replay file presence does not match "
                            + "its recorded row count")
                }
            }
            if replayExists {
                let arrays = try loadArrays(url: replayURL)
                guard let replayObservations = arrays["observations"],
                      let replayActions = arrays["actions"],
                      let replayExpertGates = arrays["expertGates"],
                      let replayStandExpertGates = arrays["standExpertGates"],
                      replayObservations.shape.count == 2,
                      replayObservations.shape[1] == obsDim,
                      replayActions.shape == [replayObservations.shape[0], actionDim],
                      replayExpertGates.shape == [replayObservations.shape[0]],
                      replayStandExpertGates.shape
                        == [replayObservations.shape[0]] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "success replay checkpoint has incompatible tensor shapes")
                }
                let sourceCount = replayObservations.shape[0]
                let replayAuxiliaryExpertGates = arrays["auxiliaryExpertGates"]
                if resume,
                   replayAuxiliaryExpertGates?.shape != [sourceCount] {
                    throw RLEnvironmentError.invalidConfiguration(
                        "PPO resume success replay is missing its auxiliary gates")
                }
                if resume, sourceCount != restoredSuccessReplayCount {
                    throw RLEnvironmentError.invalidConfiguration(
                        "PPO resume success replay tensor rows do not match "
                            + "training-state.json")
                }
                let retainedCount = min(sourceCount, successReplayCapacity)
                let sourceStart = sourceCount - retainedCount
                let observationValues = replayObservations.asArray(Float.self)
                let actionValues = replayActions.asArray(Float.self)
                let expertValues = replayExpertGates.asArray(Float.self)
                let standExpertValues = replayStandExpertGates.asArray(Float.self)
                let auxiliaryExpertValues = replayAuxiliaryExpertGates
                    .map { $0.asArray(Float.self) }
                    ?? [Float](repeating: 0, count: sourceCount)
                try Self.validateSuccessReplayCheckpointValues(
                    observations: observationValues,
                    actions: actionValues,
                    expertGates: expertValues,
                    standExpertGates: standExpertValues,
                    auxiliaryExpertGates: auxiliaryExpertValues,
                    rowCount: sourceCount,
                    observationDimension: obsDim,
                    actionDimension: actionDim)
                if retainedCount > 0 {
                    successReplayObservations.replaceSubrange(
                        0..<(retainedCount * obsDim), with: observationValues[
                            (sourceStart * obsDim)..<(sourceCount * obsDim)])
                    successReplayActions.replaceSubrange(
                        0..<(retainedCount * actionDim), with: actionValues[
                            (sourceStart * actionDim)..<(sourceCount * actionDim)])
                    successReplayExpertGates.replaceSubrange(
                        0..<retainedCount,
                        with: expertValues[sourceStart..<sourceCount])
                    successReplayStandExpertGates.replaceSubrange(
                        0..<retainedCount,
                        with: standExpertValues[sourceStart..<sourceCount])
                    successReplayAuxiliaryExpertGates.replaceSubrange(
                        0..<retainedCount,
                        with: auxiliaryExpertValues[sourceStart..<sourceCount])
                }
                successReplayCount = retainedCount
                successReplayNext = retainedCount % successReplayCapacity
                Swift.print("restored \(retainedCount) successful transition "
                    + "rows from \(replayURL.path)")
            }
        }
        var actionLogStandardDeviationBounds = configuration
            .actionLogStandardDeviationBounds(
                completedUpdate: startingUpdate)
        let actionMirrorSources = MLXArray(
            (policySymmetry?.policyActionMirrorSourceIndices
                ?? Array(0..<actionDim)).map(Int32.init))
        let actionMirrorSigns = MLXArray(
            policySymmetry?.policyActionMirrorSigns
                ?? [Float](repeating: 1, count: actionDim))

        /// Run the behavior policy with the exact row geometry recorded by the
        /// source checkpoint. Smaller final chunks repeat their last row, just
        /// like `VectorPolicyRunner`, and only real rows are returned.
        func rolloutForward(
            observations: ContiguousArray<Float>,
            expertGates: ContiguousArray<Float>,
            standExpertGates: ContiguousArray<Float>,
            auxiliaryExpertGates: ContiguousArray<Float>
        ) -> (mean: MLXArray, value: MLXArray,
              logStandardDeviation: MLXArray) {
            precondition(observations.count == n * obsDim)
            precondition(expertGates.count == n)
            precondition(standExpertGates.count == n)
            precondition(auxiliaryExpertGates.count == n)
            var means = [MLXArray]()
            var values = [MLXArray]()
            var logStandardDeviation: MLXArray?
            for range in VectorPolicyRunner.inferenceBatchRanges(
                rowCount: n,
                recordedBatchSize: policyInferenceBatchSize) {
                let actualRows = range.count
                let observationStart = range.lowerBound * obsDim
                let observationEnd = range.upperBound * obsDim
                var chunkObservations = ContiguousArray(
                    observations[observationStart..<observationEnd])
                var chunkExpertGates = ContiguousArray(expertGates[range])
                var chunkStandExpertGates = ContiguousArray(
                    standExpertGates[range])
                var chunkAuxiliaryExpertGates = ContiguousArray(
                    auxiliaryExpertGates[range])
                if actualRows < policyInferenceBatchSize {
                    let lastObservation = ContiguousArray(
                        chunkObservations.suffix(obsDim))
                    for _ in actualRows..<policyInferenceBatchSize {
                        chunkObservations.append(contentsOf: lastObservation)
                    }
                    chunkExpertGates.append(contentsOf: repeatElement(
                        chunkExpertGates.last!,
                        count: policyInferenceBatchSize - actualRows))
                    chunkStandExpertGates.append(contentsOf: repeatElement(
                        chunkStandExpertGates.last!,
                        count: policyInferenceBatchSize - actualRows))
                    chunkAuxiliaryExpertGates.append(contentsOf: repeatElement(
                        chunkAuxiliaryExpertGates.last!,
                        count: policyInferenceBatchSize - actualRows))
                }
                let chunkOut = policy.forward(
                    MLXArray(Array(chunkObservations)).reshaped(
                        [policyInferenceBatchSize, obsDim]),
                    expertGate: MLXArray(Array(chunkExpertGates)).reshaped(
                        [policyInferenceBatchSize, 1]),
                    expertActionMask: expertActionMaskArray,
                    standExpertGate: MLXArray(Array(chunkStandExpertGates))
                        .reshaped([policyInferenceBatchSize, 1]),
                    standExpertActionMask: standExpertActionMaskArray,
                    auxiliaryExpertGate: MLXArray(
                        Array(chunkAuxiliaryExpertGates))
                        .reshaped([policyInferenceBatchSize, 1]),
                    auxiliaryExpertActionMask:
                        auxiliaryExpertActionMaskArray,
                    freezeBaseActor: freezesBasePolicyExpert,
                    freezeExpertActor: freezesLowSpeedPolicyExpert,
                    freezeStandActor: freezesStandPolicyExpert)
                if actualRows == policyInferenceBatchSize {
                    means.append(chunkOut.mean)
                    values.append(chunkOut.value)
                } else {
                    means.append(chunkOut.mean[0..<actualRows, 0...])
                    values.append(chunkOut.value[0..<actualRows])
                }
                if logStandardDeviation == nil {
                    logStandardDeviation = chunkOut.logStandardDeviation
                }
            }
            return (
                means.count == 1 ? means[0] : concatenated(means, axis: 0),
                values.count == 1 ? values[0] : concatenated(values, axis: 0),
                logStandardDeviation!)
        }

        func routedExplorationScales(
            expertGates: MLXArray, standExpertGates: MLXArray,
            auxiliaryExpertGates: MLXArray
        ) -> MLXArray {
            let expertContribution = expertActionMaskArray.map {
                expertGates * $0
            } ?? expertGates
            let standContribution = standExpertActionMaskArray.map {
                standExpertGates * $0
            } ?? standExpertGates
            let auxiliaryContribution = auxiliaryExpertActionMaskArray.map {
                auxiliaryExpertGates * $0
            } ?? auxiliaryExpertGates
            let baseContribution = maximum(
                1 - expertContribution - standContribution
                    - auxiliaryContribution,
                MLXArray(Float(0)))
            var scales = auxiliaryContribution
            if !freezesBasePolicyExpert { scales = scales + baseContribution }
            if !freezesLowSpeedPolicyExpert {
                scales = scales + expertContribution
            }
            if !freezesStandPolicyExpert {
                scales = scales + standContribution
            }
            return clip(scales, min: 0, max: 1)
        }

        func routedEffectiveLogStandardDeviation(
            base: MLXArray, expertGates: MLXArray,
            standExpertGates: MLXArray,
            auxiliaryExpertGates: MLXArray
        ) -> MLXArray {
            guard configuration.resolvedRoutedExplorationMaskVersion >= 3 else {
                return base
            }
            return base + log(clip(routedExplorationScales(
                expertGates: expertGates,
                standExpertGates: standExpertGates,
                auxiliaryExpertGates: auxiliaryExpertGates),
                min: 1e-6, max: 1))
        }

        func routedLikelihoodActionMask(
            expertGates: MLXArray, standExpertGates: MLXArray,
            auxiliaryExpertGates: MLXArray
        ) -> MLXArray {
            guard configuration.resolvedRoutedExplorationMaskVersion >= 4 else {
                return MLXArray.ones([expertGates.shape[0], actionDim])
            }
            let scales = routedExplorationScales(
                expertGates: expertGates,
                standExpertGates: standExpertGates,
                auxiliaryExpertGates: auxiliaryExpertGates)
            return which(
                scales .> 0, MLXArray(Float(1)), MLXArray(Float(0)))
        }

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
            let auxiliaryExpertGates = args[11]
            let successImitationMask = args[12]
            let referencePolicyWeights = args[13]
            let referencePolicyActionWeights = args[14]
            let actorTrainingWeights = args[15]
            let oldMeans = args[16]
            let oldLogStandardDeviations = args[17]
            let replayObservations = args[18]
            let replayActions = args[19]
            let replayExpertGates = args[20]
            let replayStandExpertGates = args[21]
            let replayAuxiliaryExpertGates = args[22]
            let replayWeights = args[23]
            let out = model.forward(
                observations, expertGate: expertGates,
                expertActionMask: expertActionMaskArray,
                standExpertGate: standExpertGates,
                standExpertActionMask: standExpertActionMaskArray,
                auxiliaryExpertGate: auxiliaryExpertGates,
                auxiliaryExpertActionMask: auxiliaryExpertActionMaskArray,
                freezeBaseActor: freezesBasePolicyExpert,
                freezeExpertActor: freezesLowSpeedPolicyExpert,
                freezeStandActor: freezesStandPolicyExpert)
            let baseLogStandardDeviation = clip(
                out.logStandardDeviation,
                min: actionLogStandardDeviationBounds.minimum,
                max: actionLogStandardDeviationBounds.maximum)
            let effectiveLogStandardDeviation =
                routedEffectiveLogStandardDeviation(
                    base: baseLogStandardDeviation,
                    expertGates: expertGates,
                    standExpertGates: standExpertGates,
                    auxiliaryExpertGates: auxiliaryExpertGates)
            let likelihoodActionMask = routedLikelihoodActionMask(
                expertGates: expertGates,
                standExpertGates: standExpertGates,
                auxiliaryExpertGates: auxiliaryExpertGates)
            let logProb = Self.gaussianLogProbability(
                preTanhActions, mean: out.mean,
                logStandardDeviation: effectiveLogStandardDeviation,
                actionMask: likelihoodActionMask)
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
            let valueLoss: MLXArray
            if self.configuration.resolvedClipValueLoss {
                let clippedValue = oldValues + clip(
                    out.value - oldValues,
                    min: -self.configuration.valueClip,
                    max: self.configuration.valueClip)
                valueLoss = 0.5 * mean(maximum(
                    (out.value - returns).square(),
                    (clippedValue - returns).square()))
            } else {
                valueLoss = 0.5 * mean((out.value - returns).square())
            }
            let entropy: MLXArray
            switch actionDistribution {
            case .gaussian:
                // RSL-RL uses the analytic entropy of its unbounded diagonal
                // Gaussian action distribution.
                let entropyRows = sum((effectiveLogStandardDeviation
                    + Float(1.4189385332)) * likelihoodActionMask,
                    axis: -1)
                entropy = sum(entropyRows * actorTrainingWeights)
                    / actorTrainingDenominator
            case .squashedGaussian:
                // Entropy must belong to the policy the environment actually
                // receives. A reparameterized Monte-Carlo estimate includes
                // the tanh Jacobian and does not reward saturated ±1 actions.
                let entropyPreTanh = out.mean + entropyNoise
                    * exp(effectiveLogStandardDeviation)
                let entropyActions = tanh(entropyPreTanh)
                let entropyLogProbability = Self.gaussianLogProbability(
                    entropyPreTanh, mean: out.mean,
                    logStandardDeviation: effectiveLogStandardDeviation,
                    actionMask: likelihoodActionMask)
                    - sum(log(clip(1 - entropyActions.square(),
                                   min: 1e-6, max: 1))
                        * likelihoodActionMask, axis: -1)
                entropy = -sum(entropyLogProbability * actorTrainingWeights)
                    / actorTrainingDenominator
            }
            // Use the exact diagonal-Gaussian KL employed by RSL-RL rather
            // than a sampled importance-ratio estimate. The sampled estimator
            // becomes extremely noisy when only a sparse routed expert is
            // trainable; that previously stopped transfer updates even when
            // the deterministic joint targets changed by only a few 1e-5.
            // Symmetry-transformed rows remain excluded because they were not
            // generated by the environment behavior policy.
            let trainableKLWeights = klWeights * actorTrainingWeights
            let klDenominator = clip(sum(trainableKLWeights), min: 1,
                                     max: Float.greatestFiniteMagnitude)
            let oldVariance = exp(2 * oldLogStandardDeviations)
            let newVariance = exp(2 * effectiveLogStandardDeviation)
            let gaussianKLRows = maximum(sum((
                effectiveLogStandardDeviation - oldLogStandardDeviations
                    + (oldVariance + (oldMeans - out.mean).square())
                        / (2 * newVariance)
                    - 0.5) * likelihoodActionMask,
                axis: -1), MLXArray(Float(0)))
            let exactKL = sum(gaussianKLRows * trainableKLWeights)
                / klDenominator
            let maximumExactKL = max(gaussianKLRows * trainableKLWeights)
            let maximumBehaviorMeanReplayError = max(
                abs(oldMeans - out.mean) * likelihoodActionMask
                    * trainableKLWeights.reshaped([-1, 1]))
            let maximumBehaviorLogStandardDeviationReplayError = max(abs(
                oldLogStandardDeviations - effectiveLogStandardDeviation)
                * likelihoodActionMask
                * trainableKLWeights.reshaped([-1, 1]))
            let sampleApproximateKL = sum(
                ((ratio - 1) - logRatio) * trainableKLWeights)
                / klDenominator
            var symmetryLoss = MLXArray(Float(0))
            if usesSymmetryMirrorLoss {
                let mirroredMean = model.forward(
                    mirroredObservations, expertGate: expertGates,
                    expertActionMask: expertActionMaskArray,
                    standExpertGate: standExpertGates,
                    standExpertActionMask: standExpertActionMaskArray,
                    auxiliaryExpertGate: auxiliaryExpertGates,
                    auxiliaryExpertActionMask:
                        auxiliaryExpertActionMaskArray,
                    freezeBaseActor: freezesBasePolicyExpert,
                    freezeExpertActor: freezesLowSpeedPolicyExpert,
                    freezeStandActor: freezesStandPolicyExpert).mean
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
            let rolloutSuccessImitationLoss = sum(
                imitationActionError * trainableImitationMask)
                / imitationDenominator
            let replayOut = model.forward(
                replayObservations,
                expertGate: replayExpertGates,
                expertActionMask: expertActionMaskArray,
                standExpertGate: replayStandExpertGates,
                standExpertActionMask: standExpertActionMaskArray,
                auxiliaryExpertGate: replayAuxiliaryExpertGates,
                auxiliaryExpertActionMask: auxiliaryExpertActionMaskArray,
                freezeBaseActor: freezesBasePolicyExpert,
                freezeExpertActor: freezesLowSpeedPolicyExpert,
                freezeStandActor: freezesStandPolicyExpert)
            let replayActionError = sum(
                (actionDistribution.environmentAction(replayOut.mean)
                    - actionDistribution.environmentAction(replayActions))
                    .square(), axis: -1)
            let replayDenominator = clip(sum(replayWeights), min: 1,
                                         max: Float.greatestFiniteMagnitude)
            let replaySuccessImitationLoss = sum(
                replayActionError * replayWeights) / replayDenominator
            let successImitationLoss = rolloutSuccessImitationLoss
                + replaySuccessImitationLoss
            var referencePolicyLoss = MLXArray(Float(0))
            if let referencePolicy {
                let referenceMean = stopGradient(referencePolicy.forward(
                    observations, expertGate: expertGates,
                    expertActionMask: expertActionMaskArray,
                    standExpertGate: standExpertGates,
                    standExpertActionMask: standExpertActionMaskArray,
                    auxiliaryExpertGate: auxiliaryExpertGates,
                    auxiliaryExpertActionMask:
                        auxiliaryExpertActionMaskArray).mean)
                let referenceActionError = sum(
                    (actionDistribution.environmentAction(out.mean)
                        - actionDistribution.environmentAction(referenceMean))
                        .square() * referencePolicyActionWeights,
                    axis: -1)
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
            return [total, policyLoss, valueLoss, entropy, exactKL,
                    symmetryLoss, successImitationLoss, referencePolicyLoss,
                    sampleApproximateKL, maximumExactKL,
                    maximumBehaviorMeanReplayError,
                    maximumBehaviorLogStandardDeviationReplayError]
        }

        if resume {
            try Self.reconcileMetricsLog(
                at: metricsURL, completedUpdates: startingUpdate)
        } else {
            try Data().write(to: metricsURL, options: .atomic)
        }
        let metricsFile = try FileHandle(forWritingTo: metricsURL)
        try metricsFile.seekToEnd()
        defer { try? metricsFile.close() }

        for localUpdate in 0..<configuration.updates {
            let update = startingUpdate + localUpdate
            actionLogStandardDeviationBounds = configuration
                .actionLogStandardDeviationBounds(completedUpdate: update)
            let startTime = Date()
            optimizer.learningRate = adaptiveLearningRate

            var storedObservations = [Float](); storedObservations.reserveCapacity(batchSize * obsDim)
            var storedMirroredObservations = [Float]()
            storedMirroredObservations.reserveCapacity(batchSize * obsDim)
            var storedActions = [Float](); storedActions.reserveCapacity(batchSize * actionDim)
            var storedBehaviorMeans = [Float]()
            storedBehaviorMeans.reserveCapacity(batchSize * actionDim)
            var storedBehaviorLogStandardDeviations = [Float]()
            storedBehaviorLogStandardDeviations.reserveCapacity(
                batchSize * actionDim)
            var storedLogProb = [Float](); storedLogProb.reserveCapacity(batchSize)
            var storedValues = [Float](); storedValues.reserveCapacity(batchSize)
            var storedRewards = [Float](); storedRewards.reserveCapacity(batchSize)
            var storedDones = [Bool](); storedDones.reserveCapacity(batchSize)
            var storedKLWeights = [Float](); storedKLWeights.reserveCapacity(batchSize)
            var storedSuccessImitationMask = [Float](
                repeating: 0, count: batchSize)
            var storedReferencePolicyWeights = [Float]()
            storedReferencePolicyWeights.reserveCapacity(batchSize)
            var storedReferencePolicyActionWeights = [Float]()
            storedReferencePolicyActionWeights.reserveCapacity(
                batchSize * actionDim)
            var episodeStartSteps = [Int](repeating: 0, count: n)
            var storedExpertGates = [Float]();
            storedExpertGates.reserveCapacity(batchSize)
            var storedStandExpertGates = [Float]();
            storedStandExpertGates.reserveCapacity(batchSize)
            var storedAuxiliaryExpertGates = [Float]();
            storedAuxiliaryExpertGates.reserveCapacity(batchSize)
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
                let rolloutReferencePolicyActionWeights:
                    ContiguousArray<Float>
                if referencePolicyCoefficient > 0,
                   let referenceActionRegularizationProvider {
                    rolloutReferencePolicyActionWeights =
                        referenceActionRegularizationProvider
                            .policyReferenceActionRegularizationWeights(
                                observation.policy,
                                actionDimension: actionDim)
                } else {
                    rolloutReferencePolicyActionWeights = ContiguousArray(
                        repeating: 1, count: n * actionDim)
                }
                guard rolloutReferencePolicyActionWeights.count
                        == n * actionDim,
                      rolloutReferencePolicyActionWeights.allSatisfy({
                          $0.isFinite && (0...1).contains($0)
                      }) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "task returned invalid action-wise reference-policy weights")
                }
                storedReferencePolicyActionWeights.append(
                    contentsOf: rolloutReferencePolicyActionWeights)
                let rolloutExpertGates = usesPolicyExpertGate
                    ? policyExpertGate!.policyExpertGates(observation.policy)
                    : ContiguousArray(repeating: Float(0), count: n)
                let rolloutStandExpertGates = usesPolicyStandExpertGate
                    ? policyStandExpertGate!.policyStandExpertGates(
                        observation.policy)
                    : ContiguousArray(repeating: Float(0), count: n)
                let rolloutAuxiliaryExpertGates = usesPolicyAuxiliaryExpertGate
                    ? policyAuxiliaryExpertGate!.policyAuxiliaryExpertGates(
                        observation.policy)
                    : ContiguousArray(repeating: Float(0), count: n)
                guard Self.hasValidActorComposition(
                    expertGates: Array(rolloutExpertGates),
                    standExpertGates: Array(rolloutStandExpertGates),
                    actionDimension: actionDim,
                    expertActionMask: expertActionMask.map { Array($0) },
                    standExpertActionMask: standExpertActionMask.map {
                        Array($0)
                    }, auxiliaryExpertGates:
                        Array(rolloutAuxiliaryExpertGates),
                    auxiliaryExpertActionMask:
                        auxiliaryExpertActionMask.map { Array($0) }) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "task returned invalid or overlapping policy expert gates")
                }
                storedExpertGates.append(contentsOf: rolloutExpertGates)
                storedStandExpertGates.append(
                    contentsOf: rolloutStandExpertGates)
                storedAuxiliaryExpertGates.append(
                    contentsOf: rolloutAuxiliaryExpertGates)
                var rolloutActorTrainingWeights = Self.actorTrainingWeights(
                    expertGates: Array(rolloutExpertGates),
                    standExpertGates: Array(rolloutStandExpertGates),
                    actionDimension: actionDim,
                    expertActionMask: expertActionMask.map { Array($0) },
                    standExpertActionMask: standExpertActionMask.map {
                        Array($0)
                    },
                    freezesBaseActor: freezesBasePolicyExpert,
                    freezesExpertActor: freezesLowSpeedPolicyExpert,
                    auxiliaryExpertGates:
                        Array(rolloutAuxiliaryExpertGates),
                    auxiliaryExpertActionMask:
                        auxiliaryExpertActionMask.map { Array($0) },
                    freezesStandActor: freezesStandPolicyExpert)
                if let actorTrainingWeightProvider {
                    let taskWeights = actorTrainingWeightProvider
                        .policyActorTrainingWeights(observation.policy)
                    guard taskWeights.count == n,
                          taskWeights.allSatisfy({
                              $0.isFinite && (0...1).contains($0)
                          }) else {
                        throw RLEnvironmentError.invalidConfiguration(
                            "task returned invalid actor-training weights")
                    }
                    for environment in 0..<n {
                        rolloutActorTrainingWeights[environment] *=
                            taskWeights[environment]
                    }
                }
                storedActorTrainingWeights.append(
                    contentsOf: rolloutActorTrainingWeights)
                let rolloutGateArray = MLXArray(Array(rolloutExpertGates))
                    .reshaped([n, 1])
                let rolloutStandGateArray = MLXArray(
                    Array(rolloutStandExpertGates)).reshaped([n, 1])
                let rolloutAuxiliaryGateArray = MLXArray(
                    Array(rolloutAuxiliaryExpertGates)).reshaped([n, 1])
                var explorationActionScaleValues = configuration
                    .resolvedRoutedExplorationMaskVersion >= 2
                    ? Self.actorExplorationActionScales(
                        expertGates: Array(rolloutExpertGates),
                        standExpertGates: Array(rolloutStandExpertGates),
                        actionDimension: actionDim,
                        expertActionMask: expertActionMask.map { Array($0) },
                        standExpertActionMask: standExpertActionMask.map {
                            Array($0)
                        },
                        freezesBaseActor: freezesBasePolicyExpert,
                        freezesExpertActor: freezesLowSpeedPolicyExpert,
                        auxiliaryExpertGates:
                            Array(rolloutAuxiliaryExpertGates),
                        auxiliaryExpertActionMask:
                            auxiliaryExpertActionMask.map { Array($0) },
                        freezesStandActor: freezesStandPolicyExpert)
                    : [Float](repeating: 1, count: n * actionDim)
                if configuration.resolvedRoutedExplorationMaskVersion == 2 {
                    explorationActionScaleValues =
                        explorationActionScaleValues.map { $0 > 0 ? 1 : 0 }
                }
                let explorationActionScales = MLXArray(
                    explorationActionScaleValues)
                    .reshaped([n, actionDim])
                let out = rolloutForward(
                    observations: normalized,
                    expertGates: rolloutExpertGates,
                    standExpertGates: rolloutStandExpertGates,
                    auxiliaryExpertGates: rolloutAuxiliaryExpertGates)
                let baseLogStandardDeviation = clip(
                    out.logStandardDeviation,
                    min: actionLogStandardDeviationBounds.minimum,
                    max: actionLogStandardDeviationBounds.maximum)
                let effectiveLogStandardDeviation =
                    routedEffectiveLogStandardDeviation(
                        base: baseLogStandardDeviation,
                        expertGates: rolloutGateArray,
                        standExpertGates: rolloutStandGateArray,
                        auxiliaryExpertGates: rolloutAuxiliaryGateArray)
                let likelihoodActionMask = routedLikelihoodActionMask(
                    expertGates: rolloutGateArray,
                    standExpertGates: rolloutStandGateArray,
                    auxiliaryExpertGates: rolloutAuxiliaryGateArray)
                // An explicit update/step key makes action sampling stable
                // across process restarts without depending on global PRNG
                // state consumed during model construction.
                let randomKey = MLXRandom.key(
                    configuration.seed
                        &+ UInt64(update) &* 0x9E3779B97F4A7C15
                        &+ UInt64(rolloutStep) &* 0xD1B54A32D192ED03)
                let preTanh = out.mean
                    + MLXRandom.normal([n, actionDim], key: randomKey)
                        * exp(baseLogStandardDeviation)
                        * explorationActionScales
                let environmentActions = actionDistribution.environmentAction(
                    preTanh)
                let logProb = Self.gaussianLogProbability(
                    preTanh, mean: out.mean,
                    logStandardDeviation: effectiveLogStandardDeviation,
                    actionMask: likelihoodActionMask)
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
                    let trainingOut = rolloutForward(
                        observations: trainingNormalized,
                        expertGates: rolloutExpertGates,
                        standExpertGates: rolloutStandExpertGates,
                        auxiliaryExpertGates: rolloutAuxiliaryExpertGates)
                    eval(trainingOut.value)
                    trainingValues = ContiguousArray(
                        trainingOut.value.asArray(Float.self))
                }
                guard trainingPreTanh.allSatisfy(\.isFinite),
                      trainingValues.allSatisfy(\.isFinite) else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "non-finite symmetry-augmented PPO sample before update "
                        + "\(update + 1)")
                }
                storedActions.append(contentsOf: trainingPreTanh)
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
                try stepResult.validate(for: spec)
                for (name, values) in stepResult.metrics
                    where name.hasPrefix("reward/")
                        || name.hasPrefix("penalty/")
                        || name.hasPrefix("gait/")
                        || name.hasPrefix("curriculum/")
                        || name.hasPrefix("task/") {
                    taskMetricSums[name, default: 0] += values.reduce(0, +)
                    taskMetricCounts[name, default: 0] += values.count
                }

                var adjustedRewards = stepResult.rewards.map {
                    $0 * configuration.resolvedRewardScale
                }
                let hasBootstrapTruncation = (0..<n).contains {
                    Self.shouldBootstrapFinalObservation(
                        terminated: stepResult.terminated[$0],
                        truncated: stepResult.truncated[$0])
                }
                if hasBootstrapTruncation {
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
                    let finalValues = rolloutForward(
                        observations: final,
                        expertGates: ContiguousArray(repeating: 0, count: n),
                        standExpertGates: ContiguousArray(
                            repeating: 0, count: n),
                        auxiliaryExpertGates: ContiguousArray(
                            repeating: 0, count: n)).value
                    eval(finalValues)
                    let host = finalValues.asArray(Float.self)
                    for e in 0..<n where Self.shouldBootstrapFinalObservation(
                        terminated: stepResult.terminated[e],
                        truncated: stepResult.truncated[e]) {
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
                    }
                    let successImitationTrigger = stepResult.successes[e]
                        && !episodeSuccessImitationReached[e]
                    let imitationTrigger = successImitationTrigger
                        || stepResult.imitationMilestones[e]
                    if imitationTrigger {
                        if successImitationTrigger {
                            episodeSuccessImitationReached[e] = true
                        }
                        if successImitationCoefficient > 0 {
                            let causalStart = Self.imitationSegmentStart(
                                episodeStart: episodeStartSteps[e],
                                milestoneStep: rolloutStep,
                                historySteps: configuration
                                    .successImitationHistorySteps)
                            Self.markSuccessfulEpisodeSegment(
                                mask: &storedSuccessImitationMask,
                                environment: e, numEnvironments: n,
                                startStep: causalStart,
                                endStep: rolloutStep)
                        }
                    }
                    if done {
                        completedEpisodes += 1
                        if episodeSucceeded[e] { successfulEpisodes += 1 }
                        episodeSucceeded[e] = false
                        episodeSuccessImitationReached[e] = false
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

            if successReplayCapacity > 0 {
                for row in 0..<batchSize
                    where storedSuccessImitationMask[row] > 0
                        && storedActorTrainingWeights[row] > 0 {
                    let slot = successReplayNext
                    let observationSource = row * obsDim
                    let observationDestination = slot * obsDim
                    for index in 0..<obsDim {
                        successReplayObservations[
                            observationDestination + index] =
                            storedObservations[observationSource + index]
                    }
                    let actionSource = row * actionDim
                    let actionDestination = slot * actionDim
                    for index in 0..<actionDim {
                        successReplayActions[actionDestination + index] =
                            storedActions[actionSource + index]
                    }
                    successReplayExpertGates[slot] = storedExpertGates[row]
                    successReplayStandExpertGates[slot] =
                        storedStandExpertGates[row]
                    successReplayAuxiliaryExpertGates[slot] =
                        storedAuxiliaryExpertGates[row]
                    successReplayNext = (successReplayNext + 1)
                        % successReplayCapacity
                    successReplayCount = min(
                        successReplayCount + 1, successReplayCapacity)
                }
            }

            // MLX may select a different Metal matrix kernel for the small
            // online environment batch than for a large PPO minibatch. On a
            // deep transferred actor, the resulting floating-point reduction
            // drift was enough to make an unchanged policy report a 0.05 KL.
            // Re-evaluate the frozen behavior policy once in the exact update
            // geometry so importance ratios, value clipping, and analytic KL
            // all satisfy the required identity-at-update-start invariant.
            // The actions remain the ones physically sampled and applied by
            // the rollout; only their immutable reference likelihood/value is
            // represented in the numerical geometry used by optimization.
            var storedTrainingValues = [Float]()
            storedTrainingValues.reserveCapacity(batchSize)
            for start in stride(from: 0, to: batchSize,
                                by: configuration.minibatchSize) {
                let end = start + configuration.minibatchSize
                let rows = end - start
                let observationStart = start * obsDim
                let observationEnd = end * obsDim
                let actionStart = start * actionDim
                let actionEnd = end * actionDim
                let behaviorObservations = MLXArray(Array(
                    storedObservations[observationStart..<observationEnd]))
                    .reshaped([rows, obsDim])
                let behaviorExpertGates = MLXArray(Array(
                    storedExpertGates[start..<end])).reshaped([rows, 1])
                let behaviorStandExpertGates = MLXArray(Array(
                    storedStandExpertGates[start..<end])).reshaped([rows, 1])
                let behaviorAuxiliaryExpertGates = MLXArray(Array(
                    storedAuxiliaryExpertGates[start..<end]))
                    .reshaped([rows, 1])
                let behavior = policy.forward(
                    behaviorObservations,
                    expertGate: behaviorExpertGates,
                    expertActionMask: expertActionMaskArray,
                    standExpertGate: behaviorStandExpertGates,
                    standExpertActionMask: standExpertActionMaskArray,
                    auxiliaryExpertGate: behaviorAuxiliaryExpertGates,
                    auxiliaryExpertActionMask:
                        auxiliaryExpertActionMaskArray,
                    freezeBaseActor: freezesBasePolicyExpert,
                    freezeExpertActor: freezesLowSpeedPolicyExpert,
                    freezeStandActor: freezesStandPolicyExpert)
                let behaviorBaseLogStandardDeviation = clip(
                    behavior.logStandardDeviation,
                    min: actionLogStandardDeviationBounds.minimum,
                    max: actionLogStandardDeviationBounds.maximum)
                let behaviorLogStandardDeviation =
                    routedEffectiveLogStandardDeviation(
                        base: behaviorBaseLogStandardDeviation,
                        expertGates: behaviorExpertGates,
                        standExpertGates: behaviorStandExpertGates,
                        auxiliaryExpertGates: behaviorAuxiliaryExpertGates)
                let behaviorLikelihoodActionMask =
                    routedLikelihoodActionMask(
                        expertGates: behaviorExpertGates,
                        standExpertGates: behaviorStandExpertGates,
                        auxiliaryExpertGates: behaviorAuxiliaryExpertGates)
                let behaviorActions = MLXArray(Array(
                    storedActions[actionStart..<actionEnd]))
                    .reshaped([rows, actionDim])
                let behaviorLogProbability = Self.gaussianLogProbability(
                    behaviorActions, mean: behavior.mean,
                    logStandardDeviation: behaviorLogStandardDeviation,
                    actionMask: behaviorLikelihoodActionMask)
                eval(behavior.mean, behavior.value,
                     behaviorLogStandardDeviation, behaviorLogProbability)
                storedBehaviorMeans.append(contentsOf:
                    behavior.mean.asArray(Float.self))
                let behaviorLogStandardDeviationHost =
                    behaviorLogStandardDeviation.asArray(Float.self)
                if configuration.resolvedRoutedExplorationMaskVersion >= 3 {
                    storedBehaviorLogStandardDeviations.append(
                        contentsOf: behaviorLogStandardDeviationHost)
                } else {
                    for _ in 0..<rows {
                        storedBehaviorLogStandardDeviations.append(
                            contentsOf: behaviorLogStandardDeviationHost)
                    }
                }
                storedLogProb.append(contentsOf:
                    behaviorLogProbability.asArray(Float.self))
                storedTrainingValues.append(contentsOf:
                    behavior.value.asArray(Float.self))
            }
            guard storedBehaviorMeans.count == batchSize * actionDim,
                  storedBehaviorLogStandardDeviations.count
                    == batchSize * actionDim,
                  storedLogProb.count == batchSize,
                  storedTrainingValues.count == batchSize,
                  storedBehaviorMeans.allSatisfy(\.isFinite),
                  storedBehaviorLogStandardDeviations.allSatisfy(\.isFinite),
                  storedLogProb.allSatisfy(\.isFinite),
                  storedTrainingValues.allSatisfy(\.isFinite) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "non-finite or incomplete behavior-policy replay before "
                        + "PPO update \(update + 1)")
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
            let lastValueArray = rolloutForward(
                observations: lastNormalized,
                expertGates: ContiguousArray(repeating: 0, count: n),
                standExpertGates: ContiguousArray(repeating: 0, count: n),
                auxiliaryExpertGates: ContiguousArray(
                    repeating: 0, count: n)).value
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
            var lastSampleApproximateKL: Float = 0
            var lastMaximumExactKL: Float = 0
            var lastMaximumBehaviorMeanReplayError: Float = 0
            var lastMaximumBehaviorLogStandardDeviationReplayError: Float = 0
            var firstMinibatchHostBehaviorMeanReplayError: Float = 0
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
                    var mbAuxiliaryExpertGates = [Float]();
                    mbAuxiliaryExpertGates.reserveCapacity(m)
                    var mbSuccessImitationMask = [Float]()
                    mbSuccessImitationMask.reserveCapacity(m)
                    var mbReferencePolicyWeights = [Float]()
                    mbReferencePolicyWeights.reserveCapacity(m)
                    var mbReferencePolicyActionWeights = [Float]()
                    mbReferencePolicyActionWeights.reserveCapacity(
                        m * actionDim)
                    var mbActorTrainingWeights = [Float]()
                    mbActorTrainingWeights.reserveCapacity(m)
                    var mbBehaviorMeans = [Float]()
                    mbBehaviorMeans.reserveCapacity(m * actionDim)
                    var mbBehaviorLogStandardDeviations = [Float]()
                    mbBehaviorLogStandardDeviations.reserveCapacity(
                        m * actionDim)
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
                        mbValues.append(storedTrainingValues[i])
                        mbKLWeights.append(storedKLWeights[i])
                        mbExpertGates.append(storedExpertGates[i])
                        mbStandExpertGates.append(storedStandExpertGates[i])
                        mbAuxiliaryExpertGates.append(
                            storedAuxiliaryExpertGates[i])
                        mbSuccessImitationMask.append(
                            storedSuccessImitationMask[i])
                        mbReferencePolicyWeights.append(
                            storedReferencePolicyWeights[i])
                        mbReferencePolicyActionWeights.append(contentsOf:
                            storedReferencePolicyActionWeights[
                                (i * actionDim)..<((i + 1) * actionDim)])
                        mbActorTrainingWeights.append(
                            storedActorTrainingWeights[i])
                        mbBehaviorMeans.append(contentsOf:
                            storedBehaviorMeans[
                                (i * actionDim)..<((i + 1) * actionDim)])
                        mbBehaviorLogStandardDeviations.append(contentsOf:
                            storedBehaviorLogStandardDeviations[
                                (i * actionDim)..<((i + 1) * actionDim)])
                    }
                    let replayRows = min(
                        successReplayBatchSize, successReplayCount)
                    var mbReplayObservations = [Float]()
                    var mbReplayActions = [Float]()
                    var mbReplayExpertGates = [Float]()
                    var mbReplayStandExpertGates = [Float]()
                    var mbReplayAuxiliaryExpertGates = [Float]()
                    var mbReplayWeights = [Float]()
                    if replayRows > 0 {
                        mbReplayObservations.reserveCapacity(replayRows * obsDim)
                        mbReplayActions.reserveCapacity(replayRows * actionDim)
                        mbReplayExpertGates.reserveCapacity(replayRows)
                        mbReplayStandExpertGates.reserveCapacity(replayRows)
                        mbReplayAuxiliaryExpertGates.reserveCapacity(replayRows)
                        mbReplayWeights.reserveCapacity(replayRows)
                        let replayStart = (update &* 131
                            + epoch &* 17
                            + start / configuration.minibatchSize)
                            % successReplayCount
                        for offset in 0..<replayRows {
                            let slot = (replayStart + offset)
                                % successReplayCount
                            mbReplayObservations.append(contentsOf:
                                successReplayObservations[
                                    (slot * obsDim)..<((slot + 1) * obsDim)])
                            mbReplayActions.append(contentsOf:
                                successReplayActions[
                                    (slot * actionDim)..<((slot + 1) * actionDim)])
                            mbReplayExpertGates.append(
                                successReplayExpertGates[slot])
                            mbReplayStandExpertGates.append(
                                successReplayStandExpertGates[slot])
                            mbReplayAuxiliaryExpertGates.append(
                                successReplayAuxiliaryExpertGates[slot])
                            mbReplayWeights.append(1)
                        }
                    } else {
                        // MLX tensors cannot use a zero row in every backend.
                        // A single zero-weight row keeps the graph shape valid
                        // and contributes exactly zero replay loss.
                        mbReplayObservations = [Float](
                            repeating: 0, count: obsDim)
                        mbReplayActions = [Float](
                            repeating: 0, count: actionDim)
                        mbReplayExpertGates = [0]
                        mbReplayStandExpertGates = [0]
                        mbReplayAuxiliaryExpertGates = [0]
                        mbReplayWeights = [0]
                    }
                    let effectiveReplayRows = max(replayRows, 1)
                    let entropyKey = MLXRandom.key(
                        configuration.seed
                            &+ UInt64(update) &* 0xD1B54A32D192ED03
                            &+ UInt64(epoch) &* 0x94D049BB133111EB
                            &+ UInt64(start))
                    let entropyNoise = MLXRandom.normal(
                        [m, actionDim], key: entropyKey)
                    let mbObservationArray = MLXArray(mbObs)
                        .reshaped([m, obsDim])
                    let mbExpertGateArray = MLXArray(mbExpertGates)
                        .reshaped([m, 1])
                    let mbStandExpertGateArray = MLXArray(mbStandExpertGates)
                        .reshaped([m, 1])
                    let mbAuxiliaryExpertGateArray = MLXArray(
                        mbAuxiliaryExpertGates).reshaped([m, 1])
                    if epoch == 0 && start == 0 {
                        let replayMean = policy.forward(
                            mbObservationArray,
                            expertGate: mbExpertGateArray,
                            expertActionMask: expertActionMaskArray,
                            standExpertGate: mbStandExpertGateArray,
                            standExpertActionMask: standExpertActionMaskArray,
                            auxiliaryExpertGate: mbAuxiliaryExpertGateArray,
                            auxiliaryExpertActionMask:
                                auxiliaryExpertActionMaskArray,
                            freezeBaseActor: freezesBasePolicyExpert,
                            freezeExpertActor:
                                freezesLowSpeedPolicyExpert,
                            freezeStandActor: freezesStandPolicyExpert).mean
                        eval(replayMean)
                        let replayHost = replayMean.asArray(Float.self)
                        for row in 0..<m
                            where mbActorTrainingWeights[row] > 0 {
                            for action in 0..<actionDim {
                                let index = row * actionDim + action
                                let error = abs(replayHost[index]
                                    - mbBehaviorMeans[index])
                                if error
                                    > firstMinibatchHostBehaviorMeanReplayError {
                                    firstMinibatchHostBehaviorMeanReplayError =
                                        error
                                }
                            }
                        }
                        if firstMinibatchHostBehaviorMeanReplayError > 1e-5 {
                            throw RLEnvironmentError.invalidConfiguration(
                                "behavior-policy replay changed before PPO "
                                    + "update \(update + 1) (max actor mean "
                                    + "error "
                                    + "\(firstMinibatchHostBehaviorMeanReplayError)); "
                                    + "Metal training geometry is inconsistent")
                        }
                    }
                    let (losses, gradients) = lossAndGradient(policy, [
                        mbObservationArray,
                        MLXArray(mbActions).reshaped([m, actionDim]),
                        MLXArray(mbLogProb), MLXArray(mbAdvantages),
                        MLXArray(mbReturns), MLXArray(mbValues), entropyNoise,
                        MLXArray(mbKLWeights),
                        MLXArray(mbMirroredObs).reshaped([m, obsDim]),
                        mbExpertGateArray,
                        mbStandExpertGateArray,
                        mbAuxiliaryExpertGateArray,
                        MLXArray(mbSuccessImitationMask),
                        MLXArray(mbReferencePolicyWeights),
                        MLXArray(mbReferencePolicyActionWeights)
                            .reshaped([m, actionDim]),
                        MLXArray(mbActorTrainingWeights),
                        MLXArray(mbBehaviorMeans).reshaped([m, actionDim]),
                        MLXArray(mbBehaviorLogStandardDeviations)
                            .reshaped([m, actionDim]),
                        MLXArray(mbReplayObservations)
                            .reshaped([effectiveReplayRows, obsDim]),
                        MLXArray(mbReplayActions)
                            .reshaped([effectiveReplayRows, actionDim]),
                        MLXArray(mbReplayExpertGates)
                            .reshaped([effectiveReplayRows, 1]),
                        MLXArray(mbReplayStandExpertGates)
                            .reshaped([effectiveReplayRows, 1]),
                        MLXArray(mbReplayAuxiliaryExpertGates)
                            .reshaped([effectiveReplayRows, 1]),
                        MLXArray(mbReplayWeights),
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
                    lastSampleApproximateKL = scalarLosses[8]
                    lastMaximumExactKL = scalarLosses[9]
                    lastMaximumBehaviorMeanReplayError = scalarLosses[10]
                    lastMaximumBehaviorLogStandardDeviationReplayError =
                        scalarLosses[11]
                    klSum += lastKL
                    klCount += 1
                    if Self.shouldStopForKL(
                        minibatchKL: lastKL,
                        targetKL: configuration.targetKL,
                        schedule: configuration.resolvedKLSchedule) {
                        break epochLoop
                    }
                }
            }
            let meanKL = klCount > 0 ? klSum / Float(klCount) : 0
            if configuration.targetKL > 0
                && configuration.resolvedKLSchedule == .adaptive {
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
            if configuration.resolvedTrainingProgressUpdateVersion >= 2 {
                (task as? any TrainingModeConfigurable)?.setTrainingProgress(
                    environmentSteps: totalSteps)
            }
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
            meanTaskMetrics["training/success_replay_size"] =
                Float(successReplayCount)
            meanTaskMetrics["training/success_replay_fill_fraction"] =
                successReplayCapacity > 0
                    ? Float(successReplayCount) / Float(successReplayCapacity)
                    : 0
            meanTaskMetrics["training/reference_policy_loss"] =
                lastReferencePolicyLoss
            meanTaskMetrics["training/reference_policy_fraction"] =
                storedReferencePolicyWeights.reduce(0, +) / Float(batchSize)
            meanTaskMetrics["training/reference_policy_action_fraction"] =
                storedReferencePolicyActionWeights.reduce(0, +)
                    / Float(batchSize * actionDim)
            meanTaskMetrics["training/actor_training_fraction"] =
                storedActorTrainingWeights.reduce(0, +) / Float(batchSize)
            meanTaskMetrics["training/policy_inference_batch_size"] =
                Float(policyInferenceBatchSize)
            meanTaskMetrics["training/sample_approximate_kl"] =
                lastSampleApproximateKL
            meanTaskMetrics["training/maximum_exact_kl"] =
                lastMaximumExactKL
            meanTaskMetrics["training/behavior_mean_replay_max_abs"] =
                lastMaximumBehaviorMeanReplayError
            meanTaskMetrics[
                "training/behavior_log_std_replay_max_abs"] =
                lastMaximumBehaviorLogStandardDeviationReplayError
            meanTaskMetrics[
                "training/first_minibatch_host_mean_replay_max_abs"] =
                firstMinibatchHostBehaviorMeanReplayError
            var maximumTrainableObservationMagnitude: Float = 0
            for row in 0..<batchSize
                where storedActorTrainingWeights[row] > 0 {
                let start = row * obsDim
                let end = start + obsDim
                for value in storedObservations[start..<end] {
                    maximumTrainableObservationMagnitude = max(
                        maximumTrainableObservationMagnitude, abs(value))
                }
            }
            meanTaskMetrics["training/actor_observation_max_abs"] =
                maximumTrainableObservationMagnitude
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
                        min: actionLogStandardDeviationBounds.minimum,
                        max: actionLogStandardDeviationBounds.maximum)))
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
                    resumableSnapshotVersion: VectorPPOTrainingState
                        .currentResumableSnapshotVersion,
                    rolloutEnvironmentCount: n,
                    optimizerSteps: optimizer.step,
                    adaptiveLearningRate: adaptiveLearningRate,
                    successReplayCount: successReplayCount)
                let successReplayArrays = Self.successReplayCheckpointArrays(
                    observations: successReplayObservations,
                    actions: successReplayActions,
                    expertGates: successReplayExpertGates,
                    standExpertGates: successReplayStandExpertGates,
                    auxiliaryExpertGates:
                        successReplayAuxiliaryExpertGates,
                    count: successReplayCount, next: successReplayNext,
                    capacity: successReplayCapacity,
                    observationDimension: obsDim, actionDimension: actionDim)
                // Preserve the policy trajectory for deterministic selection.
                // A noisy on-policy update can regress after a good checkpoint;
                // overwriting the only weights makes that impossible to audit.
                // Publish the immutable generation first. The mutable root is
                // only a compatibility mirror and is never used for resume.
                _ = try VectorPolicyCheckpointDiscovery
                    .publishCompleteCheckpoint(
                        inRunDirectory: outputDirectory,
                        completedUpdates: update + 1,
                        task: spec.id, taskRevision: spec.revision
                    ) { staging in
                        try Self.save(
                            policy: policy, normalizer: normalizer,
                            optimizer: optimizer,
                            referencePolicy: referencePolicy,
                            successReplayArrays: successReplayArrays,
                            taskSpec: spec,
                            configuration: persistedConfiguration,
                            trainingState: trainingState,
                            inferenceBatchSize: policyInferenceBatchSize,
                            outputDirectory: staging.path)
                    }
                try Self.save(policy: policy, normalizer: normalizer,
                              optimizer: optimizer,
                              referencePolicy: referencePolicy,
                              successReplayArrays: successReplayArrays,
                              taskSpec: spec,
                              configuration: persistedConfiguration,
                              trainingState: trainingState,
                              inferenceBatchSize: policyInferenceBatchSize,
                              outputDirectory: outputDirectory)
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
        // Evaluation, replay, and deployment must use the same saved inference
        // row geometry. MLX can select a different matrix kernel for a one-row
        // or small evaluation batch than for the training batch; the resulting
        // last-bit reduction differences are enough to change a contact-
        // critical trajectory. VectorPolicyRunner pads to `inferenceBatchSize`
        // and returns only the requested rows, giving every frontend identical
        // deterministic actions.
        let runner = try VectorPolicyRunner(
            checkpointDirectory: checkpointDirectory)
        let policyExpertGate = task as? any PolicyExpertGateProviding
        let usesPolicyExpertGate = policyExpertGate?.usesPolicyExpertGate == true
        let expertActionMask = usesPolicyExpertGate
            ? policyExpertGate?.policyExpertActionMask : nil
        if let expertActionMask {
            guard expertActionMask.count == spec.action.elementCount,
                  expertActionMask.allSatisfy({ (0...1).contains($0) }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task expert action mask must match action dimensions")
            }
        }
        let policyStandExpertGate = task as? any PolicyStandExpertGateProviding
        let usesPolicyStandExpertGate =
            policyStandExpertGate?.usesPolicyStandExpertGate == true
        let standExpertActionMask = usesPolicyStandExpertGate
            ? policyStandExpertGate?.policyStandExpertActionMask : nil
        if let standExpertActionMask {
            guard standExpertActionMask.count == spec.action.elementCount,
                  standExpertActionMask.allSatisfy({ (0...1).contains($0) }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task stand-expert action mask must match action dimensions")
            }
        }
        let policyAuxiliaryExpertGate = task as?
            any PolicyAuxiliaryExpertGateProviding
        let usesPolicyAuxiliaryExpertGate =
            policyAuxiliaryExpertGate?.usesPolicyAuxiliaryExpertGate == true
        let auxiliaryExpertActionMask = usesPolicyAuxiliaryExpertGate
            ? policyAuxiliaryExpertGate?.policyAuxiliaryExpertActionMask : nil
        if let auxiliaryExpertActionMask {
            guard auxiliaryExpertActionMask.count == spec.action.elementCount,
                  auxiliaryExpertActionMask.allSatisfy({
                      (0...1).contains($0)
                  }) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task auxiliary-expert action mask must match action dimensions")
            }
        }
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
            let expertGate = usesPolicyExpertGate
                ? policyExpertGate!.policyExpertGates(observation.policy) : nil
            let standExpertGate = usesPolicyStandExpertGate
                ? policyStandExpertGate!.policyStandExpertGates(
                    observation.policy) : nil
            let auxiliaryExpertGate = usesPolicyAuxiliaryExpertGate
                ? policyAuxiliaryExpertGate!.policyAuxiliaryExpertGates(
                    observation.policy) : nil
            var actionValues = try runner.actions(
                for: observation.policy,
                expertGates: expertGate,
                expertActionMask: expertActionMask,
                standExpertGates: standExpertGate,
                standExpertActionMask: standExpertActionMask,
                auxiliaryExpertGates: auxiliaryExpertGate,
                auxiliaryExpertActionMask: auxiliaryExpertActionMask)
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
            try stepResult.validate(for: spec)
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
            try observation.validate(for: spec)
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
        let metrics = PPOEvaluationMetrics(
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
        try metrics.validateStructure()
        return metrics
    }

    /// Content identity for every file that changes deterministic evaluation
    /// semantics. Optimizer moments are intentionally excluded because they
    /// cannot affect policy replay.
    public static func checkpointFingerprint(directory: String) throws -> String {
        let root = URL(fileURLWithPath: directory, isDirectory: true)
        let metadataData = try Data(contentsOf: root.appendingPathComponent(
            "metadata.json"))
        let policyData = try Data(contentsOf: root.appendingPathComponent(
            "policy.safetensors"))
        let trainingStateData = try Data(contentsOf: root.appendingPathComponent(
            "training-state.json"))
        return checkpointFingerprint(
            metadataData: metadataData, policyData: policyData,
            trainingStateData: trainingStateData)
    }

    /// Hash exact bytes already read by a deployment or export operation.
    /// This is intentionally module-internal: filesystem callers use the
    /// public directory API, while trusted bundle code can avoid re-opening a
    /// mutable path between authentication and model construction.
    static func checkpointFingerprint(
        metadataData: Data,
        policyData: Data,
        trainingStateData: Data
    ) -> String {
        var hasher = SHA256()
        for (name, data) in [
            ("metadata.json", metadataData),
            ("policy.safetensors", policyData),
            ("training-state.json", trainingStateData),
        ] {
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

    /// A time-limit transition bootstraps only when the task did not also
    /// reach a true terminal state on that same control step.
    static func shouldBootstrapFinalObservation(
        terminated: Bool, truncated: Bool
    ) -> Bool {
        truncated && !terminated
    }

    public static func gaussianLogProbability(_ action: MLXArray, mean: MLXArray,
                                              logStandardDeviation: MLXArray,
                                              actionMask: MLXArray? = nil)
        -> MLXArray {
        let z = (action - mean) / exp(logStandardDeviation)
        var terms = -0.5 * z.square() - logStandardDeviation - logSqrt2Pi
        if let actionMask { terms = terms * actionMask }
        return sum(terms, axis: -1)
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

    /// Host mirror of the exact old-policy-to-new-policy diagonal Gaussian KL
    /// used by the trust-region scheduler. Keeping the arithmetic testable
    /// without a Metal test bundle guards both direction and row weighting.
    static func weightedDiagonalGaussianKL(
        oldMeans: [Float], oldLogStandardDeviations: [Float],
        newMeans: [Float], newLogStandardDeviations: [Float],
        actionDimension: Int, weights: [Float]
    ) -> Float {
        precondition(actionDimension > 0)
        precondition(oldMeans.count == newMeans.count)
        precondition(oldMeans.count == weights.count * actionDimension)
        precondition(oldLogStandardDeviations.count == oldMeans.count)
        precondition(newLogStandardDeviations.count == oldMeans.count)
        var weightedSum: Float = 0
        var weightSum: Float = 0
        for row in weights.indices {
            let weight = weights[row]
            precondition(weight >= 0 && weight.isFinite)
            var rowKL: Float = 0
            for action in 0..<actionDimension {
                let index = row * actionDimension + action
                let oldLogStd = oldLogStandardDeviations[index]
                let newLogStd = newLogStandardDeviations[index]
                let delta = oldMeans[index] - newMeans[index]
                let oldVariance = exp(2 * oldLogStd)
                let newVariance = exp(2 * newLogStd)
                rowKL += newLogStd - oldLogStd
                    + (oldVariance + delta * delta) / (2 * newVariance)
                    - 0.5
            }
            weightedSum += weight * max(rowKL, 0)
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
                             successReplayArrays: [String: MLXArray]?,
                             taskSpec: RLTaskSpec, configuration: VectorPPOConfig,
                             trainingState: VectorPPOTrainingState,
                             inferenceBatchSize: Int,
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
        let replayURL = URL(fileURLWithPath: outputDirectory)
            .appendingPathComponent(successReplayFileName)
        if let successReplayArrays {
            let replayData = try MLX.saveToData(arrays: successReplayArrays)
            try replayData.write(to: replayURL, options: .atomic)
        } else if FileManager.default.fileExists(atPath: replayURL.path) {
            try FileManager.default.removeItem(at: replayURL)
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
            inferenceBatchSize: inferenceBatchSize,
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
        let liveTaskMetrics = metrics.taskMetrics
            .filter { $0.key.hasPrefix("task/") }
            .sorted { $0.key < $1.key }
        if !liveTaskMetrics.isEmpty {
            Swift.print("  " + liveTaskMetrics.map {
                String(format: "%@ %.4f", $0.key, $0.value)
            }.joined(separator: "  "))
        }
    }
}
