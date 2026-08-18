import Foundation

// MARK: - Tensor-shaped, vectorized RL contract

public enum RLTensorDataType: String, Sendable {
    case float32
    case uint8
}

/// A named row-major tensor. The leading environment dimension is implicit:
/// an observation with shape [41] is stored as [numEnvs, 41].
public struct RLTensorSpec: Equatable, Sendable {
    public var name: String
    public var shape: [Int]
    public var dataType: RLTensorDataType
    public var lowerBound: [Float]?
    public var upperBound: [Float]?

    public init(name: String, shape: [Int], dataType: RLTensorDataType = .float32,
                lowerBound: [Float]? = nil, upperBound: [Float]? = nil) {
        precondition(!shape.isEmpty && shape.allSatisfy { $0 > 0 },
                     "tensor dimensions must be positive")
        self.name = name
        self.shape = shape
        self.dataType = dataType
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        if let lowerBound { precondition(lowerBound.count == elementCount) }
        if let upperBound { precondition(upperBound.count == elementCount) }
    }

    public var elementCount: Int { shape.reduce(1, *) }
}

/// Central compatibility namespace for physics changes shared by otherwise
/// independent task revision families.
public enum RLPhysicsContract {
    /// Fixed-gain actuator model with explicit scene-authored modes, no
    /// adaptive motor penalty, and an exact active-set effort clamp.
    /// Retained to decode and verify historical epoch-1 checkpoints.
    public static func fixedGainActuatorV2(_ taskRevision: Int) -> Int {
        precondition(taskRevision > 0 && taskRevision < 1_000_000)
        return 1_000_000 + taskRevision
    }

    /// Epoch-2 simulator contract. It includes the fixed-gain actuator model
    /// and requires exact dynamic-color dispatch, validated coloring, and
    /// fail-closed contact-capacity handling. The local task revision remains
    /// visible in the low six digits for independent task evolution.
    public static func deterministicColorSolveV1(_ taskRevision: Int) -> Int {
        precondition(taskRevision > 0 && taskRevision < 1_000_000)
        return 2_000_000 + taskRevision
    }
}

public struct RLTaskSpec: Equatable, Sendable {
    public var id: String
    /// Compatibility revision for actuator, observation, reward, termination,
    /// or success semantics that can change without changing tensor shapes.
    public var revision: Int
    public var numEnvironments: Int
    public var observation: RLTensorSpec
    public var privilegedObservation: RLTensorSpec?
    public var action: RLTensorSpec
    public var maxEpisodeSteps: Int
    public var simulationStep: Float
    public var controlDecimation: Int
    public var autoReset: Bool
    /// Semantic task options that affect observations, actions, dynamics,
    /// rewards, termination, or success. Runtime-only values such as seed,
    /// environment count, and auto-reset are deliberately excluded. Saving
    /// this map with a policy prevents a resume from silently changing its
    /// MDP while retaining the same task id and tensor shapes.
    public var configurationValues: [String: Float]

    public init(id: String, revision: Int = 1,
                numEnvironments: Int, observation: RLTensorSpec,
                privilegedObservation: RLTensorSpec? = nil, action: RLTensorSpec,
                maxEpisodeSteps: Int, simulationStep: Float,
                controlDecimation: Int, autoReset: Bool = true,
                configurationValues: [String: Float] = [:]) {
        precondition(numEnvironments > 0)
        precondition(revision > 0)
        precondition(maxEpisodeSteps > 0)
        precondition(simulationStep > 0)
        precondition(controlDecimation > 0)
        self.id = id
        self.revision = revision
        self.numEnvironments = numEnvironments
        self.observation = observation
        self.privilegedObservation = privilegedObservation
        self.action = action
        self.maxEpisodeSteps = maxEpisodeSteps
        self.simulationStep = simulationStep
        self.controlDecimation = controlDecimation
        self.autoReset = autoReset
        self.configurationValues = configurationValues
    }

    public var controlStep: Float { simulationStep * Float(controlDecimation) }
}

public enum RLEnvironmentError: Error, CustomStringConvertible,
    LocalizedError
{
    case invalidActionCount(expected: Int, actual: Int)
    case invalidActionShape(expected: [Int], actual: [Int])
    case nonFiniteAction(index: Int)
    case invalidObservationCount(expected: Int, actual: Int)
    case invalidEnvironmentIndex(Int)
    case duplicateTask(String)
    case unknownTask(String, available: [String])
    case invalidConfiguration(String)

    public var description: String {
        switch self {
        case let .invalidActionCount(expected, actual):
            return "expected \(expected) action values, got \(actual)"
        case let .invalidActionShape(expected, actual):
            return "expected action shape \(expected), got \(actual)"
        case let .nonFiniteAction(index):
            return "action value at flat index \(index) is not finite"
        case let .invalidObservationCount(expected, actual):
            return "expected \(expected) observation values, got \(actual)"
        case let .invalidEnvironmentIndex(index):
            return "environment index \(index) is out of range"
        case let .duplicateTask(id):
            return "RL task '\(id)' is already registered"
        case let .unknownTask(id, available):
            return "unknown RL task '\(id)'; available: \(available.joined(separator: ", "))"
        case let .invalidConfiguration(message):
            return "invalid RL configuration: \(message)"
        }
    }

    public var errorDescription: String? { description }
}

/// Contiguous [environment, action] values. Bounds are task metadata, not a
/// universal assumption: normalized-control tasks may use [-1, 1], while
/// Isaac/RSL position-offset tasks intentionally consume Gaussian actions.
public struct RLActionBatch: Sendable {
    public let numEnvironments: Int
    public let actionDimension: Int
    public var values: ContiguousArray<Float>

    public init(numEnvironments: Int, actionDimension: Int,
                values: ContiguousArray<Float>) throws {
        guard numEnvironments > 0, actionDimension > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "action batch dimensions must be positive")
        }
        let expected = numEnvironments * actionDimension
        guard values.count == expected else {
            throw RLEnvironmentError.invalidActionCount(expected: expected, actual: values.count)
        }
        self.numEnvironments = numEnvironments
        self.actionDimension = actionDimension
        self.values = values
    }

    public init(spec: RLTaskSpec, repeating value: Float = 0) {
        numEnvironments = spec.numEnvironments
        actionDimension = spec.action.elementCount
        values = ContiguousArray(repeating: value,
                                 count: spec.numEnvironments * spec.action.elementCount)
    }

    public subscript(environment: Int, component: Int) -> Float {
        get { values[environment * actionDimension + component] }
        set { values[environment * actionDimension + component] = newValue }
    }

    public func validate(for spec: RLTaskSpec) throws {
        guard numEnvironments == spec.numEnvironments,
              actionDimension == spec.action.elementCount else {
            throw RLEnvironmentError.invalidActionShape(
                expected: [spec.numEnvironments, spec.action.elementCount],
                actual: [numEnvironments, actionDimension])
        }
        let expected = numEnvironments * actionDimension
        guard values.count == expected else {
            throw RLEnvironmentError.invalidActionCount(expected: expected,
                                                         actual: values.count)
        }
        if let index = values.firstIndex(where: { !$0.isFinite }) {
            throw RLEnvironmentError.nonFiniteAction(index: index)
        }
    }
}

public struct RLObservationBatch: Sendable {
    public var policy: ContiguousArray<Float>
    public var privileged: ContiguousArray<Float>

    public init(spec: RLTaskSpec) {
        policy = ContiguousArray(
            repeating: 0,
            count: spec.numEnvironments * spec.observation.elementCount)
        privileged = ContiguousArray(
            repeating: 0,
            count: spec.numEnvironments * (spec.privilegedObservation?.elementCount ?? 0))
    }

    public func validate(for spec: RLTaskSpec) throws {
        let expected = spec.numEnvironments * spec.observation.elementCount
        guard policy.count == expected else {
            throw RLEnvironmentError.invalidObservationCount(expected: expected,
                                                              actual: policy.count)
        }
        let privilegedExpected = spec.numEnvironments
            * (spec.privilegedObservation?.elementCount ?? 0)
        guard privileged.count == privilegedExpected else {
            throw RLEnvironmentError.invalidObservationCount(expected: privilegedExpected,
                                                              actual: privileged.count)
        }
        if let index = policy.firstIndex(where: { !$0.isFinite }) {
            throw RLEnvironmentError.invalidConfiguration(
                "policy observation at flat index \(index) is not finite")
        }
        if let index = privileged.firstIndex(where: { !$0.isFinite }) {
            throw RLEnvironmentError.invalidConfiguration(
                "privileged observation at flat index \(index) is not finite")
        }
    }
}

/// Output storage is caller-owned and reusable. This avoids a fresh nest of
/// arrays on every control step and makes the simulation-to-MLX boundary a
/// single contiguous copy today, with a direct Metal-buffer path possible.
public struct RLStepBatch: Sendable {
    public var observations: RLObservationBatch
    public var rewards: ContiguousArray<Float>
    public var terminated: ContiguousArray<Bool>
    public var truncated: ContiguousArray<Bool>
    /// Instantaneous task-success state for this transition. Algorithms own
    /// episode reductions such as success-once; it need not coincide with a
    /// termination when a task uses fixed-horizon evaluation.
    public var successes: ContiguousArray<Bool>
    /// Task-certified transition worth retaining for self-imitation. This is
    /// deliberately separate from `successes`: curricula may preserve rare
    /// useful behavior (for example, a stable physical lift) without inflating
    /// final task success or changing Gymnasium termination semantics.
    public var imitationMilestones: ContiguousArray<Bool>
    /// Terminal state before auto-reset. Only rows marked by
    /// `hasFinalObservation` are meaningful.
    public var finalObservations: ContiguousArray<Float>
    public var hasFinalObservation: ContiguousArray<Bool>
    /// Per-step decomposed reward/diagnostic terms, each [numEnvironments].
    public var metrics: [String: ContiguousArray<Float>]

    public init(spec: RLTaskSpec) {
        observations = RLObservationBatch(spec: spec)
        rewards = ContiguousArray(repeating: 0, count: spec.numEnvironments)
        terminated = ContiguousArray(repeating: false, count: spec.numEnvironments)
        truncated = ContiguousArray(repeating: false, count: spec.numEnvironments)
        successes = ContiguousArray(repeating: false, count: spec.numEnvironments)
        imitationMilestones = ContiguousArray(
            repeating: false, count: spec.numEnvironments)
        finalObservations = ContiguousArray(
            repeating: 0,
            count: spec.numEnvironments * spec.observation.elementCount)
        hasFinalObservation = ContiguousArray(repeating: false,
                                              count: spec.numEnvironments)
        metrics = [:]
    }

    public mutating func clearSignals() {
        for i in rewards.indices { rewards[i] = 0 }
        for i in terminated.indices { terminated[i] = false }
        for i in truncated.indices { truncated[i] = false }
        for i in successes.indices { successes[i] = false }
        for i in imitationMilestones.indices { imitationMilestones[i] = false }
        for i in hasFinalObservation.indices { hasFinalObservation[i] = false }
        metrics.removeAll(keepingCapacity: true)
    }

    public func validate(for spec: RLTaskSpec) throws {
        try observations.validate(for: spec)
        let n = spec.numEnvironments
        guard rewards.count == n, terminated.count == n, truncated.count == n,
              successes.count == n, imitationMilestones.count == n,
              hasFinalObservation.count == n else {
            throw RLEnvironmentError.invalidConfiguration(
                "RLStepBatch signal storage does not match \(n) environments")
        }
        let expectedFinal = n * spec.observation.elementCount
        guard finalObservations.count == expectedFinal else {
            throw RLEnvironmentError.invalidObservationCount(expected: expectedFinal,
                                                              actual: finalObservations.count)
        }
        if let index = rewards.firstIndex(where: { !$0.isFinite }) {
            throw RLEnvironmentError.invalidConfiguration(
                "reward at environment \(index) is not finite")
        }
        if let index = finalObservations.firstIndex(where: { !$0.isFinite }) {
            throw RLEnvironmentError.invalidConfiguration(
                "final observation at flat index \(index) is not finite")
        }
        for name in metrics.keys.sorted() {
            guard let values = metrics[name] else { continue }
            guard values.count == n else {
                throw RLEnvironmentError.invalidConfiguration(
                    "metric '\(name)' has \(values.count) rows; expected \(n)")
            }
            if let index = values.firstIndex(where: { !$0.isFinite }) {
                throw RLEnvironmentError.invalidConfiguration(
                    "metric '\(name)' at environment \(index) is not finite")
            }
        }
        for environment in 0..<n {
            // Gymnasium permits termination and truncation to coincide. A
            // true terminal state must never be bootstrapped even when the
            // time limit also fired, so only a pure auto-reset truncation
            // requires the pre-reset observation side channel.
            if spec.autoReset && truncated[environment]
                && !terminated[environment]
                && !hasFinalObservation[environment] {
                throw RLEnvironmentError.invalidConfiguration(
                    "truncated environment \(environment) has no final observation")
            }
            if hasFinalObservation[environment]
                && !terminated[environment] && !truncated[environment] {
                throw RLEnvironmentError.invalidConfiguration(
                    "environment \(environment) publishes a final observation without ending")
            }
        }
    }
}

/// Isaac-Lab/MJX-shaped environment boundary. Implementations may use a
/// direct task class for speed or compose manager-like terms internally; the
/// learner only sees stable batched tensors and Gymnasium termination rules.
public protocol VectorizedRLTask: AnyObject {
    var spec: RLTaskSpec { get }
    func reset(environments: [Int]?, seed: UInt64,
               into observations: inout RLObservationBatch) throws
    func step(actions: RLActionBatch, into result: inout RLStepBatch) throws
}

/// Controller boundary shared by learned policies, classical baselines, and
/// future planners that act on a `VectorizedRLTask`. Providers own inference
/// or controller memory; tasks continue to own simulation state and physics.
/// Keeping this contract in AVBDCore prevents replay/evaluation code from
/// branching on a concrete controller implementation.
public protocol RLActionProvider: AnyObject {
    /// Stable diagnostic identity exposed to replay/evaluation front ends.
    var actionProviderID: String { get }

    /// Reset provider-owned state after the task has produced post-reset
    /// observations. `environments == nil` means the complete batch; a list
    /// supports task-side auto-reset without disturbing other controller rows.
    func reset(
        for task: any VectorizedRLTask,
        environments: [Int]?,
        observation: RLObservationBatch
    ) throws

    /// Produce one task-shaped action batch from the measured observation.
    func actions(
        for observation: RLObservationBatch,
        task: any VectorizedRLTask
    ) throws -> RLActionBatch
}

public extension RLActionProvider {
    func reset(
        for task: any VectorizedRLTask,
        environments: [Int]?,
        observation: RLObservationBatch
    ) throws {}

    /// Whole-batch convenience used after an explicit task reset.
    func reset(
        for task: any VectorizedRLTask,
        observation: RLObservationBatch
    ) throws {
        try reset(for: task, environments: nil, observation: observation)
    }

    /// Synchronize stateful provider rows after a task performs Gymnasium
    /// auto-reset inside `step`. Stateless learned policies use the default
    /// no-op reset witness; CPGs and recurrent policies can reset only rows
    /// whose returned observation belongs to a new episode.
    func resetAfterStep(
        for task: any VectorizedRLTask,
        result: RLStepBatch
    ) throws {
        guard task.spec.autoReset else { return }
        try result.validate(for: task.spec)
        let resetEnvironments = (0..<task.spec.numEnvironments).filter {
            result.terminated[$0] || result.truncated[$0]
        }
        guard !resetEnvironments.isEmpty else { return }
        try reset(
            for: task, environments: resetEnvironments,
            observation: result.observations)
    }
}

/// Optional exact symmetry of a task's policy tensors. Algorithms can use
/// this for data augmentation or actor mirror consistency without a reference
/// trajectory, gait phase, or task-specific branch in the learner. Both
/// transforms must be row-wise involutions: mirroring twice returns the
/// original tensor, and row order is preserved.
public protocol PolicySymmetryProviding: AnyObject {
    /// Signed permutation for differentiable actor-mean mirror loss. Entry j
    /// of a mirrored action is `signs[j] * action[sources[j]]`.
    var policyActionMirrorSourceIndices: [Int] { get }
    var policyActionMirrorSigns: [Float] { get }
    func mirrorPolicyObservations(
        _ observations: ContiguousArray<Float>) -> ContiguousArray<Float>
    func mirrorPolicyActions(
        _ actions: ContiguousArray<Float>) -> ContiguousArray<Float>
}

/// Optional task-side curriculum lifecycle. Trainers enable it explicitly;
/// evaluation and replay leave tasks in their full-difficulty default mode.
public protocol TrainingModeConfigurable: AnyObject {
    func setTrainingMode(_ enabled: Bool)
    /// Restore task-side curricula when training resumes from a checkpoint.
    func setTrainingProgress(environmentSteps: Int)
}

/// A task can declare conservative variance floors for observation channels
/// whose meaning or command range expands during an explicit policy transfer.
/// The trainer applies these only to the imported normalizer snapshot; all
/// other source-task statistics remain byte-for-byte unchanged. This keeps
/// transfer logic generic while the task remains the owner of observation
/// semantics.
public protocol ObservationNormalizerTransferProviding: AnyObject {
    var initializationObservationVarianceFloors: [Int: Double] { get }
}

/// Optional explicit observation-schema migration for policy transfer. The
/// returned array is indexed by destination observation channel; each value
/// names the source channel to copy, while `nil` creates a new zero-weight,
/// zero-mean channel. This lets tasks append or reorder measured state without
/// teaching the generic learner what any channel means.
public protocol ObservationSchemaTransferProviding: AnyObject {
    func initializationObservationSourceIndices(
        sourceDimension: Int) -> [Int?]?
}

/// Optional training-only mask for transfer-policy retention. Algorithms that
/// regularize a fine-tuned actor toward its frozen initialization use one
/// weight per environment. A task may set a row to zero once the state enters
/// the adaptation regime (for example, after a measured disturbance), while
/// retaining the source behavior elsewhere. The mask is never a policy input
/// and has no effect during evaluation or replay.
public protocol PolicyReferenceRegularizationProviding: AnyObject {
    func policyReferenceRegularizationWeights(
        _ observations: ContiguousArray<Float>) -> ContiguousArray<Float>
}

/// Optional action-wise mask for initialization-policy retention. Row weights
/// decide *when* retention applies; this mask decides *which actuator
/// outputs* remain anchored. The returned tensor is row-major
/// `[environment, action]`. This is useful for compositional control where a
/// verified upper-body grasp must remain fixed while a lower-body locomotion
/// branch adapts to the resulting load and center-of-mass shift.
public protocol PolicyReferenceActionRegularizationProviding: AnyObject {
    func policyReferenceActionRegularizationWeights(
        _ observations: ContiguousArray<Float>,
        actionDimension: Int
    ) -> ContiguousArray<Float>
}

/// Optional task-owned actor update mask. The generic learner still trains
/// the critic on every transition, but multiplies the PPO actor loss by one
/// weight per environment. This is useful for staged contact tasks where a
/// large prerequisite-state distribution would otherwise overwhelm the much
/// smaller adaptation regime. It does not alter exploration or physics.
public protocol PolicyActorTrainingWeightProviding: AnyObject {
    func policyActorTrainingWeights(
        _ observations: ContiguousArray<Float>) -> ContiguousArray<Float>
}

/// Optional hard routing signal for a two-expert actor. The task derives the
/// gate from its raw policy observation, while PPO remains agnostic to what
/// the modes mean. A value of one selects the specialized expert and zero
/// selects the base expert; intermediate values are allowed for smooth task
/// transitions.
public protocol PolicyExpertGateProviding: AnyObject {
    var usesPolicyExpertGate: Bool { get }
    var freezesBasePolicyExpert: Bool { get }
    /// Optional action-space mask for compositional control. A value of one
    /// routes that actuator through this expert and zero leaves it available
    /// to the base or another disjoint expert. `nil` preserves whole-action
    /// routing.
    var policyExpertActionMask: ContiguousArray<Float>? { get }
    /// An explicit transfer can initialize the routed expert as an exact copy
    /// of the source checkpoint's base actor. This makes activating a new
    /// task-owned gate behavior-identical before specialist training begins.
    var initializesPolicyExpertFromBaseOnTransfer: Bool { get }
    /// An explicit transfer can instead initialize the routed expert as the
    /// exact symmetry conjugate of the source base actor:
    /// `expert(o) = mirrorAction(base(mirrorObservation(o)))`. This is useful
    /// when a verified policy is robust on one side of a symmetric task and
    /// avoids relearning the opposite side from sparse failures.
    var initializesPolicyExpertFromMirroredBaseOnTransfer: Bool { get }
    func policyExpertGates(
        _ observations: ContiguousArray<Float>) -> ContiguousArray<Float>
}

/// Optional second hard routing signal for a three-mode actor. The ordinary
/// expert gate remains the low-speed locomotion branch; this gate selects an
/// independent exact-stand branch. Keeping the two gates task-owned lets PPO
/// train cruise, creep, and balance without embedding locomotion semantics in
/// the algorithm or blending incompatible actions at mode boundaries.
public protocol PolicyStandExpertGateProviding: PolicyExpertGateProviding {
    var usesPolicyStandExpertGate: Bool { get }
    /// Keep a verified low-speed/braking branch byte-stable while training
    /// only the independently routed stand branch.
    var freezesLowSpeedPolicyExpert: Bool { get }
    /// An explicit transfer can clone the already learned routed expert into
    /// the new third branch. The branch switch is then behavior-identical on
    /// its first transition while subsequent PPO updates remain isolated.
    var initializesPolicyStandExpertFromPolicyExpertOnTransfer: Bool { get }
    /// Initialize the third branch from the verified base actor instead. This
    /// is the useful starting point for a loaded-locomotion specialist whose
    /// upper body is supplied by a separate frozen manipulation expert.
    var initializesPolicyStandExpertFromBaseOnTransfer: Bool { get }
    /// Optional action-space mask for compositional experts. A value of one
    /// routes that action through the stand expert; zero keeps the base
    /// expert's action even while the stand mode is active. `nil` preserves
    /// whole-action routing. This lets tasks reuse a verified locomotion
    /// controller for legs while independently learning an upper-body skill.
    var policyStandExpertActionMask: ContiguousArray<Float>? { get }
    func policyStandExpertGates(
        _ observations: ContiguousArray<Float>) -> ContiguousArray<Float>
}

/// Optional fourth routed actor for compositional skills that need to preserve
/// three already verified behaviors while adapting a disjoint actuator group.
/// The task owns both the scalar gate and action mask; PPO only composes the
/// resulting convex action-wise mixture.
public protocol PolicyAuxiliaryExpertGateProviding:
    PolicyStandExpertGateProviding
{
    var usesPolicyAuxiliaryExpertGate: Bool { get }
    /// Initialize the fourth branch from the verified base actor. This gives
    /// a trainable residual skill (for example loaded legs) the source
    /// policy's real behavior rather than an unrelated stationary branch.
    var initializesPolicyAuxiliaryExpertFromBaseOnTransfer: Bool { get }
    /// Observation columns whose first-layer influence must start at zero in
    /// the transferred auxiliary branch. This is useful when an action-wise
    /// expert owns only a subsystem: the source whole-body policy may depend
    /// strongly on state from actuators now controlled by another branch.
    /// The projected columns remain ordinary trainable parameters after
    /// initialization, so the specialist can relearn useful coupling.
    var policyAuxiliaryExpertZeroedObservationIndicesOnTransfer: [Int] { get }
    /// Keep the existing third branch byte-stable while the new auxiliary
    /// branch learns. This is independent of `freezesLowSpeedPolicyExpert`,
    /// which freezes the first routed branch.
    var freezesStandPolicyExpert: Bool { get }
    var policyAuxiliaryExpertActionMask: ContiguousArray<Float>? { get }
    func policyAuxiliaryExpertGates(
        _ observations: ContiguousArray<Float>) -> ContiguousArray<Float>
}

public extension TrainingModeConfigurable {
    func setTrainingProgress(environmentSteps: Int) { _ = environmentSteps }
}

public extension PolicyExpertGateProviding {
    var policyExpertActionMask: ContiguousArray<Float>? { nil }
    var initializesPolicyExpertFromBaseOnTransfer: Bool { false }
    var initializesPolicyExpertFromMirroredBaseOnTransfer: Bool { false }
}

public extension PolicyStandExpertGateProviding {
    var initializesPolicyStandExpertFromPolicyExpertOnTransfer: Bool { false }
    var initializesPolicyStandExpertFromBaseOnTransfer: Bool { false }
    var policyStandExpertActionMask: ContiguousArray<Float>? { nil }
}

public extension PolicyAuxiliaryExpertGateProviding {
    var initializesPolicyAuxiliaryExpertFromBaseOnTransfer: Bool { false }
    var policyAuxiliaryExpertZeroedObservationIndicesOnTransfer: [Int] { [] }
    var policyAuxiliaryExpertActionMask: ContiguousArray<Float>? { nil }
    var freezesStandPolicyExpert: Bool { false }
}

/// Task-owned, algorithm-independent acceptance gate for deterministic policy
/// evaluation. This keeps "checkpoint exists" separate from "checkpoint is
/// good enough to publish" and lets every new task define measurable criteria
/// without adding branches to an RL algorithm.
public struct RLEvaluationCriteria: Equatable, Sendable {
    public var minimumSuccessRate: Float
    public var minimumMeanEpisodeLengthFraction: Float
    public var minimumTaskMetrics: [String: Float]
    public var maximumTaskMetrics: [String: Float]

    public init(minimumSuccessRate: Float,
                minimumMeanEpisodeLengthFraction: Float = 0,
                minimumTaskMetrics: [String: Float] = [:],
                maximumTaskMetrics: [String: Float] = [:]) {
        precondition(minimumSuccessRate.isFinite
            && (0...1).contains(minimumSuccessRate))
        precondition(minimumMeanEpisodeLengthFraction.isFinite
            && (0...1).contains(minimumMeanEpisodeLengthFraction))
        precondition(minimumTaskMetrics.values.allSatisfy(\.isFinite))
        precondition(maximumTaskMetrics.values.allSatisfy(\.isFinite))
        precondition(Set(minimumTaskMetrics.keys).intersection(
            maximumTaskMetrics.keys).allSatisfy {
                minimumTaskMetrics[$0]! <= maximumTaskMetrics[$0]!
            })
        self.minimumSuccessRate = minimumSuccessRate
        self.minimumMeanEpisodeLengthFraction = minimumMeanEpisodeLengthFraction
        self.minimumTaskMetrics = minimumTaskMetrics
        self.maximumTaskMetrics = maximumTaskMetrics
    }

    public func failures(successRate: Float, meanEpisodeLength: Float,
                         maxEpisodeSteps: Int,
                         taskMetrics: [String: Float]) -> [String] {
        var failures = [String]()
        let validMinimumSuccessRate = minimumSuccessRate.isFinite
            && (0...1).contains(minimumSuccessRate)
        if !validMinimumSuccessRate {
            failures.append("invalid minimum_success_rate threshold")
        }
        if !successRate.isFinite || !(0...1).contains(successRate) {
            failures.append("success_rate is not finite or outside [0, 1]")
        } else if validMinimumSuccessRate && successRate < minimumSuccessRate {
            failures.append(String(format: "success_rate %.4f < %.4f",
                                   successRate, minimumSuccessRate))
        }
        let validMinimumLengthFraction =
            minimumMeanEpisodeLengthFraction.isFinite
            && (0...1).contains(minimumMeanEpisodeLengthFraction)
        if !validMinimumLengthFraction {
            failures.append(
                "invalid minimum_mean_episode_length_fraction threshold")
        }
        if maxEpisodeSteps <= 0 {
            failures.append("max_episode_steps must be positive")
        }
        let minimumLength = validMinimumLengthFraction && maxEpisodeSteps > 0
            ? minimumMeanEpisodeLengthFraction * Float(maxEpisodeSteps) : 0
        if !meanEpisodeLength.isFinite || meanEpisodeLength < 0 {
            failures.append("mean_episode_length is not finite or is negative")
        } else if maxEpisodeSteps > 0
                    && meanEpisodeLength > Float(maxEpisodeSteps) {
            failures.append("mean_episode_length exceeds max_episode_steps")
        } else if validMinimumLengthFraction && maxEpisodeSteps > 0
                    && meanEpisodeLength < minimumLength {
            failures.append(String(format: "mean_episode_length %.3f < %.3f",
                                   meanEpisodeLength, minimumLength))
        }
        let nonFiniteMetricNames = Set(taskMetrics.compactMap {
            $0.value.isFinite ? nil : $0.key
        })
        for name in nonFiniteMetricNames.sorted() {
            failures.append("metric \(name) is not finite")
        }
        for (name, minimum) in minimumTaskMetrics.sorted(by: { $0.key < $1.key }) {
            guard minimum.isFinite else {
                failures.append("invalid minimum threshold for metric \(name)")
                continue
            }
            guard let value = taskMetrics[name] else {
                failures.append("missing metric \(name)")
                continue
            }
            guard value.isFinite else { continue }
            if value < minimum {
                failures.append(String(format: "%@ %.5f < %.5f",
                                       name, value, minimum))
            }
        }
        for (name, maximum) in maximumTaskMetrics.sorted(by: { $0.key < $1.key }) {
            guard maximum.isFinite else {
                failures.append("invalid maximum threshold for metric \(name)")
                continue
            }
            guard let value = taskMetrics[name] else {
                failures.append("missing metric \(name)")
                continue
            }
            guard value.isFinite else { continue }
            if value > maximum {
                failures.append(String(format: "%@ %.5f > %.5f",
                                       name, value, maximum))
            }
        }
        return failures
    }
}

public protocol RLEvaluationCriteriaProviding: AnyObject {
    var evaluationCriteria: RLEvaluationCriteria { get }
}

public extension VectorizedRLTask {
    func reset(seed: UInt64) throws -> RLObservationBatch {
        var result = RLObservationBatch(spec: spec)
        try reset(environments: nil, seed: seed, into: &result)
        try result.validate(for: spec)
        return result
    }

    func step(actions: RLActionBatch) throws -> RLStepBatch {
        var result = RLStepBatch(spec: spec)
        try step(actions: actions, into: &result)
        try result.validate(for: spec)
        return result
    }

    func checkedEnvironmentIDs(_ ids: [Int]?) throws -> [Int] {
        let resolved = ids ?? Array(0..<spec.numEnvironments)
        for id in resolved where id < 0 || id >= spec.numEnvironments {
            throw RLEnvironmentError.invalidEnvironmentIndex(id)
        }
        return resolved
    }
}

// MARK: - Task registration

public struct RLTaskConfiguration: Sendable {
    public var numEnvironments: Int
    public var seed: UInt64
    /// Whether terminal environments reset inside `step`. Training normally
    /// enables this for throughput. Evaluation disables it so the evaluator
    /// can assign a fixed, reproducible episode quota to every environment.
    public var autoReset: Bool
    /// Add scene-owned, policy-invisible bodies used only for interactive
    /// robustness probes such as the Policy Replay "Throw Box" control.
    /// This is deliberately not an experiment option or part of checkpoint
    /// compatibility: training/evaluation batches leave it disabled.
    public var includeInteractiveRobustnessProbes: Bool
    public var options: [String: Float]

    public init(numEnvironments: Int, seed: UInt64 = 1,
                autoReset: Bool = true,
                includeInteractiveRobustnessProbes: Bool = false,
                options: [String: Float] = [:]) {
        self.numEnvironments = numEnvironments
        self.seed = seed
        self.autoReset = autoReset
        self.includeInteractiveRobustnessProbes =
            includeInteractiveRobustnessProbes
        self.options = options
    }

    /// Fail closed when a registered task receives a misspelled or obsolete
    /// option. Experiment configuration is part of the scientific contract:
    /// silently falling back to a default can make a run look valid while it
    /// is actually testing a different task.
    public func validateOptions(
        supported: Set<String>, taskID: String
    ) throws {
        let unknown = Set(options.keys).subtracting(supported).sorted()
        guard unknown.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "task \(taskID) does not support option(s): "
                + unknown.joined(separator: ", ")
                + "; supported: "
                + supported.sorted().joined(separator: ", "))
        }
    }
}

/// Concurrent-safe registry of vectorized task factories.
///
/// Registration and lookup are linearizable. Factories are snapshotted while
/// locked and executed after unlocking, so they may safely reenter the same
/// registry. A factory can be invoked concurrently and therefore must be
/// sendable.
public final class RLTaskRegistry: @unchecked Sendable {
    public typealias Factory = @Sendable (RLTaskConfiguration) throws
        -> any VectorizedRLTask

    private let lock = NSLock()
    private var factories: [String: Factory] = [:]
    private var optionSchemas: [String: RLTaskOptionSchema] = [:]

    public init() {}

    public func register(
        _ id: String,
        optionSchema: RLTaskOptionSchema? = nil,
        factory: @escaping Factory
    ) throws {
        try Self.validateIdentifier(id)
        lock.lock()
        defer { lock.unlock() }
        guard factories[id] == nil else { throw RLEnvironmentError.duplicateTask(id) }
        factories[id] = factory
        optionSchemas[id] = optionSchema
    }

    public var taskIDs: [String] {
        lock.lock()
        defer { lock.unlock() }
        return factories.keys.sorted()
    }

    /// Returns the task's machine-readable option contract, if its factory
    /// opted into central validation.
    public func optionSchema(for id: String) -> RLTaskOptionSchema? {
        lock.lock()
        defer { lock.unlock() }
        return optionSchemas[id]
    }

    /// Reconstruct task options from checkpoint metadata without injecting a
    /// structural field into tasks that intentionally hardcode it. Exact
    /// compatibility is still checked after the task has been constructed.
    public func checkpointReplayOptions(
        for id: String,
        semanticOptions: [String: Float],
        maxEpisodeSteps: Int,
        controlDecimation: Int
    ) -> [String: Float] {
        var options = semanticOptions
        let definitions = optionSchema(for: id)?.definitions ?? [:]
        if definitions["maxEpisodeSteps"] != nil {
            options["maxEpisodeSteps"] = Float(maxEpisodeSteps)
        }
        if definitions["controlDecimation"] != nil {
            options["controlDecimation"] = Float(controlDecimation)
        }
        return options
    }

    public func make(_ id: String, configuration: RLTaskConfiguration) throws
        -> any VectorizedRLTask {
        try Self.validateIdentifier(id)
        guard configuration.numEnvironments > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "numEnvironments must be positive")
        }
        lock.lock()
        let factory = factories[id]
        let optionSchema = optionSchemas[id]
        let available = factories.keys.sorted()
        lock.unlock()
        guard let factory else {
            throw RLEnvironmentError.unknownTask(id, available: available)
        }
        try optionSchema?.validate(configuration, taskID: id)
        let task = try factory(configuration)
        guard task.spec.id == id else {
            throw RLEnvironmentError.invalidConfiguration(
                "task factory registered as '\(id)' produced task "
                + "'\(task.spec.id)'")
        }
        guard task.spec.numEnvironments == configuration.numEnvironments else {
            throw RLEnvironmentError.invalidConfiguration(
                "task '\(id)' factory produced "
                + "\(task.spec.numEnvironments) environments; requested "
                + "\(configuration.numEnvironments)")
        }
        return task
    }

    private static func validateIdentifier(_ id: String) throws {
        guard !id.isEmpty,
              id.unicodeScalars.allSatisfy({
                  !CharacterSet.whitespacesAndNewlines.contains($0)
                      && !CharacterSet.controlCharacters.contains($0)
              }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid RL task identifier '\(id)': identifiers must be "
                + "non-empty and contain no whitespace or control characters")
        }
    }
}
