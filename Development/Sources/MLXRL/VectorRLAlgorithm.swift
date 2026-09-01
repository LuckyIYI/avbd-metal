import Foundation
import SimCore
import PhysicsAVBD
import Robotics
import RL

/// How a training invocation initializes learner and environment state.
public enum VectorRLTrainingContinuation: String, Codable, Sendable {
    /// Start a new learner without reading checkpoint state from the output
    /// directory.
    case fresh

    /// Restore the latest complete learner checkpoint in the output directory.
    ///
    /// This is an update-boundary *warm* continuation, not serialized world
    /// state. Policy, optimizer, scheduler, normalizer, and progress may be
    /// restored, but simulator and task state reset from their configured seed
    /// before another rollout is collected.
    case warmResumeLatest = "warm-resume-latest"
}

/// One algorithm-neutral training invocation.
public struct VectorRLTrainingRequest {
    public let task: any VectorizedRLTask
    public let outputDirectory: String
    public let continuation: VectorRLTrainingContinuation

    public init(
        task: any VectorizedRLTask,
        outputDirectory: String,
        continuation: VectorRLTrainingContinuation = .fresh
    ) {
        self.task = task
        self.outputDirectory = outputDirectory
        self.continuation = continuation
    }
}

/// Stable lifecycle information returned after training finishes.
public struct VectorRLTrainingResult: Equatable, Sendable {
    public let algorithmID: String
    public let outputDirectory: String
    public let continuation: VectorRLTrainingContinuation

    public init(
        algorithmID: String,
        outputDirectory: String,
        continuation: VectorRLTrainingContinuation
    ) {
        self.algorithmID = algorithmID
        self.outputDirectory = outputDirectory
        self.continuation = continuation
    }
}

/// Errors at the algorithm-neutral training lifecycle boundary.
public enum VectorRLTrainingError: Error, Equatable, Sendable,
    CustomStringConvertible, LocalizedError
{
    case unsupportedContinuation(
        algorithmID: String, continuation: VectorRLTrainingContinuation)

    public var description: String {
        switch self {
        case let .unsupportedContinuation(algorithmID, continuation):
            return "RL algorithm '\(algorithmID)' does not support "
                + "training continuation '\(continuation.rawValue)'"
        }
    }

    public var errorDescription: String? { description }
}

/// A trainable algorithm that consumes any vectorized RL task.
///
/// This original protocol remains unchanged so existing conformers and witness
/// tables retain their compatibility.
public protocol VectorRLAlgorithm: AnyObject {
    var id: String { get }
    func train(task: any VectorizedRLTask, outputDirectory: String) throws
}

/// Optional capability for algorithms that support explicit continuation.
public protocol VectorRLTrainingLifecycle: VectorRLAlgorithm {
    func trainLifecycle(_ request: VectorRLTrainingRequest) throws
        -> VectorRLTrainingResult
}

public extension VectorRLAlgorithm {
    /// Executes a lifecycle request through the concrete algorithm's optional
    /// capability. Legacy algorithms support fresh training and reject warm
    /// resume rather than silently restarting from scratch.
    func train(_ request: VectorRLTrainingRequest) throws
        -> VectorRLTrainingResult
    {
        if let lifecycle = self as? any VectorRLTrainingLifecycle {
            return try lifecycle.trainLifecycle(request)
        }
        guard request.continuation == .fresh else {
            throw VectorRLTrainingError.unsupportedContinuation(
                algorithmID: id, continuation: request.continuation)
        }
        try train(
            task: request.task,
            outputDirectory: request.outputDirectory)
        return VectorRLTrainingResult(
            algorithmID: id,
            outputDirectory: request.outputDirectory,
            continuation: request.continuation)
    }
}

extension VectorPPOTrainer: VectorRLAlgorithm, VectorRLTrainingLifecycle {
    public var id: String { "ppo" }

    public func trainLifecycle(_ request: VectorRLTrainingRequest) throws
        -> VectorRLTrainingResult
    {
        let resume: Bool
        switch request.continuation {
        case .fresh: resume = false
        case .warmResumeLatest: resume = true
        }
        try train(
            task: request.task,
            outputDirectory: request.outputDirectory,
            resume: resume)
        return VectorRLTrainingResult(
            algorithmID: id,
            outputDirectory: request.outputDirectory,
            continuation: request.continuation)
    }
}

/// A typed algorithm definition with a fresh default configuration and trainer
/// factory. Runtime type erasure is private to the registry.
public struct VectorRLAlgorithmDefinition<Configuration>: Sendable
where Configuration: Codable & Sendable {
    public typealias DefaultConfigurationFactory =
        @Sendable () -> Configuration
    public typealias TrainerFactory =
        @Sendable (Configuration) throws -> any VectorRLAlgorithm

    public let id: String
    private let defaultConfigurationFactory: DefaultConfigurationFactory
    private let trainerFactory: TrainerFactory

    public init(
        id: String,
        defaultConfiguration: @escaping DefaultConfigurationFactory,
        trainerFactory: @escaping TrainerFactory
    ) {
        self.id = id
        defaultConfigurationFactory = defaultConfiguration
        self.trainerFactory = trainerFactory
    }

    public func makeDefaultConfiguration() -> Configuration {
        defaultConfigurationFactory()
    }

    public func makeTrainer(
        configuration: Configuration
    ) throws -> any VectorRLAlgorithm {
        try trainerFactory(configuration)
    }
}

/// Errors produced by ``VectorRLAlgorithmRegistry``.
public enum VectorRLAlgorithmRegistryError: Error, Equatable, Sendable,
    CustomStringConvertible, LocalizedError
{
    case invalidIdentifier(String)
    case duplicateIdentifier(String)
    case unknownIdentifier(String, available: [String])
    case factoryIdentifierMismatch(registered: String, produced: String)

    public var description: String {
        switch self {
        case let .invalidIdentifier(id):
            return "invalid RL algorithm identifier '\(id)': identifiers must "
                + "be non-empty and contain no whitespace or control characters"
        case let .duplicateIdentifier(id):
            return "RL algorithm '\(id)' is already registered"
        case let .unknownIdentifier(id, available):
            return "unknown RL algorithm '\(id)'; available: "
                + available.joined(separator: ", ")
        case let .factoryIdentifierMismatch(registered, produced):
            return "RL algorithm factory registered as '\(registered)' produced "
                + "algorithm '\(produced)'"
        }
    }

    public var errorDescription: String? { description }
}

/// Typed failures from configured algorithm construction. This is separate
/// from the original registry error enum so downstream exhaustive switches on
/// that public enum remain source-compatible.
public enum VectorRLAlgorithmConfigurationError: Error, Equatable, Sendable,
    CustomStringConvertible, LocalizedError
{
    case typeMismatch(algorithm: String, expected: String, provided: String)
    case unsupported(String)

    public var description: String {
        switch self {
        case let .typeMismatch(algorithm, expected, provided):
            return "RL algorithm '\(algorithm)' expects configuration "
                + "'\(expected)', not '\(provided)'"
        case let .unsupported(id):
            return "RL algorithm '\(id)' was registered with a fixed factory "
                + "and does not accept configured construction"
        }
    }

    public var errorDescription: String? { description }
}

/// A concurrent-safe registry of typed vectorized RL algorithm definitions.
///
/// Entries are copied while locked. Default/configured factories run only
/// after unlocking, so they may safely inspect the registry and can execute
/// concurrently. All public configuration surfaces remain statically typed;
/// heterogeneous erasure exists only inside this implementation.
public final class VectorRLAlgorithmRegistry: @unchecked Sendable {
    /// Backward-compatible factory for fixed-configuration algorithms.
    public typealias Factory = @Sendable () -> any VectorRLAlgorithm

    private struct Entry: Sendable {
        let configurationTypeID: ObjectIdentifier?
        let configurationTypeName: String?
        let makeDefault: @Sendable () throws -> any VectorRLAlgorithm
        let makeConfigured:
            (@Sendable (any Sendable) throws -> any VectorRLAlgorithm)?
    }

    private let lock = NSLock()
    private var entries: [String: Entry] = [:]

    public init() {}

    /// Registers a typed definition.
    public func register<Configuration>(
        _ definition: VectorRLAlgorithmDefinition<Configuration>
    ) throws where Configuration: Codable & Sendable {
        try Self.validateIdentifier(definition.id)
        let entry = Entry(
            configurationTypeID: ObjectIdentifier(Configuration.self),
            configurationTypeName: String(reflecting: Configuration.self),
            makeDefault: {
                try definition.makeTrainer(
                    configuration: definition.makeDefaultConfiguration())
            },
            makeConfigured: { erased in
                guard let configuration = erased as? Configuration else {
                    throw VectorRLAlgorithmConfigurationError.typeMismatch(
                        algorithm: definition.id,
                        expected: String(reflecting: Configuration.self),
                        provided: String(reflecting: type(of: erased)))
                }
                return try definition.makeTrainer(configuration: configuration)
            })
        try insert(entry, id: definition.id)
    }

    /// Registers a fixed factory through the original source-compatible API.
    public func register(_ id: String, factory: @escaping Factory) throws {
        try Self.validateIdentifier(id)
        try insert(Entry(
            configurationTypeID: nil,
            configurationTypeName: nil,
            makeDefault: factory,
            makeConfigured: nil), id: id)
    }

    public var algorithmIDs: [String] {
        lock.lock()
        defer { lock.unlock() }
        return entries.keys.sorted()
    }

    /// Creates the algorithm with its registered default configuration.
    public func make(_ id: String) throws -> any VectorRLAlgorithm {
        let registered = try entry(for: id)
        return try validate(registered.makeDefault(), registeredAs: id)
    }

    /// Creates the algorithm with an exact nominal configuration type.
    public func make<Configuration>(
        _ id: String,
        configuration: Configuration
    ) throws -> any VectorRLAlgorithm where Configuration: Sendable {
        let registered = try entry(for: id)
        guard let expectedID = registered.configurationTypeID,
              let expectedName = registered.configurationTypeName,
              let factory = registered.makeConfigured else {
            throw VectorRLAlgorithmConfigurationError.unsupported(id)
        }
        let providedID = ObjectIdentifier(Configuration.self)
        guard providedID == expectedID else {
            throw VectorRLAlgorithmConfigurationError.typeMismatch(
                algorithm: id,
                expected: expectedName,
                provided: String(reflecting: Configuration.self))
        }
        return try validate(
            factory(configuration), registeredAs: id)
    }

    /// Process-wide registry containing algorithms shipped by MLXRL.
    public static let builtIn: VectorRLAlgorithmRegistry = {
        let registry = VectorRLAlgorithmRegistry()
        let ppo = VectorRLAlgorithmDefinition(
            id: "ppo",
            defaultConfiguration: { VectorPPOConfig() },
            trainerFactory: { VectorPPOTrainer(configuration: $0) })
        try! registry.register(ppo)
        return registry
    }()

    private func validate(
        _ algorithm: any VectorRLAlgorithm,
        registeredAs id: String
    ) throws -> any VectorRLAlgorithm {
        guard algorithm.id == id else {
            throw VectorRLAlgorithmRegistryError.factoryIdentifierMismatch(
                registered: id, produced: algorithm.id)
        }
        return algorithm
    }

    private func insert(_ entry: Entry, id: String) throws {
        lock.lock()
        defer { lock.unlock() }
        guard entries[id] == nil else {
            throw VectorRLAlgorithmRegistryError.duplicateIdentifier(id)
        }
        entries[id] = entry
    }

    private func entry(for id: String) throws -> Entry {
        try Self.validateIdentifier(id)
        lock.lock()
        defer { lock.unlock() }
        guard let entry = entries[id] else {
            throw VectorRLAlgorithmRegistryError.unknownIdentifier(
                id, available: entries.keys.sorted())
        }
        return entry
    }

    private static func validateIdentifier(_ id: String) throws {
        guard !id.isEmpty,
              id.unicodeScalars.allSatisfy({
                  !CharacterSet.whitespacesAndNewlines.contains($0)
                      && !CharacterSet.controlCharacters.contains($0)
              }) else {
            throw VectorRLAlgorithmRegistryError.invalidIdentifier(id)
        }
    }
}
