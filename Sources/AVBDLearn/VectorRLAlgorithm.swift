import Foundation
import AVBDCore

/// A trainable algorithm that consumes any vectorized RL task.
///
/// The protocol is deliberately task-agnostic: adding SAC, Dreamer, AMP, or
/// distillation does not require changes to scene code.
public protocol VectorRLAlgorithm: AnyObject {
    /// Stable identifier used by experiment configuration and the CLI.
    var id: String { get }

    /// Trains on `task` and writes checkpoints and metrics below
    /// `outputDirectory`.
    func train(task: any VectorizedRLTask, outputDirectory: String) throws
}

extension VectorPPOTrainer: VectorRLAlgorithm {
    public var id: String { "ppo" }
}

/// Errors produced by ``VectorRLAlgorithmRegistry``.
public enum VectorRLAlgorithmRegistryError: Error, Equatable, Sendable,
    CustomStringConvertible, LocalizedError
{
    /// The identifier is empty or contains whitespace or control characters.
    case invalidIdentifier(String)
    /// An algorithm is already registered under the identifier.
    case duplicateIdentifier(String)
    /// No algorithm is registered under the identifier.
    case unknownIdentifier(String, available: [String])
    /// A factory returned an algorithm whose identifier differs from its key.
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

/// A concurrent-safe registry of vectorized RL algorithm factories.
///
/// Registration and lookup are linearizable. Factories are copied while the
/// registry is locked and invoked after unlocking, so a factory may safely
/// inspect or extend the same registry. A factory can be invoked concurrently
/// by multiple callers, so its captured state must satisfy Swift's sendable
/// closure contract.
public final class VectorRLAlgorithmRegistry: @unchecked Sendable {
    /// A closure that creates an independent algorithm instance.
    public typealias Factory = @Sendable () -> any VectorRLAlgorithm

    private let lock = NSLock()
    private var factories: [String: Factory] = [:]

    /// Creates an empty registry.
    public init() {}

    /// Registers `factory` under a stable identifier.
    ///
    /// - Throws: ``VectorRLAlgorithmRegistryError/invalidIdentifier(_:)`` when
    ///   `id` is not suitable for configuration, or
    ///   ``VectorRLAlgorithmRegistryError/duplicateIdentifier(_:)`` when the
    ///   identifier is already present.
    public func register(_ id: String, factory: @escaping Factory) throws {
        try Self.validateIdentifier(id)
        lock.lock()
        defer { lock.unlock() }
        guard factories[id] == nil else {
            throw VectorRLAlgorithmRegistryError.duplicateIdentifier(id)
        }
        factories[id] = factory
    }

    /// Registered identifiers in deterministic lexical order.
    public var algorithmIDs: [String] {
        lock.lock()
        defer { lock.unlock() }
        return factories.keys.sorted()
    }

    /// Creates the algorithm registered under `id`.
    ///
    /// The factory executes outside the registry lock. This prevents factory
    /// work from blocking unrelated registry reads and permits reentrant use.
    ///
    /// - Throws: A ``VectorRLAlgorithmRegistryError`` for an invalid or unknown
    ///   identifier, or when the factory returns an inconsistent identifier.
    public func make(_ id: String) throws -> any VectorRLAlgorithm {
        try Self.validateIdentifier(id)
        let (factory, available) = snapshot(for: id)
        guard let factory else {
            throw VectorRLAlgorithmRegistryError.unknownIdentifier(
                id, available: available)
        }
        let algorithm = factory()
        guard algorithm.id == id else {
            throw VectorRLAlgorithmRegistryError.factoryIdentifierMismatch(
                registered: id, produced: algorithm.id)
        }
        return algorithm
    }

    /// Process-wide registry containing algorithms shipped by AVBDLearn.
    public static let builtIn: VectorRLAlgorithmRegistry = {
        let registry = VectorRLAlgorithmRegistry()
        try! registry.register("ppo") { VectorPPOTrainer() }
        return registry
    }()

    private static func validateIdentifier(_ id: String) throws {
        guard !id.isEmpty,
              id.unicodeScalars.allSatisfy({
                  !CharacterSet.whitespacesAndNewlines.contains($0)
                      && !CharacterSet.controlCharacters.contains($0)
              })
        else {
            throw VectorRLAlgorithmRegistryError.invalidIdentifier(id)
        }
    }

    private func snapshot(for id: String) -> (Factory?, [String]) {
        lock.lock()
        defer { lock.unlock() }
        return (factories[id], factories.keys.sorted())
    }
}
