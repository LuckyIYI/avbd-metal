import AVBDCore

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
