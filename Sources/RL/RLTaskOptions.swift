import Foundation

/// The serialized scalar representation accepted by an RL task option.
///
/// Experiment files intentionally remain `[String: Float]` so they can cross
/// Swift, Python, MLX, and checkpoint metadata without a tagged-value bridge.
/// This schema restores the type information at the task boundary and rejects
/// configurations that would otherwise be silently truncated or coerced.
public enum RLTaskOptionValueKind: String, Codable, Equatable, Sendable {
    case number
    case integer
    case boolean
}

public struct RLTaskOptionDefinition: Codable, Equatable, Sendable {
    public let name: String
    public let valueKind: RLTaskOptionValueKind
    public let lowerBound: Float?
    public let upperBound: Float?

    public init(
        _ name: String,
        valueKind: RLTaskOptionValueKind = .number,
        lowerBound: Float? = nil,
        upperBound: Float? = nil
    ) {
        precondition(!name.isEmpty, "RL task option names cannot be empty")
        if let lowerBound {
            precondition(lowerBound.isFinite,
                         "RL task option lower bounds must be finite")
        }
        if let upperBound {
            precondition(upperBound.isFinite,
                         "RL task option upper bounds must be finite")
        }
        if let lowerBound, let upperBound {
            precondition(lowerBound <= upperBound,
                         "RL task option bounds must be ordered")
        }
        self.name = name
        self.valueKind = valueKind
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }

    fileprivate func validate(value: Float, taskID: String) throws {
        guard value.isFinite else {
            throw RLEnvironmentError.invalidConfiguration(
                "task \(taskID) option \(name) must be finite")
        }
        switch valueKind {
        case .number:
            break
        case .integer:
            guard value.rounded(.towardZero) == value else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task \(taskID) option \(name) must be an integer; "
                    + "got \(value)")
            }
            let wideValue = Double(value)
            guard wideValue >= Double(Int.min),
                  wideValue < Double(Int.max) else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task \(taskID) option \(name) must fit Swift Int; "
                    + "got \(value)")
            }
        case .boolean:
            guard value == 0 || value == 1 else {
                throw RLEnvironmentError.invalidConfiguration(
                    "task \(taskID) option \(name) must be 0 or 1; "
                    + "got \(value)")
            }
        }
        if let lowerBound, value < lowerBound {
            throw RLEnvironmentError.invalidConfiguration(
                "task \(taskID) option \(name) must be >= \(lowerBound); "
                + "got \(value)")
        }
        if let upperBound, value > upperBound {
            throw RLEnvironmentError.invalidConfiguration(
                "task \(taskID) option \(name) must be <= \(upperBound); "
                + "got \(value)")
        }
    }
}

/// Machine-readable option contract for a registered task.
public struct RLTaskOptionSchema: Codable, Equatable, Sendable {
    public let definitions: [String: RLTaskOptionDefinition]

    public init(_ definitions: [RLTaskOptionDefinition]) {
        var indexed = [String: RLTaskOptionDefinition]()
        indexed.reserveCapacity(definitions.count)
        for definition in definitions {
            precondition(indexed[definition.name] == nil,
                         "duplicate RL task option \(definition.name)")
            indexed[definition.name] = definition
        }
        self.definitions = indexed
    }

    public var optionNames: [String] { definitions.keys.sorted() }

    public func validate(
        _ configuration: RLTaskConfiguration,
        taskID: String
    ) throws {
        let unknown = Set(configuration.options.keys)
            .subtracting(definitions.keys)
            .sorted()
        guard unknown.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "task \(taskID) does not support option(s): "
                + unknown.joined(separator: ", ")
                + "; supported: "
                + optionNames.joined(separator: ", "))
        }
        for (name, value) in configuration.options {
            try definitions[name]!.validate(value: value, taskID: taskID)
        }
    }
}
