import Foundation

/// Validation failures in robot descriptions, hardware calibration, and
/// controller I/O. These errors deliberately do not depend on an RL runtime.
public enum RobotContractError: Error, LocalizedError, CustomStringConvertible, Equatable,
    Sendable {
    case invalidConfiguration(String)
    case invalidValueCount(label: String, expected: Int, actual: Int)
    case nonFiniteValue(label: String, index: Int)

    public var description: String {
        switch self {
        case .invalidConfiguration(let message):
            return message
        case .invalidValueCount(let label, let expected, let actual):
            return "\(label) expected \(expected) values, received \(actual)"
        case .nonFiniteValue(let label, let index):
            return "\(label) contains a non-finite value at index \(index)"
        }
    }

    public var errorDescription: String? { description }
}
