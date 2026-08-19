import Foundation

/// Result of an exact-physics proposal-mode probe. `provided` deliberately
/// covers learned, retrieved, and future generative proposal mechanisms.
public enum PhysicalFlowProposalSelection: String, Codable, Sendable {
    case notApplicable
    case provided
    case geometric
}

/// Task-agnostic result for a state-to-state physical-flow endpoint. Tasks own
/// the semantics and frozen thresholds of each component; the planner sees
/// only dimensionless normalized errors.
public struct PhysicalFlowEndpointEvaluation: Codable, Sendable {
    public var normalizedErrors: [Float]
    public var maximumNormalizedError: Float
    public var meanSquaredNormalizedError: Float
    public var bottleneckLoss: Float

    public var satisfiesEveryConstraint: Bool {
        maximumNormalizedError < 1
    }
}

public enum PhysicalFlowBalancedObjective {
    /// A bottleneck term directly targets the worst endpoint constraint while
    /// the small mean-square term prevents all non-worst components from being
    /// ignored. CEM/MPPI only require a stable ordering, not differentiability.
    public static func evaluate(
        normalizedErrors: [Float], meanSquareWeight: Float = 0.25
    ) -> PhysicalFlowEndpointEvaluation {
        precondition(!normalizedErrors.isEmpty)
        precondition(meanSquareWeight.isFinite && meanSquareWeight >= 0)
        precondition(normalizedErrors.allSatisfy { $0.isFinite && $0 >= 0 })
        let maximum = normalizedErrors.max()!
        let meanSquare = normalizedErrors.reduce(0) {
            $0 + $1 * $1
        } / Float(normalizedErrors.count)
        return PhysicalFlowEndpointEvaluation(
            normalizedErrors: normalizedErrors,
            maximumNormalizedError: maximum,
            meanSquaredNormalizedError: meanSquare,
            bottleneckLoss: maximum * maximum
                + meanSquareWeight * meanSquare)
    }
}
