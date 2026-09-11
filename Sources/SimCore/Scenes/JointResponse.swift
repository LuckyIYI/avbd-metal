import Foundation

/// Passive generalized effort versus joint coordinate, summed with the joint's other constraints.
/// Radians/N m for hinges; metres/N for slides. Positive effort increases the coordinate.
public struct JointResponse: Codable, Sendable, Equatable {
    public var knots: [[Float]]
    public var damping: Float
    public var max_effort: Float
    public init(knots: [[Float]], damping: Float = 0, maxEffort: Float) {
        self.knots=knots;self.damping=damping;self.max_effort=maxEffort
    }
    public func validate() throws {
        guard (2...16).contains(knots.count), damping.isFinite, damping>=0,
              max_effort.isFinite,max_effort>0,
              knots.allSatisfy({$0.count==2 && $0.allSatisfy(\.isFinite)}),
              zip(knots,knots.dropFirst()).allSatisfy({$0[0]<$1[0]}) else {
            throw JointResponseError.invalidCurve
        }
    }
    /// Endpoint effort is held outside the knot range. The cap includes damping.
    public func effort(position: Float, velocity: Float) -> Float {
        var force=knots[0][1]
        if position >= knots.last![0] { force=knots.last![1] }
        else if position > knots[0][0] {
            for i in 1..<knots.count where position<=knots[i][0] {
                let a=knots[i-1],b=knots[i]
                force=a[1]+(b[1]-a[1])*(position-a[0])/(b[0]-a[0]);break
            }
        }
        return min(max_effort,max(-max_effort,force-damping*velocity))
    }
}
public enum JointResponseError: Error { case invalidCurve }

/// Constraint reaction thresholds; nil leaves that channel unbreakable.
/// Force is N; torque is N m about the joint anchor, excluding lever-arm torque
/// from its linear reaction and excluding passive responses/motor effort.
public struct JointBreakLoad: Codable, Sendable, Equatable {
    public var force: Float?
    public var torque: Float?
    public init(force: Float? = nil, torque: Float? = nil) { self.force=force;self.torque=torque }
    public func validate() throws {
        let values=[force,torque].compactMap{$0}
        guard !values.isEmpty,values.allSatisfy({$0.isFinite && $0>0}) else { throw JointResponseError.invalidCurve }
    }
}
