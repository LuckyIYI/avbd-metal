import simd

/// Coordinate and command conventions for point-goal navigation.
///
/// The two modes preserve the command contracts used by the native Arachne
/// policy and the imported Isaac H1 policy. They deliberately share goal
/// geometry, arrival-speed shaping, and bounded yaw control without pretending
/// that a forward-only policy accepts a holonomic velocity command.
public enum PointGoalCommandMode: Sendable, Equatable {
    /// Project a world-horizontal goal vector through the complete measured
    /// body attitude and command velocity along that body-plane direction.
    case projectedBodyPlane
    /// Measure distance in the world horizontal plane, command forward speed,
    /// and steer toward the goal using the measured horizontal body heading.
    case forwardOnlyYaw
}

/// Parameters that completely define the reusable point-goal command law.
public struct PointGoalNavigationParameters: Sendable, Equatable {
    public let goalRadius: Float
    public let slowdownDistance: Float
    public let cruiseSpeed: Float
    public let boundarySpeed: Float
    public let yawGain: Float
    public let maximumYawRate: Float
    public let mode: PointGoalCommandMode

    public init(
        goalRadius: Float, slowdownDistance: Float,
        cruiseSpeed: Float, boundarySpeed: Float = 0,
        yawGain: Float, maximumYawRate: Float,
        mode: PointGoalCommandMode
    ) {
        precondition(goalRadius > 0 && slowdownDistance > goalRadius)
        precondition(cruiseSpeed >= boundarySpeed && boundarySpeed >= 0)
        precondition(yawGain >= 0 && maximumYawRate >= 0)
        self.goalRadius = goalRadius
        self.slowdownDistance = slowdownDistance
        self.cruiseSpeed = cruiseSpeed
        self.boundarySpeed = boundarySpeed
        self.yawGain = yawGain
        self.maximumYawRate = maximumYawRate
        self.mode = mode
    }
}

/// Fully measured result of one point-goal command update.
public struct PointGoalNavigationCommand: Sendable, Equatable {
    /// Contract revision for logs and external controller integrations.
    public let revision: Int
    public let bodyTwist: F3
    public let bodyPlanarDelta: F3
    public let worldPlanarDirection: F3
    public let remainingDistance: Float
    public let commandedSpeed: Float
    public let relativeBearing: Float
    public let proximity: Float
    public let reachedGoal: Bool
    public let desiredWorldHeading: Float
}

/// Revisioned point-goal navigation contract shared by simulation and
/// deployment. It computes only a task-space command; joint actions remain the
/// responsibility of the selected learned or classical controller.
public enum PointGoalNavigator {
    public static let revision = 1

    public static func command(
        worldGoal: F3, bodyPosition: F3, bodyRotation: Quat,
        parameters: PointGoalNavigationParameters
    ) -> PointGoalNavigationCommand {
        let worldDelta3 = worldGoal - bodyPosition
        let worldDelta = F3(worldDelta3.x, worldDelta3.y, 0)
        let worldDistance = sqrt(
            worldDelta.x * worldDelta.x + worldDelta.y * worldDelta.y)
        let worldDirection = worldDistance > 0
            ? worldDelta / max(worldDistance, 1e-6) : .zero
        let forward = bodyRotation.act(F3(1, 0, 0))
        let bodyHeading = atan2(forward.y, forward.x)

        let bodyDelta: F3
        let distance: Float
        let bearing: Float
        let desiredWorldHeading: Float
        switch parameters.mode {
        case .projectedBodyPlane:
            let projected = bodyRotation.conjugate.act(worldDelta)
            bodyDelta = F3(projected.x, projected.y, 0)
            distance = sqrt(
                bodyDelta.x * bodyDelta.x + bodyDelta.y * bodyDelta.y)
            bearing = atan2(bodyDelta.y, bodyDelta.x)
            desiredWorldHeading = worldDistance > 0
                ? atan2(worldDelta.y, worldDelta.x) : bodyHeading
        case .forwardOnlyYaw:
            distance = worldDistance
            desiredWorldHeading = distance > parameters.goalRadius
                ? atan2(worldDelta.y, worldDelta.x) : bodyHeading
            bearing = wrappedAngle(desiredWorldHeading - bodyHeading)
            bodyDelta = F3(cos(bearing) * distance,
                           sin(bearing) * distance, 0)
        }

        let reached = distance <= parameters.goalRadius
        let speed = commandSpeed(
            remainingDistance: distance,
            cruiseSpeed: parameters.cruiseSpeed,
            goalRadius: parameters.goalRadius,
            slowdownDistance: parameters.slowdownDistance,
            boundarySpeed: parameters.boundarySpeed)
        let yawRate = reached ? 0 : boundedYawRate(
            bearing: bearing, gain: parameters.yawGain,
            maximumRate: parameters.maximumYawRate)
        let velocity: F3
        switch parameters.mode {
        case .projectedBodyPlane:
            let direction = distance > parameters.goalRadius
                ? bodyDelta / max(distance, 1e-6) : .zero
            velocity = direction * speed
        case .forwardOnlyYaw:
            velocity = F3(speed, 0, 0)
        }

        return PointGoalNavigationCommand(
            revision: revision,
            bodyTwist: F3(velocity.x, velocity.y, yawRate),
            bodyPlanarDelta: bodyDelta,
            worldPlanarDirection: worldDirection,
            remainingDistance: distance,
            commandedSpeed: speed,
            relativeBearing: bearing,
            proximity: proximity(
                remainingDistance: distance,
                goalRadius: parameters.goalRadius,
                slowdownDistance: parameters.slowdownDistance),
            reachedGoal: reached,
            desiredWorldHeading: desiredWorldHeading)
    }

    public static func commandSpeed(
        remainingDistance: Float, cruiseSpeed: Float,
        goalRadius: Float, slowdownDistance: Float,
        boundarySpeed: Float = 0
    ) -> Float {
        precondition(remainingDistance >= 0 && cruiseSpeed >= 0)
        precondition(goalRadius > 0 && slowdownDistance > goalRadius)
        precondition(boundarySpeed >= 0)
        guard remainingDistance > goalRadius else { return 0 }
        let fraction = simd_clamp(
            (remainingDistance - goalRadius)
                / (slowdownDistance - goalRadius), 0, 1)
        let admissibleBoundary = min(boundarySpeed, cruiseSpeed)
        return admissibleBoundary
            + (cruiseSpeed - admissibleBoundary) * fraction
    }

    public static func proximity(
        remainingDistance: Float, goalRadius: Float,
        slowdownDistance: Float
    ) -> Float {
        precondition(remainingDistance >= 0)
        precondition(goalRadius > 0 && slowdownDistance > goalRadius)
        return 1 - simd_clamp(
            (remainingDistance - goalRadius)
                / (slowdownDistance - goalRadius), 0, 1)
    }

    public static func boundedYawRate(
        bearing: Float, gain: Float, maximumRate: Float
    ) -> Float {
        precondition(gain >= 0 && maximumRate >= 0)
        return simd_clamp(gain * bearing, -maximumRate, maximumRate)
    }

    /// Maps an angle into [-pi, pi), matching the existing H1 command law.
    public static func wrappedAngle(_ angle: Float) -> Float {
        angle - 2 * .pi * floor((angle + .pi) / (2 * .pi))
    }
}
