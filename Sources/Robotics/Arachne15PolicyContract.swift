import simd
import SimCore

/// The measured state and command consumed by the deployed Arachne policy.
/// Velocities, gravity, and command are expressed in the robot body frame;
/// joint arrays use `Arachne15PolicyContract.jointNames` order.
public struct Arachne15PolicyInput: Sendable, Equatable {
    public var bodyLinearVelocity: F3
    public var bodyAngularVelocity: F3
    public var projectedGravity: F3
    public var commandedBodyTwist: F3
    public var jointPositions: [Float]
    public var jointVelocities: [Float]
    public var previousActions: [Float]

    public init(bodyLinearVelocity: F3, bodyAngularVelocity: F3,
                projectedGravity: F3, commandedBodyTwist: F3,
                jointPositions: [Float], jointVelocities: [Float],
                previousActions: [Float]) {
        self.bodyLinearVelocity = bodyLinearVelocity
        self.bodyAngularVelocity = bodyAngularVelocity
        self.projectedGravity = projectedGravity
        self.commandedBodyTwist = commandedBodyTwist
        self.jointPositions = jointPositions
        self.jointVelocities = jointVelocities
        self.previousActions = previousActions
    }
}

/// Authoritative policy I/O schema shared by simulation and the iPhone
/// controller. Keeping encoding and actuator scaling here prevents a hardware
/// app from reimplementing the 60-channel tensor with a silent ordering or
/// coordinate-system mismatch.
public enum Arachne15PolicyContract {
    public static let observationDimension = 60
    public static let actionDimension = 16
    public static let controlFrequencyHz: Float = 50

    /// MJCF actuator order, alternating hip and knee. Robot forward is +X and
    /// left is +Y. Hardware commissioning must map servo IDs to this list.
    public static let jointNames = [
        "right_rear_hip", "right_rear_knee",
        "right_mid_rear_hip", "right_mid_rear_knee",
        "right_mid_front_hip", "right_mid_front_knee",
        "right_front_hip", "right_front_knee",
        "left_rear_hip", "left_rear_knee",
        "left_mid_rear_hip", "left_mid_rear_knee",
        "left_mid_front_hip", "left_mid_front_knee",
        "left_front_hip", "left_front_knee",
    ]

    /// Normalized actions are relative joint-position commands in radians.
    /// The mechanical assembly zero is encoded in each link's fixed transform.
    public static let actionScales: [Float] = (0..<actionDimension).map {
        $0.isMultiple(of: 2) ? 0.35 : 0.45
    }

    public static func encode(_ input: Arachne15PolicyInput) throws
        -> ContiguousArray<Float> {
        for values in [input.jointPositions, input.jointVelocities,
                       input.previousActions] where values.count != actionDimension {
            throw RobotContractError.invalidConfiguration(
                "Arachne policy joint/action arrays must contain "
                    + "\(actionDimension) values")
        }
        var output = ContiguousArray<Float>()
        output.reserveCapacity(observationDimension)
        output.append(contentsOf: [
            input.bodyLinearVelocity.x, input.bodyLinearVelocity.y,
            input.bodyLinearVelocity.z,
            input.bodyAngularVelocity.x, input.bodyAngularVelocity.y,
            input.bodyAngularVelocity.z,
            input.projectedGravity.x, input.projectedGravity.y,
            input.projectedGravity.z,
            input.commandedBodyTwist.x, input.commandedBodyTwist.y,
            input.commandedBodyTwist.z,
        ])
        output.append(contentsOf: input.jointPositions)
        output.append(contentsOf: input.jointVelocities)
        output.append(contentsOf: input.previousActions)
        guard output.count == observationDimension,
              output.allSatisfy(\.isFinite) else {
            throw RobotContractError.invalidConfiguration(
                "Arachne policy observation contains a non-finite value")
        }
        let gravityNorm = simd_length(input.projectedGravity)
        guard gravityNorm >= 0.8, gravityNorm <= 1.2 else {
            throw RobotContractError.invalidConfiguration(
                "Arachne projected gravity must be a body-frame unit vector")
        }
        return output
    }

    /// Convert a world-space goal to the exact body-twist command used during
    /// training. A future vision/gesture estimator supplies the world goal;
    /// it must not bypass this policy command contract.
    public static func pointGoalCommand(
        worldGoal: F3, rootPosition: F3, rootRotation: Quat,
        goalRadius: Float, slowdownDistance: Float,
        cruiseSpeed: Float, boundarySpeed: Float,
        maximumYawRate: Float
    ) -> F3 {
        precondition(goalRadius > 0 && slowdownDistance > goalRadius)
        precondition(cruiseSpeed >= boundarySpeed && boundarySpeed >= 0)
        precondition(maximumYawRate >= 0)
        return PointGoalNavigator.command(
            worldGoal: worldGoal,
            bodyPosition: rootPosition,
            bodyRotation: rootRotation,
            parameters: PointGoalNavigationParameters(
                goalRadius: goalRadius,
                slowdownDistance: slowdownDistance,
                cruiseSpeed: cruiseSpeed,
                boundarySpeed: boundarySpeed,
                yawGain: 2,
                maximumYawRate: maximumYawRate,
                mode: .projectedBodyPlane)).bodyTwist
    }

    /// Convert policy actions to relative joint targets, matching simulation.
    /// Hardware-specific encoder zeros and direction signs are applied only
    /// after this function and must be recorded in the calibration artifact.
    public static func relativeJointTargets(
        for actions: ContiguousArray<Float>
    ) throws -> [Float] {
        guard actions.count == actionDimension else {
            throw RobotContractError.invalidValueCount(
                label: "Arachne policy action",
                expected: actionDimension,
                actual: actions.count)
        }
        if let index = actions.firstIndex(where: { !$0.isFinite }) {
            throw RobotContractError.nonFiniteValue(
                label: "Arachne policy action", index: index)
        }
        return actions.indices.map {
            simd_clamp(actions[$0], -1, 1) * actionScales[$0]
        }
    }
}
