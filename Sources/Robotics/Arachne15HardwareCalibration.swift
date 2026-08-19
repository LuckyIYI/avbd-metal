import Foundation
import simd
import SimCore

/// Measured actuator mapping required before a policy can command physical
/// hardware. The external hardware integration owns its calibration template;
/// every runtime must refuse to arm until those values are measured and signed
/// off.
public struct Arachne15HardwareCalibration: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var robotSerial: String
    public var commissioned: Bool
    public var measuredAtUTC: String
    public var policyCheckpointFingerprint: String
    public var jointNames: [String]
    public var servoIDs: [Int]
    public var servoZeroRadians: [Float]
    public var servoDirectionSigns: [Float]
    public var jointLowerRadians: [Float]
    public var jointUpperRadians: [Float]
    public var currentLimitsMilliamps: [Int]
    public var maximumServoTemperatureCelsius: Float
    public var measuredMaximumRoundTripLatencySeconds: Float

    public init(schemaVersion: Int = 1, robotSerial: String,
                commissioned: Bool, measuredAtUTC: String,
                policyCheckpointFingerprint: String,
                jointNames: [String] = Arachne15PolicyContract.jointNames,
                servoIDs: [Int], servoZeroRadians: [Float],
                servoDirectionSigns: [Float],
                jointLowerRadians: [Float], jointUpperRadians: [Float],
                currentLimitsMilliamps: [Int],
                maximumServoTemperatureCelsius: Float,
                measuredMaximumRoundTripLatencySeconds: Float) {
        self.schemaVersion = schemaVersion
        self.robotSerial = robotSerial
        self.commissioned = commissioned
        self.measuredAtUTC = measuredAtUTC
        self.policyCheckpointFingerprint = policyCheckpointFingerprint
        self.jointNames = jointNames
        self.servoIDs = servoIDs
        self.servoZeroRadians = servoZeroRadians
        self.servoDirectionSigns = servoDirectionSigns
        self.jointLowerRadians = jointLowerRadians
        self.jointUpperRadians = jointUpperRadians
        self.currentLimitsMilliamps = currentLimitsMilliamps
        self.maximumServoTemperatureCelsius =
            maximumServoTemperatureCelsius
        self.measuredMaximumRoundTripLatencySeconds =
            measuredMaximumRoundTripLatencySeconds
    }

    public func validationFailures(
        expectedPolicyFingerprint: String
    ) -> [String] {
        var failures = [String]()
        if schemaVersion != 1 { failures.append("unsupported schemaVersion") }
        if !commissioned { failures.append("calibration is not commissioned") }
        if robotSerial.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            failures.append("robotSerial is empty")
        }
        if measuredAtUTC.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            failures.append("measuredAtUTC is empty")
        }
        if policyCheckpointFingerprint != expectedPolicyFingerprint {
            failures.append("policy checkpoint fingerprint mismatch")
        }
        if jointNames != Arachne15PolicyContract.jointNames {
            failures.append("jointNames do not match the policy contract")
        }
        let n = Arachne15PolicyContract.actionDimension
        let counts = [servoIDs.count, servoZeroRadians.count,
                      servoDirectionSigns.count, jointLowerRadians.count,
                      jointUpperRadians.count, currentLimitsMilliamps.count]
        if counts.contains(where: { $0 != n }) {
            failures.append("every actuator array must contain \(n) values")
            return failures
        }
        if Set(servoIDs).count != n
            || servoIDs.contains(where: { !(1...252).contains($0) }) {
            failures.append("servoIDs must be unique values in 1...252")
        }
        if !servoZeroRadians.allSatisfy(\.isFinite) {
            failures.append("servoZeroRadians contain a non-finite value")
        }
        if !servoDirectionSigns.allSatisfy({ $0 == -1 || $0 == 1 }) {
            failures.append("servoDirectionSigns must contain only -1 or +1")
        }
        for j in 0..<n where !jointLowerRadians[j].isFinite
            || !jointUpperRadians[j].isFinite
            || jointLowerRadians[j] >= 0 || jointUpperRadians[j] <= 0
            || jointLowerRadians[j] >= jointUpperRadians[j] {
            failures.append("joint limits are invalid at index \(j)")
        }
        if currentLimitsMilliamps.contains(where: { $0 <= 0 }) {
            failures.append("current limits must be measured positive values")
        }
        if !maximumServoTemperatureCelsius.isFinite
            || !(30...80).contains(maximumServoTemperatureCelsius) {
            failures.append("maximum servo temperature is outside 30...80 C")
        }
        if !measuredMaximumRoundTripLatencySeconds.isFinite
            || measuredMaximumRoundTripLatencySeconds <= 0
            || measuredMaximumRoundTripLatencySeconds > 0.040 {
            failures.append(
                "measured round-trip latency exceeds the trained 40 ms envelope")
        }
        return failures
    }

    public func validate(expectedPolicyFingerprint: String) throws {
        let failures = validationFailures(
            expectedPolicyFingerprint: expectedPolicyFingerprint)
        guard failures.isEmpty else {
            throw RobotContractError.invalidConfiguration(
                "Arachne hardware calibration is not deployable: "
                    + failures.joined(separator: "; "))
        }
    }

    /// Policy action to calibrated servo shaft angle. The bridge still owns
    /// rate, current, thermal, range, heartbeat, and torque-off enforcement.
    public func servoPositionRadians(
        for actions: ContiguousArray<Float>,
        expectedPolicyFingerprint: String
    ) throws -> [Float] {
        try validate(expectedPolicyFingerprint: expectedPolicyFingerprint)
        let relative = try Arachne15PolicyContract.relativeJointTargets(
            for: actions)
        return relative.indices.map { j in
            let joint = simd_clamp(relative[j], jointLowerRadians[j],
                                   jointUpperRadians[j])
            return servoZeroRadians[j] + servoDirectionSigns[j] * joint
        }
    }

    /// Calibrated Dynamixel feedback to the relative joint coordinates used
    /// in the policy observation.
    public func policyJointPositions(
        servoPositionRadians: [Float],
        expectedPolicyFingerprint: String
    ) throws -> [Float] {
        try validate(expectedPolicyFingerprint: expectedPolicyFingerprint)
        guard servoPositionRadians.count
                == Arachne15PolicyContract.actionDimension,
              servoPositionRadians.allSatisfy(\.isFinite) else {
            throw RobotContractError.invalidConfiguration(
                "servo feedback positions are invalid")
        }
        return servoPositionRadians.indices.map { j in
            servoDirectionSigns[j]
                * (servoPositionRadians[j] - servoZeroRadians[j])
        }
    }

    public func policyJointVelocities(
        servoVelocityRadiansPerSecond: [Float],
        expectedPolicyFingerprint: String
    ) throws -> [Float] {
        try validate(expectedPolicyFingerprint: expectedPolicyFingerprint)
        guard servoVelocityRadiansPerSecond.count
                == Arachne15PolicyContract.actionDimension,
              servoVelocityRadiansPerSecond.allSatisfy(\.isFinite) else {
            throw RobotContractError.invalidConfiguration(
                "servo feedback velocities are invalid")
        }
        return servoVelocityRadiansPerSecond.indices.map { j in
            servoDirectionSigns[j] * servoVelocityRadiansPerSecond[j]
        }
    }
}
