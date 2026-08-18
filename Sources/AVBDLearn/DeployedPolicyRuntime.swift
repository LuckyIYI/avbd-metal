import AVBDCore
import CryptoKit
import Foundation

/// Narrow inference surface shared by the MLX deployment runtime and safety
/// controller fakes. Hardware code does not need to depend on simulator task
/// types or optimizer state.
public protocol VectorPolicyInferencing: AnyObject {
    var observationDimension: Int { get }
    var actionDimension: Int { get }
    var controlPeriodSeconds: Double { get }
    var checkpointFingerprint: String { get }
    func actions(
        for observation: ContiguousArray<Float>
    ) throws -> ContiguousArray<Float>
}

/// A validated immutable deployment bundle plus its deterministic MLX actor.
/// Construction fails before loading the model if any file, semantic field,
/// tensor contract, or digest differs from the exported manifest.
public final class VectorPolicyDeploymentRuntime: VectorPolicyInferencing {
    public let manifest: VectorPolicyDeploymentManifest
    public let metadata: VectorPolicyMetadata
    public let trainingState: VectorPPOTrainingState

    private let runner: VectorPolicyRunner

    public var observationDimension: Int { manifest.observationDimension }
    public var actionDimension: Int { manifest.actionDimension }
    public var controlPeriodSeconds: Double {
        Double(manifest.simulationStepSeconds)
            * Double(manifest.controlDecimation)
    }
    public var checkpointFingerprint: String {
        manifest.checkpointFingerprint
    }

    public init(
        bundleDirectory: String,
        expectedTask: String? = nil,
        expectedTaskRevision: Int? = nil,
        expectedCheckpointFingerprint: String? = nil
    ) throws {
        let root = URL(fileURLWithPath: bundleDirectory, isDirectory: true)
            .standardizedFileURL
        let manifestURL = root.appendingPathComponent(
            VectorPolicyDeploymentBundle.manifestFileName)
        let manifestData = try Data(contentsOf: manifestURL)
        let decodedManifest = try JSONDecoder().decode(
            VectorPolicyDeploymentManifest.self, from: manifestData)
        guard decodedManifest.schemaVersion == 1 else {
            throw RLEnvironmentError.invalidConfiguration(
                "unsupported deployment manifest schema "
                    + "\(decodedManifest.schemaVersion)")
        }
        if let expectedTask, decodedManifest.task != expectedTask {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment task \(decodedManifest.task) does not match "
                    + "expected task \(expectedTask)")
        }
        if let expectedTaskRevision,
           decodedManifest.taskRevision != expectedTaskRevision {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment task revision \(decodedManifest.taskRevision) "
                    + "does not match expected revision \(expectedTaskRevision)")
        }
        if let expectedCheckpointFingerprint,
           decodedManifest.checkpointFingerprint
            != expectedCheckpointFingerprint {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment checkpoint fingerprint does not match the "
                    + "commissioned policy")
        }
        for name in [decodedManifest.policyFile,
                     decodedManifest.metadataFile,
                     decodedManifest.trainingStateFile] {
            guard !name.isEmpty,
                  URL(fileURLWithPath: name).lastPathComponent == name else {
                throw RLEnvironmentError.invalidConfiguration(
                    "deployment manifest contains a non-local file name")
            }
        }

        let metadataData = try Data(contentsOf: root.appendingPathComponent(
            decodedManifest.metadataFile))
        let trainingStateData = try Data(contentsOf: root.appendingPathComponent(
            decodedManifest.trainingStateFile))
        let policyData = try Data(contentsOf: root.appendingPathComponent(
            decodedManifest.policyFile))
        let decodedMetadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self, from: metadataData)
        let decodedTrainingState = try JSONDecoder().decode(
            VectorPPOTrainingState.self, from: trainingStateData)
        let digest = SHA256.hash(data: policyData).map {
            String(format: "%02x", $0)
        }.joined()
        guard digest == decodedManifest.policySHA256 else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment policy SHA-256 does not match its manifest")
        }
        let fingerprint = try VectorPPOTrainer.checkpointFingerprint(
            directory: root.path)
        guard fingerprint == decodedManifest.checkpointFingerprint else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment checkpoint fingerprint does not match its manifest")
        }
        try Self.validate(
            manifest: decodedManifest, metadata: decodedMetadata,
            trainingState: decodedTrainingState)

        manifest = decodedManifest
        metadata = decodedMetadata
        trainingState = decodedTrainingState
        runner = try VectorPolicyRunner(checkpointDirectory: root.path)
    }

    public func actions(
        for observation: ContiguousArray<Float>
    ) throws -> ContiguousArray<Float> {
        guard observation.count == observationDimension else {
            throw RLEnvironmentError.invalidObservationCount(
                expected: observationDimension, actual: observation.count)
        }
        let result = try runner.actions(for: observation)
        guard result.count == actionDimension else {
            throw RLEnvironmentError.invalidActionCount(
                expected: actionDimension, actual: result.count)
        }
        return result
    }

    private static func validate(
        manifest: VectorPolicyDeploymentManifest,
        metadata: VectorPolicyMetadata,
        trainingState: VectorPPOTrainingState
    ) throws {
        guard manifest.task == metadata.task,
              manifest.taskRevision == metadata.taskRevision,
              manifest.architectureVersion == metadata.architectureVersion,
              manifest.observationDimension == metadata.observationDimension,
              manifest.actionDimension == metadata.actionDimension,
              manifest.simulationStepSeconds == metadata.simulationStep,
              manifest.controlDecimation == metadata.controlDecimation,
              manifest.normalizesObservations
                == metadata.ppo.normalizeObservations,
              manifest.actionDistribution
                == metadata.ppo.resolvedActionDistribution.rawValue,
              manifest.taskConfiguration == metadata.taskConfiguration,
              manifest.trainingUpdates == trainingState.completedUpdates,
              manifest.trainingEnvironmentSteps
                == trainingState.environmentSteps else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment manifest disagrees with checkpoint metadata")
        }
        guard manifest.observationDimension > 0,
              manifest.actionDimension > 0,
              manifest.simulationStepSeconds.isFinite,
              manifest.simulationStepSeconds > 0,
              manifest.controlDecimation > 0,
              manifest.controlFrequencyHz.isFinite,
              manifest.controlFrequencyHz > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment timing or tensor dimensions are invalid")
        }
        let derivedFrequency = 1 / (manifest.simulationStepSeconds
            * Float(manifest.controlDecimation))
        guard abs(derivedFrequency - manifest.controlFrequencyHz) < 1e-3 else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment control frequency is internally inconsistent")
        }
        let normalizer = metadata.normalizer
        guard normalizer.mean.count == manifest.observationDimension,
              normalizer.variance.count == manifest.observationDimension,
              normalizer.count.isFinite, normalizer.count >= 0,
              normalizer.mean.allSatisfy(\.isFinite),
              normalizer.variance.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment observation normalizer is invalid")
        }
        guard VectorActorCritic.compatibleArchitectureVersions.contains(
            manifest.architectureVersion) else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment policy architecture is not supported")
        }
    }
}

public struct PolicyObservationFrame: Codable, Sendable, Equatable {
    public var sequence: UInt64
    /// Monotonic time in seconds from the controller process's clock domain.
    public var timestampSeconds: Double
    public var values: ContiguousArray<Float>

    public init(sequence: UInt64, timestampSeconds: Double,
                values: ContiguousArray<Float>) {
        self.sequence = sequence
        self.timestampSeconds = timestampSeconds
        self.values = values
    }
}

public enum PolicyActuationMode: String, Codable, Sendable {
    case active
    /// The bridge must execute its independently verified low-body ramp and
    /// torque-off sequence; these action values are diagnostic only.
    case safeStop = "safe-stop"
}

public enum PolicyControlFault: String, Codable, Sendable, Equatable {
    case notArmed = "not-armed"
    case emergencyStop = "emergency-stop"
    case staleObservation = "stale-observation"
    case futureObservation = "future-observation"
    case outOfOrderObservation = "out-of-order-observation"
    case invalidObservation = "invalid-observation"
    case inferenceFailure = "inference-failure"
    case inferenceDeadlineMiss = "inference-deadline-miss"
    case invalidAction = "invalid-action"
}

public struct PolicyCommandFrame: Codable, Sendable, Equatable {
    public var sequence: UInt64
    public var createdAtSeconds: Double
    public var validUntilSeconds: Double
    public var policyFingerprint: String
    public var mode: PolicyActuationMode
    public var values: ContiguousArray<Float>
    public var fault: PolicyControlFault?
}

public struct PolicyControlSafetyConfiguration: Sendable, Equatable {
    public var maximumObservationAgeSeconds: Double
    public var maximumFutureClockSkewSeconds: Double
    public var maximumInferenceDurationSeconds: Double
    public var maximumAbsoluteAction: Float

    public init(maximumObservationAgeSeconds: Double = 0.060,
                maximumFutureClockSkewSeconds: Double = 0.005,
                maximumInferenceDurationSeconds: Double = 0.018,
                maximumAbsoluteAction: Float = 1.0001) {
        precondition(maximumObservationAgeSeconds > 0)
        precondition(maximumFutureClockSkewSeconds >= 0)
        precondition(maximumInferenceDurationSeconds > 0)
        precondition(maximumAbsoluteAction > 0)
        self.maximumObservationAgeSeconds = maximumObservationAgeSeconds
        self.maximumFutureClockSkewSeconds = maximumFutureClockSkewSeconds
        self.maximumInferenceDurationSeconds = maximumInferenceDurationSeconds
        self.maximumAbsoluteAction = maximumAbsoluteAction
    }
}

/// Fail-closed policy supervisor for the iPhone→safety-bridge boundary. Any
/// sensor, ordering, inference, deadline, or action fault latches a safe stop;
/// recovery requires an explicit re-arm after the underlying fault is fixed.
public final class GuardedPolicyController {
    public let inference: any VectorPolicyInferencing
    public let safety: PolicyControlSafetyConfiguration

    public private(set) var isArmed = false
    public private(set) var latchedFault: PolicyControlFault?
    private var previousSequence: UInt64?

    public init(inference: any VectorPolicyInferencing,
                safety: PolicyControlSafetyConfiguration = .init()) {
        self.inference = inference
        self.safety = safety
    }

    public func arm() {
        isArmed = true
        latchedFault = nil
        previousSequence = nil
    }

    public func emergencyStop() {
        latch(.emergencyStop)
    }

    public func disarm() {
        latch(.notArmed)
    }

    public func command(
        for frame: PolicyObservationFrame,
        nowSeconds: Double = ProcessInfo.processInfo.systemUptime
    ) -> PolicyCommandFrame {
        guard isArmed, latchedFault == nil else {
            return safeStop(sequence: frame.sequence, now: nowSeconds,
                            fault: latchedFault ?? .notArmed)
        }
        guard nowSeconds.isFinite, frame.timestampSeconds.isFinite,
              frame.values.count == inference.observationDimension,
              frame.values.allSatisfy(\.isFinite) else {
            return fail(.invalidObservation, sequence: frame.sequence,
                        now: nowSeconds)
        }
        let age = nowSeconds - frame.timestampSeconds
        guard age <= safety.maximumObservationAgeSeconds else {
            return fail(.staleObservation, sequence: frame.sequence,
                        now: nowSeconds)
        }
        guard age >= -safety.maximumFutureClockSkewSeconds else {
            return fail(.futureObservation, sequence: frame.sequence,
                        now: nowSeconds)
        }
        if let previousSequence, frame.sequence <= previousSequence {
            return fail(.outOfOrderObservation, sequence: frame.sequence,
                        now: nowSeconds)
        }

        let start = ProcessInfo.processInfo.systemUptime
        let actions: ContiguousArray<Float>
        do {
            actions = try inference.actions(for: frame.values)
        } catch {
            return fail(.inferenceFailure, sequence: frame.sequence,
                        now: ProcessInfo.processInfo.systemUptime)
        }
        let completed = ProcessInfo.processInfo.systemUptime
        guard completed - start <= safety.maximumInferenceDurationSeconds else {
            return fail(.inferenceDeadlineMiss, sequence: frame.sequence,
                        now: completed)
        }
        guard actions.count == inference.actionDimension,
              actions.allSatisfy({ $0.isFinite
                && abs($0) <= safety.maximumAbsoluteAction }) else {
            return fail(.invalidAction, sequence: frame.sequence,
                        now: completed)
        }
        previousSequence = frame.sequence
        return PolicyCommandFrame(
            sequence: frame.sequence, createdAtSeconds: completed,
            validUntilSeconds: completed + inference.controlPeriodSeconds,
            policyFingerprint: inference.checkpointFingerprint,
            mode: .active, values: actions, fault: nil)
    }

    private func fail(_ fault: PolicyControlFault, sequence: UInt64,
                      now: Double) -> PolicyCommandFrame {
        latch(fault)
        return safeStop(sequence: sequence, now: now, fault: fault)
    }

    private func latch(_ fault: PolicyControlFault) {
        latchedFault = fault
        isArmed = false
    }

    private func safeStop(sequence: UInt64, now: Double,
                          fault: PolicyControlFault) -> PolicyCommandFrame {
        let safeNow = now.isFinite ? now : 0
        return PolicyCommandFrame(
            sequence: sequence, createdAtSeconds: safeNow,
            validUntilSeconds: safeNow,
            policyFingerprint: inference.checkpointFingerprint,
            mode: .safeStop,
            values: ContiguousArray(
                repeating: 0, count: inference.actionDimension),
            fault: fault)
    }
}

/// Transport-ready command for the ESP32-S3 bridge. An active frame contains
/// calibrated servo shaft targets; a safe-stop frame deliberately contains no
/// positions, requiring the bridge's independent stop routine to take over.
public struct Arachne15ServoCommandFrame: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var sequence: UInt64
    public var createdAtSeconds: Double
    public var validUntilSeconds: Double
    public var policyFingerprint: String
    public var mode: PolicyActuationMode
    public var servoIDs: [Int]
    public var servoPositionRadians: [Float]
    public var currentLimitsMilliamps: [Int]
    public var maximumServoTemperatureCelsius: Float
    public var fault: PolicyControlFault?
}

/// End-to-end Arachne deployment path: hardware state encoding, immutable MLX
/// inference, fail-closed timing supervision, and calibrated servo mapping.
/// Network framing and Dynamixel bus execution remain bridge-owned.
public final class Arachne15DeploymentController {
    public let inference: any VectorPolicyInferencing
    public let calibration: Arachne15HardwareCalibration
    public let supervisor: GuardedPolicyController

    public convenience init(
        bundleDirectory: String,
        calibration: Arachne15HardwareCalibration,
        safety: PolicyControlSafetyConfiguration = .init()
    ) throws {
        let runtime = try VectorPolicyDeploymentRuntime(
            bundleDirectory: bundleDirectory,
            expectedTask: "arachne15-goal-v0",
            expectedTaskRevision:
                Arachne15LocomotionTask.currentTaskRevision,
            expectedCheckpointFingerprint:
                calibration.policyCheckpointFingerprint)
        try self.init(inference: runtime, calibration: calibration,
                      safety: safety)
    }

    public init(
        inference: any VectorPolicyInferencing,
        calibration: Arachne15HardwareCalibration,
        safety: PolicyControlSafetyConfiguration = .init()
    ) throws {
        guard inference.observationDimension
                == Arachne15PolicyContract.observationDimension,
              inference.actionDimension
                == Arachne15PolicyContract.actionDimension,
              abs(inference.controlPeriodSeconds - 0.02) < 1e-6 else {
            throw RLEnvironmentError.invalidConfiguration(
                "Arachne deployment inference tensor or timing mismatch")
        }
        try calibration.validate(
            expectedPolicyFingerprint: inference.checkpointFingerprint)
        self.inference = inference
        self.calibration = calibration
        supervisor = GuardedPolicyController(
            inference: inference, safety: safety)
    }

    public func arm() { supervisor.arm() }
    public func emergencyStop() { supervisor.emergencyStop() }
    public func disarm() { supervisor.disarm() }

    public func command(
        for input: Arachne15PolicyInput,
        sequence: UInt64,
        sensorTimestampSeconds: Double,
        nowSeconds: Double = ProcessInfo.processInfo.systemUptime
    ) -> Arachne15ServoCommandFrame {
        let observation: ContiguousArray<Float>
        do {
            observation = try Arachne15PolicyContract.encode(input)
        } catch {
            // A wrong-sized observation forces the same latched fail-closed
            // path as corrupt or missing sensor data.
            return makeServoFrame(from: supervisor.command(
                for: .init(sequence: sequence,
                    timestampSeconds: sensorTimestampSeconds, values: []),
                nowSeconds: nowSeconds))
        }
        let policyCommand = supervisor.command(
            for: .init(sequence: sequence,
                       timestampSeconds: sensorTimestampSeconds,
                       values: observation),
            nowSeconds: nowSeconds)
        return makeServoFrame(from: policyCommand)
    }

    private func makeServoFrame(
        from command: PolicyCommandFrame
    ) -> Arachne15ServoCommandFrame {
        var positions = [Float]()
        var mode = command.mode
        var fault = command.fault
        if command.mode == .active {
            do {
                positions = try calibration.servoPositionRadians(
                    for: command.values,
                    expectedPolicyFingerprint:
                        inference.checkpointFingerprint)
            } catch {
                supervisor.emergencyStop()
                mode = .safeStop
                fault = .invalidAction
            }
        }
        return Arachne15ServoCommandFrame(
            schemaVersion: 1, sequence: command.sequence,
            createdAtSeconds: command.createdAtSeconds,
            validUntilSeconds: mode == .active
                ? command.validUntilSeconds : command.createdAtSeconds,
            policyFingerprint: command.policyFingerprint, mode: mode,
            servoIDs: calibration.servoIDs,
            servoPositionRadians: mode == .active ? positions : [],
            currentLimitsMilliamps: calibration.currentLimitsMilliamps,
            maximumServoTemperatureCelsius:
                calibration.maximumServoTemperatureCelsius,
            fault: fault)
    }
}
