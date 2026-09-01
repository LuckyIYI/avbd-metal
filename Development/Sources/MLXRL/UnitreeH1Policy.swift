import CryptoKit
import Foundation
import MLX
import MLXNN
import SimCore
import PhysicsAVBD
import Robotics
import RL

public struct UnitreeH1PolicyManifest: Codable, Sendable {
    public struct Source: Codable, Sendable {
        public var project: String
        public var url: String
        public var revision: String
        public var checkpointSHA256: String
        public var license: String
        public var licenseFile: String
        public var licenseSHA256: String
    }

    public struct Network: Codable, Sendable {
        public var kind: String
        public var observationDimension: Int
        public var hiddenDimension: Int
        public var actionDimension: Int
        public var actorHiddenDimension: Int
        public var gateOrder: String
        public var activation: String
    }

    public struct Control: Codable, Sendable {
        public var physicsTimeStep: Float
        public var controlDecimation: Int
        public var periodSeconds: Float
        public var actionScale: Float
        public var angularVelocityScale: Float
        public var jointPositionScale: Float
        public var jointVelocityScale: Float
        public var commandScale: [Float]
        public var defaultCommand: [Float]
        public var defaultJointPositions: [Float]
        public var jointNames: [String]
        public var stiffness: [Float]
        public var damping: [Float]
    }

    public struct GoldenSequence: Codable, Sendable {
        public var inputs: [[Float]]
        public var actions: [[Float]]
        public var hiddenStates: [[Float]]
        public var cellStates: [[Float]]
        public var absoluteTolerance: Float
    }

    public var schemaVersion: Int
    public var format: String
    public var robot: String
    public var source: Source
    public var weightsFile: String
    public var weightsSHA256: String
    public var network: Network
    public var control: Control
    public var observationLayout: [String]
    public var goldenSequence: GoldenSequence
}

public struct UnitreeH1PolicyVerification: Sendable {
    public var maximumActionError: Float
    public var maximumHiddenStateError: Float
    public var maximumCellStateError: Float
    public var tolerance: Float

    public var passed: Bool {
        max(maximumActionError,
            max(maximumHiddenStateError, maximumCellStateError)) <= tolerance
    }
}

public enum UnitreeH1PolicyTrust: String, Codable, Sendable, Equatable {
    /// Exact manifest bytes and every independently pinned release identity
    /// match the independent release-index trust anchor supplied by the caller.
    case knownReleaseVerified
    /// Golden vectors may still provide a useful candidate self-check, but
    /// they came from the same unanchored manifest and prove no provenance.
    case unverifiedCandidate
}

/// MLX implementation of Unitree's exported `PolicyExporterLSTM` graph.
///
/// Torch is needed only by the explicit import tool. Runtime replay loads a
/// small, architecture-checked safetensors file and executes entirely in MLX.
public final class UnitreeH1RecurrentPolicy {
    public let manifest: UnitreeH1PolicyManifest
    public let manifestSHA256: String
    public let trust: UnitreeH1PolicyTrust

    private let weightInputHidden: MLXArray
    private let weightHiddenHidden: MLXArray
    private let biasInputHidden: MLXArray
    private let biasHiddenHidden: MLXArray
    private let actorHiddenWeight: MLXArray
    private let actorHiddenBias: MLXArray
    private let actorOutputWeight: MLXArray
    private let actorOutputBias: MLXArray
    private var hiddenState: MLXArray
    private var cellState: MLXArray
    private var batchSize: Int

    public init(
        directory: String, batchSize: Int = 1,
        expectedReleaseIdentity: UnitreeH1ReleaseIdentity? = nil
    ) throws {
        guard batchSize > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "Unitree H1 policy batch size must be positive")
        }
        let root = URL(fileURLWithPath: directory)
        let manifestData = try Data(contentsOf: root.appendingPathComponent(
            "manifest.json"))
        let actualManifestSHA256 = Self.sha256(manifestData)
        if let expectedReleaseIdentity {
            try Self.verifyDigest(
                actual: actualManifestSHA256,
                expected: expectedReleaseIdentity.manifestSHA256,
                label: "Unitree H1 manifest")
        }
        let decodedManifest = try JSONDecoder().decode(
            UnitreeH1PolicyManifest.self, from: manifestData)
        if let expectedReleaseIdentity {
            try Self.verifyKnownRelease(
                manifest: decodedManifest, manifestData: manifestData,
                expected: expectedReleaseIdentity)
        }
        manifest = decodedManifest
        manifestSHA256 = actualManifestSHA256
        trust = expectedReleaseIdentity == nil
            ? .unverifiedCandidate : .knownReleaseVerified
        guard manifest.schemaVersion == 2,
              manifest.format == "avbd-unitree-h1-lstm-v1",
              manifest.robot == "unitree-h1",
              manifest.source.project == "unitreerobotics/unitree_rl_gym",
              manifest.source.license == "BSD-3-Clause",
              manifest.source.licenseFile == "LICENSE",
              manifest.weightsFile == "policy.safetensors",
              manifest.network.kind == "lstm-actor",
              manifest.network.observationDimension == 41,
              manifest.network.hiddenDimension == 64,
              manifest.network.actionDimension == 10,
              manifest.network.actorHiddenDimension == 32,
              manifest.network.gateOrder == "ifgo",
              manifest.network.activation == "elu" else {
            throw RLEnvironmentError.invalidConfiguration(
                "unsupported Unitree H1 import manifest")
        }
        let weightsURL = root.appendingPathComponent(manifest.weightsFile)
        let weightsData = try Data(contentsOf: weightsURL)
        try Self.verifySHA256(
            data: weightsData, expected: manifest.weightsSHA256,
            label: "Unitree H1 converted policy")
        try Self.verifySHA256(
            data: try Data(contentsOf: root.appendingPathComponent(
                manifest.source.licenseFile)),
            expected: manifest.source.licenseSHA256,
            label: "Unitree H1 license")
        // Execute the same immutable bytes authenticated immediately above;
        // never reopen a mutable override directory between hash and parse.
        let weights = try loadArrays(data: weightsData)
        func tensor(_ name: String, shape: [Int]) throws -> MLXArray {
            guard let value = weights[name], value.shape == shape else {
                throw RLEnvironmentError.invalidConfiguration(
                    "Unitree H1 tensor \(name) is missing or has the wrong shape")
            }
            return value
        }
        weightInputHidden = try tensor("memory.weight_ih_l0", shape: [256, 41])
        weightHiddenHidden = try tensor("memory.weight_hh_l0", shape: [256, 64])
        biasInputHidden = try tensor("memory.bias_ih_l0", shape: [256])
        biasHiddenHidden = try tensor("memory.bias_hh_l0", shape: [256])
        actorHiddenWeight = try tensor("actor.0.weight", shape: [32, 64])
        actorHiddenBias = try tensor("actor.0.bias", shape: [32])
        actorOutputWeight = try tensor("actor.2.weight", shape: [10, 32])
        actorOutputBias = try tensor("actor.2.bias", shape: [10])
        self.batchSize = batchSize
        hiddenState = MLXArray.zeros([batchSize, 64])
        cellState = MLXArray.zeros([batchSize, 64])
        eval(weightInputHidden, weightHiddenHidden, biasInputHidden,
             biasHiddenHidden, actorHiddenWeight, actorHiddenBias,
             actorOutputWeight, actorOutputBias)
    }

    public func reset(batchSize: Int? = nil) {
        let requested = batchSize ?? self.batchSize
        precondition(requested > 0)
        self.batchSize = requested
        hiddenState = MLXArray.zeros([requested, manifest.network.hiddenDimension])
        cellState = MLXArray.zeros([requested, manifest.network.hiddenDimension])
        eval(hiddenState, cellState)
    }

    public func actions(observations: ContiguousArray<Float>) throws
        -> ContiguousArray<Float> {
        let expected = batchSize * manifest.network.observationDimension
        guard observations.count == expected else {
            throw RLEnvironmentError.invalidObservationCount(
                expected: expected, actual: observations.count)
        }
        let input = MLXArray(Array(observations)).reshaped(
            [batchSize, manifest.network.observationDimension])
        let gates = matmul(input, weightInputHidden.T)
            + matmul(hiddenState, weightHiddenHidden.T)
            + biasInputHidden + biasHiddenHidden
        let pieces = split(gates, parts: 4, axis: -1)
        let inputGate = MLX.sigmoid(pieces[0])
        let forgetGate = MLX.sigmoid(pieces[1])
        let candidate = MLX.tanh(pieces[2])
        let outputGate = MLX.sigmoid(pieces[3])
        cellState = forgetGate * cellState + inputGate * candidate
        hiddenState = outputGate * MLX.tanh(cellState)
        let actorHidden = MLXNN.elu(
            matmul(hiddenState, actorHiddenWeight.T) + actorHiddenBias)
        let action = matmul(actorHidden, actorOutputWeight.T) + actorOutputBias
        eval(action, hiddenState, cellState)
        return ContiguousArray(action.asArray(Float.self))
    }

    public func recurrentState() -> (hidden: [Float], cell: [Float]) {
        eval(hiddenState, cellState)
        return (hiddenState.asArray(Float.self), cellState.asArray(Float.self))
    }

    /// Compare MLX execution to outputs captured directly from the imported
    /// TorchScript checkpoint. This verifies recurrence, not only step zero.
    public func verifyGoldenSequence() throws -> UnitreeH1PolicyVerification {
        guard batchSize == 1 else {
            throw RLEnvironmentError.invalidConfiguration(
                "golden verification requires batch size one")
        }
        let golden = manifest.goldenSequence
        guard golden.inputs.count == golden.actions.count,
              golden.inputs.count == golden.hiddenStates.count,
              golden.inputs.count == golden.cellStates.count,
              !golden.inputs.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid Unitree H1 golden sequence")
        }
        reset()
        var actionError: Float = 0
        var hiddenError: Float = 0
        var cellError: Float = 0
        for step in golden.inputs.indices {
            let output = try actions(observations: ContiguousArray(
                golden.inputs[step]))
            let state = recurrentState()
            actionError = max(actionError,
                Self.maximumAbsoluteError(output, golden.actions[step]))
            hiddenError = max(hiddenError,
                Self.maximumAbsoluteError(state.hidden,
                                          golden.hiddenStates[step]))
            cellError = max(cellError,
                Self.maximumAbsoluteError(state.cell, golden.cellStates[step]))
        }
        reset()
        return UnitreeH1PolicyVerification(
            maximumActionError: actionError,
            maximumHiddenStateError: hiddenError,
            maximumCellStateError: cellError,
            tolerance: golden.absoluteTolerance)
    }

    private static func maximumAbsoluteError<C1: Collection, C2: Collection>(
        _ lhs: C1, _ rhs: C2
    ) -> Float where C1.Element == Float, C2.Element == Float {
        guard lhs.count == rhs.count else { return .infinity }
        return zip(lhs, rhs).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }

    private static func verifySHA256(
        data: Data, expected: String, label: String
    ) throws {
        guard expected.count == 64,
              expected.allSatisfy({ $0.isHexDigit }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "\(label) manifest SHA-256 is invalid")
        }
        let digest = sha256(data)
        guard digest == expected.lowercased() else {
            throw RLEnvironmentError.invalidConfiguration(
                "\(label) SHA-256 does not match its import manifest")
        }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }
            .joined()
    }

    private static func verifyDigest(
        actual: String, expected: String, label: String
    ) throws {
        guard expected.count == 64,
              expected.allSatisfy({ $0.isHexDigit }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "expected \(label) SHA-256 is invalid")
        }
        guard actual == expected.lowercased() else {
            throw RLEnvironmentError.invalidConfiguration(
                "\(label) bytes do not match the known release")
        }
    }

    private static func verifyKnownRelease(
        manifest: UnitreeH1PolicyManifest, manifestData: Data,
        expected: UnitreeH1ReleaseIdentity
    ) throws {
        guard manifest.source.revision == expected.sourceRevision,
              manifest.source.checkpointSHA256.lowercased()
                == expected.sourceCheckpointSHA256.lowercased(),
              manifest.weightsSHA256.lowercased()
                == expected.weightsSHA256.lowercased(),
              manifest.source.licenseSHA256.lowercased()
                == expected.licenseSHA256.lowercased() else {
            throw RLEnvironmentError.invalidConfiguration(
                "Unitree H1 source, weights, or license identity does not "
                    + "match the known release")
        }
        let object = try JSONSerialization.jsonObject(with: manifestData)
        guard let dictionary = object as? [String: Any],
              let golden = dictionary["goldenSequence"] else {
            throw RLEnvironmentError.invalidConfiguration(
                "Unitree H1 manifest has no golden sequence")
        }
        let canonicalGolden = try JSONSerialization.data(
            withJSONObject: golden,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try verifyDigest(
            actual: sha256(canonicalGolden),
            expected: expected.goldenSequenceSHA256,
            label: "Unitree H1 golden sequence")
    }
}

/// Exact 41-value observation contract in Unitree's MuJoCo deployment script.
public enum UnitreeH1ObservationEncoder {
    public static func encode(
        state: HumanoidState,
        command: SIMD3<Float>,
        previousAction: some Collection<Float>,
        elapsedTime: Float,
        control: UnitreeH1PolicyManifest.Control
    ) throws -> ContiguousArray<Float> {
        guard state.jointAngles.count >= 10,
              state.jointVelocities.count >= 10,
              previousAction.count == 10,
              control.commandScale.count == 3,
              control.defaultJointPositions.count == 10 else {
            throw RLEnvironmentError.invalidConfiguration(
                "state does not match Unitree H1 observation contract")
        }
        let localAngularVelocity = state.root.rotation.inverse.act(
            state.root.angularVelocity) * control.angularVelocityScale
        let projectedGravity = state.root.rotation.inverse.act(F3(0, 0, -1))
        let period = max(control.periodSeconds, 1e-6)
        let phase = elapsedTime.truncatingRemainder(dividingBy: period) / period
        var observation = ContiguousArray<Float>()
        observation.reserveCapacity(41)
        observation.append(contentsOf: [
            localAngularVelocity.x, localAngularVelocity.y,
            localAngularVelocity.z,
            projectedGravity.x, projectedGravity.y, projectedGravity.z,
            command.x * control.commandScale[0],
            command.y * control.commandScale[1],
            command.z * control.commandScale[2],
        ])
        for joint in 0..<10 {
            observation.append((state.jointAngles[joint]
                - control.defaultJointPositions[joint])
                * control.jointPositionScale)
        }
        for joint in 0..<10 {
            observation.append(state.jointVelocities[joint]
                * control.jointVelocityScale)
        }
        observation.append(contentsOf: previousAction)
        observation.append(sin(2 * .pi * phase))
        observation.append(cos(2 * .pi * phase))
        return observation
    }
}

public struct UnitreeH1Sim2SimReport: Codable, Sendable {
    public var schemaVersion: Int
    public var checkpointSHA256: String
    public var sourceRevision: String?
    public var command: [Float]
    public var controlSteps: Int
    public var simulatedSeconds: Float
    public var forwardDistanceMeters: Float
    public var lateralDistanceMeters: Float
    public var meanForwardSpeedMetersPerSecond: Float
    public var finalPelvisHeightMeters: Float
    public var minimumPelvisHeightMeters: Float
    public var minimumUprightAlignment: Float
    public var firstFallStep: Int?
    public var firstFallTimeSeconds: Float?
    public var fell: Bool
    public var finite: Bool
    public var manifestSHA256: String?
    public var policyTrust: UnitreeH1PolicyTrust?
    public var policyVerification: UnitreeH1PolicyVerificationRecord
}

public struct UnitreeH1PolicyVerificationRecord: Codable, Sendable {
    public var maximumActionError: Float
    public var maximumHiddenStateError: Float
    public var maximumCellStateError: Float
    public var tolerance: Float
    public var passed: Bool

    init(_ verification: UnitreeH1PolicyVerification) {
        maximumActionError = verification.maximumActionError
        maximumHiddenStateError = verification.maximumHiddenStateError
        maximumCellStateError = verification.maximumCellStateError
        tolerance = verification.tolerance
        passed = verification.passed
    }
}

/// Stateful, unchanged-policy sim-to-sim session. `step()` follows the exact
/// ordering in Unitree's deploy_mujoco.py: hold the current PD target for ten
/// 2-ms physics steps, observe, advance the LSTM, then install the next target.
public final class UnitreeH1Sim2SimSession {
    public let environment: UnitreeH1Sim2SimEnv
    public let policy: UnitreeH1RecurrentPolicy
    public let command: SIMD3<Float>
    public let policyVerification: UnitreeH1PolicyVerification

    public private(set) var elapsedTime: Float = 0
    public private(set) var controlSteps = 0
    public private(set) var previousAction = ContiguousArray<Float>(
        repeating: 0, count: 10)
    public private(set) var lastObservation = ContiguousArray<Float>()

    private var jointTargets = ContiguousArray<Float>(repeating: 0, count: 10)
    private let initialPosition: F3
    private var minimumHeight: Float
    private var minimumUpright: Float = 1
    private var firstFallStep: Int?
    private var allFinite = true

    public init(policyDirectory: String,
                command: SIMD3<Float> = SIMD3(0.5, 0, 0),
                solverIterations: Int? = nil,
                expectedReleaseIdentity: UnitreeH1ReleaseIdentity? = nil) throws {
        policy = try UnitreeH1RecurrentPolicy(
            directory: policyDirectory,
            expectedReleaseIdentity: expectedReleaseIdentity)
        policyVerification = try policy.verifyGoldenSequence()
        guard policyVerification.passed else {
            throw RLEnvironmentError.invalidConfiguration(
                "imported Unitree H1 policy failed its TorchScript golden sequence")
        }
        let control = policy.manifest.control
        guard abs(control.physicsTimeStep - 0.002) < 1e-8,
              control.controlDecimation == 10,
              control.defaultJointPositions.count == 10,
              control.stiffness == [150, 150, 150, 200, 40,
                                    150, 150, 150, 200, 40],
              control.damping == [2, 2, 2, 4, 2, 2, 2, 2, 4, 2] else {
            throw RLEnvironmentError.invalidConfiguration(
                "Unitree H1 control manifest does not match the supported plant profile")
        }
        self.command = command
        environment = try UnitreeH1Sim2SimEnv(
            solverIterations: solverIterations)
        let initial = environment.state()
        initialPosition = initial.root.position
        minimumHeight = initial.root.position.z
        for index in 0..<10 {
            // The source MuJoCo model starts at q=0 while its initial PD
            // target is the crouched `default_angles` vector.
            jointTargets[index] = control.defaultJointPositions[index]
        }
    }

    @discardableResult
    public func step() throws -> HumanoidState {
        let control = policy.manifest.control
        try environment.stepChecked(
            jointPositionTargets: jointTargets,
            decimation: control.controlDecimation)
        elapsedTime += control.physicsTimeStep * Float(control.controlDecimation)
        controlSteps += 1
        let state = environment.state()
        let observation = try UnitreeH1ObservationEncoder.encode(
            state: state, command: command,
            previousAction: previousAction, elapsedTime: elapsedTime,
            control: control)
        lastObservation = observation
        let action = try policy.actions(observations: observation)
        previousAction = action
        for index in 0..<10 {
            jointTargets[index] = control.defaultJointPositions[index]
                + action[index] * control.actionScale
        }

        let upright = state.root.rotation.act(F3(0, 0, 1)).z
        minimumHeight = min(minimumHeight, state.root.position.z)
        minimumUpright = min(minimumUpright, upright)
        let finiteValues = [
            state.root.position.x, state.root.position.y,
            state.root.position.z, upright,
        ] + Array(action)
        if !finiteValues.allSatisfy(\.isFinite) { allFinite = false }
        if firstFallStep == nil
            && (state.root.position.z < 0.55 || upright < 0.5) {
            firstFallStep = controlSteps
        }
        return state
    }

    public func run(controlSteps requestedSteps: Int,
                    onStep: ((Int, HumanoidState) -> Void)? = nil) throws
        -> UnitreeH1Sim2SimReport {
        precondition(requestedSteps >= 0)
        let control = policy.manifest.control
        var state = environment.state()
        for _ in 0..<requestedSteps {
            state = try step()
            onStep?(controlSteps, state)
        }
        let duration = max(elapsedTime, 1e-8)
        let forwardDistance = state.root.position.x - initialPosition.x
        let lateralDistance = state.root.position.y - initialPosition.y
        return UnitreeH1Sim2SimReport(
            schemaVersion: 1,
            checkpointSHA256: policy.manifest.source.checkpointSHA256,
            sourceRevision: policy.manifest.source.revision,
            command: [command.x, command.y, command.z],
            controlSteps: controlSteps,
            simulatedSeconds: elapsedTime,
            forwardDistanceMeters: forwardDistance,
            lateralDistanceMeters: lateralDistance,
            meanForwardSpeedMetersPerSecond: forwardDistance / duration,
            finalPelvisHeightMeters: state.root.position.z,
            minimumPelvisHeightMeters: minimumHeight,
            minimumUprightAlignment: minimumUpright,
            firstFallStep: firstFallStep,
            firstFallTimeSeconds: firstFallStep.map {
                Float($0) * control.physicsTimeStep
                    * Float(control.controlDecimation)
            },
            fell: firstFallStep != nil,
            finite: allFinite,
            manifestSHA256: policy.manifestSHA256,
            policyTrust: policy.trust,
            policyVerification: .init(policyVerification))
    }
}
