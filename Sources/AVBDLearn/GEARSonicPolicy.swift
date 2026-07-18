import AVBDCore
import CryptoKit
import Foundation
import MLX
import MLXNN

/// Reproducible contract for NVIDIA's released GEAR-SONIC G1 controller.
public struct GEARSonicG1PolicyManifest: Codable, Sendable {
    public struct Source: Codable, Sendable {
        public var project: String
        public var url: String
        public var modelRevision: String?
        public var encoderSHA256: String
        public var decoderSHA256: String
        public var observationConfigSHA256: String
        public var codeProject: String
        public var codeURL: String
        public var codeRevision: String?
        public var license: String
        public var licenseFile: String
        public var licenseSHA256: String
    }

    public struct Quantizer: Codable, Sendable {
        public var offset: Float
        public var scale: Float
        public var halfStep: Float
        public var levels: Float
    }

    public struct Network: Codable, Sendable {
        public var kind: String
        public var compactReferenceDimension: Int
        public var tokenDimension: Int
        public var historyDimension: Int
        public var decoderInputDimension: Int
        public var actionDimension: Int
        public var encoderLayerDimensions: [Int]
        public var decoderLayerDimensions: [Int]
        public var activation: String
        public var encoderMode: String
        public var encoderModeID: Int
        public var quantizer: Quantizer
    }

    public struct Control: Codable, Sendable {
        public var physicsTimeStep: Float
        public var controlDecimation: Int
        public var periodSeconds: Float
        public var referenceFrames: Int
        public var referenceFrameStride: Int
        public var historyFrames: Int
        public var historyOldestFirst: Bool
        public var actuatorJointNames: [String]
        public var policyJointNames: [String]
        public var actuatorToPolicy: [Int]
        public var policyToActuator: [Int]
        public var defaultJointPositions: [Float]
        public var actionScale: [Float]
        /// Source Isaac Lab implicit-PD gains, also used by the deploy client.
        public var stiffness: [Float]
        public var damping: [Float]
        /// Isaac Lab training-plant actuator properties in actuator order.
        public var trainingArmature: [Float]
        public var trainingEffortLimit: [Float]
        public var trainingVelocityLimit: [Float]
        /// Force clamps from NVIDIA's MuJoCo/hardware deployment plant.
        public var deploymentEffortLimit: [Float]
    }

    public struct Golden: Codable, Sendable {
        public var referenceInputs: [[Float]]
        public var historyInputs: [[Float]]
        public var tokens: [[Float]]
        public var actions: [[Float]]
        public var tokenAbsoluteTolerance: Float
        public var actionAbsoluteTolerance: Float
        public var minimumQuantizerTieMargin: Float
    }

    public var schemaVersion: Int
    public var format: String
    public var robot: String
    public var source: Source
    public var weightsFile: String
    public var weightsSHA256: String
    public var network: Network
    public var control: Control
    public var referenceObservationLayout: [String]
    public var historyObservationLayout: [String]
    public var golden: Golden
}

public struct GEARSonicG1PolicyVerification: Sendable {
    public var maximumTokenError: Float
    public var maximumActionError: Float
    public var tokenTolerance: Float
    public var actionTolerance: Float

    public var passed: Bool {
        maximumTokenError <= tokenTolerance
            && maximumActionError <= actionTolerance
    }
}

/// Stateless, dynamically batched native-MLX execution of GEAR-SONIC.
///
/// ONNX is used by the import tool only. Runtime inference consumes the exact
/// 640-value G1 reference and 930-value history contracts and returns the 29
/// raw actions in Isaac Lab policy order.
public final class GEARSonicG1Policy {
    public let manifest: GEARSonicG1PolicyManifest

    private let encoderLayers: [Linear]
    private let decoderLayers: [Linear]

    public init(directory: String) throws {
        let root = URL(fileURLWithPath: directory, isDirectory: true)
            .standardizedFileURL
        manifest = try JSONDecoder().decode(
            GEARSonicG1PolicyManifest.self,
            from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
        try Self.validate(manifest: manifest)
        try Self.validateLocalFileName(manifest.weightsFile, field: "weightsFile")
        try Self.validateLocalFileName(
            manifest.source.licenseFile, field: "source.licenseFile")

        let weightsURL = root.appendingPathComponent(manifest.weightsFile)
        try Self.verifySHA256(
            url: weightsURL, expected: manifest.weightsSHA256,
            description: "GEAR-SONIC policy weights")
        try Self.verifySHA256(
            url: root.appendingPathComponent(manifest.source.licenseFile),
            expected: manifest.source.licenseSHA256,
            description: "GEAR-SONIC model license")

        let weights = try loadArrays(url: weightsURL)
        let expectedNames = Set((0..<5).flatMap {
            ["encoder.\($0).weight", "encoder.\($0).bias"]
        } + (0..<7).flatMap {
            ["decoder.\($0).weight", "decoder.\($0).bias"]
        })
        guard Set(weights.keys) == expectedNames else {
            let missing = expectedNames.subtracting(weights.keys).sorted()
            let unexpected = Set(weights.keys).subtracting(expectedNames).sorted()
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC tensor set mismatch; missing=\(missing), "
                    + "unexpected=\(unexpected)")
        }

        func layers(prefix: String, dimensions: [Int]) throws -> [Linear] {
            try (0..<(dimensions.count - 1)).map { index in
                let weightName = "\(prefix).\(index).weight"
                let biasName = "\(prefix).\(index).bias"
                guard let weight = weights[weightName],
                      weight.shape == [dimensions[index + 1], dimensions[index]],
                      let bias = weights[biasName],
                      bias.shape == [dimensions[index + 1]] else {
                    throw RLEnvironmentError.invalidConfiguration(
                        "GEAR-SONIC tensor \(prefix).\(index) has the wrong shape")
                }
                return Linear(weight: weight, bias: bias)
            }
        }
        encoderLayers = try layers(
            prefix: "encoder", dimensions: manifest.network.encoderLayerDimensions)
        decoderLayers = try layers(
            prefix: "decoder", dimensions: manifest.network.decoderLayerDimensions)
        eval((encoderLayers + decoderLayers).flatMap { layer in
            [layer.weight, layer.bias!]
        })
    }

    /// Encode an already-batched semantic reference matrix `[B, 640]`.
    public func tokens(referenceObservations input: MLXArray) throws -> MLXArray {
        let dimension = manifest.network.compactReferenceDimension
        guard input.ndim == 2, input.dim(0) > 0, input.dim(1) == dimension else {
            throw RLEnvironmentError.invalidActionShape(
                expected: [-1, dimension], actual: input.shape)
        }

        // The exported graph concatenates the per-frame anchor after reshaping
        // the q/qd block. This ordering is intentionally not plain q+qd+anchor.
        let batch = input.dim(0)
        let jointReference = input[0..., 0..<580].reshaped([batch, 10, 58])
        let anchorReference = input[0..., 580..<640].reshaped([batch, 10, 6])
        var value = concatenated(
            [jointReference, anchorReference], axis: 2).reshaped([batch, 640])
        for (index, layer) in encoderLayers.enumerated() {
            value = layer(value)
            if index + 1 < encoderLayers.count {
                value = MLXNN.silu(value)
            }
        }
        let quantizer = manifest.network.quantizer
        return MLX.round(
            MLX.tanh(value + quantizer.offset) * quantizer.scale
                - quantizer.halfStep) / quantizer.levels
    }

    /// Decode `[B, 64]` tokens and `[B, 930]` history into raw `[B, 29]` actions.
    public func actions(
        tokens: MLXArray,
        historyObservations history: MLXArray
    ) throws -> MLXArray {
        let tokenDimension = manifest.network.tokenDimension
        let historyDimension = manifest.network.historyDimension
        guard tokens.ndim == 2, tokens.dim(0) > 0,
              tokens.dim(1) == tokenDimension else {
            throw RLEnvironmentError.invalidActionShape(
                expected: [-1, tokenDimension], actual: tokens.shape)
        }
        guard history.ndim == 2, history.dim(0) == tokens.dim(0),
              history.dim(1) == historyDimension else {
            throw RLEnvironmentError.invalidConfiguration(
                "expected GEAR-SONIC history shape "
                    + "[\(tokens.dim(0)), \(historyDimension)], got \(history.shape)")
        }
        var value = concatenated([tokens, history], axis: 1)
        for (index, layer) in decoderLayers.enumerated() {
            // The released decoder graph is MatMul followed by Add (not Gemm).
            // Preserve that boundary instead of Linear's fused addMM path.
            value = matmul(value, layer.weight.T) + layer.bias!
            if index + 1 < decoderLayers.count {
                value = MLXNN.silu(value)
            }
        }
        return value
    }

    /// End-to-end device-resident inference for arbitrary batch size.
    public func actions(
        referenceObservations: MLXArray,
        historyObservations: MLXArray
    ) throws -> MLXArray {
        try actions(
            tokens: tokens(referenceObservations: referenceObservations),
            historyObservations: historyObservations)
    }

    /// Host-array convenience used by replay and verification paths.
    public func tokens(
        referenceObservations: ContiguousArray<Float>
    ) throws -> ContiguousArray<Float> {
        let dimension = manifest.network.compactReferenceDimension
        let batch = try Self.batchSize(
            count: referenceObservations.count, dimension: dimension,
            description: "reference")
        let output = try tokens(referenceObservations:
            MLXArray(Array(referenceObservations)).reshaped([batch, dimension]))
        eval(output)
        return try Self.finiteHostValues(output, description: "tokens")
    }

    /// Host-array convenience used by replay and verification paths.
    public func actions(
        referenceObservations: ContiguousArray<Float>,
        historyObservations: ContiguousArray<Float>
    ) throws -> ContiguousArray<Float> {
        let referenceDimension = manifest.network.compactReferenceDimension
        let historyDimension = manifest.network.historyDimension
        let batch = try Self.batchSize(
            count: referenceObservations.count, dimension: referenceDimension,
            description: "reference")
        guard historyObservations.count == batch * historyDimension else {
            throw RLEnvironmentError.invalidObservationCount(
                expected: batch * historyDimension,
                actual: historyObservations.count)
        }
        let reference = MLXArray(Array(referenceObservations)).reshaped(
            [batch, referenceDimension])
        let history = MLXArray(Array(historyObservations)).reshaped(
            [batch, historyDimension])
        let output = try actions(
            referenceObservations: reference, historyObservations: history)
        eval(output)
        return try Self.finiteHostValues(output, description: "actions")
    }

    /// Runs both importer-captured ONNX examples as one native MLX batch.
    public func verifyGoldenBatch() throws -> GEARSonicG1PolicyVerification {
        let golden = manifest.golden
        let batch = golden.referenceInputs.count
        guard batch > 0,
              golden.historyInputs.count == batch,
              golden.tokens.count == batch,
              golden.actions.count == batch,
              golden.referenceInputs.allSatisfy({
                  $0.count == manifest.network.compactReferenceDimension }),
              golden.historyInputs.allSatisfy({
                  $0.count == manifest.network.historyDimension }),
              golden.tokens.allSatisfy({
                  $0.count == manifest.network.tokenDimension }),
              golden.actions.allSatisfy({
                  $0.count == manifest.network.actionDimension }),
              golden.minimumQuantizerTieMargin >= 0.02 else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC manifest has invalid golden tensors")
        }
        let reference = ContiguousArray(golden.referenceInputs.flatMap { $0 })
        let history = ContiguousArray(golden.historyInputs.flatMap { $0 })
        let actualTokens = try tokens(referenceObservations: reference)
        let actualActions = try actions(
            referenceObservations: reference, historyObservations: history)
        return GEARSonicG1PolicyVerification(
            maximumTokenError: Self.maximumAbsoluteError(
                actualTokens, golden.tokens.flatMap { $0 }),
            maximumActionError: Self.maximumAbsoluteError(
                actualActions, golden.actions.flatMap { $0 }),
            tokenTolerance: golden.tokenAbsoluteTolerance,
            actionTolerance: golden.actionAbsoluteTolerance)
    }

    private static func validate(manifest: GEARSonicG1PolicyManifest) throws {
        let network = manifest.network
        guard manifest.schemaVersion == 1,
              manifest.format == "avbd-gear-sonic-g1-v1",
              manifest.robot == "unitree-g1-29dof",
              network.kind == "gear-sonic-g1-encoder-decoder",
              network.compactReferenceDimension == 640,
              network.tokenDimension == 64,
              network.historyDimension == 930,
              network.decoderInputDimension == 994,
              network.actionDimension == 29,
              network.encoderLayerDimensions == [640, 2048, 1024, 512, 512, 64],
              network.decoderLayerDimensions
                == [994, 2048, 2048, 1024, 1024, 512, 512, 29],
              network.activation == "silu",
              network.encoderMode == "g1", network.encoderModeID == 0,
              network.quantizer.levels > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "unsupported GEAR-SONIC G1 import manifest")
        }

        let control = manifest.control
        let count = network.actionDimension
        let vectorCounts = [
            control.actuatorJointNames.count, control.policyJointNames.count,
            control.actuatorToPolicy.count, control.policyToActuator.count,
            control.defaultJointPositions.count, control.actionScale.count,
            control.stiffness.count, control.damping.count,
            control.trainingArmature.count,
            control.trainingEffortLimit.count,
            control.trainingVelocityLimit.count,
            control.deploymentEffortLimit.count,
        ]
        let actuatorSet = Set(control.actuatorToPolicy)
        let policySet = Set(control.policyToActuator)
        guard vectorCounts.allSatisfy({ $0 == count }),
              actuatorSet == Set(0..<count), policySet == Set(0..<count),
              control.actuatorToPolicy.enumerated().allSatisfy({ actuator, policy in
                  control.policyToActuator[policy] == actuator
              }),
              abs(control.physicsTimeStep - 0.005) < 1e-8,
              control.controlDecimation == 4,
              abs(control.periodSeconds - 0.02) < 1e-8,
              control.referenceFrames == 10,
              control.referenceFrameStride == 5,
              control.historyFrames == 10,
              control.historyOldestFirst,
              (control.defaultJointPositions + control.actionScale
                + control.stiffness + control.damping
                + control.trainingArmature + control.trainingEffortLimit
                + control.trainingVelocityLimit
                + control.deploymentEffortLimit).allSatisfy(\.isFinite),
              control.actionScale.allSatisfy({ $0 > 0 }),
              control.stiffness.allSatisfy({ $0 > 0 }),
              control.damping.allSatisfy({ $0 > 0 }),
              control.trainingArmature.allSatisfy({ $0 > 0 }),
              control.trainingEffortLimit.allSatisfy({ $0 > 0 }),
              control.trainingVelocityLimit.allSatisfy({ $0 > 0 }),
              control.deploymentEffortLimit.allSatisfy({ $0 > 0 }),
              zip(control.actionScale.indices, control.actionScale).allSatisfy({ pair in
                  abs(pair.1 - 0.25 * control.trainingEffortLimit[pair.0]
                      / control.stiffness[pair.0]) < 1e-6
              }) else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid GEAR-SONIC G1 control contract")
        }
    }

    private static func validateLocalFileName(
        _ name: String, field: String
    ) throws {
        guard !name.isEmpty, URL(fileURLWithPath: name).lastPathComponent == name else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC \(field) must be a local file name")
        }
    }

    private static func verifySHA256(
        url: URL, expected: String, description: String
    ) throws {
        let digest = SHA256.hash(data: try Data(contentsOf: url)).map {
            String(format: "%02x", $0)
        }.joined()
        guard digest == expected else {
            throw RLEnvironmentError.invalidConfiguration(
                "\(description) SHA-256 does not match its manifest")
        }
    }

    private static func batchSize(
        count: Int, dimension: Int, description: String
    ) throws -> Int {
        guard count > 0, count.isMultiple(of: dimension) else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC \(description) input must contain complete "
                    + "\(dimension)-value rows")
        }
        return count / dimension
    }

    private static func finiteHostValues(
        _ value: MLXArray, description: String
    ) throws -> ContiguousArray<Float> {
        let result = ContiguousArray(value.asArray(Float.self))
        guard result.allSatisfy(\.isFinite) else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC produced non-finite \(description)")
        }
        return result
    }

    private static func maximumAbsoluteError<C1: Collection, C2: Collection>(
        _ lhs: C1, _ rhs: C2
    ) -> Float where C1.Element == Float, C2.Element == Float {
        guard lhs.count == rhs.count else { return .infinity }
        return zip(lhs, rhs).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }
}

/// Exact feature-group-major ten-frame history expected by the decoder.
public struct GEARSonicG1HistoryBuffer: Sendable {
    public let batchSize: Int
    public private(set) var storedFrameCount = 0

    private let frameCapacity = 10
    private var nextWriteFrame = 0
    private var baseAngularVelocity: [Float]
    private var jointPositionResidual: [Float]
    private var jointVelocity: [Float]
    private var previousRawAction: [Float]
    private var projectedGravity: [Float]

    public init(batchSize: Int) throws {
        guard batchSize > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC history batch size must be positive")
        }
        self.batchSize = batchSize
        baseAngularVelocity = .init(repeating: 0, count: 10 * batchSize * 3)
        jointPositionResidual = .init(repeating: 0, count: 10 * batchSize * 29)
        jointVelocity = .init(repeating: 0, count: 10 * batchSize * 29)
        previousRawAction = .init(repeating: 0, count: 10 * batchSize * 29)
        projectedGravity = .init(repeating: 0, count: 10 * batchSize * 3)
    }

    public mutating func reset() {
        storedFrameCount = 0
        nextWriteFrame = 0
        baseAngularVelocity = .init(repeating: 0, count: baseAngularVelocity.count)
        jointPositionResidual = .init(repeating: 0, count: jointPositionResidual.count)
        jointVelocity = .init(repeating: 0, count: jointVelocity.count)
        previousRawAction = .init(repeating: 0, count: previousRawAction.count)
        projectedGravity = .init(repeating: 0, count: projectedGravity.count)
    }

    public mutating func append(
        baseAngularVelocity angularVelocity: ContiguousArray<Float>,
        jointPositionResidual positionResidual: ContiguousArray<Float>,
        jointVelocity velocity: ContiguousArray<Float>,
        previousRawAction action: ContiguousArray<Float>,
        projectedGravity gravity: ContiguousArray<Float>
    ) throws {
        try requireCount(angularVelocity, width: 3, name: "base angular velocity")
        try requireCount(positionResidual, width: 29, name: "joint position")
        try requireCount(velocity, width: 29, name: "joint velocity")
        try requireCount(action, width: 29, name: "previous action")
        try requireCount(gravity, width: 3, name: "projected gravity")
        Self.write(angularVelocity, into: &baseAngularVelocity, width: 3,
                   frame: nextWriteFrame, batchSize: batchSize)
        Self.write(positionResidual, into: &jointPositionResidual, width: 29,
                   frame: nextWriteFrame, batchSize: batchSize)
        Self.write(velocity, into: &jointVelocity, width: 29,
                   frame: nextWriteFrame, batchSize: batchSize)
        Self.write(action, into: &previousRawAction, width: 29,
                   frame: nextWriteFrame, batchSize: batchSize)
        Self.write(gravity, into: &projectedGravity, width: 3,
                   frame: nextWriteFrame, batchSize: batchSize)
        nextWriteFrame = (nextWriteFrame + 1) % frameCapacity
        storedFrameCount = min(storedFrameCount + 1, frameCapacity)
    }

    /// Returns `[B, 930]`, with missing startup frames padded at the front.
    public func observations() -> ContiguousArray<Float> {
        var output = ContiguousArray<Float>()
        output.reserveCapacity(batchSize * 930)
        for environment in 0..<batchSize {
            appendField(baseAngularVelocity, width: 3,
                        environment: environment, to: &output)
            appendField(jointPositionResidual, width: 29,
                        environment: environment, to: &output)
            appendField(jointVelocity, width: 29,
                        environment: environment, to: &output)
            appendField(previousRawAction, width: 29,
                        environment: environment, to: &output)
            appendField(projectedGravity, width: 3,
                        environment: environment, to: &output)
        }
        return output
    }

    private func requireCount(
        _ values: ContiguousArray<Float>, width: Int, name: String
    ) throws {
        guard values.count == batchSize * width,
              values.allSatisfy(\.isFinite) else {
            throw RLEnvironmentError.invalidConfiguration(
                "GEAR-SONIC history \(name) must contain \(batchSize * width) "
                    + "finite values")
        }
    }

    private static func write(
        _ source: ContiguousArray<Float>, into storage: inout [Float], width: Int,
        frame: Int, batchSize: Int
    ) {
        let offset = frame * batchSize * width
        storage.replaceSubrange(offset..<(offset + source.count), with: source)
    }

    private func appendField(
        _ storage: [Float], width: Int, environment: Int,
        to output: inout ContiguousArray<Float>
    ) {
        output.append(contentsOf: repeatElement(
            Float.zero, count: (frameCapacity - storedFrameCount) * width))
        let oldestFrame = storedFrameCount == frameCapacity ? nextWriteFrame : 0
        for logicalFrame in 0..<storedFrameCount {
            let frame = (oldestFrame + logicalFrame) % frameCapacity
            let offset = (frame * batchSize + environment) * width
            output.append(contentsOf: storage[offset..<(offset + width)])
        }
    }
}

/// Converts raw policy-order actions to actuator-order position targets.
public enum GEARSonicG1Control {
    public static func jointPositionTargets(
        rawPolicyActions: ContiguousArray<Float>,
        control: GEARSonicG1PolicyManifest.Control
    ) throws -> ContiguousArray<Float> {
        let jointCount = 29
        guard !rawPolicyActions.isEmpty,
              rawPolicyActions.count.isMultiple(of: jointCount),
              rawPolicyActions.allSatisfy(\.isFinite),
              control.actuatorToPolicy.count == jointCount,
              control.defaultJointPositions.count == jointCount,
              control.actionScale.count == jointCount else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid GEAR-SONIC action-to-target input")
        }
        let batch = rawPolicyActions.count / jointCount
        var targets = ContiguousArray<Float>(
            repeating: 0, count: rawPolicyActions.count)
        for environment in 0..<batch {
            for actuator in 0..<jointCount {
                let policy = control.actuatorToPolicy[actuator]
                targets[environment * jointCount + actuator]
                    = control.defaultJointPositions[actuator]
                    + rawPolicyActions[environment * jointCount + policy]
                    * control.actionScale[actuator]
            }
        }
        return targets
    }
}
