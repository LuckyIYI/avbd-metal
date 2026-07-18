import Foundation
import AVBDCore
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

public struct PushTPhysicalFlowRejectedTeacher: Codable, Sendable {
    public var seed: UInt64
    public var teacherLoss: Float
    public var robustReplaySuccessFraction: Float
    public var infrastructureGatePassed: Bool
    public var dynamicStateReconstructionGatePassed: Bool
    public var robustReplayGatePassed: Bool
}

public struct PushTPhysicalFlowDataset: Codable, Sendable {
    public static let schemaVersion = 1

    public var schemaVersion: Int
    public var requestedSamples: Int
    public var seedStart: UInt64
    public var seedStride: UInt64
    public var inputDimension: Int
    public var outputDimension: Int
    public var samples: [PushTPhysicalFlowTeacherSample]
    public var rejected: [PushTPhysicalFlowRejectedTeacher]
}

public enum PushTPhysicalFlowDatasetBuilder {
    public static func collect(
        requestedSamples: Int,
        seedStart: UInt64,
        seedStride: UInt64 = 104_729,
        experimentConfiguration: PushTPhysicalFlowConfiguration
    ) throws -> PushTPhysicalFlowDataset {
        guard requestedSamples > 0, seedStride > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "physical-flow dataset requires positive sample count and seed stride")
        }
        var samples = [PushTPhysicalFlowTeacherSample]()
        var rejected = [PushTPhysicalFlowRejectedTeacher]()
        for index in 0..<requestedSamples {
            let seed = seedStart &+ UInt64(index) &* seedStride
            var configuration = experimentConfiguration
            configuration.seed = seed
            let report = try PushTPhysicalFlowExperiment.run(
                configuration: configuration)
            if report.teacherSample.goGatePassed {
                samples.append(report.teacherSample)
            } else {
                rejected.append(.init(
                    seed: seed,
                    teacherLoss: report.optimizedSpline.loss,
                    robustReplaySuccessFraction:
                        report.robustReplaySuccessFraction,
                    infrastructureGatePassed:
                        report.infrastructureGatePassed,
                    dynamicStateReconstructionGatePassed:
                        report.dynamicStateReconstructionGatePassed,
                    robustReplayGatePassed: report.robustReplayGatePassed))
            }
            print(String(format:
                "flow teacher %3d/%3d seed %llu  accepted %d  loss %.5g  robust %.0f%%",
                index + 1, requestedSamples, seed,
                report.teacherSample.goGatePassed ? 1 : 0,
                report.optimizedSpline.loss,
                100 * report.robustReplaySuccessFraction))
        }
        let outputDimension = 2 * (
            experimentConfiguration.controlPointCount - 2)
        precondition(samples.allSatisfy {
            $0.input.count == PushTPhysicalFlowProposalContext.inputDimension
                && $0.canonicalInternalControlPointsXY.count == outputDimension
        })
        return PushTPhysicalFlowDataset(
            schemaVersion: PushTPhysicalFlowDataset.schemaVersion,
            requestedSamples: requestedSamples,
            seedStart: seedStart,
            seedStride: seedStride,
            inputDimension: PushTPhysicalFlowProposalContext.inputDimension,
            outputDimension: outputDimension,
            samples: samples,
            rejected: rejected)
    }
}

public final class PushTPhysicalFlowProposalNetwork: Module {
    @ModuleInfo public var inputLayer: Linear
    @ModuleInfo public var hiddenLayer1: Linear
    @ModuleInfo public var hiddenLayer2: Linear
    @ModuleInfo public var outputLayer: Linear

    public let inputDimension: Int
    public let outputDimension: Int
    public let hiddenDimension: Int

    public init(inputDimension: Int, outputDimension: Int,
                hiddenDimension: Int = 128) {
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        self.hiddenDimension = hiddenDimension
        inputLayer = Linear(inputDimension, hiddenDimension)
        hiddenLayer1 = Linear(hiddenDimension, hiddenDimension)
        hiddenLayer2 = Linear(hiddenDimension, hiddenDimension)
        outputLayer = Linear(hiddenDimension, outputDimension)
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        outputLayer(gelu(hiddenLayer2(gelu(
            hiddenLayer1(gelu(inputLayer(input)))))))
    }
}

public struct PushTPhysicalFlowProposalMetadata: Codable, Sendable {
    public static let schemaVersion = 3
    public static let outputRepresentation = "geometric-residual-v1"

    public var schemaVersion: Int
    public var inputDimension: Int
    public var outputDimension: Int
    public var outputRepresentation: String
    public var hiddenDimension: Int
    public var inputMean: [Float]
    public var inputStandardDeviation: [Float]
    public var outputMean: [Float]
    public var outputStandardDeviation: [Float]
    public var trainingSeeds: [UInt64]
    public var validationSeeds: [UInt64]
    public var epochs: Int
    public var bestEpoch: Int
    public var trainingLoss: Float
    public var validationLoss: Float
}

public struct PushTPhysicalFlowProposalTrainingReport: Codable, Sendable {
    public var trainingRows: Int
    public var validationRows: Int
    public var initialTrainingLoss: Float
    public var finalTrainingLoss: Float
    public var initialValidationLoss: Float
    public var finalValidationLoss: Float
    public var bestEpoch: Int
    public var checkpointDirectory: String
}

public enum PushTPhysicalFlowProposalTrainer {
    public static func train(
        dataset: PushTPhysicalFlowDataset,
        epochs: Int = 2_000,
        hiddenDimension: Int = 128,
        learningRate: Float = 3e-4,
        seed: UInt64 = 1,
        checkpointDirectory: String
    ) throws -> PushTPhysicalFlowProposalTrainingReport {
        guard dataset.schemaVersion == PushTPhysicalFlowDataset.schemaVersion,
              dataset.inputDimension ==
                PushTPhysicalFlowProposalContext.inputDimension,
              dataset.samples.count >= 5,
              epochs > 0,
              hiddenDimension > 0,
              learningRate.isFinite, learningRate > 0 else {
            throw RLEnvironmentError.invalidConfiguration(
                "invalid physical-flow proposal training configuration or dataset")
        }
        let ordered = dataset.samples.sorted { $0.seed < $1.seed }
        let validation = ordered.enumerated().compactMap {
            $0.offset % 5 == 0 ? $0.element : nil
        }
        let training = ordered.enumerated().compactMap {
            $0.offset % 5 == 0 ? nil : $0.element
        }
        let inputMean = columnMean(training.map(\.input),
                                   dimension: dataset.inputDimension)
        let inputDeviation = columnStandardDeviation(
            training.map(\.input), mean: inputMean)
        let controlPointCount = dataset.outputDimension / 2 + 2
        func residualTarget(
            _ sample: PushTPhysicalFlowTeacherSample
        ) -> [Float] {
            let geometric = PushTPhysicalFlowProposalContext(
                input: sample.input,
                sourceBlockCenter: .zero,
                sourceBlockYaw: 0,
                sourceCommandedTipTarget: .zero,
                targetCommandedTipTarget: .zero)
                .canonicalGeometricInternalControlPointsXY(
                    controlPointCount: controlPointCount)
            return sample.canonicalInternalControlPointsXY.indices.map {
                sample.canonicalInternalControlPointsXY[$0] - geometric[$0]
            }
        }
        let outputMean = columnMean(
            training.map(residualTarget),
            dimension: dataset.outputDimension)
        let outputDeviation = columnStandardDeviation(
            training.map(residualTarget), mean: outputMean)

        func arrays(_ rows: [PushTPhysicalFlowTeacherSample])
            -> (MLXArray, MLXArray) {
            let x = rows.flatMap { normalized(
                $0.input, mean: inputMean, deviation: inputDeviation) }
            let y = rows.flatMap { normalized(
                residualTarget($0),
                mean: outputMean, deviation: outputDeviation) }
            return (
                MLXArray(x).reshaped([rows.count, dataset.inputDimension]),
                MLXArray(y).reshaped([rows.count, dataset.outputDimension]))
        }
        let (trainingInput, trainingTarget) = arrays(training)
        let (validationInput, validationTarget) = arrays(validation)
        MLXRandom.seed(seed)
        let model = PushTPhysicalFlowProposalNetwork(
            inputDimension: dataset.inputDimension,
            outputDimension: dataset.outputDimension,
            hiddenDimension: hiddenDimension)
        let optimizer = AdamW(learningRate: learningRate)
        let lossAndGradient = valueAndGrad(model: model) {
            (model: PushTPhysicalFlowProposalNetwork,
             arguments: [MLXArray]) -> [MLXArray] in
            [mean((model(arguments[0]) - arguments[1]).square())]
        }
        func loss(_ input: MLXArray, _ target: MLXArray) -> Float {
            let value = mean((model(input) - target).square())
            eval(value)
            return value.item(Float.self)
        }
        let initialTrainingLoss = loss(trainingInput, trainingTarget)
        let initialValidationLoss = loss(validationInput, validationTarget)
        var bestEpoch = 0
        var bestValidationLoss = initialValidationLoss
        var bestWeightsData = try MLX.saveToData(arrays:
            Dictionary(uniqueKeysWithValues:
                model.parameters().flattened().map { ($0.0, $0.1) }))
        var completedEpochs = 0
        let patience = max(100, min(500, epochs / 5))
        for epoch in 1...epochs {
            let (losses, gradients) = lossAndGradient(
                model, [trainingInput, trainingTarget])
            optimizer.update(model: model, gradients: gradients)
            eval(model, optimizer)
            completedEpochs = epoch
            if epoch == 1 || epoch % 10 == 0 || epoch == epochs {
                let validationLoss = loss(validationInput, validationTarget)
                if validationLoss < bestValidationLoss * (1 - 1e-5) {
                    bestValidationLoss = validationLoss
                    bestEpoch = epoch
                    bestWeightsData = try MLX.saveToData(arrays:
                        Dictionary(uniqueKeysWithValues:
                            model.parameters().flattened().map { ($0.0, $0.1) }))
                }
                if epoch == 1 || epoch % 100 == 0 || epoch == epochs {
                    print(String(format:
                        "flow proposal epoch %5d  train %.6f  validation %.6f  best %.6f@%d",
                        epoch, losses[0].item(Float.self), validationLoss,
                        bestValidationLoss, bestEpoch))
                }
                if epoch - bestEpoch >= patience, epoch >= 100 { break }
            }
        }
        let bestWeights = try loadArrays(data: bestWeightsData)
        try model.update(
            parameters: ModuleParameters.unflattened(bestWeights),
            verify: [.all])
        eval(model)
        let finalTrainingLoss = loss(trainingInput, trainingTarget)
        let finalValidationLoss = loss(validationInput, validationTarget)
        let metadata = PushTPhysicalFlowProposalMetadata(
            schemaVersion: PushTPhysicalFlowProposalMetadata.schemaVersion,
            inputDimension: dataset.inputDimension,
            outputDimension: dataset.outputDimension,
            outputRepresentation:
                PushTPhysicalFlowProposalMetadata.outputRepresentation,
            hiddenDimension: hiddenDimension,
            inputMean: inputMean,
            inputStandardDeviation: inputDeviation,
            outputMean: outputMean,
            outputStandardDeviation: outputDeviation,
            trainingSeeds: training.map(\.seed),
            validationSeeds: validation.map(\.seed),
            epochs: completedEpochs,
            bestEpoch: bestEpoch,
            trainingLoss: finalTrainingLoss,
            validationLoss: finalValidationLoss)
        try FileManager.default.createDirectory(
            atPath: checkpointDirectory,
            withIntermediateDirectories: true)
        try bestWeightsData.write(
            to: URL(fileURLWithPath:
                "\(checkpointDirectory)/proposal.safetensors"),
            options: .atomic)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(
            to: URL(fileURLWithPath:
                "\(checkpointDirectory)/metadata.json"),
            options: .atomic)
        return PushTPhysicalFlowProposalTrainingReport(
            trainingRows: training.count,
            validationRows: validation.count,
            initialTrainingLoss: initialTrainingLoss,
            finalTrainingLoss: finalTrainingLoss,
            initialValidationLoss: initialValidationLoss,
            finalValidationLoss: finalValidationLoss,
            bestEpoch: bestEpoch,
            checkpointDirectory: checkpointDirectory)
    }

    private static func columnMean(
        _ rows: [[Float]], dimension: Int
    ) -> [Float] {
        var result = [Float](repeating: 0, count: dimension)
        for row in rows {
            precondition(row.count == dimension)
            for index in result.indices { result[index] += row[index] }
        }
        for index in result.indices { result[index] /= Float(rows.count) }
        return result
    }

    private static func columnStandardDeviation(
        _ rows: [[Float]], mean: [Float]
    ) -> [Float] {
        var result = [Float](repeating: 0, count: mean.count)
        for row in rows {
            for index in result.indices {
                let delta = row[index] - mean[index]
                result[index] += delta * delta
            }
        }
        for index in result.indices {
            result[index] = max(
                sqrt(result[index] / Float(rows.count)), 1e-4)
        }
        return result
    }

    private static func normalized(
        _ values: [Float], mean: [Float], deviation: [Float]
    ) -> [Float] {
        precondition(values.count == mean.count && mean.count == deviation.count)
        return values.indices.map {
            (values[$0] - mean[$0]) / deviation[$0]
        }
    }
}

public final class PushTPhysicalFlowMLXProposal {
    public let metadata: PushTPhysicalFlowProposalMetadata
    private let model: PushTPhysicalFlowProposalNetwork

    public init(checkpointDirectory: String) throws {
        metadata = try JSONDecoder().decode(
            PushTPhysicalFlowProposalMetadata.self,
            from: Data(contentsOf: URL(fileURLWithPath:
                "\(checkpointDirectory)/metadata.json")))
        guard metadata.schemaVersion ==
                PushTPhysicalFlowProposalMetadata.schemaVersion,
              metadata.outputRepresentation ==
                PushTPhysicalFlowProposalMetadata.outputRepresentation,
              metadata.inputDimension ==
                PushTPhysicalFlowProposalContext.inputDimension else {
            throw RLEnvironmentError.invalidConfiguration(
                "incompatible physical-flow proposal checkpoint")
        }
        model = PushTPhysicalFlowProposalNetwork(
            inputDimension: metadata.inputDimension,
            outputDimension: metadata.outputDimension,
            hiddenDimension: metadata.hiddenDimension)
        let weights = try loadArrays(url: URL(fileURLWithPath:
            "\(checkpointDirectory)/proposal.safetensors"))
        try model.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
    }

    public func callAsFunction(
        _ context: PushTPhysicalFlowProposalContext
    ) throws -> [Float] {
        guard context.input.count == metadata.inputDimension else {
            throw RLEnvironmentError.invalidConfiguration(
                "physical-flow proposal input dimension mismatch")
        }
        let normalizedInput = context.input.indices.map {
            (context.input[$0] - metadata.inputMean[$0])
                / metadata.inputStandardDeviation[$0]
        }
        let output = model(MLXArray(normalizedInput).reshaped(
            [1, metadata.inputDimension]))
        eval(output)
        let normalizedOutput = output.asArray(Float.self)
        let residual = normalizedOutput.indices.map {
            normalizedOutput[$0] * metadata.outputStandardDeviation[$0]
                + metadata.outputMean[$0]
        }
        let geometric = context.canonicalGeometricInternalControlPointsXY(
            controlPointCount: metadata.outputDimension / 2 + 2)
        return residual.indices.map { geometric[$0] + residual[$0] }
    }
}

/// Nonparametric coverage baseline. It deliberately returns one verified mode
/// instead of averaging incompatible contact-side/timing solutions.
public final class PushTPhysicalFlowNearestNeighborProposal {
    public private(set) var lastNeighborSeed: UInt64?
    public private(set) var lastNormalizedDistance: Float?

    private let samples: [PushTPhysicalFlowTeacherSample]
    private let inputDeviation: [Float]

    public init(dataset: PushTPhysicalFlowDataset) throws {
        guard !dataset.samples.isEmpty,
              dataset.inputDimension ==
                PushTPhysicalFlowProposalContext.inputDimension else {
            throw RLEnvironmentError.invalidConfiguration(
                "nearest-flow proposal requires a nonempty compatible dataset")
        }
        samples = dataset.samples
        var mean = [Float](repeating: 0, count: dataset.inputDimension)
        for sample in samples {
            for index in mean.indices { mean[index] += sample.input[index] }
        }
        for index in mean.indices { mean[index] /= Float(samples.count) }
        var deviation = [Float](repeating: 0, count: mean.count)
        for sample in samples {
            for index in deviation.indices {
                let delta = sample.input[index] - mean[index]
                deviation[index] += delta * delta
            }
        }
        for index in deviation.indices {
            deviation[index] = max(
                sqrt(deviation[index] / Float(samples.count)), 1e-4)
        }
        inputDeviation = deviation
    }

    public func callAsFunction(
        _ context: PushTPhysicalFlowProposalContext
    ) -> [Float] {
        let nearest = samples.min { lhs, rhs in
            distance(lhs.input, context.input)
                < distance(rhs.input, context.input)
        }!
        let d = distance(nearest.input, context.input)
        lastNeighborSeed = nearest.seed
        lastNormalizedDistance = sqrt(d / Float(context.input.count))
        let controlPointCount = nearest.canonicalInternalControlPointsXY.count
            / 2 + 2
        let teacherContext = PushTPhysicalFlowProposalContext(
            input: nearest.input,
            sourceBlockCenter: .zero,
            sourceBlockYaw: 0,
            sourceCommandedTipTarget: .zero,
            targetCommandedTipTarget: .zero)
        let teacherGeometric = teacherContext
            .canonicalGeometricInternalControlPointsXY(
                controlPointCount: controlPointCount)
        let queryGeometric = context
            .canonicalGeometricInternalControlPointsXY(
                controlPointCount: controlPointCount)
        return nearest.canonicalInternalControlPointsXY.indices.map {
            queryGeometric[$0]
                + nearest.canonicalInternalControlPointsXY[$0]
                - teacherGeometric[$0]
        }
    }

    private func distance(_ lhs: [Float], _ rhs: [Float]) -> Float {
        lhs.indices.reduce(0) { result, index in
            let delta = (lhs[index] - rhs[index]) / inputDeviation[index]
            return result + delta * delta
        }
    }
}
