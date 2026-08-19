import Foundation
import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL
@testable import MLXRL

final class PushTPipelineValidationTests: XCTestCase {
    private var temporaryRoots: [URL] = []

    override func tearDownWithError() throws {
        for root in temporaryRoots {
            try? FileManager.default.removeItem(at: root)
        }
        temporaryRoots.removeAll(keepingCapacity: false)
        try super.tearDownWithError()
    }

    func testMalformedMetadataFailsBeforeMLXInitialization() throws {
        let dataset = try makeDatasetDirectory()
        try "not valid metadata".write(
            to: dataset.appendingPathComponent("meta.txt"),
            atomically: true,
            encoding: .utf8)

        assertInvalidConfiguration(containing: "metadata") {
            try train(dataset: dataset)
        }
    }

    func testTooShortHorizonFailsBeforeReadingPayloadsOrInitializingMLX() throws {
        let dataset = try makeDatasetDirectory()
        try "1 2 64".write(
            to: dataset.appendingPathComponent("meta.txt"),
            atomically: true,
            encoding: .utf8)

        assertInvalidConfiguration(containing: "at least 33") {
            try train(dataset: dataset)
        }
    }

    func testTruncatedPayloadsFailBeforeMLXInitialization() throws {
        let dataset = try makeDatasetDirectory()
        try "1 33 64".write(
            to: dataset.appendingPathComponent("meta.txt"),
            atomically: true,
            encoding: .utf8)
        for name in ["obs.bin", "act.bin", "state.bin"] {
            try Data([0]).write(to: dataset.appendingPathComponent(name))
        }

        assertInvalidConfiguration(
            containing: "file 'obs.bin' byte count is 1") {
            try train(dataset: dataset)
        }
    }

    func testWrongResolutionIsRejectedEvenWhenPayloadsMatchMetadata() throws {
        let dataset = try makeDatasetDirectory()
        try writeBytePerfectDataset(to: dataset, resolution: 32)

        assertInvalidConfiguration(
            containing: "requires 64x64 observations, got 32x32") {
            try train(dataset: dataset)
        }
    }

    func testNonFiniteActionPayloadFailsBeforeMLXInitialization() throws {
        let dataset = try makeDatasetDirectory()
        var actions = [Float](repeating: 0, count: 33 * 2)
        actions[17] = .nan
        try writeBytePerfectDataset(to: dataset, actions: actions)

        assertInvalidConfiguration(
            containing: "action value 17 is not finite") {
            try train(dataset: dataset)
        }
    }

    func testNonFiniteStatePayloadFailsBeforeMLXInitialization() throws {
        let dataset = try makeDatasetDirectory()
        var states = [Float](repeating: 0, count: 33 * 6)
        states[12] = .infinity
        try writeBytePerfectDataset(to: dataset, states: states)

        assertInvalidConfiguration(
            containing: "state value 12 is not finite") {
            try train(dataset: dataset)
        }
    }

    func testOutOfRangeNormalizedActionFailsBeforeMLXInitialization() throws {
        let dataset = try makeDatasetDirectory()
        var actions = [Float](repeating: 0, count: 33 * 2)
        actions[4] = 1.25
        try writeBytePerfectDataset(to: dataset, actions: actions)

        assertInvalidConfiguration(
            containing: "action value 4 is outside [-1.0, 1.0]") {
            try train(dataset: dataset)
        }
    }

    func testWorldModelConfigurationRestoresTrainedLatentDimension() throws {
        let model = try makeDatasetDirectory()
        let configuration = PushTWorldModelConfiguration(latentDimension: 32)
        try JSONEncoder().encode(configuration).write(
            to: model.appendingPathComponent(
                PushTPipeline.worldModelConfigurationFile))

        XCTAssertEqual(
            try PushTPipeline.worldModelConfiguration(modelPath: model.path),
            configuration)
        assertInvalidConfiguration(containing: "32, not 128") {
            _ = try PushTPipeline.worldModelConfiguration(
                modelPath: model.path, requestedLatent: 128)
        }
    }

    func testMalformedWorldModelConfigurationDoesNotFallBackSilently() throws {
        let model = try makeDatasetDirectory()
        try Data("{}".utf8).write(
            to: model.appendingPathComponent(
                PushTPipeline.worldModelConfigurationFile))

        assertInvalidConfiguration(containing: "contract is unreadable") {
            _ = try PushTPipeline.worldModelConfiguration(modelPath: model.path)
        }
    }

    private func makeDatasetDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "avbd-pusht-validation-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        temporaryRoots.append(root)
        return root
    }

    private func train(dataset: URL) throws {
        try PushTPipeline.train(
            dataPath: dataset.path,
            iters: 1,
            batch: 1,
            latent: 1,
            modelPath: dataset.appendingPathComponent("model").path)
    }

    private func writeBytePerfectDataset(
        to dataset: URL,
        environments: Int = 1,
        steps: Int = 33,
        resolution: Int = 64,
        actions: [Float]? = nil,
        states: [Float]? = nil
    ) throws {
        let actionValues = actions
            ?? [Float](repeating: 0, count: steps * environments * 2)
        let stateValues = states
            ?? [Float](repeating: 0, count: steps * environments * 6)
        XCTAssertEqual(actionValues.count, steps * environments * 2)
        XCTAssertEqual(stateValues.count, steps * environments * 6)

        try "\(environments) \(steps) \(resolution)".write(
            to: dataset.appendingPathComponent("meta.txt"),
            atomically: true,
            encoding: .utf8)
        try Data(
            repeating: 0,
            count: (steps + 1) * environments * resolution * resolution * 3
        ).write(to: dataset.appendingPathComponent("obs.bin"))
        try floatData(actionValues).write(
            to: dataset.appendingPathComponent("act.bin"))
        try floatData(stateValues).write(
            to: dataset.appendingPathComponent("state.bin"))
    }

    private func floatData(_ values: [Float]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }

    private func assertInvalidConfiguration(
        containing expectedText: String,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ operation: () throws -> Void
    ) {
        XCTAssertThrowsError(try operation(), file: file, line: line) { error in
            guard case let RLEnvironmentError.invalidConfiguration(message) = error else {
                return XCTFail(
                    "expected invalidConfiguration, got \(error)",
                    file: file,
                    line: line)
            }
            XCTAssertTrue(
                message.contains(expectedText),
                "expected '\(message)' to contain '\(expectedText)'",
                file: file,
                line: line)
        }
    }
}
