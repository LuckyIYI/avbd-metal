import Dispatch
import Foundation
import XCTest
@testable import AVBDCore
@testable import AVBDLearn

final class VectorRLAlgorithmLifecycleTests: XCTestCase {
    private struct StubConfiguration: Codable, Equatable, Sendable {
        var name: String
        var steps: Int
    }

    /// Same representation, deliberately different nominal type.
    private struct LookalikeConfiguration: Codable, Sendable {
        var name: String
        var steps: Int
    }

    private final class ConfiguredAlgorithm: VectorRLAlgorithm {
        let id: String
        let configuration: StubConfiguration

        init(id: String, configuration: StubConfiguration) {
            self.id = id
            self.configuration = configuration
        }

        func train(
            task: any VectorizedRLTask,
            outputDirectory: String
        ) throws {}
    }

    private final class StubTask: VectorizedRLTask {
        let spec = RLTaskSpec(
            id: "algorithm-lifecycle-test-v0",
            numEnvironments: 1,
            observation: RLTensorSpec(name: "observation", shape: [1]),
            action: RLTensorSpec(name: "action", shape: [1]),
            maxEpisodeSteps: 2,
            simulationStep: 0.01,
            controlDecimation: 1)

        func reset(
            environments: [Int]?,
            seed: UInt64,
            into observations: inout RLObservationBatch
        ) throws {}

        func step(
            actions: RLActionBatch,
            into result: inout RLStepBatch
        ) throws {}
    }

    private final class RequestRecordingAlgorithm: VectorRLTrainingLifecycle {
        let id = "recording"
        private(set) var taskIdentity: ObjectIdentifier?
        private(set) var outputDirectory: String?
        private(set) var continuation: VectorRLTrainingContinuation?

        func train(
            task: any VectorizedRLTask,
            outputDirectory: String
        ) throws {
            XCTFail("the lifecycle capability should receive the request")
        }

        func trainLifecycle(_ request: VectorRLTrainingRequest) throws
            -> VectorRLTrainingResult
        {
            taskIdentity = ObjectIdentifier(request.task)
            outputDirectory = request.outputDirectory
            continuation = request.continuation
            return VectorRLTrainingResult(
                algorithmID: id,
                outputDirectory: request.outputDirectory,
                continuation: request.continuation)
        }
    }

    private final class ConcurrentResults: @unchecked Sendable {
        private let lock = NSLock()
        private var storedAlgorithms = [ConfiguredAlgorithm]()
        private var storedErrors = [Error]()

        func append(_ algorithm: ConfiguredAlgorithm) {
            lock.lock()
            storedAlgorithms.append(algorithm)
            lock.unlock()
        }

        func append(_ error: Error) {
            lock.lock()
            storedErrors.append(error)
            lock.unlock()
        }

        var algorithms: [ConfiguredAlgorithm] {
            lock.lock()
            defer { lock.unlock() }
            return storedAlgorithms
        }

        var errors: [Error] {
            lock.lock()
            defer { lock.unlock() }
            return storedErrors
        }
    }

    private var defaults: StubConfiguration {
        StubConfiguration(name: "default", steps: 10)
    }

    private func definition(
        id: String = "typed"
    ) -> VectorRLAlgorithmDefinition<StubConfiguration> {
        let defaults = defaults
        return VectorRLAlgorithmDefinition(
            id: id,
            defaultConfiguration: { defaults },
            trainerFactory: {
                ConfiguredAlgorithm(id: id, configuration: $0)
            })
    }

    func testTypedDefaultAndConfiguredValuesReachFactoryExactly() throws {
        let registry = VectorRLAlgorithmRegistry()
        try registry.register(definition())

        let first = try XCTUnwrap(
            try registry.make("typed") as? ConfiguredAlgorithm)
        let second = try XCTUnwrap(
            try registry.make("typed") as? ConfiguredAlgorithm)
        XCTAssertEqual(first.configuration, defaults)
        XCTAssertEqual(second.configuration, defaults)
        XCTAssertNotEqual(ObjectIdentifier(first), ObjectIdentifier(second))

        let custom = StubConfiguration(name: "custom", steps: 37)
        let configured = try XCTUnwrap(
            try registry.make("typed", configuration: custom)
                as? ConfiguredAlgorithm)
        XCTAssertEqual(configured.configuration, custom)
    }

    func testConfiguredMakeRequiresExactNominalType() throws {
        let registry = VectorRLAlgorithmRegistry()
        try registry.register(definition())
        let lookalike = LookalikeConfiguration(name: "custom", steps: 37)

        XCTAssertThrowsError(
            try registry.make("typed", configuration: lookalike)
        ) { error in
            XCTAssertEqual(
                error as? VectorRLAlgorithmConfigurationError,
                .typeMismatch(
                    algorithm: "typed",
                    expected: String(reflecting: StubConfiguration.self),
                    provided: String(reflecting: LookalikeConfiguration.self)))
        }
    }

    func testConfiguredFactoryRunsOutsideLockAndIsConcurrent() throws {
        let registry = VectorRLAlgorithmRegistry()
        let defaults = defaults
        let definition = VectorRLAlgorithmDefinition(
            id: "typed",
            defaultConfiguration: { defaults },
            trainerFactory: { configuration in
                XCTAssertEqual(registry.algorithmIDs, ["typed"])
                return ConfiguredAlgorithm(
                    id: "typed", configuration: configuration)
            })
        try registry.register(definition)

        let results = ConcurrentResults()
        DispatchQueue.concurrentPerform(iterations: 32) { index in
            do {
                let configuration = StubConfiguration(
                    name: "configured-\(index)", steps: index)
                let algorithm = try registry.make(
                    "typed", configuration: configuration)
                guard let configured = algorithm as? ConfiguredAlgorithm else {
                    return XCTFail("factory returned the wrong algorithm type")
                }
                results.append(configured)
            } catch {
                results.append(error)
            }
        }

        XCTAssertTrue(results.errors.isEmpty)
        XCTAssertEqual(results.algorithms.count, 32)
        XCTAssertEqual(
            Set(results.algorithms.map(ObjectIdentifier.init)).count, 32)
        XCTAssertEqual(
            Set(results.algorithms.map(\.configuration.steps)),
            Set(0..<32))
    }

    func testFixedFactoryRejectsConfiguredConstruction() throws {
        let registry = VectorRLAlgorithmRegistry()
        let defaults = defaults
        try registry.register("fixed") {
            ConfiguredAlgorithm(id: "fixed", configuration: defaults)
        }

        XCTAssertThrowsError(try registry.make(
            "fixed", configuration: defaults)) {
            XCTAssertEqual(
                $0 as? VectorRLAlgorithmConfigurationError,
                .unsupported("fixed"))
        }
    }

    func testTrainingRequestDispatchesThroughErasedAlgorithm() throws {
        let algorithm = RequestRecordingAlgorithm()
        let erased: any VectorRLAlgorithm = algorithm
        let task = StubTask()
        let request = VectorRLTrainingRequest(
            task: task,
            outputDirectory: "/tmp/algorithm-lifecycle",
            continuation: .warmResumeLatest)

        let result = try erased.train(request)

        XCTAssertEqual(algorithm.taskIdentity, ObjectIdentifier(task))
        XCTAssertEqual(algorithm.outputDirectory, request.outputDirectory)
        XCTAssertEqual(algorithm.continuation, .warmResumeLatest)
        XCTAssertEqual(result, VectorRLTrainingResult(
            algorithmID: "recording",
            outputDirectory: request.outputDirectory,
            continuation: .warmResumeLatest))
    }

    func testLegacyBridgeNeverSilentlyRestartsWarmRequest() throws {
        let algorithm: any VectorRLAlgorithm = ConfiguredAlgorithm(
            id: "legacy", configuration: defaults)
        let task = StubTask()
        _ = try algorithm.train(VectorRLTrainingRequest(
            task: task, outputDirectory: "/tmp/fresh"))

        XCTAssertThrowsError(try algorithm.train(VectorRLTrainingRequest(
            task: task,
            outputDirectory: "/tmp/resume",
            continuation: .warmResumeLatest))) {
            XCTAssertEqual(
                $0 as? VectorRLTrainingError,
                .unsupportedContinuation(
                    algorithmID: "legacy",
                    continuation: .warmResumeLatest))
        }
    }

    func testBuiltInPPOPreservesCompleteConfigurationValues() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]

        let defaultTrainer = try XCTUnwrap(
            try VectorRLAlgorithmRegistry.builtIn.make("ppo")
                as? VectorPPOTrainer)
        XCTAssertEqual(
            try encoder.encode(defaultTrainer.configuration),
            try encoder.encode(VectorPPOConfig()))

        var custom = VectorPPOConfig()
        custom.updates = 17
        custom.learningRate = 7e-5
        custom.initialActionStd = 0.42
        custom.checkpointInterval = 3
        let configuredTrainer = try XCTUnwrap(
            try VectorRLAlgorithmRegistry.builtIn.make(
                "ppo", configuration: custom) as? VectorPPOTrainer)
        XCTAssertEqual(
            try encoder.encode(configuredTrainer.configuration),
            try encoder.encode(custom))
    }
}
