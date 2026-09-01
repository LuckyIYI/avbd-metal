import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL
@testable import MLXRL

final class RLBatchValidationTests: XCTestCase {
    private enum MalformedOutput {
        case resetObservation
        case stepReward
    }

    private final class MalformedTask: VectorizedRLTask {
        let spec = RLTaskSpec(
            id: "malformed-output-test-v0",
            numEnvironments: 2,
            observation: RLTensorSpec(name: "observation", shape: [2]),
            privilegedObservation: RLTensorSpec(
                name: "privileged_observation", shape: [1]),
            action: RLTensorSpec(name: "action", shape: [1]),
            maxEpisodeSteps: 8,
            simulationStep: 0.01,
            controlDecimation: 2)
        let malformedOutput: MalformedOutput

        init(malformedOutput: MalformedOutput) {
            self.malformedOutput = malformedOutput
        }

        func reset(
            environments: [Int]?,
            seed: UInt64,
            into observations: inout RLObservationBatch
        ) throws {
            _ = environments
            _ = seed
            if malformedOutput == .resetObservation {
                observations.policy[0] = .nan
            }
        }

        func step(
            actions: RLActionBatch,
            into result: inout RLStepBatch
        ) throws {
            _ = actions
            if malformedOutput == .stepReward {
                result.rewards[1] = .infinity
            }
        }
    }

    private var spec: RLTaskSpec {
        RLTaskSpec(
            id: "batch-validation-test-v0",
            numEnvironments: 2,
            observation: RLTensorSpec(name: "observation", shape: [2]),
            privilegedObservation: RLTensorSpec(
                name: "privileged_observation", shape: [1]),
            action: RLTensorSpec(name: "action", shape: [1]),
            maxEpisodeSteps: 8,
            simulationStep: 0.01,
            controlDecimation: 2)
    }

    private func assertInvalidConfiguration(
        containing expectedMessage: String,
        _ operation: () throws -> Void,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try operation(), file: file, line: line) { error in
            guard case RLEnvironmentError.invalidConfiguration(let message) = error else {
                return XCTFail(
                    "expected invalidConfiguration, got \(error)",
                    file: file,
                    line: line)
            }
            XCTAssertTrue(
                message.contains(expectedMessage),
                "'\(message)' does not contain '\(expectedMessage)'",
                file: file,
                line: line)
        }
    }

    func testValidObservationAndStepBatchesPassValidation() throws {
        let spec = spec
        var observations = RLObservationBatch(spec: spec)
        observations.policy = [0, 1, 2, 3]
        observations.privileged = [4, 5]
        try observations.validate(for: spec)

        var step = RLStepBatch(spec: spec)
        step.observations = observations
        step.rewards = [0.25, -0.5]
        step.terminated[0] = true
        step.truncated[1] = true
        step.hasFinalObservation = [true, true]
        step.finalObservations = [6, 7, 8, 9]
        step.metrics["diagnostic/finite"] = [10, 11]
        try step.validate(for: spec)
    }

    func testObservationBatchRejectsNonFinitePolicyAndPrivilegedValues() {
        let spec = spec
        var observations = RLObservationBatch(spec: spec)
        observations.policy[2] = .nan
        assertInvalidConfiguration(containing: "policy observation at flat index 2") {
            try observations.validate(for: spec)
        }

        observations.policy[2] = 0
        observations.privileged[1] = .infinity
        assertInvalidConfiguration(
            containing: "privileged observation at flat index 1"
        ) {
            try observations.validate(for: spec)
        }
    }

    func testStepBatchRejectsNonFiniteRewardFinalObservationAndMetric() {
        let spec = spec
        var step = RLStepBatch(spec: spec)
        step.rewards[1] = .nan
        assertInvalidConfiguration(containing: "reward at environment 1") {
            try step.validate(for: spec)
        }

        step.rewards[1] = 0
        step.finalObservations[3] = .infinity
        assertInvalidConfiguration(containing: "final observation at flat index 3") {
            try step.validate(for: spec)
        }

        step.finalObservations[3] = 0
        step.metrics["diagnostic/contact"] = [0, .nan]
        assertInvalidConfiguration(
            containing: "metric 'diagnostic/contact' at environment 1"
        ) {
            try step.validate(for: spec)
        }
    }

    func testStepBatchRejectsMetricWithWrongRowCount() {
        let spec = spec
        var step = RLStepBatch(spec: spec)
        step.metrics["diagnostic/contact"] = [1]

        assertInvalidConfiguration(
            containing: "metric 'diagnostic/contact' has 1 rows; expected 2"
        ) {
            try step.validate(for: spec)
        }
    }

    func testStepBatchAllowsCoincidentTerminationAndTruncation() throws {
        let spec = spec
        var step = RLStepBatch(spec: spec)
        step.terminated[1] = true
        step.truncated[1] = true
        try step.validate(for: spec)
    }

    func testStepBatchRejectsTruncationWithoutFinalObservation() {
        let spec = spec
        var step = RLStepBatch(spec: spec)
        step.truncated[0] = true

        assertInvalidConfiguration(
            containing: "truncated environment 0 has no final observation"
        ) {
            try step.validate(for: spec)
        }
    }

    func testNonAutoResetTruncationUsesReturnedObservation() throws {
        var spec = spec
        spec.autoReset = false
        var step = RLStepBatch(spec: spec)
        step.truncated[0] = true
        try step.validate(for: spec)
    }

    func testStepBatchRejectsFinalObservationWithoutDoneSignal() {
        let spec = spec
        var step = RLStepBatch(spec: spec)
        step.hasFinalObservation[0] = true

        assertInvalidConfiguration(
            containing: "environment 0 publishes a final observation without ending"
        ) {
            try step.validate(for: spec)
        }
    }

    func testConvenienceResetValidatesTaskOutput() {
        let task = MalformedTask(malformedOutput: .resetObservation)

        assertInvalidConfiguration(containing: "policy observation at flat index 0") {
            _ = try task.reset(seed: 123)
        }
    }

    func testConvenienceStepValidatesTaskOutput() {
        let task = MalformedTask(malformedOutput: .stepReward)

        assertInvalidConfiguration(containing: "reward at environment 1") {
            _ = try task.step(actions: RLActionBatch(spec: task.spec))
        }
    }

    func testPPOBootstrapsOnlyPureTruncations() {
        XCTAssertFalse(VectorPPOTrainer.shouldBootstrapFinalObservation(
            terminated: false, truncated: false))
        XCTAssertFalse(VectorPPOTrainer.shouldBootstrapFinalObservation(
            terminated: true, truncated: false))
        XCTAssertTrue(VectorPPOTrainer.shouldBootstrapFinalObservation(
            terminated: false, truncated: true))
        XCTAssertFalse(VectorPPOTrainer.shouldBootstrapFinalObservation(
            terminated: true, truncated: true))
    }
}
