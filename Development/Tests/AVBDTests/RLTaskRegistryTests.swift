import Dispatch
import Foundation
import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL

final class RLTaskRegistryTests: XCTestCase {
    private final class StubTask: VectorizedRLTask {
        let spec: RLTaskSpec

        init(id: String, numEnvironments: Int) {
            spec = RLTaskSpec(
                id: id,
                numEnvironments: numEnvironments,
                observation: RLTensorSpec(name: "observation", shape: [1]),
                action: RLTensorSpec(name: "action", shape: [1]),
                maxEpisodeSteps: 4,
                simulationStep: 0.01,
                controlDecimation: 1)
        }

        func reset(
            environments: [Int]?, seed: UInt64,
            into observations: inout RLObservationBatch
        ) throws {
            _ = environments
            _ = seed
            try observations.validate(for: spec)
        }

        func step(
            actions: RLActionBatch, into result: inout RLStepBatch
        ) throws {
            try actions.validate(for: spec)
            try result.validate(for: spec)
        }
    }

    private func assertInvalidConfiguration(
        containing fragment: String,
        _ operation: () throws -> Void,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try operation(), file: file, line: line) { error in
            guard case RLEnvironmentError.invalidConfiguration(let message) = error
            else {
                return XCTFail("unexpected error: \(error)", file: file, line: line)
            }
            XCTAssertTrue(message.contains(fragment), message,
                          file: file, line: line)
        }
    }

    func testRegistrationAndLookupAreDeterministic() throws {
        let registry = RLTaskRegistry()
        try registry.register("zeta") {
            StubTask(id: "zeta", numEnvironments: $0.numEnvironments)
        }
        try registry.register("alpha") {
            StubTask(id: "alpha", numEnvironments: $0.numEnvironments)
        }

        XCTAssertEqual(registry.taskIDs, ["alpha", "zeta"])
        let task = try registry.make(
            "alpha", configuration: .init(numEnvironments: 3))
        XCTAssertEqual(task.spec.id, "alpha")
        XCTAssertEqual(task.spec.numEnvironments, 3)
    }

    func testInvalidIdentifiersFailBeforeRegistrationOrLookup() {
        let registry = RLTaskRegistry()
        for id in ["", " ", "humanoid task", "task\n"] {
            assertInvalidConfiguration(containing: "invalid RL task identifier") {
                try registry.register(id) {
                    StubTask(id: id, numEnvironments: $0.numEnvironments)
                }
            }
            assertInvalidConfiguration(containing: "invalid RL task identifier") {
                _ = try registry.make(
                    id, configuration: .init(numEnvironments: 1))
            }
        }
    }

    func testFactoryMustPreserveRegisteredIdentityAndBatchSize() throws {
        let wrongID = RLTaskRegistry()
        try wrongID.register("expected") {
            StubTask(id: "different", numEnvironments: $0.numEnvironments)
        }
        assertInvalidConfiguration(containing: "produced task 'different'") {
            _ = try wrongID.make(
                "expected", configuration: .init(numEnvironments: 2))
        }

        let wrongBatch = RLTaskRegistry()
        try wrongBatch.register("expected") { _ in
            StubTask(id: "expected", numEnvironments: 1)
        }
        assertInvalidConfiguration(containing: "1 environments; requested 2") {
            _ = try wrongBatch.make(
                "expected", configuration: .init(numEnvironments: 2))
        }
    }

    func testFactoryCanReenterRegistry() throws {
        let registry = RLTaskRegistry()
        try registry.register("reentrant") { configuration in
            XCTAssertEqual(registry.taskIDs, ["reentrant"])
            return StubTask(
                id: "reentrant",
                numEnvironments: configuration.numEnvironments)
        }

        _ = try registry.make(
            "reentrant", configuration: .init(numEnvironments: 1))
    }

    func testConcurrentDuplicateRegistrationIsAtomic() {
        let registry = RLTaskRegistry()
        let resultLock = NSLock()
        var successes = 0
        var duplicateErrors = 0

        DispatchQueue.concurrentPerform(iterations: 64) { _ in
            do {
                try registry.register("task") {
                    StubTask(id: "task", numEnvironments: $0.numEnvironments)
                }
                resultLock.lock()
                successes += 1
                resultLock.unlock()
            } catch RLEnvironmentError.duplicateTask("task") {
                resultLock.lock()
                duplicateErrors += 1
                resultLock.unlock()
            } catch {
                XCTFail("unexpected error: \(error)")
            }
        }

        XCTAssertEqual(successes, 1)
        XCTAssertEqual(duplicateErrors, 63)
        XCTAssertEqual(registry.taskIDs, ["task"])
    }

    func testConcurrentLookupReturnsIndependentTasks() throws {
        let registry = RLTaskRegistry()
        try registry.register("task") {
            StubTask(id: "task", numEnvironments: $0.numEnvironments)
        }
        let resultLock = NSLock()
        var tasks: [any VectorizedRLTask] = []
        var errors: [Error] = []

        DispatchQueue.concurrentPerform(iterations: 64) { _ in
            do {
                let task = try registry.make(
                    "task", configuration: .init(numEnvironments: 2))
                resultLock.lock()
                tasks.append(task)
                resultLock.unlock()
            } catch {
                resultLock.lock()
                errors.append(error)
                resultLock.unlock()
            }
        }

        XCTAssertTrue(errors.isEmpty)
        XCTAssertEqual(tasks.count, 64)
        XCTAssertEqual(Set(tasks.map(ObjectIdentifier.init)).count, 64)
    }
}
