import Dispatch
import Foundation
import XCTest
@testable import AVBDCore
@testable import AVBDLearn

final class VectorRLAlgorithmRegistryTests: XCTestCase {
    private final class StubAlgorithm: VectorRLAlgorithm {
        let id: String

        init(id: String) {
            self.id = id
        }

        func train(
            task: any VectorizedRLTask,
            outputDirectory: String
        ) throws {}
    }

    func testRegistrationAndLookupAreDeterministic() throws {
        let registry = VectorRLAlgorithmRegistry()
        try registry.register("zeta") { StubAlgorithm(id: "zeta") }
        try registry.register("alpha") { StubAlgorithm(id: "alpha") }

        XCTAssertEqual(registry.algorithmIDs, ["alpha", "zeta"])
        XCTAssertEqual(try registry.make("alpha").id, "alpha")
    }

    func testDuplicateRegistrationReturnsTypedErrorAndKeepsFirstFactory() throws {
        let registry = VectorRLAlgorithmRegistry()
        try registry.register("ppo") { StubAlgorithm(id: "ppo") }

        XCTAssertThrowsError(
            try registry.register("ppo") { StubAlgorithm(id: "replacement") }
        ) { error in
            XCTAssertEqual(
                error as? VectorRLAlgorithmRegistryError,
                .duplicateIdentifier("ppo"))
        }
        XCTAssertEqual(try registry.make("ppo").id, "ppo")
    }

    func testInvalidIdentifiersReturnTypedErrors() {
        let registry = VectorRLAlgorithmRegistry()
        for id in ["", " ", "dreamer v3", "ppo\n"] {
            XCTAssertThrowsError(
                try registry.register(id) { StubAlgorithm(id: id) }
            ) { error in
                XCTAssertEqual(
                    error as? VectorRLAlgorithmRegistryError,
                    .invalidIdentifier(id))
            }
            XCTAssertThrowsError(try registry.make(id)) { error in
                XCTAssertEqual(
                    error as? VectorRLAlgorithmRegistryError,
                    .invalidIdentifier(id))
            }
        }
    }

    func testUnknownLookupReportsSortedSnapshot() throws {
        let registry = VectorRLAlgorithmRegistry()
        try registry.register("sac") { StubAlgorithm(id: "sac") }
        try registry.register("ppo") { StubAlgorithm(id: "ppo") }

        XCTAssertThrowsError(try registry.make("dreamer-v3")) { error in
            XCTAssertEqual(
                error as? VectorRLAlgorithmRegistryError,
                .unknownIdentifier("dreamer-v3", available: ["ppo", "sac"]))
        }
    }

    func testFactoryIdentifierMismatchReturnsTypedError() throws {
        let registry = VectorRLAlgorithmRegistry()
        try registry.register("ppo") { StubAlgorithm(id: "sac") }

        XCTAssertThrowsError(try registry.make("ppo")) { error in
            XCTAssertEqual(
                error as? VectorRLAlgorithmRegistryError,
                .factoryIdentifierMismatch(registered: "ppo", produced: "sac"))
        }
    }

    func testFactoryCanReenterRegistry() throws {
        let registry = VectorRLAlgorithmRegistry()
        try registry.register("reentrant") {
            XCTAssertEqual(registry.algorithmIDs, ["reentrant"])
            return StubAlgorithm(id: "reentrant")
        }

        XCTAssertEqual(try registry.make("reentrant").id, "reentrant")
    }

    func testConcurrentDuplicateRegistrationIsAtomic() {
        let registry = VectorRLAlgorithmRegistry()
        let resultLock = NSLock()
        var successes = 0
        var errors: [VectorRLAlgorithmRegistryError] = []

        DispatchQueue.concurrentPerform(iterations: 64) { _ in
            do {
                try registry.register("ppo") { StubAlgorithm(id: "ppo") }
                resultLock.lock()
                successes += 1
                resultLock.unlock()
            } catch let error as VectorRLAlgorithmRegistryError {
                resultLock.lock()
                errors.append(error)
                resultLock.unlock()
            } catch {
                XCTFail("unexpected error: \(error)")
            }
        }

        XCTAssertEqual(successes, 1)
        XCTAssertEqual(errors.count, 63)
        XCTAssertTrue(errors.allSatisfy { $0 == .duplicateIdentifier("ppo") })
        XCTAssertEqual(registry.algorithmIDs, ["ppo"])
    }

    func testConcurrentReadsReturnIndependentInstances() throws {
        let registry = VectorRLAlgorithmRegistry()
        try registry.register("ppo") { StubAlgorithm(id: "ppo") }
        let resultLock = NSLock()
        var algorithms: [any VectorRLAlgorithm] = []
        var errors: [Error] = []

        DispatchQueue.concurrentPerform(iterations: 64) { _ in
            do {
                let algorithm = try registry.make("ppo")
                resultLock.lock()
                algorithms.append(algorithm)
                resultLock.unlock()
            } catch {
                resultLock.lock()
                errors.append(error)
                resultLock.unlock()
            }
        }

        XCTAssertTrue(errors.isEmpty)
        XCTAssertEqual(algorithms.count, 64)
        XCTAssertTrue(algorithms.allSatisfy { $0.id == "ppo" })
        XCTAssertEqual(Set(algorithms.map(ObjectIdentifier.init)).count, 64)
    }
}
