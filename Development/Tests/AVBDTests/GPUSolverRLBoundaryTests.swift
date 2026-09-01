import XCTest
@testable import PhysicsAVBD
@testable import RL

final class GPUSolverRLBoundaryTests: XCTestCase {
    func testVectorizedTaskPropagatesSolverFailure() throws {
        let task = try PushTTask(configuration: .init(numEnvironments: 1))
        task.environment.solver.commandBufferFactoryForTesting = { nil }
        var result = RLStepBatch(spec: task.spec)

        XCTAssertThrowsError(try task.step(
            actions: RLActionBatch(spec: task.spec), into: &result)) { error in
            XCTAssertEqual(
                error as? GPUSolver.RuntimeFailure,
                .commandBufferCreation(operation: "physics", frame: 1))
        }
        XCTAssertThrowsError(try task.step(
            actions: RLActionBatch(spec: task.spec), into: &result)) { error in
            XCTAssertEqual(error as? GPUSolver.RuntimeFailure,
                           task.environment.solver.runtimeFailure)
        }
    }

    func testVectorizedTaskPropagatesAsynchronousExecutionFailure() throws {
        let task = try PushTTask(configuration: .init(numEnvironments: 1))
        let injected = GPUSolver.RuntimeFailure.commandExecution(
            operation: "physics", frame: 1, status: -1,
            domain: "AVBDTests", code: 8, message: "synthetic GPU fault")
        task.environment.solver.completionFailureForTesting = {
            operation, frame in
            operation == "physics" && frame == 1 ? injected : nil
        }
        var result = RLStepBatch(spec: task.spec)

        XCTAssertThrowsError(try task.step(
            actions: RLActionBatch(spec: task.spec), into: &result)) { error in
            XCTAssertEqual(error as? GPUSolver.RuntimeFailure, injected)
        }
    }
}
