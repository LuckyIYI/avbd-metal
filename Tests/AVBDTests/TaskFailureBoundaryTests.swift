import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL

final class TaskFailureBoundaryTests: XCTestCase {
    private func caughtSolverFailure(
        _ operation: () throws -> Void,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> GPUSolver.RuntimeFailure {
        var observed: GPUSolver.RuntimeFailure?
        XCTAssertThrowsError(try operation(), file: file, line: line) { error in
            observed = error as? GPUSolver.RuntimeFailure
            if observed == nil {
                XCTFail("expected GPUSolver.RuntimeFailure, got \(error)",
                        file: file, line: line)
            }
        }
        return try XCTUnwrap(observed, file: file, line: line)
    }

    func testPoisonedVectorTaskStepAndResetRethrowStickySolverFailure()
        throws
    {
        let concreteTask = try PushTTask(configuration: .init(
            numEnvironments: 1, controlDecimation: 1))
        let task: any VectorizedRLTask = concreteTask
        concreteTask.environment.solver.commandBufferFactoryForTesting = {
            nil
        }

        var result = RLStepBatch(spec: task.spec)
        let actions = RLActionBatch(spec: task.spec)
        let initialFailure = try caughtSolverFailure {
            try task.step(actions: actions, into: &result)
        }
        XCTAssertEqual(
            initialFailure,
            .commandBufferCreation(operation: "physics", frame: 1))
        XCTAssertEqual(concreteTask.environment.solver.runtimeFailure,
                       initialFailure)

        // Both task entry points must stop at the checked solver boundary.
        // Reaching PushT's legacy state setters after poisoning would call the
        // fail-closed compatibility wrapper instead of returning this error.
        let repeatedStepFailure = try caughtSolverFailure {
            try task.step(actions: actions, into: &result)
        }
        var observations = RLObservationBatch(spec: task.spec)
        let resetFailure = try caughtSolverFailure {
            try task.reset(environments: nil, seed: 91,
                           into: &observations)
        }

        XCTAssertEqual(repeatedStepFailure, initialFailure)
        XCTAssertEqual(resetFailure, initialFailure)
        XCTAssertEqual(concreteTask.environment.solver.runtimeFailure,
                       initialFailure)
        XCTAssertEqual(concreteTask.environment.solver.frameIndex, 0)
    }

    func testProjectileTaskChecksStickyFailureBeforeLaunchMutation() throws {
        let task = try HumanoidIsaacVelocityTask(
            configuration: .init(
                numEnvironments: 1,
                seed: 101,
                maxEpisodeSteps: 2,
                commandResamplingSteps: 1,
                initialYawRange: 0,
                observationNoise: false,
                solverIterations: 1,
                autoReset: false,
                pointGoal: true,
                minimumGoalDistance: 4,
                maximumGoalDistance: 4,
                projectileProbability: 1,
                minimumProjectileSpeed: 4,
                maximumProjectileSpeed: 4,
                minimumProjectileLaunchStep: 0,
                maximumProjectileLaunchStep: 0),
            taskID: "humanoid-isaac-goal-v0")
        XCTAssertTrue(task.hasProjectile(environment: 0))

        let solver = task.environment.solver
        solver.commandBufferFactoryForTesting = { nil }
        let poison = try caughtSolverFailure {
            try solver.submitStep()
        }
        XCTAssertEqual(
            poison,
            .commandBufferCreation(operation: "physics", frame: 1))
        XCTAssertEqual(solver.runtimeFailure, poison)

        var result = RLStepBatch(spec: task.spec)
        result.rewards[0] = 123
        result.metrics["test/sentinel"] = [7]
        let stepFailure = try caughtSolverFailure {
            try task.step(actions: RLActionBatch(spec: task.spec),
                          into: &result)
        }

        // This task is guaranteed to launch a projectile on its first step.
        // Keeping caller-owned output untouched proves that health checking
        // happens before clearSignals and, critically, before the scheduled
        // launch's legacy state reads and writes.
        XCTAssertEqual(stepFailure, poison)
        XCTAssertEqual(result.rewards[0], 123)
        XCTAssertEqual(result.metrics["test/sentinel"], [7])
        XCTAssertEqual(solver.runtimeFailure, poison)
        XCTAssertEqual(solver.frameIndex, 0)
    }
}
