import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL

final class RLActionProviderTests: XCTestCase {
    func testClassicalControllerRunsThroughGenericProviderBoundary() throws {
        let task = try Arachne15LocomotionTask(
            configuration: .init(
                numEnvironments: 2, seed: 903,
                maxEpisodeSteps: 20, initialRollPitchRange: 0,
                initialYawRange: 0, observationNoise: false,
                maximumActionLatencySteps: 0,
                domainRandomization: .init(), autoReset: false))
        var observation = try task.reset(seed: 904)
        let provider: any RLActionProvider = Arachne15ClassicalController()

        XCTAssertEqual(provider.actionProviderID,
                       "arachne-paired-ripple-cpg-ik")
        try provider.reset(for: task, observation: observation)
        let first = try provider.actions(for: observation, task: task)
        try first.validate(for: task.spec)

        let result = try task.step(actions: first)
        observation = result.observations
        let second = try provider.actions(for: observation, task: task)
        try second.validate(for: task.spec)
        XCTAssertEqual(second.numEnvironments, 2)
        XCTAssertEqual(second.actionDimension,
                       Arachne15PolicyContract.actionDimension)
    }

    func testClassicalProviderRejectsWrongTaskType() throws {
        let task = try PushTTask(configuration: .init(numEnvironments: 1))
        let observation = try task.reset(seed: 7)
        let provider: any RLActionProvider = Arachne15ClassicalController()

        XCTAssertThrowsError(
            try provider.actions(for: observation, task: task)) { error in
                XCTAssertTrue(String(describing: error).contains(
                    "requires Arachne15LocomotionTask"))
            }
    }

    func testClassicalProviderResetsOnlyAutoResetRows() throws {
        let task = try Arachne15LocomotionTask(
            configuration: .init(
                numEnvironments: 2, seed: 910,
                maxEpisodeSteps: 20, initialRollPitchRange: 0,
                initialYawRange: 0, observationNoise: false,
                maximumActionLatencySteps: 0,
                domainRandomization: .init(), pointGoal: true,
                minimumGoalDistance: 1, maximumGoalDistance: 1,
                autoReset: true),
            taskID: "arachne15-goal-v0")
        let observation = try task.reset(seed: 911)
        try task.setGoal(
            environment: 0, direction: F3(1, 0, 0), distance: 1)
        try task.setGoal(
            environment: 1, direction: F3(1, 0, 0), distance: 1)
        let controller = Arachne15ClassicalController()
        let provider: any RLActionProvider = controller
        try provider.reset(for: task, observation: observation)

        for _ in 0..<3 {
            _ = try provider.actions(for: observation, task: task)
        }
        var result = RLStepBatch(spec: task.spec)
        result.observations = observation
        result.terminated[0] = true
        try provider.resetAfterStep(for: task, result: result)
        _ = try provider.actions(for: observation, task: task)

        XCTAssertEqual(controller.diagnostics.swingPhase[0], 0.25,
                       accuracy: 1e-6)
        XCTAssertEqual(controller.diagnostics.swingPhase[1], 1,
                       accuracy: 1e-6)
    }
}
