import XCTest
@testable import AVBDCore

final class RLTaskOptionSchemaTests: XCTestCase {
    func testBuiltInRegistryPublishesTypedTaskOptions() throws {
        let schema = try XCTUnwrap(
            BuiltInRLTasks.registry.optionSchema(for: "arachne15-goal-v0"))

        XCTAssertEqual(schema.definitions["goalCommandSpeed"]?.valueKind,
                       .number)
        XCTAssertEqual(schema.definitions["goalDwellSteps"]?.valueKind,
                       .integer)
        XCTAssertEqual(schema.definitions["pointGoal"]?.valueKind,
                       .boolean)
        XCTAssertEqual(schema.definitions["maxEpisodeSteps"]?.lowerBound, 1)
        XCTAssertTrue(schema.optionNames.contains("maximumGoalDistance"))

        let encoded = try JSONEncoder().encode(schema)
        XCTAssertEqual(try JSONDecoder().decode(
            RLTaskOptionSchema.self, from: encoded), schema)
    }

    func testRegistryRejectsFractionalIntegerBeforeFactoryCoercion() {
        XCTAssertThrowsError(try BuiltInRLTasks.registry.make(
            "pusht-state-v0",
            configuration: .init(
                numEnvironments: 1,
                options: ["maxEpisodeSteps": 12.5]))) { error in
            XCTAssertTrue(String(describing: error).contains("must be an integer"))
        }
    }

    func testRegistryRejectsIntegerOutsideSwiftIntRange() {
        XCTAssertThrowsError(try BuiltInRLTasks.registry.make(
            "pusht-state-v0",
            configuration: .init(
                numEnvironments: 1,
                options: ["maxEpisodeSteps": 1e30]))) { error in
            XCTAssertTrue(String(describing: error).contains("must fit Swift Int"))
        }
    }

    func testRegistryRejectsNonCanonicalBoolean() {
        XCTAssertThrowsError(try BuiltInRLTasks.registry.make(
            "arachne15-goal-v0",
            configuration: .init(
                numEnvironments: 1,
                options: ["pointGoal": 2]))) { error in
            XCTAssertTrue(String(describing: error).contains("must be 0 or 1"))
        }
    }

    func testRegistryRejectsNonFiniteAndUnknownOptions() {
        XCTAssertThrowsError(try BuiltInRLTasks.registry.make(
            "pusht-state-v0",
            configuration: .init(
                numEnvironments: 1,
                options: ["actionScale": .infinity]))) { error in
            XCTAssertTrue(String(describing: error).contains("must be finite"))
        }
        XCTAssertThrowsError(try BuiltInRLTasks.registry.make(
            "pusht-state-v0",
            configuration: .init(
                numEnvironments: 1,
                options: ["actionScael": 0.2]))) { error in
            XCTAssertTrue(String(describing: error).contains("actionScael"))
            XCTAssertTrue(String(describing: error).contains("supported:"))
        }
    }

    func testCheckpointReplayRestoresOnlySupportedStructuralOptions() {
        let semantic = ["observationNoise": Float(0)]
        let h1 = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: "humanoid-isaac-flat-v0", semanticOptions: semantic,
            maxEpisodeSteps: 777, controlDecimation: 4)
        XCTAssertEqual(h1["maxEpisodeSteps"], 777)
        XCTAssertNil(h1["controlDecimation"])

        let arachne = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: "arachne15-goal-v0", semanticOptions: [:],
            maxEpisodeSteps: 888, controlDecimation: 12)
        XCTAssertEqual(arachne["maxEpisodeSteps"], 888)
        XCTAssertEqual(arachne["controlDecimation"], 12)
    }
}
