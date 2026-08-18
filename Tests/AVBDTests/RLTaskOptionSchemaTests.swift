import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL

final class RLTaskOptionSchemaTests: XCTestCase {
    private let humanoidBoxCarryIntegerOptions: Set<String> = [
        "maxEpisodeSteps", "solverIterations",
        "stationDistanceCurriculumControlSteps",
        "liftClearanceCurriculumControlSteps",
        "carryDistanceCurriculumControlSteps",
        "destinationBearingCurriculumControlSteps",
        "minimumTrainingSuccessDwellSteps", "successDwellSteps",
        "successDwellCurriculumControlSteps", "manipulationHandoffSteps",
        "carryHandoffSteps", "carryCommandRampSteps",
        "minimumLoadedAlternatingSteps",
    ]

    private let humanoidBoxCarryBooleanOptions: Set<String> = [
        "observationNoise", "carryHolonomicCommand",
        "freezeBasePolicyExpert", "freezeManipulationPolicyExpert",
        "freezeCarryPolicyExpert", "manipulationGatedActor",
        "initializeManipulationExpertFromBaseOnTransfer",
        "initializeCarryExpertFromManipulationExpertOnTransfer",
        "compositionalCarryController", "upperBodyCarryController",
        "carryLocomotionControlsTorso",
        "initializeCarryExpertFromBaseOnTransfer",
        "initializeCarryLocomotionExpertFromBaseOnTransfer",
        "advanceReplaySnapshotAtDestinationContact",
        "coupledCarryCommandTracking",
    ]

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

    func testHumanoidBoxCarryPublishesEveryCoercedIntegerAndBooleanType() throws {
        let schema = try XCTUnwrap(
            BuiltInRLTasks.registry.optionSchema(
                for: "humanoid-box-carry-v0"))
        let integerOptions = Set(schema.definitions.compactMap {
            $0.value.valueKind == .integer ? $0.key : nil
        })
        let booleanOptions = Set(schema.definitions.compactMap {
            $0.value.valueKind == .boolean ? $0.key : nil
        })

        XCTAssertEqual(integerOptions, humanoidBoxCarryIntegerOptions)
        XCTAssertEqual(booleanOptions, humanoidBoxCarryBooleanOptions)
    }

    func testHumanoidBoxCarryRejectsFractionalIntegerOptions() throws {
        let schema = try XCTUnwrap(
            BuiltInRLTasks.registry.optionSchema(
                for: "humanoid-box-carry-v0"))

        for name in humanoidBoxCarryIntegerOptions.sorted() {
            XCTAssertThrowsError(try schema.validate(
                .init(numEnvironments: 1, options: [name: 1.5]),
                taskID: "humanoid-box-carry-v0"), name) { error in
                let message = String(describing: error)
                XCTAssertTrue(message.contains(name), message)
                XCTAssertTrue(message.contains("must be an integer"), message)
            }
        }
    }

    func testHumanoidBoxCarryRejectsEveryNonCanonicalBooleanValue() throws {
        let schema = try XCTUnwrap(
            BuiltInRLTasks.registry.optionSchema(
                for: "humanoid-box-carry-v0"))

        for name in humanoidBoxCarryBooleanOptions.sorted() {
            for value: Float in [-1, 0.5, 2] {
                XCTAssertThrowsError(try schema.validate(
                    .init(numEnvironments: 1, options: [name: value]),
                    taskID: "humanoid-box-carry-v0"), "\(name)=\(value)") {
                    error in
                    let message = String(describing: error)
                    XCTAssertTrue(message.contains(name), message)
                    XCTAssertTrue(message.contains("must be 0 or 1"), message)
                }
            }
        }
    }

    func testHumanoidBoxCarryAcceptsCanonicalIntegerAndBooleanValues() throws {
        let schema = try XCTUnwrap(
            BuiltInRLTasks.registry.optionSchema(
                for: "humanoid-box-carry-v0"))
        var options = Dictionary(uniqueKeysWithValues:
            humanoidBoxCarryIntegerOptions.map { ($0, Float(1)) })

        for booleanValue: Float in [0, 1] {
            for name in humanoidBoxCarryBooleanOptions {
                options[name] = booleanValue
            }
            XCTAssertNoThrow(try schema.validate(
                .init(numEnvironments: 1, options: options),
                taskID: "humanoid-box-carry-v0"))
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
