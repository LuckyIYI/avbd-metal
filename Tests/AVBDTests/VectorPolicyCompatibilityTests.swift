import XCTest
import Foundation
@testable import AVBDCore
@testable import AVBDLearn

final class VectorPolicyCompatibilityTests: XCTestCase {
    func testMetadataRejectsSameShapeWrongTaskAndPhysicsContract() {
        let spec = RLTaskSpec(
            id: "destination-v0", revision: 4, numEnvironments: 1,
            observation: .init(name: "observation", shape: [3]),
            action: .init(name: "action", shape: [2]),
            maxEpisodeSteps: 100, simulationStep: 0.002,
            controlDecimation: 10,
            configurationValues: ["plant": 1])
        var metadata = VectorPolicyMetadata(
            architectureVersion: VectorActorCritic.architectureVersion,
            task: "source-v0", taskRevision: 3,
            taskConfiguration: ["plant": 2],
            observationDimension: 3, actionDimension: 2,
            simulationStep: 0.004, controlDecimation: 5,
            maxEpisodeSteps: 200, ppo: .init(),
            normalizer: .init(
                count: 0, mean: [0, 0, 0], variance: [1, 1, 1]))

        let mismatches = metadata.compatibilityMismatches(with: spec)
        for expected in [
            "task source-v0", "revision 3", "configuration [",
            "simulation step", "control decimation", "episode steps",
        ] {
            XCTAssertTrue(mismatches.contains { $0.contains(expected) })
        }
        XCTAssertFalse(mismatches.contains { $0.contains("observations") })
        XCTAssertFalse(mismatches.contains { $0.contains("actions") })

        metadata.task = spec.id
        metadata.taskRevision = spec.revision
        metadata.taskConfiguration = spec.configurationValues
        metadata.simulationStep = spec.simulationStep
        metadata.controlDecimation = spec.controlDecimation
        metadata.maxEpisodeSteps = spec.maxEpisodeSteps
        XCTAssertTrue(metadata.compatibilityMismatches(with: spec).isEmpty)

        metadata.taskConfiguration = nil
        XCTAssertTrue(metadata.compatibilityMismatches(with: spec).contains {
            $0 == "configuration metadata is missing"
        })
    }

    func testTrackedMetadataReconstructionExposesPhysicsRevisionStaleness()
        throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let metadataPaths: [(path: String, expectsStalePhysics: Bool)] = [
            ("checkpoints/humanoid-walk-v0/metadata.json", true),
            ("checkpoints/humanoid-goal-v0/metadata.json", true),
            ("checkpoints/humanoid-isaac-flat-v0/metadata.json", true),
            ("checkpoints/humanoid-isaac-goal-v0/metadata.json", true),
            ("Robots/Arachne15/policies/"
                + "arachne15-goal-r6-update-000020/metadata.json", false),
        ]

        for entry in metadataPaths {
            let metadata = try JSONDecoder().decode(
                VectorPolicyMetadata.self,
                from: Data(contentsOf: root.appendingPathComponent(entry.path)))
            let options = BuiltInRLTasks.registry.checkpointReplayOptions(
                for: metadata.task,
                semanticOptions: try XCTUnwrap(metadata.taskConfiguration),
                maxEpisodeSteps: metadata.maxEpisodeSteps,
                controlDecimation: metadata.controlDecimation)
            let task = try BuiltInRLTasks.registry.make(
                metadata.task,
                configuration: .init(
                    numEnvironments: 1, seed: 1, autoReset: false,
                    options: options))
            let mismatches = metadata.compatibilityMismatches(with: task.spec)
            if entry.expectsStalePhysics {
                XCTAssertEqual(mismatches.count, 1, entry.path)
                XCTAssertTrue(mismatches[0].hasPrefix("revision "), entry.path)
            } else {
                XCTAssertTrue(
                    mismatches.isEmpty,
                    "\(entry.path): \(mismatches.joined(separator: "; "))")
            }
        }
    }
}
