import XCTest
import Foundation
@testable import AVBDCore
@testable import AVBDLearn

final class VectorPolicyCompatibilityTests: XCTestCase {
    func testExplicitReplayCheckpointIsAnAuthoritativeSource() {
        XCTAssertEqual(
            PolicyReplayCheckpointResolution.candidates(
                explicit: "/operator/checkpoint",
                fallbacks: ["/live/latest", "/bundled", nil]),
            ["/operator/checkpoint"])
        XCTAssertEqual(
            PolicyReplayCheckpointResolution.candidates(
                explicit: nil,
                fallbacks: ["/live/latest", nil, "/bundled"]),
            ["/live/latest", "/bundled"])
    }

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

    func testTrackedReplayCatalogMakesPhysicsCompatibilityExplicit() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let selectionIDs = PolicyReplayCatalog.entries.map(\.selectionID)
        XCTAssertEqual(Set(selectionIDs).count, selectionIDs.count)
        XCTAssertFalse(selectionIDs.contains("humanoid-isaac-flat-v0"))
        XCTAssertTrue(selectionIDs.contains("humanoid-isaac-flat-v1"))
        XCTAssertEqual(
            PolicyReplayCatalog.entry(
                selectionID: "humanoid-isaac-flat-v1")?.evidenceRelativePath,
            "checkpoints/humanoid-isaac-flat-v1/requalification-manifest.json")
        XCTAssertTrue(PolicyReplayCatalog.entries.allSatisfy {
            $0.qualification != .requalificationRequired
        })
        XCTAssertEqual(
            PolicyReplayCatalog.historicalEntries.map(\.selectionID),
            ["humanoid-isaac-flat-v0"])
        XCTAssertNil(PolicyReplayCatalog.entry(
            selectionID: "humanoid-isaac-flat-v0"))
        XCTAssertEqual(
            PolicyReplayCatalog.historicalEntry(
                selectionID: "humanoid-isaac-flat-v0")?.qualification,
            .requalificationRequired)
        let allSelectionIDs = PolicyReplayCatalog.allDeclaredEntries.map(
            \.selectionID)
        XCTAssertEqual(Set(allSelectionIDs).count, allSelectionIDs.count)
        for entry in PolicyReplayCatalog.allDeclaredEntries {
            XCTAssertEqual(
                entry.runtime == .classicalController,
                entry.checkpointRelativeDirectory == nil,
                "only the explicitly non-neural baseline may omit a checkpoint")
            XCTAssertEqual(
                entry.runtime == .classicalController,
                entry.evidenceRelativePath == nil,
                "every learned replay must name machine-readable evidence")
            if let evidence = entry.evidenceRelativePath {
                XCTAssertTrue(
                    FileManager.default.fileExists(
                        atPath: root.appendingPathComponent(evidence).path),
                    evidence)
            }
        }
        let policyDirectories = PolicyReplayCatalog.allDeclaredEntries
            .filter { $0.runtime == .nativeMLX }
            .map { "checkpoints/" + $0.checkpointRelativeDirectory! }

        let checkpointRoot = root.appendingPathComponent("checkpoints")
        let actualTaskDirectories = try FileManager.default.contentsOfDirectory(
            at: checkpointRoot, includingPropertiesForKeys: nil)
            .filter {
                FileManager.default.fileExists(
                    atPath: $0.appendingPathComponent("metadata.json").path)
            }
            .map(\.lastPathComponent)
            .sorted()
        XCTAssertEqual(actualTaskDirectories, policyDirectories.map {
            URL(fileURLWithPath: $0).lastPathComponent
        }.sorted(), "deprecated or unregistered replay checkpoints must not ship")

        for entry in PolicyReplayCatalog.entries
            where entry.runtime == .unitreeRecurrentMLX {
            let directory = checkpointRoot.appendingPathComponent(
                try XCTUnwrap(entry.checkpointRelativeDirectory))
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("manifest.json").path))
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(
                    "policy.safetensors").path))
        }

        for entry in PolicyReplayCatalog.nativeLearnedEntries {
            let directory = "checkpoints/"
                + (try XCTUnwrap(entry.checkpointRelativeDirectory))
            let policyDirectory = root.appendingPathComponent(directory)
            let metadataPath = policyDirectory.appendingPathComponent(
                "metadata.json")
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: policyDirectory.appendingPathComponent(
                    "policy.safetensors").path), directory)
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: policyDirectory.appendingPathComponent(
                    "training-state.json").path), directory)
            let metadata = try JSONDecoder().decode(
                VectorPolicyMetadata.self,
                from: Data(contentsOf: metadataPath))
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
            XCTAssertTrue(
                mismatches.isEmpty,
                "\(directory): \(mismatches.joined(separator: "; "))")
        }

        for entry in PolicyReplayCatalog.historicalEntries
            where entry.runtime == .nativeMLX {
            let directory = "checkpoints/"
                + (try XCTUnwrap(entry.checkpointRelativeDirectory))
            let policyDirectory = root.appendingPathComponent(directory)
            let metadata = try JSONDecoder().decode(
                VectorPolicyMetadata.self,
                from: Data(contentsOf: policyDirectory.appendingPathComponent(
                    "metadata.json")))
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
            XCTAssertFalse(mismatches.isEmpty, directory)
            XCTAssertTrue(mismatches.contains { $0.contains("revision") },
                          "\(directory): \(mismatches)")
        }
    }

    func testAcceptedIsaacFlatReplayHasSealedZeroUpdateLineage() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let parent = root.appendingPathComponent(
            "checkpoints/humanoid-isaac-flat-v0", isDirectory: true)
        let bundle = root.appendingPathComponent(
            "checkpoints/humanoid-isaac-flat-v1", isDirectory: true)
        let metadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self,
            from: Data(contentsOf: bundle.appendingPathComponent(
                "metadata.json")))
        let options = BuiltInRLTasks.registry.checkpointReplayOptions(
            for: metadata.task,
            semanticOptions: try XCTUnwrap(metadata.taskConfiguration),
            maxEpisodeSteps: metadata.maxEpisodeSteps,
            controlDecimation: metadata.controlDecimation)
        let task = try BuiltInRLTasks.registry.make(
            metadata.task,
            configuration: .init(
                numEnvironments: 1, seed: 51_001, autoReset: false,
                options: options))
        let criteria = try XCTUnwrap(
            (task as? any RLEvaluationCriteriaProviding)?.evaluationCriteria)

        let manifest = try VectorPolicyRequalification.verify(
            targetSpec: task.spec,
            evaluationCriteria: criteria,
            bundleDirectory: bundle.path,
            parentCheckpointDirectory: parent.path)

        XCTAssertEqual(manifest.sourceTaskRevision,
                       RLPhysicsContract.fixedGainActuatorV2(10))
        XCTAssertEqual(manifest.targetTaskRevision,
                       RLPhysicsContract.fixedGainActuatorV2(11))
        XCTAssertEqual(manifest.parentPolicySHA256,
                       manifest.candidatePolicySHA256)
        XCTAssertEqual(manifest.targetTrainingUpdates, 0)
        XCTAssertEqual(manifest.targetTrainingEnvironmentSteps, 0)
        XCTAssertEqual(manifest.qualificationPlan.evaluationSeeds,
                       [51_001, 51_002, 51_003, 51_004])
        XCTAssertEqual(manifest.qualificationPlan.episodesPerReport, 512)
        XCTAssertEqual(manifest.qualification?.reports.count, 4)
        XCTAssertEqual(
            try Data(contentsOf: parent.appendingPathComponent(
                "policy.safetensors")),
            try Data(contentsOf: bundle.appendingPathComponent(
                "policy.safetensors")))
    }
}
