import XCTest
import Foundation
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL
@testable import MLXRL

final class VectorPolicyCompatibilityTests: XCTestCase {
    func testPhysicsEpochsPreserveLocalTaskRevision() {
        XCTAssertEqual(RLPhysicsContract.fixedGainActuatorV2(11), 1_000_011)
        XCTAssertEqual(
            RLPhysicsContract.deterministicColorSolveV1(11), 2_000_011)
        XCTAssertEqual(
            RLPhysicsContract.deterministicColorSolveV1(122), 2_000_122)
        XCTAssertEqual(Arachne15LocomotionTask.localTaskRevision, 6)
        XCTAssertEqual(
            Arachne15LocomotionTask.currentTaskRevision, 2_000_006)
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

    func testTrackedPolicyReleaseIndexMakesPhysicsCompatibilityExplicit()
        throws
    {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let checkpointRoot = root.appendingPathComponent(
            "checkpoints", isDirectory: true)
        let index = try PolicyBundleReleaseIndex.load(from:
            checkpointRoot.appendingPathComponent("policy-release-index.json"))
        let selectionIDs = index.releases.map(\.bundleIdentifier)
        XCTAssertEqual(Set(selectionIDs).count, selectionIDs.count)
        XCTAssertEqual(selectionIDs, [
            "humanoid-isaac-flat-v2",
            "arachne15-velocity-v1",
            "arachne15-goal-v1",
            "unitree-h1-sim2sim-v0",
        ])
        let acceptedH1 = try XCTUnwrap(index.releases.first {
            $0.bundleIdentifier == "humanoid-isaac-flat-v2"
        })
        let h1Bundle = try PolicyBundleLoader.load(directory:
            checkpointRoot.appendingPathComponent(
                acceptedH1.bundleRelativeDirectory, isDirectory: true))
        XCTAssertEqual(h1Bundle.manifest.simulation.task,
                       "humanoid-isaac-flat-v0")
        XCTAssertEqual(acceptedH1.qualification, .accepted)
        XCTAssertEqual(acceptedH1.evidenceRelativePath,
            "checkpoints/humanoid-isaac-flat-v2/requalification-manifest.json")
        XCTAssertEqual(acceptedH1.acceptanceAggregateRelativePath,
            "checkpoints/humanoid-isaac-flat-v2/qualification/aggregate.json")
        XCTAssertEqual(acceptedH1.deploymentManifestRelativePath,
            "checkpoints/humanoid-isaac-flat-v2/deployment-manifest.json")
        XCTAssertEqual(acceptedH1.expectedTaskRevision, 2_000_011)
        XCTAssertEqual(
            acceptedH1.expectedCheckpointFingerprint,
            "00bc782d1845ddde94282b46f0d7fa2732feeb4a8e52215a5abe62128bccc756")
        XCTAssertEqual(
            acceptedH1.expectedDeploymentManifestSHA256,
            "cb04233bd11bcc8dc3e0d2e1f0d6cc2e1ec27d4318344c7ea021d8b117be5d59")
        let acceptedArachne = [
            (
                selection: "arachne15-velocity-v1",
                task: "arachne15-velocity-v0",
                fingerprint:
                    "97f79641c8b7acf87c903b9d6baf739a5dc3c2536e52cb0e44121260133d79d5",
                deploymentSHA256:
                    "7295cf74dc9576a8b2bce74eeae0beb2932af96c5ff0617b0b99b82796b5039c"
            ),
            (
                selection: "arachne15-goal-v1",
                task: "arachne15-goal-v0",
                fingerprint:
                    "923e07c286f4fdb186b30a6fd95469e6848f4fec4ca1e3811320424b94c9dc02",
                deploymentSHA256:
                    "e7d747a41b3f724940bbe42d92dc38de8798dafbc7d39909e4ad1cf10ae1e127"
            ),
        ]
        for policy in acceptedArachne {
            let entry = try XCTUnwrap(index.releases.first {
                $0.bundleIdentifier == policy.selection
            })
            let bundle = try PolicyBundleLoader.load(directory:
                checkpointRoot.appendingPathComponent(
                    entry.bundleRelativeDirectory, isDirectory: true))
            XCTAssertEqual(bundle.manifest.simulation.task, policy.task)
            XCTAssertEqual(entry.qualification, .accepted)
            XCTAssertEqual(
                entry.bundleRelativeDirectory, policy.selection)
            XCTAssertEqual(entry.evidenceRelativePath,
                "checkpoints/\(policy.selection)/requalification-manifest.json")
            XCTAssertEqual(entry.acceptanceAggregateRelativePath,
                "checkpoints/\(policy.selection)/qualification/nominal/aggregate.json")
            XCTAssertEqual(entry.deploymentManifestRelativePath,
                "checkpoints/\(policy.selection)/deployment-manifest.json")
            XCTAssertEqual(entry.expectedTaskRevision, 2_000_006)
            XCTAssertEqual(
                entry.expectedCheckpointFingerprint, policy.fingerprint)
            XCTAssertEqual(
                entry.expectedDeploymentManifestSHA256,
                policy.deploymentSHA256)
        }
        let unitree = try XCTUnwrap(index.releases.first {
            $0.bundleIdentifier == "unitree-h1-sim2sim-v0"
        })
        let unitreeIdentity = try XCTUnwrap(
            unitree.unitreeH1ReleaseIdentity)
        XCTAssertEqual(
            unitreeIdentity.manifestSHA256,
            "9f434828cf2b2ede587bced686a22d30c3df6b048e631e94641bafeb7a45d117")
        XCTAssertEqual(
            unitreeIdentity.sourceRevision,
            "276801e46c5d433564f24658bac64f254b7d2d4b")
        XCTAssertEqual(
            unitreeIdentity.sourceCheckpointSHA256,
            "44a0fbceb81f3877833ae9a398d039bea1759cb0d3c8188181013885f70589eb")
        XCTAssertEqual(
            unitreeIdentity.weightsSHA256,
            "cb51db3e4ccbecc0d9a863173640f8cb8b5a5fb821bc1db9024c7957297ff4ee")
        XCTAssertEqual(
            unitreeIdentity.licenseSHA256,
            "98335465f43a20b5850e4651db6e74c4aa1e9fc8e8813d38f345178045c0da50")
        XCTAssertEqual(
            unitreeIdentity.goldenSequenceSHA256,
            "705e5d5bad46f696b4617ea440cfda1b4ff51b9504d949c5384e5d73cc14015b")
        let acceptedEvidenceURL = root.appendingPathComponent(
            acceptedH1.evidenceRelativePath)
        let acceptedManifest = try JSONDecoder().decode(
            VectorPolicyRequalificationManifest.self,
            from: Data(contentsOf: acceptedEvidenceURL))
        XCTAssertEqual(
            acceptedManifest.task, h1Bundle.manifest.simulation.task)
        XCTAssertEqual(acceptedManifest.targetTaskRevision, 2_000_011)
        XCTAssertEqual(
            acceptedManifest.candidateCheckpointFingerprint,
            "00bc782d1845ddde94282b46f0d7fa2732feeb4a8e52215a5abe62128bccc756")
        let acceptedAggregateURL = root.appendingPathComponent(
            try XCTUnwrap(acceptedH1.acceptanceAggregateRelativePath))
        let acceptedAggregate = try JSONDecoder().decode(
            PPOCheckpointEvaluationAggregate.self,
            from: Data(contentsOf: acceptedAggregateURL))
        XCTAssertEqual(acceptedAggregate.taskRevision, 2_000_011)
        XCTAssertEqual(acceptedAggregate.totalSuccesses, 2_028)
        XCTAssertEqual(acceptedAggregate.totalEpisodes, 2_048)
        XCTAssertEqual(
            acceptedAggregate.checkpointFingerprint,
            acceptedManifest.candidateCheckpointFingerprint)
        for entry in index.releases where entry.qualification == .accepted {
            let aggregatePath = try XCTUnwrap(
                entry.acceptanceAggregateRelativePath)
            let deploymentPath = try XCTUnwrap(
                entry.deploymentManifestRelativePath)
            let aggregateURL = root.appendingPathComponent(aggregatePath)
            let deploymentURL = root.appendingPathComponent(deploymentPath)
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: aggregateURL.path), aggregatePath)
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: deploymentURL.path), deploymentPath)

            let aggregate = try JSONDecoder().decode(
                PPOCheckpointEvaluationAggregate.self,
                from: Data(contentsOf: aggregateURL))
            try aggregate.validateStructure()
            let deployment = try JSONDecoder().decode(
                VectorPolicyDeploymentManifest.self,
                from: Data(contentsOf: deploymentURL))
            let bundle = try PolicyBundleLoader.load(directory:
                checkpointRoot.appendingPathComponent(
                    entry.bundleRelativeDirectory, isDirectory: true))
            XCTAssertEqual(aggregate.task, bundle.manifest.simulation.task)
            XCTAssertEqual(aggregate.task, deployment.task)
            XCTAssertEqual(aggregate.taskRevision, deployment.taskRevision)
            XCTAssertEqual(
                aggregate.checkpointFingerprint,
                deployment.checkpointFingerprint)
            XCTAssertTrue(aggregate.robustAcrossEvaluationSeeds)
        }
        for entry in index.releases
            where entry.qualification == .externalParityVerified {
            let directory = checkpointRoot.appendingPathComponent(
                entry.bundleRelativeDirectory)
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("manifest.json").path))
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(
                    "policy.safetensors").path))
        }

        for entry in index.releases where entry.qualification == .accepted {
            let policyDirectory = checkpointRoot.appendingPathComponent(
                entry.bundleRelativeDirectory)
            let metadataPath = policyDirectory.appendingPathComponent(
                "metadata.json")
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: policyDirectory.appendingPathComponent(
                    "policy.safetensors").path),
                entry.bundleRelativeDirectory)
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: policyDirectory.appendingPathComponent(
                    "training-state.json").path),
                entry.bundleRelativeDirectory)
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
                "\(entry.bundleRelativeDirectory): "
                    + mismatches.joined(separator: "; "))
        }
    }

    func testHistoricalIsaacFlatReplayRetainsSealedZeroUpdateLineage() throws {
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
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.deterministicColorSolveV1(11))
        XCTAssertTrue(metadata.compatibilityMismatches(with: task.spec)
            .contains { $0.contains("revision") })
        var historicalTargetSpec = task.spec
        historicalTargetSpec.revision = RLPhysicsContract.fixedGainActuatorV2(11)

        let manifest = try VectorPolicyRequalification.verify(
            targetSpec: historicalTargetSpec,
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

    func testAcceptedIsaacFlatV2VerifiesCurrentZeroUpdateRequalification()
        throws
    {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let parent = root.appendingPathComponent(
            "checkpoints/humanoid-isaac-flat-v1", isDirectory: true)
        let bundle = root.appendingPathComponent(
            "checkpoints/humanoid-isaac-flat-v2", isDirectory: true)
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
        XCTAssertEqual(task.spec.revision,
                       RLPhysicsContract.deterministicColorSolveV1(11))
        XCTAssertTrue(metadata.compatibilityMismatches(with: task.spec).isEmpty)

        let manifest = try VectorPolicyRequalification.verify(
            targetSpec: task.spec,
            evaluationCriteria: criteria,
            bundleDirectory: bundle.path,
            parentCheckpointDirectory: parent.path)

        XCTAssertEqual(manifest.sourceTaskRevision,
                       RLPhysicsContract.fixedGainActuatorV2(11))
        XCTAssertEqual(manifest.targetTaskRevision,
                       RLPhysicsContract.deterministicColorSolveV1(11))
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

        let aggregateFile = try XCTUnwrap(
            manifest.qualification?.aggregate.file)
        let aggregate = try JSONDecoder().decode(
            PPOCheckpointEvaluationAggregate.self,
            from: Data(contentsOf: bundle.appendingPathComponent(
                aggregateFile)))
        XCTAssertEqual(aggregate.taskRevision,
                       RLPhysicsContract.deterministicColorSolveV1(11))
        XCTAssertEqual(aggregate.checkpointFingerprint,
                       manifest.candidateCheckpointFingerprint)
        XCTAssertEqual(aggregate.totalSuccesses, 2_028)
        XCTAssertEqual(aggregate.totalEpisodes, 2_048)
        XCTAssertTrue(aggregate.robustAcrossEvaluationSeeds)
    }

    func testAcceptedArachneV1VerifiesCurrentMultiSuiteRequalification()
        throws
    {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let policies: [(
            selection: String, task: String, fingerprint: String,
            policySHA256: String,
            nominalSeeds: [UInt64], validationSeeds: [UInt64]
        )] = [
            (
                "arachne15-velocity-v1", "arachne15-velocity-v0",
                "97f79641c8b7acf87c903b9d6baf739a5dc3c2536e52cb0e44121260133d79d5",
                "a41b162b2bb922605e29a487a736f98b31416457807fc30a30c4e52014bf0638",
                [61_001, 61_002, 61_003, 61_004],
                [61_501, 61_502, 61_503, 61_504]
            ),
            (
                "arachne15-goal-v1", "arachne15-goal-v0",
                "923e07c286f4fdb186b30a6fd95469e6848f4fec4ca1e3811320424b94c9dc02",
                "9521c03cab6fc9e829cd2664fa0e086f69720d4aa46b1e5b893776a4df072c14",
                [62_001, 62_002, 62_003, 62_004],
                [63_001, 63_002, 63_003, 63_004]
            ),
        ]

        for policy in policies {
            let parent = root.appendingPathComponent(
                "checkpoints/\(policy.task)", isDirectory: true)
            let bundle = root.appendingPathComponent(
                "checkpoints/\(policy.selection)", isDirectory: true)
            let metadata = try JSONDecoder().decode(
                VectorPolicyMetadata.self,
                from: Data(contentsOf: bundle.appendingPathComponent(
                    "metadata.json")))
            let nominalOptions = BuiltInRLTasks.registry
                .checkpointReplayOptions(
                    for: metadata.task,
                    semanticOptions: try XCTUnwrap(
                        metadata.taskConfiguration),
                    maxEpisodeSteps: metadata.maxEpisodeSteps,
                    controlDecimation: metadata.controlDecimation)
            let nominalTask = try BuiltInRLTasks.registry.make(
                metadata.task,
                configuration: .init(
                    numEnvironments: 1, seed: policy.nominalSeeds[0],
                    autoReset: false, options: nominalOptions))
            var validationOptions = nominalOptions
            validationOptions["validationCollisionProfile"] = 1
            let validationTask = try BuiltInRLTasks.registry.make(
                metadata.task,
                configuration: .init(
                    numEnvironments: 1, seed: policy.validationSeeds[0],
                    autoReset: false, options: validationOptions))
            let nominalCriteria = try XCTUnwrap(
                (nominalTask as? any RLEvaluationCriteriaProviding)?
                    .evaluationCriteria)
            let validationCriteria = try XCTUnwrap(
                (validationTask as? any RLEvaluationCriteriaProviding)?
                    .evaluationCriteria)
            var nominalSuiteSpec = nominalTask.spec
            nominalSuiteSpec.numEnvironments = 128
            var validationSuiteSpec = validationTask.spec
            validationSuiteSpec.numEnvironments = 128

            let manifest = try VectorPolicyRequalification.verify(
                targetSpec: nominalTask.spec,
                evaluationCriteria: nominalCriteria,
                bundleDirectory: bundle.path,
                parentCheckpointDirectory: parent.path,
                suiteContracts: [
                    .init(
                        id: "nominal", taskSpec: nominalSuiteSpec,
                        evaluationCriteria: nominalCriteria),
                    .init(
                        id: "validation-collision",
                        taskSpec: validationSuiteSpec,
                        evaluationCriteria: validationCriteria),
                ])

            XCTAssertEqual(manifest.schemaVersion, 2)
            XCTAssertEqual(manifest.task, policy.task)
            XCTAssertEqual(manifest.sourceTaskRevision, 6)
            XCTAssertEqual(
                manifest.targetTaskRevision,
                Arachne15LocomotionTask.currentTaskRevision)
            XCTAssertEqual(
                manifest.candidateCheckpointFingerprint,
                policy.fingerprint)
            XCTAssertEqual(
                manifest.parentPolicySHA256,
                manifest.candidatePolicySHA256)
            XCTAssertEqual(
                manifest.candidatePolicySHA256, policy.policySHA256)
            XCTAssertEqual(manifest.targetTrainingUpdates, 0)
            XCTAssertEqual(manifest.targetTrainingEnvironmentSteps, 0)
            XCTAssertEqual(
                manifest.qualificationMatrix?.suites.map(\.id),
                ["nominal", "validation-collision"])
            XCTAssertEqual(
                manifest.qualificationMatrix?.suites[0].evaluationSeeds,
                policy.nominalSeeds)
            XCTAssertEqual(
                manifest.qualificationMatrix?.suites[1].evaluationSeeds,
                policy.validationSeeds)
            XCTAssertEqual(
                manifest.qualificationMatrix?.comparisons.first?
                    .maximumPooledSuccessRateDrop,
                0.05)
            XCTAssertEqual(
                manifest.qualification?.aggregate.file,
                "qualification/nominal/aggregate.json")
            XCTAssertEqual(
                manifest.qualification?.additionalSuites?.first?.id,
                "validation-collision")
            XCTAssertEqual(
                try Data(contentsOf: parent.appendingPathComponent(
                    "policy.safetensors")),
                try Data(contentsOf: bundle.appendingPathComponent(
                    "policy.safetensors")))
        }
    }
}
