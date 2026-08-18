import CryptoKit
import Foundation
import XCTest
@testable import AVBDCore
@testable import AVBDLearn

final class PolicyRequalificationTests: XCTestCase {
    private struct Fixture {
        var root: URL
        var parent: URL
        var candidate: URL
        var targetSpec: RLTaskSpec
        var preparation: VectorPolicyRequalificationManifest
        var criteria: RLEvaluationCriteria
    }

    func testPrepareCreatesByteIdenticalZeroUpdateCandidateWithoutOverwrite()
        throws
    {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        XCTAssertNotEqual(
            fixture.preparation.parentCheckpointFingerprint,
            fixture.preparation.candidateCheckpointFingerprint)
        XCTAssertEqual(
            fixture.preparation.parentPolicySHA256,
            fixture.preparation.candidatePolicySHA256)
        XCTAssertNil(fixture.preparation.qualification)
        XCTAssertEqual(
            try Data(contentsOf: fixture.parent.appendingPathComponent(
                "policy.safetensors")),
            try Data(contentsOf: fixture.candidate.appendingPathComponent(
                "policy.safetensors")))

        let metadata = try decode(
            VectorPolicyMetadata.self,
            at: fixture.candidate.appendingPathComponent("metadata.json"))
        XCTAssertEqual(metadata.taskRevision, fixture.targetSpec.revision)
        XCTAssertEqual(metadata.taskConfiguration,
                       fixture.targetSpec.configurationValues)
        XCTAssertEqual(metadata.inferenceBatchSize, 8)
        XCTAssertEqual(metadata.ppo.initializationCheckpoint,
                       fixture.parent.path)
        XCTAssertEqual(metadata.ppo.updates, 0)
        XCTAssertTrue(metadata.compatibilityMismatches(
            with: fixture.targetSpec).isEmpty)

        let state = try decode(
            VectorPPOTrainingState.self,
            at: fixture.candidate.appendingPathComponent(
                "training-state.json"))
        XCTAssertEqual(state.completedUpdates, 0)
        XCTAssertEqual(state.environmentSteps, 0)
        XCTAssertEqual(state.optimizerSteps, 0)
        XCTAssertNil(state.adaptiveLearningRate)

        XCTAssertThrowsError(try VectorPolicyRequalification.prepare(
            targetSpec: fixture.targetSpec,
            parentCheckpointDirectory: fixture.parent.path,
            outputDirectory: fixture.candidate.path,
            expectedParentCheckpointFingerprint:
                fixture.preparation.parentCheckpointFingerprint,
            inferenceBatchSize: 8,
            declaredSourceCommit: String(repeating: "a", count: 40),
            qualificationPlan: fixture.preparation.qualificationPlan,
            evaluationCriteria: fixture.criteria))

        let parentAlias = fixture.root.appendingPathComponent("parent-alias")
        try FileManager.default.createSymbolicLink(
            at: parentAlias, withDestinationURL: fixture.parent)
        XCTAssertThrowsError(try VectorPolicyRequalification.prepare(
            targetSpec: fixture.targetSpec,
            parentCheckpointDirectory: fixture.parent.path,
            outputDirectory: parentAlias.appendingPathComponent(
                "symlink-nested-candidate").path,
            expectedParentCheckpointFingerprint:
                fixture.preparation.parentCheckpointFingerprint,
            inferenceBatchSize: 8,
            declaredSourceCommit: String(repeating: "a", count: 40),
            qualificationPlan: fixture.preparation.qualificationPlan,
            evaluationCriteria: fixture.criteria))
        XCTAssertThrowsError(try VectorPolicyRequalification.prepare(
            targetSpec: fixture.targetSpec,
            parentCheckpointDirectory: fixture.parent.path,
            outputDirectory: fixture.parent.appendingPathComponent(
                "nested-candidate").path,
            expectedParentCheckpointFingerprint:
                fixture.preparation.parentCheckpointFingerprint,
            inferenceBatchSize: 8,
            declaredSourceCommit: String(repeating: "a", count: 40),
            qualificationPlan: fixture.preparation.qualificationPlan,
            evaluationCriteria: fixture.criteria))
    }

    func testPrepareRejectsWrongFingerprintAndAnyNonRevisionDrift() throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let parent = root.appendingPathComponent("parent", isDirectory: true)
        let sourceSpec = makeSpec(revision: 7)
        try writeParent(at: parent, spec: sourceSpec)
        let fingerprint = try VectorPPOTrainer.checkpointFingerprint(
            directory: parent.path)
        let plan = VectorPolicyRequalificationPlan(
            evaluationSeeds: [101, 102, 103, 104],
            evaluationEnvironments: 8)

        XCTAssertThrowsError(try VectorPolicyRequalification.prepare(
            targetSpec: makeSpec(revision: 8),
            parentCheckpointDirectory: parent.path,
            outputDirectory: root.appendingPathComponent("wrong-hash").path,
            expectedParentCheckpointFingerprint: String(
                repeating: "0", count: 64),
            inferenceBatchSize: 8,
            declaredSourceCommit: String(repeating: "b", count: 40),
            qualificationPlan: plan,
            evaluationCriteria: .init(minimumSuccessRate: 0.8)))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("wrong-hash").path))

        var configurationDrift = makeSpec(revision: 8)
        configurationDrift.configurationValues["plant"] = 2
        XCTAssertThrowsError(try VectorPolicyRequalification.prepare(
            targetSpec: configurationDrift,
            parentCheckpointDirectory: parent.path,
            outputDirectory: root.appendingPathComponent("config-drift").path,
            expectedParentCheckpointFingerprint: fingerprint,
            inferenceBatchSize: 8,
            declaredSourceCommit: String(repeating: "b", count: 40),
            qualificationPlan: plan,
            evaluationCriteria: .init(minimumSuccessRate: 0.8)))
        XCTAssertThrowsError(try VectorPolicyRequalification.prepare(
            targetSpec: sourceSpec,
            parentCheckpointDirectory: parent.path,
            outputDirectory: root.appendingPathComponent("no-revision-drift").path,
            expectedParentCheckpointFingerprint: fingerprint,
            inferenceBatchSize: 8,
            declaredSourceCommit: String(repeating: "b", count: 40),
            qualificationPlan: plan,
            evaluationCriteria: .init(minimumSuccessRate: 0.8)))
    }

    func testPublishSealsExactReportsAndEvidenceWithoutOverwrite() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let qualification = try writeQualification(for: fixture)
        let relaxedOutput = fixture.root.appendingPathComponent(
            "relaxed-publication", isDirectory: true)
        XCTAssertThrowsError(try VectorPolicyRequalification.publish(
            targetSpec: fixture.targetSpec,
            evaluationCriteria: RLEvaluationCriteria(
                minimumSuccessRate: 0,
                minimumMeanEpisodeLengthFraction: 0),
            candidateDirectory: fixture.candidate.path,
            parentCheckpointDirectory: fixture.parent.path,
            evaluationReportPaths: qualification.reports.map(\.path),
            aggregatePath: qualification.aggregate.path,
            outputDirectory: relaxedOutput.path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: relaxedOutput.path))

        let output = fixture.root.appendingPathComponent(
            "published", isDirectory: true)
        let manifest = try VectorPolicyRequalification.publish(
            targetSpec: fixture.targetSpec,
            evaluationCriteria: fixture.criteria,
            candidateDirectory: fixture.candidate.path,
            parentCheckpointDirectory: fixture.parent.path,
            evaluationReportPaths: qualification.reports.map(\.path),
            aggregatePath: qualification.aggregate.path,
            outputDirectory: output.path)

        XCTAssertEqual(manifest.qualification?.reports.count, 4)
        XCTAssertEqual(
            try VectorPPOTrainer.checkpointFingerprint(directory: output.path),
            fixture.preparation.candidateCheckpointFingerprint)
        XCTAssertTrue(FileManager.default.fileExists(atPath: output
            .appendingPathComponent("deployment-manifest.json").path))
        for seed in fixture.preparation.qualificationPlan.evaluationSeeds {
            XCTAssertTrue(FileManager.default.fileExists(atPath: output
                .appendingPathComponent(
                    "qualification/eval-seed-\(seed).json").path))
        }
        let sealed = try decode(
            VectorPolicyRequalificationManifest.self,
            at: output.appendingPathComponent(
                VectorPolicyRequalification.manifestFileName))
        XCTAssertEqual(sealed, manifest)
        XCTAssertNotNil(sealed.qualification)
        XCTAssertNoThrow(try VectorPolicyRequalification.verify(
            targetSpec: fixture.targetSpec,
            evaluationCriteria: fixture.criteria,
            bundleDirectory: output.path,
            parentCheckpointDirectory: fixture.parent.path))

        XCTAssertThrowsError(try VectorPolicyRequalification.publish(
            targetSpec: fixture.targetSpec,
            evaluationCriteria: fixture.criteria,
            candidateDirectory: fixture.candidate.path,
            parentCheckpointDirectory: fixture.parent.path,
            evaluationReportPaths: qualification.reports.map(\.path),
            aggregatePath: qualification.aggregate.path,
            outputDirectory: output.path))
        XCTAssertThrowsError(try VectorPolicyRequalification.publish(
            targetSpec: fixture.targetSpec,
            evaluationCriteria: fixture.criteria,
            candidateDirectory: fixture.candidate.path,
            parentCheckpointDirectory: fixture.parent.path,
            evaluationReportPaths: qualification.reports.map(\.path),
            aggregatePath: qualification.aggregate.path,
            outputDirectory: fixture.candidate.appendingPathComponent(
                "nested-publication").path))

        let deploymentURL = output.appendingPathComponent(
            VectorPolicyDeploymentBundle.manifestFileName)
        let originalDeployment = try Data(contentsOf: deploymentURL)
        var deployment = try decode(
            VectorPolicyDeploymentManifest.self, at: deploymentURL)
        deployment.policyFile = "different-policy.safetensors"
        try write(deployment, to: deploymentURL)
        XCTAssertThrowsError(try VectorPolicyRequalification.verify(
            targetSpec: fixture.targetSpec,
            evaluationCriteria: fixture.criteria,
            bundleDirectory: output.path,
            parentCheckpointDirectory: fixture.parent.path))
        try originalDeployment.write(to: deploymentURL, options: .atomic)

        let sealedReport = output.appendingPathComponent(
            "qualification/eval-seed-101.json")
        var reportBytes = try Data(contentsOf: sealedReport)
        reportBytes.append(0x20)
        try reportBytes.write(to: sealedReport, options: .atomic)
        XCTAssertThrowsError(try VectorPolicyRequalification.verify(
            targetSpec: fixture.targetSpec,
            evaluationCriteria: fixture.criteria,
            bundleDirectory: output.path,
            parentCheckpointDirectory: fixture.parent.path))
    }

    func testPublishRejectsIncompleteMixedRevisionInconsistentAndTamperedInputs()
        throws
    {
        do {
            let fixture = try makeFixture()
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let qualification = try writeQualification(for: fixture)
            XCTAssertThrowsError(try VectorPolicyRequalification.publish(
                targetSpec: fixture.targetSpec,
                evaluationCriteria: fixture.criteria,
                candidateDirectory: fixture.candidate.path,
                parentCheckpointDirectory: fixture.parent.path,
                evaluationReportPaths: Array(qualification.reports.prefix(3))
                    .map(\.path),
                aggregatePath: qualification.aggregate.path,
                outputDirectory: fixture.root.appendingPathComponent(
                    "incomplete-output").path))
        }

        do {
            let fixture = try makeFixture()
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            var qualification = try writeQualification(for: fixture)
            var mixed = try decode(
                PPOEvaluationMetrics.self, at: qualification.reports[3])
            mixed.taskRevision = fixture.targetSpec.revision + 1
            try write(mixed, to: qualification.reports[3])
            XCTAssertThrowsError(try VectorPolicyRequalification.publish(
                targetSpec: fixture.targetSpec,
                evaluationCriteria: fixture.criteria,
                candidateDirectory: fixture.candidate.path,
                parentCheckpointDirectory: fixture.parent.path,
                evaluationReportPaths: qualification.reports.map(\.path),
                aggregatePath: qualification.aggregate.path,
                outputDirectory: fixture.root.appendingPathComponent(
                    "mixed-output").path))

            qualification = try writeQualification(for: fixture)
            var inconsistent = try decode(
                PPOEvaluationMetrics.self, at: qualification.reports[0])
            inconsistent.successes = 511
            try write(inconsistent, to: qualification.reports[0])
            XCTAssertThrowsError(try VectorPolicyRequalification.publish(
                targetSpec: fixture.targetSpec,
                evaluationCriteria: fixture.criteria,
                candidateDirectory: fixture.candidate.path,
                parentCheckpointDirectory: fixture.parent.path,
                evaluationReportPaths: qualification.reports.map(\.path),
                aggregatePath: qualification.aggregate.path,
                outputDirectory: fixture.root.appendingPathComponent(
                    "inconsistent-output").path))
        }

        do {
            let fixture = try makeFixture()
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let qualification = try writeQualification(for: fixture)
            let policyURL = fixture.candidate.appendingPathComponent(
                "policy.safetensors")
            var policy = try Data(contentsOf: policyURL)
            policy.append(0xff)
            try policy.write(to: policyURL, options: .atomic)
            XCTAssertThrowsError(try VectorPolicyRequalification.publish(
                targetSpec: fixture.targetSpec,
                evaluationCriteria: fixture.criteria,
                candidateDirectory: fixture.candidate.path,
                parentCheckpointDirectory: fixture.parent.path,
                evaluationReportPaths: qualification.reports.map(\.path),
                aggregatePath: qualification.aggregate.path,
                outputDirectory: fixture.root.appendingPathComponent(
                    "tampered-output").path))
        }

        do {
            let fixture = try makeFixture()
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let qualification = try writeQualification(for: fixture)
            let metadataURL = fixture.candidate.appendingPathComponent(
                "metadata.json")
            var metadata = try decode(
                VectorPolicyMetadata.self, at: metadataURL)
            metadata.normalizer.mean[0] = 0.25
            try write(metadata, to: metadataURL)

            let manifestURL = fixture.candidate.appendingPathComponent(
                VectorPolicyRequalification.manifestFileName)
            var manifest = try decode(
                VectorPolicyRequalificationManifest.self, at: manifestURL)
            let metadataData = try Data(contentsOf: metadataURL)
            manifest.candidateMetadataSHA256 = sha256(metadataData)
            manifest.candidateCheckpointFingerprint = try VectorPPOTrainer
                .checkpointFingerprint(directory: fixture.candidate.path)
            try write(manifest, to: manifestURL)

            XCTAssertThrowsError(try VectorPolicyRequalification.publish(
                targetSpec: fixture.targetSpec,
                evaluationCriteria: fixture.criteria,
                candidateDirectory: fixture.candidate.path,
                parentCheckpointDirectory: fixture.parent.path,
                evaluationReportPaths: qualification.reports.map(\.path),
                aggregatePath: qualification.aggregate.path,
                outputDirectory: fixture.root.appendingPathComponent(
                    "self-consistent-metadata-tamper").path)) { error in
                XCTAssertTrue(String(describing: error).contains(
                    "zero-update transform"))
            }
        }

        do {
            let fixture = try makeFixture()
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let qualification = try writeQualification(for: fixture)
            let manifestURL = fixture.candidate.appendingPathComponent(
                VectorPolicyRequalification.manifestFileName)
            var manifest = try decode(
                VectorPolicyRequalificationManifest.self, at: manifestURL)
            manifest.sourceTaskRevision += 10
            manifest.parentTrainingUpdates += 1
            try write(manifest, to: manifestURL)
            XCTAssertThrowsError(try VectorPolicyRequalification.publish(
                targetSpec: fixture.targetSpec,
                evaluationCriteria: fixture.criteria,
                candidateDirectory: fixture.candidate.path,
                parentCheckpointDirectory: fixture.parent.path,
                evaluationReportPaths: qualification.reports.map(\.path),
                aggregatePath: qualification.aggregate.path,
                outputDirectory: fixture.root.appendingPathComponent(
                    "lineage-tamper").path))
        }
    }

    private func makeFixture() throws -> Fixture {
        let root = temporaryDirectory()
        let parent = root.appendingPathComponent("parent", isDirectory: true)
        let candidate = root.appendingPathComponent(
            "candidate", isDirectory: true)
        let sourceSpec = makeSpec(revision: 7)
        let targetSpec = makeSpec(revision: 8)
        try writeParent(at: parent, spec: sourceSpec)
        let fingerprint = try VectorPPOTrainer.checkpointFingerprint(
            directory: parent.path)
        let plan = VectorPolicyRequalificationPlan(
            evaluationSeeds: [101, 102, 103, 104],
            evaluationEnvironments: 8)
        let criteria = RLEvaluationCriteria(
            minimumSuccessRate: 0.8,
            minimumMeanEpisodeLengthFraction: 0.9)
        let preparation = try VectorPolicyRequalification.prepare(
            targetSpec: targetSpec,
            parentCheckpointDirectory: parent.path,
            outputDirectory: candidate.path,
            expectedParentCheckpointFingerprint: fingerprint,
            inferenceBatchSize: 8,
            declaredSourceCommit: String(repeating: "c", count: 40),
            qualificationPlan: plan,
            evaluationCriteria: criteria)
        return Fixture(
            root: root, parent: parent, candidate: candidate,
            targetSpec: targetSpec, preparation: preparation,
            criteria: criteria)
    }

    private func makeSpec(revision: Int) -> RLTaskSpec {
        RLTaskSpec(
            id: "fixture-task-v0", revision: revision,
            numEnvironments: 1,
            observation: .init(name: "policy", shape: [3]),
            action: .init(name: "action", shape: [2]),
            maxEpisodeSteps: 100,
            simulationStep: 0.005,
            controlDecimation: 4,
            configurationValues: ["plant": 1])
    }

    private func writeParent(at directory: URL, spec: RLTaskSpec) throws {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let metadata = VectorPolicyMetadata(
            architectureVersion: VectorActorCritic.architectureVersion,
            task: spec.id, taskRevision: spec.revision,
            taskConfiguration: spec.configurationValues,
            observationDimension: spec.observation.elementCount,
            actionDimension: spec.action.elementCount,
            simulationStep: spec.simulationStep,
            controlDecimation: spec.controlDecimation,
            maxEpisodeSteps: spec.maxEpisodeSteps,
            ppo: VectorPPOConfig(
                updates: 12, rolloutSteps: 4, updateEpochs: 1,
                minibatchSize: 4, learningRate: 1e-3,
                hiddenSize: 8, hiddenDimensions: [8, 8, 8],
                normalizeObservations: false,
                checkpointInterval: 1, seed: 77),
            normalizer: RunningNormalizerSnapshot(
                count: 0, mean: [0, 0, 0], variance: [1, 1, 1]))
        try write(metadata, to: directory.appendingPathComponent(
            "metadata.json"))
        try Data([1, 7, 2, 9, 4, 3]).write(
            to: directory.appendingPathComponent("policy.safetensors"),
            options: .atomic)
        try write(
            VectorPPOTrainingState(
                completedUpdates: 12, environmentSteps: 4_608,
                optimizerSteps: 36, adaptiveLearningRate: 1e-3),
            to: directory.appendingPathComponent("training-state.json"))
    }

    private func writeQualification(for fixture: Fixture) throws
        -> (reports: [URL], aggregate: URL)
    {
        let directory = fixture.root.appendingPathComponent(
            "qualification", isDirectory: true)
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let metadata = try decode(
            VectorPolicyMetadata.self,
            at: fixture.candidate.appendingPathComponent("metadata.json"))
        var reports = [PPOEvaluationMetrics]()
        var urls = [URL]()
        for seed in fixture.preparation.qualificationPlan.evaluationSeeds {
            let report = PPOEvaluationMetrics(
                provenanceVersion: 3,
                task: fixture.targetSpec.id,
                taskRevision: fixture.targetSpec.revision,
                checkpointTaskConfiguration:
                    fixture.targetSpec.configurationValues,
                evaluationTaskConfiguration:
                    fixture.targetSpec.configurationValues,
                taskConfigurationTransferred: false,
                checkpointDirectory: fixture.candidate.path,
                checkpointFingerprint:
                    fixture.preparation.candidateCheckpointFingerprint,
                initializationCheckpoint:
                    metadata.ppo.initializationCheckpoint,
                trainingSeed: metadata.ppo.seed,
                evaluationSeed: seed,
                evaluationEnvironments:
                    fixture.preparation.qualificationPlan
                        .evaluationEnvironments,
                trainingUpdates: 0,
                trainingEnvironmentSteps: 0,
                episodes: 512,
                successes: 512,
                successRate: 1,
                meanReturn: 25,
                meanEpisodeLength: 100,
                taskMetrics: [:],
                acceptance: PPOEvaluationAcceptance(
                    passed: true, failures: []))
            let url = directory.appendingPathComponent(
                "eval-seed-\(seed).json")
            try write(report, to: url)
            reports.append(report)
            urls.append(url)
        }
        let aggregate = try PPOCheckpointEvaluationAggregate.make(reports)
        let aggregateURL = directory.appendingPathComponent("aggregate.json")
        try write(aggregate, to: aggregateURL)
        return (urls, aggregateURL)
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(
            "avbd-policy-requalification-\(UUID().uuidString)",
            isDirectory: true)
    }

    private func write<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: url, options: .atomic)
    }

    private func decode<T: Decodable>(_ type: T.Type, at url: URL) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: url))
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
