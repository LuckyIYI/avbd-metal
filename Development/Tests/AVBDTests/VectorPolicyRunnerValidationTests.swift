import Foundation
import XCTest
@testable import MLXRL

final class VectorPolicyRunnerValidationTests: XCTestCase {
    private func trackedRelease(
        root: URL, identifier: String
    ) throws -> PolicyBundleReleaseIndex.Release {
        let index = try PolicyBundleReleaseIndex.load(from:
            root.appendingPathComponent(
                "checkpoints/policy-release-index.json"))
        return try XCTUnwrap(index.releases.first {
            $0.bundleIdentifier == identifier
        })
    }

    private func metadata(
        observationDimension: Int = 3,
        actionDimension: Int = 2,
        hiddenDimensions: [Int] = [8, 8, 8],
        initialActionStd: Float = 0.5
    ) -> VectorPolicyMetadata {
        VectorPolicyMetadata(
            architectureVersion: VectorActorCritic.architectureVersion,
            task: "runner-validation-v0",
            taskRevision: 1,
            taskConfiguration: ["fixture": 1],
            observationDimension: observationDimension,
            actionDimension: actionDimension,
            simulationStep: 0.005,
            controlDecimation: 4,
            maxEpisodeSteps: 100,
            inferenceBatchSize: 4,
            ppo: VectorPPOConfig(
                updates: 0,
                rolloutSteps: 4,
                updateEpochs: 1,
                minibatchSize: 4,
                hiddenSize: hiddenDimensions[0],
                hiddenDimensions: hiddenDimensions,
                initialActionStd: initialActionStd,
                normalizeObservations: true,
                checkpointInterval: 1),
            normalizer: RunningNormalizerSnapshot(
                count: 0,
                mean: [Double](repeating: 0, count: observationDimension),
                variance: [Double](repeating: 1,
                                   count: observationDimension)))
    }

    private func temporaryDirectory(_ stem: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "avbd-\(stem)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeRunnerFixture(
        metadata: VectorPolicyMetadata, to directory: URL
    ) throws {
        try JSONEncoder().encode(metadata).write(
            to: directory.appendingPathComponent("metadata.json"))
        // Deliberately not a safetensors file. Metadata failures must surface
        // before this payload reaches MLX.
        try Data([0xde, 0xad, 0xbe, 0xef]).write(
            to: directory.appendingPathComponent("policy.safetensors"))
    }

    func testRunnerRejectsInvalidPPOBeforeParsingPolicy() throws {
        let directory = try temporaryDirectory("runner-invalid-ppo")
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeRunnerFixture(
            metadata: metadata(initialActionStd: 0), to: directory)

        XCTAssertThrowsError(try VectorPolicyRunner(
            checkpointDirectory: directory.path)) { error in
            XCTAssertTrue(String(describing: error).contains(
                "PPO inference configuration"))
        }
    }

    func testRunnerRejectsOverflowBeforeParsingPolicy() throws {
        let directory = try temporaryDirectory("runner-overflow")
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeRunnerFixture(
            metadata: metadata(hiddenDimensions: [Int.max, 2, 2]),
            to: directory)

        XCTAssertThrowsError(try VectorPolicyRunner(
            checkpointDirectory: directory.path)) { error in
            XCTAssertTrue(String(describing: error).contains("overflows Int"))
        }
    }

    func testRunnerRejectsNormalizerReconstructionOverflowBeforePolicy()
        throws {
        let directory = try temporaryDirectory("runner-normalizer-overflow")
        defer { try? FileManager.default.removeItem(at: directory) }
        var invalidMetadata = metadata()
        invalidMetadata.normalizer = RunningNormalizerSnapshot(
            count: 3,
            mean: [Double](repeating: 0, count: 3),
            variance: [Double](
                repeating: .greatestFiniteMagnitude, count: 3))
        try writeRunnerFixture(metadata: invalidMetadata, to: directory)

        XCTAssertThrowsError(try VectorPolicyRunner(
            checkpointDirectory: directory.path)) { error in
            XCTAssertTrue(String(describing: error).contains(
                "observation normalizer"))
        }
    }

    func testRunnerRejectsNonFiniteFloat32PolicyTensor() throws {
        guard ProcessInfo.processInfo.environment[
            "AVBD_MLX_INTEGRATION_TESTS"] == "1" else {
            throw XCTSkip("requires an Xcode-packaged MLX default.metallib")
        }
        let packageRoot = TestPaths.repositoryRoot
        let source = packageRoot.appendingPathComponent(
            "checkpoints/arachne15-goal-v1", isDirectory: true)
        let directory = try temporaryDirectory("runner-non-finite-policy")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.copyItem(
            at: source.appendingPathComponent("metadata.json"),
            to: directory.appendingPathComponent("metadata.json"))

        var policyData = try Data(contentsOf: source.appendingPathComponent(
            "policy.safetensors"))
        XCTAssertGreaterThan(policyData.count, 8)
        var headerLength: UInt64 = 0
        for index in 0..<8 {
            headerLength |= UInt64(policyData[index]) << UInt64(index * 8)
        }
        let headerEnd = 8 + Int(headerLength)
        let header = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: policyData[8..<headerEnd]) as? [String: Any])
        let descriptor = try XCTUnwrap(
            header["actor1.bias"] as? [String: Any])
        let offsets = try XCTUnwrap(descriptor["data_offsets"] as? [Int])
        XCTAssertEqual(offsets.count, 2)
        let valueOffset = headerEnd + offsets[0]
        var nanBits = Float.nan.bitPattern.littleEndian
        let nanData = withUnsafeBytes(of: &nanBits) { Data($0) }
        policyData.replaceSubrange(
            valueOffset..<(valueOffset + nanData.count), with: nanData)
        try policyData.write(
            to: directory.appendingPathComponent("policy.safetensors"))

        XCTAssertThrowsError(try VectorPolicyRunner(
            checkpointDirectory: directory.path)) { error in
            XCTAssertTrue(String(describing: error).contains(
                "contains non-finite values"))
        }
    }

    func testRoutingRequiresFiniteUnitIntervalsAndConvexComposition() throws {
        XCTAssertNoThrow(try VectorPolicyRunner.validateRouting(
            rowCount: 2,
            actionDimension: 2,
            expertGates: [1, 0.5],
            expertActionMask: [1, 0],
            standExpertGates: [0.75, 0.5],
            standExpertActionMask: [0, 1],
            auxiliaryExpertGates: nil,
            auxiliaryExpertActionMask: nil))

        XCTAssertThrowsError(try VectorPolicyRunner.validateRouting(
            rowCount: 1,
            actionDimension: 1,
            expertGates: [.nan],
            expertActionMask: [1],
            standExpertGates: nil,
            standExpertActionMask: nil,
            auxiliaryExpertGates: nil,
            auxiliaryExpertActionMask: nil))
        XCTAssertThrowsError(try VectorPolicyRunner.validateRouting(
            rowCount: 1,
            actionDimension: 1,
            expertGates: [0.6],
            expertActionMask: nil,
            standExpertGates: [0.5],
            standExpertActionMask: nil,
            auxiliaryExpertGates: nil,
            auxiliaryExpertActionMask: nil)) { error in
            XCTAssertTrue(String(describing: error).contains(
                "convex actor composition"))
        }
        XCTAssertThrowsError(try VectorPolicyRunner.validateRouting(
            rowCount: 1,
            actionDimension: 1,
            expertGates: [0.5000004],
            expertActionMask: nil,
            standExpertGates: [0.5000004],
            standExpertActionMask: nil,
            auxiliaryExpertGates: nil,
            auxiliaryExpertActionMask: nil))
        XCTAssertThrowsError(try VectorPolicyRunner.validateRouting(
            rowCount: 1,
            actionDimension: 1,
            expertGates: [0],
            expertActionMask: [.infinity],
            standExpertGates: nil,
            standExpertActionMask: nil,
            auxiliaryExpertGates: nil,
            auxiliaryExpertActionMask: nil))
    }

    func testDeploymentManifestRequiresCanonicalCheckpointNames() throws {
        let directory = try temporaryDirectory("deployment-file-names")
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = VectorPolicyDeploymentManifest(
            schemaVersion: 1,
            task: "runner-validation-v0",
            taskRevision: 1,
            checkpointFingerprint: String(repeating: "0", count: 64),
            policySHA256: String(repeating: "0", count: 64),
            policyFile: "alternate-policy.safetensors",
            metadataFile: VectorPolicyDeploymentBundle.metadataFileName,
            trainingStateFile:
                VectorPolicyDeploymentBundle.trainingStateFileName,
            architectureVersion: VectorActorCritic.architectureVersion,
            observationDimension: 3,
            actionDimension: 2,
            simulationStepSeconds: 0.005,
            controlDecimation: 4,
            controlFrequencyHz: 50,
            normalizesObservations: true,
            observationNormalizationClip: 10,
            actionDistribution: "squashed-gaussian",
            taskConfiguration: ["fixture": 1],
            trainingUpdates: 0,
            trainingEnvironmentSteps: 0)
        try JSONEncoder().encode(manifest).write(
            to: directory.appendingPathComponent(
                VectorPolicyDeploymentBundle.manifestFileName))

        XCTAssertThrowsError(try VectorPolicyDeploymentRuntime(
            bundleDirectory: directory.path)) { error in
            XCTAssertTrue(String(describing: error).contains(
                "canonical checkpoint file names"))
        }
    }

    func testDeploymentRuntimeRejectsSemanticallyValidButUncommissionedManifestBytes()
        throws
    {
        let packageRoot = TestPaths.repositoryRoot
        let source = packageRoot.appendingPathComponent(
            "checkpoints/humanoid-isaac-flat-v2/deployment-manifest.json")
        guard FileManager.default.fileExists(atPath: source.path) else {
            throw XCTSkip("accepted H1 deployment manifest is not installed")
        }
        let directory = try temporaryDirectory("deployment-manifest-bytes")
        defer { try? FileManager.default.removeItem(at: directory) }
        var bytes = try Data(contentsOf: source)
        // JSON whitespace preserves every decoded claim but changes the exact
        // commissioned artifact. Rejection must happen before model loading.
        bytes.append(0x0a)
        try bytes.write(to: directory.appendingPathComponent(
            VectorPolicyDeploymentBundle.manifestFileName))
        let entry = try trackedRelease(
            root: packageRoot, identifier: "humanoid-isaac-flat-v2")

        XCTAssertThrowsError(try VectorPolicyDeploymentRuntime(
            bundleDirectory: directory.path,
            expectedTask: "humanoid-isaac-flat-v0",
            expectedTaskRevision: entry.expectedTaskRevision,
            expectedCheckpointFingerprint:
                entry.expectedCheckpointFingerprint,
            expectedDeploymentManifestSHA256:
                entry.expectedDeploymentManifestSHA256)) { error in
            XCTAssertTrue(String(describing: error).contains(
                "manifest bytes do not match the commissioned policy"))
        }
    }

    func testExportUsesRunnerMetadataContract() throws {
        let root = try temporaryDirectory("deployment-metadata-contract")
        defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = root.appendingPathComponent(
            "checkpoint", isDirectory: true)
        let output = root.appendingPathComponent("bundle", isDirectory: true)
        try FileManager.default.createDirectory(
            at: checkpoint, withIntermediateDirectories: false)
        var invalidMetadata = metadata()
        invalidMetadata.simulationStep = .greatestFiniteMagnitude
        try writeRunnerFixture(metadata: invalidMetadata, to: checkpoint)
        try JSONEncoder().encode(VectorPPOTrainingState(
            completedUpdates: 0, environmentSteps: 0)).write(
                to: checkpoint.appendingPathComponent(
                    VectorPolicyDeploymentBundle.trainingStateFileName))

        XCTAssertThrowsError(try VectorPolicyDeploymentBundle.export(
            checkpointDirectory: checkpoint.path,
            outputDirectory: output.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
    }

    func testInMemoryFingerprintMatchesDirectoryFingerprint() throws {
        let directory = try temporaryDirectory("in-memory-fingerprint")
        defer { try? FileManager.default.removeItem(at: directory) }
        let metadataData = Data([1, 2, 3])
        let policyData = Data([4, 5, 6])
        let trainingStateData = Data([7, 8, 9])
        try metadataData.write(to: directory.appendingPathComponent(
            VectorPolicyDeploymentBundle.metadataFileName))
        try policyData.write(to: directory.appendingPathComponent(
            VectorPolicyDeploymentBundle.policyFileName))
        try trainingStateData.write(to: directory.appendingPathComponent(
            VectorPolicyDeploymentBundle.trainingStateFileName))

        XCTAssertEqual(
            try VectorPPOTrainer.checkpointFingerprint(
                directory: directory.path),
            VectorPPOTrainer.checkpointFingerprint(
                metadataData: metadataData,
                policyData: policyData,
                trainingStateData: trainingStateData))
    }

    func testPackagedPolicyBundlesConstructGeneratedReplayPages() throws {
        guard ProcessInfo.processInfo.environment[
            "AVBD_MLX_INTEGRATION_TESTS"] == "1" else {
            throw XCTSkip("requires an Xcode-packaged MLX default.metallib")
        }
        let packageRoot = TestPaths.repositoryRoot
        let checkpointRoot = packageRoot.appendingPathComponent(
            "checkpoints", isDirectory: true)
        let index = try PolicyBundleReleaseIndex.load(from:
            checkpointRoot.appendingPathComponent(
                "policy-release-index.json"))

        for relative in [
            "humanoid-isaac-flat-v2",
            "arachne15-velocity-v1",
            "arachne15-goal-v1",
            "external/unitree-h1",
        ] {
            let bundle = try PolicyBundleLoader.load(directory:
                checkpointRoot.appendingPathComponent(
                    relative, isDirectory: true))
            let release = try XCTUnwrap(index.release(for: bundle))
            let session = try PolicyBundleReplayFactory.make(
                bundle: bundle, release: release)
            try session.step()
            for camera in bundle.manifest.presentation.cameraPresets {
                XCTAssertNotNil(
                    session.anchor(named: camera.anchor),
                    "\(bundle.manifest.identifier): \(camera.anchor)")
            }
            for metric in bundle.manifest.presentation.metrics {
                XCTAssertNotNil(
                    session.values[metric.source],
                    "\(bundle.manifest.identifier): \(metric.source)")
            }
        }
    }
}
