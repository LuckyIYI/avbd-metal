import Foundation
import XCTest
import SimCore
import RL
@testable import MLXRL

final class PolicyBundleTests: XCTestCase {
    private var root: URL {
        TestPaths.repositoryRoot
    }

    func testTrackedReleaseBundlesAreSelfContainedAndIndexed() throws {
        let checkpointRoot = root.appendingPathComponent(
            "checkpoints", isDirectory: true)
        let index = try PolicyBundleReleaseIndex.load(from:
            checkpointRoot.appendingPathComponent(
                "policy-release-index.json"))
        XCTAssertEqual(index.schemaVersion, 1)
        XCTAssertEqual(index.releases.count, 4)

        let expected: [String: String] = [
            "humanoid-isaac-flat-v2": "humanoid-isaac-flat-v2",
            "arachne15-velocity-v1": "arachne15-velocity-v1",
            "arachne15-goal-v1": "arachne15-goal-v1",
            "unitree-h1-sim2sim-v0": "external/unitree-h1",
        ]
        for (identifier, relativeDirectory) in expected {
            let bundle = try PolicyBundleLoader.load(
                directory: checkpointRoot.appendingPathComponent(
                    relativeDirectory, isDirectory: true))
            XCTAssertEqual(bundle.manifest.identifier, identifier)
            XCTAssertNotNil(index.release(for: bundle))
            XCTAssertFalse(bundle.manifest.presentation.cameraPresets.isEmpty)
            for role in bundle.manifest.runtime.files.keys {
                XCTAssertTrue(try bundle.fileURL(role: role)
                    .standardizedFileURL.path.hasPrefix(
                        bundle.directory.path + "/"))
            }
        }
    }

    func testBundleCannotSelfCertifyAfterManifestChange() throws {
        let source = root.appendingPathComponent(
            "checkpoints/humanoid-isaac-flat-v2", isDirectory: true)
        let temporary = try copyToTemporary(source)
        defer { try? FileManager.default.removeItem(
            at: temporary.deletingLastPathComponent()) }

        let index = try PolicyBundleReleaseIndex.load(from:
            root.appendingPathComponent(
                "checkpoints/policy-release-index.json"))
        let original = try PolicyBundleLoader.load(directory: temporary)
        XCTAssertNotNil(index.release(for: original))

        let manifestURL = temporary.appendingPathComponent(
            PolicyBundleManifest.fileName)
        var manifest = original.manifest
        manifest.summary += " changed"
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
        let modified = try PolicyBundleLoader.load(directory: temporary)
        XCTAssertNil(index.release(for: modified))
    }

    func testImportContentIdentityIncludesPolicyBytes() throws {
        let source = root.appendingPathComponent(
            "checkpoints/external/unitree-h1", isDirectory: true)
        let temporary = try copyToTemporary(source)
        defer { try? FileManager.default.removeItem(
            at: temporary.deletingLastPathComponent()) }

        let original = try PolicyBundleLoader.load(directory: temporary)
        let policyURL = temporary.appendingPathComponent("policy.safetensors")
        var policy = try Data(contentsOf: policyURL)
        policy[policy.startIndex] ^= 0x01
        try policy.write(to: policyURL, options: .atomic)
        let modified = try PolicyBundleLoader.load(directory: temporary)

        XCTAssertEqual(original.manifestSHA256, modified.manifestSHA256)
        XCTAssertNotEqual(original.contentSHA256, modified.contentSHA256)
    }

    func testRuntimePathTraversalFailsClosed() throws {
        let source = root.appendingPathComponent(
            "checkpoints/external/unitree-h1", isDirectory: true)
        let temporary = try copyToTemporary(source)
        defer { try? FileManager.default.removeItem(
            at: temporary.deletingLastPathComponent()) }
        let outside = temporary.deletingLastPathComponent()
            .appendingPathComponent("outside.safetensors")
        try Data([0]).write(to: outside)

        let manifestURL = temporary.appendingPathComponent(
            PolicyBundleManifest.fileName)
        var manifest = try JSONDecoder().decode(
            PolicyBundleManifest.self, from: Data(contentsOf: manifestURL))
        manifest.runtime.files["policy"] = "../outside.safetensors"
        try JSONEncoder().encode(manifest).write(
            to: manifestURL, options: .atomic)
        XCTAssertThrowsError(try PolicyBundleLoader.load(directory: temporary))
    }

    func testUnknownManifestFieldFailsClosed() throws {
        let source = root.appendingPathComponent(
            "checkpoints/humanoid-isaac-flat-v2", isDirectory: true)
        let temporary = try copyToTemporary(source)
        defer { try? FileManager.default.removeItem(
            at: temporary.deletingLastPathComponent()) }
        let manifestURL = temporary.appendingPathComponent(
            PolicyBundleManifest.fileName)
        var json = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as? [String: Any])
        json["futureBehavior"] = true
        try JSONSerialization.data(
            withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            .write(to: manifestURL)
        XCTAssertThrowsError(try PolicyBundleLoader.load(directory: temporary))
    }

    func testAnySymlinkInImportedBundleFailsClosed() throws {
        let source = root.appendingPathComponent(
            "checkpoints/external/unitree-h1", isDirectory: true)
        let temporary = try copyToTemporary(source)
        defer { try? FileManager.default.removeItem(
            at: temporary.deletingLastPathComponent()) }
        try FileManager.default.createSymbolicLink(
            at: temporary.appendingPathComponent("alias"),
            withDestinationURL: temporary.appendingPathComponent("LICENSE"))
        XCTAssertThrowsError(try PolicyBundleLoader.load(directory: temporary))
    }

    func testManifestSymlinkFailsBeforeManifestRead() throws {
        let source = root.appendingPathComponent(
            "checkpoints/external/unitree-h1", isDirectory: true)
        let temporary = try copyToTemporary(source)
        let parent = temporary.deletingLastPathComponent()
        defer { try? FileManager.default.removeItem(at: parent) }
        let manifestURL = temporary.appendingPathComponent(
            PolicyBundleManifest.fileName)
        let outside = parent.appendingPathComponent("outside-manifest.json")
        try FileManager.default.moveItem(at: manifestURL, to: outside)
        try FileManager.default.createSymbolicLink(
            at: manifestURL, withDestinationURL: outside)

        XCTAssertThrowsError(try PolicyBundleLoader.load(directory: temporary))
    }

    func testSymlinkedBundleRootFailsClosed() throws {
        let source = root.appendingPathComponent(
            "checkpoints/external/unitree-h1", isDirectory: true)
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: parent) }
        let link = parent.appendingPathComponent("bundle", isDirectory: true)
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: source)

        XCTAssertThrowsError(try PolicyBundleLoader.load(directory: link))
    }

    func testReleaseIndexRejectsTraversalAndIncompleteTrustAnchors() throws {
        let source = root.appendingPathComponent(
            "checkpoints/policy-release-index.json")
        var index = try JSONDecoder().decode(
            PolicyBundleReleaseIndex.self, from: Data(contentsOf: source))
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let candidate = directory.appendingPathComponent("index.json")

        index.releases[0].bundleRelativeDirectory = "../outside"
        try JSONEncoder().encode(index).write(to: candidate)
        XCTAssertThrowsError(try PolicyBundleReleaseIndex.load(from: candidate))

        index = try JSONDecoder().decode(
            PolicyBundleReleaseIndex.self, from: Data(contentsOf: source))
        index.releases[0].expectedCheckpointFingerprint = nil
        try JSONEncoder().encode(index).write(to: candidate)
        XCTAssertThrowsError(try PolicyBundleReleaseIndex.load(from: candidate))

        var json = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: Data(contentsOf: source)) as? [String: Any])
        var releases = try XCTUnwrap(json["releases"] as? [[String: Any]])
        releases[0]["unitreeH1ReleaseIdentity"] = NSNull()
        json["releases"] = releases
        try JSONSerialization.data(withJSONObject: json).write(to: candidate)
        XCTAssertThrowsError(try PolicyBundleReleaseIndex.load(from: candidate))
    }

    func testPresentationMustUseRuntimeDeclaredCapabilities() throws {
        let bundle = try PolicyBundleLoader.load(directory:
            root.appendingPathComponent(
                "checkpoints/humanoid-isaac-flat-v2", isDirectory: true))
        let capabilities = RLReplayCapabilities(
            anchors: ["robot", "course"],
            values: [
                "task/root-height", "task/planar-speed",
                "task/command-forward", "replay/control-step",
            ],
            commands: ["throw-projectile"])
        XCTAssertNoThrow(try PolicyBundleReplayFactory.validatePresentation(
            bundle.manifest.presentation, capabilities: capabilities))

        var presentation = bundle.manifest.presentation
        presentation.metrics[0].source =
            "metric/reward/tracking_linear_velocity"
        XCTAssertNoThrow(try PolicyBundleReplayFactory.validatePresentation(
            presentation, capabilities: capabilities))
        presentation.metrics[0].source = "metric/"
        XCTAssertThrowsError(try PolicyBundleReplayFactory.validatePresentation(
            presentation, capabilities: capabilities))

        presentation = bundle.manifest.presentation
        presentation.cameraPresets[0].anchor = "unknown"
        XCTAssertThrowsError(try PolicyBundleReplayFactory.validatePresentation(
            presentation, capabilities: capabilities))
        presentation = bundle.manifest.presentation
        presentation.controls[0].command = "delete-everything"
        XCTAssertThrowsError(try PolicyBundleReplayFactory.validatePresentation(
            presentation, capabilities: capabilities))
        presentation = bundle.manifest.presentation
        presentation.metrics[0].source = "task/private-value"
        XCTAssertThrowsError(try PolicyBundleReplayFactory.validatePresentation(
            presentation, capabilities: capabilities))
    }

    func testCameraTargetUsesWorldAndAnchorRelativeManifestData() {
        let world = PolicyBundleManifest.CameraPreset(
            id: "world", label: "World", anchor: "world",
            target: [4, 5, 6], offset: [0.5, -1, 2], distance: 3,
            azimuth: 0, elevation: 0)
        XCTAssertEqual(
            PolicyBundleReplayFactory.cameraTarget(
                preset: world, anchorValue: F3(100, 100, 100)),
            F3(4.5, 4, 8))

        var relative = world
        relative.anchor = "robot"
        XCTAssertEqual(
            PolicyBundleReplayFactory.cameraTarget(
                preset: relative, anchorValue: F3(10, 20, 30)),
            F3(14.5, 24, 38))
        XCTAssertNil(PolicyBundleReplayFactory.cameraTarget(
            preset: relative, anchorValue: nil))
    }

    func testUnitreeRuntimeRejectsIgnoredRobustnessProbeSetting() throws {
        let source = root.appendingPathComponent(
            "checkpoints/external/unitree-h1", isDirectory: true)
        let temporary = try copyToTemporary(source)
        defer { try? FileManager.default.removeItem(
            at: temporary.deletingLastPathComponent()) }
        let manifestURL = temporary.appendingPathComponent(
            PolicyBundleManifest.fileName)
        var manifest = try JSONDecoder().decode(
            PolicyBundleManifest.self, from: Data(contentsOf: manifestURL))
        manifest.simulation.includeInteractiveRobustnessProbes = false
        try JSONEncoder().encode(manifest).write(
            to: manifestURL, options: .atomic)

        XCTAssertThrowsError(try PolicyBundleLoader.load(directory: temporary))
    }

    func testAppSourceHasNoPolicyOrTaskSpecificReplaySwitches() throws {
        let source = try String(contentsOf: root.appendingPathComponent(
            "Development/Sources/AVBDApp/PolicyReplayLab.swift"), encoding: .utf8)
        for forbidden in [
            "PolicyReplayCatalog", "HumanoidIsaacVelocityTask",
            "Arachne15LocomotionTask", "UnitreeH1Sim2SimSession",
            "HumanoidBoxCarryTask", "GEARSonicG1Session",
        ] {
            XCTAssertFalse(source.contains(forbidden), forbidden)
        }
        XCTAssertTrue(source.contains("manifest.presentation.controls"))
        XCTAssertTrue(source.contains("manifest.presentation.metrics"))
        XCTAssertTrue(source.contains("manifest.presentation.cameraPresets"))
    }

    private func copyToTemporary(_ source: URL) throws -> URL {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: false)
        let destination = parent.appendingPathComponent(
            source.lastPathComponent, isDirectory: true)
        try FileManager.default.copyItem(at: source, to: destination)
        return destination
    }
}
