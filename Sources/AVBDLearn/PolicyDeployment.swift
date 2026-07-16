import AVBDCore
import CryptoKit
import Foundation

/// Immutable, optimizer-free artifact description for policy deployment.
/// The bundle deliberately retains the original metadata and training state
/// bytes, so its checkpoint fingerprint is identical to the evaluated source.
public struct VectorPolicyDeploymentManifest: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var task: String
    public var taskRevision: Int
    public var checkpointFingerprint: String
    public var policySHA256: String
    public var policyFile: String
    public var metadataFile: String
    public var trainingStateFile: String
    public var architectureVersion: Int
    public var observationDimension: Int
    public var actionDimension: Int
    public var simulationStepSeconds: Float
    public var controlDecimation: Int
    public var controlFrequencyHz: Float
    public var normalizesObservations: Bool
    public var observationNormalizationClip: Float?
    public var actionDistribution: String
    public var taskConfiguration: [String: Float]
    public var trainingUpdates: Int
    public var trainingEnvironmentSteps: Int
}

public enum VectorPolicyDeploymentBundle {
    public static let manifestFileName = "deployment-manifest.json"

    /// Export only deterministic inference state, preserving byte identity.
    /// Refusing to overwrite an existing directory prevents a field bundle
    /// from silently changing after bench qualification.
    @discardableResult
    public static func export(
        checkpointDirectory: String,
        outputDirectory: String
    ) throws -> VectorPolicyDeploymentManifest {
        let manager = FileManager.default
        let source = URL(fileURLWithPath: checkpointDirectory,
                         isDirectory: true)
        let destination = URL(fileURLWithPath: outputDirectory,
                              isDirectory: true)
        guard source.standardizedFileURL != destination.standardizedFileURL else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment output must differ from the source checkpoint")
        }
        guard !manager.fileExists(atPath: destination.path) else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment output already exists: \(destination.path)")
        }

        let metadataData = try Data(contentsOf: source.appendingPathComponent(
            "metadata.json"))
        let policyData = try Data(contentsOf: source.appendingPathComponent(
            "policy.safetensors"))
        let trainingStateData = try Data(contentsOf: source.appendingPathComponent(
            "training-state.json"))
        guard !metadataData.isEmpty, !policyData.isEmpty,
              !trainingStateData.isEmpty else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment source contains an empty checkpoint file")
        }
        let metadata = try JSONDecoder().decode(
            VectorPolicyMetadata.self, from: metadataData)
        let trainingState = try JSONDecoder().decode(
            VectorPPOTrainingState.self, from: trainingStateData)
        guard let taskRevision = metadata.taskRevision,
              let taskConfiguration = metadata.taskConfiguration,
              let architectureVersion = metadata.architectureVersion,
              VectorActorCritic.compatibleArchitectureVersions.contains(
                architectureVersion) else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment requires a current, fully specified checkpoint")
        }
        let fingerprint = try VectorPPOTrainer.checkpointFingerprint(
            directory: checkpointDirectory)
        let policyDigest = SHA256.hash(data: policyData).map {
            String(format: "%02x", $0)
        }.joined()
        let manifest = VectorPolicyDeploymentManifest(
            schemaVersion: 1,
            task: metadata.task,
            taskRevision: taskRevision,
            checkpointFingerprint: fingerprint,
            policySHA256: policyDigest,
            policyFile: "policy.safetensors",
            metadataFile: "metadata.json",
            trainingStateFile: "training-state.json",
            architectureVersion: architectureVersion,
            observationDimension: metadata.observationDimension,
            actionDimension: metadata.actionDimension,
            simulationStepSeconds: metadata.simulationStep,
            controlDecimation: metadata.controlDecimation,
            controlFrequencyHz: 1 / (metadata.simulationStep
                * Float(metadata.controlDecimation)),
            normalizesObservations: metadata.ppo.normalizeObservations,
            observationNormalizationClip:
                metadata.ppo.normalizeObservations ? 10 : nil,
            actionDistribution:
                metadata.ppo.resolvedActionDistribution.rawValue,
            taskConfiguration: taskConfiguration,
            trainingUpdates: trainingState.completedUpdates,
            trainingEnvironmentSteps: trainingState.environmentSteps)

        let parent = destination.deletingLastPathComponent()
        try manager.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).staging-\(UUID().uuidString)",
            isDirectory: true)
        try manager.createDirectory(at: staging, withIntermediateDirectories: false)
        var published = false
        defer {
            if !published { try? manager.removeItem(at: staging) }
        }
        try metadataData.write(
            to: staging.appendingPathComponent("metadata.json"), options: .atomic)
        try policyData.write(
            to: staging.appendingPathComponent("policy.safetensors"),
            options: .atomic)
        try trainingStateData.write(
            to: staging.appendingPathComponent("training-state.json"),
            options: .atomic)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(
            to: staging.appendingPathComponent(manifestFileName), options: .atomic)
        let copiedFingerprint = try VectorPPOTrainer.checkpointFingerprint(
            directory: staging.path)
        guard copiedFingerprint == fingerprint else {
            throw RLEnvironmentError.invalidConfiguration(
                "deployment bundle changed checkpoint identity while copying")
        }
        try manager.moveItem(at: staging, to: destination)
        published = true
        return manifest
    }
}
