import Foundation
import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import Robotics
@testable import RL
@testable import MLXRL

final class PPOCheckpointTransactionTests: XCTestCase {
    private enum ReplayPayloadGap { case leading, interior, trailing }
    private let task = "transaction-test-v0"
    private let revision = 7
    private let observationDimension = 3
    private let actionDimension = 2

    private func temporaryRun() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "avbd-ppo-transaction-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    private func metadata(successReplayCapacity: Int = 0)
        -> VectorPolicyMetadata {
        let replayEnabled = successReplayCapacity > 0
        return VectorPolicyMetadata(
            architectureVersion: VectorActorCritic.architectureVersion,
            task: task,
            taskRevision: revision,
            taskConfiguration: ["fixture": 1],
            observationDimension: observationDimension,
            actionDimension: actionDimension,
            simulationStep: 0.005,
            controlDecimation: 4,
            maxEpisodeSteps: 100,
            inferenceBatchSize: 4,
            ppo: VectorPPOConfig(
                updates: 1,
                rolloutSteps: 4,
                updateEpochs: 1,
                minibatchSize: 4,
                successImitationCoefficient: replayEnabled ? 0.1 : nil,
                successReplayCapacity:
                    replayEnabled ? successReplayCapacity : nil,
                successReplayBatchSize: replayEnabled ? 2 : nil,
                normalizeObservations: !replayEnabled,
                updateObservationNormalizer: !replayEnabled,
                checkpointInterval: 1),
            normalizer: RunningNormalizerSnapshot(
                count: 0,
                mean: [Double](
                    repeating: 0, count: observationDimension),
                variance: [Double](
                    repeating: 1, count: observationDimension)))
    }

    private func writeSuccessReplay(
        rows: Int,
        at directory: URL,
        gap: ReplayPayloadGap? = nil
    ) throws {
        let tensors: [(String, [Int], [Float])] = [
            ("observations", [rows, observationDimension],
             [Float](repeating: 0.25,
                     count: rows * observationDimension)),
            ("actions", [rows, actionDimension],
             [Float](repeating: -0.1, count: rows * actionDimension)),
            ("expertGates", [rows], [Float](repeating: 0, count: rows)),
            ("standExpertGates", [rows],
             [Float](repeating: 0, count: rows)),
            ("auxiliaryExpertGates", [rows],
             [Float](repeating: 0, count: rows)),
        ]
        var header = [String: Any]()
        var payload = gap == .leading
            ? Data(repeating: 0, count: 4) : Data()
        for (index, tensor) in tensors.enumerated() {
            let (name, shape, values) = tensor
            if gap == .interior, index == 1 {
                payload.append(Data(repeating: 0, count: 4))
            }
            let start = payload.count
            for value in values {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
            }
            header[name] = [
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, payload.count],
            ]
        }
        if gap == .trailing {
            payload.append(Data(repeating: 0, count: 4))
        }
        var headerData = try JSONSerialization.data(
            withJSONObject: header, options: [.sortedKeys])
        let padding = (8 - headerData.count % 8) % 8
        headerData.append(Data(repeating: 0x20, count: padding))
        var headerLength = UInt64(headerData.count).littleEndian
        var file = Data()
        withUnsafeBytes(of: &headerLength) { file.append(contentsOf: $0) }
        file.append(headerData)
        file.append(payload)
        try file.write(
            to: directory.appendingPathComponent(
                VectorPPOTrainer.successReplayFileName))
    }

    private func writeCandidate(
        at directory: URL,
        update: Int,
        environmentSteps: Int? = nil,
        resumableSnapshotVersion: Int? = 2,
        rolloutEnvironmentCount: Int? = 4,
        optimizerSteps: Int?,
        adaptiveLearningRate: Float?,
        successReplayCapacity: Int = 0,
        successReplayCount: Int? = 0,
        replayRows: Int? = nil,
        replayGap: ReplayPayloadGap? = nil,
        includeOptimizer: Bool,
        mutateMetadata: ((inout VectorPolicyMetadata) -> Void)? = nil
    ) throws {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data([0x50]).write(
            to: directory.appendingPathComponent("policy.safetensors"))
        var metadata = metadata(successReplayCapacity: successReplayCapacity)
        mutateMetadata?(&metadata)
        try JSONEncoder().encode(metadata).write(
            to: directory.appendingPathComponent("metadata.json"))
        let state = VectorPPOTrainingState(
            completedUpdates: update,
            environmentSteps: environmentSteps ?? update * 16,
            resumableSnapshotVersion: resumableSnapshotVersion,
            rolloutEnvironmentCount: rolloutEnvironmentCount,
            optimizerSteps: optimizerSteps,
            adaptiveLearningRate: adaptiveLearningRate,
            successReplayCount: successReplayCount)
        try JSONEncoder().encode(state).write(
            to: directory.appendingPathComponent("training-state.json"))
        if includeOptimizer {
            try Data([0x4f]).write(
                to: directory.appendingPathComponent("optimizer.safetensors"))
        }
        if let replayRows {
            try writeSuccessReplay(
                rows: replayRows, at: directory, gap: replayGap)
        }
    }

    func testAtomicPublicationExposesOnlyACompleteImmutableGeneration() throws {
        let run = try temporaryRun()
        defer { try? FileManager.default.removeItem(at: run) }

        let published = try VectorPolicyCheckpointDiscovery
            .publishCompleteCheckpoint(
                inRunDirectory: run.path,
                completedUpdates: 12,
                task: task,
                taskRevision: revision
            ) { staging in
                XCTAssertTrue(staging.lastPathComponent.hasPrefix("."))
                try self.writeCandidate(
                    at: staging, update: 12,
                    optimizerSteps: 12,
                    adaptiveLearningRate: 2e-4,
                    includeOptimizer: true)
                XCTAssertNil(VectorPolicyCheckpointDiscovery
                    .latestCompleteCheckpoint(
                        inRunDirectory: run.path,
                        task: task,
                        taskRevision: revision))
            }

        XCTAssertEqual(
            URL(fileURLWithPath: published.directory).lastPathComponent,
            "update-000012")
        XCTAssertEqual(
            VectorPolicyCheckpointDiscovery.latestCompleteCheckpoint(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision),
            published)
        let children = try FileManager.default.contentsOfDirectory(
            at: run.appendingPathComponent("checkpoints"),
            includingPropertiesForKeys: nil)
        XCTAssertEqual(children.map(\.lastPathComponent), ["update-000012"])

        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .publishCompleteCheckpoint(
                inRunDirectory: run.path,
                completedUpdates: 12,
                task: task,
                taskRevision: revision
            ) { _ in
                XCTFail("an immutable destination must be rejected before writing")
            })
        XCTAssertEqual(try Data(contentsOf: URL(
            fileURLWithPath: published.directory)
            .appendingPathComponent("policy.safetensors")), Data([0x50]))
    }

    func testFailedPublicationRemovesStagingAndCanBeRetried() throws {
        let run = try temporaryRun()
        defer { try? FileManager.default.removeItem(at: run) }

        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .publishCompleteCheckpoint(
                inRunDirectory: run.path,
                completedUpdates: 4,
                task: task,
                taskRevision: revision
            ) { staging in
                try self.writeCandidate(
                    at: staging, update: 4,
                    optimizerSteps: 4,
                    adaptiveLearningRate: 3e-4,
                    includeOptimizer: false)
            })
        let checkpoints = run.appendingPathComponent(
            "checkpoints", isDirectory: true)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(
            at: checkpoints, includingPropertiesForKeys: nil), [])

        let retry = try VectorPolicyCheckpointDiscovery
            .publishCompleteCheckpoint(
                inRunDirectory: run.path,
                completedUpdates: 4,
                task: task,
                taskRevision: revision
            ) { staging in
                try self.writeCandidate(
                    at: staging, update: 4,
                    optimizerSteps: 4,
                    adaptiveLearningRate: 3e-4,
                    includeOptimizer: true)
            }
        XCTAssertEqual(retry.completedUpdates, 4)
    }

    func testReplayDiscoveryKeepsLegacyButResumeRejectsNewestBadGeneration()
        throws {
        let run = try temporaryRun()
        defer { try? FileManager.default.removeItem(at: run) }
        let checkpoints = run.appendingPathComponent(
            "checkpoints", isDirectory: true)

        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000010"),
            update: 10, optimizerSteps: 10,
            adaptiveLearningRate: 2e-4, includeOptimizer: true)
        XCTAssertEqual(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4).completedUpdates, 10)
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000020"),
            update: 20, optimizerSteps: 20,
            adaptiveLearningRate: 2e-4, includeOptimizer: false)
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000030"),
            update: 30, optimizerSteps: nil,
            adaptiveLearningRate: nil, includeOptimizer: false)
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000040"),
            update: 40, optimizerSteps: 20,
            adaptiveLearningRate: 2e-4, includeOptimizer: true)
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000050"),
            update: 50, optimizerSteps: 0,
            adaptiveLearningRate: 2e-4, includeOptimizer: false)
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000055"),
            update: 55, optimizerSteps: 221,
            adaptiveLearningRate: 2e-4, includeOptimizer: true)
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000056"),
            update: 56, environmentSteps: 897, optimizerSteps: 56,
            adaptiveLearningRate: 2e-4, includeOptimizer: true)

        XCTAssertEqual(VectorPolicyCheckpointDiscovery
            .latestCompleteCheckpoint(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision)?.completedUpdates, 56)
        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4))

        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000060"),
            update: 60,
            resumableSnapshotVersion: nil,
            rolloutEnvironmentCount: nil,
            optimizerSteps: nil,
            adaptiveLearningRate: nil,
            successReplayCount: nil,
            includeOptimizer: false,
            mutateMetadata: { $0.taskConfiguration = nil })
        XCTAssertEqual(VectorPolicyCheckpointDiscovery
            .latestCompleteCheckpoint(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision)?.completedUpdates, 60)
        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4))

        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000070"),
            update: 70, optimizerSteps: 70,
            adaptiveLearningRate: 2e-4, includeOptimizer: true)
        XCTAssertEqual(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4).completedUpdates, 70)
        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 8))

        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000075"),
            update: 75, optimizerSteps: 75,
            adaptiveLearningRate: 2e-4, includeOptimizer: true,
            mutateMetadata: { $0.inferenceBatchSize = nil })
        XCTAssertEqual(VectorPolicyCheckpointDiscovery
            .latestCompleteCheckpoint(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision)?.completedUpdates, 75)
        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4))

        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000080"),
            update: 80, optimizerSteps: 80,
            adaptiveLearningRate: 4e-4, includeOptimizer: true)
        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4))

        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000090"),
            update: 90, optimizerSteps: 90,
            adaptiveLearningRate: 2e-4, includeOptimizer: true,
            mutateMetadata: { $0.ppo.klSchedule = PPOKLSchedule.none })
        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4))

        let freshRun = try temporaryRun()
        defer { try? FileManager.default.removeItem(at: freshRun) }
        try writeCandidate(
            at: freshRun.appendingPathComponent(
                "checkpoints/update-000000"),
            update: 0, environmentSteps: 0, optimizerSteps: 0,
            adaptiveLearningRate: 3e-4, includeOptimizer: false)
        XCTAssertEqual(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: freshRun.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4).completedUpdates, 0)
    }

    func testMalformedFullMetadataCannotHideOlderCompleteSnapshot() throws {
        let run = try temporaryRun()
        defer { try? FileManager.default.removeItem(at: run) }
        let checkpoints = run.appendingPathComponent(
            "checkpoints", isDirectory: true)
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000010"),
            update: 10, optimizerSteps: 10,
            adaptiveLearningRate: 2e-4, includeOptimizer: true)
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000020"),
            update: 20, optimizerSteps: 20,
            adaptiveLearningRate: 2e-4, includeOptimizer: true,
            mutateMetadata: { $0.observationDimension = 4 })
        let truncated = checkpoints.appendingPathComponent(
            "update-000030", isDirectory: true)
        try writeCandidate(
            at: truncated, update: 30, optimizerSteps: 30,
            adaptiveLearningRate: 2e-4, includeOptimizer: true)
        try Data("{\"task\":\"\(task)\",\"taskRevision\":7}".utf8)
            .write(to: truncated.appendingPathComponent("metadata.json"))
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000040"),
            update: 40, optimizerSteps: 40,
            adaptiveLearningRate: 2e-4, includeOptimizer: true,
            mutateMetadata: { $0.ppo.learningRate = 0 })

        XCTAssertEqual(VectorPolicyCheckpointDiscovery
            .latestCompleteCheckpoint(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision)?.completedUpdates, 10)
    }

    func testSuccessReplayPresenceAndRowsMustMatchTrainingState() throws {
        let run = try temporaryRun()
        defer { try? FileManager.default.removeItem(at: run) }
        let checkpoints = run.appendingPathComponent(
            "checkpoints", isDirectory: true)
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000010"),
            update: 10, optimizerSteps: 10,
            adaptiveLearningRate: 2e-4,
            successReplayCapacity: 4, successReplayCount: 2,
            replayRows: nil, includeOptimizer: true)
        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000020"),
            update: 20, optimizerSteps: 20,
            adaptiveLearningRate: 2e-4,
            successReplayCapacity: 4, successReplayCount: 3,
            replayRows: 2, includeOptimizer: true)
        XCTAssertEqual(VectorPolicyCheckpointDiscovery
            .latestCompleteCheckpoint(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision)?.completedUpdates, 20)
        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4))

        try writeCandidate(
            at: checkpoints.appendingPathComponent("update-000030"),
            update: 30, optimizerSteps: 30,
            adaptiveLearningRate: 2e-4,
            successReplayCapacity: 4, successReplayCount: 2,
            replayRows: 2, includeOptimizer: true)
        XCTAssertEqual(VectorPolicyCheckpointDiscovery
            .latestCompleteCheckpoint(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision)?.completedUpdates, 30)
        XCTAssertEqual(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4).completedUpdates, 30)

        for (update, gap) in [
            (40, ReplayPayloadGap.leading),
            (50, ReplayPayloadGap.interior),
            (60, ReplayPayloadGap.trailing),
        ] {
            try writeCandidate(
                at: checkpoints.appendingPathComponent(
                    try VectorPolicyCheckpointDiscovery
                        .checkpointDirectoryName(completedUpdates: update)),
                update: update, optimizerSteps: update,
                adaptiveLearningRate: 2e-4,
                successReplayCapacity: 4, successReplayCount: 2,
                replayRows: 2, replayGap: gap, includeOptimizer: true)
            XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
                .checkpointForResume(
                    inRunDirectory: run.path,
                    task: task,
                    taskRevision: revision,
                    numEnvironments: 4))
        }
        XCTAssertEqual(VectorPolicyCheckpointDiscovery
            .latestCompleteCheckpoint(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision)?.completedUpdates, 60)
    }

    func testResumeResolutionNeverFallsBackToMutableRunRoot() throws {
        let run = try temporaryRun()
        defer { try? FileManager.default.removeItem(at: run) }
        try writeCandidate(
            at: run, update: 9, optimizerSteps: 9,
            adaptiveLearningRate: 1e-4, includeOptimizer: true)

        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 4))
        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .checkpointForResume(
                inRunDirectory: run.path,
                task: task,
                taskRevision: revision,
                numEnvironments: 0))

        let snapshot = try VectorPolicyCheckpointDiscovery
            .publishCompleteCheckpoint(
                inRunDirectory: run.path,
                completedUpdates: 8,
                task: task,
                taskRevision: revision
            ) { staging in
                try self.writeCandidate(
                    at: staging, update: 8,
                    optimizerSteps: 8,
                    adaptiveLearningRate: 1e-4,
                    includeOptimizer: true)
            }
        let resolved = try VectorPolicyCheckpointDiscovery.checkpointForResume(
            inRunDirectory: run.path,
            task: task,
            taskRevision: revision,
            numEnvironments: 4)
        XCTAssertEqual(resolved, snapshot)
        XCTAssertNotEqual(resolved.directory, run.path)
    }

    func testV2TrainingStateRejectsImpossibleCounterGeometry() throws {
        var configuration = metadata().ppo
        configuration.learningRate = 2e-4
        let valid = VectorPPOTrainingState(
            completedUpdates: 10,
            environmentSteps: 160,
            resumableSnapshotVersion:
                VectorPPOTrainingState.currentResumableSnapshotVersion,
            rolloutEnvironmentCount: 4,
            optimizerSteps: 40,
            adaptiveLearningRate: 2e-4,
            successReplayCount: 0)
        XCTAssertEqual(try valid.maximumOptimizerSteps(
            rolloutSteps: 4, updateEpochs: 1, minibatchSize: 4), 40)
        XCTAssertNoThrow(try valid.validatedOptimizerResumeState(
            configuration: configuration,
            maximumOptimizerSteps: 40))

        var optimizerAhead = valid
        optimizerAhead.optimizerSteps = 41
        XCTAssertThrowsError(try optimizerAhead.validatedOptimizerResumeState(
            configuration: configuration,
            maximumOptimizerSteps: 40))
        var optimizerBehind = valid
        optimizerBehind.optimizerSteps = 9
        XCTAssertThrowsError(try optimizerBehind.validatedOptimizerResumeState(
            configuration: configuration,
            maximumOptimizerSteps: 40))
        var falseFresh = valid
        falseFresh.optimizerSteps = 0
        XCTAssertThrowsError(try falseFresh.validatedOptimizerResumeState(
            configuration: configuration,
            maximumOptimizerSteps: 40))
        var missingOptimizerCounter = valid
        missingOptimizerCounter.optimizerSteps = nil
        XCTAssertThrowsError(try missingOptimizerCounter
            .validatedOptimizerResumeState(
                configuration: configuration,
                maximumOptimizerSteps: 40))
        var adaptiveFloor = valid
        adaptiveFloor.adaptiveLearningRate = configuration.learningRate / 100
        XCTAssertNoThrow(try adaptiveFloor.validatedOptimizerResumeState(
            configuration: configuration,
            maximumOptimizerSteps: 40))
        var belowAdaptiveFloor = adaptiveFloor
        belowAdaptiveFloor.adaptiveLearningRate =
            configuration.learningRate / 101
        XCTAssertThrowsError(try belowAdaptiveFloor
            .validatedOptimizerResumeState(
                configuration: configuration,
                maximumOptimizerSteps: 40))
        var aboveAdaptiveCeiling = valid
        aboveAdaptiveCeiling.adaptiveLearningRate =
            configuration.learningRate * 1.01
        XCTAssertThrowsError(try aboveAdaptiveCeiling
            .validatedOptimizerResumeState(
                configuration: configuration,
                maximumOptimizerSteps: 40))
        var disabledAdaptiveConfiguration = configuration
        disabledAdaptiveConfiguration.targetKL = 0
        XCTAssertNoThrow(try valid.validatedOptimizerResumeState(
            configuration: disabledAdaptiveConfiguration,
            maximumOptimizerSteps: 40))
        var changedDisabledAdaptiveRate = valid
        changedDisabledAdaptiveRate.adaptiveLearningRate =
            configuration.learningRate / 2
        XCTAssertThrowsError(try changedDisabledAdaptiveRate
            .validatedOptimizerResumeState(
                configuration: disabledAdaptiveConfiguration,
                maximumOptimizerSteps: 40))
        var fixedConfiguration = configuration
        fixedConfiguration.klSchedule = .earlyStop
        XCTAssertNoThrow(try valid.validatedOptimizerResumeState(
            configuration: fixedConfiguration,
            maximumOptimizerSteps: 40))
        var changedFixedRate = valid
        changedFixedRate.adaptiveLearningRate = configuration.learningRate / 2
        XCTAssertThrowsError(try changedFixedRate
            .validatedOptimizerResumeState(
                configuration: fixedConfiguration,
                maximumOptimizerSteps: 40))
        fixedConfiguration.klSchedule = PPOKLSchedule.none
        XCTAssertThrowsError(try changedFixedRate
            .validatedOptimizerResumeState(
                configuration: fixedConfiguration,
                maximumOptimizerSteps: 40))
        var fresh = valid
        fresh.completedUpdates = 0
        fresh.environmentSteps = 0
        fresh.optimizerSteps = 0
        XCTAssertNoThrow(try fresh.validatedOptimizerResumeState(
            configuration: configuration,
            maximumOptimizerSteps: 0))
        fresh.adaptiveLearningRate = configuration.learningRate / 2
        XCTAssertThrowsError(try fresh.validatedOptimizerResumeState(
            configuration: configuration,
            maximumOptimizerSteps: 0))
        var inconsistentEnvironmentSteps = valid
        inconsistentEnvironmentSteps.environmentSteps += 1
        XCTAssertThrowsError(try inconsistentEnvironmentSteps
            .validatedRolloutEnvironmentCount(rolloutSteps: 4))
    }

    func testSuccessReplayRestoreRejectsInvalidHostValues() throws {
        let validate: (
            [Float], [Float], [Float], [Float], [Float]
        ) throws -> Void = { observations, actions, expert, stand, auxiliary in
            try VectorPPOTrainer.validateSuccessReplayCheckpointValues(
                observations: observations,
                actions: actions,
                expertGates: expert,
                standExpertGates: stand,
                auxiliaryExpertGates: auxiliary,
                rowCount: 2,
                observationDimension: 3,
                actionDimension: 2)
        }
        let observations = [Float](repeating: 0.25, count: 6)
        let actions = [Float](repeating: -0.5, count: 4)
        let gates: [Float] = [0, 1]
        XCTAssertNoThrow(try validate(
            observations, actions, gates, gates, gates))

        var nonfiniteObservations = observations
        nonfiniteObservations[2] = .nan
        XCTAssertThrowsError(try validate(
            nonfiniteObservations, actions, gates, gates, gates))
        var nonfiniteActions = actions
        nonfiniteActions[1] = .infinity
        XCTAssertThrowsError(try validate(
            observations, nonfiniteActions, gates, gates, gates))
        XCTAssertThrowsError(try validate(
            observations, actions, [-0.01, 0], gates, gates))
        XCTAssertThrowsError(try validate(
            observations, actions, gates, [0, 1.01], gates))
        XCTAssertThrowsError(try validate(
            observations, actions, gates, gates, [0, .nan]))
    }

    func testOptimizerRestoreHostValidationFailsClosed() throws {
        let expected: Set<String> = [
            "first.actor.weight", "second.actor.weight",
        ]
        XCTAssertNoThrow(try CheckpointableAdam.validateCheckpointKeys(
            actual: expected, expected: expected))
        XCTAssertThrowsError(try CheckpointableAdam.validateCheckpointKeys(
            actual: ["first.actor.weight"], expected: expected))
        XCTAssertThrowsError(try CheckpointableAdam.validateCheckpointKeys(
            actual: expected.union(["unexpected"]), expected: expected))

        XCTAssertNoThrow(try CheckpointableAdam
            .validateFirstMomentValues([0, -1, 2]))
        XCTAssertThrowsError(try CheckpointableAdam
            .validateFirstMomentValues([0, .nan]))
        XCTAssertNoThrow(try CheckpointableAdam
            .validateSecondMomentValues([0, 1, 2]))
        XCTAssertThrowsError(try CheckpointableAdam
            .validateSecondMomentValues([-0.000_001]))
        XCTAssertThrowsError(try CheckpointableAdam
            .validateSecondMomentValues([.infinity]))
    }

    func testCheckpointNameDoesNotTruncateLargeInt() throws {
        let update = Int(Int32.max) + 123
        XCTAssertEqual(try VectorPolicyCheckpointDiscovery
            .checkpointDirectoryName(completedUpdates: update),
            "update-2147483770")
        XCTAssertThrowsError(try VectorPolicyCheckpointDiscovery
            .checkpointDirectoryName(completedUpdates: -1))
    }

    func testRunLockRejectsContentionAndCanBeReacquired() throws {
        let run = try temporaryRun()
        defer { try? FileManager.default.removeItem(at: run) }
        let first = try VectorPPORunLock(runDirectory: run.path)
        XCTAssertThrowsError(try VectorPPORunLock(runDirectory: run.path))
        first.unlock()
        let reacquired = try VectorPPORunLock(runDirectory: run.path)
        reacquired.unlock()
    }
}
