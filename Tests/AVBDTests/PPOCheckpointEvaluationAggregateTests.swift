import XCTest
@testable import AVBDLearn

final class PPOCheckpointEvaluationAggregateTests: XCTestCase {
    func testAggregatePersistsRevisionAndRejectsMixedPhysicsContracts() throws {
        func report(seed: UInt64, revision: Int) -> PPOEvaluationMetrics {
            PPOEvaluationMetrics(
                provenanceVersion: 3,
                task: "humanoid-isaac-flat-v0",
                taskRevision: revision,
                checkpointTaskConfiguration: ["solverIterations": 20],
                evaluationTaskConfiguration: ["solverIterations": 20],
                taskConfigurationTransferred: false,
                checkpointDirectory: "checkpoints/humanoid-isaac-flat-v1",
                checkpointFingerprint: "immutable-policy",
                initializationCheckpoint: "checkpoints/humanoid-isaac-flat-v0",
                trainingSeed: 41_002,
                evaluationSeed: seed,
                evaluationEnvironments: 512,
                trainingUpdates: 0,
                trainingEnvironmentSteps: 0,
                episodes: 512,
                successes: 512,
                successRate: 1,
                meanReturn: 1,
                meanEpisodeLength: 1_000,
                taskMetrics: ["episode/linear_velocity_rmse": 0.05],
                acceptance: PPOEvaluationAcceptance(
                    passed: true, failures: []))
        }

        let reports = (0..<4).map {
            report(seed: 51_000 + UInt64($0), revision: 1_000_011)
        }
        let aggregate = try PPOCheckpointEvaluationAggregate.make(reports)
        XCTAssertEqual(aggregate.taskRevision, 1_000_011)
        XCTAssertTrue(aggregate.robustAcrossEvaluationSeeds)

        var mixed = reports
        mixed[3].taskRevision = 1_000_010
        XCTAssertThrowsError(
            try PPOCheckpointEvaluationAggregate.make(mixed)) { error in
                XCTAssertTrue(String(describing: error).contains(
                    "one immutable checkpoint"))
            }

        var missing = reports
        missing[0].taskRevision = nil
        XCTAssertThrowsError(
            try PPOCheckpointEvaluationAggregate.make(missing)) { error in
                XCTAssertTrue(String(describing: error).contains(
                    "explicit task revision"))
            }
    }
}
