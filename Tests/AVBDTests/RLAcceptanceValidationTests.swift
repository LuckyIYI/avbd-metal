import XCTest
@testable import AVBDCore
@testable import AVBDLearn

final class RLAcceptanceValidationTests: XCTestCase {
    func testCriteriaRejectNonFiniteThresholdsAndTaskOutputs() {
        var criteria = RLEvaluationCriteria(
            minimumSuccessRate: 0.8,
            minimumMeanEpisodeLengthFraction: 0.5,
            minimumTaskMetrics: ["episode/progress": 1],
            maximumTaskMetrics: ["episode/error": 0.2])
        criteria.minimumSuccessRate = .nan
        criteria.minimumMeanEpisodeLengthFraction = .infinity
        criteria.minimumTaskMetrics["episode/progress"] = .nan
        criteria.maximumTaskMetrics["episode/error"] = .infinity

        let failures = criteria.failures(
            successRate: .nan,
            meanEpisodeLength: .infinity,
            maxEpisodeSteps: 100,
            taskMetrics: [
                "episode/progress": .nan,
                "episode/error": .infinity,
                "episode/unbounded": .nan,
            ])

        XCTAssertTrue(failures.contains {
            $0.contains("minimum_success_rate threshold")
        })
        XCTAssertTrue(failures.contains {
            $0.contains("minimum_mean_episode_length_fraction threshold")
        })
        XCTAssertTrue(failures.contains {
            $0.contains("minimum threshold for metric episode/progress")
        })
        XCTAssertTrue(failures.contains {
            $0.contains("maximum threshold for metric episode/error")
        })
        XCTAssertTrue(failures.contains { $0.contains("success_rate") })
        XCTAssertTrue(failures.contains { $0.contains("mean_episode_length") })
        XCTAssertTrue(failures.contains {
            $0.contains("metric episode/unbounded is not finite")
        })
    }

    func testEvaluationReportRejectsNonFiniteAndInconsistentFacts() throws {
        let valid = report(
            update: 10, fingerprint: "policy-a", successes: 8,
            acceptance: .init(passed: true, failures: []))
        try valid.validateStructure()

        var inconsistentRate = valid
        inconsistentRate.successRate = 0.9
        XCTAssertThrowsError(try inconsistentRate.validateStructure())

        var nonFiniteOutput = valid
        nonFiniteOutput.taskMetrics["episode/progress"] = .nan
        XCTAssertThrowsError(try nonFiniteOutput.validateStructure())

        nonFiniteOutput = valid
        nonFiniteOutput.meanReturn = .infinity
        XCTAssertThrowsError(try nonFiniteOutput.validateStructure())

        var contradictoryAcceptance = valid
        contradictoryAcceptance.acceptance = .init(
            passed: true, failures: ["failed gate"])
        XCTAssertThrowsError(try contradictoryAcceptance.validateStructure())
    }

    func testMissingAcceptanceCannotWinCheckpointSelection() throws {
        let accepted = report(
            update: 10, fingerprint: "accepted", successes: 8,
            acceptance: .init(passed: true, failures: []))
        let missing = report(
            update: 20, fingerprint: "missing", successes: 10,
            acceptance: nil)

        let selection = try PPOCheckpointSelection.make([missing, accepted])
        XCTAssertEqual(selection.selectedCheckpointFingerprint, "accepted")
        XCTAssertEqual(
            selection.candidates.first {
                $0.checkpointFingerprint == "missing"
            }?.acceptancePassed,
            false)
        XCTAssertThrowsError(try PPOCheckpointSelection.make([missing])) {
            XCTAssertTrue(String(describing: $0).contains(
                "no checkpoint candidate passed"))
        }
    }

    func testCheckpointAggregateValidatesSerializedArithmeticAndGates() throws {
        let reports = (0..<4).map { index in
            report(
                update: 10, fingerprint: "policy-a",
                evaluationSeed: UInt64(500 + index), successes: 8,
                acceptance: .init(passed: true, failures: []))
        }
        var aggregate = try PPOCheckpointEvaluationAggregate.make(
            reports, requiredEpisodesPerRun: 10)
        try aggregate.validateStructure()

        aggregate.totalSuccesses -= 1
        XCTAssertThrowsError(try aggregate.validateStructure())

        aggregate = try PPOCheckpointEvaluationAggregate.make(
            reports, requiredEpisodesPerRun: 10)
        aggregate.robustAcrossEvaluationSeeds = false
        XCTAssertThrowsError(try aggregate.validateStructure())
    }

    func testTrainingAggregateValidatesAcceptanceAndPublicationGates() throws {
        var reports = [PPOEvaluationMetrics]()
        for index in 0..<5 {
            var value = report(
                update: 10, fingerprint: "policy-\(index)",
                evaluationSeed: UInt64(700 + index), successes: 8,
                acceptance: .init(passed: true, failures: []))
            value.trainingSeed = UInt64(800 + index)
            reports.append(value)
        }
        let aggregate = try PPOEvaluationAggregate.make(
            reports, requiredEpisodesPerRun: 10)
        try aggregate.validateStructure()

        var inconsistent = aggregate
        inconsistent.acceptedRuns -= 1
        XCTAssertThrowsError(try inconsistent.validateStructure())

        reports[0].acceptance = nil
        let missingAcceptance = try PPOEvaluationAggregate.make(
            reports, requiredEpisodesPerRun: 10)
        XCTAssertFalse(missingAcceptance.allRunsPassed)
        XCTAssertFalse(missingAcceptance.publishable)
        try missingAcceptance.validateStructure()
    }

    private func report(
        update: Int,
        fingerprint: String,
        evaluationSeed: UInt64 = 100,
        successes: Int,
        acceptance: PPOEvaluationAcceptance?
    ) -> PPOEvaluationMetrics {
        PPOEvaluationMetrics(
            provenanceVersion: 3,
            task: "test-task-v0", taskRevision: 1,
            checkpointTaskConfiguration: ["solverIterations": 4],
            evaluationTaskConfiguration: ["solverIterations": 4],
            taskConfigurationTransferred: false,
            checkpointDirectory: "run/checkpoints/update-\(update)",
            checkpointFingerprint: fingerprint,
            trainingSeed: 77, evaluationSeed: evaluationSeed,
            evaluationEnvironments: 10,
            trainingUpdates: update,
            trainingEnvironmentSteps: update * 100,
            episodes: 10, successes: successes,
            successRate: Float(successes) / 10,
            meanReturn: 5, meanEpisodeLength: 100,
            taskMetrics: ["episode/progress": 1],
            acceptance: acceptance)
    }
}
