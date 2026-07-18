import XCTest
import simd
@testable import AVBDCore
@testable import AVBDLearn

final class PushTPhysicalFlowExperimentTests: XCTestCase {
    func testTaskGeneralBalancedObjectiveTargetsWorstConstraint() {
        let evaluation = PhysicalFlowBalancedObjective.evaluate(
            normalizedErrors: [0.5, 1, 2])
        XCTAssertEqual(evaluation.maximumNormalizedError, 2)
        XCTAssertEqual(
            evaluation.meanSquaredNormalizedError, 1.75, accuracy: 1e-6)
        XCTAssertEqual(evaluation.bottleneckLoss, 4.4375, accuracy: 1e-6)
        XCTAssertFalse(evaluation.satisfiesEveryConstraint)
        XCTAssertTrue(PhysicalFlowBalancedObjective.evaluate(
            normalizedErrors: [0.999, 0.2]).satisfiesEveryConstraint)
    }

    func testClampedCubicSplineInterpolatesEndpointsAndConvexHull() {
        let points = [
            SIMD2<Float>(-1, 0.5), SIMD2<Float>(-0.4, 1.2),
            SIMD2<Float>(0.3, -0.8), SIMD2<Float>(0.9, 0.4),
            SIMD2<Float>(1.4, -0.2),
        ]
        XCTAssertEqual(PushTClampedCubicSpline.sample(
            controlPoints: points, progress: 0), points.first!)
        XCTAssertEqual(PushTClampedCubicSpline.sample(
            controlPoints: points, progress: 1), points.last!)
        for step in 0...100 {
            let progress = Float(step) / 100
            let weights = PushTClampedCubicSpline.basisWeights(
                controlPointCount: points.count, progress: progress)
            XCTAssertEqual(weights.reduce(0, +), 1, accuracy: 1e-5)
            XCTAssertTrue(weights.allSatisfy { $0 >= -1e-6 })
            let sample = PushTClampedCubicSpline.sample(
                controlPoints: points, progress: progress)
            XCTAssertGreaterThanOrEqual(sample.x, -1.00001)
            XCTAssertLessThanOrEqual(sample.x, 1.40001)
            XCTAssertGreaterThanOrEqual(sample.y, -0.80001)
            XCTAssertLessThanOrEqual(sample.y, 1.20001)
        }
    }

    func testSplineLeastSquaresFitRecoversSmoothActionSequence() {
        let sourcePoints = [
            SIMD2<Float>(-1.1, 0.2), SIMD2<Float>(-0.7, 1.0),
            SIMD2<Float>(0.1, -0.2), SIMD2<Float>(0.8, 0.5),
            SIMD2<Float>(1.2, 0.7), SIMD2<Float>(1.6, -0.1),
        ]
        let samples = (0...48).map {
            PushTClampedCubicSpline.sample(
                controlPoints: sourcePoints, progress: Float($0) / 48)
        }
        let fitted = PushTClampedCubicSpline.fit(
            samples: samples, initialControlPoint: sourcePoints[0],
            controlPointCount: sourcePoints.count)
        let reconstructed = (0...48).map {
            PushTClampedCubicSpline.sample(
                controlPoints: fitted, progress: Float($0) / 48)
        }
        let rmse = sqrt(zip(samples, reconstructed).reduce(0) {
            $0 + length_squared($1.0 - $1.1)
        } / Float(samples.count))
        XCTAssertLessThan(rmse, 1e-4)
    }

    func testMicrobatchedPhysicalFlowReplaysFromColdFork() throws {
        let report = try PushTPhysicalFlowExperiment.run(configuration: .init(
            populationSize: 40, simulationBatchSize: 8,
            generations: 1, horizon: 8,
            controlPointCount: 4, substeps: 2,
            sourcePreparationSteps: 160,
            initialStandardDeviation: 0.5, eliteFraction: 0.25,
            seed: 73))
        XCTAssertTrue(report.sourceHadActiveContact)
        XCTAssertEqual(report.populationSize, 40)
        XCTAssertEqual(report.simulationBatchSize, 8)
        XCTAssertLessThan(report.targetCloneSpreadLoss, 1e-5)
        XCTAssertLessThan(report.targetCloneMaximumStateError, 3e-4)
        XCTAssertLessThan(report.referenceReplay.loss, 1e-7)
        XCTAssertLessThan(report.selectedReplayLoss, 1e-7)
        XCTAssertLessThan(report.selectedReplayMaximumStateError, 1e-5)
        XCTAssertTrue(report.optimizedSpline.loss.isFinite)
        XCTAssertEqual(report.generationZeroSelectedProposal, .notApplicable)
        XCTAssertNil(report.providedProposalProbeBestLoss)
        XCTAssertNil(report.geometricProposalProbeBestLoss)
        XCTAssertEqual(report.generationZeroProbeCandidates, 0)
        XCTAssertEqual(report.generationZeroExploitationCandidates, 0)
        XCTAssertEqual(report.bestControlPointsXY.count, 8)
        XCTAssertEqual(
            report.teacherSample.schemaVersion,
            PushTPhysicalFlowTeacherSample.schemaVersion)
        XCTAssertEqual(
            report.teacherSample.input.count,
            PushTPhysicalFlowProposalContext.inputDimension)
        XCTAssertEqual(
            report.teacherSample.canonicalInternalControlPointsXY.count, 4)
        XCTAssertEqual(report.teacherSample.seed, 73)
    }

    func testLearnedProposalProviderReceivesStructuredStateAndDecodesKnots()
        throws {
        var observedInput = [Float]()
        let report = try PushTPhysicalFlowExperiment.run(configuration: .init(
            populationSize: 8, simulationBatchSize: 8,
            generations: 1, horizon: 8,
            controlPointCount: 4, substeps: 2,
            sourcePreparationSteps: 160,
            initialStandardDeviation: 0.5, eliteFraction: 0.25,
            seed: 79), proposalProvider: { context in
                observedInput = context.input
                return [0.1, -0.2, 0.3, 0.4]
            })
        XCTAssertEqual(
            observedInput.count,
            PushTPhysicalFlowProposalContext.inputDimension)
        XCTAssertTrue(observedInput.allSatisfy(\.isFinite))
        XCTAssertEqual(
            report.teacherSample.canonicalInternalControlPointsXY.count, 4)
        XCTAssertTrue(report.initialProposalSpline.loss.isFinite)
        XCTAssertNotEqual(
            report.generationZeroSelectedProposal, .notApplicable)
        XCTAssertNotNil(report.providedProposalProbeBestLoss)
        XCTAssertNotNil(report.geometricProposalProbeBestLoss)
        XCTAssertEqual(report.generationZeroProbeCandidates, 8)
        XCTAssertEqual(report.generationZeroExploitationCandidates, 0)
        XCTAssertLessThanOrEqual(
            report.optimizedSpline.loss,
            min(report.initialProposalSpline.loss,
                report.geometricProposalSpline.loss) + 1e-6)
    }

    func testExactPhysicsProbeRejectsBadProviderAtMatchedCandidateBudget()
        throws {
        let configuration = PushTPhysicalFlowConfiguration(
            populationSize: 16, simulationBatchSize: 8,
            generations: 1, horizon: 8,
            controlPointCount: 4, substeps: 2,
            sourcePreparationSteps: 160,
            objective: .balancedEndpointBottleneck,
            initialStandardDeviation: 0.5, eliteFraction: 0.25,
            seed: 83)
        let report = try PushTPhysicalFlowExperiment.run(
            configuration: configuration,
            proposalProvider: { _ in [2.95, 2.95, 2.95, 2.95] })

        XCTAssertEqual(report.generationZeroSelectedProposal, .geometric)
        XCTAssertEqual(report.objective, .balancedEndpointBottleneck)
        XCTAssertEqual(
            report.optimizedSpline.loss,
            report.optimizedSpline.balancedEndpointBottleneckLoss,
            accuracy: 1e-6)
        XCTAssertTrue(report.optimizedSpline.legacyWeightedLoss.isFinite)
        XCTAssertTrue(
            report.optimizedSpline.maximumNormalizedEndpointError.isFinite)
        XCTAssertEqual(report.generationZeroProbeCandidates, 8)
        XCTAssertEqual(report.generationZeroExploitationCandidates, 8)
        XCTAssertGreaterThan(
            report.providedProposalProbeBestLoss!,
            report.geometricProposalProbeBestLoss!)
        XCTAssertLessThanOrEqual(
            report.optimizedSpline.loss,
            report.geometricProposalProbeBestLoss! + 1e-6)
        let expectedControlSteps = configuration.simulationBatchSize * (
            report.sourcePreparationSteps + 3 * configuration.horizon)
            + configuration.populationSize * configuration.horizon
        XCTAssertEqual(
            report.simulatedEnvironmentControlSteps, expectedControlSteps)
    }
}
