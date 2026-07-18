import XCTest
import simd
@testable import AVBDCore
@testable import AVBDLearn

final class PushTPhysicalFlowExperimentTests: XCTestCase {
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
        XCTAssertLessThanOrEqual(
            report.optimizedSpline.loss,
            min(report.initialProposalSpline.loss,
                report.geometricProposalSpline.loss) + 1e-6)
    }
}
