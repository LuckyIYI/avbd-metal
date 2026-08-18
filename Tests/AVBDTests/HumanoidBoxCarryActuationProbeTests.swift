import XCTest
@testable import MLXRL

final class HumanoidBoxCarryActuationProbeTests: XCTestCase {
    func testProbeUsesMeanAndAntitheticBoundedCandidates() {
        var generator = ProbeRandomNumberGenerator(seed: 71)
        let mean = [Float](repeating: 0, count: 8)
        let candidates = HumanoidBoxCarryActuationProbe.sampledTargets(
            count: 9, mean: mean,
            standardDeviation: [Float](repeating: 0.1, count: 8),
            generator: &generator)

        XCTAssertEqual(candidates.count, 9)
        XCTAssertEqual(candidates[0], mean)
        XCTAssertTrue(candidates.allSatisfy {
            $0.count == 8 && $0.allSatisfy { (-0.999...0.999).contains($0) }
        })
        for arm in 0..<8 {
            XCTAssertEqual(candidates[1][arm] + candidates[2][arm], 0,
                           accuracy: 1e-6)
            XCTAssertEqual(candidates[3][arm] + candidates[4][arm], 0,
                           accuracy: 1e-6)
        }
    }

    func testProbeRejectsDegeneratePopulation() {
        let configuration = HumanoidBoxCarryActuationProbeConfiguration(
            populationSize: 1)
        XCTAssertThrowsError(try configuration.validate())
    }

    func testPhysicalLiftTierOutranksSupportedPivotScore() {
        XCTAssertTrue(HumanoidBoxCarryActuationProbe.isFeasibilityPreferred(
            succeeded: false, physicallyLifted: true, score: 1,
            overSucceeded: false, overPhysicallyLifted: false,
            overScore: 10_000))
        XCTAssertFalse(HumanoidBoxCarryActuationProbe.isFeasibilityPreferred(
            succeeded: false, physicallyLifted: false, score: 10_000,
            overSucceeded: false, overPhysicallyLifted: true,
            overScore: 1))
        XCTAssertTrue(HumanoidBoxCarryActuationProbe.isFeasibilityPreferred(
            succeeded: true, physicallyLifted: true, score: 1,
            overSucceeded: false, overPhysicallyLifted: true,
            overScore: 10_000))
    }

    func testProbeProjectsArmTargetsToBilateralH1Symmetry() {
        let projected = HumanoidBoxCarryActuationProbe
            .bilaterallySymmetricArmTarget([
                0.8, 0.6, -0.4, 0.2, 0.4, -0.2, 0.8, 0.6,
            ])
        XCTAssertEqual(projected[0], projected[4], accuracy: 1e-6)
        XCTAssertEqual(projected[1], -projected[5], accuracy: 1e-6)
        XCTAssertEqual(projected[2], -projected[6], accuracy: 1e-6)
        XCTAssertEqual(projected[3], projected[7], accuracy: 1e-6)
        XCTAssertEqual(projected[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(projected[1], 0.4, accuracy: 1e-6)
        XCTAssertEqual(projected[2], -0.6, accuracy: 1e-6)
        XCTAssertEqual(projected[3], 0.4, accuracy: 1e-6)
    }

    func testProbeTrajectoryStartsContinuouslyAndMirrorsEveryKnot() {
        let parameters: [Float] = [
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2,
        ]
        let start = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
            parameters, knotCount: 3, progress: 0)
        XCTAssertEqual(start, [Float](repeating: 0, count: 8))

        let first = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
            parameters, knotCount: 3, progress: 1.0 / 3.0)
        XCTAssertEqual(first[0], 0.1, accuracy: 1e-6)
        XCTAssertEqual(first[1], 0.2, accuracy: 1e-6)
        XCTAssertEqual(first[4], first[0], accuracy: 1e-6)
        XCTAssertEqual(first[5], -first[1], accuracy: 1e-6)
        XCTAssertEqual(first[6], -first[2], accuracy: 1e-6)
        XCTAssertEqual(first[7], first[3], accuracy: 1e-6)

        let middle = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
            parameters, knotCount: 3, progress: 0.5)
        XCTAssertEqual(middle[0], 0.3, accuracy: 1e-6)
        XCTAssertEqual(middle[3], 0.6, accuracy: 1e-6)
        let end = HumanoidBoxCarryActuationProbe.trajectoryArmDelta(
            parameters, knotCount: 3, progress: 1)
        XCTAssertEqual(Array(end.prefix(4)), [0.9, 1.0, 1.1, 1.2])
    }
}
