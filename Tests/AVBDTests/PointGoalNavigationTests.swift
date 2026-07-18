import XCTest
import simd
@testable import AVBDCore

final class PointGoalNavigationTests: XCTestCase {
    func testRevisionAndBodyFrameGeometry() {
        let quarterTurn = Quat(angle: .pi / 2, axis: F3(0, 0, 1))
        let command = PointGoalNavigator.command(
            worldGoal: F3(5, 2, 7),
            bodyPosition: F3(1, 2, -3),
            bodyRotation: quarterTurn,
            parameters: .init(
                goalRadius: 0.1, slowdownDistance: 1,
                cruiseSpeed: 0.6, yawGain: 2,
                maximumYawRate: 1.2,
                mode: .projectedBodyPlane))

        XCTAssertEqual(PointGoalNavigator.revision, 3)
        XCTAssertEqual(command.revision, PointGoalNavigator.revision)
        XCTAssertEqual(command.remainingDistance, 4, accuracy: 1e-6)
        XCTAssertEqual(command.bodyPlanarDelta.x, 0, accuracy: 1e-6)
        XCTAssertEqual(command.bodyPlanarDelta.y, -4, accuracy: 1e-6)
        XCTAssertEqual(command.bodyTwist.x, 0, accuracy: 1e-6)
        XCTAssertEqual(command.bodyTwist.y, -0.6, accuracy: 1e-6)
        XCTAssertEqual(command.bodyTwist.z, -1.2, accuracy: 1e-6)
        XCTAssertFalse(command.reachedGoal)
    }

    func testArachneAdapterIsBitExactWithPreviousCommandLaw() {
        let rotations = [
            Quat(ix: 0, iy: 0, iz: 0, r: 1),
            Quat(angle: 0.71, axis: simd_normalize(F3(0.2, -0.3, 1))),
            Quat(angle: -0.43, axis: simd_normalize(F3(1, 0.4, -0.2))),
        ]
        let positions = [F3(0, 0, 0), F3(-2.3, 1.2, 0.4)]
        let goals = [F3(3, 0, 0), F3(-1, 2, 4), F3(0.05, 0, -9)]

        for rotation in rotations {
            for position in positions {
                for goal in goals {
                    let expected = legacyArachneCommand(
                        worldGoal: goal, rootPosition: position,
                        rootRotation: rotation,
                        goalRadius: 0.12, slowdownDistance: 0.5,
                        cruiseSpeed: 0.6, boundarySpeed: 0.08,
                        maximumYawRate: 0.8)
                    let actual = Arachne15PolicyContract.pointGoalCommand(
                        worldGoal: goal, rootPosition: position,
                        rootRotation: rotation,
                        goalRadius: 0.12, slowdownDistance: 0.5,
                        cruiseSpeed: 0.6, boundarySpeed: 0.08,
                        maximumYawRate: 0.8)
                    assertBitEqual(actual, expected)
                }
            }
        }
    }

    func testForwardOnlyModeIsBitExactWithPreviousH1CommandLaw() {
        let rotations = [
            Quat(ix: 0, iy: 0, iz: 0, r: 1),
            Quat(angle: 1.1, axis: F3(0, 0, 1)),
            Quat(angle: 0.37, axis: simd_normalize(F3(0.5, -0.2, 1))),
        ]
        let position = F3(-0.7, 1.1, 0.9)
        let goals = [
            F3(4, 1.1, 0), F3(-4, -2, 3), F3(-0.7, 1.1, -8),
        ]

        for rotation in rotations {
            for goal in goals {
                let expected = legacyH1Command(
                    worldGoal: goal, rootPosition: position,
                    rootRotation: rotation,
                    goalRadius: 0.75, slowdownDistance: 2.5,
                    cruiseSpeed: 1, boundarySpeed: 0.15)
                let navigation = PointGoalNavigator.command(
                    worldGoal: goal, bodyPosition: position,
                    bodyRotation: rotation,
                    parameters: .init(
                        goalRadius: 0.75, slowdownDistance: 2.5,
                        cruiseSpeed: 1, boundarySpeed: 0.15,
                        yawGain: 0.5, maximumYawRate: 1,
                        mode: .forwardOnlyYaw))
                assertBitEqual(navigation.bodyTwist, expected.command)
                XCTAssertEqual(
                    navigation.desiredWorldHeading.bitPattern,
                    expected.targetHeading.bitPattern)
            }
        }
    }

    func testLegacyHumanoidScalarAdaptersRemainBitExact() {
        for distance in [Float(0), 1.5, 1.500_001, 2.25, 4] {
            let fraction = simd_clamp(
                (distance - 1.5) / (3 - 1.5), 0, 1)
            let expectedSpeed: Float = distance > 1.5
                ? 0.2 + (0.6 - 0.2) * fraction : 0
            let expectedProximity = 1 - fraction
            XCTAssertEqual(
                HumanoidLocomotionObjective.pointGoalCommandSpeed(
                    remainingDistance: distance, cruiseSpeed: 0.6,
                    goalRadius: 1.5, slowdownDistance: 3,
                    boundaryCommandSpeed: 0.2).bitPattern,
                expectedSpeed.bitPattern)
            XCTAssertEqual(
                HumanoidLocomotionObjective.pointGoalProximity(
                    remainingDistance: distance, goalRadius: 1.5,
                    slowdownDistance: 3).bitPattern,
                expectedProximity.bitPattern)
        }
    }

    func testForwardAlignmentGateKeepsConfiguredSlowTurningArc() {
        let command = PointGoalNavigator.command(
            worldGoal: F3(0, 2, 0), bodyPosition: .zero,
            bodyRotation: Quat(ix: 0, iy: 0, iz: 0, r: 1),
            parameters: .init(
                goalRadius: 0.05, slowdownDistance: 0.35,
                cruiseSpeed: 0.3, boundarySpeed: 0.03,
                yawGain: 0.5, maximumYawRate: 1,
                forwardAlignmentSpeedExponent: 2,
                minimumForwardAlignmentScale: 0.25,
                mode: .forwardOnlyYaw))

        XCTAssertEqual(command.bodyTwist.x, 0.075, accuracy: 1e-6)
        XCTAssertEqual(command.bodyTwist.y, 0, accuracy: 1e-6)
        XCTAssertGreaterThan(command.bodyTwist.z, 0)
    }

    private func legacyArachneCommand(
        worldGoal: F3, rootPosition: F3, rootRotation: Quat,
        goalRadius: Float, slowdownDistance: Float,
        cruiseSpeed: Float, boundarySpeed: Float,
        maximumYawRate: Float
    ) -> F3 {
        let worldDelta = worldGoal - rootPosition
        let localDelta = rootRotation.conjugate.act(
            F3(worldDelta.x, worldDelta.y, 0))
        let distance = sqrt(localDelta.x * localDelta.x
            + localDelta.y * localDelta.y)
        guard distance > goalRadius else { return .zero }
        let blend = simd_clamp(
            (distance - goalRadius) / (slowdownDistance - goalRadius), 0, 1)
        let speed = boundarySpeed + blend * (cruiseSpeed - boundarySpeed)
        let direction = SIMD2<Float>(localDelta.x, localDelta.y)
            / max(distance, 1e-6)
        let yawError = atan2(localDelta.y, localDelta.x)
        return F3(direction.x * speed, direction.y * speed,
                  simd_clamp(2 * yawError,
                             -maximumYawRate, maximumYawRate))
    }

    private func legacyH1Command(
        worldGoal: F3, rootPosition: F3, rootRotation: Quat,
        goalRadius: Float, slowdownDistance: Float,
        cruiseSpeed: Float, boundarySpeed: Float
    ) -> (command: F3, targetHeading: Float) {
        let delta = worldGoal - rootPosition
        let distance = sqrt(delta.x * delta.x + delta.y * delta.y)
        let forward = rootRotation.act(F3(1, 0, 0))
        let currentHeading = atan2(forward.y, forward.x)
        guard distance > goalRadius else {
            return (.zero, currentHeading)
        }
        let targetHeading = atan2(delta.y, delta.x)
        let blend = simd_clamp(
            (distance - goalRadius) / (slowdownDistance - goalRadius), 0, 1)
        let speed = boundarySpeed
            + blend * (cruiseSpeed - boundarySpeed)
        var error = targetHeading - currentHeading
        error -= 2 * .pi * floor((error + .pi) / (2 * .pi))
        return (F3(speed, 0, simd_clamp(0.5 * error, -1, 1)),
                targetHeading)
    }

    private func assertBitEqual(
        _ actual: F3, _ expected: F3,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.x.bitPattern, expected.x.bitPattern,
                       file: file, line: line)
        XCTAssertEqual(actual.y.bitPattern, expected.y.bitPattern,
                       file: file, line: line)
        XCTAssertEqual(actual.z.bitPattern, expected.z.bitPattern,
                       file: file, line: line)
    }
}
