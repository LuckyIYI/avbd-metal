import XCTest
import simd
import AVBDCore
import AVBDLearn

final class PublicAPICompatibilityTests: XCTestCase {
    func testLegacyCPUBodyAPIsForwardLegacyDefaults() {
        let direct = CPURigid(
            index: 7, size: F3(repeating: 0.5), density: 2,
            friction: 0.65, position: F3(1, 2, 3))
        XCTAssertEqual(direct.dynamicFriction, 0.65)
        XCTAssertEqual(direct.gravityScale, 1)

        let solver = CPUSolver()
        let inserted = solver.addBody(
            size: F3(repeating: 0.25), density: 3, friction: 0.4,
            position: F3(2, 1, 0))
        XCTAssertTrue(inserted === solver.bodies[0])
        XCTAssertEqual(inserted.dynamicFriction, 0.4)
        XCTAssertEqual(inserted.gravityScale, 1)
    }

    func testLegacySceneBodyAPIsForwardLegacyDefaults() {
        let direct = SceneBody(
            size: F3(1, 2, 3), density: 4, friction: 0.7,
            position: F3(3, 2, 1))
        XCTAssertEqual(direct.dynamicFriction, 0.7)
        XCTAssertNil(direct.mass)
        XCTAssertNil(direct.diagonalInertia)
        XCTAssertEqual(direct.gravityScale, 1)

        var scene = PhysicsScene(name: "legacy-body-api")
        let body = scene.addBody(
            size: F3(repeating: 0.4), density: 2, friction: 0.3,
            position: .zero)
        XCTAssertEqual(scene.bodies[body].dynamicFriction, 0.3)
        XCTAssertEqual(scene.colliders.count, 1)
        XCTAssertEqual(scene.colliders[0].collisionGroup, 0)
        XCTAssertTrue(scene.colliders[0].collisionEnabled)
    }

    func testLegacyJointMotorMapsToOriginalServoContract() {
        let joint = SceneJoint(
            bodyA: -1, bodyB: 0, rA: .zero, rB: .zero,
            hingeAxis: F3(0, 0, 1), motorTarget: 0.25,
            motorTorque: 12, motorRate: 1.5)
        XCTAssertEqual(joint.motorStiffness, 400)
        XCTAssertEqual(joint.motorDamping, 0)
        XCTAssertEqual(joint.motorMode, .implicitPositionPD)
        XCTAssertEqual(joint.motorRate, 1.5)

        // An old call that omitted the defaulted rate must select the legacy
        // overload rather than the modern zero-gain initializer.
        let defaultRate = SceneJoint(
            bodyA: -1, bodyB: 0, rA: .zero, rB: .zero,
            hingeAxis: F3(0, 0, 1), motorTorque: 5)
        XCTAssertEqual(defaultRate.motorStiffness, 400)
        XCTAssertEqual(defaultRate.motorRate, 0)

        let modern = SceneJoint(
            bodyA: -1, bodyB: 0, rA: .zero, rB: .zero,
            hingeAxis: F3(0, 0, 1), motorTorque: 5,
            motorStiffness: 125, motorDamping: 3,
            motorMode: .explicitTorquePD)
        XCTAssertEqual(modern.motorStiffness, 125)
        XCTAssertEqual(modern.motorDamping, 3)
        XCTAssertEqual(modern.motorMode, .explicitTorquePD)
        XCTAssertEqual(modern.motorRate, 0)
    }

    func testOriginalAVBDErrorRemainsExhaustive() {
        func category(_ error: GPUSolver.AVBDError) -> Int {
            switch error {
            case .noDevice: return 0
            case .allocFailed: return 1
            case .shaderCompile: return 2
            case .kernelMissing: return 3
            }
        }

        XCTAssertEqual(category(.noDevice), 0)
        XCTAssertEqual(category(.allocFailed("buffer")), 1)
    }

    func testOriginalPushTSignaturesRemainAddressable() {
        let initializer: (Int, UInt64, Bool) throws -> PushTEnv =
            PushTEnv.init(numEnvs:seed:goalMarkers:)
        let solver: (String, Int, UInt64, Int, Bool, Bool) throws -> Void =
            PushTPipeline.solve(
                modelPath:episodes:seed:latent:debug:oracleNull:)

        _ = initializer
        _ = solver
    }
}
