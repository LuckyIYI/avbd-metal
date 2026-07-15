import XCTest
import simd
@testable import AVBDCore

final class CPUSolverTests: XCTestCase {
    func testFrictionCombineModesMatchAuthoredMaterialRules() {
        XCTAssertEqual(FrictionCombineMode.geometricMean.combine(0.8, 0.5),
                       sqrt(0.4), accuracy: 1e-6)
        XCTAssertEqual(FrictionCombineMode.multiply.combine(0.8, 0.5),
                       0.4, accuracy: 1e-6)
        XCTAssertEqual(FrictionCombineMode.minimum.combine(0.8, 0.5), 0.5)
        XCTAssertEqual(FrictionCombineMode.maximum.combine(0.8, 0.5), 0.8)
        XCTAssertEqual(FrictionCombineMode.average.combine(0.8, 0.5), 0.65)
    }

    /// A single box resting on a static ground should settle and not sink.
    func testBoxRestsOnGround() {
        let solver = CPUSolver()
        _ = solver.addBody(size: F3(100, 100, 1), density: 0, friction: 0.5,
                           position: F3(0, 0, -0.5))
        let box = solver.addBody(size: F3(1, 1, 1), density: 1, friction: 0.5,
                                 position: F3(0, 0, 0.6))

        for _ in 0..<120 { solver.step() }

        XCTAssertEqual(box.positionLin.z, 0.5, accuracy: 0.03,
                       "box should rest at z=0.5 (got \(box.positionLin.z))")
        XCTAssertLessThan(length(box.velocityLin), 0.05, "box should be at rest")
    }

    /// Free fall matches analytic ballistic trajectory.
    func testFreeFallMatchesAnalytic() {
        let solver = CPUSolver()
        let box = solver.addBody(size: F3(1, 1, 1), density: 1, friction: 0.5,
                                 position: F3(0, 0, 100))
        let steps = 30
        for _ in 0..<steps { solver.step() }

        // Implicit Euler free fall: v_{n+1} = v_n + g*dt; x_{n+1} = x_n + v_{n+1}*dt
        var v: Float = 0, z: Float = 100
        for _ in 0..<steps {
            v += solver.gravity * solver.dt
            z += v * solver.dt
        }
        XCTAssertEqual(box.positionLin.z, z, accuracy: 0.01)
    }

    /// Hard joint constraint: pendulum link stays attached.
    func testPendulumJointError() {
        let solver = CPUSolver()
        let bob = solver.addBody(size: F3(0.5, 0.5, 0.5), density: 1, friction: 0.5,
                                 position: F3(1, 0, 0))
        // attach world anchor (0,0,0) to bob local corner (-1,0,0) -> 1m arm
        solver.addJoint(nil, bob, rA: F3(0, 0, 0), rB: F3(-1, 0, 0))

        for _ in 0..<120 { solver.step() }
        XCTAssertLessThan(solver.maxConstraintError(), 0.01,
                          "joint constraint should hold within 1cm")
    }

    /// Stack of boxes should not collapse or sink with few iterations.
    func testSmallStackStability() {
        let solver = CPUSolver()
        solver.iterations = 10
        _ = solver.addBody(size: F3(100, 100, 1), density: 0, friction: 0.7,
                           position: F3(0, 0, -0.5))
        var tops: [CPURigid] = []
        for i in 0..<5 {
            tops.append(solver.addBody(size: F3(1, 1, 1), density: 1, friction: 0.7,
                                       position: F3(0, 0, 0.5 + Float(i) * 1.0)))
        }
        for _ in 0..<300 { solver.step() }

        for (i, b) in tops.enumerated() {
            XCTAssertEqual(b.positionLin.z, 0.5 + Float(i), accuracy: 0.08,
                           "box \(i) should stay in stack at z=\(0.5 + Float(i))")
            XCTAssertLessThan(length(F3(b.positionLin.x, b.positionLin.y, 0)), 0.15,
                              "box \(i) should not drift laterally")
        }
    }
}
