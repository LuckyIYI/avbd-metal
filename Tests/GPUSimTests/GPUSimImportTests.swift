import XCTest
import GPUSim

final class GPUSimImportTests: XCTestCase {
    func testSingleImportBuildsAndStepsAScene() throws {
        var scene = PhysicsScene(name: "package-consumer-smoke")
        _ = scene.addBody(
            size: F3(10, 10, 1),
            density: 0,
            friction: 0.8,
            position: F3(0, 0, -0.5)
        )
        let dynamicBody = scene.addBody(
            size: F3(repeating: 1),
            density: 1,
            friction: 0.5,
            position: F3(0, 0, 2)
        )

        let solver = try scene.makeCPUSolverChecked()
        try solver.stepChecked()

        XCTAssertTrue(solver.bodies[dynamicBody].positionLin.z.isFinite)
    }
}
