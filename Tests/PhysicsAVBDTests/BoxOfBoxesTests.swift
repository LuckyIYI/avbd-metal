import XCTest
@testable import SimCore
@testable import PhysicsAVBD
@testable import GPUSimDemos

final class BoxOfBoxesTests: XCTestCase {
    func testSceneIsRegisteredAndSmallTierFormsDenseBox() throws {
        XCTAssertTrue(Demos.all.contains("boxofboxes"))
        let scene = try XCTUnwrap(Demos.make("boxofboxes", scale: 1))
        let cubes = scene.bodies.filter(\.isDynamic)

        XCTAssertEqual(scene.name, "boxofboxes")
        XCTAssertEqual(cubes.count, 3 * 3 * 5)
        XCTAssertTrue(cubes.allSatisfy { $0.shape == .box })
        XCTAssertEqual(Set(cubes.map(\.position.x)).count, 3)
        XCTAssertEqual(Set(cubes.map(\.position.y)).count, 3)
        XCTAssertEqual(Set(cubes.map(\.position.z)).count, 5)
    }

    func testColossalTierContainsExactlyTwoHundredThousandCubes() {
        let dimensions = Demos.boxOfBoxesDimensions(scale: 16)
        XCTAssertEqual(dimensions.x, 50)
        XCTAssertEqual(dimensions.y, 50)
        XCTAssertEqual(dimensions.z, 80)
        XCTAssertEqual(dimensions.x * dimensions.y * dimensions.z, 200_000)

        let scene = Demos.boxOfBoxes(scale: 16)
        XCTAssertEqual(scene.bodies.count, 200_001) // cubes plus the ground
        XCTAssertEqual(scene.colliders.count, 200_001)
        XCTAssertEqual(scene.bodies.dropFirst().filter(\.isDynamic).count, 200_000)
    }
}
