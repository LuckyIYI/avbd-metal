import XCTest
import simd
@testable import AVBDCore

final class DirectionalShadowProjectionTests: XCTestCase {
    func testMetalDepthConventionMapsNearAndFarPlanes() {
        let near: Float = 0.25
        let far: Float = 40
        let projection = metalOrthographicProjection(
            left: -8, right: 8, bottom: -6, top: 6,
            near: near, far: far)

        let nearClip = projection * SIMD4<Float>(0, 0, -near, 1)
        let farClip = projection * SIMD4<Float>(0, 0, -far, 1)

        XCTAssertEqual(nearClip.z / nearClip.w, 0, accuracy: 1e-6)
        XCTAssertEqual(farClip.z / farClip.w, 1, accuracy: 1e-6)
    }

    func testOrthographicBoundsMapToClipCube() {
        let projection = metalOrthographicProjection(
            left: -8, right: 12, bottom: -3, top: 7,
            near: 1, far: 21)

        let lowerNear = projection * SIMD4<Float>(-8, -3, -1, 1)
        let upperFar = projection * SIMD4<Float>(12, 7, -21, 1)

        XCTAssertEqual(lowerNear.x, -1, accuracy: 1e-6)
        XCTAssertEqual(lowerNear.y, -1, accuracy: 1e-6)
        XCTAssertEqual(lowerNear.z, 0, accuracy: 1e-6)
        XCTAssertEqual(upperFar.x, 1, accuracy: 1e-6)
        XCTAssertEqual(upperFar.y, 1, accuracy: 1e-6)
        XCTAssertEqual(upperFar.z, 1, accuracy: 1e-6)
    }
}
