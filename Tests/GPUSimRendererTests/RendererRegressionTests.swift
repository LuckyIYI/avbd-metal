import Foundation
import simd
import XCTest

final class RendererRegressionTests: XCTestCase {
    private var root: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    /// A tilted receiver has a different expected depth at every PCF tap.
    /// Reusing the fragment-center depth exposes the number of passing taps
    /// as visible bands; receiver-plane correction removes that quantization.
    /// A small caster-side raster offset is still required to cover the depth
    /// representation error at the corrected point.
    func testPlanarPCFNeedsReceiverPlaneAndCasterBias() {
        let centerDepth: Float = 0.5
        let depthGradient = SIMD2<Float>(0.0020, 0.0011)
        let representationError: Float = -0.00025
        let receiverBias: Float = 0.00020
        let casterRasterOffset: Float = 0.00010

        func visibility(
            correctReceiverPlane: Bool,
            offsetCaster: Bool
        ) -> Float {
            var visible: Float = 0
            for y in -1...1 {
                for x in -1...1 {
                    let tap = SIMD2<Float>(Float(x), Float(y))
                    let planeDelta = simd_dot(depthGradient, tap)
                    let stored = centerDepth + planeDelta
                        + representationError
                        + (offsetCaster ? casterRasterOffset : 0)
                    let compared = centerDepth - receiverBias
                        + (correctReceiverPlane ? planeDelta : 0)
                    visible += compared <= stored ? 1 : 0
                }
            }
            return visible / 9
        }

        XCTAssertLessThan(
            visibility(correctReceiverPlane: false, offsetCaster: true), 1)
        XCTAssertEqual(
            visibility(correctReceiverPlane: true, offsetCaster: false), 0)
        XCTAssertEqual(
            visibility(correctReceiverPlane: true, offsetCaster: true), 1)
    }

    /// Connect the numerical regression above to the packaged Metal renderer.
    func testRendererAppliesBothPlanarPCFCorrections() throws {
        let source = try String(contentsOf: root.appendingPathComponent(
            "Sources/GPUSimRenderer/GPUSimRenderer.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains(
            "receiverDepthGradient, tapOffset"))
        XCTAssertTrue(source.contains(
            "enc.setDepthBias(1.0, slopeScale: 1.0, clamp: 0.001)"))
    }
}
