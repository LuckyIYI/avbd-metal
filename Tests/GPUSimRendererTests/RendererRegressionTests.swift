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

    /// An 8-bit target cannot represent a tone halfway between adjacent sRGB
    /// codes.  Repeating the same rounded code creates a visible contour;
    /// deterministic sub-LSB dithering represents the tone in the local
    /// spatial average instead.
    func testSRGBDitherRepresentsSubCodeTone() {
        let lowerCode = 224
        let targetEncoded = (Float(lowerCode) + 0.42) / 255

        func noise(x: Int, y: Int) -> Float {
            func fract(_ value: Float) -> Float {
                value - floor(value)
            }
            let inner = fract(Float(x) * 0.06711056 + Float(y) * 0.00583715)
            return fract(52.9829189 * inner) - 0.5
        }

        func quantized(_ encoded: Float) -> Float {
            round(max(0, min(encoded, 1)) * 255) / 255
        }

        let undithered = quantized(targetEncoded)
        var ditheredMean: Float = 0
        let side = 64
        for y in 0..<side {
            for x in 0..<side {
                ditheredMean += quantized(
                    targetEncoded + noise(x: x, y: y) / 255)
            }
        }
        ditheredMean /= Float(side * side)

        XCTAssertGreaterThan(abs(undithered - targetEncoded), 0.35 / 255)
        XCTAssertLessThan(abs(ditheredMean - targetEncoded), 0.03 / 255)
    }

    func testRendererDithersFinalToneMappedColorForSRGB8() throws {
        let source = try String(contentsOf: root.appendingPathComponent(
            "Sources/GPUSimRenderer/GPUSimRenderer.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("inline float3 displayColorSRGB8"))
        XCTAssertEqual(
            source.components(separatedBy: "displayColorSRGB8(acesTonemap(").count - 1,
            4)
        XCTAssertTrue(source.contains("static let colorFormat = MTLPixelFormat.bgra8Unorm_srgb"))
    }
}
