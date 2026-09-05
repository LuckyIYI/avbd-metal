import XCTest
import simd
@testable import GPUSimRenderer

final class LightingAccumulationTests: XCTestCase {
    func testStableWorldAccumulatesThenHoldsAndEveryLightingInputInvalidates() {
        let world = NSObject()
        var state = LightingAccumulation()
        var revision: UInt64 = 4, camera = matrix_identity_float4x4
        var light = SIMD3<Float>(0,0,-1), size = SIMD2(800,600)
        var options = GPUSimRenderOptions.qualityBeta
        func next(reset: Bool = false) -> SIMD4<Float> {
            state.advance(world: ObjectIdentifier(world), revision: revision, camera: camera,
                          light: light, size: size, options: options, reset: reset)
        }
        XCTAssertEqual(next().z,0)
        for frame in 1...65 { XCTAssertEqual(next().z,Float(frame)) }
        let held = next()
        for _ in 0..<80 { XCTAssertEqual(next(),held) }
        // Offscreen geometry or emission changes are just as significant as
        // a receiver moving: a fixed camera must not retain obsolete lighting.
        revision += 1; XCTAssertEqual(next().z,0); XCTAssertEqual(next().z,1)
        camera.columns.3.x += 0.001; XCTAssertEqual(next().z,0); XCTAssertEqual(next().z,1)
        light.x += 0.1; XCTAssertEqual(next().z,0); XCTAssertEqual(next().z,1)
        size.x += 1; XCTAssertEqual(next().z,0); XCTAssertEqual(next().z,1)
        options.diffuseGlobalIllumination = false; XCTAssertEqual(next().z,0)
        XCTAssertEqual(next().z,1); XCTAssertEqual(next(reset: true).z,0)
    }

    func testUnknownOrChangingWorldUsesStableFreshSamples() {
        let world = NSObject()
        var state = LightingAccumulation()
        for frame in 0..<100 {
            for revision: UInt64? in [nil,UInt64(frame)] {
                let value = state.advance(world: ObjectIdentifier(world), revision: revision,
                    camera: matrix_identity_float4x4, light: SIMD3(0,0,-1), size: SIMD2(800,600),
                    options: .qualityBeta, reset: false)
                XCTAssertEqual(value,SIMD4(0,1,0,0))
            }
        }
    }
}
