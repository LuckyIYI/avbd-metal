import simd
import XCTest
@testable import GPUSimRenderer

final class SunDirectionTests: XCTestCase {
    func testSunDirectionSurvivesFastAndHQResolution() {
        for mode in [GPUSimLightingMode.lightweight, .qualityBeta] {
            let options = GPUSimRenderOptions(lightingMode: mode, sunDirection: SIMD3(-2, 0.2, -1))
            for supportsHQ in [true, false] {
                let resolved = options.resolved(supportsHQ: supportsHQ)
                XCTAssertLessThan(simd_distance(resolved.sunDirection, simd_normalize(SIMD3(-2, 0.2, -1))), 1e-6)
            }
        }
    }

    func testInvalidSunDirectionsFallbackAndExtremeFiniteInputsNormalize() {
        let standard = GPUSimRenderOptions().resolved(supportsHQ: true).sunDirection
        for direction: SIMD3<Float> in [.zero, SIMD3(.nan, 0, -1), SIMD3(0, .infinity, -1)] {
            XCTAssertEqual(GPUSimRenderOptions(sunDirection: direction).resolved(supportsHQ: true).sunDirection, standard)
        }
        for direction: SIMD3<Float> in [SIMD3(0, 1, 0), SIMD3(0, -1, 0), SIMD3(repeating: .greatestFiniteMagnitude), SIMD3(1e-35, 0, 0)] {
            let resolved = GPUSimRenderOptions(sunDirection: direction).resolved(supportsHQ: true).sunDirection
            XCTAssertEqual(simd_length(resolved), 1, accuracy: 1e-6)
        }
    }

    func testSunChangesInvalidateOptionEqualityForTemporalHistory() {
        var a = GPUSimRenderOptions.qualityBeta
        let b = a
        a.sunDirection = SIMD3(-1, 0, -1)
        XCTAssertNotEqual(a, b)
    }
}
