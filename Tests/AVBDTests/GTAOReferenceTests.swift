import XCTest
import simd
@testable import AVBDCore

/// Validates the GTAO algorithm (CPU port of the shader math) against
/// Monte-Carlo ray-traced ground-truth AO on an analytic scene, and checks
/// that the value at a fixed surface point does not depend on camera
/// distance or azimuth.
final class GTAOReferenceTests: XCTestCase {
    // analytic scene: ground plane z=0 + unit box [-0.5,0.5]^2 x [0,1]
    static let boxMin = F3(-0.5, -0.5, 0)
    static let boxMax = F3(0.5, 0.5, 1)

    func rayScene(_ o: F3, _ d: F3) -> (t: Float, n: F3)? {
        var best: (Float, F3)? = nil
        if abs(d.z) > 1e-8 {
            let t = -o.z / d.z
            if t > 1e-4 { best = (t, F3(0, 0, 1)) }
        }
        // slab test
        var t0: Float = 1e-4, t1 = Float.greatestFiniteMagnitude
        var nrm = F3.zero
        var ok = true
        for ax in 0..<3 {
            let inv = 1 / d[ax]
            var ta = (Self.boxMin[ax] - o[ax]) * inv
            var tb = (Self.boxMax[ax] - o[ax]) * inv
            var sign: Float = -1
            if ta > tb { swap(&ta, &tb); sign = 1 }
            if ta > t0 {
                t0 = ta
                nrm = .zero; nrm[ax] = sign * (d[ax] > 0 ? -1 : -1) * (sign == -1 ? -1 : 1)
                // normal = axis pointing against ray entry side
                nrm = .zero; nrm[ax] = d[ax] > 0 ? -1 : 1
            }
            t1 = min(t1, tb)
            if t0 > t1 { ok = false; break }
        }
        if ok, t0 > 1e-4, best == nil || t0 < best!.0 { best = (t0, nrm) }
        return best.map { (t: $0.0, n: $0.1) }
    }

    /// ground-truth AO: cosine-weighted hemisphere, smooth falloff radius R
    func gtAO(_ P: F3, _ N: F3, R: Float, samples: Int = 4096) -> Float {
        var t = N.x != 0 || N.y != 0 ? normalize(cross(F3(0, 0, 1), N)) : F3(1, 0, 0)
        if N.z == 1 { t = F3(1, 0, 0) }
        let b = cross(N, t)
        var rng = SplitMix64(seed: 42)
        var vis: Float = 0
        for _ in 0..<samples {
            let u1 = rng.nextFloat(), u2 = rng.nextFloat()
            let r = sqrt(u1), phi = 2 * Float.pi * u2
            let d = t * (r * cos(phi)) + b * (r * sin(phi)) + N * sqrt(max(0, 1 - u1))
            if let hit = rayScene(P + N * 1e-3, d) {
                let fade = simd_clamp(1 - (hit.t - R) / R, 0, 1)
                vis += 1 - fade
            } else { vis += 1 }
        }
        return vis / Float(samples)
    }

    // ---- CPU port of the GTAO shader ----
    struct GBuf {
        var w: Int, h: Int
        var pos: [SIMD4<Float>], nrm: [F3]
        func at(_ x: Int, _ y: Int) -> (SIMD4<Float>, F3) {
            let i = min(max(y, 0), h - 1) * w + min(max(x, 0), w - 1)
            return (pos[i], nrm[i])
        }
    }

    func render(eye: F3, look: F3, w: Int, h: Int) -> GBuf {
        let fwd = normalize(look - eye)
        let right = normalize(cross(fwd, F3(0, 0, 1)))
        let up = cross(right, fwd)
        let tanH = tan(25 * Float.pi / 180)
        var pos = [SIMD4<Float>](repeating: .zero, count: w * h)
        var nrm = [F3](repeating: .zero, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let ndcX = (Float(x) + 0.5) / Float(w) * 2 - 1
                let ndcY = 1 - (Float(y) + 0.5) / Float(h) * 2
                let d = normalize(fwd + right * (ndcX * tanH * Float(w) / Float(h))
                                      + up * (ndcY * tanH))
                if let hit = rayScene(eye, d) {
                    let p = eye + d * hit.t
                    pos[y * w + x] = SIMD4(p, 1)
                    nrm[y * w + x] = hit.n
                }
            }
        }
        return GBuf(w: w, h: h, pos: pos, nrm: nrm)
    }

    /// Exact port of `gtao_fragment` (production shader constants included).
    func gtaoPixel(_ g: GBuf, _ px: Int, _ py: Int, eye: F3,
                   camRight: F3, camUpUV: F3, pxPerUnit: Float) -> Float {
        let (P4, Nr) = g.at(px, py)
        if P4.w < 0.5 { return 1 }
        let P = F3(P4.x, P4.y, P4.z)
        let N = normalize(Nr)
        let V = normalize(eye - P)
        let dist = length(eye - P)
        let R: Float = 0.9
        var pxRadius = pxPerUnit * R / max(dist, 0.5)
        let farFade = simd_clamp((pxRadius - 2.5) / 6.0, 0, 1)
        if farFade <= 0 { return 1 }
        pxRadius = min(pxRadius, 96)
        let Reff = pxRadius * max(dist, 0.5) / pxPerUnit
        let ign = ((52.9829189 * ((0.06711056 * Float(px) + 0.00583715 * Float(py))
            .truncatingRemainder(dividingBy: 1))).truncatingRemainder(dividingBy: 1))
        let ang = abs(ign) * .pi
        let stepJit = abs((ign * 7).truncatingRemainder(dividingBy: 1))
        let SLICES = 3, STEPS = 4
        var occl: Float = 0
        for sl in 0..<SLICES {
            let phi = ang + Float(sl) * (.pi / Float(SLICES))
            let dirPx = SIMD2<Float>(cos(phi), sin(phi))
            let dirW = camRight * dirPx.x + camUpUV * dirPx.y
            var omega = dirW - V * dot(dirW, V)
            let ol = length(omega)
            if ol < 1e-4 { occl += 1; continue }
            omega /= ol
            var cosH0: Float = -1, cosH1: Float = -1
            for side in 0..<2 {
                let sgn: Float = side == 0 ? 1 : -1
                for st in 1...STEPS {
                    var t = (Float(st) - 1 + stepJit) / Float(STEPS)
                    t = max(t, 0.02)
                    let sx = px + Int((sgn * dirPx.x * t * pxRadius).rounded())
                    let sy = py + Int((sgn * dirPx.y * t * pxRadius).rounded())
                    let (Q4, _) = g.at(sx, sy)
                    if Q4.w < 0.5 { continue }
                    let wv = F3(Q4.x, Q4.y, Q4.z) - P
                    let l = length(wv)
                    if l < 1e-4 { continue }
                    var c = dot(wv / l, V)
                    let fade = simd_clamp(1 - (l - Reff) / Reff, 0, 1)
                    c = -1 + (c + 1) * fade
                    if dot(wv, omega) >= 0 { cosH0 = max(cosH0, c) }
                    else { cosH1 = max(cosH1, c) }
                }
            }
            let sliceN = cross(V, omega)
            var projN = N - sliceN * dot(N, sliceN)
            let projLen = length(projN)
            if projLen < 1e-4 { occl += 1; continue }
            projN /= projLen
            let cosNV = simd_clamp(dot(projN, V), -1, 1)
            let n = acos(cosNV) * (dot(projN, omega) >= 0 ? 1 : -1)
            var h1 = acos(simd_clamp(cosH0, -1, 1))
            var h2 = -acos(simd_clamp(cosH1, -1, 1))
            h1 = n + min(h1 - n, .pi / 2)
            h2 = n + max(h2 - n, -.pi / 2)
            let a1 = 0.25 * (-cos(2 * h1 - n) + cosNV + 2 * h1 * sin(n))
            let a2 = 0.25 * (-cos(2 * h2 - n) + cosNV + 2 * h2 * sin(n))
            occl += projLen * (a1 + a2)
        }
        var ao = simd_clamp(occl / Float(SLICES), 0, 1)
        ao = pow(ao, 1.25)
        return 1 - farFade + ao * farFade
    }

    /// average GTAO in a small pixel neighborhood (blur surrogate)
    func gtaoAt(worldPoint: F3, eye: F3, look: F3) -> Float {
        let w = 480, h = 300
        let g = render(eye: eye, look: look, w: w, h: h)
        let fwd = normalize(look - eye)
        let right = normalize(cross(fwd, F3(0, 0, 1)))
        let up = cross(right, fwd)
        let tanH = tan(25 * Float.pi / 180)
        // project the world point
        let rel = worldPoint - eye
        let zc = dot(rel, fwd)
        let xc = dot(rel, right) / (zc * tanH * Float(w) / Float(h))
        let yc = dot(rel, up) / (zc * tanH)
        let px = Int((xc * 0.5 + 0.5) * Float(w))
        let py = Int((1 - (yc * 0.5 + 0.5)) * Float(h))
        let pxPerUnit = Float(h) * 0.5 / tanH
        var acc: Float = 0; var cnt: Float = 0
        for dy in -2...2 {
            for dx in -2...2 {
                let (q4, _) = g.at(px + dx, py + dy)
                if q4.w < 0.5 { continue }
                if length(F3(q4.x, q4.y, q4.z) - worldPoint) > 0.25 { continue }
                acc += gtaoPixel(g, px + dx, py + dy, eye: eye,
                                 camRight: right, camUpUV: -up, pxPerUnit: pxPerUnit)
                cnt += 1
            }
        }
        return cnt > 0 ? acc / cnt : -1
    }

    func testGroundTruthAndDistanceInvariance() throws {
        // probe points: floor near box wall, floor far from box, wall center
        let probes: [(String, F3, F3)] = [
            ("floor near wall", F3(0.85, 0, 0), F3(0, 0, 1)),
            ("floor 1.5 from box", F3(2.0, 0, 0), F3(0, 0, 1)),
            ("open floor", F3(6, 4, 0), F3(0, 0, 1)),
        ]
        for (name, P, N) in probes {
            let gt = gtAO(P, N, R: 0.9)
            var vals: [Float] = []
            for d in [Float(6), 10, 16] {
                for az in [Float(0.5), 1.2] {
                    let eye = P + F3(cos(az) * d * 0.8, sin(az) * d * 0.8, d * 0.55)
                    let v = gtaoAt(worldPoint: P, eye: eye, look: P)
                    if v >= 0 { vals.append(v) }
                }
            }
            let mean = vals.reduce(0, +) / Float(vals.count)
            let spread = vals.map { abs($0 - mean) }.max() ?? 0
            print("\(name): GT=\(String(format: "%.3f", gt)) gtao=\(vals.map { String(format: "%.3f", $0) }.joined(separator: " ")) mean=\(String(format: "%.3f", mean)) spread=\(String(format: "%.3f", spread))")
            XCTAssertLessThan(abs(mean - gt), 0.1,
                              "\(name): GTAO mean deviates from ray-traced GT")
            XCTAssertLessThan(spread, 0.08,
                              "\(name): GTAO must not depend on camera pose")
            _ = N
        }
    }
}
