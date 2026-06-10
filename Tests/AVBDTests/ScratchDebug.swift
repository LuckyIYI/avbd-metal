import XCTest
import simd
@testable import AVBDCore

final class ScratchDebug: XCTestCase {
    func testWatchStall() throws {
        let env = try PushTEnv(numEnvs: 1, seed: 11 + 5 * 7)
        for step in 0..<300 {
            let target = env.oracleAction(0)
            env.step(actions: [target])
            if step >= 120 && step % 20 == 0 {
                let tp = env.tipPos(0)
                let (bp, _) = env.blockPose(0)
                let phase = length(tp - (bp - normalize(env.refs[0].goalPos - bp) * 0.42)) > 0.20 ? "REPOSITION" : "PUSH"
                print("s\(step) [\(phase)] tip (\(String(format: "%.2f", tp.x)), \(String(format: "%.2f", tp.y))) block (\(String(format: "%.2f", bp.x)), \(String(format: "%.2f", bp.y))) tgt (\(String(format: "%.2f", target.x)), \(String(format: "%.2f", target.y)))")
            }
            if env.success(0) { print("SOLVED \(step)"); break }
        }
    }
}
