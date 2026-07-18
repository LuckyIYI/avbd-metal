import XCTest
import simd
@testable import AVBDCore

final class PushTPhysicalBridgeTests: XCTestCase {
    func testActiveContactStateForksReplayIdenticallyAcrossReplicas() throws {
        let replicaCount = 8
        let environment = try PushTEnv(
            numEnvs: replicaCount, seed: 37)
        environment.resetAll(seed: 811)

        var foundContact = false
        for _ in 0..<120 where !foundContact {
            let states = environment.states()
            var targets = states.map(\.tipPosition)
            targets[0] = environment.oracleAction(0)
            environment.step(actions: targets, substeps: 4, maxStep: 0.16)
            foundContact = hasTipBlockContact(environment, replica: 0)
        }
        XCTAssertTrue(foundContact,
                      "the source snapshot must exercise active rigid contact")

        let source = environment.physicalStates()[0]
        environment.fork(source, into: Array(0..<replicaCount))

        let cloned = environment.physicalStates()
        for replica in 1..<replicaCount {
            assertEqual(cloned[0], cloned[replica], accuracy: 3e-5)
        }

        for step in 0..<12 {
            let direction = SIMD2<Float>(0.07, step.isMultiple(of: 2) ? 0.015 : -0.015)
            environment.step(
                actions: (0..<replicaCount).map { _ in
                    source.commandedTipTarget + direction
                },
                substeps: 4, maxStep: 0.16)
        }

        let replayed = environment.physicalStates()
        for replica in 1..<replicaCount {
            assertEqual(replayed[0], replayed[replica], accuracy: 3e-4)
        }
    }

    func testForkedBranchesRemainIndependentUnderDifferentSplinePrefixes()
        throws {
        let environment = try PushTEnv(numEnvs: 4, seed: 91)
        environment.resetAll(seed: 1234)
        let source = environment.physicalStates()[0]
        environment.fork(source, into: [0, 1, 2, 3])

        for _ in 0..<10 {
            environment.step(actions: [
                source.commandedTipTarget + SIMD2(0.5, 0.4),
                source.commandedTipTarget + SIMD2(-0.5, 0.4),
                source.commandedTipTarget + SIMD2(0.5, -0.4),
                source.commandedTipTarget + SIMD2(-0.5, -0.4),
            ], substeps: 4, maxStep: 0.16)
        }

        let states = environment.states()
        for i in states.indices {
            for j in states.indices where j > i {
                XCTAssertGreaterThan(
                    length(states[i].tipPosition - states[j].tipPosition),
                    0.2)
            }
        }
    }

    private func hasTipBlockContact(_ environment: PushTEnv,
                                    replica: Int) -> Bool {
        let reference = environment.refs[replica]
        return environment.solver.activeRigidContactPairs().contains { pair in
            (pair.0 == reference.tip
                && (pair.1 == reference.blockBar
                    || pair.1 == reference.blockStem))
                || (pair.1 == reference.tip
                    && (pair.0 == reference.blockBar
                        || pair.0 == reference.blockStem))
        }
    }

    private func assertEqual(_ lhs: PushTPhysicalState,
                             _ rhs: PushTPhysicalState,
                             accuracy: Float,
                             file: StaticString = #filePath,
                             line: UInt = #line) {
        func vector(_ a: F3, _ b: F3) {
            XCTAssertEqual(a.x, b.x, accuracy: accuracy, file: file, line: line)
            XCTAssertEqual(a.y, b.y, accuracy: accuracy, file: file, line: line)
            XCTAssertEqual(a.z, b.z, accuracy: accuracy, file: file, line: line)
        }
        func quaternion(_ a: Quat, _ b: Quat) {
            let sign: Float = simd_dot(a.vector, b.vector) < 0 ? -1 : 1
            XCTAssertEqual(a.real, sign * b.real, accuracy: accuracy,
                           file: file, line: line)
            vector(a.imag, sign * b.imag)
        }
        func rigid(_ a: GPUSolver.RigidBodyState,
                   _ b: GPUSolver.RigidBodyState) {
            vector(a.position, b.position)
            quaternion(a.rotation, b.rotation)
            vector(a.linearVelocity, b.linearVelocity)
            vector(a.angularVelocity, b.angularVelocity)
        }
        rigid(lhs.tip, rhs.tip)
        rigid(lhs.blockBar, rhs.blockBar)
        rigid(lhs.blockStem, rhs.blockStem)
        XCTAssertEqual(lhs.commandedTipTarget.x,
                       rhs.commandedTipTarget.x,
                       accuracy: accuracy, file: file, line: line)
        XCTAssertEqual(lhs.commandedTipTarget.y,
                       rhs.commandedTipTarget.y,
                       accuracy: accuracy, file: file, line: line)
    }
}
