import MLXRL
import Foundation
import XCTest

final class GEARSonicPolicyTests: XCTestCase {
    func testHistoryPadsAtFrontAndKeepsBatchesIndependent() throws {
        var history = try GEARSonicG1HistoryBuffer(batchSize: 2)
        try history.append(
            baseAngularVelocity: [1, 2, 3, 101, 102, 103],
            jointPositionResidual: values(width: 29, bases: [10, 110]),
            jointVelocity: values(width: 29, bases: [20, 120]),
            previousRawAction: values(width: 29, bases: [30, 130]),
            projectedGravity: [4, 5, 6, 104, 105, 106])

        let output = history.observations()
        XCTAssertEqual(output.count, 2 * 930)
        XCTAssertEqual(Array(output[0..<27]), Array(repeating: 0, count: 27))
        XCTAssertEqual(Array(output[27..<30]), [1, 2, 3])
        XCTAssertEqual(
            Array(output[(30 + 9 * 29)..<(30 + 10 * 29)]),
            Array(values(width: 29, bases: [10]).prefix(29)))
        XCTAssertEqual(Array(output[900..<927]), Array(repeating: 0, count: 27))
        XCTAssertEqual(Array(output[927..<930]), [4, 5, 6])

        let second = 930
        XCTAssertEqual(Array(output[(second + 27)..<(second + 30)]),
                       [101, 102, 103])
        XCTAssertEqual(
            Array(output[(second + 30 + 9 * 29)..<(second + 30 + 10 * 29)]),
            Array(values(width: 29, bases: [110]).prefix(29)))
        XCTAssertEqual(Array(output[(second + 927)..<(second + 930)]),
                       [104, 105, 106])
    }

    func testHistoryRingRetainsNewestTenFramesOldestFirst() throws {
        var history = try GEARSonicG1HistoryBuffer(batchSize: 1)
        for frame in 1...11 {
            let value = Float(frame)
            try history.append(
                baseAngularVelocity: .init(repeating: value, count: 3),
                jointPositionResidual: .init(repeating: value, count: 29),
                jointVelocity: .init(repeating: value, count: 29),
                previousRawAction: .init(repeating: value, count: 29),
                projectedGravity: .init(repeating: value, count: 3))
        }
        XCTAssertEqual(history.storedFrameCount, 10)
        let output = history.observations()
        for logicalFrame in 0..<10 {
            let expected = Float(logicalFrame + 2)
            XCTAssertEqual(
                Array(output[(logicalFrame * 3)..<(logicalFrame * 3 + 3)]),
                [expected, expected, expected])
            let positionOffset = 30 + logicalFrame * 29
            XCTAssertEqual(output[positionOffset], expected)
            XCTAssertEqual(output[positionOffset + 28], expected)
        }
    }

    func testOfficialBundleBatchedMLXParityAndControlMapping() throws {
        let metalLibrary = Bundle(for: Self.self).bundleURL
            .appendingPathComponent("Contents/Resources/mlx-swift_Cmlx.bundle")
            .appendingPathComponent("Contents/Resources/default.metallib")
        guard FileManager.default.fileExists(atPath: metalLibrary.path) else {
            throw XCTSkip(
                "requires an Xcode-packaged MLX default.metallib")
        }
        let directory = try officialBundleDirectory()
        let policy = try GEARSonicG1Policy(directory: directory.path)
        XCTAssertEqual(
            policy.manifest.source.modelRevision,
            "5e22ddc69abcea2a9aafc40536b14c232d3f9d7f")
        XCTAssertEqual(
            policy.manifest.source.encoderSHA256,
            "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3")
        XCTAssertEqual(
            policy.manifest.source.decoderSHA256,
            "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed")

        let verification = try policy.verifyGoldenBatch()
        XCTAssertTrue(verification.passed)
        XCTAssertLessThanOrEqual(
            verification.maximumTokenError, verification.tokenTolerance)
        XCTAssertLessThanOrEqual(
            verification.maximumActionError, verification.actionTolerance)

        let raw = ContiguousArray(
            (0..<58).map { Float($0 < 29 ? $0 : 100 + $0 - 29) })
        let targets = try GEARSonicG1Control.jointPositionTargets(
            rawPolicyActions: raw, control: policy.manifest.control)
        for environment in 0..<2 {
            for actuator in 0..<29 {
                let policyIndex = policy.manifest.control.actuatorToPolicy[actuator]
                let expectedRaw = Float(policyIndex + environment * 100)
                let expected = policy.manifest.control.defaultJointPositions[actuator]
                    + expectedRaw * policy.manifest.control.actionScale[actuator]
                XCTAssertEqual(
                    targets[environment * 29 + actuator], expected, accuracy: 1e-6)
            }
        }

        XCTAssertEqual(
            policy.manifest.control.trainingEffortLimit,
            [139, 139, 88, 139, 50, 50, 139, 139, 88, 139, 50, 50,
             88, 50, 50, 25, 25, 25, 25, 25, 5, 5,
             25, 25, 25, 25, 25, 5, 5])
        XCTAssertEqual(
            policy.manifest.control.deploymentEffortLimit,
            [88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50,
             88, 50, 50, 25, 25, 25, 25, 25, 5, 5,
             25, 25, 25, 25, 25, 5, 5])
        XCTAssertEqual(
            policy.manifest.control.trainingArmature[4],
            2 * 0.003609725, accuracy: 1e-9)
        XCTAssertEqual(
            policy.manifest.control.trainingArmature[13],
            2 * 0.003609725, accuracy: 1e-9)
        XCTAssertEqual(
            policy.manifest.control.stiffness[13],
            28.501246, accuracy: 1e-5)
        XCTAssertEqual(
            policy.manifest.control.stiffness[14],
            28.501246, accuracy: 1e-5)
    }

    private func values(
        width: Int, bases: [Float]
    ) -> ContiguousArray<Float> {
        ContiguousArray(bases.flatMap { base in
            (0..<width).map { base + Float($0) }
        })
    }

    private func officialBundleDirectory() throws -> URL {
        let root = TestPaths.repositoryRoot
        let directory = root.appendingPathComponent(
            "checkpoints/external/gear-sonic-g1", isDirectory: true)
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("manifest.json").path),
              FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("policy.safetensors").path)
        else {
            throw XCTSkip(
                "official GEAR-SONIC bundle is absent; run the import tool first")
        }
        return directory
    }
}
