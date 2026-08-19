import MLXRL
import Foundation
import XCTest

final class GEARSonicReferenceTests: XCTestCase {
    func testObservationUsesPolicyOrderingAndClampsFutureFrames() throws {
        let clip = try makeClip(frameCount: 13)
        let identity = quaternion(yaw: 0)
        let alignment = try clip.headingAlignment(
            robotInitialBaseQuaternionWXYZ: identity, referenceFrame: 0)
        let observation = try clip.referenceObservation640(
            currentFrame: 2, isPlaying: true,
            robotBaseQuaternionWXYZ: identity,
            headingAlignment: alignment)

        XCTAssertEqual(observation.count, 640)
        let frames = [2, 7, 12, 12, 12, 12, 12, 12, 12, 12]
        for (sample, frame) in frames.enumerated() {
            let qOffset = sample * 29
            let qdOffset = 290 + sample * 29
            XCTAssertEqual(observation[qOffset], Double(frame * 1_000))
            XCTAssertEqual(observation[qOffset + 28], Double(frame * 1_000 + 28))
            XCTAssertEqual(observation[qdOffset], Double(100_000 + frame * 1_000))
            XCTAssertEqual(
                observation[qdOffset + 28], Double(100_000 + frame * 1_000 + 28))
            XCTAssertEqual(
                Array(observation[(580 + sample * 6)..<(586 + sample * 6)]),
                [1, 0, 0, 1, 0, 0])
        }
    }

    func testPausedReferenceRepeatsPoseAndZerosAllFutureVelocity() throws {
        let clip = try makeClip(frameCount: 8) { quaternion(yaw: Double($0) * 0.1) }
        let identity = quaternion(yaw: 0)
        let alignment = try clip.headingAlignment(
            robotInitialBaseQuaternionWXYZ: identity, referenceFrame: 0)
        let observation = try clip.referenceObservation640(
            currentFrame: 3, isPlaying: false,
            robotBaseQuaternionWXYZ: identity,
            headingAlignment: alignment)

        for sample in 0..<10 {
            XCTAssertEqual(observation[sample * 29], 3_000)
            XCTAssertEqual(observation[sample * 29 + 28], 3_028)
        }
        XCTAssertTrue(observation[290..<580].allSatisfy { $0 == 0 })
        let firstAnchor = Array(observation[580..<586])
        for sample in 1..<10 {
            XCTAssertEqual(
                Array(observation[(580 + sample * 6)..<(586 + sample * 6)]),
                firstAnchor)
        }
    }

    func testHeadingAlignmentIdentityAndNonzeroYaw() throws {
        let identityClip = try makeClip(frameCount: 1)
        let identity = quaternion(yaw: 0)
        let identityAlignment = try identityClip.headingAlignment(
            robotInitialBaseQuaternionWXYZ: identity, referenceFrame: 0)
        let identityObservation = try identityClip.referenceObservation640(
            currentFrame: 0, isPlaying: true,
            robotBaseQuaternionWXYZ: identity,
            headingAlignment: identityAlignment)
        XCTAssertEqual(Array(identityObservation[580..<586]), [1, 0, 0, 1, 0, 0])

        let referenceYaw = 60.0 * Double.pi / 180
        let clip = try makeClip(frameCount: 1) { _ in quaternion(yaw: referenceYaw) }
        // qApply = yaw(90 - 30 + 15) = yaw(75). The current reference is 60,
        // and current robot is 80, so the relative encoder rotation is 55 deg.
        let alignment = try GEARSonicG1HeadingAlignment(
            robotInitialBaseQuaternionWXYZ: quaternion(yaw: degrees(90)),
            referenceInitialRootQuaternionWXYZ: quaternion(yaw: degrees(30)),
            deltaHeadingRadians: degrees(15))
        let observation = try clip.referenceObservation640(
            currentFrame: 0, isPlaying: true,
            robotBaseQuaternionWXYZ: quaternion(yaw: degrees(80)),
            headingAlignment: alignment)
        let angle = degrees(55)
        let expected = [cos(angle), -sin(angle), sin(angle), cos(angle), 0, 0]
        for index in 0..<6 {
            XCTAssertEqual(observation[580 + index], expected[index], accuracy: 1e-12)
        }
    }

    func testStrictCSVLoaderPreservesDoublesAndAcceptsExactHeaderlessRows() throws {
        let directory = try temporaryClipDirectory()
        let precise = "0.12345678901234567"
        let qHeader = (0..<29).map { "joint_\($0)" }.joined(separator: ",")
        let q = ([precise] + (1..<29).map(String.init)).joined(separator: ",")
        let qd = (0..<29).map { String(Double($0) + 0.5) }.joined(separator: ",")
        let bodyHeader = (0..<14).flatMap { body in
            ["body_\(body)_x", "body_\(body)_y", "body_\(body)_z"]
        }.joined(separator: ",")
        let body = (0..<42).map { String(Double($0) / 10) }.joined(separator: ",")
        let quaternionHeader = (0..<14).flatMap { body in
            ["body_\(body)_w", "body_\(body)_x", "body_\(body)_y", "body_\(body)_z"]
        }.joined(separator: ",")
        let quaternionRow = Array(repeating: ["1", "0", "0", "0"], count: 14)
            .flatMap { $0 }.joined(separator: ",")

        try write("\(qHeader)\n\(q)\n", named: "joint_pos.csv", to: directory)
        // Headerless input is deliberately supported, but remains exact-width.
        try write("\(qd)\n", named: "joint_vel.csv", to: directory)
        try write("\(bodyHeader)\n\(body)\n", named: "body_pos.csv", to: directory)
        try write(
            "\(quaternionHeader)\n\(quaternionRow)\n",
            named: "body_quat.csv", to: directory)

        let clip = try GEARSonicG1ReferenceClip(directory: directory)
        XCTAssertEqual(clip.frameCount, 1)
        XCTAssertEqual(clip.jointPositionsPolicyOrder[0][0], Double(precise)!)
        XCTAssertEqual(clip.jointPositionsPolicyOrder[0][28], 28)
        XCTAssertEqual(clip.jointVelocitiesPolicyOrder[0][0], 0.5)
        XCTAssertEqual(clip.bodyPositions[0][41], 4.1)
        XCTAssertEqual(clip.bodyQuaternionsWXYZ[0][52..<56], [1, 0, 0, 0])
        XCTAssertNil(clip.bodyLinearVelocities)
        XCTAssertNil(clip.bodyAngularVelocities)
        XCTAssertNil(try clip.rootLinearVelocity(at: 0))
        XCTAssertNil(try clip.rootAngularVelocity(at: 0))
    }

    func testMalformedCSVRowsAndFrameCountMismatchAreRejected() throws {
        let directory = try temporaryClipDirectory()
        try writeValidOneFrameClip(to: directory)
        let shortVelocity = (0..<28).map(String.init).joined(separator: ",")
        try write("\(shortVelocity)\n", named: "joint_vel.csv", to: directory)

        XCTAssertThrowsError(try GEARSonicG1ReferenceClip(directory: directory)) {
            guard case let GEARSonicG1ReferenceError.malformedCSV(file, line, _) = $0
            else { return XCTFail("unexpected error: \($0)") }
            XCTAssertEqual(file, "joint_vel.csv")
            XCTAssertEqual(line, 1)
        }

        try writeValidOneFrameClip(to: directory)
        let qHeader = (0..<29).map { "joint_\($0)" }.joined(separator: ",")
        let qRow = (0..<29).map(String.init).joined(separator: ",")
        try write("\(qHeader)\n\(qRow)\n\(qRow)\n", named: "joint_pos.csv", to: directory)
        XCTAssertThrowsError(try GEARSonicG1ReferenceClip(directory: directory)) {
            guard case GEARSonicG1ReferenceError.inconsistentFrameCount = $0
            else { return XCTFail("unexpected error: \($0)") }
        }
    }

    func testOptionalBodyVelocitiesPreserveDoublesAndRequireExactPair() throws {
        let directory = try temporaryClipDirectory()
        try writeValidOneFrameClip(to: directory)
        let linearHeader = bodyVelocityHeader(prefix: "vel")
        let angularHeader = bodyVelocityHeader(prefix: "angvel")
        let linear = (["0.12345678901234567", "-2.5", "3.25"]
            + Array(repeating: "0", count: 39)).joined(separator: ",")
        let angular = (["-0.75", "0.5", "1.125"]
            + Array(repeating: "0", count: 39)).joined(separator: ",")
        try write(
            "\(linearHeader)\n\(linear)\n",
            named: "body_lin_vel.csv", to: directory)
        try write(
            "\(angularHeader)\n\(angular)\n",
            named: "body_ang_vel.csv", to: directory)

        let clip = try GEARSonicG1ReferenceClip(directory: directory)
        XCTAssertEqual(
            clip.bodyLinearVelocities?[0][0],
            Double("0.12345678901234567")!)
        XCTAssertEqual(
            try clip.rootLinearVelocity(at: 0),
            GEARSonicXYZVector(
                x: Double("0.12345678901234567")!, y: -2.5, z: 3.25))
        XCTAssertEqual(
            try clip.rootAngularVelocity(at: 0),
            GEARSonicXYZVector(x: -0.75, y: 0.5, z: 1.125))

        try FileManager.default.removeItem(
            at: directory.appendingPathComponent("body_ang_vel.csv"))
        XCTAssertThrowsError(try GEARSonicG1ReferenceClip(directory: directory)) {
            guard case GEARSonicG1ReferenceError.incompleteBodyVelocityPair = $0
            else { return XCTFail("unexpected error: \($0)") }
        }

        try write(
            "\(angularHeader)\n\(angular)\n",
            named: "body_ang_vel.csv", to: directory)
        try write(
            "\(linearHeader)\n\(linear)\n\(linear)\n",
            named: "body_lin_vel.csv", to: directory)
        XCTAssertThrowsError(try GEARSonicG1ReferenceClip(directory: directory)) {
            guard case GEARSonicG1ReferenceError.inconsistentBodyVelocityFrameCount = $0
            else { return XCTFail("unexpected error: \($0)") }
        }
    }

    private func makeClip(
        frameCount: Int,
        rootQuaternion: (Int) -> GEARSonicWXYZQuaternion = { _ in
            GEARSonicWXYZQuaternion(w: 1, x: 0, y: 0, z: 0)
        }
    ) throws -> GEARSonicG1ReferenceClip {
        let q = (0..<frameCount).map { frame in
            (0..<29).map { Double(frame * 1_000 + $0) }
        }
        let qd = (0..<frameCount).map { frame in
            (0..<29).map { Double(100_000 + frame * 1_000 + $0) }
        }
        let body = (0..<frameCount).map { _ in [Double](repeating: 0, count: 42) }
        let quaternions = (0..<frameCount).map { frame -> [Double] in
            let root = rootQuaternion(frame)
            var row = [Double](repeating: 0, count: 56)
            for body in 0..<14 { row[body * 4] = 1 }
            row.replaceSubrange(0..<4, with: [root.w, root.x, root.y, root.z])
            return row
        }
        return try GEARSonicG1ReferenceClip(
            jointPositionsPolicyOrder: q,
            jointVelocitiesPolicyOrder: qd,
            bodyPositions: body,
            bodyQuaternionsWXYZ: quaternions)
    }

    private func degrees(_ value: Double) -> Double { value * .pi / 180 }

    private func quaternion(yaw: Double) -> GEARSonicWXYZQuaternion {
        GEARSonicWXYZQuaternion(
            w: cos(yaw * 0.5), x: 0, y: 0, z: sin(yaw * 0.5))
    }

    private func temporaryClipDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gear-sonic-reference-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    private func write(_ contents: String, named name: String, to directory: URL) throws {
        try Data(contents.utf8).write(to: directory.appendingPathComponent(name))
    }

    private func writeValidOneFrameClip(to directory: URL) throws {
        let qHeader = (0..<29).map { "joint_\($0)" }.joined(separator: ",")
        let qdHeader = (0..<29).map { "joint_vel_\($0)" }.joined(separator: ",")
        let qRow = (0..<29).map(String.init).joined(separator: ",")
        let bodyHeader = (0..<14).flatMap { body in
            ["body_\(body)_x", "body_\(body)_y", "body_\(body)_z"]
        }.joined(separator: ",")
        let bodyRow = Array(repeating: "0", count: 42).joined(separator: ",")
        let quaternionHeader = (0..<14).flatMap { body in
            ["body_\(body)_w", "body_\(body)_x", "body_\(body)_y", "body_\(body)_z"]
        }.joined(separator: ",")
        let quaternionRow = Array(repeating: ["1", "0", "0", "0"], count: 14)
            .flatMap { $0 }.joined(separator: ",")
        try write("\(qHeader)\n\(qRow)\n", named: "joint_pos.csv", to: directory)
        try write("\(qdHeader)\n\(qRow)\n", named: "joint_vel.csv", to: directory)
        try write("\(bodyHeader)\n\(bodyRow)\n", named: "body_pos.csv", to: directory)
        try write(
            "\(quaternionHeader)\n\(quaternionRow)\n",
            named: "body_quat.csv", to: directory)
    }

    private func bodyVelocityHeader(prefix: String) -> String {
        (0..<14).flatMap { body in
            ["body_\(body)_\(prefix)_x", "body_\(body)_\(prefix)_y",
             "body_\(body)_\(prefix)_z"]
        }.joined(separator: ",")
    }
}
