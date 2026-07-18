import Foundation

/// A scalar-first quaternion matching GEAR-SONIC's deployment convention.
public struct GEARSonicWXYZQuaternion: Sendable, Equatable {
    public let w: Double
    public let x: Double
    public let y: Double
    public let z: Double

    public init(w: Double, x: Double, y: Double, z: Double) {
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    }
}

public struct GEARSonicXYZVector: Sendable, Equatable {
    public let x: Double
    public let y: Double
    public let z: Double

    public init(x: Double, y: Double, z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public enum GEARSonicG1ReferenceError: Error, LocalizedError, Sendable {
    case unreadableFile(path: String, reason: String)
    case malformedCSV(file: String, line: Int, reason: String)
    case inconsistentFrameCount(
        jointPositions: Int, jointVelocities: Int,
        bodyPositions: Int, bodyQuaternions: Int)
    case incompleteBodyVelocityPair
    case inconsistentBodyVelocityFrameCount(
        expected: Int, linear: Int, angular: Int)
    case invalidFrameIndex(index: Int, frameCount: Int)
    case invalidQuaternion(context: String)

    public var errorDescription: String? {
        switch self {
        case let .unreadableFile(path, reason):
            return "could not read GEAR-SONIC reference file \(path): \(reason)"
        case let .malformedCSV(file, line, reason):
            return "malformed GEAR-SONIC \(file) at line \(line): \(reason)"
        case let .inconsistentFrameCount(q, qd, body, quaternion):
            return "GEAR-SONIC reference frame counts differ "
                + "(q=\(q), qd=\(qd), body_pos=\(body), body_quat=\(quaternion))"
        case .incompleteBodyVelocityPair:
            return "GEAR-SONIC body_lin_vel.csv and body_ang_vel.csv "
                + "must either both be present or both be absent"
        case let .inconsistentBodyVelocityFrameCount(expected, linear, angular):
            return "GEAR-SONIC body velocity frame counts differ "
                + "(expected=\(expected), linear=\(linear), angular=\(angular))"
        case let .invalidFrameIndex(index, count):
            return "GEAR-SONIC reference frame \(index) is outside 0..<\(count)"
        case let .invalidQuaternion(context):
            return "GEAR-SONIC \(context) quaternion is non-finite or degenerate"
        }
    }
}

/// Heading state captured when a GEAR-SONIC reference is reset.
///
/// The application rotation is exactly
/// `yaw(delta) * heading(robotInitial) * inverse(heading(referenceInitial))`.
/// Quaternion products are Hamilton products in WXYZ order.
public struct GEARSonicG1HeadingAlignment: Sendable, Equatable {
    public let robotInitialBaseQuaternionWXYZ: GEARSonicWXYZQuaternion
    public let referenceInitialRootQuaternionWXYZ: GEARSonicWXYZQuaternion
    public let deltaHeadingRadians: Double
    public let referenceAlignmentQuaternionWXYZ: GEARSonicWXYZQuaternion

    public init(
        robotInitialBaseQuaternionWXYZ robot: GEARSonicWXYZQuaternion,
        referenceInitialRootQuaternionWXYZ reference: GEARSonicWXYZQuaternion,
        deltaHeadingRadians: Double = 0
    ) throws {
        try GEARSonicReferenceMath.validate(robot, context: "initial robot base")
        try GEARSonicReferenceMath.validate(
            reference, context: "initial reference root")
        guard deltaHeadingRadians.isFinite else {
            throw GEARSonicG1ReferenceError.invalidQuaternion(
                context: "delta-heading")
        }

        let initialHeading = GEARSonicReferenceMath.headingQuaternion(robot)
        let inverseReferenceHeading = GEARSonicReferenceMath.conjugate(
            GEARSonicReferenceMath.headingQuaternion(reference))
        var application = GEARSonicReferenceMath.hamilton(
            initialHeading, inverseReferenceHeading)
        // The released controller skips this multiply when delta is exactly zero.
        if deltaHeadingRadians != 0 {
            application = GEARSonicReferenceMath.hamilton(
                GEARSonicReferenceMath.yawQuaternion(deltaHeadingRadians),
                application)
        }

        robotInitialBaseQuaternionWXYZ = robot
        referenceInitialRootQuaternionWXYZ = reference
        self.deltaHeadingRadians = deltaHeadingRadians
        referenceAlignmentQuaternionWXYZ = application
    }
}

/// The four files needed by the released GEAR-SONIC G1 encoder.
///
/// Joint CSV values are already in the encoder's 29-DoF Isaac Lab policy
/// order; this type intentionally performs no MuJoCo/actuator remapping.
/// Source values remain `Double` through loading and reference construction.
public struct GEARSonicG1ReferenceClip: Sendable {
    public static let jointCount = 29
    public static let bodyCount = 14
    public static let futureFrameCount = 10
    public static let futureFrameStride = 5
    public static let observationDimension = 640

    public let jointPositionsPolicyOrder: [[Double]]
    public let jointVelocitiesPolicyOrder: [[Double]]
    /// One flattened `[14, 3]` row per frame, in body-index then XYZ order.
    public let bodyPositions: [[Double]]
    /// One flattened `[14, 4]` row per frame, in body-index then WXYZ order.
    public let bodyQuaternionsWXYZ: [[Double]]
    /// Optional flattened `[14, 3]` rows in body-index then XYZ order.
    public let bodyLinearVelocities: [[Double]]?
    /// Optional flattened `[14, 3]` rows in body-index then XYZ order.
    public let bodyAngularVelocities: [[Double]]?

    public var frameCount: Int { jointPositionsPolicyOrder.count }

    public init(directory: String) throws {
        try self.init(directory: URL(fileURLWithPath: directory, isDirectory: true))
    }

    public init(directory: URL) throws {
        let q = try Self.readCSV(
            directory.appendingPathComponent("joint_pos.csv"),
            expectedHeader: (0..<Self.jointCount).map { "joint_\($0)" })
        let qd = try Self.readCSV(
            directory.appendingPathComponent("joint_vel.csv"),
            expectedHeader: (0..<Self.jointCount).map { "joint_vel_\($0)" })
        let body = try Self.readCSV(
            directory.appendingPathComponent("body_pos.csv"),
            expectedHeader: (0..<Self.bodyCount).flatMap { body in
                ["body_\(body)_x", "body_\(body)_y", "body_\(body)_z"]
            })
        let quaternion = try Self.readCSV(
            directory.appendingPathComponent("body_quat.csv"),
            expectedHeader: (0..<Self.bodyCount).flatMap { body in
                ["body_\(body)_w", "body_\(body)_x",
                 "body_\(body)_y", "body_\(body)_z"]
            })
        let linearVelocityURL = directory.appendingPathComponent(
            "body_lin_vel.csv")
        let angularVelocityURL = directory.appendingPathComponent(
            "body_ang_vel.csv")
        let hasLinearVelocity = FileManager.default.fileExists(
            atPath: linearVelocityURL.path)
        let hasAngularVelocity = FileManager.default.fileExists(
            atPath: angularVelocityURL.path)
        guard hasLinearVelocity == hasAngularVelocity else {
            throw GEARSonicG1ReferenceError.incompleteBodyVelocityPair
        }
        let linearVelocity = hasLinearVelocity ? try Self.readCSV(
            linearVelocityURL,
            expectedHeader: (0..<Self.bodyCount).flatMap { body in
                ["body_\(body)_vel_x", "body_\(body)_vel_y",
                 "body_\(body)_vel_z"]
            }) : nil
        let angularVelocity = hasAngularVelocity ? try Self.readCSV(
            angularVelocityURL,
            expectedHeader: (0..<Self.bodyCount).flatMap { body in
                ["body_\(body)_angvel_x", "body_\(body)_angvel_y",
                 "body_\(body)_angvel_z"]
            }) : nil
        try self.init(
            jointPositionsPolicyOrder: q,
            jointVelocitiesPolicyOrder: qd,
            bodyPositions: body,
            bodyQuaternionsWXYZ: quaternion,
            bodyLinearVelocities: linearVelocity,
            bodyAngularVelocities: angularVelocity)
    }

    public init(
        jointPositionsPolicyOrder: [[Double]],
        jointVelocitiesPolicyOrder: [[Double]],
        bodyPositions: [[Double]],
        bodyQuaternionsWXYZ: [[Double]],
        bodyLinearVelocities: [[Double]]? = nil,
        bodyAngularVelocities: [[Double]]? = nil
    ) throws {
        let counts = (
            jointPositionsPolicyOrder.count,
            jointVelocitiesPolicyOrder.count,
            bodyPositions.count,
            bodyQuaternionsWXYZ.count)
        guard counts.0 > 0,
              counts.0 == counts.1,
              counts.0 == counts.2,
              counts.0 == counts.3 else {
            throw GEARSonicG1ReferenceError.inconsistentFrameCount(
                jointPositions: counts.0, jointVelocities: counts.1,
                bodyPositions: counts.2, bodyQuaternions: counts.3)
        }
        try Self.validateRows(
            jointPositionsPolicyOrder, width: Self.jointCount,
            name: "joint_pos.csv")
        try Self.validateRows(
            jointVelocitiesPolicyOrder, width: Self.jointCount,
            name: "joint_vel.csv")
        try Self.validateRows(
            bodyPositions, width: Self.bodyCount * 3,
            name: "body_pos.csv")
        try Self.validateRows(
            bodyQuaternionsWXYZ, width: Self.bodyCount * 4,
            name: "body_quat.csv")
        for (frame, row) in bodyQuaternionsWXYZ.enumerated() {
            for body in 0..<Self.bodyCount {
                let offset = body * 4
                try GEARSonicReferenceMath.validate(
                    GEARSonicWXYZQuaternion(
                        w: row[offset], x: row[offset + 1],
                        y: row[offset + 2], z: row[offset + 3]),
                    context: "body \(body), frame \(frame)")
            }
        }
        switch (bodyLinearVelocities, bodyAngularVelocities) {
        case (nil, nil):
            break
        case let (.some(linear), .some(angular)):
            guard linear.count == counts.0, angular.count == counts.0 else {
                throw GEARSonicG1ReferenceError.inconsistentBodyVelocityFrameCount(
                    expected: counts.0, linear: linear.count,
                    angular: angular.count)
            }
            try Self.validateRows(
                linear, width: Self.bodyCount * 3,
                name: "body_lin_vel.csv")
            try Self.validateRows(
                angular, width: Self.bodyCount * 3,
                name: "body_ang_vel.csv")
        default:
            throw GEARSonicG1ReferenceError.incompleteBodyVelocityPair
        }

        self.jointPositionsPolicyOrder = jointPositionsPolicyOrder
        self.jointVelocitiesPolicyOrder = jointVelocitiesPolicyOrder
        self.bodyPositions = bodyPositions
        self.bodyQuaternionsWXYZ = bodyQuaternionsWXYZ
        self.bodyLinearVelocities = bodyLinearVelocities
        self.bodyAngularVelocities = bodyAngularVelocities
    }

    public func rootQuaternionWXYZ(
        at frame: Int
    ) throws -> GEARSonicWXYZQuaternion {
        try validate(frame: frame)
        let row = bodyQuaternionsWXYZ[frame]
        return GEARSonicWXYZQuaternion(
            w: row[0], x: row[1], y: row[2], z: row[3])
    }

    /// Root body linear velocity from the optional official velocity pair.
    public func rootLinearVelocity(
        at frame: Int
    ) throws -> GEARSonicXYZVector? {
        try validate(frame: frame)
        guard let row = bodyLinearVelocities?[frame] else { return nil }
        return GEARSonicXYZVector(x: row[0], y: row[1], z: row[2])
    }

    /// Root body angular velocity from the optional official velocity pair.
    public func rootAngularVelocity(
        at frame: Int
    ) throws -> GEARSonicXYZVector? {
        try validate(frame: frame)
        guard let row = bodyAngularVelocities?[frame] else { return nil }
        return GEARSonicXYZVector(x: row[0], y: row[1], z: row[2])
    }

    /// Convenience for capturing the released controller's reset-time state.
    public func headingAlignment(
        robotInitialBaseQuaternionWXYZ: GEARSonicWXYZQuaternion,
        referenceFrame: Int,
        deltaHeadingRadians: Double = 0
    ) throws -> GEARSonicG1HeadingAlignment {
        try GEARSonicG1HeadingAlignment(
            robotInitialBaseQuaternionWXYZ: robotInitialBaseQuaternionWXYZ,
            referenceInitialRootQuaternionWXYZ:
                rootQuaternionWXYZ(at: referenceFrame),
            deltaHeadingRadians: deltaHeadingRadians)
    }

    /// Build the released G1 encoder's semantic 640-value input.
    ///
    /// Layout is `[q(10x29), qd(10x29), baseToReferenceR6(10x6)]`.
    /// Playing samples `current + [0,5,...,45]`, clamped at the last frame.
    /// Paused replay repeats current `q` and orientation and emits zero `qd`.
    public func referenceObservation640(
        currentFrame: Int,
        isPlaying: Bool,
        robotBaseQuaternionWXYZ robotBase: GEARSonicWXYZQuaternion,
        headingAlignment: GEARSonicG1HeadingAlignment
    ) throws -> ContiguousArray<Double> {
        try validate(frame: currentFrame)
        try GEARSonicReferenceMath.validate(robotBase, context: "current robot base")

        let sampledFrames: [Int]
        if isPlaying {
            sampledFrames = (0..<Self.futureFrameCount).map {
                min(currentFrame + $0 * Self.futureFrameStride, frameCount - 1)
            }
        } else {
            sampledFrames = [Int](
                repeating: currentFrame, count: Self.futureFrameCount)
        }

        var result = ContiguousArray<Double>()
        result.reserveCapacity(Self.observationDimension)
        for frame in sampledFrames {
            result.append(contentsOf: jointPositionsPolicyOrder[frame])
        }
        if isPlaying {
            for frame in sampledFrames {
                result.append(contentsOf: jointVelocitiesPolicyOrder[frame])
            }
        } else {
            result.append(contentsOf: repeatElement(
                0, count: Self.futureFrameCount * Self.jointCount))
        }

        let inverseRobotBase = GEARSonicReferenceMath.conjugate(robotBase)
        for frame in sampledFrames {
            let alignedReference = GEARSonicReferenceMath.hamilton(
                headingAlignment.referenceAlignmentQuaternionWXYZ,
                try rootQuaternionWXYZ(at: frame))
            let robotToReference = GEARSonicReferenceMath.hamilton(
                inverseRobotBase, alignedReference)
            result.append(contentsOf:
                try GEARSonicReferenceMath.rotation6(robotToReference))
        }
        precondition(result.count == Self.observationDimension)
        return result
    }

    private func validate(frame: Int) throws {
        guard frame >= 0, frame < frameCount else {
            throw GEARSonicG1ReferenceError.invalidFrameIndex(
                index: frame, frameCount: frameCount)
        }
    }

    private static func validateRows(
        _ rows: [[Double]], width: Int, name: String
    ) throws {
        for (row, values) in rows.enumerated() {
            guard values.count == width else {
                throw GEARSonicG1ReferenceError.malformedCSV(
                    file: name, line: row + 1,
                    reason: "expected \(width) values, found \(values.count)")
            }
            guard values.allSatisfy(\.isFinite) else {
                throw GEARSonicG1ReferenceError.malformedCSV(
                    file: name, line: row + 1,
                    reason: "all values must be finite doubles")
            }
        }
    }

    /// Accept either the official exact header or a genuinely headerless file.
    /// Once selected, every data row has an exact fixed width and finite values.
    private static func readCSV(
        _ url: URL, expectedHeader: [String]
    ) throws -> [[Double]] {
        let raw: String
        do {
            raw = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw GEARSonicG1ReferenceError.unreadableFile(
                path: url.path, reason: error.localizedDescription)
        }
        let normalized = raw
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        var lines = normalized.split(
            separator: "\n", omittingEmptySubsequences: false).map(String.init)
        if lines.last == "" { lines.removeLast() }
        guard !lines.isEmpty else {
            throw GEARSonicG1ReferenceError.malformedCSV(
                file: url.lastPathComponent, line: 1, reason: "file is empty")
        }

        func fields(_ line: String) -> [String] {
            line.split(separator: ",", omittingEmptySubsequences: false).map {
                $0.trimmingCharacters(in: .whitespaces)
            }
        }
        let first = fields(lines[0])
        let isHeaderless = first.count == expectedHeader.count
            && first.allSatisfy {
                guard let value = Double($0) else { return false }
                return value.isFinite
            }
        let firstDataLine: Int
        if isHeaderless {
            firstDataLine = 0
        } else {
            guard first == expectedHeader else {
                throw GEARSonicG1ReferenceError.malformedCSV(
                    file: url.lastPathComponent, line: 1,
                    reason: "header does not match the released schema")
            }
            firstDataLine = 1
        }
        guard firstDataLine < lines.count else {
            throw GEARSonicG1ReferenceError.malformedCSV(
                file: url.lastPathComponent, line: firstDataLine + 1,
                reason: "file has no data rows")
        }

        var rows = [[Double]]()
        rows.reserveCapacity(lines.count - firstDataLine)
        for lineIndex in firstDataLine..<lines.count {
            let lineNumber = lineIndex + 1
            guard !lines[lineIndex].trimmingCharacters(in: .whitespaces).isEmpty else {
                throw GEARSonicG1ReferenceError.malformedCSV(
                    file: url.lastPathComponent, line: lineNumber,
                    reason: "blank rows are not allowed")
            }
            let columns = fields(lines[lineIndex])
            guard columns.count == expectedHeader.count else {
                throw GEARSonicG1ReferenceError.malformedCSV(
                    file: url.lastPathComponent, line: lineNumber,
                    reason: "expected \(expectedHeader.count) values, "
                        + "found \(columns.count)")
            }
            var row = [Double]()
            row.reserveCapacity(columns.count)
            for (column, text) in columns.enumerated() {
                guard let value = Double(text), value.isFinite else {
                    throw GEARSonicG1ReferenceError.malformedCSV(
                        file: url.lastPathComponent, line: lineNumber,
                        reason: "column \(column + 1) is not a finite double")
                }
                row.append(value)
            }
            rows.append(row)
        }
        return rows
    }
}

private enum GEARSonicReferenceMath {
    static func validate(
        _ q: GEARSonicWXYZQuaternion, context: String
    ) throws {
        let squaredNorm = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
        guard q.w.isFinite, q.x.isFinite, q.y.isFinite, q.z.isFinite,
              squaredNorm.isFinite, squaredNorm > 1e-24 else {
            throw GEARSonicG1ReferenceError.invalidQuaternion(context: context)
        }
    }

    static func conjugate(
        _ q: GEARSonicWXYZQuaternion
    ) -> GEARSonicWXYZQuaternion {
        GEARSonicWXYZQuaternion(w: q.w, x: -q.x, y: -q.y, z: -q.z)
    }

    /// Standard Hamilton product, with no implicit normalization.
    static func hamilton(
        _ a: GEARSonicWXYZQuaternion,
        _ b: GEARSonicWXYZQuaternion
    ) -> GEARSonicWXYZQuaternion {
        GEARSonicWXYZQuaternion(
            w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w)
    }

    static func yawQuaternion(_ angle: Double) -> GEARSonicWXYZQuaternion {
        let half = angle * 0.5
        return GEARSonicWXYZQuaternion(
            w: cos(half), x: 0, y: 0, z: sin(half))
    }

    /// Match `calc_heading_quat_d`: rotate world +X, then atan2(Y, X).
    static func headingQuaternion(
        _ q: GEARSonicWXYZQuaternion
    ) -> GEARSonicWXYZQuaternion {
        let rotatedX = 2 * (q.w * q.w + q.x * q.x) - 1
        let rotatedY = 2 * (q.x * q.y + q.w * q.z)
        return yawQuaternion(atan2(rotatedY, rotatedX))
    }

    /// Row-major flattening of the first two rotation-matrix columns.
    /// This is the only point at which the released implementation normalizes.
    static func rotation6(
        _ q: GEARSonicWXYZQuaternion
    ) throws -> [Double] {
        try validate(q, context: "relative rotation")
        let inverseNorm = 1 / sqrt(
            q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
        let w = q.w * inverseNorm
        let x = q.x * inverseNorm
        let y = q.y * inverseNorm
        let z = q.z * inverseNorm
        return [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
        ]
    }
}
