// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// This file is an altered Swift reference implementation inspired by Newton's
// support-map, MPR, GJK, and multi-contact code at commit
// 37b75a212112cebc35bdbfb521357b8a2900d6be, plus centered-box MPR support from
// commit 8e2f385a. The MPR, GJK simplex, and manifold algorithms in Newton are
// derived from Jitter Physics 2 (MIT), and MPR is in turn based on XenoCollide
// (zlib). See THIRD_PARTY_NOTICES.md.

import Foundation
import SimCore
import simd

/// A translation-stable interior point for a finite convex support cloud.
///
/// Summing source-frame CAD coordinates directly in `Float` can lose the
/// entire local offset (for example, a one-metre hull around 1e6 metres).  By
/// accumulating differences from one represented vertex, the average remains
/// a positive convex combination while all arithmetic stays at hull scale.
@inline(__always)
func stableConvexSupportCenter(_ vertices: [F3]) -> F3 {
    precondition(!vertices.isEmpty)
    let base = vertices[0]
    var localSum = F3.zero
    for vertex in vertices.dropFirst() {
        localSum += vertex - base
    }
    return base + localSum / Float(vertices.count)
}

/// Deterministic CPU reference narrow phase for bounded convex shapes.
///
/// The implementation intentionally has no dependency on `PhysicsScene`, the CPU
/// solver, or the GPU buffer layout. It is the correctness oracle for a later Metal
/// implementation. Queries run in shape A's local frame and return world-space
/// witnesses whose normal points from A to B. Signed distance is negative for
/// penetration and positive for separation.
public enum ConvexNarrowPhase {
    public enum Operand: String, Equatable, Sendable {
        case a
        case b
    }

    public enum ShapeKind: String, Equatable, Sendable {
        case box
        case sphere
        case capsule
        case convexHull
        case torus
    }

    public enum Algorithm: String, Equatable, Sendable {
        case mpr
        case gjk
    }

    public enum Stage: String, Equatable, Sendable {
        case validation
        case mpr
        case gjk
        case manifold
    }

    public enum Parameter: String, Equatable, Sendable {
        case halfExtents
        case radius
        case halfHeight
        case majorRadius
        case minorRadius
        case position
        case orientation
        case margin
        case contactThreshold
    }

    public enum Failure: Error, Equatable, Sendable {
        case unsupportedShape(operand: Operand, kind: ShapeKind)
        case nonFiniteParameter(operand: Operand?, parameter: Parameter)
        case nonPositiveParameter(operand: Operand, parameter: Parameter)
        case negativeParameter(operand: Operand, parameter: Parameter)
        case invalidOrientation(operand: Operand)
        case invalidHullVertexCount(operand: Operand, actual: Int, allowed: ClosedRange<Int>)
        case nonFiniteHullVertex(operand: Operand, index: Int)
        case degenerateHull(operand: Operand)
        case invalidIterationLimit(Int)
        case nonFiniteComputation(stage: Stage)
        case didNotConverge(stage: Stage)
    }

    public struct Pose: Sendable {
        public var position: F3
        public var orientation: Quat

        public init(position: F3 = .zero,
                    orientation: Quat = Quat(real: 1, imag: .zero)) {
            self.position = position
            self.orientation = orientation
        }

        public static var identity: Pose { Pose() }
    }

    /// Shape dimensions are local-space dimensions. Capsules use the local Z axis.
    /// Convex hull vertices must already be cooked and are bounded to 4...64 vertices.
    public enum Shape: Sendable {
        case box(halfExtents: F3)
        case sphere(radius: Float)
        case capsule(radius: Float, halfHeight: Float)
        case convexHull(vertices: [F3])

        /// Present only so unsupported torus use fails explicitly rather than silently
        /// falling through to an origin support point.
        case torus(majorRadius: Float, minorRadius: Float)

        public var kind: ShapeKind {
            switch self {
            case .box: .box
            case .sphere: .sphere
            case .capsule: .capsule
            case .convexHull: .convexHull
            case .torus: .torus
            }
        }
    }

    public struct Options: Equatable, Sendable {
        /// These margins choose Newton's anti-flicker MPR inflation. They are not
        /// subtracted from the returned true-surface signed distance.
        public var marginA: Float
        public var marginB: Float

        /// Multi-contact sampling is skipped for separation beyond this distance.
        /// A single exact GJK witness is still returned for farther queries.
        public var contactThreshold: Float
        public var maximumIterations: Int

        public init(marginA: Float = 0,
                    marginB: Float = 0,
                    contactThreshold: Float = 0.02,
                    maximumIterations: Int = 30) {
            self.marginA = marginA
            self.marginB = marginB
            self.contactThreshold = contactThreshold
            self.maximumIterations = maximumIterations
        }
    }

    public struct Contact: Equatable, Sendable {
        public var pointA: F3
        public var pointB: F3
        public var normalAtoB: F3
        public var signedDistance: Float
        public var sortKey: UInt8

        public var center: F3 { (pointA + pointB) * 0.5 }
    }

    public struct Manifold: Equatable, Sendable {
        public var algorithm: Algorithm
        public var signedDistance: Float
        public var normalAtoB: F3
        public var contacts: [Contact]

        public var intersects: Bool { signedDistance <= 0 }
    }

    /// Query two convex shapes. Every successful value is finite and contains one
    /// through five deterministic contacts. Invalid or unsupported inputs return a
    /// typed failure before arithmetic can produce NaNs.
    public static func query(
        shapeA: Shape,
        poseA: Pose = .identity,
        shapeB: Shape,
        poseB: Pose = .identity,
        options: Options = Options()
    ) -> Result<Manifold, Failure> {
        do {
            let a = try validate(shapeA, operand: .a)
            let b = try validate(shapeB, operand: .b)
            let transforms = try relativeTransform(
                poseA: poseA, sourceCenterA: a.sourceCenter,
                poseB: poseB, sourceCenterB: b.sourceCenter)
            try validate(options)

            let context = SupportContext(a: a, b: b,
                                         orientationB: transforms.relativeOrientationB,
                                         positionB: transforms.relativePositionB)
            let marginSum = options.marginA + options.marginB
            let enlarge: Float
            if marginSum <= 0 {
                enlarge = antiFlickerEpsilon
            } else if marginSum < antiFlickerEpsilon {
                enlarge = 2 * antiFlickerEpsilon
            } else {
                enlarge = 0
            }

            let adaptiveDilationCap = min(
                a.characteristicScale + b.characteristicScale,
                2 * (options.contactThreshold + 0.25) + antiFlickerEpsilon)
            let standardMPRQuery: LocalQuery?
            if case .overlap(let overlap) = solveMPR(
                context: context, extend: enlarge,
                maximumIterations: options.maximumIterations
            ) {
                let corrected = correctedMPRQuery(overlap, enlarge: enlarge)
                standardMPRQuery = isConsistentCorrectedMPR(corrected)
                    ? corrected : nil
            } else {
                standardMPRQuery = nil
            }

            let localQuery: LocalQuery
            if let standardMPRQuery {
                localQuery = standardMPRQuery
            } else {
                switch solveGJK(context: context,
                                maximumIterations: options.maximumIterations) {
                case .separated(let separated):
                    localQuery = LocalQuery(
                        algorithm: .gjk,
                        pointA: separated.pointA,
                        pointB: separated.pointB,
                        normal: separated.normal,
                        signedDistance: separated.distance)
                case .overlap:
                    // A bounded retry is reserved for the rare seam where
                    // standard MPR and exact GJK are both inconclusive. It
                    // never accepts guessed witnesses: try the standard portal
                    // from the opposite operand first, then one fixed larger
                    // dilation, and require the corrected true-surface witness
                    // to satisfy the same finite/unit/projection contract as
                    // the Metal implementation.
                    if let retry = boundedMPRFallback(
                        context: context,
                        maximumIterations: options.maximumIterations,
                        failedGJKUpperBound: 0,
                        maximumDilation: adaptiveDilationCap) {
                        localQuery = retry
                    } else {
                        throw Failure.didNotConverge(stage: .mpr)
                    }
                case .didNotConverge(let upperBound):
                    if let retry = boundedMPRFallback(
                        context: context,
                        maximumIterations: options.maximumIterations,
                        failedGJKUpperBound: upperBound,
                        maximumDilation: adaptiveDilationCap) {
                        localQuery = retry
                    } else {
                        throw Failure.didNotConverge(stage: .gjk)
                    }
                }
            }

            guard localQuery.isFinite else {
                throw Failure.nonFiniteComputation(stage: localQuery.algorithm == .mpr ? .mpr : .gjk)
            }
            let normalLengthSquared = length_squared(localQuery.normal)
            guard normalLengthSquared > minimumVectorLengthSquared else {
                throw Failure.didNotConverge(stage: localQuery.algorithm == .mpr ? .mpr : .gjk)
            }

            let normalizedQuery = localQuery.withNormalizedNormal()
            let localContacts = buildManifold(
                context: context,
                query: normalizedQuery,
                threshold: options.contactThreshold)
            guard !localContacts.isEmpty,
                  localContacts.count <= maximumContacts,
                  localContacts.allSatisfy(\.isFinite) else {
                throw Failure.nonFiniteComputation(stage: .manifold)
            }

            let worldNormal = transforms.orientationA.act(normalizedQuery.normal)
            var contacts = localContacts.map { local in
                Contact(
                    pointA: transforms.orientationA.act(local.pointA) + transforms.positionA,
                    pointB: transforms.orientationA.act(local.pointB) + transforms.positionA,
                    normalAtoB: worldNormal,
                    signedDistance: local.signedDistance,
                    sortKey: 0)
            }
            contacts.sort(by: deterministicContactOrder)
            for index in contacts.indices {
                contacts[index].sortKey = UInt8(index)
            }

            guard isFinite(worldNormal), contacts.allSatisfy({
                isFinite($0.pointA) && isFinite($0.pointB)
                    && isFinite($0.normalAtoB) && $0.signedDistance.isFinite
            }) else {
                throw Failure.nonFiniteComputation(stage: .manifold)
            }

            return .success(Manifold(
                algorithm: normalizedQuery.algorithm,
                signedDistance: normalizedQuery.signedDistance,
                normalAtoB: worldNormal,
                contacts: contacts))
        } catch let failure as Failure {
            return .failure(failure)
        } catch {
            return .failure(.nonFiniteComputation(stage: .validation))
        }
    }
}

// MARK: - Validated geometry

private extension ConvexNarrowPhase {
    static let antiFlickerEpsilon: Float = 1.0e-4
    static let mprEpsilon: Float = 1.0e-5
    static let gjkEpsilon: Float = 1.0e-4
    static let minimumVectorLengthSquared: Float = 1.0e-16
    static let mprNumericEpsilon: Float = 1.0e-16
    static let gjkSimplexEpsilon: Float = 1.0e-8
    static let maximumHullVertices = 64
    static let maximumContacts = 5
    static let boxSupportDeadband: Float = 1.0e-10
    static let centeredBoxTieEpsilon: Float = 1.0e-6

    enum ValidatedShape {
        case box(F3)
        case sphere(Float)
        case capsule(radius: Float, halfHeight: Float)
        case hull(vertices: [F3], sourceCenter: F3, scale: Float)

        var center: F3 {
            return .zero
        }

        var sourceCenter: F3 {
            if case .hull(_, let sourceCenter, _) = self {
                return sourceCenter
            }
            return .zero
        }

        var characteristicScale: Float {
            switch self {
            case .box(let half): return max(half.x, max(half.y, half.z)) * 2
            case .sphere(let radius): return radius * 2
            case .capsule(let radius, let halfHeight): return 2 * (radius + halfHeight)
            case .hull(_, _, let scale): return scale
            }
        }

        var isPolytope: Bool {
            switch self {
            case .box, .hull: true
            case .sphere, .capsule: false
            }
        }

        func support(_ direction: F3, centeredBoxTies: Bool = false) -> F3 {
            switch self {
            case .box(let half):
                let directionScale = max(abs(direction.x), max(abs(direction.y), abs(direction.z)))
                let threshold = ConvexNarrowPhase.boxSupportDeadband * directionScale
                var result = F3(
                    direction.x >= -threshold ? half.x : -half.x,
                    direction.y >= -threshold ? half.y : -half.y,
                    direction.z >= -threshold ? half.z : -half.z)
                if centeredBoxTies {
                    let contribution = abs(direction) * half
                    let tie = ConvexNarrowPhase.centeredBoxTieEpsilon
                        * (contribution.x + contribution.y + contribution.z)
                    if contribution.x <= tie { result.x = 0 }
                    if contribution.y <= tie { result.y = 0 }
                    if contribution.z <= tie { result.z = 0 }
                }
                return result

            case .sphere(let radius):
                return safeNormalize(direction, fallback: F3(1, 0, 0)) * radius

            case .capsule(let radius, let halfHeight):
                var result = safeNormalize(direction, fallback: F3(1, 0, 0)) * radius
                result.z += direction.z >= 0 ? halfHeight : -halfHeight
                return result

            case .hull(let vertices, _, _):
                var best = vertices[0]
                var bestDot = dot(best, direction)
                for vertex in vertices.dropFirst() {
                    let candidate = dot(vertex, direction)
                    if candidate > bestDot {
                        best = vertex
                        bestDot = candidate
                    }
                }
                return best
            }
        }
    }

    struct RelativeTransform {
        var positionA: F3
        var orientationA: Quat
        var relativePositionB: F3
        var relativeOrientationB: Quat
    }

    static func validate(_ shape: Shape, operand: Operand) throws -> ValidatedShape {
        switch shape {
        case .torus:
            throw Failure.unsupportedShape(operand: operand, kind: .torus)

        case .box(let half):
            guard isFinite(half) else {
                throw Failure.nonFiniteParameter(operand: operand, parameter: .halfExtents)
            }
            guard half.x > 0, half.y > 0, half.z > 0 else {
                throw Failure.nonPositiveParameter(operand: operand, parameter: .halfExtents)
            }
            return .box(half)

        case .sphere(let radius):
            guard radius.isFinite else {
                throw Failure.nonFiniteParameter(operand: operand, parameter: .radius)
            }
            guard radius > 0 else {
                throw Failure.nonPositiveParameter(operand: operand, parameter: .radius)
            }
            return .sphere(radius)

        case .capsule(let radius, let halfHeight):
            guard radius.isFinite else {
                throw Failure.nonFiniteParameter(operand: operand, parameter: .radius)
            }
            guard halfHeight.isFinite else {
                throw Failure.nonFiniteParameter(operand: operand, parameter: .halfHeight)
            }
            guard radius > 0 else {
                throw Failure.nonPositiveParameter(operand: operand, parameter: .radius)
            }
            guard halfHeight >= 0 else {
                throw Failure.negativeParameter(operand: operand, parameter: .halfHeight)
            }
            return .capsule(radius: radius, halfHeight: halfHeight)

        case .convexHull(let vertices):
            let allowed = 4...maximumHullVertices
            guard allowed.contains(vertices.count) else {
                throw Failure.invalidHullVertexCount(
                    operand: operand, actual: vertices.count, allowed: allowed)
            }
            for (index, vertex) in vertices.enumerated() where !isFinite(vertex) {
                throw Failure.nonFiniteHullVertex(operand: operand, index: index)
            }
            guard let bounds = bounds(of: vertices) else {
                throw Failure.degenerateHull(operand: operand)
            }
            let scale = length(bounds.upper - bounds.lower)
            guard scale.isFinite, scale > 0, hasThreeDimensionalRank(vertices, scale: scale) else {
                throw Failure.degenerateHull(operand: operand)
            }
            // The MPR portal origin must be inside the shape. An AABB
            // midpoint can lie outside an asymmetric convex hull; the
            // arithmetic mean is a convex combination of the vertices and is
            // therefore a guaranteed hull point for this public vertex API.
            let supportCenter = stableConvexSupportCenter(vertices)
            let centeredVertices = vertices.map { $0 - supportCenter }
            return .hull(
                vertices: centeredVertices,
                sourceCenter: supportCenter,
                scale: scale)
        }
    }

    static func validate(_ options: Options) throws {
        guard options.marginA.isFinite else {
            throw Failure.nonFiniteParameter(operand: nil, parameter: .margin)
        }
        guard options.marginB.isFinite else {
            throw Failure.nonFiniteParameter(operand: nil, parameter: .margin)
        }
        guard options.marginA >= 0, options.marginB >= 0 else {
            throw Failure.negativeParameter(operand: options.marginA < 0 ? .a : .b,
                                            parameter: .margin)
        }
        guard options.contactThreshold.isFinite else {
            throw Failure.nonFiniteParameter(operand: nil, parameter: .contactThreshold)
        }
        guard options.contactThreshold >= 0 else {
            throw Failure.negativeParameter(operand: .a, parameter: .contactThreshold)
        }
        guard (1...128).contains(options.maximumIterations) else {
            throw Failure.invalidIterationLimit(options.maximumIterations)
        }
    }

    static func relativeTransform(
        poseA: Pose, sourceCenterA: F3,
        poseB: Pose, sourceCenterB: F3
    ) throws -> RelativeTransform {
        guard isFinite(poseA.position) else {
            throw Failure.nonFiniteParameter(operand: .a, parameter: .position)
        }
        guard isFinite(poseB.position) else {
            throw Failure.nonFiniteParameter(operand: .b, parameter: .position)
        }
        let orientationA = try normalized(poseA.orientation, operand: .a)
        let orientationB = try normalized(poseB.orientation, operand: .b)
        let centeredPositionA = poseA.position
            + orientationA.act(sourceCenterA)
        let centeredPositionB = poseB.position
            + orientationB.act(sourceCenterB)
        guard isFinite(centeredPositionA), isFinite(centeredPositionB) else {
            throw Failure.nonFiniteComputation(stage: .validation)
        }
        let inverseA = orientationA.inverse
        return RelativeTransform(
            positionA: centeredPositionA,
            orientationA: orientationA,
            relativePositionB: inverseA.act(
                centeredPositionB - centeredPositionA),
            relativeOrientationB: (inverseA * orientationB).normalized)
    }

    static func normalized(_ orientation: Quat, operand: Operand) throws -> Quat {
        guard isFinite(orientation.vector) else {
            throw Failure.nonFiniteParameter(operand: operand, parameter: .orientation)
        }
        let normSquared = simd_length_squared(orientation.vector)
        guard normSquared.isFinite, normSquared > 1.0e-12 else {
            throw Failure.invalidOrientation(operand: operand)
        }
        return orientation.normalized
    }

    static func hasThreeDimensionalRank(_ vertices: [F3], scale: Float) -> Bool {
        let tolerance = max(scale * 1.0e-6, 1.0e-7)
        let toleranceSquared = tolerance * tolerance
        let p0 = vertices[0]

        guard let p1 = vertices.max(by: {
            length_squared($0 - p0) < length_squared($1 - p0)
        }), length_squared(p1 - p0) > toleranceSquared else { return false }

        let line = p1 - p0
        let lineSquared = length_squared(line)
        guard let p2 = vertices.max(by: {
            length_squared(cross(line, $0 - p0)) < length_squared(cross(line, $1 - p0))
        }), length_squared(cross(line, p2 - p0)) > toleranceSquared * lineSquared else {
            return false
        }

        let planeNormal = cross(line, p2 - p0)
        let planeLength = length(planeNormal)
        guard planeLength > 0 else { return false }
        let maximumPlaneDistance = vertices.reduce(Float.zero) {
            max($0, abs(dot($1 - p0, planeNormal)) / planeLength)
        }
        return maximumPlaneDistance > tolerance
    }

    static func bounds(of points: [F3]) -> (lower: F3, upper: F3)? {
        guard var lower = points.first else { return nil }
        var upper = lower
        for point in points.dropFirst() {
            lower = simd_min(lower, point)
            upper = simd_max(upper, point)
        }
        return (lower, upper)
    }
}

// MARK: - Support mapping and MPR

private extension ConvexNarrowPhase {
    struct SupportVertex {
        var pointB: F3
        var delta: F3 // pointA - pointB in A-local space

        var pointA: F3 { pointB + delta }
        static var zero: SupportVertex { SupportVertex(pointB: .zero, delta: .zero) }
    }

    struct SupportContext {
        var a: ValidatedShape
        var b: ValidatedShape
        var orientationB: Quat
        var positionB: F3

        var centerVertex: SupportVertex {
            let centerB = orientationB.act(b.center) + positionB
            return SupportVertex(pointB: centerB, delta: a.center - centerB)
        }

        func supportB(_ directionInA: F3, centeredBoxTies: Bool = false) -> F3 {
            let directionInB = orientationB.inverse.act(directionInA)
            return orientationB.act(b.support(directionInB,
                                                centeredBoxTies: centeredBoxTies)) + positionB
        }

        func minkowskiSupport(_ direction: F3,
                              extend: Float,
                              centeredBoxTies: Bool) -> SupportVertex {
            var pointA = a.support(direction, centeredBoxTies: centeredBoxTies)
            var pointB = supportB(-direction, centeredBoxTies: centeredBoxTies)
            if extend != 0 {
                let offset = safeNormalize(direction, fallback: F3(1, 0, 0)) * (0.5 * extend)
                pointA += offset
                pointB -= offset
            }
            return SupportVertex(pointB: pointB, delta: pointA - pointB)
        }
    }

    struct MPROverlap {
        var pointA: F3
        var pointB: F3
        var normal: F3
        var penetration: Float
        var signedDistance: Float = 0
    }

    enum MPRResult {
        case overlap(MPROverlap)
        case separated
        case inconclusive
    }

    static func solveMPR(context: SupportContext,
                         extend: Float,
                         maximumIterations: Int) -> MPRResult {
        var v0 = context.centerVertex
        if length_squared(v0.delta) < mprNumericEpsilon {
            var bestDot = -Float.greatestFiniteMagnitude
            var bestDirection = F3(1, 0, 0)
            for axis in [F3(1, 0, 0), F3(0, 1, 0), F3(0, 0, 1)] {
                let probe = context.minkowskiSupport(
                    axis, extend: extend, centeredBoxTies: true)
                let candidate = dot(probe.delta, axis)
                if candidate > bestDot {
                    bestDot = candidate
                    bestDirection = axis
                }
            }
            v0.delta = bestDirection * 1.0e-5
        }

        var normal = -v0.delta
        var v1 = context.minkowskiSupport(
            normal, extend: extend, centeredBoxTies: true)
        if dot(v1.delta, normal) <= 0 { return .separated }

        normal = cross(v1.delta, v0.delta)
        if length_squared(normal) < mprNumericEpsilon * mprNumericEpsilon {
            normal = safeNormalize(v1.delta - v0.delta, fallback: F3(1, 0, 0))
            let penetration = dot(v1.delta, normal)
            guard penetration.isFinite else { return .inconclusive }
            return .overlap(MPROverlap(
                pointA: v1.pointA, pointB: v1.pointB,
                normal: normal, penetration: penetration))
        }

        var v2 = context.minkowskiSupport(
            normal, extend: extend, centeredBoxTies: true)
        if dot(v2.delta, normal) <= 0 { return .separated }

        var temp1 = v1.delta - v0.delta
        var temp2 = v2.delta - v0.delta
        normal = cross(temp1, temp2)
        if dot(normal, v0.delta) > 0 {
            swap(&v1, &v2)
            normal = -normal
        }

        var v3 = SupportVertex.zero
        var foundPortal = false
        for _ in 0...maximumIterations {
            v3 = context.minkowskiSupport(
                normal, extend: extend, centeredBoxTies: true)
            if dot(v3.delta, normal) <= 0 { return .separated }

            temp1 = cross(v1.delta, v3.delta)
            if dot(temp1, v0.delta) < 0 {
                v2 = v3
                normal = cross(v1.delta - v0.delta, v3.delta - v0.delta)
                continue
            }

            temp1 = cross(v3.delta, v2.delta)
            if dot(temp1, v0.delta) < 0 {
                v1 = v3
                normal = cross(v3.delta - v0.delta, v2.delta - v0.delta)
                continue
            }
            foundPortal = true
            break
        }
        guard foundPortal else { return .inconclusive }

        var hit = false
        for phase in 1...(maximumIterations + 1) {
            temp1 = v2.delta - v1.delta
            temp2 = v3.delta - v1.delta
            normal = cross(temp1, temp2)
            let normalSquared = length_squared(normal)
            guard normalSquared >= mprNumericEpsilon * mprNumericEpsilon else {
                return .inconclusive
            }

            if !hit {
                hit = dot(normal, v1.delta) >= 0
            }

            let v4 = context.minkowskiSupport(
                normal, extend: extend, centeredBoxTies: true)
            let delta = dot(v4.delta - v3.delta, normal)
            var penetration = dot(v4.delta, normal)

            if delta * delta <= mprEpsilon * mprEpsilon * normalSquared
                || penetration <= 0 || phase > maximumIterations {
                guard hit else { return .separated }
                let inverseNormal = 1 / normalSquared.squareRoot()
                penetration *= inverseNormal
                normal *= inverseNormal

                var temp3 = cross(v1.delta, temp1)
                let gamma = dot(temp3, normal) * inverseNormal
                temp3 = cross(temp2, v1.delta)
                let beta = dot(temp3, normal) * inverseNormal
                let alpha = 1 - gamma - beta
                let pointA = alpha * v1.pointA + beta * v2.pointA + gamma * v3.pointA
                let pointB = alpha * v1.pointB + beta * v2.pointB + gamma * v3.pointB
                guard isFinite(pointA), isFinite(pointB), isFinite(normal), penetration.isFinite else {
                    return .inconclusive
                }
                return .overlap(MPROverlap(
                    pointA: pointA, pointB: pointB,
                    normal: normal, penetration: penetration))
            }

            temp1 = cross(v4.delta, v0.delta)
            if dot(temp1, v1.delta) >= 0 {
                if dot(temp1, v2.delta) >= 0 {
                    v1 = v4
                } else {
                    v3 = v4
                }
            } else if dot(temp1, v3.delta) >= 0 {
                v2 = v4
            } else {
                v1 = v4
            }
        }
        return .inconclusive
    }

    /// Remove symmetric portal dilation and require the reconstructed witness
    /// to agree with its signed distance. Five MPR convergence epsilons covers
    /// accumulated Float error in the barycentric reconstruction; it is a
    /// query-wide numerical contract rather than a pose-specific tolerance.
    static func correctedMPRQuery(
        _ overlap: MPROverlap, enlarge: Float
    ) -> LocalQuery {
        let correction = overlap.normal * (0.5 * enlarge)
        return LocalQuery(
            algorithm: .mpr,
            pointA: overlap.pointA - correction,
            pointB: overlap.pointB + correction,
            normal: overlap.normal,
            signedDistance: -overlap.penetration + enlarge)
    }

    static func isConsistentCorrectedMPR(_ query: LocalQuery) -> Bool {
        guard query.isFinite else { return false }
        let tolerance: Float = 5 * mprEpsilon
        let normalSquared = length_squared(query.normal)
        guard normalSquared.isFinite,
              abs(normalSquared - 1) <= tolerance else { return false }
        let delta = query.pointB - query.pointA
        let projectedResidual = abs(dot(delta, query.normal)
                                    - query.signedDistance)
        let tangentResidual = delta - query.normal * query.signedDistance
        return projectedResidual.isFinite && isFinite(tangentResidual)
            && projectedResidual <= tolerance
            && length(tangentResidual) <= tolerance
    }

    static func boundedMPRFallback(
        context: SupportContext, maximumIterations: Int,
        failedGJKUpperBound: Float?, maximumDilation: Float
    ) -> LocalQuery? {
        let inverseRelative = context.orientationB.inverse
        let swappedContext = SupportContext(
            a: context.b,
            b: context.a,
            orientationB: inverseRelative,
            positionB: inverseRelative.act(-context.positionB))

        if case .overlap(let swapped) = solveMPR(
            context: swappedContext,
            extend: antiFlickerEpsilon,
            maximumIterations: maximumIterations
        ) {
            // The swapped result is in original B-local space. Map its raw
            // inflated witnesses and B->A normal back to original A-local
            // space before correcting the dilation.
            let mapped = MPROverlap(
                pointA: context.orientationB.act(swapped.pointB)
                    + context.positionB,
                pointB: context.orientationB.act(swapped.pointA)
                    + context.positionB,
                normal: -context.orientationB.act(swapped.normal),
                penetration: swapped.penetration)
            let query = correctedMPRQuery(
                mapped, enlarge: antiFlickerEpsilon)
            if isConsistentCorrectedMPR(query) { return query }
        }

        let upper = failedGJKUpperBound ?? 0
        guard upper.isFinite, upper >= 0, maximumDilation.isFinite,
              maximumDilation > 0 else { return nil }
        // Inflating by twice the failed simplex upper bound puts the origin
        // at least one upper-bound distance inside the dilated CSO, avoiding
        // another near-boundary portal walk. The public query threshold and
        // shape scale bound this rare retry; correction and residual checks
        // still certify only the original, undilated surfaces.
        let retryEnlarge = min(
            maximumDilation,
            max(1.0e-3, 2 * upper + antiFlickerEpsilon))
        guard retryEnlarge > upper + 0.5 * antiFlickerEpsilon else {
            return nil
        }
        if case .overlap(let retry) = solveMPR(
            context: context,
            extend: retryEnlarge,
            maximumIterations: maximumIterations
        ) {
            let query = correctedMPRQuery(retry, enlarge: retryEnlarge)
            if isConsistentCorrectedMPR(query) { return query }
        }
        return nil
    }
}

// MARK: - GJK distance

private extension ConvexNarrowPhase {
    struct SimplexClosest {
        var point: F3
        var barycentric: [Float]
        var mask: UInt8
    }

    struct GJKSeparation {
        var pointA: F3
        var pointB: F3
        var normal: F3
        var distance: Float
    }

    enum GJKResult {
        case separated(GJKSeparation)
        case overlap
        case didNotConverge(upperBound: Float?)
    }

    static func solveGJK(context: SupportContext,
                         maximumIterations: Int) -> GJKResult {
        var simplex = [SupportVertex](repeating: .zero, count: 4)
        var barycentric = [Float](repeating: 0, count: 4)
        var usageMask: UInt8 = 0
        var v = context.centerVertex.delta
        var distanceSquared = length_squared(v)
        var lastSearchDirection = F3(1, 0, 0)
        var converged = false

        for _ in 0..<maximumIterations {
            if distanceSquared < gjkEpsilon * gjkEpsilon {
                return .overlap
            }

            let searchDirection = -v
            lastSearchDirection = searchDirection
            let support = context.minkowskiSupport(
                searchDirection, extend: 0, centeredBoxTies: false)
            let supportDelta = support.delta
            let dualityGap = dot(v, v - supportDelta)
            if dualityGap <= 0
                || dualityGap * dualityGap < gjkEpsilon * gjkEpsilon * distanceSquared {
                converged = true
                break
            }

            var duplicate = false
            for index in 0..<4 where usageMask & (1 << UInt8(index)) != 0 {
                if length_squared(simplex[index].delta - supportDelta) < gjkEpsilon * gjkEpsilon {
                    duplicate = true
                    break
                }
            }
            if duplicate {
                // In an unbounded distance query, a repeated retained support
                // point proves numerical stalling but not separation. Unlike
                // the bounded GPU contact query there is no positive distance
                // threshold against which both witness and support-plane
                // bounds can certify the result, so fail closed and let the
                // query's bounded MPR retry produce a checked witness.
                let upper = distanceSquared.isFinite && distanceSquared >= 0
                    ? distanceSquared.squareRoot() : nil
                return .didNotConverge(upperBound: upper)
            }

            var indices: [Int] = []
            var freeSlot = 0
            for index in 0..<4 {
                if usageMask & (1 << UInt8(index)) != 0 {
                    indices.append(index)
                } else {
                    freeSlot = index
                }
            }
            indices.append(freeSlot)
            simplex[freeSlot] = support

            let closest: SimplexClosest
            switch indices.count {
            case 1:
                var bc = [Float](repeating: 0, count: 4)
                bc[indices[0]] = 1
                closest = SimplexClosest(
                    point: simplex[indices[0]].delta,
                    barycentric: bc,
                    mask: 1 << UInt8(indices[0]))
            case 2:
                closest = closestSegment(simplex, indices[0], indices[1])
            case 3:
                closest = closestTriangle(simplex, indices[0], indices[1], indices[2])
            case 4:
                closest = closestTetrahedron(simplex)
                if closest.mask == 0b1111 { return .overlap }
            default:
                return .didNotConverge(upperBound: nil)
            }

            barycentric = closest.barycentric
            usageMask = closest.mask
            v = closest.point
            distanceSquared = length_squared(v)
            guard isFinite(v), distanceSquared.isFinite else {
                return .didNotConverge(upperBound: nil)
            }
        }

        guard converged else {
            let upper = distanceSquared.isFinite && distanceSquared >= 0
                ? distanceSquared.squareRoot() : nil
            return .didNotConverge(upperBound: upper)
        }
        let witnesses = simplexWitnesses(simplex, barycentric, usageMask)
        let delta = witnesses.pointB - witnesses.pointA
        let deltaSquared = length_squared(delta)
        let distance: Float
        let normal: F3
        if deltaSquared > minimumVectorLengthSquared {
            distance = deltaSquared.squareRoot()
            normal = delta / distance
        } else {
            distance = max(distanceSquared, 0).squareRoot()
            if distance > gjkEpsilon {
                normal = -v / distance
            } else {
                normal = safeNormalize(lastSearchDirection, fallback: F3(1, 0, 0))
            }
        }
        guard isFinite(witnesses.pointA), isFinite(witnesses.pointB),
              isFinite(normal), distance.isFinite else {
            return .didNotConverge(upperBound: nil)
        }
        return .separated(GJKSeparation(
            pointA: witnesses.pointA,
            pointB: witnesses.pointB,
            normal: normal,
            distance: distance))
    }

    static func closestSegment(_ vertices: [SupportVertex], _ i0: Int, _ i1: Int) -> SimplexClosest {
        let a = vertices[i0].delta
        let b = vertices[i1].delta
        let edge = b - a
        let edgeSquared = length_squared(edge)
        let degenerate = edgeSquared < gjkSimplexEpsilon
        let t = -dot(a, edge) / (degenerate ? gjkSimplexEpsilon : edgeSquared)
        var lambda0 = 1 - t
        var lambda1 = t
        var mask: UInt8 = (1 << UInt8(i0)) | (1 << UInt8(i1))
        if lambda0 < 0 || degenerate {
            lambda0 = 0
            lambda1 = 1
            mask = 1 << UInt8(i1)
        } else if lambda1 < 0 {
            lambda0 = 1
            lambda1 = 0
            mask = 1 << UInt8(i0)
        }
        var bc = [Float](repeating: 0, count: 4)
        bc[i0] = lambda0
        bc[i1] = lambda1
        return SimplexClosest(point: lambda0 * a + lambda1 * b,
                              barycentric: bc, mask: mask)
    }

    static func closestTriangle(_ vertices: [SupportVertex],
                                _ i0: Int, _ i1: Int, _ i2: Int) -> SimplexClosest {
        let a = vertices[i0].delta
        let b = vertices[i1].delta
        let c = vertices[i2].delta
        let u = a - b
        let w = a - c
        let normal = cross(u, w)
        let normalSquared = length_squared(normal)
        let degenerate = normalSquared < gjkSimplexEpsilon
        let inverse = 1 / (degenerate ? gjkSimplexEpsilon : normalSquared)
        let lambda2 = dot(cross(u, a), normal) * inverse
        let lambda1 = dot(cross(a, w), normal) * inverse
        let lambda0 = 1 - lambda2 - lambda1

        var candidates: [SimplexClosest] = []
        if lambda0 < 0 || degenerate { candidates.append(closestSegment(vertices, i1, i2)) }
        if lambda1 < 0 || degenerate { candidates.append(closestSegment(vertices, i0, i2)) }
        if lambda2 < 0 || degenerate { candidates.append(closestSegment(vertices, i0, i1)) }
        if let best = candidates.min(by: { length_squared($0.point) < length_squared($1.point) }) {
            return best
        }

        var bc = [Float](repeating: 0, count: 4)
        bc[i0] = lambda0
        bc[i1] = lambda1
        bc[i2] = lambda2
        return SimplexClosest(
            point: lambda0 * a + lambda1 * b + lambda2 * c,
            barycentric: bc,
            mask: (1 << UInt8(i0)) | (1 << UInt8(i1)) | (1 << UInt8(i2)))
    }

    static func determinant(_ a: F3, _ b: F3, _ c: F3, _ d: F3) -> Float {
        dot(b - a, cross(c - a, d - a))
    }

    static func closestTetrahedron(_ vertices: [SupportVertex]) -> SimplexClosest {
        let v0 = vertices[0].delta
        let v1 = vertices[1].delta
        let v2 = vertices[2].delta
        let v3 = vertices[3].delta
        let det = determinant(v0, v1, v2, v3)
        let degenerate = abs(det) < gjkSimplexEpsilon
        let inverse = 1 / (degenerate ? gjkSimplexEpsilon : det)
        let zero = F3.zero
        let lambda0 = determinant(zero, v1, v2, v3) * inverse
        let lambda1 = determinant(v0, zero, v2, v3) * inverse
        let lambda2 = determinant(v0, v1, zero, v3) * inverse
        let lambda3 = 1 - lambda0 - lambda1 - lambda2

        var candidates: [SimplexClosest] = []
        if lambda0 < 0 || degenerate { candidates.append(closestTriangle(vertices, 1, 2, 3)) }
        if lambda1 < 0 || degenerate { candidates.append(closestTriangle(vertices, 0, 2, 3)) }
        if lambda2 < 0 || degenerate { candidates.append(closestTriangle(vertices, 0, 1, 3)) }
        if lambda3 < 0 || degenerate { candidates.append(closestTriangle(vertices, 0, 1, 2)) }
        if let best = candidates.min(by: { length_squared($0.point) < length_squared($1.point) }) {
            return best
        }

        return SimplexClosest(
            point: .zero,
            barycentric: [lambda0, lambda1, lambda2, lambda3],
            mask: 0b1111)
    }

    static func simplexWitnesses(_ vertices: [SupportVertex],
                                 _ barycentric: [Float], _ mask: UInt8) -> (pointA: F3, pointB: F3) {
        var pointA = F3.zero
        var pointB = F3.zero
        for index in 0..<4 where mask & (1 << UInt8(index)) != 0 {
            pointA += vertices[index].pointA * barycentric[index]
            pointB += vertices[index].pointB * barycentric[index]
        }
        return (pointA, pointB)
    }
}

// MARK: - Deterministic sampled manifold

private extension ConvexNarrowPhase {
    struct LocalQuery {
        var algorithm: Algorithm
        var pointA: F3
        var pointB: F3
        var normal: F3
        var signedDistance: Float

        var isFinite: Bool {
            ConvexNarrowPhase.isFinite(pointA)
                && ConvexNarrowPhase.isFinite(pointB)
                && ConvexNarrowPhase.isFinite(normal)
                && signedDistance.isFinite
        }

        func withNormalizedNormal() -> LocalQuery {
            var copy = self
            copy.normal = safeNormalize(normal, fallback: F3(1, 0, 0))
            return copy
        }
    }

    struct LocalContact {
        var pointA: F3
        var pointB: F3
        var signedDistance: Float

        var center: F3 { (pointA + pointB) * 0.5 }
        var isFinite: Bool {
            ConvexNarrowPhase.isFinite(pointA)
                && ConvexNarrowPhase.isFinite(pointB)
                && signedDistance.isFinite
        }
    }

    struct Point2: Equatable {
        var x: Float
        var y: Float

        static func + (lhs: Point2, rhs: Point2) -> Point2 {
            Point2(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
        }
        static func - (lhs: Point2, rhs: Point2) -> Point2 {
            Point2(x: lhs.x - rhs.x, y: lhs.y - rhs.y)
        }
        static func * (lhs: Point2, rhs: Float) -> Point2 {
            Point2(x: lhs.x * rhs, y: lhs.y * rhs)
        }
    }

    static func buildManifold(context: SupportContext,
                              query: LocalQuery,
                              threshold: Float) -> [LocalContact] {
        let deepest = LocalContact(
            pointA: query.pointA,
            pointB: query.pointB,
            signedDistance: query.signedDistance)
        guard query.signedDistance <= threshold,
              context.a.isPolytope, context.b.isPolytope else { return [deepest] }

        let basis = orthonormalBasis(query.normal)
        let tangent1 = basis.1
        let tangent2 = basis.2
        let tilt = Float.pi * 2 / 180
        let sine = sin(tilt)
        let cosine = cos(tilt)
        var projectedA: [Point2] = []
        var projectedB: [Point2] = []
        var offsetsA: [Float] = []
        var offsetsB: [Float] = []

        for sample in 0..<5 {
            let angle = 2 * Float.pi * Float(sample) / 5
            let tangent = tangent1 * cos(angle) + tangent2 * sin(angle)
            let pointA = context.a.support(query.normal * cosine + tangent * sine)
            let pointB = context.supportB(-query.normal * cosine + tangent * sine)
            projectedA.append(Point2(x: dot(pointA, tangent1), y: dot(pointA, tangent2)))
            projectedB.append(Point2(x: dot(pointB, tangent1), y: dot(pointB, tangent2)))
            offsetsA.append(dot(pointA, query.normal))
            offsetsB.append(dot(pointB, query.normal))
        }

        let planeA = dot(context.a.support(query.normal), query.normal)
        let planeB = dot(context.supportB(-query.normal), query.normal)
        let scale = max(context.a.characteristicScale, context.b.characteristicScale)
        let planeTolerance = max(5.0e-5, scale * 1.0e-4)
        guard offsetsA.allSatisfy({ abs($0 - planeA) <= planeTolerance }),
              offsetsB.allSatisfy({ abs($0 - planeB) <= planeTolerance }) else {
            return [deepest]
        }

        let polygonA = convexHull2D(projectedA, epsilon: planeTolerance)
        let polygonB = convexHull2D(projectedB, epsilon: planeTolerance)
        guard polygonA.count >= 3, polygonB.count >= 3 else { return [deepest] }
        var intersection = intersectConvexPolygons(
            polygonA, polygonB, epsilon: planeTolerance)
        intersection = deduplicated(intersection, epsilon: planeTolerance)
        guard intersection.count >= 3,
              abs(polygonArea(intersection)) > planeTolerance * planeTolerance else {
            return [deepest]
        }

        let selected = selectAtMostFour(intersection)
        var contacts = selected.map { projected -> LocalContact in
            let tangentPoint = tangent1 * projected.x + tangent2 * projected.y
            let pointA = tangentPoint + query.normal * planeA
            let pointB = tangentPoint + query.normal * planeB
            return LocalContact(
                pointA: pointA,
                pointB: pointB,
                signedDistance: dot(pointB - pointA, query.normal))
        }

        let mergeDistanceSquared = max(1.0e-10, scale * scale * 1.0e-10)
        if contacts.count < maximumContacts,
           !contacts.contains(where: { length_squared($0.center - deepest.center) <= mergeDistanceSquared }) {
            contacts.append(deepest)
        }
        return contacts.isEmpty ? [deepest] : contacts
    }

    static func convexHull2D(_ points: [Point2], epsilon: Float) -> [Point2] {
        let sorted = deduplicated(points.sorted(by: point2Order), epsilon: epsilon)
        guard sorted.count > 2 else { return sorted }
        var lower: [Point2] = []
        for point in sorted {
            while lower.count >= 2,
                  cross2(lower[lower.count - 1] - lower[lower.count - 2],
                         point - lower[lower.count - 1]) <= epsilon {
                lower.removeLast()
            }
            lower.append(point)
        }
        var upper: [Point2] = []
        for point in sorted.reversed() {
            while upper.count >= 2,
                  cross2(upper[upper.count - 1] - upper[upper.count - 2],
                         point - upper[upper.count - 1]) <= epsilon {
                upper.removeLast()
            }
            upper.append(point)
        }
        lower.removeLast()
        upper.removeLast()
        return lower + upper
    }

    static func intersectConvexPolygons(_ subject: [Point2], _ clip: [Point2],
                                        epsilon: Float) -> [Point2] {
        var output = subject
        for index in clip.indices {
            guard !output.isEmpty else { break }
            let clipA = clip[index]
            let clipB = clip[(index + 1) % clip.count]
            let edge = clipB - clipA
            let input = output
            output = []
            var previous = input[input.count - 1]
            var previousInside = cross2(edge, previous - clipA) >= -epsilon
            for current in input {
                let currentInside = cross2(edge, current - clipA) >= -epsilon
                if currentInside != previousInside,
                   let intersection = segmentLineIntersection(
                    previous, current, clipA, edge, epsilon: epsilon) {
                    output.append(intersection)
                }
                if currentInside { output.append(current) }
                previous = current
                previousInside = currentInside
            }
            output = deduplicated(output, epsilon: epsilon)
        }
        return output
    }

    static func segmentLineIntersection(_ p0: Point2, _ p1: Point2,
                                        _ linePoint: Point2, _ lineDirection: Point2,
                                        epsilon: Float) -> Point2? {
        let segment = p1 - p0
        let denominator = cross2(lineDirection, segment)
        guard abs(denominator) > epsilon * epsilon else { return nil }
        let t = -cross2(lineDirection, p0 - linePoint) / denominator
        return p0 + segment * min(max(t, 0), 1)
    }

    static func selectAtMostFour(_ polygon: [Point2]) -> [Point2] {
        guard polygon.count > 4 else { return polygon }
        var best: [Point2] = Array(polygon.prefix(4))
        var bestArea = abs(polygonArea(best))
        for i in 0..<(polygon.count - 3) {
            for j in (i + 1)..<(polygon.count - 2) {
                for k in (j + 1)..<(polygon.count - 1) {
                    for l in (k + 1)..<polygon.count {
                        let candidate = [polygon[i], polygon[j], polygon[k], polygon[l]]
                        let area = abs(polygonArea(candidate))
                        if area > bestArea {
                            best = candidate
                            bestArea = area
                        }
                    }
                }
            }
        }
        return best
    }

    static func polygonArea(_ polygon: [Point2]) -> Float {
        guard polygon.count >= 3 else { return 0 }
        var twiceArea: Float = 0
        for index in polygon.indices {
            let next = polygon[(index + 1) % polygon.count]
            twiceArea += cross2(polygon[index], next)
        }
        return 0.5 * twiceArea
    }

    static func deduplicated(_ points: [Point2], epsilon: Float) -> [Point2] {
        let epsilonSquared = epsilon * epsilon
        var result: [Point2] = []
        for point in points where !result.contains(where: {
            let delta = point - $0
            return delta.x * delta.x + delta.y * delta.y <= epsilonSquared
        }) {
            result.append(point)
        }
        return result
    }

    static func cross2(_ a: Point2, _ b: Point2) -> Float { a.x * b.y - a.y * b.x }

    static func point2Order(_ a: Point2, _ b: Point2) -> Bool {
        a.x == b.x ? a.y < b.y : a.x < b.x
    }

    static func deterministicContactOrder(_ a: Contact, _ b: Contact) -> Bool {
        let ac = a.center
        let bc = b.center
        if ac.x != bc.x { return ac.x < bc.x }
        if ac.y != bc.y { return ac.y < bc.y }
        if ac.z != bc.z { return ac.z < bc.z }
        if a.pointA.x != b.pointA.x { return a.pointA.x < b.pointA.x }
        if a.pointA.y != b.pointA.y { return a.pointA.y < b.pointA.y }
        return a.pointA.z < b.pointA.z
    }
}

// MARK: - Finite helpers

private extension ConvexNarrowPhase {
    static func isFinite(_ value: F3) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }

    static func isFinite(_ value: SIMD4<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite && value.w.isFinite
    }

    static func safeNormalize(_ value: F3, fallback: F3) -> F3 {
        let squared = length_squared(value)
        if squared.isFinite, squared > minimumVectorLengthSquared {
            return value / squared.squareRoot()
        }
        return fallback
    }
}
