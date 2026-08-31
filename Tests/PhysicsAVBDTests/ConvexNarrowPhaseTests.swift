import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

final class ConvexNarrowPhaseTests: XCTestCase {
    func testSeparatedSpheresUseGJKAndReturnExactWitnessDistance() throws {
        let manifold = try value(ConvexNarrowPhase.query(
            shapeA: .sphere(radius: 1),
            shapeB: .sphere(radius: 1),
            poseB: .init(position: F3(3, 0, 0))))

        XCTAssertEqual(manifold.algorithm, .gjk)
        XCTAssertFalse(manifold.intersects)
        XCTAssertEqual(manifold.signedDistance, 1, accuracy: 1.0e-5)
        XCTAssertEqual(manifold.normalAtoB.x, 1, accuracy: 1.0e-5)
        XCTAssertEqual(manifold.contacts.count, 1)
        assertWitnessInvariant(manifold.contacts[0], accuracy: 1.0e-5)
    }

    func testTouchingBoxesUseCorrectedMPRInflation() throws {
        let options = ConvexNarrowPhase.Options(
            marginA: 1.0e-5, marginB: 1.0e-5,
            contactThreshold: 0.01)
        let manifold = try value(ConvexNarrowPhase.query(
            shapeA: .box(halfExtents: F3(repeating: 1)),
            shapeB: .box(halfExtents: F3(repeating: 1)),
            poseB: .init(position: F3(2, 0, 0)),
            options: options))

        XCTAssertEqual(manifold.algorithm, .mpr)
        XCTAssertEqual(manifold.signedDistance, 0, accuracy: 2.0e-5)
        XCTAssertGreaterThanOrEqual(manifold.contacts.count, 4)
        XCTAssertLessThanOrEqual(manifold.contacts.count, 5)
        for contact in manifold.contacts {
            assertWitnessInvariant(contact, accuracy: 3.0e-4)
        }
    }

    func testCoincidentCentersAndDeepPenetrationRemainFinite() throws {
        let manifold = try value(ConvexNarrowPhase.query(
            shapeA: .box(halfExtents: F3(1.2, 0.8, 0.6)),
            shapeB: .box(halfExtents: F3(0.7, 0.5, 0.4))))

        XCTAssertEqual(manifold.algorithm, .mpr)
        XCTAssertTrue(manifold.intersects)
        XCTAssertLessThan(manifold.signedDistance, -0.79)
        XCTAssertGreaterThan(length(manifold.normalAtoB), 0.999)
        assertFinite(manifold)
    }

    func testOffOriginHullUsesInteriorVertexMeanSeed() throws {
        // The authored origin is 26 m away from the hull. In world space pose A
        // moves the hull back around zero, leaving a true 1.42 mm gap to B.
        let offOriginHull = cubeVertices(center: F3(26, 0, 0), half: F3(repeating: 0.5))
        let manifold = try value(ConvexNarrowPhase.query(
            shapeA: .convexHull(vertices: offOriginHull),
            poseA: .init(position: F3(-26, 0, 0)),
            shapeB: .box(halfExtents: F3(repeating: 0.5)),
            poseB: .init(position: F3(1.00142, 0, 0))))

        XCTAssertEqual(manifold.algorithm, .gjk)
        XCTAssertEqual(manifold.signedDistance, 0.00142, accuracy: 2.0e-4)
        XCTAssertGreaterThan(manifold.normalAtoB.x, 0.999)
        assertWitnessInvariant(manifold.contacts[0], accuracy: 2.0e-4)
    }

    func testLargeSourceFrameHullUsesTranslationStableInteriorMean() throws {
        let base = F3(repeating: 1_000_000)
        var vertices: [F3] = []
        for x in 0..<4 {
            for y in 0..<4 {
                for z in 0..<4 {
                    vertices.append(base + F3(
                        Float(x) / 3, Float(y) / 3, Float(z) / 3))
                }
            }
        }

        let center = stableConvexSupportCenter(vertices)
        XCTAssertEqual(center.x, base.x + 0.5)
        XCTAssertEqual(center.y, base.y + 0.5)
        XCTAssertEqual(center.z, base.z + 0.5)

        let manifold = try value(ConvexNarrowPhase.query(
            shapeA: .convexHull(vertices: vertices),
            poseA: .init(position: -base),
            shapeB: .box(halfExtents: F3(repeating: 0.5)),
            poseB: .init(position: F3(1.50142, 0.5, 0.5))))

        XCTAssertEqual(manifold.algorithm, .gjk)
        XCTAssertEqual(manifold.signedDistance, 0.00142, accuracy: 2.0e-4)
        XCTAssertGreaterThan(manifold.normalAtoB.x, 0.999)
        assertWitnessInvariant(manifold.contacts[0], accuracy: 2.0e-4)
    }

    func testAsymmetricTetraUsesInteriorSeedAcrossAnalyticOrderings() throws {
        // Its AABB midpoint (2, 1, 0.5) is outside x/4+y/2+z<=1.
        // The vertex mean (1, 0.5, 0.25) is a guaranteed interior point.
        let tetra: ConvexNarrowPhase.Shape = .convexHull(vertices: [
            F3(0, 0, 0), F3(4, 0, 0),
            F3(0, 2, 0), F3(0, 0, 1),
        ])
        let cases: [(ConvexNarrowPhase.Shape, F3)] = [
            (.sphere(radius: 0.1), F3(0.25, 0.25, 0.25)),
            (.box(halfExtents: F3(repeating: 0.1)), F3(0.25, 0.25, 0.25)),
        ]

        for (analytic, overlapPosition) in cases {
            for (position, intersects) in [
                (overlapPosition, true),
                (F3(4.4, 0, 0), false),
            ] {
                let forward = try value(ConvexNarrowPhase.query(
                    shapeA: tetra,
                    shapeB: analytic,
                    poseB: .init(position: position)))
                let swapped = try value(ConvexNarrowPhase.query(
                    shapeA: analytic,
                    poseA: .init(position: position),
                    shapeB: tetra))

                XCTAssertEqual(forward.intersects, intersects)
                XCTAssertEqual(swapped.intersects, intersects)
                XCTAssertEqual(forward.signedDistance,
                               swapped.signedDistance, accuracy: 8.0e-4)
                XCTAssertEqual(dot(forward.normalAtoB,
                                   swapped.normalAtoB), -1,
                               accuracy: 2.0e-3)
                assertFinite(forward)
                assertFinite(swapped)
                assertWitnessInvariant(forward.contacts[0], accuracy: 8.0e-4)
                assertWitnessInvariant(swapped.contacts[0], accuracy: 8.0e-4)
            }
        }
    }

    func testHullHullOverlapBuildsDeterministicMultiContactManifold() throws {
        let hull = cubeVertices(center: .zero, half: F3(repeating: 1))
        let query = {
            ConvexNarrowPhase.query(
                shapeA: .convexHull(vertices: hull),
                shapeB: .convexHull(vertices: hull),
                poseB: .init(position: F3(1.5, 0, 0)))
        }
        let first = try value(query())
        let second = try value(query())

        XCTAssertEqual(first, second)
        XCTAssertEqual(first.algorithm, .mpr)
        XCTAssertEqual(first.signedDistance, -0.5, accuracy: 3.0e-4)
        XCTAssertGreaterThanOrEqual(first.contacts.count, 4)
        XCTAssertLessThanOrEqual(first.contacts.count, 5)
        XCTAssertEqual(first.contacts.map(\.sortKey), Array(0..<UInt8(first.contacts.count)))
        assertFinite(first)
    }

    func testHullSphereOverlapUsesGenericSupportMaps() throws {
        let hull = cubeVertices(center: .zero, half: F3(repeating: 1))
        let manifold = try value(ConvexNarrowPhase.query(
            shapeA: .convexHull(vertices: hull),
            shapeB: .sphere(radius: 0.5),
            poseB: .init(position: F3(1.25, 0, 0))))

        XCTAssertEqual(manifold.algorithm, .mpr)
        XCTAssertEqual(manifold.signedDistance, -0.25, accuracy: 4.0e-4)
        XCTAssertEqual(manifold.contacts.count, 1)
        XCTAssertGreaterThan(manifold.normalAtoB.x, 0.999)
        assertFinite(manifold)
    }

    func testCapsuleSupportUsesLocalZAxis() throws {
        let manifold = try value(ConvexNarrowPhase.query(
            shapeA: .capsule(radius: 0.25, halfHeight: 1),
            shapeB: .sphere(radius: 0.25),
            poseB: .init(position: F3(0, 0, 2))))

        XCTAssertEqual(manifold.algorithm, .gjk)
        XCTAssertEqual(manifold.signedDistance, 0.5, accuracy: 2.0e-4)
        XCTAssertGreaterThan(manifold.normalAtoB.z, 0.999)
        assertFinite(manifold)
    }

    func testTinyBoxRotationDoesNotFlipFaceNormalOrOrdering() throws {
        let tinyRotation = Quat(angle: 1.0e-7, axis: F3(0, 1, 0))
        let query = {
            ConvexNarrowPhase.query(
                shapeA: .box(halfExtents: F3(repeating: 1)),
                shapeB: .box(halfExtents: F3(repeating: 1)),
                poseB: .init(position: F3(0, 0, 2), orientation: tinyRotation))
        }
        let first = try value(query())
        let second = try value(query())

        XCTAssertEqual(first, second)
        XCTAssertGreaterThan(first.normalAtoB.z, 0.9999)
        XCTAssertGreaterThanOrEqual(first.contacts.count, 4)
        XCTAssertEqual(first.contacts.map(\.sortKey), Array(0..<UInt8(first.contacts.count)))
        assertFinite(first)
    }

    func testTorusIsExplicitlyUnsupported() {
        let result = ConvexNarrowPhase.query(
            shapeA: .torus(majorRadius: 1, minorRadius: 0.2),
            shapeB: .sphere(radius: 1))
        XCTAssertEqual(failure(result), .unsupportedShape(operand: .a, kind: .torus))
    }

    func testPlanarHullReturnsTypedDegeneracyFailure() {
        let planar = [
            F3(-1, -1, 0), F3(1, -1, 0),
            F3(1, 1, 0), F3(-1, 1, 0),
        ]
        let result = ConvexNarrowPhase.query(
            shapeA: .convexHull(vertices: planar),
            shapeB: .sphere(radius: 1))
        XCTAssertEqual(failure(result), .degenerateHull(operand: .a))
    }

    func testNonFiniteInputsReturnTypedFailuresInsteadOfNaN() {
        let invalidShape = ConvexNarrowPhase.query(
            shapeA: .sphere(radius: .nan),
            shapeB: .sphere(radius: 1))
        XCTAssertEqual(
            failure(invalidShape),
            .nonFiniteParameter(operand: .a, parameter: .radius))

        let invalidPose = ConvexNarrowPhase.query(
            shapeA: .sphere(radius: 1),
            poseA: .init(position: F3(.infinity, 0, 0)),
            shapeB: .sphere(radius: 1))
        XCTAssertEqual(
            failure(invalidPose),
            .nonFiniteParameter(operand: .a, parameter: .position))

        let invalidOrientation = ConvexNarrowPhase.query(
            shapeA: .sphere(radius: 1),
            poseA: .init(orientation: Quat(vector: .zero)),
            shapeB: .sphere(radius: 1))
        XCTAssertEqual(failure(invalidOrientation), .invalidOrientation(operand: .a))
    }

    private func value(
        _ result: Result<ConvexNarrowPhase.Manifold, ConvexNarrowPhase.Failure>,
        file: StaticString = #filePath, line: UInt = #line
    ) throws -> ConvexNarrowPhase.Manifold {
        switch result {
        case .success(let value): return value
        case .failure(let error):
            XCTFail("unexpected narrow-phase failure: \(error)", file: file, line: line)
            throw error
        }
    }

    private func failure(
        _ result: Result<ConvexNarrowPhase.Manifold, ConvexNarrowPhase.Failure>,
        file: StaticString = #filePath, line: UInt = #line
    ) -> ConvexNarrowPhase.Failure? {
        switch result {
        case .failure(let error): return error
        case .success:
            XCTFail("expected narrow-phase failure", file: file, line: line)
            return nil
        }
    }

    private func cubeVertices(center: F3, half: F3) -> [F3] {
        var vertices: [F3] = []
        for z: Float in [-1, 1] {
            for y: Float in [-1, 1] {
                for x: Float in [-1, 1] {
                    vertices.append(center + F3(x * half.x, y * half.y, z * half.z))
                }
            }
        }
        return vertices
    }

    private func assertWitnessInvariant(
        _ contact: ConvexNarrowPhase.Contact,
        accuracy: Float,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(
            dot(contact.pointB - contact.pointA, contact.normalAtoB),
            contact.signedDistance,
            accuracy: accuracy, file: file, line: line)
    }

    private func assertFinite(
        _ manifold: ConvexNarrowPhase.Manifold,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertTrue(manifold.signedDistance.isFinite, file: file, line: line)
        XCTAssertTrue(isFinite(manifold.normalAtoB), file: file, line: line)
        for contact in manifold.contacts {
            XCTAssertTrue(isFinite(contact.pointA), file: file, line: line)
            XCTAssertTrue(isFinite(contact.pointB), file: file, line: line)
            XCTAssertTrue(isFinite(contact.normalAtoB), file: file, line: line)
            XCTAssertTrue(contact.signedDistance.isFinite, file: file, line: line)
        }
    }

    private func isFinite(_ value: F3) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }
}
