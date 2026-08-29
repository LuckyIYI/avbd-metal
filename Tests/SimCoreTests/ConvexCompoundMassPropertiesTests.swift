import Foundation
import simd
import XCTest
@testable import SimCore

final class ConvexCompoundMassPropertiesTests: XCTestCase {
    func testRightTetraIntegratesCompleteTensorAndPrincipalFrame() throws {
        let compound = try makeCompound(part: makeRightTetra())
        let properties = try compound.massProperties()

        XCTAssertEqual(properties.volume, 1.0 / 6.0, accuracy: 1e-6)
        XCTAssertEqual(properties.centerOfMass.x, 0.25, accuracy: 1e-6)
        XCTAssertEqual(properties.centerOfMass.y, 0.25, accuracy: 1e-6)
        XCTAssertEqual(properties.centerOfMass.z, 0.25, accuracy: 1e-6)

        let tensor = properties.inertiaTensorAtUnitDensity
        let diagonal: Float = 1.0 / 80.0
        let product: Float = 1.0 / 480.0
        for axis in 0..<3 {
            XCTAssertEqual(tensor[axis][axis], diagonal, accuracy: 2e-7)
        }
        XCTAssertEqual(tensor[0][1], product, accuracy: 2e-7)
        XCTAssertEqual(tensor[0][2], product, accuracy: 2e-7)
        XCTAssertEqual(tensor[1][2], product, accuracy: 2e-7)

        let moments = properties.principalInertiaAtUnitDensity
        XCTAssertEqual(moments.x, 1.0 / 96.0, accuracy: 2e-7)
        XCTAssertEqual(moments.y, 1.0 / 96.0, accuracy: 2e-7)
        XCTAssertEqual(moments.z, 1.0 / 60.0, accuracy: 2e-7)
        XCTAssertLessThanOrEqual(moments.x, moments.y + moments.z)
        XCTAssertLessThanOrEqual(moments.y, moments.x + moments.z)
        XCTAssertLessThanOrEqual(moments.z, moments.x + moments.y)

        let rotation = properties.principalRotation
        let axes = (
            rotation.act(F3(1, 0, 0)),
            rotation.act(F3(0, 1, 0)),
            rotation.act(F3(0, 0, 1))
        )
        XCTAssertGreaterThan(dot(cross(axes.0, axes.1), axes.2), 0.99999)
        XCTAssertGreaterThan(
            abs(dot(axes.2, normalize(F3(repeating: 1)))), 0.99999
        )

        let sourceFromPrincipal = simd_float3x3(columns: axes)
        let reconstructed = sourceFromPrincipal
            * simd_float3x3(diagonal: moments)
            * sourceFromPrincipal.transpose
        for column in 0..<3 {
            for row in 0..<3 {
                XCTAssertEqual(
                    reconstructed[column][row], tensor[column][row],
                    accuracy: 3e-7
                )
            }
        }
    }

    func testTranslatedGeometryKeepsMomentsAndDeterministicQuaternion() throws {
        let offset = F3(10_000, -20_000, 30_000)
        let compound = try makeCompound(part: makeRightTetra(offset: offset))
        let first = try compound.massProperties()
        let second = try compound.massProperties()

        XCTAssertEqual(first.centerOfMass.x, offset.x + 0.25, accuracy: 1e-4)
        XCTAssertEqual(first.centerOfMass.y, offset.y + 0.25, accuracy: 1e-4)
        XCTAssertEqual(first.centerOfMass.z, offset.z + 0.25, accuracy: 1e-4)
        XCTAssertEqual(first.volume, 1.0 / 6.0, accuracy: 1e-6)
        XCTAssertEqual(
            first.principalInertiaAtUnitDensity,
            second.principalInertiaAtUnitDensity
        )
        XCTAssertEqual(first.principalRotation.vector,
                       second.principalRotation.vector)
        XCTAssertGreaterThanOrEqual(first.principalRotation.real, 0)
    }

    func testUniformScaleProducesCubicVolumeAndQuinticInertia() throws {
        let base = try makeCompound(part: makeRightTetra()).massProperties()
        let scale: Float = 2.5
        let scaled = try makeCompound(
            part: makeRightTetra(scale: scale)
        ).massProperties()
        let cubic = scale * scale * scale
        let quintic = cubic * scale * scale

        XCTAssertEqual(scaled.volume, base.volume * cubic, accuracy: 2e-5)
        XCTAssertEqual(
            scaled.centerOfMass.x, base.centerOfMass.x * scale,
            accuracy: 2e-6
        )
        XCTAssertEqual(
            scaled.centerOfMass.y, base.centerOfMass.y * scale,
            accuracy: 2e-6
        )
        XCTAssertEqual(
            scaled.centerOfMass.z, base.centerOfMass.z * scale,
            accuracy: 2e-6
        )
        for axis in 0..<3 {
            XCTAssertEqual(
                scaled.principalInertiaAtUnitDensity[axis],
                base.principalInertiaAtUnitDensity[axis] * quintic,
                accuracy: 3e-5
            )
        }
    }

    func testExistingCompoundMatchesItsKnownVolumeMoments() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(
                "Sources/PhysicsAVBD/Assets/convex/concave-u.avbdconvex.json"
            )
        let properties = try ConvexCompoundAsset.load(from: url)
            .massProperties()

        XCTAssertEqual(properties.volume, 7.00281, accuracy: 2e-4)
        XCTAssertEqual(properties.centerOfMass.x, -0.0001143, accuracy: 2e-5)
        XCTAssertEqual(properties.centerOfMass.y, 0, accuracy: 2e-5)
        XCTAssertEqual(properties.centerOfMass.z, -0.142732, accuracy: 2e-4)
        XCTAssertEqual(
            properties.principalInertiaAtUnitDensity.x,
            9.03712 / 1.5,
            accuracy: 2e-3
        )
        XCTAssertEqual(
            properties.principalInertiaAtUnitDensity.y,
            10.75140 / 1.5,
            accuracy: 2e-3
        )
        XCTAssertEqual(
            properties.principalInertiaAtUnitDensity.z,
            18.03782 / 1.5,
            accuracy: 2e-3
        )
    }

    private func makeRightTetra(
        offset: F3 = .zero, scale: Float = 1
    ) throws -> ConvexHullAsset {
        let vertices: [F3] = [
            offset,
            offset + F3(0, 0, scale),
            offset + F3(0, scale, 0),
            offset + F3(scale, 0, 0),
        ]
        let triangles: [SIMD3<UInt32>] = [
            SIMD3(0, 1, 2),
            SIMD3(0, 2, 3),
            SIMD3(0, 3, 1),
            SIMD3(1, 3, 2),
        ]
        let edges = [
            ConvexHullEdge(vertexA: 0, vertexB: 1, faceA: 0, faceB: 2),
            ConvexHullEdge(vertexA: 0, vertexB: 2, faceA: 0, faceB: 1),
            ConvexHullEdge(vertexA: 0, vertexB: 3, faceA: 1, faceB: 2),
            ConvexHullEdge(vertexA: 1, vertexB: 2, faceA: 0, faceB: 3),
            ConvexHullEdge(vertexA: 1, vertexB: 3, faceA: 2, faceB: 3),
            ConvexHullEdge(vertexA: 2, vertexB: 3, faceA: 1, faceB: 3),
        ]
        let digest = ConvexHullAsset.geometryDigest(
            vertices: vertices, triangles: triangles
        )
        return try ConvexHullAsset(
            vertices: vertices,
            triangles: triangles,
            edges: edges,
            boundsMin: offset,
            boundsMax: offset + F3(repeating: scale),
            boundingRadius: sqrt(3) * scale * 0.5,
            volume: scale * scale * scale / 6,
            centroid: offset + F3(repeating: scale * 0.25),
            digest: digest,
            stableID: "hull-" + digest.prefix(16)
        )
    }

    private func makeCompound(part: ConvexHullAsset) throws
        -> ConvexCompoundAsset
    {
        let fixtureURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(
                "Sources/PhysicsAVBD/Assets/convex/concave-u.avbdconvex.json"
            )
        let fixture = try ConvexCompoundAsset.load(from: fixtureURL)
        return try ConvexCompoundAsset(
            source: fixture.source,
            cooker: fixture.cooker,
            cacheKey: fixture.cacheKey,
            digest: ConvexCompoundAsset.contentDigest(parts: [part]),
            parts: [part]
        )
    }
}
