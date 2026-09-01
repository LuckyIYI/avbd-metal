@testable import PhysicsAVBD
@testable import GPUSimDemos
import SimCore
import Testing
import simd

@Suite("Cooked convex decomposition demo")
struct ConvexDecompositionDemoTests {
    @Test("connected concavity is preserved by shared compound parts")
    func sharedCompoundPreservesCavity() throws {
        #expect(Demos.resourceBundle.url(
            forResource: "concave-u", withExtension: "avbdconvex.json",
            subdirectory: "Assets/convex") != nil)
        let scene = Demos.convexDecomposition(scale: 2)

        #expect(scene.convexAssets.count == 3)
        #expect(Set(scene.convexAssets.map(\.digest)).count == 3)
        #expect(scene.rigidMeshes.count == 3)
        #expect(scene.rigidMeshes.allSatisfy { !$0.triangles.isEmpty })

        let dynamicU = scene.bodies[3]
        #expect(abs((dynamicU.mass ?? 0) - 10.50422) < 1e-3)
        #expect(length((dynamicU.diagonalInertia ?? .zero)
            - F3(9.03712, 10.75140, 18.03782)) < 2e-3)
        let dynamicVisual = scene.rigidMeshes.first { $0.body == 3 }!
        let authoredDynamicRotation = Quat(
            angle: 0.28, axis: normalize(F3(0.3, 1, 0.2)))
        #expect(length(dynamicU.position
            + dynamicU.rotation.act(dynamicVisual.localPosition)
            - F3(2.0, -0.15, 4.8)) < 2e-5)
        let reconstructedSourceRotation = (
            dynamicU.rotation * dynamicVisual.localRotation).normalized
        #expect(abs(dot(reconstructedSourceRotation.vector,
                        authoredDynamicRotation.vector)) > 0.99999)
        let staticVisual = scene.rigidMeshes[0]
        let staticU = scene.bodies[staticVisual.body]
        #expect(length(staticU.position
            + staticU.rotation.act(staticVisual.localPosition)
            - F3(-2.4, 0, 1.5)) < 1e-5)

        let assetColliders = scene.colliders.filter { $0.convexAssetID != nil }
        #expect(assetColliders.count == 9)
        #expect(assetColliders.allSatisfy {
            $0.convexHullVertices.isEmpty && !$0.isRendered
        })

        // This point is in the U's open channel. One convex hull of the source
        // would contain it; no part of the decomposition may seal it.
        let cavityPoint = F3(0, 0, 0.6)
        #expect(scene.convexAssets.allSatisfy {
            !contains(cavityPoint, in: $0)
        })
    }

    @Test("sphere enters the cavity on the Metal runtime")
    func sphereFallsInsideU() throws {
        let scene = Demos.convexDecomposition(scale: 1)
        let solver = try GPUSolver(scene: scene)
        // ground=0, static U=1, probe sphere=2
        for _ in 0..<360 {
            try solver.submitStep()
        }
        try solver.synchronize()

        let position = solver.bodyPosition(2)
        #expect(solver.runtimeFailure == nil)
        #expect(position.x.isFinite && position.y.isFinite && position.z.isFinite)
        #expect(abs(position.x + 2.4) < 0.55)
        #expect(abs(position.y) < 0.55)
        #expect(position.z > 0.9)
        #expect(position.z < 2.0)
        #expect(solver.activeRigidContactPairs().contains {
            Set([$0.0, $0.1]) == Set([1, 2])
        })
        #expect(solver.uniqueConvexAssetCount == scene.convexAssets.count)
        #expect(solver.convexDebugTriangleVertexCount > 0)
        #expect(solver.convexDebugEdgeVertexCount > 0)
    }

    private func contains(_ point: F3, in hull: ConvexHullAsset) -> Bool {
        let diagonal = length(hull.boundsMax - hull.boundsMin)
        let tolerance = max(1e-6, diagonal * 1e-5)
        for triangle in hull.triangles {
            let a = hull.vertices[Int(triangle.x)]
            let b = hull.vertices[Int(triangle.y)]
            let c = hull.vertices[Int(triangle.z)]
            let normal = cross(b - a, c - a)
            if dot(normal, point - a) > tolerance * max(length(normal), 1) {
                return false
            }
        }
        return true
    }
}
