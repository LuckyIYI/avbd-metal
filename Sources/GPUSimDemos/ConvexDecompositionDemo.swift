import Foundation
import SimCore
import simd

extension Demos {
    /// End-to-end concave collision example. The visible geometry is the
    /// original connected U-shaped triangle mesh, while physics uses one
    /// deterministic offline-cooked convex compound shared by every instance.
    /// A sphere can fall into the open cavity; replacing the compound with the
    /// source mesh's single convex hull would incorrectly seal that opening.
    public static func convexDecomposition(scale: Int = 1) -> PhysicsScene {
        let compound = loadConvexDecompositionAsset()
        let visual = loadConvexDecompositionVisual()
        let repetitions = max(1, min(scale, 8))
        let massProperties: ConvexCompoundMassProperties
        do {
            massProperties = try compound.massProperties()
        } catch {
            preconditionFailure(
                "bundled concave-U mass properties are invalid: \(error)")
        }

        var scene = PhysicsScene(name: "convexdecomp")
        scene.settings.iterations = 14
        scene.settings.betaLin = 12_000
        scene.settings.cameraDistance = 13
        scene.settings.cameraTargetZ = 2.4
        addGround(&scene, friction: 0.75)

        func addU(
            at position: F3,
            rotation: Quat = Quat(real: 1, imag: .zero),
            dynamic: Bool,
            color: F3
        ) -> Int {
            let bounds = convexCompoundBounds(compound)
            let size = bounds.max - bounds.min
            let density: Float = dynamic ? 1.5 : 0
            let mass: Float? = dynamic ? massProperties.volume * density : nil
            let diagonalInertia: F3? = dynamic
                ? massProperties.principalInertiaAtUnitDensity * density : nil
            let normalizedRotation = rotation.normalized
            let principalToSource = massProperties.principalRotation
            let sourceToPrincipal = principalToSource.inverse.normalized
            // SceneBody.position is the inertial COM and its rotation is the
            // principal frame. Keep the authored mesh origin fixed in world
            // space by applying the inverse COM/principal transform to every
            // attached collider and visual.
            let bodyPosition = dynamic
                ? position + normalizedRotation.act(massProperties.centerOfMass)
                : position
            let bodyRotation = dynamic
                ? (normalizedRotation * principalToSource).normalized
                : normalizedRotation
            let localGeometryOffset = dynamic
                ? sourceToPrincipal.act(-massProperties.centerOfMass) : .zero
            let localGeometryRotation = dynamic
                ? sourceToPrincipal : Quat(real: 1, imag: .zero)
            let body = scene.addBody(
                size: size, density: density, friction: 0.72,
                position: bodyPosition, rotation: bodyRotation,
                mass: mass, diagonalInertia: diagonalInertia,
                collisionEnabled: false)
            _ = scene.addConvexCompound(
                body: body, asset: compound, friction: 0.72,
                localPosition: localGeometryOffset,
                localRotation: localGeometryRotation,
                collisionEnabled: true, isRendered: false)
            scene.addRigidMesh(SceneRigidMesh(
                body: body, mesh: visual,
                localPosition: localGeometryOffset,
                localRotation: localGeometryRotation, color: color))
            return body
        }

        // Static receiver: the yellow sphere must enter the cavity and settle
        // on its lower bar instead of resting on an imaginary convex lid.
        _ = addU(at: F3(-2.4, 0, 1.5), dynamic: false,
                 color: F3(0.18, 0.52, 0.88))
        _ = scene.addSphere(diameter: 0.64, density: 2.0, friction: 0.55,
                            position: F3(-2.4, 0, 5.4))

        // Dynamic compound against ordinary analytic primitives.
        _ = addU(
            at: F3(2.0, -0.15, 4.8),
            rotation: Quat(angle: 0.28, axis: normalize(F3(0.3, 1, 0.2))),
            dynamic: true, color: F3(0.86, 0.28, 0.18))
        _ = scene.addBody(size: F3(1.1, 1.1, 1.1), density: 1.2,
                          friction: 0.65, position: F3(2.2, 0.1, 1.0))
        _ = scene.addCapsule(length: 1.1, radius: 0.28, density: 1.0,
                             friction: 0.6, position: F3(3.4, 0.3, 3.3),
                             rotation: Quat(angle: .pi / 2, axis: F3(0, 1, 0)))

        // Extra instances demonstrate geometry sharing and exercise compound
        // vs compound manifolds without multiplying the cooked asset payload.
        if repetitions > 1 {
            for index in 1..<repetitions {
                let x = Float(index - 1) * 1.7 - 0.8
                _ = addU(
                    at: F3(x, 3.2, 5.5 + Float(index) * 1.4),
                    rotation: Quat(
                        angle: 0.35 * Float(index),
                        axis: normalize(F3(0.2, 0.4, 1))),
                    dynamic: true,
                    color: F3(0.35 + 0.05 * Float(index), 0.72, 0.28))
            }
        }
        return scene
    }

    private static func loadConvexDecompositionAsset() -> ConvexCompoundAsset {
        guard let url = Demos.resourceBundle.url(
            forResource: "concave-u", withExtension: "avbdconvex.json",
            subdirectory: "Assets/convex") else {
            preconditionFailure("bundled concave-U convex asset is missing")
        }
        do {
            return try ConvexCompoundAsset.load(from: url)
        } catch {
            preconditionFailure("bundled concave-U convex asset is invalid: \(error)")
        }
    }

    private static func loadConvexDecompositionVisual() -> SurfaceMesh {
        guard let url = Demos.resourceBundle.url(
            forResource: "concave-u", withExtension: "obj",
            subdirectory: "Assets/convex") else {
            preconditionFailure("bundled concave-U source mesh is missing")
        }
        do {
            return try SurfaceMesh.load(path: url.path, upAxis: .z)
        } catch {
            preconditionFailure("bundled concave-U source mesh is invalid: \(error)")
        }
    }

    private static func convexCompoundBounds(
        _ compound: ConvexCompoundAsset
    ) -> (min: F3, max: F3) {
        let lo = compound.parts.reduce(F3(repeating: .infinity)) {
            simd_min($0, $1.boundsMin)
        }
        let hi = compound.parts.reduce(F3(repeating: -.infinity)) {
            simd_max($0, $1.boundsMax)
        }
        return (lo, hi)
    }

}
