import Foundation
import SimCore
import simd

package struct ClassicRigidBodySpec: Sendable {
    package let assetName: String
    package let isDynamic: Bool
    package let targetMass: Float
    package let slot: F3
    package let yaw: Float
    package let color: F3
}

extension Demos {
    package static let classicRigidTableTopSize = F3(8.6, 4.6, 0.34)
    package static let classicRigidTableTopPosition = F3(0, 0, 2.8)

    /// Metadata order matches `PhysicsScene.rigidMeshes` in
    /// `classicRigidBodies()`, allowing focused tests and tooling to identify
    /// instances without depending on incidental body indices.
    package static let classicRigidBodySpecs: [ClassicRigidBodySpec] = [
        ClassicRigidBodySpec(
            assetName: "stanford-bunny", isDynamic: true, targetMass: 1.8,
            slot: F3(-3.0, -0.52, 0), yaw: 0.18,
            color: F3(0.32, 0.52, 0.92)),
        ClassicRigidBodySpec(
            assetName: "stanford-dragon", isDynamic: true, targetMass: 2.6,
            slot: F3(-1.0, 0.50, 0), yaw: -0.20,
            color: F3(0.18, 0.68, 0.48)),
        ClassicRigidBodySpec(
            assetName: "stanford-armadillo", isDynamic: true, targetMass: 2.2,
            slot: F3(1.0, -0.48, 0), yaw: 0.16,
            color: F3(0.88, 0.38, 0.20)),
        ClassicRigidBodySpec(
            assetName: "utah-teapot", isDynamic: true, targetMass: 1.6,
            slot: F3(3.0, 0.50, 0), yaw: -0.24,
            color: F3(0.72, 0.34, 0.86)),
    ]

    /// Four classic detailed meshes on a table made only from analytic boxes.
    /// The visible source surfaces remain independent of their offline-cooked
    /// convex compounds, so the app's collision-hull overlay exposes the exact
    /// geometry used by contact generation.
    public static func classicRigidBodies(scale _: Int = 1) -> PhysicsScene {
        var scene = PhysicsScene(name: "classicrigids")
        scene.settings.iterations = 18
        scene.settings.betaLin = 16_000
        scene.settings.collisionMargin = 0.008
        scene.settings.cameraDistance = 11.2
        scene.settings.cameraTargetZ = 3.35
        scene.settings.cameraAzimuth = 0.86
        scene.settings.cameraElevation = 0.34
        addGround(&scene, friction: 0.82)

        let tableSize = classicRigidTableTopSize
        let tablePosition = classicRigidTableTopPosition
        let tabletop = F3(0.46, 0.25, 0.10)
        let legs = F3(0.34, 0.16, 0.07)
        addClassicTableBox(
            to: &scene, size: tableSize, position: tablePosition,
            friction: 0.86, color: tabletop)
        let legHeight = tablePosition.z - tableSize.z * 0.5
        let legX = tableSize.x * 0.5 - 0.62
        let legY = tableSize.y * 0.5 - 0.55
        for x in [-legX, legX] {
            for y in [-legY, legY] {
                addClassicTableBox(
                    to: &scene, size: F3(0.55, 0.55, legHeight),
                    position: F3(x, y, legHeight * 0.5),
                    friction: 0.86, color: legs)
            }
        }

        let tableTopZ = tablePosition.z + tableSize.z * 0.5
        for spec in classicRigidBodySpecs {
            let compound = loadClassicRigidCompound(named: spec.assetName)
            let visual = loadClassicRigidVisual(named: spec.assetName)
            let sourceRotation = Quat(angle: spec.yaw, axis: F3(0, 0, 1))
            let rotatedBounds = classicRotatedBounds(
                compound, rotation: sourceRotation)
            let rotatedCenter = (rotatedBounds.min + rotatedBounds.max) * 0.5
            let releaseGap: Float = spec.isDynamic ? 0.10 : 0
            let sourceOrigin = F3(
                spec.slot.x - rotatedCenter.x,
                spec.slot.y - rotatedCenter.y,
                tableTopZ + releaseGap - rotatedBounds.min.z)

            let body: Int
            let geometryPosition: F3
            let geometryRotation: Quat
            if spec.isDynamic {
                let properties: ConvexCompoundMassProperties
                do {
                    properties = try compound.massProperties()
                } catch {
                    preconditionFailure(
                        "classic rigid asset \(spec.assetName) has invalid mass properties: \(error)"
                    )
                }
                let principalToSource = properties.principalRotation
                let sourceToPrincipal = principalToSource.inverse.normalized
                let inertiaScale = spec.targetMass / properties.volume
                body = scene.addBody(
                    size: rotatedBounds.max - rotatedBounds.min,
                    density: 0, friction: 0.82,
                    position: sourceOrigin
                        + sourceRotation.act(properties.centerOfMass),
                    rotation: (sourceRotation * principalToSource).normalized,
                    mass: spec.targetMass,
                    diagonalInertia:
                        properties.principalInertiaAtUnitDensity * inertiaScale,
                    collisionEnabled: false)
                geometryPosition = sourceToPrincipal.act(
                    -properties.centerOfMass)
                geometryRotation = sourceToPrincipal
            } else {
                body = scene.addBody(
                    size: rotatedBounds.max - rotatedBounds.min,
                    density: 0, friction: 0.82,
                    position: sourceOrigin, rotation: sourceRotation,
                    collisionEnabled: false)
                geometryPosition = .zero
                geometryRotation = Quat(real: 1, imag: .zero)
            }

            _ = scene.addConvexCompound(
                body: body, asset: compound, friction: 0.82,
                localPosition: geometryPosition,
                localRotation: geometryRotation,
                collisionEnabled: true, isRendered: false)
            scene.addRigidMesh(SceneRigidMesh(
                body: body, mesh: visual,
                localPosition: geometryPosition,
                localRotation: geometryRotation,
                color: spec.color))
        }
        return scene
    }

    private static func addClassicTableBox(
        to scene: inout PhysicsScene,
        size: F3,
        position: F3,
        friction: Float,
        color: F3
    ) {
        let body = scene.addBody(
            size: size, density: 0, friction: friction,
            position: position, collisionEnabled: false)
        _ = scene.addCollider(
            body: body, size: size, friction: friction,
            collisionEnabled: true, isRendered: true,
            renderColor: color)
    }

    private static func loadClassicRigidCompound(
        named name: String
    ) -> ConvexCompoundAsset {
        guard let url = GPUSolver.physicsResourceBundle.url(
            forResource: name, withExtension: "avbdconvex.json",
            subdirectory: "Assets/convex/classic") else {
            preconditionFailure(
                "bundled classic rigid collision asset \(name) is missing"
            )
        }
        do {
            return try ConvexCompoundAsset.load(from: url)
        } catch {
            preconditionFailure(
                "bundled classic rigid collision asset \(name) is invalid: \(error)"
            )
        }
    }

    private static func loadClassicRigidVisual(named name: String) -> SurfaceMesh {
        guard let url = GPUSolver.physicsResourceBundle.url(
            forResource: name, withExtension: "obj",
            subdirectory: "Assets/classic") else {
            preconditionFailure(
                "bundled classic rigid visual asset \(name) is missing"
            )
        }
        do {
            return try SurfaceMesh.load(path: url.path, upAxis: .z)
        } catch {
            preconditionFailure(
                "bundled classic rigid visual asset \(name) is invalid: \(error)"
            )
        }
    }

    private static func classicRotatedBounds(
        _ compound: ConvexCompoundAsset,
        rotation: Quat
    ) -> (min: F3, max: F3) {
        var lower = F3(repeating: .infinity)
        var upper = F3(repeating: -.infinity)
        for part in compound.parts {
            for vertex in part.vertices {
                let point = rotation.act(vertex)
                lower = simd_min(lower, point)
                upper = simd_max(upper, point)
            }
        }
        return (lower, upper)
    }
}
