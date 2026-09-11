import SimCore
import simd

public extension Demos {
    /// Suspended cables with different bend rigidities, a rigid payload, and
    /// capsule/box contacts, all participating in the same AVBD solve.
    static func cables(count: Int = 3, segments: Int = 24) -> PhysicsScene {
        precondition(count > 0 && segments >= 4)
        var scene = PhysicsScene(name: "Elastic cables")
        scene.settings.dt = 1 / 120
        scene.settings.iterations = 24
        scene.settings.collisionMargin = 0.002
        _ = scene.addBody(size: F3(6, Float(count) * 0.5 + 2, 0.2),
                          density: 0, friction: 0.6, position: F3(0, 0, -0.1))
        for row in 0..<count {
            let y = (Float(row) - Float(count - 1) * 0.5) * 0.45
            let points = (0...segments).map { i -> F3 in
                let t = Float(i) / Float(segments)
                return F3(-1.5 + 3 * t, y, 1.5 - 0.25 * sin(.pi * t))
            }
            let material = CableMaterial(stretchRigidity: 4000, shearRigidity: 2000,
                bendRigidity: 0.01 * pow(4, Float(row % 3)), twistRigidity: 0.02,
                dampingTime: 0.015)
            // All values above are finite, statically authored demo geometry.
            let cable = try! scene.addCable(points: points, radius: 0.025,
                density: 1000, material: material, friction: 0.6,
                fixedSegments: [0, segments - 1])
            if row == 0 {
                let middle = segments / 2
                let body = cable.bodyIDs[middle]
                let anchor = scene.bodies[body].position
                let payload = scene.addBody(size: F3(repeating: 0.18), density: 500,
                    friction: 0.5, position: anchor - F3(0, 0, 0.2))
                scene.addJoint(SceneJoint(bodyA: body, bodyB: payload, rA: .zero,
                                           rB: F3(0, 0, 0.2)))
            }
        }
        return scene
    }
}
