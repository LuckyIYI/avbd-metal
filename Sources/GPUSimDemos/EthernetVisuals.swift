import SimCore
import simd

/// Display details on the robot and workbench. The connector and slot retain
/// their authored contact geometry; these finishes do not guide insertion.
enum EthernetVisuals {
    static func finish(_ scene: inout PhysicsScene, bodies: Set<Int>, table: Int) {
        for i in scene.colliders.indices {
            let c = scene.colliders[i]
            guard bodies.contains(c.body), c.shape == .box, c.isRendered else { continue }
            let bevel = min(0.0003, c.size.min() * 0.10)
            let color = c.renderColor ?? F3(0.4, 0.4, 0.4)
            let metallic: Float = abs(color.x - color.z) < 0.15 && color.x > 0.4 ? 0.72 : 0.05
            scene.addRigidMesh(
                SceneRigidMesh(
                    body: c.body,
                    mesh: chamferedBox(c.size, bevel: bevel),
                    localPosition: c.localPosition, localRotation: c.localRotation,
                    color: color, roughness: metallic > 0.5 ? 0.3 : 0.48, metallic: metallic))
            scene.colliders[i].isRendered = false
        }
        var v: [F3] = []
        var n: [F3] = []
        var t: [(Int, Int, Int)] = []
        for i in 0..<10 {
            for j in 0..<5 {
                let center = F3(Float(i) * 0.025 - 0.125, Float(j) * 0.025 - 0.05, 0.00302)
                let base = v.count
                v.append(center)
                n.append(F3(0, 0, 1))
                for k in 0...20 {
                    let a = 2 * Float.pi * Float(k) / 20
                    v.append(center + F3(cos(a), sin(a), 0) * 0.00125)
                    n.append(F3(0, 0, 1))
                }
                for k in 0..<20 { t.append((base, base + k + 1, base + k + 2)) }
            }
        }
        scene.addRigidMesh(
            SceneRigidMesh(
                body: table,
                mesh: SurfaceMesh(vertices: v, normals: n, triangles: t), color: F3(0.055, 0.065, 0.075),
                roughness: 0.8))
    }

    private static func chamferedBox(_ size: F3, bevel b: Float) -> SurfaceMesh {
        let h = size / 2
        var vertices: [F3] = []
        var normals: [F3] = []
        var triangles: [(Int, Int, Int)] = []
        func face(_ points: [F3], _ normal: F3) {
            var p = points
            if dot(cross(p[1] - p[0], p[2] - p[0]), normal) < 0 { p.reverse() }
            let base = vertices.count
            vertices += p
            normals += Array(repeating: normalize(normal), count: p.count)
            for i in 1..<p.count - 1 { triangles.append((base, base + i, base + i + 1)) }
        }
        for a in 0..<3 {
            for sign: Float in [-1, 1] {
                let u = (a + 1) % 3
                let v = (a + 2) % 3
                var points: [F3] = []
                for (su, sv): (Float, Float) in [(-1, -1), (1, -1), (1, 1), (-1, 1)] {
                    var p = F3.zero
                    p[a] = sign * h[a]
                    p[u] = su * (h[u] - b)
                    p[v] = sv * (h[v] - b)
                    points.append(p)
                }
                var n = F3.zero
                n[a] = sign
                face(points, n)
            }
        }
        for a in 0..<3 {
            for d in a + 1..<3 {
                for sa: Float in [-1, 1] {
                    for sd: Float in [-1, 1] {
                        let c = 3 - a - d
                        var points: [F3] = []
                        for (side, end): (Int, Float) in [(0, -1), (1, -1), (1, 1), (0, 1)] {
                            var p = F3.zero
                            p[a] = sa * (h[a] - (side == 0 ? 0 : b))
                            p[d] = sd * (h[d] - (side == 0 ? b : 0))
                            p[c] = end * (h[c] - b)
                            points.append(p)
                        }
                        var n = F3.zero
                        n[a] = sa
                        n[d] = sd
                        face(points, n)
                    }
                }
            }
        }
        for x: Float in [-1, 1] {
            for y: Float in [-1, 1] {
                for z: Float in [-1, 1] {
                    let sign = F3(x, y, z)
                    var points: [F3] = []
                    for a in 0..<3 {
                        var p = (h - F3(repeating: b)) * sign
                        p[a] = h[a] * sign[a]
                        points.append(p)
                    }
                    face(points, sign)
                }
            }
        }
        return SurfaceMesh(vertices: vertices, normals: normals, triangles: triangles)
    }
}
