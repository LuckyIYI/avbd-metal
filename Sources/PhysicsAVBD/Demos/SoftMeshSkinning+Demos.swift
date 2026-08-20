import Foundation
import simd
import SimCore

private struct SkinnedSoftMeshTemplate {
    var nodePositions: [F3]
    var tetIDs: [(Int, Int, Int, Int)]
    var jointPairs: [(Int, Int)]
    var skinVertices: [SceneSkinnedVertex]
    var triangles: [(Int, Int, Int)]
    var spacing: Float
}

private struct SkinnedSoftMeshTemplateKey: Hashable {
    var mesh: String
    var height: UInt32
    var res: Int
    var visualVertexLimit: Int
    var voxelVertexLimit: Int
    var minTetElements: Int
}

private final class SkinnedSoftMeshCache {
    static let shared = SkinnedSoftMeshCache()

    private let lock = NSLock()
    private var meshes: [String: SurfaceMesh] = [:]
    private var templates: [SkinnedSoftMeshTemplateKey: SkinnedSoftMeshTemplate] = [:]

    func mesh(for key: String, build: () -> SurfaceMesh) -> SurfaceMesh {
        lock.lock()
        if let cached = meshes[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let built = build()
        lock.lock()
        if let cached = meshes[key] {
            lock.unlock()
            return cached
        }
        meshes[key] = built
        lock.unlock()
        return built
    }

    func template(for key: SkinnedSoftMeshTemplateKey,
                  build: () -> SkinnedSoftMeshTemplate) -> SkinnedSoftMeshTemplate {
        lock.lock()
        if let cached = templates[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        let built = build()
        lock.lock()
        if let cached = templates[key] {
            lock.unlock()
            return cached
        }
        templates[key] = built
        lock.unlock()
        return built
    }
}

extension Demos {

    @discardableResult
    public static func addSkinnedSoftMesh(_ s: inout PhysicsScene,
                                          mesh input: SurfaceMesh,
                                          center: F3,
                                          height: Float = 1.7,
                                          res: Int = 8,
                                          visualVertexLimit: Int = 1400,
                                          voxelVertexLimit: Int = 6000,
                                          minTetElements: Int = 600,
                                          mu: Float = 2500,
                                          lambda: Float? = nil,
                                          massScale: Float = 28,
                                          friction: Float = 0.8)
        -> [Int] {
        addSkinnedSoftMesh(
            &s, mesh: input, center: center, height: height, res: res,
            visualVertexLimit: visualVertexLimit,
            voxelVertexLimit: voxelVertexLimit,
            minTetElements: minTetElements, mu: mu, lambda: lambda,
            massScale: massScale, friction: friction,
            selfCollisionEnabled: false)
    }

    @discardableResult
    public static func addSkinnedSoftMesh(_ s: inout PhysicsScene,
                                          mesh input: SurfaceMesh,
                                          center: F3,
                                          height: Float = 1.7,
                                          res: Int = 8,
                                          visualVertexLimit: Int = 1400,
                                          voxelVertexLimit: Int = 6000,
                                          minTetElements: Int = 600,
                                          mu: Float = 2500,
                                          lambda: Float? = nil,
                                          massScale: Float = 28,
                                          friction: Float = 0.8,
                                          selfCollisionEnabled: Bool)
        -> [Int] {
        let template = makeSkinnedSoftMeshTemplate(mesh: input, height: height,
                                                   res: res,
                                                   visualVertexLimit: visualVertexLimit,
                                                   voxelVertexLimit: voxelVertexLimit,
                                                   minTetElements: minTetElements)
        return instantiateSkinnedSoftMesh(&s, template: template, center: center,
                                          mu: mu, lambda: lambda ?? 10 * mu,
                                          massScale: massScale,
                                          friction: friction,
                                          selfCollisionEnabled:
                                            selfCollisionEnabled)
    }

    private static func makeSkinnedSoftMeshTemplate(mesh input: SurfaceMesh,
                                                    height: Float,
                                                    res: Int,
                                                    visualVertexLimit: Int,
                                                    voxelVertexLimit: Int,
                                                    minTetElements: Int)
        -> SkinnedSoftMeshTemplate {
        let fitted = input.fitted(height: height, center: .zero)
        let visual = fitted.simplified(maxVertices: visualVertexLimit)
        let voxelSource = fitted.simplified(maxVertices: voxelVertexLimit)
        let targetTets = max(1, minTetElements)
        var voxelRes = max(3, res)
        var surfaceBand: Float = 0.45
        var voxel = voxelSource.voxelCells(res: voxelRes,
                                           surfaceBand: surfaceBand)
        while voxelRes > 3 {
            let candidate = voxelSource.voxelCells(res: voxelRes - 1,
                                                   surfaceBand: surfaceBand)
            guard candidate.0.count * 5 >= targetTets else { break }
            voxelRes -= 1
            voxel = candidate
        }
        while voxel.0.count * 5 < targetTets
            && (voxelRes < 40 || surfaceBand < 1.05) {
            if voxelRes < 40 {
                voxelRes = min(40, voxelRes + max(1, voxelRes / 3))
            } else {
                surfaceBand = min(1.05, surfaceBand + 0.20)
            }
            voxel = voxelSource.voxelCells(res: voxelRes,
                                           surfaceBand: surfaceBand)
        }
        var cells = voxel.0
        if cells.count * 5 < targetTets {
            cells = expandedCells(cells, targetCount: (targetTets + 4) / 5)
        }
        let (_, origin, spacing) = voxel
        var tmp = PhysicsScene(name: "skin-template")
        let nodes = addSoftVoxelCells(&tmp, cells: cells, origin: origin,
                                      spacing: spacing, mu: 1,
                                      lambda: 1,
                                      massPerNode: 1,
                                      friction: 0)
        let skin = bindVisualMesh(visual, scene: tmp,
                                  tetRange: 0..<tmp.tets.count,
                                  bodyIDs: nodes)
        let nodePositions = nodes.map { tmp.bodies[$0].position }
        let joints = tmp.joints.compactMap { j -> (Int, Int)? in
            guard j.bodyA >= 0, j.bodyB >= 0 else { return nil }
            return (j.bodyA, j.bodyB)
        }
        return SkinnedSoftMeshTemplate(nodePositions: nodePositions,
                                       tetIDs: tmp.tets.map { $0.ids },
                                       jointPairs: joints,
                                       skinVertices: skin.vertices,
                                       triangles: skin.triangles,
                                       spacing: spacing)
    }

    private static func cachedSkinnedSoftMeshTemplate(mesh input: SurfaceMesh,
                                                      meshKey: String,
                                                      height: Float,
                                                      res: Int,
                                                      visualVertexLimit: Int,
                                                      voxelVertexLimit: Int,
                                                      minTetElements: Int)
        -> SkinnedSoftMeshTemplate {
        let key = SkinnedSoftMeshTemplateKey(mesh: meshKey,
                                             height: height.bitPattern,
                                             res: res,
                                             visualVertexLimit: visualVertexLimit,
                                             voxelVertexLimit: voxelVertexLimit,
                                             minTetElements: minTetElements)
        return SkinnedSoftMeshCache.shared.template(for: key) {
            makeSkinnedSoftMeshTemplate(mesh: input,
                                        height: height,
                                        res: res,
                                        visualVertexLimit: visualVertexLimit,
                                        voxelVertexLimit: voxelVertexLimit,
                                        minTetElements: minTetElements)
        }
    }

    @discardableResult
    private static func instantiateSkinnedSoftMesh(_ s: inout PhysicsScene,
                                                   template: SkinnedSoftMeshTemplate,
                                                   center: F3,
                                                   mu: Float,
                                                   lambda: Float,
                                                   massScale: Float,
                                                   friction: Float,
                                                   selfCollisionEnabled: Bool = false)
        -> [Int] {
        let mass = massScale * template.spacing * template.spacing * template.spacing
        var bodyIDs: [Int] = []
        bodyIDs.reserveCapacity(template.nodePositions.count)
        for p in template.nodePositions {
            bodyIDs.append(s.addParticle(radius: template.spacing * 0.32,
                                         mass: mass,
                                         friction: friction,
                                         position: p + center))
        }
        for ids in template.tetIDs {
            s.addTet(SceneTet(ids: (bodyIDs[ids.0], bodyIDs[ids.1],
                                    bodyIDs[ids.2], bodyIDs[ids.3]),
                              mu: mu, lambda: lambda,
                              selfCollisionEnabled: selfCollisionEnabled))
        }
        for (a, b) in template.jointPairs {
            s.addJoint(SceneJoint(bodyA: bodyIDs[a], bodyB: bodyIDs[b],
                                  rA: .zero, rB: .zero,
                                  stiffnessLin: 0, stiffnessAng: 0))
        }
        let vertices = template.skinVertices.map { v in
            SceneSkinnedVertex(ids: (bodyIDs[v.ids.0], bodyIDs[v.ids.1],
                                     bodyIDs[v.ids.2], bodyIDs[v.ids.3]),
                               weights: v.weights,
                               restNormal: v.restNormal,
                               restInv0: v.restInv0,
                               restInv1: v.restInv1,
                               restInv2: v.restInv2)
        }
        s.addSkinnedMesh(SceneSkinnedMesh(vertices: vertices,
                                          triangles: template.triangles,
                                          bodyIDs: bodyIDs))
        return bodyIDs
    }

    public static func skinnedbunny(res: Int = 8, count: Int = 1,
                                    stiffness: Float = 2500,
                                    friction: Float = 0.8,
                                    meshPath: String? = nil) -> PhysicsScene {
        var s = PhysicsScene(name: "skinnedbunny")
        s.settings.iterations = 12
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 1000
        addGround(&s, friction: friction)

        let (source, sourceKey) = loadDefaultSkinMeshWithKey(meshPath: meshPath)
        let n = max(1, count)
        let bodyHeight: Float = n > 8 ? 1.12 : 1.35
        let visualLimit = n > 12 ? 1600 : 2800
        let voxelLimit = n > 12 ? 6500 : 9000
        let template = cachedSkinnedSoftMeshTemplate(mesh: source,
                                                     meshKey: sourceKey,
                                                     height: bodyHeight,
                                                     res: res,
                                                     visualVertexLimit: visualLimit,
                                                     voxelVertexLimit: voxelLimit,
                                                     minTetElements: 1500)
        var templateMin = F3(repeating: Float.greatestFiniteMagnitude)
        var templateMax = F3(repeating: -Float.greatestFiniteMagnitude)
        for p in template.nodePositions {
            templateMin = min(templateMin, p)
            templateMax = max(templateMax, p)
        }
        let extent = max(templateMax - templateMin, F3(repeating: 1e-4))
        let planarExtent = max(extent.x, extent.y)
        let spacing = max(planarExtent * 1.38, bodyHeight * (n > 8 ? 1.22 : 1.26))
        let layerStride = max(extent.z * 1.36, bodyHeight * 1.48)
        let baseSlots = max(1, min(n, Int(ceil(Double(n) * (n > 8 ? 0.18 : 0.50)))))
        let cols = Int(ceil(sqrt(Double(baseSlots))))
        let rows = (baseSlots + cols - 1) / cols
        let layers = (n + baseSlots - 1) / baseSlots
        let layerShiftX = layers > 1 ? spacing * 0.18 : 0
        let layerShiftY = layers > 1 ? spacing * 0.12 : 0
        let halfX = Float(max(0, cols - 1)) * spacing * 0.5
            + layerShiftX + max(abs(templateMin.x), abs(templateMax.x)) + 0.10
        let halfY = Float(max(0, rows - 1)) * spacing * 0.5
            + layerShiftY + max(abs(templateMin.y), abs(templateMax.y)) + 0.10
        let baseZ = max(0.18, -templateMin.z + 0.06)
        let topZ = baseZ + Float(max(0, layers - 1)) * layerStride + templateMax.z
        let stackWallH = topZ + max(0.8, extent.z * 0.55)
        let wallH: Float = max(1.1, stackWallH / 4.0)
        addOpenBox(&s, halfX: halfX, halfY: halfY, wallHeight: wallH,
                   thickness: 0.16, friction: friction)
        var rng = SplitMix64(seed: 0x5eed_b00c)
        let jitterXY = min(0.045, spacing * 0.025)
        let jitterZ = min(0.035, layerStride * 0.018)
        for i in 0..<n {
            let slot = i % baseSlots
            let layer = i / baseSlots
            let x = Float(slot % cols) - Float(cols - 1) * 0.5
            let y = Float(slot / cols) - Float(rows - 1) * 0.5
            let sx = layers > 1 ? (layer % 2 == 0 ? -layerShiftX : layerShiftX) : 0
            let sy = layers > 1 ? Float((layer % 3) - 1) * layerShiftY : 0
            let jitter = F3((rng.nextFloat() - 0.5) * 2 * jitterXY,
                            (rng.nextFloat() - 0.5) * 2 * jitterXY,
                            (rng.nextFloat() - 0.5) * 2 * jitterZ)
            _ = instantiateSkinnedSoftMesh(&s, template: template,
                                           center: F3(x * spacing + sx,
                                                      y * spacing + sy,
                                                      baseZ + Float(layer)
                                                          * layerStride)
                                               + jitter,
                                           mu: stiffness,
                                           lambda: 10 * stiffness,
                                           massScale: 28,
                                           friction: friction)
        }

        if n <= 4 {
            _ = s.addSphere(diameter: 0.55, density: 2.2, friction: 0.6,
                            position: F3(0, 0, wallH + 2.8))
        }
        s.settings.cameraDistance = max(8, max(halfX, halfY) * 2.4)
        s.settings.cameraTargetZ = wallH * 0.52
        return s
    }

    public static func meshclothdrop(res: Int = 22, scale: Int = 1,
                                     stiffness: Float = 2500,
                                     friction: Float = 0.8,
                                     membraneMu: Float = 300,
                                     bend: Float = 5e-4,
                                     meshPath: String? = nil) -> PhysicsScene {
        var s = PhysicsScene(name: "meshclothdrop")
        s.settings.iterations = 20
        s.settings.betaLin = 20000
        s.settings.lambdaMax = 1200
        s.settings.particleDamping = 1.5
        s.settings.clothViscosity = 0.25
        addGround(&s, friction: friction)

        let (source, sourceKey) = loadDefaultSkinMeshWithKey(meshPath: meshPath)
        let k = max(1, scale)
        let n = max(14, res)
        let clothSize: Float = 3.6 + 0.32 * Float(k).squareRoot()
        let spacing = clothSize / Float(n - 1)
        let r: Float = min(0.045, 0.30 * spacing)
        let pinZ: Float = 1.55 + 0.05 * Float(k).squareRoot()
        var positions: [[F3]] = []
        for i in 0..<n {
            var row: [F3] = []
            for j in 0..<n {
                row.append(F3(Float(i) * spacing - clothSize / 2,
                              Float(j) * spacing - clothSize / 2,
                              pinZ))
            }
            positions.append(row)
        }
        let grid = addClothGrid(&s, positions: positions, thickness: r,
                                massPerNode: 0.010,
                                friction: friction,
                                structuralK: 5000,
                                hardRods: true,
                                membraneMu: membraneMu,
                                membraneBend: bend)

        for (i, j) in [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)] {
            let node = grid[i][j]
            s.addJoint(SceneJoint(bodyA: -1, bodyB: node,
                                  rA: s.bodies[node].position, rB: .zero))
        }

        let softCount: Int
        switch k {
        case 1: softCount = 2
        case 2: softCount = 3
        case 4: softCount = 5
        case 8: softCount = 8
        default: softCount = 12
        }
        let softRes = max(8, min(14, res / 2))
        let softLimit = softCount > 6 ? 1400 : 2200
        let voxelLimit = softCount > 6 ? 6500 : 9000
        let template = cachedSkinnedSoftMeshTemplate(mesh: source,
                                                     meshKey: sourceKey,
                                                     height: 0.62,
                                                     res: softRes,
                                                     visualVertexLimit: softLimit,
                                                     voxelVertexLimit: voxelLimit,
                                                     minTetElements: 600)
        let dropZ = pinZ + 2.2
        var rng = SplitMix64(seed: 0x5eed_4017)
        let cols = max(1, Int(ceil(sqrt(Double(softCount)))))
        for i in 0..<softCount {
            let gx = Float(i % cols) - Float(cols - 1) * 0.5
            let gy = Float(i / cols) - Float((softCount + cols - 1) / cols - 1) * 0.5
            let jitter = F3((rng.nextFloat() - 0.5) * 0.25,
                            (rng.nextFloat() - 0.5) * 0.25,
                            Float(i % 3) * 0.22)
            let center = F3(gx * 0.72, gy * 0.72, dropZ) + jitter
            _ = instantiateSkinnedSoftMesh(&s, template: template, center: center,
                                           mu: stiffness,
                                           lambda: 10 * stiffness,
                                           massScale: 22,
                                           friction: friction)
        }

        let rigidCount = softCount + max(2, k / 2)
        for i in 0..<rigidCount {
            let x = (rng.nextFloat() - 0.5) * clothSize * 0.62
            let y = (rng.nextFloat() - 0.5) * clothSize * 0.62
            let z = dropZ + 0.45 + Float(i % 5) * 0.42
            let q = Quat(angle: rng.nextFloat() * .pi,
                         axis: safeNormalize(F3(rng.nextFloat() - 0.5,
                                                rng.nextFloat() - 0.5,
                                                0.4),
                                             fallback: F3(0, 0, 1)))
            if i % 3 == 0 {
                _ = s.addSphere(diameter: 0.34, density: 3.8, friction: friction,
                                position: F3(x, y, z))
            } else {
                let sx = 0.28 + rng.nextFloat() * 0.16
                let sy = 0.24 + rng.nextFloat() * 0.14
                let sz = 0.22 + rng.nextFloat() * 0.16
                _ = s.addBody(size: F3(sx, sy, sz), density: 5.0,
                              friction: friction,
                              position: F3(x, y, z),
                              rotation: q)
            }
        }

        s.settings.cameraDistance = 8.5 + Float(k).squareRoot() * 1.2
        s.settings.cameraTargetZ = pinZ + 0.6
        return s
    }

    private static func addSoftVoxelCells(_ s: inout PhysicsScene,
                                          cells: Set<SIMD3<Int>>,
                                          origin: F3,
                                          spacing: Float,
                                          mu: Float,
                                          lambda: Float,
                                          massPerNode: Float,
                                          friction: Float) -> [Int] {
        var nodeId: [SIMD3<Int>: Int] = [:]
        var flat: [Int] = []
        func node(_ i: Int, _ j: Int, _ k: Int) -> Int {
            let key = SIMD3(i, j, k)
            if let id = nodeId[key] { return id }
            let p = origin + F3(Float(i), Float(j), Float(k)) * spacing
            let id = s.addParticle(radius: spacing * 0.32, mass: massPerNode,
                                   friction: friction, position: p)
            nodeId[key] = id
            flat.append(id)
            return id
        }
        func tet(_ a: Int, _ b: Int, _ c: Int, _ d: Int) {
            s.addTet(SceneTet(ids: (a, b, c, d), mu: mu, lambda: lambda))
        }
        for c in cells.sorted(by: { ($0.x, $0.y, $0.z) < ($1.x, $1.y, $1.z) }) {
            let i = c.x, j = c.y, k = c.z
            let c000 = node(i, j, k),         c100 = node(i + 1, j, k)
            let c010 = node(i, j + 1, k),     c110 = node(i + 1, j + 1, k)
            let c001 = node(i, j, k + 1),     c101 = node(i + 1, j, k + 1)
            let c011 = node(i, j + 1, k + 1), c111 = node(i + 1, j + 1, k + 1)
            if (i + j + k) % 2 == 0 {
                tet(c000, c100, c010, c001)
                tet(c110, c100, c010, c111)
                tet(c101, c100, c001, c111)
                tet(c011, c010, c001, c111)
                tet(c100, c010, c001, c111)
            } else {
                tet(c100, c000, c110, c101)
                tet(c010, c000, c110, c011)
                tet(c001, c000, c101, c011)
                tet(c111, c110, c101, c011)
                tet(c000, c110, c101, c011)
            }
        }
        for (key, a) in nodeId {
            for d in [SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1),
                      SIMD3(1, 1, 0), SIMD3(1, 0, 1), SIMD3(0, 1, 1),
                      SIMD3(1, 1, 1)] {
                if let b = nodeId[key &+ d] {
                    s.addJoint(SceneJoint(bodyA: a, bodyB: b, rA: .zero, rB: .zero,
                                          stiffnessLin: 0, stiffnessAng: 0))
                }
            }
        }
        return flat
    }

    private static func expandedCells(_ cells: Set<SIMD3<Int>>,
                                      targetCount: Int) -> Set<SIMD3<Int>> {
        var out = cells
        guard !out.isEmpty, out.count < targetCount else { return out }
        let dirs = [
            SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0),
            SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)
        ]
        var frontier = cells.sorted { ($0.x, $0.y, $0.z) < ($1.x, $1.y, $1.z) }
        while !frontier.isEmpty && out.count < targetCount {
            var next: [SIMD3<Int>] = []
            for c in frontier {
                for d in dirs {
                    let n = c &+ d
                    if out.insert(n).inserted {
                        next.append(n)
                        if out.count >= targetCount { return out }
                    }
                }
            }
            frontier = next.sorted { ($0.x, $0.y, $0.z) < ($1.x, $1.y, $1.z) }
        }
        return out
    }

    private static func addOpenBox(_ s: inout PhysicsScene,
                                   halfX: Float, halfY: Float,
                                   wallHeight: Float, thickness: Float,
                                   friction: Float) {
        _ = s.addBody(size: F3(2 * halfX + 2 * thickness,
                               2 * halfY + 2 * thickness, 0.10),
                      density: 0, friction: friction,
                      position: F3(0, 0, 0.05))
        let z = 0.05 + wallHeight * 0.5
        _ = s.addBody(size: F3(thickness, 2 * halfY + 2 * thickness, wallHeight),
                      density: 0, friction: friction,
                      position: F3(-halfX - thickness * 0.5, 0, z))
        _ = s.addBody(size: F3(thickness, 2 * halfY + 2 * thickness, wallHeight),
                      density: 0, friction: friction,
                      position: F3(halfX + thickness * 0.5, 0, z))
        _ = s.addBody(size: F3(2 * halfX, thickness, wallHeight),
                      density: 0, friction: friction,
                      position: F3(0, -halfY - thickness * 0.5, z))
        _ = s.addBody(size: F3(2 * halfX, thickness, wallHeight),
                      density: 0, friction: friction,
                      position: F3(0, halfY + thickness * 0.5, z))
    }

    private struct TetBindRecord {
        var ids: (Int, Int, Int, Int)
        var x0: F3
        var x1: F3
        var x2: F3
        var x3: F3
        var inv: simd_float3x3
        var restInv0: F3
        var restInv1: F3
        var restInv2: F3
        var lo: F3
        var hi: F3

        init?(tet: SceneTet, scene: PhysicsScene) {
            ids = tet.ids
            x0 = scene.bodies[tet.ids.0].position
            x1 = scene.bodies[tet.ids.1].position
            x2 = scene.bodies[tet.ids.2].position
            x3 = scene.bodies[tet.ids.3].position
            let d0 = x1 - x0
            let d1 = x2 - x0
            let d2 = x3 - x0
            let dm = simd_float3x3(columns: (d0, d1, d2))
            guard abs(dm.determinant) > 1e-12 else { return nil }
            inv = dm.inverse
            restInv0 = F3(inv.columns.0.x, inv.columns.1.x, inv.columns.2.x)
            restInv1 = F3(inv.columns.0.y, inv.columns.1.y, inv.columns.2.y)
            restInv2 = F3(inv.columns.0.z, inv.columns.1.z, inv.columns.2.z)
            lo = min(min(x0, x1), min(x2, x3))
            hi = max(max(x0, x1), max(x2, x3))
        }

        func barycentric(_ p: F3) -> SIMD4<Float> {
            let u = inv * (p - x0)
            return SIMD4(1 - u.x - u.y - u.z, u.x, u.y, u.z)
        }
    }

    private struct TetBindIndex {
        var records: [TetBindRecord]
        var origin: F3 = .zero
        var cellSize: Float = 1
        var dims = SIMD3<Int>(1, 1, 1)
        var cells: [[Int]] = [[]]

        init(tets: [SceneTet], scene: PhysicsScene) {
            records = tets.compactMap { TetBindRecord(tet: $0, scene: scene) }
            guard !records.isEmpty else { return }

            var lo = F3(repeating: Float.greatestFiniteMagnitude)
            var hi = F3(repeating: -Float.greatestFiniteMagnitude)
            var edgeSum: Float = 0
            for r in records {
                lo = min(lo, r.lo)
                hi = max(hi, r.hi)
                edgeSum += max(max(r.hi.x - r.lo.x, r.hi.y - r.lo.y), r.hi.z - r.lo.z)
            }
            cellSize = max(1e-5, 1.25 * edgeSum / Float(records.count))
            origin = lo - F3(repeating: cellSize)
            let extent = hi - origin + F3(repeating: cellSize)
            dims = SIMD3(max(1, Int(ceil(extent.x / cellSize))),
                         max(1, Int(ceil(extent.y / cellSize))),
                         max(1, Int(ceil(extent.z / cellSize))))
            cells = Array(repeating: [], count: dims.x * dims.y * dims.z)
            for (ri, r) in records.enumerated() {
                let mn = coord(r.lo - F3(repeating: 1e-5))
                let mx = coord(r.hi + F3(repeating: 1e-5))
                for i in mn.x...mx.x {
                    for j in mn.y...mx.y {
                        for k in mn.z...mx.z {
                            cells[index(i, j, k)].append(ri)
                        }
                    }
                }
            }
        }

        func coord(_ p: F3) -> SIMD3<Int> {
            let q = (p - origin) / cellSize
            return SIMD3(max(0, min(dims.x - 1, Int(floor(q.x)))),
                         max(0, min(dims.y - 1, Int(floor(q.y)))),
                         max(0, min(dims.z - 1, Int(floor(q.z)))))
        }

        func index(_ i: Int, _ j: Int, _ k: Int) -> Int {
            (k * dims.y + j) * dims.x + i
        }

        func candidates(near p: F3, maxRing: Int) -> [Int] {
            guard !records.isEmpty else { return [] }
            let c = coord(p)
            var seen = Set<Int>()
            var out: [Int] = []
            for r in 0...max(0, maxRing) {
                let mn = SIMD3(max(0, c.x - r), max(0, c.y - r),
                               max(0, c.z - r))
                let mx = SIMD3(min(dims.x - 1, c.x + r),
                               min(dims.y - 1, c.y + r),
                               min(dims.z - 1, c.z + r))
                for i in mn.x...mx.x {
                    for j in mn.y...mx.y {
                        for k in mn.z...mx.z {
                            for ti in cells[index(i, j, k)] where seen.insert(ti).inserted {
                                out.append(ti)
                            }
                        }
                    }
                }
                if !out.isEmpty { break }
            }
            return out
        }
    }

    private static func bindVisualMesh(_ mesh: SurfaceMesh,
                                       scene: PhysicsScene,
                                       tetRange: Range<Int>,
                                       bodyIDs: [Int]) -> SceneSkinnedMesh {
        let index = TetBindIndex(tets: tetRange.map { scene.tets[$0] },
                                 scene: scene)
        var vertices: [SceneSkinnedVertex] = []
        vertices.reserveCapacity(mesh.vertices.count)
        for (vi, p) in mesh.vertices.enumerated() {
            var best: (record: TetBindRecord, w: SIMD4<Float>, inside: Bool,
                       dist2: Float, outside: Float)?
            let candidates = index.candidates(near: p, maxRing: 3)
            let search = candidates.isEmpty ? index.records.indices.map { $0 } : candidates
            for ti in search {
                let record = index.records[ti]
                let w = record.barycentric(p)
                let minW = min(min(w.x, w.y), min(w.z, w.w))
                let outside = max(0, -minW)
                let inside = outside <= 1e-4
                let wc = clampWeights(w)
                let q = record.x0 * wc.x + record.x1 * wc.y
                    + record.x2 * wc.z + record.x3 * wc.w
                let d2 = length_squared(q - p)
                if best == nil
                    || (inside && !best!.inside)
                    || (inside == best!.inside && d2 < best!.dist2)
                    || (inside == best!.inside && d2 == best!.dist2
                        && outside < best!.outside) {
                    best = (record, w, inside, d2, outside)
                }
            }
            if let best {
                vertices.append(SceneSkinnedVertex(ids: best.record.ids,
                                                   weights: best.w,
                                                   restNormal: mesh.normals[vi],
                                                   restInv0: best.record.restInv0,
                                                   restInv1: best.record.restInv1,
                                                   restInv2: best.record.restInv2))
            } else if let first = index.records.first {
                vertices.append(SceneSkinnedVertex(ids: first.ids,
                                                   weights: SIMD4(0.25, 0.25, 0.25, 0.25),
                                                   restNormal: mesh.normals[vi],
                                                   restInv0: first.restInv0,
                                                   restInv1: first.restInv1,
                                                   restInv2: first.restInv2))
            }
        }
        return SceneSkinnedMesh(vertices: vertices, triangles: mesh.triangles,
                                bodyIDs: bodyIDs)
    }

    private static func proceduralBunnySurface(res: Int) -> SurfaceMesh {
        func ellipsoid(_ p: F3, _ c: F3, _ r: F3) -> Bool {
            let d = (p - c) / r
            return dot(d, d) <= 1
        }
        func bunny(_ p: F3) -> Bool {
            ellipsoid(p, F3(0.00, 0, 0.52), F3(0.62, 0.46, 0.44)) ||
            ellipsoid(p, F3(0.52, 0, 1.00), F3(0.30, 0.26, 0.27)) ||
            ellipsoid(p, F3(0.46, 0.14, 1.42), F3(0.11, 0.08, 0.34)) ||
            ellipsoid(p, F3(0.46, -0.14, 1.42), F3(0.11, 0.08, 0.34)) ||
            ellipsoid(p, F3(-0.60, 0, 0.42), F3(0.16, 0.16, 0.16)) ||
            ellipsoid(p, F3(0.30, 0.26, 0.16), F3(0.30, 0.13, 0.14)) ||
            ellipsoid(p, F3(0.30, -0.26, 0.16), F3(0.30, 0.13, 0.14))
        }
        let h = 1.8 / Float(max(8, res))
        let origin = F3(-1.0, -0.7, 0.0)
        let nx = Int(2.0 / h), ny = Int(1.4 / h), nz = Int(1.9 / h)
        var occ = Set<SIMD3<Int>>()
        for i in 0..<nx {
            for j in 0..<ny {
                for k in 0..<nz {
                    let p = origin + (F3(Float(i), Float(j), Float(k))
                                      + F3(repeating: 0.5)) * h
                    if bunny(p) { occ.insert(SIMD3(i, j, k)) }
                }
            }
        }
        occ = largestComponent(occ)
        var verts: [F3] = []
        var vmap: [SIMD3<Int>: Int] = [:]
        var tris: [(Int, Int, Int)] = []
        func v(_ q: SIMD3<Int>) -> Int {
            if let id = vmap[q] { return id }
            let p = origin + F3(Float(q.x), Float(q.y), Float(q.z)) * h
            let id = verts.count
            vmap[q] = id
            verts.append(p)
            return id
        }
        let faces: [(SIMD3<Int>, [SIMD3<Int>])] = [
            (SIMD3(-1, 0, 0), [SIMD3(0,0,0), SIMD3(0,0,1), SIMD3(0,1,1), SIMD3(0,1,0)]),
            (SIMD3(1, 0, 0), [SIMD3(1,0,0), SIMD3(1,1,0), SIMD3(1,1,1), SIMD3(1,0,1)]),
            (SIMD3(0, -1, 0), [SIMD3(0,0,0), SIMD3(1,0,0), SIMD3(1,0,1), SIMD3(0,0,1)]),
            (SIMD3(0, 1, 0), [SIMD3(0,1,0), SIMD3(0,1,1), SIMD3(1,1,1), SIMD3(1,1,0)]),
            (SIMD3(0, 0, -1), [SIMD3(0,0,0), SIMD3(0,1,0), SIMD3(1,1,0), SIMD3(1,0,0)]),
            (SIMD3(0, 0, 1), [SIMD3(0,0,1), SIMD3(1,0,1), SIMD3(1,1,1), SIMD3(0,1,1)])
        ]
        for c in occ {
            for (d, corners) in faces where !occ.contains(c &+ d) {
                let ids = corners.map { v(c &+ $0) }
                tris.append((ids[0], ids[1], ids[2]))
                tris.append((ids[0], ids[2], ids[3]))
            }
        }
        return SurfaceMesh(vertices: verts, normals: [], triangles: tris)
    }

    private static func loadDefaultSkinMesh(meshPath: String?) -> SurfaceMesh {
        loadDefaultSkinMeshWithKey(meshPath: meshPath).0
    }

    private static func loadDefaultSkinMeshWithKey(meshPath: String?) -> (SurfaceMesh, String) {
        let envPath = ProcessInfo.processInfo.environment["AVBD_SKIN_MESH"]
        let path = meshPath ?? envPath
        if let path {
            let key = skinMeshFileKey(path)
            let mesh = SkinnedSoftMeshCache.shared.mesh(for: key) {
                (try? SurfaceMesh.load(path: path)) ?? proceduralBunnySurface(res: 22)
            }
            return (mesh, key)
        }
        let key = "procedural-bunny:22"
        let mesh = SkinnedSoftMeshCache.shared.mesh(for: key) {
            proceduralBunnySurface(res: 22)
        }
        return (mesh, key)
    }

    private static func skinMeshFileKey(_ path: String) -> String {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let attrs = (try? FileManager.default.attributesOfItem(atPath: url.path)) ?? [:]
        let size = (attrs[.size] as? NSNumber)?.uint64Value ?? 0
        let modified = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return String(format: "file:%@:%llu:%.6f", url.path, size, modified)
    }
}

private func safeNormalize(_ v: F3, fallback: F3 = .zero) -> F3 {
    let l2 = length_squared(v)
    return l2 > 1e-20 ? v * rsqrt(l2) : fallback
}

private func largestComponent(_ occupied: Set<SIMD3<Int>>) -> Set<SIMD3<Int>> {
    var unvisited = occupied
    var keep = Set<SIMD3<Int>>()
    while let seed = unvisited.first {
        var comp = Set<SIMD3<Int>>()
        var stack = [seed]
        unvisited.remove(seed)
        while let c = stack.popLast() {
            comp.insert(c)
            for d in [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0),
                      SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)] {
                let n = c &+ d
                if unvisited.remove(n) != nil { stack.append(n) }
            }
        }
        if comp.count > keep.count { keep = comp }
    }
    return keep
}

private func barycentric(_ p: F3, tet: SceneTet, scene: PhysicsScene) -> SIMD4<Float>? {
    let x0 = scene.bodies[tet.ids.0].position
    let d0 = scene.bodies[tet.ids.1].position - x0
    let d1 = scene.bodies[tet.ids.2].position - x0
    let d2 = scene.bodies[tet.ids.3].position - x0
    let dm = simd_float3x3(columns: (d0, d1, d2))
    guard abs(dm.determinant) > 1e-12 else { return nil }
    let u = dm.inverse * (p - x0)
    return SIMD4(1 - u.x - u.y - u.z, u.x, u.y, u.z)
}

private func clampWeights(_ w: SIMD4<Float>) -> SIMD4<Float> {
    var c = max(w, SIMD4<Float>(repeating: 0))
    let s = c.x + c.y + c.z + c.w
    if s <= 1e-12 { return SIMD4(0.25, 0.25, 0.25, 0.25) }
    c /= s
    return c
}
