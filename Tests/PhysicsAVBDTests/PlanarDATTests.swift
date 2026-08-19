import XCTest
import simd
@testable import SimCore
@testable import PhysicsAVBD

final class PlanarDATTests: XCTestCase {
    private let identity = Quat(real: 1, imag: .zero)

    private struct PairKey: Hashable {
        let kind: UInt32
        let owner: UInt32
        let other: UInt32
        let flags: UInt32

        init(kind: UInt32, owner: UInt32, other: UInt32,
             flags: UInt32 = 3) {
            self.kind = kind
            self.owner = owner
            self.other = other
            self.flags = flags
        }

        init(_ raw: SIMD4<UInt32>) {
            self.init(kind: raw.x, owner: raw.y, other: raw.z,
                      flags: raw.w)
        }
    }

    @discardableResult
    private func addTriangle(
        _ scene: inout PhysicsScene,
        _ a: F3, _ b: F3, _ c: F3,
        radius: Float = 0.01,
        mass: Float,
        velocity: F3 = .zero
    ) -> (triangle: Int, vertices: [Int]) {
        let vertices = [a, b, c].map {
            scene.addParticle(
                radius: radius, mass: mass, friction: 0,
                position: $0, velocity: velocity)
        }
        let triangle = scene.tris.count
        scene.addTri(SceneTri(ids: (vertices[0], vertices[1], vertices[2])))
        return (triangle, vertices)
    }

    /// A large static target containing a much smaller moving triangle in
    /// projection. The target vertices stay outside the small triangle's DAT
    /// query radius, which isolates the moving V-T events analytically.
    private func twoTriangleScene(
        sourceHeight: Float,
        sourceVelocity: F3,
        iterations: Int = 0
    ) -> (scene: PhysicsScene, target: Int, source: [Int]) {
        var scene = PhysicsScene(name: "planar-dat-two-triangle")
        scene.settings.dt = 1
        scene.settings.gravity = 0
        scene.settings.iterations = iterations
        let target = addTriangle(
            &scene,
            F3(-1, -1, 0), F3(1, -1, 0), F3(0, 1, 0),
            mass: 0)
        let source = addTriangle(
            &scene,
            F3(-0.15, -0.10, sourceHeight),
            F3(0.15, -0.10, sourceHeight),
            F3(0, 0.15, sourceHeight),
            mass: 1, velocity: sourceVelocity)
        return (scene, target.triangle, source.vertices)
    }

    private func sharedPositions(_ solver: GPUSolver) -> UnsafeMutablePointer<SIMD4<Float>> {
        solver.posLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)
    }

    private func planarPairCount(_ solver: GPUSolver) -> Int {
        Int(solver.planarDATPairCountsBuf.contents().bindMemory(
            to: UInt32.self, capacity: 3)[0])
    }

    private func planarPairs(_ solver: GPUSolver) -> [SIMD4<UInt32>] {
        let count = min(planarPairCount(solver), solver.maxPlanarDATPairs)
        guard count > 0 else { return [] }
        let pointer = solver.planarDATPairsBuf.contents().bindMemory(
            to: SIMD4<UInt32>.self, capacity: solver.maxPlanarDATPairs)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func position(
        _ pointer: UnsafeMutablePointer<SIMD4<Float>>, _ index: UInt32
    ) -> F3 {
        let p = pointer[Int(index)]
        return F3(p.x, p.y, p.z)
    }

    private func pointTriangleDistanceSquared(
        _ p: F3, _ a: F3, _ b: F3, _ c: F3
    ) -> Float {
        let ab = b - a, ac = c - a, ap = p - a
        let d1 = dot(ab, ap), d2 = dot(ac, ap)
        if d1 <= 0 && d2 <= 0 { return length_squared(ap) }
        let bp = p - b
        let d3 = dot(ab, bp), d4 = dot(ac, bp)
        if d3 >= 0 && d4 <= d3 { return length_squared(bp) }
        let vc = d1 * d4 - d3 * d2
        if vc <= 0 && d1 >= 0 && d3 <= 0 {
            let v = d1 / (d1 - d3)
            return length_squared(p - (a + ab * v))
        }
        let cp = p - c
        let d5 = dot(ab, cp), d6 = dot(ac, cp)
        if d6 >= 0 && d5 <= d6 { return length_squared(cp) }
        let vb = d5 * d2 - d1 * d6
        if vb <= 0 && d2 >= 0 && d6 <= 0 {
            let w = d2 / (d2 - d6)
            return length_squared(p - (a + ac * w))
        }
        let va = d3 * d6 - d5 * d4
        if va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return length_squared(p - (b + (c - b) * w))
        }
        let denom = 1 / (va + vb + vc)
        let q = a + ab * (vb * denom) + ac * (vc * denom)
        return length_squared(p - q)
    }

    private func segmentSegmentDistanceSquared(
        _ p0: F3, _ p1: F3, _ q0: F3, _ q1: F3
    ) -> Float {
        let d0 = p1 - p0, d1 = q1 - q0, r = p0 - q0
        let a = dot(d0, d0), e = dot(d1, d1)
        let epsilon: Float = 1e-12
        var s: Float = 0
        var t: Float = 0
        if a <= epsilon && e <= epsilon {
            return length_squared(r)
        }
        if a <= epsilon {
            t = min(max(dot(d1, r) / e, 0), 1)
        } else {
            let c = dot(d0, r)
            if e <= epsilon {
                s = min(max(-c / a, 0), 1)
            } else {
                let b = dot(d0, d1)
                let f = dot(d1, r)
                let denominator = a * e - b * b
                if denominator != 0 {
                    s = min(max((b * f - c * e) / denominator, 0), 1)
                }
                let projectedT = (b * s + f) / e
                if projectedT < 0 {
                    t = 0
                    s = min(max(-c / a, 0), 1)
                } else if projectedT > 1 {
                    t = 1
                    s = min(max((b - c) / a, 0), 1)
                } else {
                    t = projectedT
                }
            }
        }
        return length_squared((p0 + d0 * s) - (q0 + d1 * t))
    }

    /// Brute-force accepted-pose oracle, independent of the GPU hash grid.
    /// It mirrors only the public Planar-DAT pair contract: exact rq distance,
    /// canonical E-E ownership, and one-edge-ring topology filtering.
    private func acceptedPairOracle(_ solver: GPUSolver) -> Set<PairKey> {
        let positions = sharedPositions(solver)
        let triangles = solver.trisBuf.contents().bindMemory(
            to: SIMD4<UInt32>.self, capacity: max(1, solver.numTris))
        let edges = solver.edgesBuf.contents().bindMemory(
            to: SIMD2<UInt32>.self,
            capacity: max(1, solver.numPlanarDATEdges))
        let particles = solver.particleIdxBuf.contents().bindMemory(
            to: UInt32.self, capacity: max(1, solver.numParticles))
        var oneRing = [Set<UInt32>](repeating: [], count: solver.numBodies)
        for ti in 0..<solver.numTris {
            let t = triangles[ti]
            for (u, v) in [(t.x, t.y), (t.y, t.z), (t.x, t.z)] {
                oneRing[Int(u)].insert(v)
                oneRing[Int(v)].insert(u)
            }
        }

        let rq2 = solver.params.planarDATQueryRadius
            * solver.params.planarDATQueryRadius
        var expected = Set<PairKey>()
        for pi in 0..<solver.numParticles {
            let v = particles[pi]
            let p = position(positions, v)
            for ti in 0..<solver.numTris {
                let t = triangles[ti]
                if t.x == v || t.y == v || t.z == v { continue }
                if oneRing[Int(v)].contains(t.x)
                    || oneRing[Int(v)].contains(t.y)
                    || oneRing[Int(v)].contains(t.z) { continue }
                let distance2 = pointTriangleDistanceSquared(
                    p, position(positions, t.x), position(positions, t.y),
                    position(positions, t.z))
                if distance2 <= rq2 {
                    expected.insert(PairKey(
                        kind: 0, owner: v, other: UInt32(ti)))
                }
            }
        }

        for first in 0..<solver.numPlanarDATEdges {
            let a = edges[first]
            for second in (first + 1)..<solver.numPlanarDATEdges {
                let b = edges[second]
                if a.x == b.x || a.x == b.y || a.y == b.x || a.y == b.y {
                    continue
                }
                if oneRing[Int(a.x)].contains(b.x)
                    || oneRing[Int(a.x)].contains(b.y)
                    || oneRing[Int(a.y)].contains(b.x)
                    || oneRing[Int(a.y)].contains(b.y) { continue }
                let distance2 = segmentSegmentDistanceSquared(
                    position(positions, a.x), position(positions, a.y),
                    position(positions, b.x), position(positions, b.y))
                if distance2 <= rq2 {
                    expected.insert(PairKey(
                        kind: 1, owner: UInt32(first),
                        other: UInt32(second)))
                }
            }
        }
        return expected
    }

    private func edgeIndex(
        _ solver: GPUSolver, _ first: Int, _ second: Int
    ) -> UInt32? {
        let lo = UInt32(min(first, second)), hi = UInt32(max(first, second))
        let edges = solver.edgesBuf.contents().bindMemory(
            to: SIMD2<UInt32>.self,
            capacity: max(1, solver.numPlanarDATEdges))
        for index in 0..<solver.numPlanarDATEdges {
            let edge = edges[index]
            if edge.x == lo && edge.y == hi { return UInt32(index) }
        }
        return nil
    }

    func testVertexTrianglePredictorMatchesAnalyticDivisionPlane() throws {
        let startZ: Float = 0.2
        let velocityZ: Float = -0.4
        let fixture = twoTriangleScene(
            sourceHeight: startZ,
            sourceVelocity: F3(0, 0, velocityZ))
        let solver = try GPUSolver(scene: fixture.scene)

        // The static target gets the minimum 5% allocation of the initial
        // gap. The moving vertex reaches that division plane at hit=0.475;
        // Planar-DAT applies gamma*hit because it is stricter than hit-0.001.
        let planeZ = startZ * 0.05
        let hit = (startZ - planeZ) / -velocityZ
        let t = min(
            solver.params.planarDATRelaxation * hit,
            hit - 0.001)
        let expectedZ = startZ + velocityZ * t
        let globalCap = 0.5 * solver.params.planarDATRelaxation
            * solver.params.planarDATQueryRadius
        XCTAssertLessThan(abs(velocityZ * t), globalCap,
                          "the analytic plane, not the global rq cap, must bind")

        try solver.submitStep()
        try solver.synchronize()

        let positions = sharedPositions(solver)
        for vertex in fixture.source {
            XCTAssertEqual(positions[vertex].z, expectedZ, accuracy: 2e-5)
        }
        XCTAssertGreaterThanOrEqual(solver.lastPlanarDATVertexTrianglePairs, 3)
        XCTAssertGreaterThanOrEqual(solver.lastPlanarDATTruncations, 3)
    }

    func testSameConnectedSolidVertexTrianglePairsAreSafetyOnly() throws {
        let fixture = twoTriangleScene(
            sourceHeight: 0.03, sourceVelocity: .zero)
        let solver = try GPUSolver(scene: fixture.scene)

        // White-box the force-eligibility boundary: the geometric pair and
        // topology stay unchanged, but both surfaces now represent one
        // connected volumetric solid rather than two cloth components.
        let groups = solver.clothGroupBuf.contents().bindMemory(
            to: UInt32.self, capacity: solver.numBodies)
        let cloth = solver.clothVertFlag.contents().bindMemory(
            to: UInt32.self, capacity: solver.numBodies)
        let target = fixture.scene.tris[fixture.target].ids
        for vertex in fixture.source + [target.0, target.1, target.2] {
            groups[vertex] = 7
            cloth[vertex] = 0
        }

        try solver.submitStep()
        try solver.synchronize()
        let pairs = planarPairs(solver).filter { $0.x == 0 }
        XCTAssertFalse(
            pairs.isEmpty,
            "expected same-solid V-T safety pairs; peak=\(solver.lastPlanarDATPairs) "
                + "VT=\(solver.lastPlanarDATVertexTrianglePairs)")
        XCTAssertTrue(pairs.allSatisfy { $0.w == 1 },
                      "same-solid V-T pairs must truncate without OGC force")
    }

    func testFullPairStreamKeepsSixVertexTrianglePairsForOneVertex() throws {
        var scene = PhysicsScene(name: "planar-dat-full-vt-stream")
        scene.settings.gravity = 0
        scene.settings.iterations = 0

        let targetCount = 6
        for layer in 1...targetCount {
            let z = Float(layer) * 0.1
            addTriangle(
                &scene,
                F3(-1, -1, z), F3(1, -1, z), F3(0, 1, z),
                mass: 0)
        }
        let source = addTriangle(
            &scene,
            F3(-0.15, -0.10, 0),
            F3(0.15, -0.10, 0),
            F3(0, 0.15, 0),
            mass: 1)
        let owner = source.vertices[2]
        let solver = try GPUSolver(scene: scene)

        try solver.submitStep()
        try solver.synchronize()

        // data = (kind, owner vertex/edge, triangle/edge, flags), with kind 0
        // denoting V-T. Count this owner's six authored targets in the full
        // conservative stream, not merely a global aggregate counter.
        let ownedTargets = planarPairs(solver).filter {
            $0.x == 0 && $0.y == UInt32(owner) && $0.z < UInt32(targetCount)
        }
        XCTAssertEqual(ownedTargets.count, targetCount)
        XCTAssertEqual(Set(ownedTargets.map(\.z)).count, targetCount,
                       "the compact stream must neither cap at four nor duplicate a target")
        XCTAssertGreaterThan(solver.lastPlanarDATVertexTrianglePairs, 4)
        XCTAssertEqual(solver.lastPlanarDATTruncations, 0)
    }

    func testAcceptedPairSetMatchesExactCPUOracle() throws {
        var scene = PhysicsScene(name: "planar-dat-pair-oracle")
        scene.settings.dt = 1
        scene.settings.gravity = 0
        scene.settings.iterations = 0

        // Two triangles form one connected square at z=0. A disconnected
        // layer at z=.06 must produce V-T and E-E candidates, while the
        // geometrically identical layer at z=.30 is beyond rq.
        let lowerA = addTriangle(
            &scene,
            F3(-0.05, -0.05, 0), F3(0.05, -0.05, 0),
            F3(-0.05, 0.05, 0), radius: 0.005, mass: 1)
        let lowerD = scene.addParticle(
            radius: 0.005, mass: 1, friction: 0,
            position: F3(0.05, 0.05, 0), velocity: .zero)
        let lowerBTriangle = scene.tris.count
        let lowerBVertices = [lowerA.vertices[1], lowerD, lowerA.vertices[2]]
        scene.addTri(SceneTri(ids: (
            lowerBVertices[0], lowerBVertices[1], lowerBVertices[2])))
        let lowerB = (triangle: lowerBTriangle, vertices: lowerBVertices)
        let nearLayer = addTriangle(
            &scene,
            F3(-0.05, -0.05, 0.06), F3(0.05, -0.05, 0.06),
            F3(-0.05, 0.05, 0.06), radius: 0.005, mass: 1)
        let farLayer = addTriangle(
            &scene,
            F3(-0.05, -0.05, 0.30), F3(0.05, -0.05, 0.30),
            F3(-0.05, 0.05, 0.30), radius: 0.005, mass: 1)
        let solver = try GPUSolver(scene: scene)
        solver.params.planarDATQueryRadius = 0.13

        try solver.submitStep()
        try solver.synchronize()

        let raw = planarPairs(solver)
        let actualList = raw.map(PairKey.init)
        let actual = Set(actualList)
        let expected = acceptedPairOracle(solver)
        XCTAssertEqual(actualList.count, actual.count,
                       "the final compact pair stream must contain no duplicates")
        XCTAssertEqual(actual, expected,
                       "the GPU grid must be exactly equivalent to the brute-force rq query")
        XCTAssertTrue(expected.contains { $0.kind == 0 },
                      "the fixture must exercise retained V-T pairs")
        XCTAssertTrue(expected.contains { $0.kind == 1 },
                      "the fixture must exercise retained E-E pairs")

        // lowerA.v0 is within rq of lowerB, but lowerB contains direct
        // one-ring neighbors from the connected square, so V-T is excluded.
        let excludedVT = PairKey(
            kind: 0, owner: UInt32(lowerA.vertices[0]),
            other: UInt32(lowerB.triangle))
        XCTAssertLessThan(
            pointTriangleDistanceSquared(
                scene.bodies[lowerA.vertices[0]].position,
                scene.bodies[lowerB.vertices[0]].position,
                scene.bodies[lowerB.vertices[1]].position,
                scene.bodies[lowerB.vertices[2]].position),
            solver.params.planarDATQueryRadius
                * solver.params.planarDATQueryRadius)
        XCTAssertFalse(actual.contains(excludedVT))

        // The opposite square edges are also geometrically within rq, but
        // their endpoints are connected by the one-ring mesh topology.
        let edgeAB = try XCTUnwrap(edgeIndex(
            solver, lowerA.vertices[0], lowerA.vertices[1]))
        let edgeDC = try XCTUnwrap(edgeIndex(
            solver, lowerB.vertices[1], lowerB.vertices[2]))
        let excludedEE = PairKey(
            kind: 1, owner: min(edgeAB, edgeDC),
            other: max(edgeAB, edgeDC))
        XCTAssertFalse(actual.contains(excludedEE))

        // A representative disconnected pair is accepted at .06, whereas
        // its otherwise identical .30 counterpart fails only the exact rq
        // distance predicate (not topology).
        XCTAssertTrue(actual.contains(PairKey(
            kind: 0, owner: UInt32(nearLayer.vertices[0]),
            other: UInt32(lowerA.triangle))))
        XCTAssertFalse(actual.contains(PairKey(
            kind: 0, owner: UInt32(farLayer.vertices[0]),
            other: UInt32(lowerA.triangle))))
    }

    func testAcceptedPoseQueryDropsPairThatMovedOutsideRadius() throws {
        var fixture = twoTriangleScene(
            sourceHeight: 0.5, sourceVelocity: .zero)
        fixture.scene.settings.iterations = 0
        let solver = try GPUSolver(scene: fixture.scene)
        let rq = solver.params.planarDATQueryRadius
        let startZ = 0.8 * rq
        let velocity = F3(0, 0, 2 * rq)
        solver.setBodyStates(fixture.source.map {
            .init(body: $0,
                  position: F3(fixture.scene.bodies[$0].position.x,
                               fixture.scene.bodies[$0].position.y,
                               startZ),
                  rotation: identity,
                  linearVelocity: velocity)
        })

        var passSites: [GPUSolver.PlanarDATPassSite] = []
        solver.planarDATPassObserverForTesting = { passSites.append($0) }
        try solver.submitStep()
        try solver.synchronize()

        // Moving away is safe, so only the unconditional rq cap applies:
        // 0.8rq + 0.425rq lies beyond rq. Peak telemetry proves the start
        // query saw the pair, while the final buffer proves the independent
        // accepted-pose query dropped it before the color solve.
        XCTAssertEqual(passSites, [.predictor])
        XCTAssertGreaterThan(solver.lastPlanarDATPairs, 0,
                             "the start query must see the nearby pair")
        XCTAssertEqual(planarPairCount(solver), 0,
                       "the accepted-pose query must drop a pair outside rq")

        let positions = sharedPositions(solver)
        let anchors = solver.ogcPrevBuf.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)
        for vertex in fixture.source {
            XCTAssertGreaterThan(positions[vertex].z, rq)
            XCTAssertEqual(anchors[vertex].x, positions[vertex].x, accuracy: 1e-6)
            XCTAssertEqual(anchors[vertex].y, positions[vertex].y, accuracy: 1e-6)
            XCTAssertEqual(anchors[vertex].z, positions[vertex].z, accuracy: 1e-6)
        }
    }

    func testAcceptedPoseQueryFindsEnteringContactWithCurrentMasses() throws {
        // Keep the authored edges small enough that rq is derived from the
        // physical contact reach rather than a large triangle's edge length.
        // This makes the accepted pose enter both rq and the OGC force band.
        var scene = PhysicsScene(name: "planar-dat-entering-contact")
        scene.settings.dt = 1
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        addTriangle(
            &scene,
            F3(-0.04, -0.04, 0), F3(0.04, -0.04, 0), F3(0, 0.04, 0),
            mass: 0)
        let source = addTriangle(
            &scene,
            F3(-0.01, -0.007, 0.5), F3(0.01, -0.007, 0.5),
            F3(0, 0.01, 0.5), mass: 1)
        let solver = try GPUSolver(scene: scene)
        let rq = solver.params.planarDATQueryRadius
        let startZ = 1.01 * rq
        let velocity = F3(0, 0, -2 * rq)
        solver.setBodyStates(source.vertices.map {
            .init(
                body: $0,
                position: F3(
                    scene.bodies[$0].position.x,
                    scene.bodies[$0].position.y,
                    startZ),
                rotation: identity,
                linearVelocity: velocity)
        })

        try solver.submitStep()
        try solver.synchronize()

        XCTAssertGreaterThan(startZ, rq)
        XCTAssertGreaterThanOrEqual(solver.lastNumSoft, 3,
                                    "a pair entering rq during prediction must emit at the accepted pose")
        let positions = sharedPositions(solver)
        for vertex in source.vertices {
            XCTAssertTrue(positions[vertex].z.isFinite)
            XCTAssertLessThan(positions[vertex].z, rq)
            XCTAssertEqual(
                positions[vertex].z,
                startZ - 0.5 * solver.params.planarDATRelaxation * rq,
                accuracy: 2e-5)
        }
    }

    func testAcceptedPoseVTContactStoresStartLinearizedC0() throws {
        var scene = PhysicsScene(name: "planar-dat-accepted-vt-c0")
        scene.settings.dt = 1
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        scene.settings.alpha = 0.99
        let target = addTriangle(
            &scene,
            F3(-0.04, -0.04, 0), F3(0.04, -0.04, 0),
            F3(0, 0.04, 0), mass: 0)
        let source = addTriangle(
            &scene,
            F3(-0.01, -0.007, 0.5), F3(0.01, -0.007, 0.5),
            F3(0, 0.01, 0.5), mass: 1)
        let solver = try GPUSolver(scene: scene)
        let rq = solver.params.planarDATQueryRadius
        let startZ = 1.01 * rq
        let velocity = F3(0, 0, -2 * rq)
        solver.setBodyStates(source.vertices.map {
            .init(
                body: $0,
                position: F3(
                    scene.bodies[$0].position.x,
                    scene.bodies[$0].position.y,
                    startZ),
                rotation: identity,
                linearVelocity: velocity)
        })

        try solver.submitStep()
        try solver.synchronize()

        let contacts = solver.prevSoftContacts.contents().bindMemory(
            to: SoftContactGPU.self, capacity: max(1, solver.maxSoft))
        let contactCount = min(solver.lastNumSoft, solver.maxSoft)
        let sourceVertex = UInt32(source.vertices[2])
        let targetIDs = target.vertices.map(UInt32.init)
        let contact = try XCTUnwrap((0..<contactCount).lazy.map {
            contacts[$0]
        }.first {
            let kind = ($0.anchorA.w.bitPattern >> 2) & 0x7
            return kind == 1 && $0.ids.x == sourceVertex
                && $0.ids.y == targetIDs[0]
                && $0.ids.z == targetIDs[1]
                && $0.ids.w == targetIDs[2]
        }, "the accepted-pose query must emit the selected V-T contact")

        let accepted = sharedPositions(solver)
        let starts = solver.initLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)
        let shape = solver.shape.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)
        let n = F3(contact.normal.x, contact.normal.y, contact.normal.z)
        var startRelative = F3.zero
        var acceptedRelative = F3.zero
        for slot in 0..<4 {
            let id = contact.ids[slot]
            let weight = contact.weights[slot]
            startRelative += weight * position(starts, id)
            acceptedRelative += weight * position(accepted, id)
        }
        let hSum = abs(shape[Int(contact.ids.x)].w) + max(
            abs(shape[Int(contact.ids.y)].w),
            max(abs(shape[Int(contact.ids.z)].w),
                abs(shape[Int(contact.ids.w)].w)))
        let expectedC0 = dot(n, startRelative) - hSum
            + solver.params.elemMargin
        let acceptedC = dot(n, acceptedRelative) - hSum
            + solver.params.elemMargin
        let predictorNormalDisplacement = dot(
            n, acceptedRelative - startRelative)

        XCTAssertEqual(solver.params.alpha, 0.99, accuracy: 1e-7)
        XCTAssertGreaterThan(abs(predictorNormalDisplacement), 1e-4,
                             "the fixture must exercise normal predictor motion")
        XCTAssertEqual(contact.C0.x,
                       acceptedC - predictorNormalDisplacement,
                       accuracy: 2e-5)
        XCTAssertEqual(contact.C0.x, expectedC0, accuracy: 2e-5,
                       "C0 must be back-linearized to the step-start pose")
        XCTAssertGreaterThan(abs(expectedC0), 1e-3)
        XCTAssertGreaterThan(
            abs(contact.C0.x - expectedC0 / (1 - solver.params.alpha)),
            10 * abs(expectedC0),
            "accepted-pose back-linearization must not divide C0 by 1-alpha")
    }

    func testNonfinitePredictionFreezesAtFiniteAnchorAndThrows() throws {
        let startZ: Float = 0.2
        let fixture = twoTriangleScene(
            sourceHeight: startZ, sourceVelocity: .zero)
        let solver = try GPUSolver(scene: fixture.scene)
        let velocities = solver.velLin.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: solver.numBodies)
        for vertex in fixture.source {
            velocities[vertex].z = .infinity
        }

        try solver.submitStep()
        XCTAssertThrowsError(try solver.synchronize()) { error in
            guard case let GPUSolver.RuntimeFailure.commandExecution(
                _, frame, _, domain, code, message) = error,
                  domain == GPUSolver.RuntimeFailure.planarDATFailureDomain,
                  code == GPUSolver.RuntimeFailure.planarDATInvalidAnchorCode else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(frame, 1)
            XCTAssertTrue(message.contains("Planar-DAT pairs"))
        }

        let positions = sharedPositions(solver)
        for vertex in fixture.source {
            XCTAssertTrue(positions[vertex].z.isFinite)
            XCTAssertEqual(positions[vertex].z, startZ, accuracy: 1e-6)
        }
    }

    func testPlanarDATRunsAfterEveryVBDColor() throws {
        let iterations = 2
        let fixture = twoTriangleScene(
            sourceHeight: 0.2, sourceVelocity: .zero,
            iterations: iterations)
        let solver = try GPUSolver(scene: fixture.scene)
        var observed: [GPUSolver.PlanarDATPassSite] = []
        solver.planarDATPassObserverForTesting = { observed.append($0) }

        try solver.submitStep()
        try solver.synchronize()

        var expected: [GPUSolver.PlanarDATPassSite] = [.predictor]
        for iteration in 0..<iterations {
            for color in 0..<solver.staticUsedColors {
                expected.append(.color(iteration: iteration, color: color))
            }
        }
        XCTAssertEqual(observed, expected)
    }

    func testPerColorDATRestoresAllPrimalBindings() throws {
        var scene = Demos.clothfold(res: 36)
        scene.settings.iterations = 10
        let solver = try GPUSolver(scene: scene)
        XCTAssertGreaterThanOrEqual(solver.staticUsedColors, 2)

        try solver.submitStep()
        try solver.synchronize()

        let (_, stretch) = solver.debugClothMetrics()
        XCTAssertLessThan(
            stretch, 0.01,
            "a DAT dispatch must not leave its counter buffer bound as springs for the next color")
    }

    func testBilateralVertexTrianglePlaneAllocatesRelativeMotion() throws {
        var scene = PhysicsScene(name: "planar-dat-bilateral-vt")
        scene.settings.dt = 1
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        let target = addTriangle(
            &scene,
            F3(-1, -1, 0), F3(1, -1, 0), F3(0, 1, 0),
            mass: 1, velocity: F3(0, 0, 0.1))
        let source = addTriangle(
            &scene,
            F3(-0.15, -0.10, 0.2),
            F3(0.15, -0.10, 0.2),
            F3(0, 0.15, 0.2),
            mass: 1, velocity: F3(0, 0, -0.3))
        let solver = try GPUSolver(scene: scene)

        // Relative approach is 0.4. The target receives 25% of the gap,
        // so both sides hit their shared plane at 0.5 and accept gamma*hit.
        let t: Float = 0.5 * solver.params.planarDATRelaxation
        try solver.submitStep()
        try solver.synchronize()

        let positions = sharedPositions(solver)
        for v in source.vertices {
            XCTAssertEqual(positions[v].z, 0.2 - 0.3 * t, accuracy: 2e-5)
        }
        for v in target.vertices {
            XCTAssertEqual(positions[v].z, 0.1 * t, accuracy: 2e-5)
        }
    }

    func testExactVertexTrianglePlaneHitIsStrictlyTruncated() throws {
        let fixture = twoTriangleScene(
            sourceHeight: 0.2,
            sourceVelocity: F3(0, 0, -0.19))
        let solver = try GPUSolver(scene: fixture.scene)

        // A static target receives the minimum 5% room, so -0.19 lands
        // exactly on its z=.01 division plane at t=1. The half-space is
        // strict: the accepted endpoint must be gamma-scaled, not left on it.
        let expectedZ: Float = 0.2 - 0.19 * solver.params.planarDATRelaxation
        try solver.submitStep()
        try solver.synchronize()

        let positions = sharedPositions(solver)
        for v in fixture.source {
            XCTAssertEqual(positions[v].z, expectedZ, accuracy: 2e-5)
            XCTAssertGreaterThan(positions[v].z, 0.01)
        }
    }

    func testStrictestSimultaneousDivisionPlaneWins() throws {
        var scene = PhysicsScene(name: "planar-dat-strictest-plane")
        scene.settings.dt = 1
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        addTriangle(
            &scene,
            F3(-1, -1, 0), F3(1, -1, 0), F3(0, 1, 0), mass: 0)
        addTriangle(
            &scene,
            F3(-1, -1, 0.1), F3(1, -1, 0.1), F3(0, 1, 0.1), mass: 0)
        let source = addTriangle(
            &scene,
            F3(-0.15, -0.10, 0.2),
            F3(0.15, -0.10, 0.2),
            F3(0, 0.15, 0.2),
            mass: 1, velocity: F3(0, 0, -0.4))
        let solver = try GPUSolver(scene: scene)

        let hit: Float = (0.2 - 0.105) / 0.4
        let t = min(solver.params.planarDATRelaxation * hit, hit - 0.001)
        let expectedZ: Float = 0.2 - 0.4 * t
        try solver.submitStep()
        try solver.synchronize()

        let positions = sharedPositions(solver)
        for v in source.vertices {
            XCTAssertEqual(positions[v].z, expectedZ, accuracy: 2e-5)
            XCTAssertGreaterThan(positions[v].z, 0.105)
        }
    }

    func testTangentialMotionPreservesMoreProgressThanIsotropicDAT() throws {
        func makeSolver(_ mode: GPUSolver.SurfaceTruncationMode) throws
            -> (GPUSolver, [Int], [F3]) {
            let fixture = twoTriangleScene(
                sourceHeight: 0.05,
                sourceVelocity: F3(0.1, 0, 0))
            let starts = fixture.source.map { fixture.scene.bodies[$0].position }
            let solver = try GPUSolver(scene: fixture.scene)
            solver.surfaceTruncationMode = mode
            return (solver, fixture.source, starts)
        }

        let (planar, planarIDs, starts) = try makeSolver(.planarDAT)
        let (isotropic, isotropicIDs, _) = try makeSolver(.isotropicDAT)
        try planar.submitStep()
        try planar.synchronize()
        try isotropic.submitStep()
        try isotropic.synchronize()

        let planarPositions = sharedPositions(planar)
        let isotropicPositions = sharedPositions(isotropic)
        for i in planarIDs.indices {
            let dxPlanar = planarPositions[planarIDs[i]].x - starts[i].x
            let dxIsotropic = isotropicPositions[isotropicIDs[i]].x - starts[i].x
            XCTAssertGreaterThan(dxPlanar, 0.095)
            XCTAssertLessThan(dxIsotropic, 0.05)
            XCTAssertGreaterThan(dxPlanar, 2 * dxIsotropic)
            XCTAssertEqual(planarPositions[planarIDs[i]].z, 0.05, accuracy: 1e-5)
            XCTAssertEqual(isotropicPositions[isotropicIDs[i]].z, 0.05, accuracy: 1e-5)
        }
    }

    func testVertexTriangleTruncationIsTranslationInvariant() throws {
        let offset = F3(10_000, -10_000, 10_000)
        var scene = PhysicsScene(name: "planar-dat-translated-vt")
        scene.settings.dt = 1
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        addTriangle(
            &scene,
            offset + F3(-1, -1, 0),
            offset + F3(1, -1, 0),
            offset + F3(0, 1, 0), mass: 0)
        let source = addTriangle(
            &scene,
            offset + F3(-0.15, -0.10, 0.2),
            offset + F3(0.15, -0.10, 0.2),
            offset + F3(0, 0.15, 0.2),
            mass: 1, velocity: F3(0, 0, -0.4))
        let solver = try GPUSolver(scene: scene)
        let hit: Float = 0.95 * 0.2 / 0.4
        let t = min(solver.params.planarDATRelaxation * hit, hit - 0.001)
        let expectedRelativeZ: Float = 0.2 - 0.4 * t

        try solver.submitStep()
        try solver.synchronize()

        let positions = sharedPositions(solver)
        for v in source.vertices {
            XCTAssertEqual(positions[v].z - offset.z,
                           expectedRelativeZ, accuracy: 0.002)
        }
    }

    func testEdgeEdgePredictorMatchesAnalyticDivisionPlane() throws {
        var scene = PhysicsScene(name: "planar-dat-ee")
        scene.settings.dt = 1
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        addTriangle(
            &scene,
            F3(0, -1, 0), F3(0, 1, 0), F3(0.02, -1, 0), mass: 0)
        let source = addTriangle(
            &scene,
            F3(-1, 0, 0.2), F3(1, 0, 0.2), F3(-1, 0.02, 0.2),
            mass: 1, velocity: F3(0, 0, -0.4))
        let solver = try GPUSolver(scene: scene)
        solver.params.planarDATQueryRadius = 0.5

        let hit: Float = 0.95 * 0.2 / 0.4
        let t = min(solver.params.planarDATRelaxation * hit, hit - 0.001)
        let expectedZ: Float = 0.2 - 0.4 * t
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertEqual(solver.lastPlanarDATVertexTrianglePairs, 0)
        XCTAssertGreaterThan(solver.lastPlanarDATEdgeEdgePairs, 0)
        let positions = sharedPositions(solver)
        XCTAssertEqual(positions[source.vertices[0]].z, expectedZ, accuracy: 2e-5)
        XCTAssertEqual(positions[source.vertices[1]].z, expectedZ, accuracy: 2e-5)
        XCTAssertGreaterThan(positions[source.vertices[0]].z, 0.01)
        XCTAssertGreaterThan(positions[source.vertices[1]].z, 0.01)
    }

    func testExactEdgeEdgePlaneHitIsStrictlyTruncated() throws {
        var scene = PhysicsScene(name: "planar-dat-exact-ee-hit")
        scene.settings.dt = 1
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        addTriangle(
            &scene,
            F3(0, -1, 0), F3(0, 1, 0), F3(0.02, -1, 0), mass: 0)
        let source = addTriangle(
            &scene,
            F3(-1, 0, 0.2), F3(1, 0, 0.2), F3(-1, 0.02, 0.2),
            mass: 1, velocity: F3(0, 0, -0.19))
        let solver = try GPUSolver(scene: scene)
        solver.params.planarDATQueryRadius = 0.5

        // The static edge receives the minimum 5% room, so the trial endpoint
        // lands exactly on z=.01. Planar-DAT uses strict half-spaces: equality
        // is truncated by gamma instead of being accepted as non-crossing.
        let expectedZ: Float = 0.2 - 0.19 * solver.params.planarDATRelaxation
        try solver.submitStep()
        try solver.synchronize()

        XCTAssertEqual(solver.lastPlanarDATVertexTrianglePairs, 0)
        XCTAssertGreaterThan(solver.lastPlanarDATEdgeEdgePairs, 0)
        let positions = sharedPositions(solver)
        XCTAssertEqual(positions[source.vertices[0]].z, expectedZ, accuracy: 2e-5)
        XCTAssertEqual(positions[source.vertices[1]].z, expectedZ, accuracy: 2e-5)
        XCTAssertGreaterThan(positions[source.vertices[0]].z, 0.01)
        XCTAssertGreaterThan(positions[source.vertices[1]].z, 0.01)
    }

    func testIntersectingEdgeAnchorFailsClosed() throws {
        var scene = PhysicsScene(name: "planar-dat-invalid-ee-anchor")
        scene.settings.dt = 1
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        addTriangle(
            &scene,
            F3(0, -1, 0), F3(0, 1, 0), F3(0.02, -1, 0), mass: 0)
        let source = addTriangle(
            &scene,
            F3(-1, 0, 0), F3(1, 0, 0), F3(-1, 0.02, 0),
            mass: 1)
        let solver = try GPUSolver(scene: scene)
        solver.params.planarDATQueryRadius = 0.3

        try solver.submitStep()
        XCTAssertThrowsError(try solver.synchronize()) { error in
            guard case let GPUSolver.RuntimeFailure.commandExecution(
                _, frame, _, domain, code, message) = error,
                  domain == GPUSolver.RuntimeFailure.planarDATFailureDomain,
                  code == GPUSolver.RuntimeFailure.planarDATInvalidAnchorCode else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(frame, 1)
            XCTAssertTrue(message.contains("Planar-DAT pairs"))
        }
        let positions = sharedPositions(solver)
        for v in source.vertices {
            XCTAssertEqual(positions[v].z, 0, accuracy: 1e-6)
        }
    }

    func testElementSpanOverflowFreezesAndThrows() throws {
        let fixture = twoTriangleScene(sourceHeight: 0.2, sourceVelocity: .zero)
        let solver = try GPUSolver(scene: fixture.scene)
        let target = fixture.scene.tris[fixture.target].ids.0
        solver.setBodyStates([
            .init(body: target, position: F3(-100, -1, 0),
                  rotation: identity, linearVelocity: .zero)
        ])

        try solver.submitStep()
        XCTAssertThrowsError(try solver.synchronize()) { error in
            guard case let GPUSolver.RuntimeFailure.commandExecution(
                _, frame, _, domain, code, message) = error,
                  domain == GPUSolver.RuntimeFailure.planarDATFailureDomain,
                  code == GPUSolver.RuntimeFailure.planarDATElementGridSpanCode else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(frame, 1)
            XCTAssertTrue(message.contains("element AABBs"))
        }
        let positions = sharedPositions(solver)
        XCTAssertEqual(positions[target].x, -100, accuracy: 1e-6)
        for v in fixture.source {
            XCTAssertEqual(positions[v].z, 0.2, accuracy: 1e-6)
        }
    }

    func testRigidOnlySceneBypassesPlanarDAT() throws {
        var scene = PhysicsScene(name: "planar-dat-rigid-no-op")
        scene.settings.dt = 0.25
        scene.settings.gravity = 0
        scene.settings.iterations = 0
        let body = scene.addBody(
            size: F3(repeating: 0.2), density: 1, friction: 0,
            position: .zero, velocity: F3(2, 0, 0), shape: .sphere)
        let solver = try GPUSolver(scene: scene)
        var passes: [GPUSolver.PlanarDATPassSite] = []
        solver.planarDATPassObserverForTesting = { passes.append($0) }

        try solver.submitStep()
        try solver.synchronize()

        XCTAssertTrue(passes.isEmpty)
        XCTAssertEqual(solver.lastPlanarDATPairs, 0)
        XCTAssertEqual(solver.lastPlanarDATTruncations, 0)
        let position = sharedPositions(solver)[body]
        XCTAssertEqual(position.x, 0.5, accuracy: 1e-6)
        XCTAssertEqual(position.y, 0, accuracy: 1e-6)
        XCTAssertEqual(position.z, 0, accuracy: 1e-6)
    }

    func testPairOverflowFreezesParticlesBeforeTypedFailure() throws {
        let startZ: Float = 0.2
        let fixture = twoTriangleScene(
            sourceHeight: startZ,
            sourceVelocity: F3(0, 0, -0.4),
            iterations: 2)
        let solver = try GPUSolver(scene: fixture.scene)
        solver.planarDATPairCapacityForTesting = 0

        try solver.submitStep()
        XCTAssertThrowsError(try solver.synchronize()) { error in
            guard case let GPUSolver.RuntimeFailure.commandExecution(
                _, frame, _, domain, code, _) = error,
                  domain == GPUSolver.RuntimeFailure.planarDATFailureDomain,
                  code == GPUSolver.RuntimeFailure.planarDATPairCapacityCode else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(frame, 1)
        }

        // synchronize() has completed the command before publishing the
        // terminal error, so shared memory exposes the fail-closed pose.
        let positions = sharedPositions(solver)
        for vertex in fixture.source {
            XCTAssertEqual(positions[vertex].z, startZ, accuracy: 1e-6)
        }
        XCTAssertEqual(solver.runtimeFailure,
                       .planarDATPairCapacity(
                        frame: 1, required: solver.lastPlanarDATPairs,
                        capacity: 0))
    }

    func testSoftContactOverflowRestoresStepStartPoseBeforeTypedFailure() throws {
        let startZ: Float = 0.05
        let velocity = F3(0.1, 0, 0)
        let fixture = twoTriangleScene(
            sourceHeight: startZ,
            sourceVelocity: velocity,
            iterations: 2)
        let starts = fixture.source.map { fixture.scene.bodies[$0].position }
        let solver = try GPUSolver(scene: fixture.scene)
        solver.planarDATSoftCapacityForTesting = 0

        try solver.submitStep()
        XCTAssertThrowsError(try solver.synchronize()) { error in
            guard case let GPUSolver.RuntimeFailure.softContactCapacity(
                frame, required, capacity) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(frame, 1)
            XCTAssertGreaterThan(required, 0)
            XCTAssertEqual(capacity, 0)
        }

        // Contact overflow is discovered after prediction and before the
        // color solve. The failed frame must publish neither that prediction
        // nor a partially solved pose from the clipped contact stream.
        let positions = sharedPositions(solver)
        for (index, vertex) in fixture.source.enumerated() {
            XCTAssertEqual(positions[vertex].x, starts[index].x, accuracy: 1e-6)
            XCTAssertEqual(positions[vertex].y, starts[index].y, accuracy: 1e-6)
            XCTAssertEqual(positions[vertex].z, starts[index].z, accuracy: 1e-6)
        }
        XCTAssertEqual(
            solver.runtimeFailure,
            .softContactCapacity(
                frame: 1, required: solver.lastSoftCandidates, capacity: 0))
    }

}
