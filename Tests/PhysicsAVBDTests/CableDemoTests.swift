import XCTest
import simd
import SimCore
import GPUSimDemos

final class CableDemoTests: XCTestCase {
    func testClipsHaveConformingVolumeAndNoCableAttachments() {
        let scene = Demos.cableGrippers()
        let cableBodies = Set(scene.cables.flatMap(\.bodyIDs))
        let particles = scene.bodies.indices.filter { scene.bodies[$0].isParticle }
        XCTAssertFalse(particles.isEmpty)
        XCTAssertTrue(particles.contains { scene.bodies[$0].density == 0 })
        XCTAssertTrue(particles.contains { scene.bodies[$0].density > 0 })
        for joint in scene.joints {
            // Retention/release must come from contact with the clip lips.
            XCTAssertEqual(cableBodies.contains(joint.bodyA), cableBodies.contains(joint.bodyB))
        }
        var faces: [[Int]: Int] = [:]
        var volume: Float = 0
        for tet in scene.tets {
            let ids = [tet.ids.0, tet.ids.1, tet.ids.2, tet.ids.3]
            let p = ids.map { scene.bodies[$0].position }
            let v = abs(dot(p[1] - p[0], cross(p[2] - p[0], p[3] - p[0]))) / 6
            XCTAssertGreaterThan(v, 1e-9)
            volume += v
            for omitted in 0..<4 {
                faces[ids.enumerated().filter { $0.offset != omitted }.map(\.element).sorted(), default: 0] += 1
            }
        }
        XCTAssertTrue(faces.values.allSatisfy { $0 == 1 || $0 == 2 })
        var boundaryEdges: [[Int]: Int] = [:]
        for (face, count) in faces where count == 1 {
            for (a, b) in [(0,1), (1,2), (2,0)] {
                boundaryEdges[[face[a], face[b]].sorted(), default: 0] += 1
            }
        }
        XCTAssertTrue(boundaryEdges.values.allSatisfy { $0 == 2 }, "clip skins must be watertight")
        // Four 270-degree annular clips, 120 mm long. Linear surface facets
        // approximate the analytic annulus within two percent.
        let expected: Float = 4 * 0.12 * 0.75 * .pi * (0.075 * 0.075 - 0.04 * 0.04)
        XCTAssertEqual(volume, expected, accuracy: expected * 0.02)
    }

    func testTwistFixturesDriveBothMaterialFrames() {
        let scene = Demos.cableTwisting(turnRate: 0.17)
        XCTAssertEqual(scene.cables.count, 2)
        XCTAssertEqual(scene.spinners.count, 1)
        XCTAssertEqual(scene.spinners[0].omega, 0.17 * 2 * .pi, accuracy: 1e-6)
        for cable in scene.cables {
            for body in [cable.bodyIDs[0], cable.bodyIDs.last!] {
                let weld = scene.joints.first { $0.bodyB == body && $0.cable == nil }
                XCTAssertNotNil(weld)
                XCTAssertTrue(weld?.stiffnessAng.isInfinite == true)
            }
        }
    }

    func testEthernetPlugHasConformingSoftVolumeAndAnOpenRigidSocket() throws {
        let task = Demos.ethernetInsertionTask()
        let scene = task.scene
        XCTAssertNil(Demos.make("cablethreading"))
        XCTAssertEqual(scene.cables.count, 9) // cable plus eight contact wires
        XCTAssertEqual(task.contactWires.count, 8)
        XCTAssertTrue(task.contactWires.allSatisfy { $0.bodyIDs.count == 1 })
        let particles = Set(task.plugNodes)
        XCTAssertGreaterThan(particles.count, 200)
        XCTAssertTrue(particles.allSatisfy { scene.bodies[$0].isDynamic })
        let mu: Float = 2.4e9 / (2 * (1 + 0.37))
        XCTAssertTrue(scene.tets.allSatisfy { abs($0.mu - mu) < 100 })
        var faces: [[Int]: Int] = [:]
        var volume: Float = 0
        for tet in scene.tets {
            let ids = [tet.ids.0, tet.ids.1, tet.ids.2, tet.ids.3]
            let p = ids.map { scene.bodies[$0].position }
            let v = abs(dot(p[1]-p[0], cross(p[2]-p[0], p[3]-p[0]))) / 6
            XCTAssertGreaterThan(v, 1e-15)
            volume += v
            for omitted in 0..<4 {
                faces[ids.enumerated().filter { $0.offset != omitted }.map(\.element).sorted(), default: 0] += 1
            }
        }
        XCTAssertTrue(faces.values.allSatisfy { $0 == 1 || $0 == 2 })
        var edges: [[Int]: Int] = [:]
        for (face, count) in faces where count == 1 {
            for (a,b) in [(0,1),(1,2),(2,0)] { edges[[face[a],face[b]].sorted(), default: 0] += 1 }
        }
        XCTAssertTrue(edges.values.allSatisfy { $0 == 2 }, "housing has a closed volume boundary")
        XCTAssertEqual(scene.tris.count, 20)
        XCTAssertTrue(scene.tris.allSatisfy { $0.mu > 0 && $0.bend > 0 && !$0.selfCollisionEnabled })
        let housingNodes = Set(scene.tets.flatMap { [$0.ids.0,$0.ids.1,$0.ids.2,$0.ids.3] })
        let latchNodes = Set(scene.tris.flatMap { [$0.ids.0,$0.ids.1,$0.ids.2] })
        XCTAssertEqual(housingNodes.intersection(latchNodes).count, 6, "two shared rows clamp the latch root")
        let latchVolume = scene.tris.reduce(Float(0)) { sum, tri in
            let p = [tri.ids.0,tri.ids.1,tri.ids.2].map { scene.bodies[$0].position }
            return sum + length(cross(p[1]-p[0],p[2]-p[0]))/2 * 0.00047
        }
        let mass = particles.reduce(Float(0)) { total, id in
            let b = scene.bodies[id], r = b.size.x/2
            return total + b.density * (4 * .pi/3) * r*r*r
        }
        XCTAssertEqual(mass, (volume+latchVolume) * 1200, accuracy: 1e-7)
        XCTAssertGreaterThan(mass, 0.0008)
        XCTAssertLessThan(mass, 0.002)
        XCTAssertFalse(scene.bodies[task.remoteConnectorBody].isDynamic)
        let remote = try XCTUnwrap(scene.joints.first {
            $0.bodyA == task.remoteConnectorBody && $0.bodyB == task.cable.bodyIDs[0]
        })
        let root = scene.bodies[remote.bodyA].position+remote.rA
        let command = task.command(at:EthernetInsertionTask.duration)
        let end = command.position+command.rotation.act(F3(-0.004,0,0))
        XCTAssertGreaterThan(task.cable.restLengths.reduce(0,+)-distance(root,end),0.04)
        XCTAssertFalse(scene.bodies[task.socketBody].isDynamic)
        for x: Float in [0.002, 0.0075, 0.014] {
            for c in scene.colliders where c.body == task.socketBody && c.collisionEnabled {
                let p = c.localRotation.inverse.act(F3(x,0,EthernetInsertionTask.axisHeight)
                    - scene.bodies[task.socketBody].position - c.localPosition)
                XCTAssertTrue((abs(p).x > c.size.x/2) || (abs(p).y > c.size.y/2) || (abs(p).z > c.size.z/2))
            }
        }
        let bonds = scene.joints.filter { particles.contains($0.bodyB) }
        XCTAssertGreaterThan(bonds.count, 10)
        XCTAssertLessThan(bonds.count, particles.count)
        XCTAssertTrue(bonds.allSatisfy { $0.bodyA == task.toolBody && $0.stiffnessLin == 1e5 })
        XCTAssertEqual(Set(scene.rigidMotionGroups[0]), particles.union([task.toolBody]))
        let repeated = Demos.ethernetInsertionTask().scene
        XCTAssertEqual(scene.joints.map(\.bodyA), repeated.joints.map(\.bodyA))
        XCTAssertEqual(scene.joints.map(\.bodyB), repeated.joints.map(\.bodyB), "attachment order must be reproducible")
        let replicated = scene.replicated(count: 2, spacing: F3(1,0,0), includeVisuals: true)
        XCTAssertEqual(replicated.scene.rigidMotionGroups[1], scene.rigidMotionGroups[0].map { $0 + scene.bodies.count })
        XCTAssertEqual(replicated.scene.skinnedMeshes.last?.vertices.first?.color, scene.skinnedMeshes.last?.vertices.first?.color)
        XCTAssertTrue(replicated.scene.tris.allSatisfy { !$0.selfCollisionEnabled })
    }

    func testEthernetRunStopsItsCommandClockOnExcessiveForce() {
        let task = Demos.ethernetInsertionTask()
        var run = EthernetInsertionRun(task: task)
        let command = run.command
        run.observe(toolPosition: command.position - F3(0.002,0,0), toolRotation: command.rotation, noseCenter: .zero)
        XCTAssertTrue(run.stopped)
        XCTAssertFalse(run.seated)
        XCTAssertEqual(run.step, 0)
        XCTAssertEqual(run.command.position, command.position)
        XCTAssertEqual(run.command.linearVelocity, .zero)
        XCTAssertEqual(run.command.angularVelocity, .zero)
        XCTAssertGreaterThan(run.peakForce, EthernetInsertionTask.forceLimit)
    }
}
