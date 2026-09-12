import Foundation
import Metal
import PhysicsAVBD
import SimCore
import simd

func validateCableSurfaceWitnesses(device: MTLDevice, library: MTLLibrary) throws {
    let a = F3(-0.002, -0.001, -0.001)
    let b = F3(-0.002, 0.001, -0.001)
    let c = F3(-0.002, 0, 0.001)
    let cases: [(F3, F3, Float)] = [
        (F3(-0.004, 0, 0), F3(0.004, 0, 0), 0),  // face crossing between old samples
        (F3(-0.0023, -0.002, 0), F3(-0.0023, 0.002, 0), 0.0003),  // parallel face
        (F3(-0.004, 0, 0.0013), F3(0.004, 0, 0.0013), 0.0003),  // edge/vertex minimum
        (F3(-0.002, 0, 0), F3(-0.002, 0, 0), 0),  // zero-length axis
    ]
    let rotation = Quat(angle: 0.7, axis: normalize(F3(1, 2, 3)))
    var input: [SIMD4<Float>] = []
    var expected: [Float] = []
    var tolerances: [Float] = []
    for scale: Float in [0.1, 1, 10] {
        for (p0, p1, distance) in cases {
            input += [p0, p1, a, b, c].map { SIMD4(rotation.act($0 * scale) + F3(0.005, -0.01, 0.003), 0) }
            expected.append(distance * scale)
            tolerances.append(2e-7 * max(1, scale))
        }
    }
    let data = device.makeBuffer(bytes: input, length: input.count * 16, options: .storageModeShared)!
    let output = device.makeBuffer(length: expected.count * 16, options: .storageModeShared)!
    let pipeline = try device.makeComputePipelineState(
        function: library.makeFunction(name: "cable_surface_probe")!)
    let command = device.makeCommandQueue()!.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(data, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.dispatchThreads(
        MTLSize(width: expected.count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: expected.count, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error { throw error }
    let values = output.contents().bindMemory(to: SIMD4<Float>.self, capacity: expected.count)
    for i in expected.indices {
        try require(
            values[i].w.isFinite && abs(values[i].w - expected[i]) < tolerances[i],
            "continuous capsule/triangle witness case \(i): \(values[i].w) vs \(expected[i])")
    }
    print("PASS continuous cable/triangle face, parallel, edge and degenerate-axis witnesses at three scales")
}

func validateCableSurfaceContact() throws {
    func makeScene(native: Bool) throws -> (PhysicsScene, Int, [Int]) {
        var s = PhysicsScene(name: "thin wire crossing tet face")
        s.settings.gravity = 0
        s.settings.dt = 1 / 480
        s.settings.iterations = 4
        s.settings.collisionMargin = 0.000025
        s.settings.deformableCollisionMargin = 0.000015
        let vertices = [
            F3(-0.002, -0.001, -0.001), F3(-0.002, 0.001, -0.001),
            F3(-0.002, 0, 0.001), F3(-0.003, 0, -0.0005),
        ]
        let ids = vertices.map { s.addParticle(radius: 0.000025, mass: 0, position: $0) }
        s.addTet(SceneTet(ids: (ids[0], ids[1], ids[2], ids[3]), mu: 1e6, lambda: 1e6))
        let wire: Int
        if native {
            wire = try s.addCable(
                points: [F3(-0.004, 0, 0), F3(0.004, 0, 0)], radius: 0.00015, density: 8800,
                material: CableMaterial(
                    stretchRigidity: 1000, shearRigidity: 500, bendRigidity: 0.01, twistRigidity: 0.01)
            ).bodyIDs[0]
        } else {
            wire = s.addBody(
                size: F3(0.008, 0.00015, 0), density: 8800, friction: 0.25,
                position: .zero, rotation: Quat(from: F3(0, 0, 1), to: F3(1, 0, 0)), shape: .capsule)
        }
        return (s, wire, ids)
    }
    for native in [false, true] {
        let (s, wire, vertices) = try makeScene(native: native)
        let solver = try GPUSolver(scene: s)
        try solver.submitStep()
        try solver.synchronize()
        let count = solver.debugRigidTriangleContactCount(
            colliderIDs: s.colliders.indices.filter { s.colliders[$0].body == wire }, surfaceBodies: vertices)
        try require(
            native ? count > 0 : count == 0,
            "thin face crossing native=\(native), contacts=\(count); preserve the unflagged compatibility path"
        )
    }
    print("PASS native cable catches a face between sphere samples; ordinary capsule compatibility preserved")
}

/// Legacy V-T/E-E must use the physical self-contact policy, independent of
/// whether a vertex is rendered as a sheet or as a tet boundary.
func validateSurfaceContactPolicy() throws {
    func folded(selfContact: Bool, separate: Bool) -> PhysicsScene {
        var scene = PhysicsScene(name: "folded surface policy")
        scene.settings.gravity = 0
        scene.settings.dt = 1/240
        scene.settings.iterations = 1
        scene.settings.deterministic = true
        scene.settings.deformableCollisionMargin = 0.00001
        for row in 0..<7 {
            let x = Float(row <= 3 ? row : 6-row)*0.01
            let z: Float = row <= 2 ? 0 : row == 3 ? 0.00025 : 0.0005
            for y: Float in [-0.005,0.005] {
                _ = scene.addParticle(radius:0.0004,mass:0.001,position:F3(x,y,z))
            }
        }
        for row in 0..<6 where !separate || (row != 2 && row != 3) {
            let a = 2*row, b = a+2
            scene.addTri(SceneTri(ids:(a,b,b+1),selfCollisionEnabled:selfContact))
            scene.addTri(SceneTri(ids:(a,b+1,a+1),selfCollisionEnabled:selfContact))
        }
        return scene
    }
    for (selfContact,separate) in [(false,false),(true,false),(false,true)] {
        let solver = try GPUSolver(scene: folded(selfContact:selfContact,separate:separate))
        try solver.submitStep()
        try solver.synchronize()
        let expectContact = selfContact || separate
        try require(expectContact ? solver.lastNumSoft > 0 : solver.lastNumSoft == 0,
            "surface policy self=\(selfContact) separate=\(separate) emitted \(solver.lastNumSoft)")
    }
    print("PASS surface self-contact off/on and separate-body contact with self-contact off")
}
