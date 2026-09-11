import Foundation
import Metal
import simd
import SimCore
import PhysicsAVBD

/// White-box regression for the two torsional-friction paths. Append a tiny
/// probe to the checkout's production shaders; no test kernel ships in the
/// solver library or changes production code generation.
func validateCableShaders() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw ValidationFailure.failed("Metal required for shader regressions")
    }
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    let directory = root.appendingPathComponent("Sources/PhysicsAVBD/Shaders")
    let files = try FileManager.default.contentsOfDirectory(at: directory,
        includingPropertiesForKeys: nil).filter {
            $0.pathExtension == "metal"
                && $0.lastPathComponent != "21_hierarchy_broadphase.metal"
                && $0.lastPathComponent != "31_optimized_convex_narrowphase.metal"
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    var source = "#include <metal_stdlib>\nusing namespace metal;\n"
    for file in files {
        source += try String(contentsOf: file, encoding: .utf8)
            .replacingOccurrences(of: "#include <metal_stdlib>", with: "")
            .replacingOccurrences(of: "using namespace metal;", with: "") + "\n"
    }
    source += """
    kernel void cable_surface_probe(device const float4* input [[buffer(0)]],
        device float4* output [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
        uint k=gid*5;
        float3 a=input[k+2].xyz,b=input[k+3].xyz,c=input[k+4].xyz,bary;
        float3 p=cableTriangleAxisWitness(input[k].xyz,input[k+1].xyz,a,b,c);
        float3 q=closestPtTriangle(p,a,b,c,bary);
        output[gid]=float4(p,distance(p,q));
    }
    kernel void cable_review_probe(
        device const ManifoldGPU* m [[buffer(0)]],
        device const float4* pos [[buffer(1)]],
        device const float4* rot [[buffer(2)]],
        device const float4* initial [[buffer(3)]],
        device float4* torsion [[buffer(4)]],
        device const JointGPU* joints [[buffer(5)]],
        device float4* output [[buffer(6)]],
        constant SimParams& P [[buffer(7)]],
        uint gid [[thread_position_in_grid]]) {
        if (gid >= 2) return;
        M3 lhs = m3_zero();
        float3 rhs = float3(0);
        stampTorsionalManifold(m[gid], torsion[gid], 1,
            pos, rot, pos, initial, 0, lhs, rhs);
        dual_torsion_one(pos, rot, pos, initial, m, torsion, P, gid);
        output[gid] = float4(length(rhs), fabs(torsion[gid].y),
            jointHasSolveTerms(joints[gid]) ? 1.0f : 0.0f, float(sizeof(JointGPU)));
    }
    """
    let options = MTLCompileOptions()
    if #available(macOS 15.0, iOS 18.0, *) { options.mathMode = .fast }
    let library = try device.makeLibrary(source: source, options: options)
    let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "cable_review_probe")!)
    func buffer<T>(_ values: [T]) -> MTLBuffer {
        values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    }
    // These GPU ABI records contain only scalar/vector POD fields. Start
    // from raw zeroed storage without exposing their internal initializers.
    func zeroRecord<T>(_ type: T.Type) -> T {
        Data(count: MemoryLayout<T>.stride).withUnsafeBytes { $0.loadUnaligned(as: type) }
    }
    var manifold = zeroRecord(ManifoldGPU.self)
    manifold.header = SIMD4(0,1,1,1 | 8)
    manifold.basisN = SIMD4(-1,0,0,0); manifold.basisT1 = SIMD4(0,1,0,0)
    manifold.contacts.0.rA = SIMD4(0.03,0,0,0)
    manifold.contacts.0.rB = SIMD4(-0.03,0,0,0)
    manifold.contacts.0.C0 = SIMD4(-0.001,0,0,0)
    manifold.contacts.0.penalty = SIMD4(1000,0,0,0)
    var legacy = manifold; legacy.header.w = 1
    // Axial spin alone cannot change geometry. Add a small contact-normal
    // spin so both torsional routines have to clamp to radius * normal load.
    let rotation = Quat(angle: 0.8, axis: F3(0,0,1)) * Quat(angle: 0.1, axis: F3(1,0,0))
    var cable = zeroRecord(JointGPU.self); cable.header.w = JointGPU.cableFlag
    var breakLoad = zeroRecord(JointGPU.self); breakLoad.header.w = 4 | 64
    let output = buffer([SIMD4<Float>.zero, .zero])
    let buffers = [buffer([manifold, legacy]), buffer([SIMD4<Float>.zero, .zero]),
        buffer([SIMD4<Float>(0,0,0,1), rotation.vector]),
        buffer([SIMD4<Float>(0,0,0,1), SIMD4<Float>(0,0,0,1)]),
        buffer([SIMD4<Float>(0.01,0,1000,0), SIMD4<Float>(0.01,0,1000,0)]),
        buffer([cable, breakLoad]), output]
    var params = zeroRecord(SimParamsGPU.self); params.alpha = 0; params.betaAng = 100
    let command = device.makeCommandQueue()!.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    for (index, value) in buffers.enumerated() { encoder.setBuffer(value, offset: 0, index: index) }
    encoder.setBytes(&params, length: MemoryLayout<SimParamsGPU>.stride, index: 7)
    encoder.dispatchThreads(MTLSize(width: 2,height: 1,depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 2,height: 1,depth: 1))
    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
    if let error = command.error { throw error }
    let values = output.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2)
    try require(abs(values[0].x - 0.01) < 1e-5 && abs(values[0].y - 0.01) < 1e-5,
                "torsional primal/dual must share the 1 N normal load: \(values[0])")
    try require(values[1].x < 1e-6 && values[1].y < 1e-6,
                "unflagged non-cable contact policy changed: \(values[1])")
    try require(values[0].z == 1 && values[1].z == 0,
                "zero-shear cable adjacency or break-load flag alias")
    try require(values[0].w == Float(MemoryLayout<JointGPU>.stride), "Swift/Metal joint stride mismatch")
    try validateCableSurfaceWitnesses(device: device, library: library)
    try validateCableSurfaceContact()
    print("PASS Metal torsional primal/dual contact bounds, legacy policy, explicit cable adjacency and joint ABI")
}
