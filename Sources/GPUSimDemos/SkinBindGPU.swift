import Foundation
import Metal
import simd
import SimCore

/// Hash-grid skin binding on the GPU. One thread per visual vertex walks the
/// cage's tet grid outward from its (clamped) cell, evaluates barycentric
/// coordinates in every candidate tet with the cage's precomputed inverse
/// rest matrices, and keeps the same best-candidate rule as the serial
/// binder: an enclosing tet first, then the smallest reconstruction error,
/// then the smallest barycentric excursion. The serial walk cost ~24 s for
/// 300k vertices in a debug build; this is a few milliseconds.
final class SkinBindGPU {
    static let shared: SkinBindGPU? = SkinBindGPU()

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    private struct Params {
        var origin: SIMD3<Float>
        var cellSize: Float
        var dims: SIMD3<UInt32>
        var recordCount: UInt32
        var vertexCount: UInt32
        var maxRing: UInt32
        var pad0: UInt32 = 0
        var pad1: UInt32 = 0
    }

    private struct Pick {
        var record: UInt32
        var pad: SIMD3<UInt32>
        var weights: SIMD4<Float>
    }

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
        float3 origin;
        float cellSize;
        uint3 dims;
        uint recordCount;
        uint vertexCount;
        uint maxRing;
        uint pad0;
        uint pad1;
    };

    struct Pick {
        uint record;
        uint3 pad;
        float4 weights;
    };

    struct Best {
        uint record;
        float4 w;
        bool inside;
        float dist2;
        float outside;
        bool valid;
    };

    inline void consider(uint ti,
                         float3 p,
                         device const float4* x0,
                         device const float4* x1,
                         device const float4* x2,
                         device const float4* x3,
                         device const float4* inv0,
                         device const float4* inv1,
                         device const float4* inv2,
                         thread Best& best)
    {
        float3 d = p - x0[ti].xyz;
        float3 u = float3(dot(inv0[ti].xyz, d), dot(inv1[ti].xyz, d),
                          dot(inv2[ti].xyz, d));
        float4 w = float4(1.0f - u.x - u.y - u.z, u.x, u.y, u.z);
        float minW = min(min(w.x, w.y), min(w.z, w.w));
        float outside = max(0.0f, -minW);
        bool inside = outside <= 1e-4f;
        float4 wc = max(w, float4(0.0f));
        float sum = wc.x + wc.y + wc.z + wc.w;
        wc = sum <= 1e-12f ? float4(0.25f) : wc / sum;
        float3 q = x0[ti].xyz * wc.x + x1[ti].xyz * wc.y
                 + x2[ti].xyz * wc.z + x3[ti].xyz * wc.w;
        float d2 = length_squared(q - p);
        if (!best.valid
            || (inside && !best.inside)
            || (inside == best.inside && d2 < best.dist2)
            || (inside == best.inside && d2 == best.dist2
                && outside < best.outside)) {
            best.record = ti; best.w = w; best.inside = inside;
            best.dist2 = d2; best.outside = outside; best.valid = true;
        }
    }

    kernel void skin_bind(
        device const float4* vertices  [[buffer(0)]],
        device const float4* x0        [[buffer(1)]],
        device const float4* x1        [[buffer(2)]],
        device const float4* x2        [[buffer(3)]],
        device const float4* x3        [[buffer(4)]],
        device const float4* inv0      [[buffer(5)]],
        device const float4* inv1      [[buffer(6)]],
        device const float4* inv2      [[buffer(7)]],
        device const uint* cellStart   [[buffer(8)]],
        device const uint* cellItems   [[buffer(9)]],
        constant Params& P             [[buffer(10)]],
        device Pick* picks             [[buffer(11)]],
        uint gid                       [[thread_position_in_grid]])
    {
        if (gid >= P.vertexCount) return;
        float3 p = vertices[gid].xyz;
        float3 q = (p - P.origin) / P.cellSize;
        int3 dims = int3(P.dims);
        int3 c = clamp(int3(floor(q)), int3(0), dims - 1);
        Best best;
        best.valid = false;
        for (uint r = 0; r <= P.maxRing && !best.valid; r++) {
            int ri = int(r);
            int3 mn = max(c - ri, int3(0));
            int3 mx = min(c + ri, dims - 1);
            for (int i = mn.x; i <= mx.x; i++)
            for (int j = mn.y; j <= mx.y; j++)
            for (int k = mn.z; k <= mx.z; k++) {
                uint cell = uint((k * dims.y + j) * dims.x + i);
                uint s = cellStart[cell], e = cellStart[cell + 1];
                for (uint idx = s; idx < e; idx++) {
                    consider(cellItems[idx], p, x0, x1, x2, x3,
                             inv0, inv1, inv2, best);
                }
            }
        }
        if (!best.valid) {
            for (uint ti = 0; ti < P.recordCount; ti++) {
                consider(ti, p, x0, x1, x2, x3, inv0, inv1, inv2, best);
            }
        }
        Pick pick;
        pick.record = best.valid ? best.record : 0u;
        pick.pad = uint3(0);
        pick.weights = best.valid ? best.w : float4(0.25f);
        picks[gid] = pick;
    }
    """

    private init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = try? device.makeLibrary(source: Self.source, options: nil),
              let function = library.makeFunction(name: "skin_bind"),
              let pipeline = try? device.makeComputePipelineState(function: function)
        else { return nil }
        self.device = device
        self.queue = queue
        self.pipeline = pipeline
    }

    func bind(mesh: SurfaceMesh, index: Demos.TetBindIndex) -> [Demos.SkinBindPick]? {
        let records = index.records
        guard !records.isEmpty, !mesh.vertices.isEmpty else { return nil }
        func buffer<T>(_ values: [T]) -> MTLBuffer? {
            values.withUnsafeBytes { raw in
                device.makeBuffer(bytes: raw.baseAddress!, length: max(16, raw.count),
                                  options: .storageModeShared)
            }
        }
        func pad(_ v: SIMD3<Float>) -> SIMD4<Float> { SIMD4(v, 0) }
        let cellCount = index.dims.x * index.dims.y * index.dims.z
        var cellStart = [UInt32](repeating: 0, count: cellCount + 1)
        var cellItems: [UInt32] = []
        for cell in 0..<cellCount {
            cellStart[cell] = UInt32(cellItems.count)
            cellItems.append(contentsOf: index.cells[cell].map(UInt32.init))
        }
        cellStart[cellCount] = UInt32(cellItems.count)
        guard let vertexBuf = buffer(mesh.vertices.map(pad)),
              let x0 = buffer(records.map { pad($0.x0) }),
              let x1 = buffer(records.map { pad($0.x1) }),
              let x2 = buffer(records.map { pad($0.x2) }),
              let x3 = buffer(records.map { pad($0.x3) }),
              let inv0 = buffer(records.map { pad($0.restInv0) }),
              let inv1 = buffer(records.map { pad($0.restInv1) }),
              let inv2 = buffer(records.map { pad($0.restInv2) }),
              let startBuf = buffer(cellStart),
              let itemsBuf = buffer(cellItems.isEmpty ? [0] : cellItems),
              let picksBuf = device.makeBuffer(
                length: mesh.vertices.count * MemoryLayout<Pick>.stride,
                options: .storageModeShared),
              let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder()
        else { return nil }
        var params = Params(
            origin: index.origin, cellSize: index.cellSize,
            dims: SIMD3(UInt32(index.dims.x), UInt32(index.dims.y), UInt32(index.dims.z)),
            recordCount: UInt32(records.count),
            vertexCount: UInt32(mesh.vertices.count), maxRing: 12)
        encoder.setComputePipelineState(pipeline)
        for (slot, buf) in [vertexBuf, x0, x1, x2, x3, inv0, inv1, inv2, startBuf, itemsBuf].enumerated() {
            encoder.setBuffer(buf, offset: 0, index: slot)
        }
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 10)
        encoder.setBuffer(picksBuf, offset: 0, index: 11)
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreadgroups(
            MTLSize(width: (mesh.vertices.count + width - 1) / width, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.error == nil else { return nil }
        let out = picksBuf.contents().bindMemory(to: Pick.self, capacity: mesh.vertices.count)
        return (0..<mesh.vertices.count).map {
            Demos.SkinBindPick(record: Int(out[$0].record), weights: out[$0].weights)
        }
    }
}
