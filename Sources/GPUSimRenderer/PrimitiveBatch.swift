import Metal

/// Stable GPU classification avoids launching unrelated shape vertex shaders.
/// Source IDs and the draw order within each shape remain unchanged, including
/// at coplanar depth ties. No readback or CPU instance sorting is needed.
final class PrimitiveBatch {
  enum Failure: Error { case allocation, encoder }
  private let device: MTLDevice
  private let classify, prefix, scatter: MTLComputePipelineState
  private(set) var indices: MTLBuffer!
  private(set) var arguments: MTLBuffer!
  private var ranks, blocks: MTLBuffer!
  init(device: MTLDevice) throws {
    self.device = device
    let library = try device.makeLibrary(
      source: """
        #include <metal_stdlib>
        using namespace metal;
        struct Instance { float4x4 model; float4 color; float4 params; float4 material; };
        inline uint shapeOf(float value) {
            return isfinite(value) && value>=0 && value<=3 && floor(value)==value ? uint(value) : 4;
        }
        kernel void classify(device const Instance* instances [[buffer(0)]],device uint* ranks [[buffer(1)]],
            device uint* blocks [[buffer(2)]],constant uint& count [[buffer(3)]],
            uint id [[thread_position_in_grid]],uint local [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
            threadgroup uint shapes[128];
            uint shape=id<count ? shapeOf(instances[id].color.w) : 4;
            shapes[local]=shape; threadgroup_barrier(mem_flags::mem_threadgroup);
            if (id<count && shape<4) {
                uint rank=0;
                for(uint i=0;i<local;++i) rank+=shapes[i]==shape;
                ranks[id]=rank;
            }
            if(local<4) {
                uint total=0; for(uint i=0;i<128;++i) total+=shapes[i]==local;
                blocks[group*4+local]=total;
            }
        }
        kernel void prefix(device uint* blocks [[buffer(2)]],constant uint& count [[buffer(3)]],
            device uint4* args [[buffer(4)]],uint shape [[thread_position_in_grid]]) {
            uint sum=0;
            for(uint group=0;group<(count+127)/128;++group) {
                uint n=blocks[group*4+shape]; blocks[group*4+shape]=sum; sum+=n;
            }
            const uint vertices[4]={36,\(SPHV),\(TORV),\(CAPV)};
            args[shape]=uint4(vertices[shape],sum,0,shape*count);
        }
        kernel void scatter(device const Instance* instances [[buffer(0)]],device const uint* ranks [[buffer(1)]],
            device const uint* blocks [[buffer(2)]],constant uint& count [[buffer(3)]],
            device uint* indices [[buffer(5)]],uint id [[thread_position_in_grid]]) {
            if(id>=count) return;
            uint shape=shapeOf(instances[id].color.w);
            if(shape<4) indices[shape*count+blocks[(id/128)*4+shape]+ranks[id]]=id;
        }
        """, options: nil)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else { throw Failure.encoder }
      return try device.makeComputePipelineState(function: function)
    }
    classify = try pipeline("classify")
    prefix = try pipeline("prefix")
    scatter = try pipeline("scatter")
    arguments = device.makeBuffer(length: 64, options: .storageModePrivate)
    guard arguments != nil else { throw Failure.allocation }
  }
  func encode(command: MTLCommandBuffer, instances: MTLBuffer, count: Int) throws {
    if indices == nil || indices.length < count * 16 {
      indices = device.makeBuffer(length: count * 16, options: .storageModePrivate)
      ranks = device.makeBuffer(length: count * 4, options: .storageModePrivate)
      blocks = device.makeBuffer(length: ((count + 127) / 128) * 16, options: .storageModePrivate)
    }
    guard indices != nil, ranks != nil, blocks != nil else { throw Failure.allocation }
    guard let encoder = command.makeComputeCommandEncoder() else { throw Failure.encoder }
    encoder.label = "Classify primitive instances in source order"
    encoder.setBuffer(instances, offset: 0, index: 0)
    encoder.setBuffer(ranks, offset: 0, index: 1)
    encoder.setBuffer(blocks, offset: 0, index: 2)
    encoder.setBuffer(arguments, offset: 0, index: 4)
    encoder.setBuffer(indices, offset: 0, index: 5)
    var n = UInt32(count)
    encoder.setBytes(&n, length: 4, index: 3)
    encoder.setComputePipelineState(classify)
    // Full groups make every threadgroup slot initialized, even in the tail.
    encoder.dispatchThreadgroups(
      MTLSize(width: (count + 127) / 128, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.setComputePipelineState(prefix)
    encoder.dispatchThreads(
      MTLSize(width: 4, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1))
    encoder.setComputePipelineState(scatter)
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
  }
}
