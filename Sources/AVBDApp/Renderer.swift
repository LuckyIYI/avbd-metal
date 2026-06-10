import Metal
import MetalKit
import AVBDCore
import simd

// Instanced-box renderer for AVBD body state. Instance transforms are built
// on the GPU (build_instances kernel in AVBDCore); this file owns the render
// pipeline, camera, and a ground grid.

let renderShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct RenderInstance {
    float4x4 model;
    float4 color;       // w = shape type (0 box, 1 sphere, 2 torus)
    float4 params;      // torus: x = major R, y = minor r
};

struct Uniforms {
    float4x4 viewProj;
    float4 lightDir;    // xyz used
    float4 eye;         // xyz used
};

struct VOut {
    float4 position [[position]];
    float3 normal;
    float3 world;
    float4 color;
};

// Unit cube (centered, size 1): 36 verts, position+normal
constant float3 cubeVerts[36] = {
    // +X
    float3(0.5,-0.5,-0.5), float3(0.5, 0.5,-0.5), float3(0.5, 0.5, 0.5),
    float3(0.5,-0.5,-0.5), float3(0.5, 0.5, 0.5), float3(0.5,-0.5, 0.5),
    // -X
    float3(-0.5,-0.5, 0.5), float3(-0.5, 0.5, 0.5), float3(-0.5, 0.5,-0.5),
    float3(-0.5,-0.5, 0.5), float3(-0.5, 0.5,-0.5), float3(-0.5,-0.5,-0.5),
    // +Y
    float3(-0.5, 0.5,-0.5), float3(-0.5, 0.5, 0.5), float3(0.5, 0.5, 0.5),
    float3(-0.5, 0.5,-0.5), float3(0.5, 0.5, 0.5), float3(0.5, 0.5,-0.5),
    // -Y
    float3(-0.5,-0.5, 0.5), float3(-0.5,-0.5,-0.5), float3(0.5,-0.5,-0.5),
    float3(-0.5,-0.5, 0.5), float3(0.5,-0.5,-0.5), float3(0.5,-0.5, 0.5),
    // +Z
    float3(-0.5,-0.5, 0.5), float3(0.5,-0.5, 0.5), float3(0.5, 0.5, 0.5),
    float3(-0.5,-0.5, 0.5), float3(0.5, 0.5, 0.5), float3(-0.5, 0.5, 0.5),
    // -Z
    float3(0.5,-0.5,-0.5), float3(-0.5,-0.5,-0.5), float3(-0.5, 0.5,-0.5),
    float3(0.5,-0.5,-0.5), float3(-0.5, 0.5,-0.5), float3(0.5, 0.5,-0.5),
};

constant float3 cubeNormals[6] = {
    float3(1,0,0), float3(-1,0,0), float3(0,1,0),
    float3(0,-1,0), float3(0,0,1), float3(0,0,-1),
};

vertex VOut box_vertex(uint vid [[vertex_id]],
                       uint iid [[instance_id]],
                       device const RenderInstance* instances [[buffer(0)]],
                       constant Uniforms& U [[buffer(1)]])
{
    RenderInstance inst = instances[iid];
    if (inst.color.w != 0.0) {       // non-box instance: collapse in box pass
        VOut o;
        o.position = float4(0, 0, -2, 1);
        o.normal = float3(0); o.world = float3(0); o.color = float4(0);
        return o;
    }
    float3 p = cubeVerts[vid];
    float3 n = cubeNormals[vid / 6];
    float4 world = inst.model * float4(p, 1);
    // normal via rotation part (columns normalized to undo size scaling)
    float3x3 rot = float3x3(normalize(inst.model[0].xyz),
                            normalize(inst.model[1].xyz),
                            normalize(inst.model[2].xyz));
    VOut o;
    o.position = U.viewProj * world;
    o.normal = rot * n;
    o.world = world.xyz;
    o.color = inst.color;
    return o;
}

fragment float4 box_fragment(VOut in [[stage_in]],
                             constant Uniforms& U [[buffer(1)]])
{
    float3 n = normalize(in.normal);
    float ndl = max(dot(n, -U.lightDir.xyz), 0.0);
    float3 ambient = float3(0.35);
    float3 lit = in.color.rgb * (ambient + float3(0.75) * ndl);
    // cheap distance fade for depth perception
    float d = length(in.world - U.eye.xyz);
    lit = mix(lit, float3(0.62, 0.67, 0.75), clamp(d / 400.0, 0.0, 0.6));
    return float4(lit, 1.0);
}

// Analytic UV sphere: STACKS x SLICES quads, 6 verts each
#define SPH_STACKS 12
#define SPH_SLICES 18

vertex VOut sphere_vertex(uint vid [[vertex_id]],
                          uint iid [[instance_id]],
                          device const RenderInstance* instances [[buffer(0)]],
                          constant Uniforms& U [[buffer(1)]])
{
    RenderInstance inst = instances[iid];
    if (inst.color.w != 1.0) {       // non-sphere: collapse in sphere pass
        VOut o;
        o.position = float4(0, 0, -2, 1);
        o.normal = float3(0); o.world = float3(0); o.color = float4(0);
        return o;
    }
    uint quad = vid / 6;
    uint corner = vid % 6;
    uint stack = quad / SPH_SLICES;
    uint slice = quad % SPH_SLICES;
    // two triangles: (0,0),(1,0),(1,1) and (0,0),(1,1),(0,1)
    uint2 off[6] = { uint2(0,0), uint2(1,0), uint2(1,1), uint2(0,0), uint2(1,1), uint2(0,1) };
    float v = float(stack + off[corner].y) / float(SPH_STACKS);   // 0..1 pole to pole
    float u = float(slice + off[corner].x) / float(SPH_SLICES);
    float phi = v * M_PI_F;
    float theta = u * 2.0 * M_PI_F;
    float3 n = float3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
    float3 p = n * 0.5;              // unit-diameter sphere
    float4 world = inst.model * float4(p, 1);
    float3x3 rot = float3x3(normalize(inst.model[0].xyz),
                            normalize(inst.model[1].xyz),
                            normalize(inst.model[2].xyz));
    VOut o;
    o.position = U.viewProj * world;
    o.normal = rot * n;
    o.world = world.xyz;
    o.color = float4(inst.color.rgb, 1);
    return o;
}

// Analytic torus: RINGS x SIDES quads
#define TOR_RINGS 24
#define TOR_SIDES 12

vertex VOut torus_vertex(uint vid [[vertex_id]],
                         uint iid [[instance_id]],
                         device const RenderInstance* instances [[buffer(0)]],
                         constant Uniforms& U [[buffer(1)]])
{
    RenderInstance inst = instances[iid];
    if (inst.color.w != 2.0) {       // non-torus: collapse in torus pass
        VOut o;
        o.position = float4(0, 0, -2, 1);
        o.normal = float3(0); o.world = float3(0); o.color = float4(0);
        return o;
    }
    uint quad = vid / 6;
    uint corner = vid % 6;
    uint ring = quad / TOR_SIDES;
    uint side = quad % TOR_SIDES;
    uint2 off[6] = { uint2(0,0), uint2(1,0), uint2(1,1), uint2(0,0), uint2(1,1), uint2(0,1) };
    float u = float(ring + off[corner].x) / float(TOR_RINGS) * 2.0 * M_PI_F;
    float v = float(side + off[corner].y) / float(TOR_SIDES) * 2.0 * M_PI_F;
    float R = inst.params.x;
    float r = inst.params.y;
    float3 spine = float3(R * cos(u), R * sin(u), 0);
    float3 n = float3(cos(u) * cos(v), sin(u) * cos(v), sin(v));
    float3 p = spine + n * r;
    float4 world = inst.model * float4(p, 1);
    float3x3 rot = float3x3(normalize(inst.model[0].xyz),
                            normalize(inst.model[1].xyz),
                            normalize(inst.model[2].xyz));
    VOut o;
    o.position = U.viewProj * world;
    o.normal = rot * n;
    o.world = world.xyz;
    o.color = float4(inst.color.rgb, 1);
    return o;
}

// Ground grid lines
struct GridOut { float4 position [[position]]; float4 color; };

vertex GridOut grid_vertex(uint vid [[vertex_id]],
                           constant Uniforms& U [[buffer(1)]])
{
    int line = int(vid) / 2;
    int end = int(vid) % 2;
    const int half_ = 40;
    const float spacing = 2.0;
    float4 p;
    if (line <= 2 * half_) {
        float x = float(line - half_) * spacing;
        p = float4(x, (end == 0 ? -half_ : half_) * spacing, 0.011, 1);
    } else {
        float y = float(line - 3 * half_ - 1) * spacing;
        p = float4((end == 0 ? -half_ : half_) * spacing, y, 0.011, 1);
    }
    GridOut o;
    o.position = U.viewProj * p;
    float major = (line % 5 == 0) ? 0.30 : 0.16;
    o.color = float4(float3(major), 1);
    return o;
}

fragment float4 grid_fragment(GridOut in [[stage_in]]) {
    return in.color;
}
"""

struct Uniforms {
    var viewProj: simd_float4x4
    var lightDir: SIMD4<Float>
    var eye: SIMD4<Float>
}

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    var boxPipeline: MTLRenderPipelineState!
    var spherePipeline: MTLRenderPipelineState!
    var torusPipeline: MTLRenderPipelineState!
    var gridPipeline: MTLRenderPipelineState!
    var depthState: MTLDepthStencilState!
    var instances: MTLBuffer?

    weak var model: SimulationModel?

    // Orbit camera
    var azimuth: Float = 0.9
    var elevation: Float = 0.35
    var distance: Float = 30
    var target = F3(0, 0, 3)
    var viewportSize = SIMD2<Float>(1, 1)

    init(device: MTLDevice, model: SimulationModel) throws {
        self.device = device
        self.queue = device.makeCommandQueue()!
        self.model = model
        super.init()

        let lib = try device.makeLibrary(source: renderShaderSource, options: nil)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = lib.makeFunction(name: "box_vertex")
        desc.fragmentFunction = lib.makeFunction(name: "box_fragment")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.depthAttachmentPixelFormat = .depth32Float
        boxPipeline = try device.makeRenderPipelineState(descriptor: desc)

        let sdesc = MTLRenderPipelineDescriptor()
        sdesc.vertexFunction = lib.makeFunction(name: "sphere_vertex")
        sdesc.fragmentFunction = lib.makeFunction(name: "box_fragment")
        sdesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        sdesc.depthAttachmentPixelFormat = .depth32Float
        spherePipeline = try device.makeRenderPipelineState(descriptor: sdesc)

        let tdesc = MTLRenderPipelineDescriptor()
        tdesc.vertexFunction = lib.makeFunction(name: "torus_vertex")
        tdesc.fragmentFunction = lib.makeFunction(name: "box_fragment")
        tdesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        tdesc.depthAttachmentPixelFormat = .depth32Float
        torusPipeline = try device.makeRenderPipelineState(descriptor: tdesc)

        let gdesc = MTLRenderPipelineDescriptor()
        gdesc.vertexFunction = lib.makeFunction(name: "grid_vertex")
        gdesc.fragmentFunction = lib.makeFunction(name: "grid_fragment")
        gdesc.colorAttachments[0].pixelFormat = .bgra8Unorm
        gdesc.depthAttachmentPixelFormat = .depth32Float
        gridPipeline = try device.makeRenderPipelineState(descriptor: gdesc)

        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: dd)
    }

    var viewMatrix: simd_float4x4 {
        let eye = eyePosition
        return lookAt(eye: eye, center: target, up: F3(0, 0, 1))
    }

    var eyePosition: F3 {
        target + F3(
            distance * cos(elevation) * cos(azimuth),
            distance * cos(elevation) * sin(azimuth),
            distance * sin(elevation)
        )
    }

    func projectionMatrix(aspect: Float) -> simd_float4x4 {
        perspective(fovY: 50 * .pi / 180, aspect: aspect, near: 0.1, far: 1000)
    }

    /// Ray through screen point in world space.
    func ray(at point: CGPoint, in size: CGSize) -> (origin: F3, dir: F3) {
        let aspect = Float(size.width / max(size.height, 1))
        let ndcX = Float(point.x / size.width) * 2 - 1
        let ndcY = 1 - Float(point.y / size.height) * 2
        let invVP = (projectionMatrix(aspect: aspect) * viewMatrix).inverse
        var pNear = invVP * SIMD4<Float>(ndcX, ndcY, 0, 1)
        var pFar = invVP * SIMD4<Float>(ndcX, ndcY, 1, 1)
        pNear /= pNear.w
        pFar /= pFar.w
        let o = F3(pNear.x, pNear.y, pNear.z)
        let d = normalize(F3(pFar.x, pFar.y, pFar.z) - o)
        return (o, d)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        viewportSize = SIMD2(Float(size.width), Float(size.height))
    }

    func draw(in view: MTKView) {
        guard let model,
              let solver = model.solver,
              let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer() else { return }

        model.tickIfRunning()

        let count = solver.bodyCount
        let stride = MemoryLayout<simd_float4x4>.stride + 32
        let needed = count * stride
        if instances == nil || instances!.length < count * stride {
            instances = device.makeBuffer(length: max(256, count * stride), options: .storageModePrivate)
        }
        _ = needed
        guard let instances else { return }

        solver.encodeBuildInstances(cmd, instances: instances,
                                    colorMode: model.colorByGraphColor ? 1 : 0)

        let aspect = viewportSize.x / max(viewportSize.y, 1)
        let l = normalize(F3(0.4, 0.25, -0.85))
        let e = eyePosition
        var U = Uniforms(viewProj: projectionMatrix(aspect: aspect) * viewMatrix,
                         lightDir: SIMD4(l, 0),
                         eye: SIMD4(e, 0))

        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.62, green: 0.67, blue: 0.75, alpha: 1)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        enc.setDepthStencilState(depthState)

        enc.setRenderPipelineState(gridPipeline)
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: 2 * (2 * 81))

        enc.setRenderPipelineState(boxPipeline)
        enc.setVertexBuffer(instances, offset: 0, index: 0)
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36, instanceCount: count)

        enc.setRenderPipelineState(spherePipeline)
        enc.setVertexBuffer(instances, offset: 0, index: 0)
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0,
                           vertexCount: 12 * 18 * 6, instanceCount: count)

        enc.setRenderPipelineState(torusPipeline)
        enc.setVertexBuffer(instances, offset: 0, index: 0)
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0,
                           vertexCount: 24 * 12 * 6, instanceCount: count)
        enc.endEncoding()

        cmd.present(drawable)
        cmd.commit()

        framesDrawn += 1
        if framesDrawn == 30, ProcessInfo.processInfo.environment["AVBD_MARKER"] != nil {
            // Smoke-test hook: prove the render loop is alive and the sim ran.
            let info = "frames=30 bodies=\(count) stats=\(model.statsText)"
            try? info.write(toFile: "/tmp/avbd_render_marker.txt", atomically: true, encoding: .utf8)
        }
    }

    private var framesDrawn = 0
}

func lookAt(eye: F3, center: F3, up: F3) -> simd_float4x4 {
    let f = normalize(center - eye)
    let s = normalize(cross(f, up))
    let u = cross(s, f)
    return simd_float4x4(columns: (
        SIMD4(s.x, u.x, -f.x, 0),
        SIMD4(s.y, u.y, -f.y, 0),
        SIMD4(s.z, u.z, -f.z, 0),
        SIMD4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)
    ))
}

func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let y = 1 / tan(fovY * 0.5)
    let x = y / aspect
    let z = far / (near - far)
    return simd_float4x4(columns: (
        SIMD4(x, 0, 0, 0),
        SIMD4(0, y, 0, 0),
        SIMD4(0, 0, z, -1),
        SIMD4(0, 0, z * near, 0)
    ))
}
