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

// ---------- PBR (metallic-roughness, GGX + Schlick) + ACES ----------
inline float3 acesTonemap(float3 x) {
    // Narkowicz ACES approximation
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

#define HORIZON_COL float3(0.93, 0.94, 0.96)
#define SUN_COL (float3(1.0, 0.96, 0.88) * 3.2)
#define SKY_IRR float3(0.62, 0.66, 0.72)
#define GND_IRR float3(0.52, 0.50, 0.47)

inline float3 pbrShade(float3 albedo, float3 n, float3 world, float3 eye, float3 L,
                       float rough, float metal)
{
    float3 V = normalize(eye - world);
    float3 H = normalize(L + V);
    float NdL = max(dot(n, L), 0.0);
    float NdV = max(dot(n, V), 1e-4);
    float NdH = max(dot(n, H), 0.0);
    float HdV = max(dot(H, V), 0.0);

    float a2 = rough * rough; a2 *= a2;
    float dDen = NdH * NdH * (a2 - 1.0) + 1.0;
    float D = a2 / (M_PI_F * dDen * dDen);
    float k = (rough + 1.0); k = k * k / 8.0;
    float G = (NdV / (NdV * (1.0 - k) + k)) * (NdL / (NdL * (1.0 - k) + k));
    float3 F0 = mix(float3(0.04), albedo, metal);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - HdV, 5.0);
    float3 spec = D * G * F / max(4.0 * NdV * NdL, 1e-4);
    float3 kd = (1.0 - F) * (1.0 - metal);

    float3 direct = (kd * albedo / M_PI_F + spec) * SUN_COL * NdL;

    // hemispheric irradiance + fresnel-weighted sky reflection
    float3 irr = mix(GND_IRR, SKY_IRR, n.z * 0.5 + 0.5);
    float3 ambient = kd * albedo * irr;
    float3 R = reflect(-V, n);
    float3 skyRef = mix(HORIZON_COL * 0.9, float3(0.72, 0.78, 0.88),
                        clamp(R.z, 0.0, 1.0));
    float fr = pow(1.0 - NdV, 5.0);
    ambient += skyRef * (F0 + (1.0 - F0) * fr) * (1.0 - rough * 0.7) * 0.5;

    return direct + ambient;
}

fragment float4 box_fragment(VOut in [[stage_in]],
                             constant Uniforms& U [[buffer(1)]])
{
    float3 n = normalize(in.normal);
    float3 lit = pbrShade(in.color.rgb, n, in.world, U.eye.xyz,
                          -U.lightDir.xyz, 0.42, 0.0);
    float d = length(in.world - U.eye.xyz);
    float fog = 1.0 - exp(-d * 0.010);
    lit = mix(lit, HORIZON_COL, fog * fog);
    return float4(acesTonemap(lit), 1.0);
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

// Analytic capsule: cylinder (RINGS_C x SLICES_C) + two hemisphere caps
#define CAP_SLICES 16
#define CAP_STACKS 6

vertex VOut capsule_vertex(uint vid [[vertex_id]],
                           uint iid [[instance_id]],
                           device const RenderInstance* instances [[buffer(0)]],
                           constant Uniforms& U [[buffer(1)]])
{
    RenderInstance inst = instances[iid];
    if (inst.color.w != 3.0) {
        VOut o;
        o.position = float4(0, 0, -2, 1);
        o.normal = float3(0); o.world = float3(0); o.color = float4(0);
        return o;
    }
    float halfL = inst.params.x * 0.5;
    float r = inst.params.y;
    // total stacks: CAP_STACKS (bottom cap) + 1 (cylinder) + CAP_STACKS (top)
    uint quad = vid / 6;
    uint corner = vid % 6;
    uint stack = quad / CAP_SLICES;
    uint slice = quad % CAP_SLICES;
    uint2 off[6] = { uint2(0,0), uint2(1,0), uint2(1,1), uint2(0,0), uint2(1,1), uint2(0,1) };
    uint sIdx = stack + off[corner].y;
    float u = float(slice + off[corner].x) / float(CAP_SLICES) * 2.0 * M_PI_F;
    float phi;      // -pi/2..pi/2 latitude
    float zoff;
    if (sIdx <= CAP_STACKS) {
        phi = -M_PI_F / 2.0 + float(sIdx) / float(CAP_STACKS) * (M_PI_F / 2.0);
        zoff = -halfL;
    } else {
        uint k = sIdx - CAP_STACKS - 1;
        phi = float(k) / float(CAP_STACKS) * (M_PI_F / 2.0);
        zoff = halfL;
    }
    float3 n = float3(cos(phi) * cos(u), cos(phi) * sin(u), sin(phi));
    float3 p = n * r + float3(0, 0, zoff);
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

// --- Soft blob shadows: one ground quad per body, radial falloff ---
struct ShadowOut {
    float4 position [[position]];
    float2 local;
    float strength;
};

vertex ShadowOut shadow_vertex(uint vid [[vertex_id]],
                               uint iid [[instance_id]],
                               device const RenderInstance* instances [[buffer(0)]],
                               constant Uniforms& U [[buffer(1)]])
{
    RenderInstance inst = instances[iid];
    float r = inst.params.z;               // bounding radius
    float3 c = inst.model[3].xyz;
    float h = max(c.z - r * 0.5, 0.0);     // height above ground
    float strength = inst.params.w * clamp(1.0 - h / 7.0, 0.0, 1.0) * 0.42;
    float size = r * (1.25 + h * 0.08);
    float2 corners[6] = {
        float2(-1,-1), float2(1,-1), float2(1,1),
        float2(-1,-1), float2(1,1), float2(-1,1)
    };
    float2 l = corners[vid];
    ShadowOut o;
    if (strength < 0.01 || r <= 0.0) {
        o.position = float4(0, 0, -2, 1);
        o.local = float2(0); o.strength = 0;
        return o;
    }
    float3 p = float3(c.xy + l * size, 0.012);
    o.position = U.viewProj * float4(p, 1);
    o.local = l;
    o.strength = strength;
    return o;
}

fragment float4 shadow_fragment(ShadowOut in [[stage_in]]) {
    float d = length(in.local);
    float a = in.strength * smoothstep(1.0, 0.25, d);
    return float4(0.18, 0.20, 0.26, a);
}

// --- Sky: fullscreen gradient (drawn first, no depth) ---
struct SkyOut { float4 position [[position]]; float2 uv; };

vertex SkyOut sky_vertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2) * 2.0 - 1.0;   // fullscreen tri
    SkyOut o;
    o.position = float4(p, 1.0, 1.0);
    o.uv = p;
    return o;
}

fragment float4 sky_fragment(SkyOut in [[stage_in]]) {
    float t = clamp(in.uv.y * 0.5 + 0.5, 0.0, 1.0);
    float3 horizon = float3(0.93, 0.94, 0.96);
    float3 zenith  = float3(0.72, 0.78, 0.87);
    float3 c = mix(horizon, zenith, pow(t, 1.6));
    // soft warm sun glow upper-left
    float2 sunDir = float2(-0.45, 0.55);
    float glow = exp(-3.0 * distance(in.uv, sunDir));
    c += float3(0.20, 0.17, 0.11) * glow;
    return float4(c, 1);
}

// --- Checkerboard floor: huge quad, AA checker, fades into the horizon ---
struct FloorOut {
    float4 position [[position]];
    float3 world;
};

vertex FloorOut floor_vertex(uint vid [[vertex_id]],
                             constant Uniforms& U [[buffer(1)]])
{
    const float E = 4000.0;
    float2 corners[6] = {
        float2(-E,-E), float2(E,-E), float2(E,E),
        float2(-E,-E), float2(E,E), float2(-E,E)
    };
    float3 p = float3(corners[vid], 0.005);
    FloorOut o;
    o.position = U.viewProj * float4(p, 1);
    o.world = p;
    return o;
}

fragment float4 floor_fragment(FloorOut in [[stage_in]],
                               constant Uniforms& U [[buffer(1)]])
{
    // anti-aliased checker via filter-width smoothing
    float2 c = in.world.xy / 2.0;
    float2 fw = fwidth(c);
    float2 fc = fract(c);
    float2 aa = clamp((fc - 0.5) / max(fw, 0.0001) + 0.5, 0.0, 1.0)
              - clamp(fc / max(fw, 0.0001) - 0.0, 0.0, 1.0) + 1.0;
    float2 sq = abs(aa - 1.0);
    float checker = abs(sq.x - sq.y);
    float3 tileA = float3(0.90, 0.905, 0.92);
    float3 tileB = float3(0.72, 0.755, 0.80);
    float3 avg = (tileA + tileB) * 0.5;
    float3 col = mix(tileA, tileB, checker);

    // distance "blur": melt the checker toward its average long before the
    // fog finishes — reads as depth-of-field fuzziness, kills the moire
    float d = length(in.world.xy - U.eye.xy);
    float blur = 1.0 - exp(-d * 0.022);
    col = mix(col, avg, blur);

    // full fog into the horizon color: by ~350m the floor IS the sky, so
    // there is no visible horizon step anywhere
    float3 horizon = float3(0.93, 0.94, 0.96);
    float fog = 1.0 - exp(-d * 0.011);
    col = mix(col, horizon, smoothstep(0.0, 1.0, fog));
    return float4(col, 1);
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
    var capsulePipeline: MTLRenderPipelineState!
    var skyPipeline: MTLRenderPipelineState!
    var floorPipeline: MTLRenderPipelineState!
    var shadowPipeline: MTLRenderPipelineState!
    static let sampleCount = 4
    var depthState: MTLDepthStencilState!
    var noDepthState: MTLDepthStencilState!
    var noWriteDepthState: MTLDepthStencilState!
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
        func pipe(_ v: String, _ f: String) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = lib.makeFunction(name: v)
            d.fragmentFunction = lib.makeFunction(name: f)
            d.colorAttachments[0].pixelFormat = .bgra8Unorm
            d.depthAttachmentPixelFormat = .depth32Float
            d.rasterSampleCount = Self.sampleCount
            return try device.makeRenderPipelineState(descriptor: d)
        }
        boxPipeline = try pipe("box_vertex", "box_fragment")

        spherePipeline = try pipe("sphere_vertex", "box_fragment")
        torusPipeline = try pipe("torus_vertex", "box_fragment")
        capsulePipeline = try pipe("capsule_vertex", "box_fragment")
        skyPipeline = try pipe("sky_vertex", "sky_fragment")
        floorPipeline = try pipe("floor_vertex", "floor_fragment")

        let shd = MTLRenderPipelineDescriptor()
        shd.vertexFunction = lib.makeFunction(name: "shadow_vertex")
        shd.fragmentFunction = lib.makeFunction(name: "shadow_fragment")
        shd.colorAttachments[0].pixelFormat = .bgra8Unorm
        shd.colorAttachments[0].isBlendingEnabled = true
        shd.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        shd.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        shd.depthAttachmentPixelFormat = .depth32Float
        shd.rasterSampleCount = Self.sampleCount
        shadowPipeline = try device.makeRenderPipelineState(descriptor: shd)

        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: dd)

        let nd = MTLDepthStencilDescriptor()
        nd.depthCompareFunction = .always
        nd.isDepthWriteEnabled = false
        noDepthState = device.makeDepthStencilState(descriptor: nd)

        let nw = MTLDepthStencilDescriptor()
        nw.depthCompareFunction = .less
        nw.isDepthWriteEnabled = false
        noWriteDepthState = device.makeDepthStencilState(descriptor: nw)
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

        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.87, green: 0.93, blue: 1.0, alpha: 1)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }

        // sky first (depth test/write off via noDepth state)
        enc.setDepthStencilState(noDepthState)
        enc.setRenderPipelineState(skyPipeline)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        enc.setDepthStencilState(depthState)
        enc.setRenderPipelineState(floorPipeline)
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

        // soft blob shadows (read depth, don't write)
        enc.setDepthStencilState(noWriteDepthState)
        enc.setRenderPipelineState(shadowPipeline)
        enc.setVertexBuffer(instances, offset: 0, index: 0)
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: count)
        enc.setDepthStencilState(depthState)

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

        enc.setRenderPipelineState(capsulePipeline)
        enc.setVertexBuffer(instances, offset: 0, index: 0)
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0,
                           vertexCount: (2 * 6 + 1) * 16 * 6, instanceCount: count)
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
