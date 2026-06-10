import Metal
import MetalKit
import AVBDCore
import simd

// Renderer: PBR (GGX metallic-roughness) + ACES, sRGB-correct framebuffer,
// SSAO (Alchemy-style, world-position prepass), soft blob shadows,
// horizon-only fog, analytic instanced geometry for box/sphere/torus/capsule.

let renderShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct RenderInstance {
    float4x4 model;
    float4 color;       // rgb albedo (sRGB), w = shape type (0 box 1 sphere 2 torus 3 capsule)
    float4 params;      // torus/capsule dims; z = bounding radius, w = dynamic flag
};

struct Uniforms {
    float4x4 viewProj;
    float4 lightDir;    // xyz
    float4 eye;         // xyz
    float4 screen;      // x,y = drawable size; z = px per world unit at d=1
};

struct VOut {
    float4 position [[position]];
    float3 normal;
    float3 world;
    float3 albedo;
};

#define HORIZON_LIN float3(0.78, 0.81, 0.85)
#define SUN_COL (float3(1.0, 0.95, 0.86) * 3.4)
#define SKY_IRR float3(0.30, 0.33, 0.38)
#define GND_IRR float3(0.20, 0.185, 0.17)

inline float3 srgbToLin(float3 c) { return c * c * (0.7 + 0.3 * c); } // fast approx

inline float horizonFog(float d) {
    // fog ONLY near the horizon: nothing below 90m, full by 420m
    return smoothstep(90.0, 420.0, d);
}

inline float3 acesTonemap(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Geometry vertex shaders (analytic instanced shapes)
// ---------------------------------------------------------------------------
constant float3 cubeVerts[36] = {
    float3(0.5,-0.5,-0.5), float3(0.5, 0.5,-0.5), float3(0.5, 0.5, 0.5),
    float3(0.5,-0.5,-0.5), float3(0.5, 0.5, 0.5), float3(0.5,-0.5, 0.5),
    float3(-0.5,-0.5, 0.5), float3(-0.5, 0.5, 0.5), float3(-0.5, 0.5,-0.5),
    float3(-0.5,-0.5, 0.5), float3(-0.5, 0.5,-0.5), float3(-0.5,-0.5,-0.5),
    float3(-0.5, 0.5,-0.5), float3(-0.5, 0.5, 0.5), float3(0.5, 0.5, 0.5),
    float3(-0.5, 0.5,-0.5), float3(0.5, 0.5, 0.5), float3(0.5, 0.5,-0.5),
    float3(-0.5,-0.5, 0.5), float3(-0.5,-0.5,-0.5), float3(0.5,-0.5,-0.5),
    float3(-0.5,-0.5, 0.5), float3(0.5,-0.5,-0.5), float3(0.5,-0.5, 0.5),
    float3(-0.5,-0.5, 0.5), float3(0.5,-0.5, 0.5), float3(0.5, 0.5, 0.5),
    float3(-0.5,-0.5, 0.5), float3(0.5, 0.5, 0.5), float3(-0.5, 0.5, 0.5),
    float3(0.5,-0.5,-0.5), float3(-0.5,-0.5,-0.5), float3(-0.5, 0.5,-0.5),
    float3(0.5,-0.5,-0.5), float3(-0.5, 0.5,-0.5), float3(0.5, 0.5,-0.5),
};
constant float3 cubeNormals[6] = {
    float3(1,0,0), float3(-1,0,0), float3(0,1,0),
    float3(0,-1,0), float3(0,0,1), float3(0,0,-1),
};

inline VOut collapse() {
    VOut o;
    o.position = float4(0, 0, -2, 1);
    o.normal = float3(0); o.world = float3(0); o.albedo = float3(0);
    return o;
}

inline VOut emit(float3 p, float3 n, RenderInstance inst, constant Uniforms& U) {
    float4 world = inst.model * float4(p, 1);
    float3x3 rot = float3x3(normalize(inst.model[0].xyz),
                            normalize(inst.model[1].xyz),
                            normalize(inst.model[2].xyz));
    VOut o;
    o.position = U.viewProj * world;
    o.normal = rot * n;
    o.world = world.xyz;
    o.albedo = srgbToLin(inst.color.rgb);
    return o;
}

vertex VOut box_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                       device const RenderInstance* instances [[buffer(0)]],
                       constant Uniforms& U [[buffer(1)]])
{
    RenderInstance inst = instances[iid];
    if (inst.color.w != 0.0) return collapse();
    return emit(cubeVerts[vid], cubeNormals[vid / 6], inst, U);
}

#define SPH_STACKS 12
#define SPH_SLICES 18
vertex VOut sphere_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                          device const RenderInstance* instances [[buffer(0)]],
                          constant Uniforms& U [[buffer(1)]])
{
    RenderInstance inst = instances[iid];
    if (inst.color.w != 1.0) return collapse();
    uint quad = vid / 6, corner = vid % 6;
    uint stack = quad / SPH_SLICES, slice = quad % SPH_SLICES;
    uint2 off[6] = { uint2(0,0), uint2(1,0), uint2(1,1), uint2(0,0), uint2(1,1), uint2(0,1) };
    float v = float(stack + off[corner].y) / float(SPH_STACKS);
    float u = float(slice + off[corner].x) / float(SPH_SLICES);
    float phi = v * M_PI_F, theta = u * 2.0 * M_PI_F;
    float3 n = float3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
    return emit(n * 0.5, n, inst, U);
}

#define TOR_RINGS 24
#define TOR_SIDES 12
vertex VOut torus_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                         device const RenderInstance* instances [[buffer(0)]],
                         constant Uniforms& U [[buffer(1)]])
{
    RenderInstance inst = instances[iid];
    if (inst.color.w != 2.0) return collapse();
    uint quad = vid / 6, corner = vid % 6;
    uint ring = quad / TOR_SIDES, side = quad % TOR_SIDES;
    uint2 off[6] = { uint2(0,0), uint2(1,0), uint2(1,1), uint2(0,0), uint2(1,1), uint2(0,1) };
    float u = float(ring + off[corner].x) / float(TOR_RINGS) * 2.0 * M_PI_F;
    float v = float(side + off[corner].y) / float(TOR_SIDES) * 2.0 * M_PI_F;
    float R = inst.params.x, r = inst.params.y;
    float3 n = float3(cos(u) * cos(v), sin(u) * cos(v), sin(v));
    float3 p = float3(R * cos(u), R * sin(u), 0) + n * r;
    return emit(p, n, inst, U);
}

#define CAP_SLICES 16
#define CAP_STACKS 6
vertex VOut capsule_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                           device const RenderInstance* instances [[buffer(0)]],
                           constant Uniforms& U [[buffer(1)]])
{
    RenderInstance inst = instances[iid];
    if (inst.color.w != 3.0) return collapse();
    float halfL = inst.params.x * 0.5, r = inst.params.y;
    uint quad = vid / 6, corner = vid % 6;
    uint stack = quad / CAP_SLICES, slice = quad % CAP_SLICES;
    uint2 off[6] = { uint2(0,0), uint2(1,0), uint2(1,1), uint2(0,0), uint2(1,1), uint2(0,1) };
    uint sIdx = stack + off[corner].y;
    float u = float(slice + off[corner].x) / float(CAP_SLICES) * 2.0 * M_PI_F;
    float phi, zoff;
    if (sIdx <= CAP_STACKS) {
        phi = -M_PI_F / 2.0 + float(sIdx) / float(CAP_STACKS) * (M_PI_F / 2.0);
        zoff = -halfL;
    } else {
        uint k = sIdx - CAP_STACKS - 1;
        phi = float(k) / float(CAP_STACKS) * (M_PI_F / 2.0);
        zoff = halfL;
    }
    float3 n = float3(cos(phi) * cos(u), cos(phi) * sin(u), sin(phi));
    return emit(n * r + float3(0, 0, zoff), n, inst, U);
}

// ---------------------------------------------------------------------------
// PBR main fragment (samples the SSAO texture)
// ---------------------------------------------------------------------------
fragment float4 pbr_fragment(VOut in [[stage_in]],
                             constant Uniforms& U [[buffer(1)]],
                             texture2d<float> aoTex [[texture(0)]])
{
    constexpr sampler smp(filter::linear);
    float ao = aoTex.sample(smp, in.position.xy / U.screen.xy).r;

    float3 n = normalize(in.normal);
    float3 L = -U.lightDir.xyz;
    float3 V = normalize(U.eye.xyz - in.world);
    float3 H = normalize(L + V);
    float NdL = max(dot(n, L), 0.0);
    float NdV = max(dot(n, V), 1e-4);
    float NdH = max(dot(n, H), 0.0);
    float HdV = max(dot(H, V), 0.0);

    const float rough = 0.45;
    float a2 = rough * rough; a2 *= a2;
    float dDen = NdH * NdH * (a2 - 1.0) + 1.0;
    float D = a2 / (M_PI_F * dDen * dDen);
    float k = (rough + 1.0); k = k * k / 8.0;
    float G = (NdV / (NdV * (1.0 - k) + k)) * (NdL / (NdL * (1.0 - k) + k));
    float3 F0 = float3(0.04);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - HdV, 5.0);
    float3 spec = D * G * F / max(4.0 * NdV * NdL, 1e-4);

    float3 direct = (in.albedo / M_PI_F * (1.0 - F) + spec) * SUN_COL * NdL;

    float3 irr = mix(GND_IRR, SKY_IRR, n.z * 0.5 + 0.5);
    float3 ambient = in.albedo * irr;
    float3 R = reflect(-V, n);
    float3 skyRef = mix(HORIZON_LIN, float3(0.42, 0.48, 0.58), clamp(R.z, 0.0, 1.0));
    ambient += skyRef * (F0 + (1.0 - F0) * pow(1.0 - NdV, 5.0)) * 0.30;
    ambient *= ao;

    float3 lit = direct + ambient;
    float fog = horizonFog(length(in.world - U.eye.xyz));
    lit = mix(lit, HORIZON_LIN, fog);
    return float4(acesTonemap(lit), 1.0);
}

// ---------------------------------------------------------------------------
// Prepass: world position + normal into two attachments (for SSAO)
// ---------------------------------------------------------------------------
struct PreOut {
    float4 pos  [[color(0)]];   // xyz world, w = 1 (geometry present)
    float4 norm [[color(1)]];
};

fragment PreOut prepass_fragment(VOut in [[stage_in]]) {
    PreOut o;
    o.pos = float4(in.world, 1.0);
    o.norm = float4(normalize(in.normal), 0);
    return o;
}

// ---------------------------------------------------------------------------
// SSAO (Alchemy / point-based estimator over a spiral kernel)
// ---------------------------------------------------------------------------
struct FSOut { float4 position [[position]]; float2 uv; };

vertex FSOut fs_vertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2) * 2.0 - 1.0;
    FSOut o;
    o.position = float4(p, 1.0, 1.0);
    o.uv = p * float2(0.5, -0.5) + 0.5;
    return o;
}

fragment float4 ssao_fragment(FSOut in [[stage_in]],
                              constant Uniforms& U [[buffer(1)]],
                              texture2d<float> posTex [[texture(0)]],
                              texture2d<float> normTex [[texture(1)]])
{
    constexpr sampler smp(filter::nearest, address::clamp_to_edge);
    float4 P4 = posTex.sample(smp, in.uv);
    if (P4.w < 0.5) return float4(1);
    float3 P = P4.xyz;
    float3 N = normTex.sample(smp, in.uv).xyz;
    float dist = length(P - U.eye.xyz);

    // world-space AO radius, projected to a pixel radius
    const float R = 0.9;
    float pxRadius = clamp(U.screen.z * R / max(dist, 0.5), 3.0, 90.0);

    // stable per-pixel rotation (position-hash, no temporal flicker)
    float ang = fract(sin(dot(floor(in.position.xy), float2(12.9898, 78.233)))
                      * 43758.5453) * 6.2831853;
    float ca = cos(ang), sa = sin(ang);

    const int NS = 10;
    float occ = 0.0;
    for (int i = 0; i < NS; i++) {
        float t = (float(i) + 0.5) / float(NS);
        float a = t * 6.2831853 * 2.0;     // 2 spiral turns
        float rad = t * pxRadius;
        float2 off = float2(cos(a) * ca - sin(a) * sa,
                            cos(a) * sa + sin(a) * ca) * rad;
        float2 uv2 = in.uv + off / U.screen.xy;
        float4 Q4 = posTex.sample(smp, uv2);
        if (Q4.w < 0.5) continue;
        float3 v = Q4.xyz - P;
        float d2 = dot(v, v);
        // Alchemy estimator: occlusion from samples above the surface
        occ += max(0.0, dot(N, v) - 0.02 * dist) / (d2 + 0.08);
    }
    float ao = clamp(1.0 - occ * (1.6 / float(NS)), 0.0, 1.0);
    ao = pow(ao, 1.4);
    return float4(ao, ao, ao, 1);
}

fragment float4 blur_fragment(FSOut in [[stage_in]],
                              constant Uniforms& U [[buffer(1)]],
                              texture2d<float> src [[texture(0)]])
{
    constexpr sampler smp(filter::linear, address::clamp_to_edge);
    float2 px = 1.0 / U.screen.xy;
    float s = 0.0;
    for (int y = -1; y <= 2; y++) {
        for (int x = -1; x <= 2; x++) {
            s += src.sample(smp, in.uv + float2(float(x) - 0.5, float(y) - 0.5) * px).r;
        }
    }
    return float4(s / 16.0);
}

// ---------------------------------------------------------------------------
// Sky (whitish), floor (AA checker, melts then fogs), blob shadows
// ---------------------------------------------------------------------------
fragment float4 sky_fragment(FSOut in [[stage_in]]) {
    float t = 1.0 - in.uv.y;
    float3 horizon = HORIZON_LIN;
    float3 zenith  = float3(0.50, 0.56, 0.66);
    float3 c = mix(horizon, zenith, pow(clamp(t, 0.0, 1.0), 1.7));
    float2 ndc = float2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    float glow = exp(-3.0 * distance(ndc, float2(-0.45, 0.55)));
    c += float3(0.30, 0.25, 0.16) * glow;
    return float4(acesTonemap(c), 1);
}

struct FloorOut { float4 position [[position]]; float3 world; };

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

struct FloorPre {
    float4 pos  [[color(0)]];
    float4 norm [[color(1)]];
};
fragment FloorPre floor_prepass_fragment(FloorOut in [[stage_in]]) {
    FloorPre o;
    o.pos = float4(in.world, 1.0);
    o.norm = float4(0, 0, 1, 0);
    return o;
}

fragment float4 floor_fragment(FloorOut in [[stage_in]],
                               constant Uniforms& U [[buffer(1)]],
                               texture2d<float> aoTex [[texture(0)]])
{
    constexpr sampler smp(filter::linear);
    float ao = aoTex.sample(smp, in.position.xy / U.screen.xy).r;

    float2 c = in.world.xy / 2.0;
    float2 fw = fwidth(c);
    float2 fc = fract(c);
    float2 aa = clamp((fc - 0.5) / max(fw, 0.0001) + 0.5, 0.0, 1.0)
              - clamp(fc / max(fw, 0.0001), 0.0, 1.0) + 1.0;
    float2 sq = abs(aa - 1.0);
    float checker = abs(sq.x - sq.y);
    float3 tileA = srgbToLin(float3(0.93, 0.93, 0.94));
    float3 tileB = srgbToLin(float3(0.62, 0.66, 0.72));
    float3 albedo = mix(tileA, tileB, checker);
    float3 avg = (tileA + tileB) * 0.5;

    float d = length(in.world.xy - U.eye.xy);
    // melt the pattern with distance (anti-moire softness)
    albedo = mix(albedo, avg, smoothstep(25.0, 160.0, d));

    float3 L = -U.lightDir.xyz;
    float NdL = max(L.z, 0.0);
    float3 lit = albedo * (SKY_IRR * 1.1 * ao + SUN_COL / M_PI_F * NdL * 0.85);

    float fog = horizonFog(d);
    lit = mix(lit, HORIZON_LIN, fog);
    return float4(acesTonemap(lit), 1);
}

struct ShadowOut {
    float4 position [[position]];
    float2 local;
    float strength;
};

vertex ShadowOut shadow_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                               device const RenderInstance* instances [[buffer(0)]],
                               constant Uniforms& U [[buffer(1)]])
{
    RenderInstance inst = instances[iid];
    float r = inst.params.z;
    float3 c = inst.model[3].xyz;
    float h = max(c.z - r * 0.5, 0.0);
    float strength = inst.params.w * clamp(1.0 - h / 8.0, 0.0, 1.0) * 0.26;
    float size = r * (1.45 + h * 0.10);
    float2 corners[6] = {
        float2(-1,-1), float2(1,-1), float2(1,1),
        float2(-1,-1), float2(1,1), float2(-1,1)
    };
    ShadowOut o;
    if (strength < 0.012 || r <= 0.0) {
        o.position = float4(0, 0, -2, 1);
        o.local = float2(0); o.strength = 0;
        return o;
    }
    float3 p = float3(c.xy + corners[vid] * size, 0.03);
    o.position = U.viewProj * float4(p, 1);
    o.local = corners[vid];
    o.strength = strength;
    return o;
}

fragment float4 shadow_fragment(ShadowOut in [[stage_in]]) {
    float d = length(in.local);
    // wide gaussian falloff: soft penumbra, no hard rim
    float a = in.strength * exp(-d * d * 3.2);
    return float4(0.05, 0.06, 0.09, a);
}
"""

struct Uniforms {
    var viewProj: simd_float4x4
    var lightDir: SIMD4<Float>
    var eye: SIMD4<Float>
    var screen: SIMD4<Float>
}

let SPHV = 12 * 18 * 6
let TORV = 24 * 12 * 6
let CAPV = (2 * 6 + 1) * 16 * 6

final class Renderer: NSObject, MTKViewDelegate {
    static let sampleCount = 4
    static let colorFormat = MTLPixelFormat.bgra8Unorm_srgb

    let device: MTLDevice
    let queue: MTLCommandQueue
    weak var model: SimulationModel?

    var boxP, sphereP, torusP, capsuleP: MTLRenderPipelineState!
    var boxPre, spherePre, torusPre, capsulePre, floorPreP: MTLRenderPipelineState!
    var skyP, floorP, shadowP, ssaoP, blurP: MTLRenderPipelineState!
    var depthState, noDepthState, noWriteDepthState: MTLDepthStencilState!

    var posTex, normTex, aoTexA, aoTexB, preDepthTex: MTLTexture?
    var targetSize = SIMD2<Int>(0, 0)
    var instances: MTLBuffer?

    var azimuth: Float = 0.9
    var elevation: Float = 0.35
    var distance: Float = 30
    var target = F3(0, 0, 3)
    var viewportSize = SIMD2<Float>(1, 1)
    private var framesDrawn = 0

    init(device: MTLDevice, model: SimulationModel) throws {
        self.device = device
        self.queue = device.makeCommandQueue()!
        self.model = model
        super.init()

        let lib = try device.makeLibrary(source: renderShaderSource, options: nil)
        func pipe(_ v: String, _ f: String, samples: Int = Renderer.sampleCount,
                  blend: Bool = false,
                  colorFormats: [MTLPixelFormat] = [Renderer.colorFormat],
                  depth: MTLPixelFormat = .depth32Float) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = lib.makeFunction(name: v)!
            d.fragmentFunction = lib.makeFunction(name: f)!
            for (i, fmt) in colorFormats.enumerated() {
                d.colorAttachments[i].pixelFormat = fmt
                if blend && i == 0 {
                    d.colorAttachments[i].isBlendingEnabled = true
                    d.colorAttachments[i].sourceRGBBlendFactor = .sourceAlpha
                    d.colorAttachments[i].destinationRGBBlendFactor = .oneMinusSourceAlpha
                }
            }
            d.depthAttachmentPixelFormat = depth
            d.rasterSampleCount = samples
            return try device.makeRenderPipelineState(descriptor: d)
        }

        boxP = try pipe("box_vertex", "pbr_fragment")
        sphereP = try pipe("sphere_vertex", "pbr_fragment")
        torusP = try pipe("torus_vertex", "pbr_fragment")
        capsuleP = try pipe("capsule_vertex", "pbr_fragment")
        skyP = try pipe("fs_vertex", "sky_fragment")
        floorP = try pipe("floor_vertex", "floor_fragment")
        shadowP = try pipe("shadow_vertex", "shadow_fragment", blend: true)

        let preFmt: [MTLPixelFormat] = [.rgba32Float, .rgba16Float]
        boxPre = try pipe("box_vertex", "prepass_fragment", samples: 1, colorFormats: preFmt)
        spherePre = try pipe("sphere_vertex", "prepass_fragment", samples: 1, colorFormats: preFmt)
        torusPre = try pipe("torus_vertex", "prepass_fragment", samples: 1, colorFormats: preFmt)
        capsulePre = try pipe("capsule_vertex", "prepass_fragment", samples: 1, colorFormats: preFmt)
        floorPreP = try pipe("floor_vertex", "floor_prepass_fragment", samples: 1, colorFormats: preFmt)
        ssaoP = try pipe("fs_vertex", "ssao_fragment", samples: 1,
                         colorFormats: [.r8Unorm], depth: .invalid)
        blurP = try pipe("fs_vertex", "blur_fragment", samples: 1,
                         colorFormats: [.r8Unorm], depth: .invalid)

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

    private func ensureTargets(_ size: CGSize) {
        let w = max(Int(size.width), 4), h = max(Int(size.height), 4)
        if targetSize == SIMD2(w, h), posTex != nil { return }
        targetSize = SIMD2(w, h)
        func tex(_ fmt: MTLPixelFormat) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt,
                                                             width: w, height: h,
                                                             mipmapped: false)
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)!
        }
        posTex = tex(.rgba32Float)
        normTex = tex(.rgba16Float)
        aoTexA = tex(.r8Unorm)
        aoTexB = tex(.r8Unorm)
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float,
                                                          width: w, height: h,
                                                          mipmapped: false)
        dd.usage = .renderTarget
        dd.storageMode = .private
        preDepthTex = device.makeTexture(descriptor: dd)
    }

    // MARK: - Camera

    var viewMatrix: simd_float4x4 {
        lookAt(eye: eyePosition, center: target, up: F3(0, 0, 1))
    }

    var eyePosition: F3 {
        target + F3(distance * cos(elevation) * cos(azimuth),
                    distance * cos(elevation) * sin(azimuth),
                    distance * sin(elevation))
    }

    func projectionMatrix(aspect: Float) -> simd_float4x4 {
        perspective(fovY: 50 * .pi / 180, aspect: aspect, near: 0.1, far: 1000)
    }

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

    // MARK: - Frame

    func draw(in view: MTKView) {
        guard let model,
              let solver = model.solver,
              let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer() else { return }

        model.tickIfRunning()
        ensureTargets(view.drawableSize)

        let count = solver.bodyCount
        let stride = MemoryLayout<simd_float4x4>.stride + 32
        if instances == nil || instances!.length < count * stride {
            instances = device.makeBuffer(length: max(256, count * stride),
                                          options: .storageModePrivate)
        }
        guard let instances, let posTex, let normTex, let aoTexA, let aoTexB,
              let preDepthTex else { return }

        solver.encodeBuildInstances(cmd, instances: instances,
                                    colorMode: model.colorByGraphColor ? 1 : 0)

        let aspect = viewportSize.x / max(viewportSize.y, 1)
        let pxPerUnit = viewportSize.y * 0.5 / tan(25 * Float.pi / 180)
        var U = Uniforms(viewProj: projectionMatrix(aspect: aspect) * viewMatrix,
                         lightDir: SIMD4(normalize(F3(0.4, 0.25, -0.85)), 0),
                         eye: SIMD4(eyePosition, 0),
                         screen: SIMD4(viewportSize.x, viewportSize.y, pxPerUnit, 0))

        // ---- 1. prepass: world pos + normals ----
        let pre = MTLRenderPassDescriptor()
        pre.colorAttachments[0].texture = posTex
        pre.colorAttachments[0].loadAction = .clear
        pre.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pre.colorAttachments[0].storeAction = .store
        pre.colorAttachments[1].texture = normTex
        pre.colorAttachments[1].loadAction = .clear
        pre.colorAttachments[1].storeAction = .store
        pre.depthAttachment.texture = preDepthTex
        pre.depthAttachment.loadAction = .clear
        pre.depthAttachment.clearDepth = 1
        pre.depthAttachment.storeAction = .dontCare
        if let enc = cmd.makeRenderCommandEncoder(descriptor: pre) {
            enc.setDepthStencilState(depthState)
            enc.setRenderPipelineState(floorPreP)
            enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            for (p, verts) in [(boxPre!, 36), (spherePre!, SPHV),
                               (torusPre!, TORV), (capsulePre!, CAPV)] {
                enc.setRenderPipelineState(p)
                enc.setVertexBuffer(instances, offset: 0, index: 0)
                enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: verts, instanceCount: count)
            }
            enc.endEncoding()
        }

        // ---- 2. SSAO ----
        let aoPass = MTLRenderPassDescriptor()
        aoPass.colorAttachments[0].texture = aoTexA
        aoPass.colorAttachments[0].loadAction = .dontCare
        aoPass.colorAttachments[0].storeAction = .store
        if let enc = cmd.makeRenderCommandEncoder(descriptor: aoPass) {
            enc.setRenderPipelineState(ssaoP)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(posTex, index: 0)
            enc.setFragmentTexture(normTex, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }

        // ---- 3. blur AO ----
        let blurPass = MTLRenderPassDescriptor()
        blurPass.colorAttachments[0].texture = aoTexB
        blurPass.colorAttachments[0].loadAction = .dontCare
        blurPass.colorAttachments[0].storeAction = .store
        if let enc = cmd.makeRenderCommandEncoder(descriptor: blurPass) {
            enc.setRenderPipelineState(blurP)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(aoTexA, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }

        // ---- 4. main pass ----
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }

        enc.setDepthStencilState(noDepthState)
        enc.setRenderPipelineState(skyP)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        enc.setDepthStencilState(depthState)
        enc.setRenderPipelineState(floorP)
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentTexture(aoTexB, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

        enc.setDepthStencilState(noWriteDepthState)
        enc.setRenderPipelineState(shadowP)
        enc.setVertexBuffer(instances, offset: 0, index: 0)
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6,
                           instanceCount: count)

        enc.setDepthStencilState(depthState)
        for (p, verts) in [(boxP!, 36), (sphereP!, SPHV), (torusP!, TORV), (capsuleP!, CAPV)] {
            enc.setRenderPipelineState(p)
            enc.setVertexBuffer(instances, offset: 0, index: 0)
            enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(aoTexB, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: verts, instanceCount: count)
        }
        enc.endEncoding()

        cmd.present(drawable)
        cmd.commit()

        framesDrawn += 1
        if framesDrawn == 30, ProcessInfo.processInfo.environment["AVBD_MARKER"] != nil {
            let info = "frames=30 bodies=\(count) stats=\(model.statsText)"
            try? info.write(toFile: "/tmp/avbd_render_marker.txt", atomically: true, encoding: .utf8)
        }
    }
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
