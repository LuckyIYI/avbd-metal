import Metal
import MetalKit

/// What the renderer needs from a driving model — lets the same renderer
/// serve both the physics playground and the Robotics Lab.
@MainActor
protocol RenderableModel {
    /// Stable routing key for screenshots/video. SwiftUI may instantiate and
    /// render hidden tabs, so capture must never be selected merely because a
    /// renderer exists.
    nonisolated var captureID: String { get }
    var solver: GPUSolver? { get }
    var colorByGraphColor: Bool { get }
    var statsText: String { get }
    /// Increments only when the DEMO changes — reset/size/param rebuilds
    /// keep the user's camera.
    var cameraEpoch: Int { get }
    func tickIfRunning()
    /// Stop the driving model and expose a renderer failure through that
    /// model's existing status UI. Render failures are deliberately separate
    /// from GPUSolver's terminal physics health.
    func reportRenderFailure(_ message: String)
}

extension SimulationModel {
    func reportRenderFailure(_ message: String) {
        running = false
        statsText = "render stopped: \(message)"
    }
}
import SimCore
import PhysicsAVBD
import Robotics
import RL
import MLXRL
import simd

// Renderer: PBR (GGX metallic-roughness) + ACES, sRGB-correct framebuffer,
// GTAO-style ambient occlusion from a world-position prepass, directional
// shadow mapping, horizon-only fog, and analytic instanced geometry.

let renderShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct RenderInstance {
    float4x4 model;
    float4 color;       // rgb albedo (sRGB), w = shape type (0 box 1 sphere 2 torus 3 capsule)
    float4 params;      // torus/capsule dims; z = bounding radius, w = dynamic flag
};

struct SkinRenderVertex {
    float4 position;
    float4 normal;
};

struct RigidMeshVertex {
    float4 positionBody; // xyz body-local; w stores uint body id bits
    float4 normal;
    float4 color;
};

struct Uniforms {
    float4x4 viewProj;
    float4 lightDir;    // xyz
    float4 eye;         // xyz
    float4 screen;      // x,y = drawable size; z = px per world unit at d=1
    float4 camRight;    // xyz: world dir of screen +x
    float4 camUp;       // xyz: world dir of UV +y (down on screen)
    float4x4 prevViewProj;
    float4 temporal;    // x = per-frame noise offset, y = history blend
    float4x4 shadowViewProj;
    float4 shadowParams; // x = constant bias, y = normal/slope bias
};

struct VOut {
    float4 position [[position]];
    float3 normal;
    float3 world;
    float3 albedo;
    float flatShade;
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

// Three-by-three comparison-filtered directional shadow. The light projection
// follows the camera target, so the useful texel density stays around the robot
// instead of being spent on the effectively infinite floor.
inline float shadowVisibility(float3 world, float3 normal,
                              constant Uniforms& U,
                              depth2d<float> shadowTex)
{
    float4 clip = U.shadowViewProj * float4(world, 1.0);
    if (clip.w <= 0.0) return 1.0;
    float3 ndc = clip.xyz / clip.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uv <= 0.0) || any(uv >= 1.0) || ndc.z <= 0.0 || ndc.z >= 1.0) {
        return 1.0;
    }

    float NdL = saturate(dot(normalize(normal), -U.lightDir.xyz));
    float receiverDepth = ndc.z - (U.shadowParams.x
                                  + U.shadowParams.y * (1.0 - NdL));
    float2 texel = 1.0 / float2(shadowTex.get_width(), shadowTex.get_height());
    constexpr sampler shadowSampler(coord::normalized, filter::linear,
                                    address::clamp_to_edge,
                                    compare_func::less_equal);
    float visible = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            visible += shadowTex.sample_compare(
                shadowSampler, uv + float2(float(x), float(y)) * texel,
                receiverDepth);
        }
    }
    // A small floor keeps shaded faces readable without washing out the
    // articulated silhouette. Indirect light remains governed by GTAO below.
    return mix(0.16, 1.0, visible / 9.0);
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
    o.flatShade = 0;
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
    o.flatShade = 0;
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
// Soft surface meshes (cloth sheets, tet-body boundaries): vertices live in
// the solver's posLin buffer, smooth normals come from the soft_normals
// kernel, color from the connected-component id packed in the corner.
// ---------------------------------------------------------------------------
inline float3 softPalette(uint i) {
    float h = fract(float(i) * 0.61803398875f);
    float3 p = abs(fract(h + float3(5.0f, 3.0f, 1.0f) / 6.0f) * 6.0f - 3.0f) - 1.0f;
    return clamp(p, 0.0f, 1.0f);
}

vertex VOut soft_vertex(uint vid [[vertex_id]],
                        device const uint* corners [[buffer(0)]],
                        constant Uniforms& U [[buffer(1)]],
                        device const float4* posLin [[buffer(2)]],
                        device const float4* normals [[buffer(3)]])
{
    uint packed = corners[vid];
    uint body = packed & 0x001FFFFFu;
    uint side = (packed >> 23) & 1u;
    uint rim = (packed >> 22) & 1u;
    uint comp = packed >> 24;
    float4 nt = normals[body];               // xyz smooth normal, w thickness
    // zero thickness = flat sheet (the default): only the front layer
    // exists — the back layer would z-fight and the hem rims degenerate
    if (nt.w == 0.0 && (side == 1u || rim == 1u)) return collapse();
    // extrude along the smooth normal: front +r, back -r — at full scale
    // the render skin matches the contact skin, so layered cloth touches
    float3 w = posLin[body].xyz + nt.xyz * (nt.w * (side == 0 ? 1.0 : -1.0) * 0.95);
    VOut o;
    o.position = U.viewProj * float4(w, 1);
    o.world = w;
    o.normal = nt.xyz;
    o.flatShade = ((packed >> 21) & 1u) != 0 ? 1.0 : 0.0;
    // warm fabric-ish tones, one per sheet/body
    o.albedo = srgbToLin(mix(float3(0.90), softPalette(comp * 7u + 2u), 0.72));
    return o;
}

vertex VOut skin_vertex(uint vid [[vertex_id]],
                        device const uint* corners [[buffer(0)]],
                        constant Uniforms& U [[buffer(1)]],
                        device const SkinRenderVertex* verts [[buffer(2)]])
{
    uint packed = corners[vid];
    uint v = packed & 0x00FFFFFFu;
    uint comp = packed >> 24;
    SkinRenderVertex sv = verts[v];
    VOut o;
    o.position = U.viewProj * float4(sv.position.xyz, 1);
    o.world = sv.position.xyz;
    o.normal = normalize(sv.normal.xyz);
    o.flatShade = 0.0;
    o.albedo = srgbToLin(mix(float3(0.90), softPalette(comp * 5u + 11u), 0.76));
    return o;
}

inline float3 rigidMeshRotate(float4 q, float3 v) {
    return v + 2.0f * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

vertex VOut rigid_mesh_vertex(
    uint vid [[vertex_id]],
    device const RigidMeshVertex* vertices [[buffer(0)]],
    constant Uniforms& U [[buffer(1)]],
    device const float4* posLin [[buffer(2)]],
    device const float4* posAng [[buffer(3)]])
{
    RigidMeshVertex v = vertices[vid];
    uint body = as_type<uint>(v.positionBody.w);
    float4 q = posAng[body];
    float3 world = posLin[body].xyz + rigidMeshRotate(q, v.positionBody.xyz);
    VOut o;
    o.position = U.viewProj * float4(world, 1);
    o.world = world;
    o.normal = normalize(rigidMeshRotate(q, v.normal.xyz));
    o.albedo = srgbToLin(v.color.rgb);
    o.flatShade = 0;
    return o;
}

// ---------------------------------------------------------------------------
// PBR main fragment (samples the GTAO texture)
// ---------------------------------------------------------------------------
fragment float4 pbr_fragment(VOut in [[stage_in]],
                             constant Uniforms& U [[buffer(1)]],
                             texture2d<float> aoTex [[texture(0)]],
                             depth2d<float> shadowTex [[texture(1)]])
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

    float shadow = shadowVisibility(in.world, n, U, shadowTex);
    float3 direct = (in.albedo / M_PI_F * (1.0 - F) + spec) * SUN_COL * NdL * shadow;

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
// Prepass: world position + normal into two attachments (for GTAO)
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

// Thin sheets are double-sided: shade with the normal facing the viewer.
fragment float4 soft_fragment(VOut in [[stage_in]],
                              constant Uniforms& U [[buffer(1)]],
                              texture2d<float> aoTex [[texture(0)]],
                              depth2d<float> shadowTex [[texture(1)]])
{
    constexpr sampler smp(filter::linear);
    float ao = aoTex.sample(smp, in.position.xy / U.screen.xy).r;

    float3 n = normalize(in.normal);
    if (in.flatShade > 0.5) {
        // sharp-edged soft solids (tet boundaries): per-pixel face normal
        n = normalize(cross(dfdx(in.world), dfdy(in.world)));
        // derivative orientation depends on winding in screen space
        n = dot(n, U.eye.xyz - in.world) < 0.0 ? -n : n;
    }
    float3 V = normalize(U.eye.xyz - in.world);
    if (dot(n, V) < 0.0) n = -n;
    float3 L = -U.lightDir.xyz;
    float3 H = normalize(L + V);
    float NdL = max(dot(n, L), 0.0);
    float NdV = max(dot(n, V), 1e-4);
    float NdH = max(dot(n, H), 0.0);
    float HdV = max(dot(H, V), 0.0);

    const float rough = 0.72;              // matte woven look
    float a2 = rough * rough; a2 *= a2;
    float dDen = NdH * NdH * (a2 - 1.0) + 1.0;
    float D = a2 / (M_PI_F * dDen * dDen);
    float k = (rough + 1.0); k = k * k / 8.0;
    float G = (NdV / (NdV * (1.0 - k) + k)) * (NdL / (NdL * (1.0 - k) + k));
    float3 F0 = float3(0.035);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - HdV, 5.0);
    float3 spec = D * G * F / max(4.0 * NdV * NdL, 1e-4);

    // a touch of wrap so the unlit side of folds doesn't go dead black
    float wrap = max((dot(n, L) + 0.35) / 1.35, 0.0);
    float shadow = shadowVisibility(in.world, n, U, shadowTex);
    float3 direct = (in.albedo / M_PI_F * (1.0 - F) * wrap + spec * NdL) * SUN_COL * shadow;

    float3 irr = mix(GND_IRR, SKY_IRR, n.z * 0.5 + 0.5);
    float3 ambient = in.albedo * irr * 1.15;
    ambient *= ao;

    float3 lit = direct + ambient;
    float fog = horizonFog(length(in.world - U.eye.xyz));
    lit = mix(lit, HORIZON_LIN, fog);
    return float4(acesTonemap(lit), 1.0);
}

// Prepass variant: only the FRONT layer reaches the AO/depth buffers. The
// back layer sits one skin behind it and, at grazing angles, reads as a
// second surface inside the AO radius — pure view-space dark halos along
// the cloth silhouette. Rim corners (bit 22) collapse too.
vertex VOut soft_vertex_front(uint vid [[vertex_id]],
                              device const uint* corners [[buffer(0)]],
                              constant Uniforms& U [[buffer(1)]],
                              device const float4* posLin [[buffer(2)]],
                              device const float4* normals [[buffer(3)]])
{
    uint packed = corners[vid];
    if (((packed >> 23) & 1u) != 0 || ((packed >> 22) & 1u) != 0) return collapse();
    uint body = packed & 0x001FFFFFu;
    float4 nt = normals[body];
    float3 w = posLin[body].xyz + nt.xyz * (nt.w * 0.95);
    VOut o;
    o.position = U.viewProj * float4(w, 1);
    o.world = w;
    o.normal = nt.xyz;
    o.albedo = float3(0);
    o.flatShade = ((packed >> 21) & 1u) != 0 ? 1.0 : 0.0;
    return o;
}

fragment PreOut soft_prepass_fragment(VOut in [[stage_in]],
                                      constant Uniforms& U [[buffer(1)]])
{
    float3 n = normalize(in.normal);
    if (in.flatShade > 0.5) {
        n = normalize(cross(dfdx(in.world), dfdy(in.world)));
        n = dot(n, U.eye.xyz - in.world) < 0.0 ? -n : n;
    }
    float3 V = normalize(U.eye.xyz - in.world);
    if (dot(n, V) < 0.0) n = -n;           // AO sees the visible side
    PreOut o;
    o.pos = float4(in.world, 1.0);
    o.norm = float4(n, 0);
    return o;
}

// ---------------------------------------------------------------------------
// GTAO (horizon-based estimator over screen-space slices)
// ---------------------------------------------------------------------------
struct FSOut { float4 position [[position]]; float2 uv; };

vertex FSOut fs_vertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2) * 2.0 - 1.0;
    FSOut o;
    o.position = float4(p, 1.0, 1.0);
    o.uv = p * float2(0.5, -0.5) + 0.5;
    return o;
}

fragment float4 gtao_fragment(FSOut in [[stage_in]],
                              constant Uniforms& U [[buffer(1)]],
                              texture2d<float> posTex [[texture(0)]],
                              texture2d<float> normTex [[texture(1)]])
{
    // GTAO (Jimenez et al. 2016 / XeGTAO-style): march several
    // screen-space slices, find both horizon angles per slice, clamp to the
    // normal hemisphere, then integrate the cosine-weighted visible arc.
    constexpr sampler smp(filter::nearest, address::clamp_to_edge);
    float4 P4 = posTex.sample(smp, in.uv);
    if (P4.w < 0.5) return float4(1);
    float3 P = P4.xyz;
    float3 N = normalize(normTex.sample(smp, in.uv).xyz);
    float3 V = normalize(U.eye.xyz - P);
    float3 camForward = normalize(cross(U.camRight.xyz, U.camUp.xyz));
    float viewDepth = max(dot(P - U.eye.xyz, camForward), 0.25);

    const float R = 0.9;                                  // world AO radius
    float pxRadius = U.screen.z * R / viewDepth;
    // far away the radius collapses below sampling density — fade AO out
    // instead of letting a few-pixel march invent large-scale occlusion
    float farFade = saturate((pxRadius - 2.5) / 6.0);
    if (farFade <= 0.0) return float4(1);
    pxRadius = min(pxRadius, 96.0);
    // the falloff must use the radius we ACTUALLY march (post-clamp), or
    // near-camera AO reaches past its sampled range and over-darkens
    float Reff = pxRadius * viewDepth / U.screen.z;
    float falloffRange = max(Reff * 0.65, 1e-4);

    // interleaved gradient noise: much better spectral distribution than a
    // sin-hash, so residual error is high-frequency and blurs away cleanly
    float2 px = floor(in.position.xy);
    float ign = fract(52.9829189 * fract(0.06711056 * px.x + 0.00583715 * px.y));
    // golden-ratio rotation per frame: the temporal pass averages 1/blend
    // frames of differently-rotated estimates into a converged result
    float ang = fract(ign + U.temporal.x) * M_PI_F;
    float stepJit = fract(ign * 7.0 + U.temporal.x * 5.0);

    const int SLICES = 3;
    const int STEPS = 4;
    float occlusion = 0.0;

    for (int sl = 0; sl < SLICES; sl++) {
        float phi = ang + float(sl) * (M_PI_F / float(SLICES));
        float2 dirPx = float2(cos(phi), sin(phi));

        // ANALYTIC slice tangent: the world direction this screen-space
        // march corresponds to, projected perpendicular to V. Deriving it
        // from the camera basis (not from samples) keeps the slice plane
        // exact and view-consistent.
        float3 dirW = U.camRight.xyz * dirPx.x + U.camUp.xyz * dirPx.y;
        float3 omega = dirW - V * dot(dirW, V);
        float ol = length(omega);
        if (ol < 1e-4) { occlusion += 1.0; continue; }
        omega /= ol;

        // project N into the slice plane (spanned by V and omega)
        float3 sliceN = cross(V, omega);
        float3 projN = N - sliceN * dot(N, sliceN);
        float projLen = length(projN);
        if (projLen < 1e-4) { occlusion += 1.0; continue; }
        float3 pn = projN / projLen;
        float cosNV = saturate(dot(pn, V));
        float n = acos(cosNV) * (dot(pn, omega) >= 0.0 ? 1.0 : -1.0);

        // XeGTAO fades discarded/out-of-radius samples toward the normal
        // hemisphere edge instead of -1, avoiding grazing-angle detail loss.
        float lowH0 = cos(n + M_PI_F * 0.5);
        float lowH1 = cos(n - M_PI_F * 0.5);

        // horizon cosines per WORLD side of the slice (classified by the
        // actual sample offset, so screen/UV orientation can't flip them)
        float cosH0 = lowH0, cosH1 = lowH1;
        float minS = min(0.95, 1.3 / max(pxRadius, 1e-3));
        for (int side = 0; side < 2; side++) {
            float sgn = side == 0 ? 1.0 : -1.0;
            for (int st = 1; st <= STEPS; st++) {
                float u = saturate((float(st) - 1.0 + stepJit) / float(STEPS));
                float t = minS + (1.0 - minS) * pow(u, 2.0);
                float2 uv2 = in.uv + sgn * dirPx * (t * pxRadius) / U.screen.xy;
                float4 Q4 = posTex.sample(smp, uv2);
                if (Q4.w < 0.5) continue;
                float3 w = Q4.xyz - P;
                float l = length(w);
                if (l < 1e-4) continue;
                float c = dot(w / l, V);
                float weight = saturate((Reff - l) / falloffRange);
                if (dot(w, omega) >= 0.0) {
                    cosH0 = max(cosH0, mix(lowH0, c, weight));
                } else {
                    cosH1 = max(cosH1, mix(lowH1, c, weight));
                }
            }
        }

        // signed horizon angles in the slice plane (+ = toward omega),
        // clamped to the hemisphere around the projected normal
        float h1 =  acos(clamp(cosH0, -1.0, 1.0));
        float h2 = -acos(clamp(cosH1, -1.0, 1.0));
        h1 = n + min(h1 - n,  M_PI_F / 2.0);
        h2 = n + max(h2 - n, -M_PI_F / 2.0);

        // analytic cosine-weighted visible arc (GTAO inner integral)
        float a1 = 0.25 * (-cos(2.0 * h1 - n) + cosNV + 2.0 * h1 * sin(n));
        float a2 = 0.25 * (-cos(2.0 * h2 - n) + cosNV + 2.0 * h2 * sin(n));
        occlusion += projLen * (a1 + a2);
    }
    float ao = clamp(occlusion / float(SLICES), 0.0, 1.0);
    ao = pow(ao, 1.25);    // slight contrast shaping
    ao = max(ao, 0.03);    // visible pixels should not reach total black
    ao = mix(1.0, ao, farFade);
    return float4(ao, ao, ao, 1);
}

fragment float4 temporal_fragment(FSOut in [[stage_in]],
                                  constant Uniforms& U [[buffer(1)]],
                                  texture2d<float> rawAO [[texture(0)]],
                                  texture2d<float> histAO [[texture(1)]],
                                  texture2d<float> posTex [[texture(2)]],
                                  texture2d<float> prevPosTex [[texture(3)]])
{
    // temporal accumulation with reprojection: find this pixel's world
    // point in the previous frame, reuse its accumulated AO unless the
    // surface was occluded/moved (world-position mismatch -> reject).
    constexpr sampler smp(filter::nearest, address::clamp_to_edge);
    constexpr sampler lsmp(filter::linear, address::clamp_to_edge);
    float cur = rawAO.sample(smp, in.uv).r;
    float4 P4 = posTex.sample(smp, in.uv);
    if (P4.w < 0.5 || U.temporal.y >= 1.0) return float4(cur);
    float3 P = P4.xyz;

    float4 clipPrev = U.prevViewProj * float4(P, 1.0);
    if (clipPrev.w <= 0.0) return float4(cur);
    float2 ndc = clipPrev.xy / clipPrev.w;
    float2 uvPrev = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uvPrev < 0.0) || any(uvPrev > 1.0)) return float4(cur);

    float4 Q4 = prevPosTex.sample(smp, uvPrev);
    if (Q4.w < 0.5) return float4(cur);
    // disocclusion / moving-body test: same surface point last frame?
    float reuse = saturate(1.0 - length(Q4.xyz - P) * 14.0);
    if (reuse <= 0.0) return float4(cur);

    float hist = histAO.sample(lsmp, uvPrev).r;
    float blend = mix(1.0, U.temporal.y, reuse);   // partial trust on the edge
    return float4(mix(hist, cur, blend));
}

fragment float4 blur_fragment(FSOut in [[stage_in]],
                              constant Uniforms& U [[buffer(1)]],
                              texture2d<float> src [[texture(0)]],
                              texture2d<float> posTex [[texture(1)]])
{
    // depth-aware 5x5 blur: averages AO noise without bleeding across
    // geometric edges (weights collapse when world distance jumps)
    constexpr sampler smp(filter::nearest, address::clamp_to_edge);
    float2 px = 1.0 / U.screen.xy;
    float4 c4 = posTex.sample(smp, in.uv);
    if (c4.w < 0.5) return float4(1);
    float3 cP = c4.xyz;
    float acc = 0.0, wsum = 0.0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            float2 uv2 = in.uv + float2(float(x), float(y)) * px;
            float4 q4 = posTex.sample(smp, uv2);
            if (q4.w < 0.5) continue;
            float d = length(q4.xyz - cP);
            float w = exp(-d * d * 6.0);
            acc += src.sample(smp, uv2).r * w;
            wsum += w;
        }
    }
    return float4(wsum > 0.0 ? acc / wsum : 1.0);
}

// ---------------------------------------------------------------------------
// Sky (whitish) and floor (AA checker, melts then fogs)
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
                               texture2d<float> aoTex [[texture(0)]],
                               depth2d<float> shadowTex [[texture(1)]])
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
    float shadow = shadowVisibility(in.world, float3(0, 0, 1), U, shadowTex);
    float3 lit = albedo * (SKY_IRR * 1.1 * ao
                         + SUN_COL / M_PI_F * NdL * 0.85 * shadow);

    float fog = horizonFog(d);
    lit = mix(lit, HORIZON_LIN, fog);
    return float4(acesTonemap(lit), 1);
}

"""

struct Uniforms {
    var viewProj: simd_float4x4
    var lightDir: SIMD4<Float>
    var eye: SIMD4<Float>
    var screen: SIMD4<Float>
    var camRight: SIMD4<Float>
    var camUp: SIMD4<Float>
    var prevViewProj: simd_float4x4
    var temporal: SIMD4<Float>
    var shadowViewProj: simd_float4x4
    var shadowParams: SIMD4<Float>
}

let SPHV = 12 * 18 * 6
let TORV = 24 * 12 * 6
let CAPV = (2 * 6 + 1) * 16 * 6

private enum RendererInitializationError: LocalizedError {
    case commandQueue
    case shaderFunction(String)
    case depthState(String)

    var errorDescription: String? {
        switch self {
        case .commandQueue:
            return "could not create the Metal render command queue"
        case .shaderFunction(let name):
            return "render shader function '\(name)' is missing"
        case .depthState(let name):
            return "could not create the \(name) depth state"
        }
    }
}

final class Renderer: NSObject, MTKViewDelegate {
    static let sampleCount = 4
    static let colorFormat = MTLPixelFormat.bgra8Unorm_srgb
    static let shadowMapSize = 2048

    let device: MTLDevice
    let queue: MTLCommandQueue
    weak var model: (AnyObject & RenderableModel)?

    var boxP, sphereP, torusP, capsuleP, softP, skinP, rigidMeshP: MTLRenderPipelineState!
    var boxPre, spherePre, torusPre, capsulePre, floorPreP, softPre, skinPre,
        rigidMeshPre: MTLRenderPipelineState!
    var boxShadow, sphereShadow, torusShadow, capsuleShadow, softShadow,
        skinShadow, rigidMeshShadow: MTLRenderPipelineState!
    var skyP, floorP, gtaoP, blurP, temporalP: MTLRenderPipelineState!
    var depthState, noDepthState: MTLDepthStencilState!

    var posTex, normTex, aoTexA, aoTexB, preDepthTex, shadowTex: MTLTexture?
    var histTex, prevPosTex: MTLTexture?
    var prevVP: simd_float4x4?
    var frameIdx: UInt32 = 0
    var targetSize = SIMD2<Int>(0, 0)
    var instances: MTLBuffer?

    var azimuth: Float = 0.9
    var elevation: Float = 0.35
    var distance: Float = 30
    var target = F3(0, 0, 3)
    var viewportSize = SIMD2<Float>(1, 1)
    private var framesDrawn = 0
    private var lastCameraEpoch: Int = -1
    private var lastSolverIdentity: ObjectIdentifier?
    private var envCameraSet = false
    private var runtimeFailure: String?

    private func reportFailure(_ message: String) {
        guard runtimeFailure == nil else { return }
        runtimeFailure = message
        // A failed render command can leave temporal targets partially
        // written. Ignore their history if a later frame is retried.
        prevVP = nil
        let target = model
        DispatchQueue.main.async {
            target?.reportRenderFailure(message)
        }
    }

    private func commandFailureDescription(_ command: MTLCommandBuffer) -> String? {
        guard command.status != .completed || command.error != nil else {
            return nil
        }
        let status = String(describing: command.status)
        guard let error = command.error as NSError? else {
            return "Metal command ended with status \(status)"
        }
        return "Metal command ended with status \(status): "
            + "\(error.domain) \(error.code) — \(error.localizedDescription)"
    }

    private var scriptedCaptureAllowed: Bool {
        let environment = ProcessInfo.processInfo.environment
        let selected = environment["AVBD_CAPTURE_VIEW"]
            ?? (environment["AVBD_POLICY_REPLAY"] == nil ? "playground" : "policy")
        return model?.captureID == selected
    }

    init(device: MTLDevice, model: AnyObject & RenderableModel) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw RendererInitializationError.commandQueue
        }
        self.queue = queue
        self.model = model
        super.init()
        // scripted-capture camera overrides
        let env = ProcessInfo.processInfo.environment
        if let v = env["AVBD_CAM_DIST"].flatMap(Float.init) { distance = v; envCameraSet = true }
        if let v = env["AVBD_CAM_AZ"].flatMap(Float.init) { azimuth = v; envCameraSet = true }
        if let v = env["AVBD_CAM_EL"].flatMap(Float.init) { elevation = v; envCameraSet = true }
        if let v = env["AVBD_CAM_TZ"].flatMap(Float.init) { target.z = v; envCameraSet = true }

        let lib = try device.makeLibrary(source: renderShaderSource, options: nil)
        func pipe(_ v: String, _ f: String, samples: Int = Renderer.sampleCount,
                  colorFormats: [MTLPixelFormat] = [Renderer.colorFormat],
                  depth: MTLPixelFormat = .depth32Float) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            guard let vertex = lib.makeFunction(name: v) else {
                throw RendererInitializationError.shaderFunction(v)
            }
            guard let fragment = lib.makeFunction(name: f) else {
                throw RendererInitializationError.shaderFunction(f)
            }
            d.vertexFunction = vertex
            d.fragmentFunction = fragment
            for (i, fmt) in colorFormats.enumerated() {
                d.colorAttachments[i].pixelFormat = fmt
            }
            d.depthAttachmentPixelFormat = depth
            d.rasterSampleCount = samples
            return try device.makeRenderPipelineState(descriptor: d)
        }
        func depthPipe(_ vertex: String) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.label = "Directional shadow: \(vertex)"
            guard let function = lib.makeFunction(name: vertex) else {
                throw RendererInitializationError.shaderFunction(vertex)
            }
            d.vertexFunction = function
            d.fragmentFunction = nil
            d.depthAttachmentPixelFormat = .depth32Float
            d.rasterSampleCount = 1
            return try device.makeRenderPipelineState(descriptor: d)
        }

        boxP = try pipe("box_vertex", "pbr_fragment")
        sphereP = try pipe("sphere_vertex", "pbr_fragment")
        torusP = try pipe("torus_vertex", "pbr_fragment")
        capsuleP = try pipe("capsule_vertex", "pbr_fragment")
        softP = try pipe("soft_vertex", "soft_fragment")
        skinP = try pipe("skin_vertex", "soft_fragment")
        rigidMeshP = try pipe("rigid_mesh_vertex", "pbr_fragment")
        skyP = try pipe("fs_vertex", "sky_fragment")
        floorP = try pipe("floor_vertex", "floor_fragment")

        boxShadow = try depthPipe("box_vertex")
        sphereShadow = try depthPipe("sphere_vertex")
        torusShadow = try depthPipe("torus_vertex")
        capsuleShadow = try depthPipe("capsule_vertex")
        softShadow = try depthPipe("soft_vertex_front")
        skinShadow = try depthPipe("skin_vertex")
        rigidMeshShadow = try depthPipe("rigid_mesh_vertex")

        let preFmt: [MTLPixelFormat] = [.rgba32Float, .rgba16Float]
        boxPre = try pipe("box_vertex", "prepass_fragment", samples: 1, colorFormats: preFmt)
        spherePre = try pipe("sphere_vertex", "prepass_fragment", samples: 1, colorFormats: preFmt)
        torusPre = try pipe("torus_vertex", "prepass_fragment", samples: 1, colorFormats: preFmt)
        capsulePre = try pipe("capsule_vertex", "prepass_fragment", samples: 1, colorFormats: preFmt)
        floorPreP = try pipe("floor_vertex", "floor_prepass_fragment", samples: 1, colorFormats: preFmt)
        softPre = try pipe("soft_vertex_front", "soft_prepass_fragment", samples: 1, colorFormats: preFmt)
        skinPre = try pipe("skin_vertex", "prepass_fragment", samples: 1, colorFormats: preFmt)
        rigidMeshPre = try pipe("rigid_mesh_vertex", "prepass_fragment",
                                samples: 1, colorFormats: preFmt)
        gtaoP = try pipe("fs_vertex", "gtao_fragment", samples: 1,
                         colorFormats: [.r8Unorm], depth: .invalid)
        blurP = try pipe("fs_vertex", "blur_fragment", samples: 1,
                         colorFormats: [.r8Unorm], depth: .invalid)
        temporalP = try pipe("fs_vertex", "temporal_fragment", samples: 1,
                             colorFormats: [.r8Unorm], depth: .invalid)

        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: dd) else {
            throw RendererInitializationError.depthState("geometry")
        }
        self.depthState = depthState

        let nd = MTLDepthStencilDescriptor()
        nd.depthCompareFunction = .always
        nd.isDepthWriteEnabled = false
        guard let noDepthState = device.makeDepthStencilState(descriptor: nd) else {
            throw RendererInitializationError.depthState("background")
        }
        self.noDepthState = noDepthState

    }

    @discardableResult
    private func ensureTargets(_ size: CGSize) -> Bool {
        let w = max(Int(size.width), 4), h = max(Int(size.height), 4)
        if targetSize == SIMD2(w, h), posTex != nil, normTex != nil,
           aoTexA != nil, aoTexB != nil, histTex != nil, prevPosTex != nil,
           preDepthTex != nil, shadowTex != nil { return true }
        func tex(_ fmt: MTLPixelFormat) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt,
                                                             width: w, height: h,
                                                             mipmapped: false)
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }
        let newPos = tex(.rgba32Float)
        let newNorm = tex(.rgba16Float)
        let newAOA = tex(.r8Unorm)
        let newAOB = tex(.r8Unorm)
        let newHistory = tex(.r8Unorm)
        let newPreviousPosition = tex(.rgba32Float)
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float,
                                                          width: w, height: h,
                                                          mipmapped: false)
        dd.usage = .renderTarget
        dd.storageMode = .private
        let newPreDepth = device.makeTexture(descriptor: dd)

        let sd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: Renderer.shadowMapSize,
            height: Renderer.shadowMapSize,
            mipmapped: false)
        sd.usage = [.renderTarget, .shaderRead]
        sd.storageMode = .private
        let newShadow = device.makeTexture(descriptor: sd)
        guard let newPos, let newNorm, let newAOA, let newAOB,
              let newHistory, let newPreviousPosition, let newPreDepth,
              let newShadow else { return false }
        posTex = newPos
        normTex = newNorm
        aoTexA = newAOA
        aoTexB = newAOB
        histTex = newHistory
        prevPosTex = newPreviousPosition
        preDepthTex = newPreDepth
        shadowTex = newShadow
        shadowTex?.label = "Directional shadow map"
        targetSize = SIMD2(w, h)
        prevVP = nil                      // resize invalidates history
        return true
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
        guard runtimeFailure == nil else { return }
        guard let model,
              let solver = model.solver,
              let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }

        guard let cmd = queue.makeCommandBuffer() else {
            reportFailure("could not create a Metal command buffer")
            return
        }
        cmd.label = "AVBD render frame"

        model.tickIfRunning()
        // The model owns physics failure reporting. Never relabel a poisoned
        // solver as a renderer failure or read its potentially invalid state.
        guard solver.runtimeFailure == nil else { return }
        let solverIdentity = ObjectIdentifier(solver)
        if solverIdentity != lastSolverIdentity {
            lastSolverIdentity = solverIdentity
            prevVP = nil
        }
        guard ensureTargets(view.drawableSize) else {
            reportFailure("could not allocate required Metal render targets")
            return
        }

        // Per-scene default framing (small cloth rigs drown at the rigid-rig
        // default distance). Applied only when the DEMO changes — resets and
        // size/param rebuilds keep the user's camera. AVBD_CAM_* env
        // overrides outrank it.
        if lastCameraEpoch != model.cameraEpoch {
            lastCameraEpoch = model.cameraEpoch
            if !envCameraSet {
                distance = solver.settings.cameraDistance > 0
                    ? solver.settings.cameraDistance : 30
                target.z = solver.settings.cameraTargetZ != 0
                    ? solver.settings.cameraTargetZ : 3
                target.x = solver.settings.cameraTargetX.isFinite
                    ? solver.settings.cameraTargetX : 0
                target.y = solver.settings.cameraTargetY.isFinite
                    ? solver.settings.cameraTargetY : 0
                if solver.settings.cameraAzimuth.isFinite {
                    azimuth = solver.settings.cameraAzimuth
                }
                if solver.settings.cameraElevation.isFinite {
                    elevation = solver.settings.cameraElevation
                }
            }
        }

        let bodyCount = solver.bodyCount
        let rigidCount = solver.renderRigidBodyCount
        let stride = MemoryLayout<simd_float4x4>.stride + 32
        if instances == nil || instances!.length < max(1, rigidCount) * stride {
            instances = device.makeBuffer(length: max(256, max(1, rigidCount) * stride),
                                          options: .storageModePrivate)
        }
        guard let instances, let posTex, let normTex, let aoTexA, let aoTexB,
              let histTex, let prevPosTex, let preDepthTex, let shadowTex else {
            reportFailure("could not allocate required Metal render resources")
            return
        }

        do {
            try solver.encodeBuildInstancesChecked(
                cmd, instances: instances,
                colorMode: model.colorByGraphColor ? 1 : 0)
        } catch {
            // synchronize() inside the checked API may be the first observer
            // of a physics error. Its owning model will report it on the next
            // tick; do not poison or misclassify it here.
            guard solver.runtimeFailure == nil else { return }
            reportFailure("instance-build encoder failed: \(error.localizedDescription)")
            return
        }

        let aspect = viewportSize.x / max(viewportSize.y, 1)
        let pxPerUnit = viewportSize.y * 0.5 / tan(25 * Float.pi / 180)
        let fwd = normalize(target - eyePosition)
        let camR = normalize(cross(fwd, F3(0, 0, 1)))
        let camU = cross(camR, fwd)          // screen +y in UV space is DOWN
        let vp = projectionMatrix(aspect: aspect) * viewMatrix
        let lightDirection = normalize(F3(0.4, 0.25, -0.85))

        // A camera-targeted, texel-stabilized orthographic light projection.
        // This keeps articulated robot detail crisp while preventing shadows
        // from crawling when the follow camera advances with a policy.
        let shadowExtent = max(7.0, min(distance * 0.9, 40.0))
        let lightUp = F3(0, 1, 0)
        let lightRight = normalize(cross(lightDirection, lightUp))
        let lightMapUp = cross(lightRight, lightDirection)
        let texelWorld = (2 * shadowExtent) / Float(Renderer.shadowMapSize)
        let rightCoord = (dot(target, lightRight) / texelWorld).rounded() * texelWorld
        let upCoord = (dot(target, lightMapUp) / texelWorld).rounded() * texelWorld
        let shadowCenter = target
            + lightRight * (rightCoord - dot(target, lightRight))
            + lightMapUp * (upCoord - dot(target, lightMapUp))
        let lightEye = shadowCenter - lightDirection * (shadowExtent * 2.5)
        let shadowVP = metalOrthographicProjection(
            left: -shadowExtent, right: shadowExtent,
            bottom: -shadowExtent, top: shadowExtent,
            near: 0.1, far: shadowExtent * 5.0)
            * lookAt(eye: lightEye, center: shadowCenter, up: lightUp)
        let noiseOff = Float(frameIdx % 64) * 0.6180339887
        let screenViewport = MTLViewport(originX: 0, originY: 0,
                                         width: Double(targetSize.x),
                                         height: Double(targetSize.y),
                                         znear: 0, zfar: 1)
        var U = Uniforms(viewProj: vp,
                         lightDir: SIMD4(lightDirection, 0),
                         eye: SIMD4(eyePosition, 0),
                         screen: SIMD4(viewportSize.x, viewportSize.y, pxPerUnit, 0),
                         camRight: SIMD4(camR, 0),
                         camUp: SIMD4(-camU, 0),
                         prevViewProj: prevVP ?? vp,
                         temporal: SIMD4(noiseOff.truncatingRemainder(dividingBy: 1),
                                         prevVP == nil ? 1.0 : 0.12, 0, 0),
                         shadowViewProj: shadowVP,
                         shadowParams: SIMD4(0.00008, 0.00030, 0, 0))
        // ---- 1. directional shadow depth ----
        let shadowPass = MTLRenderPassDescriptor()
        shadowPass.depthAttachment.texture = shadowTex
        shadowPass.depthAttachment.loadAction = .clear
        shadowPass.depthAttachment.clearDepth = 1
        shadowPass.depthAttachment.storeAction = .store
        guard let shadowEncoder = cmd.makeRenderCommandEncoder(
                descriptor: shadowPass) else {
            reportFailure("could not create the directional-shadow encoder")
            return
        }
        do {
            let enc = shadowEncoder
            enc.label = "Directional shadow casters"
            enc.setViewport(MTLViewport(originX: 0, originY: 0,
                                        width: Double(Renderer.shadowMapSize),
                                        height: Double(Renderer.shadowMapSize),
                                        znear: 0, zfar: 1))
            enc.setDepthStencilState(depthState)
            var shadowU = U
            shadowU.viewProj = shadowVP
            if rigidCount > 0 {
                for (p, verts) in [(boxShadow!, 36), (sphereShadow!, SPHV),
                                   (torusShadow!, TORV), (capsuleShadow!, CAPV)] {
                    enc.setRenderPipelineState(p)
                    enc.setVertexBuffer(instances, offset: 0, index: 0)
                    enc.setVertexBytes(&shadowU, length: MemoryLayout<Uniforms>.stride, index: 1)
                    enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                       vertexCount: verts, instanceCount: rigidCount)
                }
            }
            if let surf = solver.renderSurface {
                enc.setRenderPipelineState(softShadow)
                enc.setVertexBuffer(surf.tris, offset: 0, index: 0)
                enc.setVertexBytes(&shadowU, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(surf.positions, offset: 0, index: 2)
                enc.setVertexBuffer(surf.normals, offset: 0, index: 3)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: surf.triCount * 3)
            }
            if let skin = solver.renderSkinnedSurface {
                enc.setRenderPipelineState(skinShadow)
                enc.setVertexBuffer(skin.tris, offset: 0, index: 0)
                enc.setVertexBytes(&shadowU, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(skin.vertices, offset: 0, index: 2)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: skin.triCount * 3)
            }
            if let mesh = solver.renderRigidMeshSurface {
                enc.setRenderPipelineState(rigidMeshShadow)
                enc.setVertexBuffer(mesh.vertices, offset: 0, index: 0)
                enc.setVertexBytes(&shadowU, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(mesh.positions, offset: 0, index: 2)
                enc.setVertexBuffer(mesh.rotations, offset: 0, index: 3)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: mesh.vertexCount)
            }
            enc.endEncoding()
        }

        // ---- 2. prepass: world pos + normals ----
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
        guard let prepassEncoder = cmd.makeRenderCommandEncoder(
                descriptor: pre) else {
            reportFailure("could not create the world-position prepass encoder")
            return
        }
        do {
            let enc = prepassEncoder
            enc.label = "World-position and normal prepass"
            enc.setViewport(screenViewport)
            enc.setDepthStencilState(depthState)
            enc.setRenderPipelineState(floorPreP)
            enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
            if rigidCount > 0 {
                for (p, verts) in [(boxPre!, 36), (spherePre!, SPHV),
                                   (torusPre!, TORV), (capsulePre!, CAPV)] {
                    enc.setRenderPipelineState(p)
                    enc.setVertexBuffer(instances, offset: 0, index: 0)
                    enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                    enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                       vertexCount: verts, instanceCount: rigidCount)
                }
            }
            if let surf = solver.renderSurface {
                enc.setRenderPipelineState(softPre)
                enc.setVertexBuffer(surf.tris, offset: 0, index: 0)
                enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(surf.positions, offset: 0, index: 2)
                enc.setVertexBuffer(surf.normals, offset: 0, index: 3)
                enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: surf.triCount * 3)
            }
            if let skin = solver.renderSkinnedSurface {
                enc.setRenderPipelineState(skinPre)
                enc.setVertexBuffer(skin.tris, offset: 0, index: 0)
                enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(skin.vertices, offset: 0, index: 2)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: skin.triCount * 3)
            }
            if let mesh = solver.renderRigidMeshSurface {
                enc.setRenderPipelineState(rigidMeshPre)
                enc.setVertexBuffer(mesh.vertices, offset: 0, index: 0)
                enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(mesh.positions, offset: 0, index: 2)
                enc.setVertexBuffer(mesh.rotations, offset: 0, index: 3)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: mesh.vertexCount)
            }
            enc.endEncoding()
        }

        // ---- 3. GTAO ----
        let aoPass = MTLRenderPassDescriptor()
        aoPass.colorAttachments[0].texture = aoTexA
        aoPass.colorAttachments[0].loadAction = .dontCare
        aoPass.colorAttachments[0].storeAction = .store
        guard let gtaoEncoder = cmd.makeRenderCommandEncoder(
                descriptor: aoPass) else {
            reportFailure("could not create the GTAO encoder")
            return
        }
        do {
            let enc = gtaoEncoder
            enc.label = "GTAO"
            enc.setViewport(screenViewport)
            enc.setRenderPipelineState(gtaoP)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(posTex, index: 0)
            enc.setFragmentTexture(normTex, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }

        // ---- 4a. temporal resolve: raw A + reprojected history -> B ----
        let tPass = MTLRenderPassDescriptor()
        tPass.colorAttachments[0].texture = aoTexB
        tPass.colorAttachments[0].loadAction = .dontCare
        tPass.colorAttachments[0].storeAction = .store
        guard let temporalEncoder = cmd.makeRenderCommandEncoder(
                descriptor: tPass) else {
            reportFailure("could not create the temporal-AO encoder")
            return
        }
        do {
            let enc = temporalEncoder
            enc.label = "Temporal AO resolve"
            enc.setViewport(screenViewport)
            enc.setRenderPipelineState(temporalP)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(aoTexA, index: 0)
            enc.setFragmentTexture(histTex, index: 1)
            enc.setFragmentTexture(posTex, index: 2)
            enc.setFragmentTexture(prevPosTex, index: 3)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }
        // ---- 4b. save history (resolved AO + world positions) ----
        guard let historyEncoder = cmd.makeBlitCommandEncoder() else {
            reportFailure("could not create the temporal-history encoder")
            return
        }
        historyEncoder.label = "Temporal history copy"
        historyEncoder.copy(from: aoTexB, to: histTex)
        historyEncoder.copy(from: posTex, to: prevPosTex)
        historyEncoder.endEncoding()
        // ---- 4c. one depth-aware spatial blur: B -> A ----
        let blurPass = MTLRenderPassDescriptor()
        blurPass.colorAttachments[0].texture = aoTexA
        blurPass.colorAttachments[0].loadAction = .dontCare
        blurPass.colorAttachments[0].storeAction = .store
        guard let blurEncoder = cmd.makeRenderCommandEncoder(
                descriptor: blurPass) else {
            reportFailure("could not create the AO-blur encoder")
            return
        }
        do {
            let enc = blurEncoder
            enc.label = "Depth-aware AO blur"
            enc.setViewport(screenViewport)
            enc.setRenderPipelineState(blurP)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(aoTexB, index: 0)
            enc.setFragmentTexture(posTex, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }

        // ---- 5. main pass ----
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else {
            reportFailure("could not create the main render encoder")
            return
        }
        enc.label = "Main PBR pass"
        enc.setViewport(screenViewport)

        enc.setDepthStencilState(noDepthState)
        enc.setRenderPipelineState(skyP)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        enc.setDepthStencilState(depthState)
        enc.setRenderPipelineState(floorP)
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentTexture(aoTexA, index: 0)
        enc.setFragmentTexture(shadowTex, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

        enc.setDepthStencilState(depthState)
        if rigidCount > 0 {
            for (p, verts) in [(boxP!, 36), (sphereP!, SPHV), (torusP!, TORV), (capsuleP!, CAPV)] {
                enc.setRenderPipelineState(p)
                enc.setVertexBuffer(instances, offset: 0, index: 0)
                enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setFragmentTexture(aoTexA, index: 0)
                enc.setFragmentTexture(shadowTex, index: 1)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: verts, instanceCount: rigidCount)
            }
        }
        if let surf = solver.renderSurface {
            enc.setRenderPipelineState(softP)
            enc.setVertexBuffer(surf.tris, offset: 0, index: 0)
            enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setVertexBuffer(surf.positions, offset: 0, index: 2)
            enc.setVertexBuffer(surf.normals, offset: 0, index: 3)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(aoTexA, index: 0)
            enc.setFragmentTexture(shadowTex, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: surf.triCount * 3)
        }
        if let skin = solver.renderSkinnedSurface {
            enc.setRenderPipelineState(skinP)
            enc.setVertexBuffer(skin.tris, offset: 0, index: 0)
            enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setVertexBuffer(skin.vertices, offset: 0, index: 2)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(aoTexA, index: 0)
            enc.setFragmentTexture(shadowTex, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: skin.triCount * 3)
        }
        if let mesh = solver.renderRigidMeshSurface {
            enc.setRenderPipelineState(rigidMeshP)
            enc.setVertexBuffer(mesh.vertices, offset: 0, index: 0)
            enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setVertexBuffer(mesh.positions, offset: 0, index: 2)
            enc.setVertexBuffer(mesh.rotations, offset: 0, index: 3)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(aoTexA, index: 0)
            enc.setFragmentTexture(shadowTex, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: mesh.vertexCount)
        }
        enc.endEncoding()

        cmd.present(drawable)
        cmd.commit()

        // Physics and rendering own different command queues but share pose
        // buffers. Retire every visual frame before returning to the main run
        // loop, where Play/Step/Reset/goal/impact actions may mutate those
        // buffers. This is the conservative cross-queue correctness boundary;
        // a future shared-event/read-lease API can recover overlap safely.
        cmd.waitUntilCompleted()
        if let failure = commandFailureDescription(cmd) {
            reportFailure(failure)
            return
        }

        prevVP = vp
        frameIdx &+= 1

        framesDrawn += 1
        if framesDrawn == 30, ProcessInfo.processInfo.environment["AVBD_MARKER"] != nil {
            let info = "frames=30 bodies=\(bodyCount) rigidInstances=\(rigidCount) stats=\(model.statsText)"
            try? info.write(toFile: "/tmp/avbd_render_marker.txt", atomically: true, encoding: .utf8)
        }
        // Scripted capture: AVBD_SHOT=<path.png> [AVBD_SHOT_FRAME=n] writes
        // the resolved frame and exits (needs framebufferOnly = false).
        if scriptedCaptureAllowed,
           let shotPath = ProcessInfo.processInfo.environment["AVBD_SHOT"],
           framesDrawn == (Int(ProcessInfo.processInfo.environment["AVBD_SHOT_FRAME"] ?? "") ?? 90) {
            writePNG(texture: drawable.texture, to: shotPath)
            exit(0)
        }
        if scriptedCaptureAllowed,
           let directory = ProcessInfo.processInfo.environment["AVBD_VIDEO_DIR"] {
            let every = max(Int(ProcessInfo.processInfo.environment["AVBD_VIDEO_EVERY"] ?? "") ?? 3, 1)
            let maximum = max(Int(ProcessInfo.processInfo.environment["AVBD_VIDEO_FRAMES"] ?? "") ?? 600, 1)
            if framesDrawn.isMultiple(of: every) {
                let path = String(format: "%@/%05d.png", directory, framesDrawn / every)
                writePNG(texture: drawable.texture, to: path)
            }
            if framesDrawn >= maximum { exit(0) }
        }
    }

    private func writePNG(texture: MTLTexture, to path: String) {
        let w = texture.width, h = texture.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        bytes.withUnsafeMutableBytes { raw in
            texture.getBytes(raw.baseAddress!, bytesPerRow: w * 4,
                             from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        }
        // BGRA -> RGBA
        for i in stride(from: 0, to: bytes.count, by: 4) { bytes.swapAt(i, i + 2) }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let img = CGImage(width: w, height: h, bitsPerComponent: 8,
                                bitsPerPixel: 32, bytesPerRow: w * 4, space: cs,
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                provider: provider, decode: nil,
                                shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(
                  URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(dest, img, nil)
        CGImageDestinationFinalize(dest)
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
