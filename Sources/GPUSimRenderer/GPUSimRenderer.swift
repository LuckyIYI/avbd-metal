import Metal
import MetalKit
import SimCore
import PhysicsAVBD
import simd

/// Runtime presentation choices. They affect only rendering and never mutate
/// the simulation or its collision geometry.
public enum GPUSimRenderColorMode: UInt32, Sendable, Equatable {
    case bodyIndex = 0
    case constraintGraph = 1
}

public struct GPUSimRenderOptions: Sendable, Equatable {
    public var colorMode: GPUSimRenderColorMode
    public var showConvexCollisionGeometry: Bool
    public var convexCollisionWireframe: Bool
    /// Ambient occlusion (half-resolution GTAO with temporal accumulation),
    /// ~1.5 ms GPU on the reference scene. Hosts on a power budget can
    /// disable it and keep the rest of the pipeline unchanged.
    public var ambientOcclusion: Bool
    /// Disable the built-in checker ground when the host authors its own floor.
    public var showsGroundPlane: Bool
    /// Minimum time each presented frame stays on screen, in seconds; nil
    /// presents as soon as the GPU finishes. A loop running below the
    /// panel's refresh (60 fps on a 120 Hz display) otherwise lands each
    /// frame on whichever refresh its GPU work happened to finish before,
    /// so content sampled at a steady rate displays with an uneven
    /// cadence. Set it to the loop's frame period to pin the cadence.
    public var minimumFrameDuration: Double?

    public init(
        colorMode: GPUSimRenderColorMode = .bodyIndex,
        showConvexCollisionGeometry: Bool = false,
        convexCollisionWireframe: Bool = true,
        ambientOcclusion: Bool = true,
        showsGroundPlane: Bool = true,
        minimumFrameDuration: Double? = nil
    ) {
        self.colorMode = colorMode
        self.showConvexCollisionGeometry = showConvexCollisionGeometry
        self.convexCollisionWireframe = convexCollisionWireframe
        self.ambientOcclusion = ambientOcclusion
        self.showsGroundPlane = showsGroundPlane
        self.minimumFrameDuration = minimumFrameDuration
    }
}

public struct GPUSimRenderCameraHint: Sendable, Equatable {
    public var distance: Float
    public var target: F3
    public var azimuth: Float
    public var elevation: Float

    public init(
        distance: Float = 30,
        target: F3 = F3(0, 0, 3),
        azimuth: Float = 0.9,
        elevation: Float = 0.35
    ) {
        self.distance = distance
        self.target = target
        self.azimuth = azimuth
        self.elevation = elevation
    }
}

/// An application-owned look-at camera. Use this for cameras attached to a
/// robot, chase cameras, and other poses that cannot be represented by the
/// renderer's editor-style orbit controls.
public struct GPUSimCameraPose: Sendable, Equatable {
    public var position: F3
    public var target: F3
    public var up: F3

    public init(position: F3, target: F3, up: F3 = F3(0, 0, 1)) {
        precondition(position.x.isFinite && position.y.isFinite
            && position.z.isFinite && target.x.isFinite
            && target.y.isFinite && target.z.isFinite
            && up.x.isFinite && up.y.isFinite && up.z.isFinite,
            "camera pose must be finite")
        precondition(length_squared(target - position) > 1.0e-12,
                     "camera position and target must differ")
        precondition(length_squared(cross(target - position, up)) > 1.0e-12,
                     "camera up must not be parallel to its view direction")
        self.position = position
        self.target = target
        self.up = normalize(up)
    }
}

/// Per-body presentation overrides uploaded by ``GPUSimRenderer``. This is
/// also the exact 32-byte ABI passed to custom ``GPUSimRenderableScene``
/// backends: `albedo.w > 0` enables the sRGB color override, while
/// `emissive.rgb` is linear HDR radiance added after lighting.
public struct GPUSimRenderAppearance: Sendable, Equatable {
    public var albedo: SIMD4<Float>
    public var emissive: SIMD4<Float>

    public init(color: F3? = nil, emissive: F3 = .zero) {
        self.albedo = color.map { SIMD4($0, 1) } ?? .zero
        self.emissive = SIMD4(emissive, 0)
    }
}

/// Coarse world bounds of the renderable content, used to fit the
/// directional-shadow volume. Optional: without it the light volume
/// follows the camera focus, which cannot cover off-focus content.
public struct GPUSimContentBounds: Sendable, Equatable {
    public var center: F3
    public var radius: Float

    public init(center: F3, radius: Float) {
        self.center = center
        self.radius = radius
    }
}

/// Analytic geometry accepted by the auxiliary-instance rendering pass.
public enum GPUSimRenderPrimitive: Sendable, Equatable {
    case box(size: F3)
    case sphere(radius: Float)
    case torus(majorRadius: Float, minorRadius: Float)
    case capsule(length: Float, radius: Float)
}

/// ABI written into the `instances` buffer by
/// ``GPUSimRenderableScene/encodeRenderInstances(_:instances:colorMode:appearanceOverrides:)``.
public struct GPUSimRenderInstance {
    /// Column-major world transform, including primitive scale.
    public var model: simd_float4x4
    /// RGB is sRGB albedo. W selects box (0), sphere (1), torus (2), or
    /// capsule (3).
    public var color: SIMD4<Float>
    /// X/Y are torus or capsule dimensions, Z is the bounding radius, and W
    /// is nonzero for a dynamic shadow caster.
    public var parameters: SIMD4<Float>
    /// RGB is linear HDR emission. W is opacity when this value is submitted
    /// through the auxiliary-instance pass; scene geometry remains opaque.
    public var material: SIMD4<Float>

    public init(
        model: simd_float4x4,
        color: SIMD4<Float>,
        parameters: SIMD4<Float>,
        material: SIMD4<Float> = SIMD4(0, 0, 0, 1)
    ) {
        self.model = model
        self.color = color
        self.parameters = parameters
        self.material = material
    }

    /// Builds an analytic primitive for ``GPUSimRenderer/auxiliaryInstances``
    /// or ``GPUSimRendererSource/rendererAuxiliaryInstances``.
    public init(
        primitive: GPUSimRenderPrimitive,
        position: F3,
        rotation: Quat = Quat(real: 1, imag: .zero),
        color: F3,
        emissive: F3 = .zero,
        opacity: Float = 1,
        castsShadow: Bool = false
    ) {
        var model = simd_float4x4(rotation.normalized)
        model.columns.3 = SIMD4(position, 1)
        let kind: Float
        var parameters = SIMD4<Float>.zero
        switch primitive {
        case .box(let size):
            precondition(size.x >= 0 && size.y >= 0 && size.z >= 0,
                         "box size must be nonnegative")
            model.columns.0 *= size.x
            model.columns.1 *= size.y
            model.columns.2 *= size.z
            kind = 0
            parameters.z = length(size) * 0.5
        case .sphere(let radius):
            precondition(radius >= 0, "sphere radius must be nonnegative")
            let diameter = 2 * radius
            model.columns.0 *= diameter
            model.columns.1 *= diameter
            model.columns.2 *= diameter
            kind = 1
            parameters.z = radius
        case .torus(let majorRadius, let minorRadius):
            precondition(majorRadius >= 0 && minorRadius >= 0,
                         "torus radii must be nonnegative")
            kind = 2
            parameters.x = majorRadius
            parameters.y = minorRadius
            parameters.z = majorRadius + minorRadius
        case .capsule(let capsuleLength, let radius):
            precondition(capsuleLength >= 0 && radius >= 0,
                         "capsule dimensions must be nonnegative")
            kind = 3
            parameters.x = capsuleLength
            parameters.y = radius
            parameters.z = capsuleLength * 0.5 + radius
        }
        parameters.w = castsShadow ? 1 : 0
        self.init(
            model: model,
            color: SIMD4(color, kind),
            parameters: parameters,
            material: SIMD4(emissive, min(max(opacity, 0), 1)))
    }
}

/// Vertex ABI for ``GPUSimSkinnedRenderSurface``.
public struct GPUSimSkinRenderVertex {
    public var position: SIMD4<Float>
    public var normal: SIMD4<Float>

    public init(position: SIMD4<Float>, normal: SIMD4<Float>) {
        self.position = position
        self.normal = normal
    }
}

/// Vertex ABI for rigid and convex-debug surfaces. `positionBody.w` stores
/// the owning body index as a `UInt32` bit pattern. `normal.w > 0` opts into
/// metallic/roughness shading: normal.w is roughness, color.w is metallic.
/// Zero normal.w preserves the legacy dielectric, roughness-0.45 appearance.
public struct GPUSimRigidMeshRenderVertex {
    public var positionBody: SIMD4<Float>
    public var normal: SIMD4<Float>
    public var color: SIMD4<Float>

    public init(
        positionBody: SIMD4<Float>,
        normal: SIMD4<Float>,
        color: SIMD4<Float>
    ) {
        self.positionBody = positionBody
        self.normal = normal
        self.color = color
    }
}

/// `triangles` contains three packed `UInt32` corners per triangle. Bits
/// 0...20 are the position index, bit 21 requests flat shading, bit 22 marks
/// a rim, bit 23 marks the back layer, and bits 24...31 select its color.
public struct GPUSimSoftRenderSurface {
    public let triangles: MTLBuffer
    public let triangleCount: Int
    public let positions: MTLBuffer
    public let normals: MTLBuffer

    public init(
        triangles: MTLBuffer,
        triangleCount: Int,
        positions: MTLBuffer,
        normals: MTLBuffer
    ) {
        self.triangles = triangles
        self.triangleCount = triangleCount
        self.positions = positions
        self.normals = normals
    }
}

/// `triangles` contains three packed `UInt32` corners per triangle: bits
/// 0...23 are the vertex index and bits 24...31 select its color.
public struct GPUSimSkinnedRenderSurface {
    public let triangles: MTLBuffer
    public let triangleCount: Int
    public let vertices: MTLBuffer

    public init(
        triangles: MTLBuffer,
        triangleCount: Int,
        vertices: MTLBuffer
    ) {
        self.triangles = triangles
        self.triangleCount = triangleCount
        self.vertices = vertices
    }
}

public struct GPUSimRigidMeshRenderSurface {
    public let vertices: MTLBuffer
    public let indices: MTLBuffer
    public let indexCount: Int
    public let positions: MTLBuffer
    public let rotations: MTLBuffer

    public init(
        vertices: MTLBuffer,
        indices: MTLBuffer,
        indexCount: Int,
        positions: MTLBuffer,
        rotations: MTLBuffer
    ) {
        self.vertices = vertices
        self.indices = indices
        self.indexCount = indexCount
        self.positions = positions
        self.rotations = rotations
    }
}

public struct GPUSimConvexDebugRenderSurface {
    public let triangleVertices: MTLBuffer
    public let triangleVertexCount: Int
    public let edgeVertices: MTLBuffer
    public let edgeVertexCount: Int
    public let positions: MTLBuffer
    public let rotations: MTLBuffer

    public init(
        triangleVertices: MTLBuffer,
        triangleVertexCount: Int,
        edgeVertices: MTLBuffer,
        edgeVertexCount: Int,
        positions: MTLBuffer,
        rotations: MTLBuffer
    ) {
        self.triangleVertices = triangleVertices
        self.triangleVertexCount = triangleVertexCount
        self.edgeVertices = edgeVertices
        self.edgeVertexCount = edgeVertexCount
        self.positions = positions
        self.rotations = rotations
    }
}

/// Backend-neutral GPU data contract consumed by ``GPUSimRenderer``.
/// Backends can adopt it without exposing their CPU-side simulation model.
public protocol GPUSimRenderableScene: AnyObject {
    /// Device that owns every buffer returned by this scene.
    var renderDevice: MTLDevice { get }
    /// Number of stable body indices addressable by per-body appearance
    /// overrides. Zero disables that optional presentation feature.
    var renderBodyCount: Int { get }
    var renderRigidInstanceCount: Int { get }
    var rendererStateIsValid: Bool { get }
    var renderCameraHint: GPUSimRenderCameraHint { get }
    var softRenderSurface: GPUSimSoftRenderSurface? { get }
    var skinnedRenderSurface: GPUSimSkinnedRenderSurface? { get }
    var rigidMeshRenderSurface: GPUSimRigidMeshRenderSurface? { get }
    var convexDebugRenderSurface: GPUSimConvexDebugRenderSurface? { get }
    /// Coarse content bounds for shadow fitting; nil = camera-driven.
    var renderContentBounds: GPUSimContentBounds? { get }
    /// Whether the renderer must retire each frame before returning.
    /// GPUSolver requires it (render reads pose buffers another queue
    /// mutates); a scene whose buffers are all triple-buffered or written
    /// inside the frame's own command buffer can return false and let CPU
    /// and GPU pipeline.
    var renderSceneRequiresFrameRetirement: Bool { get }
    /// Encodes `renderRigidInstanceCount` values with
    /// `GPUSimRenderInstance` layout. A backend may also refresh the optional
    /// surface buffers in this command buffer before returning. When non-nil,
    /// `appearanceOverrides` contains `renderBodyCount` elements with
    /// `GPUSimRenderAppearance` layout and belongs to `renderDevice`.
    func encodeRenderInstances(
        _ commandBuffer: MTLCommandBuffer,
        instances: MTLBuffer,
        colorMode: GPUSimRenderColorMode,
        appearanceOverrides: MTLBuffer?
    ) throws
}

extension GPUSolver: GPUSimRenderableScene {
    public var renderDevice: MTLDevice { device }

    /// Live bounds of the rendered colliders: current body positions plus
    /// each collider's local offset and conservative half extent, with wide
    /// static scenery excluded. Computed per call - a few hundred entries,
    /// render-thread cheap.
    public var renderContentBounds: GPUSimContentBounds? {
        guard let bounds = renderedContentBounds else { return nil }
        return GPUSimContentBounds(center: bounds.center,
                                   radius: bounds.radius)
    }
    public var renderBodyCount: Int { bodyCount }
    public var renderRigidInstanceCount: Int { renderRigidBodyCount }
    public var rendererStateIsValid: Bool { runtimeFailure == nil }

    public var renderCameraHint: GPUSimRenderCameraHint {
        GPUSimRenderCameraHint(
            distance: settings.cameraDistance > 0 ? settings.cameraDistance : 30,
            target: F3(
                settings.cameraTargetX.isFinite ? settings.cameraTargetX : 0,
                settings.cameraTargetY.isFinite ? settings.cameraTargetY : 0,
                settings.cameraTargetZ != 0 ? settings.cameraTargetZ : 3
            ),
            azimuth: settings.cameraAzimuth.isFinite
                ? settings.cameraAzimuth : 0.9,
            elevation: settings.cameraElevation.isFinite
                ? settings.cameraElevation : 0.35
        )
    }

    public var softRenderSurface: GPUSimSoftRenderSurface? {
        renderSurface.map {
            GPUSimSoftRenderSurface(
                triangles: $0.tris,
                triangleCount: $0.triCount,
                positions: $0.positions,
                normals: $0.normals
            )
        }
    }

    public var skinnedRenderSurface: GPUSimSkinnedRenderSurface? {
        renderSkinnedSurface.map {
            GPUSimSkinnedRenderSurface(
                triangles: $0.tris,
                triangleCount: $0.triCount,
                vertices: $0.vertices
            )
        }
    }

    public var rigidMeshRenderSurface: GPUSimRigidMeshRenderSurface? {
        renderIndexedRigidMeshSurface.map {
            GPUSimRigidMeshRenderSurface(
                vertices: $0.vertices,
                indices: $0.indices,
                indexCount: $0.indexCount,
                positions: $0.positions,
                rotations: $0.rotations
            )
        }
    }

    public var convexDebugRenderSurface: GPUSimConvexDebugRenderSurface? {
        renderConvexCollisionSurface.map {
            GPUSimConvexDebugRenderSurface(
                triangleVertices: $0.triangleVertices,
                triangleVertexCount: $0.triangleVertexCount,
                edgeVertices: $0.edgeVertices,
                edgeVertexCount: $0.edgeVertexCount,
                positions: $0.positions,
                rotations: $0.rotations
            )
        }
    }

    public func encodeRenderInstances(
        _ commandBuffer: MTLCommandBuffer,
        instances: MTLBuffer,
        colorMode: GPUSimRenderColorMode,
        appearanceOverrides: MTLBuffer?
    ) throws {
        try encodeBuildInstancesChecked(
            commandBuffer,
            instances: instances,
            colorMode: colorMode.rawValue,
            appearanceOverrides: appearanceOverrides
        )
    }
}

/// Optional live source for applications that replace solvers or advance a
/// simulation immediately before each rendered frame. A fixed scene can use
/// ``GPUSimRenderer/init(device:solver:)`` and does not need a source object.
@MainActor
public protocol GPUSimRendererSource: AnyObject {
    var renderScene: (any GPUSimRenderableScene)? { get }
    var rendererOptions: GPUSimRenderOptions { get }
    /// Presentation-only body overrides keyed by the scene's stable body
    /// index. Invalid indices are ignored across scene swaps.
    var rendererBodyAppearances: [Int: GPUSimRenderAppearance] { get }
    /// World-space primitives drawn in a depth-tested alpha-blended pass
    /// after the opaque simulation scene.
    var rendererAuxiliaryInstances: [GPUSimRenderInstance] { get }
    /// Increment when a different scene should receive its default camera.
    /// Rebuilds of the same scene can keep the value stable to preserve the
    /// user's camera.
    var rendererSceneRevision: Int { get }
    func rendererWillDrawFrame()
    func rendererDidFail(_ message: String)
}

public extension GPUSimRenderableScene {
    var renderContentBounds: GPUSimContentBounds? { nil }
    var renderSceneRequiresFrameRetirement: Bool { true }
}

public extension GPUSimRendererSource {
    var rendererOptions: GPUSimRenderOptions { GPUSimRenderOptions() }
    var rendererBodyAppearances: [Int: GPUSimRenderAppearance] { [:] }
    var rendererAuxiliaryInstances: [GPUSimRenderInstance] { [] }
    var rendererSceneRevision: Int { 0 }
    func rendererWillDrawFrame() {}
    func rendererDidFail(_ message: String) {}
}

// Renderer: PBR (GGX metallic-roughness) + ACES, sRGB-correct framebuffer,
// GTAO-style ambient occlusion from a depth/normal prepass, directional
// shadow mapping, horizon-only fog, and analytic instanced geometry.

let renderShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct RenderInstance {
    float4x4 model;
    float4 color;       // rgb albedo (sRGB), w = shape type (0 box 1 sphere 2 torus 3 capsule)
    float4 params;      // torus/capsule dims; z = bounding radius, w = dynamic flag
    float4 material;    // rgb linear emission, w = opacity
};

struct RenderAppearance {
    float4 albedo;      // rgb sRGB override, w = enabled
    float4 emissive;    // rgb linear HDR radiance
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
    float4x4 invViewProj;
    float4x4 prevInvViewProj;
    float4 aoProjection; // xy: depth A/B (deviceDepth=A+B/viewZ); zw: inverse focal scales
};

// World position from depth: 4 bytes per tap instead of a 16-byte stored
// position, with full float32 precision - the position textures this
// replaces were the renderer's dominant bandwidth cost.
static inline float3 worldFromDepth(float2 uv, float depth, float4x4 invVP) {
    float4 clip = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    float4 w = invVP * clip;
    return w.xyz / w.w;
}


struct VOut {
    float4 position [[position]];
    float3 normal;
    float3 world;
    float3 albedo;
    float3 emissive;
    float opacity;
    float flatShade;
    float2 pbr; // perceptual roughness, metallic
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

// The drawable is an 8-bit sRGB target. A smooth near-white lighting ramp
// can otherwise collapse into wide one-code plateaus after tone mapping.
// Dither in encoded sRGB space, then return to linear so the attachment's
// hardware conversion produces the intended sub-LSB distribution. The noise
// is deterministic in screen space, zero-mean, and cannot shimmer over time.
inline float3 linearToSRGBExact(float3 c) {
    c = clamp(c, 0.0, 1.0);
    float3 low = c * 12.92;
    float3 high = 1.055 * pow(c, float3(1.0 / 2.4)) - 0.055;
    return select(low, high, c > 0.0031308);
}

inline float3 sRGBToLinearExact(float3 c) {
    c = clamp(c, 0.0, 1.0);
    float3 low = c / 12.92;
    float3 high = pow((c + 0.055) / 1.055, float3(2.4));
    return select(low, high, c > 0.04045);
}

inline float interleavedGradientNoise(float2 pixel) {
    float2 p = floor(pixel);
    return fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715))));
}

inline float3 displayColorSRGB8(float3 toneMappedLinear, float2 pixel) {
    float noise = interleavedGradientNoise(pixel) - 0.5;
    float3 encoded = linearToSRGBExact(toneMappedLinear);
    encoded = clamp(encoded + noise / 255.0, 0.0, 1.0);
    return sRGBToLinearExact(encoded);
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

    // Bias against the rasterized receiver plane, not the interpolated
    // shading normal.  On a smooth-shaded, coarsely tessellated shape those
    // normals differ most near the silhouette; using the shading normal there
    // produces the view-dependent triangle/stripe acne the bias is meant to
    // remove.  The capped tangent term supplies enough bias at grazing light
    // angles without detaching contact shadows.
    float3 shadeN = normalize(normal);
    float3 plane = cross(dfdx(world), dfdy(world));
    float planeLen2 = dot(plane, plane);
    float3 geomN = planeLen2 > 1e-12 ? plane * rsqrt(planeLen2) : shadeN;
    if (dot(geomN, shadeN) < 0.0) geomN = -geomN;
    float geomNdL = saturate(dot(geomN, -U.lightDir.xyz));
    float slope = min(2.0, sqrt(saturate(1.0 - geomNdL * geomNdL))
                           / max(geomNdL, 0.2));
    float receiverDepth = ndc.z - (U.shadowParams.x + U.shadowParams.y * slope);

    // A PCF tap does not sample the receiver at this fragment's light-space
    // XY: it samples a neighboring shadow texel.  Reusing the center depth
    // for every tap compares a tilted receiver against the wrong point on
    // its own plane.  The resulting 0/1 tap-count transitions are the broad,
    // discrete diagonal bands visible on otherwise flat boxes.  Reconstruct
    // dz/duv from screen derivatives and compare every tap against the depth
    // of the receiver plane at that tap instead.
    float3 shadowCoord = float3(uv, ndc.z);
    float3 shadowDx = dfdx(shadowCoord);
    float3 shadowDy = dfdy(shadowCoord);
    float determinant = shadowDx.x * shadowDy.y
                      - shadowDx.y * shadowDy.x;
    float2 receiverDepthGradient = float2(0.0);
    if (abs(determinant) > 1e-10) {
        receiverDepthGradient = float2(
            shadowDx.z * shadowDy.y - shadowDx.y * shadowDy.z,
            shadowDx.x * shadowDy.z - shadowDx.z * shadowDy.x)
            / determinant;
    }
    float2 texel = 1.0 / float2(shadowTex.get_width(), shadowTex.get_height());
    constexpr sampler shadowSampler(coord::normalized, filter::linear,
                                    address::clamp_to_edge,
                                    compare_func::less_equal);
    float visible = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 tapOffset = float2(float(x), float(y)) * texel;
            float tapReceiverDepth = receiverDepth
                + dot(receiverDepthGradient, tapOffset);
            visible += shadowTex.sample_compare(
                shadowSampler, uv + tapOffset, tapReceiverDepth);
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
    o.pbr = float2(0.45, 0.0);
    o.position = float4(0, 0, -2, 1);
    o.normal = float3(0); o.world = float3(0); o.albedo = float3(0);
    o.emissive = float3(0); o.opacity = 0;
    o.flatShade = 0;
    return o;
}

inline VOut emit(float3 p, float3 n, RenderInstance inst, constant Uniforms& U) {
    float4 world = inst.model * float4(p, 1);
    float3x3 rot = float3x3(normalize(inst.model[0].xyz),
                            normalize(inst.model[1].xyz),
                            normalize(inst.model[2].xyz));
    VOut o;
    o.pbr = float2(0.45, 0.0);
    o.position = U.viewProj * world;
    o.normal = rot * n;
    o.world = world.xyz;
    o.albedo = srgbToLin(inst.color.rgb);
    o.emissive = inst.material.rgb;
    o.opacity = inst.material.w;
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
                        device const float4* normals [[buffer(3)]],
                        device const RenderAppearance* appearanceOverrides [[buffer(4)]],
                        constant uint& hasAppearanceOverrides [[buffer(5)]])
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
    o.pbr = float2(0.45, 0.0);
    o.position = U.viewProj * float4(w, 1);
    o.world = w;
    o.normal = nt.xyz;
    o.flatShade = ((packed >> 21) & 1u) != 0 ? 1.0 : 0.0;
    // warm fabric-ish tones, one per sheet/body
    o.albedo = srgbToLin(mix(float3(0.90), softPalette(comp * 7u + 2u), 0.72));
    o.emissive = float3(0);
    o.opacity = 1;
    if (hasAppearanceOverrides != 0) {
        RenderAppearance appearance = appearanceOverrides[body];
        if (appearance.albedo.w > 0.0f) {
            o.albedo = srgbToLin(appearance.albedo.rgb);
        }
        o.emissive = appearance.emissive.rgb;
    }
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
    o.pbr = float2(0.45, 0.0);
    o.position = U.viewProj * float4(sv.position.xyz, 1);
    o.world = sv.position.xyz;
    o.normal = normalize(sv.normal.xyz);
    o.flatShade = 0.0;
    o.albedo = srgbToLin(mix(float3(0.90), softPalette(comp * 5u + 11u), 0.76));
    o.emissive = float3(0);
    o.opacity = 1;
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
    device const float4* posAng [[buffer(3)]],
    device const RenderAppearance* appearanceOverrides [[buffer(4)]],
    constant uint& hasAppearanceOverrides [[buffer(5)]])
{
    RigidMeshVertex v = vertices[vid];
    uint body = as_type<uint>(v.positionBody.w);
    float4 q = posAng[body];
    float3 world = posLin[body].xyz + rigidMeshRotate(q, v.positionBody.xyz);
    VOut o;
    o.pbr = float2(0.45, 0.0);
    o.position = U.viewProj * float4(world, 1);
    o.world = world;
    o.normal = normalize(rigidMeshRotate(q, v.normal.xyz));
    o.albedo = srgbToLin(v.color.rgb);
    if (v.normal.w > 0.0) o.pbr = float2(clamp(v.normal.w, 0.02, 1.0),
                                       clamp(v.color.w, 0.0, 1.0));
    o.emissive = float3(0);
    o.opacity = 1;
    if (hasAppearanceOverrides != 0) {
        RenderAppearance appearance = appearanceOverrides[body];
        if (appearance.albedo.w > 0.0f) {
            o.albedo = srgbToLin(appearance.albedo.rgb);
        }
        o.emissive = appearance.emissive.rgb;
    }
    o.flatShade = 0;
    return o;
}

fragment float4 convex_debug_fill_fragment(VOut in [[stage_in]])
{
    return float4(mix(in.albedo, float3(1.0), 0.12), 0.22);
}

fragment float4 convex_debug_wire_fragment(VOut in [[stage_in]])
{
    return float4(mix(in.albedo, float3(1.0), 0.58), 0.94);
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

    float rough = clamp(in.pbr.x, 0.02, 1.0);
    float metal = clamp(in.pbr.y, 0.0, 1.0);
    float a2 = rough * rough; a2 *= a2;
    float dDen = NdH * NdH * (a2 - 1.0) + 1.0;
    float D = a2 / (M_PI_F * dDen * dDen);
    float k = (rough + 1.0); k = k * k / 8.0;
    float G = (NdV / (NdV * (1.0 - k) + k)) * (NdL / (NdL * (1.0 - k) + k));
    float3 F0 = mix(float3(0.04), in.albedo, metal);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - HdV, 5.0);
    float3 spec = D * G * F / max(4.0 * NdV * NdL, 1e-4);

    float shadow = shadowVisibility(in.world, n, U, shadowTex);
    float3 direct = (in.albedo / M_PI_F * (1.0 - metal) * (1.0 - F) + spec) * SUN_COL * NdL * shadow;

    float3 irr = mix(GND_IRR, SKY_IRR, n.z * 0.5 + 0.5);
    float3 ambient = in.albedo * irr * (1.0 - metal);
    float3 R = reflect(-V, n);
    float3 skyRef = mix(HORIZON_LIN, float3(0.42, 0.48, 0.58), clamp(R.z, 0.0, 1.0));
    ambient += skyRef * (F0 + (1.0 - F0) * pow(1.0 - NdV, 5.0))
        * mix(0.50, 0.20, rough);
    ambient *= ao;

    float3 lit = direct + ambient + in.emissive;
    float fog = horizonFog(length(in.world - U.eye.xyz));
    lit = mix(lit, HORIZON_LIN, fog);
    return float4(displayColorSRGB8(acesTonemap(lit), in.position.xy), in.opacity);
}

// ---------------------------------------------------------------------------
// Prepass: normals to one attachment; the depth buffer carries position
// ---------------------------------------------------------------------------
struct PreOut {
    float4 norm [[color(0)]];
};

fragment PreOut prepass_fragment(VOut in [[stage_in]]) {
    PreOut o;
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

    float3 lit = direct + ambient + in.emissive;
    float fog = horizonFog(length(in.world - U.eye.xyz));
    lit = mix(lit, HORIZON_LIN, fog);
    return float4(displayColorSRGB8(acesTonemap(lit), in.position.xy), in.opacity);
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
    o.pbr = float2(0.45, 0.0);
    o.position = U.viewProj * float4(w, 1);
    o.world = w;
    o.normal = nt.xyz;
    o.albedo = float3(0);
    o.emissive = float3(0);
    o.opacity = 1;
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
                              depth2d<float> depthTex [[texture(0)]],
                              texture2d<float> normTex [[texture(1)]])
{
    // GTAO (Jimenez et al. 2016 / XeGTAO-style): march several
    // screen-space slices, find both horizon angles per slice, clamp to the
    // normal hemisphere, then integrate the cosine-weighted occluded arc.
    constexpr sampler smp(filter::nearest, address::clamp_to_edge);
    float dC = depthTex.sample(smp, in.uv);
    if (dC >= 1.0) return float4(1);
    float3 camForward = normalize(cross(U.camRight.xyz, U.camUp.xyz));
    float3 worldN = normalize(normTex.sample(smp, in.uv).xyz);
    // Work in the orthonormal (screen right, screen down, forward) basis.
    // Linearizing depth and scaling a ray replaces a matrix multiply per
    // tap, and subtraction stays precise when the camera is far from origin.
    float3 N = float3(dot(worldN, U.camRight.xyz), dot(worldN, U.camUp.xyz),
                      dot(worldN, camForward));
    float2 pixelToView = 2.0 * U.aoProjection.zw / U.screen.xy;
    float2 sliceScale = pixelToView / pixelToView.y;
    float centerZ = U.aoProjection.y / (dC - U.aoProjection.x);
    float3 P = float3(in.position.xy * pixelToView - U.aoProjection.zw, 1.0) * centerZ;
    float3 V = normalize(-P);
    float viewDepth = max(centerZ, 0.25);

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

    // Independent spatial angle/radius seeds; linear IGN bands and a radial
    // jitter derived from the angle seed remain correlated after denoising.
    float2 px = floor(in.position.xy);
    uint seed = uint(px.x) + uint(px.y) * 65537u;
    seed ^= seed >> 17; seed *= 0xed5ad4bbu;
    seed ^= seed >> 11; seed *= 0xac4c1b51u;
    seed ^= seed >> 15; seed *= 0x31848babu;
    seed ^= seed >> 14;
    float2 noise = float2(seed & 65535u, seed >> 16) / 65536.0;
    float ang = fract(noise.x + U.temporal.x) * M_PI_F;
    float stepJit = fract(noise.y + U.temporal.x * 5.0);

    const int SLICES = 3;
    const int STEPS = 4;
    float occlusion = 0.0;

    for (int sl = 0; sl < SLICES; sl++) {
        float phi = ang + float(sl) * (M_PI_F / float(SLICES));
        float2 dirPx = float2(cos(phi), sin(phi));

        // ANALYTIC slice tangent: the view direction this screen-space
        // march corresponds to, projected perpendicular to V. Deriving it
        // from the camera basis (not from samples) keeps the slice plane
        // exact and view-consistent.
        float3 dirV = float3(dirPx * sliceScale, 0.0);
        float3 omega = dirV - V * dot(dirV, V);
        float ol = length(omega);
        if (ol < 1e-4) continue;
        omega /= ol;

        // project N into the slice plane (spanned by V and omega)
        float3 sliceN = cross(V, omega);
        float3 projN = N - sliceN * dot(N, sliceN);
        float projLen = length(projN);
        if (projLen < 1e-4) continue;
        float3 pn = projN / projLen;
        float cosNV = saturate(dot(pn, V));
        float n = acos(cosNV) * (dot(pn, omega) >= 0.0 ? 1.0 : -1.0);

        // XeGTAO fades discarded/out-of-radius samples toward the normal
        // hemisphere edge instead of -1, avoiding grazing-angle detail loss.
        float sinN = sin(n);
        float lowH0 = -sinN;
        float lowH1 = sinN;

        // horizon cosines per side of the slice (classified by the
        // actual sample offset, so screen/UV orientation can't flip them)
        float cosH0 = lowH0, cosH1 = lowH1;
        float minS = min(0.95, 1.3 / max(pxRadius, 1e-3));
        for (int st = 1; st <= STEPS; st++) {
            // Shift each stratum independently instead of moving every
            // radius in every slice together. Share the offset across sides.
            float stepNoise = fract(stepJit + float(sl + (st - 1) * STEPS) * 0.6180339887);
            float u = (float(st) - 1.0 + stepNoise) / float(STEPS);
            float t = minS + (1.0 - minS) * (u * u);
            int2 offset = int2(round(dirPx * (t * pxRadius)));
            for (int side = 0; side < 2; side++) {
                // Depth belongs to a texel CENTER. Unprojecting a nearest
                // depth at the continuous march UV fabricates height steps
                // on tilted planes. Off-screen taps have no geometry data.
                int2 samplePixel = int2(px) + (side == 0 ? offset : -offset);
                if (any(samplePixel < 0) || any(samplePixel >= int2(U.screen.xy))) continue;
                float dQ = depthTex.read(uint2(samplePixel));
                if (dQ >= 1.0) continue;
                float sampleZ = U.aoProjection.y / (dQ - U.aoProjection.x);
                float3 w = float3((float2(samplePixel) + 0.5) * pixelToView
                                 - U.aoProjection.zw, 1.0) * sampleZ - P;
                float l = length(w);
                if (l < 1e-4) continue;
                // Rounding moves samples off the ideal slice. Coplanar or
                // below-tangent points cannot occlude the normal hemisphere,
                // even if their view cosine exceeds this slice's horizon.
                // Cover half-float normal error plus sub-mm depth roundoff;
                // this is a tangent-plane tolerance, not an AO radius bias.
                if (dot(N, w) <= 0.001 * l + 0.0001) continue;
                float c = dot(w / l, V);
                float weight = saturate((Reff - l) / falloffRange);
                if (dot(w, omega) >= 0.0) {
                    cosH0 = max(cosH0, mix(lowH0, c, weight));
                } else {
                    cosH1 = max(cosH1, mix(lowH1, c, weight));
                }
            }
        }

        // An unchanged horizon has exactly zero missing visibility. This
        // common case also avoids four inverse/trigonometric evaluations.
        if (cosH0 == lowH0 && cosH1 == lowH1) continue;

        // signed horizon angles in the slice plane (+ = toward omega),
        // clamped to the hemisphere around the projected normal
        float h1 =  acos(clamp(cosH0, -1.0, 1.0));
        float h2 = -acos(clamp(cosH1, -1.0, 1.0));
        h1 = n + min(h1 - n,  M_PI_F / 2.0);
        h2 = n + max(h2 - n, -M_PI_F / 2.0);

        // analytic cosine-weighted visible arc (GTAO inner integral)
        float a1 = 0.25 * (-cos(2.0 * h1 - n) + cosNV + 2.0 * h1 * sinN);
        float a2 = 0.25 * (-cos(2.0 * h2 - n) + cosNV + 2.0 * h2 * sinN);
        // Integrate missing visibility relative to the analytic open arc.
        // The full unoccluded hemisphere is exactly 1; estimating it with
        // only three slices and then saturating creates a normal-dependent
        // dark bias and structured noise even without any occluders.
        float unoccluded = cosNV + n * sinN;
        occlusion += projLen * max(0.0, unoccluded - a1 - a2);
    }
    float ao = saturate(1.0 - occlusion / float(SLICES));
    ao = pow(ao, 1.25);    // slight contrast shaping
    ao = max(ao, 0.03);    // visible pixels should not reach total black
    ao = mix(1.0, ao, farFade);
    return float4(ao, ao, ao, 1);
}

// Store the surface normal beside AO in the temporal target.  Octahedral
// packing keeps the history buffer compact (RGBA8) while giving the resolve
// enough orientation information to reject history from another face/object.
inline float2 encodeOctahedral(float3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    float2 signN = select(float2(-1.0), float2(1.0), n.xy >= 0.0);
    float2 e = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * signN;
    return e * 0.5 + 0.5;
}

inline float3 decodeOctahedral(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += select(float2(t), float2(-t), n.xy >= 0.0);
    return normalize(n);
}

inline float4 aoHistoryValue(float ao, float3 normal) {
    return float4(ao, encodeOctahedral(normal), 1.0);
}

fragment float4 temporal_fragment(FSOut in [[stage_in]],
                                  constant Uniforms& U [[buffer(1)]],
                                  texture2d<float> rawAO [[texture(0)]],
                                  texture2d<float> histAO [[texture(1)]],
                                  depth2d<float> depthTex [[texture(2)]],
                                  depth2d<float> prevDepthTex [[texture(3)]],
                                  texture2d<float> normTex [[texture(4)]])
{
    // temporal accumulation with reprojection: find this pixel's world
    // point in the previous frame, then reuse accumulated AO only while the
    // position and surface orientation still agree.
    constexpr sampler smp(filter::nearest, address::clamp_to_edge);
    constexpr sampler lsmp(filter::linear, address::clamp_to_edge);
    float cur = rawAO.sample(smp, in.uv).r;
    float dC = depthTex.sample(smp, in.uv);
    if (dC >= 1.0) return float4(cur, 0.5, 0.5, 0.0);
    float3 P = worldFromDepth(in.uv, dC, U.invViewProj);
    float3 N = normalize(normTex.sample(smp, in.uv).xyz);
    float4 current = aoHistoryValue(cur, N);
    if (U.temporal.y >= 1.0) return current;

    float4 clipPrev = U.prevViewProj * float4(P, 1.0);
    if (clipPrev.w <= 0.0) return current;
    float2 ndc = clipPrev.xy / clipPrev.w;
    float2 uvPrev = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uvPrev < 0.0) || any(uvPrev >= 1.0)) return current;

    uint2 historyPixel = uint2(uvPrev * U.screen.xy);
    float2 historyUV = (float2(historyPixel) + 0.5) / U.screen.xy;
    float dPrev = prevDepthTex.read(historyPixel);
    if (dPrev >= 1.0) return current;
    float3 Qw = worldFromDepth(historyUV, dPrev, U.prevInvViewProj);

    // AO is continuous and may be bilinearly reprojected.  The octahedral
    // normal is encoded metadata: filtering its packed coordinates across a
    // hard edge or the octahedral fold can decode to an unrelated direction.
    float historyAO = histAO.sample(lsmp, uvPrev).r;
    float2 historyNormal = histAO.read(historyPixel).gb;
    float3 historyN = decodeOctahedral(historyNormal);

    // A nearest reprojection naturally differs by a fraction of one world
    // pixel even for static geometry.  Derivative-scaled thresholds preserve
    // that case but reject real motion much sooner than the old fixed 7 cm
    // window, particularly for close-up objects.
    float pixelWorld = clamp(min(length(dfdx(P)), length(dfdy(P))), 0.002, 0.03);
    float motionStart = 0.002 + 0.35 * pixelWorld;
    float motionEnd = 0.012 + 1.25 * pixelWorld;
    float positionReuse = 1.0 - smoothstep(
        motionStart, motionEnd, length(Qw - P));
    float normalReuse = smoothstep(0.58, 0.92, dot(N, historyN));
    float reuse = positionReuse * normalReuse;
    if (reuse <= 0.0) return current;

    // Clip history to a cheap five-sample current-frame neighborhood.  This
    // removes dark/light trails immediately when AO changes, while stable
    // stochastic variation still benefits from temporal convergence.
    float2 pixel = 1.0 / U.screen.xy;
    constexpr int2 offsets[4] = {
        int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)
    };
    float localMin = cur, localMax = cur;
    for (int i = 0; i < 4; ++i) {
        float value = rawAO.sample(smp, in.uv + float2(offsets[i]) * pixel).r;
        localMin = min(localMin, value);
        localMax = max(localMax, value);
    }
    float padding = max(0.015, (localMax - localMin) * 0.15);
    float hist = clamp(historyAO, localMin - padding, localMax + padding);
    float blend = mix(1.0, U.temporal.y, reuse);
    return aoHistoryValue(mix(hist, cur, blend), N);
}

fragment float4 blur_fragment(FSOut in [[stage_in]],
                              constant Uniforms& U [[buffer(1)]],
                              texture2d<float> src [[texture(0)]],
                              depth2d<float> depthTex [[texture(1)]],
                              texture2d<float> normTex [[texture(2)]])
{
    // Thirteen-tap Gaussian bilateral blur.  Position and normal agreement
    // keep AO on its own surface at silhouettes and hard edges.  The diamond
    // footprint costs fewer texture reads than the old 25-tap box kernel.
    constexpr sampler smp(filter::nearest, address::clamp_to_edge);
    float2 px = 1.0 / U.screen.xy;
    float cD = depthTex.sample(smp, in.uv);
    if (cD >= 1.0) return float4(1);
    float3 cP = worldFromDepth(in.uv, cD, U.invViewProj);
    float3 cN = normalize(normTex.sample(smp, in.uv).xyz);
    constexpr int2 offsets[13] = {
        int2( 0, 0),
        int2(-1, 0), int2( 1, 0), int2( 0,-1), int2( 0, 1),
        int2(-1,-1), int2( 1,-1), int2(-1, 1), int2( 1, 1),
        int2(-2, 0), int2( 2, 0), int2( 0,-2), int2( 0, 2)
    };
    float acc = 0.0, wsum = 0.0;
    for (int i = 0; i < 13; ++i) {
        float2 tap = float2(offsets[i]);
        int2 samplePixel = int2(in.position.xy) + offsets[i];
        if (any(samplePixel < 0) || any(samplePixel >= int2(U.screen.xy))) continue;
        float2 uv2 = in.uv + tap * px;
        float qD = depthTex.sample(smp, uv2);
        if (qD >= 1.0) continue;
        float3 qN = normalize(normTex.sample(smp, uv2).xyz);
        float3 delta = worldFromDepth(uv2, qD, U.invViewProj) - cP;
        float spatialWeight = exp(-0.5 * dot(tap, tap));
        float positionWeight = exp(-dot(delta, delta) * 8.0);
        float normalWeight = pow(saturate(dot(cN, qN)), 12.0);
        float w = spatialWeight * positionWeight * normalWeight;
        acc += src.sample(smp, uv2).r * w;
        wsum += w;
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
    return float4(displayColorSRGB8(acesTonemap(c), in.position.xy), 1);
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
    float4 norm [[color(0)]];
};
fragment FloorPre floor_prepass_fragment(FloorOut in [[stage_in]]) {
    FloorPre o;
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
    return float4(displayColorSRGB8(acesTonemap(lit), in.position.xy), 1);
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
    var invViewProj: simd_float4x4
    var prevInvViewProj: simd_float4x4
    var aoProjection: SIMD4<Float>
}

let SPHV = 12 * 18 * 6
let TORV = 24 * 12 * 6
let CAPV = (2 * 6 + 1) * 16 * 6

public enum GPUSimRendererError: LocalizedError {
    /// The Metal device could not create a rendering command queue.
    case commandQueue
    /// A scene supplied buffers allocated by another Metal device.
    case sceneDeviceMismatch
    /// Runtime shader compilation did not produce a required entry point.
    case shaderFunction(String)
    /// A required depth-stencil state could not be created.
    case depthState(String)

    public var errorDescription: String? {
        switch self {
        case .commandQueue:
            return "could not create the Metal render command queue"
        case .sceneDeviceMismatch:
            return "render scene buffers belong to a different Metal device"
        case .shaderFunction(let name):
            return "render shader function '\(name)' is missing"
        case .depthState(let name):
            return "could not create the \(name) depth state"
        }
    }
}

/// Metal renderer for a live ``GPUSimRenderableScene``.
///
/// The renderer consumes the scene's GPU buffers directly, so normal frame
/// rendering does not read poses back through the CPU. Keep the renderer alive
/// for as long as it is assigned as an `MTKView` delegate.
@MainActor
public final class GPUSimRenderer: NSObject, MTKViewDelegate {
    public nonisolated static let sampleCount = 4
    public nonisolated static let colorFormat = MTLPixelFormat.bgra8Unorm_srgb
    public nonisolated static let shadowMapSize = 2048

    public let device: MTLDevice
    private let queue: MTLCommandQueue
    public weak var source: (any GPUSimRendererSource)?
    public private(set) var scene: (any GPUSimRenderableScene)?
    public var options = GPUSimRenderOptions()
    /// Presentation-only overrides keyed by simulation body index. A live
    /// source's `rendererBodyAppearances` takes precedence when present.
    public var bodyAppearances: [Int: GPUSimRenderAppearance] = [:]
    /// World-space primitives drawn after opaque scene geometry. Fully opaque
    /// values write depth; translucent values are sorted back-to-front and
    /// alpha blended without depth writes.
    public var auxiliaryInstances: [GPUSimRenderInstance] = []
    public private(set) var sceneRevision = 0
    /// Called after a frame completes. This is useful for screenshots,
    /// telemetry, and embedding without coupling the package to app policy.
    /// Set the view's `framebufferOnly` to `false` before reading texture bytes.
    public var frameCompletionHandler: (@MainActor (MTLTexture, Int) -> Void)?

    var boxP, sphereP, torusP, capsuleP, softP, skinP, rigidMeshP: MTLRenderPipelineState!
    var boxAuxP, sphereAuxP, torusAuxP, capsuleAuxP: MTLRenderPipelineState!
    var boxPre, spherePre, torusPre, capsulePre, floorPreP, softPre, skinPre,
        rigidMeshPre: MTLRenderPipelineState!
    var boxShadow, sphereShadow, torusShadow, capsuleShadow, softShadow,
        skinShadow, rigidMeshShadow: MTLRenderPipelineState!
    var skyP, floorP, gtaoP, blurP, temporalP: MTLRenderPipelineState!
    var convexDebugFillP, convexDebugWireP: MTLRenderPipelineState!
    var depthState, noDepthState, debugDepthState: MTLDepthStencilState!
    var auxiliaryOpaqueDepthState, auxiliaryTranslucentDepthState: MTLDepthStencilState!

    var normTex, aoTexA, aoTexB, preDepthTex, shadowTex: MTLTexture?
    var histTex, prevDepthTex: MTLTexture?
    private var aoTexIsWhite = false
    private var aoSize = SIMD2<Int>(0, 0)
    var prevVP: simd_float4x4?
    var frameIdx: UInt32 = 0
    var targetSize = SIMD2<Int>(0, 0)
    var instances: MTLBuffer?
    private var appearanceRing: [MTLBuffer?] = [nil, nil, nil]
    private var appearanceActiveBodies: [[Int]] = [[], [], []]
    private var appearanceCapacity = 0
    private var activeAppearanceBodies: [Int] = []
    private var auxiliaryInstanceRing: [MTLBuffer?] = [nil, nil, nil]
    /// One staging slot per frame, shared by every CPU-written renderer
    /// buffer; the in-flight semaphore caps outstanding frames to the ring
    /// depth so a slot is never rewritten while a frame still reads it.
    private var frameSlot = 0
    private let inFlightFrames = DispatchSemaphore(value: 3)

    /// Orbit azimuth in radians.
    public var azimuth: Float = 0.9
    /// Orbit elevation in radians.
    public var elevation: Float = 0.35
    /// Orbit distance in scene units.
    public var distance: Float = 30
    /// Orbit target in world coordinates.
    public var target = F3(0, 0, 3)
    /// Active app-owned camera. `nil` means the public orbit properties drive
    /// the view. Set through one of the `setCamera` methods.
    public private(set) var cameraPose: GPUSimCameraPose?
    /// When enabled, a new scene revision adopts the camera hints stored in
    /// the scene adapter. Disable this before applying an app-owned camera.
    public var automaticallyFramesScene = true
    public private(set) var viewportSize = SIMD2<Float>(1, 1)
    private var framesDrawn = 0
    private var lastCameraEpoch: Int = -1
    private var lastSceneIdentity: ObjectIdentifier?
    public private(set) var runtimeFailure: String?
    /// GPU time of the most recently completed frame, milliseconds.
    public private(set) var lastFrameGPUMilliseconds: Double = 0

    private func reportFailure(_ message: String) {
        guard runtimeFailure == nil else { return }
        runtimeFailure = message
        // A failed render command can leave temporal targets partially
        // written. Ignore their history if a later frame is retried.
        prevVP = nil
        let target = source
        DispatchQueue.main.async {
            target?.rendererDidFail(message)
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

    /// Creates a renderer for a fixed GPU render scene. Pass `nil` when the
    /// scene will be supplied later with ``setScene(_:resetCamera:)``.
    public init(
        device: MTLDevice,
        scene: (any GPUSimRenderableScene)? = nil
    ) throws {
        if let scene, scene.renderDevice.registryID != device.registryID {
            throw GPUSimRendererError.sceneDeviceMismatch
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw GPUSimRendererError.commandQueue
        }
        self.queue = queue
        self.scene = scene
        super.init()

        let lib = try device.makeLibrary(source: renderShaderSource, options: nil)
        func pipe(_ v: String, _ f: String,
                  samples: Int = GPUSimRenderer.sampleCount,
                  colorFormats: [MTLPixelFormat] = [GPUSimRenderer.colorFormat],
                  depth: MTLPixelFormat = .depth32Float,
                  blending: Bool = false) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            guard let vertex = lib.makeFunction(name: v) else {
                throw GPUSimRendererError.shaderFunction(v)
            }
            guard let fragment = lib.makeFunction(name: f) else {
                throw GPUSimRendererError.shaderFunction(f)
            }
            d.vertexFunction = vertex
            d.fragmentFunction = fragment
            for (i, fmt) in colorFormats.enumerated() {
                d.colorAttachments[i].pixelFormat = fmt
            }
            if blending, let attachment = d.colorAttachments[0] {
                attachment.isBlendingEnabled = true
                attachment.rgbBlendOperation = .add
                attachment.alphaBlendOperation = .add
                attachment.sourceRGBBlendFactor = .sourceAlpha
                attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
                attachment.sourceAlphaBlendFactor = .one
                attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }
            d.depthAttachmentPixelFormat = depth
            d.rasterSampleCount = samples
            return try device.makeRenderPipelineState(descriptor: d)
        }
        func depthPipe(_ vertex: String) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.label = "Directional shadow: \(vertex)"
            guard let function = lib.makeFunction(name: vertex) else {
                throw GPUSimRendererError.shaderFunction(vertex)
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
        boxAuxP = try pipe("box_vertex", "pbr_fragment", blending: true)
        sphereAuxP = try pipe("sphere_vertex", "pbr_fragment", blending: true)
        torusAuxP = try pipe("torus_vertex", "pbr_fragment", blending: true)
        capsuleAuxP = try pipe("capsule_vertex", "pbr_fragment", blending: true)
        softP = try pipe("soft_vertex", "soft_fragment")
        skinP = try pipe("skin_vertex", "soft_fragment")
        rigidMeshP = try pipe("rigid_mesh_vertex", "pbr_fragment")
        convexDebugFillP = try pipe(
            "rigid_mesh_vertex", "convex_debug_fill_fragment", blending: true)
        convexDebugWireP = try pipe(
            "rigid_mesh_vertex", "convex_debug_wire_fragment", blending: true)
        skyP = try pipe("fs_vertex", "sky_fragment")
        floorP = try pipe("floor_vertex", "floor_fragment")

        boxShadow = try depthPipe("box_vertex")
        sphereShadow = try depthPipe("sphere_vertex")
        torusShadow = try depthPipe("torus_vertex")
        capsuleShadow = try depthPipe("capsule_vertex")
        softShadow = try depthPipe("soft_vertex_front")
        skinShadow = try depthPipe("skin_vertex")
        rigidMeshShadow = try depthPipe("rigid_mesh_vertex")

        let preFmt: [MTLPixelFormat] = [.rgba16Float]
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
                             colorFormats: [.rgba8Unorm], depth: .invalid)

        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: dd) else {
            throw GPUSimRendererError.depthState("geometry")
        }
        self.depthState = depthState

        let nd = MTLDepthStencilDescriptor()
        nd.depthCompareFunction = .always
        nd.isDepthWriteEnabled = false
        guard let noDepthState = device.makeDepthStencilState(descriptor: nd) else {
            throw GPUSimRendererError.depthState("background")
        }
        self.noDepthState = noDepthState

        let debugDepth = MTLDepthStencilDescriptor()
        debugDepth.depthCompareFunction = .lessEqual
        debugDepth.isDepthWriteEnabled = false
        guard let debugDepthState = device.makeDepthStencilState(
                descriptor: debugDepth) else {
            throw GPUSimRendererError.depthState("convex-debug")
        }
        self.debugDepthState = debugDepthState

        let auxiliaryOpaqueDepth = MTLDepthStencilDescriptor()
        auxiliaryOpaqueDepth.depthCompareFunction = .lessEqual
        auxiliaryOpaqueDepth.isDepthWriteEnabled = true
        guard let auxiliaryOpaqueDepthState = device.makeDepthStencilState(
                descriptor: auxiliaryOpaqueDepth) else {
            throw GPUSimRendererError.depthState("opaque auxiliary geometry")
        }
        self.auxiliaryOpaqueDepthState = auxiliaryOpaqueDepthState

        let auxiliaryTranslucentDepth = MTLDepthStencilDescriptor()
        auxiliaryTranslucentDepth.depthCompareFunction = .lessEqual
        auxiliaryTranslucentDepth.isDepthWriteEnabled = false
        guard let auxiliaryTranslucentDepthState = device.makeDepthStencilState(
                descriptor: auxiliaryTranslucentDepth) else {
            throw GPUSimRendererError.depthState("translucent auxiliary geometry")
        }
        self.auxiliaryTranslucentDepthState = auxiliaryTranslucentDepthState

    }

    /// Creates a renderer whose current scene and options are supplied by a
    /// live application model.
    public convenience init(
        device: MTLDevice,
        source: any GPUSimRendererSource
    ) throws {
        try self.init(device: device, scene: nil)
        self.source = source
    }

    /// Convenience for the package's current Metal simulation backend.
    public convenience init(device: MTLDevice, solver: GPUSolver) throws {
        try self.init(device: device, scene: solver)
    }

    /// Applies the formats required by the renderer and installs it as the
    /// view delegate. The caller must retain this renderer because `delegate`
    /// is weak.
    public func configure(_ view: MTKView, preferredFramesPerSecond: Int? = nil) {
        view.device = device
        view.colorPixelFormat = Self.colorFormat
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = Self.sampleCount
        if let preferredFramesPerSecond {
            view.preferredFramesPerSecond = preferredFramesPerSecond
        }
        viewportSize = SIMD2(
            Float(max(view.drawableSize.width, 1)),
            Float(max(view.drawableSize.height, 1))
        )
        view.delegate = self
    }

    /// Replaces the fixed scene. Set `resetCamera` when this is a different
    /// scene; temporal history is always invalidated across scene identity.
    public func setScene(
        _ scene: (any GPUSimRenderableScene)?,
        resetCamera: Bool = true
    ) throws {
        if let scene, scene.renderDevice.registryID != device.registryID {
            throw GPUSimRendererError.sceneDeviceMismatch
        }
        self.scene = scene
        if resetCamera { sceneRevision &+= 1 }
        prevVP = nil
        lastSceneIdentity = nil
    }

    /// Discards temporal AO history, for example after a camera teleport.
    public func resetTemporalHistory() {
        prevVP = nil
    }

    @discardableResult
    private func ensureTargets(_ size: CGSize) -> Bool {
        let w = max(Int(size.width), 4), h = max(Int(size.height), 4)
        if targetSize == SIMD2(w, h), normTex != nil,
           aoTexA != nil, aoTexB != nil, histTex != nil, prevDepthTex != nil,
           preDepthTex != nil, shadowTex != nil { return true }
        // The AO chain runs at half resolution: GTAO is a low-frequency
        // signal already smoothed by a bilateral blur and temporal
        // accumulation, and full-resolution marching over a 16-byte
        // position texture was measured at ~23 ms of a 24 ms frame.
        let aw = max(w / 2, 4), ah = max(h / 2, 4)
        func tex(_ fmt: MTLPixelFormat) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt,
                                                             width: aw,
                                                             height: ah,
                                                             mipmapped: false)
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }
        aoTexIsWhite = false
        aoSize = SIMD2(aw, ah)
        let newNorm = tex(.rgba16Float)
        let newAOA = tex(.r8Unorm)
        // Resolved AO carries an octahedrally-packed normal in GB so history
        // validation needs no extra texture or pass.  The final blurred AO
        // remains the compact single-channel target.
        let newAOB = tex(.rgba8Unorm)
        let newHistory = tex(.rgba8Unorm)
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float,
                                                          width: aw, height: ah,
                                                          mipmapped: false)
        dd.usage = [.renderTarget, .shaderRead]
        dd.storageMode = .private
        let newPreDepth = device.makeTexture(descriptor: dd)
        let pd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: aw, height: ah,
            mipmapped: false)
        pd.usage = .shaderRead
        pd.storageMode = .private
        let newPrevDepth = device.makeTexture(descriptor: pd)

        let sd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: GPUSimRenderer.shadowMapSize,
            height: GPUSimRenderer.shadowMapSize,
            mipmapped: false)
        sd.usage = [.renderTarget, .shaderRead]
        sd.storageMode = .private
        let newShadow = device.makeTexture(descriptor: sd)
        guard let newNorm, let newAOA, let newAOB,
              let newHistory, let newPrevDepth, let newPreDepth,
              let newShadow else { return false }
        normTex = newNorm
        aoTexA = newAOA
        aoTexB = newAOB
        histTex = newHistory
        prevDepthTex = newPrevDepth
        preDepthTex = newPreDepth
        shadowTex = newShadow
        shadowTex?.label = "Directional shadow map"
        targetSize = SIMD2(w, h)
        prevVP = nil                      // resize invalidates history
        return true
    }

    private func prepareAppearanceBuffer(
        bodyCount: Int,
        appearances: [Int: GPUSimRenderAppearance]
    ) -> MTLBuffer? {
        guard bodyCount > 0,
              appearances.keys.contains(where: { $0 >= 0 && $0 < bodyCount })
        else { return nil }
        // same in-flight ring as the auxiliary staging: a non-retired frame
        // may still be reading the previous slot
        let slot = frameSlot
        if appearanceRing[slot] == nil || appearanceCapacity < bodyCount {
            appearanceCapacity = max(bodyCount, max(16, appearanceCapacity))
            for k in appearanceRing.indices where appearanceRing[k] == nil
                || appearanceRing[k]!.length < appearanceCapacity
                    * MemoryLayout<GPUSimRenderAppearance>.stride {
                let buffer = device.makeBuffer(
                    length: appearanceCapacity
                        * MemoryLayout<GPUSimRenderAppearance>.stride,
                    options: .storageModeShared)
                buffer?.label = "GPU Sim body appearance overrides"
                if let buffer {
                    memset(buffer.contents(), 0, buffer.length)
                }
                appearanceRing[k] = buffer
                appearanceActiveBodies[k].removeAll(keepingCapacity: true)
            }
        }
        guard let buffer = appearanceRing[slot] else { return nil }
        let values = buffer.contents().bindMemory(
            to: GPUSimRenderAppearance.self, capacity: appearanceCapacity)
        for body in appearanceActiveBodies[slot] { values[body] = .init() }
        appearanceActiveBodies[slot].removeAll(keepingCapacity: true)
        for (body, appearance) in appearances
        where body >= 0 && body < bodyCount {
            values[body] = appearance
            appearanceActiveBodies[slot].append(body)
        }
        return buffer
    }

    static func auxiliaryRenderOrder(
        _ values: [GPUSimRenderInstance],
        viewedFrom eye: F3
    ) -> (opaque: [GPUSimRenderInstance], casterCount: Int,
          translucent: [GPUSimRenderInstance]) {
        var opaque: [GPUSimRenderInstance] = []
        var translucent: [(index: Int, distanceSquared: Float,
                           instance: GPUSimRenderInstance)] = []
        opaque.reserveCapacity(values.count)
        translucent.reserveCapacity(values.count)
        for (index, instance) in values.enumerated() {
            let opacity = instance.material.w
            guard opacity.isFinite && opacity > 0 else { continue }
            if opacity >= 1 {
                opaque.append(instance)
                continue
            }
            let translation = instance.model.columns.3
            let delta = F3(translation.x, translation.y, translation.z) - eye
            let distanceSquared = length_squared(delta)
            translucent.append((
                index: index,
                distanceSquared: distanceSquared.isNaN ? -.infinity
                    : distanceSquared,
                instance: instance))
        }
        translucent.sort {
            if $0.distanceSquared == $1.distanceSquared {
                return $0.index < $1.index
            }
            return $0.distanceSquared > $1.distanceSquared
        }
        // shadow casters lead the opaque run so the shadow pass draws a
        // stable prefix; relative order is preserved on both sides
        let casters = opaque.filter { $0.parameters.w > 0.5 }
        let nonCasters = opaque.filter { $0.parameters.w <= 0.5 }
        return (casters + nonCasters, casters.count,
                translucent.map(\.instance))
    }

    private func prepareAuxiliaryInstanceBuffer(
        _ values: [GPUSimRenderInstance],
        viewedFrom eye: F3
    ) -> (buffer: MTLBuffer, opaqueCount: Int, casterCount: Int,
          translucent: [GPUSimRenderInstance])? {
        let order = Self.auxiliaryRenderOrder(values, viewedFrom: eye)
        let ordered = order.opaque + order.translucent
        guard !ordered.isEmpty else { return nil }
        // A ring, not a single buffer: scenes that opt out of per-frame
        // retirement can have frames in flight, and a CPU memcpy into a
        // buffer the previous frame still reads is a race.
        if auxiliaryInstanceRing[frameSlot] == nil
            || auxiliaryInstanceRing[frameSlot]!.length
                < ordered.count * MemoryLayout<GPUSimRenderInstance>.stride {
            let capacity = max(ordered.count * 2, 16)
            let buffer = device.makeBuffer(
                length: capacity * MemoryLayout<GPUSimRenderInstance>.stride,
                options: .storageModeShared)
            buffer?.label = "GPU Sim auxiliary instances"
            auxiliaryInstanceRing[frameSlot] = buffer
        }
        guard let buffer = auxiliaryInstanceRing[frameSlot]
        else { return nil }
        _ = ordered.withUnsafeBytes { bytes in
            memcpy(buffer.contents(), bytes.baseAddress!, bytes.count)
        }
        return (buffer, order.opaque.count, order.casterCount,
                order.translucent)
    }

    // MARK: - Camera

    public var viewMatrix: simd_float4x4 {
        if let cameraPose {
            return lookAt(
                eye: cameraPose.position,
                center: cameraPose.target,
                up: cameraPose.up)
        }
        return lookAt(eye: eyePosition, center: target, up: F3(0, 0, 1))
    }

    public var eyePosition: F3 {
        if let cameraPose { return cameraPose.position }
        return target + F3(distance * cos(elevation) * cos(azimuth),
                           distance * cos(elevation) * sin(azimuth),
                           distance * sin(elevation))
    }

    private var cameraFocus: F3 { cameraPose?.target ?? target }
    private var cameraUp: F3 { cameraPose?.up ?? F3(0, 0, 1) }

    /// Activates an application-owned camera and disables automatic scene
    /// framing. Continuous camera motion can preserve temporal history;
    /// request a reset for cuts or teleports.
    public func setCamera(
        _ pose: GPUSimCameraPose,
        resetTemporalHistory: Bool = false
    ) {
        cameraPose = pose
        automaticallyFramesScene = false
        if resetTemporalHistory { self.resetTemporalHistory() }
    }

    public func setCamera(
        position: F3,
        target: F3,
        up: F3 = F3(0, 0, 1),
        resetTemporalHistory: Bool = false
    ) {
        setCamera(
            GPUSimCameraPose(position: position, target: target, up: up),
            resetTemporalHistory: resetTemporalHistory)
    }

    /// Activates an app-owned camera from a world-to-view matrix. The focus
    /// distance determines the target used for camera-relative effects and
    /// the directional-shadow region.
    public func setCamera(
        viewMatrix: simd_float4x4,
        focusDistance: Float = 1,
        resetTemporalHistory: Bool = false
    ) {
        precondition(matrixIsFinite(viewMatrix)
            && abs(simd_determinant(viewMatrix)) > 1.0e-12,
            "camera view matrix must be finite and invertible")
        precondition(focusDistance.isFinite && focusDistance > 0,
                     "camera focus distance must be finite and positive")
        let world = viewMatrix.inverse
        let homogeneousEye = world.columns.3
        precondition(abs(homogeneousEye.w) > 1.0e-12,
                     "camera view matrix has no finite world-space origin")
        let eye = F3(homogeneousEye.x, homogeneousEye.y, homogeneousEye.z)
            / homogeneousEye.w
        let forward = normalize(-F3(
            world.columns.2.x, world.columns.2.y, world.columns.2.z))
        let up = normalize(F3(
            world.columns.1.x, world.columns.1.y, world.columns.1.z))
        setCamera(
            position: eye,
            target: eye + forward * focusDistance,
            up: up,
            resetTemporalHistory: resetTemporalHistory)
    }

    /// Returns control to the public orbit properties. This does not change
    /// `automaticallyFramesScene`; callers choose whether later scene
    /// revisions should replace the orbit values.
    public func useOrbitCamera(resetTemporalHistory: Bool = true) {
        cameraPose = nil
        if resetTemporalHistory { self.resetTemporalHistory() }
    }

    public func projectionMatrix(aspect: Float) -> simd_float4x4 {
        perspective(fovY: 50 * .pi / 180, aspect: aspect, near: 0.1, far: 1000)
    }

    /// Builds a world ray from a pixel point whose origin is the view's
    /// upper-left corner.
    public func ray(at point: CGPoint, in size: CGSize) -> (origin: F3, dir: F3) {
        let width = max(size.width, 1)
        let height = max(size.height, 1)
        let aspect = Float(width / height)
        let ndcX = Float(point.x / width) * 2 - 1
        let ndcY = 1 - Float(point.y / height) * 2
        let invVP = (projectionMatrix(aspect: aspect) * viewMatrix).inverse
        var pNear = invVP * SIMD4<Float>(ndcX, ndcY, 0, 1)
        var pFar = invVP * SIMD4<Float>(ndcX, ndcY, 1, 1)
        pNear /= pNear.w
        pFar /= pFar.w
        let o = F3(pNear.x, pNear.y, pNear.z)
        let d = normalize(F3(pFar.x, pFar.y, pFar.z) - o)
        return (o, d)
    }

    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        viewportSize = SIMD2(Float(size.width), Float(size.height))
    }

    // MARK: - Frame

    public func draw(in view: MTKView) {
        guard runtimeFailure == nil else { return }
        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable else { return }

        source?.rendererWillDrawFrame()
        guard let renderScene = source?.renderScene ?? scene else { return }

        guard renderScene.renderDevice.registryID == device.registryID else {
            reportFailure(
                GPUSimRendererError.sceneDeviceMismatch.localizedDescription
            )
            return
        }

        let activeOptions = source?.rendererOptions ?? options
        let activeBodyAppearances = source?.rendererBodyAppearances
            ?? bodyAppearances
        let activeAuxiliaryInstances = source?.rendererAuxiliaryInstances
            ?? auxiliaryInstances
        let activeSceneRevision = source?.rendererSceneRevision ?? sceneRevision

        // cap frames in flight to the staging ring depth; every exit that
        // does not commit must release the slot
        inFlightFrames.wait()
        frameSlot = (frameSlot + 1) % auxiliaryInstanceRing.count
        var releasedByCompletion = false
        defer { if !releasedByCompletion { inFlightFrames.signal() } }

        guard let cmd = queue.makeCommandBuffer() else {
            reportFailure("could not create a Metal command buffer")
            return
        }
        cmd.label = "GPU Sim render frame"

        // The source owns physics failure reporting. Never relabel a poisoned
        // scene as a renderer failure or read its potentially invalid state.
        guard renderScene.rendererStateIsValid else { return }
        let sceneIdentity = ObjectIdentifier(renderScene)
        if sceneIdentity != lastSceneIdentity {
            lastSceneIdentity = sceneIdentity
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
        if lastCameraEpoch != activeSceneRevision {
            lastCameraEpoch = activeSceneRevision
            if automaticallyFramesScene {
                let hint = renderScene.renderCameraHint
                cameraPose = nil
                distance = hint.distance
                target = hint.target
                azimuth = hint.azimuth
                elevation = hint.elevation
            }
        }

        let rigidCount = renderScene.renderRigidInstanceCount
        let stride = MemoryLayout<GPUSimRenderInstance>.stride
        if instances == nil || instances!.length < max(1, rigidCount) * stride {
            instances = device.makeBuffer(length: max(256, max(1, rigidCount) * stride),
                                          options: .storageModePrivate)
        }
        guard let instances, let normTex, let aoTexA, let aoTexB,
              let histTex, let prevDepthTex, let preDepthTex,
              let shadowTex else {
            reportFailure("could not allocate required Metal render resources")
            return
        }

        let hasValidAppearance = activeBodyAppearances.keys.contains {
            $0 >= 0 && $0 < renderScene.renderBodyCount
        }
        let appearanceOverrides = prepareAppearanceBuffer(
            bodyCount: renderScene.renderBodyCount,
            appearances: activeBodyAppearances)
        if hasValidAppearance && appearanceOverrides == nil {
            reportFailure("could not allocate body appearance overrides")
            return
        }

        let activeEye = eyePosition
        let hasVisibleAuxiliary = activeAuxiliaryInstances.contains {
            $0.material.w.isFinite && $0.material.w > 0
        }
        let auxiliaryBatch = prepareAuxiliaryInstanceBuffer(
            activeAuxiliaryInstances, viewedFrom: activeEye)
        if hasVisibleAuxiliary && auxiliaryBatch == nil {
            reportFailure("could not allocate auxiliary render instances")
            return
        }

        do {
            try renderScene.encodeRenderInstances(
                cmd, instances: instances,
                colorMode: activeOptions.colorMode,
                appearanceOverrides: appearanceOverrides)
        } catch {
            // synchronize() inside the checked API may be the first observer
            // of a physics error. Its owning model will report it on the next
            // tick; do not poison or misclassify it here.
            guard renderScene.rendererStateIsValid else { return }
            reportFailure("instance-build encoder failed: \(error.localizedDescription)")
            return
        }

        let aspect = viewportSize.x / max(viewportSize.y, 1)
        let pxPerUnit = viewportSize.y * 0.5 / tan(25 * Float.pi / 180)
        let activeFocus = cameraFocus
        let fwd = normalize(activeFocus - activeEye)
        let camR = normalize(cross(fwd, cameraUp))
        let camU = cross(camR, fwd)          // screen +y in UV space is DOWN
        let projection = projectionMatrix(aspect: aspect)
        let vp = projection * viewMatrix
        let lightDirection = normalize(F3(0.4, 0.25, -0.85))

        // A camera-targeted, texel-stabilized orthographic light projection.
        // This keeps articulated robot detail crisp while preventing shadows
        // from crawling when the follow camera advances with a policy.
        // Fit the light volume to the CONTENT when the scene declares its
        // bounds (a tabletop rig gets millimetre texels and every object
        // casts, wherever the camera looks); fall back to camera-focus
        // fitting for scenes without bounds or too large to cover crisply.
        // The old fixed 7 m floor gave tabletop scenes 7 mm texels and
        // biases larger than a cube's contact shadow.
        var contentBounds = renderScene.renderContentBounds
        // opaque auxiliary casters live in app space; a skeleton or marker
        // outside the scene's own bounds must not fall off the light
        // volume - and a backend without bounds of its own still gets a
        // fitted volume from its casters alone
        for instance in activeAuxiliaryInstances
        where instance.material.w >= 1 && instance.parameters.w > 0.5 {
            let t = instance.model.columns.3
            let p = F3(t.x, t.y, t.z)
            if var bounds = contentBounds {
                let reach = length(p - bounds.center) + instance.parameters.z
                bounds.radius = max(bounds.radius, reach)
                contentBounds = bounds
            } else {
                contentBounds = GPUSimContentBounds(
                    center: p, radius: max(instance.parameters.z, 0.5))
            }
        }
        let shadowFollowsContent = contentBounds.map { $0.radius <= 25 } ?? false
        let shadowFocus = shadowFollowsContent
            ? contentBounds!.center : activeFocus
        let shadowExtent = shadowFollowsContent
            ? max(1.0, contentBounds!.radius * 1.15)
            : max(1.5, min(length(activeFocus - activeEye) * 0.9, 40.0))
        let lightUp = F3(0, 1, 0)
        let lightRight = normalize(cross(lightDirection, lightUp))
        let lightMapUp = cross(lightRight, lightDirection)
        let texelWorld = (2 * shadowExtent) / Float(GPUSimRenderer.shadowMapSize)
        let rightCoord = (dot(shadowFocus, lightRight) / texelWorld).rounded() * texelWorld
        let upCoord = (dot(shadowFocus, lightMapUp) / texelWorld).rounded() * texelWorld
        let shadowCenter = shadowFocus
            + lightRight * (rightCoord - dot(shadowFocus, lightRight))
            + lightMapUp * (upCoord - dot(shadowFocus, lightMapUp))
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
        let aoViewport = MTLViewport(originX: 0, originY: 0,
                                     width: Double(aoSize.x),
                                     height: Double(aoSize.y),
                                     znear: 0, zfar: 1)
        // AO re-enabled after a disabled stretch: the frozen history and
        // the stale previous-frame matrices must fall together, and BEFORE
        // the uniforms capture prevVP and the temporal blend for this frame
        if activeOptions.ambientOcclusion, aoTexIsWhite {
            prevVP = nil
            aoTexIsWhite = false
        }
        var U = Uniforms(viewProj: vp,
                         lightDir: SIMD4(lightDirection, 0),
                         eye: SIMD4(activeEye, 0),
                         screen: SIMD4(viewportSize.x, viewportSize.y, pxPerUnit, 0),
                         camRight: SIMD4(camR, 0),
                         camUp: SIMD4(-camU, 0),
                         prevViewProj: prevVP ?? vp,
                         temporal: SIMD4(noiseOff.truncatingRemainder(dividingBy: 1),
                                         prevVP == nil ? 1.0 : 0.20, 0, 0),
                         shadowViewProj: shadowVP,
                         shadowParams: SIMD4(
                             0.30 * texelWorld / (shadowExtent * 5),
                             1.15 * texelWorld / (shadowExtent * 5), 0, 0),
                         invViewProj: vp.inverse,
                         prevInvViewProj: (prevVP ?? vp).inverse,
                         aoProjection: SIMD4(-projection.columns.2.z, projection.columns.3.z,
                                             1 / projection.columns.0.x, 1 / projection.columns.1.y))
        func bindAppearance(
            _ encoder: MTLRenderCommandEncoder,
            enabled: Bool = true
        ) {
            var hasAppearance: UInt32 = enabled && appearanceOverrides != nil
                ? 1 : 0
            encoder.setVertexBuffer(
                enabled ? (appearanceOverrides ?? instances) : instances,
                offset: 0, index: 4)
            encoder.setVertexBytes(
                &hasAppearance, length: 4, index: 5)
        }
        var Uh = U
        Uh.screen = SIMD4(Float(aoSize.x), Float(aoSize.y),
                          U.screen.z * Float(aoSize.y) / U.screen.y, U.screen.w)

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
                                        width: Double(GPUSimRenderer.shadowMapSize),
                                        height: Double(GPUSimRenderer.shadowMapSize),
                                        znear: 0, zfar: 1))
            enc.setDepthStencilState(depthState)
            // Offset caster depth in the shadow rasterizer as the primary
            // self-shadow defense. Receiver bias alone cannot cover the
            // sub-texel depth quantization of a sloped triangle consistently,
            // which exposes the PCF tap count as broad tonal stripes.
            enc.setDepthBias(1.0, slopeScale: 1.0, clamp: 0.001)
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
            if let auxiliaryBatch, auxiliaryBatch.casterCount > 0 {
                for (p, verts) in [(boxShadow!, 36), (sphereShadow!, SPHV),
                                   (torusShadow!, TORV), (capsuleShadow!, CAPV)] {
                    enc.setRenderPipelineState(p)
                    enc.setVertexBuffer(auxiliaryBatch.buffer, offset: 0,
                                        index: 0)
                    enc.setVertexBytes(&shadowU,
                                       length: MemoryLayout<Uniforms>.stride,
                                       index: 1)
                    enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                       vertexCount: verts,
                                       instanceCount: auxiliaryBatch.casterCount)
                }
            }
            if let surf = renderScene.softRenderSurface {
                enc.setRenderPipelineState(softShadow)
                enc.setVertexBuffer(surf.triangles, offset: 0, index: 0)
                enc.setVertexBytes(&shadowU, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(surf.positions, offset: 0, index: 2)
                enc.setVertexBuffer(surf.normals, offset: 0, index: 3)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: surf.triangleCount * 3)
            }
            if let skin = renderScene.skinnedRenderSurface {
                enc.setRenderPipelineState(skinShadow)
                enc.setVertexBuffer(skin.triangles, offset: 0, index: 0)
                enc.setVertexBytes(&shadowU, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(skin.vertices, offset: 0, index: 2)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: skin.triangleCount * 3)
            }
            if let mesh = renderScene.rigidMeshRenderSurface {
                enc.setRenderPipelineState(rigidMeshShadow)
                enc.setVertexBuffer(mesh.vertices, offset: 0, index: 0)
                enc.setVertexBytes(&shadowU, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(mesh.positions, offset: 0, index: 2)
                enc.setVertexBuffer(mesh.rotations, offset: 0, index: 3)
                bindAppearance(enc)
                enc.drawIndexedPrimitives(
                    type: .triangle, indexCount: mesh.indexCount,
                    indexType: .uint32, indexBuffer: mesh.indices,
                    indexBufferOffset: 0)
            }
            enc.endEncoding()
        }

        // ---- 2. prepass: world pos + normals ----
        if !activeOptions.ambientOcclusion {
            if !aoTexIsWhite {
                let clearPass = MTLRenderPassDescriptor()
                clearPass.colorAttachments[0].texture = aoTexA
                clearPass.colorAttachments[0].loadAction = .clear
                clearPass.colorAttachments[0].clearColor = MTLClearColor(
                    red: 1, green: 1, blue: 1, alpha: 1)
                clearPass.colorAttachments[0].storeAction = .store
                cmd.makeRenderCommandEncoder(descriptor: clearPass)?
                    .endEncoding()
                aoTexIsWhite = true
            }
        } else {
            let pre = MTLRenderPassDescriptor()
            pre.colorAttachments[0].texture = normTex
            pre.colorAttachments[0].loadAction = .clear
            pre.colorAttachments[0].clearColor = MTLClearColor(
                red: 0, green: 0, blue: 0, alpha: 0)
            pre.colorAttachments[0].storeAction = .store
            pre.depthAttachment.texture = preDepthTex
            pre.depthAttachment.loadAction = .clear
            pre.depthAttachment.clearDepth = 1
            pre.depthAttachment.storeAction = .store
            guard let prepassEncoder = cmd.makeRenderCommandEncoder(
                    descriptor: pre) else {
                reportFailure("could not create the world-position prepass encoder")
                return
            }
            do {
                let enc = prepassEncoder
                enc.label = "World-position and normal prepass"
                enc.setViewport(aoViewport)
                enc.setDepthStencilState(depthState)
                enc.setVertexBytes(&Uh, length: MemoryLayout<Uniforms>.stride, index: 1)
                if activeOptions.showsGroundPlane {
                    enc.setRenderPipelineState(floorPreP)
                    enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
                }
                if rigidCount > 0 {
                    for (p, verts) in [(boxPre!, 36), (spherePre!, SPHV),
                                       (torusPre!, TORV), (capsulePre!, CAPV)] {
                        enc.setRenderPipelineState(p)
                        enc.setVertexBuffer(instances, offset: 0, index: 0)
                        enc.setVertexBytes(&Uh, length: MemoryLayout<Uniforms>.stride, index: 1)
                        enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                           vertexCount: verts, instanceCount: rigidCount)
                    }
                }
                if let surf = renderScene.softRenderSurface {
                    enc.setRenderPipelineState(softPre)
                    enc.setVertexBuffer(surf.triangles, offset: 0, index: 0)
                    enc.setVertexBytes(&Uh, length: MemoryLayout<Uniforms>.stride, index: 1)
                    enc.setVertexBuffer(surf.positions, offset: 0, index: 2)
                    enc.setVertexBuffer(surf.normals, offset: 0, index: 3)
                    enc.setFragmentBytes(&Uh, length: MemoryLayout<Uniforms>.stride, index: 1)
                    enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                       vertexCount: surf.triangleCount * 3)
                }
                if let skin = renderScene.skinnedRenderSurface {
                    enc.setRenderPipelineState(skinPre)
                    enc.setVertexBuffer(skin.triangles, offset: 0, index: 0)
                    enc.setVertexBytes(&Uh, length: MemoryLayout<Uniforms>.stride, index: 1)
                    enc.setVertexBuffer(skin.vertices, offset: 0, index: 2)
                    enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                       vertexCount: skin.triangleCount * 3)
                }
                if let mesh = renderScene.rigidMeshRenderSurface {
                    enc.setRenderPipelineState(rigidMeshPre)
                    enc.setVertexBuffer(mesh.vertices, offset: 0, index: 0)
                    enc.setVertexBytes(&Uh, length: MemoryLayout<Uniforms>.stride, index: 1)
                    enc.setVertexBuffer(mesh.positions, offset: 0, index: 2)
                    enc.setVertexBuffer(mesh.rotations, offset: 0, index: 3)
                    bindAppearance(enc)
                    enc.drawIndexedPrimitives(
                        type: .triangle, indexCount: mesh.indexCount,
                        indexType: .uint32, indexBuffer: mesh.indices,
                        indexBufferOffset: 0)
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
                enc.setViewport(aoViewport)
                enc.setRenderPipelineState(gtaoP)
                enc.setFragmentBytes(&Uh, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setFragmentTexture(preDepthTex, index: 0)
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
                enc.setViewport(aoViewport)
                enc.setRenderPipelineState(temporalP)
                enc.setFragmentBytes(&Uh, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setFragmentTexture(aoTexA, index: 0)
                enc.setFragmentTexture(histTex, index: 1)
                enc.setFragmentTexture(preDepthTex, index: 2)
                enc.setFragmentTexture(prevDepthTex, index: 3)
                enc.setFragmentTexture(normTex, index: 4)
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
            historyEncoder.copy(from: preDepthTex, to: prevDepthTex)
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
                enc.setViewport(aoViewport)
                enc.setRenderPipelineState(blurP)
                enc.setFragmentBytes(&Uh, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setFragmentTexture(aoTexB, index: 0)
                enc.setFragmentTexture(preDepthTex, index: 1)
                enc.setFragmentTexture(normTex, index: 2)
                enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                enc.endEncoding()
            }
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
        if activeOptions.showsGroundPlane { enc.setRenderPipelineState(floorP) }
        enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentTexture(aoTexA, index: 0)
        enc.setFragmentTexture(shadowTex, index: 1)
        if activeOptions.showsGroundPlane {
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        }

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
        if let surf = renderScene.softRenderSurface {
            enc.setRenderPipelineState(softP)
            enc.setVertexBuffer(surf.triangles, offset: 0, index: 0)
            enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setVertexBuffer(surf.positions, offset: 0, index: 2)
            enc.setVertexBuffer(surf.normals, offset: 0, index: 3)
            bindAppearance(enc)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(aoTexA, index: 0)
            enc.setFragmentTexture(shadowTex, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: surf.triangleCount * 3)
        }
        if let skin = renderScene.skinnedRenderSurface {
            enc.setRenderPipelineState(skinP)
            enc.setVertexBuffer(skin.triangles, offset: 0, index: 0)
            enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setVertexBuffer(skin.vertices, offset: 0, index: 2)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(aoTexA, index: 0)
            enc.setFragmentTexture(shadowTex, index: 1)
            enc.drawPrimitives(type: .triangle, vertexStart: 0,
                               vertexCount: skin.triangleCount * 3)
        }
        if let mesh = renderScene.rigidMeshRenderSurface {
            enc.setRenderPipelineState(rigidMeshP)
            enc.setVertexBuffer(mesh.vertices, offset: 0, index: 0)
            enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setVertexBuffer(mesh.positions, offset: 0, index: 2)
            enc.setVertexBuffer(mesh.rotations, offset: 0, index: 3)
            bindAppearance(enc)
            enc.setFragmentBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
            enc.setFragmentTexture(aoTexA, index: 0)
            enc.setFragmentTexture(shadowTex, index: 1)
            enc.drawIndexedPrimitives(
                type: .triangle, indexCount: mesh.indexCount,
                indexType: .uint32, indexBuffer: mesh.indices,
                indexBufferOffset: 0)
        }
        if activeOptions.showConvexCollisionGeometry,
           let debug = renderScene.convexDebugRenderSurface {
            enc.setDepthStencilState(debugDepthState)
            // Pull the diagnostic surface very slightly toward the camera so
            // coincident visual and collision meshes do not z-fight. This is
            // render state only; collision geometry and poses are untouched.
            enc.setDepthBias(-0.0002, slopeScale: -0.35, clamp: -0.002)
            if debug.triangleVertexCount > 0 {
                enc.setRenderPipelineState(convexDebugFillP)
                enc.setVertexBuffer(debug.triangleVertices, offset: 0, index: 0)
                enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(debug.positions, offset: 0, index: 2)
                enc.setVertexBuffer(debug.rotations, offset: 0, index: 3)
                bindAppearance(enc, enabled: false)
                enc.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: debug.triangleVertexCount)
            }
            if activeOptions.convexCollisionWireframe, debug.edgeVertexCount > 0 {
                enc.setRenderPipelineState(convexDebugWireP)
                enc.setVertexBuffer(debug.edgeVertices, offset: 0, index: 0)
                enc.setVertexBytes(&U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setVertexBuffer(debug.positions, offset: 0, index: 2)
                enc.setVertexBuffer(debug.rotations, offset: 0, index: 3)
                bindAppearance(enc, enabled: false)
                enc.drawPrimitives(type: .line, vertexStart: 0,
                                   vertexCount: debug.edgeVertexCount)
            }
            enc.setDepthBias(0, slopeScale: 0, clamp: 0)
        }
        if let auxiliaryBatch {
            // Opaque auxiliary geometry participates in depth just like the
            // scene, so multi-part app geometry occludes itself correctly.
            // Guarded per SECTION, not for the whole batch: an
            // all-translucent batch (a lone landing ring, say) must skip
            // only the opaque draws - instanceCount 0 is a Metal validation
            // failure - while its translucent pieces still render.
            if auxiliaryBatch.opaqueCount > 0 {
                enc.setDepthStencilState(auxiliaryOpaqueDepthState)
                for (pipeline, vertices) in [
                    (boxP!, 36),
                    (sphereP!, SPHV),
                    (torusP!, TORV),
                    (capsuleP!, CAPV),
                ] {
                    enc.setRenderPipelineState(pipeline)
                    enc.setVertexBuffer(
                        auxiliaryBatch.buffer, offset: 0, index: 0)
                    enc.setVertexBytes(
                        &U, length: MemoryLayout<Uniforms>.stride, index: 1)
                    enc.setFragmentBytes(
                        &U, length: MemoryLayout<Uniforms>.stride, index: 1)
                    enc.setFragmentTexture(aoTexA, index: 0)
                    enc.setFragmentTexture(shadowTex, index: 1)
                    enc.drawPrimitives(
                        type: .triangle, vertexStart: 0,
                        vertexCount: vertices,
                        instanceCount: auxiliaryBatch.opaqueCount)
                }
            }

            // Blended geometry cannot write depth without incorrectly hiding
            // farther translucent layers. Submit it strictly back-to-front;
            // individual draws preserve that order across primitive types.
            enc.setDepthStencilState(auxiliaryTranslucentDepthState)
            let translucentOffset = auxiliaryBatch.opaqueCount
                * MemoryLayout<GPUSimRenderInstance>.stride
            for (index, instance) in auxiliaryBatch.translucent.enumerated() {
                let pipeline: MTLRenderPipelineState
                let vertices: Int
                switch instance.color.w {
                case 0: (pipeline, vertices) = (boxAuxP, 36)
                case 1: (pipeline, vertices) = (sphereAuxP, SPHV)
                case 2: (pipeline, vertices) = (torusAuxP, TORV)
                case 3: (pipeline, vertices) = (capsuleAuxP, CAPV)
                default: continue
                }
                enc.setRenderPipelineState(pipeline)
                enc.setVertexBuffer(
                    auxiliaryBatch.buffer,
                    offset: translucentOffset
                        + index * MemoryLayout<GPUSimRenderInstance>.stride,
                    index: 0)
                enc.setVertexBytes(
                    &U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setFragmentBytes(
                    &U, length: MemoryLayout<Uniforms>.stride, index: 1)
                enc.setFragmentTexture(aoTexA, index: 0)
                enc.setFragmentTexture(shadowTex, index: 1)
                enc.drawPrimitives(
                    type: .triangle, vertexStart: 0,
                    vertexCount: vertices, instanceCount: 1)
            }
        }
        enc.endEncoding()

        #if targetEnvironment(simulator)
        // the simulator SDK has no present(afterMinimumDuration:); pacing is
        // a device concern anyway
        cmd.present(drawable)
        #else
        if let hold = activeOptions.minimumFrameDuration {
            cmd.present(drawable, afterMinimumDuration: hold)
        } else {
            cmd.present(drawable)
        }
        #endif
        let retire = renderScene.renderSceneRequiresFrameRetirement
        framesDrawn += 1
        let frameNumber = framesDrawn
        let completedTexture = drawable.texture
        let inFlight = inFlightFrames
        cmd.addCompletedHandler { [weak self] finished in
            inFlight.signal()
            guard !retire else { return }
            // pipelined mode: the frame is only now finished, so timing,
            // failure inspection, AND the completion callback (documented as
            // running after the frame completes, safe for readback) all
            // happen here
            let ms = (finished.gpuEndTime - finished.gpuStartTime) * 1000
            let status = finished.status
            let error = finished.error as NSError?
            Task { @MainActor in
                guard let self else { return }
                self.lastFrameGPUMilliseconds = ms
                if status != .completed || error != nil {
                    let detail = error.map {
                        "\($0.domain) \($0.code) — \($0.localizedDescription)"
                    } ?? String(describing: status)
                    self.reportFailure(
                        "Metal command ended with status \(detail)")
                    return
                }
                self.frameCompletionHandler?(completedTexture, frameNumber)
            }
        }
        releasedByCompletion = true
        cmd.commit()

        // Physics and rendering own different command queues but share pose
        // buffers: GPUSolver-backed scenes retire every visual frame before
        // returning to the run loop that may mutate them. Scenes that own
        // their buffers (declared via renderSceneRequiresFrameRetirement)
        // keep CPU and GPU pipelined instead.
        if retire {
            cmd.waitUntilCompleted()
            // the frame is complete: timing is current, and the completion
            // callback runs with finished pixels, before this call returns
            lastFrameGPUMilliseconds =
                (cmd.gpuEndTime - cmd.gpuStartTime) * 1000
            if let failure = commandFailureDescription(cmd) {
                reportFailure(failure)
                return
            }
            frameCompletionHandler?(completedTexture, frameNumber)
        }

        prevVP = vp
        frameIdx &+= 1
    }
}

private func matrixIsFinite(_ matrix: simd_float4x4) -> Bool {
    for column in [
        matrix.columns.0, matrix.columns.1,
        matrix.columns.2, matrix.columns.3,
    ] where !column.x.isFinite || !column.y.isFinite
        || !column.z.isFinite || !column.w.isFinite {
        return false
    }
    return true
}

private func lookAt(eye: F3, center: F3, up: F3) -> simd_float4x4 {
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

private func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
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
