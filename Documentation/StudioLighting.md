# Configurable environment and finite-area lighting

The renderer keeps its existing Fast and HQ defaults: no external environment,
no area lights, opaque materials, one glossy ray and the existing adaptive HQ
diffuse budget. A caller can opt into richer lighting independently of physics,
asset import, or procedural material authoring. No images are downloaded and no
studio preset is embedded in the renderer.

See [measured costs and validation](StudioLightingBenchmarks.md) for default-mode
comparisons and a consumer batch-rendering example.

## HDR environment

```swift
let image = try GPUSimMaterialLibrary.loadTexture(
    device: device, url: environmentURL, sRGB: false)
let environment = try GPUSimEnvironmentLight(
    device: device, texture: image,
    intensity: 1, rotation: .pi / 4, showsBackground: true)
let materials = try GPUSimMaterialLibrary(
    device: device, materials: authoredMaterials, environment: environment)
let renderer = try GPUSimRenderer(device: device, scene: scene, materials: materials)
```

Use a linear floating-point texture for HDR radiance. Encoded SDR images should
be loaded with `sRGB: true`; Metal decodes the sampler once. Material factors,
procedural shader outputs, environment radiance and area-light radiance are
linear. The existing vertex/body color ABI is sRGB; encode a linear palette
before passing it through that ABI. The sRGB output attachment handles the final
display encoding. Do not add another gamma conversion to saved BGRA images.

`renderer.options.displayExposure` adjusts exposure in stops before the existing
ACES-style tone curve, independently of lighting and reconstruction history.
It defaults to zero; finite inputs clamp to -16...16 and nonfinite inputs resolve
to zero. Callers can also supply an optional [display program and 3D lookup
table](DisplayTransforms.md), including an OCIO-exported AgX processor, without
changing the ray budget. Exposure alone does not reproduce AgX.

The equirectangular map uses longitude `atan2(y,x)` and latitude `acos(z)` with
north at +Z and a top-left UV origin. Positive `rotation` rotates the map around
world +Z. `showsBackground` changes camera visibility independently of lighting.
Intensity scales both diffuse and specular environment radiance.

Nine cosine-convolved spherical-harmonic coefficients are prepared once from
4,096 spherical samples. `diffuseSamples` on the environment constructor changes
only that preparation cost. The result approximates irradiance divided by pi;
floating-point HDR values above one are retained. Diffuse lighting uses those
coefficients for rigid, soft and floor surfaces. Raster specular fallback uses
the caller's mip chain; HQ uses GGX ray sampling and actual scene intersections.
Environment textures participate in the material library's existing deduplicated
texture budget. Argument-buffer Tier 2 is required.

## Finite emitters

```swift
renderer.options.areaLights = [try GPUSimAreaLight(
    position: F3(3, -4, 6), normal: F3(-3, 4, -5.3),
    size: SIMD2(4, 4), radiance: F3(18, 16, 14), shape: .disk)]
renderer.options.sunIntensity = 0
renderer.options.lightingMode = .qualityBeta
renderer.options.rayTracingQuality = .balanced
renderer.options.rayTracingQuality.areaLightSamples = 8
```

A light has a position, facing normal, in-plane up direction, full width/height,
linear RGB radiance and either a rectangle or disk shape. Unequal disk axes
produce an ellipse. Radiance is per unit projected emitting area, per steradian;
it does not change when the emitter grows. For total radiant power `P`, a
one-sided diffuse emitter of area `A` has radiance `P / (pi * A)`. Scale each RGB
channel by the light color. `twoSided` enables emission on both sides at the same
radiance. Invalid and degenerate frames are rejected. Up to eight emitters are
used; the options resolver takes the first eight entries.

HQ evaluates finite distance, emitter orientation, GGX response, and bounded
shadow rays toward samples on each emitter. Emitters are visible in the sky
background and to reflection rays, but contribute no collision geometry or mass.
Camera-visible emitters use the background pass: opaque raster geometry covers
them even when it lies behind the light. Direct-emitter
radiance is excluded from the separate opaque bounce estimators to avoid counting
it twice. Secondary surfaces receive area lighting, giving a bounded diffuse
bounce and reflected illuminated geometry. The primary area pass recovers the
actual scene intersection before launching visibility rays, avoiding depth-buffer
quantization acne with distant cameras. Fast mode uses deterministic quadrature
for area illumination without area-light occlusion. Existing directional shadows
remain available in Fast.

## Quality and diagnostics

| Setting | Real-time default | Balanced | High | Bound |
|---|---:|---:|---:|---:|
| Directional shadow samples | 1 | 4 | 16 | 1–64 |
| Glossy samples | 1 | 2 | 8 | 1–64 |
| Diffuse samples | 0 (adaptive) | 4 | 16 | 0–64 |
| Samples per area emitter | 1 | 4 | 16 | 1–64 |
| Dielectric interfaces | 12 | 12 | 24 | 0 (off), 2–32 |

These are ray budgets within one frame, not accumulated-frame counts. Area
lighting costs grow with emitter count and samples per emitter. Arbitrary diffuse
budgets use a shifted Hammersley set; the original one/four-ray pattern remains
unchanged at the default. `sunAngularRadius` enables soft directional shadows in
HQ, in radians; zero retains hard directional shadows. `sunIntensity` is a linear
multiplier, with one preserving the original sun.

Set `rayTracingDenoising = false` to display the HDR lighting before MetalFX. This
forces native resolution and bypasses the neural/temporal scaler. Surface-aware
visibility reconstruction remains active. This diagnostic mode still uses the
HQ guide infrastructure and device requirements; it is not a separate portable
path tracer. Compare it against denoised output with identical ray counts,
resolution and scene state. The default remains denoising enabled.

`verticalFieldOfView` defaults to 50 degrees. `nearClipDistance` and
`farClipDistance` default to 0.1 and 1000 world units. A narrow lens aimed at a
distant subject benefits from moving the near clip plane forward. All camera
changes reset reconstruction history; focal scale follows the selected lens.

## Material additions and limits

`GPUSimSurfaceMaterial.transmission` defaults to zero. HQ can follow a camera ray
through closed air/material interfaces with the configured index of refraction,
Snell refraction, total internal reflection, and first-interface Fresnel
reflection. Interface count bounds the work; zero disables dielectric camera transport. It is a preview dielectric model:
it does not implement nested media, rough transmission, volumetric absorption,
or caustics. Lightweight rendering retains the opaque appearance. Area shadow
rays currently treat material boundaries as opaque; the directional path has a
bounded straight-ray transmission approximation.

`clearcoat` and `sheen` are optional approximate preview lobes on primary surfaces,
not a complete layered BSDF. Their default weights are zero. Existing texture
maps and caller-authored programs continue to supply detailed color, roughness,
metallicity, emission and normal perturbations.

This is a bounded hybrid renderer, not Cycles parity. Three-band SH cannot resolve
all high-frequency diffuse environment detail, ordinary box mipmaps are not a GGX
prefilter, and uniform finite-emitter sampling can be noisy for sharp highlights.
More rays and denoising trade throughput for quality; they do not add missing
light paths. The default ACES-style display curve remains available; a caller's
display program can match Blender's AgX independently of those transport limits.
