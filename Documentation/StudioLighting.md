# Configurable environment and finite-area lighting

The renderer keeps its existing Fast and HQ defaults: no external environment,
no area lights, opaque materials, one glossy ray and the existing adaptive HQ
diffuse budget. A caller can opt into richer lighting independently of physics,
asset import, or procedural material authoring. No images are downloaded and no
studio preset is embedded in the renderer.

## HDR environment

```swift
let image = try GPUSimMaterialLibrary.loadTexture(
    device: device, url: environmentURL, sRGB: false)
let environment = try GPUSimEnvironmentLight(
    device: device, texture: image)
let materials = try GPUSimMaterialLibrary(
    device: device, materials: authoredMaterials)
let renderer = try GPUSimRenderer(
    device: device, scene: scene, materials: materials, environment: environment)
renderer.options.environmentIntensity = 1
renderer.options.environmentRotation = .pi / 4
renderer.options.showsEnvironmentBackground = true
// Swapping lighting preserves material shaders and geometry acceleration structures.
try renderer.setEnvironment(anotherPreparedEnvironment)
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
north at +Z and a top-left UV origin. Positive `environmentRotation` rotates the map around
world +Z. `showsEnvironmentBackground` changes camera visibility independently of lighting.
Intensity scales both diffuse and specular environment radiance.

Nine cosine-convolved spherical-harmonic coefficients are prepared once from
4,096 spherical samples. `diffuseSamples` on the environment constructor changes
only that preparation cost. The result approximates irradiance divided by pi;
floating-point HDR values above one are retained. Diffuse lighting uses those
coefficients for rigid, soft and floor surfaces. Raster specular fallback uses
the caller's mip chain; HQ uses GGX ray sampling and actual scene intersections.
Each renderer combines shared material resources with its own immutable environment
bindings. Environment textures participate in the combined deduplicated texture
budget; rejected replacements leave existing lighting unchanged. Submitted frames
retain their original bindings. Image/SH preparation happens once; intensity and
rotation are per-frame settings. Lighting changes reset HQ history, while display
exposure does not. Argument-buffer Tier 2 is required.

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
supported. `options.validateLighting()` throws for an excess count; drawing an
invalid configuration reports a renderer failure rather than silently dropping lights.

HQ evaluates finite distance, emitter orientation, GGX response, and bounded
shadow rays toward samples on each emitter. Emitters are visible in the sky
background and to reflection rays, but contribute no collision geometry or mass.
Smooth receivers get the emitter's specular image from those reflection rays;
the direct area-light pass supplies specular only where reflection rays are not
traced (roughness at or above the reflection cutoff), so no emitter is counted twice.
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

The denoiser receives signed world-space normals, view-dependent Schlick specular
albedo, and specular hit distances from the existing reflection rays; no extra
rays are traced for guides. Distances use world units and average evaluated
specular samples. Misses use the finite tracing horizon; pixels without an
evaluated specular lobe use zero. Transmission previews use the first refracted
segment. These are approximate guides for the combined lighting signal,
particularly where reflection and transmission overlap. Fast mode allocates no
specular-distance texture.

`verticalFieldOfView` defaults to 50 degrees. `nearClipDistance` and
`farClipDistance` default to 0.1 and 1000 world units. A narrow lens aimed at a
distant subject benefits from moving the near clip plane forward. All camera
changes reset reconstruction history; focal scale follows the selected lens. Primary
ray queries respect the same clip planes, including off-axis ray distances.

## Material additions and limits

`GPUSimSurfaceMaterial.previewOptics.transmission` defaults to zero. HQ can follow a camera ray
through closed air/material interfaces with the configured index of refraction,
Snell refraction, total internal reflection, and first-interface Fresnel
reflection. Interface count bounds the work; zero disables dielectric camera transport. It is a preview dielectric model:
it does not implement nested media, rough transmission, volumetric absorption,
or caustics. Lightweight rendering retains the opaque appearance. Area shadow
rays currently treat material boundaries as opaque; the directional path has a
bounded straight-ray transmission approximation.

`previewOptics.clearcoat` and `previewOptics.sheen` are optional approximate preview lobes on primary surfaces,
not a complete layered BSDF. Their default weights are zero. Existing texture
maps and caller-authored programs continue to supply detailed color, roughness,
metallicity, emission and normal perturbations.

This is a bounded hybrid renderer, not Cycles parity. Three-band SH cannot resolve
all high-frequency diffuse environment detail, ordinary box mipmaps are not a GGX
prefilter, and uniform finite-emitter sampling can be noisy for sharp highlights.
More rays and denoising trade throughput for quality; they do not add missing
light paths. The default ACES-style display curve remains available; a caller's
display program can match Blender's AgX independently of those transport limits.


## Bounded finite-emitter sampling

The default remains stratified sampling of every light. `areaLightSamples` is a
**per-light** budget in this mode. Secondary diffuse/reflection hits previously
inherited this budget, multiplying visibility work by both bounce samples and
light count. Callers can now choose an independent budget:

```swift
var quality = GPUSimRayTracingQuality.balanced
quality.secondaryAreaLightSamples = 1
renderer.options.rayTracingQuality = quality
```

Zero (the default) inherits the primary budget; positive budgets clamp to 1...64.
Note that any body appearance override makes `GPUSimEnvironmentBatch` capture each
reference to a shared solver separately (one synchronized snapshot per reference per
frame); highlight bodies sparingly in large repeated-environment views.
This changes variance, not the lighting model. More noise at secondary hits is
possible. It does not lower primary direct-light sampling or turn off shadows.

For larger light sets, callers can opt into `quality.areaLightSampling =
.powerWeighted`. Both primary and secondary counts then specify **total** light
samples, independent of emitter count. Each sample selects an emitter with a
probability proportional to emitted luminance times area, with a small uniform
floor to preserve support. Its contribution is divided by that probability.
Emitter position sampling uses separate random dimensions. This is ordinary
importance sampling, not reservoir sampling or ReSTIR; there is no persistent
history, neighbor lookup, reservoir allocation or spatial reuse pass.

Power weighting ignores receiver orientation, distance and occlusion. A dominant
but hidden emitter can increase variance; per-light stratification can be better
for a few studio lights. The mode is opt-in and should be evaluated at equal time
and acceptable image quality, not just equal numerical sample counts.

Both modes conservatively reject emitters wholly below the receiver hemisphere,
black emitters, and one-sided emitters facing away. Secondary directional shadow
queries are omitted when the directional light has zero intensity. These skips
remove zero-contribution work; they do not relax visibility or geometric bias.


### Exploratory measurements

See `LightingEfficiencyMeasurements.json` for all runs, including unfavorable
ones. On a local Apple M5 cabinet fixture at native 1600×1200, high quality,
one frame and 64 primary area samples per light, changing the inherited secondary
budget from 64 to 1 reduced median GPU time from 1446.6 to 289.5 ms. The single
saved-frame mean absolute display-JPEG difference was 0.197/255; that is not an
unbiased or converged ground-truth quality metric. The same-budget median was
approximately unchanged (1470.2 vs 1489.4 ms in the paired repeat).

At 800×600 internal resolution with MetalFX output at 1600×1200, a two-frame
balanced preset with 8 primary and 1 secondary samples per light measured 62.9 ms
median / 78.0 ms maximum per saved image after initialization over 14 views.
The first render took 151.3 ms, excluding application startup. Grain remains.
A higher-budget two-frame trial exceeded 100 ms. The optional power selector's
low-budget gain did not repeat; it is an experimental alternative for evaluating
larger light sets, not a recommended default for a three-light studio.

These are application-level observations using external procedural fixtures.
They do not establish portable performance or a universal quality improvement.
The 19 related GPU correctness tests cover area-light energy and occlusion,
material transport, environment lighting and MetalFX reconstruction.
