# Caller-authored display transforms

`GPUSimDisplayTransform` replaces the final display curve without changing
lighting, ray counts, HDR accumulation or material inputs. No transform preserves
the existing ACES-style curve. The renderer does not embed Blender, AgX, OCIO or
an application color configuration.

```swift
let program = GPUSimDisplayProgram(body: "return radiance / (1 + radiance);")
let display = try GPUSimDisplayTransform(device: device, program: program)
let renderer = try GPUSimRenderer(device: device, displayTransform: display)
renderer.options.displayExposure = 0.5  // stops, before the display program
```

The Metal body receives `radiance` in scene-linear RGB after exposure and an
optional `texture3d<float> lut`. Return **linear display RGB**; the existing sRGB
attachment encodes the result once. `sRGBToLinearExact` and `linearToSRGBExact`
are available. If an exported display program already returns encoded sRGB,
decode that result before returning it. Do not gamma-correct saved BGRA bytes.

`supportingSource` supplies helper functions in a private shader namespace. For
example, a caller can export an OCIO processor as MSL, upload its 3D table, and
wrap its entry point in `body`. The caller owns the input/output color-space
contract, table layout, sampling and any conversion outside the display program.
This hook supports one optional 3D table, not every resource layout OCIO can
generate. Analytic programs need no table. Only use trusted shader source.

## Resources and lifetime

Construct the renderer with the intended program. Programs are immutable and
participate in the material library's shader-cache key. A different program
requires a new renderer. With the same program, `setDisplayTransform` can switch
the table or temporarily disable the transform with `nil`; neither operation
invalidates scene-linear reconstruction history. Re-enabling the original
transform restores it without compilation. An incompatible program or device
throws before changing the active transform.

Tables must belong to the renderer's Metal device, be cubic with an edge of
2–129 texels, use RGBA16Float or RGBA32Float, have one mip level and be readable
by shaders. Memoryless storage is rejected. The default allocated texture budget
is 64 MiB and source text is limited to 256 KiB. The hook uses fragment texture
slot 8, independently of the material argument buffer. Nil compiles a wrapper
around the existing display function with no LUT sample.

The custom curve is applied to opaque meshes, soft surfaces, the floor,
environment background, HDR composite and MetalFX presentation. Display exposure
remains independent of ambient lighting exposure. Debug color overlays are not
physical radiance and retain their existing behavior.

## Validation

`DisplayTransformTests` checks GPU lookup axes, texel centers, exposure,
out-of-range clamping, sRGB encoding, texture validation/budgets, shader-cache
identity and safe enable/disable behavior. The default renderer suite remains
unchanged by a caller who does not opt in.

An optional independent OCIO fixture can be supplied with
`AVBD_DISPLAY_LUT_FIXTURE=/path/to/fixture swift test --filter DisplayTransformTests`.
The directory contains `display-lut.json` (`size`, `file`, `shader_file`, `body`),
the cubic RGBA32Float table, supporting MSL source, and two tightly packed
Float32 RGB files: `display-probe-input.f32` and `display-probe-reference.f32`.
The reference must come from the corresponding CPU OCIO processor. The GPU test
requires maximum error below 0.05 of one 8-bit code and p99 below 0.01 code.

A consumer validation on Apple M5 used Blender 5.2.1's bundled OCIO 2.5.0,
AgX / None / sRGB, exposure 0, gamma 1, no curves and no white balance. The
exported processor uses a 57³ table and its own tetrahedral MSL interpolation.
Across 4,096 HDR, gray, saturated and black probes, GPU versus CPU error was
0.00370 code maximum and 0.00251 at p99. Blender's actual 8-bit `save_render`
output agreed within 0.5 code, including quantization. Six scene comparisons
used identical cameras, light geometry/radiance, world, floor and normals;
maximum unjittered camera disagreement was below 0.001 pixel at 900×900.

These checks establish color and camera agreement, not BSDF or light-transport
parity. Cloudy procedural wood, regular stone patterns, reflection/transmission
differences and low-sample shadow noise require separate material or lighting
work. More display precision does not resolve them.
