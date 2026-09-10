# Lighting validation and measured costs

Measured on Apple M5, 2026-09-10. These are local measurements, not performance
guarantees for other devices. Baseline is upstream
`4f5be0581a2b6aa7e408a125c37a89bfef7f1344`.

## Unchanged defaults

Both revisions ran the existing `material-preview` fixture at 1024×768, with no
environment, area lights, or optical material lobes. Both used Swift debug builds
and the same toolchain. Trials alternated baseline and changed revision, with
three trials per configuration. Numbers below are medians of trial means.

| Configuration | Baseline GPU ms | Changed GPU ms | Baseline synchronized ms/frame | Changed synchronized ms/frame |
|---|---:|---:|---:|---:|
| Fast, 60 frames | 2.426 | 3.495 | 8.48 | 8.71 |
| Fast, 600 frames | 3.210 | 2.999 | 8.62 | 8.55 |
| HQ, 60 frames | 3.881 | 4.070 | 8.62 | 8.76 |

Short runs showed substantial GPU timing variation. Longer Fast runs overlap:
baseline trial means were 3.025–3.226 ms and changed means were 2.995–3.274 ms.
The device clock and temperature were not controlled. These measurements do not
establish either a speedup or zero regression. The synchronized column includes
readback and presentation scheduling; it is not pure GPU cost.

The third 60-frame Fast outputs were pixel-identical. HQ mean absolute difference
was 0.00362 of one 8-bit code, with a maximum difference of one code. Default ray
budgets and visual appearance are retained; optional lighting adds work when used.
All trial values, including the slower first changed HQ run, are retained in
[StudioLightingBenchmarks.json](StudioLightingBenchmarks.json).

Reproduce each mode on both checkouts, without overlapping runs:

```sh
swift build --product material-preview --jobs 1
.build/debug/material-preview --fast --frames 600 --output /tmp/fast.png
.build/debug/material-preview --frames 60 --output /tmp/hq.png
swift test --jobs 1 --filter GPUSimRendererTests
```

## Consumer catalog

A separate consumer rendered 1,000 distinct generated meshes in ten sequential
100-object processes. It reused the renderer, environment and material library
within each process and reset history between objects. The consumer and its
catalog are not bundled with this renderer.

- 1920×1080 output, 1286×722 internal resolution, MetalFX reconstruction.
- Three finite disk lights, a constant HDR world, two accumulated frames per
  object, one sample per emitter per frame, real-time glossy/diffuse budgets.
- Native process wall time totaled 44.06 seconds, including initialization,
  scene preparation, synchronized readback and JPEG writing: 44.06 ms per image.
- Outer guarded invocation took 47.49 seconds. GPU frame median was 14.75 ms;
  p95 was 18.38 ms. Two GPU frames are used per saved image.
- Maximum recorded Metal allocation was 538.64 MiB. Observed process-tree RSS
  including the consumer coordinator peaked at 401.73 MiB. These are different
  accounting measures and are not additive.

On the same 22-object pilot, one/two/four/sixteen accumulated frames averaged
33.5/45.9/82.4/294.3 ms per saved image, including native startup and writing.
Two frames removed much of the visible single-frame noise while keeping the
catalog throughput in tens of milliseconds. These times exclude geometry
generation, renderer compilation and final video encoding. Camera and material
models still differ from Cycles; exposure is not an AgX implementation.

## Checks

The renderer suite executed 106 tests: 105 passed, no failures, and the existing
opt-in AO timing benchmark was skipped. New GPU tests cover finite-emitter
orientation, inverse-square behavior, geometry occlusion, linear HDR environment
energy, sRGB palette conversion, exposure before tone mapping, dielectric slab
refraction and bypassing the MetalFX scaler without modifying raw HDR. Existing
floor, AO, SSR, visibility, reconstruction and renderer regressions passed.
