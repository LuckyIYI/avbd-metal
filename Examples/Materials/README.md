# Material preview

Real image maps, independently sized and mipmapped. The optional fixture is
[Wood Table 001](https://polyhaven.com/a/wood_table_001), by Dimitrios Savva and
Rico Cilliers, published by Poly Haven under CC0. Images are downloaded explicitly,
not bundled into the engine or fetched during builds/tests.

```sh
python3 Examples/Materials/fetch-example.py
swift run -c release material-preview \
  --base-color /tmp/avbd-material-example/wood_table_001_diff_1k.jpg \
  --roughness /tmp/avbd-material-example/wood_table_001_rough_1k.jpg \
  --normal /tmp/avbd-material-example/wood_table_001_nor_gl_1k.jpg --normal-gl \
  --output /tmp/material-hq.png
```

Add `--fast` to exercise Fast rendering. `--procedural` demonstrates a caller-owned
Metal program. `--asset path.obj` (or USD) exercises the importer; the fixed preview
camera may need adjustment for differently scaled assets. `--frames 240` reports
synchronized rendering/readback time, not an interactive frame-rate claim.

Lighting controls use the same public API as a consuming application:

```sh
swift run -c release material-preview --area-light --quality balanced --frames 4
swift run -c release material-preview --environment /path/to/linear-environment.exr
swift run -c release material-preview --area-light --no-denoise --frames 1
```

Use `--environment-srgb` for an encoded SDR environment image. The default expects
linear HDR input. `--no-denoise` displays native-resolution HDR lighting before
MetalFX; ray budgets remain controlled by `--quality`. See
[StudioLighting.md](../../Documentation/StudioLighting.md) for limits and costs.
