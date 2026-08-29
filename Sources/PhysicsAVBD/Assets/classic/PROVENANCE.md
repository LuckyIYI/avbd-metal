# Classic rigid-mesh provenance

This directory contains the visual/source meshes used by the classic rigid-body
scene. They are checked in at their runtime size, in meters, with Z as up and
the lowest vertex on `Z=0`. The collision cooker therefore uses a baked scale
of `[1, 1, 1]` and `upAxis: "z"`; rendering and collision consume the same
coordinates.

The SHA-256 values below identify bytes, not merely filenames. The Stanford
terms are restrictive: Bunny, Dragon, and Armadillo are for research and free
redistribution with acknowledgement, and may not be used commercially or in a
product for sale without Stanford's permission. Read [NOTICE.md](NOTICE.md).

## Checked conversion inputs

| Asset | Direct normalization input | Bytes | Vertices / triangles | Up axis | Input SHA-256 |
| --- | --- | ---: | ---: | --- | --- |
| Bunny | Existing user-supplied MeshLab OBJ, `Object bun_zipper.obj` | 2,408,417 | 35,947 / 69,451 | Y | `1eb35d1e21ce99e5ce911353b6be278990713448dd9e8f5c9387f9de39b32205` |
| Dragon | Trimesh 4.11.5 OBJ of the largest connected component of Stanford `dragon_vrip_res3.ply` | 3,409,874 | 22,739 / 45,533 | Y | `c778737b12332e2e5ae48624a4f2e4995aa044b99375ac6631bcc80d452e154a` |
| Armadillo | Pinned pre-existing Stanford-derived OBJ mirror | 4,634,780 | 49,990 / 99,976 | Y | `d2fd040334c12b16804fbcb9962391232cc24708244617f931baaa871210563e` |
| Teapot | Trimesh 4.11.5 OBJ converted from the solid STL | 663,388 | 4,719 / 9,438 | Z | `d1d1a31dd8e6dfade0caf94a25847c47c650f2cd249e9c86acdaaa3ecc4be00f` |

The Bunny input is byte-identical to the OBJ supplied for the earlier skinned
demo. Its object name and counts identify Stanford's full `bun_zipper`
reconstruction, but the historical MeshLab export recipe is not documented;
this table does not claim otherwise. Stanford's official
[`bunny.tar.gz`](https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz) was
also archived during provenance review: 4,894,286 bytes, SHA-256
`a5720bd96d158df403d153381b8411a727a1d73cff2f33dc9b212d6f75455b84`.
Its `bunny/reconstruction/bun_zipper.ply` payload is 3,033,195 bytes, SHA-256
`b1acc63bece78444aa2e15bdcc72371a201279b98c6f5d4b74c993d02f0566fe`.

The Dragon archive is Stanford's
[`dragon_recon.tar.gz`](https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz):
11,197,764 bytes, SHA-256
`74ac1d90989c9b1732edee82d57e9ce71452144cf4355f108d8c9c616d28d02f`.
Its `dragon_vrip_res3.ply` payload is 1,665,018 bytes, 22,998 vertices and
47,794 triangles, SHA-256
`f32b87762894bde78cd45dc05aa9fde0f5ad390c944168a96e97191ff1fc6d45`.
The conversion retained the largest component and removed 2,261 triangles of
detached scan debris; it did not decimate that component.

The Armadillo direct input is exactly
[`armadillo.obj`](https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/2f03ad344cb01ab67ce698b004dc901ce416c105f/data/armadillo.obj)
at commit `2f03ad344cb01ab67ce698b004dc901ce416c105f`. It is a pre-existing
approximately 50k-vertex Stanford-derived LOD; its decimation recipe is not
documented, and AVBD does not claim to have generated it. For traceability,
Stanford's official
[`Armadillo.ply.gz`](https://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz)
is 3,874,291 bytes, SHA-256
`8b9b56cc36e66d54429b1e1e75bd89e833645bfe0dc7c1afd1205877a7356a3f`;
the decompressed 172,974-vertex / 345,944-triangle payload has SHA-256
`483563fca686b66168e1e45339c50e71a5dc8a526a04a545a16b9237d5eaa023`.

The Teapot input was converted without decimation from Nik Clark's
[`Utah teapot (solid).stl`](https://commons.wikimedia.org/wiki/File:Utah_teapot_(solid).stl),
released under CC0 1.0. The STL is 471,984 bytes, SHA-256
`0ba068b3ffd2a927b72d5037a163fcc17860bc474c06f0e85a78ad34faccce15`.
Although its exported triangle mesh is edge-watertight, CoACD detects triangle
self-intersection and therefore uses its deterministic preprocessing path.

### Topology audit

Each checked conversion input has one connected component. The Bunny retains
the reconstruction's five open underside holes (223 boundary edges), so its
cook explicitly preprocesses at resolution 80. The Dragon largest-component
mesh still contains scan defects (21 boundary edges and 30 non-manifold edges),
so it explicitly preprocesses at resolution 100. The pinned Armadillo is a
closed, consistently oriented two-manifold and passes CoACD's automatic
manifold check without preprocessing. The Teapot is edge-watertight but has a
triangle self-intersection; CoACD's `auto` check detects it and activates
preprocessing at resolution 80. No cook silently combines disconnected source
objects.

## Deterministic normalization

`Tools/normalize_classic_obj.py` strictly parses positions and the triangular
faces in these inputs. It performs all geometry operations in float32, applies
a uniform scale, rotates the declared source up axis into the right-handed
Z-up frame, centers
the X/Y bounds, moves the minimum Z to zero, and emits only `v` and `f` records.

| Output | Scale | Post-bake translation `(x, y, z)` | Bounds min / max | Bytes | Output SHA-256 |
| --- | ---: | --- | --- | ---: | --- |
| `stanford-bunny.obj` | 5 | `(0.0842024982, -0.00768499076, -0.164934993)` | `(-0.389247507, -0.301684976, 0)` / `(0.389247507, 0.301684976, 0.771670043)` | 2,756,301 | `e0f8157b25b0a876583b0205655471dbec6b6c47a8434c967387500818d8db1c` |
| `stanford-dragon.obj` | 5 | `(0.0292262584, -0.0230619907, -0.263871491)` | `(-0.509138763, -0.227192998, 0)` / `(0.509138763, 0.227192998, 0.719348490)` | 1,751,169 | `a86bf3b22d299d5baae35493d2ff56794bd8c4518044c6a66c09757d1c674252` |
| `stanford-armadillo.obj` | 0.005 | `(-0.000050380826, -0.0000451058149, 0.271100044)` | `(-0.317536652, -0.288527489, 0)` / `(0.317536652, 0.288527489, 0.756535172)` | 3,932,692 | `ce9866e4804c4ffee0fef5043aaf0d5f14780b6a5c13e948d8254e34771fde1a` |
| `utah-teapot.obj` | 0.075 | `(-0.0468178988, -0.00703190267, -0.0000000015)` | `(-0.659109652, -0.409763932, 0)` / `(0.659109652, 0.409763932, 0.642864585)` | 343,301 | `83591f7524e6626da5dd019aa85f76782cb1f07b601440f61224c24821f5c6b7` |

Example invocation (substitute the exact checked input named in the first
table for `$INPUT`):

```bash
python3 Tools/normalize_classic_obj.py \
  --input "$INPUT" \
  --output Sources/PhysicsAVBD/Assets/classic/stanford-bunny.obj \
  --label "Stanford Bunny (zipper reconstruction)" \
  --source-up-axis y \
  --uniform-scale 5
```

All four normalization commands were run twice from their exact input bytes;
each second OBJ was byte-identical to its checked-in counterpart.

## Offline convex cooking

The canonical collision JSON and exact debug OBJ are in `../convex/classic/`.
They were produced by `Tools/cook_convex_asset.py` with CoACD 1.0.11. Runtime
code does not link or invoke CoACD. All cooks use seed 0, real-metric mode,
convex-hull approximation, merge and decimate enabled, PCA and extrusion
disabled, a 2,000 sampling resolution, MCTS 20 nodes / 150 iterations / depth
3, relative weld tolerance `1e-7`, absolute weld tolerance `1e-9`, baked scale
`1 1 1`, and `--up-axis z`.

| Asset | Threshold | Preprocess | Part cap / result | Vertex cap | Sum of part volumes (m³) |
| --- | ---: | --- | ---: | ---: | ---: |
| Bunny | 0.120 m | `on`, resolution 80 | 32 / 21 | 24 | 0.05160232787602581 |
| Dragon | 0.055 m | `on`, resolution 100 | 48 / 39 | 24 | 0.06522370866605343 |
| Armadillo | 0.040 m | `auto`, resolution 80 | 32 / 21 | 24 | 0.02764676432707347 |
| Teapot | 0.025 m | `auto`, resolution 80 | 24 / 14 | 24 | 0.15818233663594583 |

The aggregate is 95 convex parts. No asset reaches its part cap. In particular,
the Dragon was re-run with a 48-part cap: a 0.040 m cook naturally produced 47
parts, proving that an earlier 39/40 cook was cap-constrained. Raising its
threshold to 0.055 m produced 39/48 without forcing merges. A 24-vertex cap was
selected only after every resulting merged coplanar face passed the runtime's
16-vertex face-loop limit; it materially reduced surface under-approximation
compared with the initially tested 16-vertex hulls.

| Asset | JSON bytes / SHA-256 | Debug OBJ bytes / SHA-256 | Compound digest | Cache key |
| --- | --- | --- | --- | --- |
| Bunny | 118,303 / `ba14d9486477f261d5f36f24f9b421b12012ff696f1a87da6b12690c4c2c3c77` | 33,031 / `ced184df11ee24214040743e0d1f2f6452949073c83a3e3d778c0ab9ca5f7814` | `bacac8091e9e1eabea1183d9abfa2cfd298d93331d26bc8a8abc2d0287cd22a1` | `5e871545e70839f609a4d9fdc722d34a636232115fc504f1bb9daaf55800754c` |
| Dragon | 218,530 / `a1df287ac6a7debecfac19487b678859db274430484515d86a4ac5044770ec31` | 61,557 / `1bca90b7d0177506e278256dbb6a947ef126e7cd746f1d03af6d5f181fef7d1f` | `eefb3d70331ae9b87ecedc367ec5e60ba9f647ddd040933152c9b0180fe5caa7` | `4fb5acaac29f9e342bbfd518156b6f904ecb01aee675f62887bff23b527a3167` |
| Armadillo | 118,197 / `0a02f6acb74f17518f1fe73db4c0e8eb3849f509d95992935e0c71ac86f9eaa0` | 33,039 / `38199c3279bfce1b51a1a7c633f6eb98d953b966193c1a758e006b532fc294c2` | `7e730459134ab092c5e9872c3e5a9e1404f7b3af681bc3e0a670ad16bf31ec48` | `c2c2dad4a2d57897e77b77e3001eef686396b0cc94a3f8f8a94be5d349c865c5` |
| Teapot | 78,947 / `c284d669a4fb8994839df4deef00b35d8a7db7fe42c1bc902733975e8816e8be` | 21,813 / `22e59f788d80c3c494f8d725123331bc5a2266a823b5b3ebd35a2ec959c13147` | `77461425636283ed3a21e51dbe329feda1f28ff9aa34984d7662126f3767a1ba` | `a2a3f24667e8e58389c59ed6824060cec37825ca681c595b90fa77f1da5cf11a` |

The exact commands use this shared argument set followed by the per-asset
invocations below:

```sh
COMMON_ARGS=(
  --method coacd --scale 1 1 1 --up-axis z
  --max-vertices-per-hull 24 --seed 0
  --weld-relative 1e-7 --weld-absolute 1e-9
  --coacd-mcts-nodes 20 --coacd-mcts-iterations 150
  --coacd-mcts-max-depth 3 --coacd-resolution 2000
)

python Tools/cook_convex_asset.py \
  --input Sources/PhysicsAVBD/Assets/classic/stanford-bunny.obj \
  --output Sources/PhysicsAVBD/Assets/convex/classic/stanford-bunny.avbdconvex.json \
  --debug-obj Sources/PhysicsAVBD/Assets/convex/classic/stanford-bunny.debug.obj \
  --source-uri classic/stanford-bunny.obj --threshold-meters 0.12 \
  --max-parts 32 --coacd-preprocess-mode on \
  --coacd-preprocess-resolution 80 "${COMMON_ARGS[@]}"

python Tools/cook_convex_asset.py \
  --input Sources/PhysicsAVBD/Assets/classic/stanford-dragon.obj \
  --output Sources/PhysicsAVBD/Assets/convex/classic/stanford-dragon.avbdconvex.json \
  --debug-obj Sources/PhysicsAVBD/Assets/convex/classic/stanford-dragon.debug.obj \
  --source-uri classic/stanford-dragon.obj --threshold-meters 0.055 \
  --max-parts 48 --coacd-preprocess-mode on \
  --coacd-preprocess-resolution 100 "${COMMON_ARGS[@]}"

python Tools/cook_convex_asset.py \
  --input Sources/PhysicsAVBD/Assets/classic/stanford-armadillo.obj \
  --output Sources/PhysicsAVBD/Assets/convex/classic/stanford-armadillo.avbdconvex.json \
  --debug-obj Sources/PhysicsAVBD/Assets/convex/classic/stanford-armadillo.debug.obj \
  --source-uri classic/stanford-armadillo.obj --threshold-meters 0.04 \
  --max-parts 32 --coacd-preprocess-mode auto \
  --coacd-preprocess-resolution 80 "${COMMON_ARGS[@]}"

python Tools/cook_convex_asset.py \
  --input Sources/PhysicsAVBD/Assets/classic/utah-teapot.obj \
  --output Sources/PhysicsAVBD/Assets/convex/classic/utah-teapot.avbdconvex.json \
  --debug-obj Sources/PhysicsAVBD/Assets/convex/classic/utah-teapot.debug.obj \
  --source-uri classic/utah-teapot.obj --threshold-meters 0.025 \
  --max-parts 24 --coacd-preprocess-mode auto \
  --coacd-preprocess-resolution 80 "${COMMON_ARGS[@]}"
```

The per-asset settings, hashes, volume, and compound-center sanity checks are
also recorded by the checked fixture test in
`Tools/tests/test_cook_convex_asset.py`. The runtime polyhedral mass/inertia
integrator has analytic tensor and principal-frame gates in
`Tests/SimCoreTests/ConvexCompoundMassPropertiesTests.swift`; the classic-scene
gate additionally requires every dynamic asset to produce finite positive
principal inertia satisfying the physical triangle inequalities.
`make verify-convex-assets` validates the canonical source seal, cache key,
convex topology, face-loop workspace limit, positive finite volume and bounds,
debug OBJ byte parity, non-saturated part caps, aggregate part budget, and
expected compound digests. The full cook was then repeated into a temporary
directory and both JSON and debug OBJ were byte-compared with the checked-in
files.

The Dragon has no rig, motion, morph, or deformation data. In the classic
tabletop scene it remains an intact rigid mesh; the simulator changes only its
whole-body pose, like moving a physical reproduction of the scanned object.
