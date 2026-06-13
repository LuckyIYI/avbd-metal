# AVBD Metal

Swift + Metal implementation of **Augmented Vertex Block Descent**
([Giles, Diaz, Yuksel — SIGGRAPH 2025](https://graphics.cs.utah.edu/research/projects/avbd/))
grown into a unified solver where **rigid bodies, volumetric soft bodies, and
cloth are first-class citizens** in one solve loop — plus a robotics/world-model
research playground built on top of it.

## Papers & techniques

| Source | What's used |
|---|---|
| **AVBD** (Giles, Diaz, Yuksel, SIGGRAPH 2025) | Core solver: per-color 6×6 LDLᵀ block primal, bounded AL duals, penalty ramping, α-stabilization, γ warm-start |
| **VBD** (Chen et al., SIGGRAPH 2024) | Block-descent structure the GPU loop follows; trust-region step caps |
| **OGC** (Chen et al., SIGGRAPH 2025, `refs/ogc_paper.pdf`) | Contact model alignment (face blocks ⊥, radial boundaries) + the penetration-free machinery: 2-ring-excluded conservative bounds, Eq-28 warmstart truncation, counter-driven in-loop bound refresh (indirect dispatch, no CPU sync), divergent cloth-cloth log barrier |
| **Stable Neo-Hookean** (Smith et al. 2018) | Tet FEM material with per-vertex SPD Hessian |
| **Bergou et al. 2006** | Quadratic bending; hinge K derived numerically from the intrinsic unfolded shape |
| **IPC / Codimensional IPC** (Li et al.) | Lagged friction formulation; contact-radius framing |
| Ericson, *Real-Time Collision Detection* | Closest-point primitives (point-triangle, segment-segment) |

Engine techniques: scene-adaptive coloring (static topology palette for soft
scenes, contact-aware dynamic GS coloring for rigid stacks), lane-split SIMD
primal (8 threads/body, `simd_shuffle_xor` reduction — rigids included),
counting-sort spatial hash, CAS open-addressing persistence maps, sign-memory
crossing protection with boundary release, runtime shader concatenation
(`00_common` … `60_robotics`).

## Targets

| Target | Purpose |
|---|---|
| `AVBDCore` | Solver library: CPU reference + GPU implementation + scenes |
| `avbd` | CLI: `run`, `bench`, `parity`, `profile` (per-kernel GPU timings), `clothgate` (gap/stretch/KE gates), `rodexp` |
| `AVBDApp` | macOS app: viewer, demos, Robotics Lab |
| `AVBDTests` | Full battery (16 suites) |

## Quick start

```bash
swift test                                    # full battery
make app && open AVBD.app                     # interactive app
.build/release/avbd run bed --frames 300
.build/release/avbd profile clothfold --scale 16 --frames 80
.build/release/avbd clothgate drape --frames 300
```

Demos: stacks/walls/pyramids, ratio stack, friction-sweep slope, jenga,
dominoes, bridge, tensegrity, chainmail, treadmill, car, marble run,
trebuchet, wrecking ball, Rube Goldberg, brick android statue, soft bunny,
steel-and-rubber cart (`softwheel`), cloth fold/drape/multidrape/hammock/
ribbons/box-on-cloth, `twist` (a hanging sheet winds into a rope against a
spinning bar — the OGC stress test), and `bed` (rigid frame + soft
mattress/pillows + cloth blanket in one solve).

## Performance (M1 Ultra, release, 20 iterations)

| Scene | Bodies / verts | ms/frame |
|---|---|---|
| clothfold Colossal | 26k cloth verts | 15.3 |
| drape on sphere | 20k cloth verts | 9.3 |
| bed Colossal | 5.5k (rigid+soft+cloth) | 13.8 |
| softwheel | 1.4k (4 rubber tires) | 6.5 |
| boxpile ×12 (10 iters) | 28.8k rigid | 9.4 |
| boxpile ×24 (4 iters) | 115k rigid | 33.6 |

## Notes

- `refs/` (gitignored) holds reference material: the AVBD & OGC papers and
  third-party sample code. It is not part of the repository history.
- Metal fast math folds `isinf()` — hard-constraint/fracture logic uses flag
  bits, never inf compares.
- Solver parameters (paper Table 2): β ramp 5e3 lin / 1e2 ang, α 0.99,
  γ 0.999, 10–20 iterations depending on scene; all live-tunable in the app.
