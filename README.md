# AVBD Metal

Swift + Metal implementation of **Augmented Vertex Block Descent**
([Giles, Diaz, Yuksel — SIGGRAPH 2025](https://graphics.cs.utah.edu/research/projects/avbd/))
grown into a unified solver where **rigid bodies, volumetric soft bodies, and
cloth are first-class citizens** in one solve loop — plus a robotics/world-model
research playground built on top of it.

## Status

**Done**
- **Rigid bodies**: boxes/spheres/tori/capsules, SAT narrowphase with feature-ID
  warm-started manifolds, hard & soft joints, hinges with servo/velocity motors
  and limits, springs, fracture, kinematic spinners. CPU reference (port of
  [avbd-demo3d](https://github.com/savant117/avbd-demo3d)) backs GPU parity tests.
- **Soft bodies**: 3-DOF particles (exact 3×3 block solve), stable Neo-Hookean
  tets, voxelized implicit shapes (bunny, rubber tires); tet boundary faces are
  collision triangles, so soft–soft and rigid-face–soft contact just work.
- **Cloth**: StVK membrane + quadratic (Bergou) bending elements, hard
  inextensible rods (AL duals) for structural edges, V-T / E-E / rigid-feature-T
  contacts on a unified 4-slot stencil with persistent warm-started λ/penalty,
  friction cones, and topological exclusions.
- **Detection at scale**: Voronoi temporal tracking (per-vertex/edge Best-4
  closest-element sets, topology propagation, relative-velocity-staggered grid
  reseeds), AABB multi-cell element binning, rigid-cell-flagged broadphase.
- **Rendering**: instanced analytic shapes, soft surfaces with angle×area
  weighted normals (flat default for cloth, opt-in thickness), GTAO, ACES.
- **App**: live solver parameters, per-demo material tunables (membrane µ,
  bending, tet stiffness, friction, …), Small→Colossal scaling, mouse drag,
  camera-stable resets.
- **Tests**: 16 suites — accuracy, convergence, stability, friction sweep,
  mass-ratio stacking, fracture, GPU↔CPU parity, plus adversarial cloth/soft
  gates (fold interpenetration, forced strip crossing, 8× box-on-cloth,
  hammock inextensibility, whip energy envelope, soft-block stacking).

**WIP / research**
- OGC log-barrier contact stage implemented but compiled out (`OGC_BARRIER 0`):
  needs the paper's per-iteration re-detection cadence to be safe (see in-kernel
  notes for the full experiment record).
- Solve-primal dispatch floor (~3 ms = colors × iterations serial chain) —
  next lever is palette reduction (Jacobi-accepted tets/rods).
- Robotics Lab: gantry-pusher Push-T environments (11k env-steps/s @ 256),
  BC visuomotor policy ~90% from pixels; latent world-model (LeWM) planning
  still below the scripted oracle.

## Papers & techniques

| Source | What's used |
|---|---|
| **AVBD** (Giles, Diaz, Yuksel, SIGGRAPH 2025) | Core solver: per-color 6×6 LDLᵀ block primal, bounded AL duals, penalty ramping, α-stabilization, γ warm-start |
| **VBD** (Chen et al., SIGGRAPH 2024) | Block-descent structure the GPU loop follows; trust-region step caps |
| **OGC** (Chen et al., SIGGRAPH 2025, `refs/ogc_paper.pdf`) | Contact model alignment: face blocks push along ±n with persistent side, boundary closest points use radial block directions; 2-stage activation implemented (dormant) |
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
ribbons/box-on-cloth, and `bed` (rigid frame + soft mattress/pillows + cloth
blanket in one solve).

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
