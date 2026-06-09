# AVBD Metal

Production-oriented Swift + Metal implementation of **Augmented Vertex Block
Descent** (Giles, Diaz, Yuksel — SIGGRAPH 2025,
[paper](https://graphics.cs.utah.edu/research/projects/avbd/)) for 3D rigid
body dynamics with hard constraints, frictional contact, joints, springs,
and fracture.

## What's here

| Target | Purpose |
|---|---|
| `AVBDCore` | The solver library: CPU reference + Metal GPU implementation |
| `avbd` | Headless CLI: run/bench/parity on demo scenes |
| `AVBDApp` | macOS app: real-time viewer, demos, live parameter tuning, mouse dragging |
| `AVBDTests` | 18 tests: analytic accuracy, convergence, stability, friction, fracture, determinism, GPU↔CPU parity |

## Quick start

```bash
swift test                      # full test suite
make app && open AVBD.app      # interactive app
make build
.build/release/avbd run stack --frames 300
.build/release/avbd bench boxpile --scale 6
.build/release/avbd parity wall --frames 120
```

In the app: drag bodies with the mouse, ⌥-drag orbits, right-drag pans,
scroll zooms. All AVBD parameters (iterations, α, β, γ, gravity, time scale)
are live-tunable. Every demo has Small/Medium/Large/Giant/Colossal sizes
(CLI: `--scale 1/2/4/8/16`) for stress testing.

## Architecture

**CPU reference** (`Sources/AVBDCore/CPU`) is a faithful port of the official
[avbd-demo3d](https://github.com/savant117/avbd-demo3d) sample — same math,
same 6×6 LDLᵀ block solve, same SAT box collision with feature-ID contact
persistence. It is the ground truth for the GPU parity tests.

**GPU solver** (`Sources/AVBDCore/GPU` + `Shaders/`) runs the entire step on
the GPU in a single command buffer:

1. **Broadphase** — spatial-hash *grouping* (count → exclusive scan →
   scatter, no sort needed), 27-cell pair generation; statics/oversized
   bodies live in a small brute-forced "global" list so grid cells stay
   tight. Jointed pairs are excluded via binary search in a sorted list.
2. **Narrowphase** — OBB SAT with reference-face clipping / edge-edge
   closest points, one thread per pair. Contacts warm-start from the
   previous frame through a CAS-based pair hash map + feature-ID matching;
   sticking contacts keep their anchor points for static friction.
3. **Warm start** (paper Eq. 19) — λ ← αγλ, k ← max(k·γ, k_start), adaptive
   inertial prediction.
4. **CSR adjacency** rebuilt on GPU (degree count → scan → scatter) and
   **incremental parallel greedy coloring** (paper §4) with a compaction
   rule that keeps the color palette dense; only dynamic–dynamic edges
   constrain colors.
5. **Solver loop** — per-color primal kernels (force/Hessian gathering,
   SPD-projected geometric stiffness (§3.5), 6×6 LDLᵀ in registers,
   quaternion update Eq. 21) dispatched indirectly so empty colors cost
   ~nothing; then dual kernels (λ/k updates Eq. 11/16, friction-cone
   clamping §3.2–3.3, stiffness-bounded ramping, fracture).
6. **BDF1 velocity finalize**.

No CPU↔GPU sync inside a step; the one readback (counters page, shared
memory) happens after completion and feeds stats + the next frame's color
loop bound.

Shader sources are concatenated in filename order and compiled at runtime
(`00_common` math/types, `10_scan`, `20_broadphase`, `30_narrowphase`,
`40_solver`, `50_render`).

### Gotchas worth knowing

- Metal fast math folds `isinf()`/`isfinite()` — hard-constraint and
  fracture checks use explicit flag bits (`JointGPU.header.w`), and the
  NaN guard in the primal kernel tests exponent bits directly.
- The 6×6 LDLᵀ is Jacobi-preconditioned (unit diagonal) so fp32 survives
  penalty (10¹⁰) vs. tiny-rod-inertia scale gaps; pivots are clamped and
  the primal step is trust-region capped (body radius / 0.5 rad per
  iteration), which only engages in violent transients.

## Performance (M1 Ultra, release)

| Scene | Bodies | Iterations | ms/frame |
|---|---|---|---|
| wall ×3 | 433 | 10 | 2.9 |
| boxpile ×3 | 1,801 | 10 | 4.0 |
| boxpile ×6 | 7,201 | 10 | 3.7 |
| boxpile ×12 | 28,801 | 10 | 9.4 |
| boxpile ×24 | 115,201 | 4 | 33.6 |

## Parameters (paper Table 2)

| | Default | Meaning |
|---|---|---|
| β | 5×10³ (lin) / 10² (ang) | Penalty ramp speed; convergence rate only, not the converged result. Too high destabilizes long chains under the parallel (colored) solver |
| α | 0.99 | Stabilization: portion of pre-existing error ignored (prevents explosive correction) |
| γ | 0.999 | Warm-start decay for k and λ |
| iterations | 10 | Per-frame primal/dual sweeps |
