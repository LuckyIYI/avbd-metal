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
| **[OGC](https://graphics.cs.utah.edu/research/projects/ogc/)** (Chen et al., SIGGRAPH 2025) | Deformable contact-force model alignment (face blocks ⊥, radial boundaries), persistence, and divergent cloth-cloth log barrier; its isotropic conservative bound remains available as the legacy truncation mode |
| **[Divide and Truncate / Planar-DAT](https://arxiv.org/abs/2604.15513)** (Chen et al., 2026) | Direction-aware V–T/E–E truncation after prediction and every VBD color; OGC remains the contact-force model. Automatic mode selects Planar-DAT for shell-only scenes and opt-in volumetric self-contact, while ordinary closed tet bodies and mixed scenes use the lower-overhead isotropic bound. Signed tet-volume prevention is a separate experimental opt-in; rigid curved-trajectory and animated-DoF DAT are not implemented |
| **Stable Neo-Hookean** (Smith et al. 2018) | Tet FEM material with per-vertex SPD Hessian |
| **Bergou et al. 2006** | Quadratic bending; hinge K derived numerically from the intrinsic unfolded shape |
| **IPC / Codimensional IPC** (Li et al.) | Lagged friction formulation; contact-radius framing |
| **[Newton](https://github.com/newton-physics/newton) / Jitter Physics 2** | Convex support maps, MPR overlap, GJK separation, deterministic clipped contact manifolds, and persistent convex feature identities; analytic non-convex torus contact remains a separate path |
| **[CoACD](https://github.com/SarahWeiii/CoACD)** | Optional offline decomposition of connected concave source meshes into validated, canonical, content-addressed convex compounds; ordinary builds consume only checked-in cooked assets |
| Ericson, *Real-Time Collision Detection* | Closest-point primitives (point-triangle, segment-segment) |

Engine techniques: scene-adaptive coloring (static topology palette for soft
scenes, contact-aware dynamic GS coloring for rigid stacks), lane-split SIMD
primal (8 threads/body, `simd_shuffle_xor` reduction — rigids included),
counting-sort spatial hash, CAS open-addressing persistence maps, sign-memory
crossing protection with boundary release, runtime shader concatenation
(`00_common` … `60_robotics`).

Rigid contact materials keep geometry and constitutive response separate.
`friction` / `dynamicFriction` bound tangential force; the optional
`torsionalFriction` on a body or collider is an effective contact radius in
scene-length units and bounds one aggregate manifold torque by
`normalLoad * torsionalFriction`. The two shape values are averaged (matching
[Newton XPBD's material rule](https://github.com/newton-physics/newton/blob/main/docs/solvers/index.rst)); zero is the source-compatible default. Enabling it
does not manufacture sphere or capsule witnesses: a sphere on a plane remains
one exact contact, a flat box keeps its clipped manifold, and both feed the
same solver-level twist mode. Rolling friction is not yet implemented.

## Targets

| Target | Purpose |
|---|---|
| `SimCore` | Backend-neutral math, scene descriptions, geometry, mesh import, and deterministic utilities |
| `PhysicsAVBD` | Concrete AVBD CPU/Metal solver, tuned demo scenes, and shader resources; depends only on `SimCore` |
| `Robotics` | Robot models, MJCF import, calibration, and hardware-facing contracts; depends only on `SimCore` |
| `RL` | Vector task contracts, task registry, rewards, and the current AVBD-backed environments |
| `MLXRL` | MLX learning, checkpoint/evidence formats, policy runtime, and research pipelines |
| `avbd` | CLI: `run`, `bench`, `parity`, `profile` (per-kernel GPU timings), `clothgate` (gap/stretch/KE gates), `rodexp` |
| `AVBDApp` | macOS app: physics playground and data-driven Policy Replay bundle viewer |
| `AVBDTests` | Cross-layer and MLX integration tests; lower layers also have dependency-limited test targets |

The production dependency graph is intentionally small: `SimCore` is the
foundation; `PhysicsAVBD` and `Robotics` are independent siblings; `RL`
composes them; and `MLXRL` is the only layer that depends on MLX. Policy
metadata and runtime stay together with learning until they have an independent
consumer or release cadence. Clients depend on the owning products directly;
the `AVBD` name is reserved for the concrete physics backend rather than used
as a generic simulator or learning namespace.

## Robot learning

The vector RL path is task-agnostic and runs the simulator and MLX learner on
Apple silicon. Built-in tasks include Cartesian Push-T, randomized articulated
arm Push-T, the full seven-axis Panda port of ManiSkill PushT-v1, the imported
19-DoF Unitree H1 tasks aligned to Isaac Lab, the printable 16-DoF Arachne-15
spider, and earlier native humanoid research tasks. Policy Replay loads
portable bundle directories whose manifest owns the policy runtime, exact
simulation configuration, cameras, controls, and metrics. The app has no
policy-specific page switch: choose **Import Bundle** to add an unverified
bundle, or package it through the independent release index to advertise a
qualified release. The currently supported runtime ABIs and exact release
evidence are documented with [the policy bundles](checkpoints/README.md).
The Arachne-15 CAD, hardware, and device-qualification workspace is maintained
as a separate project; this repository contains only its reviewed simulator
runtime snapshot and sealed policy releases.

```bash
make build
.build/release/avbd list-rl
.build/release/avbd rl-smoke pusht-state-v0 --envs 256 --frames 200
.build/release/avbd rl-smoke arm-pusht-v0 --envs 128 --frames 200
.build/release/avbd rl-smoke humanoid-isaac-flat-v0 --envs 128 --frames 200
.build/release/avbd rl-smoke arachne15-velocity-v0 --envs 128 --frames 200

# MLX's Metal library is only produced by xcodebuild, not plain SwiftPM.
make ml-tool
.xcbuild/Build/Products/Release/avbd train-rl humanoid-isaac-flat-v0 \
  --run my-h1-seed-1 --seed 1 --envs 4096 --updates 300 \
  --horizon 24 --epochs 5 --batch 24576 --lr 0.001 \
  --gamma 0.99 --gae-lambda 0.95 --entropy 0.01 --target-kl 0.01 \
  --action-std 1 --hidden-layers 128,128,128 \
  --action-distribution gaussian --no-observation-normalization \
  --no-symmetry-augmentation
.xcbuild/Build/Products/Release/avbd eval-rl humanoid-isaac-flat-v0 \
  --checkpoint runs/humanoid-isaac-flat-v0/my-h1-seed-1/checkpoints/update-000300 \
  --envs 256 --episodes 512 --seed 10001 --json \
  --output runs/humanoid-isaac-flat-v0/my-h1-seed-1/eval.json

# Open Policy Replay. The native H1 Flat actor is qualified on the current
# BSD-source collision hulls; the source-verified Unitree actor remains an
# independent Sim2Sim reference.
make app-ml
AVBD_POLICY_REPLAY=1 open AVBD.app
# Or import a bundle directory directly while developing:
AVBD_POLICY_BUNDLE=/absolute/path/to/my-policy-bundle open AVBD.app
```

## Quick start

```bash
swift test                                    # full battery
swift test --filter RLFrameworkTests          # vector RL contract + GAE
make verify-release                           # full arm64 Mac release gate
make app && open AVBD.app                     # interactive app
.build/release/avbd run convexdecomp --frames 300 # cooked convex-compound demo
.build/release/avbd run classicrigids --frames 720 # classic meshes on a primitive table
.build/release/avbd run bed --frames 300
.build/release/avbd profile clothfold --scale 16 --frames 80
.build/release/avbd clothgate drape --frames 300
```

## Convex mesh pipeline

Concave triangle meshes are decomposed offline; simulator and app builds load
only the validated, content-addressed result. CoACD is pinned to `1.0.11` and
is never a runtime dependency.

```bash
python3 -m venv .venv-convex
.venv-convex/bin/pip install -r Tools/requirements-convex.txt
.venv-convex/bin/python Tools/cook_convex_asset.py \
  --input model.obj --output model.avbdconvex.json \
  --debug-obj model.convex-debug.obj --method coacd --seed 0 \
  --up-axis y --scale 1 1 1
python3 -S Tools/cook_convex_asset.py \
  --verify model.avbdconvex.json --input model.obj \
  --debug-obj model.convex-debug.obj
```

Load the result with `ConvexCompoundAsset.load(from:)`, create the owning body
with its mass and inertia once, then call
`PhysicsScene.addConvexCompound(body:asset:...)`. Replicas share the immutable
hull table; compound parts add collision geometry, not mass. The current
bounded runtime accepts up to 256 parts, 64 vertices per hull, and 16 vertices
per maximal coplanar face. In the app, **Collision hulls** and **Hull
wireframe** visualize the exact cooked parts independently of the visual mesh.
Scale components are applied in source coordinates, then `--up-axis` is baked
with a right-handed rotation into the runtime's Z-up frame. Load the visual
mesh with the same source up-axis and apply the identical source-axis scale to
its vertices; both paths then produce matching geometry. Hulls collide with
hulls, boxes, spheres, capsules, and deformable triangle/tet surfaces. Torus is
non-convex: any potentially colliding torus-hull pair is rejected explicitly
when either CPU or GPU solver is constructed rather than silently omitted.
