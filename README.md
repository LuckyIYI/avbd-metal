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
| **[OGC](https://graphics.cs.utah.edu/research/projects/ogc/)** (Chen et al., SIGGRAPH 2025) | Contact model alignment (face blocks ⊥, radial boundaries) + the penetration-free machinery: 2-ring-excluded conservative bounds, Eq-28 warmstart truncation, counter-driven in-loop bound refresh (indirect dispatch, no CPU sync), divergent cloth-cloth log barrier |
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
| `SimCore` | Backend-neutral math, scene descriptions, geometry, mesh import, and deterministic utilities |
| `PhysicsAVBD` | Concrete AVBD CPU/Metal solver, tuned demo scenes, and shader resources; depends only on `SimCore` |
| `Robotics` | Robot models, MJCF import, calibration, and hardware-facing contracts; depends only on `SimCore` |
| `RL` | Vector task contracts, task registry, rewards, and the current AVBD-backed environments |
| `MLXRL` | MLX learning, checkpoint/evidence formats, policy runtime, and research pipelines |
| `avbd` | CLI: `run`, `bench`, `parity`, `profile` (per-kernel GPU timings), `clothgate` (gap/stretch/KE gates), `rodexp` |
| `AVBDApp` | macOS app: viewer, demos, Robotics Lab |
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
spider, and earlier native humanoid research tasks. Policy Replay is narrower:
it exposes only maintained examples with a packaged current-contract policy.
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
```

The packaged Robotics Lab launches its bundled `avbd` helper and writes local
training runs under `~/Library/Application Support/AVBD`. Set
`AVBD_WORKSPACE=/absolute/path` before launch to use a different workspace;
source-tree builds continue to use the repository working directory.

Every task returns contiguous `[environment, feature]` tensors, normalized
bounded actions, decomposed reward metrics, distinct termination/time-limit
signals, and pre-reset final observations. Checkpoints include Safetensors,
observation statistics, trainer progress, the full PPO configuration, task
timing, and JSONL training metrics. Deterministic evaluation applies task-owned
publish gates and exits nonzero when a policy stands still, falls early, misses
Push-T, or otherwise fails its contract. The tracked extension points are
[`VectorizedRLTask`](Sources/RL/RLEnvironment.swift) and
[`VectorRLAlgorithm`](Sources/MLXRL/VectorRLAlgorithm.swift); shipped
checkpoint contracts are indexed in [the checkpoint catalog](checkpoints/README.md).

The tracked **H1 Flat v2** actor is an accepted native replay on the epoch-2
deterministic-color solver contract (`taskRevision=2000011`). The registry task
ID remains `humanoid-isaac-flat-v0`; `humanoid-isaac-flat-v2` is the Policy
Replay selection and sealed bundle ID, not a different task. V2 is an exact
zero-update requalification of the immutable epoch-1 v1 bundle: policy bytes
are unchanged and no target training or optimizer steps were performed.
Four manifest-locked, fixed-seed 512-episode evaluations passed 2028/2048
episodes (99.02%); the worst run passed 98.63%, while worst-run linear and
yaw-rate RMSE were
0.089 m/s and 0.132 rad/s. The sealed bundle binds the unchanged policy, old
and new task revisions, fixed seeds, raw reports, aggregate, and producing
commit. Its fingerprint is `00bc782d...c756`. V1 remains the immutable
epoch-1 requalification source and is historical/nonselectable; the
revision-1000010 v0 bundle remains immutable older-hull evidence and is also
not selectable.

The tracked **H1 Goal** replay is also epoch-1 historical evidence and requires
requalification before replay on the current solver. Its recorded robustness
result remains:
it walks 4--8 m to a sampled point goal while one physical 8 kg box is launched
at 4--6 m/s into the full articulated body. The sealed seed-42010 test reached
400/512 goals (78.12%) and survived 80.66%; every box launched and made
physical contact. It is intentionally not labeled an accepted result yet
because unconditional final/minimum goal distances, 1.208/1.179 m, exceed the
1.125/0.750 m gates. Its current fingerprint is `15710d3f...113f6`.

The selectable **Arachne Straight Walk v1** and **Arachne Goal v1** actors are
exact zero-update requalifications on the epoch-2 deterministic-color solver
(`taskRevision=2000006`). Their task IDs remain `arachne15-velocity-v0` and
`arachne15-goal-v0`; v1 names the immutable replay bundles, not new tasks.
Each bundle seals four fixed-seed nominal reports and four full-collision
validation reports, with both suites required to pass independently and a
predeclared five-point maximum pooled-success degradation guard. Straight Walk
v1 passed 2,048/2,048 episodes in each suite. Goal v1 passed
1,942/2,048 nominal episodes (94.824%) and 1,935/2,048 validation episodes
(94.482%), a 0.342 percentage-point drop, with 100% survival in both suites.
The v0 parents and the older Arachne Goal report directory remain immutable,
nonselectable epoch-1 evidence. Goal v1 is also the exact policy/revision
boundary consumed by the fail-closed hardware deployment controller.
The selectable **Arachne Classical** mode also exposes separate **Fold** and
**Unfold & Walk** actions. Its physical motor/contact sequence reduces the
articulated footprint by 41.0%, holds it there, then deploys and hands the
measured state back to the controller. It is not a render animation and does
not make either historical native actor selectable.

The earlier native humanoids and obsolete two-link Arm policy were removed from
Policy Replay. The full Panda Push-T task remains trainable, but its best
current held-out policy does not pass the 80% task-success gate and is therefore
not shown as a solved example. Replay refuses incompatible checkpoints instead
of silently applying zero actions.

For H1 replay, the colored floor bars delimit the current velocity-command
segment: its direction is the sampled absolute heading and its length is the
distance implied by the commanded speed until the next scheduled resample.
They are visual-only and are never observations, rewards, or success inputs.
The H1 gate requires a full 20-second episode plus low linear and yaw tracking
error; merely standing upright cannot pass.

The current Flat actor's tracked multi-seed evidence is
[`checkpoints/humanoid-isaac-flat-v2/qualification/aggregate.json`](checkpoints/humanoid-isaac-flat-v2/qualification/aggregate.json);
its exact zero-update lineage is sealed by the adjacent
[`requalification-manifest.json`](checkpoints/humanoid-isaac-flat-v2/requalification-manifest.json).
The epoch-1
[`v1 parent`](checkpoints/humanoid-isaac-flat-v1/requalification-manifest.json)
and revision-1000010
[`historical evaluation`](checkpoints/humanoid-isaac-flat-v0/evaluation.json)
are retained as nonselectable lineage evidence.
The Arachne v1 lineage and both named evaluation suites are sealed by the
velocity
[`requalification manifest`](checkpoints/arachne15-velocity-v1/requalification-manifest.json)
and goal
[`requalification manifest`](checkpoints/arachne15-goal-v1/requalification-manifest.json).
Replay videos and frame-locked single-episode reports are generated outputs
under the ignored `artifacts/` directory; they are deliberately not required
by a clean clone or treated as release evidence.

## Quick start

```bash
swift test                                    # full battery
swift test --filter RLFrameworkTests          # vector RL contract + GAE
make verify-release                           # full arm64 Mac release gate
make app && open AVBD.app                     # interactive app
.build/release/avbd run bed --frames 300
.build/release/avbd profile clothfold --scale 16 --frames 80
.build/release/avbd clothgate drape --frames 300
```
