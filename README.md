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

## Robot learning

The vector RL path is task-agnostic and runs the simulator and MLX learner on
Apple silicon. Built-in tasks include Cartesian Push-T, randomized articulated
arm Push-T, the full seven-axis Panda port of ManiSkill PushT-v1, the imported
19-DoF Unitree H1 tasks aligned to Isaac Lab, the printable 16-DoF Arachne-15
spider, and earlier native humanoid research tasks. Policy Replay is narrower:
it exposes only maintained examples with a packaged current-contract policy.

```bash
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

# Replay the identical checkpoint/task path in Metal UI. H1 Flat Walk is the
# default Policy Replay task and executes only its packaged Safetensors actor.
make app-ml
AVBD_POLICY_REPLAY=1 open AVBD.app
```

Every task returns contiguous `[environment, feature]` tensors, normalized
bounded actions, decomposed reward metrics, distinct termination/time-limit
signals, and pre-reset final observations. Checkpoints include Safetensors,
observation statistics, trainer progress, the full PPO configuration, task
timing, and JSONL training metrics. Deterministic evaluation applies task-owned
publish gates and exits nonzero when a policy stands still, falls early, misses
Push-T, or otherwise fails its contract. See
[the RL architecture and research notes](docs/RL_ARCHITECTURE.md).

The packaged H1 Flat actor has been transferred onto the corrected fixed-gain
actuator contract (`taskRevision=1000010`) and passes its documented held-out
gate. A sealed 512-episode test at seed 41010 achieved 507/512 successes
(99.02%), 0.089 m/s linear RMSE, and 0.134 rad/s yaw-rate RMSE. Its immutable
fingerprint is `d6b5d416...e777ab`. This transfer descends from one training
seed and is therefore not a multi-training-seed algorithm claim.

The packaged **H1 Goal** replay is also on the corrected actuator contract and
is the current experimental robustness best:
it walks 4--8 m to a sampled point goal while one physical 8 kg box is launched
at 4--6 m/s into the full articulated body. The sealed seed-42010 test reached
400/512 goals (78.12%) and survived 80.66%; every box launched and made
physical contact. It is intentionally not labeled an accepted result yet
because unconditional final/minimum goal distances, 1.208/1.179 m, exceed the
1.125/0.750 m gates. Its current fingerprint is `15710d3f...113f6`.

The packaged **Arachne Straight Walk** actor is an accepted deterministic
0.15 m/s regression benchmark on the corrected revision-6 foot collider. Its
sealed seed-45010 test passed 512/512 episodes, with 0.068 m/s linear and 0.284
rad/s yaw-rate RMSE and no control steps deeper than 1 mm below the floor.
**Arachne Goal** remains the randomized, arbitrary-direction sim-to-real policy;
its multi-seed qualification reports are shipped beside the robot assets.

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

A frame-locked replay of packaged update 300 at seed 21001 is under
`artifacts/replays/humanoid-isaac-flat-v0`. It contains 503 frames at 25 fps
(20.12 seconds) and the matching deterministic one-episode report. That exact
episode completed all 1,000 controls, traveled 12.78 m, and passed its tracking
gate; `h1-flat-update300-seed21001-replay.json` records the linkage between
video, seed, checkpoint fingerprint, and metrics.

## Quick start

```bash
swift test                                    # full battery
swift test --filter RLFrameworkTests          # vector RL contract + GAE
make app && open AVBD.app                     # interactive app
.build/release/avbd run bed --frames 300
.build/release/avbd profile clothfold --scale 16 --frames 80
.build/release/avbd clothgate drape --frames 300
```
