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
arm Push-T, the imported 19-DoF Unitree H1 Flat velocity task aligned to Isaac
Lab, the printable 16-DoF Arachne-15 spider, and earlier native humanoid
experiments:

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

The packaged revision-10 H1 PPO checkpoint passes its documented held-out
gate. Update 300 was chosen across three validation seeds (758/768 successes),
then achieved 507/512 successes on untouched seed 28001, with 0.471 m/s
achieved versus 0.495 m/s commanded velocity, 0.089 m/s linear RMSE, 0.133
rad/s yaw-rate RMSE, and 9.38 m mean forward path. Its immutable fingerprint is
`a26559e...16fb8`. This is one training seed and therefore not yet the required
multi-training-seed algorithm claim. The older revision-35 native humanoid also
has a separately accepted checkpoint. The articulated-arm checkpoint remains
rejected because its legacy policy exposed staged controller features. Replay
refuses incompatible checkpoints instead of silently applying zero actions.

The packaged **H1 Goal** replay is the current experimental robustness best:
it walks 4--8 m to a sampled point goal while one physical 8 kg box is launched
at 4--6 m/s into the full articulated body. Across four 512-episode evaluation
seeds it reached 1,604/2,048 goals (78.32%) with 82.03% median survival; every
box launched and made physical contact. It is intentionally not labeled an
accepted result yet because the unconditional final/minimum goal-distance
medians, 1.234/1.200 m, exceed the 1.125/0.750 m gates. Select **H1 Goal** in
Policy Replay and press **Load Latest** to run fingerprint
`b6e449d...7afa`.

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
