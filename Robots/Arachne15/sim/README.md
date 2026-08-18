# Arachne-15 simulation asset pipeline

The simulation source of truth is the same measured architecture as the CAD,
but visual and collision geometry deliberately have different jobs:

- Packaged snapshots of generated printable STLs are scaled from millimetres
  to metres and used only for rendering. Ignored `build/` outputs are never a
  runtime dependency.
- Collision is an authored union of boxes and capsules per rigid
  link. No dynamic link collides as a raw triangle mesh.
- Each TPU foot uses an oriented 17.5 × 14 × 8 mm box contact matching its
  printable support envelope. This prevents tilted feet or tibia tips from
  rendering below the floor while retaining primitive-only batched contact.
- Every body has explicit mass, centre of mass, and diagonal inertia. The
  initial values come from the design mass budget and component envelopes;
  measured assembled-link values must replace them after fabrication.
- The policy interface is stable: one floating base, eight hip-yaw joints,
  eight knee-pitch joints, and sixteen motors in deterministic XML order.

After changing CAD, rebuild it to install fresh packaged visual meshes. After
changing only the model generator, regenerate the MJCF directly:

```sh
Robots/Arachne15/scripts/build_cad.sh
```

```sh
make generate-arachne-assets
```

CI and pre-commit checks should run `make verify-arachne-assets`. That target is
read-only and works in a clean clone: it fails if either generated MJCF profile
or any packaged mesh is missing, then structurally validates both source and
bundled models.

This produces and validates two MJCF files:

| Asset | Intended use | Root collision detail |
|---|---|---|
| `arachne15_training.xml` | Batched RL | ring, spine, phone, battery |
| `arachne15_validation.xml` | Replay and sim-to-real checks | training set plus hip pods, servo housings, phone guides, camera plateau, and electronics |

Both profiles contain 17 bodies, 16 joints, 16 actuators, the same mass and
inertia tensors, and the same visual mesh instances. Only non-task-critical
base collision detail changes. Feet and leg collision geometry are identical.
The generator also installs byte-identical XML and mesh resources under
`Sources/AVBDCore/Assets/arachne15`, so SwiftPM, the CLI, tests, and the app all
consume the same checked asset rather than depending on a developer CAD path.

## AVBD/MLX task

`arachne15-velocity-v0` is a normal `VectorizedRLTask`; PPO and checkpointing
need no spider-specific learner code. The policy contract is:

- observation `[60]`: body-frame linear/angular velocity, projected gravity,
  commanded planar twist, 16 encoder angles, 16 encoder velocities, and the
  16 previous applied actions;
- action `[16]`: bounded joint-position offsets in MJCF actuator order;
- 2 ms physics with decimation 10 (50 Hz policy control);
- eight physical foot contacts, no gait clock and no reference trajectory;
- command-direction progress from measured root displacement, capped at the
  commanded displacement per step and paired with velocity tracking, so
  neither standing nor overspeed can maximize locomotion reward;
- distinct termination/time-limit signals, decomposed metrics, terminal
  observations, and a task-owned held-out evaluation gate.

Large batches omit detailed CAD render buffers while retaining identical
inertia and collisions. One-to-four-environment replay includes them, and the
Policy Replay task selector exposes **Arachne-15** even before a checkpoint is
available. Run the actual task boundary with:

```sh
.build/release/avbd rl-smoke arachne15-velocity-v0 \
  --envs 128 --frames 200
```

The default training population deterministically samples mass, inertia,
friction, torque, servo gains, reflected inertia, and 0--2 control ticks of
latency. Disable the initial envelope for nominal-plant debugging with
`--task-option domainRandomization=0 --task-option maximumActionLatencySteps=0`.
Custom Swift tasks can supply any `ArticulationDomainRandomization` ranges.
See [SIM_TO_REAL.md](SIM_TO_REAL.md) for the calibration and acceptance plan.

### Random point-goal navigation

`arachne15-goal-v0` keeps the exact `[60]` observation and `[16]` action
interface. At every reset it samples a uniformly random bearing and a target
distance in `0.60...2.0 m`. A task-owned command provider continuously converts
the world target into a body-frame planar-velocity/yaw command, slows near the
target, and commands zero inside the `0.12 m` acceptance radius. Success
requires remaining inside that radius below the arrival-speed limit for 15
control steps; merely crossing the marker at speed is not success.

Replay scenes render the start as a flat square and the destination as a
beacon sphere plus an acceptance-radius ring. Both are static, non-colliding
visualization primitives. Their poses
never enter policy observations, contacts, or reward computation, and large
training batches omit them completely. This prevents privileged target data
from leaking into the motor policy. A future iPhone camera/gesture module can
replace the command provider with an estimated goal while preserving the same
locomotion and actuator interfaces.

```sh
.build/release/avbd rl-smoke arachne15-goal-v0 \
  --envs 128 --frames 200
```

After a velocity run passes evaluation, reuse its locomotion network explicitly
instead of relearning contact dynamics from scratch (the hidden-layer layout
must match):

```sh
.xcbuild/Build/Products/Release/avbd train-rl arachne15-goal-v0 \
  --run arachne-goal-seed-1 --seed 1 --envs 2048 --updates 500 \
  --horizon 32 --epochs 5 --batch 16384 --hidden-layers 256,256,128 \
  --initialize-from runs/arachne15-velocity-v0/arachne-velocity-seed-1
```

Policy Replay exposes **Arachne Goal**. **Sample Task Goal** draws another
seeded task target; bearing and distance controls install a deterministic
target for diagnosis. The UI never supplies joint targets.

### Physical fold and reveal

Every Arachne Policy Replay scene exposes separate **Fold** and
**Unfold & Walk** actions. Fold resets to the authored assembly, sends the
folding portion of a bounded commissioning trajectory to the same sixteen
torque-limited motors used for walking, and physically holds the compact pose.
Unfold & Walk resumes at the deployment phase. The renderer never
edits the root or link poses. Folding uses two balanced diagonal
lift/sweep/plant waves; unfolding uses a shared-load knee crouch and the
complementary waves so no final two-leg pair has to raise the iPhone and
chassis alone. When the
sequence completes, the normal 20%-stall walking effort budget is restored and
the selected learned or classical controller receives the measured physical
observation.

The compact targets are outside the learned action envelope but inside the
authored MJCF reserve limits. Therefore this feature does not alter task revision 6,
the deployed collision model, or the checkpoint tensor contract. The exact
pre-check is part of `make verify-arachne-assets`; focused Swift tests require
the compact pose to be reached by motor/contact dynamics, remain upright, stay
near its start point, return to the loaded neutral stance, and survive the
first post-commissioning task step.

### Classical CPG + IK baseline

**Arachne Classical** runs without MLX or a checkpoint. A four-phase paired
ripple oscillator lifts two diagonally separated legs while six legs execute a
powered stance sweep. Exact yaw/pitch leg IK converts each foot trajectory into
the same bounded 16 joint-position targets used by RL. Those targets go through
the same randomized latency, servo gains, and torque limits; the body moves
only through AVBD contact and friction. There is no root translation,
visual-only animation, reference replay, or policy inference.

Run the reproducible batched evaluator:

```sh
.build/release/avbd eval-arachne-classical --envs 32 --seed 21001 \
  --output runs/arachne15-classical/seed-21001.json
```

The default controller uses a 16-tick cycle at 50 Hz, 16 mm swing clearance, a
350 ms half-stride horizon, and a 45 mm planar placement bound. The CLI exposes
`--gait-swing-steps`, `--gait-swing-height`,
`--gait-placement-horizon`, and `--gait-maximum-placement` for explicit
experiments. Evaluation commands write ignored run output; the tracked
2026-07-16 two-seed snapshot under `Robots/Arachne15/baselines/classical/`
reached 56/64 randomized goals (87.5%; seed rates 81.25% and 93.75%), survived
64/64 episodes, and measured 0.28 mm mean foot-penetration RMSE. This is a
useful baseline, not a deployment qualification: the immutable learned
controller reaches about 95% on thousands of held-out episodes and remains the
stronger candidate.

Tests cover exact FK↔IK round trips and require a deterministic front goal to
be physically reached. The latter fails if joint targets animate without
producing body displacement through contacts.

## Why primitives instead of decomposed printable meshes?

This is the standard robust robot path, rather than a limitation of the CAD.
Isaac Sim's current URDF/MJCF converters default to a convex hull when they are
asked to synthesize collision geometry from visuals. Isaac Lab's generic mesh
converter defaults to convex decomposition. Both support explicit collision
geometry, which is preferable here because it is deterministic, inspectable,
cheap to batch, and consistent between engines.

MuJoCo likewise treats an ordinary mesh collider as its convex hull; concave
objects are represented as unions of convex geoms. A printable ring or slotted
link therefore cannot be used directly without either filling its holes or
decomposing it. The authored compounds preserve the important external shape
without coupling contacts to a decomposition tool/version.

SDF collision is useful for a genuinely non-convex dynamic object whose exact
surface interaction is task-critical. It is not the default choice for robot
links: it adds cooking/memory cost, is harder to keep identical across engines,
and provides little value for Arachne's mostly box/capsule geometry. Static
terrain should use native planes, height fields, or static triangle meshes.

## Fidelity ladder

1. Train with explicit primitive colliders and self-collision disabled.
2. Replay checkpoints with the validation profile and selected self-collision.
3. Randomize foot friction, payload/COM, joint damping, motor strength, latency,
   battery voltage, and terrain—not mesh tessellation.
4. Identify actuator and friction parameters from a one-leg physical rig.
5. Compare the same command/state logs in hardware and simulation, then update
   the measured mass, inertia, torque-speed, backlash, and delay parameters.

Primary references checked 2026-07-15:

- NVIDIA Isaac Sim URDF importer: https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html
- Isaac Lab converter configuration: https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/api/lab/isaaclab.sim.converters.html
- NVIDIA PhysX rigid-body collision: https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/docs/RigidBodyCollision.html
- MuJoCo model overview: https://mujoco.readthedocs.io/en/stable/overview.html
