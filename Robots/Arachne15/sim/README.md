# Arachne-15 simulation asset pipeline

The simulation source of truth is the same measured architecture as the CAD,
but visual and collision geometry deliberately have different jobs:

- Generated printable STLs are scaled from millimetres to metres and used only
  for rendering.
- Collision is an authored union of boxes, capsules, and spheres per rigid
  link. No dynamic link collides as a raw triangle mesh.
- Every body has explicit mass, centre of mass, and diagonal inertia. The
  initial values come from the design mass budget and component envelopes;
  measured assembled-link values must replace them after fabrication.
- The policy interface is stable: one floating base, eight hip-yaw joints,
  eight knee-pitch joints, and sixteen motors in deterministic XML order.

Run:

```sh
../scripts/build_sim.sh
```

This produces and validates two MJCF files:

| Asset | Intended use | Root collision detail |
|---|---|---|
| `arachne15_training.xml` | Batched RL | ring, spine, phone, battery |
| `arachne15_validation.xml` | Replay and sim-to-real checks | training set plus hip pods, servo housings, phone guides, camera plateau, and electronics |

Both profiles contain 17 bodies, 16 joints, 16 actuators, the same mass and
inertia tensors, and the same visual mesh instances. Only non-task-critical
base collision detail changes. Feet and leg collision geometry are identical.

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
