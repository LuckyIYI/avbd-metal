# Unitree H1 MJCF dynamics asset

`h1.xml` is the bundled Unitree H1 dynamics model from Google DeepMind's
MuJoCo Menagerie:
https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_h1

The original URDF and assets were provided by Unitree Robotics under the
BSD-3-Clause license reproduced in `LICENSE`. AVBD uses the source inertial,
joint, actuator, and primitive collision data. Visual STL meshes are omitted
because they are not used by the physics/training path.

The bounded ankle and torso support sets used by the three-collider H1 replay
profile are generated directly from three STL files pinned independently at
revision `71f066ad0be9cd271f7ed58c030243ef157af9f4`.
`COLLISION_HULLS_PROVENANCE.json` records their source URLs and hashes, the
dependency-free reduction contract, output hashes, and measured support error.
Regeneration is an explicit offline operation; normal builds consume the
checked-in Swift data and never download assets:

```sh
python3 Tools/generate_unitree_h1_collision_hulls.py --verify

# Full source-to-output audit when the pinned STLs are available locally.
python3 Tools/generate_unitree_h1_collision_hulls.py \
  --source-dir /path/to/pinned/unitree_h1/assets
python3 Tools/generate_unitree_h1_collision_hulls.py \
  --source-dir /path/to/pinned/unitree_h1/assets --check
```
