# Unitree H1 MJCF dynamics asset

`h1.xml` is the Unitree H1 model from Google DeepMind's MuJoCo Menagerie:
https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_h1

The original URDF and assets were provided by Unitree Robotics under the
BSD-3-Clause license reproduced in `LICENSE`. AVBD uses the source inertial,
joint, actuator, and primitive collision data. Visual STL meshes are omitted
because they are not used by the physics/training path.
