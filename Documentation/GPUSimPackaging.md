# GPU Sim package extraction

`GPUSim` is the backend-neutral package entry point. It currently combines the
scene model in `SimCore` with the CPU and Metal implementations in
`PhysicsAVBD`, so clients can start with one dependency and one import while
the backend boundary continues to evolve.

## Current boundary

- `GPUSim`: stable consumer-facing facade.
- `SimCore`: backend-neutral math, geometry, and scene descriptions.
- `PhysicsAVBD`: current CPU reference and Metal production backends, plus the
  Metal shader resources they require.
- `Robotics`, `RL`, `MLXRL`, the CLI, and the app remain higher-level sibling
  targets and are not re-exported by `GPUSim`.

The existing `SimCore` and `PhysicsAVBD` products remain available as
compatibility and advanced-use entry points. New simulator consumers should
prefer `GPUSim` so adding another backend does not require changing imports.

## Remaining work before an independent release

1. Move demo scenes and their large sample assets out of `PhysicsAVBD` into an
   optional `GPUSimDemos` product. Solver clients should receive shaders, but
   not demonstration meshes.
2. Move the app, CLI, robotics, RL, and MLX integration into a workspace or a
   second package. SwiftPM resolves all dependencies declared by a manifest,
   so this is what will let simulator-only consumers avoid resolving MLX.
3. Replace the underscored re-export compatibility shim with explicit public
   API ownership when the project adopts a Swift toolchain with supported
   access-level imports.
4. Add a tiny external consumer fixture to CI and establish semantic-versioning
   policy for the `GPUSim` API.
5. Publish from a repository whose package identity is `gpu-sim`; until the
   repository is renamed, downstream manifests use the current repository URL
   while importing the `GPUSim` product.
