# GPU Sim package structure

The repository root is a standalone Swift package named `gpu-sim`. Its
recommended library product and module are both named `GPUSim`. Depending on
that product does not resolve MLX, build the app or CLI, or copy sample meshes.

## Public simulator package

The root [Package.swift](../Package.swift) owns four products:

- `GPUSim` is the stable, single-import facade. It re-exports the neutral scene
  API and the available solver backends.
- `SimCore` owns backend-neutral math, geometry, assets, and scene descriptions.
- `PhysicsAVBD` owns the current CPU reference and Metal production backends.
  Its resource bundle contains only required Metal shaders.
- `GPUSimDemos` is optional. It owns tuned sample scenes and all demonstration
  meshes, cooked convex fixtures, and their attribution records.

The root manifest intentionally has no package dependencies. Both
`PhysicsAVBD` and `GPUSimDemos` depend downward on `SimCore`; neither depends on
the other. A client selecting only `GPUSim` therefore compiles the scene layer,
the CPU/Metal backend, and shader resources—nothing from the demo or research
workspace.

The `SimCore` and `PhysicsAVBD` products remain available for clients that need
an explicit lower-level module boundary. New clients should prefer `GPUSim` so
future backend additions do not require changing their imports.

## Development workspace

[Development/Package.swift](../Development/Package.swift) is a separate Swift
package for code developed in this repository but not shipped as part of the
simulator dependency:

- `Robotics` depends on the root `SimCore` product.
- `RL` composes the root simulator, optional demos, and robotics.
- `MLXRL` is the only layer that depends on MLX.
- `avbd`, `AVBDApp`, and `AVBDTests` remain in this package.

Its local dependency points upward to the repository root. The root manifest
has no reference back to `Development`, preventing a dependency cycle or an
accidental MLX resolution for simulator consumers.

## Verification

Three independent checks protect the boundary:

1. `Tests/GPUSimTests` imports only `GPUSim` and exercises a complete CPU scene
   step.
2. `IntegrationTests/PackageConsumer` is a separate Swift package that depends
   on the repository root, imports only `GPUSim`, steps the CPU backend, and
   resolves the Metal backend type.
3. `Tools/verify_architecture.py` evaluates both manifests in isolated caches,
   requires zero root dependencies, verifies exact target/resource ownership,
   rejects reverse imports, and confines MLX to the development package.

Run the package gates with:

```bash
swift test
swift run --package-path IntegrationTests/PackageConsumer PackageConsumer
swift test --package-path Development
python3 Tools/verify_architecture.py
make verify-gpusim-ios
```

## Distribution notes

The manifests require Swift 5.10 and compile project code in Swift 5 language
mode. Swift has no supported declaration-level syntax that
fully re-exports another module: `public import` exposes a dependency for API
signatures but does not put its declarations in a downstream client's lookup
scope. The small `GPUSim` facade therefore deliberately uses Swift's established
`@_exported import` compatibility mechanism. The external-package compile gate
guards this behavior; all other production targets are forbidden from using it.
Supported deployment targets are macOS 14 and iOS 17.

The repository URL still gives remote dependencies the identity `avbd-metal`.
This is independent of the `GPUSim` product/module name and does not prevent
distribution. A semantic-version tag should be created when the public API is
declared release-stable; until then consumers can select `main` or an exact
revision.

Demo assets remain in the same Git repository, so they are present in a source
checkout even though SwiftPM does not build or copy them for `GPUSim` clients.
Moving those assets to a separate repository or artifact host would optimize
checkout size, but is not required for package or dependency isolation.
