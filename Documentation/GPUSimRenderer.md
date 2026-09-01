# GPU Sim renderer

`GPUSimRenderer` is an optional MetalKit library product. It renders live GPU
simulation buffers with PBR lighting, a directional shadow map, temporally
accumulated GTAO, rigid primitives, soft surfaces, skinned meshes, and indexed
rigid visual meshes. It does not depend on demos, robotics, RL, MLX, SwiftUI,
or the development app.

## Minimal setup

Select both products in the consuming target:

```swift
.product(name: "GPUSim", package: "avbd-metal"),
.product(name: "GPUSimRenderer", package: "avbd-metal"),
```

Create and retain one renderer per view:

```swift
import GPUSim
import GPUSimRenderer
import MetalKit

enum AppError: Error { case metalUnavailable }

@MainActor
final class SimulationViewController {
    let view: MTKView
    let solver: GPUSolver
    let renderer: GPUSimRenderer

    init(scene: PhysicsScene) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw AppError.metalUnavailable
        }
        solver = try GPUSolver(scene: scene, device: device)
        renderer = try GPUSimRenderer(device: device, solver: solver)
        view = MTKView(frame: .zero, device: device)
        renderer.configure(view, preferredFramesPerSecond: 60)
    }
}
```

`MTKView.delegate` is weak, so the owner must retain the renderer. The
renderer and its configuration API are main-actor isolated, matching AppKit,
UIKit, and MetalKit view ownership.

## Camera and interaction

The public orbit camera consists of `target`, `distance`, `azimuth`, and
`elevation`. New scenes use camera hints from their render-scene adapter by
default. Set `automaticallyFramesScene = false` before applying a camera owned
by the application.

Use `ray(at:in:)` to turn a view-space point into a world-space ray for picking
or simulation interaction. Call `resetTemporalHistory()` after a camera
teleport; ordinary continuous orbit motion is reprojected automatically.

## Live application models

For an app that advances the simulator during rendering or swaps scenes,
implement `GPUSimRendererSource` and use `init(device:source:)`. Only
`renderScene` is required. The protocol supplies defaults for options, scene
revision, frame advance, and error reporting.
The renderer holds its `source` weakly, so the application must retain that
model as well as the renderer.

- `rendererWillDrawFrame()` runs before the renderer captures the scene.
- `rendererSceneRevision` should change only when a new scene should adopt its
  default camera.
- `rendererOptions` controls graph coloring and collision-hull diagnostics.
- `rendererDidFail(_:)` lets the host stop its loop and present the error.

Physics and rendering currently use separate Metal command queues. The AVBD
adapter synchronizes before rendering, and the renderer retires its frame
before returning, preventing either queue from mutating shared pose buffers
while the other reads them.

## Other GPU backends

`GPUSimRenderableScene` is the renderer-facing boundary. It describes compact
Metal buffers for rigid instances, soft surfaces, skinned surfaces, and rigid
meshes, plus camera hints and one instance-encoding operation. Public Swift
mirror types document the required buffer ABI:

- `GPUSimRenderInstance`
- `GPUSimSkinRenderVertex`
- `GPUSimRigidMeshRenderVertex`
- the packed corner formats on `GPUSimSoftRenderSurface` and
  `GPUSimSkinnedRenderSurface`

`GPUSolver` already conforms. A future backend can conform without exposing
its CPU-side simulation model or changing view/controller code.
All buffers must belong to the scene's `renderDevice`; the renderer rejects a
scene from a different Metal device before encoding it.
