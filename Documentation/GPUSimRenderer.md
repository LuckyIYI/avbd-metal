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
default.

For a camera owned by the application, supply a look-at pose. This supports
wrist, chase, and egocentric cameras and disables automatic scene framing:

```swift
renderer.setCamera(
    position: wristCameraPosition,
    target: wristCameraTarget,
    up: wristCameraUp)
```

`setCamera(viewMatrix:focusDistance:)` accepts an existing world-to-view
matrix. Camera setters preserve temporal history by default so a continuously
moving camera can reproject normally; pass `resetTemporalHistory: true` for a
cut or teleport. `useOrbitCamera()` returns control to the orbit properties
and resets temporal history by default.

`configure(_:)` connects rendering to an `MTKView`; it deliberately does not
install platform gestures. The host maps its own mouse, trackpad, touch, or
controller input onto the orbit properties. A typical drag/zoom mapping is:

```swift
renderer.azimuth -= dragDeltaX * 0.008
renderer.elevation = min(max(
    renderer.elevation + dragDeltaY * 0.008, -1.5), 1.55)
renderer.distance = min(max(
    renderer.distance * (1 - scrollDeltaY * 0.02), 2), 400)
```

Use `ray(at:in:)` to turn a view-space point into a world-space ray for picking
or simulation interaction.

## Live appearance and auxiliary geometry

Per-body presentation is keyed by the same stable body indices returned by
scene authoring. A color replaces the authored collider or visual-mesh color;
emissive values are linear HDR radiance and may exceed one:

```swift
renderer.bodyAppearances[gripperBody] = GPUSimRenderAppearance(
    color: F3(0.2, 0.85, 0.35),
    emissive: F3(0.0, 0.8, 0.1))
```

Overrides apply to analytic rigid geometry, rigid visual meshes, and soft
surface vertices. Skinned meshes do not have a single body identity after
deformation and retain their authored material.

`GPUSimRenderAppearance` changes material presentation only; it does not hide
a body, and a missing color means "keep the authored color." Hide an analytic
collision proxy at scene-authoring time with `SceneCollider.isRendered = false`
or the `isRendered: false` argument on collider builders. Collision remains
enabled.

Guides, targets, ghost poses, and other app-owned world geometry use the
auxiliary pass. It is depth-tested against the scene and supports alpha plus
emission:

```swift
renderer.auxiliaryInstances = [GPUSimRenderInstance(
    primitive: .torus(majorRadius: 0.18, minorRadius: 0.008),
    position: dropTarget,
    color: F3(0.1, 0.8, 1.0),
    emissive: F3(0.0, 0.5, 1.5),
    opacity: 0.65)]
```

The built-in primitives are boxes, spheres, tori, and capsules. Instances with
opacity 1 write depth and therefore form correctly occluding multi-part solid
geometry. Translucent instances are sorted back-to-front from the active
camera, alpha blended, and do not write depth. Auxiliary geometry does not
participate in shadows or GTAO.

The pass is a deliberately bounded extension point: it preserves renderer
ownership of its pipelines, uniforms, attachments, and synchronization instead
of exposing a raw encoder whose contract would be easy to break.

## Rigid visual-mesh authoring

Rigid visual meshes are a supported public scene-authoring path, not an
importer-only implementation detail. Attach a `SurfaceMesh` to a body before
constructing the solver; it follows that body's live GPU pose but contributes
no collision shape or mass:

```swift
let visual = SurfaceMesh(
    vertices: vertices,
    normals: normals,
    triangles: triangles)
scene.addRigidMesh(SceneRigidMesh(
    body: armLink,
    mesh: visual,
    localPosition: meshOffset,
    localRotation: meshRotation,
    color: F3(0.24, 0.28, 0.34)))
```

Author collision geometry separately on the body or its colliders. Set
`isRendered: false` when adding those colliders if the visual mesh should
replace, rather than accompany, their analytic proxy:

```swift
scene.addCollider(
    body: armLink,
    size: collisionSize,
    friction: 0.7,
    shape: .capsule,
    isRendered: false)
```

Mesh vertices and normals are body-local; `localPosition` and
`localRotation` place the source mesh in that body frame. This is the intended
path for detailed CAD or a slim visual skeleton over simpler, hidden collision
proxies.

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
- `rendererBodyAppearances` supplies per-frame body color/emission overrides.
- `rendererAuxiliaryInstances` supplies per-frame app-owned world geometry.
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
- `GPUSimRenderAppearance`
- `GPUSimSkinRenderVertex`
- `GPUSimRigidMeshRenderVertex`
- the packed corner formats on `GPUSimSoftRenderSurface` and
  `GPUSimSkinnedRenderSurface`

`GPUSolver` already conforms. A future backend can conform without exposing
its CPU-side simulation model or changing view/controller code.
`renderBodyCount` defines the valid appearance index range, and
`encodeRenderInstances` receives the optional 32-byte-per-body appearance
buffer. Backends copy those overrides into their public 112-byte instance ABI;
rigid mesh shaders consume the same buffer directly.
All buffers must belong to the scene's `renderDevice`; the renderer rejects a
scene from a different Metal device before encoding it.
