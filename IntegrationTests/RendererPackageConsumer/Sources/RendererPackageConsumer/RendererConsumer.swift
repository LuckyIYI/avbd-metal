import GPUSim
import GPUSimRenderer
import MetalKit

@main
struct RendererConsumer {
    @MainActor
    static func main() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("GPUSimRenderer consumer compiled; Metal is unavailable")
            return
        }

        var scene = PhysicsScene(name: "renderer-package-consumer")
        let body = scene.addBody(
            size: F3(repeating: 1),
            density: 1,
            friction: 0.5,
            position: F3(0, 0, 2),
            collisionEnabled: false
        )
        scene.addCollider(
            body: body,
            size: F3(repeating: 1),
            shape: .box,
            isRendered: false)
        scene.addRigidMesh(SceneRigidMesh(
            body: body,
            mesh: SurfaceMesh(
                vertices: [F3(-0.2, 0, 0), F3(0.2, 0, 0), F3(0, 0, 0.3)],
                normals: [F3(0, -1, 0), F3(0, -1, 0), F3(0, -1, 0)],
                triangles: [(0, 1, 2)]),
            color: F3(0.3, 0.4, 0.7)))

        let solver = try GPUSolver(scene: scene, device: device)
        precondition(solver.renderRigidBodyCount == 0)
        precondition(solver.rigidMeshRenderSurface != nil)
        precondition(solver.brokenJointIndices().isEmpty)
        solver.repairJoints()
        let renderer = try GPUSimRenderer(device: device, solver: solver)
        renderer.setCamera(
            position: F3(3, -3, 2), target: F3(0, 0, 1),
            resetTemporalHistory: true)
        renderer.bodyAppearances[body] = GPUSimRenderAppearance(
            color: F3(0.2, 0.7, 0.4), emissive: F3(0.1, 0.2, 0.1))
        renderer.auxiliaryInstances = [GPUSimRenderInstance(
            primitive: .torus(majorRadius: 0.4, minorRadius: 0.02),
            position: F3(0, 0, 0.02),
            color: F3(0.2, 0.8, 1),
            emissive: F3(0.1, 0.6, 1),
            opacity: 0.7)]
        let view = MTKView(frame: .zero, device: device)
        renderer.configure(view, preferredFramesPerSecond: 60)

        precondition(view.delegate === renderer)
        let configuredScene = renderer.scene
        precondition(configuredScene === solver)
        print("GPUSimRenderer consumer ready")
    }
}
