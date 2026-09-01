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
        scene.addBody(
            size: F3(repeating: 1),
            density: 1,
            friction: 0.5,
            position: F3(0, 0, 2)
        )

        let solver = try GPUSolver(scene: scene, device: device)
        let renderer = try GPUSimRenderer(device: device, solver: solver)
        let view = MTKView(frame: .zero, device: device)
        renderer.configure(view, preferredFramesPerSecond: 60)

        precondition(view.delegate === renderer)
        let configuredScene = renderer.scene
        precondition(configuredScene === solver)
        print("GPUSimRenderer consumer ready")
    }
}
