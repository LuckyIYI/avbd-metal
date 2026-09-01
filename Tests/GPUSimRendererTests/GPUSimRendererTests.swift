import Metal
import MetalKit
import SimCore
import PhysicsAVBD
import XCTest
@testable import GPUSimRenderer

@MainActor
private final class MinimalRenderSource: GPUSimRendererSource {
    var renderScene: (any GPUSimRenderableScene)?
}

@MainActor
final class GPUSimRendererTests: XCTestCase {
    func testDefaultOptionsAreMinimal() {
        XCTAssertEqual(GPUSimRenderOptions(), GPUSimRenderOptions(
            colorMode: .bodyIndex,
            showConvexCollisionGeometry: false,
            convexCollisionWireframe: true
        ))
        XCTAssertEqual(MemoryLayout<GPUSimRenderInstance>.stride, 112)
        XCTAssertEqual(MemoryLayout<GPUSimRenderAppearance>.stride, 32)
        XCTAssertEqual(MemoryLayout<GPUSimSkinRenderVertex>.stride, 32)
        XCTAssertEqual(MemoryLayout<GPUSimRigidMeshRenderVertex>.stride, 48)
    }

    func testLiveSourceDefaultsRequireOnlyAScene() {
        let source = MinimalRenderSource()
        XCTAssertNil(source.renderScene)
        XCTAssertEqual(source.rendererOptions, GPUSimRenderOptions())
        XCTAssertEqual(source.rendererBodyAppearances, [:])
        XCTAssertTrue(source.rendererAuxiliaryInstances.isEmpty)
        XCTAssertEqual(source.rendererSceneRevision, 0)
        source.rendererWillDrawFrame()
        source.rendererDidFail("ignored by default")
    }

    func testRendererConfiguresRequiredViewFormats() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is unavailable")
        }
        let renderer = try GPUSimRenderer(device: device)
        let view = MTKView(frame: .zero, device: device)
        renderer.configure(view, preferredFramesPerSecond: 45)

        XCTAssertEqual(view.colorPixelFormat, GPUSimRenderer.colorFormat)
        XCTAssertEqual(view.depthStencilPixelFormat, .depth32Float)
        XCTAssertEqual(view.sampleCount, GPUSimRenderer.sampleCount)
        XCTAssertEqual(view.preferredFramesPerSecond, 45)
        XCTAssertTrue(view.delegate === renderer)
    }

    func testCenterRayIsFiniteAndForwardFacing() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is unavailable")
        }
        let renderer = try GPUSimRenderer(device: device)
        let ray = renderer.ray(
            at: CGPoint(x: 320, y: 180),
            in: CGSize(width: 640, height: 360)
        )

        XCTAssertTrue(ray.origin.x.isFinite)
        XCTAssertTrue(ray.origin.y.isFinite)
        XCTAssertTrue(ray.origin.z.isFinite)
        XCTAssertGreaterThan(dot(ray.dir, renderer.target - renderer.eyePosition), 0)
    }

    func testApplicationCameraPoseAndMatrixOverride() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is unavailable")
        }
        let renderer = try GPUSimRenderer(device: device)
        let position = F3(1, -2, 3)
        let target = F3(1.5, 0, 2.5)
        renderer.setCamera(position: position, target: target)

        XCTAssertFalse(renderer.automaticallyFramesScene)
        XCTAssertEqual(renderer.cameraPose?.position, position)
        XCTAssertEqual(renderer.cameraPose?.target, target)
        XCTAssertEqual(renderer.eyePosition, position)

        let view = renderer.viewMatrix
        renderer.setCamera(viewMatrix: view, focusDistance: length(target - position))
        XCTAssertEqual(renderer.eyePosition.x, position.x, accuracy: 1e-5)
        XCTAssertEqual(renderer.eyePosition.y, position.y, accuracy: 1e-5)
        XCTAssertEqual(renderer.eyePosition.z, position.z, accuracy: 1e-5)
        renderer.useOrbitCamera()
        XCTAssertNil(renderer.cameraPose)
    }

    func testAuxiliaryPrimitiveBuilderCarriesAlphaAndEmission() {
        let instance = GPUSimRenderInstance(
            primitive: .sphere(radius: 0.25),
            position: F3(1, 2, 3),
            color: F3(0.2, 0.4, 0.8),
            emissive: F3(2, 1, 0),
            opacity: 0.35)

        XCTAssertEqual(instance.color, SIMD4(0.2, 0.4, 0.8, 1))
        XCTAssertEqual(instance.parameters.z, 0.25)
        XCTAssertEqual(instance.material, SIMD4(2, 1, 0, 0.35))
        XCTAssertEqual(instance.model.columns.3, SIMD4(1, 2, 3, 1))
    }

    func testAuxiliaryRenderOrderPartitionsOpaqueAndSortsTranslucent() {
        func instance(
            _ primitive: GPUSimRenderPrimitive,
            z: Float,
            opacity: Float
        ) -> GPUSimRenderInstance {
            GPUSimRenderInstance(
                primitive: primitive,
                position: F3(0, 0, z),
                color: F3(repeating: 0.5),
                opacity: opacity)
        }

        let order = GPUSimRenderer.auxiliaryRenderOrder([
            instance(.sphere(radius: 1), z: 1, opacity: 0.5),
            instance(.box(size: F3(repeating: 1)), z: 2, opacity: 1),
            instance(.capsule(length: 1, radius: 0.1), z: 9, opacity: 0),
            instance(.torus(majorRadius: 1, minorRadius: 0.1),
                     z: 5, opacity: 0.5),
        ], viewedFrom: .zero)

        XCTAssertEqual(order.opaque.map(\.color.w), [0])
        XCTAssertEqual(order.translucent.map(\.color.w), [2, 1],
                       "translucent primitive types must retain global far-to-near order")
    }

    func testSolverEncodesBodyAppearanceIntoPublicInstanceABI() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is unavailable")
        }
        var scene = PhysicsScene(name: "render-appearance")
        _ = scene.addBody(
            size: F3(repeating: 0.2), density: 1_000, friction: 0,
            position: F3(0, 0, 1))
        let solver = try GPUSolver(scene: scene, device: device)
        guard let queue = device.makeCommandQueue(),
              let command = queue.makeCommandBuffer(),
              let instances = device.makeBuffer(
                length: MemoryLayout<GPUSimRenderInstance>.stride,
                options: .storageModeShared),
              let appearances = device.makeBuffer(
                length: MemoryLayout<GPUSimRenderAppearance>.stride,
                options: .storageModeShared)
        else { return XCTFail("could not allocate Metal test resources") }
        appearances.contents().bindMemory(
            to: GPUSimRenderAppearance.self, capacity: 1).pointee =
                GPUSimRenderAppearance(
                    color: F3(0.1, 0.3, 0.7),
                    emissive: F3(3, 2, 1))

        try solver.encodeRenderInstances(
            command, instances: instances, colorMode: .bodyIndex,
            appearanceOverrides: appearances)
        command.commit()
        command.waitUntilCompleted()

        XCTAssertEqual(command.status, .completed)
        let instance = instances.contents().bindMemory(
            to: GPUSimRenderInstance.self, capacity: 1).pointee
        XCTAssertEqual(instance.color, SIMD4(0.1, 0.3, 0.7, 0))
        XCTAssertEqual(instance.material, SIMD4(3, 2, 1, 1))
    }
}
