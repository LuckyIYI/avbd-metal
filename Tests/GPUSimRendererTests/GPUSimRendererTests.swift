import Metal
import MetalKit
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
        XCTAssertEqual(MemoryLayout<GPUSimRenderInstance>.stride, 96)
        XCTAssertEqual(MemoryLayout<GPUSimSkinRenderVertex>.stride, 32)
        XCTAssertEqual(MemoryLayout<GPUSimRigidMeshRenderVertex>.stride, 48)
    }

    func testLiveSourceDefaultsRequireOnlyAScene() {
        let source = MinimalRenderSource()
        XCTAssertNil(source.renderScene)
        XCTAssertEqual(source.rendererOptions, GPUSimRenderOptions())
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
}
