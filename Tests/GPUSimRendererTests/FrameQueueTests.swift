import MetalKit
import XCTest
@testable import GPUSimRenderer

@MainActor
final class FrameQueueTests: XCTestCase {
    private final class PipelinedScene: GPUSimRenderableScene {
        let renderDevice: MTLDevice
        var renderBodyCount: Int { 0 }
        var renderRigidInstanceCount: Int { 0 }
        var rendererStateIsValid: Bool { true }
        var renderSceneRequiresFrameRetirement: Bool { false }
        var renderCameraHint: GPUSimRenderCameraHint { .init() }
        var softRenderSurface: GPUSimSoftRenderSurface? { nil }
        var skinnedRenderSurface: GPUSimSkinnedRenderSurface? { nil }
        var rigidMeshRenderSurface: GPUSimRigidMeshRenderSurface? { nil }
        var convexDebugRenderSurface: GPUSimConvexDebugRenderSurface? { nil }
        var frames: [MTLCommandBuffer] = []
        var gate: MTLSharedEvent?
        var precedingQueueFrame: MTLCommandBuffer?

        init(device: MTLDevice) { renderDevice = device }

        func encodeRenderInstances(_ command: MTLCommandBuffer, instances: MTLBuffer,
                                   colorMode: GPUSimRenderColorMode, appearanceOverrides: MTLBuffer?) throws {
            // Includes topology preparation, which can write a backend's
            // surface buffers before the renderer encodes its visual frame.
            if let previous = precedingQueueFrame {
                XCTAssertEqual(previous.status, .completed, "queue transitions must retire earlier readers before refreshing scene data")
            }
            if command.label == "GPU Sim render frame" {
                if let gate { command.encodeWaitForEvent(gate, value: 1) }
                frames.append(command)
            }
        }
    }

    func testMetalFXModeSwitchAndResizeKeepSubmittingValidFrames() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              MetalFXReconstruction.supports(device: device, denoising: true) else { throw XCTSkip("MetalFX denoising unavailable") }
        let scene = PipelinedScene(device: device)
        let renderer = try GPUSimRenderer(device: device, scene: scene)
        let view = MTKView(frame: CGRect(x: 0,y: 0,width: 128,height: 96),device: device)
        renderer.configure(view); view.isPaused = true
        for (index, mode) in [GPUSimLightingMode.lightweight,.qualityBeta,.qualityBeta,.lightweight,.lightweight].enumerated() {
            renderer.options = GPUSimRenderOptions(lightingMode: mode,reconstruction: index == 4 ? .legacy : .metalFX)
            if index == 2 { view.drawableSize = CGSize(width: 160,height: 112) }
            let count = scene.frames.count
            view.draw()
            XCTAssertNil(renderer.runtimeFailure)
            XCTAssertEqual(scene.frames.count,count+1)
            scene.frames.last?.waitUntilCompleted()
            XCTAssertEqual(scene.frames.last?.status,.completed)
            XCTAssertEqual(renderer.activeReconstruction,index == 4 ? .legacy : .metalFX)
        }
    }

    func testLightingAndSceneQueueChangesRetirePipelinedFrames() throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.supportsRaytracing else {
            throw XCTSkip("Metal ray tracing is unavailable")
        }
        let renderer = try GPUSimRenderer(device: device)
        let view = MTKView(frame: CGRect(x: 0, y: 0, width: 96, height: 64), device: device)
        renderer.configure(view)
        view.isPaused = true
        let a = PipelinedScene(device: device), b = PipelinedScene(device: device)
        // Compile both worlds before arming a GPU wait, so shader compilation
        // cannot hide a missing queue dependency by delaying the second draw.
        let worlds = try [a,b].map { try RayTracingScene.shared(scene: $0) }
        defer { withExtendedLifetime(worlds) {} }
        var previous: MTLCommandBuffer?
        var previousGate: MTLSharedEvent?
        var gates: [MTLSharedEvent] = []
        defer {
            for gate in gates { gate.signaledValue = 1 }
            for frame in a.frames + b.frames {
                frame.waitUntilCompleted()
                XCTAssertEqual(frame.status, .completed, "\(String(describing: frame.error))")
            }
        }
        let draws = [(a, GPUSimRenderOptions.lightweight), (a, .lightweight), (a, .qualityBeta), (b, .qualityBeta), (b, .lightweight)]
        for (index, (scene, options)) in draws.enumerated() {
            let changesQueue = index > 1
            if let previous, let previousGate {
                XCTAssertNotEqual(previous.status, .completed, "the outgoing frame must still be in flight")
                if changesQueue {
                    scene.precedingQueueFrame = previous
                    // Release outside the main actor while draw() waits.
                    DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) { previousGate.signaledValue = 1 }
                }
            }
            let gate = try XCTUnwrap(index == 1 ? previousGate : device.makeSharedEvent())
            gates.append(gate)
            scene.gate = gate
            // Bound failure time if a regression starts retiring every frame.
            DispatchQueue.global().asyncAfter(deadline: .now() + 5) { gate.signaledValue = 1 }
            try renderer.setScene(scene)
            renderer.options = options
            let count = scene.frames.count
            view.draw()
            XCTAssertNil(renderer.runtimeFailure)
            XCTAssertEqual(scene.frames.count, count + 1, "the MTKView must submit a real frame")
            if index == 1 {
                XCTAssertNotEqual(previous?.status, .completed, "same-queue frames must remain pipelined")
            }
            previous = try XCTUnwrap(scene.frames.last)
            previousGate = gate
            scene.precedingQueueFrame = nil
        }
    }
}
