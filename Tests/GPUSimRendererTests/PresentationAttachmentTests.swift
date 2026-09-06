import CoreGraphics
import Metal
import XCTest
@testable import GPUSimRenderer

@MainActor
final class PresentationAttachmentTests: XCTestCase {
    func testPresentationKeepsGeometryCoverageAndSSRDepthAcrossPasses() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let renderer = try GPUSimRenderer(device: device)
        let transient: MTLStorageMode = device.supportsFamily(.apple1) ? .memoryless : .private
        func destination(width: Int) throws -> MTLRenderPassDescriptor {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: GPUSimRenderer.colorFormat,
                width: width, height: 48, mipmapped: false)
            d.usage = [.renderTarget, .shaderRead]; d.storageMode = .private
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = try XCTUnwrap(device.makeTexture(descriptor: d))
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            return pass
        }
        let target = try destination(width: 64)
        let fast = try renderer.presentationPass(using: target, options: .lightweight, hasNativeOverlays: false)
        XCTAssertEqual(fast.colorAttachments[0].texture?.sampleCount, 4)
        XCTAssertEqual(fast.colorAttachments[0].texture?.storageMode, transient)
        XCTAssertTrue(fast.colorAttachments[0].resolveTexture === target.colorAttachments[0].texture)
        XCTAssertEqual(fast.depthAttachment.texture?.sampleCount, 4)
        XCTAssertEqual(fast.depthAttachment.texture?.storageMode, transient)
        XCTAssertEqual(fast.depthAttachment.storeAction, .dontCare)

        let ssrOptions = GPUSimRenderOptions(screenSpaceReflections: true)
        let ssr = try renderer.presentationPass(using: target, options: ssrOptions, hasNativeOverlays: true)
        XCTAssertEqual(ssr.depthAttachment.texture?.storageMode, .private,
            "SSR must retain native coverage depth across its separate lighting/composite passes")
        try renderer.screenSpace.prepare(size: CGSize(width: 64, height: 48), options: ssrOptions)
        let scene = renderer.screenSpace.scenePass(using: ssr)
        XCTAssertTrue(scene.depthAttachment.texture === ssr.depthAttachment.texture)
        XCTAssertEqual(scene.depthAttachment.storeAction, .storeAndMultisampleResolve)
        XCTAssertTrue(scene.depthAttachment.resolveTexture === renderer.screenSpace.sceneDepth)

        let hq = try renderer.presentationPass(using: target, options: .qualityBeta, hasNativeOverlays: false)
        XCTAssertTrue(hq.colorAttachments[0].texture === target.colorAttachments[0].texture)
        XCTAssertNil(hq.colorAttachments[0].resolveTexture)
        XCTAssertEqual(hq.colorAttachments[0].storeAction, .store)
        XCTAssertEqual(hq.depthAttachment.texture?.sampleCount, 1)
        XCTAssertEqual(hq.depthAttachment.texture?.storageMode, transient)
        XCTAssertEqual(hq.depthAttachment.loadAction, .clear)
        XCTAssertEqual(hq.depthAttachment.storeAction, .dontCare)

        let overlays = try renderer.presentationPass(using: target, options: .qualityBeta, hasNativeOverlays: true)
        XCTAssertEqual(overlays.colorAttachments[0].texture?.sampleCount, 4,
            "Native translucent/debug edges retain four-sample coverage after MetalFX")
        XCTAssertEqual(overlays.depthAttachment.texture?.sampleCount, 4)
        XCTAssertEqual(overlays.depthAttachment.texture?.storageMode, transient)
        let resized = try renderer.presentationPass(using: destination(width: 79), options: .lightweight, hasNativeOverlays: false)
        XCTAssertEqual(resized.colorAttachments[0].texture?.width, 79)
        XCTAssertEqual(resized.depthAttachment.texture?.width, 79)
        XCTAssertFalse(resized.depthAttachment.texture === overlays.depthAttachment.texture)
    }
}
