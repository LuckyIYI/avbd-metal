import Metal
import XCTest
@testable import GPUSimRenderer

@MainActor
final class FrameReadbackTests: XCTestCase {
    func testDelayedCallbacksKeepTheirOwnPixelsBeyondDrawablePoolDepth() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 4, height: 4, mipmapped: false)
        d.storageMode = .private; d.usage = .renderTarget
        let drawableSurface = try XCTUnwrap(device.makeTexture(descriptor: d))
        let readback = FrameReadback()
        var delayed: [MTLTexture] = []
        for frame in 0..<8 {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawableSurface
            pass.colorAttachments[0].loadAction = .clear; pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(red: Double(frame)/255, green: 0, blue: 0, alpha: 1)
            try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass)).endEncoding()
            delayed.append(try readback.copy(drawableSurface, command: command))
            command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status, .completed)
        }
        for (frame, texture) in delayed.enumerated() {
            var pixel: UInt32 = 0
            texture.getBytes(&pixel, bytesPerRow: 4, from: MTLRegionMake2D(0,0,1,1), mipmapLevel: 0)
            XCTAssertEqual(pixel & 255, UInt32(frame), "Later rendering must not overwrite a delayed callback's pixels")
            readback.recycle(texture)
        }
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let reused = try readback.copy(drawableSurface, command: command)
        XCTAssertTrue(delayed.contains { $0 === reused }, "Completed callbacks should release textures for reuse")
        command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed)

        // Fill all idle slots at the old size, then resize. The new size
        // must reuse its completed texture rather than allocate every frame.
        readback.recycle(reused)
        d.width = 8
        let resizedSurface = try XCTUnwrap(device.makeTexture(descriptor: d))
        let resizeCommand = try XCTUnwrap(queue.makeCommandBuffer())
        let resized = try readback.copy(resizedSurface, command: resizeCommand)
        resizeCommand.commit(); resizeCommand.waitUntilCompleted()
        XCTAssertEqual(resizeCommand.status, .completed)
        readback.recycle(resized)
        let nextCommand = try XCTUnwrap(queue.makeCommandBuffer())
        let next = try readback.copy(resizedSurface, command: nextCommand)
        XCTAssertTrue(next === resized, "Resizing must not turn pooled capture into a per-frame allocation")
        nextCommand.commit(); nextCommand.waitUntilCompleted()
        XCTAssertEqual(nextCommand.status, .completed)
    }
}
