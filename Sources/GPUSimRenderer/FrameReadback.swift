import Metal

/// Textures remain checked out until the CPU callback returns, independently
/// of both GPU completion and CAMetalLayer's drawable recycling.
@MainActor
final class FrameReadback {
    enum Failure: Error { case allocation, encoder, framebufferOnly }
    private var available: [MTLTexture] = []

    func copy(_ source: MTLTexture, command: MTLCommandBuffer) throws -> MTLTexture {
        guard !source.isFramebufferOnly else { throw Failure.framebufferOnly }
        let texture: MTLTexture
        if let index = available.firstIndex(where: {
            $0.width == source.width && $0.height == source.height && $0.pixelFormat == source.pixelFormat
        }) {
            texture = available.remove(at: index)
        } else {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: source.pixelFormat,
                width: source.width, height: source.height, mipmapped: false)
            d.storageMode = .shared
            #if os(macOS)
            if !source.device.hasUnifiedMemory { d.storageMode = .managed }
            #endif
            d.usage = .shaderRead
            guard let allocated = source.device.makeTexture(descriptor: d) else { throw Failure.allocation }
            texture = allocated
            texture.label = "Completed frame readback"
        }
        guard let blit = command.makeBlitCommandEncoder() else { throw Failure.encoder }
        blit.label = "Preserve completed frame for CPU callback"
        blit.copy(from: source, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                  sourceSize: MTLSize(width: source.width, height: source.height, depth: 1),
                  to: texture, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin())
        #if os(macOS)
        if texture.storageMode == .managed { blit.synchronize(resource: texture) }
        #endif
        blit.endEncoding()
        return texture
    }

    func recycle(_ texture: MTLTexture) {
        // Bound idle storage even if the main actor was delayed for many frames.
        if available.count < 3 { available.append(texture) }
    }
}
