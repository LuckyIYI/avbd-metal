import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class VisibilityFilterTests: XCTestCase {
    func testDirectOnlyFilterPreservesVisibilityAtSurfaceBoundaries() throws {
        try compareSpecialization(ambientOnly: false)
    }

    func testAmbientOnlyFilterPreservesEveryR16ValueAtSurfaceBoundaries() throws {
        try compareSpecialization(ambientOnly: true)
    }

    func testFilterAlgebraPreservesRotatedCurvedSurfaces() throws {
        for angle in [Float(0.4), 1.27, 2.31] {
            try compareSpecialization(ambientOnly: false, referenceRotation: angle)
        }
    }

    // Reuse the shipped tap traversal, changing only the independently
    // equivalent expressions being checked: rotate both normals before dot,
    // and evaluate the two nonnegative square roots separately.
    private func viewSpaceReferenceSource() throws -> String {
        let begin = try XCTUnwrap(renderShaderSource.range(of: "template <bool wantAmbient, bool wantDirect>"))
        let end = try XCTUnwrap(renderShaderSource.range(of: "inline float3 reflectionModulation", range: begin.upperBound..<renderShaderSource.endIndex))
        let source = String(renderShaderSource[begin.lowerBound..<end.lowerBound])
            .replacingOccurrences(of: "filterVisibility", with: "referenceFilterVisibility")
            .replacingOccurrences(of: "visibility_fragment", with: "reference_visibility_fragment")
            .replacingOccurrences(of: "QNW = normal.read(uint2(q)).xyz", with: "QN = screenVector(normal.read(uint2(q)).xyz, U)")
            .replacingOccurrences(of: "dot(NW, QNW)", with: "dot(N, QN)")
            .replacingOccurrences(of: "0.5 * sqrt(dot(tangent, tangent) * (1.0 - agreement))",
                                  with: "length(tangent) * 0.5 * sqrt(1.0 - agreement)")
        XCTAssertTrue(source.contains("QN = screenVector("))
        XCTAssertTrue(source.contains("length(tangent) * 0.5 * sqrt(1.0 - agreement)"))
        XCTAssertFalse(source.contains("QNW"), "The oracle must rotate both normals instead of testing the same expression")
        return source
    }

    private func compareSpecialization(ambientOnly: Bool, referenceRotation: Float? = nil) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let rotation = simd_float3x3(simd_quatf(angle: referenceRotation ?? 0, axis: normalize(SIMD3<Float>(0.3,1,0.6))))
        let forward = normalize(rotation * SIMD3<Float>(0,0,-1))
        let right = normalize(cross(forward, rotation * SIMD3<Float>(0,1,0)))
        let down = cross(right, forward) * -1
        func vector(_ v: SIMD3<Float>) -> String { "float3(\(v.x),\(v.y),\(v.z))" }
        let curvedNormal = """
            float3 N = normalize(float3(0.2 * sin(float(p.x) * 0.13), 0.3 * cos(float(p.y) * 0.17), 1));
            if (p.x > 131u) N = normalize(float3(0.6,0.2,1));
            o.normal.xyz = N.x * \(vector(right)) + N.y * \(vector(down)) + N.z * \(vector(cross(right,down)));
        """
        let reference = try referenceRotation == nil ? "" : viewSpaceReferenceSource()
        let library = try device.makeLibrary(source: renderShaderSource + reference + """
        struct DirectFilterFixture {
            float4 normal [[color(0)]];
            float contact [[color(1)]];
            float depth [[depth(any)]];
        };
        fragment DirectFilterFixture direct_filter_fixture(FSOut in [[stage_in]]) {
            uint2 p = uint2(in.position.xy);
            DirectFilterFixture o;
            o.normal = float4(normalize(p.x > 31u ? float3(0.6,0.2,1) : float3(0,0,1)), 0.5);
            \(referenceRotation == nil ? "" : curvedNormal)
            float z = (p.x > 31u ? 2.4 : 2.0) + float(p.y) * 0.005;
            o.depth = p.x < 3u || p.y > \(referenceRotation == nil ? 42 : 190)u ? 1.0 : 1.0 - 0.1 / z;
            o.contact = float((p.x * 29u + p.y * 17u) & 255u) / 255.0;
            return o;
        }
        """, options: nil)
        let width = referenceRotation == nil ? 63 : 253, height = referenceRotation == nil ? 47 : 197
        let outputFormat: MTLPixelFormat = ambientOnly ? .r16Float : .rg8Unorm
        func texture(_ format: MTLPixelFormat) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
            d.usage = [.renderTarget, .shaderRead]; d.storageMode = .private
            return try XCTUnwrap(device.makeTexture(descriptor: d))
        }
        func pipeline(_ name: String, fixture: Bool = false) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = library.makeFunction(name: "fs_vertex")
            d.fragmentFunction = library.makeFunction(name: name)
            d.colorAttachments[0].pixelFormat = fixture ? .rgba16Float : outputFormat
            if fixture {
                d.colorAttachments[1].pixelFormat = .r8Unorm
                d.depthAttachmentPixelFormat = .depth32Float
            }
            return try device.makeRenderPipelineState(descriptor: d)
        }
        let normal = try texture(.rgba16Float), contact = try texture(.r8Unorm)
        let depth = try texture(.depth32Float), ambient = try texture(.rg8Unorm)
        let original = try texture(outputFormat), specialized = try texture(outputFormat)
        let originalPipeline = try pipeline(referenceRotation == nil ? "visibility_fragment" : "reference_visibility_fragment")
        let specializedPipeline = try pipeline(referenceRotation == nil
            ? (ambientOnly ? "ambient_visibility_fragment" : "direct_visibility_fragment") : "visibility_fragment")
        let fixturePipeline = try pipeline("direct_filter_fixture", fixture: true)
        let state = MTLDepthStencilDescriptor(); state.depthCompareFunction = .always; state.isDepthWriteEnabled = true
        let depthState = try XCTUnwrap(device.makeDepthStencilState(descriptor: state))
        let queue = try XCTUnwrap(device.makeCommandQueue()), command = try XCTUnwrap(queue.makeCommandBuffer())
        let clear = MTLRenderPassDescriptor()
        clear.colorAttachments[0].texture = ambient; clear.colorAttachments[0].loadAction = .clear
        clear.colorAttachments[0].storeAction = .store
        clear.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)
        let clearEncoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: clear)); clearEncoder.endEncoding()
        let guides = MTLRenderPassDescriptor()
        for (i, t) in [normal, contact].enumerated() {
            guides.colorAttachments[i].texture = t; guides.colorAttachments[i].loadAction = .dontCare
            guides.colorAttachments[i].storeAction = .store
        }
        guides.depthAttachment.texture = depth; guides.depthAttachment.loadAction = .dontCare
        guides.depthAttachment.storeAction = .store
        let guideEncoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: guides))
        guideEncoder.setRenderPipelineState(fixturePipeline); guideEncoder.setDepthStencilState(depthState)
        guideEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3); guideEncoder.endEncoding()
        var u = Uniforms(viewProj: matrix_identity_float4x4, lightDir: SIMD4(0,0,1,0), eye: .zero,
            screen: SIMD4(Float(width),Float(height),Float(height),0), camRight: SIMD4(right,0), camUp: SIMD4(down,0),
            prevViewProj: matrix_identity_float4x4, temporal: .zero, shadowViewProj: matrix_identity_float4x4,
            shadowParams: .zero, invViewProj: matrix_identity_float4x4, prevInvViewProj: matrix_identity_float4x4,
            effects: SIMD4(1,ambientOnly ? 0 : 0.2,0,0.65), rayTracing: SIMD4(1,0,0,1), aoProjection: SIMD4(1,-0.1,1,1))
        for (target, filter) in [(original, originalPipeline), (specialized, specializedPipeline)] {
            let pass = MTLRenderPassDescriptor(); pass.colorAttachments[0].texture = target
            pass.colorAttachments[0].loadAction = .dontCare; pass.colorAttachments[0].storeAction = .store
            let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass))
            encoder.setRenderPipelineState(filter)
            encoder.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            // The AO-only fixture uses the varying R8 signal, not neutral AO.
            for (i, t) in [ambientOnly || referenceRotation != nil ? contact : ambient, contact, depth, normal].enumerated() { encoder.setFragmentTexture(t, index: i) }
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3); encoder.endEncoding()
        }
        let row = (width * 2 + 255) / 256 * 256
        let readbacks = try [original, specialized].map { _ in
            try XCTUnwrap(device.makeBuffer(length: row * height, options: .storageModeShared))
        }
        let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
        for (i, t) in [original, specialized].enumerated() {
            blit.copy(from: t, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: width, height: height, depth: 1), to: readbacks[i], destinationOffset: 0,
                destinationBytesPerRow: row, destinationBytesPerImage: row * height)
        }
        blit.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        let a = readbacks[0].contents().assumingMemoryBound(to: UInt8.self)
        let b = readbacks[1].contents().assumingMemoryBound(to: UInt8.self)
        var changedChannels = 0
        for y in 0..<height { for x in 0..<width {
            let offset = y * row + x * 2
            if referenceRotation != nil {
                for channel in 0..<2 {
                    let difference = abs(Int(b[offset + channel]) - Int(a[offset + channel]))
                    XCTAssertLessThanOrEqual(difference, 1, "Equivalent weights may differ only at a quantization boundary")
                    if difference != 0 { changedChannels += 1 }
                }
            } else if ambientOnly {
                XCTAssertEqual(b[offset],a[offset])
                XCTAssertEqual(b[offset + 1],a[offset + 1],
                    "AO-only filtering must preserve the combined filter's exact R16 bits at borders and depth/normal discontinuities")
            } else {
                XCTAssertEqual(b[offset], 255, "HQ has neutral screen-space ambient visibility")
                XCTAssertEqual(b[offset + 1], a[offset + 1], "Direct filtering must retain every RG8 value, including borders and depth/normal discontinuities")
            }
        } }
        if referenceRotation != nil {
            XCTAssertLessThanOrEqual(Double(changedChannels) / Double(width * height * 2), 0.0001,
                "At least 99.99% of RG8 channels must match the view-space, two-square-root reference")
            for channel in 0..<2 {
                let sample = a[16 * row + 16 * 2 + channel]
                XCTAssertGreaterThan(sample, 0)
                XCTAssertLessThan(sample, 255, "The oracle must exercise both visibility signals")
            }
        }
        if ambientOnly {
            let sample = readbacks[0].contents().advanced(by: 16 * row + 16 * 2)
                .assumingMemoryBound(to: Float16.self).pointee
            XCTAssertGreaterThan(Float(sample),0.05)
            XCTAssertLessThan(Float(sample),0.95,
                "The equivalence fixture must exercise nontrivial ambient filtering")
        }
    }
}
