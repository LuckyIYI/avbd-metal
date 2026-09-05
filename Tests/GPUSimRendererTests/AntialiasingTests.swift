import Metal
import simd
import XCTest
@testable import GPUSimRenderer

@MainActor
final class AntialiasingTests: XCTestCase {
    private func uniforms(width: Int, height: Int) -> Uniforms {
        Uniforms(viewProj: matrix_identity_float4x4, lightDir: SIMD4(0,0,1,0), eye: SIMD4(0,0,2,0),
            screen: SIMD4(Float(width),Float(height),Float(height),0), camRight: SIMD4(1,0,0,0), camUp: SIMD4(0,-1,0,0),
            prevViewProj: matrix_identity_float4x4, temporal: SIMD4(0,1,0,0), shadowViewProj: matrix_identity_float4x4,
            shadowParams: .zero, invViewProj: matrix_identity_float4x4, prevInvViewProj: matrix_identity_float4x4,
            effects: SIMD4(1,0,0,0.65), rayTracing: SIMD4(1,0,0,1), aoProjection: SIMD4(1,-0.1,1,1))
    }

    func testSpecularFootprintReducesSubpixelHighlightError() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let source = """
        fragment float4 highlight_fixture(FSOut in [[stage_in]]) {
            float angle = (in.position.x-64+fract(in.position.y/8))*0.035;
            float3 n = float3(sin(angle),0,cos(angle));
            float rough = 0.055;
            float filtered = specularRoughness(n,rough,1);
            float a2 = pow(rough,4.0), b2 = pow(filtered,4.0), nh2 = n.z*n.z;
            return float4(a2/(M_PI_F*pow(nh2*(a2-1)+1,2.0)),
                b2/(M_PI_F*pow(nh2*(b2-1)+1,2.0)),filtered,1);
        }
        """
        let library = try device.makeLibrary(source: renderShaderSource+source, options: nil)
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = library.makeFunction(name: "fs_vertex"); d.fragmentFunction = library.makeFunction(name: "highlight_fixture")
        d.colorAttachments[0].pixelFormat = .rgba32Float
        let pipeline = try device.makeRenderPipelineState(descriptor: d)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,width: 128,height: 8,mipmapped: false)
        descriptor.usage = [.renderTarget]; descriptor.storageMode = .shared
        let target = try XCTUnwrap(device.makeTexture(descriptor: descriptor))
        let queue = try XCTUnwrap(device.makeCommandQueue()), command = try XCTUnwrap(queue.makeCommandBuffer())
        let pass = MTLRenderPassDescriptor(); pass.colorAttachments[0].texture = target; pass.colorAttachments[0].storeAction = .store
        let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass))
        encoder.setRenderPipelineState(pipeline); encoder.drawPrimitives(type: .triangle,vertexStart: 0,vertexCount: 3)
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status,.completed,"\(String(describing: command.error))")
        var pixels = [SIMD4<Float>](repeating: .zero,count: 128*8)
        target.getBytes(&pixels,bytesPerRow: 128*16,from: MTLRegionMake2D(0,0,128,8),mipmapLevel: 0)
        var rawError = 0.0, filteredError = 0.0
        for y in 0..<8 { for x in 55..<73 {
            var reference = 0.0
            // Independently integrate the narrow GGX highlight across a pixel.
            for i in 0..<2048 {
                let angle = (Double(x)+(Double(i)+0.5)/2048-64+(Double(y)+0.5)/8)*0.035
                let a2 = pow(0.055,4.0), denominator = pow(cos(angle),2)*(a2-1)+1
                reference += a2/(Double.pi*denominator*denominator)/2048
            }
            let value = pixels[y*128+x]
            rawError += pow(Double(value.x)-reference,2)
            filteredError += pow(Double(value.y)-reference,2)
            XCTAssertTrue(value.y.isFinite)
        } }
        print("Specular AA integrated-highlight MSE ratio: \(filteredError/rawError)")
        XCTAssertLessThan(filteredError/rawError,0.5)
    }

    func testReflectionRespectsMSAACoverageAndDoesNotApplyDiffuseAOToMetal() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let source = """
        vertex VOut coverage_fixture(uint i [[vertex_id]]) {
            const float2 p[3] = {float2(-0.83,-0.75),float2(0.89,-0.75),float2(-0.83,0.91)};
            VOut o = {}; o.position = float4(p[i],0.5,1); o.world = o.position.xyz;
            o.normal = float3(0,0,1); o.albedo = float3(1); o.pbr = float2(0.3,1); o.opacity = 1;
            return o;
        }
        """
        let library = try device.makeLibrary(source: renderShaderSource+source, options: nil)
        let effects = try ScreenSpacePipeline(device: device,library: library)
        let width = 127, height = 95
        try effects.prepare(size: CGSize(width: width,height: height),options: .qualityBeta)
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = library.makeFunction(name: "coverage_fixture"); d.fragmentFunction = library.makeFunction(name: "pbr_fragment")
        d.colorAttachments[0].pixelFormat = .rgba16Float; d.rasterSampleCount = 4
        let pipeline = try device.makeRenderPipelineState(descriptor: d)
        let samples = try effects.texture(.rgba16Float,width: width,height: height,label: "Covered reflection",samples: 4)
        let output = try effects.texture(.rgba16Float,width: width,height: height,label: "Resolved reflection")
        let queue = try XCTUnwrap(device.makeCommandQueue())
        func render(reflection: Bool, ao: Double, neighborNormal: Double = 1) throws -> [SIMD4<Float>] {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            for (texture,color) in [(effects.normal!,MTLClearColor(red: 0,green: 0,blue: neighborNormal,alpha: 0.3)),
                                    (effects.material!,MTLClearColor(red: 1,green: 1,blue: 1,alpha: 1)),
                                    (effects.reflection!,MTLClearColor(red: 0.125,green: 0.125,blue: 0.125,alpha: 1)),
                                    (effects.visibility!,MTLClearColor(red: ao,green: 1,blue: 0,alpha: 1))] {
                let clear = MTLRenderPassDescriptor(); clear.colorAttachments[0].texture = texture
                clear.colorAttachments[0].loadAction = .clear; clear.colorAttachments[0].storeAction = .store
                clear.colorAttachments[0].clearColor = color
                let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: clear)); encoder.endEncoding()
            }
            let clear = MTLRenderPassDescriptor(); clear.depthAttachment.texture = effects.depth
            clear.depthAttachment.loadAction = .clear; clear.depthAttachment.storeAction = .store; clear.depthAttachment.clearDepth = 0.5
            let clearEncoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: clear)); clearEncoder.endEncoding()
            let pass = MTLRenderPassDescriptor(); pass.colorAttachments[0].texture = samples
            pass.colorAttachments[0].resolveTexture = output; pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,0); pass.colorAttachments[0].storeAction = .multisampleResolve
            let encoder = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass))
            var U = uniforms(width: width,height: height); U.rayTracing.w = reflection ? 1 : 0
            encoder.setRenderPipelineState(pipeline)
            encoder.setFragmentBytes(&U,length: MemoryLayout<Uniforms>.stride,index: 1)
            for (i,texture) in [effects.visibility!,effects.depth!,effects.reflection!,effects.normal!,effects.depth!,effects.material!].enumerated() {
                encoder.setFragmentTexture(texture,index: i)
            }
            encoder.drawPrimitives(type: .triangle,vertexStart: 0,vertexCount: 3); encoder.endEncoding()
            let row = (width*8+255)/256*256
            let buffer = try XCTUnwrap(device.makeBuffer(length: row*height,options: .storageModeShared))
            let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
            blit.copy(from: output,sourceSlice: 0,sourceLevel: 0,sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: width,height: height,depth: 1),to: buffer,destinationOffset: 0,
                destinationBytesPerRow: row,destinationBytesPerImage: row*height)
            blit.endEncoding(); command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status,.completed,"\(String(describing: command.error))")
            let bytes = buffer.contents().assumingMemoryBound(to: UInt16.self)
            return (0..<width*height).map { i in
                let j = i/width*(row/2)+i%width*4
                return SIMD4(Float(Float16(bitPattern: bytes[j])),Float(Float16(bitPattern: bytes[j+1])),
                    Float(Float16(bitPattern: bytes[j+2])),Float(Float16(bitPattern: bytes[j+3])))
            }
        }
        let base = try render(reflection: false,ao: 1), reflected = try render(reflection: true,ao: 1)
        let occluded = try render(reflection: true,ao: 0.2), wrongSurface = try render(reflection: true,ao: 1,neighborNormal: -1)
        var partial = 0
        for i in base.indices {
            let coverage = base[i].w
            if coverage > 0 && coverage < 1 { partial += 1 }
            XCTAssertEqual(reflected[i].x-base[i].x,0.125*coverage,accuracy: 0.001,
                "Reflection must cover exactly the same MSAA samples as its receiver")
            XCTAssertEqual(occluded[i].x,reflected[i].x,accuracy: 0.001,"World specular visibility must not be multiplied by diffuse AO")
            XCTAssertEqual(wrongSurface[i].x,base[i].x,accuracy: 0.001,"An unrelated half-resolution surface must contribute no reflection")
        }
        XCTAssertGreaterThan(partial,50)
    }

    func testDisplayEdgesApproachAreaCoverageWithoutBlurringFlatRegions() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let library = try device.makeLibrary(source: renderShaderSource, options: nil)
        let effects = try ScreenSpacePipeline(device: device, library: library)
        let width = 129, height = 95, row = 768
        try effects.prepare(size: CGSize(width: width, height: height), options: .lightweight)
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let multisample = try effects.texture(.bgra8Unorm_srgb, width: width, height: height, label: "AA samples", samples: 4)
        let depth = try effects.texture(.depth32Float, width: width, height: height, label: "AA depth", samples: 4)
        let output = try effects.texture(.bgra8Unorm_srgb, width: width, height: height, label: "AA output")
        var beforeError = 0.0, afterError = 0.0
        for slope in [0.37, -0.61, 1.4] {
            var pixels = [UInt8](repeating: 255, count: row*height)
            var reference = [Double](repeating: 0, count: width*height)
            let boundary = slope < 0 ? 85.0 : 18.0
            for y in 0..<height { for x in 0..<width {
                func inside(_ px: Double, _ py: Double) -> Bool { px > boundary+slope*py }
                let value: UInt8 = inside(Double(x)+0.5,Double(y)+0.5) ? 204 : 51
                for channel in 0..<3 { pixels[y*row+x*4+channel] = value }
                var coverage = 0
                // Independent 32x32 area integration of a subpixel diagonal.
                if abs(Double(x)+0.5-boundary-slope*(Double(y)+0.5)) < 2 {
                    for sy in 0..<32 { for sx in 0..<32 {
                        if inside(Double(x)+(Double(sx)+0.5)/32,Double(y)+(Double(sy)+0.5)/32) { coverage += 1 }
                    } }
                } else { coverage = value == 204 ? 1024 : 0 }
                reference[y*width+x] = 51+153*Double(coverage)/1024
            } }
            let upload = try XCTUnwrap(pixels.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
            let readback = try XCTUnwrap(device.makeBuffer(length: row*height, options: .storageModeShared))
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let uploadEncoder = try XCTUnwrap(command.makeBlitCommandEncoder())
            uploadEncoder.copy(from: upload, sourceOffset: 0, sourceBytesPerRow: row, sourceBytesPerImage: row*height,
                sourceSize: MTLSize(width: width,height: height,depth: 1), to: effects.displayColor!, destinationSlice: 0,
                destinationLevel: 0, destinationOrigin: MTLOrigin())
            uploadEncoder.endEncoding()
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = multisample
            pass.colorAttachments[0].resolveTexture = output
            pass.colorAttachments[0].storeAction = .multisampleResolve
            pass.depthAttachment.texture = depth
            try effects.encodeAntialiasing(command: command, destination: pass)
            let readEncoder = try XCTUnwrap(command.makeBlitCommandEncoder())
            readEncoder.copy(from: output, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                sourceSize: MTLSize(width: width,height: height,depth: 1), to: readback, destinationOffset: 0,
                destinationBytesPerRow: row, destinationBytesPerImage: row*height)
            readEncoder.endEncoding()
            command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
            let values = readback.contents().assumingMemoryBound(to: UInt8.self)
            for y in 3..<(height-3) { for x in 3..<(width-3) {
                let expected = reference[y*width+x], i = y*row+x*4
                beforeError += pow(Double(pixels[i])-expected,2)
                afterError += pow(Double(values[i])-expected,2)
                if abs(Double(x)+0.5-boundary-slope*(Double(y)+0.5)) > 4 {
                    XCTAssertEqual(values[i],pixels[i], "flat shading must retain its exact display value")
                }
            } }
        }
        print("Display AA area-coverage MSE ratio: \(afterError/beforeError)")
        XCTAssertLessThan(afterError/beforeError,0.65)
        var disabled = GPUSimRenderOptions.lightweight; disabled.edgeAntialiasing = false
        try effects.prepare(size: CGSize(width: width,height: height), options: disabled)
        XCTAssertNil(effects.displayColor, "disabled AA releases its only additional full-resolution surface")
    }
}
