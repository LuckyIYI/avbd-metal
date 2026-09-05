import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class MetalFXReconstructionTests: XCTestCase {
    func testNeuralReconstructionReducesFreshNoiseAndResetsHistory() throws {
        guard let device = MTLCreateSystemDefaultDevice(), MetalFXReconstruction.supports(device: device, denoising: true) else { throw XCTSkip("MetalFX denoising unavailable") }
        let size = SIMD2(128, 96)
        let fx = try MetalFXReconstruction(device: device, size: size, denoising: true)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: size.x, height: size.y, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite, .renderTarget]; descriptor.storageMode = .private
        let color = try XCTUnwrap(device.makeTexture(descriptor: descriptor))
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void noise(texture2d<float,access::write> output [[texture(0)]], constant uint& frame [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
            uint h=p.x+p.y*65537u+frame*977u; h^=h>>17; h*=0xed5ad4bbu; h^=h>>11; h*=0xac4c1b51u; h^=h>>15;
            float n=float(h&65535u)/65535.0-0.5;
            output.write(float4(float3(0.5+n*0.6),1),p);
        }
        """
        let lib = try device.makeLibrary(source: source, options: nil)
        let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "noise")))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        for frame in 0..<32 {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            _ = fx.beginFrame(camera: matrix_identity_float4x4, options: .qualityBeta, invalidate: frame == 0)
            XCTAssertEqual(fx.reset, frame == 0)
            let pass = fx.guidePass()
            let values = [MTLClearColor(), MTLClearColor(red: 0,green: 0,blue: 1,alpha: 0), MTLClearColor(red: 0.5,green: 0.5,blue: 0.5,alpha: 1), MTLClearColor(red: 0.04,green: 0.04,blue: 0.04,alpha: 1), MTLClearColor(red: 0.7,green: 0,blue: 0,alpha: 0)]
            for (i, value) in values.enumerated() { pass.colorAttachments[i].clearColor = value }
            pass.depthAttachment.clearDepth = 0.5
            try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: pass)).endEncoding()
            let e = try XCTUnwrap(command.makeComputeCommandEncoder())
            e.setComputePipelineState(pipeline); e.setTexture(color, index: 0)
            var f = UInt32(frame); e.setBytes(&f, length: 4, index: 0)
            e.dispatchThreads(MTLSize(width: size.x,height: size.y,depth: 1), threadsPerThreadgroup: MTLSize(width: 8,height: 8,depth: 1)); e.endEncoding()
            try fx.finishFrame(command: command, color: color)
            command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status,.completed,"\(String(describing: command.error))")
        }
        let raw = try read(color, device: device), denoised = try read(fx.output, device: device)
        func mse(_ data: [Float16]) -> Double {
            var error = 0.0
            for y in 16..<80 { for x in 16..<112 { let v=Double(data[(y*128+x)*4]); XCTAssertTrue(v.isFinite); error += pow(v-0.5,2) } }
            return error/(64*96)
        }
        print("MetalFX flat irradiance MSE: raw \(mse(raw)), denoised \(mse(denoised))")
        XCTAssertGreaterThan(mse(raw),0.02)
        XCTAssertLessThan(mse(denoised),mse(raw)*0.15)
        _ = fx.beginFrame(camera: matrix_identity_float4x4, options: .lightweight, invalidate: false)
        XCTAssertTrue(fx.reset)
    }

    func testRigidMotionGuidesIncludeObjectAndCameraMotionWithoutJitter() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let source = makeRenderShaderSource(motionGuides: true)
        let lib = try device.makeLibrary(source: source, options: nil)
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = lib.makeFunction(name: "rigid_mesh_vertex"); d.fragmentFunction = lib.makeFunction(name: "reconstruction_fragment")
        for (i, format) in MetalFXReconstruction.guideFormats.enumerated() { d.colorAttachments[i].pixelFormat = format }
        d.depthAttachmentPixelFormat = .depth32Float
        let pipeline = try device.makeRenderPipelineState(descriptor: d)
        let fx = try MetalFXReconstruction(device: device,size: SIMD2(64,64),denoising: false)
        func buffer<T>(_ values: [T]) throws -> MTLBuffer { try XCTUnwrap(values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!,length: $0.count,options: .storageModeShared) }) }
        // Each rigid vertex is three float4 values, with body ID 0 packed in position.w.
        let vertices = try buffer([SIMD4<Float>(-0.6,-0.6,0.5,0),SIMD4(0,0,1,0.5),SIMD4(0.7,0.7,0.7,0), SIMD4(0.6,-0.6,0.5,0),SIMD4(0,0,1,0.5),SIMD4(0.7,0.7,0.7,0), SIMD4(0,0.6,0.5,0),SIMD4(0,0,1,0.5),SIMD4(0.7,0.7,0.7,0)])
        let current = try buffer([SIMD4<Float>(0.125,0,0,0)]), previous = try buffer([SIMD4<Float>.zero]), rotation = try buffer([SIMD4<Float>(0,0,0,1)])
        var camera = matrix_identity_float4x4; camera.columns.3.y = 0.125
        var guide = MetalFXReconstruction.GuideUniforms(current: camera,previous: matrix_identity_float4x4,size: SIMD4(64,64,0,0))
        var u = Uniforms(viewProj: camera,lightDir: .zero,eye: SIMD4(0,0,2,0),screen: SIMD4(64,64,64,0),camRight: SIMD4(1,0,0,0),camUp: SIMD4(0,-1,0,0),prevViewProj: matrix_identity_float4x4,temporal: .zero,shadowViewProj: matrix_identity_float4x4,shadowParams: .zero,invViewProj: camera.inverse,prevInvViewProj: matrix_identity_float4x4,aoProjection: SIMD4(1,-0.1,1,1))
        // Deliberately jitter rasterization; the expected guide remains exactly (-4,+4) pixels.
        u.viewProj.columns.3.x += 0.4/32; u.viewProj.columns.3.y -= 0.3/32
        let queue = try XCTUnwrap(device.makeCommandQueue()), command = try XCTUnwrap(queue.makeCommandBuffer())
        let e = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: fx.guidePass()))
        e.setRenderPipelineState(pipeline)
        e.setVertexBuffer(vertices,offset: 0,index: 0); e.setVertexBytes(&u,length: MemoryLayout<Uniforms>.stride,index: 1)
        e.setVertexBuffer(current,offset: 0,index: 2); e.setVertexBuffer(rotation,offset: 0,index: 3)
        e.setVertexBuffer(previous,offset: 0,index: 6); e.setVertexBuffer(rotation,offset: 0,index: 7)
        var noAppearance: UInt32 = 0; e.setVertexBuffer(vertices,offset: 0,index: 4); e.setVertexBytes(&noAppearance,length: 4,index: 5)
        e.setFragmentBytes(&u,length: MemoryLayout<Uniforms>.stride,index: 1); e.setFragmentBytes(&guide,length: MemoryLayout<MetalFXReconstruction.GuideUniforms>.stride,index: 2)
        e.drawPrimitives(type: .triangle,vertexStart: 0,vertexCount: 3); e.endEncoding(); command.commit(); command.waitUntilCompleted()
        XCTAssertEqual(command.status,.completed,"\(String(describing: command.error))")
        let data = try read(fx.motion,device: device,channels: 2)
        for y in 28..<34 { for x in 30..<36 { let i=(y*64+x)*2; XCTAssertEqual(Float(data[i]),-4,accuracy: 0.01); XCTAssertEqual(Float(data[i+1]),4,accuracy: 0.01) } }
    }

    func testDeformingClothAndSkinUsePreviousVertices() throws {
        guard let device = MTLCreateSystemDefaultDevice(), MetalFXReconstruction.supports(device: device,denoising: false) else { throw XCTSkip("MetalFX unavailable") }
        let lib = try device.makeLibrary(source: makeRenderShaderSource(motionGuides: true),options: nil)
        let fx = try MetalFXReconstruction(device: device,size: SIMD2(64,64),denoising: false)
        func buffer<T>(_ values: [T]) throws -> MTLBuffer { try XCTUnwrap(values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!,length: $0.count,options: .storageModeShared) }) }
        let p: [SIMD4<Float>] = [SIMD4(-0.6,-0.6,0.5,0),SIMD4(0.6,-0.6,0.5,0),SIMD4(0,0.6,0.5,0)]
        let old = p.enumerated().map { i,v in v-SIMD4<Float>(0.125,i == 2 ? 0.25 : 0,0,0) }
        let n = [SIMD4<Float>](repeating: SIMD4(0,0,1,0),count: 3)
        let corners = try buffer([UInt32(0),1,2]), normals = try buffer(n)
        var guide = MetalFXReconstruction.GuideUniforms(current: matrix_identity_float4x4,previous: matrix_identity_float4x4,size: SIMD4(64,64,0,0))
        var u = Uniforms(viewProj: matrix_identity_float4x4,lightDir: .zero,eye: SIMD4(0,0,2,0),screen: SIMD4(64,64,64,0),camRight: SIMD4(1,0,0,0),camUp: SIMD4(0,-1,0,0),prevViewProj: matrix_identity_float4x4,temporal: .zero,shadowViewProj: matrix_identity_float4x4,shadowParams: .zero,invViewProj: matrix_identity_float4x4,prevInvViewProj: matrix_identity_float4x4,aoProjection: SIMD4(1,-0.1,1,1))
        for skin in [false,true] {
            let current = try buffer(skin ? (0..<3).flatMap { [p[$0],n[$0]] } : p)
            let previous = try buffer(skin ? (0..<3).flatMap { [old[$0],n[$0]] } : old)
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = lib.makeFunction(name: skin ? "skin_vertex" : "soft_vertex")
            d.fragmentFunction = lib.makeFunction(name: "soft_reconstruction_fragment")
            for (i,format) in MetalFXReconstruction.guideFormats.enumerated() { d.colorAttachments[i].pixelFormat = format }
            d.depthAttachmentPixelFormat = .depth32Float
            let pipeline = try device.makeRenderPipelineState(descriptor: d)
            let command = try XCTUnwrap(device.makeCommandQueue()?.makeCommandBuffer())
            let e = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: fx.guidePass()))
            e.setRenderPipelineState(pipeline); e.setVertexBuffer(corners,offset: 0,index: 0)
            e.setVertexBytes(&u,length: MemoryLayout<Uniforms>.stride,index: 1)
            e.setVertexBuffer(current,offset: 0,index: 2); e.setVertexBuffer(normals,offset: 0,index: 3)
            e.setVertexBuffer(previous,offset: 0,index: 6); e.setVertexBuffer(normals,offset: 0,index: 7)
            var noAppearance: UInt32 = 0; e.setVertexBuffer(current,offset: 0,index: 4); e.setVertexBytes(&noAppearance,length: 4,index: 5)
            e.setFragmentBytes(&u,length: MemoryLayout<Uniforms>.stride,index: 1); e.setFragmentBytes(&guide,length: MemoryLayout<MetalFXReconstruction.GuideUniforms>.stride,index: 2)
            e.drawPrimitives(type: .triangle,vertexStart: 0,vertexCount: 3); e.endEncoding(); command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status,.completed)
            let values = try read(fx.motion,device: device,channels: 2)
            for y in 28..<34 { for x in 30..<34 {
                let worldY = 1-(Float(y)+0.5)/32
                let expectedY = 8*(worldY+0.6)/1.2
                XCTAssertEqual(Float(values[(y*64+x)*2]),-4,accuracy: 0.01)
                XCTAssertEqual(Float(values[(y*64+x)*2+1]),expectedY,accuracy: 0.01)
            } }
        }
    }

    private func read(_ texture: MTLTexture, device: MTLDevice, channels: Int = 4) throws -> [Float16] {
        let bytesPerRow = ((texture.width*channels*2+255)/256)*256
        let buffer = try XCTUnwrap(device.makeBuffer(length: bytesPerRow*texture.height,options: .storageModeShared))
        let queue = try XCTUnwrap(device.makeCommandQueue()), command = try XCTUnwrap(queue.makeCommandBuffer()), b = try XCTUnwrap(command.makeBlitCommandEncoder())
        b.copy(from: texture,sourceSlice: 0,sourceLevel: 0,sourceOrigin: MTLOrigin(),sourceSize: MTLSize(width: texture.width,height: texture.height,depth: 1),to: buffer,destinationOffset: 0,destinationBytesPerRow: bytesPerRow,destinationBytesPerImage: bytesPerRow*texture.height)
        b.endEncoding(); command.commit(); command.waitUntilCompleted()
        let values = buffer.contents().assumingMemoryBound(to: Float16.self)
        return (0..<texture.height).flatMap { y in Array(UnsafeBufferPointer(start: values+y*bytesPerRow/2,count: texture.width*channels)) }
    }
}
