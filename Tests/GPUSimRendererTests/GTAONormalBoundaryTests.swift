import Metal
import simd
import XCTest
@testable import GPUSimRenderer

final class GTAONormalBoundaryTests: XCTestCase {
    func testMissingSecondNeighborUsesComparableErrorsOnBothSides() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let library = try device.makeLibrary(source: renderShaderSource + """
        struct BoundaryDepth { float depth [[depth(any)]]; };
        fragment BoundaryDepth normal_fixture_depth(FSOut in [[stage_in]], device const float* values [[buffer(0)]]) {
            BoundaryDepth out; out.depth = values[uint(in.position.y)*9u+uint(in.position.x)]; return out;
        }
        fragment float4 normal_boundary_fixture(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
            depth2d<float> depth [[texture(0)]]) {
            uint2 pixel = uint2(in.position.xy);
            float d = depth.read(pixel);
            float3 P = screenPosition(in.uv,d,U);
            return float4(gtaoGeometricNormal(pixel,d,P,float3(0,0,-1),U,depth),1);
        }
        """, options:nil)
        func pipeline(_ name:String, depth:Bool) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = library.makeFunction(name:"fs_vertex")
            d.fragmentFunction = library.makeFunction(name:name)
            if depth { d.depthAttachmentPixelFormat = .depth32Float }
            else { d.colorAttachments[0].pixelFormat = .rgba32Float }
            return try device.makeRenderPipelineState(descriptor:d)
        }
        let depthPipeline = try pipeline("normal_fixture_depth",depth:true)
        let normalPipeline = try pipeline("normal_boundary_fixture",depth:false)
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat:.depth32Float,width:9,height:9,mipmapped:false)
        td.usage = [.renderTarget,.shaderRead]
        let depth = try XCTUnwrap(device.makeTexture(descriptor:td))
        td.pixelFormat = .rgba32Float; td.storageMode = .shared
        let output = try XCTUnwrap(device.makeTexture(descriptor:td))
        let ds = MTLDepthStencilDescriptor(); ds.depthCompareFunction = .always; ds.isDepthWriteEnabled = true
        let depthState = try XCTUnwrap(device.makeDepthStencilState(descriptor:ds))
        let queue = try XCTUnwrap(device.makeCommandQueue())
        var u = Uniforms(viewProj:matrix_identity_float4x4,lightDir:.zero,eye:.zero,
            screen:SIMD4(9,9,4.5,0),camRight:SIMD4(1,0,0,0),camUp:SIMD4(0,1,0,0),
            prevViewProj:matrix_identity_float4x4,temporal:.zero,shadowViewProj:matrix_identity_float4x4,
            shadowParams:.zero,invViewProj:matrix_identity_float4x4,prevInvViewProj:matrix_identity_float4x4,
            aoProjection:SIMD4(1,-0.1,1,1))
        for axis in 0...1 { for side in [-1,1] { for background in [false,true] {
            var center = SIMD2<Int>(4,4)
            center[axis] = background ? 4 : (side < 0 ? 1 : 7)
            var values = [Float](repeating:0.6,count:81)
            var near = center; near[axis] += side
            var far = center; far[axis] += side*2
            var other = center; other[axis] -= side
            var other2 = center; other2[axis] -= side*2
            values[near.y*9+near.x] = 0.59
            if background { values[far.y*9+far.x] = 1 }
            values[other.y*9+other.x] = 0.64
            values[other2.y*9+other2.x] = 0.68
            let buffer = try XCTUnwrap(values.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:$0.count,options:.storageModeShared) })
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let pass = MTLRenderPassDescriptor(); pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear; pass.depthAttachment.storeAction = .store
            let e = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor:pass))
            e.setRenderPipelineState(depthPipeline); e.setDepthStencilState(depthState)
            e.setFragmentBuffer(buffer,offset:0,index:0)
            e.drawPrimitives(type:.triangle,vertexStart:0,vertexCount:3); e.endEncoding()
            let result = MTLRenderPassDescriptor(); result.colorAttachments[0].texture = output
            result.colorAttachments[0].storeAction = .store
            let f = try XCTUnwrap(command.makeRenderCommandEncoder(descriptor:result))
            f.setRenderPipelineState(normalPipeline); f.setFragmentTexture(depth,index:0)
            f.setFragmentBytes(&u,length:MemoryLayout<Uniforms>.stride,index:1)
            f.drawPrimitives(type:.triangle,vertexStart:0,vertexCount:3); f.endEncoding()
            command.commit(); command.waitUntilCompleted()
            XCTAssertEqual(command.status,.completed,"\(String(describing:command.error))")
            func position(_ p:SIMD2<Int>, _ d:Float) -> SIMD3<Float> {
                let z = -0.1/(d-1)
                return SIMD3((Float(p.x)+0.5)/9*2-1,(Float(p.y)+0.5)/9*2-1,1)*z
            }
            let P = position(center,0.6)
            let tangent = (position(near,0.59)-P)*Float(side)
            let perpendicular = axis == 0 ? SIMD3<Float>(0,1,0) : SIMD3<Float>(1,0,0)
            var expected = normalize(cross(tangent,perpendicular))
            if dot(expected,-P) < 0 { expected = -expected }
            var pixel = SIMD4<Float>.zero
            output.getBytes(&pixel,bytesPerRow:16,from:MTLRegionMake2D(center.x,center.y,1,1),mipmapLevel:0)
            XCTAssertGreaterThan(dot(SIMD3(pixel.x,pixel.y,pixel.z),expected),0.9999,
                "Missing second neighbor: axis \(axis), side \(side), background \(background)")
        } } }
    }
}
