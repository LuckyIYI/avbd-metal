import MetalKit
import CoreGraphics
import ImageIO
import XCTest
@testable import GPUSimRenderer

@MainActor
final class WallCornerTests: XCTestCase {
    private final class Scene: GPUSimRenderableScene {
        let renderDevice: MTLDevice
        var renderBodyCount: Int { 0 }
        var renderRigidInstanceCount: Int { 0 }
        var rendererStateIsValid: Bool { true }
        var renderCameraHint: GPUSimRenderCameraHint { .init() }
        var rigidMeshRenderSurface: GPUSimRigidMeshRenderSurface? { nil }
        var softRenderSurface: GPUSimSoftRenderSurface? { nil }
        var skinnedRenderSurface: GPUSimSkinnedRenderSurface? { nil }
        var convexDebugRenderSurface: GPUSimConvexDebugRenderSurface? { nil }
        init(_ device: MTLDevice) { renderDevice = device }
        func encodeRenderInstances(_ commandBuffer: MTLCommandBuffer, instances: MTLBuffer,
            colorMode: GPUSimRenderColorMode, appearanceOverrides: MTLBuffer?) throws {}
    }

    func testSlantedWallJointKeepsOcclusionThroughPixelGridCrossings() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        let scene = Scene(device)
        let renderer = try GPUSimRenderer(device: device)
        try renderer.setScene(scene)
        renderer.auxiliaryInstances = [
            .init(primitive: .box(size: SIMD3(0.1,3.2,2.6)), position: SIMD3(1.27,0,1.3), color: SIMD3(repeating: 0.75)),
            .init(primitive: .box(size: SIMD3(3.6,0.09,2.6)), position: SIMD3(-0.48,-1.645,1.3), color: SIMD3(repeating: 0.75))
        ]
        let w = 1120, h = 760
        let view = MTKView(frame: CGRect(x:0,y:0,width:w,height:h), device:device)
        renderer.configure(view); view.isPaused = true
        view.autoResizeDrawable = false; view.drawableSize = CGSize(width:w,height:h)
        renderer.automaticallyFramesScene = false
        renderer.options = .init(contactShadows:false, showsGroundPlane:false)
        renderer.options.sunDirection = normalize(SIMD3(-1,0.18,-0.75))
        var pixels: [UInt8] = []
        renderer.frameCompletionHandler = { texture, _ in
            pixels = [UInt8](repeating:0,count:w*h*4)
            pixels.withUnsafeMutableBytes { texture.getBytes($0.baseAddress!, bytesPerRow:w*4,
                from:MTLRegionMake2D(0,0,w,h), mipmapLevel:0) }
        }
        for targetX: Float in [1.10, 1.11, 1.34] {
            renderer.setCamera(position: SIMD3(-0.15,-0.15,1.65),
                target: SIMD3(targetX,-1.6,1.35), up: SIMD3(0,0,1))
            renderer.options.ambientOcclusion = true
            view.draw()
            XCTAssertNil(renderer.runtimeFailure)
            XCTAssertEqual(pixels.count, w*h*4)
            guard pixels.count == w*h*4 else { return }
            let shaded = pixels
            try save(shaded, width:w, height:h, name:"joint-\(targetX)")
            renderer.options.ambientOcclusion = false
            view.draw()
            XCTAssertNil(renderer.runtimeFailure)

            // Project the known joint, independently of its AO. Testing an
            // off-center vertical line crosses every subpixel edge alignment.
            let vp = renderer.projectionMatrix(aspect:Float(w)/Float(h)) * renderer.viewMatrix
            func project(_ z: Float) -> SIMD2<Float> {
                let clip = vp * SIMD4<Float>(1.22,-1.6,z,1)
                return SIMD2((clip.x/clip.w*0.5+0.5)*Float(w), (0.5-clip.y/clip.w*0.5)*Float(h))
            }
            let bottom = project(0), top = project(2.6)
            var peak: Float = 0
            for y in 80..<(h-80) {
                let t = (Float(y)+0.5-bottom.y)/(top.y-bottom.y)
                let x = Int(bottom.x + t*(top.x-bottom.x))
                for q in (x-2)...(x+2) {
                    let index = (y*w+q)*4
                    peak = max(peak, Float(shaded[index])/Float(max(pixels[index],1)))
                }
            }
            print("Wall joint maximum AO-on / AO-off brightness at \(targetX): \(peak)")
            // A perpendicular wall blocks substantial ambient light at the
            // crease. The old nearest-depth normal invented bright cracks
            // above 0.9; allow a generous margin for quantization and MSAA.
            XCTAssertLessThan(peak, 0.85, "The joint must remain occluded across pixel-grid crossings")
        }
    }

    private func save(_ pixels:[UInt8],width:Int,height:Int,name:String) throws {
        guard let path = ProcessInfo.processInfo.environment["GTAO_TEST_OUTPUT"] else { return }
        try FileManager.default.createDirectory(atPath:path,withIntermediateDirectories:true)
        let gray = stride(from:0,to:pixels.count,by:4).map { pixels[$0] }
        let provider = try XCTUnwrap(CGDataProvider(data:Data(gray) as CFData))
        let image = try XCTUnwrap(CGImage(width:width,height:height,bitsPerComponent:8,bitsPerPixel:8,
            bytesPerRow:width,space:CGColorSpaceCreateDeviceGray(),bitmapInfo:[],provider:provider,
            decode:nil,shouldInterpolate:false,intent:.defaultIntent))
        let url = URL(fileURLWithPath:path).appendingPathComponent(name+".png")
        let target = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL,"public.png" as CFString,1,nil))
        CGImageDestinationAddImage(target,image,nil)
        XCTAssertTrue(CGImageDestinationFinalize(target))
    }
}
