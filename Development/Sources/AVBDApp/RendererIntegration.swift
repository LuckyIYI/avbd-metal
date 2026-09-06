import Foundation
import ImageIO
import Metal
import PhysicsAVBD
import GPUSimRenderer

/// App policy layered over the package renderer. The public renderer source
/// requires only a solver; these additional values belong to this UI.
@MainActor
protocol RenderableModel: GPUSimRendererSource {
    nonisolated var captureID: String { get }
    var solver: GPUSolver? { get }
    var rayTracingEnabled: Bool { get }
    var screenSpaceReflectionsEnabled: Bool { get }
    var colorByGraphColor: Bool { get }
    var showConvexCollisionGeometry: Bool { get }
    var convexCollisionWireframe: Bool { get }
    var statsText: String { get }
    var cameraEpoch: Int { get }
    func tickIfRunning()
    func reportRenderFailure(_ message: String)
}

extension RenderableModel {
    var showConvexCollisionGeometry: Bool { false }
    var convexCollisionWireframe: Bool { true }
    var supportsRayTracing: Bool {
        (solver?.device ?? MTLCreateSystemDefaultDevice()).map { GPUSimRenderer.supportsHQ(device: $0) } ?? false
    }

    var renderScene: (any GPUSimRenderableScene)? { solver }
    var rendererOptions: GPUSimRenderOptions {
        GPUSimRenderOptions(
            colorMode: colorByGraphColor ? .constraintGraph : .bodyIndex,
            showConvexCollisionGeometry: showConvexCollisionGeometry,
            convexCollisionWireframe: convexCollisionWireframe,
            screenSpaceReflections: screenSpaceReflectionsEnabled,
            lightingMode: rayTracingEnabled ? .qualityBeta : .lightweight
        )
    }

    var rendererSceneRevision: Int { cameraEpoch }
    func rendererWillDrawFrame() { tickIfRunning() }
    func rendererDidFail(_ message: String) { reportRenderFailure(message) }
}

extension SimulationModel {
    func reportRenderFailure(_ message: String) {
        running = false
        statsText = "render stopped: \(message)"
    }
}

/// Keeps scripted screenshots/video and the smoke-test marker out of the
/// reusable renderer package.
@MainActor
func configureAppCapture(
    renderer: GPUSimRenderer,
    model: any RenderableModel
) {
    let environment = ProcessInfo.processInfo.environment
    var cameraWasOverridden = false
    if let value = environment["AVBD_CAM_DIST"].flatMap(Float.init) {
        renderer.distance = value
        cameraWasOverridden = true
    }
    if let value = environment["AVBD_CAM_AZ"].flatMap(Float.init) {
        renderer.azimuth = value
        cameraWasOverridden = true
    }
    if let value = environment["AVBD_CAM_EL"].flatMap(Float.init) {
        renderer.elevation = value
        cameraWasOverridden = true
    }
    if let value = environment["AVBD_CAM_TZ"].flatMap(Float.init) {
        renderer.target.z = value
        cameraWasOverridden = true
    }
    if cameraWasOverridden {
        renderer.automaticallyFramesScene = false
    }

    guard environment["AVBD_MARKER"] != nil
            || environment["AVBD_SHOT"] != nil
            || environment["AVBD_VIDEO_DIR"] != nil else { return }

    renderer.frameCompletionHandler = { [weak model] texture, frame in
        guard let model else { return }
        if frame == 30, environment["AVBD_MARKER"] != nil {
            let bodies = model.solver?.bodyCount ?? 0
            let instances = model.solver?.renderRigidBodyCount ?? 0
            let info = "frames=30 bodies=\(bodies) rigidInstances=\(instances) "
                + "stats=\(model.statsText)"
            try? info.write(
                toFile: "/tmp/avbd_render_marker.txt",
                atomically: true,
                encoding: .utf8
            )
        }

        let selected = environment["AVBD_CAPTURE_VIEW"]
            ?? (environment["AVBD_POLICY_REPLAY"] == nil ? "playground" : "policy")
        guard model.captureID == selected else { return }

        if let path = environment["AVBD_SHOT"],
           frame == (Int(environment["AVBD_SHOT_FRAME"] ?? "") ?? 90) {
            writePNG(texture: texture, to: path)
            exit(0)
        }
        if let directory = environment["AVBD_VIDEO_DIR"] {
            let every = max(Int(environment["AVBD_VIDEO_EVERY"] ?? "") ?? 3, 1)
            let maximum = max(Int(environment["AVBD_VIDEO_FRAMES"] ?? "") ?? 600, 1)
            if frame.isMultiple(of: every) {
                let path = String(format: "%@/%05d.png", directory, frame / every)
                writePNG(texture: texture, to: path)
            }
            if frame >= maximum { exit(0) }
        }
    }
}

private func writePNG(texture: MTLTexture, to path: String) {
    let width = texture.width
    let height = texture.height
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    bytes.withUnsafeMutableBytes { raw in
        texture.getBytes(
            raw.baseAddress!,
            bytesPerRow: width * 4,
            from: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0
        )
    }
    for index in stride(from: 0, to: bytes.count, by: 4) {
        bytes.swapAt(index, index + 2)
    }
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let provider = CGDataProvider(data: Data(bytes) as CFData),
          let image = CGImage(
              width: width,
              height: height,
              bitsPerComponent: 8,
              bitsPerPixel: 32,
              bytesPerRow: width * 4,
              space: colorSpace,
              bitmapInfo: CGBitmapInfo(
                  rawValue: CGImageAlphaInfo.noneSkipLast.rawValue
              ),
              provider: provider,
              decode: nil,
              shouldInterpolate: false,
              intent: .defaultIntent
          ),
          let destination = CGImageDestinationCreateWithURL(
              URL(fileURLWithPath: path) as CFURL,
              "public.png" as CFString,
              1,
              nil
          ) else { return }
    CGImageDestinationAddImage(destination, image, nil)
    CGImageDestinationFinalize(destination)
}
