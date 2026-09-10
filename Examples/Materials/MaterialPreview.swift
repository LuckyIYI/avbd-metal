#if os(macOS)
  import AppKit
  import MetalKit
  import ImageIO
  import GPUSimRenderer
  import PhysicsAVBD
  import SimCore

  @MainActor
  func run() async throws {
    let args = CommandLine.arguments
    func value(_ key: String) -> String? {
      guard let i = args.firstIndex(of: key), i + 1 < args.count else { return nil }
      return args[i + 1]
    }
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal unavailable") }
    var scene = PhysicsScene(name: "Material preview")
    let body = scene.addBody(
      size: F3(repeating: 0.001), density: 0, friction: 0.5, position: value("--asset") == nil ? F3(0, 0, -5) : .zero)
    var materials: [GPUSimSurfaceMaterial] = []
    var programs: [GPUSimMaterialProgram] = []
    if let path = value("--asset") {
      let asset = try GPUSimAssetImporter.load(url: URL(fileURLWithPath: path), device: device)
      for part in asset.parts { scene.addRigidMesh(part.rigidMesh(body: body)) }
      materials = asset.materials
      for message in asset.diagnostics { print(message) }
    } else {
      var material = GPUSimSurfaceMaterial()
      func texture(_ flag: String, srgb: Bool) throws -> MTLTexture? {
        guard let path = value(flag) else { return nil }
        return try GPUSimMaterialLibrary.loadTexture(
          device: device, url: URL(fileURLWithPath: path), sRGB: srgb)
      }
      material.baseColorTexture = try texture("--base-color", srgb: true)
      material.roughnessTexture = try texture("--roughness", srgb: false)
      material.normalTexture = try texture("--normal", srgb: false)
      // Top-left UVs: a +Y tangent normal map must invert its green axis.
      material.invertNormalGreen = args.contains("--normal-gl")
      material.roughness = material.roughnessTexture == nil ? 0.4 : 1
      if args.contains("--procedural") {
        programs = [
          .init(
            body: """
              // Caller-owned demonstration, not a built-in engine material.
              float stripe = smoothstep(0.35,0.65,0.5+0.5*sin(context.uv.x*24));
              surface.color = mix(float3(0.025,0.1,0.2),float3(0.7,0.35,0.035),stripe);
              surface.roughness = mix(0.15,0.75,stripe);
              """)
        ]
        material.program = 1
      }
      materials = [material]
      let mesh = SurfaceMesh(
        vertices: [F3(-1, -1, 5), F3(1, -1, 5), F3(1, 1, 5), F3(-1, 1, 5)],
        normals: Array(repeating: F3(0, 0, 1), count: 4), triangles: [(0, 1, 2), (0, 2, 3)])
      scene.addRigidMesh(
        SceneRigidMesh(
          body: body, mesh: mesh, color: F3(repeating: 1),
          textureCoordinates: [SIMD2(0, 1), SIMD2(1, 1), SIMD2(1, 0), SIMD2(0, 0)], materialID: 1))
    }
    let environment: GPUSimEnvironmentLight?
    if let path = value("--environment") {
      let texture = try GPUSimMaterialLibrary.loadTexture(
        device: device, url: URL(fileURLWithPath: path), sRGB: args.contains("--environment-srgb"))
      environment = try GPUSimEnvironmentLight(device: device, texture: texture)
    } else { environment = nil }
    let resources = try GPUSimMaterialLibrary(
      device: device, materials: materials, programs: programs)
    let solver = try GPUSolver(scene: scene, device: device)
    let renderer = try GPUSimRenderer(device: device, scene: solver, materials: resources, environment: environment)
    renderer.automaticallyFramesScene = false
    renderer.options = args.contains("--fast") ? .lightweight : .qualityBeta
    renderer.options.showsGroundPlane = false
    renderer.options.rayTracingDenoising = !args.contains("--no-denoise")
    switch value("--quality") {
    case "high": renderer.options.rayTracingQuality = .high
    case "balanced": renderer.options.rayTracingQuality = .balanced
    default: break
    }
    if args.contains("--area-light") {
      renderer.options.areaLights = [try GPUSimAreaLight(
        position: F3(1, -1, 3), normal: F3(-1, 1, -3),
        size: SIMD2(2, 2), radiance: F3(8, 7, 6), shape: .disk)]
      renderer.options.sunIntensity = 0
    }
    renderer.options.sunDirection = normalize(F3(-0.6, 0.25, -0.6))
    renderer.setCamera(position: F3(0, -1.5, 2.6), target: .zero, up: F3(0, 0, 1))
    let width = 1024
    let height = 768
    let view = MTKView(frame: NSRect(x: 0, y: 0, width: width, height: height), device: device)
    renderer.configure(view)
    view.isPaused = true
    view.autoResizeDrawable = false
    view.drawableSize = CGSize(width: width, height: height)
    var pixels = [UInt8]()
    var pendingFrame: CheckedContinuation<Void, Never>?
    renderer.frameCompletionHandler = { texture, _ in
      defer {
        pendingFrame?.resume()
        pendingFrame = nil
      }
      pixels = [UInt8](repeating: 0, count: width * height * 4)
      pixels.withUnsafeMutableBytes {
        texture.getBytes(
          $0.baseAddress!, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height),
          mipmapLevel: 0)
      }
    }
    let frames = max(1, Int(value("--frames") ?? "32") ?? 32)
    func draw() async {
      await withCheckedContinuation { continuation in
        pendingFrame = continuation
        view.draw()
        if let failure = renderer.runtimeFailure { fatalError(failure) }
      }
    }
    await draw()
    let start = Date()
    var gpuMilliseconds = 0.0
    for _ in 0..<frames {
      await draw()
      gpuMilliseconds += renderer.lastFrameGPUMilliseconds
    }
    let elapsed = Date().timeIntervalSince(start)
    print("Lighting mode: \(renderer.activeLightingMode); mean GPU time: \(String(format: "%.3f", gpuMilliseconds / Double(frames))) ms")
    if let failure = renderer.runtimeFailure { fatalError(failure) }
    guard !pixels.isEmpty else { fatalError("No rendered frame") }
    // Renderer output uses BGRA8; PNG expects RGBA8.
    for i in stride(from: 0, to: pixels.count, by: 4) { pixels.swapAt(i, i + 2) }
    let provider = CGDataProvider(data: Data(pixels) as CFData)!
    let image = CGImage(
      width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
      space: CGColorSpace(name: CGColorSpace.sRGB)!,
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
      decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let output = URL(fileURLWithPath: value("--output") ?? "/tmp/material-preview.png")
    let destination = CGImageDestinationCreateWithURL(
      output as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { fatalError("Image write failed") }
    print(
      "\(output.path): \(frames) frames, \(String(format:"%.2f",elapsed*1000/Double(frames))) ms/frame including synchronized readback; \(resources.textureBytes) texture bytes"
    )
  }
  @main
  struct MaterialPreview {
    @MainActor static func main() async throws { try await run() }
  }
#else
  @main
  struct MaterialPreview {
    static func main() {
      print("material-preview runs on macOS; GPUSimRenderer material APIs also support iOS.")
    }
  }
#endif
