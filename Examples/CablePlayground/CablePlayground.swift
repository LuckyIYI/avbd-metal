#if os(macOS)
import AppKit
import SwiftUI
import MetalKit
import ImageIO
import simd
import SimCore
import PhysicsAVBD
import GPUSimDemos
import GPUSimRenderer

private let demoNames = ["cablethreading", "cabletwisting", "cablegrippers", "cableplastic"]

@MainActor
final class Playground: ObservableObject, GPUSimRendererSource {
    @Published var name: String
    @Published var running = true
    @Published var failure: String?
    @Published var parameters: [String: Float] = [:]
    var solver: GPUSolver!
    var dragSlot = 0
    var grab: (body: Int, local: F3)?
    var rendererSceneRevision = 0
    var rendererBodyAppearances: [Int: GPUSimRenderAppearance] = [:]
    var renderScene: (any GPUSimRenderableScene)? { solver }
    var rendererOptions: GPUSimRenderOptions {
        var options = GPUSimRenderOptions.lightweight
        options.showsGroundPlane = false
        return options
    }
    private var lastTime: Double = 0
    private var accumulator: Double = 0
    private var timestep: Double = 1 / 120

    init(name: String) throws {
        self.name = name
        try load()
    }

    func load() throws {
        var scene = Demos.make(name, params: parameters)!
        dragSlot = scene.addDragSlot()
        solver = try GPUSolver(scene: scene)
        timestep = Double(scene.settings.dt)
        rendererBodyAppearances = [:]
        for collider in scene.colliders {
            if scene.bodies[collider.body].isParticle, let color = collider.renderColor {
                rendererBodyAppearances[collider.body] = GPUSimRenderAppearance(color: color)
            }
        }
        grab = nil
        lastTime = 0
        accumulator = 0
        rendererSceneRevision += 1
        failure = nil
    }

    func reset() {
        do { try load() } catch { rendererDidFail(error.localizedDescription) }
    }

    func rendererWillDrawFrame() {
        let now = ProcessInfo.processInfo.systemUptime
        defer { lastTime = now }
        guard running, failure == nil, lastTime > 0 else { return }
        accumulator += min(now - lastTime, 0.05)
        do {
            while accumulator >= timestep {
                try solver.submitStep()
                accumulator -= timestep
            }
        } catch { rendererDidFail(error.localizedDescription) }
    }

    func rendererDidFail(_ message: String) { failure = message; running = false }

    func beginDrag(origin: F3, direction: F3) {
        grab = solver.pick(origin: origin, dir: direction).map { ($0.body, $0.local) }
        updateDrag(origin: origin, direction: direction)
    }

    func updateDrag(origin: F3, direction: F3) {
        guard let grab else { return }
        let anchor = solver.bodyPosition(grab.body) + solver.bodyRotation(grab.body).act(grab.local)
        let depth = max(dot(anchor - origin, direction), 1)
        solver.setDrag(jointIndex: dragSlot, body: grab.body,
            worldTarget: origin + direction * depth, localAnchor: grab.local,
            stiffness: 50 * max(1, solver.bodyMass(grab.body)))
    }

    func endDrag() {
        solver.setDrag(jointIndex: dragSlot, body: nil, worldTarget: .zero, localAnchor: .zero)
        grab = nil
    }
}

@MainActor
final class CableView: MTKView {
    var model: Playground!
    var renderer: GPUSimRenderer!
    var framedName = ""
    override var acceptsFirstResponder: Bool { true }

    func frameScene() {
        guard renderer != nil, framedName != model.name else { return }
        framedName = model.name
        renderer.automaticallyFramesScene = false
        renderer.azimuth = model.name == "cablegrippers" ? -2.25 : -1.85
        renderer.elevation = model.name == "cablegrippers" ? 0.24 : 0.5
        renderer.distance = model.name == "cablethreading" ? 4.8 : 4.2
        renderer.target = F3(-0.2, 0, (model.name == "cablegrippers" || model.name == "cableplastic") ? 1.5 : 0.8)
    }

    func ray(_ event: NSEvent) -> (F3, F3) {
        var p = convert(event.locationInWindow, from: nil)
        p.y = bounds.height - p.y
        return renderer.ray(at: p, in: bounds.size)
    }
    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        guard !event.modifierFlags.contains(.option) else { return }
        let (o, d) = ray(event)
        model.beginDrag(origin: o, direction: d)
    }
    override func mouseDragged(with event: NSEvent) {
        if event.modifierFlags.contains(.option) || model.grab == nil {
            renderer.azimuth -= Float(event.deltaX) * 0.008
            renderer.elevation = min(max(renderer.elevation + Float(event.deltaY) * 0.008, -1.4), 1.5)
        } else {
            let (o, d) = ray(event)
            model.updateDrag(origin: o, direction: d)
        }
    }
    override func mouseUp(with event: NSEvent) { model.endDrag() }
    override func rightMouseDragged(with event: NSEvent) {
        let scale = renderer.distance * 0.0015
        renderer.target -= F3(-sin(renderer.azimuth), cos(renderer.azimuth), 0) * Float(event.deltaX) * scale
        renderer.target.z += Float(event.deltaY) * scale
    }
    override func scrollWheel(with event: NSEvent) {
        renderer.distance = min(max(renderer.distance * (1 - Float(event.scrollingDeltaY) * 0.02), 0.6), 30)
    }
    override func keyDown(with event: NSEvent) {
        if event.keyCode == 49 { model.running.toggle() }
        else if event.charactersIgnoringModifiers == "r" { model.reset() }
        else { super.keyDown(with: event) }
    }
}

struct MetalCanvas: NSViewRepresentable {
    let model: Playground
    func makeNSView(context: Context) -> CableView {
        let view = CableView(frame: .zero, device: model.solver.device)
        view.model = model
        do {
            view.renderer = try GPUSimRenderer(device: model.solver.device, source: model)
            view.renderer.configure(view)
            view.frameScene()
        } catch { model.rendererDidFail(error.localizedDescription) }
        return view
    }
    func updateNSView(_ view: CableView, context: Context) { view.frameScene() }
}

struct PlaygroundContent: View {
    @ObservedObject var model: Playground
    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("CABLE LAB").font(.headline).tracking(3)
                    Picker("Scene", selection: $model.name) {
                        ForEach(demoNames, id: \.self) { Text(Demos.cableDemoTitle($0)!).tag($0) }
                    }.frame(maxWidth: 430).onChange(of: model.name) { _, _ in
                        model.parameters = [:]
                        model.reset()
                    }
                    Spacer()
                    Button(model.running ? "Pause" : "Play") { model.running.toggle() }
                    Button("Reset") { model.reset() }
                }
                Text(Demos.cableDemoInstructions(model.name) ?? "").font(.callout)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 320))], alignment: .leading, spacing: 8) {
                    ForEach(Demos.tunables(model.name)) { parameter in
                        HStack {
                            Text(parameter.label).font(.caption)
                            Slider(value: Binding(
                                get: { model.parameters[parameter.key] ?? parameter.def },
                                set: { model.parameters[parameter.key] = $0 }
                            ), in: parameter.range, onEditingChanged: { editing in
                                if !editing { model.reset() }
                            }).frame(width: 140)
                            Text(String(format: "%.2f", model.parameters[parameter.key] ?? parameter.def))
                                .font(.caption.monospacedDigit())
                        }
                    }
                    Spacer()
                }
                Text("Drag to grab · Option-drag to orbit · Right-drag to pan · Scroll to zoom · Space to pause · R to reset")
                    .font(.caption).foregroundStyle(.secondary)
                if let failure = model.failure { Text(failure).foregroundStyle(.red) }
            }.padding(16)
            MetalCanvas(model: model)
        }.frame(minWidth: 900, minHeight: 600)
    }
}

@MainActor
private func showWindow(_ model: Playground) {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)
    let menu = NSMenu()
    let item = NSMenuItem()
    menu.addItem(item)
    item.submenu = NSMenu()
    item.submenu?.addItem(withTitle: "Quit Cable Lab", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
    app.mainMenu = menu
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
        styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
    window.title = "AVBD Metal — Cable Lab"
    window.contentView = NSHostingView(rootView: PlaygroundContent(model: model))
    window.center()
    window.makeKeyAndOrderFront(nil)
    app.activate(ignoringOtherApps: true)
    app.run()
}

@MainActor
private func snapshot(_ model: Playground, path: String, steps: Int,
                      bendRelease: Bool) async throws {
    model.running = false
    if bendRelease && model.name == "cableplastic" {
        // Reproducible visual comparison using the same spring as mouse input.
        let scene = Demos.make(model.name, params: model.parameters)!
        for _ in 0..<240 { try model.solver.submitStep() }
        for cable in scene.cables {
            let body = cable.bodyIDs.last!
            let initial = model.solver.bodyPosition(body)
                + model.solver.bodyRotation(body).act(cable.endAnchor)
            for frame in 0..<300 {
                let t = min(Float(frame) / 180, 1)
                model.solver.setDrag(jointIndex: model.dragSlot, body: body,
                    worldTarget: initial + F3(0, -0.9 * t, -0.65 * t),
                    localAnchor: cable.endAnchor, stiffness: 50)
                for _ in 0..<4 { try model.solver.submitStep() }
            }
            model.endDrag()
        }
    }
    for _ in 0..<steps { try model.solver.submitStep() }
    try model.solver.synchronize()
    let renderer = try GPUSimRenderer(device: model.solver.device, source: model)
    renderer.options = model.rendererOptions
    renderer.automaticallyFramesScene = false
    if model.name == "cablegrippers" || model.name == "cableplastic" {
        renderer.setCamera(position: F3(-1.5, -4.0, 2.4), target: F3(-0.1, 0, 1.5), up: F3(0, 0, 1))
    } else {
        renderer.setCamera(position: F3(-2.3, -3.8, 3.0), target: F3(-0.25, 0, 0.8), up: F3(0, 0, 1))
    }
    let width = 1200, height = 800
    let view = MTKView(frame: NSRect(x: 0, y: 0, width: width, height: height), device: model.solver.device)
    renderer.configure(view)
    view.isPaused = true
    view.autoResizeDrawable = false
    view.drawableSize = CGSize(width: width, height: height)
    var pixels = [UInt8](repeating: 0, count: width * height * 4)
    var pending: CheckedContinuation<Void, Never>?
    renderer.frameCompletionHandler = { texture, _ in
        pixels.withUnsafeMutableBytes {
            texture.getBytes($0.baseAddress!, bytesPerRow: width * 4,
                from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        pending?.resume(); pending = nil
    }
    for _ in 0..<8 {
        await withCheckedContinuation { continuation in
            pending = continuation
            view.draw()
            if let failure = renderer.runtimeFailure { fatalError(failure) }
        }
    }
    for i in stride(from: 0, to: pixels.count, by: 4) { pixels.swapAt(i, i + 2) }
    let provider = CGDataProvider(data: Data(pixels) as CFData)!
    let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
        decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
        "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { fatalError("Cannot write snapshot: \(path)") }
    print("Saved \(path) after \(steps) simulation steps\(bendRelease ? " following bend/release" : "")")
}

@main struct CablePlayground {
    @MainActor static func main() async throws {
        let args = CommandLine.arguments
        if args.contains("--help") {
            print("cable-playground [cablethreading|cabletwisting|cablegrippers|cableplastic] [--snapshot /path/image.png --steps 120] [--bend-release]")
            return
        }
        let name = args.dropFirst().first(where: { demoNames.contains($0) }) ?? "cablegrippers"
        let model = try Playground(name: name)
        if let index = args.firstIndex(of: "--snapshot"), index + 1 < args.count {
            let steps = args.firstIndex(of: "--steps").flatMap { $0 + 1 < args.count ? Int(args[$0 + 1]) : nil } ?? 120
            try await snapshot(model, path: args[index + 1], steps: max(0, steps),
                               bendRelease: args.contains("--bend-release"))
        } else { showWindow(model) }
    }
}
#else
@main struct CablePlayground {
    static func main() { print("cable-playground runs on macOS; cable scenes also support iOS.") }
}
#endif
