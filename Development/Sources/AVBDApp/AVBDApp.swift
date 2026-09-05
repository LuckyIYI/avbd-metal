import SwiftUI
import MetalKit
import SimCore
import PhysicsAVBD
import GPUSimDemos
import GPUSimRenderer
import Robotics
import RL
import MLXRL
import simd

@main
struct AVBDApp: App {
    @StateObject var model = SimulationModel()

    init() {
        // Make a bare SPM executable behave like a real app: dock icon,
        // key window, foreground activation.
        NSApplication.shared.setActivationPolicy(.regular)
        DispatchQueue.main.async {
            NSApplication.shared.activate(ignoringOtherApps: true)
        }
    }

    var body: some SwiftUI.Scene {
        WindowGroup("AVBD Metal") {
            ContentView(model: model)
                // Policy Replay reserves enough room for both an inspectable
                // robot viewport and its non-clipped 520-point control panel.
                .frame(minWidth: 1200, minHeight: 680)
        }
    }
}

struct ContentView: View {
    @ObservedObject var model: SimulationModel
    @State private var selectedTab = ProcessInfo.processInfo.environment[
        "AVBD_POLICY_REPLAY"] == nil ? 0 : 1

    var body: some View {
        TabView(selection: $selectedTab) {
            HSplitView {
                MetalView(model: model)
                    .frame(minWidth: 600)
                controls
                    .frame(width: 280)
                    .padding(12)
            }
            .tabItem { Text("Playground") }
            .tag(0)
            PolicyReplayLabView()
                .tabItem { Text("Policy Replay") }
                .tag(1)
        }
    }

    var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("AVBD Metal").font(.title2).bold()
                RenderingModePicker(rayTracingEnabled: $model.rayTracingEnabled,
                                    supportsRayTracing: model.supportsRayTracing)

                GroupBox("Demo") {
                    Picker("Scene", selection: $model.demoName) {
                        ForEach(Demos.all, id: \.self) {
                            Text($0 == "gaudifunicular" ? "Gaudí Funicular"
                                : $0 == "classicrigids" ? "Classic Rigid Bodies"
                                : $0 == "boxofboxes" ? "Box of Boxes"
                                : $0)
                        }
                    }
                    if Demos.supportsScale(model.demoName) {
                        Picker("Size", selection: $model.scale) {
                            Text("Small").tag(1)
                            Text("Medium").tag(2)
                            Text("Large").tag(4)
                            Text("Giant").tag(8)
                            Text("Colossal").tag(16)
                        }
                    }
                    HStack {
                        Button(model.running ? "Pause" : "Play") {
                            model.running.toggle()
                        }
                        Button("Step") { model.singleStep() }
                        Button("Reset") { model.reset() }
                    }
                    let tunables = Demos.tunables(model.demoName)
                    if !tunables.isEmpty {
                        DisclosureGroup("Demo parameters") {
                            ForEach(tunables) { p in
                                HStack {
                                    Text(p.label).font(.caption)
                                    Slider(value: Binding(
                                        get: { model.demoParams[p.key] ?? p.def },
                                        set: { model.demoParams[p.key] = $0 }
                                    ), in: p.range, onEditingChanged: { editing in
                                        if !editing { model.reset() }
                                    })
                                    Text(String(format: "%.2f",
                                                model.demoParams[p.key] ?? p.def))
                                        .font(.caption.monospacedDigit())
                                        .frame(width: 44, alignment: .trailing)
                                }
                            }
                        }
                    }
                }

                if model.demoName == "gaudifunicular" {
                    GroupBox("Gaudí's gravity computer") {
                        VStack(alignment: .leading, spacing: 7) {
                            Text("TENSION MODEL")
                                .font(.caption2.weight(.bold))
                                .foregroundStyle(.orange)
                            Text("The overhead lattice is the foundation plan. Gravity pulls low-mass cord and weighted shot bags into pure-tension load paths. Invert those paths and they become compression-only columns and vaults.")
                                .font(.caption)
                            Divider()
                            Text("Blueprint basis")
                                .font(.caption.weight(.semibold))
                            Text("7.5 m module · 90 m interior\n45 m naves · 60 m transept\n15 m central nave")
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(.secondary)
                            Text("Historically, the physical polyfunicular model was for Colònia Güell. Gaudí applied its corresponding graphical method at Sagrada Família.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Text("Try it: drag a bag or knot and release it. The whole network recomputes the force geometry under gravity.")
                                .font(.caption2.weight(.medium))
                            Text("Damping is explicit in Demo parameters. Set both damping sliders to 0 for an undamped run.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }

                GroupBox("Solver") {
                    labeledSlider("Iterations", $model.iterations, 1...60, "%.0f")
                    labeledSlider("α stabilization", $model.alpha, 0...1, "%.2f")
                    labeledSlider("β stiffness ramp", $model.betaLin, 10...100000, "%.0f", log: true)
                    labeledSlider("γ warm start", $model.gamma, 0...1, "%.4f")
                    labeledSlider("Gravity", $model.gravity, -30...0, "%.1f")
                    labeledSlider("Time scale", $model.timeScale, 0.05...2, "%.2f")
                }

                GroupBox("Display") {
                    Toggle("Color by graph color", isOn: $model.colorByGraphColor)
                    Toggle("Collision hulls",
                           isOn: $model.showConvexCollisionGeometry)
                    Toggle("Hull wireframe",
                           isOn: $model.convexCollisionWireframe)
                        .disabled(!model.showConvexCollisionGeometry)
                }

                Text(model.statsText)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)

                Text("Drag bodies with the mouse.\n⌥-drag orbits, scroll zooms,\nspace/right-drag pans.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                Spacer()
            }
        }
    }

    func labeledSlider(_ label: String, _ value: Binding<Double>,
                       _ range: ClosedRange<Double>, _ fmt: String,
                       log: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption)
                Spacer()
                Text(String(format: fmt, value.wrappedValue)).font(.caption.monospacedDigit())
            }
            if log {
                Slider(value: Binding(
                    get: { Foundation.log10(value.wrappedValue) },
                    set: { value.wrappedValue = pow(10, $0) }
                ), in: Foundation.log10(range.lowerBound)...Foundation.log10(range.upperBound))
            } else {
                Slider(value: value, in: range)
            }
        }
    }
}

struct RenderingModePicker: View {
    @Binding var rayTracingEnabled: Bool
    let supportsRayTracing: Bool

    var body: some View {
        GroupBox("Rendering") {
            VStack(alignment: .leading, spacing: 6) {
                Picker("Rendering", selection: $rayTracingEnabled) {
                    Text("Fast").tag(false)
                    Text("RT Beta").tag(true)
                }
                .pickerStyle(.segmented)
                .disabled(!supportsRayTracing)
                Text(!supportsRayTracing ? "Ray tracing is unavailable on this Mac."
                     : rayTracingEnabled ? "Ray-traced shadows, reflections and bounced light. Experimental."
                     : "Raster shadows and ambient occlusion.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

// MARK: - Metal view with orbit/drag interaction

struct MetalView: NSViewRepresentable {
    @ObservedObject var model: SimulationModel

    func makeCoordinator() -> Coordinator {
        Coordinator(model: model)
    }

    func makeNSView(context: Context) -> InteractiveMTKView {
        let device = model.solver?.device ?? MTLCreateSystemDefaultDevice()
        let view = InteractiveMTKView(frame: .zero, device: device)
        if ProcessInfo.processInfo.environment["AVBD_SHOT"] != nil
            || ProcessInfo.processInfo.environment["AVBD_VIDEO_DIR"] != nil {
            view.framebufferOnly = false
        }
        guard let device else {
            model.reportRenderFailure("no Metal device is available")
            view.isPaused = true
            view.coordinator = context.coordinator
            return view
        }
        do {
            let renderer = try GPUSimRenderer(device: device, source: model)
            configureAppCapture(renderer: renderer, model: model)
            renderer.configure(view, preferredFramesPerSecond: 60)
            context.coordinator.renderer = renderer
        } catch {
            model.reportRenderFailure(error.localizedDescription)
            view.isPaused = true
        }
        view.coordinator = context.coordinator
        return view
    }

    func updateNSView(_ view: InteractiveMTKView, context: Context) {}

    final class Coordinator {
        let model: SimulationModel
        var renderer: GPUSimRenderer?
        var dragging = false

        init(model: SimulationModel) {
            self.model = model
        }
    }
}

final class InteractiveMTKView: MTKView {
    weak var coordinator: MetalView.Coordinator?
    private var spaceDown = false

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 49 { spaceDown = true } else { super.keyDown(with: event) }
    }

    override func keyUp(with event: NSEvent) {
        if event.keyCode == 49 { spaceDown = false } else { super.keyUp(with: event) }
    }

    private func pan(_ event: NSEvent) {
        guard let r = coordinator?.renderer else { return }
        let s = r.distance * 0.0015
        let right = F3(-sin(r.azimuth), cos(r.azimuth), 0)
        r.target -= right * Float(event.deltaX) * s
        r.target += F3(0, 0, 1) * Float(event.deltaY) * s
    }

    private func rayAt(_ event: NSEvent) -> (F3, F3)? {
        guard let r = coordinator?.renderer else { return nil }
        var p = convert(event.locationInWindow, from: nil)
        p.y = bounds.height - p.y
        return r.ray(at: p, in: bounds.size)
    }

    override func mouseDown(with event: NSEvent) {
        guard let c = coordinator else { return }
        if event.modifierFlags.contains(.option) || spaceDown { return }
        if let (o, d) = rayAt(event) {
            c.model.beginDrag(origin: o, dir: d)
            c.dragging = true
        }
    }

    override func mouseDragged(with event: NSEvent) {
        guard let c = coordinator, let r = c.renderer else { return }
        if spaceDown {
            pan(event)
        } else if event.modifierFlags.contains(.option) || !c.dragging {
            r.azimuth -= Float(event.deltaX) * 0.008
            r.elevation = min(max(r.elevation + Float(event.deltaY) * 0.008, -1.5), 1.55)
        } else if let (o, d) = rayAt(event) {
            c.model.updateDrag(origin: o, dir: d)
        }
    }

    override func mouseUp(with event: NSEvent) {
        coordinator?.model.endDrag()
        coordinator?.dragging = false
    }

    override func rightMouseDragged(with event: NSEvent) {
        pan(event)
    }

    override func scrollWheel(with event: NSEvent) {
        guard let r = coordinator?.renderer else { return }
        r.distance = min(max(r.distance * (1 - Float(event.scrollingDeltaY) * 0.02), 2), 400)
    }
}
