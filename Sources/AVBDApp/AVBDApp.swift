import SwiftUI
import MetalKit
import AVBDCore
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
                .frame(minWidth: 1000, minHeight: 640)
        }
    }
}

struct ContentView: View {
    @ObservedObject var model: SimulationModel

    var body: some View {
        HSplitView {
            MetalView(model: model)
                .frame(minWidth: 600)
            controls
                .frame(width: 280)
                .padding(12)
        }
    }

    var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("AVBD Metal").font(.title2).bold()

                GroupBox("Demo") {
                    Picker("Scene", selection: $model.demoName) {
                        ForEach(Demos.all, id: \.self) { Text($0) }
                    }
                    Picker("Size", selection: $model.scale) {
                        Text("Small").tag(1)
                        Text("Medium").tag(2)
                        Text("Large").tag(4)
                        Text("Giant").tag(8)
                        Text("Colossal").tag(16)
                    }
                    HStack {
                        Button(model.running ? "Pause" : "Play") {
                            model.running.toggle()
                        }
                        Button("Step") { model.singleStep() }
                        Button("Reset") { model.reset() }
                    }
                }

                GroupBox("Solver") {
                    labeledSlider("Iterations", $model.iterations, 1...30, "%.0f")
                    labeledSlider("α stabilization", $model.alpha, 0...1, "%.2f")
                    labeledSlider("β stiffness ramp", $model.betaLin, 10...100000, "%.0f", log: true)
                    labeledSlider("γ warm start", $model.gamma, 0...0.9999, "%.4f")
                    labeledSlider("Gravity", $model.gravity, -30...0, "%.1f")
                    labeledSlider("Time scale", $model.timeScale, 0.05...2, "%.2f")
                }

                GroupBox("Display") {
                    Toggle("Color by graph color", isOn: $model.colorByGraphColor)
                }

                Text(model.statsText)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)

                Text("Drag bodies with the mouse.\n⌥-drag orbits, scroll zooms.")
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

// MARK: - Metal view with orbit/drag interaction

struct MetalView: NSViewRepresentable {
    @ObservedObject var model: SimulationModel

    func makeCoordinator() -> Coordinator {
        Coordinator(model: model)
    }

    func makeNSView(context: Context) -> InteractiveMTKView {
        let device = model.solver?.device ?? MTLCreateSystemDefaultDevice()!
        let view = InteractiveMTKView(frame: .zero, device: device)
        view.colorPixelFormat = Renderer.colorFormat
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = Renderer.sampleCount
        view.preferredFramesPerSecond = 60
        let renderer = try! Renderer(device: device, model: model)
        context.coordinator.renderer = renderer
        view.delegate = renderer
        view.coordinator = context.coordinator
        return view
    }

    func updateNSView(_ view: InteractiveMTKView, context: Context) {}

    final class Coordinator {
        let model: SimulationModel
        var renderer: Renderer?
        var dragging = false

        init(model: SimulationModel) {
            self.model = model
        }
    }
}

final class InteractiveMTKView: MTKView {
    weak var coordinator: MetalView.Coordinator?

    override var acceptsFirstResponder: Bool { true }

    private func rayAt(_ event: NSEvent) -> (F3, F3)? {
        guard let r = coordinator?.renderer else { return nil }
        var p = convert(event.locationInWindow, from: nil)
        p.y = bounds.height - p.y
        return r.ray(at: p, in: bounds.size)
    }

    override func mouseDown(with event: NSEvent) {
        guard let c = coordinator else { return }
        if event.modifierFlags.contains(.option) { return }
        if let (o, d) = rayAt(event) {
            c.model.beginDrag(origin: o, dir: d)
            c.dragging = true
        }
    }

    override func mouseDragged(with event: NSEvent) {
        guard let c = coordinator, let r = c.renderer else { return }
        if event.modifierFlags.contains(.option) || !c.dragging {
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
        guard let r = coordinator?.renderer else { return }
        // pan target in view plane
        let s = r.distance * 0.0015
        let azim = r.azimuth
        let right = F3(-sin(azim), cos(azim), 0)
        let upish = F3(0, 0, 1)
        r.target -= right * Float(event.deltaX) * s
        r.target += upish * Float(event.deltaY) * s
    }

    override func scrollWheel(with event: NSEvent) {
        guard let r = coordinator?.renderer else { return }
        r.distance = min(max(r.distance * (1 - Float(event.scrollingDeltaY) * 0.02), 2), 400)
    }
}
