import SwiftUI
import MetalKit
import AVBDCore
import simd

// The Robotics Lab: a dedicated view for manipulation experiments and for
// debugging learned world models — separate from the physics playground.

@MainActor
final class RoboticsModel: ObservableObject, RenderableModel {
    enum DriveMode: String, CaseIterable {
        case manual = "Manual (IK mouse)"
        case oracle = "Oracle"
        case policy = "LeWM policy"
    }

    @Published var mode: DriveMode = .manual
    @Published var running = true
    @Published var motorTorque: Double = 260 { didSet { applyMotorSettings() } }
    @Published var speedLimit: Double = 0.05
    @Published var topView = false { didSet { applyCamera?() } }
    var applyCamera: (() -> Void)? = nil
    @Published var statsText = ""
    @Published var obsImage: CGImage? = nil
    @Published var episodes = 0
    @Published var solves = 0
    @Published var policyStatus = "policy: not loaded"
    var colorByGraphColor = false

    var env: PushTEnv?
    var solver: GPUSolver? { env?.solver }
    var tipTarget = SIMD2<Float>(1.4, 0)
    private var lastStepTime = CACurrentMediaTime()
    private var stepAccumulator: Double = 0
    private var frame = 0
    /// hook for the learned policy (installed by PolicyBridge when MLX works)
    var policyAction: ((PushTEnv) -> SIMD2<Float>?)? = nil

    init() {
        reset()
        PolicyBridge.install(into: self)
    }

    func reset() {
        // in-place: production-style episode reset, no scene rebuild
        if let env {
            env.reset(0, seed: UInt64.random(in: 1...999_999_999))
        } else {
            env = try? PushTEnv(numEnvs: 1, seed: UInt64.random(in: 1...999_999),
                                goalMarkers: true)
            applyMotorSettings()
        }
        tipTarget = SIMD2(1.4, 0)
        episodes += 1
        lastStepTime = CACurrentMediaTime()
    }

    func applyMotorSettings() {
        guard let env else { return }
        for j in env.refs[0].motorJoints {
            env.solver.setMotorTorque(j, torque: Float(motorTorque))
        }
    }

    func tickIfRunning() {
        guard running, let env, let solver else { return }
        let now = CACurrentMediaTime()
        let elapsed = min(now - lastStepTime, 0.1)
        lastStepTime = now
        stepAccumulator += elapsed
        let dt = Double(solver.settings.dt)
        var steps = 0
        while stepAccumulator >= dt && steps < 4 {
            env.jointSpeedLimit = Float(speedLimit)
            switch mode {
            case .manual:
                env.setTipTarget(0, tipTarget)
            case .oracle:
                env.setTipTarget(0, env.oracleAction(0))
            case .policy:
                if let act = policyAction?(env) {
                    env.setTipTarget(0, act * 2)
                } else {
                    policyStatus = "policy unavailable: build with `make app-ml` and train a model first"
                }
            }
            solver.step()
            stepAccumulator -= dt
            steps += 1
        }
        if steps == 4 { stepAccumulator = 0 }

        frame += 1
        if frame % 8 == 0 { refreshDebug() }
        if env.success(0) && frame % 30 == 0 {
            solves += 1
            reset()
        }
    }

    private func refreshDebug() {
        guard let env else { return }
        let (bp, byaw) = env.blockPose(0)
        let r = env.refs[0]
        let js = r.motorJoints
        let angles = js.map { env.solver.motorAngle($0) }
        statsText = String(format: """
            block (%.2f, %.2f)  yaw %.2f
            goal  (%.2f, %.2f)  dist %.2f
            joints  yaw %.2f  shoulder %.2f  elbow %.2f
            episodes %d   solved %d
            """, bp.x, bp.y, byaw, r.goalPos.x, r.goalPos.y,
            length(bp - r.goalPos), angles[0], angles[1], angles[2],
            episodes, solves)

        // what the model sees: the actual observation tensor as an image
        let res = env.obsRes
        let obs = env.observations()
        var pixels = [UInt8](repeating: 255, count: res * res * 4)
        for i in 0..<(res * res) {
            pixels[i * 4 + 0] = obs[i * 3 + 0]
            pixels[i * 4 + 1] = obs[i * 3 + 1]
            pixels[i * 4 + 2] = obs[i * 3 + 2]
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &pixels, width: res, height: res,
                            bitsPerComponent: 8, bytesPerRow: res * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        obsImage = ctx?.makeImage()
    }
}

struct RoboticsLabView: View {
    @ObservedObject var model: RoboticsModel

    var body: some View {
        HSplitView {
            RoboticsMetalView(model: model)
                .frame(minWidth: 600)
            controls
                .frame(width: 300)
                .padding(12)
        }
    }

    var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Robotics Lab").font(.title2).bold()

                GroupBox("Drive") {
                    Picker("Mode", selection: $model.mode) {
                        ForEach(RoboticsModel.DriveMode.allCases, id: \.self) {
                            Text($0.rawValue).tag($0)
                        }
                    }
                    .pickerStyle(.radioGroup)
                    HStack {
                        Button(model.running ? "Pause" : "Run") { model.running.toggle() }
                        Button("Reset episode") { model.reset() }
                    }
                }

                GroupBox("Robot settings") {
                    slider("Motor torque", $model.motorTorque, 60...900)
                    slider("Joint speed limit", $model.speedLimit, 0.01...0.15)
                    Toggle("Top-down camera", isOn: $model.topView)
                }

                if model.mode == .policy {
                    Text(model.policyStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                GroupBox("Model input (64\u{00D7}64)") {
                    if let img = model.obsImage {
                        Image(decorative: img, scale: 1)
                            .interpolation(.none)
                            .resizable()
                            .frame(width: 192, height: 192)
                            .border(.gray.opacity(0.4))
                    } else {
                        Text("—").frame(width: 192, height: 192)
                    }
                }

                Text(model.statsText)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)

                Text("Manual mode: click/drag the floor —\nthe arm chases your cursor.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
            }
        }
    }

    func slider(_ label: String, _ v: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption)
                Spacer()
                Text(String(format: "%.2f", v.wrappedValue)).font(.caption.monospacedDigit())
            }
            Slider(value: v, in: range)
        }
    }
}

// Metal viewport for the lab: mouse commands the tip (manual mode).
struct RoboticsMetalView: NSViewRepresentable {
    @ObservedObject var model: RoboticsModel

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    func makeNSView(context: Context) -> LabMTKView {
        let device = model.solver?.device ?? MTLCreateSystemDefaultDevice()!
        let view = LabMTKView(frame: .zero, device: device)
        view.colorPixelFormat = Renderer.colorFormat
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = Renderer.sampleCount
        view.preferredFramesPerSecond = 60
        let renderer = try! Renderer(device: device, model: model)
        context.coordinator.renderer = renderer
        view.delegate = renderer
        view.coordinator = context.coordinator
        model.applyCamera = { [weak renderer, weak model] in
            guard let r = renderer, let m = model else { return }
            if m.topView {
                r.elevation = 1.52
                r.azimuth = -.pi / 2
                r.distance = 9
            } else {
                r.elevation = 0.6
                r.azimuth = -.pi / 3
                r.distance = 10
            }
        }
        return view
    }

    func updateNSView(_ view: LabMTKView, context: Context) {}

    @MainActor
    final class Coordinator {
        let model: RoboticsModel
        var renderer: Renderer?
        init(model: RoboticsModel) { self.model = model }
    }
}

final class LabMTKView: MTKView {
    weak var coordinator: RoboticsMetalView.Coordinator?
    override var acceptsFirstResponder: Bool { true }

    private func command(_ event: NSEvent) {
        guard let c = coordinator, let r = c.renderer,
              let env = c.model.env else { return }
        var p = convert(event.locationInWindow, from: nil)
        p.y = bounds.height - p.y
        let (o, d) = r.ray(at: p, in: bounds.size)
        guard d.z < -1e-4 else { return }
        let t = (0.42 - o.z) / d.z
        let hit = o + d * t
        let local = hit - env.refs[0].center
        c.model.tipTarget = simd_clamp(SIMD2(local.x, local.y),
                                       SIMD2(-1.95, -1.95), SIMD2(1.95, 1.95))
    }

    override func mouseDown(with event: NSEvent) {
        if event.modifierFlags.contains(.option) { return }
        command(event)
    }

    override func mouseDragged(with event: NSEvent) {
        guard let r = coordinator?.renderer else { return }
        if event.modifierFlags.contains(.option) {
            r.azimuth -= Float(event.deltaX) * 0.008
            r.elevation = min(max(r.elevation + Float(event.deltaY) * 0.008, -1.5), 1.55)
        } else {
            command(event)
        }
    }

    override func scrollWheel(with event: NSEvent) {
        guard let r = coordinator?.renderer else { return }
        r.distance = min(max(r.distance * (1 - Float(event.scrollingDeltaY) * 0.02), 2), 400)
    }
}
