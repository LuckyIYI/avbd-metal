import SwiftUI
import MetalKit
import AVBDCore
import simd

/// Spawns and supervises the ML pipeline CLI (collect/train/solve) so the
/// Lab can launch massively parallel sim + training runs with live logs.
@MainActor
final class TrainingRunner: ObservableObject {
    @Published var log = "ready — packaged runs use the bundled MLX trainer"
    @Published var running = false
    @Published var envs: Double = 128
    @Published var steps: Double = 900
    @Published var iters: Double = 3000
    @Published var batch: Double = 256
    @Published var latent: Double = 128
    @Published var lr: Double = 3e-4
    @Published var episodes: Double = 10
    private var proc: Process?

    static var workspaceURL: URL {
        let environment = ProcessInfo.processInfo.environment
        if let override = environment["AVBD_WORKSPACE"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        if Bundle.main.bundleURL.pathExtension == "app",
           let applicationSupport = FileManager.default.urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask).first {
            return applicationSupport.appendingPathComponent(
                "AVBD", isDirectory: true)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath,
                   isDirectory: true)
    }

    var binaryURL: URL? {
        let bundled = Bundle.main.bundleURL
            .appendingPathComponent("Contents/MacOS/avbd")
        let development = [
            URL(fileURLWithPath: ".xcbuild/Build/Products/Release/avbd"),
            URL(fileURLWithPath: ".build/release/avbd"),
        ]
        return ([bundled] + development).first {
            FileManager.default.isExecutableFile(atPath: $0.path)
        }
    }

    func launch(_ phase: String) {
        guard !running else { return }
        if phase == "train-wm" || phase == "solve-pusht" {
            do {
                try PolicyBridge.requireMLXRuntime()
            } catch {
                log = "trainer unavailable: \(error.localizedDescription)"
                return
            }
        }
        guard let binaryURL else {
            log = "trainer unavailable — build the ML-enabled app with `make app-ml`"
            return
        }
        var args: [String] = [phase]
        switch phase {
        case "collect":
            args += ["--envs", "\(Int(envs))", "--frames", "\(Int(steps))"]
        case "train-wm":
            args += ["--frames", "\(Int(iters))", "--batch", "\(Int(batch))",
                     "--latent", "\(Int(latent))", "--lr", "\(lr)"]
        case "solve-pusht", "oracle-pusht":
            // Solve reads the architecture contract written by train-wm;
            // oracle-pusht is non-neural and ignores latent dimensions.
            args += ["--episodes", "\(Int(episodes))"]
        default: break
        }
        let workspace = Self.workspaceURL
        do {
            try FileManager.default.createDirectory(
                at: workspace, withIntermediateDirectories: true)
        } catch {
            log = "workspace unavailable: \(error.localizedDescription)"
            return
        }
        let p = Process()
        p.executableURL = binaryURL
        p.arguments = args
        p.currentDirectoryURL = workspace
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let str = String(data: h.availableData, encoding: .utf8) ?? ""
            guard !str.isEmpty else { return }
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.log = String((self.log + str).suffix(2400))
            }
        }
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor in
                self?.running = false
                self?.log += "\n[\(phase) exited \(proc.terminationStatus)]"
            }
        }
        log = "$ avbd \(args.joined(separator: " "))\n"
            + "workspace: \(workspace.path)\n"
        do { try p.run(); running = true; proc = p }
        catch { log += "launch failed: \(error.localizedDescription)" }
    }

    func stop() { proc?.terminate() }
}

// The Robotics Lab: a dedicated view for manipulation experiments and for
// debugging learned world models — separate from the physics playground.

@MainActor
final class RoboticsModel: ObservableObject, RenderableModel {
    nonisolated let captureID = "robotics"
    let cameraEpoch = 0      // lab keeps its own camera; frame once at start
    enum DriveMode: String, CaseIterable {
        case manual = "Manual (IK mouse)"
        case oracle = "Oracle"
        case policy = "LeWM policy"
    }

    @Published var mode: DriveMode = .manual
    @Published var running = true
    @Published var pusherForceNote = "force-limited Cartesian actuator"
    @Published var speedLimit: Double = 0.16
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
    var policyAction: ((PushTEnv) throws -> SIMD2<Float>?)? = nil

    init() {
        reset()
        PolicyBridge.install(into: self)
    }

    func reset() {
        // In-place reset is valid only while the solver is healthy. A Metal
        // runtime failure is terminal for that GPUSolver, so Reset rebuilds
        // the one-environment scene instead of invoking legacy setters on a
        // poisoned solver.
        if let env, env.solver.runtimeFailure == nil {
            env.reset(0, seed: UInt64.random(in: 1...999_999_999))
        } else {
            do {
                env = try PushTEnv(
                    numEnvs: 1,
                    seed: UInt64.random(in: 1...999_999),
                    goalMarkers: true)
            } catch {
                env = nil
                running = false
                statsText = "Push-T scene unavailable: \(error.localizedDescription)"
                return
            }
        }
        tipTarget = SIMD2(1.4, 0)
        episodes += 1
        settleGrace = 12        // let contacts settle before control engages
        lastStepTime = CACurrentMediaTime()
    }



    private var settleGrace = 0

    func reportRenderFailure(_ message: String) {
        running = false
        stepAccumulator = 0
        statsText = "render stopped: \(message)"
    }

    func tickIfRunning() {
        guard running, let env, let solver else { return }
        if let failure = solver.runtimeFailure {
            running = false
            stepAccumulator = 0
            statsText = "Push-T simulation stopped: \(failure.localizedDescription)"
            return
        }
        let now = CACurrentMediaTime()
        let elapsed = min(now - lastStepTime, 0.1)
        lastStepTime = now
        stepAccumulator += elapsed
        // one CONTROL tick = 4 physics substeps, all through stepChecked's
        // rate-limited actuator path — same cadence the training pipeline
        // uses; driving the actuator per-substep made the oracle whip
        let controlDt = Double(solver.settings.dt) * 4
        var ticks = 0
        do {
            while stepAccumulator >= controlDt && ticks < 2 {
                env.tipSpeedLimit = Float(speedLimit)
                if settleGrace > 0 {
                    settleGrace -= 1
                    try env.stepChecked(
                        actions: [env.tipPos(0)], substeps: 4)
                } else {
                    let action: SIMD2<Float>
                    switch mode {
                    case .manual:
                        action = tipTarget
                    case .oracle:
                        action = env.oracleAction(0)
                    case .policy:
                        if let policyAction,
                           let act = try policyAction(env) {
                            action = act * 3
                        } else {
                            policyStatus = "policy unavailable: build with `make app-ml` and train a model first"
                            action = env.tipPos(0)
                        }
                    }
                    try env.stepChecked(actions: [action], substeps: 4)
                }
                stepAccumulator -= controlDt
                ticks += 1
            }
            if ticks == 2 { stepAccumulator = 0 }

            frame += 1
            if frame % 8 == 0 { try refreshDebug() }
            if env.success(0) && frame % 30 == 0 {
                solves += 1
                reset()
            }
        } catch {
            running = false
            stepAccumulator = 0
            statsText = "Push-T simulation stopped: \(error.localizedDescription)"
        }
    }

    private func refreshDebug() throws {
        guard let env else { return }
        let (bp, byaw) = env.blockPose(0)
        let r = env.refs[0]
        let tp = env.tipPos(0)
        statsText = String(format: """
            block (%.2f, %.2f)  yaw %.2f
            goal  (%.2f, %.2f)  dist %.2f
            tool head (%.2f, %.2f)
            episodes %d   solved %d
            """, bp.x, bp.y, byaw, r.goalPos.x, r.goalPos.y,
            length(bp - r.goalPos), tp.x, tp.y,
            episodes, solves)

        // what the model sees: the actual observation tensor as an image
        let res = env.obsRes
        let obs = try env.observationsChecked()
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
    @StateObject var trainer = TrainingRunner()

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
                    slider("Tool-head speed", $model.speedLimit, 0.04...0.4)
                    Toggle("Top-down camera", isOn: $model.topView)
                    Text(model.pusherForceNote).font(.caption2)
                        .foregroundStyle(.secondary)
                }

                if model.mode == .policy {
                    VStack(alignment: .leading, spacing: 5) {
                        Text(model.policyStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Button("Reload trained policy") {
                            PolicyBridge.install(into: model)
                        }
                    }
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

                GroupBox("Training (parallel sim + MLX)") {
                    slider("Parallel envs", $trainer.envs, 16...1024)
                    slider("Collect steps", $trainer.steps, 100...4000)
                    slider("Train iters", $trainer.iters, 200...20000)
                    slider("Batch size", $trainer.batch, 32...1024)
                    slider("Latent dim", $trainer.latent, 32...512)
                    slider("Eval episodes", $trainer.episodes, 5...50)
                    HStack {
                        Button("Collect") { trainer.launch("collect") }
                        Button("Train") { trainer.launch("train-wm") }
                        Button("Evaluate") { trainer.launch("solve-pusht") }
                    }.disabled(trainer.running)
                    HStack {
                        Button("Oracle eval") { trainer.launch("oracle-pusht") }
                            .disabled(trainer.running)
                        if trainer.running {
                            Button("Stop") { trainer.stop() }
                        }
                    }
                    ScrollView {
                        Text(trainer.log)
                            .font(.system(size: 9, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(height: 130)
                    .background(.black.opacity(0.85))
                    .foregroundStyle(.green)
                }

                Text("Manual mode: click/drag the floor —\nthe tool head chases your cursor.")
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
        let device = model.solver?.device ?? MTLCreateSystemDefaultDevice()
        let view = LabMTKView(frame: .zero, device: device)
        if ProcessInfo.processInfo.environment["AVBD_CAPTURE_VIEW"] == "robotics",
           ProcessInfo.processInfo.environment["AVBD_SHOT"] != nil
            || ProcessInfo.processInfo.environment["AVBD_VIDEO_DIR"] != nil {
            view.framebufferOnly = false
        }
        view.colorPixelFormat = Renderer.colorFormat
        view.depthStencilPixelFormat = .depth32Float
        view.sampleCount = Renderer.sampleCount
        view.preferredFramesPerSecond = 60
        view.coordinator = context.coordinator
        guard let device else {
            model.reportRenderFailure("no Metal device is available")
            view.isPaused = true
            return view
        }
        do {
            let renderer = try Renderer(device: device, model: model)
            context.coordinator.renderer = renderer
            view.delegate = renderer
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
        } catch {
            model.reportRenderFailure(error.localizedDescription)
            view.isPaused = true
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
