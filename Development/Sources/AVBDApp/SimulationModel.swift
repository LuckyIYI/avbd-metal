import Foundation
import QuartzCore
import SimCore
import PhysicsAVBD
import GPUSimDemos
import Robotics
import RL
import MLXRL
import simd

/// Observable simulation state driving the UI and the renderer.
final class SimulationModel: ObservableObject, RenderableModel {
    nonisolated let captureID = "playground"
    @Published var demoName = ProcessInfo.processInfo.environment["AVBD_DEMO"] ?? "stack" {
        didSet {
            demoParams = Self.defaultParams(for: demoName)
            cameraEpoch += 1                  // reframe only on demo change
            reset(adoptSceneDefaults: true)   // new demo = its own tuning
        }
    }
    /// Bumped only when the demo changes; the renderer reframes the camera
    /// on epoch change (reset/size/param rebuilds keep the user's view).
    private(set) var cameraEpoch = 0
    @Published var scale = Int(ProcessInfo.processInfo.environment["AVBD_SIZE"] ?? "") ?? 1 {
        didSet { reset() }
    }
    /// Per-demo tunables (friction etc.) — edited via the expandable
    /// "Demo parameters" section; applied on scene rebuild.
    @Published var demoParams: [String: Float] = SimulationModel.defaultParams(
        for: ProcessInfo.processInfo.environment["AVBD_DEMO"] ?? "stack")

    static func defaultParams(for name: String) -> [String: Float] {
        Demos.tunables(name).reduce(into: [:]) { $0[$1.key] = $1.def }
    }
    @Published var running = true
    @Published var rayTracingEnabled = false
    @Published var screenSpaceReflectionsEnabled = false
    @Published var colorByGraphColor = false
    @Published var showConvexCollisionGeometry = false
    @Published var convexCollisionWireframe = true

    // Live-tunable solver settings
    @Published var iterations: Double = 10 { didSet { push() } }
    @Published var alpha: Double = 0.99 { didSet { push() } }
    @Published var betaLin: Double = 5000 { didSet { push() } }
    @Published var gamma: Double = 0.999 { didSet { push() } }
    @Published var gravity: Double = -10 { didSet { push() } }
    private var adopting = false
    @Published var timeScale: Double = 1.0

    // Stats (updated as sim runs)
    @Published var statsText = ""

    private(set) var solver: GPUSolver?
    private(set) var dragJoint: Int = -1
    private var dragBody: Int? = nil
    private var dragLocal = F3.zero
    private var stepAccumulator: Double = 0
    private var lastStepTime: Double = CACurrentMediaTime()
    private var msEMA: Double = 0
    private var frameCounter = 0
    private var resetGeneration = 0

    init() {
        reset(adoptSceneDefaults: true)
    }

    /// Rebuild the scene. By default the USER'S current solver sliders and
    /// demo parameters survive the restart — only switching to a different
    /// demo adopts that scene's own tuned defaults (each demo is tuned;
    /// e.g. the castle needs its own iterations/beta).
    func reset(adoptSceneDefaults: Bool = false) {
        resetGeneration += 1
        let generation = resetGeneration
        let name = demoName
        let sceneScale = scale
        let params = demoParams
        let current = (iterations: iterations, alpha: alpha, betaLin: betaLin,
                       gamma: gamma, gravity: gravity)
        solver = nil
        dragJoint = -1
        dragBody = nil
        statsText = "loading..."
        lastStepTime = CACurrentMediaTime()

        DispatchQueue.global(qos: .userInitiated).async {
            guard var scene = Demos.make(name, scale: sceneScale, params: params) else {
                DispatchQueue.main.async {
                    guard generation == self.resetGeneration else { return }
                    self.statsText = "unknown demo: \(name)"
                }
                return
            }
            let dragSlot = scene.addDragSlot()
            if !adoptSceneDefaults {
                // restart with the user's current settings, as set up in the UI
                scene.settings.iterations = Int(current.iterations)
                scene.settings.alpha = Float(current.alpha)
                scene.settings.betaLin = Float(current.betaLin)
                scene.settings.gamma = Float(current.gamma)
                scene.settings.gravity = Float(current.gravity)
            }
            let adoptedSettings = scene.settings
            let result: Result<GPUSolver, Error> = Result {
                try GPUSolver(scene: scene)
            }
            DispatchQueue.main.async {
                guard generation == self.resetGeneration else { return }
                switch result {
                case .success(let newSolver):
                    if adoptSceneDefaults {
                        self.adopting = true
                        self.iterations = Double(adoptedSettings.iterations)
                        self.alpha = Double(adoptedSettings.alpha)
                        self.betaLin = Double(adoptedSettings.betaLin)
                        self.gamma = Double(adoptedSettings.gamma)
                        self.gravity = Double(adoptedSettings.gravity)
                        self.adopting = false
                    }
                    self.solver = newSolver
                    self.dragJoint = dragSlot
                    self.dragBody = nil
                    self.stepAccumulator = 0
                    self.lastStepTime = CACurrentMediaTime()
                    self.statsText = ""
                case .failure(let error):
                    self.statsText = "solver init failed: \(error)"
                    self.solver = nil
                    self.dragJoint = -1
                    self.dragBody = nil
                }
            }
        }
    }

    private func push() {
        guard !adopting, let solver else { return }
        solver.settings.iterations = Int(iterations)
        solver.settings.alpha = Float(alpha)
        solver.settings.betaLin = Float(betaLin)
        solver.settings.gamma = Float(gamma)
        solver.settings.gravity = Float(gravity)
    }

    /// Called from the renderer each display frame.
    func tickIfRunning() {
        guard running, let solver else { return }
        let now = CACurrentMediaTime()
        let elapsed = min(now - lastStepTime, 0.1) * timeScale
        lastStepTime = now
        stepAccumulator += elapsed

        let dt = Double(solver.settings.dt)
        var steps = 0
        do {
            let batchStart = CACurrentMediaTime()
            while stepAccumulator >= dt && steps < 4 {
                try solver.submitStep()
                stepAccumulator -= dt
                steps += 1
            }
            // Physics and rendering use separate Metal queues. Retiring the
            // submitted steps here is the explicit cross-queue boundary and
            // prevents rendering partially updated shared poses.
            try solver.synchronize()
            if steps > 0 {
                let ms = (CACurrentMediaTime() - batchStart) * 1000
                    / Double(steps)
                msEMA = msEMA == 0 ? ms : msEMA * 0.95 + ms * 0.05
            }
        } catch {
            running = false
            statsText = "solver stopped: \(error.localizedDescription)"
            return
        }
        if steps == 4 { stepAccumulator = 0 }  // avoid spiral of death

        frameCounter += 1
        if frameCounter % 15 == 0 {
            let colors = solver.lastColorCounts.filter { $0 > 0 }.count
            statsText = String(
                format: "%d bodies   %.2f ms/step   %d pairs   %d colors",
                solver.bodyCount, msEMA, solver.lastNumPairs, colors)
            if solver.uniqueConvexAssetCount > 0 {
                statsText += "   \(solver.uniqueConvexAssetCount) shared hulls"
            }

        }
    }

    func singleStep() {
        running = false
        guard let solver else { return }
        do {
            try solver.submitStep()
            try solver.synchronize()
        } catch {
            statsText = "solver stopped: \(error.localizedDescription)"
        }
    }

    // MARK: - Mouse dragging

    func beginDrag(origin: F3, dir: F3) {
        guard let solver, solver.runtimeFailure == nil else {
            dragBody = nil
            return
        }
        guard let (body, local) = solver.pick(origin: origin, dir: dir) else { return }
        dragBody = body
        dragLocal = local
        updateDrag(origin: origin, dir: dir)
    }

    func updateDrag(origin: F3, dir: F3) {
        guard let solver, solver.runtimeFailure == nil,
              let body = dragBody else {
            dragBody = nil
            return
        }
        // target: keep the grabbed point at fixed distance along the ray
        let anchor = solver.bodyPosition(body) + solver.bodyRotation(body).act(dragLocal)
        let depth = dot(anchor - origin, dir)
        let target = origin + dir * max(depth, 1)
        solver.setDrag(jointIndex: dragJoint, body: body,
                       worldTarget: target, localAnchor: dragLocal,
                       stiffness: 50 * max(1, solverMass(body)))
    }

    func endDrag() {
        defer { dragBody = nil }
        guard let solver, solver.runtimeFailure == nil else { return }
        solver.setDrag(jointIndex: dragJoint, body: nil, worldTarget: .zero, localAnchor: .zero)
    }

    private func solverMass(_ body: Int) -> Float {
        solver?.bodyMass(body) ?? 1
    }
}
