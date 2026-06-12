import Foundation
import QuartzCore
import AVBDCore
import simd

/// Observable simulation state driving the UI and the renderer.
final class SimulationModel: ObservableObject, RenderableModel {
    @Published var demoName = ProcessInfo.processInfo.environment["AVBD_DEMO"] ?? "stack" {
        didSet {
            demoParams = Self.defaultParams(for: demoName)
            reset(adoptSceneDefaults: true)   // new demo = its own tuning
        }
    }
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
    @Published var colorByGraphColor = false

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

    init() {
        reset(adoptSceneDefaults: true)
    }

    /// Rebuild the scene. By default the USER'S current solver sliders and
    /// demo parameters survive the restart — only switching to a different
    /// demo adopts that scene's own tuned defaults (each demo is tuned;
    /// e.g. the castle needs its own iterations/beta).
    func reset(adoptSceneDefaults: Bool = false) {
        guard var scene = Demos.make(demoName, scale: scale, params: demoParams)
        else { return }
        dragJoint = scene.addDragSlot()
        if adoptSceneDefaults {
            adopting = true
            iterations = Double(scene.settings.iterations)
            alpha = Double(scene.settings.alpha)
            betaLin = Double(scene.settings.betaLin)
            gamma = Double(scene.settings.gamma)
            gravity = Double(scene.settings.gravity)
            adopting = false
        } else {
            // restart with the user's current settings, as set up in the UI
            scene.settings.iterations = Int(iterations)
            scene.settings.alpha = Float(alpha)
            scene.settings.betaLin = Float(betaLin)
            scene.settings.gamma = Float(gamma)
            scene.settings.gravity = Float(gravity)
        }
        do {
            solver = try GPUSolver(scene: scene)
        } catch {
            statsText = "solver init failed: \(error)"
            solver = nil
        }
        dragBody = nil
        lastStepTime = CACurrentMediaTime()
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
        while stepAccumulator >= dt && steps < 4 {
            let t0 = CACurrentMediaTime()
            solver.step()
            let ms = (CACurrentMediaTime() - t0) * 1000
            msEMA = msEMA == 0 ? ms : msEMA * 0.95 + ms * 0.05
            stepAccumulator -= dt
            steps += 1
        }
        if steps == 4 { stepAccumulator = 0 }  // avoid spiral of death

        frameCounter += 1
        if frameCounter % 15 == 0 {
            let colors = solver.lastColorCounts.filter { $0 > 0 }.count
            statsText = String(
                format: "%d bodies   %.2f ms/step   %d pairs   %d colors",
                solver.bodyCount, msEMA, solver.lastNumPairs, colors)

        }
    }

    func singleStep() {
        running = false
        solver?.step()
    }

    // MARK: - Mouse dragging

    func beginDrag(origin: F3, dir: F3) {
        guard let solver else { return }
        guard let (body, local) = solver.pick(origin: origin, dir: dir) else { return }
        dragBody = body
        dragLocal = local
        updateDrag(origin: origin, dir: dir)
    }

    func updateDrag(origin: F3, dir: F3) {
        guard let solver, let body = dragBody else { return }
        // target: keep the grabbed point at fixed distance along the ray
        let anchor = solver.bodyPosition(body) + solver.bodyRotation(body).act(dragLocal)
        let depth = dot(anchor - origin, dir)
        let target = origin + dir * max(depth, 1)
        solver.setDrag(jointIndex: dragJoint, body: body,
                       worldTarget: target, localAnchor: dragLocal,
                       stiffness: 50 * max(1, solverMass(body)))
    }

    func endDrag() {
        guard let solver else { return }
        solver.setDrag(jointIndex: dragJoint, body: nil, worldTarget: .zero, localAnchor: .zero)
        dragBody = nil
    }

    private func solverMass(_ body: Int) -> Float {
        solver?.bodyMass(body) ?? 1
    }
}
