import AppKit
import Combine
import Foundation
import MetalKit
import SwiftUI
import UniformTypeIdentifiers
import SimCore
import PhysicsAVBD
import GPUSimRenderer
import RL
import MLXRL

@MainActor
final class PolicyReplayModel: ObservableObject, RenderableModel {
    struct Page: Identifiable {
        let id: String
        let bundle: LoadedPolicyBundle
        let release: PolicyBundleReleaseIndex.Release?
        let isPackaged: Bool

        var trust: PolicyBundleTrust {
            guard isPackaged, let release else { return .importedUnverified }
            switch release.qualification {
            case .accepted: return .qualified
            case .externalParityVerified: return .externalParityVerified
            }
        }
    }

    nonisolated let captureID = "policy"

    @Published private(set) var pages = [Page]()
    @Published var selectedPageID = "" {
        didSet {
            guard selectedPageID != oldValue else { return }
            UserDefaults.standard.set(
                selectedPageID, forKey: "AVBDPolicyBundlePage")
            loadSelectedPage()
        }
    }
    @Published private(set) var trust = PolicyBundleTrust.importedUnverified
    @Published private(set) var status = "Discovering policy bundles…"
    @Published private(set) var interactionStatus = ""
    @Published private(set) var statsText = ""
    @Published var running = true
    @Published var rayTracingEnabled = false
    @Published var metalFXEnabled = false
    @Published var playbackRate: Double = 1
    @Published var selectedCameraID = "" {
        didSet {
            guard selectedCameraID != oldValue else { return }
            cameraEpoch += 1
        }
    }
    @Published var controlValues = [String: Float]()
    @Published var isImporting = false

    private(set) var session: (any PolicyBundleReplaySession)?
    private var releaseIndex: PolicyBundleReleaseIndex?
    private var packagedRoot: URL?
    private var lastTime = CACurrentMediaTime()
    private var accumulator = 0.0

    var solver: GPUSolver? { session?.solver }
    var colorByGraphColor: Bool { false }
    var cameraEpoch = 0
    var selectedPage: Page? { pages.first { $0.id == selectedPageID } }
    var manifest: PolicyBundleManifest? { selectedPage?.bundle.manifest }
    var hasReplayScene: Bool { session != nil }

    var selectedCamera: PolicyBundleManifest.CameraPreset? {
        let cameras = manifest?.presentation.cameraPresets ?? []
        return cameras.first { $0.id == selectedCameraID } ?? cameras.first
    }

    var cameraTarget: F3 {
        guard let camera = selectedCamera else { return .zero }
        return PolicyBundleReplayFactory.cameraTarget(
            preset: camera,
            anchorValue: session?.anchor(named: camera.anchor)) ?? .zero
    }

    init() {
        playbackRate = ProcessInfo.processInfo.environment[
            "AVBD_REPLAY_RATE"].flatMap(Double.init) ?? 1
        refreshLibrary()
    }

    func refreshLibrary(selecting preferredDirectory: URL? = nil) {
        do {
            let roots = try discoveryRoots()
            packagedRoot = roots.packaged
            releaseIndex = try roots.releaseIndex.flatMap(Self.loadReleaseIndex)
            var discovered = [LoadedPolicyBundle]()
            var seen = Set<String>()
            for root in roots.bundleRoots where seen.insert(root.path).inserted {
                discovered.append(contentsOf: PolicyBundleLoader.discover(
                    beneath: root))
            }
            if let explicit = ProcessInfo.processInfo.environment[
                    "AVBD_POLICY_BUNDLE"] {
                let bundle = try PolicyBundleLoader.load(
                    directory: URL(fileURLWithPath: explicit,
                                   isDirectory: true))
                if !discovered.contains(where: {
                    $0.directory == bundle.directory
                }) {
                    discovered.append(bundle)
                }
            }
            var ids = Set<String>()
            var nextPages = [Page]()
            for bundle in discovered {
                let id = bundle.directory.standardizedFileURL.path
                guard ids.insert(id).inserted else { continue }
                let packaged = roots.packaged.map {
                    Self.isDescendant(bundle.directory, of: $0)
                } ?? false
                let release: PolicyBundleReleaseIndex.Release?
                if packaged {
                    guard let root = roots.packaged,
                          let index = releaseIndex,
                          let candidate = index.release(for: bundle),
                          Self.relativePath(
                            bundle.directory, beneath: root)
                            == candidate.bundleRelativeDirectory else {
                        throw PolicyBundleError.invalid(
                            "packaged bundle '\(bundle.manifest.identifier)' "
                                + "is not authenticated by its release index")
                    }
                    release = candidate
                } else {
                    release = nil
                }
                nextPages.append(Page(
                    id: id, bundle: bundle, release: release,
                    isPackaged: packaged))
            }
            pages = nextPages.sorted {
                if $0.bundle.manifest.title == $1.bundle.manifest.title {
                    return $0.id < $1.id
                }
                return $0.bundle.manifest.title < $1.bundle.manifest.title
            }
            guard !pages.isEmpty else {
                session = nil
                selectedPageID = ""
                status = "No policy bundles found. Choose Import Bundle to add one."
                return
            }
            let preferred = preferredDirectory?.standardizedFileURL.path
                ?? ProcessInfo.processInfo.environment["AVBD_POLICY_BUNDLE"]
                ?? UserDefaults.standard.string(
                    forKey: "AVBDPolicyBundlePage")
            let next = pages.first { $0.id == preferred }?.id
                ?? pages.first!.id
            if selectedPageID == next {
                loadSelectedPage()
            } else {
                selectedPageID = next
            }
        } catch {
            pages = []
            session = nil
            trust = .importedUnverified
            controlValues = [:]
            status = "bundle discovery failed: \(error.localizedDescription)"
            running = false
        }
    }

    func importBundle(from source: URL) {
        let securityScoped = source.startAccessingSecurityScopedResource()
        defer { if securityScoped { source.stopAccessingSecurityScopedResource() } }
        do {
            let candidate = try PolicyBundleLoader.load(directory: source)
            let root = try Self.importedBundleRoot()
            try FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            let name = candidate.manifest.identifier + "-"
                + String(candidate.contentSHA256.prefix(12))
            let destination = root.appendingPathComponent(name, isDirectory: true)
            if !FileManager.default.fileExists(atPath: destination.path) {
                let staging = root.appendingPathComponent(
                    ".\(name).staging-\(UUID().uuidString)", isDirectory: true)
                defer { try? FileManager.default.removeItem(at: staging) }
                try FileManager.default.copyItem(at: source, to: staging)
                _ = try PolicyBundleLoader.load(directory: staging)
                try FileManager.default.moveItem(at: staging, to: destination)
            }
            _ = try PolicyBundleLoader.load(directory: destination)
            interactionStatus = "Imported \(candidate.manifest.title)"
            refreshLibrary(selecting: destination)
        } catch {
            interactionStatus = "import failed: \(error.localizedDescription)"
        }
    }

    func togglePlayback() {
        if session?.episodeFinished == true {
            resetEpisode()
        } else {
            running.toggle()
            lastTime = CACurrentMediaTime()
            accumulator = 0
        }
    }

    func singleStep() {
        guard let session, !session.episodeFinished else { return }
        do {
            try session.step()
            running = false
            refreshStats()
        } catch {
            stop(with: "step failed: \(error.localizedDescription)")
        }
    }

    func resetEpisode() {
        do {
            if solver?.runtimeFailure != nil {
                loadSelectedPage()
                return
            }
            try session?.reset()
            running = true
            accumulator = 0
            lastTime = CACurrentMediaTime()
            interactionStatus = ""
            refreshStats()
        } catch {
            stop(with: "reset failed: \(error.localizedDescription)")
        }
    }

    func invoke(_ control: PolicyBundleManifest.Control) {
        guard control.kind == .button, let command = control.command,
              let session else { return }
        var arguments = [String: Float]()
        for (name, reference) in control.arguments ?? [:] {
            if let value = controlValues[reference] {
                arguments[name] = value
            }
        }
        do {
            try session.perform(command: command, arguments: arguments)
            running = !session.episodeFinished
            accumulator = 0
            lastTime = CACurrentMediaTime()
            interactionStatus = control.label
            refreshStats()
        } catch {
            stop(with: "\(control.label) failed: \(error.localizedDescription)")
        }
    }

    func tickIfRunning() {
        guard running, let session else { return }
        if let failure = session.solver.runtimeFailure {
            stop(with: "simulation stopped: \(failure.localizedDescription)")
            return
        }
        let now = CACurrentMediaTime()
        let wallStep = Double(session.controlStepSeconds)
            / max(playbackRate, 0.1)
        accumulator += min(now - lastTime, 0.1)
        lastTime = now
        var ticks = 0
        while accumulator >= wallStep, ticks < 3, !session.episodeFinished {
            do {
                try session.step()
            } catch {
                stop(with: "simulation stopped: \(error.localizedDescription)")
                return
            }
            accumulator -= wallStep
            ticks += 1
        }
        if session.episodeFinished { running = false }
        refreshStats()
    }

    func reportRenderFailure(_ message: String) {
        stop(with: "render stopped: \(message)")
    }

    private func loadSelectedPage() {
        guard let page = selectedPage else { return }
        session = nil
        running = false
        trust = page.trust
        controlValues = [:]
        do {
            try MLXRuntimeResources.requireDefaultMetalLibrary()
            let created = try PolicyBundleReplayFactory.make(
                bundle: page.bundle, release: page.release)
            session = created
            trust = page.trust
            controlValues = Dictionary(uniqueKeysWithValues:
                page.bundle.manifest.presentation.controls.compactMap {
                    control in control.defaultValue.map { (control.id, $0) }
                })
            selectedCameraID =
                page.bundle.manifest.presentation.cameraPresets.first!.id
            status = trust.displayLabel + " · "
                + page.bundle.directory.lastPathComponent
            interactionStatus = ""
            running = true
            accumulator = 0
            lastTime = CACurrentMediaTime()
            cameraEpoch += 1
            refreshStats()
        } catch {
            status = "replay unavailable: \(error.localizedDescription)"
            statsText = status
        }
    }

    private func refreshStats() {
        guard let session, let manifest else {
            statsText = status
            return
        }
        statsText = manifest.presentation.metrics.map { metric in
            let value = session.values[metric.source]
            let rendered = value.map {
                Self.format($0, pattern: metric.format)
            } ?? "—"
            return "\(metric.label): \(rendered)"
                + (metric.unit.map { " \($0)" } ?? "")
        }.joined(separator: "\n")
        if session.episodeFinished {
            statsText += statsText.isEmpty ? "episode finished"
                : "\nepisode finished"
        }
    }

    private func stop(with message: String) {
        running = false
        accumulator = 0
        status = message
        statsText = message
    }

    private static func loadReleaseIndex(
        _ url: URL
    ) throws -> PolicyBundleReleaseIndex {
        try PolicyBundleReleaseIndex.load(from: url)
    }

    private func discoveryRoots() throws -> (
        packaged: URL?, releaseIndex: URL?, bundleRoots: [URL]
    ) {
        let manager = FileManager.default
        var roots = [URL]()
        var packaged: URL?
        if Bundle.main.bundleURL.pathExtension.lowercased() == "app",
           let resources = Bundle.main.resourceURL {
            let root = resources.appendingPathComponent(
                "checkpoints", isDirectory: true)
            if manager.fileExists(atPath: root.path) {
                packaged = root
                roots.append(root)
            }
        } else {
            let development = URL(fileURLWithPath:
                manager.currentDirectoryPath, isDirectory: true)
                .appendingPathComponent("checkpoints", isDirectory: true)
            if manager.fileExists(atPath: development.path) {
                roots.append(development)
            }
        }
        let imported = try Self.importedBundleRoot()
        if manager.fileExists(atPath: imported.path) { roots.append(imported) }
        if let override = ProcessInfo.processInfo.environment[
                "AVBD_POLICY_BUNDLE_ROOT"] {
            roots.append(URL(fileURLWithPath: override, isDirectory: true))
        }
        let releaseRoot = packaged ?? roots.first
        let releaseIndex = releaseRoot?.appendingPathComponent(
            "policy-release-index.json")
        return (
            packaged,
            releaseIndex.flatMap {
                manager.fileExists(atPath: $0.path) ? $0 : nil
            }, roots)
    }

    private static func importedBundleRoot() throws -> URL {
        let support = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true)
        return support.appendingPathComponent(
            "AVBD/PolicyBundles", isDirectory: true)
    }

    private static func isDescendant(_ child: URL, of parent: URL) -> Bool {
        let root = parent.standardizedFileURL.path + "/"
        return child.standardizedFileURL.path.hasPrefix(root)
    }

    private static func relativePath(_ child: URL, beneath parent: URL)
        -> String?
    {
        let root = parent.standardizedFileURL.path + "/"
        let path = child.standardizedFileURL.path
        guard path.hasPrefix(root) else { return nil }
        return String(path.dropFirst(root.count))
    }

    static func format(_ value: Float, pattern: String) -> String {
        // Bundle strings never become printf format strings. Only decimal
        // precision, an optional explicit sign, and a literal suffix are read.
        guard value.isFinite else { return "—" }
        let signed = pattern.contains("%+")
        let decimals: Int = {
            guard let dot = pattern.firstIndex(of: "."),
                  let f = pattern[dot...].firstIndex(of: "f") else { return 2 }
            return Int(pattern[pattern.index(after: dot)..<f]) ?? 2
        }()
        let formatter = NumberFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.numberStyle = .decimal
        formatter.minimumFractionDigits = max(0, min(decimals, 6))
        formatter.maximumFractionDigits = max(0, min(decimals, 6))
        formatter.usesGroupingSeparator = false
        if signed { formatter.positivePrefix = "+" }
        let number = formatter.string(from: NSNumber(value: value)) ?? "—"
        guard let f = pattern.firstIndex(of: "f") else { return number }
        return number + pattern[pattern.index(after: f)...]
    }
}

struct PolicyReplayLabView: View {
    @StateObject private var model = PolicyReplayModel()

    var body: some View {
        HSplitView {
            ZStack {
                PolicyReplayMetalView(model: model)
                if !model.hasReplayScene {
                    Color(nsColor: .windowBackgroundColor)
                    VStack(spacing: 10) {
                        Image(systemName: "shippingbox")
                            .font(.system(size: 32))
                            .foregroundStyle(.secondary)
                        Text("Load a policy bundle").font(.headline)
                        Text(model.status)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 460)
                    }
                    .padding(24)
                }
            }
            .frame(minWidth: 650)

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Policy Replay").font(.title2).bold()
                    RenderingModePicker(rayTracingEnabled: $model.rayTracingEnabled,
                                    metalFXEnabled: $model.metalFXEnabled,
                                        supportsRayTracing: model.supportsRayTracing)
                    if let manifest = model.manifest {
                        Text(manifest.summary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    HStack {
                        Picker("Bundle", selection: $model.selectedPageID) {
                            ForEach(model.pages) { page in
                                Text(page.bundle.manifest.title).tag(page.id)
                            }
                        }
                        .pickerStyle(.menu)
                        Button("Import Bundle…") { model.isImporting = true }
                    }
                    Text(model.trust.displayLabel)
                        .font(.caption.bold())
                        .foregroundStyle(model.trust.isVerified
                            ? Color.green : Color.orange)
                    Text(model.status)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)

                    HStack {
                        Button(model.session?.episodeFinished == true
                            ? "Replay" : (model.running ? "Pause" : "Play")) {
                            model.togglePlayback()
                        }
                        Button("Step") { model.singleStep() }
                            .disabled(model.session?.episodeFinished == true)
                        Button("Reset") { model.resetEpisode() }
                    }
                    .disabled(!model.hasReplayScene)

                    if let cameras = model.manifest?.presentation.cameraPresets,
                       cameras.count > 1 {
                        Picker("Camera", selection: $model.selectedCameraID) {
                            ForEach(cameras) { camera in
                                Text(camera.label).tag(camera.id)
                            }
                        }
                        .pickerStyle(.segmented)
                    }
                    HStack {
                        Text("Playback").font(.caption)
                        Slider(value: $model.playbackRate, in: 0.1...1)
                        Text(String(format: "%.2fx", model.playbackRate))
                            .font(.caption.monospacedDigit())
                            .frame(width: 42)
                    }

                    if let controls = model.manifest?.presentation.controls,
                       !controls.isEmpty {
                        GroupBox("Bundle Controls") {
                            VStack(alignment: .leading, spacing: 10) {
                                ForEach(controls) { control in
                                    controlView(control)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }

                    Text(model.statsText)
                        .font(.system(.caption, design: .monospaced))
                        .fixedSize(horizontal: false, vertical: true)
                    if !model.interactionStatus.isEmpty {
                        Text(model.interactionStatus)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    Text("Bundles supply policy, simulation, cameras, controls, and metrics. Imported bundles are always unverified unless the app's independent release index authenticates them.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(14)
            }
            .frame(minWidth: 430, idealWidth: 520, maxWidth: 620)
        }
        .fileImporter(
            isPresented: $model.isImporting,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            if case let .success(urls) = result, let url = urls.first {
                model.importBundle(from: url)
            }
        }
    }

    @ViewBuilder
    private func controlView(
        _ control: PolicyBundleManifest.Control
    ) -> some View {
        switch control.kind {
        case .slider:
            let minimum = Double(control.minimum ?? 0)
            let maximum = Double(control.maximum ?? 1)
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(control.label).font(.caption)
                    Spacer()
                    Text(PolicyReplayModel.format(
                        model.controlValues[control.id]
                            ?? control.defaultValue ?? 0,
                        pattern: control.format ?? "%.2f"))
                        .font(.caption.monospacedDigit())
                }
                Slider(value: Binding(
                    get: { Double(model.controlValues[control.id]
                        ?? control.defaultValue ?? 0) },
                    set: { model.controlValues[control.id] = Float($0) }
                ), in: minimum...maximum, step: Double(control.step ?? 0.001))
            }
        case .toggle:
            Toggle(control.label, isOn: Binding(
                get: { (model.controlValues[control.id] ?? 0) != 0 },
                set: { model.controlValues[control.id] = $0 ? 1 : 0 }
            ))
        case .button:
            Button(control.label) { model.invoke(control) }
        }
    }
}

private struct PolicyReplayMetalView: NSViewRepresentable {
    @ObservedObject var model: PolicyReplayModel

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> PolicyOrbitMTKView {
        let device = model.solver?.device ?? MTLCreateSystemDefaultDevice()
        let view = PolicyOrbitMTKView(frame: .zero, device: device)
        if ProcessInfo.processInfo.environment["AVBD_SHOT"] != nil
            || ProcessInfo.processInfo.environment["AVBD_VIDEO_DIR"] != nil {
            view.framebufferOnly = false
        }
        guard let device else {
            model.reportRenderFailure("no Metal device is available")
            view.isPaused = true
            return view
        }
        do {
            let renderer = try GPUSimRenderer(device: device, source: model)
            configureAppCapture(renderer: renderer, model: model)
            renderer.configure(view, preferredFramesPerSecond: 30)
            context.coordinator.renderer = renderer
            view.renderer = renderer
        } catch {
            model.reportRenderFailure(error.localizedDescription)
            view.isPaused = true
        }
        return view
    }

    func updateNSView(_ view: PolicyOrbitMTKView, context: Context) {
        guard let renderer = context.coordinator.renderer,
              let camera = model.selectedCamera else { return }
        if context.coordinator.cameraID != camera.id
            || context.coordinator.solver !== model.solver {
            context.coordinator.cameraID = camera.id
            context.coordinator.solver = model.solver
            renderer.azimuth = camera.azimuth
            renderer.elevation = camera.elevation
            renderer.distance = camera.distance
        }
        renderer.target = model.cameraTarget
    }

    @MainActor final class Coordinator {
        var renderer: GPUSimRenderer?
        var cameraID: String?
        weak var solver: GPUSolver?
    }
}

/// Camera-only input. Policy actions remain the sole robot controls.
private final class PolicyOrbitMTKView: MTKView {
    weak var renderer: GPUSimRenderer?
    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }

    override func mouseDragged(with event: NSEvent) {
        guard let renderer else { return }
        renderer.azimuth -= Float(event.deltaX) * 0.008
        renderer.elevation = min(max(
            renderer.elevation + Float(event.deltaY) * 0.008, -1.5), 1.55)
    }

    override func rightMouseDragged(with event: NSEvent) {
        guard let renderer else { return }
        let scale = renderer.distance * 0.0015
        let right = F3(-sin(renderer.azimuth), cos(renderer.azimuth), 0)
        renderer.target -= right * Float(event.deltaX) * scale
        renderer.target += F3(0, 0, 1) * Float(event.deltaY) * scale
    }

    override func scrollWheel(with event: NSEvent) {
        guard let renderer else { return }
        renderer.distance = min(max(
            renderer.distance * (1 - Float(event.scrollingDeltaY) * 0.02),
            0.2), 100)
    }
}
