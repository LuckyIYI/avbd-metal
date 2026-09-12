#if os(macOS)
import AppKit
import SwiftUI

/// Exercise the actual SwiftUI -> NSView subscription offscreen. Calling
/// updateNSView directly would miss a view that never observes its model.
@MainActor
func validateCameraUpdates() throws {
    let model = try Playground(name: "cablegrippers")
    model.running = false
    let host = NSHostingView(rootView: MetalCanvas(model: model))
    host.frame = NSRect(x: 0, y: 0, width: 1200, height: 600)
    func settle() {
        for _ in 0..<5 {
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.02))
        }
    }
    func canvas(_ view: NSView) -> CableView? {
        if let view = view as? CableView { return view }
        return view.subviews.lazy.compactMap { canvas($0) }.first
    }
    settle()
    guard let view = canvas(host) else {
        throw NSError(domain: "CableCameraValidation", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "SwiftUI did not create the viewport"])
    }
    func check(_ distance: Float, _ scale: Float, _ label: String) throws {
        settle()
        guard abs(view.renderer.distance - distance) < 0.0001,
              view.renderer.sceneLengthScale == scale else {
            throw NSError(domain: "CableCameraValidation", code: 2,
                          userInfo: [NSLocalizedDescriptionKey:
                            "\(label): distance=\(view.renderer.distance), scale=\(view.renderer.sceneLengthScale)"])
        }
    }
    try check(4.2, 1, "initial snap-fit")
    model.name = "cableethernet"
    model.reset()
    try check(0.13, 0.01, "switch to SI insertion")
    model.overview = true
    try check(0.40, 0.01, "whole cable")
    view.renderer.distance = 1
    model.cameraRevision += 1
    try check(0.40, 0.01, "reset view")
    model.overview = false
    try check(0.13, 0.01, "connector close-up")
    model.name = "cabletwisting"
    model.reset()
    try check(4.2, 1, "switch back to large scene")
    print("PASS live SwiftUI camera subscription, scene changes, close-up/overview and reset")
}
#endif
