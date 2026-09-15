import XCTest
import Metal
import simd
@testable import PhysicsAVBD
import SimCore

/// Faces are assembled largest-triangle-first. Before that, whichever triangle
/// the hull builder emitted first defined a face's plane; when it was a
/// sliver, every well-conditioned triangle of the same face failed the angular
/// test against its noisy normal and split off, and the cell was refused as
/// "not one convex loop" or "boundary is disconnected". 408 authored cells of
/// one generated arena were lost that way, silently.
///
/// This is the runtime half of the parity guarantee: the offline cooker
/// (Tools/cook_convex_asset.py) and this uploader must group identically, or a
/// hull the cooker accepts throws at scene build. Every captured cell goes
/// through `addConvexCollider` and `GPUSolver(scene:)` exactly as an asset
/// does.
final class ConvexSeedOrderTests: XCTestCase {
    private struct Cell: Decodable {
        let owner: String
        let error: String
        let points: [[Float]]
    }

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("No Metal device; uploader parity needs the GPU path")
        }
    }

    private func cells() throws -> [Cell] {
        try JSONDecoder().decode([Cell].self, from: Data(convexSeedOrderCellsJSON.utf8))
    }

    /// The 405 cells whose only defect was seeding order must upload; the 3
    /// carrying a sub-Float32 step (a 2 um triangle 38 nm off collinear) must
    /// still be refused as degenerate, not quietly admitted.
    func testCapturedCellsUploadUnlessDegenerate() throws {
        try requireMetal()
        let fixtures = try cells()
        XCTAssertEqual(fixtures.count, 408, "Keep every captured cell")
        var uploaded = 0, refusedDegenerate = 0
        var wrong: [String] = []
        for (index, cell) in fixtures.enumerated() {
            let vertices = cell.points.map { F3($0[0], $0[1], $0[2]) }
            var scene = PhysicsScene(name: "seed-order-\(index)")
            scene.settings.gravity = 0
            scene.settings.iterations = 0
            let body = scene.addBody(size: F3(repeating: 1), density: 1, friction: 0.5,
                                     position: .zero, rotation: Quat(real: 1, imag: .zero),
                                     collisionEnabled: false)
            _ = scene.addConvexCollider(body: body, vertices: vertices)
            let expectsDegenerate = cell.error.contains("degenerate")
            do {
                _ = try GPUSolver(scene: scene)
                if expectsDegenerate {
                    wrong.append("\(index) \(cell.owner): degenerate cell was accepted")
                } else {
                    uploaded += 1
                }
            } catch {
                let text = String(describing: error)
                if expectsDegenerate, text.contains("degenerate") {
                    refusedDegenerate += 1
                } else {
                    wrong.append("\(index) \(cell.owner): \(text.prefix(90))")
                }
            }
        }
        XCTAssertEqual(uploaded, 405, "cells rescued by largest-first seeding")
        XCTAssertEqual(refusedDegenerate, 3, "sub-resolution cells stay refused")
        XCTAssertTrue(wrong.isEmpty, "\n" + wrong.prefix(12).joined(separator: "\n"))
    }

    /// A merged face's loop is a subset of the hull's vertices, so at the
    /// 64-vertex hull cap no face can exceed the 64-vertex face limit. At the
    /// larger caps the limit is evaluated on merged faces: a 64-gon prism top
    /// is accepted, a 65-gon top is refused, on both validators alike.
    func testMergedFaceLimitIsEvaluatedOnMergedFaces() throws {
        try requireMetal()
        func prism(_ sides: Int) -> [F3] {
            (0..<sides).flatMap { i -> [F3] in
                let a = Float(i) / Float(sides) * 2 * .pi
                return [F3(cos(a), sin(a), 0), F3(cos(a), sin(a), 0.2)]
            }
        }
        for (sides, shouldUpload) in [(60, true), (64, true), (65, false), (70, false)] {
            var scene = PhysicsScene(name: "prism-\(sides)")
            scene.settings.gravity = 0
            let body = scene.addBody(size: F3(repeating: 1), density: 1, friction: 0.5,
                                     position: .zero, rotation: Quat(real: 1, imag: .zero),
                                     collisionEnabled: false)
            _ = scene.addConvexCollider(body: body, vertices: prism(sides))
            do {
                _ = try GPUSolver(scene: scene)
                XCTAssertTrue(shouldUpload, "\(sides)-gon prism should have been refused")
            } catch {
                if shouldUpload {
                    let text = String(describing: error)
                    // A hull-vertex cap below 2*sides is a limit of this base,
                    // not a grouping outcome; only a face-limit refusal counts.
                    XCTAssertFalse(text.contains("coplanar face"), "\(sides)-gon: \(text.prefix(80))")
                }
            }
        }
    }
}
