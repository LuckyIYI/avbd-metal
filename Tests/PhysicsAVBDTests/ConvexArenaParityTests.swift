import XCTest
import Foundation
import Metal
import simd
@testable import PhysicsAVBD
import SimCore

/// Every hull asset of a cooked document must decode and upload; the two
/// runtime copies of the coplanar-face check must agree with the cooker.
/// Runs when CLATTER_ARENA_JSON names a document.
final class ConvexArenaParityTests: XCTestCase {
    private struct Document: Decodable { let hullAssets: [String: JSONValue] }
    private struct JSONValue: Decodable {
        let raw: Data
        init(from decoder: Decoder) throws {
            // Keep the raw object so each asset can be decoded on its own and
            // one refusal cannot mask the rest.
            let container = try decoder.singleValueContainer()
            let any = try container.decode(AnyCodable.self)
            raw = try JSONSerialization.data(withJSONObject: any.value)
        }
    }
    private struct AnyCodable: Decodable {
        let value: Any
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let v = try? c.decode([String: AnyCodable].self) { value = v.mapValues { $0.value } }
            else if let v = try? c.decode([AnyCodable].self) { value = v.map { $0.value } }
            else if let v = try? c.decode(Double.self) { value = v }
            else if let v = try? c.decode(String.self) { value = v }
            else if let v = try? c.decode(Bool.self) { value = v }
            else { value = NSNull() }
        }
    }

    func testEveryArenaHullDecodesAndUploads() throws {
        guard let path = ProcessInfo.processInfo.environment["CLATTER_ARENA_JSON"] else {
            throw XCTSkip("set CLATTER_ARENA_JSON to an arena.json to run population parity")
        }
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device") }
        let document = try JSONDecoder().decode(Document.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        var decodeRefused: [String] = [], uploadRefused: [String] = []
        var assets: [(String, ConvexHullAsset)] = []
        for key in document.hullAssets.keys.sorted() {
            do { assets.append((key, try JSONDecoder().decode(ConvexHullAsset.self, from: document.hullAssets[key]!.raw))) }
            catch { decodeRefused.append("\(key): \(String(describing: error).prefix(80))") }
        }
        for (key, asset) in assets {
            var scene = PhysicsScene(name: "arena-parity-\(key)")
            scene.settings.gravity = 0
            scene.settings.iterations = 0
            let body = scene.addBody(size: F3(repeating: 1), density: 1, friction: 0.5,
                                     position: .zero, rotation: Quat(real: 1, imag: .zero),
                                     collisionEnabled: false)
            // Upload the stored asset, triangulation included, as a document does.
            _ = scene.addConvexCollider(body: body, asset: asset,
                                        localPosition: .zero,
                                        localRotation: Quat(real: 1, imag: .zero))
            do { _ = try GPUSolver(scene: scene) }
            catch { uploadRefused.append("\(key): \(String(describing: error).prefix(80))") }
        }
        print("arena parity: \(document.hullAssets.count) hulls, decode refused \(decodeRefused.count), upload refused \(uploadRefused.count)")
        XCTAssertTrue(decodeRefused.isEmpty, "decoder refused:\n" + decodeRefused.prefix(10).joined(separator: "\n"))
        XCTAssertTrue(uploadRefused.isEmpty, "uploader refused:\n" + uploadRefused.prefix(10).joined(separator: "\n"))
    }
}
