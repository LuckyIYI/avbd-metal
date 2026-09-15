import XCTest
import Foundation
import Metal
import simd
@testable import PhysicsAVBD
import SimCore

/// Population-level parity: every hull asset of a cooked arena document must
/// decode (SimCore's ConvexHullAsset validator) AND upload (the GPU
/// uploader's face grouping) - the two runtime copies of the cooker's
/// coplanar-face check. A cell the offline cooker wrote and either runtime
/// copy refuses is exactly the failure this pins; it was found in the wild
/// as `DecodingError.dataCorrupted hullAssets.h02423: coplanar face 36
/// boundary is disconnected` after only the uploader had been fixed.
///
/// Skips unless CLATTER_ARENA_JSON names a document; runs on every hull it
/// contains and names each refusal.
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
            // The stored asset, triangulation and all - the path a cooked
            // document actually takes. The vertices-only overload rebuilds a
            // hull with the package's own builder and tests that builder, not
            // the uploader; it refused 15 of these 3,033 point sets while the
            // game loaded every one of them.
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
