import Foundation
import XCTest
@testable import AVBDLearn

final class MLXRuntimeResourcesTests: XCTestCase {
    private var temporaryRoots: [URL] = []

    override func tearDownWithError() throws {
        for root in temporaryRoots {
            try? FileManager.default.removeItem(at: root)
        }
        temporaryRoots.removeAll(keepingCapacity: false)
        try super.tearDownWithError()
    }

    func testMissingMetalLibraryFailsClosedAndReportsSearchedPaths() throws {
        let app = try makeTemporaryRoot().appendingPathComponent(
            "AVBD.app", isDirectory: true)
        try FileManager.default.createDirectory(
            at: app, withIntermediateDirectories: true)

        let availability = MLXRuntimeResources.inspect(mainBundleURL: app)

        XCTAssertFalse(availability.isAvailable)
        XCTAssertNil(availability.defaultMetalLibraryURL)
        XCTAssertEqual(availability.searchedURLs.count, 1)
        XCTAssertTrue(
            availability.unavailableDescription.contains("default.metallib"))
        XCTAssertTrue(
            availability.unavailableDescription.contains("make app-ml"))
        XCTAssertThrowsError(
            try MLXRuntimeResources.requireDefaultMetalLibrary(
                mainBundleURL: app)) { error in
            guard let resourceError = error as? MLXRuntimeResourceError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(resourceError.availability, availability)
            XCTAssertEqual(
                resourceError.localizedDescription,
                availability.unavailableDescription)
        }
    }

    func testEmptyDirectoryNamedMetalLibraryIsRejected() throws {
        let app = try makeTemporaryRoot().appendingPathComponent(
            "AVBD.app", isDirectory: true)
        let fakeLibrary = wrappedLibraryURL(app: app)
        try FileManager.default.createDirectory(
            at: fakeLibrary, withIntermediateDirectories: true)

        let availability = MLXRuntimeResources.inspect(mainBundleURL: app)

        XCTAssertFalse(availability.isAvailable)
        XCTAssertNil(availability.defaultMetalLibraryURL)
    }

    func testOrdinaryFileNamedMetalLibraryIsRejected() throws {
        let app = try makeTemporaryRoot().appendingPathComponent(
            "AVBD.app", isDirectory: true)
        let fakeLibrary = wrappedLibraryURL(app: app)
        try FileManager.default.createDirectory(
            at: fakeLibrary.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try Data(repeating: 0xa5, count: 64).write(to: fakeLibrary)

        let availability = MLXRuntimeResources.inspect(mainBundleURL: app)

        XCTAssertFalse(availability.isAvailable)
        XCTAssertNil(availability.defaultMetalLibraryURL)
    }

    func testHeaderOnlyMetalLibraryIsRejectedByMetal() throws {
        let app = try makeTemporaryRoot().appendingPathComponent(
            "AVBD.app", isDirectory: true)
        let fakeLibrary = wrappedLibraryURL(app: app)
        try writePlaceholderLibrary(at: fakeLibrary)

        let availability = MLXRuntimeResources.inspect(mainBundleURL: app)

        XCTAssertFalse(availability.isAvailable)
        XCTAssertNil(availability.defaultMetalLibraryURL)
    }

    func testWrappedAppMetalLibraryIsAccepted() throws {
        let app = try makeTemporaryRoot().appendingPathComponent(
            "AVBD.app", isDirectory: true)
        let library = wrappedLibraryURL(app: app)
        try writePlaceholderLibrary(at: library)

        let availability = MLXRuntimeResources.inspect(
            mainBundleURL: app, libraryValidator: { _ in true })

        XCTAssertTrue(availability.isAvailable)
        XCTAssertEqual(availability.defaultMetalLibraryURL, library)
        XCTAssertEqual(
            try MLXRuntimeResources.requireDefaultMetalLibrary(
                mainBundleURL: app, libraryValidator: { _ in true }),
            library)
    }

    func testWrappedAppDoesNotAcceptSiblingResourceBundle() throws {
        let root = try makeTemporaryRoot()
        let app = root.appendingPathComponent("AVBD.app", isDirectory: true)
        try FileManager.default.createDirectory(
            at: app, withIntermediateDirectories: true)
        let sibling = root.appendingPathComponent(
            "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
        try writePlaceholderLibrary(at: sibling)

        let availability = MLXRuntimeResources.inspect(
            mainBundleURL: app, libraryValidator: { _ in true })

        XCTAssertFalse(availability.isAvailable)
        XCTAssertEqual(availability.searchedURLs, [wrappedLibraryURL(app: app)])
    }

    func testSiblingXcodeResourceBundleMetalLibraryIsAccepted() throws {
        let products = try makeTemporaryRoot().appendingPathComponent(
            "Build/Products/Release", isDirectory: true)
        let executable = products.appendingPathComponent("AVBDApp")
        let library = products.appendingPathComponent(
            "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
        try writePlaceholderLibrary(at: library)

        let availability = MLXRuntimeResources.inspect(
            mainBundleURL: executable, libraryValidator: { _ in true })

        XCTAssertTrue(availability.isAvailable)
        XCTAssertEqual(availability.defaultMetalLibraryURL, library)
    }

    private func makeTemporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "avbd-mlx-resource-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        temporaryRoots.append(root)
        return root
    }

    private func wrappedLibraryURL(app: URL) -> URL {
        app.appendingPathComponent(
            "Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/"
                + "default.metallib")
    }

    private func writePlaceholderLibrary(at url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        var data = Data([0x4d, 0x54, 0x4c, 0x42])
        data.append(Data(repeating: 0, count: 60))
        try data.write(to: url)
    }
}
