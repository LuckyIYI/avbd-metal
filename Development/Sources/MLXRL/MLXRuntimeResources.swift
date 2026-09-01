import Foundation
import Metal

/// Fail-closed discovery of the Metal library required by MLX. Access is
/// package-scoped because applications in this package need the probe, while
/// the location of an embedded dependency bundle is not public AVBD API.
package struct MLXRuntimeResourceAvailability: Sendable, Equatable {
    package let defaultMetalLibraryURL: URL?
    package let searchedURLs: [URL]

    package var isAvailable: Bool { defaultMetalLibraryURL != nil }

    package var unavailableDescription: String {
        let searched = searchedURLs.map(\.path).joined(separator: ", ")
        return "MLX runtime unavailable: mlx-swift's default.metallib is "
            + "missing; build the ML-enabled app with `make app-ml`"
            + (searched.isEmpty ? "" : " (searched: \(searched))")
    }
}

package struct MLXRuntimeResourceError: Error, LocalizedError, Sendable,
    Equatable
{
    package let availability: MLXRuntimeResourceAvailability

    package var errorDescription: String? {
        availability.unavailableDescription
    }
}

package enum MLXRuntimeResources {
    private static let libraryInResourceBundle =
        "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
    private static let metalLibraryMagic = Data([0x4d, 0x54, 0x4c, 0x42])

    /// Inspect the layouts produced by a wrapped macOS app, an Xcode test
    /// bundle, and an unwrapped Xcode executable. A directory named
    /// `default.metallib` is rejected: touching MLX without the compiled file
    /// aborts the process instead of throwing a Swift error.
    package static func inspect(
        mainBundleURL: URL = Bundle.main.bundleURL,
        fileManager: FileManager = .default,
        libraryValidator: ((URL) -> Bool)? = nil
    ) -> MLXRuntimeResourceAvailability {
        let candidates: [URL]
        if mainBundleURL.pathExtension.lowercased() == "app" {
            // The generated Cmlx resource accessor searches Bundle.main's
            // resource directory. It does not search beside a wrapped app.
            candidates = [mainBundleURL
                .appendingPathComponent("Contents/Resources", isDirectory: true)
                .appendingPathComponent(libraryInResourceBundle)]
        } else {
            // Xcode/SwiftPM place dependency resource bundles either under
            // the main product directory or beside an unwrapped executable.
            candidates = uniqueURLs([
                mainBundleURL.appendingPathComponent(libraryInResourceBundle),
                mainBundleURL.deletingLastPathComponent()
                    .appendingPathComponent(libraryInResourceBundle),
            ])
        }
        let validate = libraryValidator ?? canLoadMetalLibrary
        let library = candidates.first {
            isMetalLibraryFile(
                $0, fileManager: fileManager, libraryValidator: validate)
        }
        return MLXRuntimeResourceAvailability(
            defaultMetalLibraryURL: library, searchedURLs: candidates)
    }

    @discardableResult
    package static func requireDefaultMetalLibrary(
        mainBundleURL: URL = Bundle.main.bundleURL,
        fileManager: FileManager = .default,
        libraryValidator: ((URL) -> Bool)? = nil
    ) throws -> URL {
        let availability = inspect(
            mainBundleURL: mainBundleURL,
            fileManager: fileManager,
            libraryValidator: libraryValidator)
        guard let library = availability.defaultMetalLibraryURL else {
            throw MLXRuntimeResourceError(availability: availability)
        }
        return library
    }

    private static func uniqueURLs(_ urls: [URL]) -> [URL] {
        var seen = Set<String>()
        return urls.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    private static func isMetalLibraryFile(
        _ url: URL,
        fileManager: FileManager,
        libraryValidator: (URL) -> Bool
    ) -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(
                atPath: url.path, isDirectory: &isDirectory),
              !isDirectory.boolValue,
              let attributes = try? fileManager.attributesOfItem(
                atPath: url.path),
              let size = attributes[.size] as? NSNumber,
              size.uint64Value >= 16,
              let handle = try? FileHandle(forReadingFrom: url)
        else { return false }
        defer { try? handle.close() }
        guard (try? handle.read(upToCount: metalLibraryMagic.count))
                == metalLibraryMagic else { return false }
        return libraryValidator(url)
    }

    private static func canLoadMetalLibrary(_ url: URL) -> Bool {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return (try? device.makeLibrary(URL: url)) != nil
    }
}
