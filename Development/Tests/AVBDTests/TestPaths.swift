import Foundation

enum TestPaths {
    static let repositoryRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    static let developmentPackage = repositoryRoot
        .appendingPathComponent("Development", isDirectory: true)
}
