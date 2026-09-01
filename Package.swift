// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "gpu-sim",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "GPUSim", targets: ["GPUSim"]),
        .library(name: "GPUSimDemos", targets: ["GPUSimDemos"]),
        .library(name: "GPUSimRenderer", targets: ["GPUSimRenderer"]),
        .library(name: "SimCore", targets: ["SimCore"]),
        .library(name: "PhysicsAVBD", targets: ["PhysicsAVBD"]),
    ],
    targets: [
        .target(name: "SimCore"),
        .target(
            name: "PhysicsAVBD",
            dependencies: ["SimCore"],
            resources: [.copy("Shaders")]
        ),
        .target(
            name: "GPUSim",
            dependencies: ["SimCore", "PhysicsAVBD"]
        ),
        .target(
            name: "GPUSimDemos",
            dependencies: ["SimCore"],
            resources: [.copy("Assets")]
        ),
        .target(
            name: "GPUSimRenderer",
            dependencies: ["SimCore", "PhysicsAVBD"]
        ),
        .testTarget(
            name: "SimCoreTests",
            dependencies: ["SimCore"]
        ),
        .testTarget(
            name: "PhysicsAVBDTests",
            dependencies: ["SimCore", "PhysicsAVBD", "GPUSimDemos"]
        ),
        .testTarget(
            name: "GPUSimTests",
            dependencies: ["GPUSim"]
        ),
        .testTarget(
            name: "GPUSimRendererTests",
            dependencies: ["GPUSimRenderer"]
        ),
    ],
    swiftLanguageVersions: [.v5]
)
