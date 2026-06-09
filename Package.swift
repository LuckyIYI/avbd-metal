// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "avbd-metal",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AVBDCore", targets: ["AVBDCore"]),
        .executable(name: "avbd", targets: ["avbd"]),
        .executable(name: "AVBDApp", targets: ["AVBDApp"]),
    ],
    targets: [
        .target(
            name: "AVBDCore",
            resources: [.process("Shaders")]
        ),
        .executableTarget(
            name: "avbd",
            dependencies: ["AVBDCore"]
        ),
        .executableTarget(
            name: "AVBDApp",
            dependencies: ["AVBDCore"]
        ),
        .testTarget(
            name: "AVBDTests",
            dependencies: ["AVBDCore"]
        ),
    ]
)
