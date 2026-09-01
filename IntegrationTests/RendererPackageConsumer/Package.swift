// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "GPUSimRendererPackageConsumer",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(name: "gpu-sim", path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "RendererPackageConsumer",
            dependencies: [
                .product(name: "GPUSim", package: "gpu-sim"),
                .product(name: "GPUSimRenderer", package: "gpu-sim"),
            ]
        ),
    ],
    swiftLanguageVersions: [.v5]
)
