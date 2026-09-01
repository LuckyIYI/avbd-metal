// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "GPUSimPackageConsumer",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(name: "gpu-sim", path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "PackageConsumer",
            dependencies: [
                .product(name: "GPUSim", package: "gpu-sim"),
            ]
        ),
    ],
    swiftLanguageVersions: [.v5]
)
