// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "avbd-metal",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AVBDCore", targets: ["AVBDCore"]),
        .library(name: "AVBDLearn", targets: ["AVBDLearn"]),
        .executable(name: "avbd", targets: ["avbd"]),
        .executable(name: "AVBDApp", targets: ["AVBDApp"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
    ],
    targets: [
        .target(
            name: "AVBDCore",
            resources: [.copy("Shaders"), .copy("Assets")]
        ),
        .target(
            name: "AVBDLearn",
            dependencies: [
                "AVBDCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "avbd",
            dependencies: ["AVBDCore", "AVBDLearn"]
        ),
        .executableTarget(
            name: "AVBDApp",
            dependencies: ["AVBDCore", "AVBDLearn"]
        ),
        .testTarget(
            name: "AVBDTests",
            dependencies: ["AVBDCore", "AVBDLearn"]
        ),
    ]
)
