// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "avbd-metal",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "Robotics", targets: ["Robotics"]),
        .library(name: "RL", targets: ["RL"]),
        .library(name: "MLXRL", targets: ["MLXRL"]),
        .executable(name: "avbd", targets: ["avbd"]),
        .executable(name: "AVBDApp", targets: ["AVBDApp"]),
    ],
    dependencies: [
        .package(name: "gpu-sim", path: ".."),
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            from: "0.21.0"
        ),
    ],
    targets: [
        .target(
            name: "Robotics",
            dependencies: [
                .product(name: "SimCore", package: "gpu-sim"),
            ],
            resources: [.copy("Assets")]
        ),
        .target(
            name: "RL",
            dependencies: [
                .product(name: "SimCore", package: "gpu-sim"),
                .product(name: "PhysicsAVBD", package: "gpu-sim"),
                .product(name: "GPUSimDemos", package: "gpu-sim"),
                "Robotics",
            ]
        ),
        .target(
            name: "MLXRL",
            dependencies: [
                .product(name: "SimCore", package: "gpu-sim"),
                .product(name: "PhysicsAVBD", package: "gpu-sim"),
                "Robotics",
                "RL",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "avbd",
            dependencies: [
                .product(name: "SimCore", package: "gpu-sim"),
                .product(name: "PhysicsAVBD", package: "gpu-sim"),
                .product(name: "GPUSimDemos", package: "gpu-sim"),
                "Robotics",
                "RL",
                "MLXRL",
            ]
        ),
        .executableTarget(
            name: "AVBDApp",
            dependencies: [
                .product(name: "SimCore", package: "gpu-sim"),
                .product(name: "PhysicsAVBD", package: "gpu-sim"),
                .product(name: "GPUSimDemos", package: "gpu-sim"),
                .product(name: "GPUSimRenderer", package: "gpu-sim"),
                "Robotics",
                "RL",
                "MLXRL",
            ]
        ),
        .testTarget(
            name: "AVBDTests",
            dependencies: [
                .product(name: "SimCore", package: "gpu-sim"),
                .product(name: "PhysicsAVBD", package: "gpu-sim"),
                .product(name: "GPUSimDemos", package: "gpu-sim"),
                "Robotics",
                "RL",
                "MLXRL",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ],
    swiftLanguageVersions: [.v5]
)
