// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "avbd-metal",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "SimCore", targets: ["SimCore"]),
        .library(name: "PhysicsAVBD", targets: ["PhysicsAVBD"]),
        .library(name: "Robotics", targets: ["Robotics"]),
        .library(name: "RL", targets: ["RL"]),
        .library(name: "MLXRL", targets: ["MLXRL"]),
        .executable(name: "avbd", targets: ["avbd"]),
        .executable(name: "AVBDApp", targets: ["AVBDApp"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
    ],
    targets: [
        .target(name: "SimCore"),
        .target(
            name: "PhysicsAVBD",
            dependencies: ["SimCore"],
            resources: [.copy("Shaders"), .copy("Assets")]
        ),
        .target(
            name: "Robotics",
            dependencies: ["SimCore"],
            resources: [.copy("Assets")]
        ),
        .target(
            name: "RL",
            dependencies: ["SimCore", "PhysicsAVBD", "Robotics"]
        ),
        .target(
            name: "MLXRL",
            dependencies: [
                "SimCore", "PhysicsAVBD", "Robotics", "RL",
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
                "SimCore", "PhysicsAVBD", "Robotics", "RL", "MLXRL",
            ]
        ),
        .executableTarget(
            name: "AVBDApp",
            dependencies: [
                "SimCore", "PhysicsAVBD", "Robotics", "RL", "MLXRL",
            ]
        ),
        .testTarget(
            name: "AVBDTests",
            dependencies: [
                "SimCore", "PhysicsAVBD", "Robotics", "RL", "MLXRL",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "SimCoreTests",
            dependencies: ["SimCore"]
        ),
        .testTarget(
            name: "PhysicsAVBDTests",
            dependencies: ["SimCore", "PhysicsAVBD"]
        ),
    ]
)
