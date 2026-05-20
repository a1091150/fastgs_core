// swift-tools-version: 5.12

import PackageDescription

let package = Package(
    name: "FastGSSwift",
    platforms: [
        .macOS("14.0"),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "FastGSSwift",
            targets: ["FastGSSwift"]
        )
    ],
    dependencies: [
        .package(path: "../../submodules/mlx-swift")
    ],
    targets: [
        .target(
            name: "FastGSSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ]
        ),
        .testTarget(
            name: "FastGSSwiftTests",
            dependencies: ["FastGSSwift"]
        ),
    ]
)
