// swift-tools-version:6.2
import PackageDescription

let release: [SwiftSetting] = [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]

let package = Package(
    name: "instantllm",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "instantllm", targets: ["instantllm"]),
    ],
    targets: [
        .target(name: "Tokenizer", swiftSettings: release),
        .target(name: "Engine", swiftSettings: release),
        .executableTarget(name: "instantllm", dependencies: ["Engine", "Tokenizer"], swiftSettings: release),
        .executableTarget(name: "tokcheck", dependencies: ["Tokenizer"], swiftSettings: release),
    ],
    swiftLanguageModes: [.v5]
)
