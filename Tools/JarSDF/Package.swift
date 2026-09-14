// swift-tools-version: 5.10
import PackageDescription
let package = Package(name:"jar-sdf-pilot", platforms:[.macOS(.v14)], dependencies:[.package(name:"avbd-metal",path:"../..")], targets:[.executableTarget(name:"jar-sdf-pilot",dependencies:[.product(name:"GPUSim",package:"avbd-metal")],path:"Sources")])
