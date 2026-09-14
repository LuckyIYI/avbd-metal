// swift-tools-version: 5.10
import PackageDescription
let package = Package(name:"implicit-contact-check", platforms:[.macOS(.v14)], dependencies:[.package(name:"avbd-metal",path:"../..")], targets:[.executableTarget(name:"implicit-contact-check",dependencies:[.product(name:"GPUSim",package:"avbd-metal")],path:"Sources")])
