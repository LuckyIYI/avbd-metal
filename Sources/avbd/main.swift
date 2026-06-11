import AVBDCore
import AVBDLearn
import Metal
import Foundation
import simd

// avbd — headless CLI for the Metal AVBD solver.
//
// Usage:
//   avbd run <demo> [--frames N] [--iterations N] [--cpu] [--json] [--scale N]
//            [--dt T] [--watch BODY] [--stats-every N]
//   avbd bench <demo> [--frames N] [--scale N] [--iterations N]
//   avbd list
//   avbd parity <demo> [--frames N]

struct Options {
    var frames = 300
    var iterations: Int? = nil
    var scale = 1
    var dt: Float? = nil
    var useCPU = false
    var json = false
    var watch: Int? = nil
    var statsEvery = 60
    var envs = 64
    var batch = 256
    var latent = 128
    var lr: Float = 3e-4
    var lambda: Float = 0.5
    var episodes = 10
}

func parseOptions(_ args: [String]) -> Options {
    var o = Options()
    var i = 0
    while i < args.count {
        switch args[i] {
        case "--frames": i += 1; o.frames = Int(args[i]) ?? o.frames
        case "--iterations": i += 1; o.iterations = Int(args[i])
        case "--scale": i += 1; o.scale = Int(args[i]) ?? 1
        case "--dt": i += 1; o.dt = Float(args[i])
        case "--cpu": o.useCPU = true
        case "--json": o.json = true
        case "--watch": i += 1; o.watch = Int(args[i])
        case "--stats-every": i += 1; o.statsEvery = Int(args[i]) ?? 60
        case "--envs": i += 1; o.envs = Int(args[i]) ?? 64
        case "--batch": i += 1; o.batch = Int(args[i]) ?? 256
        case "--latent": i += 1; o.latent = Int(args[i]) ?? 128
        case "--lr": i += 1; o.lr = Float(args[i]) ?? 3e-4
        case "--lambda": i += 1; o.lambda = Float(args[i]) ?? 0.5
        case "--episodes": i += 1; o.episodes = Int(args[i]) ?? 10
        default: break
        }
        i += 1
    }
    return o
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(("error: " + msg + "\n").data(using: .utf8)!)
    exit(1)
}

func makeScene(_ name: String, _ o: Options) -> PhysicsScene {
    guard var scene = Demos.make(name, scale: o.scale) else {
        fail("unknown demo '\(name)'. Available: \(Demos.all.joined(separator: ", "))")
    }
    if let it = o.iterations { scene.settings.iterations = it }
    if let dt = o.dt { scene.settings.dt = dt }
    return scene
}

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    print("""
    avbd — Augmented Vertex Block Descent (Metal)

    Commands:
      run <demo>     Simulate a demo scene headlessly
      bench <demo>   Benchmark ms/frame
      parity <demo>  Compare GPU vs CPU trajectories
      list           List demo scenes

    Options: --frames N --iterations N --scale N --dt T --cpu --json --watch BODY --stats-every N
    """)
    exit(0)
}

switch command {
case "list":
    for d in Demos.all { print(d) }

case "run":
    guard args.count > 1 else { fail("usage: avbd run <demo>") }
    let o = parseOptions(Array(args.dropFirst(2)))
    let scene = makeScene(args[1], o)

    if o.useCPU {
        let solver = scene.makeCPUSolver()
        let t0 = Date()
        for f in 0..<o.frames {
            solver.step()
            if !o.json && (f + 1) % o.statsEvery == 0 {
                let err = solver.maxConstraintError()
                print(String(format: "frame %5d  err %.5f  forces %d", f + 1, err, solver.forces.count))
            }
        }
        let ms = Date().timeIntervalSince(t0) * 1000 / Double(o.frames)
        let err = solver.maxConstraintError()
        if o.json {
            print("{\"backend\":\"cpu\",\"demo\":\"\(scene.name)\",\"frames\":\(o.frames),\"msPerFrame\":\(ms),\"maxConstraintError\":\(err)}")
        } else {
            print(String(format: "cpu: %d bodies, %.3f ms/frame, final err %.5f",
                         scene.bodies.count, ms, err))
        }
    } else {
        let solver = try GPUSolver(scene: scene)
        let t0 = Date()
        for f in 0..<o.frames {
            solver.step()
            if !o.json && (f + 1) % o.statsEvery == 0 {
                let err = solver.maxConstraintError()
                var extra = ""
                if let w = o.watch {
                    let p = solver.bodyPosition(w)
                    extra = String(format: "  body%d (%.3f, %.3f, %.3f)", w, p.x, p.y, p.z)
                }
                let colors = solver.lastColorCounts.filter { $0 > 0 }.count
                print(String(format: "frame %5d  err %.5f  pairs %5d  colors %2d%@",
                             f + 1, err, solver.lastNumPairs, colors, extra))
            }
        }
        let ms = Date().timeIntervalSince(t0) * 1000 / Double(o.frames)
        let err = solver.maxConstraintError()
        if o.json {
            print("{\"backend\":\"gpu\",\"demo\":\"\(scene.name)\",\"bodies\":\(scene.bodies.count),\"frames\":\(o.frames),\"msPerFrame\":\(ms),\"maxConstraintError\":\(err),\"pairs\":\(solver.lastNumPairs)}")
        } else {
            print(String(format: "gpu: %d bodies, %.3f ms/frame, final err %.5f, %d pairs",
                         scene.bodies.count, ms, err, solver.lastNumPairs))
        }
    }

case "collect":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.collect(envs: o.envs, steps: o.frames,
                              path: "runs/pusht/data", bc: args.contains("--bc"))

case "train-bc":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.trainBC(dataPath: "runs/pusht/data", iters: o.frames,
                              batch: o.batch, latent: o.latent, lr: o.lr,
                              modelPath: "runs/pusht/model")

case "train-lawm":
    let o = parseOptions(Array(args.dropFirst(1)))
    try LatentActionPipeline.train(dataPath: "runs/pusht/data", iters: o.frames,
                                   batch: o.batch, latent: o.latent, lr: o.lr,
                                   modelPath: "runs/pusht/model")

case "solve-lawm":
    let o = parseOptions(Array(args.dropFirst(1)))
    try LatentActionPipeline.solve(modelPath: "runs/pusht/model",
                                   episodes: o.episodes, latent: o.latent)

case "solve-bc":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.solveBC(modelPath: "runs/pusht/model",
                              episodes: o.episodes, latent: o.latent)

case "train-wm":
    let o = parseOptions(Array(args.dropFirst(1)))
    if args.contains("--ensemble") {
        for m in 0..<3 {
            print("=== ensemble member \(m) ===")
            try PushTPipeline.train(dataPath: "runs/pusht/data", iters: o.frames,
                                    batch: o.batch, latent: o.latent, lr: o.lr,
                                    lambda: o.lambda, modelPath: "runs/pusht/model",
                                    member: m)
        }
    } else {
        try PushTPipeline.train(dataPath: "runs/pusht/data", iters: o.frames,
                                batch: o.batch, latent: o.latent, lr: o.lr,
                                lambda: o.lambda, modelPath: "runs/pusht/model")
    }

case "probe-wm":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.probe(dataPath: "runs/pusht/data",
                            modelPath: "runs/pusht/model", latent: o.latent)

case "oracle-pusht":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.oracle(episodes: o.episodes)

case "solve-pusht":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.solve(modelPath: "runs/pusht/model", episodes: o.episodes,
                            seed: UInt64(o.watch ?? 11),
                            latent: o.latent, debug: args.contains("--debug"),
                            oracleNull: args.contains("--oracle-null"))

case "bench":
    guard args.count > 1 else { fail("usage: avbd bench <demo>") }
    let o = parseOptions(Array(args.dropFirst(2)))
    let scene = makeScene(args[1], o)
    let solver = try GPUSolver(scene: scene)
    // warmup
    for _ in 0..<10 { solver.step() }
    if args.contains("--capture") {
        let mgr = MTLCaptureManager.shared()
        let cd = MTLCaptureDescriptor()
        cd.captureObject = solver.device
        cd.destination = .gpuTraceDocument
        cd.outputURL = URL(fileURLWithPath: "avbd-\(scene.name).gputrace")
        try mgr.startCapture(with: cd)
        for _ in 0..<3 { solver.step() }
        mgr.stopCapture()
        print("wrote avbd-\(scene.name).gputrace (open in Xcode)")
    }
    solver.profiling = args.contains("--profile")
    solver.resetProfile()
    let t0 = Date()
    var encodeS = 0.0
    let syncEach = args.contains("--syncstep")
    for _ in 0..<o.frames {
        let e0 = Date()
        solver.step()
        if syncEach { solver.sync() }
        encodeS += Date().timeIntervalSince(e0)
    }
    solver.sync()
    let ms = Date().timeIntervalSince(t0) * 1000 / Double(o.frames)
    print(String(format: "  cpu encode: %.3f ms/frame", encodeS * 1000 / Double(o.frames)))
    print(String(format: "%@: %d bodies, %d iterations, %.3f ms/frame (%.1f FPS)",
                 scene.name, scene.bodies.count, scene.settings.iterations, ms, 1000 / ms))
    if solver.profiling, solver.profileFrames > 0 {
        let n = Double(solver.profileFrames)
        let total = solver.profileNS.values.reduce(0, +)
        print(String(format: "GPU stage breakdown (%d frames, %.3f ms GPU/frame):",
                     solver.profileFrames, total / n / 1e6))
        for (name, ns) in solver.profileNS.sorted(by: { $0.value > $1.value }) {
            print(String(format: "  %-20s %8.3f ms  %5.1f%%",
                         (name as NSString).utf8String!, ns / n / 1e6,
                         ns / total * 100))
        }
        print(String(format: "  pairs %d  colors %d",
                     solver.lastNumPairs, solver.lastMaxColorUsed + 1))
    }

case "parity":
    guard args.count > 1 else { fail("usage: avbd parity <demo>") }
    let o = parseOptions(Array(args.dropFirst(2)))
    let scene = makeScene(args[1], o)
    let cpu = scene.makeCPUSolver()
    let gpu = try GPUSolver(scene: scene)
    var maxDiff: Float = 0
    for f in 0..<o.frames {
        cpu.step()
        gpu.step()
        var diff: Float = 0
        for i in 0..<scene.bodies.count {
            diff = max(diff, length(cpu.bodies[i].positionLin - gpu.bodyPosition(i)))
        }
        maxDiff = max(maxDiff, diff)
        if (f + 1) % o.statsEvery == 0 {
            print(String(format: "frame %5d  maxDiff %.5f", f + 1, diff))
        }
    }
    print(String(format: "max position divergence over %d frames: %.5f", o.frames, maxDiff))

default:
    fail("unknown command '\(command)'")
}
