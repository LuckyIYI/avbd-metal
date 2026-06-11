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
    var res: Int? = nil
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
        case "--res": i += 1; o.res = Int(args[i])
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
    guard var scene = Demos.make(name, scale: o.scale, res: o.res) else {
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

case "rodexp":
    // Inextensibility experiment: flagwhip under three structural-edge
    // regimes. Reports per-window KE maxima (envelope must decay), worst
    // structural stretch, and the rod dual magnitude.
    let o = parseOptions(Array(args.dropFirst(1)))
    let variants: [(String, Float, Bool, Float)] = [
        ("stiff-5k", 5000, false, 0),          // current default (toy: stretches)
        ("stiff-2e5", 2e5, false, 0),          // honest stiffness candidate
        ("hard-rods", 0, true, 0),             // AL rods, no decay (pump repro)
        ("hard-decay", 0, true, 16),           // AL rods + rotation decay
    ]
    let frames = o.frames > 300 ? o.frames : 3600
    for (name, k, hard, decay) in variants {
        var scene = Demos.flagwhip(res: o.res ?? 16,
                                   structuralK: hard ? 5000 : k,
                                   hardRods: hard)
        scene.settings.rodDecayPow = decay
        if let it = o.iterations { scene.settings.iterations = it }
        let solver = try GPUSolver(scene: scene)
        var windowMax: [Float] = []
        var cur: Float = 0
        var worstStretch: Float = 0
        for f in 0..<frames {
            solver.step()
            if f % 10 == 0 {
                var ke: Float = 0
                for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
                    ke += 0.5 * solver.bodyMass(b) * length_squared(solver.bodyVelocity(b))
                }
                cur = max(cur, ke)
            }
            if (f + 1) % 600 == 0 {
                windowMax.append(cur)
                cur = 0
                if f > frames / 2 {
                    let (_, st) = solver.debugClothMetrics()
                    worstStretch = max(worstStretch, st)
                }
            }
        }
        let envelope = windowMax.map { String(format: "%.3f", $0) }.joined(separator: " ")
        let growing = windowMax.count >= 3
            && windowMax.last! > 1.5 * windowMax[1]
        print(String(format: "%@: KE windows [%@] %@  stretch %.4f",
                     name, envelope, growing ? "GROWING(PUMP)" : "decaying",
                     worstStretch))
    }

case "clothgate":
    // Cloth gate runner: step a demo and report element-contact metrics
    // (worst V-T clearance, worst structural stretch, soft contact count).
    guard args.count > 1 else { fail("usage: avbd clothgate <demo>") }
    let o = parseOptions(Array(args.dropFirst(2)))
    let scene = makeScene(args[1], o)
    let solver = try GPUSolver(scene: scene)
    print("bodies \(scene.bodies.count)  tris \(scene.tris.count)  persistent-capacity \(solver.persistentCapacity)")
    var worstGap: Float = .greatestFiniteMagnitude
    var worstStretch: Float = 0
    var ke: Float = 0
    for f in 0..<o.frames {
        solver.step()
        if (f + 1) % o.statsEvery == 0 || f == o.frames - 1 {
            let (gap, stretch) = solver.debugClothMetrics()
            // KE over particles (cloth energy envelope) + fastest node
            ke = 0
            var vmax: Float = 0
            var fastest = -1
            for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
                let v = solver.bodyVelocity(b)
                let s2 = length_squared(v)
                ke += 0.5 * solver.bodyMass(b) * s2
                if s2 > vmax * vmax { vmax = s2.squareRoot(); fastest = b }
            }
            var loc = ""
            if fastest >= 0 {
                let p = solver.bodyPosition(fastest)
                loc = String(format: "  vmax %6.2f @%d (%.2f, %.2f, %.2f)",
                             vmax, fastest, p.x, p.y, p.z)
            }
            if ProcessInfo.processInfo.environment["AVBD_ZONES"] != nil {
                var zk = [Float](repeating: 0, count: 3)
                for b in 0..<scene.bodies.count where scene.bodies[b].isParticle {
                    let z = solver.bodyPosition(b).z
                    let v = solver.bodyVelocity(b)
                    let e = 0.5 * solver.bodyMass(b) * length_squared(v)
                    zk[z < 0.3 ? 0 : (z < 1.0 ? 1 : 2)] += e
                }
                loc += String(format: "  zones KE [pool %.3f skirt %.3f crown %.3f]",
                              zk[0], zk[1], zk[2])
            }
            if f > o.frames / 2 {       // settled-half metrics
                worstGap = min(worstGap, gap)
                worstStretch = max(worstStretch, stretch)
            }
            print(String(format: "frame %5d  gap %+.4f  stretch %.4f  soft %5d  pairs %5d  KE %.4f%@",
                         f + 1, gap, stretch, solver.lastNumSoft, solver.lastNumPairs, ke, loc))
        }
    }
    print(String(format: "settled-half: worstGap %+.4f  worstStretch %.4f",
                 worstGap == .greatestFiniteMagnitude ? 0 : worstGap, worstStretch))
    let (wa, wb) = solver.lastWorstSpring
    if wa >= 0 {
        let p1 = solver.bodyPosition(wa), p2 = solver.bodyPosition(wb)
        print(String(format: "worst spring %d-%d  (%.2f,%.2f,%.2f) - (%.2f,%.2f,%.2f)",
                     wa, wb, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z))
        if solver.lastWorstSpringIdx >= 0 {
            let (lam, pen, c0, rest) = solver.debugSpringDual(solver.lastWorstSpringIdx)
            print(String(format: "  dual: lambda %.3f  penalty %.1f  C0 %.4f  rest %.4f  len %.4f",
                         lam, pen, c0, rest, distance(p1, p2)))
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

case "collect-dagger":
    let o = parseOptions(Array(args.dropFirst(1)))
    try PushTPipeline.collectDagger(envs: o.envs, steps: o.frames,
                                    path: "runs/pusht/data",
                                    policyPath: "runs/pusht/model",
                                    latent: o.latent)

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
