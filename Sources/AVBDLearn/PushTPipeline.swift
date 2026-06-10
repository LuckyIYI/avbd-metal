import Foundation
import simd
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import MLXLinalg
import AVBDCore

// End-to-end Push-T pipeline: parallel data collection from the AVBD
// simulator, LeWM training, and CEM planning in latent space.

/// Stateful LeWM + CEM planner for interactive use (the Robotics Lab).
public final class LeWMPlanner {
    let model: LeWorldModel
    var mu: MLXArray
    var zGoal: MLXArray? = nil
    var goalEnvSeedStamp: Int = -1
    let horizon = 6
    let candidates = 128
    let elites = 16
    let cemIters = 2

    public init(modelPath: String) throws {
        model = LeWorldModel()
        let weights = try loadArrays(url: URL(fileURLWithPath: "\(modelPath)/lewm.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        mu = MLXArray.zeros([horizon, 2])
    }

    /// One MPC step from the env's current pixels. Returns action in [-1,1].
    public func action(_ env: PushTEnv) -> SIMD2<Float>? {
        let res = env.obsRes
        if zGoal == nil || goalEnvSeedStamp != env.refs[0].tip {
            // render the goal embedding once per episode
            let r = env.refs[0]
            let savedBar = (env.solver.bodyPosition(r.blockBar), env.solver.bodyRotation(r.blockBar))
            let savedStem = (env.solver.bodyPosition(r.blockStem), env.solver.bodyRotation(r.blockStem))
            let gq = Quat(angle: r.goalYaw, axis: F3(0, 0, 1))
            let gc = r.center + F3(r.goalPos.x, r.goalPos.y, 0)
            env.solver.setBodyPose(r.blockBar, position: gc + gq.act(F3(0, 0.125, 0)) + F3(0, 0, 0.09), rotation: gq)
            env.solver.setBodyPose(r.blockStem, position: gc + gq.act(F3(0, -0.325, 0)) + F3(0, 0, 0.09), rotation: gq)
            zGoal = model.encoder(PushTPipeline.obsArray(env, res))
            eval(zGoal!)
            env.solver.setBodyPose(r.blockBar, position: savedBar.0, rotation: savedBar.1)
            env.solver.setBodyPose(r.blockStem, position: savedStem.0, rotation: savedStem.1)
            goalEnvSeedStamp = r.tip
        }
        let z0 = model.encoder(PushTPipeline.obsArray(env, res))
        var sigma = MLXArray.ones([horizon, 2]) * 0.5
        for _ in 0..<cemIters {
            let noise = MLXRandom.normal([candidates, horizon, 2])
            let acts = clip(mu.expandedDimensions(axis: 0) + noise * sigma.expandedDimensions(axis: 0),
                            min: -1, max: 1)
            var z = tiled(z0, repetitions: [candidates, 1])
            var cost = MLXArray.zeros([candidates])
            for h in 0..<horizon {
                z = model.predictor(z, acts[0..., h, 0...])
                cost = cost + mean((z - zGoal!).square(), axis: -1)
                    * (h == horizon - 1 ? 2.0 : 0.3)
            }
            eval(cost)
            let order = argSort(cost)
            let elite = acts[order[0..<elites], 0..., 0...]
            mu = mean(elite, axis: 0)
            sigma = sqrt(variance(elite, axis: 0)) + 0.02
        }
        eval(mu)
        let a = SIMD2(mu[0, 0].item(Float.self), mu[0, 1].item(Float.self))
        mu = concatenated([mu[1..., 0...], MLXArray.zeros([1, 2])], axis: 0)
        return a
    }
}

public enum PushTPipeline {

    // ---------------- data collection ----------------
    /// Random smooth waypoint policy across parallel envs. Saves
    /// (obs, action, next-obs implicit by sequence) as raw binary.
    public static func collect(envs numEnvs: Int, steps: Int,
                               path: String, seed: UInt64 = 1) throws {
        let env = try PushTEnv(numEnvs: numEnvs, seed: seed)
        let res = env.obsRes
        var rng = SplitMix64(seed: seed &+ 99)
        var targets = (0..<numEnvs).map { _ in
            SIMD2<Float>(rng.nextFloat() * 3.6 - 1.8, rng.nextFloat() * 3.6 - 1.8)
        }
        // settle
        for _ in 0..<10 { env.step(actions: targets) }

        var obsData = Data()
        var actData = Data()
        var stateData = Data()          // diagnostics only (probes), 6 floats
        obsData.reserveCapacity(numEnvs * steps * res * res * 3)
        for t in 0..<steps {
            // behavior mix (DAgger-style): mostly the goal-directed oracle
            // (the world model must see SOLVING behavior), some exploration
            for e in 0..<numEnvs {
                let roll = rng.nextFloat()
                if roll < 0.65 {
                    targets[e] = env.oracleAction(e)
                } else if roll < 0.72 {
                    targets[e] = SIMD2(rng.nextFloat() * 4.4 - 2.2,
                                       rng.nextFloat() * 4.4 - 2.2)
                } // else: keep previous waypoint
                // episode hygiene: solved envs get a fresh layout
                if env.success(e) {
                    env.reset(e, seed: rng.next())
                }
            }
            let obs = env.observations()
            obsData.append(contentsOf: obs)
            for e in 0..<numEnvs {
                // normalize over the FULL workspace
                var a = targets[e] / 3.0
                withUnsafeBytes(of: &a) { actData.append(contentsOf: $0) }
                let (bp, byaw) = env.blockPose(e)
                let tp = env.tipPos(e)
                var st: [Float] = [bp.x, bp.y, sin(byaw), cos(byaw), tp.x, tp.y]
                st.withUnsafeBytes { stateData.append(contentsOf: $0) }
            }
            env.step(actions: targets)
            if t % 50 == 0 { print("collect step \(t)/\(steps)") }
        }
        // final obs for the last next-obs
        obsData.append(contentsOf: env.observations())
        try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        try obsData.write(to: URL(fileURLWithPath: "\(path)/obs.bin"))
        try actData.write(to: URL(fileURLWithPath: "\(path)/act.bin"))
        try stateData.write(to: URL(fileURLWithPath: "\(path)/state.bin"))
        let meta = "\(numEnvs) \(steps) \(res)"
        try meta.write(toFile: "\(path)/meta.txt", atomically: true, encoding: .utf8)
        print("collected \(numEnvs * steps) transitions -> \(path)")
    }

    // ---------------- training ----------------
    public static func train(dataPath: String, iters: Int, batch: Int = 256,
                             latent: Int = 128, lr: Float = 3e-4,
                             lambda: Float = 0.5, modelPath: String) throws {
        let meta = try String(contentsOfFile: "\(dataPath)/meta.txt", encoding: .utf8)
            .split(separator: " ").map { Int($0)! }
        let (numEnvs, steps, res) = (meta[0], meta[1], meta[2])
        let obsRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/obs.bin"))
        let actRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/act.bin"))
        let frameBytes = res * res * 3
        print("dataset: \(numEnvs) envs x \(steps) steps")

        let model = LeWorldModel(latent: latent)
        let opt = AdamW(learningRate: lr)

        let K = 4   // rollout depth
        // index transitions where the BLOCK moves during the K-window:
        // without oversampling them, "the block never moves" is a free
        // prediction and the encoder learns to ignore the block entirely
        // (measured: block R^2 0.3 vs tip 0.95)
        var moving: [(Int, Int)] = []
        if let stRaw = try? Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/state.bin")) {
            stRaw.withUnsafeBytes { raw in
                let f = raw.bindMemory(to: Float.self)
                for e in 0..<numEnvs {
                    for t in 0..<(steps - K) {
                        let a = (t * numEnvs + e) * 6
                        let b = ((t + K) * numEnvs + e) * 6
                        let dx = f[b] - f[a], dy = f[b + 1] - f[a + 1]
                        if dx * dx + dy * dy > 0.0009 { moving.append((e, t)) }
                    }
                }
            }
            print("block-motion transitions: \(moving.count) / \(numEnvs * (steps - K))")
        }

        func minibatch() -> [MLXArray] {
            var frames = [[UInt8]](repeating: [], count: K + 1)
            var acts = [[Float]](repeating: [], count: K)
            for _ in 0..<batch {
                var e = Int.random(in: 0..<numEnvs)
                var t = Int.random(in: 0..<(steps - K))
                if !moving.isEmpty && Bool.random() {
                    (e, t) = moving[Int.random(in: 0..<moving.count)]
                }
                for k in 0...K {
                    let off = ((t + k) * numEnvs + e) * frameBytes
                    frames[k].append(contentsOf: obsRaw[off..<(off + frameBytes)])
                }
                for k in 0..<K {
                    let aOff = ((t + k) * numEnvs + e) * 8
                    actRaw.withUnsafeBytes { raw in
                        let f = raw.baseAddress!.advanced(by: aOff).assumingMemoryBound(to: Float.self)
                        acts[k].append(f[0]); acts[k].append(f[1])
                    }
                }
            }
            var out: [MLXArray] = frames.map {
                MLXArray($0).reshaped([batch, res, res, 3]).asType(.float32) / 255.0
            }
            out.append(contentsOf: acts.map { MLXArray($0).reshaped([batch, 2]) })
            return out
        }

        let lossAndGrad = valueAndGrad(model: model) {
            (m: LeWorldModel, args: [MLXArray]) -> [MLXArray] in
            let obsSeq = Array(args[0...K])
            let actSeq = Array(args[(K + 1)...])
            return [m.loss(obsSeq: obsSeq, actSeq: actSeq, lambda: lambda)]
        }
        for it in 0..<iters {
            let mb = minibatch()
            let (loss, grads) = lossAndGrad(model, mb)
            opt.update(model: model, gradients: grads)
            eval(model, opt)
            if it % 20 == 0 {
                print(String(format: "iter %4d  loss %.5f", it, loss[0].item(Float.self)))
            }
        }
        try FileManager.default.createDirectory(atPath: modelPath, withIntermediateDirectories: true)
        let flat = Dictionary(uniqueKeysWithValues:
            model.parameters().flattened().map { ($0.0, $0.1) })
        try save(arrays: flat, url: URL(fileURLWithPath: "\(modelPath)/lewm.safetensors"))
        print("model saved -> \(modelPath)/lewm.safetensors")
    }

    // ---------------- planning (latent MPC with CEM) ----------------
    public static func solve(modelPath: String, episodes: Int, seed: UInt64 = 11,
                             latent: Int = 128) throws {
        let model = LeWorldModel(latent: latent)
        let weights = try loadArrays(url: URL(fileURLWithPath: "\(modelPath)/lewm.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)

        var successes = 0
        for ep in 0..<episodes {
            let env = try PushTEnv(numEnvs: 1, seed: seed &+ UInt64(ep) * 7)
            let res = env.obsRes
            // goal observation: teleport the block to the goal pose, render
            let r = env.refs[0]
            let savedBar = (env.solver.bodyPosition(r.blockBar), env.solver.bodyRotation(r.blockBar))
            let savedStem = (env.solver.bodyPosition(r.blockStem), env.solver.bodyRotation(r.blockStem))
            let gq = Quat(angle: r.goalYaw, axis: F3(0, 0, 1))
            let gc = r.center + F3(r.goalPos.x, r.goalPos.y, 0)
            env.solver.setBodyPose(r.blockBar,
                                   position: gc + gq.act(F3(0, 0.125, 0)) + F3(0, 0, 0.09),
                                   rotation: gq)
            env.solver.setBodyPose(r.blockStem,
                                   position: gc + gq.act(F3(0, -0.325, 0)) + F3(0, 0, 0.09),
                                   rotation: gq)
            // GOAL ENSEMBLE: average the goal embedding over many tool-head
            // positions (all in-distribution). Tip features average out;
            // block-at-goal features dominate the cost direction.
            let savedTip = (env.solver.bodyPosition(r.tip), env.solver.bodyRotation(r.tip))
            var goalZs: [MLXArray] = []
            for k in 0..<8 {
                let a = Float(k) / 8 * 2 * .pi
                let tipP = r.center + F3(cos(a) * 1.9, sin(a) * 1.9, PushTEnv.tipHeight)
                env.solver.setBodyPose(r.tip, position: tipP, rotation: savedTip.1)
                goalZs.append(model.encoder(obsArray(env, res)))
            }
            let goalObs = obsArray(env, res)   // placeholder shape only
            _ = goalObs
            env.solver.setBodyPose(r.tip, position: savedTip.0, rotation: savedTip.1)
            let zGoal = mean(concatenated(goalZs.map { $0.expandedDimensions(axis: 0) },
                                          axis: 0), axis: 0)
            eval(zGoal)
            env.solver.setBodyPose(r.blockBar, position: savedBar.0, rotation: savedBar.1)
            env.solver.setBodyPose(r.blockStem, position: savedStem.0, rotation: savedStem.1)

            // MPC loop
            let horizon = 8, candidates = 320, elites = 32, cemIters = 4
            var mu = MLXArray.zeros([horizon, 2])
            for _ in 0..<140 {                      // control steps
                let z0 = model.encoder(obsArray(env, res))
                var sigma = MLXArray.ones([horizon, 2]) * 0.6
                for _ in 0..<cemIters {
                    let noise = MLXRandom.normal([candidates, horizon, 2])
                    let acts = clip(mu.expandedDimensions(axis: 0) + noise * sigma.expandedDimensions(axis: 0),
                                    min: -1, max: 1)
                    var z = tiled(z0, repetitions: [candidates, 1])
                    var cost = MLXArray.zeros([candidates])
                    for h in 0..<horizon {
                        z = model.predictor(z, acts[0..., h, 0...])
                        let d = mean((z - zGoal).square(), axis: -1)
                        cost = cost + d * (h == horizon - 1 ? 2.0 : 0.3)
                    }
                    eval(cost)
                    let order = argSort(cost)
                    let elite = acts[order[0..<elites], 0..., 0...]
                    mu = mean(elite, axis: 0)
                    sigma = sqrt(variance(elite, axis: 0)) + 0.02
                }
                eval(mu)
                let a0x = mu[0, 0].item(Float.self)
                let a0y = mu[0, 1].item(Float.self)
                env.step(actions: [SIMD2(a0x * 3, a0y * 3)])
                // receding horizon: shift
                mu = concatenated([mu[1..., 0...], MLXArray.zeros([1, 2])], axis: 0)
                if env.success(0) { break }
            }
            let ok = env.success(0)
            successes += ok ? 1 : 0
            let (bp, byaw) = env.blockPose(0)
            print("episode \(ep): \(ok ? "SUCCESS" : "fail") block (\(String(format: "%.2f", bp.x)), \(String(format: "%.2f", bp.y))) yaw \(String(format: "%.2f", byaw)) goal (\(r.goalPos.x), \(r.goalPos.y))")
        }
        print("success rate: \(successes)/\(episodes)")
    }

    /// Oracle: greedy geometric push controller (no learning). Establishes
    /// whether the task is feasible within the episode budget — separates
    /// environment/control issues from world-model issues.
    public static func oracle(episodes: Int = 10, seed: UInt64 = 11,
                              controlSteps: Int = 300) throws {
        var successes = 0
        for ep in 0..<episodes {
            let env = try PushTEnv(numEnvs: 1, seed: seed &+ UInt64(ep) * 7)
            let r = env.refs[0]
            for _ in 0..<controlSteps {
                let target = env.oracleAction(0)
                env.step(actions: [target])
                if env.success(0) { break }
            }
            let ok = env.success(0)
            successes += ok ? 1 : 0
            let (bp, byaw) = env.blockPose(0)
            let dGoal = length(bp - env.refs[0].goalPos)
            let nearWall = max(abs(bp.x), abs(bp.y)) > 2.45
            let cls = ok ? "SUCCESS" : (nearWall ? "fail[WALL-PINNED]"
                : dGoal < 0.45 ? "fail[NEAR-MISS]" : "fail[OPEN-FLOOR]")
            print("oracle ep \(ep): \(cls) dist \(String(format: "%.2f", dGoal)) block (\(String(format: "%.2f", bp.x)), \(String(format: "%.2f", bp.y))) yaw \(String(format: "%.2f", byaw))")
        }
        print("oracle success rate: \(successes)/\(episodes)")
    }

    public static func obsArray(_ env: PushTEnv, _ res: Int) -> MLXArray {
        let obs = env.observations()
        let arr = MLXArray([UInt8](obs))
        return arr.reshaped([1, res, res, 3]).asType(.float32) / 255.0
    }
}

// MARK: - Diagnostics (Milestone 1)

extension PushTPipeline {
    /// (a) Linear probe: does z linearly decode the block pose? (R^2 per
    ///     state dim via ridge regression on held-out data)
    /// (b) Predictor drift: ||rollout(z,a..) - enc(o_future)|| vs horizon,
    ///     normalized by the typical distance between random latents.
    public static func probe(dataPath: String, modelPath: String,
                             latent: Int = 128) throws {
        let meta = try String(contentsOfFile: "\(dataPath)/meta.txt", encoding: .utf8)
            .split(separator: " ").map { Int($0)! }
        let (numEnvs, steps, res) = (meta[0], meta[1], meta[2])
        let obsRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/obs.bin"))
        let stRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/state.bin"))
        let actRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/act.bin"))
        let frameBytes = res * res * 3

        let model = LeWorldModel(latent: latent)
        let weights = try loadArrays(url: URL(fileURLWithPath: "\(modelPath)/lewm.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)

        // ---- sample N transitions, encode, regress ----
        let N = 6000
        var frames = [UInt8](); frames.reserveCapacity(N * frameBytes)
        var states = [Float]()
        var idxs: [(Int, Int)] = []
        for _ in 0..<N {
            let e = Int.random(in: 0..<numEnvs)
            let t = Int.random(in: 0..<steps)
            idxs.append((e, t))
            let off = (t * numEnvs + e) * frameBytes
            frames.append(contentsOf: obsRaw[off..<(off + frameBytes)])
            let sOff = (t * numEnvs + e) * 24
            stRaw.withUnsafeBytes { raw in
                let f = raw.baseAddress!.advanced(by: sOff).assumingMemoryBound(to: Float.self)
                for k in 0..<6 { states.append(f[k]) }
            }
        }
        let obs = MLXArray(frames).reshaped([N, res, res, 3]).asType(.float32) / 255.0
        // encode in chunks
        var zs: [MLXArray] = []
        for c in stride(from: 0, to: N, by: 512) {
            let hi = min(c + 512, N)
            zs.append(model.encoder(obs[c..<hi]))
        }
        let z = concatenated(zs, axis: 0)
        eval(z)
        let y = MLXArray(states).reshaped([N, 6])

        // ridge regression on the first 5000, evaluate on the rest
        let nTr = 5000
        let zTr = concatenated([z[0..<nTr], MLXArray.ones([nTr, 1])], axis: 1)
        let zTe = concatenated([z[nTr...], MLXArray.ones([N - nTr, 1])], axis: 1)
        let yTr = y[0..<nTr], yTe = y[nTr...]
        let A = matmul(zTr.transposed(), zTr)
            + MLXArray.eye(latent + 1) * 1e-2
        let B = matmul(zTr.transposed(), yTr)
        let W = MLXLinalg.solve(A, B, stream: .cpu)
        let pred = matmul(zTe, W)
        let err = (pred - yTe).square().mean(axis: 0)
        let varY = yTe.variance(axis: 0)
        let r2 = 1.0 - err / varY
        eval(r2)
        let names = ["block_x", "block_y", "sin_yaw", "cos_yaw", "tip_x", "tip_y"]
        print("LATENT PROBE (R^2, higher=better, >0.8 means the encoder sees it):")
        for k in 0..<6 {
            print(String(format: "  %-8@ %.3f", names[k] as NSString, r2[k].item(Float.self)))
        }

        // ---- predictor drift over horizons ----
        let H = 8, M = 256
        var o0 = [UInt8](); var futs = [[UInt8]](repeating: [], count: H)
        var actsH = [[Float]](repeating: [], count: H)
        for _ in 0..<M {
            let e = Int.random(in: 0..<numEnvs)
            let t = Int.random(in: 0..<(steps - H))
            let off = (t * numEnvs + e) * frameBytes
            o0.append(contentsOf: obsRaw[off..<(off + frameBytes)])
            for h in 0..<H {
                let fo = ((t + h + 1) * numEnvs + e) * frameBytes
                futs[h].append(contentsOf: obsRaw[fo..<(fo + frameBytes)])
                let aOff = ((t + h) * numEnvs + e) * 8
                actRaw.withUnsafeBytes { raw in
                    let f = raw.baseAddress!.advanced(by: aOff).assumingMemoryBound(to: Float.self)
                    actsH[h].append(f[0]); actsH[h].append(f[1])
                }
            }
        }
        var zr = model.encoder(MLXArray(o0).reshaped([M, res, res, 3]).asType(.float32) / 255.0)
        // scale reference: typical distance between unrelated latents
        let perm = MLXArray((0..<M).map { Int32(($0 + 37) % M) })
        let zShuf = zr[perm]
        let refD = sqrt(mean((zr - zShuf).square())).item(Float.self)
        print("PREDICTOR DRIFT (latent err / unrelated-pair distance \(String(format: "%.3f", refD))):")
        for h in 0..<H {
            zr = model.predictor(zr, MLXArray(actsH[h]).reshaped([M, 2]))
            let zTrue = model.encoder(MLXArray(futs[h]).reshaped([M, res, res, 3]).asType(.float32) / 255.0)
            let d = sqrt(mean((zr - zTrue).square())).item(Float.self)
            print(String(format: "  h=%d  rel.err %.3f", h + 1, d / refD))
        }
    }
}
