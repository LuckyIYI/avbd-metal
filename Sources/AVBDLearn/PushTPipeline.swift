import Foundation
import simd
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
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
                var a = targets[e] / 2.0          // normalize to [-1, 1]
                withUnsafeBytes(of: &a) { actData.append(contentsOf: $0) }
            }
            env.step(actions: targets)
            if t % 50 == 0 { print("collect step \(t)/\(steps)") }
        }
        // final obs for the last next-obs
        obsData.append(contentsOf: env.observations())
        try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        try obsData.write(to: URL(fileURLWithPath: "\(path)/obs.bin"))
        try actData.write(to: URL(fileURLWithPath: "\(path)/act.bin"))
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

        let K = 3   // rollout depth
        func minibatch() -> [MLXArray] {
            var frames = [[UInt8]](repeating: [], count: K + 1)
            var acts = [[Float]](repeating: [], count: K)
            for _ in 0..<batch {
                let e = Int.random(in: 0..<numEnvs)
                let t = Int.random(in: 0..<(steps - K))
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
            // the goal image must not contain the ARM (a big blue blob would
            // dominate the latent distance and the planner would chase arm
            // poses instead of pushing the block) — park the finger offscreen
            let savedTip = (env.solver.bodyPosition(r.tip), env.solver.bodyRotation(r.tip))
            env.solver.setBodyPose(r.tip, position: r.center + F3(50, 50, 0.4),
                                   rotation: savedTip.1)
            let goalObs = obsArray(env, res)
            env.solver.setBodyPose(r.tip, position: savedTip.0, rotation: savedTip.1)
            let zGoal = model.encoder(goalObs)
            eval(zGoal)
            env.solver.setBodyPose(r.blockBar, position: savedBar.0, rotation: savedBar.1)
            env.solver.setBodyPose(r.blockStem, position: savedStem.0, rotation: savedStem.1)

            // MPC loop
            let horizon = 6, candidates = 192, elites = 24, cemIters = 3
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
                env.step(actions: [SIMD2(a0x * 2, a0y * 2)])
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
            print("oracle ep \(ep): \(ok ? "SUCCESS" : "fail") block (\(String(format: "%.2f", bp.x)), \(String(format: "%.2f", bp.y))) yaw \(String(format: "%.2f", byaw))")
        }
        print("oracle success rate: \(successes)/\(episodes)")
    }

    public static func obsArray(_ env: PushTEnv, _ res: Int) -> MLXArray {
        let obs = env.observations()
        let arr = MLXArray([UInt8](obs))
        return arr.reshaped([1, res, res, 3]).asType(.float32) / 255.0
    }
}
