import Foundation
import simd
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import AVBDCore

// True RL: clipped PPO + GAE on Push-T, pixels in / action out.
// The recipe follows pufferlib/CleanRL: vectorized rollouts, GAE(λ),
// per-minibatch advantage normalization, clipped policy + clipped value
// loss, entropy bonus, global grad-norm clipping, cosine LR anneal,
// time-limit value bootstrapping. The actor shares the BC policy's exact
// architecture and parameter names (encoder/h1/h2) so it warm-starts
// from bc.safetensors; value head + logStd are fresh.

/// Actor-critic: BC backbone + Gaussian head + value head.
public final class PPOPolicy: Module {
    @ModuleInfo public var encoder: LeWMEncoder
    @ModuleInfo var h1: Linear
    @ModuleInfo var h2: Linear
    @ModuleInfo var v1: Linear
    @ModuleInfo var v2: Linear
    @ParameterInfo var logStd: MLXArray

    public init(latent: Int = 192) {
        encoder = LeWMEncoder(latent: latent, inChannels: 6)
        h1 = Linear(latent, 256)
        h2 = Linear(256, 2)
        v1 = Linear(latent, 256)
        v2 = Linear(256, 1)
        logStd = MLXArray([Float](repeating: -2.3, count: 2))   // std ≈ 0.1
    }

    /// (mean action in [-1,1], value, clamped logStd)
    public func forward(_ obs: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let z = encoder(obs)
        let mu = tanh(h2(gelu(h1(z))))
        let v = v2(gelu(v1(z))).squeezed(axis: -1)
        return (mu, v, clip(logStd, min: -4, max: 0.5))
    }
}

public enum PPOPipeline {

    static let LOG_SQRT_2PI: Float = 0.9189385332

    static func gaussLogProb(_ a: MLXArray, _ mu: MLXArray, _ logStd: MLXArray) -> MLXArray {
        let std = exp(logStd)
        let t = (a - mu) / std
        return sum(-0.5 * t.square() - logStd - LOG_SQRT_2PI, axis: -1)
    }

    /// Global-norm gradient clipping (pufferlib: max_grad_norm).
    static func clipGradNorm(_ grads: ModuleParameters, maxNorm: Float) -> ModuleParameters {
        let flat = grads.flattened()
        var total = MLXArray(Float(0))
        for (_, g) in flat { total = total + g.square().sum() }
        let norm = sqrt(total)
        let scale = minimum(MLXArray(maxNorm) / (norm + 1e-6), MLXArray(Float(1)))
        return ModuleParameters.unflattened(flat.map { ($0.0, $0.1 * scale) })
    }

    /// PPO training. Macro-action cadence (one decision = 8 control steps)
    /// identical to BC/eval. Reward: per-macro-step block→goal progress +
    /// success bonus; success resets the env in place.
    public static func train(envs numEnvs: Int = 128, updates: Int = 200,
                             horizon: Int = 32, epochs: Int = 2,
                             minibatch: Int = 256, lr: Float = 3e-5,
                             latent: Int = 192, seed: UInt64 = 31,
                             initFrom: String? = "runs/pusht/model/bc.safetensors",
                             modelPath: String = "runs/pusht/model",
                             valueWarmup: Int = 10) throws {
        let gamma: Float = 0.99, gaeLambda: Float = 0.95
        let clipCoef: Float = 0.2, vfCoef: Float = 0.5, entCoef: Float = 0.005
        let maxGradNorm: Float = 0.5, maxEpLen = 60, macro = 8
        let bonus: Float = 5.0, stepPenalty: Float = 0.01

        let policy = PPOPolicy(latent: latent)
        if let p = initFrom, FileManager.default.fileExists(atPath: p) {
            let w = try loadArrays(url: URL(fileURLWithPath: p))
            try policy.update(parameters: ModuleParameters.unflattened(w), verify: [])
            print("warm-started actor from \(p)")
        } else {
            print("training from scratch (no BC init found)")
        }
        eval(policy)

        let env = try PushTEnv(numEnvs: numEnvs, seed: seed)
        let res = env.obsRes
        let frameBytes = res * res * 3
        var rng = SplitMix64(seed: seed &+ 17)
        MLXRandom.seed(seed)

        let opt = AdamW(learningRate: lr)

        // helpers ---------------------------------------------------------
        func captureFrames() -> [UInt8] { [UInt8](env.observations()) }
        func stackMLX(_ prev: [UInt8], _ cur: [UInt8], rows: [Int]? = nil) -> MLXArray {
            // [B,res,res,6] float in [0,1], channel order (prev,cur) as in BC
            let idx = rows ?? Array(0..<numEnvs)
            var bytes = [UInt8](); bytes.reserveCapacity(idx.count * 2 * frameBytes)
            for e in idx {
                bytes.append(contentsOf: prev[(e * frameBytes)..<((e + 1) * frameBytes)])
                bytes.append(contentsOf: cur[(e * frameBytes)..<((e + 1) * frameBytes)])
            }
            let o = MLXArray(bytes).reshaped([idx.count, 2, res, res, 3]).asType(.float32) / 255.0
            return concatenated([o[0..., 0, 0..., 0..., 0...],
                                 o[0..., 1, 0..., 0..., 0...]], axis: 3)
        }
        func goalDist(_ e: Int) -> Float {
            let (bp, _) = env.blockPose(e)
            return length(env.refs[e].goalPos - bp)
        }

        // rollout state ---------------------------------------------------
        env.resetAll(seed: seed)
        var targets = (0..<numEnvs).map { _ in SIMD2<Float>(1.4, 0) }
        for _ in 0..<10 { env.step(actions: targets) }
        var prevBuf = captureFrames()
        var curBuf = captureFrames()
        var epLen = [Int](repeating: 0, count: numEnvs)
        var epRet = [Float](repeating: 0, count: numEnvs)
        var dist = (0..<numEnvs).map { goalDist($0) }

        var totalEpisodes = 0, totalSuccesses = 0

        // PPO loss --------------------------------------------------------
        let lossAndGrad = valueAndGrad(model: policy) {
            (m: PPOPolicy, args: [MLXArray]) -> [MLXArray] in
            let (obs, act, oldLogp) = (args[0], args[1], args[2])
            let (adv, ret, oldV, pgCoef) = (args[3], args[4], args[5], args[6])
            let (mu, v, lsd) = m.forward(obs)
            let logp = gaussLogProb(act, mu, lsd)
            let ratio = exp(logp - oldLogp)
            let pg1 = -adv * ratio
            let pg2 = -adv * clip(ratio, min: 1 - clipCoef, max: 1 + clipCoef)
            let pgLoss = mean(maximum(pg1, pg2))
            let vClipped = oldV + clip(v - oldV, min: -clipCoef, max: clipCoef)
            let vLoss = 0.5 * mean(maximum((v - ret).square(), (vClipped - ret).square()))
            let entropy = sum(lsd + 0.5 + LOG_SQRT_2PI)
            return [pgCoef * pgLoss + vfCoef * vLoss - entCoef * entropy, pgLoss, vLoss]
        }

        let batchSize = numEnvs * horizon
        print("PPO: \(numEnvs) envs × \(horizon) macro steps = \(batchSize) batch, " +
              "\(epochs) epochs × mb \(minibatch), lr \(lr) cosine")

        for update in 0..<updates {
            // anneal (pufferlib: CosineAnnealingLR with min ratio)
            let frac = Float(update) / Float(max(updates - 1, 1))
            let curLR = lr * (0.1 + 0.9 * 0.5 * (1 + cos(.pi * frac)))
            opt.learningRate = curLR

            // ---- rollout ----
            var obsPrevSteps = [[UInt8]](); var obsCurSteps = [[UInt8]]()
            var actions = [Float](); var logps = [Float](); var values = [Float]()
            var rewards = [Float](); var dones = [Float]()
            obsPrevSteps.reserveCapacity(horizon); obsCurSteps.reserveCapacity(horizon)

            let t0 = Date()
            for _ in 0..<horizon {
                obsPrevSteps.append(prevBuf)
                obsCurSteps.append(curBuf)
                let (mu, v, lsd) = policy.forward(stackMLX(prevBuf, curBuf))
                let std = exp(lsd)
                let a = mu + MLXRandom.normal([numEnvs, 2]) * std
                let lp = gaussLogProb(a, mu, lsd)
                eval(a, v, lp)
                let aHost = a.asArray(Float.self)
                values.append(contentsOf: v.asArray(Float.self))
                logps.append(contentsOf: lp.asArray(Float.self))
                actions.append(contentsOf: aHost)

                for e in 0..<numEnvs {
                    targets[e] = SIMD2(simd_clamp(aHost[e * 2] * 3, -3, 3),
                                       simd_clamp(aHost[e * 2 + 1] * 3, -3, 3))
                }
                var nextPrev = curBuf
                for k in 0..<macro {
                    if k == macro - 1 { nextPrev = captureFrames() }
                    env.step(actions: targets)
                }
                let nextCur = captureFrames()

                // rewards / dones / resets
                var truncRows = [Int]()
                var stepDone = [Float](repeating: 0, count: numEnvs)
                var stepRew = [Float](repeating: 0, count: numEnvs)
                for e in 0..<numEnvs {
                    let nd = goalDist(e)
                    var r = (dist[e] - nd) - stepPenalty
                    epLen[e] += 1
                    if env.success(e) {
                        r += bonus
                        stepDone[e] = 1
                        totalSuccesses += 1
                    } else if epLen[e] >= maxEpLen {
                        stepDone[e] = 1
                        truncRows.append(e)   // bootstrap through the time limit
                    }
                    stepRew[e] = r
                    epRet[e] += r
                    dist[e] = nd
                }
                // time-limit bootstrap: r += γ·V(s_next) before the reset
                if !truncRows.isEmpty {
                    let (_, vT, _) = policy.forward(stackMLX(nextPrev, nextCur, rows: truncRows))
                    eval(vT)
                    let vH = vT.asArray(Float.self)
                    for (i, e) in truncRows.enumerated() {
                        stepRew[e] += gamma * vH[i]
                        epRet[e] += gamma * vH[i]
                    }
                }
                var anyReset = false
                for e in 0..<numEnvs where stepDone[e] == 1 {
                    totalEpisodes += 1
                    env.reset(e, seed: rng.next())
                    epLen[e] = 0; epRet[e] = 0
                    anyReset = true
                }
                rewards.append(contentsOf: stepRew)
                dones.append(contentsOf: stepDone)
                prevBuf = nextPrev
                curBuf = nextCur
                if anyReset {
                    // refresh frames + dist for reset envs (post-teleport)
                    let fresh = captureFrames()
                    for e in 0..<numEnvs where stepDone[e] == 1 {
                        let rge = (e * frameBytes)..<((e + 1) * frameBytes)
                        prevBuf.replaceSubrange(rge, with: fresh[rge])
                        curBuf.replaceSubrange(rge, with: fresh[rge])
                        dist[e] = goalDist(e)
                    }
                }
            }

            // bootstrap value for the last state
            let (_, vLast, _) = policy.forward(stackMLX(prevBuf, curBuf))
            eval(vLast)
            let lastV = vLast.asArray(Float.self)

            // ---- GAE(λ) ----
            var advantages = [Float](repeating: 0, count: batchSize)
            var lastGae = [Float](repeating: 0, count: numEnvs)
            for t in stride(from: horizon - 1, through: 0, by: -1) {
                for e in 0..<numEnvs {
                    let i = t * numEnvs + e
                    let nonTerm = 1 - dones[i]
                    let nextV = t == horizon - 1
                        ? lastV[e]
                        : values[(t + 1) * numEnvs + e]
                    let delta = rewards[i] + gamma * nextV * nonTerm - values[i]
                    lastGae[e] = delta + gamma * gaeLambda * nonTerm * lastGae[e]
                    advantages[i] = lastGae[e]
                }
            }
            let returns = zip(advantages, values).map(+)

            // ---- update ----
            let pgCoef: Float = update < valueWarmup ? 0 : 1
            var lastPG: Float = 0, lastVL: Float = 0
            var order = Array(0..<batchSize)
            for _ in 0..<epochs {
                order.shuffle()
                for start in stride(from: 0, to: batchSize - minibatch + 1, by: minibatch) {
                    let idx = Array(order[start..<(start + minibatch)])
                    var frames = [UInt8](); frames.reserveCapacity(minibatch * 2 * frameBytes)
                    var mbA = [Float](); var mbLP = [Float]()
                    var mbAdv = [Float](); var mbRet = [Float](); var mbV = [Float]()
                    for i in idx {
                        let (t, e) = (i / numEnvs, i % numEnvs)
                        let rge = (e * frameBytes)..<((e + 1) * frameBytes)
                        frames.append(contentsOf: obsPrevSteps[t][rge])
                        frames.append(contentsOf: obsCurSteps[t][rge])
                        mbA.append(actions[i * 2]); mbA.append(actions[i * 2 + 1])
                        mbLP.append(logps[i])
                        mbAdv.append(advantages[i])
                        mbRet.append(returns[i])
                        mbV.append(values[i])
                    }
                    // per-minibatch advantage normalization (pufferlib)
                    let am = mbAdv.reduce(0, +) / Float(minibatch)
                    let av = mbAdv.map { ($0 - am) * ($0 - am) }.reduce(0, +) / Float(minibatch)
                    let asd = max(sqrt(av), 1e-8)
                    let advN = mbAdv.map { ($0 - am) / asd }

                    let o = MLXArray(frames).reshaped([minibatch, 2, res, res, 3])
                        .asType(.float32) / 255.0
                    let obs = concatenated([o[0..., 0, 0..., 0..., 0...],
                                            o[0..., 1, 0..., 0..., 0...]], axis: 3)
                    let (losses, grads) = lossAndGrad(policy, [
                        obs,
                        MLXArray(mbA).reshaped([minibatch, 2]),
                        MLXArray(mbLP), MLXArray(advN),
                        MLXArray(mbRet), MLXArray(mbV),
                        MLXArray(pgCoef),
                    ])
                    opt.update(model: policy, gradients: clipGradNorm(grads, maxNorm: maxGradNorm))
                    eval(policy, opt)
                    lastPG = losses[1].item(Float.self)
                    lastVL = losses[2].item(Float.self)
                }
            }

            // explained variance (pufferlib monitoring)
            let rm = returns.reduce(0, +) / Float(batchSize)
            let rVar = returns.map { ($0 - rm) * ($0 - rm) }.reduce(0, +)
            var resid: Float = 0
            for i in 0..<batchSize {
                let d = returns[i] - values[i]
                resid += d * d
            }
            let ev = rVar > 1e-8 ? 1 - resid / rVar : 0
            let sr = totalEpisodes > 0 ? Float(totalSuccesses) / Float(totalEpisodes) : 0
            let std0 = exp(min(max(policy.logStd[0].item(Float.self), -4), 0.5))
            print(String(format:
                "upd %3d  lr %.1e  pg %+.4f  vf %.4f  ev %.2f  std %.3f  " +
                "eps %d  succ %.1f%%  %.1fs%@",
                update, curLR, lastPG, lastVL, ev, std0,
                totalEpisodes, sr * 100, -t0.timeIntervalSinceNow,
                pgCoef == 0 ? "  [value warmup]" : ""))
            totalEpisodes = 0; totalSuccesses = 0

            if update % 20 == 19 || update == updates - 1 {
                try saveModel(policy, modelPath: modelPath)
            }
        }
        try saveModel(policy, modelPath: modelPath)
        print("PPO done -> \(modelPath)/ppo.safetensors (+ bc_rl.safetensors actor export)")
    }

    static func saveModel(_ policy: PPOPolicy, modelPath: String) throws {
        try FileManager.default.createDirectory(atPath: modelPath,
                                                withIntermediateDirectories: true)
        let flat = Dictionary(uniqueKeysWithValues:
            policy.parameters().flattened().map { ($0.0, $0.1) })
        try save(arrays: flat, url: URL(fileURLWithPath: "\(modelPath)/ppo.safetensors"))
        // actor-only export, loadable by BCPolicy / the Lab runner
        let actor = flat.filter { $0.key.hasPrefix("encoder") || $0.key.hasPrefix("h1")
            || $0.key.hasPrefix("h2") }
        try save(arrays: actor, url: URL(fileURLWithPath: "\(modelPath)/bc_rl.safetensors"))
    }

    /// Deterministic (mean-action) eval, identical harness to solveBC.
    public static func solve(modelPath: String, episodes: Int,
                             seed: UInt64 = 11, latent: Int = 192) throws {
        let policy = PPOPolicy(latent: latent)
        let w = try loadArrays(url: URL(fileURLWithPath: "\(modelPath)/ppo.safetensors"))
        try policy.update(parameters: ModuleParameters.unflattened(w), verify: [.all])
        eval(policy)

        var successes = 0
        for ep in 0..<episodes {
            let env = try PushTEnv(numEnvs: 1, seed: seed &+ UInt64(ep) * 7)
            let res = env.obsRes
            var prevFrame = PushTPipeline.obsArray(env, res)
            env.step(actions: [env.tipPos(0)], substeps: 4)
            for _ in 0..<60 {
                let cur = PushTPipeline.obsArray(env, res)
                let (mu, _, _) = policy.forward(concatenated([prevFrame, cur], axis: 3))
                eval(mu)
                let act = SIMD2(mu[0, 0].item(Float.self) * 3, mu[0, 1].item(Float.self) * 3)
                for k in 0..<8 {
                    if k == 7 { prevFrame = PushTPipeline.obsArray(env, res) }
                    env.step(actions: [act])
                    if env.success(0) { break }
                }
                if env.success(0) { break }
            }
            let ok = env.success(0)
            successes += ok ? 1 : 0
            print("ppo episode \(ep): \(ok ? "SUCCESS" : "fail")")
        }
        print("PPO success rate: \(successes)/\(episodes)")
    }
}
