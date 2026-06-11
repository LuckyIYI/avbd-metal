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
                               path: String, seed: UInt64 = 1,
                               bc: Bool = false) throws {
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
            // behavior mix tuned for CAUSAL contact learning: the predictor
            // must learn that blocks move ONLY when the tool path intersects
            // them, so we need counterfactuals — near-misses and feints —
            // not just oracle pushes (those teach the spurious shortcut
            // 'aiming at the block moves it')
            // decisions ALIGNED to the macro grid: every window is then a
            // constant-action macro transition (random switching left ~90%
            // of windows unusable after the constancy filter)
            if t % 8 == 0 {
            for e in 0..<numEnvs {
                var roll = rng.nextFloat()
                if bc { roll = roll < 0.92 ? 0.0 : 0.95 }   // expert + DART noise
                if roll < 0.50 {
                    targets[e] = env.oracleAction(e)
                } else if roll < 0.70 {
                    // NEAR-MISS counterfactual: waypoint beside the block,
                    // passing close without (usually) touching
                    let (bp, _) = env.blockPose(e)
                    let off = SIMD2<Float>(rng.nextFloat() - 0.5,
                                           rng.nextFloat() - 0.5)
                    let miss = bp + normalize(off + SIMD2(1e-4, 0))
                        * (0.75 + rng.nextFloat() * 0.8)
                    targets[e] = simd_clamp(miss, SIMD2(-2.9, -2.9), SIMD2(2.9, 2.9))
                } else if roll < 0.85 {
                    targets[e] = SIMD2(rng.nextFloat() * 6.0 - 3.0,
                                       rng.nextFloat() * 6.0 - 3.0)
                } // else: hold previous waypoint
                if env.success(e) {
                    env.reset(e, seed: rng.next())
                }
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
                             lambda: Float = 0.5, modelPath: String,
                             member: Int = -1) throws {
        if member >= 0 { MLXRandom.seed(UInt64(1234 + member * 777)) }
        let meta = try String(contentsOfFile: "\(dataPath)/meta.txt", encoding: .utf8)
            .split(separator: " ").map { Int($0)! }
        let (numEnvs, steps, res) = (meta[0], meta[1], meta[2])
        let obsRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/obs.bin"))
        let actRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/act.bin"))
        let frameBytes = res * res * 3
        print("dataset: \(numEnvs) envs x \(steps) steps")

        let model = LeWorldModel(latent: latent, stack: 2)
        let opt = AdamW(learningRate: lr)

        let K = 3   // rollout depth (macro-steps)
        let S = 8   // control steps per macro-step: one action ~ one reached waypoint
        // index transitions where the BLOCK moves during the K-window:
        // without oversampling them, "the block never moves" is a free
        // prediction and the encoder learns to ignore the block entirely
        // (measured: block R^2 0.3 vs tip 0.95)
        var moving: [(Int, Int)] = []
        var constOK = Set<Int>()
        // constant-action macro windows + block-motion index
        actRaw.withUnsafeBytes { araw in
            let af = araw.bindMemory(to: Float.self)
            (try? Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/state.bin")))?
                .withUnsafeBytes { raw in
                let f = raw.bindMemory(to: Float.self)
                for e in 0..<numEnvs {
                    for t in S..<(steps - K * S) {
                        // constant action over the whole K*S window?
                        var ok = true
                        let a0x = af[(t * numEnvs + e) * 2]
                        let a0y = af[(t * numEnvs + e) * 2 + 1]
                        for u in stride(from: t, to: t + K * S, by: 2) {
                            let ax = af[(u * numEnvs + e) * 2]
                            let ay = af[(u * numEnvs + e) * 2 + 1]
                            if abs(ax - a0x) > 1e-5 || abs(ay - a0y) > 1e-5 { ok = false; break }
                        }
                        if !ok { continue }
                        constOK.insert(t * numEnvs + e)
                        let a = (t * numEnvs + e) * 6
                        let b = ((t + K * S) * numEnvs + e) * 6
                        let dx = f[b] - f[a], dy = f[b + 1] - f[a + 1]
                        if dx * dx + dy * dy > 0.0016 { moving.append((e, t)) }
                    }
                }
            }
        }
        let constList = constOK.map { ($0 % numEnvs, $0 / numEnvs) }
        print("constant-action macro windows: \(constList.count); with block motion: \(moving.count)")

        func minibatch() -> [MLXArray] {
            // stacked observations: channels = [frame_{t-1}, frame_t]
            var framesA = [[UInt8]](repeating: [], count: K + 1)  // t-1 parts
            var framesB = [[UInt8]](repeating: [], count: K + 1)  // t parts
            var acts = [[Float]](repeating: [], count: K)
            for _ in 0..<batch {
                var (e, t) = constList.isEmpty ? (0, S) : constList[Int.random(in: 0..<constList.count)]
                if !moving.isEmpty && Bool.random() {
                    (e, t) = moving[Int.random(in: 0..<moving.count)]
                }
                for k in 0...K {
                    let u = t + k * S
                    let offPrev = ((u - S) * numEnvs + e) * frameBytes
                    let off = (u * numEnvs + e) * frameBytes
                    framesA[k].append(contentsOf: obsRaw[offPrev..<(offPrev + frameBytes)])
                    framesB[k].append(contentsOf: obsRaw[off..<(off + frameBytes)])
                }
                for k in 0..<K {
                    let aOff = ((t + k * S) * numEnvs + e) * 8
                    actRaw.withUnsafeBytes { raw in
                        let f = raw.baseAddress!.advanced(by: aOff).assumingMemoryBound(to: Float.self)
                        acts[k].append(f[0]); acts[k].append(f[1])
                    }
                }
            }
            var out: [MLXArray] = []
            for k in 0...K {
                let a = MLXArray(framesA[k]).reshaped([batch, res, res, 3]).asType(.float32) / 255.0
                let b = MLXArray(framesB[k]).reshaped([batch, res, res, 3]).asType(.float32) / 255.0
                out.append(concatenated([a, b], axis: 3))
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
        let name = member >= 0 ? "lewm-\(member).safetensors" : "lewm.safetensors"
        try save(arrays: flat, url: URL(fileURLWithPath: "\(modelPath)/\(name)"))
        print("model saved -> \(modelPath)/\(name)")
    }

    // ---------------- planning (latent MPC with CEM) ----------------
    public static func solve(modelPath: String, episodes: Int, seed: UInt64 = 11,
                             latent: Int = 128, debug: Bool = false,
                             oracleNull: Bool = false) throws {
        // load an ensemble if present (lewm-0/1/2...), else the single model.
        // Disagreement across independently-initialized models marks the OOD
        // regions the planner exploits (PETS-style epistemic penalty).
        var models: [LeWorldModel] = []
        for m in 0..<5 {
            let url = URL(fileURLWithPath: "\(modelPath)/lewm-\(m).safetensors")
            guard FileManager.default.fileExists(atPath: url.path) else { break }
            let mm = LeWorldModel(latent: latent, stack: 2)
            try mm.update(parameters: ModuleParameters.unflattened(try loadArrays(url: url)),
                          verify: [.all])
            eval(mm)
            models.append(mm)
        }
        let model: LeWorldModel
        if models.isEmpty {
            print("(single model)")
            model = LeWorldModel(latent: latent, stack: 2)
            let weights = try loadArrays(url: URL(fileURLWithPath: "\(modelPath)/lewm.safetensors"))
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
            eval(model)
        } else {
            print("(ensemble of \(models.count))")
            model = models[0]
        }
        // belief decoder (probe ridge matrix), for debug introspection only
        let probeW = try? loadArrays(url: URL(fileURLWithPath: "\(modelPath)/probe.safetensors"))["W"]
        func decodeBlock(_ z: MLXArray) -> SIMD2<Float> {
            guard let W = probeW else { return SIMD2(0, 0) }
            let zb = concatenated([z, MLXArray.ones([z.dim(0), 1])], axis: 1)
            let st = matmul(zb, W)
            eval(st)
            return SIMD2(st[0, 0].item(Float.self), st[0, 1].item(Float.self))
        }

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
            // GOAL IMAGE: block at the goal pose, tool head parked at a
            // natural solved-state position just behind it. (An ensemble
            // averaged over a tip ring creates a 'park at the ring centroid'
            // attractor in the cost that fights distant approaches.)
            let savedTip = (env.solver.bodyPosition(r.tip), env.solver.bodyRotation(r.tip))
            let tipGoalP = r.center + F3(r.goalPos.x - 0.62, r.goalPos.y - 0.62,
                                         PushTEnv.tipHeight)
            env.solver.setBodyPose(r.tip, position: tipGoalP, rotation: savedTip.1)
            let g = obsArray(env, res)
            let gStack = concatenated([g, g], axis: 3)
            // each ensemble member lives in its OWN latent space: encode the
            // goal separately per member
            let goalZs = models.isEmpty ? [model.encoder(gStack)]
                                        : models.map { $0.encoder(gStack) }
            env.solver.setBodyPose(r.tip, position: savedTip.0, rotation: savedTip.1)
            let zGoalList = goalZs
            let zGoal = goalZs[0]
            for z in zGoalList { eval(z) }
            env.solver.setBodyPose(r.blockBar, position: savedBar.0, rotation: savedBar.1)
            env.solver.setBodyPose(r.blockStem, position: savedStem.0, rotation: savedStem.1)

            // Macro-MPC: each planned action is a waypoint HELD for `rep`
            // control steps (the model was trained on exactly these
            // macro-transitions) — one slot can cross half the arena, so
            // contact-making plans are densely sampled.
            let horizon = 6, candidates = 320, elites = 32, cemIters = 4
            let rep = 8
            var mu = MLXArray.zeros([horizon, 2])
            var prevFrame = obsArray(env, res)
            for _ in 0..<60 {                       // planning steps
                let curFrame = obsArray(env, res)
                let curStack = concatenated([prevFrame, curFrame], axis: 3)
                let z0List = models.isEmpty ? [model.encoder(curStack)]
                                            : models.map { $0.encoder(curStack) }
                let z0 = z0List[0]
                // HYBRID PLANNER (PLDM-style): CEM seeds a coarse solution,
                // then GRADIENT DESCENT through the smooth predictor refines
                // it — sampling alone cannot localize contact-making plans,
                // descent alone gets stuck in flat regions
                var sigma = MLXArray.ones([horizon, 2]) * 0.6
                func planCost(_ acts: MLXArray) -> MLXArray {
                    // acts: (B, horizon, 2) in [-1, 1]
                    let B = acts.dim(0)
                    if models.count >= 2 {
                        // ensemble: each member rolls out in ITS OWN latent
                        // space (own z0, own goal); disagreement across the
                        // per-member costs marks epistemic uncertainty
                        var dists = [MLXArray]()
                        for (mi, mm) in models.enumerated() {
                            var z = tiled(z0List[mi], repetitions: [B, 1])
                            var c = MLXArray.zeros([B])
                            for h in 0..<horizon {
                                z = mm.predictor(z, acts[0..., h, 0...])
                                let d = mean((z - zGoalList[mi]).square(), axis: -1)
                                c = c + d * (h == horizon - 1 ? 2.0 : 0.3)
                                c = c + abs(mean(z.square(), axis: -1) - 1.0) * 1.5
                            }
                            dists.append(c)
                        }
                        let stackC = concatenated(dists.map { $0.expandedDimensions(axis: 0) }, axis: 0)
                        return mean(stackC, axis: 0) + sqrt(variance(stackC, axis: 0)) * 2.0
                    }
                    var z = tiled(z0, repetitions: [B, 1])
                    var cost = MLXArray.zeros([B])
                    for h in 0..<horizon {
                        z = model.predictor(z, acts[0..., h, 0...])
                        let d = mean((z - zGoal).square(), axis: -1)
                        cost = cost + d * (h == horizon - 1 ? 2.0 : 0.3)
                        let energy = abs(mean(z.square(), axis: -1) - 1.0)
                        cost = cost + energy * 1.5
                    }
                    return cost
                }
                for _ in 0..<cemIters {
                    let deltas = MLXRandom.normal([candidates, horizon, 2]) * 0.45
                    let noise = cumsum(deltas, axis: 1)
                    let acts = clip(mu.expandedDimensions(axis: 0) + noise * sigma.expandedDimensions(axis: 0),
                                    min: -1, max: 1)
                    let cost = planCost(acts)
                    eval(cost)
                    let order = argSort(cost)
                    let elite = acts[order[0..<elites], 0..., 0...]
                    mu = mean(elite, axis: 0)
                    sigma = sqrt(variance(elite, axis: 0)) + 0.02
                }
                // gradient refinement from the CEM solution
                var a = mu.expandedDimensions(axis: 0)
                let lossFn: ([MLXArray]) -> [MLXArray] = { args in
                    [planCost(clip(args[0], min: -1, max: 1)).sum()]
                }
                let gradFn = grad(lossFn)
                for _ in 0..<24 {
                    let g = gradFn([a])[0]
                    a = clip(a - g * 0.08, min: -1, max: 1)
                }
                eval(a)
                mu = a[0]
                eval(mu)
                if debug {
                    var zb = z0
                    for h in 0..<horizon {
                        zb = model.predictor(zb, mu[h..<(h+1), 0...])
                    }
                    let believedBlock = decodeBlock(zb)
                    let believedNow = decodeBlock(z0)
                    let (bpD, _) = env.blockPose(0)
                    let tpD = env.tipPos(0)
                    print(String(format: "  plan: tip(%.2f,%.2f) block(%.2f,%.2f) bel-now(%.2f,%.2f) bel-end(%.2f,%.2f) act(%.2f,%.2f)",
                                 tpD.x, tpD.y, bpD.x, bpD.y,
                                 believedNow.x, believedNow.y,
                                 believedBlock.x, believedBlock.y,
                                 mu[0,0].item(Float.self) * 3, mu[0,1].item(Float.self) * 3))
                }
                var a0x = mu[0, 0].item(Float.self)
                var a0y = mu[0, 1].item(Float.self)
                if oracleNull {                  // harness null test
                    let oa = env.oracleAction(0)
                    a0x = oa.x / 3; a0y = oa.y / 3
                }
                // execute; capture prevFrame ONE control step before the end
                // so the next stack matches the training spacing (a
                // duplicated-frame stack is OOD and scrambles the encoder)
                for k in 0..<rep {
                    if k == rep - 1 { prevFrame = obsArray(env, res) }
                    env.step(actions: [SIMD2(a0x * 3, a0y * 3)])
                    if env.success(0) { break }
                }
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

        let model = LeWorldModel(latent: latent, stack: 2)
        let weights = try loadArrays(url: URL(fileURLWithPath: "\(modelPath)/lewm.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)

        // ---- sample N transitions, encode (stacked), regress ----
        let N = 6000
        var frames = [UInt8](); frames.reserveCapacity(N * frameBytes * 2)
        var states = [Float]()
        var idxs: [(Int, Int)] = []
        for _ in 0..<N {
            let e = Int.random(in: 0..<numEnvs)
            let t = Int.random(in: 1..<steps)
            idxs.append((e, t))
            let offPrev = ((t - 1) * numEnvs + e) * frameBytes
            frames.append(contentsOf: obsRaw[offPrev..<(offPrev + frameBytes)])
            let off = (t * numEnvs + e) * frameBytes
            frames.append(contentsOf: obsRaw[off..<(off + frameBytes)])
            let sOff = (t * numEnvs + e) * 24
            stRaw.withUnsafeBytes { raw in
                let f = raw.baseAddress!.advanced(by: sOff).assumingMemoryBound(to: Float.self)
                for k in 0..<6 { states.append(f[k]) }
            }
        }
        let obs = MLXArray(frames).reshaped([N, 2, res, res, 3]).asType(.float32) / 255.0
        // interleave the two frames as channels
        let obsStacked = concatenated([obs[0..., 0, 0..., 0..., 0...],
                                       obs[0..., 1, 0..., 0..., 0...]], axis: 3)
        // encode in chunks
        var zs: [MLXArray] = []
        for c in stride(from: 0, to: N, by: 512) {
            let hi = min(c + 512, N)
            zs.append(model.encoder(obsStacked[c..<hi]))
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
        try? save(arrays: ["W": W], url: URL(fileURLWithPath: "\(modelPath)/probe.safetensors"))
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
            let t = Int.random(in: 1..<(steps - H))
            let offP = ((t - 1) * numEnvs + e) * frameBytes
            o0.append(contentsOf: obsRaw[offP..<(offP + frameBytes)])
            let off = (t * numEnvs + e) * frameBytes
            o0.append(contentsOf: obsRaw[off..<(off + frameBytes)])
            for h in 0..<H {
                let fp = ((t + h) * numEnvs + e) * frameBytes
                futs[h].append(contentsOf: obsRaw[fp..<(fp + frameBytes)])
                let fo = ((t + h + 1) * numEnvs + e) * frameBytes
                futs[h].append(contentsOf: obsRaw[fo..<(fo + frameBytes)])
                let aOff = ((t + h) * numEnvs + e) * 8
                actRaw.withUnsafeBytes { raw in
                    let f = raw.baseAddress!.advanced(by: aOff).assumingMemoryBound(to: Float.self)
                    actsH[h].append(f[0]); actsH[h].append(f[1])
                }
            }
        }
        let o0r = MLXArray(o0).reshaped([M, 2, res, res, 3]).asType(.float32) / 255.0
        var zr = model.encoder(concatenated([o0r[0..., 0, 0..., 0..., 0...],
                                             o0r[0..., 1, 0..., 0..., 0...]], axis: 3))
        // scale reference: typical distance between unrelated latents
        let perm = MLXArray((0..<M).map { Int32(($0 + 37) % M) })
        let zShuf = zr[perm]
        let refD = sqrt(mean((zr - zShuf).square())).item(Float.self)
        print("PREDICTOR DRIFT (latent err / unrelated-pair distance \(String(format: "%.3f", refD))):")
        for h in 0..<H {
            zr = model.predictor(zr, MLXArray(actsH[h]).reshaped([M, 2]))
            let fr = MLXArray(futs[h]).reshaped([M, 2, res, res, 3]).asType(.float32) / 255.0
            let zTrue = model.encoder(concatenated([fr[0..., 0, 0..., 0..., 0...],
                                                    fr[0..., 1, 0..., 0..., 0...]], axis: 3))
            let d = sqrt(mean((zr - zTrue).square())).item(Float.self)
            print(String(format: "  h=%d  rel.err %.3f", h + 1, d / refD))
        }
    }
}

// MARK: - Behavior cloning (pixels -> action, oracle demonstrations)

/// Goal-conditioned visuomotor policy: the LeWM encoder architecture with an
/// action head. Pixels in, action out — no privileged state at inference.
public final class BCPolicy: Module {
    @ModuleInfo public var encoder: LeWMEncoder
    @ModuleInfo var h1: Linear
    @ModuleInfo var h2: Linear

    public init(latent: Int = 192) {
        encoder = LeWMEncoder(latent: latent, inChannels: 6)
        h1 = Linear(latent, 256)
        h2 = Linear(256, 2)
    }

    public func callAsFunction(_ obs: MLXArray) -> MLXArray {
        tanh(h2(gelu(h1(encoder(obs)))))
    }
}

extension PushTPipeline {
    /// Train the BC policy on macro-boundary (stacked obs, expert action)
    /// pairs from a --bc collection.
    public static func trainBC(dataPath: String, iters: Int, batch: Int = 256,
                               latent: Int = 192, lr: Float = 3e-4,
                               modelPath: String) throws {
        let meta = try String(contentsOfFile: "\(dataPath)/meta.txt", encoding: .utf8)
            .split(separator: " ").map { Int($0)! }
        let (numEnvs, steps, res) = (meta[0], meta[1], meta[2])
        let obsRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/obs.bin"))
        let actRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/act.bin"))
        let frameBytes = res * res * 3

        // macro decision points: t % 8 == 0, t >= 8 (need the prev frame)
        var points: [(Int, Int)] = []
        for e in 0..<numEnvs {
            for t in stride(from: 8, to: steps, by: 8) { points.append((e, t)) }
        }
        print("BC dataset: \(points.count) macro decisions")

        let policy = BCPolicy(latent: latent)
        let opt = AdamW(learningRate: lr)
        func minibatch() -> (MLXArray, MLXArray) {
            var frames = [UInt8](); var acts = [Float]()
            for _ in 0..<batch {
                let (e, t) = points[Int.random(in: 0..<points.count)]
                let offPrev = ((t - 1) * numEnvs + e) * frameBytes
                frames.append(contentsOf: obsRaw[offPrev..<(offPrev + frameBytes)])
                let off = (t * numEnvs + e) * frameBytes
                frames.append(contentsOf: obsRaw[off..<(off + frameBytes)])
                let aOff = (t * numEnvs + e) * 8
                actRaw.withUnsafeBytes { raw in
                    let f = raw.baseAddress!.advanced(by: aOff).assumingMemoryBound(to: Float.self)
                    acts.append(f[0]); acts.append(f[1])
                }
            }
            let o = MLXArray(frames).reshaped([batch, 2, res, res, 3]).asType(.float32) / 255.0
            let stacked = concatenated([o[0..., 0, 0..., 0..., 0...],
                                        o[0..., 1, 0..., 0..., 0...]], axis: 3)
            return (stacked, MLXArray(acts).reshaped([batch, 2]))
        }
        let lossAndGrad = valueAndGrad(model: policy) {
            (m: BCPolicy, args: [MLXArray]) -> [MLXArray] in
            [mean((m(args[0]) - args[1]).square())]
        }
        for it in 0..<iters {
            let (o, a) = minibatch()
            let (loss, grads) = lossAndGrad(policy, [o, a])
            opt.update(model: policy, gradients: grads)
            eval(policy, opt)
            if it % 100 == 0 {
                print(String(format: "bc iter %5d  loss %.5f", it, loss[0].item(Float.self)))
            }
        }
        try FileManager.default.createDirectory(atPath: modelPath, withIntermediateDirectories: true)
        let flat = Dictionary(uniqueKeysWithValues:
            policy.parameters().flattened().map { ($0.0, $0.1) })
        try save(arrays: flat, url: URL(fileURLWithPath: "\(modelPath)/bc.safetensors"))
        print("policy saved -> \(modelPath)/bc.safetensors")
    }

    /// Evaluate the BC policy: pixels in, action out, macro cadence.
    public static func solveBC(modelPath: String, episodes: Int,
                               seed: UInt64 = 11, latent: Int = 192) throws {
        let policy = BCPolicy(latent: latent)
        let weights = try loadArrays(url: URL(fileURLWithPath: "\(modelPath)/bc.safetensors"))
        try policy.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(policy)

        var successes = 0
        for ep in 0..<episodes {
            let env = try PushTEnv(numEnvs: 1, seed: seed &+ UInt64(ep) * 7)
            let res = env.obsRes
            var prevFrame = obsArray(env, res)
            // settle one macro step so the stack is honest
            env.step(actions: [env.tipPos(0)], substeps: 4)
            for _ in 0..<60 {
                let cur = obsArray(env, res)
                let a = policy(concatenated([prevFrame, cur], axis: 3))
                eval(a)
                let act = SIMD2(a[0, 0].item(Float.self) * 3, a[0, 1].item(Float.self) * 3)
                for k in 0..<8 {
                    if k == 7 { prevFrame = obsArray(env, res) }
                    env.step(actions: [act])
                    if env.success(0) { break }
                }
                if env.success(0) { break }
            }
            let ok = env.success(0)
            successes += ok ? 1 : 0
            let (bp, _) = env.blockPose(0)
            print("bc episode \(ep): \(ok ? "SUCCESS" : "fail") block (\(String(format: "%.2f", bp.x)), \(String(format: "%.2f", bp.y)))")
        }
        print("BC success rate: \(successes)/\(episodes)")
    }
}
