import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom
import simd
import SimCore
import PhysicsAVBD
import Robotics
import RL

// la-leWorldModel-style discrete latent actions (after the-puzzler/le-wm):
// an inverse-dynamics module infers what "action" happened between two
// latent states, a VQ codebook quantizes it to one of NUM_CODES discrete
// codes, and the predictor is conditioned on the code. Planning searches
// over discrete code sequences — with only 8 codes, every conditioning is
// densely covered by training data, so the planner CANNOT probe the model
// out-of-distribution (the failure mode that defeated continuous CEM).

public final class VectorQuantizer: Module {
    @ParameterInfo public var codebook: MLXArray
    public let numCodes: Int

    public init(numCodes: Int = 8, dim: Int = 32) {
        self.numCodes = numCodes
        let scale = 1.0 / Float(numCodes)
        _codebook = ParameterInfo(wrappedValue:
            MLXRandom.uniform(low: -scale, high: scale, [numCodes, dim]))
    }

    /// returns (quantized with straight-through, indices, vqLoss)
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // x: (B, dim)
        let d = (x.expandedDimensions(axis: 1) - codebook.expandedDimensions(axis: 0))
            .square().sum(axis: -1)                       // (B, numCodes)
        let idx = argMin(d, axis: -1)                     // (B)
        let q = codebook[idx]                             // (B, dim)
        let codebookLoss = mean((q - stopGradient(x)).square())
        let commitLoss = mean((stopGradient(q) - x).square()) * 0.25
        let st = x + stopGradient(q - x)                  // straight-through
        return (st, idx, codebookLoss + commitLoss)
    }
}

public final class LatentActionWM: Module {
    @ModuleInfo public var encoder: LeWMEncoder
    @ModuleInfo public var invDyn: Linear        // (z, z') -> action embedding
    @ModuleInfo public var invDyn2: Linear
    @ModuleInfo public var vq: VectorQuantizer
    @ModuleInfo public var codeEmbed: Linear     // action code vec -> conditioning
    @ModuleInfo public var p1: Linear
    @ModuleInfo public var p2: Linear
    @ModuleInfo public var p3: Linear
    @ModuleInfo public var t1: Linear            // action translator
    @ModuleInfo public var t2: Linear
    public let latent: Int
    public let aDim = 32

    public init(latent: Int = 192, numCodes: Int = 8) {
        self.latent = latent
        encoder = LeWMEncoder(latent: latent, inChannels: 6)
        invDyn = Linear(latent * 2, 256)
        invDyn2 = Linear(256, aDim)
        vq = VectorQuantizer(numCodes: numCodes, dim: aDim)
        codeEmbed = Linear(aDim, 128)
        p1 = Linear(latent + 128, 384)
        p2 = Linear(384, 384)
        p3 = Linear(384, latent)
        t1 = Linear(latent + aDim, 128)
        t2 = Linear(128, 2)
    }

    public func predict(_ z: MLXArray, codeVec: MLXArray) -> MLXArray {
        let c = gelu(codeEmbed(codeVec))
        var h = concatenated([z, c], axis: -1)
        h = gelu(p1(h))
        h = gelu(p2(h))
        return z + p3(h)
    }

    public func translate(_ z: MLXArray, codeVec: MLXArray) -> MLXArray {
        tanh(t2(gelu(t1(concatenated([z, codeVec], axis: -1)))))
    }

    /// joint loss: prediction + SIGReg + VQ + action translation
    public func loss(obsSeq: [MLXArray], actSeq: [MLXArray],
                     lambda: Float = 0.09) -> MLXArray {
        let zs = obsSeq.map { encoder($0) }
        var pred = MLXArray(Float(0))
        var vqL = MLXArray(Float(0))
        var transL = MLXArray(Float(0))
        var z = zs[0]
        for k in 0..<actSeq.count {
            // infer the latent action between TRUE consecutive states
            let aEmb = invDyn2(gelu(invDyn(concatenated([zs[k], zs[k + 1]], axis: -1))))
            let (q, _, vql) = vq(aEmb)
            vqL = vqL + vql
            pred = pred + mean((predict(z, codeVec: q) - zs[k + 1]).square())
            // translator grounds codes in real env actions
            transL = transL + mean((translate(zs[k], codeVec: q) - actSeq[k]).square())
            z = predict(z, codeVec: q)
        }
        let n = Float(actSeq.count)
        let reg = LeWorldModel.sigReg(concatenated(zs, axis: 0), sketches: 256)
        return pred / n + vqL / n + transL / n + lambda * reg
    }
}


public enum LatentActionPipeline {
    /// Train the discrete-latent-action world model on macro transitions.
    public static func train(dataPath: String, iters: Int, batch: Int = 192,
                             latent: Int = 192, lr: Float = 3e-4,
                             modelPath: String) throws {
        let meta = try String(contentsOfFile: "\(dataPath)/meta.txt", encoding: .utf8)
            .split(separator: " ").map { Int($0)! }
        let (numEnvs, steps, res) = (meta[0], meta[1], meta[2])
        let obsRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/obs.bin"))
        let actRaw = try Data(contentsOf: URL(fileURLWithPath: "\(dataPath)/act.bin"))
        let frameBytes = res * res * 3
        let K = 3, S = 8
        let model = LatentActionWM(latent: latent)
        let opt = AdamW(learningRate: lr)

        func minibatch() -> [MLXArray] {
            var framesA = [[UInt8]](repeating: [], count: K + 1)
            var framesB = [[UInt8]](repeating: [], count: K + 1)
            var acts = [[Float]](repeating: [], count: K)
            for _ in 0..<batch {
                let e = Int.random(in: 0..<numEnvs)
                let tBase = Int.random(in: 1..<((steps - K * S - S) / S))
                let t = tBase * S
                for k in 0...K {
                    let u = t + k * S
                    let offPrev = ((u - 1) * numEnvs + e) * frameBytes
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
            (m: LatentActionWM, args: [MLXArray]) -> [MLXArray] in
            let obsSeq = Array(args[0...K])
            let actSeq = Array(args[(K + 1)...])
            return [m.loss(obsSeq: obsSeq, actSeq: actSeq)]
        }
        for it in 0..<iters {
            let mb = minibatch()
            let (loss, grads) = lossAndGrad(model, mb)
            opt.update(model: model, gradients: grads)
            eval(model, opt)
            if it % 100 == 0 {
                print(String(format: "lawm iter %5d  loss %.5f", it, loss[0].item(Float.self)))
            }
        }
        try FileManager.default.createDirectory(atPath: modelPath, withIntermediateDirectories: true)
        let flat = Dictionary(uniqueKeysWithValues:
            model.parameters().flattened().map { ($0.0, $0.1) })
        try save(arrays: flat, url: URL(fileURLWithPath: "\(modelPath)/lawm.safetensors"))
        print("model saved -> \(modelPath)/lawm.safetensors")
    }

    /// Plan over DISCRETE code sequences: random shooting + refinement.
    /// 8^horizon is too many to enumerate, but with 8 codes a few thousand
    /// samples cover the space densely — and every code is in-distribution.
    public static func solve(modelPath: String, episodes: Int, seed: UInt64 = 11,
                             latent: Int = 192) throws {
        let model = LatentActionWM(latent: latent)
        let weights = try loadArrays(url: URL(fileURLWithPath: "\(modelPath)/lawm.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        let numCodes = model.vq.numCodes

        var successes = 0
        for ep in 0..<episodes {
            let env = try PushTEnv(numEnvs: 1, seed: seed &+ UInt64(ep) * 7)
            let res = env.obsRes
            let r = env.refs[0]
            // goal image (solved-state tip pose)
            let savedTip = (env.solver.bodyPosition(r.tip), env.solver.bodyRotation(r.tip))
            let tipGoalP = r.center + F3(r.goalPos.x - 0.62, r.goalPos.y - 0.62, PushTEnv.tipHeight)
            env.solver.setBodyPose(r.tip, position: tipGoalP, rotation: savedTip.1)
            let g = try PushTPipeline.obsArrayChecked(env, res)
            let zGoal = model.encoder(concatenated([g, g], axis: 3))
            eval(zGoal)
            env.solver.setBodyPose(r.tip, position: savedTip.0, rotation: savedTip.1)

            let horizon = 6, samples = 1024
            var prevFrame = try PushTPipeline.obsArrayChecked(env, res)
            try env.stepChecked(actions: [env.tipPos(0)], substeps: 4)
            for _ in 0..<60 {
                let cur = try PushTPipeline.obsArrayChecked(env, res)
                let z0 = model.encoder(concatenated([prevFrame, cur], axis: 3))
                // random shooting over code sequences
                let codeIdx = MLXRandom.randInt(low: 0, high: numCodes,
                                                [samples, horizon])
                var z = tiled(z0, repetitions: [samples, 1])
                var cost = MLXArray.zeros([samples])
                for h in 0..<horizon {
                    let vecs = model.vq.codebook[codeIdx[0..., h]]
                    z = model.predict(z, codeVec: vecs)
                    cost = cost + mean((z - zGoal).square(), axis: -1)
                        * (h == horizon - 1 ? 2.0 : 0.3)
                }
                eval(cost)
                let best = argMin(cost, axis: 0).item(Int.self)
                let bestCode = model.vq.codebook[codeIdx[best, 0..<1]]
                // translate the chosen code into a continuous action
                let a = model.translate(z0, codeVec: bestCode)
                eval(a)
                let act = SIMD2(a[0, 0].item(Float.self) * 3, a[0, 1].item(Float.self) * 3)
                for k in 0..<8 {
                    if k == 7 { prevFrame = try PushTPipeline.obsArrayChecked(env, res) }
                    try env.stepChecked(actions: [act])
                    if env.success(0) { break }
                }
                if env.success(0) { break }
            }
            let ok = env.success(0)
            successes += ok ? 1 : 0
            let (bp, _) = env.blockPose(0)
            print("lawm episode \(ep): \(ok ? "SUCCESS" : "fail") block (\(String(format: "%.2f", bp.x)), \(String(format: "%.2f", bp.y)))")
        }
        print("LAWM success rate: \(successes)/\(episodes)")
    }
}
