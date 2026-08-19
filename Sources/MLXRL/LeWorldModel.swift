import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

// LeWorldModel (LeCun et al., arXiv 2603.19312): a deliberately minimal
// JEPA world model trained end-to-end from pixels with exactly two losses:
//   1. next-embedding prediction MSE:  || pred(enc(o_t), a_t) - enc(o_{t+1}) ||^2
//   2. SIGReg (from LeJEPA): sketched isotropic Gaussian regularization —
//      random 1D projections of the latent batch are pushed toward N(0,1)
//      via the Epps–Pulley characteristic-function statistic.
// No EMA, no stop-gradient, no frozen encoders. One hyperparameter (lambda).

public final class LeWMEncoder: Module, UnaryLayer {
    @ModuleInfo var c1: Conv2d
    @ModuleInfo var c2: Conv2d
    @ModuleInfo var c3: Conv2d
    @ModuleInfo var c4: Conv2d
    @ModuleInfo var head: Linear

    public init(latent: Int = 128, inChannels: Int = 6) {
        c1 = Conv2d(inputChannels: inChannels, outputChannels: 32, kernelSize: 4, stride: 2, padding: 1)
        c2 = Conv2d(inputChannels: 32, outputChannels: 64, kernelSize: 4, stride: 2, padding: 1)
        c3 = Conv2d(inputChannels: 64, outputChannels: 128, kernelSize: 4, stride: 2, padding: 1)
        c4 = Conv2d(inputChannels: 128, outputChannels: 128, kernelSize: 4, stride: 2, padding: 1)
        head = Linear(128 * 4 * 4, latent)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, 64, 64, 3) in [0, 1]
        var h = gelu(c1(x))
        h = gelu(c2(h))
        h = gelu(c3(h))
        h = gelu(c4(h))
        h = h.reshaped([h.dim(0), -1])
        return head(h)
    }
}

public final class LeWMPredictor: Module {
    @ModuleInfo var l1: Linear
    @ModuleInfo var l2: Linear
    @ModuleInfo var l3: Linear

    public init(latent: Int = 128, actionDim: Int = 2, hidden: Int = 256) {
        l1 = Linear(latent + actionDim, hidden)
        l2 = Linear(hidden, hidden)
        l3 = Linear(hidden, latent)
    }

    public func callAsFunction(_ z: MLXArray, _ a: MLXArray) -> MLXArray {
        var h = concatenated([z, a], axis: -1)
        h = gelu(l1(h))
        h = gelu(l2(h))
        return z + l3(h)          // residual: predict the latent delta
    }
}

public final class LeWorldModel: Module {
    @ModuleInfo public var encoder: LeWMEncoder
    @ModuleInfo public var predictor: LeWMPredictor
    public let latent: Int

    /// stack: number of consecutive frames per observation (velocity
    /// observability — single frames make dynamics fundamentally ambiguous)
    public init(latent: Int = 128, actionDim: Int = 2, stack: Int = 2) {
        self.latent = latent
        encoder = LeWMEncoder(latent: latent, inChannels: 3 * stack)
        predictor = LeWMPredictor(latent: latent, actionDim: actionDim)
    }

    /// SIGReg: Epps–Pulley statistic of random 1D projections vs N(0,1).
    public static func sigReg(_ z: MLXArray, sketches: Int = 64) -> MLXArray {
        let d = z.dim(-1)
        var dirs = MLXRandom.normal([d, sketches])
        dirs = dirs / sqrt(sum(dirs * dirs, axis: 0, keepDims: true))
        let proj = matmul(z, dirs)                       // (B, S)
        // characteristic function grid
        let ts = MLXArray(stride(from: Float(0.25), through: 3.0, by: 0.25).map { $0 })
        let w = exp(-ts * ts / 2) * 0.25                 // N(0,1)-weighted dt
        let tx = proj.expandedDimensions(axis: 2) * ts   // (B, S, T)
        let reC = mean(cos(tx), axis: 0)                 // (S, T)
        let imC = mean(sin(tx), axis: 0)
        let target = exp(-ts * ts / 2)                   // (T)
        let ep = (reC - target).square() + imC.square()  // (S, T)
        return mean(sum(ep * w, axis: 1))
    }

    /// The two-term LeWM loss, with a K-step latent rollout: compounding
    /// prediction sharpens action-relevant features (block motion).
    /// obsSeq: (K+1) arrays of (B, H, W, 3); actSeq: K arrays of (B, A).
    public func loss(obsSeq: [MLXArray], actSeq: [MLXArray],
                     lambda: Float = 0.5) -> MLXArray {
        let zs = obsSeq.map { encoder($0) }
        var z = zs[0]
        var pred = MLXArray(Float(0))
        for k in 0..<actSeq.count {
            z = predictor(z, actSeq[k])
            pred = pred + mean((z - zs[k + 1]).square())
        }
        pred = pred / Float(actSeq.count)
        let reg = Self.sigReg(concatenated(zs, axis: 0))
        return pred + lambda * reg
    }
}
