/// Deterministic RNG shared by reproducible scenes, robot randomization, and
/// reinforcement-learning tasks.
public struct SplitMix64 {
    var state: UInt64

    public init(seed: UInt64) {
        state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    public mutating func nextFloat() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}
