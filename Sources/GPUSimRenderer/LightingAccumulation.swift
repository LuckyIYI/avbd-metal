import simd

/// Finite accumulation is safe only while every input to lighting is unchanged.
/// Unknown or advancing world revisions use deterministic fresh samples instead.
struct LightingAccumulation {
    private struct Key: Equatable {
        var world: ObjectIdentifier
        var revision: UInt64
        var camera: [SIMD4<Float>]
        var light: SIMD3<Float>
        var size: SIMD2<Int>
        var options: GPUSimRenderOptions
    }
    private var previous: Key?
    private var frame = 0

    mutating func advance(world: ObjectIdentifier?, revision: UInt64?, camera: simd_float4x4,
                          light: SIMD3<Float>, size: SIMD2<Int>, options: GPUSimRenderOptions,
                          reset: Bool) -> SIMD4<Float> {
        guard let world, let revision else {
            previous = nil; frame = 0
            return SIMD4(0,1,0,0)
        }
        let key = Key(world: world, revision: revision,
                      camera: [camera.columns.0,camera.columns.1,camera.columns.2,camera.columns.3],
                      light: light, size: size, options: options)
        if reset || previous != key { frame = 0 } else { frame = min(frame+1,65) }
        previous = key
        // Frame zero is the responsive motion path. The next 64 unchanged
        // frames integrate independent samples, then stop modifying history.
        let phase = Float(max(0,min(frame,64)-1))*0.6180339887
        return SIMD4(phase.truncatingRemainder(dividingBy: 1),frame<=1 ? 1 : 1/Float(frame),Float(frame),0)
    }
}
