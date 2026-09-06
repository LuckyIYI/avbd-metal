/// Periodic spatial quadrature for Fast GTAO, reconstructed by a matching box.
let gtaoSamplingShaderSource = """
// A 4x4 Bayer ordering distributes the cross angles throughout each quad.
// The radial dimension is the bit-reversed (Hammersley) ordering. Every 4x4
// window has all 16 angle/radius pairs, regardless of its pixel alignment.
// No frame index: Fast has no temporal AA to integrate changing samples.
inline float2 gtaoSampleNoise(uint2 pixel) {
    uint x = pixel.x & 3u, y = pixel.y & 3u;
    uint index = ((x ^ y) & 1u) * 8u + (y & 1u) * 4u
               + ((x ^ y) & 2u) + ((y & 2u) >> 1u);
    // Scramble the spatial ordering, preserving the same 16 sample pairs.
    // Without this each 2x2 quad uses only one radial quartile, producing
    // rectangular hit/miss clusters next to thin occluders.
    index ^= index >> 2u;
    uint reversed = ((index & 1u) << 3u) | ((index & 2u) << 1u)
                  | ((index & 4u) >> 1u) | ((index & 8u) >> 3u);
    return (float2(index, reversed) + 0.5) / 16.0;
}
"""
