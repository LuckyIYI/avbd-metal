/// Spatially stratified two-dimensional samples for GTAO's slice and radius.
let gtaoSamplingShaderSource = """
// Hilbert ordering keeps nearby pixels close in a low-discrepancy sequence.
// The R2 sequence and temporal stride follow XeGTAO's sampling design:
// https://github.com/GameTechDev/XeGTAO
inline float2 gtaoSampleNoise(uint2 pixel, uint frame) {
    uint2 p = pixel & uint2(63);
    uint index = 0;
    for (uint level = 32; level > 0; level >>= 1) {
        uint rx = (p.x & level) != 0;
        uint ry = (p.y & level) != 0;
        index += level * level * ((3 * rx) ^ ry);
        if (ry == 0) {
            if (rx != 0) p = uint2(63) - p;
            p = p.yx;
        }
    }
    index += 288 * (frame & 63);
    return fract(0.5 + float(index) * float2(0.7548776662466928, 0.5698402909980533));
}
"""
