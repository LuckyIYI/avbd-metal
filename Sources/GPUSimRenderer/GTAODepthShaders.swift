/// Paired linear depths keep the finite occluder interval in one texture read.
let gtaoDepthShaderSource = """
fragment float4 gtao_depth_fragment(FSOut in [[stage_in]],
    constant Uniforms& U [[buffer(1)]],
    depth2d<float> frontDepth [[texture(0)]], depth2d<float> backDepth [[texture(1)]]) {
    uint2 pixel = uint2(in.position.xy);
    float front = frontDepth.read(pixel), back = backDepth.read(pixel);
    float frontZ = front < 1.0 ? U.aoProjection.y / (front - U.aoProjection.x) : 1e20;
    float backZ = back < 1.0 ? U.aoProjection.y / (back - U.aoProjection.x) : 1e20;
    return float4(frontZ, backZ, 0, 0);
}
"""
