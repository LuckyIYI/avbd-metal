/// Reconstruct half-resolution visibility on the surface being shaded, before
/// multisample coverage combines foreground and background fragments.
let surfaceVisibilityShaderSource = """
inline float2 surfaceVisibility(float2 uv, float3 P, float3 N, constant Uniforms& U,
    texture2d<float> visibility, depth2d<float> depth, texture2d<float> normal) {
    float2 size = float2(depth.get_width(), depth.get_height());
    // HQ guides, world rays and single-sample opaque shading share the same
    // jittered projection and pixel grid. uv comes from fragment position /
    // U.screen.xy, so this receiver's visibility is already at its pixel.
    if (U.reconstruction.x > 0 && all(U.screen.xy == size)) {
        uint2 pixel = min(uint2(uv * size), uint2(size) - 1u);
        return visibility.read(pixel).rg;
    }
    float2 location = uv * size - 0.5;
    float2 fraction = fract(location);
    int2 base = int2(floor(location));

    // Shading normals can lean away from the actual pan base or flat triangle.
    // Use its geometric plane for ownership, keeping shading-normal agreement
    // as a separate check at creases and silhouettes.
    float3 planeCross = cross(dfdx(P), dfdy(P));
    float crossLength = length(planeCross);
    float3 planeNormal = crossLength > 1e-12 ? planeCross / crossLength : N;
    float viewZ = max(dot(P - U.eye.xyz, cross(U.camRight.xyz, U.camUp.xyz)), 1e-4);
    float texelWorld = viewZ / max(U.screen.z, 1.0) * max(U.screen.x / size.x, U.screen.y / size.y);
    float planeEpsilon = max(0.00025, texelWorld * 0.15);

    float2 sum = float2(0);
    float totalWeight = 0;
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
        int2 q = base + int2(x, y);
        if (any(q < 0) || any(q >= int2(size))) continue;
        float d = depth.read(uint2(q));
        if (d >= 1.0) continue;
        float3 QN = normal.read(uint2(q)).xyz;
        float normalAgreement = saturate(dot(N, QN));
        if (normalAgreement <= 0.5) continue;
        float3 Q = worldFromDepth((float2(q) + 0.5) / size, d, U.invViewProj);
        float3 delta = Q - P;
        float planeDistance = abs(dot(delta, planeNormal));
        float tangentDistance = length(delta - planeNormal * dot(delta, planeNormal));
        // Allow the small departure of an adjacent curved triangle, without
        // accepting a parallel surface behind a thin rack wire.
        float tolerance = planeEpsilon + tangentDistance * 0.25 * sqrt(1.0 - normalAgreement);
        float weight = (x ? fraction.x : 1.0 - fraction.x) * (y ? fraction.y : 1.0 - fraction.y);
        weight *= saturate(1.0 - planeDistance / tolerance) * smoothstep(0.5, 0.9, normalAgreement);
        sum += visibility.read(uint2(q)).rg * weight;
        totalWeight += weight;
    }
    // A subpixel foreground object may have no representative in the guides.
    // It must not inherit an unrelated background object's AO or contact shadow.
    return totalWeight > 1e-5 ? sum / totalWeight : float2(1);
}
"""
