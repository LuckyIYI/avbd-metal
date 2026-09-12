/// Finite-thickness screen-space AO with cosine-weighted visibility bitmasks.
/// GTAO slice integration: Jimenez et al. 2016.
/// Finite intervals: Therrien et al. 2023, https://arxiv.org/abs/2301.11376.
let ambientOcclusionShaderSource = gtaoSamplingShaderSource + """
inline float3 gtaoGeometricNormal(uint2 pixel, float centerDepth, float3 P, float3 N,
    constant Uniforms& U, depth2d<float> depthTex) {
    float worldScale = U.screen.w > 0 ? U.screen.w : 1.0;
    float centerZ = P.z;
    float2 pixelToView = 2.0 * U.aoProjection.zw / U.screen.xy;
    // A closer depth can belong to the other face of a concave crease.
    // Select the side whose two depths extrapolate back to this receiver.
    // Hardware depth is affine over a projected plane; linear view Z is not.
    // Smoothed normals remain the shading basis.
    float3 neighbors[4];
    constexpr int2 axisOffsets[4] = { int2(-1,0), int2(1,0), int2(0,-1), int2(0,1) };
    float distances[4], adjacentErrors[4];
    bool hasSecond[4];
    for (int i = 0; i < 4; ++i) {
        int2 q = int2((float2(pixel) + 0.5)) + axisOffsets[i];
        bool valid = all(q >= 0) && all(q < int2(U.screen.xy));
        float dq = valid ? depthTex.read(uint2(q)) : 1.0;
        valid = valid && dq < 1.0;
        float zq = valid ? U.aoProjection.y / (dq - U.aoProjection.x) : centerZ;
        neighbors[i] = float3((float2(q) + 0.5 - U.reconstruction.yz * U.screen.xy)
            * pixelToView - U.aoProjection.zw, 1.0) * zq;
        int2 q2 = int2(pixel) + axisOffsets[i] * 2;
        bool valid2 = all(q2 >= 0) && all(q2 < int2(U.screen.xy));
        float d2 = valid2 ? depthTex.read(uint2(q2)) : 1.0;
        valid2 = valid2 && d2 < 1.0;
        adjacentErrors[i] = valid ? abs(dq - centerDepth) : INFINITY;
        hasSecond[i] = valid && valid2;
        distances[i] = hasSecond[i] ? abs((2.0 * dq - d2) - centerDepth) : INFINITY;
    }
    // Compare like metrics within each axis. An extrapolation residual on
    // an unrelated plane can be zero; it must not outrank an adjacent-depth
    // difference merely because the other second tap is off-screen or empty.
    for (int axis = 0; axis < 2; ++axis) {
        int a = axis * 2, b = a + 1;
        if (!(hasSecond[a] && hasSecond[b])) {
            distances[a] = adjacentErrors[a];
            distances[b] = adjacentErrors[b];
        }
    }
    float3 dx = distances[0] < distances[1] ? P-neighbors[0] : neighbors[1]-P;
    float3 dy = distances[2] < distances[3] ? P-neighbors[2] : neighbors[3]-P;
    float3 geometricN = cross(dx, dy);
    float geometricLength = length(geometricN);
    bool hasPlane = isfinite(min(distances[0], distances[1]))
                 && isfinite(min(distances[2], distances[3])) && geometricLength > 1e-10 * worldScale * worldScale;
    geometricN = hasPlane ? geometricN/geometricLength : N;
    return geometricN * (dot(geometricN, -P) < 0.0 ? -1.0 : 1.0);
}

// Integral of cos(theta-n) * abs(sin(theta)). Mapping angle to this
// cumulative energy makes every visibility bit carry the same cosine
// weight, including oblique receivers.
inline float gtaoArcEnergy(float theta, float n, float cosN, float sinN) {
    return sign(theta) * 0.25 * (-cos(2.0 * theta - n) + cosN + 2.0 * theta * sinN);
}

inline uint gtaoIntervalMask(float lower, float upper, float phase) {
    uint first = uint(clamp(ceil(lower * 32.0 - phase), 0.0, 32.0));
    uint end = uint(clamp(ceil(upper * 32.0 - phase), 0.0, 32.0));
    if (end <= first) return 0u;
    uint low = first == 0u ? 0u : (0xFFFFFFFFu >> (32u - first));
    uint high = end == 32u ? 0xFFFFFFFFu : (0xFFFFFFFFu >> (32u - end));
    return high & ~low;
}

fragment float4 gtao_fragment(FSOut in [[stage_in]],
                              constant Uniforms& U [[buffer(1)]],
                              depth2d<float> depthTex [[texture(0)]],
                              texture2d<float> normTex [[texture(1)]],
                              texture2d<float> linearDepth [[texture(2)]])
{
    // Front/back depth bounds finite occluders in each view-aligned slice.
    // A wire blocks only its occupied interval; light can pass behind it.
    constexpr sampler smp(filter::nearest, address::clamp_to_edge);
    float dC = depthTex.sample(smp, in.uv);
    if (dC >= 1.0) return float4(1);
    float3 camForward = normalize(cross(U.camRight.xyz, U.camUp.xyz));
    float3 worldN = normalize(normTex.sample(smp, in.uv).xyz);
    // Work in the orthonormal (screen right, screen down, forward) basis.
    // Linearizing depth and scaling a ray replaces a matrix multiply per
    // tap, and subtraction stays precise when the camera is far from origin.
    float3 N = float3(dot(worldN, U.camRight.xyz), dot(worldN, U.camUp.xyz),
                      dot(worldN, camForward));
    float2 pixelToView = 2.0 * U.aoProjection.zw / U.screen.xy;
    float2 sliceScale = pixelToView / pixelToView.y;
    float centerZ = U.aoProjection.y / (dC - U.aoProjection.x);
    float3 P = float3((in.position.xy - U.reconstruction.yz * U.screen.xy) * pixelToView - U.aoProjection.zw, 1.0) * centerZ;
    float3 V = normalize(-P);
    float worldScale = U.screen.w > 0 ? U.screen.w : 1.0;
    float viewDepth = max(centerZ, 0.25 * worldScale);

    const float R = 0.9 * worldScale;                                  // world AO radius
    float pxRadius = U.screen.z * R / viewDepth;
    // far away the radius collapses below sampling density — fade AO out
    // instead of letting a few-pixel march invent large-scale occlusion
    float farFade = saturate((pxRadius - 2.5) / 6.0);
    if (farFade <= 0.0) return float4(1);
    float3 geometricN = gtaoGeometricNormal(uint2(in.position.xy), dC, P, N, U, depthTex);
    pxRadius = min(pxRadius, 96.0);
    // the falloff must use the radius we ACTUALLY march (post-clamp), or
    // near-camera AO reaches past its sampled range and over-darkens
    float Reff = pxRadius * viewDepth / U.screen.z;
    float falloffRange = max(Reff * 0.65, 1e-4 * worldScale);

    float2 px = floor(in.position.xy);
    float2 noise = gtaoSampleNoise(uint2(px));
    float stepJit = noise.y;

    // A spatial cross and matched 4x4 reconstruction distribute angles
    // across nearby receivers. Radial coverage preserves thin contacts.
    // Front and back remain paired at the same original depth texel.
    const int SLICES = 2;
    const int STEPS = 12;
    float phi = noise.x * (M_PI_F / 2.0);
    float2 baseDirection = float2(cos(phi), sin(phi));
    float occlusion = 0.0;

    for (int sl = 0; sl < SLICES; sl++) {
        float2 dirPx = sl == 0 ? baseDirection : float2(-baseDirection.y, baseDirection.x);

        // ANALYTIC slice tangent: the view direction this screen-space
        // march corresponds to, projected perpendicular to V. Deriving it
        // from the camera basis (not from samples) keeps the slice plane
        // exact and view-consistent.
        float3 dirV = float3(dirPx * sliceScale, 0.0);
        float3 omega = dirV - V * dot(dirV, V);
        float ol = length(omega);
        if (ol < 1e-4) continue;
        omega /= ol;

        // project N into the slice plane (spanned by V and omega)
        float3 sliceN = cross(V, omega);
        float3 projN = N - sliceN * dot(N, sliceN);
        float projLen = length(projN);
        if (projLen < 1e-4) continue;
        float3 pn = projN / projLen;
        float cosNV = saturate(dot(pn, V));
        float n = acos(cosNV) * (dot(pn, omega) >= 0.0 ? 1.0 : -1.0);

        float sinN = sin(n);
        float arcStart = n - M_PI_F / 2.0;
        float arcEnd = n + M_PI_F / 2.0;
        float energyStart = gtaoArcEnergy(arcStart, n, cosNV, sinN);
        float openEnergy = cosNV + n * sinN;
        float geometricAngle = atan2(dot(geometricN, omega), dot(geometricN, V));
        float lowerAngle = max(arcStart, geometricAngle - M_PI_F / 2.0);
        float upperAngle = min(arcEnd, geometricAngle + M_PI_F / 2.0);
        if (upperAngle <= lowerAngle) continue;

        // Four nested occupancy masks represent distance attenuation.
        uint4 occupied = uint4(0u);
        float minS = min(0.95, 1.3 / max(pxRadius, 1e-3));
        for (int st = 1; st <= STEPS; st++) {
            float u = (float(st) - 1.0 + stepJit) / float(STEPS);
            float t = minS + (1.0 - minS) * (u * u);
            float2 offset = dirPx * (t * pxRadius);
            float2 depthSize = float2(linearDepth.get_width(), linearDepth.get_height());
            for (int side = 0; side < 2; side++) {
                float2 sampleUV = (in.position.xy + (side == 0 ? offset : -offset)) / U.screen.xy;
                if (any(sampleUV < 0.0) || any(sampleUV >= 1.0)) continue;
                uint2 samplePixel = uint2(sampleUV * depthSize);
                float2 sampleDepth = linearDepth.read(samplePixel).rg;
                float sampleZ = sampleDepth.x;
                if (sampleZ > 1e9) continue;
                // Reconstruct at the sampled depth texel's center. Using the
                // continuous search UV with a nearest depth fabricates steps
                // on tilted surfaces.
                float2 sampleCenter = (float2(samplePixel) + 0.5) / depthSize;
                float3 w = float3((sampleCenter - U.reconstruction.yz) * 2.0 * U.aoProjection.zw
                                 - U.aoProjection.zw, 1.0) * sampleZ - P;
                float l = length(w);
                if (l < 1e-4 * worldScale) continue;
                // Rounding moves samples off the ideal slice. Coplanar or
                // below-tangent points cannot occlude the normal hemisphere.
                // Cover half-float normal error plus sub-mm depth roundoff;
                // this is a tangent-plane tolerance, not an AO radius bias.
                if (dot(N, w) <= 0.001 * l + 0.0001 * worldScale || dot(geometricN, w) <= 0.001 * l + 0.0001 * worldScale) continue;
                float weight = saturate((Reff - l) / falloffRange);
                if (weight <= 0.0) continue;
                float exitZ = sampleDepth.y < 1e9 ? sampleDepth.y : sampleZ + Reff;
                // An open/double-sided sheet has no paired back face. Keep
                // a finite footprint instead of extruding a wire to infinity.
                if (exitZ <= sampleZ + 1e-5 * worldScale) exitZ = sampleZ + max(0.001 * worldScale, sampleZ / U.screen.z);
                float3 wBack = (w + P) * (exitZ / sampleZ) - P;
                float frontAngle = atan2(dot(w, omega), dot(w, V));
                float backAngle = atan2(dot(wBack, omega), dot(wBack, V));
                float lo = clamp(min(frontAngle, backAngle), lowerAngle, upperAngle);
                float hi = clamp(max(frontAngle, backAngle), lowerAngle, upperAngle);
                float loEnergy = (gtaoArcEnergy(lo, n, cosNV, sinN) - energyStart) / openEnergy;
                float hiEnergy = (gtaoArcEnergy(hi, n, cosNV, sinN) - energyStart) / openEnergy;
                uint mask = gtaoIntervalMask(loEnergy, hiEnergy, noise.y);
                // Nested bitplanes retain the angular support when fading
                // distance, and union overlapping samples by maximum weight.
                uint levels = uint(weight * 4.0 + noise.x);
                occupied |= select(uint4(0u), uint4(mask), uint4(levels) >= uint4(1u, 2u, 3u, 4u));
            }
        }
        uint4 counts = popcount(occupied);
        occlusion += projLen * openEnergy * (float(counts.x + counts.y + counts.z + counts.w) / 128.0);
    }
    float ao = saturate(1.0 - occlusion / float(SLICES));
    ao = pow(ao, 1.25);    // slight contrast shaping
    ao = max(ao, 0.03);    // visible pixels should not reach total black
    ao = mix(1.0, ao, farFade);
    return float4(ao, ao, ao, 1);
}

"""
