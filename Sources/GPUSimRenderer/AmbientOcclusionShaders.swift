/// GTAO estimator and its independently validated temporal visibility history.
let ambientOcclusionShaderSource = """
fragment float4 gtao_fragment(FSOut in [[stage_in]],
                              constant Uniforms& U [[buffer(1)]],
                              depth2d<float> depthTex [[texture(0)]],
                              texture2d<float> normTex [[texture(1)]])
{
    // GTAO (Jimenez et al. 2016 / XeGTAO-style): march several
    // screen-space slices, find both horizon angles per slice, clamp to the
    // normal hemisphere, then integrate the cosine-weighted occluded arc.
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
    float3 P = float3(in.position.xy * pixelToView - U.aoProjection.zw, 1.0) * centerZ;
    float3 V = normalize(-P);
    float viewDepth = max(centerZ, 0.25);

    const float R = 0.9;                                  // world AO radius
    float pxRadius = U.screen.z * R / viewDepth;
    // far away the radius collapses below sampling density — fade AO out
    // instead of letting a few-pixel march invent large-scale occlusion
    float farFade = saturate((pxRadius - 2.5) / 6.0);
    if (farFade <= 0.0) return float4(1);
    pxRadius = min(pxRadius, 96.0);
    // the falloff must use the radius we ACTUALLY march (post-clamp), or
    // near-camera AO reaches past its sampled range and over-darkens
    float Reff = pxRadius * viewDepth / U.screen.z;
    float falloffRange = max(Reff * 0.65, 1e-4);

    // Independent spatial angle/radius seeds; linear IGN bands and a radial
    // jitter derived from the angle seed remain correlated after denoising.
    float2 px = floor(in.position.xy);
    uint seed = uint(px.x) + uint(px.y) * 65537u;
    seed ^= seed >> 17; seed *= 0xed5ad4bbu;
    seed ^= seed >> 11; seed *= 0xac4c1b51u;
    seed ^= seed >> 15; seed *= 0x31848babu;
    seed ^= seed >> 14;
    float2 noise = float2(seed & 65535u, seed >> 16) / 65536.0;
    float ang = fract(noise.x + U.temporal.x) * M_PI_F;
    float stepJit = fract(noise.y + U.temporal.x * 5.0);

    const int SLICES = 3;
    const int STEPS = 4;
    float occlusion = 0.0;

    for (int sl = 0; sl < SLICES; sl++) {
        float phi = ang + float(sl) * (M_PI_F / float(SLICES));
        float2 dirPx = float2(cos(phi), sin(phi));

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

        // XeGTAO fades discarded/out-of-radius samples toward the normal
        // hemisphere edge instead of -1, avoiding grazing-angle detail loss.
        float sinN = sin(n);
        float lowH0 = -sinN;
        float lowH1 = sinN;

        // horizon cosines per side of the slice (classified by the
        // actual sample offset, so screen/UV orientation can't flip them)
        float cosH0 = lowH0, cosH1 = lowH1;
        float minS = min(0.95, 1.3 / max(pxRadius, 1e-3));
        for (int st = 1; st <= STEPS; st++) {
            // Shift each stratum independently instead of moving every
            // radius in every slice together. Share the offset across sides.
            float stepNoise = fract(stepJit + float(sl + (st - 1) * STEPS) * 0.6180339887);
            float u = (float(st) - 1.0 + stepNoise) / float(STEPS);
            float t = minS + (1.0 - minS) * (u * u);
            int2 offset = int2(round(dirPx * (t * pxRadius)));
            for (int side = 0; side < 2; side++) {
                // Depth belongs to a texel CENTER. Unprojecting a nearest
                // depth at the continuous march UV fabricates height steps
                // on tilted planes. Off-screen taps have no geometry data.
                int2 samplePixel = int2(px) + (side == 0 ? offset : -offset);
                if (any(samplePixel < 0) || any(samplePixel >= int2(U.screen.xy))) continue;
                float dQ = depthTex.read(uint2(samplePixel));
                if (dQ >= 1.0) continue;
                float sampleZ = U.aoProjection.y / (dQ - U.aoProjection.x);
                float3 w = float3((float2(samplePixel) + 0.5) * pixelToView
                                 - U.aoProjection.zw, 1.0) * sampleZ - P;
                float l = length(w);
                if (l < 1e-4) continue;
                // Rounding moves samples off the ideal slice. Coplanar or
                // below-tangent points cannot occlude the normal hemisphere,
                // even if their view cosine exceeds this slice's horizon.
                // Cover half-float normal error plus sub-mm depth roundoff;
                // this is a tangent-plane tolerance, not an AO radius bias.
                if (dot(N, w) <= 0.001 * l + 0.0001) continue;
                float c = dot(w / l, V);
                float weight = saturate((Reff - l) / falloffRange);
                if (dot(w, omega) >= 0.0) {
                    cosH0 = max(cosH0, mix(lowH0, c, weight));
                } else {
                    cosH1 = max(cosH1, mix(lowH1, c, weight));
                }
            }
        }

        // An unchanged horizon has exactly zero missing visibility. This
        // common case also avoids four inverse/trigonometric evaluations.
        if (cosH0 == lowH0 && cosH1 == lowH1) continue;

        // signed horizon angles in the slice plane (+ = toward omega),
        // clamped to the hemisphere around the projected normal
        float h1 =  acos(clamp(cosH0, -1.0, 1.0));
        float h2 = -acos(clamp(cosH1, -1.0, 1.0));
        h1 = n + min(h1 - n,  M_PI_F / 2.0);
        h2 = n + max(h2 - n, -M_PI_F / 2.0);

        // analytic cosine-weighted visible arc (GTAO inner integral)
        float a1 = 0.25 * (-cos(2.0 * h1 - n) + cosNV + 2.0 * h1 * sinN);
        float a2 = 0.25 * (-cos(2.0 * h2 - n) + cosNV + 2.0 * h2 * sinN);
        // Integrate missing visibility relative to the analytic open arc.
        // The full unoccluded hemisphere is exactly 1; estimating it with
        // only three slices and then saturating creates a normal-dependent
        // dark bias and structured noise even without any occluders.
        float unoccluded = cosNV + n * sinN;
        occlusion += projLen * max(0.0, unoccluded - a1 - a2);
    }
    float ao = saturate(1.0 - occlusion / float(SLICES));
    ao = pow(ao, 1.25);    // slight contrast shaping
    ao = max(ao, 0.03);    // visible pixels should not reach total black
    ao = mix(1.0, ao, farFade);
    return float4(ao, ao, ao, 1);
}

// Store the surface normal beside AO in the temporal target.  Octahedral
// packing keeps the history buffer compact (RGBA8) while giving the resolve
// enough orientation information to reject history from another face/object.
inline float2 encodeOctahedral(float3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    float2 signN = select(float2(-1.0), float2(1.0), n.xy >= 0.0);
    float2 e = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * signN;
    return e * 0.5 + 0.5;
}

inline float3 decodeOctahedral(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += select(float2(t), float2(-t), n.xy >= 0.0);
    return normalize(n);
}

inline float4 aoHistoryValue(float ao, float3 normal) {
    return float4(ao, encodeOctahedral(normal), 1.0);
}

fragment float4 temporal_fragment(FSOut in [[stage_in]],
                                  constant Uniforms& U [[buffer(1)]],
                                  texture2d<float> rawAO [[texture(0)]],
                                  texture2d<float> histAO [[texture(1)]],
                                  depth2d<float> depthTex [[texture(2)]],
                                  depth2d<float> prevDepthTex [[texture(3)]],
                                  texture2d<float> normTex [[texture(4)]])
{
    // temporal accumulation with reprojection: find this pixel's world
    // point in the previous frame, then reuse accumulated AO only while the
    // position and surface orientation still agree.
    constexpr sampler smp(filter::nearest, address::clamp_to_edge);
    constexpr sampler lsmp(filter::linear, address::clamp_to_edge);
    float cur = rawAO.sample(smp, in.uv).r;
    float dC = depthTex.sample(smp, in.uv);
    if (dC >= 1.0) return float4(cur, 0.5, 0.5, 0.0);
    float3 P = worldFromDepth(in.uv, dC, U.invViewProj);
    float3 N = normalize(normTex.sample(smp, in.uv).xyz);
    float4 current = aoHistoryValue(cur, N);
    if (U.temporal.y >= 1.0) return current;

    float4 clipPrev = U.prevViewProj * float4(P, 1.0);
    if (clipPrev.w <= 0.0) return current;
    float2 ndc = clipPrev.xy / clipPrev.w;
    float2 uvPrev = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (any(uvPrev < 0.0) || any(uvPrev >= 1.0)) return current;

    uint2 historyPixel = uint2(uvPrev * U.screen.xy);
    float2 historyUV = (float2(historyPixel) + 0.5) / U.screen.xy;
    float dPrev = prevDepthTex.read(historyPixel);
    if (dPrev >= 1.0) return current;
    float3 Qw = worldFromDepth(historyUV, dPrev, U.prevInvViewProj);

    // AO is continuous and may be bilinearly reprojected.  The octahedral
    // normal is encoded metadata: filtering its packed coordinates across a
    // hard edge or the octahedral fold can decode to an unrelated direction.
    float historyAO = histAO.sample(lsmp, uvPrev).r;
    float2 historyNormal = histAO.read(historyPixel).gb;
    float3 historyN = decodeOctahedral(historyNormal);

    // A nearest reprojection naturally differs by a fraction of one world
    // pixel even for static geometry.  Derivative-scaled thresholds preserve
    // that case but reject real motion much sooner than the old fixed 7 cm
    // window, particularly for close-up objects.
    float pixelWorld = clamp(min(length(dfdx(P)), length(dfdy(P))), 0.002, 0.03);
    float motionStart = 0.002 + 0.35 * pixelWorld;
    float motionEnd = 0.012 + 1.25 * pixelWorld;
    float positionReuse = 1.0 - smoothstep(
        motionStart, motionEnd, length(Qw - P));
    float normalReuse = smoothstep(0.58, 0.92, dot(N, historyN));
    float reuse = positionReuse * normalReuse;
    if (reuse <= 0.0) return current;

    // Clip history to a cheap five-sample current-frame neighborhood.  This
    // removes dark/light trails immediately when AO changes, while stable
    // stochastic variation still benefits from temporal convergence.
    float2 pixel = 1.0 / U.screen.xy;
    constexpr int2 offsets[4] = {
        int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)
    };
    float localMin = cur, localMax = cur;
    for (int i = 0; i < 4; ++i) {
        float value = rawAO.sample(smp, in.uv + float2(offsets[i]) * pixel).r;
        localMin = min(localMin, value);
        localMax = max(localMax, value);
    }
    float padding = max(0.015, (localMax - localMin) * 0.15);
    float hist = clamp(historyAO, localMin - padding, localMax + padding);
    float blend = mix(1.0, U.temporal.y, reuse);
    return aoHistoryValue(mix(hist, cur, blend), N);
}

"""
