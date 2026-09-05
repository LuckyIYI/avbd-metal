/// Camera-space coordinates use screen right, screen down and camera forward.
/// Shared reconstruction keeps every depth lookup tied to its texel center.
let screenSpaceCommonShaderSource = """
struct FSOut { float4 position [[position]]; float2 uv; };

vertex FSOut fs_vertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2) * 2.0 - 1.0;
    FSOut o;
    o.position = float4(p, 1.0, 1.0);
    o.uv = p * float2(0.5, -0.5) + 0.5;
    return o;
}

inline float3 screenVector(float3 w, constant Uniforms& U) {
    return float3(dot(w, U.camRight.xyz), dot(w, U.camUp.xyz),
                  dot(w, cross(U.camRight.xyz, U.camUp.xyz)));
}
inline float screenDepth(float d, constant Uniforms& U) {
    return U.aoProjection.y / (d - U.aoProjection.x);
}
inline float3 screenPosition(float2 uv, float d, constant Uniforms& U) {
    return float3(((uv - U.reconstruction.yz) * 2.0 - 1.0) * U.aoProjection.zw, 1.0) * screenDepth(d, U);
}
inline float2 screenProject(float3 p, constant Uniforms& U) {
    return (p.xy / (p.z * U.aoProjection.zw)) * 0.5 + 0.5 + U.reconstruction.yz;
}
inline bool screenInside(float2 uv) { return all(uv >= 0.0) && all(uv < 1.0); }
inline float screenNoise(uint2 p) {
    uint h = p.x + p.y * 65537u;
    h ^= h >> 17; h *= 0xed5ad4bbu;
    h ^= h >> 11; h *= 0xac4c1b51u;
    h ^= h >> 15; h *= 0x31848babu;
    h ^= h >> 14;
    return float(h & 65535u) / 65536.0;
}
inline float3 screenEnvironment(float3 R) {
    return mix(HORIZON_LIN, float3(0.42, 0.48, 0.58), clamp(R.z, 0.0, 1.0));
}

// Tokuyoshi/Kaplanyan (I3D 2019), normal-based isotropic NDF filtering.
// Add pixel-footprint variance to alpha squared, not perceptual roughness.
inline float specularRoughness(float3 n, float rough, float pixelScale) {
    float3 dx = dfdx(n)*pixelScale, dy = dfdy(n)*pixelScale;
    float varianceRoughness = min(0.25*(dot(dx,dx)+dot(dy,dy)),0.18);
    float alpha = rough*rough;
    return sqrt(sqrt(saturate(alpha*alpha+varianceRoughness)));
}

inline float3 reflectionFactor(float3 P, float3 N, float rough, float3 albedo, float metal, constant Uniforms& U) {
    float NdV = saturate(dot(N,normalize(U.eye.xyz-P)));
    float3 F0 = mix(float3(0.04),albedo,metal);
    return max((F0+(1-F0)*pow(1-NdV,5.0))*mix(0.50,0.20,rough),float3(0.001));
}

inline float specularOcclusion(float NdV, float ao, float rough, constant Uniforms& U) {
    // World rays already evaluate visibility along the specular lobe. Diffuse
    // hemisphere AO must not darken that result a second time. Keep the same
    // unoccluded environment baseline when applying the signed ray correction.
    // Lagarde's approximation uses microfacet roughness (perceptual squared).
    float untraced = saturate(pow(NdV+ao,exp2(-16*rough*rough-1))-1+ao);
    float traced = U.rayTracing.x > 0 ? 1-smoothstep(U.effects.w-0.15,U.effects.w,rough) : 0;
    return mix(untraced,1.0,traced);
}

inline float3 surfaceReflection(float2 uv, float3 P, float3 N, float rough, float3 albedo, float metal,
    constant Uniforms& U, texture2d<float> reflection, texture2d<float> normal,
    depth2d<float> depth, texture2d<float> material) {
    if (rough >= U.effects.w) return float3(0);
    float2 size = float2(depth.get_width(),depth.get_height());
    float2 p = uv*size-0.5, f = fract(p);
    int2 base = int2(floor(p));
    float z = dot(P-U.eye.xyz,cross(U.camRight.xyz,U.camUp.xyz));
    float tolerance = max(0.0005,z/U.screen.z*0.75);
    float3 sum = float3(0); float weight = 0;
    for (int y=0; y<2; ++y) for (int x=0; x<2; ++x) {
        int2 q = base+int2(x,y);
        if (any(q<0) || any(q>=int2(size))) continue;
        float d = depth.read(uint2(q)); float4 nr = normal.read(uint2(q));
        if (d>=1 || abs(nr.w-rough)>0.08) continue;
        float3 Q = worldFromDepth((float2(q)+0.5)/size,d,U.invViewProj);
        float plane = max(abs(dot(Q-P,N)),abs(dot(Q-P,nr.xyz)));
        float w = (x ? f.x : 1-f.x)*(y ? f.y : 1-f.y);
        w *= saturate(1-plane/tolerance)*pow(saturate(dot(N,nr.xyz)),32.0);
        float4 mat = material.read(uint2(q));
        sum += reflection.read(uint2(q)).rgb/reflectionFactor(Q,nr.xyz,nr.w,mat.rgb,mat.a,U)*w;
        weight += w;
    }
    return weight > 1e-5 ? sum/weight*reflectionFactor(P,N,rough,albedo,metal,U)*saturate(weight*4) : float3(0);
}

""" + diffuseCommonShaderSource + """
inline float3 pbrRadiance(float3 albedo, float rough, float metal, float3 emissive,
                         float3 n, float3 V, float ao, float shadow, constant Uniforms& U) {
    float3 L = -U.lightDir.xyz;
    float3 H = normalize(L + V);
    float NdL = max(dot(n, L), 0.0);
    float NdV = max(dot(n, V), 1e-4);
    float NdH = max(dot(n, H), 0.0);
    float HdV = max(dot(H, V), 0.0);

    float a2 = rough * rough; a2 *= a2;
    float dDen = NdH * NdH * (a2 - 1.0) + 1.0;
    float D = a2 / (M_PI_F * dDen * dDen);
    float k = (rough + 1.0); k = k * k / 8.0;
    float G = (NdV / (NdV * (1.0 - k) + k)) * (NdL / (NdL * (1.0 - k) + k));
    float3 F0 = mix(float3(0.04), albedo, metal);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - HdV, 5.0);
    float3 spec = D * G * F / max(4.0 * NdV * NdL, 1e-4);

    float3 direct = (albedo / M_PI_F * (1.0 - metal) * (1.0 - F) + spec) * SUN_COL * NdL * shadow;

    float3 irr = mix(GND_IRR, SKY_IRR, n.z * 0.5 + 0.5);
    float3 ambient = albedo * irr * (1.0 - metal) * ao;
    float3 R = reflect(-V, n);
    float3 skyRef = screenEnvironment(R);
    ambient += skyRef * (F0 + (1.0 - F0) * pow(1.0 - NdV, 5.0))
        * mix(0.50, 0.20, rough) * specularOcclusion(NdV,ao,rough,U);

    return direct + ambient + emissive;
}

inline float3 clothRadiance(float3 albedo, float3 emissive, float3 n, float3 V,
                           float ao, float shadow, constant Uniforms& U) {
    float3 L = -U.lightDir.xyz;
    float3 H = normalize(L + V);
    float NdL = max(dot(n, L), 0.0);
    float NdV = max(dot(n, V), 1e-4);
    float NdH = max(dot(n, H), 0.0);
    float HdV = max(dot(H, V), 0.0);

    const float rough = 0.72;              // matte woven look
    float a2 = rough * rough; a2 *= a2;
    float dDen = NdH * NdH * (a2 - 1.0) + 1.0;
    float D = a2 / (M_PI_F * dDen * dDen);
    float k = (rough + 1.0); k = k * k / 8.0;
    float G = (NdV / (NdV * (1.0 - k) + k)) * (NdL / (NdL * (1.0 - k) + k));
    float3 F0 = float3(0.035);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - HdV, 5.0);
    float3 spec = D * G * F / max(4.0 * NdV * NdL, 1e-4);

    // a touch of wrap so the unlit side of folds doesn't go dead black
    float wrap = max((dot(n, L) + 0.35) / 1.35, 0.0);
    float3 direct = (albedo / M_PI_F * (1.0 - F) * wrap + spec * NdL) * SUN_COL * shadow;

    float3 irr = mix(GND_IRR, SKY_IRR, n.z * 0.5 + 0.5);
    float3 ambient = albedo * irr * 1.15;
    ambient *= ao;

    return direct + ambient + emissive;
}


"""

let screenSpaceShaderSource = """
struct SurfaceOut { float4 normal [[color(0)]]; float4 material [[color(1)]]; };
fragment SurfaceOut surface_fragment(VOut in [[stage_in]]) {
    SurfaceOut o;
    float3 n = normalize(in.normal);
    o.normal = float4(n, specularRoughness(n,clamp(in.pbr.x,0.02,1.0),0.5));
    o.material = float4(in.albedo, saturate(in.pbr.y));
    return o;
}
fragment SurfaceOut soft_surface_fragment(VOut in [[stage_in]], constant Uniforms& U [[buffer(1)]]) {
    float3 n = normalize(in.normal);
    if (in.flatShade > 0.5) n = normalize(cross(dfdx(in.world), dfdy(in.world)));
    if (dot(n, U.eye.xyz - in.world) < 0.0) n = -n;
    SurfaceOut o;
    o.normal = float4(n, 0.72);
    o.material = float4(0);
    return o;
}
fragment SurfaceOut floor_surface_fragment(FloorOut in [[stage_in]]) {
    SurfaceOut o;
    o.normal = float4(0, 0, 1, 1);
    o.material = float4(0);
    return o;
}

fragment float4 contact_fragment(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
    depth2d<float> depth [[texture(0)]], texture2d<float> normal [[texture(1)]]) {
    uint2 pixel = uint2(in.position.xy);
    float d = depth.read(pixel);
    if (d >= 1.0 || U.effects.y <= 0.0) return float4(1);
    float2 size = float2(depth.get_width(), depth.get_height());
    float3 P = screenPosition(in.uv, d, U);
    float3 N = screenVector(normal.read(pixel).xyz, U);
    float3 L = screenVector(-U.lightDir.xyz, U);
    if (dot(N, L) <= 0.0) return float4(1);
    float pixelWorld = P.z / U.screen.z;
    float bias = max(0.0005, pixelWorld * 0.2);
    float thickness = max(0.003, pixelWorld * 0.8);
    float visibility = 1.0;
    // Bounded short rays complement the shadow map; no screen-edge clamping.
    for (int i = 0; i < 12; ++i) {
        float t = (float(i) + 0.5 + 0.5 * screenNoise(pixel)) / 12.0;
        float3 Q = P + N * bias + L * (t * U.effects.y);
        if (Q.z <= screenDepth(0.0, U)) break;
        float2 uv = screenProject(Q, U);
        if (!screenInside(uv)) break;
        uint2 q = uint2(uv * size);
        float dq = depth.read(q);
        if (dq >= 1.0) continue;
        float3 S = screenPosition((float2(q) + 0.5) / size, dq, U);
        // The receiver itself cannot occlude its upper hemisphere. This
        // rejects quantization on flat surfaces before the depth comparison.
        if (dot(S - P, N) <= bias) continue;
        float gap = Q.z - S.z;
        if (gap > bias && gap < thickness) {
            float edge = min(min(uv.x, uv.y), min(1.0-uv.x, 1.0-uv.y));
            float confidence = saturate(edge * 40.0) * (1.0 - smoothstep(0.7, 1.0, t));
            visibility = min(visibility, 1.0 - confidence);
        }
    }
    return float4(visibility);
}

kernel void screen_depth_copy(depth2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x < output.get_width() && p.y < output.get_height()) output.write(float4(source.read(p)), p);
}
kernel void screen_depth_reduce(texture2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]], uint2 p [[thread_position_in_grid]]) {
    uint2 size = uint2(output.get_width(), output.get_height());
    if (any(p >= size)) return;
    uint2 inputSize = uint2(source.get_width(), source.get_height());
    // Include the extra row/column on odd-sized levels. Dropping it can hide
    // an edge occluder from all coarser levels.
    uint2 lo = p * inputSize / size;
    uint2 hi = min(((p + 1u) * inputSize + size - 1u) / size, inputSize);
    float nearest = 1.0;
    for (uint y = lo.y; y < hi.y; ++y) for (uint x = lo.x; x < hi.x; ++x)
        nearest = min(nearest, source.read(uint2(x,y)).r);
    output.write(float4(nearest), p);
}

// A perspective-correct segment: interpolate reciprocal Z along the projected
// line, then reconstruct on the exact screen ray. Linear Z interpolation would
// bend the ray and fabricate intersections on tilted surfaces.
inline float3 reflectionPoint(float2 uv, float inverseZ, constant Uniforms& U) {
    return float3(((uv - U.reconstruction.yz) * 2.0 - 1.0) * U.aoProjection.zw, 1) / inverseZ;
}
fragment float4 reflection_fragment(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
    depth2d<float> depth [[texture(0)]], texture2d<float> normal [[texture(1)]],
    texture2d<float> material [[texture(2)]], texture2d<float> scene [[texture(3)]],
    depth2d<float> sceneDepth [[texture(4)]], texture2d<float> ao [[texture(5)]],
    texture2d<float> hierarchy [[texture(6)]]) {
    uint2 pixel = uint2(in.position.xy);
    float d = depth.read(pixel);
    float4 nr = normal.read(pixel);
    float rough = nr.w;
    if (d >= 1.0 || rough >= U.effects.w || U.effects.z <= 0.0) return float4(0);
    float3 P = screenPosition(in.uv, d, U);
    float3 N = normalize(screenVector(nr.xyz, U));
    float3 V = normalize(-P);
    if (dot(N, V) <= 0.0) return float4(0);
    float3 R = reflect(-V, N);
    float pixelWorld = P.z / U.screen.z;
    float bias = max(0.001, pixelWorld * 0.3);
    float3 start = P + N * bias;
    float rayLength = U.effects.z;
    float nearZ = screenDepth(0.0, U) * 1.01;
    if (R.z < 0.0) rayLength = min(rayLength, (start.z - nearZ) / -R.z);
    if (rayLength <= bias) return float4(0);
    float3 end = start + R * rayLength;
    float2 uv0 = screenProject(start, U), uv1 = screenProject(end, U);
    float2 delta = uv1 - uv0;
    // Clip against the viewport before allocating the fixed march budget.
    float limit = 1.0;
    for (int axis = 0; axis < 2; ++axis) {
        if (delta[axis] > 0.0) limit = min(limit, (0.99999 - uv0[axis]) / delta[axis]);
        if (delta[axis] < 0.0) limit = min(limit, (0.00001 - uv0[axis]) / delta[axis]);
    }
    if (limit <= 0.0) return float4(0);
    float2 fullSize = float2(sceneDepth.get_width(), sceneDepth.get_height());
    float2 halfSize = float2(depth.get_width(), depth.get_height());
    float depth0 = U.aoProjection.x + U.aoProjection.y / start.z;
    float depthDelta = U.aoProjection.y / end.z + U.aoProjection.x - depth0;
    float t = 0.0;
    int mip = min(4, int(hierarchy.get_num_mip_levels()) - 1);
    float2 hitUV = float2(0);
    float3 hitP = float3(0);
    bool found = false;
    // Traverse coarse empty cells, descending only where a depth interval can
    // contain geometry. The fixed iteration cap bounds worst-case work.
    for (int iteration = 0; iteration < 64 && t < limit; ++iteration) {
        float2 uv = uv0 + delta * t;
        if (!screenInside(uv)) break;
        float2 cellSize = float2(hierarchy.get_width(uint(mip)), hierarchy.get_height(uint(mip)));
        uint2 cell = min(uint2(uv * cellSize), uint2(cellSize) - 1u);
        float2 boundary = (float2(cell) + select(float2(0), float2(1), delta > 0.0)) / cellSize;
        float2 crossing = select(float2(1e20), (boundary - uv0) / delta, abs(delta) > 1e-8);
        float nextT = min(limit, min(crossing.x, crossing.y));
        nextT = max(nextT, t + 1e-7);
        float nearestDepth = hierarchy.read(cell, uint(mip)).r;
        float rayD0 = depth0 + depthDelta * t;
        float rayD1 = depth0 + depthDelta * nextT;
        if (nearestDepth < 1.0 && max(rayD0, rayD1) >= nearestDepth) {
            if (mip > 0) { --mip; continue; }
            float z = screenDepth(nearestDepth, U);
            float z0 = screenDepth(rayD0, U), z1 = screenDepth(rayD1, U);
            float thickness = max(0.006, pixelWorld);
            if (min(z0, z1) <= z + thickness) {
                float hitT = abs(depthDelta) > 1e-8 ? clamp((nearestDepth-depth0)/depthDelta, t, nextT) : (t+nextT)*0.5;
                hitUV = clamp(uv0 + delta*hitT, float2(0.00001), float2(0.99999));
                hitP = reflectionPoint(hitUV, mix(1.0/start.z, 1.0/end.z, hitT), U);
                float3 hitN = screenVector(normal.read(min(uint2(hitUV * halfSize), uint2(halfSize)-1u)).xyz, U);
                if (length(hitP - P) > bias * 3.0 && dot(hitP - P, N) > bias * 2.0
                    && dot(hitN, R) < 0.0) { found = true; break; }
            }
        }
        t = nextT + 1e-6 / max(max(abs(delta.x), abs(delta.y)), 1e-6);
        mip = min(mip + 1, int(hierarchy.get_num_mip_levels()) - 1);
    }
    if (!found) return float4(0);
    float edge = min(min(hitUV.x, hitUV.y), min(1.0-hitUV.x, 1.0-hitUV.y));
    float travel = length(hitP - P);
    float confidence = saturate(edge * 20.0)
        * (1.0 - smoothstep(U.effects.z * 0.75, U.effects.z, travel))
        * (1.0 - smoothstep(U.effects.w - 0.15, U.effects.w, rough));
    // Cone footprint provides stable rough reflections without frame history.
    // The renderer's current-frame radiance never feeds back into itself.
    float footprint = rough * rough * travel * U.screen.z * (fullSize.y / halfSize.y) / max(hitP.z, 0.1) * 0.25;
    float lod = clamp(log2(max(footprint, 1.0)), 0.0, float(scene.get_num_mip_levels() - 1));
    constexpr sampler linearMip(filter::linear, mip_filter::linear, address::clamp_to_edge);
    float3 incoming = scene.sample(linearMip, hitUV, level(lod)).rgb;
    float4 mat = material.read(pixel);
    float3 F0 = mix(float3(0.04), mat.rgb, mat.a);
    float NdV = max(dot(N, V), 1e-4);
    float3 response = (F0 + (1.0-F0) * pow(1.0-NdV, 5.0)) * mix(0.50, 0.20, rough);
    float3 worldR = reflect(normalize((P.x * U.camRight.xyz + P.y * U.camUp.xyz
                            + P.z * cross(U.camRight.xyz, U.camUp.xyz))), normalize(nr.xyz));
    // Replace the covered portion of the existing environment term instead
    // of adding a second copy of specular illumination.
    float3 correction = (incoming - screenEnvironment(worldR)) * response
        * specularOcclusion(NdV,ao.read(pixel).r,rough,U);
    correction *= 1.0 - horizonFog(length(P));
    return float4(correction * confidence, confidence);
}

// One spatial resolve per signal. A 5x5 tent averages stochastic visibility
// without a preferred axis; plane/normal weights preserve contact boundaries.
inline float screenFilterWeight(float3 P, float3 N, float3 Q, float3 QN, float tolerance) {
    float agreement = max(dot(N, QN), 0.0);
    return pow(agreement, 16.0) * saturate(1.0 - abs(dot(Q-P, N)) / tolerance);
}
fragment float4 visibility_fragment(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
    texture2d<float> ambient [[texture(0)]], texture2d<float> contact [[texture(1)]],
    depth2d<float> depth [[texture(2)]], texture2d<float> normal [[texture(3)]]) {
    uint2 pixel = uint2(in.position.xy);
    float d = depth.read(pixel);
    if (d >= 1.0) return float4(1);
    float3 P = screenPosition(in.uv, d, U), N = screenVector(normal.read(pixel).xyz, U);
    float tolerance = max(0.003, P.z / U.screen.z * 1.5);
    float2 sum = float2(0); float weights = 0.0;
    int2 size = int2(depth.get_width(), depth.get_height());
    constexpr sampler nearest(filter::nearest);
    for (int y = -2; y <= 2; ++y) for (int x = -2; x <= 2; ++x) {
        int2 q = int2(pixel) + int2(x,y);
        if (any(q < 0) || any(q >= size)) continue;
        float dq = depth.read(uint2(q));
        if (dq >= 1.0) continue;
        float2 uv = (float2(q) + 0.5) / float2(size);
        float3 Q = screenPosition(uv, dq, U), QN = screenVector(normal.read(uint2(q)).xyz, U);
        float w = screenFilterWeight(P, N, Q, QN, tolerance) * float((3-abs(x))*(3-abs(y)));
        float a = ambient.sample(nearest, uv).r;
        float c = U.effects.y > 0.0 ? contact.read(uint2(q)).r : 1.0;
        sum += float2(a, c) * w; weights += w;
    }
    return float4(weights > 0.0 ? sum / weights : float2(1), 0, 1);
}
inline float3 reflectionModulation(float3 P, float4 nr, float4 material, constant Uniforms& U) {
    float NdV = saturate(dot(screenVector(nr.xyz, U), normalize(-P)));
    float3 F0 = mix(float3(0.04), material.rgb, material.a);
    return max((F0 + (1.0-F0)*pow(1.0-NdV, 5.0))*mix(0.50, 0.20, nr.w), float3(0.001));
}
fragment float4 reflection_filter_fragment(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
    constant float4& filter [[buffer(2)]], texture2d<float> raw [[texture(0)]],
    depth2d<float> depth [[texture(1)]], texture2d<float> normal [[texture(2)]],
    texture2d<float> material [[texture(3)]]) {
    uint2 pixel = uint2(in.position.xy);
    float d = depth.read(pixel); float4 nr = normal.read(pixel);
    if (d >= 1.0 || nr.w >= U.effects.w) return float4(0);
    float3 P = screenPosition(in.uv, d, U), N = screenVector(nr.xyz, U);
    float3 modulation = reflectionModulation(P, nr, material.read(pixel), U);
    int step = int(filter.x);
    if (nr.w < 0.06 || (step > 1 && nr.w < 0.16)) {
        float4 value = raw.read(pixel);
        if (filter.y > 0) value.rgb /= modulation;
        if (filter.z > 0) value.rgb *= modulation;
        return value;
    }
    float tolerance = max(0.001, P.z / U.screen.z * 0.75);
    float4 sum = float4(0); float weights = 0.0;
    int2 size = int2(depth.get_width(), depth.get_height());
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        int2 q = int2(pixel) + int2(x, y) * step;
        if (any(q < 0) || any(q >= size)) continue;
        float dq = depth.read(uint2(q)); float4 qn = normal.read(uint2(q));
        if (dq >= 1.0 || abs(qn.w - nr.w) > 0.05) continue;
        float3 Q = screenPosition((float2(q) + 0.5) / float2(size), dq, U);
        float3 QN = screenVector(qn.xyz, U);
        float planeDistance = max(abs(dot(Q-P, N)), abs(dot(Q-P, QN)));
        float w = pow(saturate(dot(N, QN)), mix(256.0, 32.0, nr.w));
        w *= saturate(1.0-planeDistance/tolerance) * (x == 0 ? 2.0 : 1.0) * (y == 0 ? 2.0 : 1.0);
        float4 value = raw.read(uint2(q));
        // Filter incident lighting separately from the receiving material's
        // Fresnel color, then restore that color once on the final pass.
        if (filter.y > 0) value.rgb /= reflectionModulation(Q, qn, material.read(uint2(q)), U);
        sum += value * w; weights += w;
    }
    float4 result = weights > 0.0 ? sum / weights : float4(0);
    if (filter.z > 0) result.rgb *= modulation;
    return result;
}

fragment float4 screen_composite_fragment(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
    texture2d<float> scene [[texture(0)]], texture2d<float> reflection [[texture(1)]],
    depth2d<float> depth [[texture(2)]], texture2d<float> normal [[texture(3)]],
    depth2d<float> fullDepth [[texture(4)]]) {
    uint2 pixel = uint2(in.position.xy);
    float3 color = scene.read(pixel).rgb;
    float d = fullDepth.read(pixel);
    uint2 np = min(uint2(in.uv * float2(normal.get_width(), normal.get_height())),
                   uint2(normal.get_width()-1, normal.get_height()-1));
    if (U.rayTracing.w == 0 && d < 1.0 && normal.read(np).w < U.effects.w) {
        float3 P = screenPosition(in.uv, d, U);
        float2 size = float2(depth.get_width(), depth.get_height());
        float2 p = in.uv * size - 0.5;
        int2 base = int2(floor(p));
        float2 f = fract(p);
        float tolerance = max(0.001, P.z * U.aoProjection.w * 2.0 / size.y);
        float3 sum = float3(0); float weight = 0.0;
        for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
            int2 q = base + int2(x, y);
            if (any(q < 0) || any(q >= int2(size))) continue;
            float dq = depth.read(uint2(q));
            if (dq >= 1.0) continue;
            float3 Q = screenPosition((float2(q) + 0.5) / size, dq, U);
            float3 N = screenVector(normal.read(uint2(q)).xyz, U);
            float w = (x ? f.x : 1.0-f.x) * (y ? f.y : 1.0-f.y);
            w *= saturate(1.0 - abs(dot(Q - P, N)) / tolerance);
            sum += reflection.read(uint2(q)).rgb * w;
            weight += w;
        }
        if (weight > 1e-5) color += sum / weight;
    }
    return float4(displayColorSRGB8(acesTonemap(max(color, 0.0)), in.position.xy), 1);
}
""" + antialiasingShaderSource + reconstructionShaderSource
