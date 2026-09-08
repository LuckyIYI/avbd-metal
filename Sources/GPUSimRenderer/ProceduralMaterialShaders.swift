/// Continuous object-space textures: no UV seams, texture uploads, or added draws.
let proceduralMaterialShaderSource = """
inline float materialHash(float3 p) {
    p = fract(p * 0.1031); p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}
inline float materialNoise(float3 p) {
    float3 i = floor(p), f = fract(p); f = f*f*(3-2*f);
    return mix(mix(mix(materialHash(i),materialHash(i+float3(1,0,0)),f.x),
                   mix(materialHash(i+float3(0,1,0)),materialHash(i+float3(1,1,0)),f.x),f.y),
               mix(mix(materialHash(i+float3(0,0,1)),materialHash(i+float3(1,0,1)),f.x),
                   mix(materialHash(i+float3(0,1,1)),materialHash(i+float3(1,1,1)),f.x),f.y),f.z);
}
inline float filteredMaterialNoise(float3 p, float footprint) {
    if (footprint >= 1.2) return 0.5;
    return mix(materialNoise(p),0.5,smoothstep(0.35,1.2,footprint));
}
inline float materialPattern(float3 p, float4 detail, float footprint) {
    if (detail.x < 0.5) return 0.5;
    float scale = clamp(detail.y,0.1,10.0); p *= scale; footprint *= scale;
    if (detail.x < 1.5) {
        // Honed limestone: broad mineral clouds, fine aggregate, sparse pores.
        float cloud = materialNoise(p*7);
        float grain = filteredMaterialNoise(p*180,footprint*180);
        float pore = smoothstep(0.73,0.87,filteredMaterialNoise(p*420,footprint*420));
        return clamp(0.5+(cloud-0.5)*0.7+(grain-0.5)*0.9-pore*0.7,0.0,1.0);
    }
    if (detail.x < 2.5 || detail.x > 3.5) {
        // Grain axes: 2 local X, 4 local Y, 5 local Z. Irregular fibres avoid
        // repeating sine-wave stripes on long joinery pieces.
        if (detail.x > 4.5) p = p.zyx;
        else if (detail.x > 3.5) p = p.yxz;
        float warp = materialNoise(p*float3(1.4,5,5));
        float3 q = p + float3(0,warp*0.009,warp*0.006);
        float growth = filteredMaterialNoise(q*float3(1.8,48,48),footprint*48);
        float fibres = filteredMaterialNoise(q*float3(4,380,120),footprint*380);
        return clamp(growth*0.65+fibres*0.35,0.0,1.0);
    }
    return 0.5+(filteredMaterialNoise(p*95,footprint*95)-0.5)*0.6;
}
inline float4 applyMaterialPattern(float pattern, float4 detail, float3 albedo, float roughness) {
    if (detail.x < 0.5) return float4(albedo,roughness);
    float variation = (pattern-0.5)*clamp(detail.z,0.0,1.0);
    albedo *= max(0.05,1.0+variation*2);
    roughness = clamp(roughness+variation*0.35,0.04,1.0);
    return float4(albedo,roughness);
}
inline VOut texturedSurface(VOut input) {
    if (input.surfaceDetail.x < 0.5) return input;
    float footprint = max(length(dfdx(input.materialPosition)),length(dfdy(input.materialPosition)));
    float pattern = materialPattern(input.materialPosition,input.surfaceDetail,footprint);
    float4 material = applyMaterialPattern(pattern,input.surfaceDetail,input.albedo,input.pbr.x);
    input.albedo = material.rgb; input.pbr.x = material.w;
    float3 n = normalize(input.normal), dx = dfdx(input.world), dy = dfdy(input.world);
    float3 r1 = cross(dy,n), r2 = cross(n,dx); float determinant = dot(dx,r1);
    if (abs(determinant)>1e-12) {
        float3 gradient = (dfdx(pattern)*r1+dfdy(pattern)*r2)/determinant;
        input.normal = normalize(n-clamp(input.surfaceDetail.w,0.0,0.002)*gradient);
    }
    return input;
}
"""
