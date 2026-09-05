let rayTracingShaderSource = """
#include <metal_raytracing>
using namespace raytracing;
struct RTVertex { float4 position; float4 normal; float4 albedo; float4 emissive; };
struct RTObject { uint vertexStart; uint source; uint index; uint asset; };
struct RTInstance {
    packed_float3 transform[4];
    uint options; uint mask; uint intersectionOffset; uint accelerationStructureIndex;
};
inline float3 rtVector(device const RTInstance& instance, float3 v) {
    return float3(instance.transform[0])*v.x + float3(instance.transform[1])*v.y + float3(instance.transform[2])*v.z;
}
inline float3 rtNormal(device const RTInstance& instance, float3 n) {
    // Match the raster primitive transform; mesh instances are rigid rotations.
    return normalize(float3(instance.transform[0]))*n.x + normalize(float3(instance.transform[1]))*n.y
        + normalize(float3(instance.transform[2]))*n.z;
}
kernel void rt_instances(device const RTObject* objects [[buffer(0)]], device RTInstance* output [[buffer(1)]],
    device const RenderInstance* rigid [[buffer(2)]], device const RenderInstance* auxiliary [[buffer(3)]],
    device const float4* positions [[buffer(4)]], device const float4* rotations [[buffer(5)]],
    constant uint& count [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    RTObject object = objects[i];
    float4x4 m = float4x4(1);
    uint mask = object.source == 5 ? 0 : 3;
    if (object.source <= 1) {
        RenderInstance instance = object.source == 0 ? rigid[object.index] : auxiliary[object.index];
        m = instance.model;
        if (object.source == 1 && instance.params.w <= 0.5) mask = 2;
        if (object.source == 1 && instance.material.w < 1.0) mask = 0;
    } else if (object.source == 2) {
        float4 q = rotations[object.index];
        m[0] = float4(rigidMeshRotate(q,float3(1,0,0)),0);
        m[1] = float4(rigidMeshRotate(q,float3(0,1,0)),0);
        m[2] = float4(rigidMeshRotate(q,float3(0,0,1)),0);
        m[3] = float4(positions[object.index].xyz,1);
    }
    // Degenerate scale cannot be inverted by traversal; mask it and use an
    // identity transform until the host gives the primitive a valid size.
    if (abs(determinant(float3x3(m[0].xyz,m[1].xyz,m[2].xyz))) < 1e-15) { mask = 0; m = float4x4(1); }
    for (uint c = 0; c < 4; ++c) output[i].transform[c] = packed_float3(m[c].xyz);
    output[i].options = 0;
    output[i].mask = mask;
    output[i].intersectionOffset = 0;
    output[i].accelerationStructureIndex = object.asset;
}
kernel void rt_deform(device RTVertex* output [[buffer(0)]], device const uint* corners [[buffer(1)]],
    device const float4* positions [[buffer(2)]], device const float4* normals [[buffer(3)]],
    constant uint4& config [[buffer(4)]], device const RenderAppearance* appearances [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= config.x) return;
    uint packed = corners[i], comp = packed >> 24;
    RTVertex v; v.emissive = float4(0);
    if (config.y == 1) {
        uint body = packed & 0x001FFFFFu, side = (packed >> 23) & 1u, rim = (packed >> 22) & 1u;
        float4 nt = normals[body];
        float3 p = positions[body].xyz + nt.xyz * (nt.w * (side == 0 ? 1.0 : -1.0) * 0.95);
        if (nt.w == 0 && (side != 0 || rim != 0)) p = float3(0);
        v.position = float4(p,1);
        v.normal = float4(nt.xyz,0.72);
        v.emissive.w = float((packed >> 21) & 1u);
        v.albedo = float4(srgbToLin(mix(float3(0.90),softPalette(comp*7u+2u),0.72)),0);
        if (config.z != 0) {
            RenderAppearance a = appearances[body];
            if (a.albedo.w > 0) v.albedo.rgb = srgbToLin(a.albedo.rgb);
            v.emissive.rgb = a.emissive.rgb;
        }
    } else {
        uint index = (packed & 0x00FFFFFFu)*2u;
        v.position = positions[index];
        v.normal = float4(normalize(positions[index+1].xyz),0.72);
        v.albedo = float4(srgbToLin(mix(float3(0.90),softPalette(comp*5u+11u),0.76)),0);
    }
    output[i] = v;
}
inline float rtVisibility(float3 P, float3 N, float3 L, float bias, instance_acceleration_structure scene) {
    ray r; r.origin = P + N*bias; r.direction = L; r.min_distance = bias*0.25; r.max_distance = 1000;
    intersector<triangle_data, instancing> query;
    query.assume_geometry_type(geometry_type::triangle);
    query.force_opacity(forced_opacity::opaque);
    query.accept_any_intersection(true);
    return query.intersect(r, scene, 1).type == intersection_type::none ? 1.0 : 0.16;
}
kernel void rt_shadows(instance_acceleration_structure scene [[buffer(0)]], constant Uniforms& U [[buffer(1)]],
    depth2d<float,access::read> depth [[texture(0)]], texture2d<float,access::read> normal [[texture(1)]],
    texture2d<float,access::write> output [[texture(2)]], uint2 pixel [[thread_position_in_grid]]) {
    if (any(pixel >= uint2(output.get_width(),output.get_height()))) return;
    float d = depth.read(pixel);
    float3 N = normal.read(pixel).xyz, L = -U.lightDir.xyz;
    float visibility = 1;
    if (d < 1.0 && dot(N,L) > 0) {
        float2 uv = (float2(pixel)+0.5)/float2(output.get_width(),output.get_height());
        float3 P = worldFromDepth(uv,d,U.invViewProj);
        float bias = max(0.0001, screenDepth(d,U)/U.screen.z*0.02);
        visibility = rtVisibility(P,normalize(N),L,bias,scene);
    }
    output.write(float4(visibility),pixel);
}
inline float3 rtLit(float3 P, float3 N, float3 V, float3 albedo, float rough, float metal,
                    float3 emissive, uint source, constant Uniforms& U, instance_acceleration_structure scene) {
    float3 L = -U.lightDir.xyz;
    float visibility = dot(N,L) > 0 ? rtVisibility(P,N,L,0.0002,scene) : 1;
    if (source == 3) return clothRadiance(albedo,emissive,N,V,1,visibility,U);
    if (source == 4) return albedo*(SKY_IRR*1.1+SUN_COL/M_PI_F*max(L.z,0.0)*0.85*visibility);
    return pbrRadiance(albedo,rough,metal,emissive,N,V,1,visibility,U);
}
// Visible-normal GGX sampling: stretch the view, sample the projected
// hemisphere, then transform the normal back (Heitz 2018, JCGT 7(4):1).
// https://jcgt.org/published/0007/04/01/
inline float3 rtGlossyNormal(float3 view, float alpha, float2 sample) {
    float3 stretched = normalize(float3(view.xy*alpha, view.z));
    float3 tangent = stretched.z < 0.9999 ? normalize(cross(float3(0,0,1),stretched)) : float3(1,0,0);
    float3 bitangent = cross(stretched,tangent);
    float2 disk = sqrt(sample.x)*float2(cos(2*M_PI_F*sample.y),sin(2*M_PI_F*sample.y));
    disk.y = mix(sqrt(max(0.0,1-disk.x*disk.x)),disk.y,0.5+0.5*stretched.z);
    float3 hemisphere = tangent*disk.x + bitangent*disk.y + stretched*sqrt(max(0.0,1-dot(disk,disk)));
    return normalize(float3(hemisphere.xy*alpha,max(hemisphere.z,0.0)));
}
inline float rtSmithLambda(float cosine, float alpha) {
    return 0.5*(sqrt(1+alpha*alpha*max(0.0,1-cosine*cosine)/max(cosine*cosine,1e-8))-1);
}
kernel void rt_reflections(instance_acceleration_structure scene [[buffer(0)]], constant Uniforms& U [[buffer(1)]],
    device const RTVertex* vertices [[buffer(2)]], device const RTObject* objects [[buffer(3)]],
    device const RTInstance* instances [[buffer(4)]], device const RenderInstance* rigid [[buffer(5)]],
    device const RenderInstance* auxiliary [[buffer(6)]], device const RenderAppearance* appearances [[buffer(7)]],
    constant uint& hasAppearance [[buffer(8)]], depth2d<float,access::read> depth [[texture(0)]],
    texture2d<float,access::read> normal [[texture(1)]], texture2d<float,access::read_write> output [[texture(2)]],
    texture2d<float,access::read> material [[texture(3)]], texture2d<float,access::read> visibility [[texture(4)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if (any(pixel >= uint2(output.get_width(),output.get_height()))) return;
    float d = depth.read(pixel); float4 nr = normal.read(pixel);
    if (d >= 1 || nr.w >= U.effects.w) { output.write(float4(0),pixel); return; }
    // A confident current-frame screen hit may avoid the world ray. Partial
    // screen hits are replaced by a fresh world result, without history reuse.
    if (U.rayTracing.y > 0 && nr.w < 0.15 && output.read(pixel).a > 0.95) return;
    float2 uv = (float2(pixel)+0.5)/float2(output.get_width(),output.get_height());
    float3 P = worldFromDepth(uv,d,U.invViewProj), N = normalize(nr.xyz), V = normalize(U.eye.xyz-P);
    if (dot(N,V) <= 0) { output.write(float4(0),pixel); return; }
    float3 tangent = normalize(cross(abs(N.z) < 0.99 ? float3(0,0,1) : float3(0,1,0),N));
    float3 bitangent = cross(N,tangent);
    float3 localV = float3(dot(V,tangent),dot(V,bitangent),dot(V,N));
    float alpha = nr.w*nr.w;
    float4 receiver = material.read(pixel);
    float3 F0 = mix(float3(0.04),receiver.rgb,receiver.a);
    float3 correctionSum = float3(0); float coverage = 0;
    // Stratified samples, followed by a surface-aware spatial resolve.
    // A fixed seed prevents idle flicker and requires no stale frame history.
    const uint sampleCount = 4;
    for (uint sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
    float2 random = float2((float(sampleIndex)+screenNoise(pixel+uint2(0,17)))/float(sampleCount),
                          fract(screenNoise(pixel+uint2(31,0))+float(sampleIndex)*0.6180339887));
    float3 localH = rtGlossyNormal(localV,alpha,random);
    float3 H = tangent*localH.x+bitangent*localH.y+N*localH.z;
    float3 R = reflect(-V,H);
    float NdR = dot(N,R);
    if (NdR <= 0) continue;
    float bias = max(0.0001,screenDepth(d,U)/U.screen.z*0.02);
    ray r; r.origin = P+N*bias; r.direction = R; r.min_distance = bias; r.max_distance = 100;
    intersector<triangle_data,instancing> query;
    query.assume_geometry_type(geometry_type::triangle);
    query.force_opacity(forced_opacity::opaque);
    auto hit = query.intersect(r,scene,2);
    if (hit.type == intersection_type::none) continue;
    RTObject object = objects[hit.instance_id];
    uint base = object.vertexStart + hit.primitive_id*3;
    float3 bary = float3(1-hit.triangle_barycentric_coord.x-hit.triangle_barycentric_coord.y,hit.triangle_barycentric_coord);
    RTVertex a = vertices[base], b = vertices[base+1], c = vertices[base+2];
    float4 nm = a.normal*bary.x+b.normal*bary.y+c.normal*bary.z;
    float4 mat = a.albedo*bary.x+b.albedo*bary.y+c.albedo*bary.z;
    float3 emission = a.emissive.rgb*bary.x+b.emissive.rgb*bary.y+c.emissive.rgb*bary.z;
    float3 hitP = r.origin + R*hit.distance;
    float3 e1 = rtVector(instances[hit.instance_id],b.position.xyz-a.position.xyz);
    float3 e2 = rtVector(instances[hit.instance_id],c.position.xyz-a.position.xyz);
    float3 geomN = normalize(cross(e1,e2));
    float3 hitN = normalize(rtNormal(instances[hit.instance_id],nm.xyz));
    if (object.source == 3 && a.emissive.w > 0.5) hitN = geomN;
    if (dot(hitN,-R) < 0) hitN = -hitN;
    if (dot(geomN,hitN) < 0) geomN = -geomN;
    if (object.source <= 1) {
        RenderInstance instance = object.source == 0 ? rigid[object.index] : auxiliary[object.index];
        mat.rgb = srgbToLin(instance.color.rgb); emission = instance.material.rgb;
    } else if (object.source == 2 && hasAppearance != 0) {
        RenderAppearance appearance = appearances[object.index];
        if (appearance.albedo.w > 0) mat.rgb = srgbToLin(appearance.albedo.rgb);
        emission = appearance.emissive.rgb;
    } else if (object.source == 4) {
        float checker = float((int(floor(hitP.x))+int(floor(hitP.y))) & 1);
        mat.rgb = mix(srgbToLin(float3(0.93,0.93,0.94)),srgbToLin(float3(0.62,0.66,0.72)),checker);
    }
    float3 incoming = rtLit(hitP+geomN*0.0001,hitN,-R,mat.rgb,clamp(nm.w,0.02,1.0),mat.a,emission,object.source,U,scene);
    float viewLambda = rtSmithLambda(localV.z,alpha);
    float masking = (1+viewLambda)/(1+viewLambda+rtSmithLambda(NdR,alpha));
    // BRDF*cos/pdf for visible-normal sampling simplifies to Fresnel*G2/G1.
    // Retain the raster environment's gain while replacing covered radiance.
    float3 response = (F0+(1-F0)*pow(1-saturate(dot(V,H)),5.0))*masking*mix(0.50,0.20,nr.w);
    correctionSum += (incoming-screenEnvironment(R))*response;
    coverage += 1;
    }
    float confidence = 1-smoothstep(U.effects.w-0.15,U.effects.w,nr.w);
    correctionSum *= (1.0/float(sampleCount))*visibility.read(pixel).r*(1-horizonFog(length(P-U.eye.xyz)))*confidence;
    output.write(float4(correctionSum,coverage/float(sampleCount)*confidence),pixel);
}
"""
