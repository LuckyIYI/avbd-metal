/// Diffuse radiance is stored without receiver albedo so spatial reconstruction
/// cannot smear material colors. Rays use world geometry, including offscreen hits.
let diffuseCommonShaderSource = """
inline float3 diffuseAmbient(float3 N) { return mix(GND_IRR,SKY_IRR,N.z*0.5+0.5); }
inline float3 diffuseEnvironment(float3 R) {
    // The cosine-weighted integral of R is 2/3 N. This linear environment
    // therefore integrates to the renderer's existing analytic diffuse sky.
    return mix(GND_IRR,SKY_IRR,R.z*0.75+0.5);
}
inline uint2 diffuseSurfacePixel(uint2 p, uint2 size) { return min(p*2u+1u,size-1u); }
inline float4 surfaceDiffuse(float2 uv, float3 P, float3 N, constant Uniforms& U,
    texture2d<float> diffuse, texture2d<float> normal, depth2d<float> depth) {
    if (U.reconstruction.x > 0) {
        uint2 pixel = min(uint2(uv*float2(diffuse.get_width(),diffuse.get_height())), uint2(diffuse.get_width()-1,diffuse.get_height()-1));
        return diffuse.read(pixel);
    }
    float2 size = float2(depth.get_width(),depth.get_height());
    float2 p = (uv*size-1.5)*0.5, f = fract(p);
    int2 base = int2(floor(p)), outputSize = int2(diffuse.get_width(),diffuse.get_height());
    float z = dot(P-U.eye.xyz,cross(U.camRight.xyz,U.camUp.xyz));
    float tolerance = max(0.0005,z/U.screen.z*1.5);
    float3 sum = float3(0); float weight = 0;
    for (int y=0;y<2;++y) for (int x=0;x<2;++x) {
        int2 q = base+int2(x,y);
        if (any(q<0) || any(q>=outputSize)) continue;
        uint2 surface = diffuseSurfacePixel(uint2(q),uint2(size));
        float d = depth.read(surface); float3 QN = normal.read(surface).xyz;
        float4 value = diffuse.read(uint2(q));
        if (d>=1 || value.a==0) continue;
        float3 Q = worldFromDepth((float2(surface)+0.5)/size,d,U.invViewProj);
        float plane = max(abs(dot(Q-P,N)),abs(dot(Q-P,QN)));
        float w = (x ? f.x : 1-f.x)*(y ? f.y : 1-f.y);
        w *= saturate(1-plane/tolerance)*pow(saturate(dot(N,QN)),32.0);
        sum += value.rgb*w; weight += w;
    }
    // A quarter-resolution irradiance sample cannot represent a subpixel rim.
    // Hand those rapidly changing normals back to the existing local lighting.
    float footprint = saturate(1-length(fwidth(N))*8);
    return weight>1e-5 ? float4(sum/weight,saturate(weight*4)*footprint) : float4(0);
}
"""

let diffuseFilterShaderSource = """
fragment float4 diffuse_temporal_fragment(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
    texture2d<float> current [[texture(0)]], texture2d<float> history [[texture(1)]]) {
    uint2 p = uint2(in.position.xy);
    float4 value = current.read(p);
    if (U.temporal.z<=1) return value;
    return float4(mix(history.read(p).rgb,value.rgb,1.0/U.temporal.z),value.a);
}

fragment float4 diffuse_filter_fragment(FSOut in [[stage_in]], constant Uniforms& U [[buffer(1)]],
    constant float4& filter [[buffer(2)]], texture2d<float> raw [[texture(0)]],
    depth2d<float> depth [[texture(1)]], texture2d<float> normal [[texture(2)]]) {
    uint2 pixel = uint2(in.position.xy), size = uint2(raw.get_width(),raw.get_height());
    uint2 surfaceSize = uint2(depth.get_width(),depth.get_height());
    uint2 s = diffuseSurfacePixel(pixel,surfaceSize);
    float d = depth.read(s);
    if (d>=1 || raw.read(pixel).a==0) return float4(0);
    float3 P = screenPosition((float2(s)+0.5)/float2(surfaceSize),d,U);
    float3 N = screenVector(normal.read(s).xyz,U);
    float tolerance = max(0.001,P.z/U.screen.z*0.75);
    float3 sum = float3(0); float weights = 0;
    for (int y=-1;y<=1;++y) for (int x=-1;x<=1;++x) {
        int2 q = int2(pixel)+int2(x,y)*int(filter.x);
        if (any(q<0) || any(q>=int2(size))) continue;
        uint2 qs = diffuseSurfacePixel(uint2(q),surfaceSize);
        float dq = depth.read(qs); float4 value = raw.read(uint2(q));
        if (dq>=1 || value.a==0) continue;
        float3 Q = screenPosition((float2(qs)+0.5)/float2(surfaceSize),dq,U);
        float3 QN = screenVector(normal.read(qs).xyz,U);
        float plane = max(abs(dot(Q-P,N)),abs(dot(Q-P,QN)));
        float w = pow(saturate(dot(N,QN)),32.0)*saturate(1-plane/tolerance);
        w *= (x==0 ? 2.0 : 1.0)*(y==0 ? 2.0 : 1.0);
        sum += value.rgb*w; weights += w;
    }
    return float4(weights>0 ? sum/weights : float3(0),1);
}
"""

let diffuseRayShaderSource = """
// Sixteen strata per receiver, with complementary sub-strata over every
// 4x4 receiver neighborhood. The two tent passes integrate those neighbors
// with balanced weights on a flat surface. Shift the sub-strata during
// accumulation so each individual receiver also covers the full domain.
inline float2 diffuseSample(uint2 pixel, uint i, uint frame, float phase) {
    uint2 cell = uint2(i&3u,i>>2u);
    uint2 subcell = uint2((pixel.x+2u*pixel.y+cell.y+(frame&3u))&3u,
                         (pixel.y+cell.x+((frame>>2u)&3u))&3u);
    uint stratum = subcell.x+subcell.y*4u;
    float2 jitter = fract(float2(screenNoise(uint2(i,stratum+113u)),screenNoise(uint2(i,stratum+331u)))
                         +float2(phase,phase*5));
    return (float2(cell)+(float2(subcell)+jitter)*0.25)*0.25;
}
// Stratify scarce diffuse rays where indirect light dominates the visible signal.
inline float2 sparseDiffuseSample(uint2 pixel, uint i, uint samples, uint frame) {
    float2 xi = float2(screenNoise(pixel+uint2(frame*103u,frame*71u)),
                       screenNoise(pixel+uint2(frame*53u+231u,frame*97u+17u)));
    return samples == 4u ? (float2(i&1u,i>>1u)+xi)*0.5 : xi;
}
kernel void rt_diffuse(instance_acceleration_structure scene [[buffer(0)]], constant Uniforms& U [[buffer(1)]],
    device const RTVertex* vertices [[buffer(2)]], device const RTObject* objects [[buffer(3)]],
    device const RTInstance* instances [[buffer(4)]], device const RenderInstance* rigid [[buffer(5)]],
    device const RenderInstance* auxiliary [[buffer(6)]], device const RenderAppearance* appearances [[buffer(7)]],
    constant uint& hasAppearance [[buffer(8)]], depth2d<float,access::read> depth [[texture(0)]],
    texture2d<float,access::read> normal [[texture(1)]], texture2d<float,access::write> output [[texture(2)]],
    texture2d<float,access::read> material [[texture(3)]],
    texture2d<float,access::read> visibility [[texture(4)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if (any(pixel>=uint2(output.get_width(),output.get_height()))) return;
    uint2 size = uint2(depth.get_width(),depth.get_height()), s = U.reconstruction.x > 0 ? pixel : diffuseSurfacePixel(pixel,size);
    float d = depth.read(s);
    float3 N = normalize(normal.read(s).xyz);
    if (d>=1 || material.read(s).a>=0.99) { output.write(float4(0),pixel); return; }
    float3 cameraP = screenPosition((float2(s)+0.5)/float2(size),d,U);
    // Reconstruct relative to the camera before adding its world translation.
    // A homogeneous inverse-VP divide amplifies cancellation on grazing planes.
    float3 P = U.eye.xyz+U.camRight.xyz*cameraP.x+U.camUp.xyz*cameraP.y
        +cross(U.camRight.xyz,U.camUp.xyz)*cameraP.z;
    float3 tangent = normalize(cross(abs(N.z)<0.99 ? float3(0,0,1) : float3(0,1,0),N));
    float3 bitangent = cross(N,tangent);
    float3 sum = float3(0);
    uint samples = 16u;
    if (U.reconstruction.x > 0) {
        float direct = max(dot(N,-U.lightDir.xyz),0.0)*visibility.read(s).g;
        samples = direct < 0.2 ? 4u : 1u;
    }
    for (uint i=0;i<samples;++i) {
        float2 xi = diffuseSample(pixel,i,uint(max(U.temporal.z-1,0.0)),U.temporal.x);
        if (U.reconstruction.x > 0) {
            xi = sparseDiffuseSample(pixel,i,samples,uint(U.reconstruction.w));
        }
        float phi = 2*M_PI_F*xi.y;
        float3 R = tangent*(sqrt(xi.x)*cos(phi))+bitangent*(sqrt(xi.x)*sin(phi))+N*sqrt(1-xi.x);
        float bias = max(0.0001,screenDepth(d,U)/U.screen.z*0.02);
        ray r; r.origin = P+N*bias; r.direction = R; r.min_distance = bias; r.max_distance = 100;
        float4 hit = rtIncoming(r,scene,U,vertices,objects,instances,rigid,auxiliary,appearances,hasAppearance,true);
        // Control variate: open sky is integrated analytically and contributes
        // exactly zero noise. Only geometry changes the diffuse lighting.
        if (U.reconstruction.x > 0) {
            // A one-sample control-variate correction can make ambient + sample
            // negative. Clipping that before denoising would bias dark rooms
            // brighter. Sample nonnegative incoming radiance instead, then
            // subtract the exact baseline expected by the compositor.
            sum += (hit.w>0 ? hit.rgb : diffuseEnvironment(R))-diffuseAmbient(N);
        } else if (hit.w>0) sum += hit.rgb-diffuseEnvironment(R);
    }
    // cos(theta)/pi cancels the cosine-hemisphere PDF: no extra pi or albedo.
    output.write(float4(sum/float(samples),1),pixel);
}
"""
