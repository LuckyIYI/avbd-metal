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
    texture2d<float> current [[texture(0)]], texture2d<float> history [[texture(1)]],
    texture2d<float> previousSurface [[texture(2)]], texture2d<float> surface [[texture(3)]]) {
    uint2 pixel = uint2(in.position.xy), size = uint2(current.get_width(),current.get_height());
    float4 value = current.read(pixel), geometry = surface.read(pixel);
    if (value.a==0 || U.diffuse.y>=1) return value;
    uint2 s = diffuseSurfacePixel(pixel,uint2(U.screen.xy));
    float3 P = worldFromDepth((float2(s)+0.5)/U.screen.xy,geometry.w,U.invViewProj);
    float4 clip = U.prevViewProj*float4(P,1);
    if (clip.w<=0) return value;
    float2 uv = clip.xy/clip.w*float2(0.5,-0.5)+0.5;
    float2 q = (uv*U.screen.xy-1.5)*0.5, f = fract(q);
    int2 base = int2(floor(q));
    float3 sum = float3(0); float weight = 0;
    float pixelWorld = max(0.0005,screenDepth(geometry.w,U)/U.screen.z);
    for (int y=0;y<2;++y) for (int x=0;x<2;++x) {
        int2 h = base+int2(x,y);
        if (any(h<0) || any(h>=int2(size))) continue;
        float4 oldGeometry = previousSurface.read(uint2(h)), old = history.read(uint2(h));
        if (old.a==0 || oldGeometry.w>=1 || dot(geometry.xyz,oldGeometry.xyz)<0.95) continue;
        uint2 hs = diffuseSurfacePixel(uint2(h),uint2(U.screen.xy));
        float3 Q = worldFromDepth((float2(hs)+0.5)/U.screen.xy,oldGeometry.w,U.prevInvViewProj);
        float plane = max(abs(dot(Q-P,geometry.xyz)),abs(dot(Q-P,oldGeometry.xyz)));
        if (length(Q-P)>pixelWorld*4 || plane>pixelWorld*0.3) continue;
        float w = (x ? f.x : 1-f.x)*(y ? f.y : 1-f.y);
        sum += old.rgb*w; weight += w;
    }
    if (weight<0.25) return value;
    // Clip history to current reconstructed lighting so newly exposed light
    // or moving occluders cannot retain an unrelated old color indefinitely.
    float3 lo = value.rgb, hi = value.rgb;
    for (int y=-1;y<=1;++y) for (int x=-1;x<=1;++x) {
        int2 q = clamp(int2(pixel)+int2(x,y),int2(0),int2(size)-1);
        float4 tap = current.read(uint2(q));
        if (tap.a>0) { lo=min(lo,tap.rgb); hi=max(hi,tap.rgb); }
    }
    float3 previous = clamp(sum/weight,lo-0.005,hi+0.005);
    return float4(mix(previous,value.rgb,U.diffuse.y),value.a);
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
kernel void rt_diffuse(instance_acceleration_structure scene [[buffer(0)]], constant Uniforms& U [[buffer(1)]],
    device const RTVertex* vertices [[buffer(2)]], device const RTObject* objects [[buffer(3)]],
    device const RTInstance* instances [[buffer(4)]], device const RenderInstance* rigid [[buffer(5)]],
    device const RenderInstance* auxiliary [[buffer(6)]], device const RenderAppearance* appearances [[buffer(7)]],
    constant uint& hasAppearance [[buffer(8)]], depth2d<float,access::read> depth [[texture(0)]],
    texture2d<float,access::read> normal [[texture(1)]], texture2d<float,access::write> output [[texture(2)]],
    texture2d<float,access::read> material [[texture(3)]], texture2d<float,access::write> surface [[texture(4)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if (any(pixel>=uint2(output.get_width(),output.get_height()))) return;
    uint2 size = uint2(depth.get_width(),depth.get_height()), s = diffuseSurfacePixel(pixel,size);
    float d = depth.read(s);
    float3 N = normalize(normal.read(s).xyz);
    surface.write(float4(N,d),pixel);
    if (d>=1 || material.read(s).a>=0.99) { output.write(float4(0),pixel); return; }
    float3 cameraP = screenPosition((float2(s)+0.5)/float2(size),d,U);
    // Reconstruct relative to the camera before adding its world translation.
    // A homogeneous inverse-VP divide amplifies cancellation on grazing planes.
    float3 P = U.eye.xyz+U.camRight.xyz*cameraP.x+U.camUp.xyz*cameraP.y
        +cross(U.camRight.xyz,U.camUp.xyz)*cameraP.z;
    float3 tangent = normalize(cross(abs(N.z)<0.99 ? float3(0,0,1) : float3(0,1,0),N));
    float3 bitangent = cross(N,tangent);
    float3 sum = float3(0);
    const uint samples = 4;
    float2 jitter = fract(float2(screenNoise(pixel+uint2(113,17)),screenNoise(pixel+uint2(31,91)))
                         +float2(U.temporal.x,U.temporal.x*5));
    for (uint i=0;i<samples;++i) {
        float2 xi = float2((float(i)+jitter.x)/float(samples),fract(jitter.y+float(i)*0.6180339887));
        float phi = 2*M_PI_F*xi.y;
        float3 R = tangent*(sqrt(xi.x)*cos(phi))+bitangent*(sqrt(xi.x)*sin(phi))+N*sqrt(1-xi.x);
        float bias = max(0.0001,screenDepth(d,U)/U.screen.z*0.02);
        ray r; r.origin = P+N*bias; r.direction = R; r.min_distance = bias; r.max_distance = 100;
        float4 hit = rtIncoming(r,scene,U,vertices,objects,instances,rigid,auxiliary,appearances,hasAppearance,true);
        // Control variate: open sky is integrated analytically and contributes
        // exactly zero noise. Only geometry changes the diffuse lighting.
        if (hit.w>0) sum += hit.rgb-diffuseEnvironment(R);
    }
    // cos(theta)/pi cancels the cosine-hemisphere PDF: no extra pi or albedo.
    output.write(float4(sum/float(samples),1),pixel);
}
"""
