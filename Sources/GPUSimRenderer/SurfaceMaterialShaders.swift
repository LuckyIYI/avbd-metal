/// One material evaluator shared by raster surfaces, reconstruction guides and
/// secondary rays. Scene-specific patterns live in caller programs, not here.
///
/// `argumentBuffers: false` emits a stub `MaterialResources` with no texture
/// table for argument-buffer Tier 1 devices. Such libraries cannot hold
/// materials (see `GPUSimMaterialLibrary.init`), so the evaluator is the
/// identity there and every pipeline stays within Tier 1 resource limits.
func makeSurfaceMaterialShaderSource(programs: [GPUSimMaterialProgram], argumentBuffers: Bool = true) -> String {
  let capacity = GPUSimMaterialLibrary.textureCapacity
  let functions = programs.enumerated().map { index, program in
    "namespace materialProgram\(index + 1) {\n\(program.supportingSource)\ninline void evaluate(MaterialContext context, thread MaterialSample& surface) {\n\(program.body)\n}\n}"
  }.joined(separator: "\n")
  let cases = programs.indices.map {
    "case \($0 + 1): materialProgram\($0 + 1)::evaluate(context,surface); break;"
  }.joined(separator: "\n")
  let common = """
    struct MaterialRecord { float4 color; float4 emission; float4 uv; uint4 maps; uint4 extra; float4 parameters; uint4 channels; float4 optics; };
    struct MaterialContext { float3 position; float2 uv; float footprint; float4 parameters; };
    struct MaterialSample { float3 color; float roughness; float metallic; float3 emission; float3 normal; };
    // Material IDs travel through the rasterizer as an interpolated float; round
    // so a sub-ulp interpolation error cannot select the previous material.
    inline uint materialIndex(float encoded) { return uint(max(encoded, 0.0) + 0.5); }
    inline float3 screenEnvironment(float3 R, constant Uniforms& U);
    inline float3 diffuseAmbient(float3 N, constant Uniforms& U);
    inline float3 diffuseEnvironment(float3 R, constant Uniforms& U);
    """
  guard argumentBuffers else {
    return common + """

      struct MaterialResources { uint4 info; };
      inline float2 materialOptics(uint id, constant MaterialResources& resources) { return float2(0,1.5); }
      inline float2 materialLobes(uint id, constant MaterialResources& resources) { return float2(0); }
      inline bool hasEnvironment(constant MaterialResources& resources) { return false; }
      inline bool environmentBackground(constant Uniforms& U,constant MaterialResources& resources) { return false; }
      inline float3 materialEnvironment(float3 R,constant Uniforms& U,constant MaterialResources& resources,float rough=0) { return screenEnvironment(R,U); }
      inline float3 materialDiffuseAmbient(float3 N,constant Uniforms& U,constant MaterialResources& resources) { return diffuseAmbient(N,U); }
      inline float3 materialDiffuseEnvironment(float3 R,constant Uniforms& U,constant MaterialResources& resources) { return diffuseEnvironment(R,U); }
      inline float3 materialHorizon(float3 direction,constant Uniforms& U,constant MaterialResources& resources) { return HORIZON_LIN; }
      inline MaterialSample evaluateMaterial(uint id, MaterialContext context, MaterialSample surface, constant MaterialResources& resources) { return surface; }
      inline float3 materialNormal(float3 n, float3 t, float3 b, float3 map) { return n; }
      inline VOut texturedSurface(VOut input, constant MaterialResources& resources) { return input; }
      """
  }
  return common + """

    struct MaterialResources {
        array<texture2d<float>,\(capacity)> maps [[id(0)]];
        device const MaterialRecord* records [[id(\(capacity))]];
        uint count [[id(\(capacity + 1))]];
        texture2d<float> environmentMap [[id(\(capacity + 2))]];
        device const float4* environmentSH [[id(\(capacity + 3))]];
        float4 environmentSettings [[id(\(capacity + 4))]];
    };
    \(GPUSimEnvironmentLight.basisSource)
    inline bool hasEnvironment(constant MaterialResources& resources) { return resources.environmentSettings.w>0; }
    inline bool environmentBackground(constant Uniforms& U,constant MaterialResources& resources) { return hasEnvironment(resources) && U.environmentSettings.z==0; }
    inline float3 environmentDirection(float3 R,constant Uniforms& U) {
        float a=U.environmentSettings.y,c=cos(a),s=sin(a);
        return float3(c*R.x+s*R.y,-s*R.x+c*R.y,R.z);
    }
    inline float3 materialEnvironment(float3 R,constant Uniforms& U,constant MaterialResources& resources,float rough=0) {
        if (!hasEnvironment(resources)) return screenEnvironment(R,U);
        R=environmentDirection(normalize(R),U);
        float2 uv=float2(atan2(R.y,R.x)/(2*M_PI_F)+0.5,acos(clamp(R.z,-1.0,1.0))/M_PI_F);
        constexpr sampler s(coord::normalized,s_address::repeat,t_address::clamp_to_edge,filter::linear,mip_filter::linear);
        float lod=rough*rough*float(resources.environmentMap.get_num_mip_levels()-1);
        float3 color=resources.environmentMap.sample(s,uv,level(lod)).rgb;
        return max(select(float3(0),color,isfinite(color)),float3(0))*U.environmentSettings.x;
    }
    inline float3 materialDiffuseAmbient(float3 N,constant Uniforms& U,constant MaterialResources& resources) {
        if (!hasEnvironment(resources)) return diffuseAmbient(N,U);
        N=environmentDirection(normalize(N),U);
        float3 color=float3(0);
        for(uint i=0;i<9;++i) color+=resources.environmentSH[i].rgb*environmentBasis(i,N);
        return max(color,float3(0))*U.environmentSettings.x;
    }
    inline float3 materialDiffuseEnvironment(float3 R,constant Uniforms& U,constant MaterialResources& resources) {
        return hasEnvironment(resources) ? materialEnvironment(R,U,resources) : diffuseEnvironment(R,U);
    }
    // Distance fog target: the drawn sky at the horizon in this direction. With
    // an environment background that is a coarse environment sample, so fogged
    // geometry meets the sky without a band; otherwise the analytic horizon.
    inline float3 materialHorizon(float3 direction,constant Uniforms& U,constant MaterialResources& resources) {
        if (!environmentBackground(U,resources)) return HORIZON_LIN;
        float2 flat=direction.xy;
        if (dot(flat,flat)<1e-8) flat=float2(1,0);
        return materialEnvironment(float3(normalize(flat),0),U,resources,0.8);
    }
    \(functions)
    inline float2 materialOptics(uint id, constant MaterialResources& resources) {
        return id > 0 && id <= resources.count ? resources.records[id-1].optics.xy : float2(0,1.5);
    }
    inline float2 materialLobes(uint id, constant MaterialResources& resources) {
        return id > 0 && id <= resources.count ? resources.records[id-1].optics.zw : float2(0);
    }
    inline float4 materialTex(uint index, MaterialContext context, uint clampUV, constant MaterialResources& resources) {
        if (index == 0xffffffffu || index >= \(capacity)) return float4(1);
        constexpr sampler repeatSampler(coord::normalized,address::repeat,filter::linear,mip_filter::linear);
        constexpr sampler clampSampler(coord::normalized,address::clamp_to_edge,filter::linear,mip_filter::linear);
        float size = max(resources.maps[index].get_width(),resources.maps[index].get_height());
        float lod = max(0.0,log2(max(context.footprint*size,1.0)));
        return clampUV ? resources.maps[index].sample(clampSampler,context.uv,level(lod))
                       : resources.maps[index].sample(repeatSampler,context.uv,level(lod));
    }
    inline MaterialSample evaluateMaterial(uint id, MaterialContext context, MaterialSample surface, constant MaterialResources& resources) {
        if (id == 0 || id > resources.count) return surface;
        MaterialRecord m = resources.records[id-1];
        context.uv = context.uv*m.uv.xy+m.uv.zw;
        context.footprint *= max(abs(m.uv.x),abs(m.uv.y));
        context.parameters = m.parameters;
        surface.color *= m.color.rgb*materialTex(m.maps.x,context,m.extra.w,resources).rgb;
        surface.roughness = m.color.w*materialTex(m.maps.y,context,m.extra.w,resources)[m.channels.x];
        surface.metallic = m.emission.w*materialTex(m.maps.z,context,m.extra.w,resources)[m.channels.y];
        surface.emission += m.emission.rgb*materialTex(m.extra.x,context,m.extra.w,resources).rgb;
        if (m.maps.w != 0xffffffffu) {
            surface.normal = materialTex(m.maps.w,context,m.extra.w,resources).xyz*2-1;
            surface.normal.xy *= as_type<float>(m.extra.z)*sign(m.uv.xy);
            if (m.channels.z) surface.normal.y = -surface.normal.y;
        }
        switch (m.extra.y) { \(cases) default: break; }
        surface.color = max(surface.color,float3(0));
        surface.emission = max(surface.emission,float3(0));
        surface.roughness = clamp(surface.roughness,0.02,1.0);
        surface.metallic = saturate(surface.metallic);
        return surface;
    }
    inline float3 materialNormal(float3 n, float3 t, float3 b, float3 map) {
        t -= n*dot(n,t);
        if (dot(t,t)<1e-12 || dot(b,b)<1e-12 || dot(map,map)<1e-12) return n;
        t = normalize(t);
        float3 bitangent = cross(n,t)*(dot(cross(n,t),b)<0 ? -1.0 : 1.0);
        return normalize(t*map.x+bitangent*map.y+n*map.z);
    }
    inline VOut texturedSurface(VOut input, constant MaterialResources& resources) {
        uint id = materialIndex(input.uvMaterial.z);
        if (id == 0) return input;
        // Derivatives are taken before any data-dependent branch so they stay
        // defined when a caller program perturbs the normal per pixel.
        float2 u = dfdx(input.uvMaterial.xy), v = dfdy(input.uvMaterial.xy);
        float3 dx = dfdx(input.world), dy = dfdy(input.world);
        MaterialContext context = { input.materialPosition,input.uvMaterial.xy,max(length(u),length(v)),float4(0) };
        MaterialSample surface = { input.albedo,input.pbr.x,input.pbr.y,input.emissive,float3(0,0,1) };
        surface = evaluateMaterial(id,context,surface,resources);
        input.albedo = surface.color; input.pbr = float2(surface.roughness,surface.metallic); input.emissive = surface.emission;
        // An unperturbed tangent normal leaves the geometric normal unchanged;
        // skip the tangent-frame reconstruction for records without a normal map.
        if (all(surface.normal == float3(0,0,1))) return input;
        float determinant = u.x*v.y-u.y*v.x;
        if (abs(determinant)>1e-12) {
            input.normal = materialNormal(normalize(input.normal),(dx*v.y-dy*u.y)/determinant,(dy*u.x-dx*v.x)/determinant,surface.normal);
        }
        return input;
    }
    """
}
