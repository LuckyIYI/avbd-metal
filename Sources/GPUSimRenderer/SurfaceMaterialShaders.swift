/// One material evaluator shared by raster surfaces, reconstruction guides and
/// secondary rays. Scene-specific patterns live in caller programs, not here.
func makeSurfaceMaterialShaderSource(programs: [GPUSimMaterialProgram]) -> String {
  let functions = programs.enumerated().map { index, program in
    "namespace materialProgram\(index + 1) {\n\(program.supportingSource)\ninline void evaluate(MaterialContext context, thread MaterialSample& surface) {\n\(program.body)\n}\n}"
  }.joined(separator: "\n")
  let cases = programs.indices.map {
    "case \($0 + 1): materialProgram\($0 + 1)::evaluate(context,surface); break;"
  }.joined(separator: "\n")
  return """
    struct MaterialRecord { float4 color; float4 emission; float4 uv; uint4 maps; uint4 extra; float4 parameters; uint4 channels; };
    struct MaterialResources {
        array<texture2d<float>,128> maps [[id(0)]];
        device const MaterialRecord* records [[id(128)]];
        uint count [[id(129)]];
    };
    struct MaterialContext { float3 position; float2 uv; float footprint; float4 parameters; };
    struct MaterialSample { float3 color; float roughness; float metallic; float3 emission; float3 normal; };
    \(functions)
    inline float4 materialTex(uint index, MaterialContext context, uint clampUV, constant MaterialResources& resources) {
        if (index == 0xffffffffu || index >= 128) return float4(1);
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
        if (input.uvMaterial.z < 1) return input;
        MaterialContext context = { input.materialPosition,input.uvMaterial.xy,
            max(length(dfdx(input.uvMaterial.xy)),length(dfdy(input.uvMaterial.xy))),float4(0) };
        MaterialSample surface = { input.albedo,input.pbr.x,input.pbr.y,input.emissive,float3(0,0,1) };
        surface = evaluateMaterial(uint(input.uvMaterial.z),context,surface,resources);
        input.albedo = surface.color; input.pbr = float2(surface.roughness,surface.metallic); input.emissive = surface.emission;
        float2 u = dfdx(input.uvMaterial.xy), v = dfdy(input.uvMaterial.xy);
        float determinant = u.x*v.y-u.y*v.x;
        if (abs(determinant)>1e-12) {
            float3 dx = dfdx(input.world), dy = dfdy(input.world);
            input.normal = materialNormal(normalize(input.normal),(dx*v.y-dy*u.y)/determinant,(dy*u.x-dx*v.x)/determinant,surface.normal);
        }
        return input;
    }
    """
}
