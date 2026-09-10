/// Shared finite-emitter geometry and BRDF for primary and secondary shading.
let areaLightingShaderSource = """
  inline float3 areaNormal(AreaLight light) { return cross(light.right.xyz,light.up.xyz); }
  inline float areaSize(AreaLight light) { return light.right.w*light.up.w*(light.position.w>0 ? M_PI_F : 4.0); }
  inline float3 areaPoint(AreaLight light,float2 xi) {
      float2 p=xi*2-1;
      if (light.position.w>0) p=sqrt(xi.x)*float2(cos(2*M_PI_F*xi.y),sin(2*M_PI_F*xi.y));
      return light.position.xyz+light.right.xyz*(p.x*light.right.w)+light.up.xyz*(p.y*light.up.w);
  }
  // RGB radiance and nearest distance. Analytic emitters never enter collision
  // geometry, and a back-facing one-sided emitter cannot illuminate a ray.
  inline float4 areaIntersection(float3 origin,float3 direction,constant Uniforms& U) {
      float4 result=float4(0,0,0,1e20);
      for(uint i=0;i<uint(U.areaSettings.x);++i) {
          AreaLight light=U.areaLights[i];float3 n=areaNormal(light);
          float denominator=dot(direction,n);
          if (abs(denominator)<1e-8 || (light.radiance.w==0 && denominator>=0)) continue;
          float t=dot(light.position.xyz-origin,n)/denominator;
          if (t<=0.00002 || t>=result.w) continue;
          float3 delta=origin+direction*t-light.position.xyz;
          float2 uv=float2(dot(delta,light.right.xyz)/light.right.w,dot(delta,light.up.xyz)/light.up.w);
          if (light.position.w>0 ? dot(uv,uv)>1 : any(abs(uv)>1)) continue;
          result=float4(light.radiance.rgb,t);
      }
      return result;
  }
  inline float3 areaBRDF(float3 albedo,float rough,float metal,float3 N,float3 V,float3 L,bool diffuseOnly) {
      float nl=saturate(dot(N,L)),nv=max(dot(N,V),1e-4);
      if (nl<=0) return float3(0);
      float3 sum=L+V,H=sum*rsqrt(max(dot(sum,sum),1e-8));
      float nh=saturate(dot(N,H)),vh=saturate(dot(V,H));
      float3 f0=mix(float3(0.04),albedo,metal),F=f0+(1-f0)*pow(1-vh,5.0);
      float3 diffuse=albedo*(1-metal)*(1-F)/M_PI_F;
      if (diffuseOnly) return diffuse*nl;
      float a2=max(rough*rough*rough*rough,1e-8),den=nh*nh*(a2-1)+1;
      float D=a2/(M_PI_F*den*den),k=(rough+1)*(rough+1)/8;
      float G=nv/(nv*(1-k)+k)*nl/(nl*(1-k)+k);
      return (diffuse+D*G*F/max(4*nv*nl,1e-4))*nl;
  }
  // Fast-mode fallback: fixed quadrature of finite emitters, without area shadows.
  inline float3 rasterAreaLighting(float3 P,float3 N,float3 V,float3 albedo,float rough,float metal,constant Uniforms& U) {
      float3 total=float3(0);
      for(uint i=0;i<uint(U.areaSettings.x);++i) {
          AreaLight light=U.areaLights[i];
          for(uint j=0;j<4;++j) {
              float3 delta=areaPoint(light,(float2(j&1u,j>>1u)+0.5)*0.5)-P;
              float d2=max(dot(delta,delta),1e-8);float3 L=delta*rsqrt(d2);
              float cosine=dot(areaNormal(light),-L);
              cosine=light.radiance.w>0 ? abs(cosine) : max(cosine,0.0);
              total+=areaBRDF(albedo,rough,metal,N,V,L,false)*light.radiance.rgb*(cosine*areaSize(light)/(4*d2));
          }
      }
      return total;
  }
  """

let areaRayShaderSource = """
  inline float3 rtAreaLighting(float3 P,float3 N,float3 V,float3 albedo,float rough,float metal,
      instance_acceleration_structure scene,constant Uniforms& U,uint2 pixel,bool diffuseOnly=false) {
      float3 total=float3(0);
      uint count=uint(max(U.areaSettings.y,1.0));
      uint frame=uint(U.reconstruction.w);
      intersector<triangle_data,instancing> query;
      query.assume_geometry_type(geometry_type::triangle);query.force_opacity(forced_opacity::opaque);
      query.accept_any_intersection(true);
      for(uint i=0;i<uint(U.areaSettings.x);++i) {
          AreaLight light=U.areaLights[i];
          float2 shift=float2(screenNoise(pixel+uint2(i*733u+frame*103u,frame*71u)),screenNoise(pixel+uint2(frame*53u,i*977u+frame*97u)));
          for(uint j=0;j<count;++j) {
              float2 xi=fract(float2(float(j)/float(count),float(reverse_bits(j))*2.3283064365386963e-10)+shift);
              float3 delta=areaPoint(light,xi)-P;float d2=dot(delta,delta);
              if (d2<1e-8) continue;
              float distance=sqrt(d2);float3 L=delta/distance;
              float cosine=dot(areaNormal(light),-L);
              cosine=light.radiance.w>0 ? abs(cosine) : max(cosine,0.0);
              if (cosine<=0 || dot(N,L)<=0) continue;
              ray r;r.origin=P+N*0.0001;r.direction=L;r.min_distance=0.00002;r.max_distance=max(0.00003,distance-0.0002);
              if (rtGroundDistance(r,U)>=0 || query.intersect(r,scene,1).type!=intersection_type::none) continue;
              total+=areaBRDF(albedo,rough,metal,N,V,L,diffuseOnly)*light.radiance.rgb*(cosine*areaSize(light)/(d2*float(count)));
          }
      }
      return total;
  }
  inline float3 rtPrimaryPosition(float3 P,instance_acceleration_structure scene,constant Uniforms& U) {
      // A depth32 camera buffer can quantize a distant surface behind its actual
      // triangle. Recover the primary intersection before shadow rays rather than
      // increasing the normal bias until thin contacts lose their shadows.
      ray camera;camera.origin=U.eye.xyz;camera.direction=normalize(P-U.eye.xyz);
      float2 clip=cameraRayInterval(camera.direction,U);
      // Restrict recovery to the raster receiver's depth neighborhood, so a
      // different surface cannot supply its position with this receiver's material.
      float expected=length(P-U.eye.xyz), tolerance=max(0.001,expected*0.001);
      camera.min_distance=max(clip.x,expected-tolerance);
      camera.max_distance=min(clip.y,expected+tolerance);
      if (camera.max_distance<=camera.min_distance) return P;
      intersector<triangle_data,instancing> primary;
      primary.assume_geometry_type(geometry_type::triangle);primary.force_opacity(forced_opacity::opaque);
      auto hit=primary.intersect(camera,scene,2);
      float ground=rtGroundDistance(camera,U);
      float distance=hit.type==intersection_type::none ? 1e20 : hit.distance;
      if (ground>=0) distance=min(distance,ground);
      return distance<1e19 ? camera.origin+camera.direction*distance : P;
  }
  kernel void rt_area_lighting(instance_acceleration_structure scene [[buffer(0)]],constant Uniforms& U [[buffer(1)]],
      depth2d<float,access::read> depth [[texture(0)]],texture2d<float,access::read> normal [[texture(1)]],
      texture2d<float,access::write> output [[texture(2)]],texture2d<float,access::read> material [[texture(3)]],
      uint2 pixel [[thread_position_in_grid]]) {
      if (any(pixel>=uint2(output.get_width(),output.get_height()))) return;
      float d=depth.read(pixel);
      if (d>=1) {output.write(float4(0),pixel);return;}
      float2 uv=(float2(pixel)+0.5)/float2(depth.get_width(),depth.get_height());
      float3 P=worldFromDepth(uv,d,U.invViewProj);float4 nr=normal.read(pixel),m=material.read(pixel);
      float3 N=normalize(nr.xyz),V=normalize(U.eye.xyz-P);
      P=rtPrimaryPosition(P,scene,U);
      output.write(float4(rtAreaLighting(P,N,V,m.rgb,nr.w,m.a,scene,U,pixel),1),pixel);
  }
  """
