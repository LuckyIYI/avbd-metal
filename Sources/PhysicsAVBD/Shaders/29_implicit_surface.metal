#ifdef AVBD_IMPLICIT
struct ImplicitPatch {float3 a,b,c; uint depth;};
struct ImplicitHit {float3 fieldPoint, otherPoint, normal;float separation;};
inline float3 impClosestTriangle(float3 p,float3 a,float3 b,float3 c) {
    float3 ab=b-a,ac=c-a,ap=p-a;float d1=dot(ab,ap),d2=dot(ac,ap);
    if(d1<=0&&d2<=0)return a;
    float3 bp=p-b;float d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0&&d4<=d3)return b;
    float vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0)return a+ab*(d1/(d1-d3));
    float3 cp=p-c;float d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0&&d5<=d6)return c;
    float vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0)return a+ac*(d2/(d2-d6));
    float va=d3*d6-d5*d4;if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0)return b+(c-b)*((d4-d3)/((d4-d3)+(d5-d6)));
    float inv=1/(va+vb+vc);return a+ab*(vb*inv)+ac*(vc*inv);
}
// Returns -1 when a finite edge, tilted support or missing witness needs the
// general search. Bounding support certifies separation/contact depth; witnesses
// are evaluated on the actual field surface when the shader is compiled.
inline int implicitPlaneContacts(uint fi,uint oi,float3 fp,float4 fq,float3 op,float4 oq,
 float margin,thread ImplicitHit* hits) {
    if(!implicit_plane_acceleration())return -1;
    float3 lo=implicit_bounds_min(fi),hi=implicit_bounds_max(fi),h=(hi-lo)*0.5f;
    float3 center=q_rotate(q_conj(oq),fp+q_rotate(fq,(lo+hi)*0.5f)-op);
    float3 ax=q_rotate(q_conj(oq),q_rotate(fq,float3(1,0,0)));
    float3 ay=q_rotate(q_conj(oq),q_rotate(fq,float3(0,1,0)));
    float3 az=q_rotate(q_conj(oq),q_rotate(fq,float3(0,0,1)));
    float3 extent=abs(ax)*h.x+abs(ay)*h.y+abs(az)*h.z;
    float3 bh=implicit_native_box_half(oi),normal=float3(0);float plane=0;
    if(bh.x>0) {
        int face=-1;
        for(int a=0;a<3;a++) {
            int u=(a+1)%3,v=(a+2)%3;
            if(abs(center[a])>=bh[a] && abs(center[u])+extent[u]<bh[u]-margin && abs(center[v])+extent[v]<bh[v]-margin && abs(center[a])-extent[a]>-bh[a]+margin) {face=a;break;}
        }
        if(face<0)return -1;
        normal[face]=center[face]>=0 ? 1.0f : -1.0f;plane=bh[face];
    } else {
        float2 rect=implicit_rectangle_half(oi);
        if(!implicit_infinite_plane(oi) && (rect.x<=0 || any(abs(center.xy)+extent.xy>rect-margin)))return -1;
        if(abs(center.z)<1e-7f)return -1;
        normal.z=center.z>0 ? 1.0f : -1.0f;
    }
    float3 worldN=q_rotate(oq,normal),localN=q_rotate(q_conj(fq),worldN);
    float bound=dot(normal,center)-dot(abs(normal),extent)-plane;
    if(bound>margin)return 0;
    // Flat support patches only. Near-tilted point/edge manifolds must retain
    // the general search rather than switch between incompatible supports.
    float3 alignment=abs(localN);float dominant=max(alignment.x,max(alignment.y,alignment.z));
    if(alignment.x+alignment.y+alignment.z-dominant>1e-7f)return -1;
    uint count=implicit_support_count(fi);if(count==0)return -1;
    float best=INFINITY;int deepest=-1;
    for(uint k=0;k<count;k++) {
        float3 p=implicit_support_vertex(fi,k);
        float d=dot(worldN,fp-op)+dot(localN,p)-plane;
        if(d<best){best=d;deepest=int(k);}
    }
    const float certificateTolerance=2e-6f;
    if(best-bound>certificateTolerance)return -1;
    if(best>margin)return -1;
    // Choose a small, spatially spread support manifold, keeping actual points.
    int selected[4];int nout=0;
    for(int slot=0;slot<4;slot++) {
        int chosen=-1;float farthest=-1;
        for(uint k=0;k<count;k++) {
            bool used=false;for(int j=0;j<nout;j++)if(selected[j]==int(k))used=true;
            if(used)continue;
            float3 p=implicit_support_vertex(fi,k);
            float d=dot(worldN,fp-op)+dot(localN,p)-plane;
            if(d>margin || d>best+certificateTolerance)continue;
            float score=slot==0 ? (int(k)==deepest ? 1.0f : 0.0f) : INFINITY;
            for(int j=0;j<nout;j++)score=min(score,distance(p,implicit_support_vertex(fi,uint(selected[j]))));
            if(score>farthest){farthest=score;chosen=int(k);}
        }
        if(chosen<0)break;
        selected[nout]=chosen;
        float3 p=fp+q_rotate(fq,implicit_support_vertex(fi,uint(chosen)));
        float d=dot(worldN,p-op)-plane;
        hits[nout++]={p,p-worldN*d,-worldN,d};
    }
    if(nout<3)return -1;
    float area=0;
    for(int j=1;j<nout;j++)for(int k=j+1;k<nout;k++)area=max(area,length(cross(hits[j].fieldPoint-hits[0].fieldPoint,hits[k].fieldPoint-hits[0].fieldPoint)));
    if(area<0.01f*length_squared(h))return -1;
    return nout;
}
// Adaptive triangle coverage uses the 1-Lipschitz bound of a metric SDF.
// Exhaustion is reported, never interpreted as separation.
inline int implicitSurfaceContacts(uint fi,uint oi,float3 fp,float4 fq,float3 op,float4 oq,
 float3 halfBounds,float margin,thread ImplicitHit* hits,thread uint& failed) {
    int planeHits=implicitPlaneContacts(fi,oi,fp,fq,op,oq,margin,hits);
    if(planeHits>=0)return planeHits;
    float3 boundMin=implicit_bounds_min(fi),boundMax=implicit_bounds_max(fi);
    halfBounds=(boundMax-boundMin)*0.5f;
    ImplicitHit candidates[32];int count=0;uint work=0;
    float coverage=max(0.001f,length(halfBounds)*0.22f);
    float tolerance=max(1e-6f,min(2e-5f,margin*0.25f));
    float4 inverse=q_conj(fq);
    uint nt=implicit_triangle_count(oi);
    for(uint ti=0;ti<nt;ti++) {
        float3 a=q_rotate(inverse,op+q_rotate(oq,implicit_triangle_vertex(oi,ti*3))-fp);
        float3 b=q_rotate(inverse,op+q_rotate(oq,implicit_triangle_vertex(oi,ti*3+1))-fp);
        float3 c=q_rotate(inverse,op+q_rotate(oq,implicit_triangle_vertex(oi,ti*3+2))-fp);
        if(implicit_infinite_plane(oi)) {
            float3 pc=q_rotate(q_conj(oq),fp+q_rotate(fq,(boundMin+boundMax)*0.5f)-op);
            float3 ex=q_rotate(q_conj(fq),q_rotate(oq,float3(1,0,0)));
            float3 ey=q_rotate(q_conj(fq),q_rotate(oq,float3(0,1,0)));
            float2 extent=float2(dot(abs(ex),halfBounds),dot(abs(ey),halfBounds))+margin*2;
            float3 ua=implicit_triangle_vertex(oi,ti*3),ub=implicit_triangle_vertex(oi,ti*3+1),uc=implicit_triangle_vertex(oi,ti*3+2);
            a=q_rotate(inverse,op+q_rotate(oq,float3(pc.xy+ua.xy*extent,0))-fp);
            b=q_rotate(inverse,op+q_rotate(oq,float3(pc.xy+ub.xy*extent,0))-fp);
            c=q_rotate(inverse,op+q_rotate(oq,float3(pc.xy+uc.xy*extent,0))-fp);
        }
        ImplicitPatch stack[40];int pending=1;stack[0]={a,b,c,0};
        while(pending>0) {
            if(++work>8192u){failed=1;return 0;}
            ImplicitPatch t=stack[--pending];
            if(any(min(t.a,min(t.b,t.c))>boundMax+margin)||any(max(t.a,max(t.b,t.c)) < boundMin-margin))continue;
            float3 p=(t.a+t.b+t.c)/3;
            float radius=max(length(p-t.a),max(length(p-t.b),length(p-t.c)));
            float4 q=implicit_query(fi,p);
            if(!all(isfinite(q))){failed=2;return 0;}
            if(q.w-radius>margin)continue;
            if(radius<=coverage) {
                // Refine a minimum on this patch, including edges/interior.
                for(int it=0;it<24;it++) {
                    float step=radius;bool improved=false;
                    for(int bt=0;bt<10;bt++) {
                        float3 trial=impClosestTriangle(p-q.xyz*step,a,b,c);
                        float4 next=implicit_query(fi,trial);
                        if(!all(isfinite(next))){failed=2;return 0;}
                        if(next.w<q.w-1e-9f){p=trial;q=next;improved=true;break;}
                        step*=0.5f;
                    }
                    if(!improved)break;
                }
                if(q.w<=margin+tolerance) {
                    float gl=length(q.xyz);if(gl<1e-6f){failed=3;return 0;}
                    ImplicitHit h;h.otherPoint=p;
                    float3 planeN=normalize(cross(b-a,c-a));
                    if(dot(planeN,p-implicit_interior_seed(fi))<0)planeN=-planeN;
                    h.normal=planeN;h.separation=q.w;h.fieldPoint=p-planeN*q.w;
                    if(q.w<0) {
                        // A triangle supplies its face normal. Find an actual
                        // SDF boundary witness along it, rather than mixing
                        // unrelated nearest-exit normals around the rim.
                        float travel=0;float residual=q.w;
                        for(int it=0;it<64 && abs(residual)>tolerance;it++) {
                            travel+=abs(residual);
                            residual=implicit_query(fi,p+planeN*travel).w;
                            if(!isfinite(residual)){failed=2;return 0;}
                        }
                        if(abs(residual)>tolerance){failed=7;return 0;}
                        h.fieldPoint=p+planeN*travel;h.separation=-travel;
                    } else if(abs(implicit_query(fi,h.fieldPoint).w)>tolerance) {continue;}

                    // Bound candidate storage while preserving spatial extent.
                    int near=-1;float closest=coverage*0.18f;
                    for(int k=0;k<count;k++){float d=distance(candidates[k].otherPoint,p);if(d<closest){closest=d;near=k;}}
                    if(near>=0){if(q.w<candidates[near].separation)candidates[near]=h;}
                    else if(count<32)candidates[count++]=h;
                    else {
                        int replace=-1;float weakest=INFINITY;
                        for(int k=0;k<count;k++){float d=INFINITY;for(int j=0;j<count;j++)if(j!=k)d=min(d,distance(candidates[k].otherPoint,candidates[j].otherPoint));if(d<weakest){weakest=d;replace=k;}}
                        float nearest=INFINITY;for(int k=0;k<count;k++)nearest=min(nearest,distance(candidates[k].otherPoint,p));
                        if(nearest>weakest)candidates[replace]=h;
                    }
                    continue;
                }
                if(radius<=tolerance)continue; // Separation resolved within tolerance.
            }
            if(t.depth>=30u||pending+2>40){failed=4;return 0;}
            float ab=length_squared(t.a-t.b),bc=length_squared(t.b-t.c),ca=length_squared(t.c-t.a);
            if(ab>=bc&&ab>=ca){float3 m=(t.a+t.b)*0.5f;stack[pending++]={t.a,m,t.c,t.depth+1};stack[pending++]={m,t.b,t.c,t.depth+1};}
            else if(bc>=ca){float3 m=(t.b+t.c)*0.5f;stack[pending++]={t.a,t.b,m,t.depth+1};stack[pending++]={t.a,m,t.c,t.depth+1};}
            else {float3 m=(t.c+t.a)*0.5f;stack[pending++]={t.a,t.b,m,t.depth+1};stack[pending++]={m,t.b,t.c,t.depth+1};}
        }
    }
    if(count==0 && implicit_closed_surface(oi)) {
        // Fully contained field: no boundary crossing is NOT proof of separation.
        float3 seed=q_rotate(q_conj(oq),fp+q_rotate(fq,implicit_interior_seed(fi))-op);
        bool inside=true;
        for(uint t=0;t<nt;t++) {float3 a=implicit_triangle_vertex(oi,t*3),b=implicit_triangle_vertex(oi,t*3+1),c=implicit_triangle_vertex(oi,t*3+2);if(dot(normalize(cross(b-a,c-a)),seed-a)>0){inside=false;break;}}
        if(inside){failed=5;return 0;}
    }
    if(count==0)return 0;
    int first=0;for(int k=1;k<count;k++)if(candidates[k].separation<candidates[first].separation)first=k;
    float3 n=candidates[first].normal;uint used=0;int nout=0;
    for(int slot=0;slot<8;slot++) {
        int best=-1;float farthest=-1;
        for(int k=0;k<count;k++) {
            if(used&(1u<<k))continue;
            if(dot(candidates[k].normal,n)<0.8f){
                // One AVBD manifold has one basis. Refuse competing deep patches.
                if(candidates[k].separation < -margin){failed=6;return 0;}continue;
            }
            float score=slot==0 ? -candidates[k].separation : INFINITY;
            for(int j=0;j<nout;j++)score=min(score,distance(hits[j].otherPoint,candidates[k].otherPoint));
            if(score>farthest){best=k;farthest=score;}
        }
        if(best<0)break;used|=1u<<best;hits[nout++]=candidates[best];
    }
    for(int k=0;k<nout;k++){hits[k].fieldPoint=fp+q_rotate(fq,hits[k].fieldPoint);hits[k].otherPoint=fp+q_rotate(fq,hits[k].otherPoint);hits[k].normal=q_rotate(fq,hits[k].normal);}
    return nout;
}
kernel void implicit_append_planes(device uint2* pairs [[buffer(0)]],device atomic_uint* counters [[buffer(1)]],constant SimParams& P [[buffer(2)]],device atomic_uint* poison [[buffer(3)]]) {
    for(uint i=0;i<implicit_global_pair_count();i++) {
        uint slot=atomic_fetch_add_explicit(&counters[CTR_PAIRS],1u,memory_order_relaxed);
        if(slot>=P.maxPairs){latchConvexQueryFailure(counters,poison);return;}
        pairs[slot]=implicit_global_pair(i);
    }
}
#endif
