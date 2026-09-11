// Optional coarse motion correction for stiff, closed tetrahedral assemblies.
// Compiled separately: it does not alter the established solver's codegen.
// One 64-lane workgroup performs a six-DOF subspace Newton step. Local VBD
// still solves all vertex deformation DOFs. Internal elastic energies are
// invariant under this common SE(3) transform and contribute no stiffness.
inline PrimalAccum motionZero() {
    return PrimalAccum{m3_zero(),m3_zero(),m3_zero(),float3(0),float3(0)};
}
inline PrimalAccum motionAdd(PrimalAccum a, PrimalAccum b) {
    return PrimalAccum{m3_add(a.lhsLin,b.lhsLin),m3_add(a.lhsAng,b.lhsAng),
        m3_add(a.lhsCross,b.lhsCross),a.rhsLin+b.rhsLin,a.rhsAng+b.rhsAng};
}
inline PrimalAccum motionProject(PrimalAccum a, float3 r) {
    M3 B=m3_skew(-r), BT=m3_transpose(B);
    M3 C=m3_mulm(a.lhsCross,B);
    return PrimalAccum{a.lhsLin,
        m3_add(a.lhsAng,m3_add(m3_mulm(m3_mulm(BT,a.lhsLin),B),m3_add(C,m3_transpose(C)))),
        m3_add(a.lhsCross,m3_mulm(BT,a.lhsLin)),a.rhsLin,a.rhsAng+cross(r,a.rhsLin)};
}
inline bool motionMember(uint b, uint group, device const uint* owners) {
    return b!=WORLD_BODY && owners[b]==group;
}
kernel void rigid_motion_solve(
    device float4* posLin [[buffer(0)]], device float4* posAng [[buffer(1)]],
    device const float4* initLin [[buffer(2)]], device const float4* initAng [[buffer(3)]],
    device const float4* inertLin [[buffer(4)]], device const float4* inertAng [[buffer(5)]],
    device const float4* props [[buffer(6)]], device const JointGPU* joints [[buffer(7)]],
    device const SpringGPU* springs [[buffer(8)]], device const ManifoldGPU* manifolds [[buffer(9)]],
    device const uint* adjStart [[buffer(10)]], device const uint* adjCount [[buffer(11)]],
    device const uint* adjList [[buffer(12)]], device const float4* shape [[buffer(13)]],
    device const SoftContactGPU* soft [[buffer(14)]], device const uint* members [[buffer(15)]],
    device const uint* owners [[buffer(16)]], constant uint2& span [[buffer(17)]],
    constant uint& group [[buffer(18)]], constant SimParams& P [[buffer(19)]],
    device const uint* bounds [[buffer(20)]], device const float4* ogcPrev [[buffer(21)]],
    device atomic_uint* counters [[buffer(22)]], uint lane [[thread_index_in_threadgroup]])
{
    threadgroup PrimalAccum partial[64];
    threadgroup float fractions[64];
    threadgroup float3 translation, angular, origin;
    if(lane==0) origin=posLin[members[span.x]].xyz;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    PrimalAccum total=motionZero();
    for(uint i=lane;i<span.y;i+=64) {
        uint body=members[span.x+i];
        float4 pl=posLin[body];
        float m=pl.w/(P.dt*P.dt);
        PrimalAccum a=motionZero();
        a.lhsLin=m3_diag(float3(m)); a.rhsLin=(pl.xyz-inertLin[body].xyz)*m;
        if(shape[body].w>=0) {
            a.lhsAng=m3_scale(world_inertia(posAng[body],props[body].xyz),1/(P.dt*P.dt));
            a.rhsAng=m3_mul(a.lhsAng,q_sub(posAng[body],inertAng[body]));
        }
        for(uint k=adjStart[body];k<adjStart[body]+adjCount[body];++k) {
            uint entry=adjList[k], kind=entry>>ADJ_KIND_SHIFT, idx=entry&ADJ_INDEX_MASK;
            if(kind==FK_JOINT) {
                device const JointGPU& j=joints[idx];
                if(motionMember(j.header.x,group,owners) && motionMember(j.header.y,group,owners)) {
                    // Isotropic attachment penalties are rotation invariant.
                    // Preserve the residual AL torque (and anisotropic part)
                    // instead of adding their enormous cancelling diagonals.
                    if(body==j.header.x) {
                        float3 c=xform(posLin[j.header.x].xyz,posAng[j.header.x],j.rA.xyz)
                            -xform(posLin[j.header.y].xyz,posAng[j.header.y],j.rB.xyz);
                        float3 f0=j.lambdaLin.xyz-((j.header.w&1u)?P.alpha*j.C0Lin.xyz*j.penaltyLin.xyz:float3(0));
                        total.rhsAng+=cross(c,c*j.penaltyLin.xyz+f0);
                        float h=length(c)*length(f0)+dot(c,c)*(max(max(j.penaltyLin.x,j.penaltyLin.y),j.penaltyLin.z)
                            -min(min(j.penaltyLin.x,j.penaltyLin.y),j.penaltyLin.z));
                        total.lhsAng=m3_add(total.lhsAng,m3_diag(float3(h)));
                    }
                } else stampJoint(j,body,posLin,posAng,initAng,P.alpha,P.dt,a);
            } else if(kind==FK_SPRING) {
                stampSpring(springs[idx],body,posLin,posAng,a,P.alpha);
            } else if(kind==FK_SOFT) {
                device const SoftContactGPU& sc=soft[idx];
                uint first=WORLD_BODY;
                for(uint j=0;j<4;++j) if(motionMember(sc.ids[j],group,owners)) first=min(first,sc.ids[j]);
                if(body!=first) continue;
                // Assemble the combined stencil ONCE, including off-diagonal
                // terms. Summing vertex-diagonal Hessians loses barycentric
                // cross terms and oversteps a face contact by up to 4x.
                float3 n,t1,t2,ra; bool rigid,round;
                float3 c=softContactC(sc,posLin,posAng,initLin,initAng,P.alpha,n,t1,t2,ra,rigid,round);
                float fs,bnd,kn; float3 f=softContactForce(sc,c,fs,bnd,kn);
                float w=0; float3 lever=0;
                for(uint j=0;j<4;++j) if(motionMember(sc.ids[j],group,owners)) {
                    w+=sc.weights[j];
                    lever+=sc.weights[j]*(posLin[sc.ids[j]].xyz-origin+((j==0&&rigid)?ra:float3(0)));
                }
                M3 L=M3{w*n,w*t1,w*t2}, A=M3{cross(lever,n),cross(lever,t1),cross(lever,t2)};
                M3 K=m3_diag(float3(kn,sc.penalty.y,sc.penalty.z));
                M3 LT=m3_transpose(L), AT=m3_transpose(A);
                PrimalAccum contact={m3_mulm(m3_mulm(LT,K),L),m3_mulm(m3_mulm(AT,K),A),
                    m3_mulm(m3_mulm(AT,K),L),m3_mul(LT,f),m3_mul(AT,f)};
                total=motionAdd(total,contact);
            } else if(kind!=FK_TET && kind!=FK_MEMBRANE && kind!=FK_BEND) {
                device const ManifoldGPU& mf=manifolds[idx];
                if(!(motionMember(mf.header.x,group,owners)&&motionMember(mf.header.y,group,owners)))
                    stampManifold(mf,body,posLin,posAng,initLin,initAng,P.alpha,a);
            }
        }
        total=motionAdd(total,motionProject(a,pl.xyz-origin));
    }
    partial[lane]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=32;stride>0;stride>>=1) {
        if(lane<stride) partial[lane]=motionAdd(partial[lane],partial[lane+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0) {
        PrimalAccum a=partial[0]; float3 v=0,w=0;
        solve6x6(a.lhsLin,a.lhsAng,a.lhsCross,-a.rhsLin,-a.rhsAng,v,w);
        if(!finite3(v)||!finite3(w)) {v=0;w=0;}
        float scale=min(0.8f,0.1f/max(length(w),1e-20f));
        translation=v*scale; angular=w*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // One common step fraction preserves every internal tet's determinant.
    // Backtrack against the same particle trust regions and absolute OGC
    // balls as local VBD. No vertex is individually warped by this pass.
    float fraction=1;
    for(uint i=lane;i<span.y;i+=64) {
        uint body=members[span.x+i]; float3 p=posLin[body].xyz;
        float d2=as_type<float>(bounds[body]);
        float bound=max(0.45f*sqrt(max(d2,0.0f)),-0.2f*shape[body].w);
        float maxMove=0.35f*fabs(shape[body].w);
        for(uint bt=0;bt<20;++bt) {
            float4 dq=normalize(float4(0.5f*angular*fraction,1));
            float3 trial=origin+translation*fraction+q_rotate(dq,p-origin);
            bool limited=length(trial-p)>maxMove;
            if(shape[body].w<0 && d2<1e37f)
                limited=limited || length(trial-ogcPrev[body].xyz)>max(bound,length(p-ogcPrev[body].xyz)+1e-9f);
            if(!limited) break;
            fraction*=0.5f;
        }
    }
    fractions[lane]=fraction;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=32;stride>0;stride>>=1) {
        if(lane<stride) fractions[lane]=min(fractions[lane],fractions[lane+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    fraction=fractions[0];
    if(fraction<1 && lane==0) atomic_fetch_add_explicit(&counters[CTR_OGC],span.y,memory_order_relaxed);
    float4 dq=normalize(float4(0.5f*angular*fraction,1));
    for(uint i=lane;i<span.y;i+=64) {
        uint b=members[span.x+i]; float4 pl=posLin[b];
        posLin[b]=float4(origin+translation*fraction+q_rotate(dq,pl.xyz-origin),pl.w);
        if(shape[b].w>=0) posAng[b]=normalize(q_mul(dq,posAng[b]));
    }
}

// A stiff assembly's contact rows must be conditioned against its coarse
// inertia, not only a single milligram vertex. This is an AL penalty floor;
// it does not change material stiffness, contact slop, masses or force caps.
kernel void rigid_motion_contact_penalties(
    device ManifoldGPU* manifolds [[buffer(0)]],
    device SoftContactGPU* soft [[buffer(1)]],
    device const uint* owners [[buffer(2)]],
    device const float* groupMass [[buffer(3)]],
    device atomic_uint* counters [[buffer(4)]],
    constant SimParams& P [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    uint nm=min(P.maxPairs,atomic_load_explicit(&counters[CTR_PAIRS],memory_order_relaxed));
    uint ns=min(P.maxSoft,atomic_load_explicit(&counters[CTR_SOFT],memory_order_relaxed));
    if(gid<nm) {
        device ManifoldGPU& m=manifolds[gid];
        if(m.header.z==0) return;
        float mass=max(groupMass[owners[m.header.x]],groupMass[owners[m.header.y]]);
        if(mass<=0) return;
        float k=min(PENALTY_MAX_T,mass/(P.dt*P.dt));
        for(uint i=0;i<m.header.z;++i) m.contacts[i].penalty.xyz=max(m.contacts[i].penalty.xyz,float3(k));
    } else if(gid<nm+ns) {
        device SoftContactGPU& c=soft[gid-nm];
        float mass=0;
        for(uint i=0;i<4;++i) if(c.ids[i]!=WORLD_BODY) mass=max(mass,groupMass[owners[c.ids[i]]]);
        if(mass>0) c.penalty.xyz=max(c.penalty.xyz,float3(min(PENALTY_MAX_T,mass/(P.dt*P.dt))));
    }
}
