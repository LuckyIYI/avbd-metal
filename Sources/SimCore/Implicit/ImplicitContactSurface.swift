import Foundation
import simd

/// Zero-thickness, two-sided triangles used by implicit contact queries.
/// Vertices are collider-local. They do not become thin convex prisms.
public struct ImplicitContactSurface {
    public var isInfinitePlane: Bool = false
    public var triangles: [SIMD3<Float>]
    public init(triangles: [SIMD3<Float>]) { self.triangles = triangles }
    public static var infinitePlane: Self { var s=Self(triangles:[]);s.isInfinitePlane=true;return s }
    /// Finite plane patch. Extents are explicit; this is not an infinite plane.
    public static func plane(halfExtents: SIMD2<Float>) -> Self {
        let x=halfExtents.x,y=halfExtents.y
        return .init(triangles:[.init(-x,-y,0),.init(x,-y,0),.init(x,y,0),
                               .init(-x,-y,0),.init(x,y,0),.init(-x,y,0)])
    }
}

extension PhysicsScene {
    func implicitSurfaceSource() throws -> String {
        let fields=colliders.indices.filter {colliders[$0].implicitField != nil}
        var surfaces=[Int:[SIMD3<Float>]](), closed=Set<Int>()
        let boxIndices=[0,2,3,0,3,1, 4,5,7,4,7,6, 0,1,5,0,5,4,
                        2,6,7,2,7,3, 0,4,6,0,6,2, 1,3,7,1,7,5]
        for i in colliders.indices {
            let c=colliders[i]
            guard fields.contains(where:{canPotentiallyCollide(colliderA:$0,colliderB:i)}) else {continue}
            if c.implicitField != nil || c.shape == .sphere {continue}
            let vertices:[SIMD3<Float>]
            if let surface=c.implicitContactSurface {
                guard c.convexAssetID == nil, c.convexHullVertices.isEmpty else {throw ImplicitCollisionError.invalidCollider(i)}
                for j in colliders.indices where canPotentiallyCollide(colliderA:i,colliderB:j) {
                    guard colliders[j].implicitField != nil else {throw ImplicitCollisionError.unsupportedPair(i,j)}
                }
                vertices=surface.isInfinitePlane ? Self.unitImplicitPlaneVertices : surface.triangles
            } else if let id=c.convexAssetID {
                guard convexAssets.indices.contains(id) else {throw ImplicitCollisionError.invalidCollider(i)}
                let a=convexAssets[id]
                let center=(a.boundsMin+a.boundsMax)*0.5
                vertices=a.triangles.flatMap {t in [a.vertices[Int(t.x)]-center,a.vertices[Int(t.y)]-center,a.vertices[Int(t.z)]-center]}
                closed.insert(i)
            } else {
                // Inline hulls must be cooked first; never substitute their AABB.
                guard c.shape == .box, c.convexHullVertices.isEmpty else {throw ImplicitCollisionError.invalidCollider(i)}
                let h=c.size*0.5
                var corners=[SIMD3<Float>]()
                for x:Float in [-1,1] {for y:Float in [-1,1] {for z:Float in [-1,1] {corners.append(h*SIMD3(x,y,z))}}}
                vertices=stride(from:0,to:boxIndices.count,by:3).flatMap { [corners[boxIndices[$0]],corners[boxIndices[$0+2]],corners[boxIndices[$0+1]]] };closed.insert(i)
            }
            guard !vertices.isEmpty,vertices.count%3==0,vertices.count<=1524,
                  vertices.allSatisfy({v in v.x.isFinite && v.y.isFinite && v.z.isFinite && (c.implicitContactSurface?.isInfinitePlane == true || (0..<3).allSatisfy {abs(v[$0]) <= c.size[$0]*0.5+1e-6})})
            else {throw ImplicitCollisionError.invalidCollider(i)}
            for j in stride(from:0,to:vertices.count,by:3) {
                guard simd_length_squared(simd_cross(vertices[j+1]-vertices[j],vertices[j+2]-vertices[j]))>1e-16 else {throw ImplicitCollisionError.invalidCollider(i)}
            }
            surfaces[i]=vertices
        }
        func literal(_ p:SIMD3<Float>)->String {"float3(\(p.x)f,\(p.y)f,\(p.z)f)"}
        // Immutable geometry belongs in Metal constant memory, not per-thread arrays.
        var tables=""
        var text="inline uint implicit_triangle_count(uint id) {switch(id) {\n"
        for i in surfaces.keys.sorted() {text+="case \(i): return \(surfaces[i]!.count/3)u;\n"}
        text+="default:return 0u;}}\ninline float3 implicit_triangle_vertex(uint id,uint v) {switch(id) {\n"
        for i in surfaces.keys.sorted() {
            tables += "constant float3 implicit_tri_\(i)[]={"+surfaces[i]!.map(literal).joined(separator:",")+"};\n"
            text += "case \(i):return implicit_tri_\(i)[v];\n"
        }
        text+="default:return float3(NAN);}}\ninline bool implicit_closed_surface(uint id) {switch(id) {\n"
        for i in closed.sorted() {text+="case \(i):return true;\n"}
        text+="default:return false;}}\ninline float3 implicit_interior_seed(uint id) {switch(id) {\n"
        var seeds=[Data:SIMD3<Float>]()
        let seedEncoder=JSONEncoder();seedEncoder.outputFormatting = .sortedKeys
        for i in fields {
            let f=colliders[i].implicitField!, lo=SIMD3(f.bounds[0][0],f.bounds[0][1],f.bounds[0][2]),hi=SIMD3(f.bounds[1][0],f.bounds[1][1],f.bounds[1][2])
            let key=try seedEncoder.encode(f)
            if let seed=seeds[key] {text+="case \(i):return \(literal(seed));\n";continue}
            var best=Float.infinity,seed=SIMD3<Float>.zero
            for x in 0..<8 {for y in 0..<8 {for z in 0..<8 {
                let p=lo+(hi-lo)*SIMD3((Float(x)+0.5)/8,(Float(y)+0.5)/8,(Float(z)+0.5)/8)
                let d=try f.evaluate(p).distance
                if d<best {best=d;seed=p}
            }}}
            guard best<0 else {throw ImplicitCollisionError.invalidCollider(i)}
            seeds[key]=seed
            text+="case \(i):return \(literal(seed));\n"
        }
        text += "default:return float3(NAN);}}\ninline bool implicit_infinite_plane(uint id) {switch(id) {\n"
        for i in surfaces.keys.sorted() where colliders[i].implicitContactSurface?.isInfinitePlane == true {text += "case \(i):return true;\n"}
        text += "default:return false;}}\n"
        for (name,row) in [("implicit_bounds_min",0),("implicit_bounds_max",1)] {
            text += "inline float3 \(name)(uint id){switch(id){\n"
            for i in fields {let b=colliders[i].implicitField!.bounds[row];text += "case \(i):return \(literal(SIMD3(b[0],b[1],b[2])));\n"}
            text += "default:return float3(NAN);}}\n"
        }
        let pairs=implicitPlanePairs()
        text += "inline uint implicit_global_pair_count(){return \(pairs.count)u;}\ninline uint2 implicit_global_pair(uint i){"
        if pairs.isEmpty {text += "return uint2(0);}"}
        else {text += "const uint2 p[]={"+pairs.map {"uint2(\($0.x),\($0.y))"}.joined(separator:",")+"};return p[i];}"}
        text += "\ninline bool implicit_plane_acceleration(){return \(implicitPlaneAcceleration ? "true" : "false");}\n"
        text += "inline float3 implicit_native_box_half(uint id){switch(id){\n"
        for i in surfaces.keys.sorted() where colliders[i].implicitContactSurface == nil && colliders[i].convexAssetID == nil {
            text += "case \(i):return \(literal(colliders[i].size*0.5));\n"
        }
        text += "default:return float3(0);}}\ninline float2 implicit_rectangle_half(uint id){switch(id){\n"
        for i in surfaces.keys.sorted() {
            if let surface=colliders[i].implicitContactSurface, !surface.isInfinitePlane {
                let h=SIMD2(colliders[i].size.x,colliders[i].size.y)*0.5
                if surface.triangles == ImplicitContactSurface.plane(halfExtents:h).triangles {
                    text += "case \(i):return float2(\(h.x)f,\(h.y)f);\n"
                }
            }
        }
        text += "default:return float2(0);}}\n"
        // A convex hull is the intersection of its unique outward face
        // halfspaces. Merge coplanar triangulation without approximating edges.
        var facePlanes=[Int:[SIMD4<Float>]]()
        for i in closed.sorted() where colliders[i].convexAssetID != nil {
            let vertices=surfaces[i]!
            var planes=[SIMD4<Float>]()
            for j in stride(from:0,to:vertices.count,by:3) {
                let a=vertices[j],n=simd_normalize(simd_cross(vertices[j+1]-a,vertices[j+2]-a)),d=simd_dot(n,a)
                guard vertices.allSatisfy({simd_dot(n,$0)<=d+1e-6}) else {throw ImplicitCollisionError.invalidCollider(i)}
                if !planes.contains(where:{simd_length(SIMD3($0.x,$0.y,$0.z)-n)<1e-6 && abs($0.w-d)<1e-7}) {planes.append(SIMD4(n,d))}
            }
            facePlanes[i]=planes
        }
        text += "inline uint implicit_hull_face_count(uint id){switch(id){\n"
        for i in facePlanes.keys.sorted() {text += "case \(i):return \(facePlanes[i]!.count)u;\n"}
        text += "default:return 0u;}}\ninline float4 implicit_hull_face(uint id,uint f){switch(id){\n"
        for i in facePlanes.keys.sorted() {
            let data=facePlanes[i]!.map {"float4(\($0.x)f,\($0.y)f,\($0.z)f,\($0.w)f)"}.joined(separator:",")
            tables += "constant float4 implicit_faces_\(i)[]={\(data)};\n"
            text += "case \(i):return implicit_faces_\(i)[f];\n"
        }
        text += "default:return float4(NAN);}}\n"
        // Surface witnesses on bounding faces certify the support bound. No
        // sampled convex hull is substituted for the implicit collision shape.
        var witnessCache=[Data:[SIMD3<Float>]](), witnessByID=[Int:[SIMD3<Float>]]()
        for i in fields {
            let f=colliders[i].implicitField!,key=try seedEncoder.encode(f)
            if let found=witnessCache[key] {witnessByID[i]=found;continue}
            let lo=SIMD3(f.bounds[0][0],f.bounds[0][1],f.bounds[0][2]),hi=SIMD3(f.bounds[1][0],f.bounds[1][1],f.bounds[1][2])
            var witnesses=[SIMD3<Float>]()
            for axis in 0..<3 {for side in 0..<2 {
                let u=(axis+1)%3,v=(axis+2)%3
                var candidates=[SIMD3<Float>]()
                for a in 0...8 {for b in 0...8 {
                    var p=lo;p[axis]=side==0 ? lo[axis] : hi[axis]
                    p[u]=lo[u]+(hi[u]-lo[u])*Float(a)/8;p[v]=lo[v]+(hi[v]-lo[v])*Float(b)/8
                    if abs(try f.evaluate(p).distance)<=1e-7 {candidates.append(p)}
                }}
                var selected=[SIMD3<Float>]()
                while !candidates.isEmpty && selected.count<8 {
                    let index=candidates.indices.max {a,b in
                        let da=selected.map {simd_length_squared($0-candidates[a])}.min() ?? simd_length_squared(candidates[a]-(lo+hi)*0.5)
                        let db=selected.map {simd_length_squared($0-candidates[b])}.min() ?? simd_length_squared(candidates[b]-(lo+hi)*0.5)
                        return da<db
                    }!
                    selected.append(candidates.remove(at:index))
                }
                witnesses += selected
            }}
            witnessCache[key]=witnesses;witnessByID[i]=witnesses
        }
        text += "inline uint implicit_support_count(uint id){switch(id){\n"
        for i in fields {text += "case \(i):return \(witnessByID[i]!.count)u;\n"}
        text += "default:return 0u;}}\ninline float3 implicit_support_vertex(uint id,uint v){switch(id){\n"
        for i in fields where !witnessByID[i]!.isEmpty {
            tables += "constant float3 implicit_witness_\(i)[]={"+witnessByID[i]!.map(literal).joined(separator:",")+"};\n"
            text += "case \(i):return implicit_witness_\(i)[v];\n"
        }
        text += "default:return float3(NAN);}}\n"
        // Author-certified regions where the distance field is exactly planar.
        text += "inline float implicit_region_triangle_min(uint id,float3 a,float3 b,float3 c){switch(id){\n"
        for i in fields {
            text += "case \(i):{\n"
            for r in colliders[i].implicitField!.planarRegions ?? [] {
                guard r.normal.count==3,r.bounds.count==2,r.bounds.allSatisfy({$0.count==3}),r.offset.isFinite else {throw ImplicitCollisionError.invalidCollider(i)}
                let n=SIMD3(r.normal[0],r.normal[1],r.normal[2]),lo=SIMD3(r.bounds[0][0],r.bounds[0][1],r.bounds[0][2]),hi=SIMD3(r.bounds[1][0],r.bounds[1][1],r.bounds[1][2])
                guard abs(simd_length(n)-1)<1e-5,(0..<3).allSatisfy({lo[$0].isFinite && hi[$0].isFinite && lo[$0]<hi[$0]}) else {throw ImplicitCollisionError.invalidCollider(i)}
                text += "if(all(min(a,min(b,c))>=\(literal(lo)))&&all(max(a,max(b,c))<=\(literal(hi))))return min(dot(\(literal(n)),a),min(dot(\(literal(n)),b),dot(\(literal(n)),c)))-\(r.offset)f;\n"
            }
            text += "return -INFINITY;}\n"
        }
        text += "default:return -INFINITY;}}\n"
        // A convex empty region is a separation certificate for an entire
        // triangle when every vertex lies inside its margin erosion. Each
        // radial halfspace is convex and 1-Lipschitz (nonnegative radius term).
        text += "inline bool implicit_void_triangle_clear(uint id,float3 a,float3 b,float3 c,float margin){switch(id){\n"
        for i in fields {
            text += "case \(i):{\n"
            for r in colliders[i].implicitField!.radialVoids ?? [] {
                text += "{bool clear=true;\n"
                for p in r.planes {
                    text += "clear=clear && max(\(p[0])f*length(a.xy)+\(p[1])f*a.z,max(\(p[0])f*length(b.xy)+\(p[1])f*b.z,\(p[0])f*length(c.xy)+\(p[1])f*c.z))<\(p[2])f-margin;\n"
                }
                text += "if(clear)return true;}\n"
            }
            text += "return false;}\n"
        }
        text += "default:return false;}}\n"
        return tables+text+"\n"
    }
}

public extension PhysicsScene {
    static var unitImplicitPlaneVertices:[SIMD3<Float>] {ImplicitContactSurface.plane(halfExtents:SIMD2(1,1)).triangles}
    func implicitPlanePairs()->[SIMD2<UInt32>] {
        let fields=colliders.indices.filter {colliders[$0].implicitField != nil}
        let planes=colliders.indices.filter {colliders[$0].implicitContactSurface?.isInfinitePlane == true}
        return fields.flatMap {f in planes.compactMap {p in canPotentiallyCollide(colliderA:f,colliderB:p) ? SIMD2(UInt32(min(f,p)),UInt32(max(f,p))) : nil}}
    }
}
