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
        var text="inline uint implicit_triangle_count(uint id) {switch(id) {\n"
        for i in surfaces.keys.sorted() {text+="case \(i): return \(surfaces[i]!.count/3)u;\n"}
        text+="default:return 0u;}}\ninline float3 implicit_triangle_vertex(uint id,uint v) {switch(id) {\n"
        for i in surfaces.keys.sorted() {
            text+="case \(i): {const float3 p[]={"+surfaces[i]!.map(literal).joined(separator:",")+"};return p[v];}\n"
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
        return text+"\n"
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
