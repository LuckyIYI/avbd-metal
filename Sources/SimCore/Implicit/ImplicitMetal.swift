import Foundation
import simd

public extension ImplicitField {
    /// Compiled straight-line arithmetic. No dynamic dispatch inside evaluation.
    func metalSource(name: String) throws -> String {
        try validate()
        guard name.range(of: "^[a-zA-Z_][a-zA-Z_0-9]*$", options:.regularExpression) != nil else { throw Failure.invalidProgram }
        var lines = ["inline float4 \(name)_analytic(float3 p) {"]
        for (i,n) in nodes.enumerated() {
            let a = n.args.isEmpty ? "0.0f" : "v\(n.args[0])"
            let da = n.args.isEmpty ? "float3(0)" : "g\(n.args[0])"
            let b = n.args.count < 2 ? "0.0f" : "v\(n.args[1])"
            let db = n.args.count < 2 ? "float3(0)" : "g\(n.args[1])"
            let v: String, g: String
            switch n.op {
            case "constant": v = "\(n.value)f"; g = "float3(0)"
            case "x": v = "p.x"; g = "float3(1,0,0)"
            case "y": v = "p.y"; g = "float3(0,1,0)"
            case "z": v = "p.z"; g = "float3(0,0,1)"
            case "add": v = "\(a)+\(b)"; g = "\(da)+\(db)"
            case "mul": v = "\(a)*\(b)"; g = "\(da)*\(b)+\(db)*\(a)"
            case "div": v = "\(a)/\(b)"; g = "(\(da)*\(b)-\(db)*\(a))/(\(b)*\(b))"
            case "sqrt": v = "sqrt(\(a))"; g = "v\(i)>0 ? \(da)/(2*v\(i)) : float3(0)"
            case "sign": v = "sign(\(a))"; g = "float3(0)"
            case "abs": v = "abs(\(a))"; g = "\(da)*sign(\(a))"
            case "sin": v = "sin(\(a))"; g = "\(da)*cos(\(a))"
            case "cos": v = "cos(\(a))"; g = "-\(da)*sin(\(a))"
            case "min", "max":
                v = "\(n.op)(\(a),\(b))"
                g = "\(a)\(n.op == "min" ? "<=" : ">=")\(b) ? \(da) : \(db)"
            default: throw Failure.invalidProgram
            }
            lines += ["float v\(i) = \(v);", "float3 g\(i) = \(g);"]
        }
        let grad = gradient.map { "float3(" + $0.map {"v\($0)"}.joined(separator:",") + ")" } ?? "g\(output)"
        lines += ["return float4(\(grad),v\(output));", "}", "inline float4 \(name)(float3 p) {", "float4 q = \(name)_analytic(p);"]
        if gradient == nil {
            if autodiff {lines.append("if(abs(q.w)<=\(epsilon)f && dot(q.xyz,q.xyz)<1e-12f){")}
            lines.append("const float h = \(epsilon)f;")
            for i in 0..<3 {
                var xyz = ["0","0","0"]; xyz[i] = "h"
                let step = "float3(" + xyz.joined(separator:",") + ")"
                lines.append("q[\(i)] = (\(name)_analytic(p+\(step)).w-\(name)_analytic(p-\(step)).w)/(2*h);")
            }
            if autodiff {lines.append("}")}
        }
        return (lines + ["return q;", "}"]).joined(separator:"\n")
    }
}

public enum ImplicitCollisionError: Error { case unsupportedPair(Int,Int), invalidCollider(Int) }
public extension PhysicsScene {
    func implicitCollisionPreamble() throws -> String {
        let ids = colliders.indices.filter { colliders[$0].implicitField != nil }
        if ids.isEmpty {
            if colliders.contains(where: {$0.implicitContactSurface != nil}) {throw ImplicitCollisionError.invalidCollider(0)}
            return ""
        }
        var source = "#define AVBD_IMPLICIT 1\n"
        var shared = [Data:Int](), names = [Int:Int]()
        let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
        for i in ids {
            let c = colliders[i], f = c.implicitField!
            try f.validate()
            // Bounds are centred on the collider frame. Arbitrary offsets go
            // in localPosition; don't under-bound an asymmetric field.
            guard f.exactDistance, c.shape == .box, c.convexAssetID == nil,
                  c.convexHullVertices.isEmpty, !c.usesWorldSpaceRoundAnchor,
                  (0..<3).allSatisfy({ max(abs(f.bounds[0][$0]),abs(f.bounds[1][$0])) <= c.size[$0]*0.5+1e-7 })
            else { throw ImplicitCollisionError.invalidCollider(i) }
            for j in colliders.indices where canPotentiallyCollide(colliderA:i,colliderB:j) {
                let other = colliders[j]
                guard other.implicitField == nil, (other.shape == .sphere || other.shape == .box || other.convexAssetID != nil || other.implicitContactSurface != nil)
                else { throw ImplicitCollisionError.unsupportedPair(i,j) }
            }
            let key = try encoder.encode(f)
            if let previous = shared[key] { names[i] = previous }
            else { shared[key] = i; names[i] = i; source += try f.metalSource(name:"implicit_\(i)") + "\n" }
        }
        source += "inline float4 implicit_query(uint id,float3 p) { switch(id) {\n"
        for i in ids {
            source += "case \(i): {\n"
            for r in colliders[i].implicitField!.planarRegions ?? [] {
                func lit(_ v:[Float])->String { "float3("+v.map{"\($0)f"}.joined(separator:",")+")" }
                source += "if(all(p>=\(lit(r.bounds[0])))&&all(p<=\(lit(r.bounds[1]))))return float4(\(lit(r.normal)),dot(\(lit(r.normal)),p)-\(r.offset)f);\n"
            }
            source += "return implicit_\(names[i]!)(p); }\n"
        }
        return source + "default: return float4(NAN); }}\n" + (try implicitSurfaceSource())
    }
}
