import Foundation
import simd

/// Portable scalar expression DAG. Coordinates and distance are in metres.
/// Negative means inside. CSG fields are not necessarily metric distances.
/// This query API does not automatically enable rigid contact pairs.
public struct ImplicitField: Codable {
    public struct Node: Codable {
        public var op: String
        public var args: [Int]
        public var value: Float
    }
    public struct Native: Codable {
        public var shape: String
        public var size: [Float]
    }
    public var nodes: [Node]
    public var output: Int
    public var gradient: [Int]?
    public var autodiff: Bool
    public var bounds: [[Float]]
    public var exactDistance: Bool
    public var epsilon: Float
    public var native: Native?
    public struct PlanarRegion: Codable {
        public var normal: [Float]
        public var offset: Float
        public var bounds: [[Float]]
    }
    public var planarRegions: [PlanarRegion]? = nil
    public struct RadialVoid: Codable { public var planes: [[Float]] }
    public var radialVoids: [RadialVoid]? = nil

    public enum Failure: Error { case invalidProgram, nonFinite, undefinedNormal }

    public func validate() throws {
        let arities = ["constant":0,"x":0,"y":0,"z":0,"add":2,"mul":2,
                       "div":2,"sqrt":1,"sign":1,"abs":1,"sin":1,"cos":1,"min":2,"max":2]
        guard !nodes.isEmpty, nodes.count <= 512, nodes.indices.contains(output),
              bounds.count == 2, bounds.allSatisfy({$0.count == 3 && $0.allSatisfy(\.isFinite)}),
              (0..<3).allSatisfy({bounds[0][$0] < bounds[1][$0]}),
              epsilon.isFinite, epsilon > 0 else { throw Failure.invalidProgram }
        if let gradient {
            guard gradient.count == 3, gradient.allSatisfy({nodes.indices.contains($0)})
            else { throw Failure.invalidProgram }
        }
        for (i,n) in nodes.enumerated() {
            guard let arity = arities[n.op], n.args.count == arity, n.value.isFinite,
                  n.args.allSatisfy({$0 >= 0 && $0 < i}) else { throw Failure.invalidProgram }
        }
        for r in planarRegions ?? [] {
            guard r.normal.count == 3, r.normal.allSatisfy(\.isFinite), r.offset.isFinite,
                r.bounds.count == 2, r.bounds.allSatisfy({$0.count == 3 && $0.allSatisfy(\.isFinite)}),
                (0..<3).allSatisfy({r.bounds[0][$0] < r.bounds[1][$0]}),
                abs(simd_length(SIMD3(r.normal[0],r.normal[1],r.normal[2]))-1)<1e-5
            else { throw Failure.invalidProgram }
        }
        for region in radialVoids ?? [] {
            guard !region.planes.isEmpty, region.planes.count <= 64,
                region.planes.allSatisfy({p in p.count == 3 && p.allSatisfy(\.isFinite)
                    && p[0]>=0 && abs(p[0]*p[0]+p[1]*p[1]-1)<1e-5})
            else {throw Failure.invalidProgram}
        }
        // Native is only an authoring hint. Importers must verify the formula
        // before selecting a native collider; never trust arbitrary JSON tags.
    }

    private func run(_ p: SIMD3<Float>) throws -> ([Float], [SIMD3<Float>]) {
        var v = [Float](); var g = [SIMD3<Float>]()
        v.reserveCapacity(nodes.count); g.reserveCapacity(nodes.count)
        for n in nodes {
            let a: Float = n.args.isEmpty ? 0 : v[n.args[0]]
            let da = n.args.isEmpty ? .zero : g[n.args[0]]
            let b: Float = n.args.count < 2 ? 0 : v[n.args[1]]
            let db = n.args.count < 2 ? .zero : g[n.args[1]]
            var x: Float = 0; var dx = SIMD3<Float>.zero
            switch n.op {
            case "constant": x = n.value
            case "x": x = p.x; dx.x = 1
            case "y": x = p.y; dx.y = 1
            case "z": x = p.z; dx.z = 1
            case "add": x = a+b; dx = da+db
            case "mul": x = a*b; dx = da*b+db*a
            case "div": x = a/b; dx = (da*b-db*a)/(b*b)
            case "sqrt": x = sqrt(a); dx = x > 0 ? da/(2*x) : .zero
            case "sign": x = a == 0 ? 0 : (a > 0 ? 1 : -1); dx = .zero
            case "abs": x = abs(a); dx = a == 0 ? .zero : (a > 0 ? da : -da)
            case "sin": x = sin(a); dx = da*cos(a)
            case "cos": x = cos(a); dx = -da*sin(a)
            case "min": x = min(a,b); dx = a <= b ? da : db
            case "max": x = max(a,b); dx = a >= b ? da : db
            default: throw Failure.invalidProgram
            }
            guard x.isFinite, dx.x.isFinite, dx.y.isFinite, dx.z.isFinite else { throw Failure.nonFinite }
            v.append(x); g.append(dx)
        }
        return (v,g)
    }

    /// Analytic chain rule by default, explicit supplied derivatives when
    /// present, central differences only when the author disables autodiff.
    public func evaluate(_ p: SIMD3<Float>) throws -> (distance: Float, gradient: SIMD3<Float>) {
        try validate()
        guard p.x.isFinite, p.y.isFinite, p.z.isFinite else { throw Failure.nonFinite }
        let (v,g) = try run(p)
        var derivative = g[output]
        if let gradient {
            derivative = SIMD3(v[gradient[0]],v[gradient[1]],v[gradient[2]])
        } else if !autodiff || (abs(v[output])<=epsilon && simd_length_squared(derivative)<1e-12) {
            for i in 0..<3 {
                var step = SIMD3<Float>.zero; step[i] = epsilon
                let plus = try run(p+step).0[output]
                let minus = try run(p-step).0[output]
                derivative[i] = (plus-minus)/(2*epsilon)
            }
        }
        return (v[output],derivative)
    }

    /// Exact sphere witness only for a metric SDF with a defined gradient.
    /// General CSG/custom fields require iterative closest-surface queries.
    public func sphereWitness(center: SIMD3<Float>, radius: Float) throws
        -> (separation: Float, surface: SIMD3<Float>, normal: SIMD3<Float>) {
        guard exactDistance, radius.isFinite, radius > 0 else { throw Failure.invalidProgram }
        let q = try evaluate(center)
        let l = simd_length(q.gradient)
        guard l > 1e-6 else { throw Failure.undefinedNormal }
        let n = q.gradient/l
        return (q.distance-radius, center-q.distance*n, n)
    }
}
