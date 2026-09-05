import simd

extension RayTracingScene {
    /// Uses the raster tessellation and dimensions, so ray visibility matches
    /// the rendered surface instead of tracing the collision proxy.
    static func primitiveVertices(_ instance: GPUSimRenderInstance) -> [Vertex] {
        let kind = Int(instance.color.w)
        func vertex(_ p: SIMD3<Float>, _ n: SIMD3<Float>) -> Vertex {
            Vertex(position: SIMD4(p, 1), normal: SIMD4(n, 0.45), albedo: SIMD4(1,1,1,0))
        }
        let corners = [(0,0), (1,0), (1,1), (0,0), (1,1), (0,1)]
        if kind == 0 {
            let axes = [SIMD3<Float>(1,0,0), SIMD3(-1,0,0), SIMD3(0,1,0), SIMD3(0,-1,0), SIMD3(0,0,1), SIMD3(0,0,-1)]
            return axes.flatMap { n -> [Vertex] in
                let t = abs(n.x) > 0.5 ? SIMD3<Float>(0,1,0) : SIMD3<Float>(1,0,0)
                let b = cross(n, t)
                return corners.map { x,y in vertex(n*0.5 + t*(Float(x)-0.5) + b*(Float(y)-0.5), n) }
            }
        }
        let slices = kind == 1 ? 18 : (kind == 2 ? 12 : 16)
        let stacks = kind == 1 ? 12 : (kind == 2 ? 24 : 13)
        return (0..<(slices*stacks)).flatMap { quad -> [Vertex] in
            let stack = quad / slices, slice = quad % slices
            return corners.map { x,y in
                if kind == 1 {
                    let phi = Float(stack+y) / 12 * Float.pi
                    let theta = Float(slice+x) / 18 * 2 * Float.pi
                    let n = SIMD3(sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi))
                    return vertex(n*0.5, n)
                }
                if kind == 2 {
                    let u = Float(stack+x) / 24 * 2 * Float.pi
                    let v = Float(slice+y) / 12 * 2 * Float.pi
                    let n = SIMD3(cos(u)*cos(v), sin(u)*cos(v), sin(v))
                    return vertex(SIMD3(instance.parameters.x*cos(u), instance.parameters.x*sin(u), 0) + n*instance.parameters.y, n)
                }
                let s = stack+y
                let phi = s <= 6 ? -Float.pi/2 + Float(s)/6 * Float.pi/2 : Float(s-7)/6 * Float.pi/2
                let z = (s <= 6 ? -0.5 : Float(0.5)) * instance.parameters.x
                let u = Float(slice+x)/16 * 2 * Float.pi
                let n = SIMD3(cos(phi)*cos(u), cos(phi)*sin(u), sin(phi))
                return vertex(n*instance.parameters.y + SIMD3(0,0,z), n)
            }
        }
    }
}
