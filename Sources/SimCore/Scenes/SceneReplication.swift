import simd

/// Index mapping for one independent copy in a replicated GPU scene.
public struct PhysicsSceneReplica: Sendable, Equatable {
    public var index: Int
    public var worldOffset: F3
    public var bodyBase: Int
    public var colliderBase: Int
    public var jointBase: Int
    public var springBase: Int

    public func body(_ source: Int) -> Int { bodyBase + source }
    public func collider(_ source: Int) -> Int { colliderBase + source }
    public func joint(_ source: Int) -> Int { jointBase + source }
    public func spring(_ source: Int) -> Int { springBase + source }
}

public extension PhysicsScene {
    /// Replicate a complete authored scene into one Metal solver submission.
    ///
    /// Copies retain their original collision-domain topology and are spaced
    /// far enough apart that broad phase cannot pair them. This is required
    /// for scenes such as MicroDuck where group 0 is shared ground, group 1 is
    /// coarse robot geometry, and group 2 is the mouth/soft payload domain;
    /// flattening those groups into one replica id would silently change the
    /// grasp model. Headless optimization may omit visual meshes while keeping
    /// every physical body, collider, joint, tet and triangle.
    func replicated(
        count: Int, spacing: F3,
        columns: Int = 1,
        includeVisuals: Bool = false
    ) -> (scene: PhysicsScene, replicas: [PhysicsSceneReplica]) {
        precondition(count > 0, "scene replication count must be positive")
        precondition(columns > 0, "scene replication columns must be positive")
        precondition(spacing.x.isFinite && spacing.y.isFinite
            && spacing.z.isFinite, "scene replication spacing must be finite")

        let source = self
        var output = PhysicsScene(name: source.name + "-batch-\(count)")
        output.settings = source.settings
        let assetMap = source.convexAssets.map {
            output.registerConvexAsset($0)
        }
        var replicas: [PhysicsSceneReplica] = []
        replicas.reserveCapacity(count)

        for replicaIndex in 0..<count {
            // A 2-D layout keeps coordinates O(sqrt(worlds)) instead of
            // pushing the final replica hundreds of metres from the origin.
            // That preserves Float contact precision for large policy
            // batches while the default one-column layout remains exactly
            // source-compatible with the original API.
            let column = replicaIndex % columns
            let row = replicaIndex / columns
            let offset = F3(
                spacing.x * Float(column),
                spacing.y * Float(row),
                spacing.z * Float(replicaIndex))
            let mapping = PhysicsSceneReplica(
                index: replicaIndex, worldOffset: offset,
                bodyBase: output.bodies.count,
                colliderBase: output.colliders.count,
                jointBase: output.joints.count,
                springBase: output.springs.count)
            replicas.append(mapping)

            for var body in source.bodies {
                body.position += offset
                output.bodies.append(body)
            }
            for var collider in source.colliders {
                collider.body = mapping.body(collider.body)
                if let sourceAsset = collider.convexAssetID {
                    precondition(assetMap.indices.contains(sourceAsset),
                        "collider references an invalid convex asset")
                    collider.convexAssetID = assetMap[sourceAsset]
                }
                output.colliders.append(collider)
            }
            for var joint in source.joints {
                if joint.bodyA >= 0 {
                    joint.bodyA = mapping.body(joint.bodyA)
                } else {
                    joint.rA += offset
                }
                joint.bodyB = mapping.body(joint.bodyB)
                output.joints.append(joint)
            }
            for var spring in source.springs {
                if spring.bodyA >= 0 {
                    spring.bodyA = mapping.body(spring.bodyA)
                } else {
                    spring.rA += offset
                }
                if spring.bodyB >= 0 {
                    spring.bodyB = mapping.body(spring.bodyB)
                } else {
                    spring.rB += offset
                }
                output.springs.append(spring)
            }
            for tet in source.tets {
                output.tets.append(SceneTet(
                    ids: (
                        mapping.body(tet.ids.0), mapping.body(tet.ids.1),
                        mapping.body(tet.ids.2), mapping.body(tet.ids.3)),
                    mu: tet.mu, lambda: tet.lambda,
                    selfCollisionEnabled: tet.selfCollisionEnabled))
            }
            for triangle in source.tris {
                output.tris.append(SceneTri(
                    ids: (
                        mapping.body(triangle.ids.0),
                        mapping.body(triangle.ids.1),
                        mapping.body(triangle.ids.2)),
                    mu: triangle.mu, lambda: triangle.lambda,
                    bend: triangle.bend))
            }
            for spinner in source.spinners {
                output.spinners.append(SceneSpinner(
                    body: mapping.body(spinner.body),
                    axis: spinner.axis, omega: spinner.omega))
            }
            for exclusion in source.collisionExclusions {
                output.collisionExclusions.append(SceneCollisionExclusion(
                    bodyA: mapping.body(exclusion.bodyA),
                    bodyB: mapping.body(exclusion.bodyB)))
            }

            guard includeVisuals else { continue }
            for mesh in source.rigidMeshes {
                output.rigidMeshes.append(SceneRigidMesh(
                    body: mapping.body(mesh.body),
                    mesh: SurfaceMesh(
                        vertices: mesh.vertices, normals: mesh.normals,
                        triangles: mesh.triangles),
                    localPosition: mesh.localPosition,
                    localRotation: mesh.localRotation, color: mesh.color))
            }
            for mesh in source.skinnedMeshes {
                let vertices = mesh.vertices.map { vertex in
                    SceneSkinnedVertex(
                        ids: (
                            mapping.body(vertex.ids.0),
                            mapping.body(vertex.ids.1),
                            mapping.body(vertex.ids.2),
                            mapping.body(vertex.ids.3)),
                        weights: vertex.weights,
                        restNormal: vertex.restNormal,
                        restInv0: vertex.restInv0,
                        restInv1: vertex.restInv1,
                        restInv2: vertex.restInv2)
                }
                output.skinnedMeshes.append(SceneSkinnedMesh(
                    vertices: vertices, triangles: mesh.triangles,
                    bodyIDs: mesh.bodyIDs.map(mapping.body)))
            }
        }
        return (output, replicas)
    }
}
