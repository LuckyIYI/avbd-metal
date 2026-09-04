import XCTest
import Metal
import simd
@testable import SimCore
@testable import PhysicsAVBD

final class RigidMeshMaterialTests:XCTestCase {
    func testMaterialsSurviveReplicationAndGPUUploadWithoutGrowingABI() throws {
        var scene=PhysicsScene(name:"material upload")
        let body=scene.addBody(size:F3(repeating:0.1),density:0,friction:0.5,position:.zero)
        let mesh=SurfaceMesh(vertices:[F3(0,0,0),F3(1,0,0),F3(0,1,0)],normals:Array(repeating:F3(0,0,1),count:3),triangles:[(0,1,2)])
        scene.addRigidMesh(SceneRigidMesh(body:body,mesh:mesh,color:F3(0.6,0.7,0.8),roughness:0.23,metallic:0.91))
        let replicated=scene.replicated(count:2,spacing:F3(2,0,0),includeVisuals:true).scene
        XCTAssertEqual(replicated.rigidMeshes.count,2)
        XCTAssertTrue(replicated.rigidMeshes.allSatisfy{$0.roughness==0.23 && $0.metallic==0.91})
        let solver=try GPUSolver(scene:scene)
        let surface=try XCTUnwrap(solver.renderRigidMeshSurface)
        let vertices=surface.vertices.contents().bindMemory(to:RigidMeshVertexGPU.self,capacity:surface.vertexCount)
        XCTAssertEqual(vertices[0].normal.w,0.23)
        XCTAssertEqual(vertices[0].color.w,0.91)
        XCTAssertEqual(MemoryLayout<RigidMeshVertexGPU>.stride,48)
    }
}
