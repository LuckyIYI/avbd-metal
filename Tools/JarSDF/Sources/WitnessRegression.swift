import Foundation
import GPUSim
import simd
private struct Capture:Decodable {
 struct Shape:Decodable {let vertices:[[Float]];let center_kind:[Float];let rotation:[Float];let dimensions:[Float]}
 let shapes:[Shape]
}
func runCapturedWitness(_ args:[String],_ report:inout [String:Any]) throws {
 let capture=try JSONDecoder().decode(Capture.self,from:Data(contentsOf:URL(fileURLWithPath:args[1])))
 var scene=PhysicsScene(name:"captured convex witness");scene.settings.gravity=0;scene.settings.dt=1/240;scene.settings.iterations=4
 for (i,s) in capture.shapes.enumerated() {
  let body=scene.addBody(size:F3(s.dimensions[0],s.dimensions[1],s.dimensions[2]),density:i==0 ? 0:1000,friction:0,position:F3(s.center_kind[0],s.center_kind[1],s.center_kind[2]),rotation:Quat(vector:SIMD4(s.rotation[0],s.rotation[1],s.rotation[2],s.rotation[3])),collisionEnabled:false)
  _=scene.addCollider(body:body,size:F3(s.dimensions[0],s.dimensions[1],s.dimensions[2]),shape:.box,convexHullVertices:s.vertices.map{F3($0[0],$0[1],$0[2])},convexAssetID:nil)
 }
 let solver=try GPUSolver(scene:scene);var contacts=0
 for _ in 0..<16 {try solver.submitStep();try solver.synchronize();contacts=max(contacts,solver.activeRigidContactCounts().reduce(0){$0+$1.contacts})}
 report["passed"]=contacts>0;report["max_contacts"]=contacts;report["steps"]=16
}
