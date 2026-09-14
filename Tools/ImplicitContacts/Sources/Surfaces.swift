import Foundation
import GPUSim
import simd
extension Main {
 static func runSurface(field:ImplicitField,mode inputMode:String,count:Int,out:String) throws {
  let mode=inputMode.replacingOccurrences(of:"-generic",with:"")
  var scene=PhysicsScene(name:mode);scene.settings.dt=1.0/240;scene.settings.iterations=4;scene.settings.collisionMargin=0.0001;scene.settings.gravity = -9.81
  scene.implicitPlaneAcceleration = !inputMode.hasSuffix("-generic")
  let ramp=mode.hasSuffix("-ramp"),slope=Quat(angle:0.15,axis:F3(0,1,0))
  var moving=[Int]()
  for i in 0..<count {
   let offset=F3(Float(i%8),Float(i/8),0),group=UInt32(i+1)
   let hole=mode=="surface-hole",contain=mode=="surface-contain"
   let far=mode=="surface-infinite-far" ? F3(4,2,0) : mode=="surface-edge-miss" ? F3(0.7,0,0) : .zero
   let ring=scene.addBody(size:F3(0.2,0.2,0.04),density:hole || contain ? 0 : 1000,friction:0.5,position:offset+far+F3(0,0,hole || contain ? 0 : 0.1),collisionEnabled:false)
   if mode=="surface-tilted" {scene.bodies[ring].rotation=Quat(angle:0.25,axis:F3(0,1,0))}
   if ramp {scene.bodies[ring].rotation=slope;scene.bodies[ring].position=offset+slope.act(F3(0,0,0.1))}
   let rc=scene.addCollider(body:ring,size:F3(0.2,0.2,0.04),shape:.box,convexAssetID:nil,collisionGroup:group)
   if mode=="surface-boxes-baseline" {
    scene.colliders[rc].collisionEnabled=false
    for j in 0..<32 {
     let a=Float(j)*2*Float.pi/32
     _=scene.addCollider(body:ring,size:F3(0.06,0.2*tan(Float.pi/32),0.04),localPosition:F3(0.07*cos(a),0.07*sin(a),0),localRotation:Quat(angle:a,axis:F3(0,0,1)),shape:.box,convexAssetID:nil,collisionGroup:group)
    }
   } else {scene.colliders[rc].implicitField=field}
   let other=scene.addBody(size:hole ? F3(repeating:0.02) : F3(0.8,0.8,0.05),density:hole || contain ? 1000 : 0,friction:0.5,position:offset+F3(0,0,hole ? 0.1 : contain ? 0 : -0.025),collisionEnabled:false)
   if ramp {scene.bodies[other].rotation=slope;scene.bodies[other].position=offset+slope.act(F3(0,0,-0.025))}
   let c=scene.addCollider(body:other,size:hole ? F3(repeating:0.02) : contain ? F3(repeating:0.8) : F3(0.8,0.8,0.05),shape:.box,convexAssetID:nil,collisionGroup:group)
   if mode.hasPrefix("surface-hull") {
    scene.colliders[c].collisionEnabled=false
    let asset=try JSONDecoder().decode(ConvexHullAsset.self,from:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[1]).deletingLastPathComponent().appendingPathComponent(mode.contains("octagon") ? "octagon-hull.json" : "floor-hull.json")))
    _=scene.addConvexCollider(body:other,asset:asset,collisionGroup:group)
   }
   if mode.hasPrefix("surface-infinite") {scene.colliders[c].localPosition.z=0.025;scene.colliders[c].implicitContactSurface = .infinitePlane}
   if mode=="surface-plane" || mode=="surface-edge-miss" {scene.colliders[c].localPosition.z=0.025;scene.colliders[c].implicitContactSurface = .plane(halfExtents:SIMD2(0.4,0.4))}
   if mode=="surface-triangle" {scene.colliders[c].localPosition.z=0.025;scene.colliders[c].implicitContactSurface = .init(triangles:[F3(-0.4,-0.4,0),F3(0.4,-0.4,0),F3(0,0.4,0)])}
   moving.append(hole ? other : ring)
  }
  let start=Date();let solver=try GPUSolver(scene:scene,maxPairsPerBody:64);let initialization=Date().timeIntervalSince(start)
  let clock=Date();var trace=[[[Float]]]();var minimumZ=Float.infinity;var maximumPenetration:Float=0
  do { for step in 0..<240 {try solver.submitStep();try solver.synchronize();let current=solver.bodyStates(moving);minimumZ=min(minimumZ,current.map{$0.position.z}.min()!);
   if mode != "surface-hole" && mode != "surface-edge-miss" {
    for (j,b) in current.enumerated() {let worldN=ramp ? slope.act(F3(0,0,1)) : F3(0,0,1);let n=b.rotation.inverse.act(worldN);let support=0.1*sqrt(n.x*n.x+n.y*n.y)+0.02*abs(n.z);let origin=F3(Float(j%8),Float(j/8),0);maximumPenetration=max(maximumPenetration,support-dot(worldN,b.position-origin))}
   }
   if step%12==0 {trace.append(solver.bodyStates(moving).map{[$0.position.x,$0.position.y,$0.position.z]})}}
  } catch {Main.failureEvidence=solver.implicitFailureEvidence() ?? [:];throw error}
  let seconds=Date().timeIntervalSince(clock),states=solver.bodyStates(moving)
  let z=states.map{$0.position.z}
  let planeGaps=states.enumerated().map {j,b -> Float in
   let worldN=ramp ? slope.act(F3(0,0,1)) : F3(0,0,1),n=b.rotation.inverse.act(worldN)
   return dot(worldN,b.position-F3(Float(j%8),Float(j/8),0))-(0.1*sqrt(n.x*n.x+n.y*n.y)+0.02*abs(n.z))
  }
  let freeFall=mode=="surface-hole" || mode=="surface-edge-miss"
  let passed=z.allSatisfy{$0.isFinite} && (freeFall ? z.allSatisfy{$0 < -1} : planeGaps.allSatisfy{abs($0)<0.001} && maximumPenetration<0.001)
  let report:[String:Any]=["final_plane_gaps":planeGaps,"maximum_plane_penetration_m":maximumPenetration,"strict_impact_pass":maximumPenetration<0.001,"minimum_center_z":minimumZ,"mode":inputMode,"replicas":count,"passed":passed,"final_z":z,"step_seconds":seconds,"steps_per_second":240/seconds,"initialization_seconds":initialization,"trace":trace]
  try JSONSerialization.data(withJSONObject:report,options:.prettyPrinted).write(to:URL(fileURLWithPath:out));print("\(mode) passed=\(passed); \(240/seconds) steps/s; z=\(z)")
  if !passed {exit(2)}
 }
}
