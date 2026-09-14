import Foundation
import Darwin
import GPUSim
import simd
struct Collider: Decodable {let asset:ConvexHullAsset;let center:[Float]}
struct Hardware: Decodable {let name:String;let colliders:[Collider];let hulls:[[[Float]]];let mass:Float;let inertia:[Float]}
struct Fixture: Decodable {let field:ImplicitField;let convex_hulls:[[[Float]]];let cooked_jar:[ConvexHullAsset];let hardware:[Hardware];let parameters:[String:Double]}
func v(_ a:[Float])->F3 {F3(a[0],a[1],a[2])}
@main struct Main {
 static func main() {
  let args=CommandLine.arguments
  guard args.count >= 4, ["sdf", "convex", "witness"].contains(args[2]) else {
   print("Usage: jar-sdf-pilot FIXTURE sdf|convex|witness REPORT [SEED] [STEPS] [COUNT]")
   exit(2)
  }
  var report:[String:Any]=["passed":false,"mode":args[2]]
  do {try run(args,&report)} catch {report["error"]=String(describing:error)}
  if let data=try? JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]) {try? data.write(to:URL(fileURLWithPath:args[3]))}
  print("\(args[2]): \(report["passed"] ?? false) \(report["error"] ?? "")")
  if report["passed"] as? Bool != true { exit(1) }
 }
 static func run(_ args:[String],_ report:inout [String:Any]) throws {
  if args[2]=="witness" {try runCapturedWitness(args,&report);return}
  let f=try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:URL(fileURLWithPath:args[1])))
  let sdf=args[2]=="sdf", seed=args.count>4 ? Int(args[4])! : 0, steps=args.count>5 ? Int(args[5])! : 1440
  let count=args.count>6 ? Int(args[6])! : 12
  let spheres=args.count>7 && args[7]=="spheres"
  var scene=PhysicsScene(name:"jar matched pilot");let dt:Float=1/Float(ProcessInfo.processInfo.environment["JAR_HZ"] ?? "240")!
  let iterations=Int(ProcessInfo.processInfo.environment["JAR_ITERATIONS"] ?? "4")!
  scene.settings.dt=dt;scene.settings.iterations=iterations;scene.settings.gravity = -9.81;scene.settings.collisionMargin=0.0001
  let radius=Float(f.parameters["radius"]!),height=Float(f.parameters["height"]!)
  _=scene.addBody(size:F3(1.6,1.2,0.04),density:0,friction:0.45,position:F3(0,0,-0.022))
  let jar=scene.addBody(size:F3(radius*2,radius*2,height),density:600,friction:0.4,position:.zero,collisionEnabled:false)
  scene.bodies[jar].gravityScale=0
  let drivePoints=[F3.zero,F3(0.07,0,0),F3(0,0.07,0)]
  for point in drivePoints {scene.addJoint(SceneJoint(bodyA:-1,bodyB:jar,rA:point,rB:point,stiffnessLin:.infinity))}
  if sdf {let c=scene.addCollider(body:jar,size:F3(radius*2,radius*2,height*2),shape:.box,convexAssetID:nil);scene.colliders[c].implicitField=f.field}
  else {for hull in f.cooked_jar {_=scene.addConvexCollider(body:jar,asset:hull)}}
  var ids=[Int]();var witnesses=[[F3]]()
  for i in 0..<count {
   let a=f.hardware[i%f.hardware.count];let angle=Float(seed)*0.23+Float(i)*0.71
   let slot=Float(i%3)*2*Float.pi/3
   let pos=F3(0.034*cos(slot),0.034*sin(slot),0.03+Float(i/3)*0.028)
   let body=scene.addBody(size:F3(0.05,0.02,0.02),density:1000,friction:0.4,position:pos,rotation:Quat(angle:angle,axis:F3(0,0,1)),collisionEnabled:false)
   scene.bodies[body].mass=a.mass;scene.bodies[body].diagonalInertia=v(a.inertia)
   if spheres {_=scene.addCollider(body:body,size:F3(repeating:0.01),shape:.sphere,convexAssetID:nil)} else {for c in a.colliders {_=scene.addConvexCollider(body:body,asset:c.asset,localPosition:v(c.center))}}
   ids.append(body);witnesses.append(spheres ? [F3(0,0,-0.005),F3(0,0,0.005),F3(0.005,0,0),F3(-0.005,0,0),F3(0,0.005,0),F3(0,-0.005,0)] : a.hulls.flatMap{$0.map(v)})
  }
  let largestSpan=witnesses.map { points -> Float in
   var lo=F3(repeating:Float.infinity),hi=F3(repeating:-Float.infinity)
   for p in points {lo=simd_min(lo,p);hi=simd_max(hi,p)}
   return length(hi-lo)
  }.max() ?? 0
  // Keep the tipped mouth above the full fastener envelope, not just its
  // thickness: a diagonal bolt must be able to leave without hitting the table.
  let liftHeight=max(Float(0.18),radius*sin(Float(2.2))-height*cos(Float(2.2))+largestSpan+0.02)
  report["lift_height_m"]=liftHeight;report["largest_fastener_span_m"]=largestSpan
  report["actuator"]="three world-anchor constraints; no per-step pose resets";report["probe_kind"]=spheres ? "sphere diagnostic" : "saved demo hardware";report["contents_count"]=count;report["seed"]=seed;report["dt"]=dt;report["iterations"]=iterations;report["jar_colliders"]=sdf ? 1:f.convex_hulls.count;report["total_colliders"]=scene.colliders.count
  let fieldCount=scene.colliders.filter{$0.implicitField != nil}.count
  precondition(fieldCount == (sdf ? 1 : 0), "SDF import lost; clean rebuild required")
  report["authored_field_count"]=fieldCount;
  let initStart=Date();let solver=try GPUSolver(scene:scene,maxPairsPerBody:128);report["initialization_seconds"]=Date().timeIntervalSince(initStart)
  var times=[Double](),trace=[[[Float]]]();var penetration:Float=0;var maxLocation:[String:Any]=[:];var escaped=Set<Int>();var maxContacts=0;var lastPos=F3.zero;var lastAngle:Float=0
  let clock=Date()
  do { for i in 0..<steps {
   let start=Date();let t=Float(i)*dt
   func smooth(_ x:Float)->Float {let u=max(0,min(1,x));return u*u*(3-2*u)}
   let pos=F3(0,0,liftHeight*smooth((t-1.5)/1.0));let angle:Float=2.2*smooth((t-3)/1.3);let rot=Quat(angle:angle,axis:F3(0,1,0))
   solver.setJointWorldAnchors(drivePoints.enumerated().map{.init(joint:$0.offset,point:pos+rot.act($0.element))});lastPos=pos;lastAngle=angle
   try solver.submitStep();try solver.synchronize();times.append(Date().timeIntervalSince(start))
   if i%12==0 {
    if i%120==0 && ProcessInfo.processInfo.environment["JAR_PROGRESS"]=="1" {FileHandle.standardError.write(Data("step \(i)/\(steps)\n".utf8))}
    let jarState=solver.bodyStates([jar])[0];let actualRot=jarState.rotation;let actualPos=jarState.position;
    let states=solver.bodyStates(ids);trace.append(states.map{[$0.position.x,$0.position.y,$0.position.z]})
    maxContacts=max(maxContacts,solver.activeRigidContactCounts().reduce(0){$0+$1.contacts})
    for (j,b) in states.enumerated() {
     let local=actualRot.inverse.act(b.position-actualPos)
     if t<3 && (local.z < 0 || hypot(local.x,local.y)>radius+0.008) {escaped.insert(j)}
     for vertex in witnesses[j] {let q=actualRot.inverse.act(b.position+b.rotation.act(vertex)-actualPos);let d=try f.field.evaluate(q).distance;if -d>penetration {penetration = -d;maxLocation=["step":i,"body":j,"jar_deviation_m":length(actualPos-pos),"local_point":[q.x,q.y,q.z]]}}
    }
   }
  }} catch {report["maximum_penetration_location"]=maxLocation;report["maximum_sampled_vertex_penetration_m"]=penetration;report["containment_escapes"]=escaped.sorted();report["failure_evidence"]=solver.implicitFailureEvidence() ?? solver.convexFailureEvidence() ?? [:];report["completed_steps"]=times.count;report["trace"]=trace;throw error}
  let elapsed=Date().timeIntervalSince(clock);let sorted=times.sorted()
  let finalJar=solver.bodyStates([jar])[0]
  let finalStates=solver.bodyStates(ids)
  report["final_jar_position"]=[finalJar.position.x,finalJar.position.y,finalJar.position.z];report["final_tracking_position_error_m"]=length(finalJar.position-lastPos);report["final_tracking_rotation_error_rad"]=2*acos(min(1,abs(dot(finalJar.rotation.vector,Quat(angle:lastAngle,axis:F3(0,1,0)).vector))));
  let discharged=finalStates.filter { b in let q=finalJar.rotation.inverse.act(b.position-finalJar.position);return hypot(q.x,q.y)>radius+0.01 || q.z>height+0.01 }.count
  var tablePenetration:Float=0
  for (i,b) in finalStates.enumerated() {for vertex in witnesses[i] {tablePenetration=max(tablePenetration,-0.002-(b.position+b.rotation.act(vertex)).z)}}
  report["final_table_penetration_m"]=tablePenetration
  let trackingOK=length(finalJar.position-lastPos)<0.005 && abs(dot(finalJar.rotation.vector,Quat(angle:lastAngle,axis:F3(0,1,0)).vector))>cos(0.025)
  report["discharged_count"]=discharged;report["pour_attempted"]=Float(steps)*dt>4.3
  report["passed"]=escaped.isEmpty && penetration<0.001 && tablePenetration<0.001 && trackingOK && (Float(steps)*dt<=4.3 || discharged==count);report["containment_escapes"]=escaped.sorted();report["maximum_sampled_vertex_penetration_m"]=penetration;report["maximum_penetration_location"]=maxLocation
  report["physics_steps_per_second"]=Double(steps)/times.reduce(0,+);report["wall_seconds_including_checks"]=elapsed;report["p95_step_ms"]=sorted[Int(Double(sorted.count-1)*0.95)]*1000;report["worst_step_ms"]=sorted.last!*1000;report["max_sampled_contacts"]=maxContacts;report["trace"]=trace
  report["scope"]="Saved demo fasteners or explicit sphere control; constraint-actuated dynamic jar; sampled vertex penetration is not a continuous collision certificate; contact memory and isolated narrowphase timing not measured"
 }
}
