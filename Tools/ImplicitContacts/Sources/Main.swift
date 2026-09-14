import Foundation
import GPUSim
import simd

@main struct Main {
 static var failureEvidence:[String:Any] = [:]
 static func main() {
  do { try run() } catch {
   let report:[String:Any] = ["passed":false,"error":String(describing:error),"evidence":failureEvidence]
   if CommandLine.arguments.count>4, let data=try? JSONSerialization.data(withJSONObject:report,options:.prettyPrinted) { try? data.write(to:URL(fileURLWithPath:CommandLine.arguments[4])) }
   print("FAILED: \(error)"); exit(1)
  }
 }
 static func run() throws {
  let args=CommandLine.arguments
  let field=try JSONDecoder().decode(ImplicitField.self,from:Data(contentsOf:URL(fileURLWithPath:args[1])))
  let mode=args[2], count=Int(args[3]) ?? 16
  if mode.hasPrefix("surface-") {try runSurface(field:field,mode:mode,count:count,out:args[4]);return}
  let steps=240
  let dynamic = args.count>5 && args[5]=="dynamic"
  var scene=PhysicsScene(name:"implicit-contact")
  scene.settings.dt=1.0/240;scene.settings.iterations=4;scene.settings.collisionMargin=0.0001
  scene.settings.gravity = dynamic ? 0 : -9.81
  var balls=[Int](), rings=[Int]()
  for k in 0..<count {
   let offset=F3(Float(k%8)*0.4,Float(k/8)*0.4,0)
   let group=UInt32(k+1)
   let ring=scene.addBody(size:F3(0.2,0.2,0.04),density:dynamic ? 1000 : 0,friction:0.4,position:offset,collisionEnabled:false)
   rings.append(ring)
   if mode=="implicit" || mode=="reject" {
    let c=scene.addCollider(body:ring,size:F3(0.2,0.2,0.04),shape:.box,convexAssetID:nil,collisionGroup:group)
    scene.colliders[c].implicitField=field
   } else if mode=="boxes" {
    for j in 0..<32 {
     let a=Float(j)*2*Float.pi/32
     _=scene.addCollider(body:ring,size:F3(0.06,0.2*tan(Float.pi/32),0.04),localPosition:F3(0.07*cos(a),0.07*sin(a),0),localRotation:Quat(angle:a,axis:F3(0,0,1)),shape:.box,convexAssetID:nil,collisionGroup:group)
    }
   } else {
    for j in 0..<32 {
     let a=Float(j)*2*Float.pi/32, b=Float(j+1)*2*Float.pi/32
     var vertices=[F3]()
     for z:Float in [-0.02,0.02] { for r:Float in [0.04,0.1] { for t in [a,b] { vertices.append(F3(r*cos(t),r*sin(t),z)) } } }
     _=scene.addCollider(body:ring,size:F3(0.2,0.2,0.04),shape:.box,convexHullVertices:vertices,convexAssetID:nil,collisionGroup:group)
    }
   }
   // Every fourth sphere must pass through the unobstructed axle bore.
   let x:Float=k%4==0 ? 0 : 0.07
   let ball=scene.addBody(size:F3(repeating:0.012),density:1000,friction:0.4,position:offset+(dynamic ? F3(0.14,0,0) : F3(x*cos(0.1),x*sin(0.1),0.06)),velocity:dynamic ? F3(-0.3,0,0) : .zero,shape:.sphere,collisionEnabled:false)
   _=scene.addCollider(body:ball,size:F3(repeating:0.012),shape:mode=="reject" ? .box : .sphere,convexAssetID:nil,collisionGroup:group)
   balls.append(ball)
  }
  let start=Date()
  let solver=try GPUSolver(scene:scene,maxPairsPerBody:64)
  let initSeconds=Date().timeIntervalSince(start)
  var trace=[[[Float]]]();let clock=Date()
  for i in 0..<steps {
   try solver.submitStep();try solver.synchronize()
   if i%12==0 { trace.append(solver.bodyStates(balls).map { [$0.position.x,$0.position.y,$0.position.z] }) }
  }
  let seconds=Date().timeIntervalSince(clock)
  let states=solver.bodyStates(balls)
  let z=states.map { $0.position.z }
  let ringStates=solver.bodyStates(rings)
  let sphereMass:Float = 1000*4*Float.pi*pow(0.006,3)/3
  let momentumErrors=zip(ringStates,states).map { r,b in abs(1.6*r.linearVelocity.x+sphereMass*b.linearVelocity.x-(-0.3*sphereMass)) }
  let dynamicPass=ringStates.allSatisfy { $0.linearVelocity.x < -1e-6 && $0.position.x.isFinite } && momentumErrors.allSatisfy { $0 < 0.00001 }
  let pass=dynamic ? dynamicPass : z.enumerated().allSatisfy { i,z in z.isFinite && (i%4==0 ? z < -1 : abs(z-0.026)<0.001) }
  let report:[String:Any]=["mode":mode,"dynamic":dynamic,"momentum_error":momentumErrors,"ring_velocity_x":ringStates.map { $0.linearVelocity.x },"replicas":count,"steps":steps,"dt":1.0/240,"iterations":4,"colliders":scene.colliders.count,"initialization_seconds":initSeconds,"step_wall_seconds":seconds,"steps_per_second":Double(steps)/seconds,"final_z":z,"passed":pass,"trace":trace]
  try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:args[4]))
  print("\(mode): passed=\(pass), \(Double(steps)/seconds) steps/s; init=\(initSeconds)s")
  if !pass { exit(2) }
 }
}
