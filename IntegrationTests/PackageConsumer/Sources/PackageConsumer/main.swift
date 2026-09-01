import GPUSim

var scene = PhysicsScene(name: "external-package-consumer")
let body = scene.addBody(
    size: F3(repeating: 1),
    density: 1,
    friction: 0.5,
    position: F3(0, 0, 2)
)

let solver = try scene.makeCPUSolverChecked()
try solver.stepChecked()
precondition(solver.bodies[body].positionLin.z.isFinite)

// Compile both concrete backends through the single public package import.
let _: GPUSolver.Type = GPUSolver.self
