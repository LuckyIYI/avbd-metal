/// The backend-neutral public entry point for GPU Sim.
///
/// Importing `GPUSim` exposes the scene model together with the currently
/// available CPU and Metal solver backends. The compatibility modules remain
/// available for clients that need to import a specific implementation layer.
///
/// Swift's access-level imports make dependency modules available to clients,
/// but do not introduce their declarations into the client's lookup scope.
/// These re-exports preserve the intended single-import package contract.
@_exported import SimCore
@_exported import PhysicsAVBD
