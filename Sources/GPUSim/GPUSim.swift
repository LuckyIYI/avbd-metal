/// The backend-neutral public entry point for GPU Sim.
///
/// Importing `GPUSim` exposes the scene model together with the currently
/// available CPU and Metal solver backends. The compatibility modules remain
/// available for clients that need to import a specific implementation layer.
@_exported import SimCore
@_exported import PhysicsAVBD
