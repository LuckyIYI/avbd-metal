import SimCore

/// Principal inertia of a solid capsule of total mass `mass`, cylinder
/// length `cylinderLength` along local z and radius `radius`, including the
/// two hemispherical caps (parallel-axis shifted to the capsule centre).
/// Returns (perpendicular, perpendicular, axial). Shared by the CPU and GPU
/// backends so a legacy capsule body has one inertia tensor.
func capsuleInertia(mass: Float, cylinderLength L: Float, radius r: Float) -> F3 {
    guard mass > 0, r > 0 else { return .zero }
    // Volume fractions: cylinder pi r^2 L, two hemispheres 4/3 pi r^3.
    let total = L + 4 * r / 3
    let mCyl = mass * L / total
    let mHemi = mass * (2 * r / 3) / total       // each cap
    let axial = 0.5 * mCyl * r * r + 2 * (0.4 * mHemi * r * r)
    let perp = mCyl * (L * L / 12 + r * r / 4)
        + 2 * mHemi * (0.4 * r * r + L * L / 4 + 3 * L * r / 8)
    return F3(perp, perp, axial)
}
