# Convex capacity and scene-sized clipping

Canonical convex assets may contain up to 256 vertices, 508 triangles and 762
edges. Maximal coplanar face boundaries may contain up to 64 vertices. The cooker
continues to default to 64 vertices per hull; callers explicitly choose a larger
`max_vertices_per_hull` when preserving higher-resolution authored convex cells.
These bounds describe collision geometry, independently of visual mesh detail.

The solver selects its Metal clipping workspace from the largest uploaded face:

| Largest source face | Clipping workspace |
| --- | --- |
| 0–16 vertices | 32 vertices |
| 17–32 vertices | 64 vertices |
| 33–64 vertices | 128 vertices |

Clipping two convex polygons may need the sum of their boundary sizes. The
source limit and GPU workspace must therefore change together. Ordinary scenes
retain the established shader preamble and 32-entry arrays. Larger workspaces
specialize only the optimized convex passes when the scene contains a potential
rigid convex pair. `GPUSolver.convexClipWorkspaceVertices` exposes the selection.
Support points and topology are stored in variable-length buffers; they are not
padded to the new maximum for every hull. Larger hulls still cost more support,
face and edge work. This is a capacity feature, not a promise that additional
collision detail is free.

Validation integrates volume, centroid and face-plane calculations in Double
from the stored Float vertices, retaining the existing acceptance tolerances.
GPU upload derives topology before support-center recentering, avoiding a second
Float rounding changing coplanar face connectivity. Mouse picking clips against
actual convex planes, with a small Float-roundoff allowance at shared cell seams;
physical contact tolerances remain unchanged.

Regressions cover rotated 16-, 32- and 64-sided prism contact, settling, selection
of each workspace, malformed metadata, boundary rejection and convex mouse rays
that intersect a bounding sphere while missing the actual hull. On the development
laptop, the 180-step prism tests measured roughly 1.0, 1.5 and 4.0 ms per step.
Geometry complexity changes with each case, so these are not isolated workspace
cost measurements. A generated 24-object room with 3,968 authored convex cells
held 59.98 fps at 2560×1406 in Fast mode, with 7.49 ms mean GPU and 4.48 ms physics
time across 180 live frames. HQ rendering has a separate quality/performance budget.
