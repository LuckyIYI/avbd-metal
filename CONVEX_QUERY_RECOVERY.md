# Bounded convex-query recovery

The ordinary MPR/GJK path is unchanged. When its existing retries cannot provide a witness, polyhedral pairs now have a complete separating-axis fallback: both sets of face normals and every edge/edge cross axis. It runs in the first shape's local frame.

Overlap requires completion of the entire axis set and an actual surface witness from the manifold builder, checked for normal/distance/tangential consistency. For a separated pair, SAT alone is only a lower bound on Euclidean distance. An edge pair or manifold surface witness must attain that bound within the existing 50-micrometre local witness tolerance. Otherwise the query still fails closed. This is a bounded recovery path, not a replacement for a general exact-distance algorithm.

The fallback rejects more than 2,048 faces or edges per shape, or more than 131,072 edge pairs. It never truncates the separating-axis set to fit the budget. These are fallback work limits, not changes to accepted asset topology. The original typed failure, poisoned queued-successor behavior and failed-frame pose restoration remain active.

The poison buffer retains its first-word safety contract and reserves a small first-failure payload. A single GPU writer captures collider IDs and query-space inputs before rollback. `convexFailureEvidence()` returns the captured world-space shape poses and hull vertices after a typed query failure has retired, without calling the legacy trapping synchronization accessor. Other failure sources can legitimately have no captured pair.

## Regression evidence

Four captured pairs: two small hulls embedded in large floor boxes, a separated hull at a floor edge, and a small hull near another furniture hull. Tests cover each pair in both operand orders and at both the recorded origin and an origin near zero. They require contacts, finite unit normals, and upward floor response for the floor cases. The forced-query-failure regression also verifies diagnostic readback and unchanged pose rollback/queued poisoning.

The three original captures all failed the old implementation. The first floor-only recovery passed the user captures but failed the separated rim pair; a subsequent room run exposed the hull/hull case. Those failures are retained in the application repository's `out/convex-recurrence-v2` and `out/convex-recurrence-v3` reports.

The final room stress ran 2,400 solver steps across 16 rooms / 1,069 bodies without a convex-query stop (previous failures at steps 208 and 1,167). Its final-position check still failed for a sofa and pillow after the repeatedly offset drag carried them out of the floor region. This is not a stability or complete scene-physics certification. Mean physics time was 16.15 ms/step in that run; it is not a controlled performance comparison with the old implementation.

This branch also promotes the detailed-convex capacity/precision changes: all four captures now fit the supported domain and must run without skips. The fixture retains an explicit capacity guard for reuse on older engine bases. It additionally includes opt-in rigid sleeping and selected-body collision debug geometry previously used only by the application integration. Renderer features remain in PR #33.

Validation: 191 targeted XCTest cases and two Swift Testing cases passed with no skips or failures; 23 Python cooker tests and 16 subtests passed. Logs are in the application repository at `out/convex-recurrence-v3/promoted-targeted-tests.log` and `cooker-tests.log`. A broader run was stopped during unrelated full-resolution bunny remeshing; its partial log is retained as `promoted-engine-tests.log` and is not a full-suite pass. Renderer sources are already represented by PR #33; `PhysicsAVBD` and `SimCore` source trees match the application integration after this promotion.

## Rounded floor / eight-millimeter detail recurrence

A sixth captured room pair contains two 120-vertex rounded hulls, each uploading 300 edges. Their 90,000 edge pairs exceeded the previous 65,536 recovery budget before SAT could run. Both native narrowphase implementations now allow 131,072 complete edge-pair checks; larger queries still fail closed. No axes are truncated and no witness acceptance tolerance changes.

The exact capture failed on merged main in both collider orders and both coordinate frames. After this change all four configurations and all 48 ConvexGPURuntimeTests pass. This is a rare recovery work-budget increase, not a general narrowphase throughput improvement or complete room stability certification.
