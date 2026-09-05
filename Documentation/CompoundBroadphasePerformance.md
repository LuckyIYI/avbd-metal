# Compound broadphase performance

Measured on 2026-09-05, Apple M5 (10 GPU cores), release Swift build. Baseline: `d4b687ac1e38b8f4480fe25b737912c1d07bf8a1`, the revision pinned by the TeleopKit dishwasher app.

## Change

Large rigid compounds now expose disjoint BVH subtrees of at most 16 leaves to the spatial grid. Each hierarchy thread handles at most 256 leaf combinations with a 16-entry traversal stack. Small compounds retain their existing tree and traversal order.

The count pass records accepted pairs as one-byte proxy-local leaf indices in traversal order. Emit copies these cached pairs after the exclusive scan; it no longer repeats transforms, sphere tests, or the tree walk. The leaf-node sphere already equals the previous collider sphere predicate, including hull padding, so the redundant second leaf test is removed.

The hierarchy count/scan capacity is bounded by `min(maxPairs, proxyCount * (proxyCount - 1) / 2)` (with a one-entry empty-case allocation). Emit dispatches only the live proxy pairs. Explicit encoder boundaries protect scratch-buffer reuse and GPU-written indirect arguments during unprofiled execution.

## Dishwasher results

The harness copies the app’s `buildSimulation` and engine-independent `DishwasherDesign` sources. It constructs default generated layouts with seeds 0–2 and the reference layout, retaining all authored collision geometry, collision exclusions, margin, 20 solver iterations, 1/60 timestep, and 128 pair slots per collider. Each measurement restores the same initial solver snapshot, with 3 warmup frames and 12 profiled frames; baseline and optimized runs execute sequentially. No rendering or networking runs in the harness.

| Layout | Candidates, both builds | Hierarchy before | Hierarchy after | Speedup | Profiled GPU total before → after |
|---|---:|---:|---:|---:|---:|
| 0 | 32,178 | 4.048 ms | 0.300 ms | 13.5× | 6.191 → 2.630 ms |
| 1 | 53,353 | 3.375 ms | 0.228 ms | 14.8× | 5.846 → 2.991 ms |
| 2 | 52,722 | 4.240 ms | 0.259 ms | 16.4× | 6.700 → 2.530 ms |
| reference | 35,534 | 2.729 ms | 0.329 ms | 8.3× | 5.080 → 2.592 ms |

Seed 0 has 37 bodies and 656 enabled colliders. Proxies increase from 37 to 64; the largest original compound has 124 colliders. Hierarchy scan entries fall from 83,968 to 2,016. Count falls from 1.961 to 0.259 ms; emit falls from 2.087 to 0.041 ms. The cache costs 516,096 bytes plus a 4,096-byte proxy-to-leaf table in this scene.

GPU totals sum the existing encoder timestamp measurements. They are instrumented physics timings, not application frame times or FPS. Timing varies with GPU load. Snapshot restore wall time includes CPU copies and is deliberately excluded from the performance claim. These are frozen-state comparisons; trajectories need not match the previous build bit-for-bit because splitting large compounds changes pair enumeration order. Determinism within the new implementation remains tested.

## Validation

40 Metal GPU tests passed: the full `ConvexGPURuntimeTests` suite, contact-rich trajectory determinism, exact pair capacity, and overflow rejection. New checks cover:

- Complete pair-set equality against a brute-force sphere oracle for mixed, rotated, uneven compounds with collision domains, shared geometry, and owner exclusions.
- Disjoint, complete leaf coverage and the maximum subtree size.
- Repeated snapshot restores preserving pair order without duplicates.
- Full 16×16 cached tiles, exactly 4,096 output pairs, and checked overflow at 4,225.
- A populated frame followed by an empty frame and contacts returning, with both unprofiled and profiled dispatch paths.

```sh
swift test --filter 'ConvexGPURuntimeTests|GPUSolverTests.testExactRigidPairCapacity|GPUSolverTests.testRigidPairOverflow|GPUSolverTests.testContactRichTrajectory'
```

Tests must run with Metal access; a sandbox that hides the GPU skips the Metal cases. Local raw profiles, the benchmark Swift source, machine-readable summary, and test log are under `artifacts/compound-broadphase/` (ignored generated artifacts).

## Scope and tradeoffs

The optimization applies to rigid compound scenes generally. Analytic scenes without compounds and deformable collision paths keep their existing broad phase. It uses the existing conservative sphere bounds and hull padding; tighter geometric bounds remain a separate improvement. More proxies can increase grid pair-generation work, which is included in the GPU totals above. The compact cache requires up to 256 bytes per reserved hierarchy proxy-pair entry, capped by the existing output pair capacity, plus 64 bytes per proxy.

The TeleopKit app remains pinned to the baseline revision; these results use a separate headless harness linked to the optimization worktree.
