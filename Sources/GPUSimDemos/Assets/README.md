# PhysicsAVBD assets

`stanford-bunny.obj` is the original-scale, Y-up Stanford Bunny used by the
skinned-soft-body demo. The `classic/` directory contains deterministic Z-up,
meter-scale visual meshes for the rigid tabletop scene. Matching collision
compounds and exact debug meshes live in `convex/classic/`; runtime builds load
the checked-in compounds and never invoke CoACD.

See [`classic/PROVENANCE.md`](classic/PROVENANCE.md) for byte hashes, source
URLs, conversion and normalization details, and cook parameters. See
[`classic/NOTICE.md`](classic/NOTICE.md) before redistributing these assets.

The Stanford Bunny, Dragon, and Armadillo are research assets from the Stanford
Computer Graphics Laboratory. Stanford permits research use and free
redistribution with acknowledgement, but does **not** permit commercial use or
inclusion in a product for sale without permission. They are not unrestricted
or permissively licensed. The solid Utah Teapot source is released under CC0.
