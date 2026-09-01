# Arachne-15 runtime snapshot

This directory is the reviewed simulator import boundary. The Arachne-15 CAD,
BOM, generators, hardware integration, and device qualification live in an
external project and are intentionally not vendored here. Changes arrive as
an explicit runtime-asset import and must update the repository-owned verifier.

Validate the imported snapshot with:

```sh
make verify-arachne-assets
```

The printable meshes are rendering assets only. Physics uses the explicit
primitive collision compounds authored in each MJCF file. `training` and
`validation` preserve identical articulation, actuator, mass, and inertia
contracts while selecting different root-link collision detail.
