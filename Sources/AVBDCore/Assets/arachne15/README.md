# Arachne-15 simulation assets

These files are generated from `Robots/Arachne15/sim/generate_model.py`.
Do not edit the XML or bundled STL copies by hand. Run:

```sh
Robots/Arachne15/scripts/build_sim.sh
```

The printable meshes are rendering assets only. Physics uses the explicit
primitive collision compounds authored in each MJCF file. `training` and
`validation` preserve identical articulation, actuator, mass, and inertia
contracts while selecting different root-link collision detail.
