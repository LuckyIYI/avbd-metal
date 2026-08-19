# Arachne-15 simulation assets

The XML files are generated from `Robots/Arachne15/sim/generate_model.py`.
The packaged STL files are installed from the parametric CAD by
`Robots/Arachne15/scripts/build_cad.sh`. Do not edit either snapshot by hand.
To regenerate and validate MJCF without rebuilding CAD, run:

```sh
Robots/Arachne15/scripts/build_sim.sh
```

The printable meshes are rendering assets only. Physics uses the explicit
primitive collision compounds authored in each MJCF file. `training` and
`validation` preserve identical articulation, actuator, mass, and inertia
contracts while selecting different root-link collision detail.
