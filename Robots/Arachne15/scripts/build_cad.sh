#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SCAD="$ROOT/cad/arachne15.scad"
OUT="$ROOT/build"
STL="$OUT/stl"

mkdir -p "$STL"

for part in chassis phone_tray retainer_clip coxa_link tibia_link battery_cradle foot_pad
do
  openscad --export-format binstl -D "PART=\"$part\"" \
    -o "$STL/$part.stl" "$SCAD"
done

openscad --render --imgsize 1800,1200 --viewall --autocenter \
  --projection perspective --colorscheme "Tomorrow Night" \
  -D 'PART="assembly"' -o "$OUT/arachne15-assembly.png" "$SCAD"

python3 "$ROOT/analysis/load_case.py"
python3 "$ROOT/analysis/mesh_metrics.py" "$STL"/*.stl

echo "CAD outputs: $OUT"
