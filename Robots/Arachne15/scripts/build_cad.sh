#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SCAD="$ROOT/cad/arachne15.scad"
OUT="$ROOT/build"
STL="$OUT/stl"

mkdir -p "$STL"

for part in chassis phone_guide retainer_clip coxa_link tibia_link battery_cradle foot_pad
do
  openscad --export-format binstl -D "PART=\"$part\"" \
    -o "$STL/$part.stl" "$SCAD"
done

openscad --render --imgsize 1800,1200 --viewall --autocenter \
  --projection perspective --colorscheme "Tomorrow Night" \
  -D 'PART="assembly"' -o "$OUT/arachne15-assembly.png" "$SCAD"

openscad --render --imgsize 1800,1200 --viewall --autocenter \
  --projection perspective --colorscheme "Tomorrow Night" \
  -D 'PART="assembly"' -D 'POSE="folded"' \
  -o "$OUT/arachne15-folded.png" "$SCAD"

openscad --render --imgsize 1600,1000 --viewall \
  --camera 450,0,110,0,0,80 --projection ortho \
  --colorscheme "Tomorrow Night" -D 'PART="assembly"' \
  -o "$OUT/arachne15-front.png" "$SCAD"

openscad --render --imgsize 1600,1200 --viewall \
  --camera 0,0,600,0,0,65 --projection ortho \
  --colorscheme "Tomorrow Night" -D 'PART="assembly"' \
  -o "$OUT/arachne15-top.png" "$SCAD"

python3 "$ROOT/analysis/load_case.py"
python3 "$ROOT/analysis/reveal_pose.py"
python3 "$ROOT/analysis/mesh_metrics.py" "$STL"/*.stl
python3 "$ROOT/sim/generate_model.py" --install-visual-meshes "$STL"
python3 "$ROOT/sim/generate_model.py" --check
python3 "$ROOT/sim/validate_model.py"

echo "CAD outputs: $OUT"
