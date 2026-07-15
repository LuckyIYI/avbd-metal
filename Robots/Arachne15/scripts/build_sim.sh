#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

python3 "$ROOT/sim/generate_model.py"
python3 "$ROOT/sim/generate_model.py" --check
python3 "$ROOT/sim/validate_model.py"
