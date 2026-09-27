#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

python -m compileall -q 06_code/sitne_walk 06_code/run_sitne_walk.py \
  07_experiments/sitne_walk_results
pytest -q 06_code/tests/sitne_walk 07_experiments/sitne_walk_results
