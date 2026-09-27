#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

config="06_code/configs/sitne_walk.portable_smoke.yaml"
python 06_code/run_sitne_walk.py inspect --config "$config" --project-root "$root_dir"
python 06_code/run_sitne_walk.py train --config "$config" --project-root "$root_dir"

echo "Portable smoke completed. Outputs are diagnostic only: 08_results/portable_smoke/"
