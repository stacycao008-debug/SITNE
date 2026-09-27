#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/frozen_formal_config.yaml" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$1"
if [ ! -f "$config" ]; then
  echo "Config does not exist: $config" >&2
  exit 2
fi

cd "$root_dir"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing formal GPU run")
print(torch.cuda.get_device_name(0))
PY

python 06_code/run_sitne_walk.py inspect --config "$config" --project-root "$root_dir"
python 06_code/run_sitne_walk.py train --config "$config" --project-root "$root_dir"
