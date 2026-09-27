#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

python - <<'PY'
import numpy
import pandas
import scipy
import sklearn
import torch
import yaml

print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("scikit-learn", sklearn.__version__)
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch.cuda.is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.device", torch.cuda.get_device_name(0))
print("pyyaml", yaml.__version__)
PY
