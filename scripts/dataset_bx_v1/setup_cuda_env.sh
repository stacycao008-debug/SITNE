#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
venv_dir="$project_root/.venv"
requirements="$project_root/requirements.bx.txt"
cuda_flavor="cu128"
reuse=0

usage() {
  cat <<'EOF'
Usage: setup_cuda_env.sh [--cuda cu126|cu128|cu130|auto] [--reuse]

The default is cu128. "auto" uses the environment probe recommendation.
The virtual environment is always created inside the bundle as .venv and is
excluded from the server ZIP manifest. Existing .venv is never overwritten;
pass --reuse to install/validate in an existing environment.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cuda)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --cuda requires a value" >&2
        exit 2
      fi
      cuda_flavor="$2"
      shift 2
      ;;
    --reuse)
      reuse=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$cuda_flavor" in
  cu126|cu128|cu130|auto) ;;
  *)
    echo "ERROR: --cuda must be cu126, cu128, cu130, or auto" >&2
    exit 2
    ;;
esac
[ -f "$requirements" ] || {
  echo "ERROR: missing requirements file: $requirements" >&2
  exit 1
}

probe_output="$(bash "$script_dir/probe_cuda_environment.sh")"
printf '%s\n' "$probe_output"
if [ "$cuda_flavor" = "auto" ]; then
  cuda_flavor="$(printf '%s\n' "$probe_output" | sed -n 's/^SITNE_BX_RECOMMENDED_WHEEL=//p')"
fi
supported_wheels="$(printf '%s\n' "$probe_output" | sed -n 's/^SITNE_BX_SUPPORTED_WHEELS=//p')"
case ",$supported_wheels," in
  *",$cuda_flavor,"*) ;;
  *)
    echo "ERROR: $cuda_flavor is not supported by this NVIDIA driver; supported=$supported_wheels" >&2
    exit 1
    ;;
esac

if [ -L "$venv_dir" ]; then
  echo "ERROR: refusing symlinked bundle environment: $venv_dir" >&2
  exit 1
fi
if [ -e "$venv_dir" ]; then
  if [ "$reuse" -ne 1 ]; then
    echo "ERROR: $venv_dir already exists; refusing to overwrite it" >&2
    echo "Use --reuse only if this is the intended bundle environment." >&2
    exit 1
  fi
  [ -x "$venv_dir/bin/python" ] || {
    echo "ERROR: existing .venv has no executable bin/python" >&2
    exit 1
  }
  [ -d "$venv_dir" ] || {
    echo "ERROR: existing .venv is not a directory" >&2
    exit 1
  }
else
  python3 -m venv "$venv_dir"
fi

venv_python="$venv_dir/bin/python"
"$venv_python" -m pip install --upgrade "pip>=25,<27" "setuptools>=75" wheel
torch_index="https://download.pytorch.org/whl/$cuda_flavor"
"$venv_python" -m pip install --index-url "$torch_index" "torch==2.10.0"
"$venv_python" -m pip install --extra-index-url "$torch_index" -r "$requirements"
"$venv_python" -m pip check

expected_cuda=""
case "$cuda_flavor" in
  cu126) expected_cuda="12.6" ;;
  cu128) expected_cuda="12.8" ;;
  cu130) expected_cuda="13.0" ;;
esac

SITNE_BX_EXPECTED_CUDA="$expected_cuda" "$venv_python" - <<'PY'
import os
import platform
import torch

expected = os.environ["SITNE_BX_EXPECTED_CUDA"]
public_version = torch.__version__.split("+", 1)[0]
if public_version != "2.10.0":
    raise SystemExit(f"expected torch 2.10.0, got {torch.__version__}")
if torch.version.cuda is None:
    raise SystemExit("installed torch is CPU-only; refusing fallback")
if not torch.version.cuda.startswith(expected):
    raise SystemExit(
        f"wheel CUDA runtime mismatch: expected {expected}, got {torch.version.cuda}"
    )
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false; refusing fallback")
if torch.cuda.device_count() < 1:
    raise SystemExit("no CUDA devices are visible")

device = torch.device("cuda:0")
probe = torch.ones(1024, device=device)
if probe.device.type != "cuda" or not torch.isfinite(probe.sum()):
    raise SystemExit("CUDA tensor allocation/operation failed")
torch.cuda.synchronize(device)
print("SITNE_BX_ENVIRONMENT=VERIFIED")
print("python", platform.python_version())
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("device_count", torch.cuda.device_count())
print("device_0", torch.cuda.get_device_name(0))
PY

PYTHONDONTWRITEBYTECODE=1 "$venv_python" "$project_root/06_code/run_sitne_bx.py" --help >/dev/null
echo "DEPENDENCY_CHECK=PASS"
echo "SITNE_BX_CLI_IMPORT=PASS"
echo "VENV_READY=$venv_dir"
echo "ACTIVATE=source .venv/bin/activate"
