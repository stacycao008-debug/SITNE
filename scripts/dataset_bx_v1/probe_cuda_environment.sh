#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

os_name="$(uname -s)"
machine_name="$(uname -m)"
[ "$os_name" = "Linux" ] || fail "Linux is required; detected $os_name"
case "$machine_name" in
  x86_64|amd64) ;;
  *) fail "Linux x86_64 is required; detected $machine_name" ;;
esac

command -v python3 >/dev/null 2>&1 || fail "python3 is not available"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not available"

python_version="$(python3 -c 'import platform; print(platform.python_version())')"
python3 - <<'PY' || fail "Python 3.10 or newer is required"
import sys
if sys.version_info < (3, 10):
    raise SystemExit(1)
PY

nvidia_output="$(nvidia-smi)" || fail "nvidia-smi could not communicate with the NVIDIA driver"
cuda_driver_capability="$({ printf '%s\n' "$nvidia_output" | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p'; } | sed -n '1p')"
[ -n "$cuda_driver_capability" ] || fail "could not parse CUDA Version from nvidia-smi"

recommendation="$({ CUDA_DRIVER_CAPABILITY="$cuda_driver_capability" python3 - <<'PY'
import os

parts = os.environ["CUDA_DRIVER_CAPABILITY"].split(".")
version = tuple(int(part) for part in parts[:2])
if version >= (12, 8):
    recommended = "cu128"
elif version >= (12, 6):
    recommended = "cu126"
else:
    raise SystemExit(
        "NVIDIA driver CUDA capability is below 12.6; no supported wheel can be selected"
    )

supported = []
if version >= (12, 6):
    supported.append("cu126")
if version >= (12, 8):
    supported.append("cu128")
if version >= (13, 0):
    supported.append("cu130")
print(recommended)
print(",".join(supported))
PY
})" || fail "the NVIDIA driver is too old for cu126/cu128/cu130"

recommended_wheel="$(printf '%s\n' "$recommendation" | sed -n '1p')"
supported_wheels="$(printf '%s\n' "$recommendation" | sed -n '2p')"
if gpu_rows="$(nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null)"; then
  gpu_inventory_fields="index,name,driver_version,memory.total,compute_cap"
else
  gpu_rows="$(nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader)" \
    || fail "could not query GPU inventory"
  gpu_inventory_fields="index,name,driver_version,memory.total"
fi

glibc_version="unknown"
if command -v ldd >/dev/null 2>&1; then
  glibc_version="$(ldd --version 2>&1 | sed -n '1p')"
fi

echo "SITNE_BX_CUDA_PROBE=PASS"
echo "OS=$os_name"
echo "ARCH=$machine_name"
echo "PYTHON=$python_version"
echo "GLIBC=$glibc_version"
echo "NVIDIA_DRIVER_CUDA_CAPABILITY=$cuda_driver_capability"
echo "SITNE_BX_RECOMMENDED_WHEEL=$recommended_wheel"
echo "SITNE_BX_SUPPORTED_WHEELS=$supported_wheels"
echo "GPU_INVENTORY_FIELDS=$gpu_inventory_fields"
echo "GPU_INVENTORY_BEGIN"
printf '%s\n' "$gpu_rows"
echo "GPU_INVENTORY_END"
echo "NOTE=nvidia-smi CUDA Version is driver capability, not a locally installed CUDA toolkit"
