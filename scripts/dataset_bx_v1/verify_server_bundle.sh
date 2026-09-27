#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_root="$(cd "$script_dir/../.." && pwd)"
project_root="${1:-$default_root}"

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [unpacked_bundle_root]" >&2
  exit 2
fi
if [ ! -d "$project_root" ]; then
  echo "ERROR: bundle root is not a directory: $project_root" >&2
  exit 2
fi
project_root="$(cd "$project_root" && pwd)"

required_paths=(
  "DATASET_BX_CUDA_SERVER_README_V1.md"
  "requirements.bx.txt"
  "02_data_canonical/dataset_B/v1"
  "02_data_canonical/dataset_Bstar/v2"
  "04_data_audit/dataset_B_research_package/v1"
  "06_code/run_sitne_bx.py"
  "06_code/sitne_bx"
  "06_code/configs/sitne_bx.binary_fp32.yaml"
  "06_code/configs/sitne_bx.fine_fp32.yaml"
  "06_code/configs/sitne_bx.coarse_fp32.yaml"
  "06_code/configs/sitne_bx.fine_fp16.yaml"
  "06_code/configs/sitne_bx.fine_bf16.yaml"
  "10_reproducibility/dataset_bx_v1/BUNDLE_SHA256SUMS.txt"
)

missing=0
for relative_path in "${required_paths[@]}"; do
  if [ ! -e "$project_root/$relative_path" ]; then
    echo "MISSING_REQUIRED_PATH=$relative_path" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo "ERROR: server bundle structure is incomplete" >&2
  exit 1
fi

python3 "$project_root/scripts/dataset_bx_v1/bundle_manifest.py" verify \
  --root "$project_root" \
  --manifest "10_reproducibility/dataset_bx_v1/BUNDLE_SHA256SUMS.txt"

echo "SERVER_BUNDLE_VERIFIED=root:$project_root"
echo "NEXT=bash scripts/dataset_bx_v1/probe_cuda_environment.sh"
