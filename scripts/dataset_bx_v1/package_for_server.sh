#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
manifest_relative="10_reproducibility/dataset_bx_v1/BUNDLE_SHA256SUMS.txt"
default_output="$(dirname "$project_root")/SITNE-Walk-BX-CUDA-server-v1.zip"
output_path="$default_output"
dry_run=0

usage() {
  cat <<'EOF'
Usage:
  package_for_server.sh --dry-run
  package_for_server.sh [--output /absolute/path/bundle.zip]

--dry-run only prints inclusion/exclusion statistics. It creates no manifest,
temporary staging directory, checksum, or ZIP file. Normal mode refuses to
overwrite either the ZIP or its adjacent .zip.sha256 file.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --output)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --output requires a path" >&2
        exit 2
      fi
      output_path="$2"
      shift 2
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

if [ "$dry_run" -eq 1 ]; then
  exec python3 "$script_dir/bundle_manifest.py" scan \
    --root "$project_root" \
    --manifest "$manifest_relative"
fi

exec python3 "$script_dir/bundle_manifest.py" package \
  --root "$project_root" \
  --output "$output_path" \
  --manifest "$manifest_relative" \
  --archive-prefix "SITNE-Walk-BX-CUDA-server-v1"
