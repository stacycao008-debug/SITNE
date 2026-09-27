#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package="$root_dir/07_experiments/sitne_walk_results"

for section in \
  r1_walk_correction \
  r2_main_performance \
  r3_ablation_semantic_signal \
  r4_robustness_and_boundaries; do
  python "$package/$section/run.py" \
    --config "$package/$section/smoke.yaml" \
    --project-root "$root_dir"
done

echo "Results-analysis smoke completed; all generated artifacts have scientific_evidence=false."
