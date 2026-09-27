# SITNE

**Shortcut-Invariant Typed Network Embedding**

SITNE is research software for typed protein–protein interaction (PPI)
networks. The repository contains two implementations: SITNE-Walk, which ranks
interaction types, and SITNE-BX, which evaluates binary PPI prediction under
inductive splits. SITNE-Walk learns topological and semantic protein embeddings
from typed random walks and combines them in a relation-ranking model.

In the audited five-fold, pair-grouped benchmark, SITNE-Walk achieved a
filtered mean reciprocal rank (MRR) of **0.9808 ± 0.0008** across 15
seed-fold runs. The strongest reference baselines were typed Skip-Gram
(0.9735), a frequency prior (0.9734), and metapath2vec (0.9717). These
results are specific to the benchmark and evaluation protocol described in the
project. The MRR difference from the frequency prior was modest (+0.0074;
paired p=0.062), and the evaluation does not establish generalization to
unseen proteins.

SITNE-Walk uses free-node embeddings and cannot represent proteins absent from
training. Such pairs are reported as `unsupported`; they are not evidence of
inductive generalization.

## Repository structure

```
06_code/          SITNE-Walk / SITNE-BX packages, CLI entry points,
                  configuration files and unit tests.
scripts/          End-to-end pipeline: environment checks, training,
                  ablations, baselines, robustness probes, enrichment,
                  case studies, and figure/table generation.
analysis/         Audited analysis outputs:
    figures/        Figures 1–7 (PNG + PDF)
    tables/         Tables 1–5
    final_tables/   Rerun tables and the manuscript results summary
    results/        Claim/artifact and source-data registries
data/             Small synthetic smoke-test data (NOT scientific evidence)
                  plus the data card describing all real datasets.
docs/             Environment and reproducibility documentation.
data_download/    Scripts to fetch and rebuild the PPI datasets locally.

Local manuscript drafts, internal handoff notes, and case-study figures pending
expert review are excluded from the public release through `.gitignore`.
```

## Environment

- Python 3.10+
- PyTorch ≥ 2.0 (CUDA build matching your driver; see `docs/environment.md`)
- numpy, pandas, scipy, scikit-learn, pyyaml, matplotlib, seaborn, tqdm, pytest

```bash
# Option A — conda
conda env create -f environment.yml && conda activate ppi-type-prediction

# Option B — venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Verify the CUDA stack:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Quick start (code-path smoke test)

The smoke run validates the software path only and sets
`scientific_evidence = false`:

```bash
./scripts/01_check_environment.sh
./scripts/02_run_tests.sh                    # pytest -q 06_code/tests/sitne_walk
./scripts/03_run_portable_smoke.sh
./scripts/05_run_results_analysis_smoke.sh
```

## Reproducing the paper results

The pipeline reads canonical data and splits, trains, and writes results under
`08_results/sitne_walk_paper_v2/`. Run order:

1. **Data** — obtain and prepare the datasets (see `data/DATA_CARD.md` and
   `data_download/`). Dataset B is not redistributed here. Dataset B*
   (BioGRID-derived, CC BY-SA) must be rebuilt locally.
2. **Splits** — freeze pair-grouped 5-fold splits (SHA-256 recorded):
   `python 06_code/prepare_typed_ranking_folds.py`
3. **Train** — `python scripts/train_all_folds_v2.py`
4. **Baselines** —
   `python scripts/run_baseline_{distmult,complex,metapath2vec,skipgram}.py`
5. **Ablations** — `python scripts/run_ablation_v2.py`
6. **R1 calibration** — `python scripts/run_r1_benchmark.py`,
   `python scripts/run_r1_formula_comparison.py`
7. **R4 probes / robustness** — `python scripts/run_r4_analysis.py`,
   `python scripts/run_r4_probes_fix.py`
8. **Statistics** — `python scripts/run_statistics.py`
9. **Biological enrichment** — `python scripts/run_biological_enrichment.py`
10. **Case studies** — `python scripts/run_case_studies.py` (requires domain-expert
    evidence curation for the final report)
11. **Figures / tables** — `python scripts/generate_figures_tables.py`,
    `python scripts/generate_figures_456.py`,
    `python scripts/generate_table3_figure3.py`,
    `python scripts/generate_figure7.py`, `python scripts/generate_figure8.py`

Every run creates a fresh output directory, refuses to overwrite existing
files, and records environment, input hashes, and RNG state for provenance.

## Data availability

- **Code**: available in this repository under the MIT License.
- **Dataset B**: not redistributed here; see Bernett et al. (DOI
  10.1093/bib/bbae076) for the original dataset and its access terms.
- **Model checkpoints**: not included in this repository.
- **Dataset B\***: derived from BioGRID 4.4.240 (CC BY-SA 4.0) combined with
  the Bernett split; the combined derivative is **not publicly redistributed**
  pending a combined-source license review. Raw interactions are available
  from BioGRID (https://downloads.thebiogrid.org) and can be rebuilt with
  `data_download/fetch_and_build_datasets.sh`.
- **External annotations** (GO, Reactome, UniProt): public databases; versions
  and accessions are listed in `data/DATA_CARD.md`.

## License

- **Code**: MIT (see `LICENSE`).
- **Data**: per-source licenses apply; see `data/DATA_CARD.md`.

## Citation

If you use SITNE, please cite the software and manuscript as described in
[`CITATION.cff`](CITATION.cff):

> Cao, M.-Y., & Zainudin, S. (2026). *SITNE-Walk: Shortcut-Invariant Typed
> Network Embedding with Corrected Walks*.

Please also cite the underlying datasets and tools: Bernett et al.
(DOI 10.1093/bib/bbae076), BioGRID, STRING, IntAct, UniProt, the Gene Ontology
and Reactome.
