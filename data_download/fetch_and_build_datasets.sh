#!/usr/bin/env bash
# Fetch and rebuild the SITNE-Walk PPI datasets from public sources.
#
# The real datasets are NOT bundled in this repository for license reasons
# (see data/DATA_CARD.md). This script documents how to obtain them.
#
# 1) Dataset B (Bernett et al., DOI 10.1093/bib/bbae076, CC BY 4.0):
#    obtain from GigaDB / Zenodo (accession to be added), then place the
#    contents under 02_data_canonical/dataset_B/v1/.
#
# 2) Dataset B* (BioGRID 5.0.259, IntAct R252; license review pending):
#    download source interaction records and rebuild the typed split locally.
#    BioGRID is recorded as CC BY-SA 4.0, pending author confirmation against
#    the downloaded release. Do NOT redistribute the combined derivative yet.
#
# 3) External annotations (GO, GOA, Reactome → UniProt).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/02_data_canonical/external_annotations"

echo "[1/3] BioGRID 5.0.259 interaction export"
echo "  Download from https://downloads.thebiogrid.org (registration may be"
echo "  required for the archive). Place the MITAB/PSI-MI file under:"
echo "    02_data_canonical/dataset_Bstar/v2/edges/"

echo "[2/3] External annotations"
# Gene Ontology (CC BY 4.0)
curl -L -o "$ROOT/02_data_canonical/external_annotations/go-basic.obo" \
  "http://purl.obolibrary.org/obo/go/go-basic.obo" || true
# GOA human annotations (large)
curl -L -o "$ROOT/02_data_canonical/external_annotations/goa_human.gaf.gz" \
  "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz" || true
# Reactome → UniProt mapping (CC0)
curl -L -o "$ROOT/02_data_canonical/external_annotations/UniProt2Reactome.txt" \
  "https://reactome.org/download/current/UniProt2Reactome.txt" || true

echo "[3/3] Rebuild typed splits"
echo "  Once canonical data is in place, run:"
echo "    python 06_code/prepare_typed_ranking_folds.py"
echo "  to regenerate the pair-grouped 5-fold splits (SHA-256 recorded)."

echo "Done. Review data/DATA_CARD.md before redistributing any derived data."
