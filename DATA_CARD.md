# Dataset B* — BERNETT_V3_FULL_PAPER_SPLIT_XSTAR_V2

Binary parent: `BERNETT_V3_FULL_PAPER_SPLIT_V1`.

This release adds typed interaction evidence as a third-entity relation while preserving the original binary pair IDs, labels, and split. Only non-complex-expanded, unambiguous train-positive events from the audited A_type allow-list are model inputs. Train proteins, sequences, and type vocabulary are materialized as train-only projections; validation/test artifacts remain under targets and are default-denied during training. Static projection audits pass, but runtime loader enforcement remains pending. Unknown pairs are not negatives.

The four observed coarse groups are `physical`, `spatial`, `general`, and `enzymatic`. They are an audited legacy grouping, not an official PSI-MI hierarchy. Rare enzymatic test targets and fine-type open-set targets require uncertainty-aware reporting.

## Source versions and license

| Source | Locked version | License | Redistribution |
|---|---|---|---|
| Bernett (binary parent `BERNETT_V3_FULL_PAPER_SPLIT_V1`) | `Bernett_protocols_v1` (DOI 10.1093/bib/bbae076) | CC BY 4.0 | Attribution required |
| IntAct (typed evidence) | `R252` | Not reverified | Original files not redistributed |
| BioGRID (typed evidence) | `5.0.259` | MIT | Permissive; retain notice |

**Redistribution status:** project-generated and project-derived benchmark /
reproducibility artifacts are released under CC0 1.0. Original third-party raw
database distributions (Bernett, BioGRID, IntAct, GO/GOA, Reactome, PSI-MI)
are not redistributed and remain subject to their respective source terms.

IntAct Release R252 was used as an upstream source during benchmark
construction. The exact historical license designation was not independently
reverified, and original IntAct distribution files are not redistributed.

These versions describe the typed evidence used to construct the benchmark. The manuscript's separate
case-study lookup used BioGRID `5.0.260` multi-validated physical records and IntAct's current PSICQUIC
endpoint; those lookup sources are not the dataset-construction snapshots.
