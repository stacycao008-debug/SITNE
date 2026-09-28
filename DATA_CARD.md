# Dataset B* — BERNETT_V3_FULL_PAPER_SPLIT_XSTAR_V2

Binary parent: `BERNETT_V3_FULL_PAPER_SPLIT_V1`.

This release adds typed interaction evidence as a third-entity relation while preserving the original binary pair IDs, labels, and split. Only non-complex-expanded, unambiguous train-positive events from the audited A_type allow-list are model inputs. Train proteins, sequences, and type vocabulary are materialized as train-only projections; validation/test artifacts remain under targets and are default-denied during training. Static projection audits pass, but runtime loader enforcement remains pending. Unknown pairs are not negatives.

The four observed coarse groups are `physical`, `spatial`, `general`, and `enzymatic`. They are an audited legacy grouping, not an official PSI-MI hierarchy. Rare enzymatic test targets and fine-type open-set targets require uncertainty-aware reporting.

## Source versions and license

| Source | Locked version | License | Redistribution |
|---|---|---|---|
| Bernett (binary parent `BERNETT_V3_FULL_PAPER_SPLIT_V1`) | `Bernett_protocols_v1` (DOI 10.1093/bib/bbae076) | CC BY 4.0 | Attribution required |
| IntAct (typed evidence) | `R252` | CC0 1.0 | Public domain |
| BioGRID (typed evidence) | `5.0.259` | CC BY-SA 4.0 (prior-snapshot record; PENDING author confirmation) | Combined-source review pending |

**Combined-source redistribution status:** `LOCAL_INTERNAL_ONLY_PENDING_COMBINED_SOURCE_REVIEW`.

> Legal blocker: if the downloaded BioGRID release is in fact CC BY-SA 4.0, its share-alike term
> propagates to this derived benchmark, so a journal CC0 release is legally impossible. Confirm the
> actual BioGRID download license at <https://downloads.thebiogrid.org/BioGRID> before any public
> redistribution. IntAct is CC0 and imposes no such constraint.

Canonical source-version records are `data_inventory.csv` and this card. Both record BioGRID `5.0.259`
and IntAct `R252`, with an access date of `2026-07-17`; re-verify the access date and downloaded release
files before publication. The inventory records BioGRID as CC BY-SA 4.0, pending author confirmation
against the downloaded release terms. Do not treat that record as a final legal determination or
redistribute the combined benchmark before the license review is complete.

These versions describe the typed evidence used to construct the benchmark. The manuscript's separate
case-study lookup used BioGRID `5.0.260` multi-validated physical records and IntAct's current PSICQUIC
endpoint; those lookup sources are not the dataset-construction snapshots.
