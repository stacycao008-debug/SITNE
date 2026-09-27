#!/usr/bin/env python3
"""Prepare 5-fold pair-grouped CV splits for typed interaction-type ranking.

Reads Dataset B* fine-type TSVs, converts to sitne_walk format (protein_i,
protein_j, type_name), and creates outer/inner pair-grouped folds.

Output: 05_splits/typed_ranking_v1/ with:
  - fold_0/train.tsv, fold_0/val.tsv, fold_0/test.tsv ... fold_4/...
  - all_known_positives.tsv (full positive set for filtered eval)
  - split_manifest.json (per-fold metadata & hashes)

Preserves B* identity by recording source hashes in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BSTAR_ROOT = PROJECT_ROOT / "02_data_canonical" / "dataset_Bstar" / "v2"
OUTPUT_ROOT = PROJECT_ROOT / "05_splits" / "typed_ranking_v1"

OUTER_FOLDS = 5
INNER_FOLDS = 3
SPLIT_SEED = 20260729  # separate from training seeds
RANDOM_STATE = np.random.RandomState(SPLIT_SEED)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_bstar_fine_tsv(path: Path) -> pd.DataFrame:
    """Load a B* fine-type TSV and convert to sitne_walk triple format."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Check columns
    for col in ["protein_a", "protein_b", "relation_name"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing column: {col}")
    result = pd.DataFrame({
        "protein_i": df["protein_a"].astype(str).str.strip(),
        "protein_j": df["protein_b"].astype(str).str.strip(),
        "type_name": df["relation_name"].astype(str).str.strip(),
    })
    # Ensure canonical unordered pair
    mask = result["protein_i"] > result["protein_j"]
    result.loc[mask, ["protein_i", "protein_j"]] = result.loc[mask, ["protein_j", "protein_i"]].values
    result["pair_key"] = result["protein_i"] + "|" + result["protein_j"]
    result = result.drop_duplicates().reset_index(drop=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load B* train, val, test fine-type data
    print("Loading Dataset B* fine-type facts...")
    train = load_bstar_fine_tsv(BSTAR_ROOT / "edges" / "train_pair_type_fine.tsv")
    val = load_bstar_fine_tsv(BSTAR_ROOT / "targets" / "validation_pair_type_fine.tsv")
    test = load_bstar_fine_tsv(BSTAR_ROOT / "targets" / "test_pair_type_fine.tsv")

    print(f"  Train: {len(train)} facts, {train['pair_key'].nunique()} pairs, "
          f"{train['protein_i'].nunique()} proteins, {train['type_name'].nunique()} types")
    print(f"  Val:   {len(val)} facts, {val['pair_key'].nunique()} pairs, "
          f"{val['protein_i'].nunique()} proteins, {val['type_name'].nunique()} types")
    print(f"  Test:  {len(test)} facts, {test['pair_key'].nunique()} pairs, "
          f"{test['protein_i'].nunique()} proteins, {test['type_name'].nunique()} types")

    # Build all_known_positives from ALL splits (train + val + test)
    all_positives = pd.concat([train, val, test], ignore_index=True)
    all_positives = all_positives[["protein_i", "protein_j", "type_name", "pair_key"]].drop_duplicates()
    print(f"\nAll known positives: {len(all_positives)} facts, {all_positives['pair_key'].nunique()} pairs")

    # Check pair overlap between B* splits
    train_pairs = set(train["pair_key"].unique())
    val_pairs = set(val["pair_key"].unique())
    test_pairs = set(test["pair_key"].unique())

    overlap_tv = len(train_pairs & val_pairs)
    overlap_tt = len(train_pairs & test_pairs)
    overlap_vt = len(val_pairs & test_pairs)
    print(f"\nPair overlap: train∩val={overlap_tv}, train∩test={overlap_tt}, val∩test={overlap_vt}")

    # For pair-grouped CV, we merge ALL data and re-split by pair
    all_data = pd.concat([train, val, test], ignore_index=True)
    all_data = all_data[["protein_i", "protein_j", "type_name", "pair_key"]].drop_duplicates()

    # Create 5 outer folds using GroupKFold on pair_key
    unique_pairs = all_data[["pair_key"]].drop_duplicates()
    unique_pairs = unique_pairs.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    gkf = GroupKFold(n_splits=OUTER_FOLDS)
    pair_to_outer = {}
    for fold_idx, (_, test_idx) in enumerate(gkf.split(unique_pairs, groups=unique_pairs["pair_key"])):
        for pair in unique_pairs.iloc[test_idx]["pair_key"]:
            pair_to_outer[pair] = fold_idx

    print(f"\nPairs per outer fold:")
    for fold_idx in range(OUTER_FOLDS):
        count = sum(1 for v in pair_to_outer.values() if v == fold_idx)
        print(f"  Fold {fold_idx}: {count} pairs")

    if args.dry_run:
        print("\nDry-run OK. Would create splits in:", OUTPUT_ROOT)
        return 0

    # Create output directory structure
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "builder": "prepare_typed_ranking_folds.py",
        "source_data": "Dataset B* v2 fine-type",
        "dataset_uid": "BERNETT_V3_FULL_PAPER_SPLIT_XSTAR_V2",
        "source_train_hash": sha256_file(BSTAR_ROOT / "edges" / "train_pair_type_fine.tsv"),
        "source_val_hash": sha256_file(BSTAR_ROOT / "targets" / "validation_pair_type_fine.tsv"),
        "source_test_hash": sha256_file(BSTAR_ROOT / "targets" / "test_pair_type_fine.tsv"),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "split_seed": SPLIT_SEED,
        "pair_overlap_train_val": overlap_tv,
        "pair_overlap_train_test": overlap_tt,
        "pair_overlap_val_test": overlap_vt,
        "folds": {},
    }

    # Write all_known_positives (without pair_key column - only triple columns)
    all_pos_path = OUTPUT_ROOT / "all_known_positives.tsv"
    all_positives[["protein_i", "protein_j", "type_name"]].to_csv(all_pos_path, sep="\t", index=False)
    manifest["all_known_positives"] = {
        "path": str(all_pos_path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(all_pos_path),
        "rows": len(all_positives),
        "pairs": int(all_positives["pair_key"].nunique()),
    }

    # Create per-fold splits
    for outer_fold in range(OUTER_FOLDS):
        fold_dir = OUTPUT_ROOT / f"fold_{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Test = outer fold pairs
        test_pairs = [p for p, f in pair_to_outer.items() if f == outer_fold]
        train_val_pairs = [p for p, f in pair_to_outer.items() if f != outer_fold]

        test_data = all_data[all_data["pair_key"].isin(test_pairs)]
        train_val_data = all_data[all_data["pair_key"].isin(train_val_pairs)]

        # Create inner folds within train_val for validation
        tv_unique = train_val_data[["pair_key"]].drop_duplicates()
        tv_unique = tv_unique.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        # Use the last inner fold as validation, rest as train
        inner_gkf = GroupKFold(n_splits=INNER_FOLDS)
        inner_pair_to_fold = {}
        for inner_idx, (_, val_idx) in enumerate(
            inner_gkf.split(tv_unique, groups=tv_unique["pair_key"])
        ):
            for pair in tv_unique.iloc[val_idx]["pair_key"]:
                inner_pair_to_fold[pair] = inner_idx

        # Use inner fold 0 as validation, 1-2 as training
        val_pairs_fold = [p for p, f in inner_pair_to_fold.items() if f == 0]
        train_pairs_fold = [p for p, f in inner_pair_to_fold.items() if f != 0]

        train_subset = train_val_data[train_val_data["pair_key"].isin(train_pairs_fold)]
        val_subset = train_val_data[train_val_data["pair_key"].isin(val_pairs_fold)]

        # Write files (triple columns only)
        train_path = fold_dir / "train.tsv"
        val_path = fold_dir / "val.tsv"
        test_path = fold_dir / "test.tsv"

        train_subset[["protein_i", "protein_j", "type_name"]].to_csv(train_path, sep="\t", index=False)
        val_subset[["protein_i", "protein_j", "type_name"]].to_csv(val_path, sep="\t", index=False)
        test_data[["protein_i", "protein_j", "type_name"]].to_csv(test_path, sep="\t", index=False)

        fold_info = {
            "train": {
                "path": str(train_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(train_path),
                "rows": len(train_subset),
                "pairs": int(train_subset["pair_key"].nunique()),
                "proteins": int(train_subset[["protein_i", "protein_j"]].stack().nunique()),
                "types": int(train_subset["type_name"].nunique()),
            },
            "val": {
                "path": str(val_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(val_path),
                "rows": len(val_subset),
                "pairs": int(val_subset["pair_key"].nunique()),
                "proteins": int(val_subset[["protein_i", "protein_j"]].stack().nunique()),
                "types": int(val_subset["type_name"].nunique()),
            },
            "test": {
                "path": str(test_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(test_path),
                "rows": len(test_data),
                "pairs": int(test_data["pair_key"].nunique()),
                "proteins": int(test_data[["protein_i", "protein_j"]].stack().nunique()),
                "types": int(test_data["type_name"].nunique()),
            },
            "test_pair_overlap_with_train": int(len(set(test_pairs) & set(train_pairs_fold))),
            "test_pair_overlap_with_val": int(len(set(test_pairs) & set(val_pairs_fold))),
        }
        manifest["folds"][f"fold_{outer_fold}"] = fold_info

        # Coverage check: all test proteins/relations should appear in train
        train_proteins = set(train_subset["protein_i"].unique()) | set(train_subset["protein_j"].unique())
        test_proteins_set = set(test_data["protein_i"].unique()) | set(test_data["protein_j"].unique())
        train_types = set(train_subset["type_name"].unique())
        test_types_set = set(test_data["type_name"].unique())

        fold_info["test_protein_coverage"] = round(
            len(test_proteins_set & train_proteins) / max(len(test_proteins_set), 1), 4
        )
        fold_info["test_relation_coverage"] = round(
            len(test_types_set & train_types) / max(len(test_types_set), 1), 4
        )

        print(f"\nFold {outer_fold}:")
        print(f"  Train: {len(train_subset)} rows, {train_subset['pair_key'].nunique()} pairs, "
              f"{train_subset['type_name'].nunique()} types")
        print(f"  Val:   {len(val_subset)} rows, {val_subset['pair_key'].nunique()} pairs")
        print(f"  Test:  {len(test_data)} rows, {test_data['pair_key'].nunique()} pairs")
        print(f"  Coverage: proteins={fold_info['test_protein_coverage']}, relations={fold_info['test_relation_coverage']}")

    # Write manifest
    manifest_path = OUTPUT_ROOT / "split_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\nManifest written to: {manifest_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
