# Formal-code identity and publication byte identity

**Scope:** how the formal-run implementation identities relate to the public
release tree. This document describes the **publication release state** only; the
pre-publication state is recorded separately as historical evidence and is not
rewritten.

**Status:** applies to release `v1.0.0` onward. Release `v1.0.1` corrected the
public disclosure wording; it did not change any code, data, split or result.

## 1. Formal identities

Formal-run provenance records a SHA-256 identity for each of the **17
implementation files** (the SITNE-Walk package plus its CLI entry point). These
identities are taken from the 15 formal main-performance runs (5 folds × seeds
{42, 123, 456}) and are the authority for formal code identity:
**17/17 formal SHA-256 identities survive.**

## 2. Disclosure states

### PRE-PUBLICATION / release-candidate state

In the pre-publication current server source tree, **12 of the 17** implementation
files were byte-identical to their formal hashes. This is the historical state
recorded in the release candidate and in its formal-code identity manifest
(`byte_status` column: 12 `FORMAL_HASH_MATCH`, 5 `POST_FORMAL_MODIFIED`). It is
preserved as historical evidence and is **not** a statement about the public
release tree.

### PUBLICATION RELEASE state

In the public release tree, **11 of the 17** implementation files are
byte-identical to their formal hashes. The count is one lower than the
pre-publication count because one file, `sitne_walk/split_builder.py`, received a
**release-time default-path relocation** when the public repository was prepared.

## 3. The three categories

| Category | Count | Files |
|---|---|---|
| `FORMAL_HASH_MATCH` | 11 | `run_sitne_walk.py`, `sitne_walk/{__init__,__main__,contexts,data,evaluator,gradient_reversal,graph,losses,model,walks}.py` |
| `HISTORICAL_FORMAL_BYTES_UNAVAILABLE_POST_FORMAL_MODIFIED` | 5 | `sitne_walk/{cli,config,optuna_search,provenance,trainer}.py` |
| `PUBLICATION_PATH_RELOCATION` | 1 | `sitne_walk/split_builder.py` |

Per-file hashes, byte-match flags and notes:
[`PUBLIC_V1_FORMAL_CODE_COMPARISON.tsv`](PUBLIC_V1_FORMAL_CODE_COMPARISON.tsv).

### Five historically unrecoverable formal source versions

The exact formal source bytes are unavailable for `cli.py`, `config.py`,
`optuna_search.py`, `provenance.py` and `trainer.py`. Their published copies are
post-formal modifications (a rerun-repair change to non-finite / `fail_fast` /
bfloat16 handling, 2026-09-05/06, after the 2026-08-11 formal runs). Their formal
identities survive as hashes only; no reconstruction is attempted.

### One additional publication-only difference

`sitne_walk/split_builder.py` differs from its formal hash **only** because its
default paths were relocated during public-release preparation:

| | Value |
|---|---|
| formal identity (SHA-256) | `673c49621d93426dbb161b88a4a3f1e6352bc5fd9621f37e87ac0639c90bd886` |
| public release bytes (SHA-256) | `188c213deaa063083e1f70692a4d645358631e4cf32477b8a6a6def8b127ec7c` |

The change replaced absolute project-root default paths with module-relative
default paths. It affects only default values used when no explicit arguments
are supplied; no computed value, model logic, metric, loss, optimisation step,
hyperparameter, seed or output definition differs. Its **formal identity remains
known and preserved** and it must **not** be described as an unavailable formal
source.

## 4. What is not claimed

- No exact formal source snapshot is claimed to exist.
- The five files above do not reproduce their formal bytes.
- The `split_builder.py` difference is not claimed to be scientific; it is a
  path-default change, and no scientific calculation or reported result changed.
- No claim is made here about scientific equivalence beyond what the manuscript
  states.
